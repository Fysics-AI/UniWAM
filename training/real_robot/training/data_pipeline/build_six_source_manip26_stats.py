#!/usr/bin/env python3
"""Build fixed-reference 20D stats for the six-source camera-frame mix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rot6_to_matrix(values: np.ndarray) -> np.ndarray:
    rows = values.reshape(*values.shape[:-1], 2, 3)
    r0 = rows[..., 0, :]
    r1 = rows[..., 1, :]
    r0 = r0 / np.maximum(np.linalg.norm(r0, axis=-1, keepdims=True), 1e-12)
    r1 = r1 - r0 * np.sum(r0 * r1, axis=-1, keepdims=True)
    r1 = r1 / np.maximum(np.linalg.norm(r1, axis=-1, keepdims=True), 1e-12)
    return np.stack((r0, r1, np.cross(r0, r1)), axis=-2)


def matrix_to_rot6(matrix: np.ndarray) -> np.ndarray:
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def fixed_delta(target: np.ndarray, snapshot: np.ndarray) -> np.ndarray:
    out = np.empty_like(target, dtype=np.float32)
    out[..., :3] = target[..., :3] - snapshot[..., None, :3]
    out[..., 10:13] = target[..., 10:13] - snapshot[..., None, 10:13]
    left = rot6_to_matrix(target[..., 3:9]) @ np.swapaxes(rot6_to_matrix(snapshot[..., 3:9]), -1, -2)[..., None, :, :]
    right = rot6_to_matrix(target[..., 13:19]) @ np.swapaxes(rot6_to_matrix(snapshot[..., 13:19]), -1, -2)[..., None, :, :]
    out[..., 3:9] = matrix_to_rot6(left)
    out[..., 13:19] = matrix_to_rot6(right)
    out[..., 9] = target[..., 9]
    out[..., 19] = target[..., 19]
    return out


def relative_nav(target: np.ndarray, snapshot: np.ndarray) -> np.ndarray:
    dx = target[..., 0] - snapshot[..., None, 0]
    dy = target[..., 1] - snapshot[..., None, 1]
    yaw = snapshot[..., None, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.empty_like(target, dtype=np.float32)
    out[..., 0] = c * dx + s * dy
    out[..., 1] = -s * dx + c * dy
    out[..., 2] = (target[..., 2] - snapshot[..., None, 2] + np.pi) % (2 * np.pi) - np.pi
    return out


class Running:
    def __init__(self, dim: int, reservoir_size: int, rng: np.random.Generator):
        self.dim = dim
        self.count = 0
        self.total = np.zeros(dim, np.float64)
        self.square = np.zeros(dim, np.float64)
        self.minimum = np.full(dim, np.inf, np.float64)
        self.maximum = np.full(dim, -np.inf, np.float64)
        self.reservoir = np.empty((reservoir_size, dim), np.float32)
        self.reservoir_size = reservoir_size
        self.rng = rng

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32).reshape(-1, self.dim)
        if values.size == 0:
            return
        finite = np.isfinite(values).all(axis=1)
        values = values[finite]
        if values.size == 0:
            return
        self.count += len(values)
        values64 = values.astype(np.float64)
        self.total += values64.sum(axis=0)
        self.square += np.square(values64).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values64.min(axis=0))
        self.maximum = np.maximum(self.maximum, values64.max(axis=0))
        start = self.count - len(values)
        for local, value in enumerate(values):
            index = start + local
            if index < self.reservoir_size:
                self.reservoir[index] = value
            else:
                slot = int(self.rng.integers(0, index + 1))
                if slot < self.reservoir_size:
                    self.reservoir[slot] = value

    def finish(self, rotation_slots: tuple[int, ...]) -> dict:
        mean = self.total / max(self.count, 1)
        variance = np.maximum(self.square / max(self.count, 1) - mean * mean, 0.0)
        n = min(self.count, self.reservoir_size)
        q01 = np.quantile(self.reservoir[:n], 0.01, axis=0)
        q99 = np.quantile(self.reservoir[:n], 0.99, axis=0)
        for dim in rotation_slots:
            q01[dim] = -1.0
            q99[dim] = 1.0
        return {
            "global_min": self.minimum.astype(np.float32).tolist(),
            "global_max": self.maximum.astype(np.float32).tolist(),
            "global_mean": mean.astype(np.float32).tolist(),
            "global_std": np.sqrt(variance).astype(np.float32).tolist(),
            "global_q01": q01.astype(np.float32).tolist(),
            "global_q99": q99.astype(np.float32).tolist(),
            "sample_count": int(self.count),
            "quantile_reservoir": int(n),
        }


def load_rows(root: Path, window_path: Path, branch: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(window_path)
    episode = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    start = np.asarray(table["start_frame"].to_numpy(), dtype=np.int64)
    weight = np.asarray(table["sampling_weight"].to_numpy(), dtype=np.float64) if "sampling_weight" in table.column_names else np.ones(len(start), np.float64)
    if branch is not None:
        phase = pq.read_table(window_path, columns=["branch"]) if "branch" in table.column_names else None
        if phase is None:
            raise ValueError(f"Branch column missing from {window_path}")
        keep = np.asarray(phase["branch"].to_pylist(), dtype=object) == branch
        episode, start, weight = episode[keep], start[keep], weight[keep]
    return episode, start, weight


def sample_records(root: Path, window_path: Path, branch: str | None, count: int, rng: np.random.Generator):
    episode, start, weight = load_rows(root, window_path, branch)
    if len(start) == 0:
        raise ValueError(f"No {branch} rows in {window_path}")
    probability = np.maximum(weight, 0.0)
    probability /= probability.sum()
    chosen = rng.choice(len(start), size=count, replace=True, p=probability)
    return episode[chosen], start[chosen]


def read_episode(root: Path, episode: int, *, need_nav: bool) -> dict[str, np.ndarray]:
    path = root / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
    columns = ["observation.state.camera_dual_arm", "action.manip.camera_dual_arm"]
    if need_nav:
        columns.extend(["observation.state", "action.nav"])
    table = pq.read_table(path, columns=columns)
    return {name: np.asarray(table[name].to_pylist(), dtype=np.float32) for name in columns}


def accumulate(root: Path, episodes: np.ndarray, starts: np.ndarray, manip: Running, state: Running, nav: Running | None, offset: int) -> None:
    order = np.argsort(episodes, kind="stable")
    episodes, starts = episodes[order], starts[order]
    for episode in np.unique(episodes):
        selected = starts[episodes == episode]
        data = read_episode(root, int(episode), need_nav=nav is not None)
        state_values = data["observation.state.camera_dual_arm"][selected]
        state.update(state_values)
        indices = selected[:, None] + offset + np.arange(32, dtype=np.int64)[None, :]
        if indices.max() >= len(data["action.manip.camera_dual_arm"]):
            raise ValueError(f"Offset/horizon crosses episode {episode} in {root}")
        snapshot = state_values
        target = data["action.manip.camera_dual_arm"][indices]
        manip.update(fixed_delta(target, snapshot))
        if nav is not None:
            nav_target = data["action.nav"][indices]
            nav.update(relative_nav(nav_target, data["observation.state"][selected]))


def accumulate_nav(root: Path, episodes: np.ndarray, starts: np.ndarray, nav: Running) -> None:
    order = np.argsort(episodes, kind="stable")
    episodes, starts = episodes[order], starts[order]
    for episode in np.unique(episodes):
        selected = starts[episodes == episode]
        data = read_episode(root, int(episode), need_nav=True)
        indices = selected[:, None] + np.arange(32, dtype=np.int64)[None, :]
        if indices.max() >= len(data["action.nav"]):
            raise ValueError(f"Navigation horizon crosses episode {episode} in {root}")
        nav.update(relative_nav(data["action.nav"][indices], data["observation.state"][selected]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows-per-weight-unit", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--group", choices=("global", "franka", "piper"), default="global")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["sources"]
    expected_names = (
        "franka_cup_pyramid", "agx_cup_tray", "agx_move_white_box",
        "agx_color_blocks", "agx_ordered_color_blocks", "agilex7000_front30",
    )
    if tuple(row["name"] for row in rows) != expected_names:
        raise ValueError("Manifest sources are not in the six-source parent order")
    weights = [float(value) for value in manifest["source_sampling_weights"]]
    if weights != [3.0, 1.0, 1.0, 1.0, 1.0, 7.0]:
        raise ValueError(f"Unexpected source sampling weights: {weights}")
    sources = [
        (
            Path(row["dataset_root"]),
            Path(row["phase_index"] if 1 <= index <= 3 else row["window_index"]),
            "manip" if 1 <= index <= 3 else None,
            weights[index],
            index in (1, 2),
        )
        for index, row in enumerate(rows)
    ]
    if args.group == "franka":
        sources = sources[:1]
    elif args.group == "piper":
        sources = sources[1:]
    rng = np.random.default_rng(args.seed)
    state_stats = Running(20, 500000, rng)
    manip_stats = Running(20, 500000, rng)
    nav_stats = Running(3, 500000, rng)
    audits = []
    for root, window, branch, source_weight, has_nav in sources:
        count = int(round(args.windows_per_weight_unit * source_weight))
        episodes, starts = sample_records(root, window, branch, count, rng)
        accumulate(root, episodes, starts, manip_stats, state_stats, None, 1)
        if has_nav:
            nav_episodes, nav_starts = sample_records(root, window, "nav", count, rng)
            accumulate_nav(root, nav_episodes, nav_starts, nav_stats)
        audits.append({"root": str(root), "window_index": str(window), "branch": branch or "all", "source_weight": source_weight, "sampled_windows": count})
        print(f"sampled {root.name} windows={count}", flush=True)
    if nav_stats.count:
        nav_payload = nav_stats.finish(())
    else:
        nav_payload = {
            "global_min": [0.0, 0.0, 0.0], "global_max": [0.0, 0.0, 0.0],
            "global_mean": [0.0, 0.0, 0.0], "global_std": [1.0, 1.0, 1.0],
            "global_q01": [-1.0, -1.0, -1.0], "global_q99": [1.0, 1.0, 1.0],
            "sample_count": 0, "quantile_reservoir": 0,
        }
    output = {
        "state": {"default": state_stats.finish((3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18))},
        "action": {"manip": manip_stats.finish((3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18)), "nav": nav_payload},
        "contract": {
            "status": "PASS", "manip_action_dim": 20, "proprio_dim": 20,
            "manip_relative_frame": "fixed_reference", "manip_action_offset": 1,
            "action_horizon": 32, "rotation_representation": "row_major_rot6d",
            "source_weights": [float(item[3]) for item in sources],
            "windows_per_weight_unit": args.windows_per_weight_unit,
            "seed": args.seed,
        },
        "audit": audits,
        "provenance": {"builder": str(Path(__file__).resolve()), "builder_sha256": sha256(Path(__file__).resolve())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(args.output), "manip_rows": manip_stats.count, "state_rows": state_stats.count, "nav_rows": nav_stats.count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
