#!/usr/bin/env python3
"""Create an immutable derivative dataset for VLASH future-state training."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from uniwam.geometry.se2 import integrate_body_twist, se2_path_to_body_twist


DEFAULT_HORIZON = 48


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    return parser.parse_args()


def fixed_list(values: np.ndarray) -> pa.Array:
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": values.min(axis=0).astype(np.float32).tolist(),
        "max": values.max(axis=0).astype(np.float32).tolist(),
        "mean": values.mean(axis=0).astype(np.float32).tolist(),
        "std": values.std(axis=0).astype(np.float32).tolist(),
        "count": [int(values.shape[0])],
    }


def rewrite_episode(source: Path, target: Path, dt: float) -> tuple[dict, dict]:
    table = pq.read_table(source)
    names = table.column_names
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    episode_stats = {"observation.state": None}
    audit = {"frames": int(table.num_rows), "has_navigation": "action.nav" in names}
    arrays = {name: table[name] for name in names}
    if "action.nav" in names:
        velocity = np.asarray(table["action.nav"].to_pylist(), dtype=np.float64)
        poses = integrate_body_twist(velocity, dt).astype(np.float32)
        recovered = se2_path_to_body_twist(poses.astype(np.float64), dt)
        error = np.abs(recovered - velocity)
        state[:, :3] = poses[:-1]
        arrays["action.nav"] = fixed_list(poses[1:])
        arrays["action.nav_velocity"] = fixed_list(velocity)
        episode_stats["action.nav"] = stats(poses[1:])
        episode_stats["action.nav_velocity"] = stats(velocity)
        audit.update(
            max_velocity_roundtrip_abs=float(error.max(initial=0.0)),
            mean_velocity_roundtrip_abs=float(error.mean()),
            terminal_pose=poses[-1].tolist(),
        )
    else:
        state[:, :3] = 0.0
    arrays["observation.state"] = fixed_list(state)
    episode_stats["observation.state"] = stats(state)
    ordered = []
    for name in names:
        ordered.append(arrays[name])
        if name == "action.nav" and "action.nav_velocity" in arrays:
            ordered.append(arrays["action.nav_velocity"])
    output = pa.Table.from_arrays(ordered, names=[
        name for original in names for name in (
            (original, "action.nav_velocity") if original == "action.nav" and "action.nav_velocity" in arrays else (original,)
        )
    ])
    temporary = target.with_suffix(target.suffix + ".tmp")
    pq.write_table(output, temporary, compression="zstd")
    os.replace(temporary, target)
    return episode_stats, audit


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if args.horizon <= 0:
        raise ValueError(f"horizon must be positive, got {args.horizon}")
    staging = output.with_name(f".{output.name}.building-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    shutil.copytree(source, staging, copy_function=os.link)

    info_path = staging / "meta/info.json"
    info = json.loads(info_path.read_text())
    fps = float(info["fps"])
    if abs(fps - float(args.fps)) > 1.0e-6:
        raise ValueError(f"Dataset fps={fps}, requested fps={args.fps}")
    features = info["features"]
    state_names = list(features["observation.state"]["names"][0])
    state_names[:3] = ["base_x", "base_y", "base_yaw"]
    features["observation.state"]["names"] = [state_names]
    if "action.nav" in features:
        features["action.nav"]["names"] = [["base_x_target", "base_y_target", "base_yaw_target"]]
        features["action.nav_velocity"] = {
            "dtype": "float32",
            "shape": [3],
            "names": [["base_vx", "base_vy", "base_wz"]],
        }
    info["robot_type"] = str(info.get("robot_type", "agx")) + "_vlash_futurestate_se2"
    info["fastwam_vlash_future_state_contract"] = {
        "attribution": "VLASH SharedObservationVLASHDataset semantics adapted to FastWAM phase-split",
        "shared_observation": True,
        "max_delay_steps": 12,
        "action_horizon": int(args.horizon),
        "state_reference": "observation snapshot t",
        "navigation_state": "episode cumulative SE2 pose before command",
        "navigation_action": "episode cumulative SE2 pose after command",
        "navigation_velocity_source": "action.nav_velocity body twist",
        "integration": "exact SE2 exponential at 30Hz",
    }
    info_path.unlink()
    info_path.write_text(json.dumps(info, indent=2) + "\n")

    old_stats = {
        int(row["episode_index"]): row
        for row in map(json.loads, (source / "meta/episodes_stats.jsonl").read_text().splitlines())
    }
    audit_rows = []
    stats_rows = []
    data_files = sorted((staging / "data").glob("chunk-*/episode_*.parquet"))
    for index, target in enumerate(data_files):
        episode_index = int(target.stem.split("_")[-1])
        new_stats, audit = rewrite_episode(source / target.relative_to(staging), target, 1.0 / fps)
        row = old_stats[episode_index]
        row["stats"].update(new_stats)
        stats_rows.append(row)
        audit_rows.append({"episode_index": episode_index, **audit})
        if (index + 1) % 100 == 0:
            print(f"converted {index + 1}/{len(data_files)}", flush=True)
    stats_path = staging / "meta/episodes_stats.jsonl"
    stats_path.unlink()
    stats_path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in stats_rows))

    episode_lengths = {row["episode_index"]: int(row["stats"]["observation.state"]["count"][0]) for row in stats_rows}
    old_window = pq.read_table(source / "meta/window_index.parquet")
    source_episode = {}
    if "source_episode_index" in old_window.column_names:
        for episode, src_episode in zip(old_window["episode_index"].to_pylist(), old_window["source_episode_index"].to_pylist()):
            source_episode.setdefault(int(episode), int(src_episode))
    windows = []
    for episode_index, length in sorted(episode_lengths.items()):
        for start in range(max(0, length - int(args.horizon))):
            windows.append({
                "episode_index": episode_index,
                "source_episode_index": source_episode.get(episode_index, episode_index),
                "start_frame": start,
                "horizon": int(args.horizon),
                "sample_type": "unrouted",
                "nav_loss_valid": "action.nav" in features,
                "manip_loss_valid": "action.manip" in features,
                "nav_active_count": 0,
                "manip_active_count": 0,
            })
    window_path = staging / "meta/window_index.parquet"
    window_path.unlink()
    pq.write_table(pa.Table.from_pylist(windows), window_path, compression="zstd")
    stale_weights = staging / "meta/window_sampling_weights.parquet"
    if stale_weights.exists():
        stale_weights.unlink()
    for stale_phase in (staging / "meta").glob("phase_split*"):
        if stale_phase.is_dir():
            shutil.rmtree(stale_phase)

    conversion = staging / "conversion"
    conversion.mkdir(exist_ok=True)
    summary = {
        "source": str(source),
        "output": str(output),
        "episodes": len(audit_rows),
        "frames": int(sum(row["frames"] for row in audit_rows)),
        "window_horizon": int(args.horizon),
        "windows": len(windows),
        "max_velocity_roundtrip_abs": max((row.get("max_velocity_roundtrip_abs", 0.0) for row in audit_rows), default=0.0),
        "contract": info["fastwam_vlash_future_state_contract"],
    }
    (conversion / "conversion_audit.json").write_text(json.dumps({"summary": summary, "episodes": audit_rows}, indent=2) + "\n")
    os.replace(staging, output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
