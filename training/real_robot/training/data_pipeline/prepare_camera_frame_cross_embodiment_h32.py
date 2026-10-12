#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CACHE: Path
OUT: Path
REMAP: Path
FRANKA: Path
SOURCES: dict[str, dict]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def video_path(root: Path, episode: int) -> Path:
    return (
        root
        / f"videos/chunk-{episode // 1000:03d}"
        / "observation.images.cam_manip_high"
        / f"episode_{episode:06d}.mp4"
    )


def video_hash_map(root: Path) -> tuple[list[dict], dict[str, int]]:
    episodes = jsonl(root / "meta/episodes.jsonl")
    mapping: dict[str, int] = {}
    for row in episodes:
        episode = int(row["episode_index"])
        digest = sha256(video_path(root, episode))
        if digest in mapping:
            raise ValueError(f"Duplicate main-camera video hash in {root}: {digest}")
        mapping[digest] = episode
    return episodes, mapping


def failed_episodes(root: Path) -> set[int]:
    info = json.loads((root / "meta/info.json").read_text())
    return {
        int(index)
        for index, label in info.get("failed_episode_labels", {}).items()
        if label.get("status") == "FAILED"
    }


def cache_locations(source: str, branch: str, count: int) -> tuple[np.ndarray, np.ndarray]:
    rank_map = np.full(count, -1, dtype=np.int16)
    row_map = np.full(count, -1, dtype=np.int32)
    for rank in range(8):
        path = CACHE / source / branch / f"rank_{rank:02d}_source_indices.npy"
        indices = np.asarray(np.load(path), dtype=np.int64)
        if indices.size and (indices.min() < 0 or indices.max() >= count):
            raise ValueError(f"Cache index outside old window table: {path}")
        if np.any(rank_map[indices] >= 0):
            raise ValueError(f"Duplicate cached source indices: {path}")
        rank_map[indices] = rank
        row_map[indices] = np.arange(indices.size, dtype=np.int32)
    return rank_map, row_map


def write_remap(
    source: str,
    branches: tuple[str, ...],
    old_count: int,
    kept_old_rows: np.ndarray,
    base_remap: Path | None,
) -> dict:
    report = {}
    for branch in branches:
        if base_remap is None:
            old_rank, old_row = cache_locations(source, branch, old_count)
        else:
            base = base_remap / source / branch
            old_rank = np.asarray(np.load(base / "rank_map.npy"), dtype=np.int16)
            old_row = np.asarray(np.load(base / "row_map.npy"), dtype=np.int32)
            if old_rank.shape != (old_count,) or old_row.shape != (old_count,):
                raise ValueError(f"Base remap shape mismatch for {source}/{branch}")
        rank_map = old_rank[kept_old_rows]
        row_map = old_row[kept_old_rows]
        target = REMAP / source / branch
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "rank_map.npy", rank_map)
        np.save(target / "row_map.npy", row_map)
        report[branch] = {
            "mapped_windows": int(np.count_nonzero(rank_map >= 0)),
            "unmapped_windows": int(np.count_nonzero(rank_map < 0)),
        }
    return report


def prepare_agilex() -> dict:
    output = {}
    remap_report = {"status": "PASS", "sources": {}}
    for source, spec in SOURCES.items():
        old_episodes, _ = video_hash_map(spec["old_root"])
        camera_episodes, camera_hashes = video_hash_map(spec["camera_root"])
        old_to_camera = {}
        for row in old_episodes:
            old_episode = int(row["episode_index"])
            digest = sha256(video_path(spec["old_root"], old_episode))
            if digest in camera_hashes:
                new_episode = camera_hashes[digest]
                if int(row["length"]) != int(camera_episodes[new_episode]["length"]):
                    raise ValueError(f"Video-matched episode length differs: {source}/{old_episode}")
                old_to_camera[old_episode] = new_episode

        failed = failed_episodes(spec["camera_root"])
        old_window = pq.read_table(spec["window"])
        old_episode = np.asarray(old_window["episode_index"].to_numpy(), dtype=np.int64)
        mapped_episode = np.asarray(
            [old_to_camera.get(int(value), -1) for value in old_episode], dtype=np.int64
        )
        keep = (mapped_episode >= 0) & ~np.isin(mapped_episode, list(failed))
        kept_old_rows = np.flatnonzero(keep).astype(np.int64)
        new_window = old_window.filter(pa.array(keep))
        new_episode = mapped_episode[keep]
        episode_column = new_window.schema.get_field_index("episode_index")
        new_window = new_window.set_column(
            episode_column, "episode_index", pa.array(new_episode, type=pa.int64())
        )
        if "source_episode_index" in new_window.column_names:
            source_ids = np.asarray(
                [camera_episodes[int(value)].get("source_episode_index", int(value)) for value in new_episode],
                dtype=np.int64,
            )
            source_column = new_window.schema.get_field_index("source_episode_index")
            new_window = new_window.set_column(
                source_column, "source_episode_index", pa.array(source_ids, type=pa.int64())
            )

        source_out = OUT / source
        source_out.mkdir(parents=True, exist_ok=True)
        window_out = source_out / "window_index_h32.parquet"
        pq.write_table(new_window, window_out, compression="zstd")

        old_to_new_row = np.full(old_window.num_rows, -1, dtype=np.int64)
        old_to_new_row[kept_old_rows] = np.arange(kept_old_rows.size, dtype=np.int64)
        phase = pq.read_table(spec["phase"])
        phase_source = np.asarray(phase["source_window_index"].to_numpy(), dtype=np.int64)
        phase_keep = old_to_new_row[phase_source] >= 0
        new_phase = phase.filter(pa.array(phase_keep))
        phase_source_new = old_to_new_row[phase_source[phase_keep]]
        phase_source_column = new_phase.schema.get_field_index("source_window_index")
        new_phase = new_phase.set_column(
            phase_source_column,
            "source_window_index",
            pa.array(phase_source_new, type=pa.int64()),
        )
        phase_episode_column = new_phase.schema.get_field_index("episode_index")
        new_phase = new_phase.set_column(
            phase_episode_column,
            "episode_index",
            pa.array(new_episode[phase_source_new], type=pa.int64()),
        )
        phase_out = source_out / "phase_branch_index_h32.parquet"
        pq.write_table(new_phase, phase_out, compression="zstd")

        branches = write_remap(
            source, spec["branches"], old_window.num_rows, kept_old_rows, spec["base_remap"]
        )
        output[source] = {
            "old_episodes": len(old_episodes),
            "camera_episodes": len(camera_episodes),
            "mapped_episodes": len(old_to_camera),
            "failed_camera_episodes": sorted(failed),
            "old_windows": old_window.num_rows,
            "kept_windows": new_window.num_rows,
            "removed_windows": int(old_window.num_rows - new_window.num_rows),
            "old_phase_rows": phase.num_rows,
            "kept_phase_rows": new_phase.num_rows,
            "window_index": str(window_out),
            "window_sha256": sha256(window_out),
            "phase_index": str(phase_out),
            "phase_sha256": sha256(phase_out),
            "branches": branches,
        }
        remap_report["sources"][source] = {
            "old_window_index_sha256": sha256(spec["window"]),
            "current_window_index_sha256": sha256(window_out),
            "branches": branches,
        }

    REMAP.mkdir(parents=True, exist_ok=True)
    (REMAP / "remap_report.json").write_text(
        json.dumps(remap_report, indent=2, sort_keys=True) + "\n"
    )
    return output


def prepare_franka() -> dict:
    episodes = jsonl(FRANKA / "meta/episodes.jsonl")
    rows = {
        "episode_index": [],
        "source_episode_index": [],
        "start_frame": [],
        "horizon": [],
        "sample_type": [],
        "nav_loss_valid": [],
        "manip_loss_valid": [],
    }
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        source_episode_index = int(episode.get("source_episode_index", episode_index))
        for start in range(0, length - 32, 4):
            rows["episode_index"].append(episode_index)
            rows["source_episode_index"].append(source_episode_index)
            rows["start_frame"].append(start)
            rows["horizon"].append(32)
            rows["sample_type"].append("pure_manip")
            rows["nav_loss_valid"].append(False)
            rows["manip_loss_valid"].append(True)
    table = pa.table(
        {
            "episode_index": pa.array(rows["episode_index"], type=pa.int64()),
            "source_episode_index": pa.array(rows["source_episode_index"], type=pa.int64()),
            "start_frame": pa.array(rows["start_frame"], type=pa.int64()),
            "horizon": pa.array(rows["horizon"], type=pa.int64()),
            "sample_type": pa.array(rows["sample_type"]),
            "nav_loss_valid": pa.array(rows["nav_loss_valid"]),
            "manip_loss_valid": pa.array(rows["manip_loss_valid"]),
        }
    )
    target = OUT / "franka_cup_pyramid"
    target.mkdir(parents=True, exist_ok=True)
    path = target / "window_index_h32_stride4.parquet"
    pq.write_table(table, path, compression="zstd")
    return {
        "episodes": len(episodes),
        "windows": table.num_rows,
        "stride": 4,
        "window_index": str(path),
        "window_sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    settings = payload["index_preparation"]
    expected = (
        "agx_cup_tray_mobile_rot6d_vlash_se2_h48_phase_split_v3",
        "agx_move_white_box_mobile_rot6d_vlash_se2_h48_phase_split_v2",
        "agx_color_blocks_mobile_rot6d_vlash_se2_h48_phase_split_v3",
    )
    if tuple(settings["sources"]) != expected:
        raise ValueError("Index preparation must list cup-tray, mobile-white-box, and ordinary color in order")
    global CACHE, OUT, REMAP, FRANKA, SOURCES
    CACHE = Path(settings["latent_cache_root"])
    FRANKA = Path(payload["sources"][0]["dataset_root"])
    OUT = args.output_root
    REMAP = OUT / "latent_remap"
    SOURCES = {}
    for name, item in settings["sources"].items():
        branches = tuple(item["branches"])
        if branches != (("manip",) if name == expected[2] else ("manip", "nav")):
            raise ValueError(f"Unexpected branch contract for {name}: {branches}")
        SOURCES[name] = {
            "old_root": Path(item["old_root"]),
            "camera_root": Path(item["camera_root"]),
            "window": Path(item["window_index"]),
            "phase": Path(item["phase_index"]),
            "branches": branches,
            "base_remap": Path(item["base_remap_root"]) if item.get("base_remap_root") else None,
        }
    if OUT.exists():
        raise FileExistsError(f"Refusing to replace existing index output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "PASS",
        "agilex": prepare_agilex(),
        "franka": prepare_franka(),
    }
    path = OUT / "camera_frame_h32_index_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
