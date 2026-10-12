#!/usr/bin/env python3
"""Map the immutable H48 latent cache onto the current cleaned H32 indices."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--old-index-root", type=Path, required=True)
    parser.add_argument("--current-index-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Restrict remapping to one or more source names.",
    )
    parser.add_argument(
        "--allow-missing-required",
        action="store_true",
        help="Write the remap and report missing routed rows instead of failing.",
    )
    return parser.parse_args()


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_keys(table_path: Path) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    table = pq.read_table(table_path, columns=["source_episode_index", "start_frame"])
    source_episode = np.asarray(table["source_episode_index"].to_numpy(), dtype=np.int64)
    start = np.asarray(table["start_frame"].to_numpy(), dtype=np.int64)
    keys = [(int(ep), int(frame)) for ep, frame in zip(source_episode, start, strict=True)]
    lookup = {key: index for index, key in enumerate(keys)}
    if len(lookup) != len(keys):
        raise ValueError(f"Non-unique stable window key in {table_path}.")
    return keys, lookup


def required_source_indices(current_index_root: Path, source_name: str, branch: str) -> np.ndarray:
    source_root = current_index_root / source_name
    phase_candidates = (
        source_root / "phase_h32_current/phase_branch_index.parquet",
        source_root / "phase_h32_episode80_clean/phase_branch_index.parquet",
        source_root / "phase_h32/phase_branch_index.parquet",
        source_root / "phase_branch_index_cache_aligned_h32.parquet",
    )
    phase_path = next((path for path in phase_candidates if path.is_file()), None)
    if phase_path is None:
        # The pure-manipulation source routes every base window directly.
        table = pq.read_table(current_index_root / source_name / "window_index_h32.parquet")
        return np.arange(table.num_rows, dtype=np.int64)
    phase = pq.read_table(phase_path, columns=["branch", "source_window_index"])
    branches = np.asarray(phase["branch"].to_pylist(), dtype=object)
    indices = np.asarray(phase["source_window_index"].to_numpy(), dtype=np.int64)
    return np.unique(indices[branches == branch])


def main() -> int:
    args = parse_args()
    manifest = json.loads((args.cache_root / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Cache manifest is not complete: {manifest.get('status')!r}.")
    world_size = int(manifest["world_size"])
    manifest_source_names = {str(source["name"]) for source in manifest["sources"]}
    requested_sources = manifest_source_names if args.source is None else set(args.source)
    unknown_sources = requested_sources - manifest_source_names
    if unknown_sources:
        raise ValueError(f"Requested sources are absent from cache: {sorted(unknown_sources)}")
    report: dict[str, object] = {
        "status": "PASS",
        "cache_root": str(args.cache_root),
        "old_index_root": str(args.old_index_root),
        "current_index_root": str(args.current_index_root),
        "mapping_key": ["source_episode_index", "start_frame"],
        "requested_sources": sorted(requested_sources),
        "sources": {},
    }

    for source in manifest["sources"]:
        source_name = str(source["name"])
        if source_name not in requested_sources:
            continue
        old_path = args.old_index_root / source_name / "window_index_h32.parquet"
        current_path = args.current_index_root / source_name / "window_index_h32.parquet"
        old_keys, _ = stable_keys(old_path)
        current_keys, current_lookup = stable_keys(current_path)
        source_report: dict[str, object] = {
            "old_windows": len(old_keys),
            "current_windows": len(current_keys),
            "old_window_index_sha256": sha256(old_path),
            "current_window_index_sha256": sha256(current_path),
            "branches": {},
        }

        for branch in source["branches"]:
            rank_map = np.full(len(current_keys), -1, dtype=np.int16)
            row_map = np.full(len(current_keys), -1, dtype=np.int32)
            old_cached_rows = 0
            removed_cached_rows = 0
            for rank in range(world_size):
                indices_path = (
                    args.cache_root
                    / source_name
                    / branch
                    / f"rank_{rank:02d}_source_indices.npy"
                )
                old_indices = np.load(indices_path, mmap_mode="r")
                for row, old_index_value in enumerate(old_indices):
                    old_index = int(old_index_value)
                    if not 0 <= old_index < len(old_keys):
                        raise IndexError(f"Cache index {old_index} is outside {old_path}.")
                    current_index = current_lookup.get(old_keys[old_index])
                    old_cached_rows += 1
                    if current_index is None:
                        removed_cached_rows += 1
                        continue
                    if rank_map[current_index] >= 0:
                        raise ValueError(
                            f"Duplicate cache mapping for {source_name}/{branch}/"
                            f"current window {current_index}."
                        )
                    rank_map[current_index] = rank
                    row_map[current_index] = row

            required = required_source_indices(args.current_index_root, source_name, branch)
            missing_required = required[rank_map[required] < 0]
            if missing_required.size and not args.allow_missing_required:
                preview = missing_required[:20].tolist()
                raise ValueError(
                    f"Missing {missing_required.size} required cached windows for "
                    f"{source_name}/{branch}; first={preview}."
                )
            output_dir = args.output_root / source_name / branch
            atomic_npy(output_dir / "rank_map.npy", rank_map)
            atomic_npy(output_dir / "row_map.npy", row_map)
            mapped = int(np.count_nonzero(rank_map >= 0))
            branch_report = {
                "old_cached_rows": old_cached_rows,
                "mapped_current_windows": mapped,
                "removed_cached_rows": removed_cached_rows,
                "required_current_windows": int(required.size),
                "missing_required_windows": int(missing_required.size),
                "missing_required_preview": missing_required[:20].tolist(),
            }
            source_report["branches"][branch] = branch_report
            print(f"[latent-remap] PASS {source_name}/{branch}: {branch_report}", flush=True)
        report["sources"][source_name] = source_report

    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "remap_report.json"
    temporary = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, report_path)
    print(f"[latent-remap] report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
