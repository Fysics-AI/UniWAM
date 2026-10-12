#!/usr/bin/env python3
"""Build complete H32 window tables directly from episode lengths."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def atomic_parquet(table: pa.Table, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, output)


def build_source(source_name: str, nav_available: bool, output_root: Path, horizon: int, data_root: Path) -> None:
    dataset_root = data_root / source_name
    episode_paths = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode parquet files under {dataset_root}")

    columns: dict[str, list] = {
        "episode_index": [],
        "source_episode_index": [],
        "start_frame": [],
        "horizon": [],
        "sample_type": [],
        "nav_loss_valid": [],
        "manip_loss_valid": [],
        "nav_active_count": [],
        "manip_active_count": [],
    }
    for path in episode_paths:
        metadata = pq.read_metadata(path)
        length = int(metadata.num_rows)
        if length <= horizon:
            continue
        identity = pq.read_table(
            path, columns=["episode_index", "source_episode_index"]
        ).slice(0, 1)
        episode_index = int(identity["episode_index"][0].as_py())
        source_episode_index = int(identity["source_episode_index"][0].as_py())
        count = length - horizon
        starts = range(count)
        columns["episode_index"].extend([episode_index] * count)
        columns["source_episode_index"].extend([source_episode_index] * count)
        columns["start_frame"].extend(starts)
        columns["horizon"].extend([horizon] * count)
        columns["sample_type"].extend(["unrouted"] * count)
        columns["nav_loss_valid"].extend([nav_available] * count)
        columns["manip_loss_valid"].extend([True] * count)
        columns["nav_active_count"].extend([0] * count)
        columns["manip_active_count"].extend([0] * count)

    table = pa.table(
        {
            "episode_index": pa.array(columns["episode_index"], type=pa.int64()),
            "source_episode_index": pa.array(columns["source_episode_index"], type=pa.int64()),
            "start_frame": pa.array(columns["start_frame"], type=pa.int64()),
            "horizon": pa.array(columns["horizon"], type=pa.int64()),
            "sample_type": pa.array(columns["sample_type"], type=pa.string()),
            "nav_loss_valid": pa.array(columns["nav_loss_valid"], type=pa.bool_()),
            "manip_loss_valid": pa.array(columns["manip_loss_valid"], type=pa.bool_()),
            "nav_active_count": pa.array(columns["nav_active_count"], type=pa.int64()),
            "manip_active_count": pa.array(columns["manip_active_count"], type=pa.int64()),
        }
    )
    output = output_root / source_name / "window_index_h32.parquet"
    atomic_parquet(table, output)

    old_rows = pq.read_metadata(dataset_root / "meta/window_index.parquet").num_rows
    expected_gain = 16 * len(episode_paths)
    actual_gain = table.num_rows - old_rows
    if actual_gain != expected_gain:
        raise AssertionError(
            f"{source_name}: expected H32-H48 gain {expected_gain}, got {actual_gain}."
        )
    print(
        f"[true-h32] PASS {source_name}: episodes={len(episode_paths)} "
        f"windows={table.num_rows} h48_windows={old_rows} gain={actual_gain}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="Directory containing the source LeRobot dataset directories.",
    )
    parser.add_argument(
        "--source", action="append", required=True, metavar="NAME:nav|manip",
        help="Dataset directory and whether navigation is available; repeat for each source.",
    )
    parser.add_argument("--horizon", type=int, default=32)
    args = parser.parse_args()
    if args.horizon != 32:
        raise ValueError(f"This contract requires horizon=32, got {args.horizon}.")
    for item in args.source:
        source_name, separator, mode = item.rpartition(":")
        if not separator or not source_name or mode not in {"nav", "manip"}:
            parser.error(f"Invalid --source {item!r}; expected NAME:nav or NAME:manip")
        nav_available = mode == "nav"
        build_source(source_name, nav_available, args.output_root, args.horizon, args.data_root)


if __name__ == "__main__":
    main()
