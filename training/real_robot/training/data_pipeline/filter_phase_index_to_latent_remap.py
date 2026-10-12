#!/usr/bin/env python3
"""Filter routed rows to cached latent coverage and restore branch mass."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-index", type=Path, required=True)
    parser.add_argument("--remap-root", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-nav-mass", type=float, default=0.4)
    args = parser.parse_args()

    table = pq.read_table(args.phase_index)
    branches = np.asarray(table["branch"].to_pylist(), dtype=object)
    source_indices = np.asarray(table["source_window_index"].to_numpy(), dtype=np.int64)
    weights = np.asarray(table["sampling_weight"].to_numpy(), dtype=np.float64)
    keep = np.zeros(table.num_rows, dtype=bool)
    missing_by_branch: dict[str, int] = {}
    for branch in ("manip", "nav"):
        rank_map = np.load(
            args.remap_root / args.source / branch / "rank_map.npy", mmap_mode="r"
        )
        branch_rows = branches == branch
        covered = rank_map[source_indices[branch_rows]] >= 0
        keep[branch_rows] = covered
        missing_by_branch[branch] = int(np.count_nonzero(~covered))

    filtered = table.filter(pa.array(keep))
    filtered_branches = np.asarray(filtered["branch"].to_pylist(), dtype=object)
    filtered_weights = np.asarray(filtered["sampling_weight"].to_numpy(), dtype=np.float64)
    nav = filtered_branches == "nav"
    manip = filtered_branches == "manip"
    nav_mass_before = float(filtered_weights[nav].sum())
    manip_mass = float(filtered_weights[manip].sum())
    if nav_mass_before <= 0.0 or manip_mass <= 0.0:
        raise ValueError("Both nav and manipulation branches must retain positive mass.")
    nav_scale = (args.target_nav_mass / (1.0 - args.target_nav_mass)) * manip_mass / nav_mass_before
    filtered_weights[nav] *= nav_scale
    filtered = filtered.set_column(
        filtered.schema.get_field_index("sampling_weight"),
        "sampling_weight",
        pa.array(filtered_weights),
    )

    actual_nav_mass = float(filtered_weights[nav].sum() / filtered_weights.sum())
    if abs(actual_nav_mass - args.target_nav_mass) > 1e-12:
        raise AssertionError(f"Failed to restore nav mass: {actual_nav_mass}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "phase_branch_index.parquet"
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    pq.write_table(filtered, temporary, compression="zstd")
    os.replace(temporary, output)
    report = {
        "status": "PASS",
        "source_phase_index": str(args.phase_index),
        "latent_remap_root": str(args.remap_root),
        "input_rows": table.num_rows,
        "output_rows": filtered.num_rows,
        "dropped_rows": int(table.num_rows - filtered.num_rows),
        "missing_rows_by_branch": missing_by_branch,
        "target_nav_mass": args.target_nav_mass,
        "actual_nav_mass": actual_nav_mass,
        "nav_weight_scale_after_filter": nav_scale,
    }
    (args.output_dir / "latent_coverage_filter_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
