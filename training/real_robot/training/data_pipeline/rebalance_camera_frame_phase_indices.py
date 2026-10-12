#!/usr/bin/env python3
"""Restore the July-30 phase sampling contract after failed-episode filtering."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
INDEX_ROOT = ROOT / "data_indices_camera_frame_h32"
SOURCES = (
    "agx_cup_tray_mobile_rot6d_vlash_se2_h48_phase_split_v3",
    "agx_move_white_box_mobile_rot6d_vlash_se2_h48_phase_split_v2",
)
BRANCH_TARGETS = {"manip": 0.60, "nav": 0.40}
HOLD_KINDS = {"manip": "manip_hold_in_nav", "nav": "nav_hold_in_manip"}
HOLD_TARGETS = {"manip": 0.15, "nav": 0.20}
BACKUP_SUFFIX = ".pre_rebalance_20260821.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def group_scale(weights: np.ndarray, mask: np.ndarray, target_mass: float) -> None:
    mass = float(weights[mask].sum())
    if mass <= 0.0:
        raise ValueError("Cannot assign positive target mass to an empty sampling group.")
    weights[mask] *= target_mass / mass


def summarize(branches: np.ndarray, kinds: np.ndarray, weights: np.ndarray) -> dict:
    total = float(weights.sum())
    result = {"total_mass": total, "branches": {}}
    for branch in ("manip", "nav"):
        branch_mask = branches == branch
        branch_mass = float(weights[branch_mask].sum())
        hold_mask = branch_mask & (kinds == HOLD_KINDS[branch])
        result["branches"][branch] = {
            "mass_fraction": branch_mass / total,
            "hold_mass_fraction_within_branch": float(weights[hold_mask].sum()) / branch_mass,
            "rows": int(np.count_nonzero(branch_mask)),
            "hold_rows": int(np.count_nonzero(hold_mask)),
        }
    return result


def rebalance(path: Path) -> dict:
    table = pq.read_table(path)
    branches = np.asarray(table["branch"].to_pylist(), dtype=object)
    kinds = np.asarray(table["sample_kind"].to_pylist(), dtype=object)
    original = np.asarray(table["sampling_weight"].to_numpy(), dtype=np.float64)
    if not np.isfinite(original).all() or np.any(original <= 0.0):
        raise ValueError(f"Invalid source weights: {path}")
    weights = original.copy()
    before = summarize(branches, kinds, weights)

    # Preserve all relative event weights inside hold/primary groups. Only their
    # aggregate masses and the aggregate manip/nav masses are restored.
    for branch in ("manip", "nav"):
        branch_mask = branches == branch
        hold_mask = branch_mask & (kinds == HOLD_KINDS[branch])
        primary_mask = branch_mask & ~hold_mask
        group_scale(weights, hold_mask, HOLD_TARGETS[branch])
        group_scale(weights, primary_mask, 1.0 - HOLD_TARGETS[branch])
        group_scale(weights, branch_mask, BRANCH_TARGETS[branch])

    weights /= weights.sum()
    after = summarize(branches, kinds, weights)
    for branch in ("manip", "nav"):
        actual_branch = after["branches"][branch]["mass_fraction"]
        actual_hold = after["branches"][branch]["hold_mass_fraction_within_branch"]
        if not np.isclose(actual_branch, BRANCH_TARGETS[branch], atol=1e-10):
            raise AssertionError((branch, actual_branch))
        if not np.isclose(actual_hold, HOLD_TARGETS[branch], atol=1e-10):
            raise AssertionError((branch, actual_hold))

    backup = path.with_name(path.stem + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
    column = table.schema.get_field_index("sampling_weight")
    output = table.set_column(column, "sampling_weight", pa.array(weights.astype(np.float64)))
    pq.write_table(output, path, compression="zstd")
    return {
        "path": str(path),
        "backup": str(backup),
        "before_sha256": sha256(backup),
        "after_sha256": sha256(path),
        "before": before,
        "after": after,
    }


def main() -> int:
    report = {
        "status": "PASS",
        "contract": {
            "manip_mass": 0.60,
            "nav_mass": 0.40,
            "manip_hold_mass_within_manip": 0.15,
            "nav_hold_mass_within_nav": 0.20,
        },
        "sources": [],
    }
    for source in SOURCES:
        report["sources"].append(
            rebalance(INDEX_ROOT / source / "phase_branch_index_h32.parquet")
        )
    output = ROOT / "audits" / "camera_frame_phase_rebalance_20260821.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
