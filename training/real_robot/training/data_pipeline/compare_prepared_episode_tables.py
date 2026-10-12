#!/usr/bin/env python3
"""Compare prepared LeRobot episode columns against a parent-run snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_COLUMNS = (
    "observation.state.camera_dual_arm",
    "action.manip.camera_dual_arm",
)


def episodes(root: Path) -> dict[Path, Path]:
    data_root = root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing episode directory: {data_root}")
    files = {path.relative_to(data_root): path for path in data_root.glob("chunk-*/episode_*.parquet")}
    if not files:
        raise ValueError(f"No episode Parquet files under {data_root}")
    return files


def compare(candidate: Path, reference: Path, columns: tuple[str, ...], atol: float = 0.0) -> dict:
    if atol < 0 or not np.isfinite(atol):
        raise ValueError("atol must be finite and non-negative")
    candidate_files = episodes(candidate)
    reference_files = episodes(reference)
    missing = sorted(str(path) for path in reference_files.keys() - candidate_files.keys())
    extra = sorted(str(path) for path in candidate_files.keys() - reference_files.keys())
    differing: list[dict] = []
    matched = 0
    tolerance_matched = 0
    max_abs_error = 0.0
    for relative in sorted(candidate_files.keys() & reference_files.keys()):
        candidate_path = candidate_files[relative]
        reference_path = reference_files[relative]
        candidate_schema = pq.read_schema(candidate_path)
        reference_schema = pq.read_schema(reference_path)
        absent = [name for name in columns if name not in candidate_schema.names or name not in reference_schema.names]
        if absent:
            differing.append({"episode": str(relative), "missing_columns": absent})
            continue
        current = pq.read_table(candidate_path, columns=list(columns))
        parent = pq.read_table(reference_path, columns=list(columns))
        if current.equals(parent, check_metadata=True):
            matched += 1
            continue
        if atol > 0 and current.num_rows == parent.num_rows:
            errors = []
            for name in columns:
                left = np.asarray(current[name].to_pylist(), dtype=np.float64)
                right = np.asarray(parent[name].to_pylist(), dtype=np.float64)
                if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
                    break
                errors.append(float(np.max(np.abs(left - right))))
            if len(errors) == len(columns):
                max_abs_error = max(max_abs_error, *errors)
                if max(errors) <= atol:
                    tolerance_matched += 1
                    continue
        changed = [name for name in columns if not current[name].equals(parent[name])]
        differing.append({
            "episode": str(relative),
            "candidate_rows": current.num_rows,
            "reference_rows": parent.num_rows,
            "changed_columns": changed,
        })
    return {
        "status": "EPISODE_VALUES_MATCH" if not (missing or extra or differing) else "EPISODE_VALUES_MISMATCH",
        "candidate_root": str(candidate),
        "reference_root": str(reference),
        "columns": list(columns),
        "candidate_episode_count": len(candidate_files),
        "reference_episode_count": len(reference_files),
        "matched_episode_count": matched,
        "tolerance_matched_episode_count": tolerance_matched,
        "absolute_tolerance": atol,
        "max_abs_error_among_numeric_comparisons": max_abs_error,
        "missing_episodes": missing,
        "extra_episodes": extra,
        "differing_episodes": differing,
        "video_contents_verified": False,
        "latent_contents_verified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--column", action="append", dest="columns")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.candidate_root, args.reference_root, tuple(args.columns or DEFAULT_COLUMNS), args.atol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("differing_episodes", "missing_episodes", "extra_episodes")}, indent=2))
    print(f"mismatches={len(result['differing_episodes'])} missing={len(result['missing_episodes'])} extra={len(result['extra_episodes'])}")
    return int(result["status"] != "EPISODE_VALUES_MATCH")


if __name__ == "__main__":
    raise SystemExit(main())
