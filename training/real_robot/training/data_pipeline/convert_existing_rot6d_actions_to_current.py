#!/usr/bin/env python3
"""Make converted Rot6D datasets use action[t] = state[t] without touching videos."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def hardlink_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite {dst}")
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)


def current_action(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    if state.ndim != 2 or state.shape[1] != 23 or action.shape != (state.shape[0], 20):
        raise ValueError(f"Expected state [N,23], action [N,20], got {state.shape}/{action.shape}")
    expected = np.concatenate((state[:, 3:13], state[:, 13:23]), axis=1).astype(np.float32)
    # This conversion intentionally makes the output independent of whether the
    # input action was current-frame or next-frame aligned.
    return expected


def main() -> int:
    cfg = parse_args()
    src, dst = cfg.input_root.resolve(), cfg.output_root.resolve()
    if not src.is_dir():
        raise FileNotFoundError(src)
    parquet_paths = sorted((src / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No episode parquet files under {src / 'data'}")
    records = []
    changed = 0
    max_error = 0.0
    if not cfg.dry_run:
        hardlink_tree(src, dst)
    for path in parquet_paths:
        table = pq.read_table(path)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action.manip"].to_pylist(), dtype=np.float32)
        new_action = current_action(state, action)
        delta = float(np.abs(new_action - action).max(initial=0.0))
        max_error = max(max_error, delta)
        changed += int(delta > 0.0)
        if not cfg.dry_run:
            action_column = pa.FixedSizeListArray.from_arrays(
                pa.array(new_action.reshape(-1), type=pa.float32()), 20
            )
            columns = [action_column if name == "action.manip" else table[name] for name in table.column_names]
            pq.write_table(
                pa.Table.from_arrays(columns, schema=table.schema),
                dst / path.relative_to(src),
                compression="zstd",
            )
        records.append({"input": str(path), "rows": int(table.num_rows), "max_abs_change": delta})
    manifest = {
        "status": "PASS",
        "dry_run": bool(cfg.dry_run),
        "input_root": str(src),
        "output_root": str(dst),
        "action_alignment": "current",
        "contract": "action.manip[t] == concat(observation.state[t,3:13], observation.state[t,13:23])",
        "episodes": len(records),
        "changed_episodes": changed,
        "max_abs_change": max_error,
        "records": records,
    }
    if not cfg.dry_run:
        meta = dst / "meta"
        (meta / "action_alignment.json").write_text(json.dumps({
            "status": "PASS", "action_alignment": "current", "state_frame": "t",
            "action_frame": "t", "source_dataset": str(src)
        }, indent=2) + "\n")
        (dst / "conversion" / "action_alignment_conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
