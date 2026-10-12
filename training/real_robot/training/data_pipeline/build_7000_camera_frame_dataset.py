#!/usr/bin/env python3
"""Materialize the validated 7000 front-30 camera-frame dataset.

The source episodes and videos are preserved.  Only the parquet state/action
columns and filtered H32 window metadata are materialized.  Validation rows
are already camera-frame poses in xyz+RPY; this script converts RPY to the
training row-major Rot6D convention and appends the original gripper values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.moveaxis(np.asarray(rpy, dtype=np.float64), -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.stack(
        (
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        ), axis=-1,
    ).reshape((-1, 3, 3))


def cam6_to_rot6d(cam6: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    cam6 = np.asarray(cam6, dtype=np.float64)
    if cam6.ndim != 2 or cam6.shape[1] != 6:
        raise ValueError(f"Expected camera pose [T,6] xyz+RPY, got {cam6.shape}")
    matrix = rpy_to_matrix(cam6[:, 3:6])
    rot6d = matrix[:, :2, :].reshape((-1, 6))
    return np.concatenate((cam6[:, :3], rot6d, np.asarray(gripper, dtype=np.float64).reshape((-1, 1))), axis=1).astype(np.float32)


def add_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    return table.append_column(name, pa.array(values.tolist(), type=pa.list_(pa.float32(), list_size=20)))


def update_info_features(info_path: Path) -> None:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.setdefault("features", {})
    features["observation.state.camera_dual_arm"] = {
        "dtype": "float32",
        "shape": [20],
        "names": [[
            "left_x", "left_y", "left_z", "left_rot6d_r0c0", "left_rot6d_r0c1",
            "left_rot6d_r0c2", "left_rot6d_r1c0", "left_rot6d_r1c1", "left_rot6d_r1c2",
            "left_gripper", "right_x", "right_y", "right_z", "right_rot6d_r0c0",
            "right_rot6d_r0c1", "right_rot6d_r0c2", "right_rot6d_r1c0", "right_rot6d_r1c1",
            "right_rot6d_r1c2", "right_gripper",
        ]],
    }
    features["action.manip.camera_dual_arm"] = {
        "dtype": "float32",
        "shape": [20],
        "names": [[
            "left_x_target", "left_y_target", "left_z_target", "left_rot6d_r0c0_target",
            "left_rot6d_r0c1_target", "left_rot6d_r0c2_target", "left_rot6d_r1c0_target",
            "left_rot6d_r1c1_target", "left_rot6d_r1c2_target", "left_gripper_target",
            "right_x_target", "right_y_target", "right_z_target", "right_rot6d_r0c0_target",
            "right_rot6d_r0c1_target", "right_rot6d_r0c2_target", "right_rot6d_r1c0_target",
            "right_rot6d_r1c1_target", "right_rot6d_r1c2_target", "right_gripper_target",
        ]],
    }
    info["fastwam_camera_frame_contract"] = {
        "state_action_source": "09_validation/lerobot_data/episode_XXXXXX.parquet",
        "pose_format": "xyz_plus_rpy_converted_to_row_major_rot6d",
        "extra_transform": "none",
        "selected_episode_policy": "front30_joined_validation_and_prompt_success",
    }
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--force-simple", type=Path, required=True)
    ap.add_argument("--window-index", type=Path, required=True)
    ap.add_argument("--window-weights", type=Path, required=True)
    ap.add_argument("--latent-root", type=Path, required=True)
    ap.add_argument("--latent-source-name", required=True)
    ap.add_argument("--remap-root", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--episode-start", type=int, default=0)
    ap.add_argument("--episode-stop", type=int, default=None)
    ap.add_argument("--skip-copy", action="store_true")
    ap.add_argument("--skip-finalize", action="store_true")
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--copy-only", action="store_true")
    args = ap.parse_args()

    manifest = [json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    eligible = {int(x["episode_index"]): x for x in manifest if x["eligible"]}
    if not eligible:
        raise RuntimeError("No eligible 7000 episodes.")
    out = args.output_root
    if args.episode_start < 0 or (args.episode_stop is not None and args.episode_stop <= args.episode_start):
        raise ValueError("Invalid episode range")
    if args.finalize_only:
        args.skip_copy = True
        args.skip_finalize = False
    if args.copy_only:
        args.skip_copy = False
        args.skip_finalize = True
    if out.exists() and args.overwrite and not args.skip_copy and args.episode_start == 0:
        for name in ("data", "meta", "videos"):
            path = out / name
            if path.is_symlink() or path.is_file(): path.unlink()
            elif path.exists(): shutil.rmtree(path)
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_copy:
        shutil.copytree(args.source_root / "meta", out / "meta", dirs_exist_ok=True)
        update_info_features(out / "meta/info.json")
        video_link = out / "videos"
        if not video_link.exists(): video_link.symlink_to(args.source_root / "videos", target_is_directory=True)

    force_rows = [json.loads(x) for x in args.force_simple.read_text().splitlines() if x.strip()]
    force_keys = {(int(x["source_dataset"]), int(x["source_episode_index"])) for x in force_rows if x.get("force_simple") is True}
    force_current = [
        {"episode_index": int(row["episode_index"]), "force_simple": True}
        for row in manifest
        if row["eligible"] and (int(row["source_dataset_index"]), int(row["source_episode_index"])) in force_keys
    ]
    if not args.skip_copy:
        (out / "meta/force_simple_episode_indices.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in force_current), encoding="utf-8"
        )

    if args.copy_only:
        print(json.dumps({"status": "COPIED", "output": str(out), "force_simple": len(force_current)}))
        return 0

    # Keep the original episode numbering and video paths.  Rewriting all
    # parquet files gives the loader one uniform schema; ineligible rows are
    # never referenced by the filtered H32 index.
    output_data = out / "data"
    episode_stop = len(manifest) if args.episode_stop is None else min(int(args.episode_stop), len(manifest))
    for src in sorted((args.source_root / "data").glob("chunk-*/*.parquet")):
        episode = int(src.stem.split("_")[-1])
        if args.finalize_only or episode < args.episode_start or episode >= episode_stop:
            continue
        table = pq.read_table(src)
        row = eligible.get(episode)
        if row is not None:
            valid = pq.read_table(Path(row["validation_parquet"]))
            if valid.num_rows != table.num_rows:
                raise ValueError(f"Frame mismatch episode={episode}: source={table.num_rows} validation={valid.num_rows}")
            state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            action = np.asarray(table["action.manip"].to_pylist(), dtype=np.float32)
            left_state = cam6_to_rot6d(np.asarray(valid["observation.state.cam_left"].to_pylist()), np.asarray(valid["observation.state.gripper_position"].to_pylist())[:, 0])
            right_state = cam6_to_rot6d(np.asarray(valid["observation.state.cam_right"].to_pylist()), np.asarray(valid["observation.state.gripper_position"].to_pylist())[:, 1])
            left_action = cam6_to_rot6d(np.asarray(valid["action.cam_left"].to_pylist()), np.asarray(valid["action.gripper_position"].to_pylist())[:, 0])
            right_action = cam6_to_rot6d(np.asarray(valid["action.cam_right"].to_pylist()), np.asarray(valid["action.gripper_position"].to_pylist())[:, 1])
            state_cam = np.concatenate((left_state, right_state), axis=1)
            action_cam = np.concatenate((left_action, right_action), axis=1)
            if not np.isfinite(state_cam).all() or not np.isfinite(action_cam).all():
                raise ValueError(f"Non-finite camera state/action episode={episode}")
        else:
            # Never sampled; maintain a valid uniform schema only.
            state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            action = np.asarray(table["action.manip"].to_pylist(), dtype=np.float32)
            state_cam, action_cam = state.copy(), action.copy()
        table = add_column(table, "observation.state.camera_dual_arm", state_cam)
        table = add_column(table, "action.manip.camera_dual_arm", action_cam)
        target = output_data / src.relative_to(args.source_root / "data")
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target, compression="zstd")

    if args.skip_finalize:
        print(json.dumps({"status": "PARTIAL", "episode_start": args.episode_start, "episode_stop": episode_stop}))
        return 0

    if args.episode_start != 0 or episode_stop != len(manifest):
        raise ValueError("Finalization requires the complete episode range")

    # Filter the already validated front-30 window and task-balanced weights.
    window = pq.read_table(args.window_index)
    keep = np.isin(np.asarray(window["episode_index"].to_numpy()), np.asarray(sorted(eligible)))
    filtered = window.filter(pa.array(keep))
    pq.write_table(filtered, out / "meta/window_index_h32_stride4_filtered.parquet", compression="zstd")
    weights = pq.read_table(args.window_weights)
    w_keep = np.isin(np.asarray(weights["episode_index"].to_numpy()), np.asarray(sorted(eligible)))
    filtered_weights = weights.filter(pa.array(w_keep))
    if "window_index" in filtered_weights.column_names:
        window_index_column = filtered_weights.schema.get_field_index("window_index")
        filtered_weights = filtered_weights.set_column(
            window_index_column,
            "window_index",
            pa.array(np.arange(filtered_weights.num_rows, dtype=np.int64), type=pa.int64()),
        )
    pq.write_table(filtered_weights, out / "meta/window_sampling_weights_task_balanced_filtered.parquet", compression="zstd")

    # Build a cache remap: filtered row i maps to its original source window row.
    old_source_indices = np.asarray(filtered["episode_index"].to_numpy()) * 0  # overwritten below
    # Existing window_index row order is the cache's source index.  The filtered
    # table retains original row order, so use the original row positions.
    source_row_positions = np.flatnonzero(keep).astype(np.int64)
    cache_manifest = json.loads((args.latent_root / "manifest.json").read_text())
    source = next(s for s in cache_manifest["sources"] if s["name"] == args.latent_source_name)
    world = int(cache_manifest["world_size"])
    cache_dir = args.latent_root / args.latent_source_name / "manip"
    rank_map = np.full(len(source_row_positions), -1, dtype=np.int16)
    row_map = np.full(len(source_row_positions), -1, dtype=np.int32)
    old_to_rank = {}
    for rank in range(world):
        indices = np.load(cache_dir / f"rank_{rank:02d}_source_indices.npy")
        for row, source_idx in enumerate(indices.tolist()): old_to_rank[int(source_idx)] = (rank, row)
    for i, old in enumerate(source_row_positions.tolist()):
        if old in old_to_rank: rank_map[i], row_map[i] = old_to_rank[old]
    remap_dir = args.remap_root / args.latent_source_name / "manip"
    remap_dir.mkdir(parents=True, exist_ok=True)
    np.save(remap_dir / "rank_map.npy", rank_map)
    np.save(remap_dir / "row_map.npy", row_map)
    report = {
        "status": "PASS" if np.all(rank_map >= 0) else "FAIL",
        "sources": {args.latent_source_name: {
            "old_window_index_sha256": source.get("window_index_sha256"),
            "current_window_index_sha256": sha256(out / "meta/window_index_h32_stride4_filtered.parquet"),
            "branches": {"manip": int(np.count_nonzero(rank_map >= 0))},
        }}
    }
    args.remap_root.mkdir(parents=True, exist_ok=True)
    (args.remap_root / "remap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"eligible_episodes": len(eligible), "filtered_windows": filtered.num_rows, "cache_remap": report}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
