#!/usr/bin/env python3
"""Create immutable 6D-rotation derivatives of cleaned quaternion LeRobot data.

The source datasets already contain the cleaned, trimmed camera streams.  This
converter validates every quaternion again, rewrites only parquet state/action
columns, and hard-links the immutable video files.  It never opens or mutates
the source dataset.  Runtime relative actions are later built as
``T_window_start^-1 @ T_target`` by the dataset loader.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from uniwam.datasets.lerobot.rot6d import (
    ROT6D_CONVENTION,
    matrix_to_rotation_6d_np,
    quaternion_xyzw_to_matrix_np,
    relative_pose_rot6d_np,
    rotation_6d_to_matrix_np,
    rotation_geodesic_angle_np,
)


STATE_ROT6D_NAMES = [
    "base_vx", "base_vy", "base_wz",
    "left_x", "left_y", "left_z",
    "left_rot6d_r0c0", "left_rot6d_r0c1", "left_rot6d_r0c2",
    "left_rot6d_r1c0", "left_rot6d_r1c1", "left_rot6d_r1c2",
    "left_gripper",
    "right_x", "right_y", "right_z",
    "right_rot6d_r0c0", "right_rot6d_r0c1", "right_rot6d_r0c2",
    "right_rot6d_r1c0", "right_rot6d_r1c1", "right_rot6d_r1c2",
    "right_gripper",
]
MANIP_ROT6D_NAMES = [
    "left_x_target", "left_y_target", "left_z_target",
    "left_rot6d_r0c0_target", "left_rot6d_r0c1_target", "left_rot6d_r0c2_target",
    "left_rot6d_r1c0_target", "left_rot6d_r1c1_target", "left_rot6d_r1c2_target",
    "left_gripper_target",
    "right_x_target", "right_y_target", "right_z_target",
    "right_rot6d_r0c0_target", "right_rot6d_r0c1_target", "right_rot6d_r0c2_target",
    "right_rot6d_r1c0_target", "right_rot6d_r1c1_target", "right_rot6d_r1c2_target",
    "right_gripper_target",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def fixed_list(values: np.ndarray) -> pa.FixedSizeListArray:
    array = np.ascontiguousarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(array.reshape(-1)), array.shape[1])


def column_to_numpy(table: pa.Table, name: str) -> np.ndarray:
    if name not in table.column_names:
        raise KeyError(f"Missing required parquet column {name!r}.")
    values = np.asarray(table[name].to_pylist(), dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{name} must decode to [T,D], got {values.shape}.")
    return values


def field_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": values.min(axis=0).astype(np.float32).tolist(),
        "max": values.max(axis=0).astype(np.float32).tolist(),
        "mean": values.mean(axis=0).astype(np.float32).tolist(),
        "std": values.std(axis=0).astype(np.float32).tolist(),
        "count": [int(values.shape[0])],
    }


def max_quaternion_norm_error(values: np.ndarray) -> float:
    return float(np.abs(np.linalg.norm(values.astype(np.float64), axis=-1) - 1.0).max(initial=0.0))


def convert_state(state: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    if state.ndim != 2 or state.shape[1] != 19:
        raise ValueError(f"Expected quaternion state [T,19], got {state.shape}.")
    if not np.isfinite(state).all():
        raise ValueError("State contains non-finite values.")
    left_matrix = quaternion_xyzw_to_matrix_np(state[:, 6:10])
    right_matrix = quaternion_xyzw_to_matrix_np(state[:, 14:18])
    result = np.empty((state.shape[0], 23), dtype=np.float32)
    result[:, :3] = state[:, :3]
    result[:, 3:6] = state[:, 3:6]
    result[:, 6:12] = matrix_to_rotation_6d_np(left_matrix)
    result[:, 12] = state[:, 10]
    result[:, 13:16] = state[:, 11:14]
    result[:, 16:22] = matrix_to_rotation_6d_np(right_matrix)
    result[:, 22] = state[:, 18]
    left_roundtrip = rotation_6d_to_matrix_np(result[:, 6:12])
    right_roundtrip = rotation_6d_to_matrix_np(result[:, 16:22])
    return result, {
        "max_input_quaternion_norm_error": max(
            max_quaternion_norm_error(state[:, 6:10]),
            max_quaternion_norm_error(state[:, 14:18]),
        ),
        "max_state_rotation_roundtrip_rad": float(
            max(
                rotation_geodesic_angle_np(left_matrix, left_roundtrip).max(initial=0.0),
                rotation_geodesic_angle_np(right_matrix, right_roundtrip).max(initial=0.0),
            )
        ),
    }


def convert_manip_action(action: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    if action.ndim != 2 or action.shape[1] != 16:
        raise ValueError(f"Expected quaternion action.manip [T,16], got {action.shape}.")
    if not np.isfinite(action).all():
        raise ValueError("action.manip contains non-finite values.")
    left_matrix = quaternion_xyzw_to_matrix_np(action[:, 3:7])
    right_matrix = quaternion_xyzw_to_matrix_np(action[:, 11:15])
    result = np.empty((action.shape[0], 20), dtype=np.float32)
    result[:, 0:3] = action[:, 0:3]
    result[:, 3:9] = matrix_to_rotation_6d_np(left_matrix)
    result[:, 9] = action[:, 7]
    result[:, 10:13] = action[:, 8:11]
    result[:, 13:19] = matrix_to_rotation_6d_np(right_matrix)
    result[:, 19] = action[:, 15]
    left_roundtrip = rotation_6d_to_matrix_np(result[:, 3:9])
    right_roundtrip = rotation_6d_to_matrix_np(result[:, 13:19])
    return result, {
        "max_input_quaternion_norm_error": max(
            max_quaternion_norm_error(action[:, 3:7]),
            max_quaternion_norm_error(action[:, 11:15]),
        ),
        "max_action_rotation_roundtrip_rad": float(
            max(
                rotation_geodesic_angle_np(left_matrix, left_roundtrip).max(initial=0.0),
                rotation_geodesic_angle_np(right_matrix, right_roundtrip).max(initial=0.0),
            )
        ),
    }


def validate_anchor_se3(state: np.ndarray, action: np.ndarray) -> dict[str, float]:
    """Prove the loader's local-frame action can reconstruct the absolute target."""
    count = min(64, state.shape[0])
    if count == 0:
        return {"max_anchor_se3_position_error_m": 0.0, "max_anchor_se3_rotation_error_rad": 0.0}
    state = state[:count]
    action = action[:count]
    left_xyz, left_rot = relative_pose_rot6d_np(
        state[:, 3:6], state[:, 6:12], action[:, 0:3], action[:, 3:9]
    )
    right_xyz, right_rot = relative_pose_rot6d_np(
        state[:, 13:16], state[:, 16:22], action[:, 10:13], action[:, 13:19]
    )
    left_current = rotation_6d_to_matrix_np(state[:, 6:12])
    right_current = rotation_6d_to_matrix_np(state[:, 16:22])
    left_rebuilt_xyz = state[:, 3:6] + np.matmul(left_current, left_xyz[..., None])[..., 0]
    right_rebuilt_xyz = state[:, 13:16] + np.matmul(right_current, right_xyz[..., None])[..., 0]
    left_rebuilt_rotation = left_current @ rotation_6d_to_matrix_np(left_rot)
    right_rebuilt_rotation = right_current @ rotation_6d_to_matrix_np(right_rot)
    return {
        "max_anchor_se3_position_error_m": float(
            max(
                np.linalg.norm(left_rebuilt_xyz - action[:, 0:3], axis=-1).max(initial=0.0),
                np.linalg.norm(right_rebuilt_xyz - action[:, 10:13], axis=-1).max(initial=0.0),
            )
        ),
        "max_anchor_se3_rotation_error_rad": float(
            max(
                rotation_geodesic_angle_np(left_rebuilt_rotation, rotation_6d_to_matrix_np(action[:, 3:9])).max(initial=0.0),
                rotation_geodesic_angle_np(right_rebuilt_rotation, rotation_6d_to_matrix_np(action[:, 13:19])).max(initial=0.0),
            )
        ),
    }


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.column_names.index(name)
    return table.set_column(index, name, fixed_list(values))


def link_tree(source: Path, destination: Path) -> int:
    if not source.is_dir():
        raise FileNotFoundError(source)
    count = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file():
            raise ValueError(f"Unsupported non-file video entry: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
        count += 1
    return count


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def update_info(source_info: Path, destination_info: Path) -> None:
    with source_info.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    features = info["features"]
    features["observation.state"] = {
        **features["observation.state"],
        "shape": [23],
        "names": [STATE_ROT6D_NAMES],
    }
    features["action.manip"] = {
        **features["action.manip"],
        "shape": [20],
        "names": [MANIP_ROT6D_NAMES],
    }
    info["robot_type"] = f"{info.get('robot_type', 'agilex')}_rot6d_anchor_se3"
    info["fastwam_rotation_contract"] = {
        "state_rotation": ROT6D_CONVENTION,
        "stored_action_rotation": ROT6D_CONVENTION,
        "runtime_relative_action": "T_window_start^-1 @ T_target",
        "quaternion_input_order": "xyzw",
    }
    write_json(destination_info, info)


def copy_meta_without_phase_indices(source: Path, destination: Path) -> None:
    for name in ("episodes.jsonl", "tasks.jsonl", "window_index.parquet", "window_sampling_weights.parquet"):
        path = source / "meta" / name
        if path.is_file():
            target = destination / "meta" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def convert_dataset(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(source / "meta" / "info.json")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite destination: {output}")
    staging = output.with_name(f"{output.name}.building-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")

    staging.mkdir(parents=True)
    episode_audit: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    total_frames = 0
    try:
        for parquet_path in sorted((source / "data").rglob("*.parquet")):
            table = pq.read_table(parquet_path)
            state, state_report = convert_state(column_to_numpy(table, "observation.state"))
            action, action_report = convert_manip_action(column_to_numpy(table, "action.manip"))
            if state.shape[0] != action.shape[0]:
                raise ValueError(f"State/action length mismatch in {parquet_path}")
            table = replace_column(table, "observation.state", state)
            table = replace_column(table, "action.manip", action)
            target = staging / parquet_path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target, compression="zstd")
            episode_index = int(parquet_path.stem.split("_")[-1])
            se3_report = validate_anchor_se3(state, action)
            episode_audit.append(
                {
                    "episode_index": episode_index,
                    "frames": int(state.shape[0]),
                    **state_report,
                    **action_report,
                    **se3_report,
                }
            )
            stats = {"observation.state": field_stats(state), "action.manip": field_stats(action)}
            if "action.nav" in table.column_names:
                stats["action.nav"] = field_stats(column_to_numpy(table, "action.nav"))
            episode_stats.append({"episode_index": episode_index, "stats": stats})
            total_frames += int(state.shape[0])

        copy_meta_without_phase_indices(source, staging)
        update_info(source / "meta" / "info.json", staging / "meta" / "info.json")
        with (staging / "meta" / "episodes_stats.jsonl").open("w", encoding="utf-8") as handle:
            for payload in episode_stats:
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        video_files = link_tree(source / "videos", staging / "videos")
        report = {
            "status": "PASS",
            "source_root": str(source),
            "output_root": str(output),
            "rotation_contract": {
                "stored_rotation": ROT6D_CONVENTION,
                "runtime_relative_pose": "T_window_start^-1 @ T_target",
                "stored_quaternion_source_order": "xyzw",
            },
            "episodes": len(episode_audit),
            "frames": total_frames,
            "hardlinked_video_files": video_files,
            "max_input_quaternion_norm_error": float(
                max((row["max_input_quaternion_norm_error"] for row in episode_audit), default=0.0)
            ),
            "max_rotation_roundtrip_rad": float(
                max(
                    (
                        max(row["max_state_rotation_roundtrip_rad"], row["max_action_rotation_roundtrip_rad"])
                        for row in episode_audit
                    ),
                    default=0.0,
                )
            ),
            "max_anchor_se3_position_error_m": float(
                max((row["max_anchor_se3_position_error_m"] for row in episode_audit), default=0.0)
            ),
            "max_anchor_se3_rotation_error_rad": float(
                max((row["max_anchor_se3_rotation_error_rad"] for row in episode_audit), default=0.0)
            ),
            "episode_audit": episode_audit,
        }
        write_json(staging / "conversion" / "conversion_audit_rot6d.json", report)
        os.replace(staging, output)
        return report
    except Exception:
        print(f"Conversion failed; preserving staging directory for inspection: {staging}", file=sys.stderr)
        raise


def main() -> int:
    args = parse_args()
    if len(args.source_root) != len(args.output_root):
        raise ValueError("Every --source-root requires one --output-root.")
    reports = [
        convert_dataset(source, output)
        for source, output in zip(args.source_root, args.output_root, strict=True)
    ]
    summary = {
        "status": "PASS",
        "datasets": reports,
        "total_frames": int(sum(report["frames"] for report in reports)),
        "total_hardlinked_video_files": int(sum(report["hardlinked_video_files"] for report in reports)),
    }
    write_json(args.report, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
