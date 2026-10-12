#!/usr/bin/env python3
"""Materialize ordered camera-frame poses and build the Franka EEF sidecar.

The camera-frame pose conversion is applied once to the parquet columns.  The
EEF image sidecar is kept separate because it is an auxiliary action target,
not robot state.  Both outputs are fail-closed and accompanied by reports.
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


ARM_NAMES = ("left", "right")
POSE_DIM = 10
ROBOT_DIM = 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rot6d_to_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    rows = values[..., :6].reshape(*values.shape[:-1], 2, 3)
    r0 = rows[..., 0, :]
    r1 = rows[..., 1, :]
    r0 = r0 / np.maximum(np.linalg.norm(r0, axis=-1, keepdims=True), 1e-12)
    r1 = r1 - r0 * np.sum(r0 * r1, axis=-1, keepdims=True)
    r1 = r1 / np.maximum(np.linalg.norm(r1, axis=-1, keepdims=True), 1e-12)
    r2 = np.cross(r0, r1)
    return np.stack((r0, r1, r2), axis=-2)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64)[..., :2, :].reshape(*matrix.shape[:-2], 6)


def transform_pose(pose: np.ndarray, camera_from_base: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != POSE_DIM:
        raise ValueError(f"Expected [T,10] arm pose, got {pose.shape}")
    rotation = rot6d_to_matrix(pose[:, 3:9])
    camera_rotation = camera_from_base[:3, :3] @ rotation
    camera_position = pose[:, :3] @ camera_from_base[:3, :3].T + camera_from_base[:3, 3]
    output = np.empty_like(pose)
    output[:, :3] = camera_position
    output[:, 3:9] = matrix_to_rot6d(camera_rotation)
    output[:, 9] = pose[:, 9]
    return output.astype(np.float32)


def append_fixed_list(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    return table.append_column(
        name, pa.array(np.asarray(values, dtype=np.float32).tolist(), type=pa.list_(pa.float32(), list_size=20))
    )


def feature_names(prefix: str) -> list[str]:
    names = []
    for arm in ARM_NAMES:
        names.extend([
            f"{arm}_x{prefix}", f"{arm}_y{prefix}", f"{arm}_z{prefix}",
            f"{arm}_rot6d_r0c0{prefix}", f"{arm}_rot6d_r0c1{prefix}",
            f"{arm}_rot6d_r0c2{prefix}", f"{arm}_rot6d_r1c0{prefix}",
            f"{arm}_rot6d_r1c1{prefix}", f"{arm}_rot6d_r1c2{prefix}",
            f"{arm}_gripper{prefix}",
        ])
    return names


def update_info(info_path: Path) -> None:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.setdefault("features", {})
    features["observation.state.camera_dual_arm"] = {
        "dtype": "float32", "shape": [20], "names": [feature_names("")],
    }
    features["action.manip.camera_dual_arm"] = {
        "dtype": "float32", "shape": [20], "names": [feature_names("_target")],
    }
    info["fastwam_camera_frame_contract"] = {
        "source": "ordered_color_blocks_period2_optimized_extrinsic",
        "pose_conversion": "camera_from_base @ base_pose",
        "state_action": "absolute_camera_frame_xyz_row_major_rot6d_plus_gripper",
        "extra_transform_at_loader": "none",
    }
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def materialize_ordered(source: Path, tracks: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    shutil.copytree(source / "meta", output / "meta")
    (output / "videos").symlink_to(source / "videos", target_is_directory=True)
    update_info(output / "meta/info.json")
    rows = 0
    max_state_error = 0.0
    transform_ids: set[str] = set()
    for src in sorted((source / "data").glob("chunk-*/*.parquet")):
        episode = int(src.stem.split("_")[-1])
        track_path = tracks / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.npz"
        if not track_path.is_file():
            raise FileNotFoundError(track_path)
        with np.load(track_path, allow_pickle=False) as data:
            left_tf = np.asarray(data["left_base_to_camera"], dtype=np.float64)
            right_rel = np.asarray(data["right_base_to_left_base"], dtype=np.float64)
            calibration_id = str(data["calibration_id"].item())
            provisional = bool(data["calibration_provisional"].item())
        if provisional or left_tf.shape != (4, 4) or right_rel.shape != (4, 4):
            raise ValueError(f"Invalid ordered calibration in {track_path}")
        right_tf = left_tf @ right_rel
        table = pq.read_table(src)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action.manip"].to_pylist(), dtype=np.float32)
        if state.shape[1] != 23 or action.shape[1] != 20 or state.shape[0] != action.shape[0]:
            raise ValueError(f"Unexpected ordered shapes at episode {episode}: {state.shape}/{action.shape}")
        state_cam = np.concatenate((
            transform_pose(state[:, 3:13], left_tf),
            transform_pose(state[:, 13:23], right_tf),
        ), axis=1)
        action_cam = np.concatenate((
            transform_pose(action[:, :10], left_tf),
            transform_pose(action[:, 10:20], right_tf),
        ), axis=1)
        if not np.isfinite(state_cam).all() or not np.isfinite(action_cam).all():
            raise ValueError(f"Non-finite converted pose at episode {episode}")
        max_state_error = max(max_state_error, float(np.max(np.abs(state_cam[:, :3]))))
        transform_ids.add(calibration_id)
        target = output / "data" / src.relative_to(source / "data")
        target.parent.mkdir(parents=True, exist_ok=True)
        table = append_fixed_list(table, "observation.state.camera_dual_arm", state_cam)
        table = append_fixed_list(table, "action.manip.camera_dual_arm", action_cam)
        pq.write_table(table, target, compression="zstd")
        rows += int(state.shape[0])
    report = {
        "status": "PASS",
        "source": str(source),
        "output": str(output),
        "episodes": len(list((source / "data").glob("chunk-*/*.parquet"))),
        "frames": rows,
        "calibration_ids": sorted(transform_ids),
        "extra_transform_after_materialization": False,
        "conversion": "per-episode left_base_to_camera and right_base_to_camera=left_base_to_camera@right_base_to_left_base",
        "max_abs_camera_position": max_state_error,
    }
    (output / "meta/camera_frame_conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_franka_sidecar(dataset_root: Path, track_root: Path, output_root: Path) -> dict:
    source_name = "franka_cup_pyramid_base_state_action_v1"
    destination = output_root / source_name
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing sidecar: {destination}")
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    episodes = int(info["total_episodes"])
    total_frames = int(info["total_frames"])
    values = np.lib.format.open_memmap(destination.parent / f".{source_name}.values.npy", mode="w+", dtype=np.float32, shape=(total_frames, 6))
    masks = np.lib.format.open_memmap(destination.parent / f".{source_name}.mask.npy", mode="w+", dtype=np.bool_, shape=(total_frames, 6))
    values[:] = 0.0
    masks[:] = False
    offset = 0
    valid_counts = np.zeros(2, dtype=np.int64)
    visible_counts = np.zeros(2, dtype=np.int64)
    for episode in range(episodes):
        parquet = dataset_root / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
        track = track_root / f"chunk-{episode // 1000:03d}/episode_{episode:06d}.npz"
        rows = int(pq.ParquetFile(parquet).metadata.num_rows)
        with np.load(track, allow_pickle=False) as data:
            frame_index = np.asarray(data["frame_index"], dtype=np.int64)
            xy = np.asarray(data["track_xy_geom"], dtype=np.float32)
            valid = np.asarray(data["operation_track_geometric_in_fov"], dtype=bool)
            image_size = np.asarray(data["image_size"], dtype=np.int64)
        if rows != len(frame_index) or xy.shape != (rows, 2, 2) or valid.shape != (rows, 2):
            raise ValueError(f"Franka track mismatch at episode {episode}")
        if not np.array_equal(frame_index, np.arange(rows, dtype=np.int64)):
            raise ValueError(f"Non-contiguous Franka track at episode {episode}")
        width, height = map(int, image_size.tolist())
        scale = np.asarray([width - 1, height - 1], dtype=np.float32)
        if np.any(valid & ~np.isfinite(xy).all(axis=-1)):
            raise ValueError(f"Non-finite valid Franka EEF point at episode {episode}")
        xy = np.clip(xy, 0.0, scale)
        model_xy = np.empty_like(xy)
        model_xy[..., 0] = (xy[..., 0] + 0.5) * (384.0 / width) - 0.5
        model_xy[..., 1] = (xy[..., 1] + 0.5) * (216.0 / height) - 0.5 - 2.0
        model_valid = valid & np.all(
            (model_xy >= 0.0) & (model_xy <= np.asarray([383.0, 319.0])), axis=-1
        )
        normalized = 2.0 * model_xy / np.asarray([383.0, 319.0], dtype=np.float32) - 1.0
        aux = np.zeros((rows, 6), dtype=np.float32)
        aux[:, :4] = normalized.reshape(rows, 4)
        # The Franka geometry export has no independent DINO visibility field.
        # For this source, visibility is the calibrated geometric in-FOV label.
        aux[:, 4:] = np.where(model_valid, 1.0, -1.0)
        mask = np.zeros((rows, 6), dtype=bool)
        mask[:, :4] = np.repeat(model_valid, 2, axis=1)
        mask[:, 4:] = True
        values[offset:offset + rows] = aux
        masks[offset:offset + rows] = mask
        valid_counts += model_valid.sum(axis=0)
        visible_counts += model_valid.sum(axis=0)
        offset += rows
    if offset != total_frames:
        raise ValueError(f"Franka frame total mismatch: {offset} != {total_frames}")
    values.flush(); masks.flush(); del values, masks
    destination.mkdir(parents=True)
    values_path = destination / "eef_xy_visible_normalized.npy"
    masks_path = destination / "feature_mask.npy"
    shutil.move(destination.parent / f".{source_name}.values.npy", values_path)
    shutil.move(destination.parent / f".{source_name}.mask.npy", masks_path)
    manifest = {
        "schema_version": "fastwam_eef_xy_sidecar_v1",
        "status": "complete",
        "source_name": source_name,
        "dataset_root": str(dataset_root),
        "track_root": str(track_root),
        "total_episodes": episodes,
        "total_frames": total_frames,
        "fps": int(info["fps"]),
        "eef_xy_order": ["left_x", "left_y", "right_x", "right_y", "left_visible", "right_visible"],
        "coordinate_frame": "fastwam_manipulation_mosaic_384x320_xy_and_visibility",
        "source_coordinate_field": "track_xy_geom",
        "normalization": "same 384x216 main-camera resize and y=2 mosaic crop as calibrated_v5",
        "invalid_fill": {"xy": 0.0, "visible": -1.0},
        "validity_masks": {"xy": "operation_track_geometric_in_fov", "visible": "operation_track_geometric_in_fov"},
        "visibility_source": "geometric_in_fov; Franka v1 geometry has no independent track_visible field",
        "geometric_in_fov_counts": valid_counts.tolist(),
        "visible_counts": visible_counts.tolist(),
        "files": {"eef_xy_visible_normalized.npy": sha256(values_path), "feature_mask.npy": sha256(masks_path)},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordered-source", type=Path, required=True)
    parser.add_argument("--ordered-tracks", type=Path, required=True)
    parser.add_argument("--ordered-output", type=Path, required=True)
    parser.add_argument("--franka-dataset", type=Path, required=True)
    parser.add_argument("--franka-tracks", type=Path, required=True)
    parser.add_argument("--eef-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.eef_output.mkdir(parents=True, exist_ok=True)
    ordered = materialize_ordered(args.ordered_source, args.ordered_tracks, args.ordered_output)
    franka = build_franka_sidecar(args.franka_dataset, args.franka_tracks, args.eef_output)
    report = {"status": "PASS", "ordered": ordered, "franka_eef_sidecar": franka}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
