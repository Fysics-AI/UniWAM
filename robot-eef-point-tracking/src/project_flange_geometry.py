#!/usr/bin/env python3
"""Re-project calibrated main-view tracks with explicit flange semantics.

The upstream geometry is intentionally left untouched. This stage reads the final
Cartesian state again, keeps the validated calibration matrices, and writes an
output with the canonical J6 frame, a topology-backed visible flange-face
candidate, and the old diagnostic auxiliary point. It also supports
narrowly-scoped, evidence-backed calibration
overrides for reviewed AgileX7000 camera outliers.

The script is CPU-only and therefore does not compete with DINO/SAM jobs for
GPU memory.  A manifest is written only after every selected episode has an
atomic NPZ/report pair.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq


ALGORITHM_VERSION = "agx-eef-geometry-v5.4"
SCHEMA_VERSION = 2
POINT_DEFINITION_VERSION = "agilex-piper-j6-plus-visible-z56-v1"

# Production target: the logged Cartesian link6/J6 origin. Canonical Piper
# defines the gripper-base fixed frame at exactly this origin, and observation
# joint FK agrees with logged XYZ at sub-millimetre scale. The custom AGILEX
# mesh does not establish a unique physical surface-center offset from J6.
# The old +Z 135.03 mm point is retained only as an explicitly non-canonical
# auxiliary coordinate; it is not called TCP, contact point, or flange.
LEGACY_AUX_OFFSET = np.asarray([0.0, 0.0, 0.13503], dtype=np.float64)
VISUAL_FLANGE_FACE_OFFSET = np.asarray([0.0, 0.0, 0.056], dtype=np.float64)
VISUAL_FLANGE_EVIDENCE = (
    Path(__file__).resolve().parent.parent
    / "assets" / "physical_flange_evidence.json"
)
VISUAL_FLANGE_EVIDENCE_SHA256 = "896c846c39ea5e44c5d92b8b7408693086a7a28ec1df85e40301eb86a5f4d798"
AGX_ARM_MOUNTS = np.asarray(
    [[0.23875, 0.3, 0.775], [0.23875, -0.3, 0.775]], dtype=np.float64
)
ARM_NAMES = np.asarray(["left", "right"])
# The URDF is an external input used for point-semantic review, not runtime FK.
URDF_PATH = "external AGILEX URDF; see assets/physical_flange_evidence.json"
URDF_SHA256 = "d4a115493af6a67ef969cfb8686b54f2970225b617ec6dd99cf682c0c4c7b6e9"


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, path)
        return digest
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def fingerprint_matches(saved: Any) -> bool:
    if not isinstance(saved, dict) or not saved.get("path"):
        return False
    path = Path(saved["path"])
    if not path.is_file():
        return False
    current = fingerprint(path)
    return (
        current["size_bytes"] == saved.get("size_bytes")
        and current["mtime_ns"] == saved.get("mtime_ns")
    )


def rot6d_rows_to_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"expected [T,6] row-major rot6d, got {values.shape}")
    first = values[:, :3]
    second = values[:, 3:]
    first = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second = second - np.sum(first * second, axis=1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("degenerate/non-finite rotation rows")
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=1)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate(
        (points, np.ones((*points.shape[:-1], 1), dtype=np.float64)), axis=-1
    )
    return np.einsum("ij,...j->...i", np.asarray(transform, dtype=np.float64), homogeneous)[..., :3]


def project_points(points: np.ndarray, intrinsic: np.ndarray, base_to_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = transform_points(base_to_camera, points)
    image = np.einsum("ij,...j->...i", intrinsic, camera)
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = image[..., :2] / image[..., 2:3]
    return xy.astype(np.float32), camera[..., 2].astype(np.float32)


def in_fov(xy: np.ndarray, depth: np.ndarray, width: int, height: int) -> np.ndarray:
    return (
        np.isfinite(xy).all(axis=-1)
        & np.isfinite(depth)
        & (depth > 0.0)
        & (xy[..., 0] >= 0.0)
        & (xy[..., 0] < width)
        & (xy[..., 1] >= 0.0)
        & (xy[..., 1] < height)
    )


def masked(xy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.asarray(xy, dtype=np.float32).copy()
    result[~valid] = np.nan
    return result


def read_state(path: Path) -> np.ndarray:
    table = pq.read_table(path, columns=["observation.state"], use_threads=False)
    column = table["observation.state"].combine_chunks()
    width = int(column.type.list_size)
    values = column.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float64).reshape(-1, width)


def scalar(value: np.ndarray | None) -> str | None:
    if value is None:
        return None
    array = np.asarray(value)
    return str(array.item()) if array.ndim == 0 else None


def matrix_digest(*matrices: np.ndarray) -> str:
    digest = hashlib.sha256()
    for matrix in matrices:
        array = np.ascontiguousarray(matrix, dtype="<f8")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def matrix_value_sha256(matrix: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(matrix, dtype="<f8").reshape(-1).tobytes(order="C")
    ).hexdigest()


def geometry_content_identity(
    arrays: dict[str, np.ndarray], dataset: str, episode: int
) -> str:
    digest = hashlib.sha256()
    digest.update(ALGORITHM_VERSION.encode("ascii"))
    digest.update(POINT_DEFINITION_VERSION.encode("ascii"))
    digest.update(dataset.encode("utf-8"))
    digest.update(np.asarray(episode, dtype="<i8").tobytes())
    for key in (
        "frame_index", "operation_mask", "eef_xy_geom", "track_xy_geom",
        "legacy_aux_xy_geom", "eef_geometric_in_fov",
        "track_geometric_in_fov", "intrinsic", "extrinsic",
        "right_base_to_left_base",
    ):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def source_parquet_from_report(report: dict[str, Any]) -> Path:
    # AGX reports call the final state parquet ``input_parquet``; AgileX
    # reports retain both source and final paths and the final state is needed.
    for key in ("final_parquet", "input_parquet"):
        value = report.get(key)
        if value and Path(value).is_file():
            return Path(value)
    raise FileNotFoundError(f"no readable state parquet in report {report.get('geometry_report')}")


def load_override(path: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported override schema in {path}")
    key = (str(payload["dataset"]), int(payload["final_episode_index"]))
    matrix = np.asarray(payload["camera_to_left_base"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid override E in {path}")
    calibration_columns = [
        "camera_intrinsics.camera_front",
        "camera_front_in_arm_left_base",
        "arm_right_base_in_arm_left_base",
    ]
    portable = payload.get("portable_release_calibration")
    if portable:
        replacement = None
        replacement_values = {
            "camera_intrinsics.camera_front": np.asarray(
                portable["intrinsic"], dtype=np.float64
            ).reshape(3, 3),
            "camera_front_in_arm_left_base": matrix,
        }
        if portable.get("right_base_to_left_base") is not None:
            replacement_values["arm_right_base_in_arm_left_base"] = np.asarray(
                portable["right_base_to_left_base"], dtype=np.float64
            )
    else:
        replacement = Path(payload["replacement_source_parquet"])
        if (
            not replacement.is_file()
            or sha256_file(replacement) != payload["replacement_source_parquet_sha256"]
        ):
            raise ValueError(f"override replacement parquet fingerprint changed: {replacement}")
        replacement_table = pq.read_table(
            replacement, columns=calibration_columns, use_threads=False
        )
        replacement_values = {
            name: np.asarray(replacement_table[name][0].as_py(), dtype=np.float64)
            for name in calibration_columns
        }
    replacement_matrix = replacement_values[
        "camera_front_in_arm_left_base"
    ].reshape(4, 4)
    if not np.array_equal(matrix, replacement_matrix):
        raise ValueError(f"embedded override E differs from {replacement}")
    for name, value in replacement_values.items():
        expected_sha = payload["replacement_matrix_sha256"][name]
        if matrix_value_sha256(value) != expected_sha:
            raise ValueError(f"override replacement matrix fingerprint changed: {name}")
    replacement_b = (
        portable.get("right_base_to_left_base") if portable
        else payload.get("right_base_to_left_base")
    )
    if replacement_b is not None:
        replacement_b = np.asarray(replacement_b, dtype=np.float64)
        if replacement_b.shape != (4, 4) or not np.array_equal(
            replacement_b,
            replacement_values["arm_right_base_in_arm_left_base"].reshape(4, 4),
        ):
            raise ValueError(f"embedded override B differs from {replacement}")
    fields = payload.get("fields")
    if fields is None:
        fields = [payload.get("field")]
    replace_b_declared = (
        "arm_right_base_in_arm_left_base" in fields
        or bool(payload.get("decision", {}).get("replace_B", False))
    )
    if replace_b_declared != (replacement_b is not None):
        raise ValueError(
            f"override B policy/content mismatch in {path}: "
            f"declared={replace_b_declared}, embedded={replacement_b is not None}"
        )
    if "camera_front_in_arm_left_base" not in " ".join(str(x) for x in fields):
        raise ValueError(f"override does not explicitly declare E replacement: {path}")
    evidence = payload.get("evidence", {})
    if portable:
        score_report = Path(__file__).resolve().parent.parent / portable["evidence_file"]
        expected_evidence_sha = portable["evidence_sha256"]
    else:
        score_report = Path(evidence["full_frame_score_report"])
        expected_evidence_sha = evidence["full_frame_score_report_sha256"]
    if not score_report.is_file() or sha256_file(score_report) != expected_evidence_sha:
        raise ValueError(f"override evidence fingerprint changed: {score_report}")
    payload = dict(payload)
    payload["_matrix"] = matrix
    payload["_right_to_left"] = replacement_b
    payload["_replacement_intrinsic"] = replacement_values[
        "camera_intrinsics.camera_front"
    ].reshape(3, 3)
    payload["_path"] = str(path.resolve())
    payload["_sha256"] = sha256_file(path)
    return {key: payload}


def video_size(path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    try:
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def geometry_for_agx(states: np.ndarray, arrays: dict[str, np.ndarray], width: int, height: int) -> dict[str, np.ndarray]:
    if states.shape[1] != 23:
        raise ValueError(f"AGX state must be 23D, got {states.shape}")
    frame_count = len(states)
    flange = np.empty((frame_count, 2, 3), dtype=np.float64)
    visual_flange = np.empty_like(flange)
    legacy_aux = np.empty_like(flange)
    for arm, (p0, r0) in enumerate(((3, 6), (13, 16))):
        position = states[:, p0 : p0 + 3] + AGX_ARM_MOUNTS[arm]
        rotation = rot6d_rows_to_matrix(states[:, r0 : r0 + 6])
        flange[:, arm] = position
        visual_flange[:, arm] = position + np.einsum(
            "tij,j->ti", rotation, VISUAL_FLANGE_FACE_OFFSET
        )
        legacy_aux[:, arm] = position + np.einsum(
            "tij,j->ti", rotation, LEGACY_AUX_OFFSET
        )
    K = np.asarray(arrays["intrinsic"], dtype=np.float64)
    E = np.asarray(arrays.get("extrinsic"), dtype=np.float64)
    if E.shape != (4, 4):
        raise ValueError("AGX geometry is missing 4x4 extrinsic")
    return make_projected_arrays(
        flange, visual_flange, legacy_aux, K, E, width, height, arrays
    )


def geometry_for_agilex(states: np.ndarray, arrays: dict[str, np.ndarray], width: int, height: int, base_to_camera: np.ndarray) -> dict[str, np.ndarray]:
    if states.shape[1] != 20:
        raise ValueError(f"AgileX state must be 20D, got {states.shape}")
    frame_count = len(states)
    flange = np.empty((frame_count, 2, 3), dtype=np.float64)
    visual_flange = np.empty_like(flange)
    legacy_aux = np.empty_like(flange)
    B = np.asarray(arrays["right_base_to_left_base"], dtype=np.float64)
    for arm, (p0, r0) in enumerate(((0, 3), (10, 13))):
        local = states[:, p0 : p0 + 3]
        rotation = rot6d_rows_to_matrix(states[:, r0 : r0 + 6])
        local_visual_flange = local + np.einsum(
            "tij,j->ti", rotation, VISUAL_FLANGE_FACE_OFFSET
        )
        local_legacy_aux = local + np.einsum(
            "tij,j->ti", rotation, LEGACY_AUX_OFFSET
        )
        transform = np.eye(4) if arm == 0 else B
        flange[:, arm] = transform_points(transform, local)
        visual_flange[:, arm] = transform_points(transform, local_visual_flange)
        legacy_aux[:, arm] = transform_points(transform, local_legacy_aux)
    K = np.asarray(arrays["intrinsic"], dtype=np.float64)
    return make_projected_arrays(
        flange, visual_flange, legacy_aux, K, base_to_camera, width, height, arrays
    )


def geometry_from_input_points(
    arrays: dict[str, np.ndarray],
    intrinsic: np.ndarray,
    base_to_camera: np.ndarray,
    effective_right_to_left: np.ndarray,
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    """Reuse the upstream stored 3D points without reopening the state parquet."""
    if "flange_footprint" in arrays and "grasp_center_footprint" in arrays:
        flange = np.asarray(arrays["flange_footprint"], dtype=np.float64)
        legacy_aux = np.asarray(
            arrays["grasp_center_footprint"], dtype=np.float64
        )
    elif "flange_left_base" in arrays and "grasp_center_left_base" in arrays:
        flange = np.asarray(arrays["flange_left_base"], dtype=np.float64).copy()
        legacy_aux = np.asarray(
            arrays["grasp_center_left_base"], dtype=np.float64
        ).copy()
        original_right_to_left = np.asarray(
            arrays["right_base_to_left_base"], dtype=np.float64
        )
        if not np.array_equal(original_right_to_left, effective_right_to_left):
            left_to_original_right = np.linalg.inv(original_right_to_left)
            for points in (flange, legacy_aux):
                local_right = transform_points(
                    left_to_original_right, points[:, 1]
                )
                points[:, 1] = transform_points(
                    effective_right_to_left, local_right
                )
    else:
        raise KeyError("upstream geometry has no reusable 3D flange/auxiliary point pair")
    if flange.shape != legacy_aux.shape or flange.ndim != 3:
        raise ValueError(
            f"invalid reusable input point shapes: {flange.shape}, {legacy_aux.shape}"
        )
    ratio = float(VISUAL_FLANGE_FACE_OFFSET[2] / LEGACY_AUX_OFFSET[2])
    visual_flange = flange + ratio * (legacy_aux - flange)
    return make_projected_arrays(
        flange,
        visual_flange,
        legacy_aux,
        intrinsic,
        base_to_camera,
        width,
        height,
        arrays,
    )


def make_projected_arrays(
    flange: np.ndarray,
    visual_flange: np.ndarray,
    legacy_aux: np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
    width: int,
    height: int,
    source_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    flange_xy, flange_depth = project_points(flange, K, E)
    visual_flange_xy, visual_flange_depth = project_points(visual_flange, K, E)
    legacy_aux_xy, legacy_aux_depth = project_points(legacy_aux, K, E)
    flange_fov = in_fov(flange_xy, flange_depth, width, height)
    visual_flange_fov = in_fov(
        visual_flange_xy, visual_flange_depth, width, height
    )
    legacy_aux_fov = in_fov(legacy_aux_xy, legacy_aux_depth, width, height)
    operation = np.asarray(source_arrays.get("operation_mask", np.ones(len(flange), dtype=bool)), dtype=bool)
    if operation.shape != (len(flange),):
        raise ValueError(f"operation mask has shape {operation.shape}")
    op_flange = operation[:, None] & flange_fov
    op_visual_flange = operation[:, None] & visual_flange_fov
    op_legacy_aux = operation[:, None] & legacy_aux_fov
    return {
        "frame_index": np.arange(len(flange), dtype=np.int32),
        # Canonical flange/EFF point requested by the user. The logged
        # Cartesian state and Piper API both define it at J6/link6 origin.
        "flange_footprint": flange.astype(np.float32),
        "j6_origin_footprint": flange.astype(np.float32),
        "eef_footprint": flange.astype(np.float32),
        "track_footprint": visual_flange.astype(np.float32),
        "visual_flange_face_candidate_footprint": visual_flange.astype(np.float32),
        # Explicit non-canonical compatibility point used by upstream geometry.
        "legacy_aux_footprint": legacy_aux.astype(np.float32),
        "legacy_grasp_center_footprint": legacy_aux.astype(np.float32),
        "flange_xy_geom": flange_xy,
        "j6_origin_xy_geom": flange_xy.copy(),
        "eef_xy_geom": flange_xy.copy(),
        "flange_projected_xy": flange_xy.copy(),
        "track_xy_geom": visual_flange_xy,
        "visual_flange_face_candidate_xy_geom": visual_flange_xy.copy(),
        "legacy_aux_xy_geom": legacy_aux_xy,
        "legacy_grasp_center_xy_geom": legacy_aux_xy.copy(),
        "flange_xy": masked(flange_xy, op_flange),
        "j6_origin_xy": masked(flange_xy, op_flange),
        "eef_xy": masked(flange_xy, op_flange),
        "track_xy": masked(visual_flange_xy, op_visual_flange),
        "visual_flange_face_candidate_xy": masked(
            visual_flange_xy, op_visual_flange
        ),
        "legacy_aux_xy": masked(legacy_aux_xy, op_legacy_aux),
        "legacy_grasp_center_xy": masked(legacy_aux_xy, op_legacy_aux),
        "flange_geometric_in_fov": flange_fov,
        "j6_origin_geometric_in_fov": flange_fov.copy(),
        "eef_geometric_in_fov": flange_fov.copy(),
        "track_geometric_in_fov": visual_flange_fov,
        "visual_flange_face_candidate_geometric_in_fov": visual_flange_fov.copy(),
        "geometric_in_fov": visual_flange_fov.copy(),
        "geometric_in_frame": visual_flange_fov.copy(),
        "legacy_aux_geometric_in_fov": legacy_aux_fov,
        "legacy_grasp_center_geometric_in_fov": legacy_aux_fov.copy(),
        "operation_geometric_in_fov": op_visual_flange,
        "operation_track_geometric_in_fov": op_visual_flange,
        "operation_flange_geometric_in_fov": op_flange,
        "operation_eef_geometric_in_fov": op_flange,
        "operation_legacy_aux_geometric_in_fov": op_legacy_aux,
        "flange_camera_depth": flange_depth,
        "j6_origin_camera_depth": flange_depth.copy(),
        "eef_camera_depth": flange_depth.copy(),
        "track_camera_depth": visual_flange_depth,
        "visual_flange_face_candidate_camera_depth": visual_flange_depth.copy(),
        "legacy_aux_camera_depth": legacy_aux_depth,
        "legacy_grasp_center_camera_depth": legacy_aux_depth.copy(),
        "operation_mask": operation,
        "operation_range": np.asarray(
            source_arrays.get("operation_range", [0, len(flange) - 1]), dtype=np.int32
        ),
    }


def output_path(root: Path, dataset: str, episode: int, suffix: str) -> Path:
    # Every source dataset in this project uses chunks of 1000 episodes.
    return root / dataset / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}{suffix}"


def process_entry(task: dict[str, Any]) -> dict[str, Any]:
    entry = task["entry"]
    input_root = Path(task["input_root"])
    output_root = Path(task["output_root"])
    final_root = Path(task["final_root"])
    overrides = task.get("overrides", {})
    reuse_input_points = bool(task.get("reuse_input_points", False))
    dataset = str(entry["dataset"])
    episode = int(entry["episode_index"])
    geometry_in = Path(entry["geometry_npz"])
    report_in = Path(entry["geometry_report"])
    geometry_out = output_path(output_root, dataset, episode, ".npz")
    report_out = output_path(output_root, dataset, episode, ".report.json")
    final_out = output_path(final_root, dataset, episode, ".npz")
    final_report_out = output_path(final_root, dataset, episode, ".report.json")
    started = time.monotonic()
    try:
        old_report = json.loads(report_in.read_text(encoding="utf-8"))
        state_path = source_parquet_from_report(old_report)
        with np.load(geometry_in, allow_pickle=False) as archive:
            old = {key: archive[key] for key in archive.files}
        states = read_state(state_path)
        frame_count = len(states)
        if frame_count != len(old["frame_index"]):
            raise ValueError("final state/upstream geometry frame-count mismatch")
        width, height = np.asarray(old["image_size"], dtype=int).tolist()
        video_frames, video_width, video_height = video_size(Path(entry["video"]))
        if (video_frames, video_width, video_height) != (frame_count, width, height):
            raise ValueError(
                "video/state/image-size mismatch: "
                f"video={(video_frames, video_width, video_height)} "
                f"state/geometry={(frame_count, width, height)}"
            )
        if int(entry.get("frame_count", frame_count)) != frame_count:
            raise ValueError("manifest/upstream geometry frame-count mismatch")
        state_dimension = int(states.shape[1])
        override = overrides.get((dataset, episode))
        K = np.asarray(old["intrinsic"], dtype=np.float64)
        is_agx = state_dimension == 23
        if not is_agx and state_dimension != 20:
            raise ValueError(f"unsupported final state dimension {state_dimension}")
        E = np.asarray(
            old["extrinsic"] if is_agx else old["left_base_to_camera"],
            dtype=np.float64,
        )
        original_E = E.copy()
        original_B = None
        effective_B = np.asarray(
            old.get("right_base_to_left_base", np.eye(4)), dtype=np.float64
        )
        if is_agx:
            if override is not None:
                raise ValueError("an AgileX7000 calibration override was attached to an AGX state")
            camera_to_left = None
        else:
            original_B = np.asarray(
                old["right_base_to_left_base"], dtype=np.float64
            ).copy()
            effective_B = original_B.copy()
            camera_to_left = np.asarray(old.get("camera_to_left_base", np.linalg.inv(E)), dtype=np.float64)
            if override is not None:
                target_source_value = old_report.get("source_parquet")
                target_source = Path(target_source_value) if target_source_value else None
                if not override.get("portable_release_calibration") and (
                    target_source is None
                    or not target_source.is_file()
                    or sha256_file(target_source)
                    != override["target_source_parquet_sha256"]
                ):
                    raise ValueError("override target source parquet fingerprint changed")
                expected = np.asarray(override.get("expected_original_camera_to_left_base", []), dtype=np.float64)
                if expected.shape == (4, 4) and not np.allclose(camera_to_left, expected, atol=2e-6, rtol=0):
                    raise ValueError("ep1952 override original E does not match the upstream source geometry")
                expected_k = np.asarray(
                    override.get("expected_intrinsic", []), dtype=np.float64
                )
                if expected_k.shape == (3, 3) and not np.array_equal(K, expected_k):
                    raise ValueError("override expected target K does not match upstream geometry")
                if not np.array_equal(K, override["_replacement_intrinsic"]):
                    raise ValueError("override requires retaining K but replacement K differs")
                expected_b = np.asarray(
                    override.get("expected_original_right_base_to_left_base", []),
                    dtype=np.float64,
                )
                if expected_b.shape == (4, 4) and not np.array_equal(
                    original_B, expected_b
                ):
                    raise ValueError("override expected target B does not match upstream geometry")
                target_matrix_sha = override["target_matrix_sha256"]
                target_values = {
                    "camera_intrinsics.camera_front": K,
                    "camera_front_in_arm_left_base": camera_to_left,
                    "arm_right_base_in_arm_left_base": np.asarray(
                        old["right_base_to_left_base"], dtype=np.float64
                    ),
                }
                for name, value in target_values.items():
                    if matrix_value_sha256(value) != target_matrix_sha[name]:
                        raise ValueError(f"override target matrix fingerprint changed: {name}")
                camera_to_left = np.asarray(override["_matrix"], dtype=np.float64)
                E = np.linalg.inv(camera_to_left)
                if override.get("_right_to_left") is not None:
                    effective_B = np.asarray(
                        override["_right_to_left"], dtype=np.float64
                    )
        if reuse_input_points:
            geometry = geometry_from_input_points(
                old, K, E, effective_B, width, height
            )
        elif is_agx:
            geometry = geometry_for_agx(states, old, width, height)
        else:
            state_geometry_inputs = dict(old)
            state_geometry_inputs["intrinsic"] = K
            state_geometry_inputs["right_base_to_left_base"] = effective_B
            geometry = geometry_for_agilex(
                states, state_geometry_inputs, width, height, E
            )

        # Do not carry ambiguous ``tcp/grasp_center`` names into the output. The
        # corresponding +Z 135.03 mm arrays are re-emitted only under explicit
        # ``legacy_aux`` names below.
        dropped_prefixes = ("tcp_", "grasp_center_")
        dropped_exact = {
            "tcp_xy", "grasp_center_xy", "visual_visible", "visible", "quality",
            "geometric_in_fov", "geometric_in_frame",
        }
        arrays = {
            key: value
            for key, value in old.items()
            if key not in dropped_exact
            and not any(key.startswith(prefix) for prefix in dropped_prefixes)
        }
        arrays.update(geometry)
        calibration_signature_v5 = matrix_digest(
            K, E, effective_B
        )
        arrays.update({
            "algorithm_version": np.asarray(ALGORITHM_VERSION),
            "geometry_algorithm_version": np.asarray(ALGORITHM_VERSION),
            "dataset": np.asarray(dataset),
            "episode_index": np.asarray(episode, dtype=np.int32),
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
            "point_definition_version": np.asarray(POINT_DEFINITION_VERSION),
            "point_names": np.asarray(
                ["flange_j6", "visual_flange_face_candidate", "legacy_aux_unknown"]
            ),
            "preferred_track_point": np.asarray("visual_flange_face_candidate"),
            "visual_flange_face_candidate_offset_local": (
                VISUAL_FLANGE_FACE_OFFSET.astype(np.float32)
            ),
            "visual_flange_face_candidate_confidence": np.asarray("medium"),
            "visual_flange_face_candidate_evidence": np.asarray(
                str(VISUAL_FLANGE_EVIDENCE)
            ),
            "visual_flange_face_candidate_evidence_sha256": np.asarray(
                VISUAL_FLANGE_EVIDENCE_SHA256
            ),
            "legacy_aux_offset_local": LEGACY_AUX_OFFSET.astype(np.float32),
            "urdf_path": np.asarray(URDF_PATH),
            "urdf_sha256": np.asarray(URDF_SHA256),
            "intrinsic": K,
            "left_base_to_camera": E,
            "extrinsic": E,
            "right_base_to_left_base": effective_B,
            "calibration_signature": np.asarray(calibration_signature_v5),
            "calibration_signature_v5": np.asarray(calibration_signature_v5),
        })
        if camera_to_left is not None:
            arrays["camera_to_left_base"] = camera_to_left
        source_override_version = scalar(old.get("calibration_override_version")) or ""
        source_override_method = scalar(old.get("calibration_method")) or ""
        old_override_report = old_report.get("calibration_override") or {}
        source_override_sha256 = (
            scalar(old.get("calibration_override_sha256"))
            or str(old_override_report.get("sha256", ""))
        )
        source_override_evidence = str(
            old_report.get("calibration_evidence", "")
        )
        v5_override_version = str(override["override_version"]) if override else ""
        v5_override_sha256 = str(override["_sha256"]) if override else ""
        if override is not None:
            canonical_override_version = v5_override_version
            canonical_override_sha256 = v5_override_sha256
        else:
            canonical_override_version = source_override_version
            canonical_override_sha256 = source_override_sha256
        arrays["source_calibration_override_version"] = np.asarray(
            source_override_version
        )
        arrays["source_calibration_override_method"] = np.asarray(
            source_override_method
        )
        arrays["source_calibration_override_sha256"] = np.asarray(
            source_override_sha256
        )
        arrays["source_calibration_override_evidence"] = np.asarray(
            source_override_evidence
        )
        arrays["v5_calibration_override_version"] = np.asarray(v5_override_version)
        arrays["v5_calibration_override_sha256"] = np.asarray(v5_override_sha256)
        arrays["calibration_override_version"] = np.asarray(
            canonical_override_version
        )
        arrays["calibration_override_sha256"] = np.asarray(
            canonical_override_sha256
        )
        content_identity = geometry_content_identity(arrays, dataset, episode)
        arrays["geometry_content_identity"] = np.asarray(content_identity)

        digest = atomic_save_npz(geometry_out, arrays)
        input_geometry_sha256 = sha256_file(geometry_in)
        state_sha256 = sha256_file(state_path)
        report = {
            "status": "complete",
            "algorithm_version": ALGORITHM_VERSION,
            "schema_version": SCHEMA_VERSION,
            "point_definition_version": POINT_DEFINITION_VERSION,
            "dataset": dataset,
            "episode_index": episode,
            "frames": frame_count,
            "video": str(entry["video"]),
            "input_geometry_npz": str(geometry_in),
            "input_geometry_report": str(report_in),
            "state_parquet": str(state_path),
            "geometry_npz": str(geometry_out),
            "final_output_npz": str(final_out),
            "final_output_report": str(final_report_out),
            "image_size": [width, height],
            "source_geometry_algorithm_version": scalar(old.get("algorithm_version")),
            "state_dimension": state_dimension,
            "geometry_content_identity": content_identity,
            "input_geometry_sha256": input_geometry_sha256,
            "state_sha256": state_sha256,
            "calibration": {
                "original_left_base_to_camera": original_E.tolist() if original_E is not None else None,
                "original_right_base_to_left_base": (
                    original_B.tolist() if original_B is not None else None
                ),
                "effective_left_base_to_camera": E.tolist(),
                "effective_camera_to_left_base": camera_to_left.tolist() if camera_to_left is not None else None,
                "effective_right_base_to_left_base": (
                    effective_B.tolist() if original_B is not None else None
                ),
                "signature": str(arrays["calibration_signature_v5"].item()),
                "source_override": (
                    {
                        "version": source_override_version,
                        "method": source_override_method,
                        "sha256": source_override_sha256,
                        "evidence": source_override_evidence,
                    }
                    if source_override_version
                    else None
                ),
                "v5_override": (
                    {"version": override["override_version"], "path": override["_path"], "sha256": override["_sha256"],
                     "replacement_source_episode_index": override.get("replacement_source_episode_index"),
                     "fields": override.get("fields", [override.get("field")]),
                     "matrix_policy": override.get("matrix_policy"),
                     "reason": override.get("reason", override.get("decision")),
                     "evidence": override.get("evidence")}
                    if override is not None else None
                ),
                "override": (
                    {
                        "version": canonical_override_version,
                        "sha256": canonical_override_sha256,
                    }
                    if canonical_override_version
                    else None
                ),
            },
            "point_definition": {
                "flange": (
                    "logged Cartesian link6/J6 origin; canonical Piper flange/EEF frame"
                ),
                "preferred_track": (
                    "link6 local +Z 56 mm visible flange/front-plate candidate; "
                    "one connected annular +Z mesh face with central bore and bolt holes"
                ),
                "preferred_track_field": "visual_flange_face_candidate_xy_geom",
                "preferred_track_offset_local_m": (
                    VISUAL_FLANGE_FACE_OFFSET.tolist()
                ),
                "preferred_track_confidence": "medium",
                "preferred_track_evidence": str(VISUAL_FLANGE_EVIDENCE),
                "preferred_track_evidence_sha256": (
                    VISUAL_FLANGE_EVIDENCE_SHA256
                ),
                "legacy_aux": (
                    "V4 +Z 135.03 mm auxiliary point retained for diagnostics; "
                    "not asserted to be TCP, contact point, or physical flange"
                ),
                "legacy_aux_offset_local_m": LEGACY_AUX_OFFSET.tolist(),
                "urdf": URDF_PATH,
                "urdf_sha256": URDF_SHA256,
            },
            "geometric_in_fov_counts": {
                "flange": np.asarray(geometry["flange_geometric_in_fov"]).sum(axis=0).astype(int).tolist(),
                "preferred_track": np.asarray(geometry["track_geometric_in_fov"]).sum(axis=0).astype(int).tolist(),
                "legacy_aux": np.asarray(geometry["legacy_aux_geometric_in_fov"]).sum(axis=0).astype(int).tolist(),
            },
            "operation_range": np.asarray(arrays["operation_range"]).astype(int).tolist(),
            "input_fingerprint": fingerprint(geometry_in),
            "state_fingerprint": fingerprint(state_path),
            "video_fingerprint": fingerprint(Path(entry["video"])),
            "output_npz_sha256": digest,
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_write_json(report_out, report)
        result = dict(entry)
        result.update({
            "algorithm_version": ALGORITHM_VERSION,
            "geometry_version": ALGORITHM_VERSION,
            "point_definition_version": POINT_DEFINITION_VERSION,
            "geometry_npz": str(geometry_out),
            "geometry_report": str(report_out),
            "output_npz": str(final_out),
            "output_report": str(final_report_out),
            "calibration_override_version": str(override["override_version"]) if override else "",
            "calibration_status": "validated",
            "status": "complete",
        })
        return {"status": "complete", "episode_index": episode, "manifest": result, "seconds": report["elapsed_seconds"]}
    except Exception as error:
        error_report = {
            "status": "failed", "algorithm_version": ALGORITHM_VERSION,
            "dataset": dataset, "episode_index": episode, "error_type": type(error).__name__,
            "error": str(error), "input_geometry_npz": str(geometry_in),
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_write_json(report_out, error_report)
        return {"status": "failed", "episode_index": episode, "error": repr(error)}


def discover_manifests(input_root: Path, agilex_dataset: str) -> list[Path]:
    paths = [input_root / "geometry_manifest.jsonl", input_root / agilex_dataset / "geometry_manifest.jsonl"]
    return [path for path in paths if path.is_file()]


def read_entries(paths: list[Path]) -> list[dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["dataset"]), int(row["episode_index"]))
            if key in result:
                raise ValueError(f"duplicate manifest entry {key}")
            # Upstream AgileX manifest rows lack an operation_start but are all
            # manipulation frames; the old geometry NPZ carries the mask.
            result[key] = row
    return [result[key] for key in sorted(result)]


def parse_episode_filter(values: list[str] | None) -> set[int] | None:
    if not values:
        return None
    selected: set[int] = set()
    for value in values:
        if ":" in value:
            start, stop = (int(x) for x in value.split(":", 1))
            selected.update(range(start, stop))
        else:
            selected.add(int(value))
    return selected


def reusable_output(
    entry: dict[str, Any],
    output: Path,
    report_path: Path,
    override: dict[str, Any] | None,
) -> bool:
    if not output.is_file() or not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_override_sha = override["_sha256"] if override is not None else ""
        actual_v5_override = report.get("calibration", {}).get("v5_override")
        actual_override_sha = (
            actual_v5_override.get("sha256")
            if isinstance(actual_v5_override, dict)
            else ""
        )
        current_geometry_path = Path(entry["geometry_npz"]).resolve()
        current_geometry_report = Path(entry["geometry_report"]).resolve()
        current_video = Path(entry["video"]).resolve()
        saved_input = report.get("input_fingerprint", {})
        saved_video = report.get("video_fingerprint", {})
        return (
            report.get("status") == "complete"
            and report.get("algorithm_version") == ALGORITHM_VERSION
            and report.get("point_definition_version") == POINT_DEFINITION_VERSION
            and report.get("dataset") == entry["dataset"]
            and int(report.get("episode_index", -1)) == int(entry["episode_index"])
            and actual_override_sha == expected_override_sha
            and Path(report.get("input_geometry_npz", "")).resolve()
            == current_geometry_path
            and Path(report.get("input_geometry_report", "")).resolve()
            == current_geometry_report
            and Path(report.get("video", "")).resolve() == current_video
            and Path(saved_input.get("path", "")).resolve() == current_geometry_path
            and Path(saved_video.get("path", "")).resolve() == current_video
            and fingerprint_matches(report.get("input_fingerprint"))
            and fingerprint_matches(report.get("state_fingerprint"))
            and fingerprint_matches(report.get("video_fingerprint"))
            and report.get("input_geometry_sha256")
            == sha256_file(current_geometry_path)
            and report.get("output_npz_sha256") == sha256_file(output)
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/calibrated_geometry_inputs"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/flange_geometry"))
    parser.add_argument("--final-root", type=Path, default=Path("outputs/eef_tracks"))
    parser.add_argument("--agilex-dataset", default="agilex7000_manip_short30_rot6d_rowmajor_h32_v4_prompt_reviewed")
    parser.add_argument("--override", type=Path, action="append", default=[])
    parser.add_argument("--episode", action="append")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-input-points",
        action="store_true",
        help="Fast diagnostic path; production defaults to recomputing final state.",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--geometry-manifest",
        type=Path,
        help="Explicit output manifest path; partial runs otherwise use a .partial file.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if (
        not VISUAL_FLANGE_EVIDENCE.is_file()
        or sha256_file(VISUAL_FLANGE_EVIDENCE)
        != VISUAL_FLANGE_EVIDENCE_SHA256
    ):
        raise SystemExit(
            f"visual flange evidence is missing or changed: {VISUAL_FLANGE_EVIDENCE}"
        )
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    final_root = args.final_root.resolve()
    all_entries = read_entries(discover_manifests(input_root, args.agilex_dataset))
    if not all_entries:
        raise SystemExit("input geometry manifests contain no episodes")
    entries = list(all_entries)
    selected = parse_episode_filter(args.episode)
    if selected is not None:
        entries = [row for row in entries if int(row["episode_index"]) in selected]
    if args.max_episodes is not None:
        entries = entries[: args.max_episodes]
    if not entries:
        raise SystemExit("episode selection is empty; refusing to publish a manifest")
    partial_run = len(entries) != len(all_entries)
    manifest_output = (
        args.geometry_manifest.resolve()
        if args.geometry_manifest is not None
        else output_root
        / ("geometry_manifest.partial.jsonl" if partial_run else "geometry_manifest.jsonl")
    )
    overrides: dict[tuple[str, int], dict[str, Any]] = {}
    for path in args.override:
        overrides.update(load_override(path.resolve()))
    tasks = []
    resumed = 0
    for entry in entries:
        out = output_path(output_root, str(entry["dataset"]), int(entry["episode_index"]), ".npz")
        rep = output_path(output_root, str(entry["dataset"]), int(entry["episode_index"]), ".report.json")
        override = overrides.get((str(entry["dataset"]), int(entry["episode_index"])))
        if not args.overwrite and reusable_output(entry, out, rep, override):
            resumed += 1
            continue
        tasks.append({
            "entry": entry,
            "input_root": str(input_root),
            "output_root": str(output_root),
            "final_root": str(final_root),
            "overrides": overrides,
            "reuse_input_points": args.reuse_input_points,
        })
    print(json.dumps({"algorithm_version": ALGORITHM_VERSION, "selected": len(entries), "resumed": resumed, "queued": len(tasks), "workers": args.workers}, ensure_ascii=False), flush=True)
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    if args.workers <= 1:
        iterator = map(process_entry, tasks)
        for index, result in enumerate(iterator, 1):
            results.append(result)
            if index % args.progress_every == 0 or index == len(tasks):
                print(json.dumps({"progress": index, "total": len(tasks), "status_counts": dict(Counter(x["status"] for x in results))}), flush=True)
    elif tasks:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            for index, result in enumerate(pool.imap_unordered(process_entry, tasks, chunksize=1), 1):
                results.append(result)
                if index % args.progress_every == 0 or index == len(tasks):
                    print(json.dumps({"progress": index, "total": len(tasks), "status_counts": dict(Counter(x["status"] for x in results))}), flush=True)
    failed = [x for x in results if x["status"] != "complete"]
    # Re-read resumed manifests and merge atomically.  A failed run never
    # publishes a partial manifest.
    if failed:
        print(json.dumps({"status": "failed", "failures": failed[:20]}, ensure_ascii=False), flush=True)
        return 2
    manifest_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in entries:
        out = output_path(output_root, str(entry["dataset"]), int(entry["episode_index"]), ".npz")
        rep = output_path(output_root, str(entry["dataset"]), int(entry["episode_index"]), ".report.json")
        if out.is_file() and rep.is_file():
            try:
                report = json.loads(rep.read_text(encoding="utf-8"))
                if report.get("status") == "complete":
                    # Prefer the newly generated row if available; otherwise
                    # reconstruct from the original manifest entry.
                    row = dict(entry)
                    row.update({"algorithm_version": ALGORITHM_VERSION, "geometry_version": ALGORITHM_VERSION,
                                "point_definition_version": POINT_DEFINITION_VERSION,
                                "frame_count": int(report["frames"]),
                                "image_size": report["image_size"],
                                "geometry_npz": str(out), "geometry_report": str(rep),
                                "geometry_npz_sha256": report["output_npz_sha256"],
                                "geometry_content_identity": report["geometry_content_identity"],
                                "output_npz": str(output_path(final_root, str(entry["dataset"]), int(entry["episode_index"]), ".npz")),
                                "output_report": str(output_path(final_root, str(entry["dataset"]), int(entry["episode_index"]), ".report.json")),
                                "status": "complete", "calibration_status": "validated"})
                    override = overrides.get((str(entry["dataset"]), int(entry["episode_index"])))
                    calibration = report.get("calibration", {})
                    canonical_override = calibration.get("override") or {}
                    source_override = calibration.get("source_override") or {}
                    v5_override = calibration.get("v5_override") or {}
                    row["calibration_override_version"] = canonical_override.get(
                        "version", ""
                    )
                    row["calibration_override_sha256"] = canonical_override.get(
                        "sha256", ""
                    )
                    row["source_calibration_override_version"] = source_override.get(
                        "version", ""
                    )
                    row["source_calibration_override_method"] = source_override.get(
                        "method", ""
                    )
                    row["source_calibration_override_sha256"] = source_override.get(
                        "sha256", ""
                    )
                    row["source_calibration_override_evidence"] = source_override.get(
                        "evidence", ""
                    )
                    row["v5_calibration_override_version"] = v5_override.get(
                        "version", ""
                    )
                    row["v5_calibration_override_sha256"] = v5_override.get(
                        "sha256", ""
                    )
                    manifest_rows[(str(entry["dataset"]), int(entry["episode_index"]))] = row
            except Exception:
                pass
    if len(manifest_rows) != len(entries):
        missing = sorted(set((str(x["dataset"]), int(x["episode_index"])) for x in entries) - set(manifest_rows))
        print(json.dumps({"status": "failed", "missing_manifest_rows": missing[:20]}), flush=True)
        return 2
    ordered = [manifest_rows[key] for key in sorted(manifest_rows)]
    atomic_write_jsonl(manifest_output, ordered)
    summary = {
        "status": "complete", "algorithm_version": ALGORITHM_VERSION,
        "point_definition_version": POINT_DEFINITION_VERSION,
        "episodes": len(ordered), "frames": int(sum(json.loads((output_root / str(row["dataset"]) / f"chunk-{int(row['episode_index']) // 1000:03d}" / f"episode_{int(row['episode_index']):06d}.report.json").read_text())["frames"] for row in ordered)),
        "elapsed_seconds": time.monotonic() - started,
        "geometry_manifest": str(manifest_output),
        "partial_run": partial_run,
        "input_manifest_episodes": len(all_entries),
        "output_root": str(output_root), "final_root": str(final_root),
        "overrides": [{"dataset": k[0], "episode_index": k[1], "version": v["override_version"], "path": v["_path"], "sha256": v["_sha256"]} for k, v in sorted(overrides.items())],
        "urdf": {"path": URDF_PATH, "sha256": URDF_SHA256},
        "counts": dict(Counter(str(row["dataset"]) for row in ordered)),
    }
    summary_path = output_root / (
        "run_summary.partial.json" if partial_run else "run_summary.json"
    )
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
