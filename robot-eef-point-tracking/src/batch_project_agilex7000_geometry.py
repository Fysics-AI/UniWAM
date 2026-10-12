#!/usr/bin/env python3
"""Batch-project AgileX7000 flange and provisional grasp-center geometry.

The source calibration convention is:

* ``C = camera_front_in_arm_left_base`` maps camera -> left-arm base.
* ``B = arm_right_base_in_arm_left_base`` maps right-arm base -> left-arm base.
* All flattened calibration matrices are reshaped in C order.

This program produces geometry only.  Being geometrically inside the image does
not imply that an end effector is visually visible or unoccluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ALGORITHM_VERSION = "agilex7000-calibrated-geometry-v2.2"
MATRIX_SIGNATURE_VERSION = "matrix-f64-c-order-v1"
CAMERA_KEY = "observation.images.camera_front"
K_COLUMN = "camera_intrinsics.camera_front"
C_COLUMN = "camera_front_in_arm_left_base"
B_COLUMN = "arm_right_base_in_arm_left_base"
CALIBRATION_COLUMNS = {"K": K_COLUMN, "E": C_COLUMN, "B": B_COLUMN}
SIGNATURE_LABELS = {
    "K": "K",
    "E": "E_camera_to_left_base",
    "B": "B_right_base_to_left_base",
}
GRASP_CENTER_OFFSET = np.array([0.0, 0.0, 0.13503], dtype=np.float64)
ARM_NAMES = np.asarray(["left", "right"])

DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "EEF_AGILEX7000_DATASET",
        "data/lerobot/agilex7000_manip_short30_rot6d_rowmajor_h32_v4_prompt_reviewed",
    )
)
DEFAULT_OUTPUT_PARENT = Path(
    os.environ.get("EEF_GEOMETRY_OUTPUT_ROOT", "outputs/calibrated_geometry_inputs")
)


class GeometryError(ValueError):
    """Input state or projection geometry is invalid."""


class CalibrationError(ValueError):
    """A calibration matrix is present but invalid."""


@dataclass(frozen=True)
class MatrixQA:
    orthogonality_error_fro: float
    determinant: float
    determinant_error: float
    bottom_row_error_max: float

    def as_dict(self) -> dict[str, float]:
        return {
            "orthogonality_error_fro": self.orthogonality_error_fro,
            "determinant": self.determinant,
            "determinant_error": self.determinant_error,
            "bottom_row_error_max": self.bottom_row_error_max,
        }


@dataclass(frozen=True)
class CalibrationBundle:
    intrinsic: np.ndarray
    camera_to_left_base: np.ndarray
    right_base_to_left_base: np.ndarray
    left_base_to_camera: np.ndarray
    signatures: dict[str, str]
    qa: dict[str, dict[str, float]]


_WORKER_CONFIG: dict[str, Any] = {}
_CALIBRATION_CACHE: dict[str, CalibrationBundle] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_array_signature(label: str, values: np.ndarray) -> str:
    """Return a stable, exact signature for a numeric matrix."""
    array = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(MATRIX_SIGNATURE_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(label.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def combined_calibration_signature(signatures: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for key in ("K", "E", "B"):
        digest.update(key.encode("ascii"))
        digest.update(b"=")
        digest.update(signatures[key].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_rigid_transform(
    transform: np.ndarray,
    name: str,
    *,
    orthogonality_tolerance: float = 1e-4,
    determinant_tolerance: float = 1e-4,
    bottom_row_tolerance: float = 1e-6,
) -> MatrixQA:
    """Validate a homogeneous rigid transform and return audit metrics."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise CalibrationError(f"{name} must have shape (4, 4), got {transform.shape}")
    if not np.isfinite(transform).all():
        raise CalibrationError(f"{name} contains non-finite values")
    rotation = transform[:3, :3]
    orthogonality_error = float(
        np.linalg.norm(rotation @ rotation.T - np.eye(3), ord="fro")
    )
    determinant = float(np.linalg.det(rotation))
    determinant_error = abs(determinant - 1.0)
    bottom_row_error = float(
        np.max(np.abs(transform[3] - np.array([0.0, 0.0, 0.0, 1.0])))
    )
    failures = []
    if orthogonality_error >= orthogonality_tolerance:
        failures.append(
            f"orthogonality Frobenius error {orthogonality_error:.3g} "
            f">= {orthogonality_tolerance:.3g}"
        )
    if determinant_error >= determinant_tolerance:
        failures.append(
            f"|det(R)-1| {determinant_error:.3g} >= {determinant_tolerance:.3g}"
        )
    if bottom_row_error >= bottom_row_tolerance:
        failures.append(
            f"bottom-row error {bottom_row_error:.3g} >= {bottom_row_tolerance:.3g}"
        )
    if failures:
        raise CalibrationError(f"invalid {name}: " + "; ".join(failures))
    return MatrixQA(
        orthogonality_error_fro=orthogonality_error,
        determinant=determinant,
        determinant_error=determinant_error,
        bottom_row_error_max=bottom_row_error,
    )


def validate_intrinsic(intrinsic: np.ndarray) -> dict[str, float]:
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise CalibrationError(f"K must have shape (3, 3), got {intrinsic.shape}")
    if not np.isfinite(intrinsic).all():
        raise CalibrationError("K contains non-finite values")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise CalibrationError("K focal lengths must be positive")
    bottom_error = float(
        np.max(np.abs(intrinsic[2] - np.array([0.0, 0.0, 1.0])))
    )
    if bottom_error >= 1e-6:
        raise CalibrationError(f"invalid K homogeneous row: error {bottom_error:.3g}")
    return {
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "bottom_row_error_max": bottom_error,
    }


def rot6d_rows_to_matrix(values: np.ndarray, *, epsilon: float = 1e-9) -> np.ndarray:
    """Convert two row-major rotation rows to a proper rotation matrix."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != 6:
        raise GeometryError(f"rot6d last dimension must be 6, got {values.shape}")
    if not np.isfinite(values).all():
        raise GeometryError("rot6d contains non-finite values")
    first = values[..., :3]
    second = values[..., 3:6]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm <= epsilon):
        raise GeometryError("rot6d first row has near-zero norm")
    first = first / first_norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm <= epsilon):
        raise GeometryError("rot6d rows are collinear")
    second = second / second_norm
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-2)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to arbitrary leading point dimensions."""
    transform = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if transform.shape != (4, 4) or points.shape[-1] != 3:
        raise GeometryError(
            f"expected transform (4,4) and points (...,3), got {transform.shape}, {points.shape}"
        )
    homogeneous = np.concatenate(
        (points, np.ones((*points.shape[:-1], 1), dtype=np.float64)), axis=-1
    )
    return np.einsum("ij,...j->...i", transform, homogeneous)[..., :3]


def project_points(
    points_left_base: np.ndarray,
    intrinsic: np.ndarray,
    left_base_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project left-base 3D points into the undistorted pinhole image."""
    points_camera = transform_points(left_base_to_camera, points_left_base)
    image_homogeneous = np.einsum("ij,...j->...i", intrinsic, points_camera)
    depth = points_camera[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = image_homogeneous[..., :2] / image_homogeneous[..., 2:3]
    return xy, depth


def points_in_fov(
    xy: np.ndarray, depth: np.ndarray, image_width: int, image_height: int
) -> np.ndarray:
    return (
        np.isfinite(xy).all(axis=-1)
        & np.isfinite(depth)
        & (depth > 0.0)
        & (xy[..., 0] >= 0.0)
        & (xy[..., 0] < image_width)
        & (xy[..., 1] >= 0.0)
        & (xy[..., 1] < image_height)
    )


def prepare_projected_point_output(
    xy: np.ndarray,
    depth: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quantize a projection and derive a mask consistent with stored values."""
    xy_output = np.asarray(xy, dtype=np.float32)
    depth_output = np.asarray(depth, dtype=np.float32)
    in_fov = points_in_fov(
        xy_output, depth_output, image_width, image_height
    )
    masked_xy = xy_output.copy()
    masked_xy[~in_fov] = np.nan
    return xy_output, depth_output, in_fov, masked_xy


def reconstruct_geometry(
    states: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_left_base: np.ndarray,
    right_base_to_left_base: np.ndarray,
    image_width: int,
    image_height: int,
    grasp_center_offset: np.ndarray = GRASP_CENTER_OFFSET,
    *,
    left_base_to_camera: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Pure dual-arm state-to-image projection used by the batch runner."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 20:
        raise GeometryError(f"observation.state must have shape (T,20), got {states.shape}")
    if not np.isfinite(states).all():
        invalid = int((~np.isfinite(states)).any(axis=1).sum())
        raise GeometryError(f"observation.state has {invalid} non-finite rows")
    if image_width <= 0 or image_height <= 0:
        raise GeometryError(f"invalid image size {image_width}x{image_height}")
    grasp_center_offset = np.asarray(grasp_center_offset, dtype=np.float64)
    if grasp_center_offset.shape != (3,) or not np.isfinite(grasp_center_offset).all():
        raise GeometryError("grasp_center_offset must be a finite 3-vector")

    frame_count = len(states)
    flange_left_base = np.empty((frame_count, 2, 3), dtype=np.float64)
    grasp_center_left_base = np.empty_like(flange_left_base)
    for arm, (xyz_start, rotation_start) in enumerate(((0, 3), (10, 13))):
        local_flange = states[:, xyz_start : xyz_start + 3]
        local_rotation = rot6d_rows_to_matrix(
            states[:, rotation_start : rotation_start + 6]
        )
        local_grasp_center = local_flange + np.einsum(
            "tij,j->ti", local_rotation, grasp_center_offset
        )
        if arm == 0:
            flange_left_base[:, arm] = local_flange
            grasp_center_left_base[:, arm] = local_grasp_center
        else:
            flange_left_base[:, arm] = transform_points(
                right_base_to_left_base, local_flange
            )
            grasp_center_left_base[:, arm] = transform_points(
                right_base_to_left_base, local_grasp_center
            )

    if left_base_to_camera is None:
        left_base_to_camera = np.linalg.inv(camera_to_left_base)
    else:
        left_base_to_camera = np.asarray(left_base_to_camera, dtype=np.float64)
        if left_base_to_camera.shape != (4, 4):
            raise GeometryError(
                f"left_base_to_camera must have shape (4,4), got {left_base_to_camera.shape}"
            )
    flange_xy_geom, flange_depth = project_points(
        flange_left_base, intrinsic, left_base_to_camera
    )
    grasp_center_xy_geom, grasp_center_depth = project_points(
        grasp_center_left_base, intrinsic, left_base_to_camera
    )
    flange_in_fov = points_in_fov(
        flange_xy_geom, flange_depth, image_width, image_height
    )
    grasp_center_in_fov = points_in_fov(
        grasp_center_xy_geom, grasp_center_depth, image_width, image_height
    )
    flange_xy = flange_xy_geom.copy()
    grasp_center_xy = grasp_center_xy_geom.copy()
    flange_xy[~flange_in_fov] = np.nan
    grasp_center_xy[~grasp_center_in_fov] = np.nan
    return {
        "flange_left_base": flange_left_base,
        "grasp_center_left_base": grasp_center_left_base,
        "flange_xy_geom": flange_xy_geom,
        "grasp_center_xy_geom": grasp_center_xy_geom,
        "flange_xy": flange_xy,
        "grasp_center_xy": grasp_center_xy,
        "flange_depth": flange_depth,
        "grasp_center_depth": grasp_center_depth,
        "flange_geometric_in_fov": flange_in_fov,
        "grasp_center_geometric_in_fov": grasp_center_in_fov,
        "left_base_to_camera": left_base_to_camera,
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return path.stat().st_size
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def list_column_to_numpy(table: pa.Table, column: str, width: int) -> np.ndarray:
    chunked = table[column]
    if chunked.null_count:
        raise GeometryError(f"{column} contains {chunked.null_count} null rows")
    values = chunked.to_pylist()
    if any(len(row) != width for row in values):
        lengths = sorted({len(row) for row in values})
        raise GeometryError(f"{column} row widths are {lengths}, expected {width}")
    return np.asarray(values, dtype=np.float64)


def read_first_list(
    parquet_file: pq.ParquetFile,
    column: str,
    expected_size: int,
    verify_static: bool,
) -> np.ndarray:
    batches = parquet_file.iter_batches(batch_size=1, columns=[column])
    try:
        value = next(batches).column(0)[0].as_py()
    except StopIteration as exc:
        raise CalibrationError(f"source parquet is empty while reading {column}") from exc
    if value is None or len(value) != expected_size:
        size = None if value is None else len(value)
        raise CalibrationError(f"{column} has size {size}, expected {expected_size}")
    first = np.asarray(value, dtype=np.float64)
    if verify_static:
        all_values = list_column_to_numpy(
            parquet_file.read(columns=[column]), column, expected_size
        )
        if not np.array_equal(all_values, np.broadcast_to(first, all_values.shape)):
            different = int(np.any(all_values != first, axis=1).sum())
            raise CalibrationError(
                f"{column} is not episode-static: {different} rows differ from row 0"
            )
    return first


def read_calibration(
    parquet_file: pq.ParquetFile, verify_static: bool
) -> tuple[CalibrationBundle, bool]:
    intrinsic = read_first_list(parquet_file, K_COLUMN, 9, verify_static).reshape(
        3, 3, order="C"
    )
    camera_to_left_base = read_first_list(
        parquet_file, C_COLUMN, 16, verify_static
    ).reshape(4, 4, order="C")
    right_base_to_left_base = read_first_list(
        parquet_file, B_COLUMN, 16, verify_static
    ).reshape(4, 4, order="C")
    signatures = {
        "K": canonical_array_signature(SIGNATURE_LABELS["K"], intrinsic),
        "E": canonical_array_signature(SIGNATURE_LABELS["E"], camera_to_left_base),
        "B": canonical_array_signature(SIGNATURE_LABELS["B"], right_base_to_left_base),
    }
    signatures["combined"] = combined_calibration_signature(signatures)
    cached = _CALIBRATION_CACHE.get(signatures["combined"])
    if cached is not None:
        return cached, True
    qa = {
        "K": validate_intrinsic(intrinsic),
        "E": validate_rigid_transform(
            camera_to_left_base, "E (camera -> left base)"
        ).as_dict(),
        "B": validate_rigid_transform(
            right_base_to_left_base, "B (right base -> left base)"
        ).as_dict(),
    }
    bundle = CalibrationBundle(
        intrinsic=intrinsic,
        camera_to_left_base=camera_to_left_base,
        right_base_to_left_base=right_base_to_left_base,
        left_base_to_camera=np.linalg.inv(camera_to_left_base),
        signatures=signatures,
        qa=qa,
    )
    _CALIBRATION_CACHE[signatures["combined"]] = bundle
    return bundle, False


def inspect_available_calibration(
    parquet_file: pq.ParquetFile,
    present: set[str],
) -> tuple[dict[str, str], dict[str, str]]:
    statuses: dict[str, str] = {}
    signatures: dict[str, str] = {}
    sizes = {"K": 9, "E": 16, "B": 16}
    shapes = {"K": (3, 3), "E": (4, 4), "B": (4, 4)}
    for key, column in CALIBRATION_COLUMNS.items():
        if column not in present:
            statuses[key] = "missing"
            continue
        try:
            matrix = read_first_list(parquet_file, column, sizes[key], False).reshape(
                shapes[key], order="C"
            )
            signatures[key] = canonical_array_signature(SIGNATURE_LABELS[key], matrix)
            if key == "K":
                validate_intrinsic(matrix)
            else:
                validate_rigid_transform(matrix, key)
            statuses[key] = "valid"
        except Exception as exc:  # The report must preserve partial calibration state.
            statuses[key] = f"invalid:{type(exc).__name__}"
    return statuses, signatures


def output_paths(output_root: Path, episode_index: int, chunk_size: int) -> tuple[Path, Path]:
    chunk = episode_index // chunk_size
    stem = f"episode_{episode_index:06d}"
    directory = output_root / f"chunk-{chunk:03d}"
    return directory / f"{stem}.npz", directory / f"{stem}.report.json"


def final_paths(dataset_root: Path, episode_index: int, chunk_size: int) -> tuple[Path, Path]:
    chunk = episode_index // chunk_size
    parquet = dataset_root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    video = (
        dataset_root
        / f"videos/chunk-{chunk:03d}/{CAMERA_KEY}/episode_{episode_index:06d}.mp4"
    )
    return parquet, video


def make_manifest_record(report: dict[str, Any]) -> dict[str, Any]:
    matrix_status = report["calibration_status"]
    if all(value == "valid" for value in matrix_status.values()):
        overall_calibration_status = "valid"
    elif any(value == "missing" for value in matrix_status.values()):
        overall_calibration_status = "missing"
    else:
        overall_calibration_status = "invalid"
    record = {
        "dataset": report["dataset"],
        "episode_index": report["episode_index"],
        "video": report["video"],
        "geometry_npz": report.get("geometry_npz"),
        "geometry_report": report["geometry_report"],
        "calibration_status": overall_calibration_status,
        "calibration_matrices": matrix_status,
        "source_parquet": report["source_parquet"],
        "status": report["status"],
    }
    # Preserve audited override provenance when a scoped backfill report is
    # reused by the normal batch runner.  Ordinary source-calibrated reports
    # remain byte-for-byte schema compatible because these keys are optional.
    for key in (
        "calibration_method",
        "calibration_evidence",
        "calibration_override_version",
        "calibration_signature",
    ):
        if key in report:
            record[key] = report[key]
    override = report.get("calibration_override") or {}
    if "calibration_override_version" not in record and override.get("version"):
        record["calibration_override_version"] = override["version"]
    calibration = report.get("calibration") or {}
    signatures = calibration.get("signatures") or {}
    if "calibration_signature" not in record and signatures.get("combined"):
        record["calibration_signature"] = signatures["combined"]
    return record


def base_report(task: dict[str, Any], final_parquet: Path, video: Path, report_path: Path) -> dict[str, Any]:
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "status": "running",
        "created_at": utc_now(),
        "dataset": _WORKER_CONFIG["dataset_name"],
        "episode_index": task["episode_index"],
        "source_dataset": task.get("source_dataset"),
        "source_episode_index": task.get("source_episode_index"),
        "source_episode_name": task.get("source_episode_name"),
        "source_parquet": task["source_parquet"],
        "final_parquet": str(final_parquet),
        "video": str(video),
        "geometry_npz": None,
        "geometry_report": str(report_path),
        "geometry_only": True,
        "visual_visibility_computed": False,
        "calibration_status": {"K": "unknown", "E": "unknown", "B": "unknown"},
        "matrix_convention": {
            "reshape_order": "C",
            "E": "camera_front -> arm_left_base; inverted for projection",
            "B": "arm_right_base -> arm_left_base",
        },
    }


def report_result(report: dict[str, Any]) -> dict[str, Any]:
    signatures = report.get("calibration", {}).get("signatures", {})
    return {
        "episode_index": report["episode_index"],
        "status": report["status"],
        "manifest": make_manifest_record(report),
        "signatures": signatures,
        "calibration": report.get("calibration"),
        "frames": report.get("frames", 0),
        "elapsed_seconds": report.get("elapsed_seconds", 0.0),
    }


def process_episode(task: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    dataset_root = Path(_WORKER_CONFIG["dataset_root"])
    output_root = Path(_WORKER_CONFIG["output_root"])
    episode_index = int(task["episode_index"])
    chunk_size = int(_WORKER_CONFIG["chunk_size"])
    final_parquet, video = final_paths(dataset_root, episode_index, chunk_size)
    npz_path, report_path = output_paths(output_root, episode_index, chunk_size)
    report = base_report(task, final_parquet, video, report_path)
    source_parquet = Path(task["source_parquet"])
    try:
        report["source_file"] = file_fingerprint(source_parquet)
        report["final_file"] = file_fingerprint(final_parquet)
        report["video_file"] = file_fingerprint(video)
        source_file = pq.ParquetFile(source_parquet)
        source_rows = source_file.metadata.num_rows
        present = set(source_file.schema_arrow.names)
        missing = [
            column for column in CALIBRATION_COLUMNS.values() if column not in present
        ]
        if missing:
            statuses, signatures = inspect_available_calibration(source_file, present)
            report.update(
                {
                    "status": "missing_calibration",
                    "calibration_status": statuses,
                    "missing_calibration_fields": missing,
                    "source_rows": source_rows,
                    "expected_frames_from_provenance": task.get("frames"),
                    "calibration": {
                        "signatures": signatures,
                        "read_policy": "row_0_after_dataset_static-calibration_audit",
                    },
                    "reason": "required source calibration column is absent; no geometry was emitted",
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            atomic_write_json(report_path, report)
            return report_result(report)

        calibration, cache_hit = read_calibration(
            source_file, bool(_WORKER_CONFIG["verify_static_calibration"])
        )
        report["calibration_status"] = {"K": "valid", "E": "valid", "B": "valid"}
        final_table = pq.read_table(final_parquet, columns=["observation.state"])
        states = list_column_to_numpy(final_table, "observation.state", 20)
        frame_count = len(states)
        if source_rows != frame_count:
            raise GeometryError(
                f"provenance row mismatch: source={source_rows}, final={frame_count}"
            )
        if task.get("frames") is not None and int(task["frames"]) != frame_count:
            raise GeometryError(
                f"provenance frames={task['frames']} but final parquet has {frame_count}"
            )
        geometry = reconstruct_geometry(
            states,
            calibration.intrinsic,
            calibration.camera_to_left_base,
            calibration.right_base_to_left_base,
            int(_WORKER_CONFIG["image_width"]),
            int(_WORKER_CONFIG["image_height"]),
            left_base_to_camera=calibration.left_base_to_camera,
        )
        flange_xy_geom, flange_depth, flange_in_fov, flange_xy = (
            prepare_projected_point_output(
                geometry["flange_xy_geom"],
                geometry["flange_depth"],
                int(_WORKER_CONFIG["image_width"]),
                int(_WORKER_CONFIG["image_height"]),
            )
        )
        grasp_xy_geom, grasp_depth, grasp_in_fov, grasp_xy = (
            prepare_projected_point_output(
                geometry["grasp_center_xy_geom"],
                geometry["grasp_center_depth"],
                int(_WORKER_CONFIG["image_width"]),
                int(_WORKER_CONFIG["image_height"]),
            )
        )
        arrays = {
            "algorithm_version": np.asarray(ALGORITHM_VERSION),
            "episode_index": np.asarray(episode_index, dtype=np.int32),
            "frame_index": np.arange(frame_count, dtype=np.int32),
            "operation_mask": np.ones(frame_count, dtype=bool),
            "operation_range": np.asarray([0, frame_count - 1], dtype=np.int32),
            "arm_names": ARM_NAMES,
            "image_size": np.asarray(
                [_WORKER_CONFIG["image_width"], _WORKER_CONFIG["image_height"]],
                dtype=np.int32,
            ),
            "flange_xy_geom": flange_xy_geom,
            "grasp_center_xy_geom": grasp_xy_geom,
            "flange_xy": flange_xy,
            "grasp_center_xy": grasp_xy,
            "tcp_xy": grasp_xy,
            "flange_geometric_in_fov": flange_in_fov,
            "grasp_center_geometric_in_fov": grasp_in_fov,
            "geometric_in_fov": grasp_in_fov,
            "geometric_in_frame": grasp_in_fov,
            "visual_visible": np.zeros_like(grasp_in_fov, dtype=bool),
            "flange_camera_depth": flange_depth,
            "grasp_center_camera_depth": grasp_depth,
            "flange_left_base": geometry["flange_left_base"].astype(np.float32),
            "grasp_center_left_base": geometry["grasp_center_left_base"].astype(np.float32),
            "grasp_center_offset_local": GRASP_CENTER_OFFSET.astype(np.float32),
            "intrinsic": calibration.intrinsic,
            "camera_to_left_base": calibration.camera_to_left_base,
            "left_base_to_camera": calibration.left_base_to_camera,
            "right_base_to_left_base": calibration.right_base_to_left_base,
            "calibration_signature": np.asarray(calibration.signatures["combined"]),
        }
        npz_size = atomic_save_npz(npz_path, arrays)
        elapsed = time.monotonic() - started
        report.update(
            {
                "status": "success",
                "frames": frame_count,
                "source_rows": source_rows,
                "image_size": [
                    int(_WORKER_CONFIG["image_width"]),
                    int(_WORKER_CONFIG["image_height"]),
                ],
                "geometry_npz": str(npz_path),
                "npz_size_bytes": npz_size,
                "calibration": {
                    "signatures": calibration.signatures,
                    "qa": calibration.qa,
                    "cache_hit_in_worker": cache_hit,
                    "read_policy": (
                        "all_rows_verified_static"
                        if _WORKER_CONFIG["verify_static_calibration"]
                        else "row_0_after_dataset_static-calibration_audit"
                    ),
                },
                "point_definition": {
                    "flange_xy": "source Cartesian link6/flange origin",
                    "grasp_center_xy": (
                        "provisional flange-local +Z offset 0.13503 m; "
                        "not a visually fitted fingertip point"
                    ),
                    "geometric_in_fov": (
                        "alias of grasp_center_geometric_in_fov; pinhole bounds and "
                        "positive depth, evaluated after float32 output quantization"
                    ),
                },
                "geometric_in_fov_counts": {
                    "flange": flange_in_fov.sum(axis=0).astype(int).tolist(),
                    "grasp_center": grasp_in_fov.sum(axis=0).astype(int).tolist(),
                },
                "geometry_only_warning": (
                    "geometric_in_fov is not visual visibility; run an occlusion/appearance filter"
                ),
                "operation_range": [0, frame_count - 1],
                "operation_contract": "all frames belong to the final manipulation episode",
                "elapsed_seconds": elapsed,
            }
        )
        atomic_write_json(report_path, report)
        return report_result(report)
    except CalibrationError as exc:
        report.update(
            {
                "status": "invalid_calibration",
                "reason": str(exc),
                "error_type": type(exc).__name__,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    except Exception as exc:
        report.update(
            {
                "status": "error",
                "reason": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    try:
        atomic_write_json(report_path, report)
    except Exception as report_exc:
        return {
            "episode_index": episode_index,
            "status": "fatal_report_write_error",
            "error": f"{type(report_exc).__name__}: {report_exc}",
            "frames": 0,
            "elapsed_seconds": time.monotonic() - started,
        }
    return report_result(report)


def _init_worker(config: dict[str, Any]) -> None:
    global _WORKER_CONFIG, _CALIBRATION_CACHE
    _WORKER_CONFIG = config
    _CALIBRATION_CACHE = {}
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def read_info(dataset_root: Path) -> tuple[dict[str, Any], int, int, int, int]:
    info_path = dataset_root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    total_episodes = int(info["total_episodes"])
    chunk_size = int(info.get("chunks_size", 1000))
    feature = info["features"][CAMERA_KEY]
    camera_info = feature.get("info", {})
    width = int(camera_info.get("video.width", feature["shape"][-1]))
    height = int(camera_info.get("video.height", feature["shape"][-2]))
    return info, total_episodes, chunk_size, width, height


def read_provenance(dataset_root: Path, total_episodes: int) -> dict[int, dict[str, Any]]:
    path = dataset_root / "provenance/episode_source_map.jsonl"
    records: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            episode_index = int(raw["episode_index"])
            if episode_index in records:
                raise ValueError(f"duplicate provenance episode {episode_index} at line {line_number}")
            records[episode_index] = {
                "episode_index": episode_index,
                "source_dataset": raw.get("source_dataset"),
                "source_episode_index": raw.get("source_episode_index"),
                "source_episode_name": raw.get("source_episode_name"),
                "source_parquet": raw["source_parquet"],
                "frames": raw.get("frames"),
            }
    expected = set(range(total_episodes))
    actual = set(records)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"provenance is not a complete 0:{total_episodes} map; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return records


def fingerprints_match(report: dict[str, Any], key: str, path: Path) -> bool:
    saved = report.get(key)
    if not isinstance(saved, dict) or not path.exists():
        return False
    current = file_fingerprint(path)
    return (
        saved.get("size_bytes") == current["size_bytes"]
        and saved.get("mtime_ns") == current["mtime_ns"]
    )


def reusable_report(
    task: dict[str, Any],
    dataset_root: Path,
    output_root: Path,
    chunk_size: int,
) -> dict[str, Any] | None:
    final_parquet, _ = final_paths(dataset_root, task["episode_index"], chunk_size)
    npz_path, report_path = output_paths(output_root, task["episode_index"], chunk_size)
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if report.get("algorithm_version") != ALGORITHM_VERSION:
        return None
    if not fingerprints_match(report, "source_file", Path(task["source_parquet"])):
        return None
    if not fingerprints_match(report, "final_file", final_parquet):
        return None
    _, video = final_paths(dataset_root, task["episode_index"], chunk_size)
    if not fingerprints_match(report, "video_file", video):
        return None
    status = report.get("status")
    if status == "success":
        if not npz_path.is_file() or report.get("npz_size_bytes") != npz_path.stat().st_size:
            return None
    elif status == "missing_calibration":
        if npz_path.exists():
            return None
    else:
        return None
    return report


def load_existing_manifest(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return records
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    records[int(record["episode_index"])] = record
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {}
    return records


def update_signature_catalog(
    catalog: dict[str, dict[str, Any]], result: dict[str, Any]
) -> None:
    signatures = result.get("signatures") or {}
    combined = signatures.get("combined")
    if not combined:
        return
    entry = catalog.setdefault(
        combined,
        {
            "combined_signature": combined,
            "matrix_signatures": {key: signatures[key] for key in ("K", "E", "B")},
            "episode_count": 0,
            "first_episode_index": result["episode_index"],
            "last_episode_index": result["episode_index"],
            "qa": (result.get("calibration") or {}).get("qa"),
        },
    )
    entry["episode_count"] += 1
    entry["first_episode_index"] = min(entry["first_episode_index"], result["episode_index"])
    entry["last_episode_index"] = max(entry["last_episode_index"], result["episode_index"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-project calibrated AgileX7000 dual-arm end-effector geometry."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--stop-episode", type=int, help="Exclusive; defaults to total_episodes")
    parser.add_argument("--episode", type=int, action="append", help="Process only this episode; repeatable")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify-static-calibration",
        action="store_true",
        help="Read every source calibration row and require exact episode-static values (slow).",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else (DEFAULT_OUTPUT_PARENT / dataset_root.name).resolve()
    )
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be >= 1")
    info, total_episodes, chunk_size, width, height = read_info(dataset_root)
    provenance = read_provenance(dataset_root, total_episodes)
    stop = total_episodes if args.stop_episode is None else args.stop_episode
    if args.episode:
        indices = sorted(set(args.episode))
    else:
        if not (0 <= args.start_episode <= stop <= total_episodes):
            raise SystemExit(
                f"invalid range [{args.start_episode}, {stop}) for {total_episodes} episodes"
            )
        indices = list(range(args.start_episode, stop))
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("require --num-shards >= 1 and 0 <= --shard-index < --num-shards")
    indices = [index for index in indices if index % args.num_shards == args.shard_index]
    if any(index < 0 or index >= total_episodes for index in indices):
        raise SystemExit(f"episode selection must be within [0, {total_episodes})")
    if args.max_episodes is not None:
        indices = indices[: args.max_episodes]

    config = {
        "dataset_root": str(dataset_root),
        "dataset_name": dataset_root.name,
        "output_root": str(output_root),
        "chunk_size": chunk_size,
        "image_width": width,
        "image_height": height,
        "verify_static_calibration": args.verify_static_calibration,
    }
    tasks: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for index in indices:
        task = provenance[index]
        reusable = None if args.overwrite else reusable_report(
            task, dataset_root, output_root, chunk_size
        )
        if reusable is None:
            tasks.append(task)
        else:
            result = report_result(reusable)
            result["resumed"] = True
            results.append(result)

    print(
        json.dumps(
            {
                "algorithm_version": ALGORITHM_VERSION,
                "dataset": str(dataset_root),
                "output": str(output_root),
                "selected": len(indices),
                "resumed": len(results),
                "queued": len(tasks),
                "workers": args.workers,
                "image_size": [width, height],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    started = time.monotonic()
    completed_new = 0
    if args.workers == 1:
        _init_worker(config)
        iterator = map(process_episode, tasks)
        for result in iterator:
            results.append(result)
            completed_new += 1
            if completed_new % args.progress_every == 0 or completed_new == len(tasks):
                counts = Counter(item["status"] for item in results)
                print(
                    f"completed={len(results)}/{len(indices)} new={completed_new}/{len(tasks)} "
                    f"status={dict(counts)}",
                    flush=True,
                )
    elif tasks:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=args.workers, initializer=_init_worker, initargs=(config,)
        ) as pool:
            for result in pool.imap_unordered(process_episode, tasks, chunksize=1):
                results.append(result)
                completed_new += 1
                if completed_new % args.progress_every == 0 or completed_new == len(tasks):
                    counts = Counter(item["status"] for item in results)
                    print(
                        f"completed={len(results)}/{len(indices)} new={completed_new}/{len(tasks)} "
                        f"status={dict(counts)}",
                        flush=True,
                    )

    results.sort(key=lambda item: item["episode_index"])
    manifest_path = output_root / "geometry_manifest.jsonl"
    manifest = load_existing_manifest(manifest_path)
    for result in results:
        if "manifest" in result:
            manifest[result["episode_index"]] = result["manifest"]
    atomic_write_jsonl(manifest_path, (manifest[index] for index in sorted(manifest)))

    signature_catalog: dict[str, dict[str, Any]] = {}
    for result in results:
        update_signature_catalog(signature_catalog, result)
    signature_payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "dataset": str(dataset_root),
        "selection": {
            "episodes": len(indices),
            "start": min(indices) if indices else None,
            "stop_inclusive": max(indices) if indices else None,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
        "unique_combined_signatures": len(signature_catalog),
        "unique_matrix_signatures": {
            key: len(
                {
                    entry["matrix_signatures"][key]
                    for entry in signature_catalog.values()
                }
            )
            for key in ("K", "E", "B")
        },
        "calibrations": sorted(
            signature_catalog.values(), key=lambda entry: entry["first_episode_index"]
        ),
    }
    suffix = (
        ""
        if args.num_shards == 1
        else f".shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    )
    atomic_write_json(
        output_root / f"calibration_signatures{suffix}.json", signature_payload
    )

    counts = Counter(item["status"] for item in results)
    missing_indices = [
        item["episode_index"] for item in results if item["status"] == "missing_calibration"
    ]
    elapsed = time.monotonic() - started
    summary = {
        "algorithm_version": ALGORITHM_VERSION,
        "status": (
            "failed"
            if any(key in counts for key in ("error", "invalid_calibration", "fatal_report_write_error"))
            else "completed_with_missing_calibration"
            if counts.get("missing_calibration", 0)
            else "success"
        ),
        "dataset": str(dataset_root),
        "output_root": str(output_root),
        "selected_episodes": len(indices),
        "queued_episodes": len(tasks),
        "resumed_episodes": len(results) - len(tasks),
        "status_counts": dict(sorted(counts.items())),
        "frames_projected": int(
            sum(item.get("frames", 0) for item in results if item["status"] == "success")
        ),
        "missing_calibration_episode_indices": missing_indices,
        "known_dataset_gap_11910_11969_matches": (
            missing_indices == list(range(11910, 11970))
            if indices == list(range(total_episodes))
            else None
        ),
        "geometry_manifest": str(manifest_path),
        "elapsed_seconds": elapsed,
        "episodes_per_second_for_queued_work": len(tasks) / max(elapsed, 1e-9),
        "generated_at": utc_now(),
    }
    atomic_write_json(output_root / f"run_summary{suffix}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 2 if summary["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
