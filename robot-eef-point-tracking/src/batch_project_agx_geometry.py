#!/usr/bin/env python3
"""Batch-project AGX J6/flange and grasp-center states into the main camera."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np
import pyarrow.parquet as pq

from agx_calibration_registry import (
    Calibration,
    CalibrationRegistry,
    DATASET_SPECS,
    DEFAULT_CALIBRATION_ROOT,
    DatasetSpec,
    read_jsonl,
    resolve_dataset_spec,
)


ALGORITHM_VERSION = "agx_geometry_projection_v1.3.0"
SCHEMA_VERSION = 1
ARM_MOUNTS = np.asarray(
    [[0.23875, 0.3, 0.775], [0.23875, -0.3, 0.775]], dtype=np.float64
)
GRASP_CENTER_OFFSET = np.asarray([0.0, 0.0, 0.13503], dtype=np.float64)
ARM_NAMES = np.asarray(["left", "right"])


@dataclass(frozen=True)
class WorkItem:
    dataset_key: str
    dataset_name: str
    dataset_root: str
    episode_index: int
    source_episode_index: int
    expected_frames: int
    parquet_path: str
    video_path: str
    image_size: tuple[int, int]
    operation_start: int
    phase_status: str
    phase_metadata_path: str | None
    output_npz: str
    output_report: str


@dataclass(frozen=True)
class WorkerConfig:
    calibration_root: str
    overwrite: bool
    verify_existing: bool


_WORKER_CONFIG: WorkerConfig | None = None
_WORKER_REGISTRY: CalibrationRegistry | None = None


def rot6d_rows_to_matrix(values: np.ndarray) -> np.ndarray:
    """Decode first two row vectors using Gram-Schmidt normalization."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != 6:
        raise ValueError(f"rot6d last dimension must be 6, got {values.shape}")
    first = values[..., :3]
    second = values[..., 3:6]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-9):
        raise ValueError("rot6d contains a degenerate first row")
    first = first / first_norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-9):
        raise ValueError("rot6d contains collinear rows")
    second = second / second_norm
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-2)


def camera_project(
    points_footprint: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_footprint, dtype=np.float64)
    hom = np.concatenate((points, np.ones((*points.shape[:-1], 1))), axis=-1)
    camera = np.einsum("ij,...j->...i", extrinsic, hom)[..., :3]
    image_hom = np.einsum("ij,...j->...i", intrinsic, camera)
    depth = camera[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = image_hom[..., :2] / image_hom[..., 2:3]
    xy[~np.isfinite(xy).all(axis=-1)] = np.nan
    return xy.astype(np.float32), depth.astype(np.float32)


def geometric_in_fov(
    xy: np.ndarray, depth: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray:
    width, height = image_size
    return (
        np.isfinite(xy).all(axis=-1)
        & np.isfinite(depth)
        & (depth > 1e-6)
        & (xy[..., 0] >= 0.0)
        & (xy[..., 0] < width)
        & (xy[..., 1] >= 0.0)
        & (xy[..., 1] < height)
    )


def compute_geometry(
    states: np.ndarray,
    calibration: Calibration,
    image_size: tuple[int, int],
    operation_start: int,
) -> dict[str, np.ndarray]:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 23:
        raise ValueError(f"expected observation.state [T,23], got {states.shape}")
    frame_count = len(states)
    flange_footprint = np.empty((frame_count, 2, 3), dtype=np.float64)
    grasp_footprint = np.empty_like(flange_footprint)
    for arm, (xyz_start, rotation_start) in enumerate(((3, 6), (13, 16))):
        position = states[:, xyz_start : xyz_start + 3]
        rotation = rot6d_rows_to_matrix(states[:, rotation_start : rotation_start + 6])
        flange_footprint[:, arm] = position + ARM_MOUNTS[arm]
        grasp_footprint[:, arm] = flange_footprint[:, arm] + np.einsum(
            "tij,j->ti", rotation, GRASP_CENTER_OFFSET
        )

    flange_raw, flange_depth = camera_project(
        flange_footprint, calibration.intrinsic, calibration.extrinsic
    )
    grasp_raw, grasp_depth = camera_project(
        grasp_footprint, calibration.intrinsic, calibration.extrinsic
    )
    flange_fov = geometric_in_fov(flange_raw, flange_depth, image_size)
    grasp_fov = geometric_in_fov(grasp_raw, grasp_depth, image_size)

    operation_mask = np.zeros(frame_count, dtype=bool)
    if 0 <= operation_start < frame_count:
        operation_mask[operation_start:] = True
    flange_xy = flange_raw.copy()
    grasp_xy = grasp_raw.copy()
    flange_xy[~(operation_mask[:, None] & flange_fov)] = np.nan
    grasp_xy[~(operation_mask[:, None] & grasp_fov)] = np.nan

    # ``geometric_in_fov`` intentionally remains independent of operation_mask.
    return {
        "frame_index": np.arange(frame_count, dtype=np.int32),
        "flange_xy": flange_xy,
        "grasp_center_xy": grasp_xy,
        "tcp_xy": grasp_xy.copy(),
        "flange_xy_geom": flange_raw.copy(),
        "grasp_center_xy_geom": grasp_raw.copy(),
        "flange_projected_xy": flange_raw,
        "grasp_center_projected_xy": grasp_raw,
        "geometric_in_fov": grasp_fov,
        "flange_geometric_in_fov": flange_fov,
        "grasp_center_geometric_in_fov": grasp_fov.copy(),
        "operation_mask": operation_mask,
        "operation_geometric_in_fov": operation_mask[:, None] & grasp_fov,
        "operation_flange_geometric_in_fov": operation_mask[:, None] & flange_fov,
        "flange_camera_depth": flange_depth,
        "grasp_center_camera_depth": grasp_depth,
        "flange_depth": flange_depth.copy(),
        "grasp_center_depth": grasp_depth.copy(),
        "flange_footprint": flange_footprint.astype(np.float32),
        "grasp_center_footprint": grasp_footprint.astype(np.float32),
    }


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        digest = _sha256_file(temporary)
        os.replace(temporary, path)
        return digest
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_states(item: WorkItem) -> np.ndarray:
    parquet_path = Path(item.parquet_path)
    parquet_file = pq.ParquetFile(parquet_path)
    available = set(parquet_file.schema_arrow.names)
    columns = ["observation.state"]
    if "source_episode_index" in available:
        columns.append("source_episode_index")
    table = pq.read_table(parquet_path, columns=columns, use_threads=False)
    state_column = table.column("observation.state").combine_chunks()
    state_dim = state_column.type.list_size
    states = state_column.values.to_numpy().reshape(-1, state_dim).astype(np.float64)
    if len(states) != item.expected_frames:
        raise ValueError(
            f"{parquet_path}: meta length {item.expected_frames} != parquet rows {len(states)}"
        )
    if "source_episode_index" in columns:
        source_values = table.column("source_episode_index").to_numpy()
        if not np.all(source_values == item.source_episode_index):
            unique = np.unique(source_values).tolist()
            raise ValueError(
                f"{parquet_path}: source ids {unique} disagree with final episodes metadata "
                f"({item.source_episode_index})"
            )
    return states


def _existing_is_current(
    item: WorkItem, calibration: Calibration, verify_sha256: bool
) -> bool:
    npz_path = Path(item.output_npz)
    report_path = Path(item.output_report)
    if not npz_path.is_file() or not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected = (
            report.get("status") == "complete"
            and report.get("algorithm_version") == ALGORITHM_VERSION
            and report.get("schema_version") == SCHEMA_VERSION
            and report.get("dataset") == item.dataset_name
            and int(report.get("episode_index", -1)) == item.episode_index
            and int(report.get("source_episode_index", -1))
            == item.source_episode_index
            and report.get("calibration", {}).get("id")
            == calibration.calibration_id
            and report.get("calibration", {}).get("sha256")
            == calibration.calibration_sha256
        )
        if not expected:
            return False
        if verify_sha256 and report.get("output_npz_sha256") != _sha256_file(npz_path):
            return False
        with np.load(npz_path, allow_pickle=False) as archive:
            return (
                str(archive["algorithm_version"].item()) == ALGORITHM_VERSION
                and int(archive["episode_index"].item()) == item.episode_index
                and int(archive["source_episode_index"].item())
                == item.source_episode_index
                and str(archive["calibration_id"].item())
                == calibration.calibration_id
                and str(archive["generation_id"].item())
                == str(report.get("generation_id"))
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def project_episode(
    item: WorkItem,
    registry: CalibrationRegistry,
    overwrite: bool = False,
    verify_existing: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    calibration = registry.select(item.dataset_key, item.source_episode_index)
    if not overwrite and _existing_is_current(item, calibration, verify_existing):
        return {
            "status": "skipped",
            "dataset": item.dataset_name,
            "episode_index": item.episode_index,
            "source_episode_index": item.source_episode_index,
            "output_npz": item.output_npz,
        }

    states = _read_states(item)
    arrays = compute_geometry(
        states, calibration, item.image_size, item.operation_start
    )
    generation_id = uuid.uuid4().hex
    operation_mask = arrays["operation_mask"]
    flange_fov = arrays["flange_geometric_in_fov"]
    grasp_fov = arrays["geometric_in_fov"]
    operation_range = np.asarray(
        [item.operation_start, len(states) - 1]
        if operation_mask.any()
        else [-1, -1],
        dtype=np.int32,
    )
    arrays.update(
        {
            "image_size": np.asarray(item.image_size, dtype=np.int32),
            "arm_names": ARM_NAMES,
            "point_names": np.asarray(["flange_j6", "grasp_center"]),
            "operation_range": operation_range,
            "intrinsic": calibration.intrinsic,
            "extrinsic": calibration.extrinsic,
            "arm_mounts": ARM_MOUNTS,
            "grasp_center_offset": GRASP_CENTER_OFFSET,
            "algorithm_version": np.asarray(ALGORITHM_VERSION),
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
            "dataset": np.asarray(item.dataset_name),
            "episode_index": np.asarray(item.episode_index, dtype=np.int32),
            "source_episode_index": np.asarray(
                item.source_episode_index, dtype=np.int32
            ),
            "calibration_id": np.asarray(calibration.calibration_id),
            "calibration_sha256": np.asarray(calibration.calibration_sha256),
            "calibration_provisional": np.asarray(calibration.provisional),
            "generation_id": np.asarray(generation_id),
        }
    )

    npz_path = Path(item.output_npz)
    output_digest = _atomic_save_npz(npz_path, arrays)
    elapsed = time.monotonic() - started
    report = {
        "status": "complete",
        "algorithm_version": ALGORITHM_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generation_id": generation_id,
        "dataset": item.dataset_name,
        "dataset_key": item.dataset_key,
        "episode_index": item.episode_index,
        "source_episode_index": item.source_episode_index,
        "frames": len(states),
        "image_size": list(item.image_size),
        "input_parquet": item.parquet_path,
        "calibration": {
            "id": calibration.calibration_id,
            "unit": calibration.calibration_unit,
            "path": str(calibration.calibration_path),
            "sha256": calibration.calibration_sha256,
            "yaml_status": calibration.yaml_status,
            "manifest_status": calibration.manifest_status,
            "selection_mode": calibration.selection_mode,
            "selection_key": "source_episode_index",
            "provisional": calibration.provisional,
            "provisional_reason": calibration.provisional_reason,
            "method": calibration.calibration_method,
            "evidence": calibration.evidence,
        },
        "phase": {
            "metadata_path": item.phase_metadata_path,
            "status": item.phase_status,
            "operation_start": item.operation_start,
            "operation_frames": int(operation_mask.sum()),
            "excluded_navigation_or_unusable_frames": int((~operation_mask).sum()),
        },
        "point_definition": {
            "flange_xy": "Piper J6 / fl_link6 / fr_link6 origin",
            "grasp_center_xy": (
                "flange-local +Z 0.13503 m; tcp_xy is an exact compatibility alias"
            ),
        },
        "field_contract": {
            "geometric_in_fov": (
                "grasp-center positive camera depth and pixel inside image; independent "
                "of operation_mask and visual occlusion"
            ),
            "flange_geometric_in_fov": (
                "J6/flange positive camera depth and pixel inside image; independent "
                "of operation_mask and visual occlusion"
            ),
            "operation_mask": "false for navigation or unusable phase frames",
            "flange_xy_and_grasp_center_xy": (
                "operation-only, independently FOV-masked coordinates; raw all-frame "
                "projections are retained in *_xy_geom and *_projected_xy"
            ),
        },
        "counts": {
            "flange_geometric_in_fov": flange_fov.sum(axis=0).tolist(),
            "grasp_center_geometric_in_fov": grasp_fov.sum(axis=0).tolist(),
            "operation_flange_geometric_in_fov": (
                operation_mask[:, None] & flange_fov
            ).sum(axis=0).tolist(),
            "operation_grasp_center_geometric_in_fov": (
                operation_mask[:, None] & grasp_fov
            ).sum(axis=0).tolist(),
        },
        "output_npz": str(npz_path),
        "output_npz_sha256": output_digest,
        "elapsed_seconds": elapsed,
    }
    # The report is the commit marker and is replaced only after the NPZ is durable.
    _atomic_write_json(Path(item.output_report), report)
    return {
        "status": "completed",
        "dataset": item.dataset_name,
        "episode_index": item.episode_index,
        "source_episode_index": item.source_episode_index,
        "calibration_id": calibration.calibration_id,
        "provisional": calibration.provisional,
        "frames": len(states),
        "operation_frames": int(operation_mask.sum()),
        "elapsed_seconds": elapsed,
        "output_npz": item.output_npz,
    }


def _phase_by_episode(spec: DatasetSpec, dataset_root: Path) -> tuple[dict[int, dict], str | None]:
    if spec.phase_file is None:
        return {}, None
    phase_path = dataset_root / spec.phase_file
    if not phase_path.is_file():
        raise FileNotFoundError(phase_path)
    rows = read_jsonl(phase_path)
    result: dict[int, dict] = {}
    for row in rows:
        episode_index = int(row["episode_index"])
        if episode_index in result:
            raise ValueError(f"duplicate phase entry {episode_index} in {phase_path}")
        result[episode_index] = row
    return result, str(phase_path)


def _operation_start(
    spec: DatasetSpec, episode_index: int, frame_count: int, phase_rows: dict[int, dict]
) -> tuple[int, str]:
    if spec.phase_file is None:
        return 0, "manipulation_only_dataset"
    if episode_index not in phase_rows:
        raise KeyError(f"missing phase metadata for {spec.key} episode {episode_index}")
    row = phase_rows[episode_index]
    status = str(row.get("status", "unknown"))
    phase_frame_count = int(row.get("frame_count", -1))
    if status in {"ok", "no_navigation_motion"} and phase_frame_count != frame_count:
        raise ValueError(
            f"phase frame count mismatch for {spec.key} episode {episode_index}"
        )
    if status == "no_navigation_motion":
        return 0, status
    if status == "ok":
        split = int(row["split_frame"])
        if not 0 <= split < frame_count:
            raise ValueError(
                f"invalid split_frame {split} for {spec.key} episode {episode_index}"
            )
        return split, status
    # Phase-split rejected episodes (currently one insufficient-terminal-static Cup
    # episode) are excluded instead of accidentally labeling navigation as operation.
    if phase_frame_count != frame_count:
        status = f"{status}_frame_count_mismatch"
    return -1, status


def build_work_items(
    data_root: Path,
    output_root: Path,
    dataset_values: list[str],
    registry: CalibrationRegistry,
    episode_indices: set[int] | None = None,
    limit: int | None = None,
) -> tuple[list[WorkItem], list[dict[str, Any]]]:
    items: list[WorkItem] = []
    audits: list[dict[str, Any]] = []
    seen_dataset_keys: set[str] = set()
    for value in dataset_values:
        spec = resolve_dataset_spec(value)
        if spec.key in seen_dataset_keys:
            raise ValueError(f"dataset {spec.key!r} was requested more than once")
        seen_dataset_keys.add(spec.key)
        dataset_root = data_root / spec.dataset_name
        info = json.loads((dataset_root / "meta" / "info.json").read_text("utf-8"))
        episodes_path = dataset_root / "meta" / "episodes.jsonl"
        episodes = read_jsonl(episodes_path)
        phase_rows, phase_path = _phase_by_episode(spec, dataset_root)
        camera_key = "observation.images.cam_manip_high"
        camera_feature = info["features"][camera_key]
        image_size = (int(camera_feature["shape"][2]), int(camera_feature["shape"][1]))
        chunk_size = int(info.get("chunks_size", 1000))
        calibration_counts: dict[str, int] = {}
        phase_counts: dict[str, int] = {}
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            if episode_indices is not None and episode_index not in episode_indices:
                continue
            if "source_episode_index" not in episode:
                raise ValueError(f"source_episode_index missing in {episodes_path}")
            source_episode_index = int(episode["source_episode_index"])
            calibration = registry.select(spec.key, source_episode_index)
            calibration_counts[calibration.calibration_id] = (
                calibration_counts.get(calibration.calibration_id, 0) + 1
            )
            frame_count = int(episode["length"])
            operation_start, phase_status = _operation_start(
                spec, episode_index, frame_count, phase_rows
            )
            phase_counts[phase_status] = phase_counts.get(phase_status, 0) + 1
            chunk = episode_index // chunk_size
            parquet_path = dataset_root / str(info["data_path"]).format(
                episode_chunk=chunk, episode_index=episode_index
            )
            if not parquet_path.is_file():
                raise FileNotFoundError(parquet_path)
            video_path = dataset_root / str(info["video_path"]).format(
                episode_chunk=chunk,
                episode_index=episode_index,
                video_key=camera_key,
            )
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            output_dir = output_root / spec.dataset_name / f"chunk-{chunk:03d}"
            items.append(
                WorkItem(
                    dataset_key=spec.key,
                    dataset_name=spec.dataset_name,
                    dataset_root=str(dataset_root),
                    episode_index=episode_index,
                    source_episode_index=source_episode_index,
                    expected_frames=frame_count,
                    parquet_path=str(parquet_path),
                    video_path=str(video_path),
                    image_size=image_size,
                    operation_start=operation_start,
                    phase_status=phase_status,
                    phase_metadata_path=phase_path,
                    output_npz=str(output_dir / f"episode_{episode_index:06d}.npz"),
                    output_report=str(
                        output_dir / f"episode_{episode_index:06d}.report.json"
                    ),
                )
            )
            if limit is not None and len(items) >= limit:
                break
        audits.append(
            {
                "dataset_key": spec.key,
                "dataset_name": spec.dataset_name,
                "selected_episodes": sum(calibration_counts.values()),
                "calibration_counts": calibration_counts,
                "phase_status_counts": phase_counts,
                "provisional": spec.provisional,
                "provisional_reason": spec.provisional_reason,
                "episodes_metadata": str(episodes_path),
                "phase_metadata": phase_path,
            }
        )
        if limit is not None and len(items) >= limit:
            break
    return items, audits


def geometry_manifest_rows(
    items: list[WorkItem],
    registry: CalibrationRegistry,
    final_output_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for work_index, item in enumerate(items):
        calibration = registry.select(item.dataset_key, item.source_episode_index)
        raw_statuses = [calibration.yaml_status, calibration.manifest_status or ""]
        normalized_status = "invalid"
        if calibration.provisional:
            normalized_status = "provisional"
        elif all("INVALID" not in value.upper() for value in raw_statuses) and any(
            token in value.upper()
            for value in raw_statuses
            for token in ("VALIDATED", "CANARY_PASS")
        ):
            normalized_status = "validated"
        chunk_name = Path(item.output_npz).parent.name
        final_dir = final_output_root / item.dataset_name / chunk_name
        rows.append(
            {
                "work_index": work_index,
                "algorithm_version": ALGORITHM_VERSION,
                "dataset": item.dataset_name,
                "dataset_key": item.dataset_key,
                "episode_index": item.episode_index,
                "source_episode_index": item.source_episode_index,
                "frame_count": item.expected_frames,
                "image_size": list(item.image_size),
                "camera_key": "observation.images.cam_manip_high",
                "video": item.video_path,
                "geometry_npz": item.output_npz,
                "geometry_report": item.output_report,
                "output_npz": str(
                    final_dir / f"episode_{item.episode_index:06d}.npz"
                ),
                "output_report": str(
                    final_dir / f"episode_{item.episode_index:06d}.report.json"
                ),
                "calibration_id": calibration.calibration_id,
                "calibration_status": normalized_status,
                "calibration_yaml_status": calibration.yaml_status,
                "calibration_provisional": calibration.provisional,
                "calibration_provisional_reason": calibration.provisional_reason,
                "calibration_method": calibration.calibration_method,
                "calibration_evidence": calibration.evidence,
                "phase_status": item.phase_status,
                "operation_start": item.operation_start,
            }
        )
    return rows


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _init_worker(config: WorkerConfig) -> None:
    global _WORKER_CONFIG, _WORKER_REGISTRY
    _WORKER_CONFIG = config
    _WORKER_REGISTRY = CalibrationRegistry(Path(config.calibration_root))


def _worker(item: WorkItem) -> dict[str, Any]:
    assert _WORKER_CONFIG is not None and _WORKER_REGISTRY is not None
    try:
        return project_episode(
            item,
            _WORKER_REGISTRY,
            overwrite=_WORKER_CONFIG.overwrite,
            verify_existing=_WORKER_CONFIG.verify_existing,
        )
    except Exception as error:
        return {
            "status": "failed",
            "dataset": item.dataset_name,
            "episode_index": item.episode_index,
            "source_episode_index": item.source_episode_index,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _parse_episode_indices(values: list[str] | None) -> set[int] | None:
    if not values:
        return None
    result: set[int] = set()
    for value in values:
        if ":" in value:
            start_text, end_text = value.split(":", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end <= start:
                raise ValueError(f"empty episode range {value!r}")
            result.update(range(start, end))
        else:
            result.add(int(value))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Project final AGX Cartesian states using source-episode calibrations."
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/lerobot")
    )
    parser.add_argument(
        "--calibration-root", type=Path, default=DEFAULT_CALIBRATION_ROOT
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/calibrated_geometry_inputs"),
    )
    parser.add_argument(
        "--final-output-root",
        type=Path,
        default=Path("outputs/eef_tracks"),
        help="Suggested downstream DINO-filter output root recorded in the manifest.",
    )
    parser.add_argument(
        "--geometry-manifest",
        type=Path,
        help="Default: <output-root>/geometry_manifest.jsonl.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_SPECS),
        help="Short keys (cup white color ordered) or exact final dataset names.",
    )
    parser.add_argument(
        "--episode",
        action="append",
        help="Final episode id or half-open range (for example 0 or 10:20).",
    )
    parser.add_argument("--limit", type=int, help="Global work-item limit for smoke tests.")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--chunksize", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.workers < 1 or args.chunksize < 1:
        parser.error("--workers and --chunksize must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        selected_episodes = _parse_episode_indices(args.episode)
        registry = CalibrationRegistry(args.calibration_root)
        items, audits = build_work_items(
            args.data_root.resolve(),
            args.output_root.resolve(),
            args.datasets,
            registry,
            selected_episodes,
            args.limit,
        )
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))

    audit_payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "work_items": len(items),
        "datasets": audits,
    }
    if args.audit_only:
        print(json.dumps(audit_payload, indent=2, sort_keys=True))
        return 0

    args.output_root = args.output_root.resolve()
    args.final_output_root = args.final_output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    geometry_manifest = (
        args.geometry_manifest or (args.output_root / "geometry_manifest.jsonl")
    ).resolve()
    manifest_rows = geometry_manifest_rows(items, registry, args.final_output_root)
    planned_manifest = geometry_manifest.with_name(
        f"{geometry_manifest.stem}.planned{geometry_manifest.suffix}"
    )
    _atomic_write_jsonl(planned_manifest, manifest_rows)
    _atomic_write_json(
        geometry_manifest.with_suffix(".summary.json"),
        {
            **audit_payload,
            "geometry_manifest": str(geometry_manifest),
            "planned_geometry_manifest": str(planned_manifest),
            "geometry_output_root": str(args.output_root),
            "suggested_final_output_root": str(args.final_output_root),
        },
    )
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    config = WorkerConfig(
        calibration_root=str(args.calibration_root),
        overwrite=args.overwrite,
        verify_existing=args.verify_existing,
    )
    if args.workers == 1:
        _init_worker(config)
        iterator = map(_worker, items)
        for done, result in enumerate(iterator, start=1):
            results.append(result)
            print(json.dumps({"progress": f"{done}/{len(items)}", **result}), flush=True)
            if args.fail_fast and result["status"] == "failed":
                break
    else:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=args.workers, initializer=_init_worker, initargs=(config,)
        ) as pool:
            iterator = pool.imap_unordered(_worker, items, chunksize=args.chunksize)
            for done, result in enumerate(iterator, start=1):
                results.append(result)
                print(
                    json.dumps({"progress": f"{done}/{len(items)}", **result}),
                    flush=True,
                )
                if args.fail_fast and result["status"] == "failed":
                    pool.terminate()
                    break

    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    failures = [result for result in results if result["status"] == "failed"]
    ready_keys = {
        (str(result["dataset"]), int(result["episode_index"]))
        for result in results
        if result["status"] in {"completed", "skipped"}
    }
    ready_manifest_rows = [
        row
        for row in manifest_rows
        if (str(row["dataset"]), int(row["episode_index"])) in ready_keys
    ]
    _atomic_write_jsonl(geometry_manifest, ready_manifest_rows)
    summary = {
        **audit_payload,
        "geometry_manifest": str(geometry_manifest),
        "manifest_ready_episodes": len(ready_manifest_rows),
        "status": "failed" if failures else "complete",
        "status_counts": status_counts,
        "elapsed_seconds": time.monotonic() - started,
        "failures": failures,
    }
    _atomic_write_json(args.output_root / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
