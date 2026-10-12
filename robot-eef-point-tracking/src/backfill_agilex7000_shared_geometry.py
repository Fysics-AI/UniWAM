#!/usr/bin/env python3
"""Backfill the 60 AgileX7000 episodes with a validated shared camera E.

This tool is intentionally scoped to final episodes 11910..11969.  K and B
must still be present and exactly match the override contract in every source
episode.  E is injected only after validating the right-base invariant audit.
The geometry manifest is published only after all 60 outputs pass a second,
independent on-disk validation pass.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq

import batch_project_agilex7000_geometry as core


BACKFILL_VERSION = "agilex7000-shared-e-backfill-v1"
DEFAULT_OVERRIDE = (
    Path(__file__).resolve().parent.parent
    / "configs/overrides/agilex7000_hanging_shared_right_base_invariant.json"
)
REQUIRED_ARRAYS = {
    "frame_index", "operation_mask", "operation_range", "arm_names", "image_size",
    "flange_xy_geom", "grasp_center_xy_geom", "flange_xy", "grasp_center_xy", "tcp_xy",
    "flange_geometric_in_fov", "grasp_center_geometric_in_fov", "geometric_in_fov",
    "geometric_in_frame", "visual_visible", "flange_camera_depth", "grasp_center_camera_depth",
    "flange_left_base", "grasp_center_left_base", "grasp_center_offset_local", "intrinsic",
    "camera_to_left_base", "left_base_to_camera", "right_base_to_left_base",
    "calibration_signature", "calibration_method", "calibration_override_version",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_records_digest(records: dict[int, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for index in sorted(records):
        digest.update(str(index).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(records[index], sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_override(path: Path, dataset_name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset") != dataset_name:
        raise ValueError(f"override dataset {payload.get('dataset')!r} != {dataset_name!r}")
    if payload.get("calibration_status") != "valid":
        raise ValueError("override calibration_status must be 'valid'")
    if payload.get("calibration_method") != "shared_right_base_invariant":
        raise ValueError("unexpected calibration_method")
    intrinsic = np.asarray(payload["intrinsic"], dtype=np.float64)
    right_to_left = np.asarray(payload["right_base_to_left_base"], dtype=np.float64)
    camera_to_right = np.asarray(payload["camera_to_right_base"], dtype=np.float64)
    camera_to_left = np.asarray(payload["camera_to_left_base"], dtype=np.float64)
    core.validate_intrinsic(intrinsic)
    core.validate_rigid_transform(right_to_left, "override B")
    core.validate_rigid_transform(camera_to_right, "override camera_to_right")
    core.validate_rigid_transform(camera_to_left, "override E")
    if not np.allclose(right_to_left @ camera_to_right, camera_to_left, atol=1e-12, rtol=0):
        raise ValueError("override E is not exactly B @ camera_to_right within 1e-12")
    evidence = Path(__file__).resolve().parent.parent / payload["evidence"]["initializer_report"]
    expected_sha = payload["evidence"]["initializer_report_sha256"]
    if not evidence.is_file() or sha256_file(evidence) != expected_sha:
        raise ValueError("override evidence file is absent or its SHA-256 changed")
    payload["_path"] = str(path.resolve())
    payload["_sha256"] = sha256_file(path)
    payload["_intrinsic"] = intrinsic
    payload["_right_to_left"] = right_to_left
    payload["_camera_to_right"] = camera_to_right
    payload["_camera_to_left"] = camera_to_left
    return payload


def calibration_bundle(override: dict[str, Any]) -> core.CalibrationBundle:
    intrinsic = override["_intrinsic"]
    camera_to_left = override["_camera_to_left"]
    right_to_left = override["_right_to_left"]
    signatures = {
        "K": core.canonical_array_signature(core.SIGNATURE_LABELS["K"], intrinsic),
        "E": core.canonical_array_signature(core.SIGNATURE_LABELS["E"], camera_to_left),
        "B": core.canonical_array_signature(core.SIGNATURE_LABELS["B"], right_to_left),
    }
    signatures["combined"] = core.combined_calibration_signature(signatures)
    qa = {
        "K": core.validate_intrinsic(intrinsic),
        "E": core.validate_rigid_transform(camera_to_left, "override E").as_dict(),
        "B": core.validate_rigid_transform(right_to_left, "source/override B").as_dict(),
    }
    return core.CalibrationBundle(
        intrinsic=intrinsic,
        camera_to_left_base=camera_to_left,
        right_base_to_left_base=right_to_left,
        left_base_to_camera=np.linalg.inv(camera_to_left),
        signatures=signatures,
        qa=qa,
    )


def video_metadata(path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"could not open video {path}")
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def build_arrays(
    episode_index: int,
    states: np.ndarray,
    calibration: core.CalibrationBundle,
    image_width: int,
    image_height: int,
    override: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    geometry = core.reconstruct_geometry(
        states, calibration.intrinsic, calibration.camera_to_left_base,
        calibration.right_base_to_left_base, image_width, image_height,
        left_base_to_camera=calibration.left_base_to_camera,
    )
    flange_xy_geom, flange_depth, flange_in_fov, flange_xy = core.prepare_projected_point_output(
        geometry["flange_xy_geom"], geometry["flange_depth"], image_width, image_height,
    )
    grasp_xy_geom, grasp_depth, grasp_in_fov, grasp_xy = core.prepare_projected_point_output(
        geometry["grasp_center_xy_geom"], geometry["grasp_center_depth"], image_width, image_height,
    )
    frame_count = len(states)
    arrays = {
        "algorithm_version": np.asarray(core.ALGORITHM_VERSION),
        "backfill_version": np.asarray(BACKFILL_VERSION),
        "episode_index": np.asarray(episode_index, dtype=np.int32),
        "frame_index": np.arange(frame_count, dtype=np.int32),
        "operation_mask": np.ones(frame_count, dtype=bool),
        "operation_range": np.asarray([0, frame_count - 1], dtype=np.int32),
        "arm_names": core.ARM_NAMES,
        "image_size": np.asarray([image_width, image_height], dtype=np.int32),
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
        "grasp_center_offset_local": core.GRASP_CENTER_OFFSET.astype(np.float32),
        "intrinsic": calibration.intrinsic,
        "camera_to_left_base": calibration.camera_to_left_base,
        "left_base_to_camera": calibration.left_base_to_camera,
        "right_base_to_left_base": calibration.right_base_to_left_base,
        "calibration_signature": np.asarray(calibration.signatures["combined"]),
        "calibration_method": np.asarray(override["calibration_method"]),
        "calibration_override_version": np.asarray(override["override_version"]),
    }
    counts = {
        "flange": flange_in_fov.sum(axis=0).astype(int).tolist(),
        "grasp_center": grasp_in_fov.sum(axis=0).astype(int).tolist(),
    }
    return arrays, counts


def reusable_success(report_path: Path, npz_path: Path, override: dict[str, Any]) -> bool:
    if not report_path.is_file() or not npz_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return (
            report.get("status") == "success"
            and report.get("calibration_method") == override["calibration_method"]
            and report.get("calibration_override", {}).get("sha256") == override["_sha256"]
            and report.get("geometry_npz") == str(npz_path)
        )
    except Exception:
        return False


def project_episode(
    task: dict[str, Any], dataset_root: Path, output_root: Path, chunk_size: int,
    image_width: int, image_height: int, override: dict[str, Any], overwrite: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    episode_index = int(task["episode_index"])
    final_parquet, video = core.final_paths(dataset_root, episode_index, chunk_size)
    npz_path, report_path = core.output_paths(output_root, episode_index, chunk_size)
    if not overwrite and reusable_success(report_path, npz_path, override):
        return {"episode_index": episode_index, "status": "success", "resumed": True,
                "npz": str(npz_path), "report": str(report_path)}
    source_parquet = Path(task["source_parquet"])
    report: dict[str, Any] = {
        "algorithm_version": core.ALGORITHM_VERSION,
        "backfill_version": BACKFILL_VERSION,
        "status": "running",
        "created_at": core.utc_now(),
        "dataset": dataset_root.name,
        "episode_index": episode_index,
        "source_dataset": task.get("source_dataset"),
        "source_episode_index": task.get("source_episode_index"),
        "source_episode_name": task.get("source_episode_name"),
        "source_parquet": str(source_parquet),
        "final_parquet": str(final_parquet),
        "video": str(video),
        "geometry_npz": str(npz_path),
        "geometry_report": str(report_path),
        "geometry_only": True,
        "visual_visibility_computed": False,
        "calibration_status": {"K": "valid", "E": "valid", "B": "valid"},
        "calibration_method": override["calibration_method"],
        "calibration_evidence": override["evidence"]["initializer_report"],
        "calibration_override": {
            "path": override["_path"], "sha256": override["_sha256"],
            "version": override["override_version"],
        },
        "matrix_convention": {
            "reshape_order": "C", "E": "camera_front -> arm_left_base; inverted for projection",
            "B": "arm_right_base -> arm_left_base",
        },
    }
    try:
        source_dataset = str(task.get("source_dataset", ""))
        if not any(source_dataset.endswith(suffix) for suffix in override["source_dataset_suffixes"]):
            raise ValueError(f"episode source dataset is outside override scope: {source_dataset}")
        source_file = pq.ParquetFile(source_parquet)
        present = set(source_file.schema_arrow.names)
        if core.C_COLUMN in present:
            raise ValueError("source E is present; shared-E backfill must only handle missing-E sources")
        for column in (core.K_COLUMN, core.B_COLUMN):
            if column not in present:
                raise ValueError(f"required source column missing: {column}")
        intrinsic = core.read_first_list(source_file, core.K_COLUMN, 9, True).reshape(3, 3)
        right_to_left = core.read_first_list(source_file, core.B_COLUMN, 16, True).reshape(4, 4)
        if not np.array_equal(intrinsic, override["_intrinsic"]):
            raise ValueError("source K does not exactly match override K")
        if not np.array_equal(right_to_left, override["_right_to_left"]):
            raise ValueError("source B does not exactly match override B")
        calibration = calibration_bundle(override)
        final_table = pq.read_table(final_parquet, columns=["observation.state"])
        states = core.list_column_to_numpy(final_table, "observation.state", 20)
        frame_count = len(states)
        source_rows = source_file.metadata.num_rows
        if source_rows != frame_count or int(task.get("frames", -1)) != frame_count:
            raise ValueError(
                f"frame mismatch source={source_rows}, final={frame_count}, provenance={task.get('frames')}"
            )
        video_frames, video_width, video_height = video_metadata(video)
        if (video_frames, video_width, video_height) != (frame_count, image_width, image_height):
            raise ValueError(
                f"video mismatch frames/size={(video_frames, video_width, video_height)} "
                f"expected={(frame_count, image_width, image_height)}"
            )
        arrays, counts = build_arrays(
            episode_index, states, calibration, image_width, image_height, override,
        )
        npz_size = core.atomic_save_npz(npz_path, arrays)
        report.update({
            "status": "success", "frames": frame_count, "source_rows": source_rows,
            "image_size": [image_width, image_height], "npz_size_bytes": npz_size,
            "npz_sha256": sha256_file(npz_path),
            "source_file": core.file_fingerprint(source_parquet),
            "final_file": core.file_fingerprint(final_parquet),
            "video_file": core.file_fingerprint(video),
            "calibration": {
                "signatures": calibration.signatures, "qa": calibration.qa,
                "read_policy": "K/B_all_rows_exact_static; E_validated_shared_override",
                "matrix_sources": {"K": "source_parquet", "E": "override", "B": "source_parquet"},
                "source_missing_fields": [core.C_COLUMN],
            },
            "point_definition": {
                "flange_xy": "source Cartesian link6/flange origin",
                "grasp_center_xy": "provisional flange-local +Z offset 0.13503 m; not a visually fitted fingertip point",
                "geometric_in_fov": "alias of grasp_center_geometric_in_fov; pinhole bounds and positive depth after float32 quantization",
            },
            "geometric_in_fov_counts": counts,
            "geometry_only_warning": "geometric_in_fov is not visual visibility; run the DINO appearance filter",
            "operation_range": [0, frame_count - 1],
            "operation_contract": "all frames belong to the final episode; appearance filtering decides visibility",
            "elapsed_seconds": time.monotonic() - started,
        })
        core.atomic_write_json(report_path, report)
        return {"episode_index": episode_index, "status": "success", "resumed": False,
                "npz": str(npz_path), "report": str(report_path)}
    except Exception as exc:
        report.update({
            "status": "error", "error_type": type(exc).__name__, "reason": str(exc),
            "elapsed_seconds": time.monotonic() - started,
        })
        core.atomic_write_json(report_path, report)
        return {"episode_index": episode_index, "status": "error", "error": repr(exc),
                "npz": str(npz_path), "report": str(report_path)}


def validate_output(
    task: dict[str, Any], dataset_root: Path, output_root: Path, chunk_size: int,
    image_width: int, image_height: int, override: dict[str, Any],
) -> dict[str, Any]:
    episode_index = int(task["episode_index"])
    final_parquet, video = core.final_paths(dataset_root, episode_index, chunk_size)
    npz_path, report_path = core.output_paths(output_root, episode_index, chunk_size)
    errors: list[str] = []
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "success" or report.get("calibration_method") != override["calibration_method"]:
            errors.append("report status/method mismatch")
        if report.get("calibration_override", {}).get("sha256") != override["_sha256"]:
            errors.append("report override SHA mismatch")
        if report.get("calibration_status") != {"K": "valid", "E": "valid", "B": "valid"}:
            errors.append("report calibration status mismatch")
        final_table = pq.read_table(final_parquet, columns=["observation.state"])
        states = core.list_column_to_numpy(final_table, "observation.state", 20)
        frame_count = len(states)
        source_file = pq.ParquetFile(Path(task["source_parquet"]))
        source_present = set(source_file.schema_arrow.names)
        if core.C_COLUMN in source_present:
            errors.append("source unexpectedly contains E")
        try:
            source_k = core.read_first_list(source_file, core.K_COLUMN, 9, True).reshape(3, 3)
            source_b = core.read_first_list(source_file, core.B_COLUMN, 16, True).reshape(4, 4)
            if not np.array_equal(source_k, override["_intrinsic"]):
                errors.append("source K differs from override")
            if not np.array_equal(source_b, override["_right_to_left"]):
                errors.append("source B differs from override")
            if source_file.metadata.num_rows != frame_count:
                errors.append("source/final row mismatch")
        except Exception as exc:
            errors.append(f"source calibration validation failed: {exc}")
        video_frames, video_width, video_height = video_metadata(video)
        if (video_frames, video_width, video_height) != (frame_count, image_width, image_height):
            errors.append("video/final frame or image-size mismatch")
        calibration = calibration_bundle(override)
        expected_geometry = core.reconstruct_geometry(
            states, calibration.intrinsic, calibration.camera_to_left_base,
            calibration.right_base_to_left_base, image_width, image_height,
            left_base_to_camera=calibration.left_base_to_camera,
        )
        with np.load(npz_path, allow_pickle=False) as data:
            missing = REQUIRED_ARRAYS - set(data.files)
            if missing:
                errors.append(f"missing arrays: {sorted(missing)}")
            if str(data["algorithm_version"]) != core.ALGORITHM_VERSION:
                errors.append("NPZ algorithm version mismatch")
            if str(data["calibration_method"]) != override["calibration_method"]:
                errors.append("NPZ calibration method mismatch")
            if int(data["episode_index"]) != episode_index:
                errors.append("NPZ episode mismatch")
            expected_shapes = {
                "frame_index": (frame_count,), "operation_mask": (frame_count,),
                "flange_xy_geom": (frame_count, 2, 2), "grasp_center_xy_geom": (frame_count, 2, 2),
                "flange_xy": (frame_count, 2, 2), "grasp_center_xy": (frame_count, 2, 2),
                "flange_geometric_in_fov": (frame_count, 2),
                "grasp_center_geometric_in_fov": (frame_count, 2),
                "visual_visible": (frame_count, 2), "flange_left_base": (frame_count, 2, 3),
                "grasp_center_left_base": (frame_count, 2, 3),
            }
            for key, shape in expected_shapes.items():
                if key in data.files and data[key].shape != shape:
                    errors.append(f"{key} shape {data[key].shape} != {shape}")
            if not np.array_equal(data["frame_index"], np.arange(frame_count)):
                errors.append("frame_index is not contiguous")
            if not data["operation_mask"].all() or data["visual_visible"].any():
                errors.append("operation/visual-visible initialization mismatch")
            for key, expected in (
                ("intrinsic", calibration.intrinsic),
                ("camera_to_left_base", calibration.camera_to_left_base),
                ("left_base_to_camera", calibration.left_base_to_camera),
                ("right_base_to_left_base", calibration.right_base_to_left_base),
            ):
                if not np.array_equal(data[key], expected):
                    errors.append(f"{key} differs from validated calibration")
            if str(data["calibration_signature"]) != calibration.signatures["combined"]:
                errors.append("calibration signature mismatch")
            geometry_keys = (
                ("flange_xy_geom", "flange_xy_geom"),
                ("grasp_center_xy_geom", "grasp_center_xy_geom"),
                ("flange_left_base", "flange_left_base"),
                ("grasp_center_left_base", "grasp_center_left_base"),
            )
            for stored, expected_key in geometry_keys:
                if not np.allclose(data[stored], expected_geometry[expected_key], atol=2e-5, rtol=1e-6):
                    errors.append(f"{stored} differs from independent reconstruction")
            for prefix in ("flange", "grasp_center"):
                raw = data[f"{prefix}_xy_geom"]
                depth = data[f"{prefix}_camera_depth"]
                fov = data[f"{prefix}_geometric_in_fov"].astype(bool)
                expected_fov = core.points_in_fov(raw, depth, image_width, image_height)
                if not np.array_equal(fov, expected_fov):
                    errors.append(f"{prefix} FOV mismatch")
                masked = data[f"{prefix}_xy"]
                if not np.array_equal(np.isfinite(masked).all(axis=2), fov):
                    errors.append(f"{prefix} masked coordinate/FOV mismatch")
                if not np.array_equal(masked[fov], raw[fov]):
                    errors.append(f"{prefix} in-FOV coordinates differ from raw")
            if not np.array_equal(data["tcp_xy"], data["grasp_center_xy"], equal_nan=True):
                errors.append("tcp_xy compatibility alias mismatch")
            if not np.array_equal(data["geometric_in_fov"], data["grasp_center_geometric_in_fov"]):
                errors.append("geometric_in_fov alias mismatch")
        if report.get("npz_sha256") != sha256_file(npz_path):
            errors.append("NPZ SHA-256 mismatch")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {"episode_index": episode_index, "valid": not errors, "errors": errors,
            "frames": report.get("frames", 0) if 'report' in locals() else 0}


def manifest_record(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": report["dataset"], "episode_index": report["episode_index"],
        "video": report["video"], "geometry_npz": report["geometry_npz"],
        "geometry_report": report["geometry_report"], "calibration_status": "valid",
        "calibration_matrices": {"K": "valid", "E": "valid", "B": "valid"},
        "calibration_method": report["calibration_method"],
        "calibration_evidence": report["calibration_evidence"],
        "calibration_override_version": report["calibration_override"]["version"],
        "calibration_signature": report["calibration"]["signatures"]["combined"],
        "source_parquet": report["source_parquet"], "status": "success",
    }


def publish_manifest(
    manifest_path: Path, target_indices: list[int], output_root: Path, chunk_size: int,
    expected_total: int,
) -> dict[str, Any]:
    existing = core.load_existing_manifest(manifest_path)
    if len(existing) != expected_total:
        raise RuntimeError(f"manifest has {len(existing)} entries, expected {expected_total}")
    target_set = set(target_indices)
    non_target_before = {index: row for index, row in existing.items() if index not in target_set}
    before_digest = canonical_records_digest(non_target_before)
    for index in target_indices:
        _, report_path = core.output_paths(output_root, index, chunk_size)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        existing[index] = manifest_record(report)
    core.atomic_write_jsonl(manifest_path, (existing[index] for index in sorted(existing)))
    published = core.load_existing_manifest(manifest_path)
    non_target_after = {index: row for index, row in published.items() if index not in target_set}
    after_digest = canonical_records_digest(non_target_after)
    if len(published) != expected_total or before_digest != after_digest:
        raise RuntimeError("manifest post-write verification failed or a non-target record changed")
    for index in target_indices:
        row = published[index]
        if (row.get("calibration_status") != "valid"
                or row.get("calibration_method") != "shared_right_base_invariant"
                or row.get("status") != "success"):
            raise RuntimeError(f"manifest target row {index} failed post-write validation")
    return {
        "path": str(manifest_path), "entries": len(published),
        "non_target_entries": len(non_target_after),
        "non_target_digest_before": before_digest, "non_target_digest_after": after_digest,
        "sha256": sha256_file(manifest_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=core.DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--override", type=Path, default=DEFAULT_OVERRIDE)
    parser.add_argument("--episode", type=int, action="append", default=[])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--publish-manifest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = (args.output_root.resolve() if args.output_root else
                   (core.DEFAULT_OUTPUT_PARENT / dataset_root.name).resolve())
    info, total_episodes, chunk_size, width, height = core.read_info(dataset_root)
    override = load_override(args.override.resolve(), dataset_root.name)
    start = int(override["episode_range"]["start_inclusive"])
    stop = int(override["episode_range"]["stop_exclusive"])
    full_indices = list(range(start, stop))
    indices = sorted(set(args.episode)) if args.episode else full_indices
    if not indices or any(index not in full_indices for index in indices):
        raise SystemExit(f"episode selection must be a nonempty subset of [{start}, {stop})")
    if args.publish_manifest and indices != full_indices:
        raise SystemExit("--publish-manifest requires the complete 60-episode override scope")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    provenance = core.read_provenance(dataset_root, total_episodes)
    tasks = [provenance[index] for index in indices]
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(
            project_episode, task, dataset_root, output_root, chunk_size, width, height,
            override, args.overwrite,
        ) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result(); results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    results.sort(key=lambda item: item["episode_index"])
    if any(result["status"] != "success" for result in results):
        raise SystemExit("projection failed; manifest was not modified")
    validations = [
        validate_output(provenance[index], dataset_root, output_root, chunk_size, width, height, override)
        for index in indices
    ]
    invalid = [item for item in validations if not item["valid"]]
    if invalid:
        print(json.dumps({"invalid": invalid}, ensure_ascii=False, indent=2))
        raise SystemExit("on-disk validation failed; manifest was not modified")
    manifest_audit = None
    if args.publish_manifest:
        manifest_audit = publish_manifest(
            output_root / "geometry_manifest.jsonl", indices, output_root, chunk_size, total_episodes,
        )
    summary = {
        "backfill_version": BACKFILL_VERSION, "status": "success",
        "dataset": str(dataset_root), "output_root": str(output_root),
        "calibration_method": override["calibration_method"],
        "calibration_override": {"path": override["_path"], "sha256": override["_sha256"]},
        "selected_episodes": len(indices), "episode_range": [indices[0], indices[-1]],
        "projected": len(results), "resumed": sum(bool(item.get("resumed")) for item in results),
        "validated": len(validations), "invalid": len(invalid),
        "frames": int(sum(item["frames"] for item in validations)),
        "manifest_published": bool(args.publish_manifest), "manifest_audit": manifest_audit,
        "elapsed_seconds": time.monotonic() - started, "generated_at": core.utc_now(),
    }
    summary_path = output_root / "shared_right_base_invariant_backfill.summary.json"
    core.atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
