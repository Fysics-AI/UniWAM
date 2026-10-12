#!/usr/bin/env python3
"""Validate every preferred-flange geometry and visibility output pair."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from dino_visibility_filter import geometry_identity


GEOMETRY_VERSION = "agx-eef-geometry-v5.4"
VISIBILITY_VERSION = "calibrated-j6-flange-v5-dino-visible-v2"
POINT_VERSION = "agilex-piper-j6-plus-visible-z56-v1"
EXPECTED_DATASET_COUNTS = {
    "agilex7000_manip_short30_rot6d_rowmajor_h32_v4_prompt_reviewed": 12150,
    "agx_color_blocks_mobile_rot6d_vlash_se2_h48_phase_split_v3_action_current_v1": 274,
    "agx_cup_tray_mobile_rot6d_vlash_se2_h48_phase_split_v3_action_current_v1": 534,
    "agx_move_white_box_mobile_rot6d_vlash_se2_h48_phase_split_v2_action_current_v1": 477,
    "agx_ordered_color_blocks_rot6d_vlash_se2_h32_v2_action_current_v1": 187,
}
EXPECTED_OVERRIDES = {
    1952: (
        "agilex7000-ep1952-e217-v1",
        "235ce5d85488d89ecfe8c6a3a068681d0f8fe62bcee9e38d6c89ba83d5eee6db",
    ),
    2829: (
        "agilex7000-ep2829-source44-eb-v1",
        "5823962ad17be49b45ea96ea3d6d42bef16dec80dc4239d9ab273f8c75a95949",
    ),
}
SHARED_OVERRIDE_VERSION = "agilex7000-hanging-shared-right-base-invariant-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arrays_equal(first: np.ndarray, second: np.ndarray) -> bool:
    first = np.asarray(first)
    second = np.asarray(second)
    if first.dtype.kind in "fc" and second.dtype.kind in "fc":
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = [(str(row["dataset"]), int(row["episode_index"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("geometry manifest contains duplicate dataset/episode keys")
    return rows


def output_paths(root: Path, row: dict[str, Any]) -> tuple[Path, Path]:
    episode = int(row["episode_index"])
    directory = root / str(row["dataset"]) / f"chunk-{episode // 1000:03d}"
    return (
        directory / f"episode_{episode:06d}.npz",
        directory / f"episode_{episode:06d}.report.json",
    )


def validate_one(task: tuple[dict[str, Any], str, bool]) -> dict[str, Any]:
    row, output_root_text, verify_sha = task
    dataset = str(row["dataset"])
    episode = int(row["episode_index"])
    output, report_path = output_paths(Path(output_root_text), row)
    result: dict[str, Any] = {
        "dataset": dataset,
        "episode_index": episode,
        "status": "invalid",
        "errors": [],
    }
    errors: list[str] = result["errors"]
    if not output.is_file():
        errors.append("missing output NPZ")
    if not report_path.is_file():
        errors.append("missing output report")
    if errors:
        result["status"] = "missing"
        return result
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            errors.append(f"report status={report.get('status')!r}")
        if report.get("algorithm_version") != VISIBILITY_VERSION:
            errors.append("report visibility algorithm mismatch")
        if int(report.get("episode_index", -1)) != episode:
            errors.append("report episode mismatch")
        if report.get("dataset") != dataset:
            errors.append("report dataset mismatch")
        geometry_path = Path(row["geometry_npz"])
        geometry_report_path = Path(row["geometry_report"])
        if not geometry_path.is_file():
            errors.append("missing geometry NPZ")
            return result
        if not geometry_report_path.is_file():
            errors.append("missing geometry report")
            return result
        geometry_report = json.loads(
            geometry_report_path.read_text(encoding="utf-8")
        )
        geometry_sha = sha256_file(geometry_path)
        if row.get("geometry_npz_sha256") != geometry_sha:
            errors.append("manifest geometry SHA-256 mismatch")
        if geometry_report.get("output_npz_sha256") != geometry_sha:
            errors.append("geometry report SHA-256 mismatch")
        if geometry_report.get("geometry_content_identity") != row.get(
            "geometry_content_identity"
        ):
            errors.append("manifest/report geometry content identity mismatch")
        if int(geometry_report.get("frames", -1)) != int(row["frame_count"]):
            errors.append("manifest/geometry-report frame count mismatch")
        with np.load(geometry_path, allow_pickle=False) as geometry_file:
            geometry_arrays = {
                key: geometry_file[key] for key in geometry_file.files
            }
        expected_geometry_identity = geometry_identity(geometry_arrays)
        if report.get("geometry_identity") != expected_geometry_identity:
            errors.append("report geometry identity mismatch")
        if report.get("geometry_npz_sha256") != geometry_sha:
            errors.append("visibility report geometry SHA-256 mismatch")
        if report.get("video") != row.get("video"):
            errors.append("manifest/visibility-report video mismatch")
        video_path = Path(row["video"])
        if not video_path.is_file():
            errors.append("manifest video is missing")
        elif report.get("video_fingerprint") != file_fingerprint(video_path):
            errors.append("visibility report video fingerprint mismatch")
        for name in ("state_fingerprint", "video_fingerprint"):
            saved = geometry_report.get(name)
            saved_path = Path(saved.get("path", "")) if isinstance(saved, dict) else Path("")
            if not isinstance(saved, dict) or not saved_path.is_file():
                errors.append(f"geometry report {name} is missing/invalid")
            elif saved != file_fingerprint(saved_path):
                errors.append(f"geometry report {name} changed")

        with np.load(output, allow_pickle=False) as archive:
            files = set(archive.files)
            required = {
                "algorithm_version", "geometry_algorithm_version",
                "point_definition_version", "preferred_track_point",
                "frame_index", "operation_mask", "operation_range", "image_size",
                "eef_xy_geom", "track_xy_geom", "legacy_aux_xy_geom",
                "eef_geometric_in_fov", "track_geometric_in_fov",
                "operation_track_geometric_in_fov", "track_xy", "track_visible",
                "dino_track_score", "occluded_or_uncertain", "visible", "quality",
                "geometry_identity", "fusion_config_identity",
                "calibration_signature", "calibration_override_version",
                "calibration_override_sha256",
                "geometry_content_identity", "dataset", "episode_index",
                "arm_names", "schema_version", "dino_track_margin",
                "dino_track_positive_score", "dino_track_negative_score",
            }
            missing = sorted(required - files)
            if missing:
                errors.append(f"missing arrays: {missing}")
                return result
            if str(archive["algorithm_version"].item()) != VISIBILITY_VERSION:
                errors.append("NPZ visibility algorithm mismatch")
            if str(archive["geometry_algorithm_version"].item()) != GEOMETRY_VERSION:
                errors.append("NPZ geometry algorithm mismatch")
            if str(archive["point_definition_version"].item()) != POINT_VERSION:
                errors.append("point definition mismatch")
            if int(archive["schema_version"].item()) != 2:
                errors.append("schema version mismatch")
            if str(archive["dataset"].item()) != dataset:
                errors.append("NPZ dataset mismatch")
            if int(archive["episode_index"].item()) != episode:
                errors.append("NPZ episode mismatch")
            if archive["arm_names"].tolist() != ["left", "right"]:
                errors.append("arm order mismatch")
            if str(archive["preferred_track_point"].item()) != "visual_flange_face_candidate":
                errors.append("preferred track point mismatch")
            if str(archive["geometry_identity"].item()) != expected_geometry_identity:
                errors.append("NPZ geometry identity mismatch")

            frame_index = archive["frame_index"]
            frame_count = len(frame_index)
            expected_frames = int(row.get("frame_count", frame_count))
            if frame_count != expected_frames:
                errors.append(f"frame count {frame_count} != {expected_frames}")
            if not np.array_equal(frame_index, np.arange(frame_count, dtype=frame_index.dtype)):
                errors.append("frame_index is not contiguous 0..T-1")
            operation = archive["operation_mask"].astype(bool)
            raw_fov = archive["track_geometric_in_fov"].astype(bool)
            operation_fov = archive["operation_track_geometric_in_fov"].astype(bool)
            visible = archive["track_visible"].astype(bool)
            visible_alias = archive["visible"].astype(bool)
            uncertain = archive["occluded_or_uncertain"].astype(bool)
            quality = archive["quality"]
            track_geom = archive["track_xy_geom"]
            track_visual = archive["track_xy"]
            eef_geom = archive["eef_xy_geom"]
            legacy = archive["legacy_aux_xy_geom"]
            for key in (
                "frame_index", "operation_mask", "operation_range", "image_size",
                "arm_names", "eef_xy_geom", "track_xy_geom", "legacy_aux_xy_geom",
                "eef_geometric_in_fov", "track_geometric_in_fov",
                "visual_flange_face_candidate_offset_local", "intrinsic",
                "extrinsic", "right_base_to_left_base", "calibration_signature",
                "calibration_override_version", "geometry_content_identity",
            ):
                if key not in geometry_arrays:
                    errors.append(f"geometry missing immutable array: {key}")
                elif not arrays_equal(archive[key], geometry_arrays[key]):
                    errors.append(f"final output changed immutable geometry array: {key}")
            expected_shape = (frame_count, 2)
            for name, value in (
                ("operation_mask", operation),
                ("track_geometric_in_fov", raw_fov),
                ("operation_track_geometric_in_fov", operation_fov),
                ("track_visible", visible),
                ("quality", quality),
            ):
                shape = (frame_count,) if name == "operation_mask" else expected_shape
                if value.shape != shape:
                    errors.append(f"{name} shape {value.shape} != {shape}")
            for name, value in (
                ("track_xy_geom", track_geom),
                ("track_xy", track_visual),
                ("eef_xy_geom", eef_geom),
                ("legacy_aux_xy_geom", legacy),
            ):
                if value.shape != (frame_count, 2, 2):
                    errors.append(f"{name} shape mismatch: {value.shape}")
            if not np.array_equal(operation_fov, raw_fov & operation[:, None]):
                errors.append("operation/FOV composition mismatch")
            if np.any(visible & ~operation_fov):
                errors.append("visible is true outside operation/FOV")
            if not np.array_equal(visible, visible_alias):
                errors.append("visible alias mismatch")
            if not np.array_equal(uncertain, operation_fov & ~visible):
                errors.append("uncertain mask mismatch")
            expected_quality = np.where(
                visible, 1, np.where(operation_fov, 2, 0)
            ).astype(np.uint8)
            if not np.array_equal(quality, expected_quality):
                errors.append("quality contract mismatch")
            finite_visual = np.isfinite(track_visual).all(axis=-1)
            if not np.array_equal(finite_visual, visible):
                errors.append("track_xy finite mask differs from visible")
            if not np.isnan(track_visual[~visible]).all():
                errors.append("invisible track coordinates are not strict [NaN, NaN]")
            if not np.array_equal(
                track_visual[visible], track_geom[visible], equal_nan=True
            ):
                errors.append("visibility stage moved preferred track coordinates")
            if not np.isfinite(track_geom).all():
                errors.append("track geometry contains non-finite coordinates")
            margin = archive["dino_track_margin"]
            positive_score = archive["dino_track_positive_score"]
            negative_score = archive["dino_track_negative_score"]
            for name, value in (
                ("dino_track_margin", margin),
                ("dino_track_positive_score", positive_score),
                ("dino_track_negative_score", negative_score),
            ):
                if value.shape != expected_shape:
                    errors.append(f"{name} shape mismatch: {value.shape}")
            if np.any(visible & ~np.isfinite(margin)):
                errors.append("visible point has no finite DINO margin")
            finite_margin = np.isfinite(margin)
            if np.any(finite_margin & ~operation_fov):
                errors.append("DINO margin exists outside operation/FOV")
            if finite_margin.any() and (
                float(np.nanmin(margin)) < -2.0001
                or float(np.nanmax(margin)) > 2.0001
            ):
                errors.append("DINO margin is outside cosine-difference range")
            for name, value in (
                ("positive", positive_score), ("negative", negative_score)
            ):
                finite = np.isfinite(value)
                if finite.any() and (
                    float(np.nanmin(value)) < -1.0001
                    or float(np.nanmax(value)) > 1.0001
                ):
                    errors.append(f"DINO {name} score is outside cosine range")
            ambiguous = sorted(
                key
                for key in files
                if key.startswith("tcp_")
                or (key.startswith("grasp_center_") and not key.startswith("legacy_"))
            )
            if ambiguous:
                errors.append(f"ambiguous retired fields remain: {ambiguous}")
            if episode in EXPECTED_OVERRIDES and dataset.startswith("agilex7000_"):
                actual_override = str(archive["calibration_override_version"].item())
                actual_override_sha = str(
                    archive["calibration_override_sha256"].item()
                )
                expected_version, expected_sha = EXPECTED_OVERRIDES[episode]
                if actual_override != expected_version:
                    errors.append(
                        f"override {actual_override!r} != {expected_version!r}"
                    )
                if actual_override_sha != expected_sha:
                    errors.append("override SHA-256 mismatch")
            canonical_override = str(
                archive["calibration_override_version"].item()
            )
            canonical_override_sha = str(
                archive["calibration_override_sha256"].item()
            )
            if row.get("calibration_override_version", "") != canonical_override:
                errors.append("manifest/NPZ canonical override version mismatch")
            if row.get("calibration_override_sha256", "") != canonical_override_sha:
                errors.append("manifest/NPZ canonical override SHA mismatch")
            report_override = geometry_report.get("calibration", {}).get("override") or {}
            if report_override.get("version", "") != canonical_override:
                errors.append("geometry report/NPZ canonical override mismatch")
            if report_override.get("sha256", "") != canonical_override_sha:
                errors.append("geometry report/NPZ override SHA mismatch")
            if dataset.startswith("agilex7000_") and 11910 <= episode < 11970:
                if canonical_override != SHARED_OVERRIDE_VERSION:
                    errors.append("shared-calibration episode lost source override provenance")
                source_version = str(
                    archive["source_calibration_override_version"].item()
                )
                if source_version != SHARED_OVERRIDE_VERSION:
                    errors.append("shared-calibration source override field mismatch")
            result["frames"] = frame_count
            result["visible_counts"] = visible.sum(axis=0).astype(int).tolist()
            result["operation_fov_counts"] = (
                operation_fov.sum(axis=0).astype(int).tolist()
            )
            result["uncertain_counts"] = (
                uncertain.sum(axis=0).astype(int).tolist()
            )
        if verify_sha and report.get("output_npz_sha256") != sha256_file(output):
            errors.append("output SHA-256 mismatch")
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    result["status"] = "complete" if not errors else "invalid"
    return result


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    rows = read_manifest(args.manifest.resolve())
    manifest_contract_errors: list[str] = []
    if not rows:
        manifest_contract_errors.append("manifest is empty")
    required_manifest_fields = {
        "dataset", "episode_index", "frame_count", "image_size", "video",
        "geometry_npz", "geometry_report", "geometry_npz_sha256",
        "geometry_content_identity", "output_npz", "output_report",
        "calibration_override_version", "calibration_override_sha256",
    }
    for index, row in enumerate(rows):
        missing = sorted(required_manifest_fields - set(row))
        if missing:
            manifest_contract_errors.append(
                f"manifest row {index} missing fields {missing}"
            )
            if len(manifest_contract_errors) >= 100:
                break
    manifest_dataset_counts = Counter(str(row.get("dataset")) for row in rows)
    if args.require_complete:
        if not args.verify_sha256:
            manifest_contract_errors.append(
                "--require-complete requires --verify-sha256"
            )
        if dict(manifest_dataset_counts) != EXPECTED_DATASET_COUNTS:
            manifest_contract_errors.append(
                "manifest dataset counts do not match the five-dataset full contract: "
                f"actual={dict(manifest_dataset_counts)}"
            )
    tasks = [(row, str(args.output_root.resolve()), args.verify_sha256) for row in rows]
    if args.workers <= 1:
        results = list(map(validate_one, tasks))
    else:
        with mp.get_context("fork").Pool(args.workers) as pool:
            results = list(pool.imap_unordered(validate_one, tasks, chunksize=4))
    status_counts = Counter(result["status"] for result in results)
    invalid = [result for result in results if result["status"] != "complete"]
    dataset_counts = Counter(result["dataset"] for result in results)
    dataset_summary: dict[str, dict[str, Any]] = {}
    for dataset in sorted(dataset_counts):
        selected = [
            result
            for result in results
            if result["dataset"] == dataset and result["status"] == "complete"
        ]
        dataset_summary[dataset] = {
            "episodes": len(selected),
            "frames": int(sum(result.get("frames", 0) for result in selected)),
            "track_visible_counts": np.sum(
                [result.get("visible_counts", [0, 0]) for result in selected], axis=0
            ).astype(int).tolist(),
            "operation_track_geometric_in_fov_counts": np.sum(
                [result.get("operation_fov_counts", [0, 0]) for result in selected],
                axis=0,
            ).astype(int).tolist(),
            "occluded_or_uncertain_counts": np.sum(
                [result.get("uncertain_counts", [0, 0]) for result in selected], axis=0
            ).astype(int).tolist(),
        }
    payload = {
        "status": (
            "complete" if not invalid and not manifest_contract_errors else "incomplete"
        ),
        "geometry_version": GEOMETRY_VERSION,
        "visibility_version": VISIBILITY_VERSION,
        "point_definition_version": POINT_VERSION,
        "manifest": str(args.manifest.resolve()),
        "output_root": str(args.output_root.resolve()),
        "episodes": len(results),
        "frames": int(sum(result.get("frames", 0) for result in results)),
        "status_counts": dict(status_counts),
        "dataset_counts": dict(dataset_counts),
        "datasets": dataset_summary,
        "invalid_count": len(invalid),
        "invalid": invalid[:100],
        "manifest_contract_errors": manifest_contract_errors,
        "verify_sha256": args.verify_sha256,
    }
    atomic_write_json(args.report.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_complete and (invalid or manifest_contract_errors):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
