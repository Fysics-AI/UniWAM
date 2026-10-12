#!/usr/bin/env python3
"""Flatten calibrated per-episode EEF image tracks into mmap training sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


EEF_XY_ORDER = [
    "left_x", "left_y", "right_x", "right_y",
    "left_visible", "right_visible",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_path(root: Path, episode: int, suffix: str) -> Path:
    return root / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.{suffix}"


def build_source(
    *,
    track_root: Path,
    dataset_root: Path,
    output_root: Path,
    summary_path: Path,
    validation_path: Path,
) -> dict:
    source_name = dataset_root.name
    source_tracks = track_root / source_name
    info_path = dataset_root / "meta/info.json"
    if not source_tracks.is_dir() or not info_path.is_file():
        raise FileNotFoundError(f"Missing source tracks or metadata for {source_name}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = int(info["total_episodes"])
    total_frames = int(info["total_frames"])
    if int(info["fps"]) != 30:
        raise ValueError(f"EEF-XY training requires 30 Hz data, got {info['fps']} for {source_name}")

    final_dir = output_root / source_name
    building_dir = output_root / f".{source_name}.building"
    if final_dir.exists() or building_dir.exists():
        raise FileExistsError(f"Refusing to replace existing sidecar: {final_dir}")
    building_dir.mkdir(parents=True)
    values_path = building_dir / "eef_xy_visible_normalized.npy"
    feature_mask_path = building_dir / "feature_mask.npy"
    values = np.lib.format.open_memmap(
        values_path, mode="w+", dtype=np.float32, shape=(total_frames, 6)
    )
    feature_mask = np.lib.format.open_memmap(
        feature_mask_path, mode="w+", dtype=np.bool_, shape=(total_frames, 6)
    )
    values[:] = 0.0
    feature_mask[:] = False

    offset = 0
    image_sizes: Counter[tuple[int, int]] = Counter()
    valid_counts = np.zeros(2, dtype=np.int64)
    visible_counts = np.zeros(2, dtype=np.int64)
    boundary_clipped_counts = np.zeros(2, dtype=np.int64)
    for episode in range(episodes):
        parquet = episode_path(dataset_root / "data", episode, "parquet")
        track = episode_path(source_tracks, episode, "npz")
        if not parquet.is_file() or not track.is_file():
            raise FileNotFoundError(f"Missing episode pair: {parquet} / {track}")
        parquet_rows = int(pq.ParquetFile(parquet).metadata.num_rows)
        with np.load(track, allow_pickle=False) as data:
            required = {
                "dataset", "episode_index", "frame_index", "image_size",
                "track_xy_geom", "operation_track_geometric_in_fov", "track_visible",
            }
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{track} is missing fields {sorted(missing)}")
            if str(data["dataset"].item()) != source_name:
                raise ValueError(f"Dataset identity mismatch in {track}")
            if int(data["episode_index"].item()) != episode:
                raise ValueError(f"Episode identity mismatch in {track}")
            frame_index = np.asarray(data["frame_index"], dtype=np.int64)
            xy = np.asarray(data["track_xy_geom"], dtype=np.float32)
            valid = np.asarray(data["operation_track_geometric_in_fov"], dtype=bool)
            is_visible = np.asarray(data["track_visible"], dtype=bool)
            width, height = map(int, np.asarray(data["image_size"]).tolist())

        expected_frames = parquet_rows
        if xy.shape != (expected_frames, 2, 2):
            raise ValueError(f"Track shape mismatch in {track}: {xy.shape}")
        if valid.shape != (expected_frames, 2) or is_visible.shape != valid.shape:
            raise ValueError(f"Mask shape mismatch in {track}: {valid.shape}/{is_visible.shape}")
        if not np.array_equal(frame_index, np.arange(expected_frames, dtype=np.int64)):
            raise ValueError(f"Non-contiguous frame_index in {track}")
        if width <= 1 or height <= 1:
            raise ValueError(f"Invalid image_size in {track}: {(width, height)}")
        if np.any(is_visible & ~valid):
            raise ValueError(f"Visible point is not geometrically in-FOV in {track}")
        if not np.isfinite(xy[valid]).all():
            raise ValueError(f"Valid EEF points contain non-finite values in {track}")
        scale = np.asarray([width - 1, height - 1], dtype=np.float32)
        # V5 defines continuous-image FOV as [0,width) x [0,height), while
        # pixel-center normalization ends at width-1/height-1. Preserve valid
        # near-edge labels by clipping at most one pixel to the final center.
        outside_pixel_centers = valid & np.any((xy < 0.0) | (xy > scale), axis=-1)
        beyond_continuous_image = valid & np.any((xy < 0.0) | (xy >= scale + 1.0), axis=-1)
        if np.any(beyond_continuous_image):
            raise ValueError(f"Valid EEF point exceeds the continuous image boundary in {track}")
        boundary_clipped_counts += outside_pixel_centers.sum(axis=0)

        # All five manipulation sources use the same FastWAM manipulation
        # canvas: main camera -> 384x216, wrist row -> 384x108, then crop
        # rows [2:322] from the 384x324 canvas. Transform point centers with
        # the exact resize/crop geometry used by build_matched_fastwam_mosaic.
        x_model = (xy[..., 0] + 0.5) * (384.0 / float(width)) - 0.5
        y_model = (xy[..., 1] + 0.5) * (216.0 / float(height)) - 0.5 - 2.0
        model_xy = np.stack((x_model, y_model), axis=-1)
        model_valid = valid & np.all(
            (model_xy >= 0.0) & (model_xy <= np.asarray([383.0, 319.0])), axis=-1
        )
        normalized = 2.0 * model_xy / np.asarray([383.0, 319.0], dtype=np.float32) - 1.0
        xy_valid = is_visible & model_valid
        normalized[~xy_valid] = 0.0
        # Flat action order is [left_xy, right_xy, left_visible, right_visible],
        # matching the model contract rather than the per-arm [xy,visible] layout.
        aux = np.zeros((expected_frames, 6), dtype=np.float32)
        aux[:, :4] = normalized.reshape(expected_frames, 4)
        aux[:, 4:] = np.where(is_visible, 1.0, -1.0)
        aux_mask = np.zeros((expected_frames, 6), dtype=bool)
        aux_mask[:, :4] = np.repeat(xy_valid, 2, axis=1)
        # The source contains an explicit visibility field on every frame.
        # Visibility itself is supervised even when the point is absent.
        aux_mask[:, 4:] = True
        stop = offset + expected_frames
        if stop > total_frames:
            raise ValueError(f"Episode frames exceed metadata total for {source_name}")
        values[offset:stop] = aux
        feature_mask[offset:stop] = aux_mask
        offset = stop
        image_sizes[(width, height)] += 1
        valid_counts += model_valid.sum(axis=0)
        visible_counts += is_visible.sum(axis=0)
        if (episode + 1) % 500 == 0 or episode + 1 == episodes:
            print(f"[{source_name}] {episode + 1}/{episodes} episodes", flush=True)

    if offset != total_frames:
        raise ValueError(f"Frame total mismatch for {source_name}: {offset} != {total_frames}")
    values.flush()
    feature_mask.flush()
    del values, feature_mask

    manifest = {
        "schema_version": "fastwam_eef_xy_sidecar_v1",
        "status": "complete",
        "source_name": source_name,
        "dataset_root": str(dataset_root),
        "track_root": str(source_tracks),
        "total_episodes": episodes,
        "total_frames": total_frames,
        "fps": 30,
        "eef_xy_order": EEF_XY_ORDER,
        "coordinate_frame": "fastwam_manipulation_mosaic_384x320_xy_and_visibility",
        "source_coordinate_field": "track_xy_geom",
        "normalization": "xy: resize main to 384x216, crop mosaic y=2, then 2 * mosaic_xy / [383,319] - 1; visibility: -1/+1",
        "invalid_fill": {"xy": 0.0, "visible": -1.0},
        "validity_masks": {
            "xy": "track_visible AND operation_track_geometric_in_fov",
            "visible": "track_visible",
        },
        "arm_order": ["left", "right"],
        "image_size_episode_counts": {
            f"{width}x{height}": count
            for (width, height), count in sorted(image_sizes.items())
        },
        "geometric_in_fov_counts": valid_counts.tolist(),
        "visible_counts": visible_counts.tolist(),
        "boundary_clipped_counts": boundary_clipped_counts.tolist(),
        "boundary_clip_contract": "clip valid continuous-FOV coordinates from [0,width) to pixel centers [0,width-1]",
        "provenance": {
            "v5_summary": str(summary_path),
            "v5_summary_sha256": sha256(summary_path),
            "v5_validation": str(validation_path),
            "v5_validation_sha256": sha256(validation_path),
        },
        "files": {
            "eef_xy_visible_normalized.npy": sha256(values_path),
            "feature_mask.npy": sha256(feature_mask_path),
        },
    }
    (building_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(building_dir, final_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track-root", type=Path, required=True,
    )
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
    )
    parser.add_argument(
        "--output-root", type=Path, required=True,
    )
    args = parser.parse_args()
    track_root = args.track_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    summary = track_root / "final_summary.json"
    validation = track_root / "full_validation.json"
    if not summary.is_file() or not validation.is_file():
        raise FileNotFoundError("V5 summary/validation files are required")
    validation_payload = json.loads(validation.read_text(encoding="utf-8"))
    expected_sources = sorted(validation_payload["datasets"])
    output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for source_name in expected_sources:
        reports.append(
            build_source(
                track_root=track_root,
                dataset_root=dataset_root / source_name,
                output_root=output_root,
                summary_path=summary,
                validation_path=validation,
            )
        )
    aggregate = {
        "schema_version": "fastwam_eef_xy_visible_sidecar_collection_v1",
        "status": "complete",
        "sources": expected_sources,
        "total_episodes": sum(item["total_episodes"] for item in reports),
        "total_frames": sum(item["total_frames"] for item in reports),
        "source_manifests": [str(output_root / name / "manifest.json") for name in expected_sources],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
