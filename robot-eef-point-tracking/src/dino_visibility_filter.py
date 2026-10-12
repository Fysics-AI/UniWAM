#!/usr/bin/env python3
"""Fuse calibrated end-effector projections with DINOv2 visual visibility scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ALGORITHM_VERSION = "calibrated-eef-v5-dino-visible-v1"
CPU_THREADS = int(os.environ.get("EEF_DINO_CPU_THREADS", "4"))
cv2.setNumThreads(CPU_THREADS)
torch.set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

GEOMETRY_IDENTITY_KEYS = (
    "algorithm_version", "calibration_sha256", "calibration_signature",
    "calibration_signature_v5", "point_definition_version", "urdf_sha256",
    "calibration_id", "calibration_override_version", "geometry_content_identity",
    "dataset", "episode_index",
)


def first_array(arrays: dict[str, np.ndarray], *keys: str) -> np.ndarray:
    for key in keys:
        if key in arrays:
            return arrays[key]
    raise KeyError(f"none of the required arrays are present: {keys}")


def geometry_identity(arrays: dict[str, np.ndarray]) -> str:
    payload = {}
    for key in GEOMETRY_IDENTITY_KEYS:
        if key in arrays:
            value = arrays[key]
            payload[key] = str(value.item()) if value.ndim == 0 else value.tolist()
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def fusion_config_identity(args: argparse.Namespace) -> str:
    checkpoint = Path(args.checkpoint)
    stat = checkpoint.stat()
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "prototype": args.prototype,
        "grasp_prototype": args.grasp_prototype or args.prototype,
        "stride": args.stride,
        "input_size": args.input_size,
        "feature_radius": args.feature_radius,
        "enter_threshold": args.enter_threshold,
        "exit_threshold": args.exit_threshold,
        "max_false_gap": args.max_false_gap,
        "min_true_run": args.min_true_run,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov2-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prototype", nargs=4, action="append", required=True,
        metavar=("VIDEO", "FRAME", "X", "Y"),
        help="Reviewed flange prototype; may be repeated for pose diversity.",
    )
    parser.add_argument(
        "--grasp-prototype", nargs=4, action="append",
        metavar=("VIDEO", "FRAME", "X", "Y"),
        help="Reviewed physical gripper prototype; defaults to flange prototypes.",
    )
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=392)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-radius", type=int, default=2)
    parser.add_argument("--enter-threshold", type=float, default=0.38)
    parser.add_argument("--exit-threshold", type=float, default=0.30)
    parser.add_argument("--max-false-gap", type=int, default=15)
    parser.add_argument("--min-true-run", type=int, default=10)
    return parser.parse_args()


def read_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index}: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def read_selected_frames(path: Path, indices: np.ndarray) -> list[np.ndarray]:
    wanted = set(int(value) for value in indices)
    decoded: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path))
    frame_index = 0
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index in wanted:
            decoded[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            wanted.remove(frame_index)
        frame_index += 1
    cap.release()
    missing = [int(value) for value in indices if int(value) not in decoded]
    if missing:
        raise RuntimeError(f"video ended before requested frames: {missing[:10]}")
    return [decoded[int(value)] for value in indices]


def preprocess(frames: list[np.ndarray], size: int) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    tensors = []
    for frame in frames:
        resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensors.append((tensor - mean) / std)
    return torch.stack(tensors)


class DINOVisibilityScorer:
    def __init__(
        self,
        repo: Path,
        checkpoint: Path,
        prototypes: list[list[str]],
        input_size: int,
        batch_size: int,
        feature_radius: int,
    ) -> None:
        sys.path.insert(0, str(repo))
        self.model = torch.hub.load(
            str(repo), "dinov2_vitb14", source="local", trust_repo=True, pretrained=False
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state.get("model", state), strict=True)
        self.model.cuda().eval()
        self.input_size = input_size
        self.batch_size = batch_size
        self.feature_radius = feature_radius
        self.templates = self._encode_prototypes(prototypes)

    @torch.inference_mode()
    def _features(self, frames: list[np.ndarray]) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feature = self.model.get_intermediate_layers(
                preprocess(frames, self.input_size).cuda(), n=1, reshape=True
            )[0].float()
        return F.normalize(feature, dim=1)

    def _encode_prototypes(self, prototypes: list[list[str]]) -> torch.Tensor:
        frames, locations = [], []
        for video, frame, x, y in prototypes:
            image = read_frame(Path(video), int(frame))
            frames.append(image)
            locations.append((float(x), float(y), image.shape[1], image.shape[0]))
        feature = self._features(frames)
        height, width = feature.shape[2:]
        vectors = []
        for index, (x, y, image_width, image_height) in enumerate(locations):
            grid_x = int(np.clip(round(x / image_width * width - 0.5), 0, width - 1))
            grid_y = int(np.clip(round(y / image_height * height - 0.5), 0, height - 1))
            vectors.append(feature[index, :, grid_y, grid_x])
        return torch.stack(vectors)

    def encode_prototypes(self, prototypes: list[list[str]]) -> torch.Tensor:
        return self._encode_prototypes(prototypes)

    def score_multi(
        self,
        video_path: Path,
        point_specs: list[tuple[np.ndarray, np.ndarray, torch.Tensor]],
        stride: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return sampled scores with shape [point_set, sample, arm]."""
        if not point_specs:
            raise ValueError("at least one point set is required")
        first_points = point_specs[0][0]
        if first_points.ndim != 3 or first_points.shape[1] < 1 or first_points.shape[2] != 2:
            raise ValueError(f"expected points [T,A,2] with A >= 1, got {first_points.shape}")
        frame_count, arm_count = first_points.shape[:2]
        for points_xy, in_fov, _ in point_specs:
            if points_xy.shape != (frame_count, arm_count, 2):
                raise ValueError(f"expected points [T,{arm_count},2], got {points_xy.shape}")
            if in_fov.shape != (frame_count, arm_count):
                raise ValueError(f"expected FOV mask [T,{arm_count}], got {in_fov.shape}")
        sample_indices = np.arange(0, frame_count, stride, dtype=np.int32)
        frames = read_selected_frames(video_path, sample_indices)
        sampled_scores = np.full(
            (len(point_specs), len(sample_indices), arm_count), np.nan, dtype=np.float32
        )
        started = time.time()
        for batch_start in range(0, len(frames), self.batch_size):
            batch_frames = frames[batch_start : batch_start + self.batch_size]
            feature = self._features(batch_frames)
            feature_height, feature_width = feature.shape[2:]
            for local_index, frame_index in enumerate(
                sample_indices[batch_start : batch_start + len(batch_frames)]
            ):
                image_height, image_width = batch_frames[local_index].shape[:2]
                for set_index, (points_xy, in_fov, templates) in enumerate(point_specs):
                    for arm in range(arm_count):
                        if (
                            not in_fov[frame_index, arm]
                            or not np.isfinite(points_xy[frame_index, arm]).all()
                        ):
                            continue
                        x, y = points_xy[frame_index, arm]
                        grid_x = int(np.clip(round(x / image_width * feature_width - 0.5), 0, feature_width - 1))
                        grid_y = int(np.clip(round(y / image_height * feature_height - 0.5), 0, feature_height - 1))
                        radius = self.feature_radius
                        region = feature[
                            local_index, :,
                            max(0, grid_y - radius) : min(feature_height, grid_y + radius + 1),
                            max(0, grid_x - radius) : min(feature_width, grid_x + radius + 1),
                        ]
                        similarities = torch.einsum("pc,chw->phw", templates, region)
                        sampled_scores[set_index, batch_start + local_index, arm] = float(
                            similarities.max().cpu()
                        )
        return sample_indices, sampled_scores, time.time() - started

    def score(
        self,
        video_path: Path,
        flange_xy_geom: np.ndarray,
        geometric_in_fov: np.ndarray,
        stride: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        sample_indices, scores, elapsed = self.score_multi(
            video_path,
            [(flange_xy_geom, geometric_in_fov, self.templates)],
            stride,
        )
        return sample_indices, scores[0], elapsed


def interpolate_scores(
    frame_count: int,
    sample_indices: np.ndarray,
    sampled_scores: np.ndarray,
    geometric_in_fov: np.ndarray,
) -> np.ndarray:
    arm_count = geometric_in_fov.shape[1]
    scores = np.full((frame_count, arm_count), np.nan, dtype=np.float32)
    for arm in range(arm_count):
        in_fov = np.asarray(geometric_in_fov[:, arm], dtype=bool)
        index = 0
        while index < frame_count:
            if not in_fov[index]:
                index += 1
                continue
            start = index
            while index < frame_count and in_fov[index]:
                index += 1
            stop = index
            sampled_in_run = (
                (sample_indices >= start)
                & (sample_indices < stop)
                & np.isfinite(sampled_scores[:, arm])
            )
            if not sampled_in_run.any():
                # Never borrow appearance evidence across an out-of-FOV or
                # navigation gap. Short runs without a sampled frame remain
                # uncertain/NaN.
                continue
            run_indices = sample_indices[sampled_in_run]
            run_scores = sampled_scores[sampled_in_run, arm]
            timeline = np.arange(start, stop)
            scores[start:stop, arm] = np.interp(
                timeline,
                run_indices,
                run_scores,
                left=float(run_scores[0]),
                right=float(run_scores[-1]),
            )
    return scores


def hysteresis_visibility(
    scores: np.ndarray,
    geometric_in_fov: np.ndarray,
    enter_threshold: float,
    exit_threshold: float,
) -> np.ndarray:
    visible = np.zeros_like(geometric_in_fov, dtype=bool)
    for arm in range(geometric_in_fov.shape[1]):
        active = False
        for frame_index in range(len(scores)):
            if not geometric_in_fov[frame_index, arm] or not np.isfinite(scores[frame_index, arm]):
                active = False
            elif active:
                active = bool(scores[frame_index, arm] >= exit_threshold)
            else:
                active = bool(scores[frame_index, arm] >= enter_threshold)
            visible[frame_index, arm] = active
    return visible


def morphology_visibility(
    visible: np.ndarray,
    geometric_in_fov: np.ndarray,
    *,
    max_false_gap: int = 15,
    min_true_run: int = 10,
) -> np.ndarray:
    """Remove one-sample flicker without bridging real out-of-FOV gaps.

    Geometry is the hard bound.  Short low-score gaps are filled only while
    the point remains geometrically in-frame; isolated positive runs are
    removed. Defaults correspond to three and two DINO samples at stride five,
    expressed on the interpolated source-frame timeline.
    """
    result = np.asarray(visible, dtype=bool).copy()
    fov = np.asarray(geometric_in_fov, dtype=bool)
    for arm in range(result.shape[1]):
        # Fill short false gaps between two confirmed runs, but never across
        # an out-of-FOV interval.
        index = 0
        while index < len(result):
            if result[index, arm] or not fov[index, arm]:
                index += 1
                continue
            start = index
            while index < len(result) and not result[index, arm] and fov[index, arm]:
                index += 1
            if (
                start > 0
                and index < len(result)
                and result[start - 1, arm]
                and result[index, arm]
                and index - start <= max_false_gap
            ):
                result[start:index, arm] = True
        # Remove very short positive bursts that are characteristic of a
        # background/camera false match.
        index = 0
        while index < len(result):
            if not result[index, arm]:
                index += 1
                continue
            start = index
            while index < len(result) and result[index, arm]:
                index += 1
            if index - start < min_true_run:
                result[start:index, arm] = False
    return result & fov


def atomic_save(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    geometry_file = np.load(args.geometry)
    arrays = {key: geometry_file[key] for key in geometry_file.files}
    geometry_id = geometry_identity(arrays)
    config_id = fusion_config_identity(args)
    flange_geom = first_array(arrays, "flange_xy_geom", "flange_xy").copy()
    grasp_geom = first_array(
        arrays,
        "grasp_tip_midpoint_xy_geom",
        "contact_midpoint_xy_geom",
        "grasp_tip_xy_geom",
        "grasp_center_xy_geom",
        "grasp_center_xy",
        "tcp_xy",
    ).copy()
    grasp_in_fov = first_array(
        arrays,
        "grasp_tip_midpoint_geometric_in_fov",
        "contact_midpoint_geometric_in_fov",
        "grasp_tip_geometric_in_fov",
        "grasp_center_geometric_in_fov",
        "geometric_in_fov",
        "geometric_in_frame",
    ).astype(bool)
    flange_in_fov = arrays.get("flange_geometric_in_fov", grasp_in_fov).astype(bool)
    operation_mask = arrays.get("operation_mask")
    if operation_mask is None:
        operation_mask = np.zeros(len(flange_geom), dtype=bool)
        operation_start, operation_end = arrays["operation_range"].astype(int)
        operation_mask[max(0, operation_start) : min(len(operation_mask), operation_end + 1)] = True
    operation_mask = operation_mask.astype(bool)
    grasp_in_fov &= operation_mask[:, None]
    flange_in_fov &= operation_mask[:, None]
    scorer = DINOVisibilityScorer(
        args.dinov2_repo, args.checkpoint, args.prototype,
        args.input_size, args.batch_size, args.feature_radius,
    )
    grasp_templates = scorer.encode_prototypes(args.grasp_prototype or args.prototype)
    sample_indices, sampled_score_sets, elapsed = scorer.score_multi(
        args.video,
        [
            (flange_geom, flange_in_fov, scorer.templates),
            (grasp_geom, grasp_in_fov, grasp_templates),
        ],
        args.stride,
    )
    flange_sampled_scores, grasp_sampled_scores = sampled_score_sets
    flange_scores = interpolate_scores(
        len(flange_geom), sample_indices, flange_sampled_scores, flange_in_fov
    )
    grasp_scores = interpolate_scores(
        len(flange_geom), sample_indices, grasp_sampled_scores, grasp_in_fov
    )
    flange_visible = hysteresis_visibility(
        flange_scores, flange_in_fov, args.enter_threshold, args.exit_threshold
    )
    grasp_visible = hysteresis_visibility(
        grasp_scores, grasp_in_fov, args.enter_threshold, args.exit_threshold
    )
    flange_visible = morphology_visibility(
        flange_visible,
        flange_in_fov,
        max_false_gap=args.max_false_gap,
        min_true_run=args.min_true_run,
    )
    grasp_visible = morphology_visibility(
        grasp_visible,
        grasp_in_fov,
        max_false_gap=args.max_false_gap,
        min_true_run=args.min_true_run,
    )
    occluded_or_uncertain = grasp_in_fov & ~grasp_visible
    flange_visual = flange_geom.copy()
    grasp_visual = grasp_geom.copy()
    flange_visual[~flange_visible] = np.nan
    grasp_visual[~grasp_visible] = np.nan
    arrays.update(
        {
            "geometry_algorithm_version": np.asarray(
                str(arrays.get("algorithm_version", np.asarray("unknown")).item())
            ),
            "algorithm_version": np.asarray(ALGORITHM_VERSION),
            "visibility_algorithm_version": np.asarray(ALGORITHM_VERSION),
            "geometry_identity": np.asarray(geometry_id),
            "fusion_config_identity": np.asarray(config_id),
            "flange_xy_geom": flange_geom,
            "j6_origin_xy_geom": flange_geom,
            "eef_xy_geom": flange_geom,
            "grasp_center_xy_geom": grasp_geom,
            "grasp_tip_midpoint_xy_geom": grasp_geom,
            "tcp_xy_geom": grasp_geom,
            "flange_geometric_in_fov": flange_in_fov,
            "j6_origin_geometric_in_fov": flange_in_fov,
            "eef_geometric_in_fov": flange_in_fov,
            "grasp_center_geometric_in_fov": grasp_in_fov,
            "grasp_tip_midpoint_geometric_in_fov": grasp_in_fov,
            "tcp_geometric_in_fov": grasp_in_fov,
            "geometric_in_fov": grasp_in_fov,
            "dino_flange_score": flange_scores,
            "dino_grasp_score": grasp_scores,
            "dino_score": grasp_scores,
            "flange_visible": flange_visible,
            "j6_origin_visible": flange_visible,
            "eef_visible": flange_visible,
            "grasp_center_visible": grasp_visible,
            "visual_visible": grasp_visible,
            "occluded_or_uncertain": occluded_or_uncertain,
            "flange_xy": flange_visual,
            "j6_origin_xy": flange_visual,
            "eef_xy": flange_visual,
            "tcp_xy": grasp_visual,
            "grasp_center_xy": grasp_visual,
            "grasp_tip_midpoint_xy": grasp_visual,
            "visible": grasp_visible,
            "quality": np.where(grasp_visible, 1, np.where(grasp_in_fov, 2, 0)).astype(np.uint8),
            "dino_sample_frame": sample_indices,
            "dino_flange_sample_score": flange_sampled_scores,
            "dino_grasp_sample_score": grasp_sampled_scores,
            "dino_sample_score": grasp_sampled_scores,
        }
    )
    atomic_save(args.output, arrays)
    report = {
        "algorithm_version": ALGORITHM_VERSION,
        "video": str(args.video),
        "geometry": str(args.geometry),
        "geometry_identity": geometry_id,
        "fusion_config_identity": config_id,
        "checkpoint": str(args.checkpoint),
        "prototypes": args.prototype,
        "grasp_prototypes": args.grasp_prototype or args.prototype,
        "frames": len(flange_geom),
        "flange_visible_counts": flange_visible.sum(axis=0).tolist(),
        "grasp_center_visible_counts": grasp_visible.sum(axis=0).tolist(),
        "visual_visible_counts": grasp_visible.sum(axis=0).tolist(),
        "flange_geometric_in_fov_counts": flange_in_fov.sum(axis=0).tolist(),
        "geometric_in_fov_counts": grasp_in_fov.sum(axis=0).tolist(),
        "occluded_or_uncertain_counts": occluded_or_uncertain.sum(axis=0).tolist(),
        "dino_stride": args.stride,
        "dino_elapsed_seconds_excluding_model_load": elapsed,
        "equivalent_source_fps": len(flange_geom) / max(elapsed, 1e-9),
        "thresholds": {"enter": args.enter_threshold, "exit": args.exit_threshold},
        "temporal_filter": {
            "max_false_gap_frames": args.max_false_gap,
            "min_true_run_frames": args.min_true_run,
        },
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
