#!/usr/bin/env python3
"""Appearance-gate the preferred flange track in a geometry manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from dino_visibility_filter import (
    DINOVisibilityScorer,
    first_array,
    geometry_identity,
    hysteresis_visibility,
    interpolate_scores,
    morphology_visibility,
)


ALGORITHM_VERSION = "calibrated-j6-flange-v5-dino-visible-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dinov2-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prototype", nargs=4, action="append", required=True)
    parser.add_argument(
        "--negative-prototype", nargs=4, action="append", required=True
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=392)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-radius", type=int, default=0)
    parser.add_argument("--enter-threshold", type=float, default=0.20)
    parser.add_argument("--exit-threshold", type=float, default=0.10)
    parser.add_argument("--max-false-gap", type=int, default=15)
    parser.add_argument("--min-true-run", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--events", type=Path)
    return parser.parse_args()


def read_entries(paths: list[Path]) -> list[dict[str, Any]]:
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["dataset"]), int(row["episode_index"]))
            if key in entries:
                raise ValueError(f"duplicate geometry manifest entry {key}")
            entries[key] = row
    ordered = [entries[key] for key in sorted(entries)]
    for work_index, row in enumerate(ordered):
        row["visibility_work_index"] = work_index
    return ordered


def output_paths(root: Path, entry: dict[str, Any]) -> tuple[Path, Path]:
    episode = int(entry["episode_index"])
    directory = root / str(entry["dataset"]) / f"chunk-{episode // 1000:03d}"
    return (
        directory / f"episode_{episode:06d}.npz",
        directory / f"episode_{episode:06d}.report.json",
    )


def config_identity(args: argparse.Namespace) -> str:
    checkpoint = args.checkpoint.resolve()
    stat = checkpoint.stat()
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "prototype": args.prototype,
        "negative_prototype": args.negative_prototype,
        "prototype_video_fingerprints": {
            str(Path(item[0]).resolve()): file_fingerprint(Path(item[0]))
            for item in args.prototype + args.negative_prototype
        },
        "prototype_video_sha256": {
            str(Path(item[0]).resolve()): sha256_file(Path(item[0]))
            for item in args.prototype + args.negative_prototype
        },
        "code_sha256": {
            "batch": sha256_file(Path(__file__).resolve()),
            "scorer": sha256_file(
                Path(__file__).with_name("dino_visibility_filter.py").resolve()
            ),
        },
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
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, path)
        return digest
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
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
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def append_event(path: Path | None, value: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def is_current(
    output: Path,
    report_path: Path,
    geometry_path: Path,
    target_video: Path,
    expected_config: str,
) -> bool:
    if not output.is_file() or not report_path.is_file() or not geometry_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        geometry_sha = sha256_file(geometry_path)
        with np.load(geometry_path, allow_pickle=False) as archive:
            current_geometry_identity = geometry_identity(
                {key: archive[key] for key in archive.files}
            )
        video_path = Path(report.get("video", ""))
        if (
            not video_path.is_file()
            or video_path.resolve() != target_video.resolve()
        ):
            return False
        current_video = file_fingerprint(video_path)
        if report.get("video_fingerprint") != current_video:
            return False
        if report.get("output_npz_sha256") != sha256_file(output):
            return False
        with np.load(output, allow_pickle=False) as final_file:
            output_identity_ok = (
                str(final_file["algorithm_version"].item()) == ALGORITHM_VERSION
                and str(final_file["geometry_identity"].item())
                == current_geometry_identity
                and str(final_file["fusion_config_identity"].item())
                == expected_config
            )
        return (
            report.get("status") == "complete"
            and report.get("algorithm_version") == ALGORITHM_VERSION
            and report.get("fusion_config_identity") == expected_config
            and report.get("geometry_identity") == current_geometry_identity
            and report.get("geometry_npz_sha256") == geometry_sha
            and output_identity_ok
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def process_entry(
    scorer: DINOVisibilityScorer,
    entry: dict[str, Any],
    output: Path,
    report_path: Path,
    args: argparse.Namespace,
    expected_config: str,
) -> dict[str, Any]:
    started = time.monotonic()
    geometry_path = Path(entry["geometry_npz"])
    video_path = Path(entry["video"])
    with np.load(geometry_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    geometry_version = str(arrays["algorithm_version"].item())
    geometry_id = geometry_identity(arrays)
    geometry_sha = sha256_file(geometry_path)
    track_geom = first_array(
        arrays,
        "track_xy_geom",
        "visual_flange_face_candidate_xy_geom",
        "eef_xy_geom",
        "j6_origin_xy_geom",
        "flange_xy_geom",
    ).copy()
    raw_in_fov = first_array(
        arrays,
        "track_geometric_in_fov",
        "visual_flange_face_candidate_geometric_in_fov",
        "eef_geometric_in_fov",
        "j6_origin_geometric_in_fov",
        "flange_geometric_in_fov",
    ).astype(bool)
    operation_mask = np.asarray(arrays["operation_mask"], dtype=bool)
    operation_in_fov = raw_in_fov & operation_mask[:, None]

    sample_indices, score_sets, dino_seconds = scorer.score_multi(
        video_path,
        [
            (track_geom, operation_in_fov, scorer.templates),
            (track_geom, operation_in_fov, scorer.negative_templates),
        ],
        args.stride,
    )
    positive_sampled_scores, negative_sampled_scores = score_sets
    sampled_margins = positive_sampled_scores - negative_sampled_scores
    scores = interpolate_scores(
        len(track_geom), sample_indices, sampled_margins, operation_in_fov
    )
    positive_scores = interpolate_scores(
        len(track_geom), sample_indices, positive_sampled_scores, operation_in_fov
    )
    negative_scores = interpolate_scores(
        len(track_geom), sample_indices, negative_sampled_scores, operation_in_fov
    )
    visible = hysteresis_visibility(
        scores, operation_in_fov, args.enter_threshold, args.exit_threshold
    )
    visible = morphology_visibility(
        visible,
        operation_in_fov,
        max_false_gap=args.max_false_gap,
        min_true_run=args.min_true_run,
    )
    uncertain = operation_in_fov & ~visible
    track_visual = track_geom.copy()
    track_visual[~visible] = np.nan

    arrays.update(
        {
            "geometry_algorithm_version": np.asarray(geometry_version),
            "algorithm_version": np.asarray(ALGORITHM_VERSION),
            "visibility_algorithm_version": np.asarray(ALGORITHM_VERSION),
            "geometry_identity": np.asarray(geometry_id),
            "fusion_config_identity": np.asarray(expected_config),
            "track_xy_geom": track_geom,
            "visual_flange_face_candidate_xy_geom": track_geom,
            "track_geometric_in_fov": raw_in_fov,
            "visual_flange_face_candidate_geometric_in_fov": raw_in_fov,
            "operation_track_geometric_in_fov": operation_in_fov,
            "dino_track_score": scores,
            "dino_track_margin": scores.copy(),
            "dino_track_positive_score": positive_scores,
            "dino_track_negative_score": negative_scores,
            "dino_sample_frame": sample_indices,
            "dino_track_sample_score": sampled_margins,
            "dino_track_positive_sample_score": positive_sampled_scores,
            "dino_track_negative_sample_score": negative_sampled_scores,
            "track_visible": visible,
            "visual_flange_face_candidate_visible": visible,
            "visual_visible": visible,
            "visible": visible,
            "occluded_or_uncertain": uncertain,
            "track_xy": track_visual,
            "visual_flange_face_candidate_xy": track_visual,
            "quality": np.where(
                visible, 1, np.where(operation_in_fov, 2, 0)
            ).astype(np.uint8),
        }
    )
    output_sha = atomic_save_npz(output, arrays)
    report = {
        "status": "complete",
        "algorithm_version": ALGORITHM_VERSION,
        "geometry_algorithm_version": geometry_version,
        "dataset": entry["dataset"],
        "episode_index": int(entry["episode_index"]),
        "video": str(video_path),
        "video_fingerprint": file_fingerprint(video_path),
        "geometry_npz": str(geometry_path),
        "geometry_npz_sha256": geometry_sha,
        "geometry_identity": geometry_id,
        "fusion_config_identity": expected_config,
        "checkpoint": str(args.checkpoint),
        "prototypes": args.prototype,
        "negative_prototypes": args.negative_prototype,
        "output_npz": str(output),
        "output_npz_sha256": output_sha,
        "frames": len(track_geom),
        "track_visible_counts": visible.sum(axis=0).astype(int).tolist(),
        "track_geometric_in_fov_counts": raw_in_fov.sum(axis=0).astype(int).tolist(),
        "operation_track_geometric_in_fov_counts": (
            operation_in_fov.sum(axis=0).astype(int).tolist()
        ),
        "occluded_or_uncertain_counts": uncertain.sum(axis=0).astype(int).tolist(),
        "dino_stride": args.stride,
        "dino_seconds_excluding_model_load": dino_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "thresholds": {"enter": args.enter_threshold, "exit": args.exit_threshold},
        "temporal_filter": {
            "max_false_gap_frames": args.max_false_gap,
            "min_true_run_frames": args.min_true_run,
        },
        "preferred_track_point": str(
            arrays.get(
                "preferred_track_point", np.asarray("canonical_j6_flange")
            ).item()
        ),
        "point_contract": (
            "preferred visible flange track; canonical J6 fields remain separate and "
            "no TCP/contact point is published"
        ),
    }
    atomic_write_json(report_path, report)
    return report


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("require num_shards >= 1 and 0 <= shard_id < num_shards")
    all_entries = read_entries(args.manifest)
    entries = [
        row
        for row in all_entries
        if int(row["visibility_work_index"]) % args.num_shards == args.shard_id
    ]
    expected_config = config_identity(args)
    pending = []
    for entry in entries:
        output, report_path = output_paths(args.output_root.resolve(), entry)
        geometry = Path(entry["geometry_npz"])
        target_video = Path(entry["video"])
        if args.overwrite or not is_current(
            output, report_path, geometry, target_video, expected_config
        ):
            pending.append((entry, output, report_path))
    if args.max_episodes is not None:
        pending = pending[: args.max_episodes]
    print(
        json.dumps(
            {"shard": args.shard_id, "assigned": len(entries), "pending": len(pending)}
        ),
        flush=True,
    )
    if not pending:
        return 0
    scorer = DINOVisibilityScorer(
        args.dinov2_repo,
        args.checkpoint,
        args.prototype,
        args.input_size,
        args.batch_size,
        args.feature_radius,
    )
    scorer.negative_templates = scorer.encode_prototypes(args.negative_prototype)
    failures = 0
    for position, (entry, output, report_path) in enumerate(pending, start=1):
        try:
            report = process_entry(
                scorer, entry, output, report_path, args, expected_config
            )
            event = {
                "status": "complete",
                "shard": args.shard_id,
                "position": position,
                "pending": len(pending),
                "dataset": entry["dataset"],
                "episode_index": int(entry["episode_index"]),
                "frames": report["frames"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
        except Exception as error:
            failures += 1
            event = {
                "status": "failed",
                "shard": args.shard_id,
                "position": position,
                "pending": len(pending),
                "dataset": entry.get("dataset"),
                "episode_index": entry.get("episode_index"),
                "error": repr(error),
            }
        print(json.dumps(event, sort_keys=True), flush=True)
        append_event(args.events, event)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
