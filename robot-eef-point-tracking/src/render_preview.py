#!/usr/bin/env python3
"""Render a compact review video from an end-effector NPZ track."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--trail", type=int, default=45)
    parser.add_argument(
        "--primary",
        choices=("track", "eef", "legacy"),
        default="track",
        help="Point family to render; the visible flange-face track is the default.",
    )
    parser.add_argument(
        "--show-j6-reference",
        action="store_true",
        help="Also draw the canonical J6 origin as a small cross.",
    )
    return parser.parse_args()


def finite_point(point: np.ndarray) -> bool:
    return bool(np.isfinite(point).all())


def first_present(tracks: np.lib.npyio.NpzFile, *keys: str) -> np.ndarray:
    for key in keys:
        if key in tracks.files:
            return tracks[key]
    raise KeyError(f"none of the point arrays are present: {keys}")


def main() -> int:
    args = parse_args()
    tracks = np.load(args.tracks)
    if args.primary == "track":
        primary = first_present(
            tracks,
            "track_xy",
            "visual_flange_face_candidate_xy",
            "eef_xy",
            "flange_xy",
        )
        point_label = "FACE"
    elif args.primary == "eef":
        primary = first_present(tracks, "eef_xy", "j6_origin_xy", "flange_xy")
        point_label = "J6"
    else:
        primary = first_present(tracks, "legacy_aux_xy")
        point_label = "LEGACY"
    if primary.ndim != 3 or primary.shape[1] < 1 or primary.shape[2] != 2:
        raise ValueError(f"expected point coordinates [T,A,2] with A >= 1, got {primary.shape}")
    j6_reference = (
        first_present(tracks, "eef_xy", "j6_origin_xy", "flange_xy")
        if args.show_j6_reference
        and any(key in tracks.files for key in ("eef_xy", "j6_origin_xy", "flange_xy"))
        else None
    )
    quality = (
        tracks["quality"]
        if args.primary == "track" and "quality" in tracks.files
        else np.isfinite(primary).all(axis=2).astype(np.uint8)
    )
    if "operation_range" in tracks.files:
        operation_start, operation_end = tracks["operation_range"].tolist()
    else:
        operation_frames = np.flatnonzero(tracks["operation_mask"])
        operation_start = int(operation_frames[0]) if len(operation_frames) else len(primary)
        operation_end = int(operation_frames[-1]) if len(operation_frames) else -1
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frames != len(primary):
        cap.release()
        raise ValueError(
            f"video/track length mismatch: video={video_frames}, track={len(primary)}"
        )
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        cap.release()
        raise RuntimeError("ffmpeg and ffprobe are required for H.264 preview output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(
        f".{args.output.stem}.{uuid.uuid4().hex}.mp4"
    )
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / max(1, args.stride),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"could not open temporary video writer: {temporary}")
    colors = ((80, 220, 80), (80, 120, 255), (220, 180, 80), (200, 100, 200))
    arm_count = primary.shape[1]
    trails = [deque(maxlen=args.trail) for _ in range(arm_count)]
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index >= len(primary):
            break
        for arm in range(arm_count):
            point = primary[frame_index, arm]
            if finite_point(point):
                trails[arm].append(tuple(np.rint(point).astype(int)))
            else:
                trails[arm].clear()
        if frame_index % max(1, args.stride) == 0:
            if frame_index < operation_start or frame_index > operation_end:
                overlay = np.zeros_like(frame)
                frame = cv2.addWeighted(frame, 0.35, overlay, 0.65, 0)
                cv2.putText(frame, "navigation/excluded", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            for arm in range(arm_count):
                color = colors[arm % len(colors)]
                previous = None
                for point in trails[arm]:
                    if point is None:
                        previous = None
                        continue
                    if previous is not None:
                        cv2.line(frame, previous, point, color, 2, cv2.LINE_AA)
                    previous = point
                primary_point = primary[frame_index, arm]
                if j6_reference is not None:
                    reference_point = j6_reference[frame_index, arm]
                    if finite_point(reference_point):
                        rp = tuple(np.rint(reference_point).astype(int))
                        cv2.drawMarker(
                            frame, rp, (210, 210, 210), cv2.MARKER_CROSS, 10, 1
                        )
                if finite_point(primary_point):
                    tp = tuple(np.rint(primary_point).astype(int))
                    cv2.circle(frame, tp, 5, color, -1, cv2.LINE_AA)
                    if arm_count == 1:
                        arm_label = "EEF"
                    elif arm_count == 2:
                        arm_label = ("L", "R")[arm]
                    else:
                        arm_label = f"EEF{arm + 1}"
                    label = f"{arm_label} {point_label}"
                    if args.primary == "track":
                        label += f" q={int(quality[frame_index, arm])}"
                    cv2.putText(frame, label, (tp[0] + 7, max(16, tp[1] - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"frame {frame_index}", (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
        frame_index += 1
    cap.release()
    writer.release()
    if frame_index != video_frames:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"video decode ended early: decoded={frame_index}, expected={video_frames}"
        )
    encoded = args.output.with_name(
        f".{args.output.stem}.{uuid.uuid4().hex}.h264.mp4"
    )
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(temporary),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(encoded),
    ]
    try:
        subprocess.run(command, check=True)
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,nb_frames",
                "-of",
                "json",
                str(encoded),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        if (
            stream.get("codec_name") != "h264"
            or stream.get("pix_fmt") != "yuv420p"
            or int(stream.get("nb_frames", -1))
            != (video_frames + max(1, args.stride) - 1) // max(1, args.stride)
        ):
            raise RuntimeError(f"encoded preview failed codec/frame contract: {stream}")
        os.replace(encoded, args.output)
    finally:
        temporary.unlink(missing_ok=True)
        encoded.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
