#!/usr/bin/env python3
"""Run one prefix=0 request from a recorded edge observation and render bbox8."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from uniwam_piper_robot.protocol import recv_message, send_message


BBOX_WIDTH = 424
BBOX_HEIGHT = 240
COLORS = {"left": (0, 220, 110), "right": (255, 70, 70)}


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def latest_observation(session: Path) -> dict[str, Any]:
    events = session / "events.jsonl"
    if not events.is_file():
        raise FileNotFoundError(f"Missing recording: {events}")
    selected = None
    with events.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "observation" and row.get("image_paths"):
                selected = row
    if selected is None:
        raise ValueError(f"No recorded observation with images in {events}")
    required = {"cam_nav", "cam_manip_high", "cam_left_wrist", "cam_right_wrist"}
    missing = required - set(selected["image_paths"])
    if missing:
        raise ValueError(f"Latest observation is missing images: {sorted(missing)}")
    return selected


def build_request(session: Path, observation: dict[str, Any], prompt: str) -> dict[str, Any]:
    images = {
        key: (session / relative).read_bytes()
        for key, relative in observation["image_paths"].items()
    }
    state = np.asarray(observation["state23_rot6d"], dtype=np.float32)
    joints = observation.get("joint_state")
    return {
        "type": "observation",
        "request_id": int(time.time() * 1000) % 2_000_000_000,
        "stamp": time.time(),
        "instruction": prompt or str(observation.get("instruction", "")),
        "inference_mode": "manip_only",
        "eef_state": state,
        "joint_state": None if joints is None else np.asarray(joints, dtype=np.float32),
        "images": images,
        "image_encoding": str(observation.get("image_encoding", "jpeg")),
        "prefix_mode": "head",
        "prefix_eef_actions": None,
        "prefix_joint_actions": None,
        "prefix_bbox_actions": None,
    }


def bbox_pixels(values: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    box = np.asarray(values, dtype=np.float32).reshape(4)
    if not np.all(np.isfinite(box)) or np.any(box < -0.5):
        return None
    clipped = np.clip(box, 0.0, 1.0)
    x1, y1, x2, y2 = clipped
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        min(width - 1, int(round(float(x1) * width))),
        min(height - 1, int(round(float(y1) * height))),
        min(width - 1, int(round(float(x2) * width))),
        min(height - 1, int(round(float(y2) * height))),
    )


def render_frames(image_path: Path, bbox: np.ndarray, output: Path) -> Path:
    base = Image.open(image_path).convert("RGB")
    if base.size != (BBOX_WIDTH, BBOX_HEIGHT):
        raise ValueError(
            f"cam_manip_high must be {BBOX_WIDTH}x{BBOX_HEIGHT}, got {base.size}; "
            "bbox coordinates are defined in the original main-camera frame."
        )
    frames = output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for index, row in enumerate(np.asarray(bbox, dtype=np.float32)):
        frame = base.copy()
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, 150, 18), fill=(0, 0, 0))
        draw.text((5, 4), f"predicted action step {index:02d}", fill=(255, 255, 255), font=font)
        for arm, values in (("left", row[:4]), ("right", row[4:8])):
            pixels = bbox_pixels(values, *base.size)
            if pixels is None:
                continue
            draw.rectangle(pixels, outline=COLORS[arm], width=3)
            draw.text((pixels[0] + 3, max(2, pixels[1] + 3)), arm, fill=COLORS[arm], font=font)
        frame.save(frames / f"frame_{index:04d}.png")
    return frames


def encode_video(frames: Path, output: Path, fps: float) -> Path:
    video = output / "bbox_prediction.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(float(fps)), "-i", str(frames / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to create the MP4; PNG frames were retained") from exc
    return video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path, help="Recorded edge session containing events.jsonl")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=18020)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    session = args.session.expanduser().resolve()
    output = (args.output or session / f"bbox_infer_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    observation = latest_observation(session)
    request = build_request(session, observation, args.prompt)
    with socket.create_connection((args.server_host, args.server_port), timeout=args.timeout) as sock:
        sock.settimeout(args.timeout)
        send_message(sock, request)
        response = recv_message(sock)
    if not response.get("ok", False):
        raise RuntimeError(f"Cloud inference failed: {response.get('error')}")
    bbox = np.asarray(response.get("bbox_actions"), dtype=np.float32)
    manip = np.asarray(response.get("manip_actions"), dtype=np.float32)
    if bbox.ndim != 2 or bbox.shape[1] != 8:
        raise ValueError(f"Expected bbox_actions [T,8], got {bbox.shape}")
    if manip.shape != (bbox.shape[0], 28) or not np.allclose(manip[:, 20:28], bbox):
        raise ValueError(f"Manip/bbox response contract mismatch: {manip.shape} vs {bbox.shape}")

    image_path = session / observation["image_paths"]["cam_manip_high"]
    frames = render_frames(image_path, bbox, output)
    video = encode_video(frames, output, args.fps)
    with (output / "predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "coordinate_contract": {
                    "camera": "cam_manip_high", "width": BBOX_WIDTH,
                    "height": BBOX_HEIGHT, "order": "left_xyxy,right_xyxy",
                    "normalized": True, "invalid_sentinel": [-1, -1, -1, -1],
                },
                "request": {"instruction": request["instruction"], "prefix_length": 0},
                "response": jsonable(response),
            },
            handle, indent=2, ensure_ascii=True,
        )
    print(f"OUTPUT_DIR={output}")
    print(f"VIDEO={video}")
    print(f"PREDICTIONS={output / 'predictions.json'}")


if __name__ == "__main__":
    main()
