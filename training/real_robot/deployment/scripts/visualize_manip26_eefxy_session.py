#!/usr/bin/env python3
"""Render EEF-XY+visibility predictions from one edge recording.

The stored XY values are normalized coordinates on the 384x320 UniWAM
mosaic.  The visualization maps points from the upper 384x216 main-camera
panel back to the recorded raw cam_manip_high image.  Points whose visibility
slot is <= 0, or that lie outside the main panel, are deliberately omitted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {"left": (50, 220, 110), "right": (255, 80, 80)}


def latest_session(root: Path) -> Path:
    sessions = [p.parent for p in root.glob("session_*/events.jsonl")]
    if not sessions:
        raise FileNotFoundError(f"No session events.jsonl under {root}")
    return max(sessions, key=lambda p: p.stat().st_mtime)


def mosaic_to_main(point: np.ndarray, image_size: tuple[int, int]) -> tuple[int, int] | None:
    point = np.asarray(point, dtype=np.float32).reshape(2)
    if not np.isfinite(point).all():
        return None
    mosaic = (point + 1.0) * 0.5 * np.asarray([383.0, 319.0], dtype=np.float32)
    x_model, y_model = map(float, mosaic)
    # Main camera occupies the upper 384x216 panel before the 2px crop.
    if not (0.0 <= x_model <= 383.0 and 0.0 <= y_model <= 215.0):
        return None
    width, height = image_size
    x = (x_model + 0.5) * width / 384.0 - 0.5
    y = (y_model + 2.0 + 0.5) * height / 216.0 - 0.5
    if not (0.0 <= x < width and 0.0 <= y < height):
        return None
    return int(round(x)), int(round(y))


def event_image(session: Path, row: dict, request_images: dict[int, Path]) -> Path | None:
    direct = row.get("cam_manip_high_path")
    if direct:
        path = session / str(direct)
        if path.is_file():
            return path
    request_id = row.get("source_request_id")
    return request_images.get(int(request_id)) if request_id is not None else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path, nargs="?", default=None)
    parser.add_argument("--records", type=Path, default=Path("~/uniwam_manip26_eefxy_visible_records").expanduser())
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    session = (args.session.expanduser() if args.session else latest_session(args.records)).resolve()
    events_path = session / "events.jsonl"
    if not events_path.is_file():
        raise FileNotFoundError(events_path)
    rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    request_images: dict[int, Path] = {}
    for row in rows:
        if row.get("event") != "observation":
            continue
        request_id = row.get("request_id")
        relative = row.get("image_paths", {}).get("cam_manip_high")
        if request_id is not None and relative and (session / str(relative)).is_file():
            request_images[int(request_id)] = session / str(relative)

    output = (args.output.expanduser() if args.output else session / "eefxy_visualization").resolve()
    if output.exists():
        suffix = 1
        base = output
        while output.exists():
            output = Path(f"{base}_{suffix}")
            suffix += 1
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)
    font = ImageFont.load_default()
    manifest = []
    skipped = 0
    for row in rows:
        if row.get("event") != "publish" or row.get("bbox_action") is None:
            continue
        values = np.asarray(row["bbox_action"], dtype=np.float32).reshape(-1)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid bbox_action/EEF-XY action at {row}")
        image_path = event_image(session, row, request_images)
        if image_path is None:
            skipped += 1
            continue
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        drawn = {}
        for arm, xy_slice, visible_index in (("left", slice(0, 2), 4), ("right", slice(2, 4), 5)):
            visible = float(values[visible_index]) > 0.0
            point = mosaic_to_main(values[xy_slice], image.size) if visible else None
            drawn[arm] = {"visible_slot": float(values[visible_index]), "drawn": point is not None}
            if point is not None:
                radius = max(5, image.width // 100)
                color = COLORS[arm]
                draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), outline=color, width=3)
                draw.line((point[0] - radius, point[1], point[0] + radius, point[1]), fill=color, width=2)
                draw.line((point[0], point[1] - radius, point[0], point[1] + radius), fill=color, width=2)
        label = f"EEF-XY action={row.get('action_index')} request={row.get('source_request_id')} step={row.get('source_chunk_step')}"
        draw.rectangle((0, 0, min(image.width, 7 * len(label) + 10), 20), fill=(0, 0, 0))
        draw.text((5, 4), label, fill=(255, 255, 255), font=font)
        frame_path = frames_dir / f"frame_{len(manifest):06d}.png"
        image.save(frame_path)
        manifest.append({"action_index": row.get("action_index"), "request_id": row.get("source_request_id"), "chunk_step": row.get("source_chunk_step"), "model_space_eef_xy_visible": values.tolist(), "drawn": drawn, "source_image": str(image_path)})

    if not manifest:
        raise ValueError("No publish EEF-XY records with cam_manip_high images found")
    video = output / "eefxy_visualization.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(args.fps), "-i", str(frames_dir / "frame_%06d.png"), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], check=True)
    payload = {"status": "PASS", "coordinate_contract": "normalized 384x320 mosaic -> upper main 384x216 panel -> raw cam_manip_high", "visibility_contract": "slot > 0 is drawn; slot <= 0 is omitted", "session": str(session), "frames": len(manifest), "skipped_without_image": skipped, "video": str(video), "records": manifest}
    (output / "visualization_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"OUTPUT_DIR={output}\nVIDEO={video}\nFRAMES={len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
