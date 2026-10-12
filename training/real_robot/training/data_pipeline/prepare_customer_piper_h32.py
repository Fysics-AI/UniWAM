#!/usr/bin/env python3
"""Validate a camera-frame Piper LeRobot source and build H32 training windows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from prepare_customer_piper_views import camera_state


ARM_COLUMNS = ("observation.state.camera_dual_arm", "action.manip.camera_dual_arm")
CAMERA_KEYS = (
    "observation.images.cam_manip_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _calibration(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text())
    result = {}
    for key in ("camera_from_left_base", "camera_from_right_base"):
        value = np.asarray(payload[key], dtype=np.float64)
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError(f"{key} must be a finite 4x4 matrix")
        if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8):
            raise ValueError(f"{key} has an invalid homogeneous row")
        rotation = value[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-4
        ):
            raise ValueError(f"{key} is not a proper rigid transform")
        result[key] = value
    return result


def _check_arm(values: np.ndarray, label: str) -> None:
    if values.ndim != 2 or values.shape[1] != 20 or not np.isfinite(values).all():
        raise ValueError(f"{label} must contain finite [T,20] values, got {values.shape}")
    if len(values) == 0:
        raise ValueError(f"{label} is empty")
    for offset in (3, 13):
        rows = values[:, offset : offset + 6].reshape(-1, 2, 3)
        lengths = np.linalg.norm(rows, axis=-1)
        dot = np.sum(rows[:, 0] * rows[:, 1], axis=-1)
        if np.max(np.abs(lengths - 1.0)) > 0.1 or np.max(np.abs(dot)) > 0.1:
            raise ValueError(f"{label} contains invalid row-major Rot6D arm poses")
    gripper = values[:, [9, 19]]
    if np.min(gripper) < -0.005 or np.max(gripper) > 0.12:
        raise ValueError(f"{label} grippers must be in meters, approximately [0,0.105]")


def prepare(dataset_root: Path, calibration_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    info = json.loads((dataset_root / "meta/info.json").read_text())
    recorded = info.get("fastwam_camera_frame_conversion")
    if recorded is None:
        raise ValueError("Dataset lacks camera-frame conversion metadata; run prepare_customer_piper_views.py first")
    if recorded.get("rotation_serialization") != "row_major_rot6d_v1":
        raise ValueError("Dataset was not built with the customer row-major Rot6D converter")
    input_frame = recorded.get("input_frame", "base")
    if input_frame not in ("base", "camera"):
        raise ValueError(f"Unsupported customer input frame: {input_frame!r}")
    calibration = _calibration(calibration_path)
    for key, value in calibration.items():
        if not np.allclose(np.asarray(recorded[key], dtype=np.float64), value, atol=1e-6):
            raise ValueError(f"Training dataset {key} differs from {calibration_path}")
    features = info.get("features", {})
    missing = set(ARM_COLUMNS + CAMERA_KEYS) - set(features)
    if missing:
        raise ValueError(f"LeRobot metadata lacks required columns/cameras: {sorted(missing)}")
    tasks = dataset_root / "meta/tasks.jsonl"
    if not tasks.is_file() or not any(line.strip() for line in tasks.read_text().splitlines()):
        raise ValueError("Dataset needs nonempty meta/tasks.jsonl with customer prompts")

    episodes = [json.loads(line) for line in (dataset_root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    if not episodes:
        raise ValueError("Dataset has no episodes")
    rows: dict[str, list] = {
        "episode_index": [], "source_episode_index": [], "start_frame": [],
        "horizon": [], "sample_type": [], "nav_loss_valid": [],
        "manip_loss_valid": [], "nav_active_count": [], "manip_active_count": [],
    }
    seen = set()
    for episode in episodes:
        index = int(episode["episode_index"])
        if index in seen:
            raise ValueError(f"Duplicate episode_index={index}")
        seen.add(index)
        path = dataset_root / f"data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet"
        table = pq.read_table(path, columns=[
            *ARM_COLUMNS, "observation.state", "action.manip", "episode_index", "frame_index",
        ])
        length = table.num_rows
        if length != int(episode["length"]):
            raise ValueError(f"Episode {index} metadata length disagrees with Parquet")
        if not np.all(np.asarray(table["episode_index"].to_pylist()) == index):
            raise ValueError(f"Episode {index} has inconsistent episode_index values")
        if not np.array_equal(np.asarray(table["frame_index"].to_pylist()), np.arange(length)):
            raise ValueError(f"Episode {index} frame_index is not contiguous from zero")
        for name in ARM_COLUMNS:
            _check_arm(np.asarray(table[name].to_pylist(), dtype=np.float32), f"episode {index} {name}")
        raw_state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        raw_action = np.asarray(table["action.manip"].to_pylist(), dtype=np.float32)
        expected_state_dim = int(recorded.get("input_state_dim", raw_state.shape[-1]))
        if expected_state_dim not in (20, 23) or raw_state.shape != (length, expected_state_dim) or raw_action.shape != (length, 20):
            raise ValueError(f"Episode {index} requires 20D dual-arm state and 20D action; optional base3 makes state 23D")
        if input_frame == "camera" and expected_state_dim != 20:
            raise ValueError("Camera-frame input must contain only the 20D dual-arm state")
        for name, raw in ((ARM_COLUMNS[0], raw_state[:, -20:]), (ARM_COLUMNS[1], raw_action)):
            rebuilt = (
                camera_state(raw, calibration["camera_from_left_base"], calibration["camera_from_right_base"])
                if input_frame == "base" else raw
            )
            stored = np.asarray(table[name].to_pylist(), dtype=np.float32)
            if not np.allclose(rebuilt, stored, atol=1e-4):
                raise ValueError(f"Episode {index} {name} disagrees with the declared {input_frame}-frame arm pose")
        count = max(length - 32, 0)
        values = {
            "episode_index": index,
            "source_episode_index": int(episode.get("source_episode_index", index)),
            "horizon": 32,
            "sample_type": "pure_manip",
            "nav_loss_valid": False,
            "manip_loss_valid": True,
            "nav_active_count": 0,
            "manip_active_count": 32,
        }
        for key, value in values.items():
            rows[key].extend([value] * count)
        rows["start_frame"].extend(range(count))
    if not rows["start_frame"]:
        raise ValueError("No H32 windows; each usable episode needs at least 33 frames")
    arrays = {
        key: pa.array(values, type=(pa.string() if key == "sample_type" else
                                    pa.bool_() if key.endswith("_valid") else pa.int64()))
        for key, values in rows.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    try:
        pq.write_table(pa.table(arrays), temporary, compression="zstd")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"episodes": len(episodes), "windows": len(rows["start_frame"]), "model_state_dim": 20, "input_frame": input_frame, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset_root, args.calibration, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
