#!/usr/bin/env python3
"""Convert Piper arm poses to the row-major camera-frame contract for new data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prepare_agilex_camera_views import build_source, load_transform


def camera_state(base_arm: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    base_arm = np.asarray(base_arm, dtype=np.float32)
    if base_arm.ndim != 2 or base_arm.shape[1] != 20 or not np.isfinite(base_arm).all():
        raise ValueError("Expected finite [T,20] Piper arm poses")
    output = np.empty_like(base_arm)
    for start, transform in ((0, left), (10, right)):
        rows = base_arm[:, start + 3 : start + 9].reshape(-1, 2, 3).astype(np.float64)
        first_norm = np.linalg.norm(rows[:, 0], axis=-1, keepdims=True)
        if np.any(first_norm < 1e-6):
            raise ValueError("Input Rot6D has a degenerate first row")
        first = rows[:, 0] / first_norm
        second = rows[:, 1] - np.sum(first * rows[:, 1], axis=-1, keepdims=True) * first
        second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
        if np.any(second_norm < 1e-6):
            raise ValueError("Input Rot6D has parallel or degenerate rows")
        second /= second_norm
        rotation = np.stack((first, second, np.cross(first, second)), axis=-2)
        camera_rotation = transform[:3, :3] @ rotation
        output[:, start : start + 3] = base_arm[:, start : start + 3] @ transform[:3, :3].T + transform[:3, 3]
        output[:, start + 3 : start + 9] = camera_rotation[:, :2, :].reshape(-1, 6)
        output[:, start + 9] = base_arm[:, start + 9]
    return output


def already_camera_state(arm: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    del left, right
    arm = np.asarray(arm, dtype=np.float32)
    if arm.ndim != 2 or arm.shape[1] != 20 or not np.isfinite(arm).all():
        raise ValueError("Expected finite [T,20] camera-frame arm poses")
    return arm.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--input-frame", choices=("base", "camera"), default="base")
    parser.add_argument("--raw-state-layout", choices=("arm20", "base3_arm20"), default="arm20")
    args = parser.parse_args()
    if args.input_frame == "camera" and args.raw_state_layout != "arm20":
        parser.error("Camera-frame input must contain only the 20D dual-arm state")
    calibration = json.loads(args.calibration.read_text())
    left = load_transform(calibration["camera_from_left_base"], "camera_from_left_base")
    right = load_transform(calibration["camera_from_right_base"], "camera_from_right_base")
    build_source(
        args.source_root, args.output_root, left, right,
        convert_state=camera_state if args.input_frame == "base" else already_camera_state,
        rotation_serialization="row_major_rot6d_v1",
        input_state_dim=20 if args.raw_state_layout == "arm20" else 23,
        input_frame=args.input_frame,
    )


if __name__ == "__main__":
    main()
