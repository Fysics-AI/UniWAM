#!/usr/bin/env python3
"""Create a dry-run Piper client config from the training camera calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def generate(base: Path, dataset_root: Path, calibration_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    calibration = json.loads(calibration_path.read_text())
    conversion = json.loads((dataset_root / "meta/info.json").read_text()).get(
        "fastwam_camera_frame_conversion"
    )
    if conversion is None:
        raise ValueError("Dataset lacks camera-frame conversion metadata")
    cfg = OmegaConf.load(base)
    for key in ("camera_from_left_base", "camera_from_right_base"):
        value = np.asarray(calibration[key], dtype=np.float64)
        recorded = np.asarray(conversion[key], dtype=np.float64)
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key}")
        rotation = value[:3, :3]
        if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8) or not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-4
        ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
            raise ValueError(f"{key} must be a rigid transform")
        if not np.allclose(value, recorded, atol=1e-6):
            raise ValueError(f"{key} does not match the training dataset")
        cfg.camera_frame[key] = value.tolist()
    cfg.camera_frame.extrinsics_profile = "customer_piper_camera_frame"
    cfg.inference_mode = "manip_only"
    cfg.control.dry_run = True
    cfg.base.dry_run = True
    cfg.base.enabled = False
    cfg.task_prompt = "Set the exact task prompt from your dataset before running"
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[1] / "configs/uniwam_robot_client_camera_frame_h32.yaml")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.base, args.dataset_root, args.calibration, args.output)
    print(f"CUSTOMER_PIPER_DRY_RUN_CONFIG_READY {args.output}")


if __name__ == "__main__":
    main()
