#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from uniwam.datasets.lerobot.dual_expert_robot_video_dataset import (
    _manip_abs_to_chunk_delta_np,
)
from uniwam.datasets.lerobot.rot6d import (
    apply_fixed_frame_relative_pose_rot6d_np,
    rotation_6d_to_matrix_np,
    rotation_geodesic_angle_np,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/camera_frame_cross_embodiment_h32_from_scratch.yaml"
STATS = (
    ROOT
    / "data_indices_camera_frame_h32"
    / "camera_frame_cross_embodiment_h32_fixed_reference_q01q99.json"
)
OLD_STATS = (
    ROOT
    / "data_indices_camera_frame_h32"
    / "camera_frame_cross_embodiment_h32_future_state_q01q99.json"
)


def child_dataset(wrapper):
    dataset = getattr(wrapper, "dataset", None)
    if dataset is None:
        raise TypeError(f"Expected routed/annotated dataset, got {type(wrapper).__name__}.")
    return dataset


def representative_source_index(wrapper) -> int:
    if hasattr(wrapper, "window_branch"):
        candidates = np.flatnonzero(np.asarray(wrapper.window_branch, dtype=object) == "manip")
        if not candidates.size:
            raise ValueError("Phase-routed source has no manipulation windows.")
        return int(np.asarray(wrapper.source_window_index, dtype=np.int64)[candidates[0]])
    return 0


def reconstruction_error(base, source_index: int) -> tuple[float, float]:
    episode = int(base.window_episode_index[source_index])
    start = int(base.window_start_frame[source_index])
    state, actions, _ = base._cached_episode_arrays(episode)
    indices = np.minimum(
        start + np.arange(base.action_horizon, dtype=np.int64),
        len(actions["manip"]) - 1,
    )
    absolute = actions["manip"][indices]
    current = state[start]
    delta = _manip_abs_to_chunk_delta_np(
        absolute[None],
        current[None],
        relative_frame="fixed_reference",
    )[0]

    left_start, right_start = (3, 13) if current.shape[-1] == 23 else (0, 10)
    xyz_errors = []
    rotation_errors = []
    for state_start, action_start in ((left_start, 0), (right_start, 10)):
        rebuilt_xyz, rebuilt_rot6d = apply_fixed_frame_relative_pose_rot6d_np(
            current[state_start : state_start + 3],
            current[state_start + 3 : state_start + 9],
            delta[:, action_start : action_start + 3],
            delta[:, action_start + 3 : action_start + 9],
        )
        xyz_errors.append(
            float(
                np.linalg.norm(
                    rebuilt_xyz - absolute[:, action_start : action_start + 3], axis=-1
                ).max()
            )
        )
        rebuilt_rotation = rotation_6d_to_matrix_np(rebuilt_rot6d)
        target_rotation = rotation_6d_to_matrix_np(
            absolute[:, action_start + 3 : action_start + 9]
        )
        rotation_errors.append(
            float(rotation_geodesic_angle_np(rebuilt_rotation, target_rotation).max())
        )
    return max(xyz_errors), max(rotation_errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    stats = json.loads(STATS.read_text(encoding="utf-8"))
    old_stats = json.loads(OLD_STATS.read_text(encoding="utf-8"))
    if stats["contract"]["status"] != "PASS":
        raise ValueError("Fixed-reference stats contract is not PASS.")
    if stats["contract"]["manip_relative_frame"] != "fixed_reference":
        raise ValueError("Stats do not declare fixed_reference manipulation actions.")
    if stats["state"] != old_stats["state"]:
        raise ValueError("State stats changed while rebuilding manipulation actions.")
    if stats["action"]["nav"] != old_stats["action"]["nav"]:
        raise ValueError("Navigation stats changed while rebuilding manipulation actions.")
    manip_stats = stats["action"]["manip"]
    for begin, end in ((3, 9), (13, 19)):
        if manip_stats["global_q01"][begin:end] != [-1.0] * 6:
            raise ValueError("Rot6D q01 must remain -1 in every rotation dimension.")
        if manip_stats["global_q99"][begin:end] != [1.0] * 6:
            raise ValueError("Rot6D q99 must remain +1 in every rotation dimension.")

    data_cfg = OmegaConf.load(CONFIG)
    cfg = OmegaConf.create({"data": data_cfg})
    train = instantiate(cfg.data.train)
    if len(train.datasets) != 4:
        raise ValueError(f"Expected four sources, got {len(train.datasets)}.")

    source_results = []
    for source_id, wrapper in enumerate(train.datasets):
        base = child_dataset(wrapper)
        if base.manip_relative_frame != "fixed_reference":
            raise ValueError(
                f"Source {source_id} uses {base.manip_relative_frame}, not fixed_reference."
            )
        if base.action_horizon != 32 or base.manip_action_dim != 20:
            raise ValueError(f"Source {source_id} is not H32/20D.")
        source_index = representative_source_index(wrapper)
        xyz_error, rotation_error = reconstruction_error(base, source_index)
        if xyz_error > 2.0e-6 or rotation_error > 5.0e-4:
            raise ValueError(
                f"Source {source_id} reconstruction failed: "
                f"xyz={xyz_error}, rotation={rotation_error}."
            )
        source_results.append(
            {
                "source_id": source_id,
                "dataset_root": str(base.dataset_root),
                "source_window_index": source_index,
                "max_xyz_reconstruction_error": xyz_error,
                "max_rotation_reconstruction_error_rad": rotation_error,
            }
        )

    report = {
        "status": "PASS",
        "action_contract": (
            "fixed main-camera reference: dp=p_target-p_start; "
            "dR=R_target@R_start^T"
        ),
        "state_contract": "20D absolute dual-arm pose in manipulation main-camera frame",
        "rotation_normalization": "disabled via q01/q99=-1/+1 for both Rot6D blocks",
        "state_stats_unchanged": True,
        "nav_stats_unchanged": True,
        "sources": source_results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
