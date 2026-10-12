"""Queue-state and target-path operations shared by cloud and edge deployment."""

from __future__ import annotations

import numpy as np
import torch

from uniwam.datasets.lerobot.rot6d import relative_pose_rot6d_np

from .se2 import integrate_body_twist, se2_path_to_body_twist


def queued_nav_future_pose(commands, control_hz: float = 30.0):
    """Return the snapshot-relative SE(2) pose at the end of queued commands."""
    if control_hz <= 0:
        raise ValueError("control_hz must be positive.")
    if commands.shape[-1] != 3:
        raise ValueError(f"commands must end in 3, got {tuple(commands.shape)}")
    if commands.shape[0] == 0:
        return (
            torch.zeros(3, dtype=commands.dtype, device=commands.device)
            if isinstance(commands, torch.Tensor)
            else np.zeros(3, dtype=np.asarray(commands).dtype)
        )
    return integrate_body_twist(commands, 1.0 / float(control_hz))[-1]


def nav_target_path_to_body_velocity(
    future_pose,
    target_path,
    control_hz: float = 30.0,
):
    """Convert snapshot-relative SE(2) targets after a queue endpoint to body twists."""
    if target_path.ndim != 2 or target_path.shape[-1] != 3:
        raise ValueError(f"target_path must be [H,3], got {tuple(target_path.shape)}")
    if isinstance(target_path, torch.Tensor):
        path = torch.cat((future_pose.reshape(1, 3).to(target_path), target_path), dim=0)
    else:
        path = np.concatenate(
            (np.asarray(future_pose, dtype=target_path.dtype).reshape(1, 3), target_path), axis=0
        )
    return se2_path_to_body_twist(path, 1.0 / float(control_hz))


def snapshot_relative_robot_state_rot6d(
    snapshot_state: np.ndarray,
    future_base_pose: np.ndarray,
    future_manip_target: np.ndarray,
) -> np.ndarray:
    """Build the model's 23D future-state condition from queued endpoints.

    ``snapshot_state`` is the measured absolute arm state. ``future_base_pose``
    is already relative to the snapshot. ``future_manip_target`` is the queued
    absolute 20D dual-arm Cartesian target.
    """
    snapshot = np.asarray(snapshot_state, dtype=np.float32)
    manip = np.asarray(future_manip_target, dtype=np.float32)
    if snapshot.shape != (23,) or manip.shape != (20,):
        raise ValueError(f"Expected snapshot [23] and manip [20], got {snapshot.shape}/{manip.shape}.")
    output = np.empty(23, dtype=np.float32)
    output[:3] = np.asarray(future_base_pose, dtype=np.float32).reshape(3)
    output[3:6], output[6:12] = relative_pose_rot6d_np(
        snapshot[3:6], snapshot[6:12], manip[0:3], manip[3:9]
    )
    output[12] = manip[9]
    output[13:16], output[16:22] = relative_pose_rot6d_np(
        snapshot[13:16], snapshot[16:22], manip[10:13], manip[13:19]
    )
    output[22] = manip[19]
    return output
