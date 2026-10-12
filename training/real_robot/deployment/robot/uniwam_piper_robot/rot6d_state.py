"""ROS-edge state assembly for the 23D SE(3)-anchored rot6D policy."""

from __future__ import annotations

import numpy as np

from uniwam_piper_common.rotation6d import (
    matrix_to_rotation_6d,
    quaternion_xyzw_to_matrix,
    rotation_6d_to_matrix,
    rpy_xyz_to_matrix,
)


STATE_DIM = 23


def _vector(name: str, value: np.ndarray, length: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or Inf.")
    return result


def compose_state23(
    base_command: np.ndarray,
    left_pose_eef7: np.ndarray,
    left_joint7: np.ndarray,
    right_pose_eef7: np.ndarray,
    right_joint7: np.ndarray,
) -> np.ndarray:
    """Build ``base3 + left(xyz,rot6d,gripper) + right(...)`` from ROS data."""
    base = _vector("base_command", base_command, 3)
    left_pose = _vector("left_pose_eef7", left_pose_eef7, 7)
    left_joint = _vector("left_joint7", left_joint7, 7)
    right_pose = _vector("right_pose_eef7", right_pose_eef7, 7)
    right_joint = _vector("right_joint7", right_joint7, 7)
    result = np.empty(STATE_DIM, dtype=np.float32)
    result[:3] = base
    result[3:6] = left_pose[:3]
    result[6:12] = matrix_to_rotation_6d(quaternion_xyzw_to_matrix(left_pose[3:7]))
    result[12] = left_joint[6]
    result[13:16] = right_pose[:3]
    result[16:22] = matrix_to_rotation_6d(quaternion_xyzw_to_matrix(right_pose[3:7]))
    result[22] = right_joint[6]
    return result


def state23_from_rpy(base_command: np.ndarray, eef14_rpy: np.ndarray) -> np.ndarray:
    """Convert a legacy wire state into the exact 23D training representation."""
    base = _vector("base_command", base_command, 3)
    eef = _vector("eef14_rpy", eef14_rpy, 14)
    result = np.empty(STATE_DIM, dtype=np.float32)
    result[:3] = base
    result[3:6] = eef[:3]
    result[6:12] = matrix_to_rotation_6d(rpy_xyz_to_matrix(eef[3:6]))
    result[12] = eef[6]
    result[13:16] = eef[7:10]
    result[16:22] = matrix_to_rotation_6d(rpy_xyz_to_matrix(eef[10:13]))
    result[22] = eef[13]
    return result


def validate_state23(value: np.ndarray) -> np.ndarray:
    """Validate a received 23D state and return an owning float32 copy."""
    state = _vector("state23", value, STATE_DIM).copy()
    rotation_6d_to_matrix(state[6:12])
    rotation_6d_to_matrix(state[16:22])
    return state
