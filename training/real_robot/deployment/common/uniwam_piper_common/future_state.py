"""VLASH queue-endpoint state construction shared by cloud and ROS edge."""

from __future__ import annotations

import numpy as np

from .rotation6d import matrix_to_rotation_6d, relative_pose, rpy_xyz_to_matrix


def _se2_exp(integrated_twist: np.ndarray) -> np.ndarray:
    twist = np.asarray(integrated_twist)
    theta = twist[..., 2]
    small = np.abs(theta) < 1.0e-4
    safe = np.where(small, 1.0, theta)
    theta2 = theta * theta
    a = np.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, np.sin(theta) / safe)
    b = np.where(small, theta / 2.0 - theta * theta2 / 24.0, (1.0 - np.cos(theta)) / safe)
    return np.stack((a * twist[..., 0] - b * twist[..., 1],
                     b * twist[..., 0] + a * twist[..., 1], theta), axis=-1)


def _compose_se2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left, right = np.asarray(left), np.asarray(right)
    c, s = np.cos(left[..., 2]), np.sin(left[..., 2])
    return np.stack((left[..., 0] + c * right[..., 0] - s * right[..., 1],
                     left[..., 1] + s * right[..., 0] + c * right[..., 1],
                     left[..., 2] + right[..., 2]), axis=-1)


def _inverse_se2(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose)
    c, s = np.cos(pose[..., 2]), np.sin(pose[..., 2])
    return np.stack((-c * pose[..., 0] - s * pose[..., 1],
                     s * pose[..., 0] - c * pose[..., 1], -pose[..., 2]), axis=-1)


def _se2_log(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose)
    theta = pose[..., 2]
    small = np.abs(theta) < 1.0e-4
    safe = np.where(small, 1.0, theta)
    theta2 = theta * theta
    a = np.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, np.sin(theta) / safe)
    b = np.where(small, theta / 2.0 - theta * theta2 / 24.0, (1.0 - np.cos(theta)) / safe)
    denominator = np.maximum(a * a + b * b, np.finfo(pose.dtype).eps)
    return np.stack(((a * pose[..., 0] + b * pose[..., 1]) / denominator,
                     (-b * pose[..., 0] + a * pose[..., 1]) / denominator, theta), axis=-1)


def queued_nav_future_pose(commands: np.ndarray, control_hz: float = 30.0) -> np.ndarray:
    commands = np.asarray(commands, dtype=np.float32)
    if commands.ndim != 2 or commands.shape[1] != 3:
        raise ValueError(f"commands must be [T,3], got {commands.shape}.")
    if control_hz <= 0:
        raise ValueError("control_hz must be positive.")
    pose = np.zeros(3, dtype=commands.dtype)
    for command in commands:
        pose = _compose_se2(pose, _se2_exp(command / float(control_hz)))
    return pose.astype(np.float32)


def queued_nav_target_path(commands: np.ndarray, control_hz: float = 30.0) -> np.ndarray:
    """Integrate queued body commands into one snapshot-relative SE(2) target per step."""
    commands = np.asarray(commands, dtype=np.float32)
    if commands.ndim != 2 or commands.shape[1] != 3:
        raise ValueError(f"commands must be [T,3], got {commands.shape}.")
    if control_hz <= 0:
        raise ValueError("control_hz must be positive.")
    pose = np.zeros(3, dtype=np.float32)
    output = np.empty((commands.shape[0], 3), dtype=np.float32)
    for index, command in enumerate(commands):
        pose = _compose_se2(pose, _se2_exp(command / float(control_hz))).astype(np.float32)
        output[index] = pose
    return output


def nav_target_path_to_body_velocity(
    future_pose: np.ndarray, target_path: np.ndarray, control_hz: float = 30.0
) -> np.ndarray:
    target = np.asarray(target_path, dtype=np.float32)
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(f"target_path must be [H,3], got {target.shape}.")
    path = np.concatenate((np.asarray(future_pose, dtype=np.float32).reshape(1, 3), target), axis=0)
    relative = _compose_se2(_inverse_se2(path[:-1]), path[1:])
    return (_se2_log(relative) * float(control_hz)).astype(np.float32)


def eef14_rpy_to_manip20(eef: np.ndarray) -> np.ndarray:
    eef = np.asarray(eef, dtype=np.float32).reshape(14)
    result = np.empty(20, dtype=np.float32)
    result[0:3] = eef[0:3]
    result[3:9] = matrix_to_rotation_6d(rpy_xyz_to_matrix(eef[3:6]))
    result[9] = eef[6]
    result[10:13] = eef[7:10]
    result[13:19] = matrix_to_rotation_6d(rpy_xyz_to_matrix(eef[10:13]))
    result[19] = eef[13]
    return result


def state23_absolute_manip(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(23)
    return np.concatenate((state[3:13], state[13:23])).astype(np.float32)


def snapshot_relative_robot_state_rot6d(
    snapshot_state: np.ndarray, future_base_pose: np.ndarray, future_manip_target: np.ndarray
) -> np.ndarray:
    snapshot = np.asarray(snapshot_state, dtype=np.float32).reshape(23)
    manip = np.asarray(future_manip_target, dtype=np.float32).reshape(20)
    output = np.empty(23, dtype=np.float32)
    output[:3] = np.asarray(future_base_pose, dtype=np.float32).reshape(3)
    output[3:6], output[6:12] = relative_pose(
        snapshot[3:6], snapshot[6:12], manip[0:3], manip[3:9]
    )
    output[12] = manip[9]
    output[13:16], output[16:22] = relative_pose(
        snapshot[13:16], snapshot[16:22], manip[10:13], manip[13:19]
    )
    output[22] = manip[19]
    if not np.all(np.isfinite(output)):
        raise ValueError("future_proprio contains NaN or Inf.")
    return output
