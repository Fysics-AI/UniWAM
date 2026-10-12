"""Exact planar rigid-body operations used by VLASH future-state training.

Poses are ``[..., (x, y, yaw)]`` and twists are body-frame
``[..., (vx, vy, wz)]``. Angles remain unwrapped when trajectories are
integrated; only relative pose construction uses rotation matrices, so the
short-horizon model target has a unique principal-angle representation.
"""

from __future__ import annotations

import numpy as np
import torch


def _coefficients_np(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta)
    small = np.abs(theta) < 1.0e-4
    theta2 = theta * theta
    safe_theta = np.where(small, 1.0, theta)
    a = np.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, np.sin(theta) / safe_theta)
    b = np.where(
        small,
        theta / 2.0 - theta * theta2 / 24.0 + theta * theta2 * theta2 / 720.0,
        (1.0 - np.cos(theta)) / safe_theta,
    )
    return a, b


def _coefficients_torch(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    small = theta.abs() < 1.0e-4
    theta2 = theta.square()
    safe_theta = torch.where(small, torch.ones_like(theta), theta)
    a_exact = torch.sin(theta) / safe_theta
    b_exact = (1.0 - torch.cos(theta)) / safe_theta
    a = torch.where(small, 1.0 - theta2 / 6.0 + theta2.square() / 120.0, a_exact)
    b = torch.where(
        small,
        theta / 2.0 - theta * theta2 / 24.0 + theta * theta2.square() / 720.0,
        b_exact,
    )
    return a, b


def se2_exp(twist):
    """Map an integrated body twist ``[vx*dt, vy*dt, wz*dt]`` to an SE(2) pose."""
    if isinstance(twist, torch.Tensor):
        theta = twist[..., 2]
        a, b = _coefficients_torch(theta)
        x = a * twist[..., 0] - b * twist[..., 1]
        y = b * twist[..., 0] + a * twist[..., 1]
        return torch.stack((x, y, theta), dim=-1)
    twist = np.asarray(twist)
    theta = twist[..., 2]
    a, b = _coefficients_np(theta)
    x = a * twist[..., 0] - b * twist[..., 1]
    y = b * twist[..., 0] + a * twist[..., 1]
    return np.stack((x, y, theta), axis=-1)


def se2_log(pose):
    """Map an SE(2) pose to its integrated body twist on the principal branch."""
    if isinstance(pose, torch.Tensor):
        theta = pose[..., 2]
        a, b = _coefficients_torch(theta)
        denominator = (a.square() + b.square()).clamp_min(torch.finfo(pose.dtype).eps)
        vx = (a * pose[..., 0] + b * pose[..., 1]) / denominator
        vy = (-b * pose[..., 0] + a * pose[..., 1]) / denominator
        return torch.stack((vx, vy, theta), dim=-1)
    pose = np.asarray(pose)
    theta = pose[..., 2]
    a, b = _coefficients_np(theta)
    denominator = np.maximum(a * a + b * b, np.finfo(pose.dtype).eps)
    vx = (a * pose[..., 0] + b * pose[..., 1]) / denominator
    vy = (-b * pose[..., 0] + a * pose[..., 1]) / denominator
    return np.stack((vx, vy, theta), axis=-1)


def compose_se2(left, right):
    """Compose planar poses, preserving the unwrapped sum of their headings."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        left = torch.as_tensor(left)
        right = torch.as_tensor(right, device=left.device, dtype=left.dtype)
        c, s = torch.cos(left[..., 2]), torch.sin(left[..., 2])
        x = left[..., 0] + c * right[..., 0] - s * right[..., 1]
        y = left[..., 1] + s * right[..., 0] + c * right[..., 1]
        return torch.stack((x, y, left[..., 2] + right[..., 2]), dim=-1)
    left, right = np.asarray(left), np.asarray(right)
    c, s = np.cos(left[..., 2]), np.sin(left[..., 2])
    x = left[..., 0] + c * right[..., 0] - s * right[..., 1]
    y = left[..., 1] + s * right[..., 0] + c * right[..., 1]
    return np.stack((x, y, left[..., 2] + right[..., 2]), axis=-1)


def inverse_se2(pose):
    if isinstance(pose, torch.Tensor):
        c, s = torch.cos(pose[..., 2]), torch.sin(pose[..., 2])
        x = -c * pose[..., 0] - s * pose[..., 1]
        y = s * pose[..., 0] - c * pose[..., 1]
        return torch.stack((x, y, -pose[..., 2]), dim=-1)
    pose = np.asarray(pose)
    c, s = np.cos(pose[..., 2]), np.sin(pose[..., 2])
    x = -c * pose[..., 0] - s * pose[..., 1]
    y = s * pose[..., 0] - c * pose[..., 1]
    return np.stack((x, y, -pose[..., 2]), axis=-1)


def relative_se2(anchor, target):
    return compose_se2(inverse_se2(anchor), target)


def integrate_body_twist(velocity, dt: float, initial_pose=None):
    """Integrate ``T`` body-frame commands and return ``T+1`` poses."""
    is_torch = isinstance(velocity, torch.Tensor)
    velocity = velocity if is_torch else np.asarray(velocity)
    if velocity.ndim != 2 or velocity.shape[-1] != 3:
        raise ValueError(f"velocity must be [T,3], got {tuple(velocity.shape)}")
    if initial_pose is None:
        initial_pose = (
            torch.zeros(3, dtype=velocity.dtype, device=velocity.device)
            if is_torch
            else np.zeros(3, dtype=velocity.dtype)
        )
    poses = [initial_pose]
    for command in velocity:
        poses.append(compose_se2(poses[-1], se2_exp(command * float(dt))))
    return torch.stack(poses) if is_torch else np.stack(poses)


def se2_path_to_body_twist(path, dt: float):
    """Recover body velocity between consecutive poses of an SE(2) path."""
    if path.ndim != 2 or path.shape[-1] != 3 or path.shape[0] < 2:
        raise ValueError(f"path must be [T+1,3], got {tuple(path.shape)}")
    return se2_log(relative_se2(path[:-1], path[1:])) / float(dt)
