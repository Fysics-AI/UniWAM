"""GR00T-compatible row-major 6D rotation helpers.

Isaac-GR00T N1.7 serializes a rotation as the first two *rows* of its
3x3 matrix.  This module deliberately keeps that convention everywhere so
dataset conversion, relative-action construction, statistics, and deployment
all agree on the six scalar ordering.
"""

from __future__ import annotations

import numpy as np
import torch


ROT6D_CONVENTION = "first_two_rotation_matrix_rows_row_major"


def _normalize_np(values: np.ndarray, *, eps: float, name: str) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm < eps):
        raise ValueError(f"{name} contains a zero-length or non-finite vector.")
    return values / norm


def quaternion_xyzw_to_matrix_np(quaternion: np.ndarray, *, eps: float = 1.0e-8) -> np.ndarray:
    """Convert normalized or unnormalized xyzw quaternions to rotation matrices."""
    q = _normalize_np(np.asarray(quaternion, dtype=np.float64), eps=eps, name="quaternion")
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    return np.stack(
        (
            1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw),
            2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw),
            2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def matrix_to_rotation_6d_np(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in [3,3], got {matrix.shape}.")
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,)).astype(np.float32)


def rotation_6d_to_matrix_np(rotation_6d: np.ndarray, *, eps: float = 1.0e-8) -> np.ndarray:
    """Decode row-major 6D rotations with GR00T's Gram--Schmidt rule."""
    value = np.asarray(rotation_6d, dtype=np.float64)
    if value.shape[-1] != 6:
        raise ValueError(f"rotation_6d must end in 6 values, got {value.shape}.")
    row1 = _normalize_np(value[..., :3], eps=eps, name="rotation_6d first row")
    row2 = value[..., 3:6] - np.sum(row1 * value[..., 3:6], axis=-1, keepdims=True) * row1
    row2 = _normalize_np(row2, eps=eps, name="rotation_6d second row")
    row3 = np.cross(row1, row2)
    return np.stack((row1, row2, row3), axis=-2).astype(np.float32)


def rotation_6d_to_matrix_torch(rotation_6d: torch.Tensor, *, eps: float = 1.0e-8) -> torch.Tensor:
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"rotation_6d must end in 6 values, got {tuple(rotation_6d.shape)}.")
    row1 = torch.nn.functional.normalize(rotation_6d[..., :3], dim=-1, eps=eps)
    row2 = rotation_6d[..., 3:6] - (row1 * rotation_6d[..., 3:6]).sum(dim=-1, keepdim=True) * row1
    row2 = torch.nn.functional.normalize(row2, dim=-1, eps=eps)
    row3 = torch.cross(row1, row2, dim=-1)
    return torch.stack((row1, row2, row3), dim=-2)


def matrix_to_rotation_6d_torch(matrix: torch.Tensor) -> torch.Tensor:
    if tuple(matrix.shape[-2:]) != (3, 3):
        raise ValueError(f"rotation matrix must end in [3,3], got {tuple(matrix.shape)}.")
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,))


def relative_pose_rot6d_np(
    current_xyz: np.ndarray,
    current_rot6d: np.ndarray,
    target_xyz: np.ndarray,
    target_rot6d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``T_current^-1 @ T_target`` as local xyz and row-major 6D rotation."""
    current_rotation = rotation_6d_to_matrix_np(current_rot6d)
    target_rotation = rotation_6d_to_matrix_np(target_rot6d)
    relative_rotation = np.matmul(np.swapaxes(current_rotation, -1, -2), target_rotation)
    delta_xyz = np.asarray(target_xyz, dtype=np.float64) - np.asarray(current_xyz, dtype=np.float64)
    relative_xyz = np.matmul(
        np.swapaxes(current_rotation, -1, -2), delta_xyz[..., None]
    )[..., 0]
    return relative_xyz.astype(np.float32), matrix_to_rotation_6d_np(relative_rotation)


def relative_pose_rot6d_torch(
    current_xyz: torch.Tensor,
    current_rot6d: torch.Tensor,
    target_xyz: torch.Tensor,
    target_rot6d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch equivalent of :func:`relative_pose_rot6d_np`."""
    current_rotation = rotation_6d_to_matrix_torch(current_rot6d)
    target_rotation = rotation_6d_to_matrix_torch(target_rot6d)
    current_inverse = current_rotation.transpose(-1, -2)
    relative_rotation = current_inverse @ target_rotation
    relative_xyz = (current_inverse @ (target_xyz - current_xyz).unsqueeze(-1)).squeeze(-1)
    return relative_xyz, matrix_to_rotation_6d_torch(relative_rotation)


def fixed_frame_relative_pose_rot6d_np(
    current_xyz: np.ndarray,
    current_rot6d: np.ndarray,
    target_xyz: np.ndarray,
    target_rot6d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return translation and spatial rotation deltas in one fixed reference frame.

    All input poses must already be expressed in the same fixed frame, such as
    the manipulation main-camera frame. The returned values satisfy
    ``p_target = p_current + delta_p`` and
    ``R_target = delta_R @ R_current``.
    """
    current_rotation = rotation_6d_to_matrix_np(current_rot6d)
    target_rotation = rotation_6d_to_matrix_np(target_rot6d)
    relative_rotation = np.matmul(
        target_rotation, np.swapaxes(current_rotation, -1, -2)
    )
    relative_xyz = np.asarray(target_xyz, dtype=np.float64) - np.asarray(
        current_xyz, dtype=np.float64
    )
    return relative_xyz.astype(np.float32), matrix_to_rotation_6d_np(relative_rotation)


def fixed_frame_relative_pose_rot6d_torch(
    current_xyz: torch.Tensor,
    current_rot6d: torch.Tensor,
    target_xyz: torch.Tensor,
    target_rot6d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch equivalent of :func:`fixed_frame_relative_pose_rot6d_np`."""
    current_rotation = rotation_6d_to_matrix_torch(current_rot6d)
    target_rotation = rotation_6d_to_matrix_torch(target_rot6d)
    relative_rotation = target_rotation @ current_rotation.transpose(-1, -2)
    return target_xyz - current_xyz, matrix_to_rotation_6d_torch(relative_rotation)


def apply_relative_pose_rot6d_np(
    anchor_xyz: np.ndarray,
    anchor_rot6d: np.ndarray,
    relative_xyz: np.ndarray,
    relative_rot6d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``T_target = T_anchor @ T_relative`` for deployment reconstruction."""
    anchor_rotation = rotation_6d_to_matrix_np(anchor_rot6d)
    relative_rotation = rotation_6d_to_matrix_np(relative_rot6d)
    target_rotation = np.matmul(anchor_rotation, relative_rotation)
    target_xyz = np.asarray(anchor_xyz, dtype=np.float64) + np.matmul(
        anchor_rotation, np.asarray(relative_xyz, dtype=np.float64)[..., None]
    )[..., 0]
    return target_xyz.astype(np.float32), matrix_to_rotation_6d_np(target_rotation)


def apply_fixed_frame_relative_pose_rot6d_np(
    anchor_xyz: np.ndarray,
    anchor_rot6d: np.ndarray,
    relative_xyz: np.ndarray,
    relative_rot6d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a target from fixed-reference translation and rotation deltas."""
    anchor_rotation = rotation_6d_to_matrix_np(anchor_rot6d)
    relative_rotation = rotation_6d_to_matrix_np(relative_rot6d)
    target_rotation = np.matmul(relative_rotation, anchor_rotation)
    target_xyz = np.asarray(anchor_xyz, dtype=np.float64) + np.asarray(
        relative_xyz, dtype=np.float64
    )
    return target_xyz.astype(np.float32), matrix_to_rotation_6d_np(target_rotation)


def rotation_geodesic_angle_np(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Geodesic angle between matrices or row-major 6D rotations, in radians."""
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    left_matrix = (
        rotation_6d_to_matrix_np(left_array)
        if left_array.shape[-1] == 6
        else left_array
    )
    right_matrix = (
        rotation_6d_to_matrix_np(right_array)
        if right_array.shape[-1] == 6
        else right_array
    )
    # Keep the comparison in float64 and use atan2(sin, cos), not acos(trace),
    # because the latter loses almost all precision for float32 matrices near
    # identity and makes exact 6D round trips look spuriously nonzero.
    relative = np.matmul(
        np.swapaxes(np.asarray(left_matrix, dtype=np.float64), -1, -2),
        np.asarray(right_matrix, dtype=np.float64),
    )
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    sine = 0.5 * np.linalg.norm(
        np.stack(
            (
                relative[..., 2, 1] - relative[..., 1, 2],
                relative[..., 0, 2] - relative[..., 2, 0],
                relative[..., 1, 0] - relative[..., 0, 1],
            ),
            axis=-1,
        ),
        axis=-1,
    )
    return np.arctan2(sine, cosine).astype(np.float32)
