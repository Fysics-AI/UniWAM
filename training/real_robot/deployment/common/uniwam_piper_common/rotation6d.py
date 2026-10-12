"""Shared SE(3) and GR00T-compatible row-major 6D rotation utilities.

The training project stores rotations as the first two rows of a 3x3 rotation
matrix.  Keeping this small module in the deployment package prevents the
cloud policy and the ROS edge from silently using different conventions.
"""

from __future__ import annotations

import numpy as np


ROT6D_CONVENTION = "first_two_rotation_matrix_rows_row_major"


def _finite_array(name: str, value: np.ndarray, last_dim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if last_dim is not None and array.shape[-1] != last_dim:
        raise ValueError(f"{name} must end in {last_dim} values, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf.")
    return array


def _normalize(name: str, value: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    array = _finite_array(name, value)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norm < eps):
        raise ValueError(f"{name} contains a near-zero vector.")
    return array / norm


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a normalized or unnormalized xyzw quaternion to a matrix."""
    q = _normalize("quaternion_xyzw", _finite_array("quaternion_xyzw", quaternion, 4))
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    return np.stack(
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - zw),
            2.0 * (xz + yw),
            2.0 * (xy + zw),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - xw),
            2.0 * (xz - yw),
            2.0 * (yz + xw),
            1.0 - 2.0 * (xx + yy),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3)).astype(np.float32)


def rpy_xyz_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert XYZ Euler angles to a rotation matrix, matching scipy 'xyz'."""
    angles = _finite_array("rpy", rpy, 3)
    roll, pitch, yaw = np.moveaxis(angles, -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.stack(
        (
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        ),
        axis=-1,
    ).reshape(angles.shape[:-1] + (3, 3)).astype(np.float32)


def matrix_to_rpy_xyz(matrix: np.ndarray) -> np.ndarray:
    """Return the principal XYZ Euler representation of valid rotation matrices."""
    rotation = _finite_array("rotation_matrix", matrix)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotation_matrix must end in [3,3], got {rotation.shape}.")
    pitch = np.arcsin(np.clip(-rotation[..., 2, 0], -1.0, 1.0))
    roll = np.arctan2(rotation[..., 2, 1], rotation[..., 2, 2])
    yaw = np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0])
    return np.stack((roll, pitch, yaw), axis=-1).astype(np.float32)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """Serialize a matrix as its first two rows in row-major order."""
    rotation = _finite_array("rotation_matrix", matrix)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotation_matrix must end in [3,3], got {rotation.shape}.")
    return rotation[..., :2, :].reshape(rotation.shape[:-2] + (6,)).astype(np.float32)


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Decode row-major 6D rotations using the training-time Gram-Schmidt rule."""
    value = _finite_array("rotation_6d", rotation_6d, 6)
    row1 = _normalize("rotation_6d first row", value[..., :3])
    row2_raw = value[..., 3:6]
    row2 = row2_raw - np.sum(row1 * row2_raw, axis=-1, keepdims=True) * row1
    row2 = _normalize("rotation_6d second row", row2)
    row3 = np.cross(row1, row2)
    return np.stack((row1, row2, row3), axis=-2).astype(np.float32)


def relative_pose(
    anchor_xyz: np.ndarray,
    anchor_rot6d: np.ndarray,
    target_xyz: np.ndarray,
    target_rot6d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``inverse(T_anchor) @ T_target`` in the anchor EEF frame."""
    anchor_position = _finite_array("anchor_xyz", anchor_xyz, 3)
    target_position = _finite_array("target_xyz", target_xyz, 3)
    anchor_rotation = rotation_6d_to_matrix(anchor_rot6d)
    target_rotation = rotation_6d_to_matrix(target_rot6d)
    inverse_anchor = np.swapaxes(anchor_rotation, -1, -2)
    relative_xyz = np.matmul(
        inverse_anchor, (target_position - anchor_position)[..., None]
    )[..., 0]
    relative_rotation = np.matmul(inverse_anchor, target_rotation)
    return relative_xyz.astype(np.float32), matrix_to_rotation_6d(relative_rotation)


def absolute_pose(
    anchor_xyz: np.ndarray,
    anchor_rot6d: np.ndarray,
    relative_xyz: np.ndarray,
    relative_rot6d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Undo ``relative_pose`` and return world-frame XYZ and a rotation matrix."""
    anchor_position = _finite_array("anchor_xyz", anchor_xyz, 3)
    local_position = _finite_array("relative_xyz", relative_xyz, 3)
    anchor_rotation = rotation_6d_to_matrix(anchor_rot6d)
    local_rotation = rotation_6d_to_matrix(relative_rot6d)
    absolute_xyz = anchor_position + np.matmul(anchor_rotation, local_position[..., None])[..., 0]
    absolute_rotation = np.matmul(anchor_rotation, local_rotation)
    return absolute_xyz.astype(np.float32), absolute_rotation.astype(np.float32)


def fixed_reference_relative_pose(anchor_xyz, anchor_rot6d, target_xyz, target_rot6d):
    anchor_position = _finite_array("anchor_xyz", anchor_xyz, 3)
    target_position = _finite_array("target_xyz", target_xyz, 3)
    anchor_rotation = rotation_6d_to_matrix(anchor_rot6d)
    target_rotation = rotation_6d_to_matrix(target_rot6d)
    delta_rotation = np.matmul(target_rotation, np.swapaxes(anchor_rotation, -1, -2))
    return (target_position - anchor_position).astype(np.float32), matrix_to_rotation_6d(delta_rotation)


def fixed_reference_absolute_pose(anchor_xyz, anchor_rot6d, delta_xyz, delta_rot6d):
    anchor_position = _finite_array("anchor_xyz", anchor_xyz, 3)
    anchor_rotation = rotation_6d_to_matrix(anchor_rot6d)
    delta_rotation = rotation_6d_to_matrix(delta_rot6d)
    return (
        (anchor_position + _finite_array("delta_xyz", delta_xyz, 3)).astype(np.float32),
        np.matmul(delta_rotation, anchor_rotation).astype(np.float32),
    )


def continuous_rpy_xyz_from_matrix_sequence(
    matrices: np.ndarray,
    initial_rpy: np.ndarray,
) -> np.ndarray:
    """Choose equivalent XYZ Euler triples nearest the preceding target."""
    rotation = _finite_array("rotation_matrices", matrices)
    if rotation.ndim != 3 or rotation.shape[1:] != (3, 3):
        raise ValueError(f"rotation_matrices must be [T,3,3], got {rotation.shape}.")
    reference = _finite_array("initial_rpy", initial_rpy, 3).reshape(3)
    principal = matrix_to_rpy_xyz(rotation).astype(np.float64)
    output = np.empty_like(principal)
    two_pi = 2.0 * np.pi
    for index, row in enumerate(principal):
        candidates = (
            row,
            np.asarray([row[0] + np.pi, np.pi - row[1], row[2] + np.pi]),
        )
        aligned = [candidate + two_pi * np.round((reference - candidate) / two_pi) for candidate in candidates]
        selected = aligned[int(np.argmin([np.linalg.norm(candidate - reference) for candidate in aligned]))]
        output[index] = selected
        reference = selected
    return output.astype(np.float32)
