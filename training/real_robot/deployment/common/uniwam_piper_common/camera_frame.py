"""Rigid transforms between dual-arm base frames and a fixed main camera."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .rotation6d import (
    matrix_to_rotation_6d,
    matrix_to_rpy_xyz,
    rotation_6d_to_matrix,
    rpy_xyz_to_matrix,
)


def validate_se3(name: str, value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size == 16:
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix or 16 row-major values, got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains NaN or Inf.")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-6):
        raise ValueError(f"{name} has an invalid homogeneous last row: {matrix[3].tolist()}.")
    rotation = matrix[:3, :3]
    orthogonal_error = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    if orthogonal_error > 2.0e-4 or abs(determinant - 1.0) > 2.0e-4:
        raise ValueError(
            f"{name} rotation is not SO(3): orthogonal_error={orthogonal_error:.3e}, "
            f"det={determinant:.8f}."
        )
    return matrix.astype(np.float32)


@dataclass(frozen=True)
class DualArmCameraExtrinsics:
    """Transforms from each arm's base frame into one fixed camera frame."""

    profile: str
    camera_from_left_base: np.ndarray
    camera_from_right_base: np.ndarray
    source: str = "profile"

    @classmethod
    def from_values(
        cls,
        *,
        profile: str,
        camera_from_left_base: Any,
        camera_from_right_base: Any,
        source: str = "profile",
    ) -> "DualArmCameraExtrinsics":
        return cls(
            profile=str(profile),
            camera_from_left_base=validate_se3(
                "camera_from_left_base", camera_from_left_base
            ),
            camera_from_right_base=validate_se3(
                "camera_from_right_base", camera_from_right_base
            ),
            source=str(source),
        )


def _transform_pose(
    xyz: np.ndarray,
    rotation: np.ndarray,
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(xyz, dtype=np.float32)
    rotations = np.asarray(rotation, dtype=np.float32)
    matrix = validate_se3("transform", transform)
    transformed_xyz = np.einsum("ij,...j->...i", matrix[:3, :3], points) + matrix[:3, 3]
    transformed_rotation = np.einsum("ij,...jk->...ik", matrix[:3, :3], rotations)
    return transformed_xyz.astype(np.float32), transformed_rotation.astype(np.float32)


def invert_se3(transform: np.ndarray) -> np.ndarray:
    matrix = validate_se3("transform", transform)
    inverse = np.eye(4, dtype=np.float32)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -(matrix[:3, :3].T @ matrix[:3, 3])
    return inverse


def fixed_reference_relative_pose(
    anchor_xyz: np.ndarray,
    anchor_rot6d: np.ndarray,
    target_xyz: np.ndarray,
    target_rot6d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return camera-axis ``dp`` and left-multiplied ``dR`` used in training."""
    anchor_position = np.asarray(anchor_xyz, dtype=np.float32)
    target_position = np.asarray(target_xyz, dtype=np.float32)
    anchor_rotation = rotation_6d_to_matrix(anchor_rot6d)
    target_rotation = rotation_6d_to_matrix(target_rot6d)
    relative_xyz = target_position - anchor_position
    relative_rotation = np.matmul(
        target_rotation, np.swapaxes(anchor_rotation, -1, -2)
    )
    return relative_xyz.astype(np.float32), matrix_to_rotation_6d(relative_rotation)


def fixed_reference_absolute_pose(
    anchor_xyz: np.ndarray,
    anchor_rot6d: np.ndarray,
    relative_xyz: np.ndarray,
    relative_rot6d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Undo ``fixed_reference_relative_pose`` in the fixed camera frame."""
    anchor_position = np.asarray(anchor_xyz, dtype=np.float32)
    anchor_rotation = rotation_6d_to_matrix(anchor_rot6d)
    relative_position = np.asarray(relative_xyz, dtype=np.float32)
    relative_rotation = rotation_6d_to_matrix(relative_rot6d)
    absolute_xyz = anchor_position + relative_position
    absolute_rotation = np.matmul(relative_rotation, anchor_rotation)
    return absolute_xyz.astype(np.float32), absolute_rotation.astype(np.float32)


def state23_base_to_camera(
    state23: np.ndarray,
    extrinsics: DualArmCameraExtrinsics,
) -> np.ndarray:
    state = np.asarray(state23, dtype=np.float32).reshape(23)
    result = state.copy()
    for xyz_slice, rot_slice, transform in (
        (slice(3, 6), slice(6, 12), extrinsics.camera_from_left_base),
        (slice(13, 16), slice(16, 22), extrinsics.camera_from_right_base),
    ):
        xyz, rotation = _transform_pose(
            state[xyz_slice], rotation_6d_to_matrix(state[rot_slice]), transform
        )
        result[xyz_slice] = xyz
        result[rot_slice] = matrix_to_rotation_6d(rotation)
    return result


def eef14_base_to_camera(
    eef14: np.ndarray,
    extrinsics: DualArmCameraExtrinsics,
) -> np.ndarray:
    return _transform_eef14(eef14, extrinsics, inverse=False)


def eef14_camera_to_base(
    eef14: np.ndarray,
    extrinsics: DualArmCameraExtrinsics,
) -> np.ndarray:
    return _transform_eef14(eef14, extrinsics, inverse=True)


def _transform_eef14(
    eef14: np.ndarray,
    extrinsics: DualArmCameraExtrinsics,
    *,
    inverse: bool,
) -> np.ndarray:
    poses = np.asarray(eef14, dtype=np.float32)
    single = poses.ndim == 1
    poses = poses.reshape(-1, 14)
    if not np.all(np.isfinite(poses)):
        raise ValueError("EEF targets contain NaN or Inf.")
    result = poses.copy()
    for start, camera_from_base in (
        (0, extrinsics.camera_from_left_base),
        (7, extrinsics.camera_from_right_base),
    ):
        transform = invert_se3(camera_from_base) if inverse else camera_from_base
        xyz, rotation = _transform_pose(
            poses[:, start : start + 3],
            rpy_xyz_to_matrix(poses[:, start + 3 : start + 6]),
            transform,
        )
        result[:, start : start + 3] = xyz
        result[:, start + 3 : start + 6] = matrix_to_rpy_xyz(rotation)
    return result[0] if single else result


def action17_base_to_camera(
    actions: np.ndarray,
    extrinsics: DualArmCameraExtrinsics,
) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32).reshape(-1, 17).copy()
    values[:, :14] = eef14_base_to_camera(values[:, :14], extrinsics)
    return values


def action17_camera_to_base(
    actions: np.ndarray,
    extrinsics: DualArmCameraExtrinsics,
) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32).reshape(-1, 17).copy()
    values[:, :14] = eef14_camera_to_base(values[:, :14], extrinsics)
    return values
