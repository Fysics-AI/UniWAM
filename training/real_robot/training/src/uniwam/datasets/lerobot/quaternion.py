from __future__ import annotations

import numpy as np
import torch


def canonicalize_quaternion_np(quaternion: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize xyzw quaternions and choose one deterministic hemisphere."""
    q = np.asarray(quaternion)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < eps):
        raise ValueError("Cannot normalize a zero-length quaternion.")
    q = q / norm

    # Prefer w > 0. At exactly w == 0, use the first non-zero xyz component.
    sign = np.ones(q.shape[:-1], dtype=q.dtype)
    unresolved = np.abs(q[..., 3]) <= eps
    sign[q[..., 3] < -eps] = -1
    for component in (2, 1, 0):
        negative = unresolved & (q[..., component] < -eps)
        positive = unresolved & (q[..., component] > eps)
        sign[negative] = -1
        unresolved &= ~(negative | positive)
    return q * sign[..., None]


def canonicalize_quaternion_torch(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Torch equivalent of :func:`canonicalize_quaternion_np` for xyzw quaternions."""
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if bool(torch.any(norm < eps)):
        raise ValueError("Cannot normalize a zero-length quaternion.")
    q = quaternion / norm

    sign = torch.ones_like(q[..., 3])
    unresolved = q[..., 3].abs() <= eps
    sign = torch.where(q[..., 3] < -eps, -sign, sign)
    for component in (2, 1, 0):
        negative = unresolved & (q[..., component] < -eps)
        positive = unresolved & (q[..., component] > eps)
        sign = torch.where(negative, -torch.ones_like(sign), sign)
        unresolved = unresolved & ~(negative | positive)
    return q * sign.unsqueeze(-1)


def quaternion_multiply_np(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for xyzw quaternions: result = left tensor-product right."""
    lx, ly, lz, lw = np.moveaxis(np.asarray(left), -1, 0)
    rx, ry, rz, rw = np.moveaxis(np.asarray(right), -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def quaternion_multiply_torch(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Torch Hamilton product for xyzw quaternions."""
    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def quaternion_inverse_np(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion)
    norm_sq = np.sum(np.square(q), axis=-1, keepdims=True)
    if np.any(norm_sq < 1e-24):
        raise ValueError("Cannot invert a zero-length quaternion.")
    conjugate = np.concatenate((-q[..., :3], q[..., 3:4]), axis=-1)
    return conjugate / norm_sq


def quaternion_inverse_torch(quaternion: torch.Tensor) -> torch.Tensor:
    norm_sq = torch.sum(quaternion.square(), dim=-1, keepdim=True)
    if bool(torch.any(norm_sq < 1e-24)):
        raise ValueError("Cannot invert a zero-length quaternion.")
    conjugate = torch.cat((-quaternion[..., :3], quaternion[..., 3:4]), dim=-1)
    return conjugate / norm_sq


def relative_quaternion_np(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return canonical q_rel satisfying target = current tensor-product q_rel."""
    return canonicalize_quaternion_np(
        quaternion_multiply_np(quaternion_inverse_np(current), target)
    )


def relative_quaternion_torch(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Torch q_rel satisfying target = current tensor-product q_rel."""
    return canonicalize_quaternion_torch(
        quaternion_multiply_torch(quaternion_inverse_torch(current), target)
    )


def quaternion_angular_error_np(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Sign-invariant angular distance in radians between xyzw quaternions."""
    left = canonicalize_quaternion_np(np.asarray(left, dtype=np.float64))
    right = canonicalize_quaternion_np(np.asarray(right, dtype=np.float64))
    error = quaternion_multiply_np(quaternion_inverse_np(left), right)
    return 2.0 * np.arctan2(
        np.linalg.norm(error[..., :3], axis=-1),
        np.abs(error[..., 3]),
    )


def repair_isolated_quaternion_spikes_np(
    quaternion: np.ndarray,
    *,
    jump_threshold_rad: float = 0.5,
    bridge_threshold_rad: float = 0.3,
    max_passes: int = 4,
) -> tuple[np.ndarray, dict[str, object]]:
    """Replace isolated impossible orientation spikes with neighboring midpoint poses."""
    repaired = canonicalize_quaternion_np(np.asarray(quaternion, dtype=np.float64)).astype(np.float32)
    repaired_indices: set[int] = set()
    initial_jump = (
        quaternion_angular_error_np(repaired[:-1], repaired[1:])
        if repaired.shape[0] > 1
        else np.zeros(0, dtype=np.float32)
    )
    for _pass in range(max_passes):
        if repaired.shape[0] < 3:
            break
        previous_jump = quaternion_angular_error_np(repaired[1:-1], repaired[:-2])
        next_jump = quaternion_angular_error_np(repaired[1:-1], repaired[2:])
        bridge_jump = quaternion_angular_error_np(repaired[:-2], repaired[2:])
        candidates = np.flatnonzero(
            (previous_jump > jump_threshold_rad)
            & (next_jump > jump_threshold_rad)
            & (bridge_jump < bridge_threshold_rad)
        ) + 1
        if candidates.size == 0:
            break
        for index in candidates.tolist():
            left = repaired[index - 1].astype(np.float64)
            right = repaired[index + 1].astype(np.float64)
            if float(np.dot(left, right)) < 0.0:
                right = -right
            repaired[index] = canonicalize_quaternion_np(left + right).astype(np.float32)
            repaired_indices.add(int(index))
    final_jump = (
        quaternion_angular_error_np(repaired[:-1], repaired[1:])
        if repaired.shape[0] > 1
        else np.zeros(0, dtype=np.float32)
    )
    return repaired, {
        "repaired_count": len(repaired_indices),
        "repaired_indices": sorted(repaired_indices),
        "max_jump_before_rad": float(initial_jump.max(initial=0.0)),
        "max_jump_after_rad": float(final_jump.max(initial=0.0)),
    }
