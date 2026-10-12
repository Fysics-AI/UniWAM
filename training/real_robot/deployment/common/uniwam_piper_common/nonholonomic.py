"""Plan smooth nonholonomic commands from snapshot-relative SE(2) paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .future_state import _compose_se2, _inverse_se2, _se2_exp, nav_target_path_to_body_velocity


def _wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class NonholonomicPlannerConfig:
    longitudinal_gain: float = 1.5
    heading_gain: float = 2.0
    cross_track_gain: float = 1.5
    cross_track_lookahead_m: float = 0.08
    smoothing_alpha: float = 0.45
    forward_only: bool = True
    max_forward_speed_mps: float = 0.0
    max_yaw_rate_radps: float = 0.0

    @classmethod
    def from_mapping(cls, value: Any) -> "NonholonomicPlannerConfig":
        if value is None:
            return cls()
        return cls(**{name: value.get(name, field.default) for name, field in cls.__dataclass_fields__.items()})

    def validate(self) -> None:
        if min(self.longitudinal_gain, self.heading_gain, self.cross_track_gain) < 0.0:
            raise ValueError("Planner gains must be non-negative.")
        if self.cross_track_lookahead_m <= 0.0:
            raise ValueError("cross_track_lookahead_m must be positive.")
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0,1].")
        if self.max_forward_speed_mps < 0.0 or self.max_yaw_rate_radps < 0.0:
            raise ValueError("Optional limits must be non-negative; zero disables them.")


@dataclass(frozen=True)
class NonholonomicPlan:
    commands: np.ndarray
    raw_body_twist: np.ndarray
    planned_path: np.ndarray
    pose_error: np.ndarray


def plan_nonholonomic_commands(
    future_pose: np.ndarray,
    target_path: np.ndarray,
    *,
    control_hz: float = 30.0,
    initial_command: np.ndarray | None = None,
    config: NonholonomicPlannerConfig | None = None,
) -> NonholonomicPlan:
    """Track an SE(2) path using forward velocity and yaw rate only."""
    target = np.asarray(target_path, dtype=np.float32)
    endpoint = np.asarray(future_pose, dtype=np.float32).reshape(3)
    if target.ndim != 2 or target.shape[1] != 3 or not target.shape[0]:
        raise ValueError(f"target_path must be non-empty [H,3], got {target.shape}.")
    if control_hz <= 0.0 or not np.all(np.isfinite(target)) or not np.all(np.isfinite(endpoint)):
        raise ValueError("Invalid SE(2) planner input.")
    cfg = config or NonholonomicPlannerConfig()
    cfg.validate()
    raw = nav_target_path_to_body_velocity(endpoint, target, control_hz)
    seed = np.zeros(3, dtype=np.float32) if initial_command is None else np.asarray(initial_command, dtype=np.float32).reshape(3)
    if not np.all(np.isfinite(seed)):
        raise ValueError("initial_command contains NaN or Inf.")

    dt = 1.0 / float(control_hz)
    pose = endpoint.astype(np.float64)
    previous_v, previous_w = float(seed[0]), float(seed[2])
    commands = np.zeros((target.shape[0], 3), dtype=np.float64)
    planned = np.zeros_like(target, dtype=np.float64)
    errors = np.zeros_like(target, dtype=np.float64)
    for index, desired in enumerate(target.astype(np.float64)):
        feedforward_v = max(0.0, float(raw[index, 0])) if cfg.forward_only else float(raw[index, 0])
        feedforward = np.asarray([feedforward_v, 0.0, float(raw[index, 2])], dtype=np.float64)
        nominal_pose = np.asarray(
            _compose_se2(pose, _se2_exp(feedforward * dt)), dtype=np.float64
        )
        error = np.asarray(_compose_se2(_inverse_se2(nominal_pose), desired), dtype=np.float64)
        error[2] = _wrap_angle(float(error[2]))
        errors[index] = error
        v_target = feedforward_v + cfg.longitudinal_gain * float(error[0])
        if cfg.forward_only:
            v_target = max(0.0, v_target)
        cross_heading = np.arctan2(float(error[1]), cfg.cross_track_lookahead_m + max(0.0, float(error[0])))
        w_target = float(raw[index, 2]) + cfg.heading_gain * float(error[2]) + cfg.cross_track_gain * float(cross_heading)
        alpha = cfg.smoothing_alpha
        v = alpha * v_target + (1.0 - alpha) * previous_v
        w = alpha * w_target + (1.0 - alpha) * previous_w
        if cfg.forward_only:
            v = max(0.0, v)
        if cfg.max_forward_speed_mps > 0.0:
            v = min(v, cfg.max_forward_speed_mps) if cfg.forward_only else float(np.clip(v, -cfg.max_forward_speed_mps, cfg.max_forward_speed_mps))
        if cfg.max_yaw_rate_radps > 0.0:
            w = float(np.clip(w, -cfg.max_yaw_rate_radps, cfg.max_yaw_rate_radps))
        command = np.asarray([v, 0.0, w], dtype=np.float64)
        commands[index] = command
        pose = np.asarray(_compose_se2(pose, _se2_exp(command * dt)), dtype=np.float64)
        pose[2] = _wrap_angle(float(pose[2]))
        planned[index] = pose
        previous_v, previous_w = v, w
    if not all(np.all(np.isfinite(value)) for value in (commands, raw, planned, errors)):
        raise ValueError("Nonholonomic planner produced NaN or Inf.")
    return NonholonomicPlan(commands.astype(np.float32), raw.astype(np.float32), planned.astype(np.float32), errors.astype(np.float32))
