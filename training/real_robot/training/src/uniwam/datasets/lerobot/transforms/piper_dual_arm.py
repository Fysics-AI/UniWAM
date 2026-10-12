from __future__ import annotations

from typing import Iterable

import torch


class PiperDualArmEefTransform:
    """Map 14D dual-arm EEF vectors to FastWAM's compact training space.

    Raw layout:
      0..6: left EEF xyz(m), rpy(rad), gripper
      7..13: right EEF xyz(m), rpy(rad), gripper

    The LeRobot action is an absolute future EEF target. For training we use
    pose deltas against the current chunk state and keep grippers absolute.
    """

    def __init__(
        self,
        keys: Iterable[str] = ("default",),
        angle_unit: str = "rad",
    ):
        self.keys = list(keys)
        self.angle_unit = str(angle_unit).lower()
        if self.angle_unit not in {"rad", "deg"}:
            raise ValueError(f"Unsupported angle_unit={angle_unit!r}; expected 'rad' or 'deg'.")

    def _wrap_angle(self, x: torch.Tensor) -> torch.Tensor:
        if self.angle_unit == "deg":
            return torch.remainder(x + 180.0, 360.0) - 180.0
        return torch.remainder(x + torch.pi, 2.0 * torch.pi) - torch.pi

    def _aligned_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.ndim == action.ndim and state.shape[-2] == action.shape[-2]:
            return state
        if state.ndim == action.ndim and state.shape[-2] == 1:
            return state.expand(*action.shape[:-1], state.shape[-1])
        return state[..., :1, :].expand(*action.shape[:-1], state.shape[-1])

    def forward(self, batch: dict) -> dict:
        for key in self.keys:
            state = batch["state"][key]
            if state.shape[-1] != 14:
                raise ValueError(f"Expected raw state dim 14 for `{key}`, got {state.shape[-1]}.")
            batch["state"][key] = state

            if "action" not in batch:
                continue

            action = batch["action"][key]
            if action.shape[-1] != 14:
                raise ValueError(f"Expected raw action dim 14 for `{key}`, got {action.shape[-1]}.")

            aligned_state = self._aligned_state(state, action)
            action_out = action.clone()
            for start in (0, 7):
                delta = action[..., start : start + 6] - aligned_state[..., start : start + 6]
                delta[..., 3:6] = self._wrap_angle(delta[..., 3:6])
                action_out[..., start : start + 6] = delta
                action_out[..., start + 6 : start + 7] = action[..., start + 6 : start + 7]
            batch["action"][key] = action_out

        return batch

    def backward(self, batch: dict) -> dict:
        for key in self.keys:
            state = batch["state"][key]
            action = batch["action"][key]
            if state.shape[-1] != 14:
                raise ValueError(f"Expected compact state dim 14 for `{key}`, got {state.shape[-1]}.")
            if action.shape[-1] != 14:
                raise ValueError(f"Expected compact action dim 14 for `{key}`, got {action.shape[-1]}.")

            raw_action = action.clone()
            for start in (0, 7):
                abs_pose = action[..., start : start + 6] + state[..., :1, start : start + 6]
                abs_pose[..., 3:6] = self._wrap_angle(abs_pose[..., 3:6])
                raw_action[..., start : start + 6] = abs_pose
                raw_action[..., start + 6 : start + 7] = action[..., start + 6 : start + 7]
            batch["action"][key] = raw_action
            batch["state"][key] = state

        return batch
