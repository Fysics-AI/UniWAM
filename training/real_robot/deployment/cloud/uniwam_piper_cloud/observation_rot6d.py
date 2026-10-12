"""Online observation construction for the arm-only rot6D/SE(2) prefix policy."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from uniwam.vision import build_matched_fastwam_mosaic
from uniwam_piper_common.rotation6d import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    rpy_xyz_to_matrix,
)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
STATE_DIM = 23


def decode_image_to_tensor(data: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    array = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


def state23_from_rpy(base_command: np.ndarray, eef14_rpy: np.ndarray) -> np.ndarray:
    base = np.asarray(base_command, dtype=np.float32).reshape(3)
    eef = np.asarray(eef14_rpy, dtype=np.float32).reshape(14)
    if not np.all(np.isfinite(np.concatenate((base, eef)))):
        raise ValueError("Observation state contains NaN or Inf.")
    state = np.empty(STATE_DIM, dtype=np.float32)
    state[:3] = base
    state[3:6] = eef[:3]
    state[6:12] = matrix_to_rotation_6d(rpy_xyz_to_matrix(eef[3:6]))
    state[12] = eef[6]
    state[13:16] = eef[7:10]
    state[16:22] = matrix_to_rotation_6d(rpy_xyz_to_matrix(eef[10:13]))
    state[22] = eef[13]
    return state


def validate_state23(value: np.ndarray) -> np.ndarray:
    state = np.asarray(value, dtype=np.float32).reshape(-1)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"Expected 23D rot6D state, got {state.shape}.")
    if not np.all(np.isfinite(state)):
        raise ValueError("Observation state contains NaN or Inf.")
    rotation_6d_to_matrix(state[6:12])
    rotation_6d_to_matrix(state[16:22])
    return state.copy()


class Rot6DObservationBuilder:
    """Build exact paired 384x320 inference inputs from raw ROS camera frames."""

    def __init__(self, *, data_cfg: Any, normalizer: Any) -> None:
        self.data_cfg = data_cfg
        self.normalizer = normalizer
        self.video_size = tuple(int(v) for v in data_cfg.video_size)
        self.context_len = int(data_cfg.context_len)
        self.text_embedding_cache_dir = Path(str(data_cfg.text_embedding_cache_dir))
        self.override_instruction = data_cfg.get("override_instruction", None)
        self._text_context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def canonical_state(state_input: np.ndarray) -> np.ndarray:
        """Accept the new 23D state and legacy RPY forms for offline tooling."""
        state = np.asarray(state_input, dtype=np.float32).reshape(-1)
        if state.shape == (STATE_DIM,):
            return validate_state23(state)
        if state.shape == (17,):
            return state23_from_rpy(state[:3], state[3:])
        if state.shape == (14,):
            return state23_from_rpy(np.zeros(3, dtype=np.float32), state)
        raise ValueError(
            "Expected eef14, [base3,eef14]17, or 23D rot6D state; "
            f"got {state.shape}."
        )

    def make_sample(
        self,
        *,
        images: dict[str, bytes],
        image_encoding: str,
        eef_state: np.ndarray,
        instruction: str | None,
    ) -> dict[str, torch.Tensor]:
        del image_encoding
        required = {"cam_nav", "cam_manip_high", "cam_left_wrist", "cam_right_wrist"}
        missing = required - set(images)
        if missing:
            raise ValueError(f"Missing camera payloads: {sorted(missing)}")

        cam_nav = decode_image_to_tensor(images["cam_nav"])
        cam_manip = decode_image_to_tensor(images["cam_manip_high"])
        cam_left = decode_image_to_tensor(images["cam_left_wrist"])
        cam_right = decode_image_to_tensor(images["cam_right_wrist"])
        nav_image = build_matched_fastwam_mosaic(cam_nav, cam_left, cam_right)
        manip_image = build_matched_fastwam_mosaic(cam_manip, cam_left, cam_right)
        if tuple(nav_image.shape[-2:]) != self.video_size:
            raise ValueError(
                f"Navigation mosaic is {tuple(nav_image.shape[-2:])}, expected {self.video_size}."
            )
        if tuple(manip_image.shape[-2:]) != self.video_size:
            raise ValueError(
                f"Manipulation mosaic is {tuple(manip_image.shape[-2:])}, expected {self.video_size}."
            )

        state_raw = self.canonical_state(eef_state)
        # Navigation targets are local SE(2) paths anchored at this snapshot.
        # Feeding an episode-global base pose would be arbitrary, so the model
        # receives only the two absolute arm poses and gripper apertures.
        state_batch = {"state": {"default": torch.from_numpy(state_raw[3:]).unsqueeze(0)}}
        proprio = self.normalizer.forward(state_batch)["state"]["default"]
        task = self.override_instruction if self.override_instruction is not None else instruction
        if not task:
            raise ValueError("Task instruction is empty.")
        prompt = DEFAULT_PROMPT.format(task=str(task))
        context, context_mask = self._get_cached_text_context(prompt)
        context = context.clone()
        context_mask = context_mask.clone()
        # This checkpoint was trained with upstream Wan/UniWAM semantics:
        # padded embeddings are zero, while all 128 cross-attention keys remain visible.
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        return {
            "nav_image": ((nav_image - 0.5) / 0.5).contiguous(),
            "manip_image": ((manip_image - 0.5) / 0.5).contiguous(),
            "proprio": proprio,
            "state_raw": torch.from_numpy(state_raw),
            "context": context,
            "context_mask": context_mask,
            "prompt": prompt,
        }

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._text_context_cache.get(prompt)
        if cached is not None:
            return cached
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = self.text_embedding_cache_dir / f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt"
        if not path.is_file():
            raise FileNotFoundError(
                "Missing the exact prompt embedding cache: "
                f"{path}. Run scripts/precompute_inference_prompt_async_prefix12_rot6d.sh "
                "with TASK_PROMPT set to this task before starting the cloud server."
            )
        payload = torch.load(path, map_location="cpu")
        context = payload["context"].float().contiguous()
        mask = payload["mask"].bool().contiguous()
        if context.ndim != 2 or context.shape[0] != self.context_len:
            raise ValueError(f"Invalid cached context shape: {tuple(context.shape)}")
        if mask.shape != (self.context_len,):
            raise ValueError(f"Invalid cached context mask shape: {tuple(mask.shape)}")
        self._text_context_cache[prompt] = (context, mask)
        return context, mask
