"""Public HM3D navigation dataset wrapper around the Joint8 index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .hm3d_joint8_delivery_dataset import (
    HM3DJoint8DeliveryDataset,
    IMAGE_PREPROCESS_PROFILES,
    _mix_uint64,
)


BASE_IMAGE_PROFILE = "resize_480x256_base"
IMAGE_PREPROCESS_PROFILES.setdefault(
    BASE_IMAGE_PROFILE,
    {
        "resize_wh": (453, 256),
        "canvas_wh": (480, 256),
        "pad_ltrb": (5, 0, 22, 0),
    },
)


class HM3DNavigationDataset(HM3DJoint8DeliveryDataset):
    """Sample a training episode and a valid H32 start."""

    def __init__(
        self,
        index_dir: str,
        episode_index_dir: str,
        normalization_path: str,
        text_embedding_cache_dirs,
        schedule_windows: int = 64_000_000,
        sampling_seed: int = 20260820,
        exclude_all_zero_starts: bool = True,
        context_len: int = 128,
        video_backend: str = "torchcodec",
        decode_retries: int = 8,
        episode_cache_size: int = 16,
        simple_text_only: bool = False,
        instructions_only: bool = False,
        image_preprocess: str = BASE_IMAGE_PROFILE,
        allow_norm_stats_override: bool = False,
        validate_all_text_caches: bool = True,
    ) -> None:
        self.episode_index_dir = Path(episode_index_dir)
        self.simple_text_only = bool(simple_text_only)
        self.instructions_only = bool(instructions_only)
        self.allow_norm_stats_override = bool(allow_norm_stats_override)
        if self.simple_text_only and self.instructions_only:
            raise ValueError("simple_text_only and instructions_only are mutually exclusive")
        index_manifest = json.loads(
            (self.episode_index_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if index_manifest.get("schema_version") not in {
            "hm3d_navigation_index_v1",
        }:
            raise ValueError(f"Unexpected navigation index: {self.episode_index_dir}")
        def manifest_path(value: str) -> Path:
            path = Path(value)
            return (path if path.is_absolute() else self.episode_index_dir / path).resolve()

        if manifest_path(index_manifest["source_index"]) != Path(index_dir).resolve():
            raise ValueError("Navigation/source index mismatch")
        if (
            manifest_path(index_manifest["normalization_path"])
            != Path(normalization_path).resolve()
            and not self.allow_norm_stats_override
        ):
            raise ValueError("Navigation/stage-1 normalization mismatch")

        super().__init__(
            index_dir=index_dir,
            norm_stats=None,
            text_embedding_cache_dirs=text_embedding_cache_dirs,
            schedule_windows=schedule_windows,
            sampling_seed=sampling_seed,
            sampling_mode="natural_episode",
            image_preprocess=image_preprocess,
            exclude_all_zero_starts=exclude_all_zero_starts,
            context_len=context_len,
            video_backend=video_backend,
            decode_retries=decode_retries,
            episode_cache_size=episode_cache_size,
            validate_all_text_caches=False,
        )

        stats = json.loads(Path(normalization_path).read_text(encoding="utf-8"))
        expected_order = self.index_manifest["action_contract"]["order"]
        if stats.get("action_order") != expected_order or stats.get("action_dim") != 9:
            raise ValueError(f"Unexpected stage-1 action stats: {normalization_path}")
        if stats.get("sampling_mode", "natural_episode") != "natural_episode":
            raise ValueError("Public navigation training requires natural-episode action stats")
        if stats.get("image_preprocess", self.image_preprocess) != self.image_preprocess:
            raise ValueError("Public navigation training image preprocessing/stats mismatch")
        if float(stats.get("missing_coord")) != self.MISSING_COORD:
            raise ValueError("Stage-1 missing-coordinate sentinel mismatch")
        q01 = torch.tensor(stats["q01"], dtype=torch.float32)
        q99 = torch.tensor(stats["q99"], dtype=torch.float32)
        if q01.shape != (9,) or q99.shape != (9,) or not torch.all(q99 > q01):
            raise ValueError(f"Invalid stage-1 q01/q99: {normalization_path}")
        self.action_scale = 2.0 / (q99 - q01)
        self.action_offset = -1.0 - self.action_scale * q01

        self.episode_ids = np.load(
            self.episode_index_dir / "episode_ids.npy", mmap_mode="r"
        )
        if len(self.episode_ids) != int(index_manifest["episode_count"]):
            raise ValueError("Navigation episode count mismatch")
        self.target_ids = {
            int(row["target_id"])
            for row in map(
                json.loads,
                (self.episode_index_dir / "targets.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines(),
            )
        }
        if validate_all_text_caches:
            for target_id in self.target_ids:
                row = self.prompts[target_id]
                if self.instructions_only:
                    tasks = tuple(row["instructions"])
                else:
                    tasks = (row["point_description"],)
                    if not self.simple_text_only:
                        tasks = (*tasks, *row["instructions"])
                for task in tasks:
                    self._cache_path(str(task))

    def _instruction(self, target_id: int, index: int) -> tuple[str, int]:
        if self.simple_text_only:
            return str(self.prompts[target_id]["point_description"]), -1
        if self.instructions_only:
            instructions = self.prompts[target_id]["instructions"]
            if len(instructions) != 5:
                raise ValueError(
                    f"Expected five instructions for target {target_id}, got {len(instructions)}"
                )
            choice = int(_mix_uint64(int(index) ^ self.sampling_seed) % len(instructions))
            return str(instructions[choice]), choice
        return super()._instruction(target_id, index)

    def _sample_metadata(self, index: int) -> tuple[int, int, int, dict[str, Any]]:
        value = _mix_uint64(int(index) ^ self.sampling_seed)
        for attempt in range(64):
            candidate = _mix_uint64(value + attempt * 0x9E3779B97F4A7C15)
            episode_id = int(self.episode_ids[candidate % len(self.episode_ids)])
            payload = self._load_episode(episode_id)
            frame_count = self._natural_frame_count(
                payload["commands"],
                payload["terminal_start"],
                self.exclude_all_zero_starts,
            )
            if frame_count <= 0:
                continue
            frame_value = _mix_uint64(candidate ^ 0xA0761D6478BD642F)
            return episode_id, int(frame_value % frame_count), -1, payload
        raise RuntimeError("Unable to sample a valid public navigation training trajectory")


class HM3DNavigationVelocityDataset(HM3DNavigationDataset):
    """Public navigation input pipeline with native FastWAM ``[vx, vyaw]`` actions."""

    def __init__(self, velocity_norm_stats: str, **kwargs) -> None:
        super().__init__(**kwargs)
        stats = json.loads(Path(velocity_norm_stats).read_text(encoding="utf-8"))
        action_stats = stats.get("action", {}).get("default", {})
        action_min = torch.tensor(action_stats.get("global_min", []), dtype=torch.float32)
        action_max = torch.tensor(action_stats.get("global_max", []), dtype=torch.float32)
        if action_min.shape != (2,) or action_max.shape != (2,) or not torch.all(action_max > action_min):
            raise ValueError(f"Invalid [vx, vyaw] min/max stats: {velocity_norm_stats}")
        self.action_scale = 2.0 / (action_max - action_min)
        self.action_offset = -1.0 - self.action_scale * action_min
        self.velocity_norm_stats = str(velocity_norm_stats)

    def _raw_action(self, payload: dict[str, Any], frame_index: int) -> np.ndarray:
        count = len(payload["commands"])
        command_indices = frame_index + np.arange(self.HORIZON, dtype=np.int64)
        command_real = command_indices < count
        commands = np.zeros((self.HORIZON, 2), dtype=np.float32)
        commands[command_real] = payload["commands"][command_indices[command_real]]
        return commands
