"""Manifest-native HM3D Joint8 H32 navigation dataset."""

from __future__ import annotations

import hashlib
import json
import mmap
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from fastwam.datasets.hm3d_nav15hz_bbox_se2_video_dataset import (
    PROMPT_TEMPLATE,
    integrate_relative_se2,
)
from fastwam.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames


STAGE_NAMES = (
    "joint_initial_yaw",
    "joint_curve",
    "joint_straight",
    "canonical_active",
    "terminal_partial_stop",
    "terminal_all_zero",
)
STAGE_WEIGHTS = (15, 30, 25, 20, 7, 3)
STAGE_CUMULATIVE = np.cumsum(STAGE_WEIGHTS)
SAMPLING_MODES = ("stage_balanced", "natural_episode")
IMAGE_PREPROCESS_PROFILES = {
    "resize_480x256_centered": {
        "resize_wh": (453, 256),
        "canvas_wh": (480, 256),
        "pad_ltrb": (13, 0, 14, 0),
    },
    "native_576x320_centered": {
        "resize_wh": (566, 320),
        "canvas_wh": (576, 320),
        "pad_ltrb": (5, 0, 5, 0),
    },
}


def _mix_uint64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (int(value) + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


class HM3DJoint8DeliveryDataset(Dataset):
    """Sample balanced H32 windows directly from the Joint8 delivery."""

    MICROBLOCK = 8
    HORIZON = 32
    SOURCE_HZ = 15.0
    SOURCE_WH = (566, 320)
    VIDEO_OFFSETS = (0, 8, 16, 24, 32)
    MISSING_COORD = -0.1

    def __init__(
        self,
        index_dir: str,
        norm_stats: str | None,
        text_embedding_cache_dirs: list[str] | tuple[str, ...],
        schedule_windows: int = 64_000_000,
        sampling_seed: int = 20260820,
        sampling_mode: str | None = None,
        image_preprocess: str | None = None,
        exclude_all_zero_starts: bool = True,
        context_len: int = 128,
        video_backend: str = "torchcodec",
        decode_retries: int = 8,
        episode_cache_size: int = 16,
        validate_all_text_caches: bool = True,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_manifest = json.loads(
            (self.index_dir / "index_manifest.json").read_text(encoding="utf-8")
        )
        if self.index_manifest.get("schema_version") not in {
            "hm3d_joint8_h32_aux9_compact_index_v1", "map_bench_navigation_index_v1"
        }:
            raise ValueError(f"Unexpected index contract: {self.index_dir}")
        if tuple(self.index_manifest["image_contract"]["source_wh"]) != self.SOURCE_WH:
            raise ValueError("Source image contract mismatch")
        # Older compact indices carry the superseded 480x256 preprocessing
        # metadata. Their episode/path CSR arrays are still valid; the runtime
        # image contract below is the authoritative one for this delivery.
        if sampling_mode != "natural_episode" and self.index_manifest["sampling_weights"] != dict(
            zip(STAGE_NAMES, STAGE_WEIGHTS)
        ):
            raise ValueError("Sampling weights do not match the approved contract")

        self.schedule_windows = int(schedule_windows)
        self.sampling_seed = int(sampling_seed)
        if sampling_mode is None:
            raise ValueError("sampling_mode must be explicit for Joint8 training")
        self.sampling_mode = str(sampling_mode).strip().lower()
        self.exclude_all_zero_starts = bool(exclude_all_zero_starts)
        if self.sampling_mode not in SAMPLING_MODES:
            raise ValueError(
                f"Unsupported sampling_mode={sampling_mode!r}; expected one of {SAMPLING_MODES}."
            )
        if image_preprocess is None:
            raise ValueError("image_preprocess must be explicit for Joint8 training")
        self.image_preprocess = str(image_preprocess).strip().lower()
        profile = IMAGE_PREPROCESS_PROFILES.get(self.image_preprocess)
        if profile is None:
            raise ValueError(
                f"Unsupported image_preprocess={image_preprocess!r}; "
                f"expected one of {tuple(IMAGE_PREPROCESS_PROFILES)}"
            )
        self.RESIZE_WH = tuple(profile["resize_wh"])
        self.CANVAS_WH = tuple(profile["canvas_wh"])
        self.PAD_LTRB = tuple(profile["pad_ltrb"])
        if self.CANVAS_WH[0] % 32 or self.CANVAS_WH[1] % 32:
            raise ValueError(f"Wan canvas must be divisible by 32, got {self.CANVAS_WH}")
        if (
            self.RESIZE_WH[0] + self.PAD_LTRB[0] + self.PAD_LTRB[2]
            != self.CANVAS_WH[0]
            or self.RESIZE_WH[1] + self.PAD_LTRB[1] + self.PAD_LTRB[3]
            != self.CANVAS_WH[1]
        ):
            raise ValueError("Resize and padding do not exactly fill the configured canvas")
        self.context_len = int(context_len)
        self.video_backend = str(video_backend)
        self.decode_retries = int(decode_retries)
        self.episode_cache_size = int(episode_cache_size)
        if self.schedule_windows <= 0 or self.decode_retries <= 0 or self.episode_cache_size <= 0:
            raise ValueError("Schedule, retry, and cache sizes must be positive")

        if norm_stats is None:
            self.action_scale = torch.ones(9, dtype=torch.float32)
            self.action_offset = torch.zeros(9, dtype=torch.float32)
        else:
            stats = json.loads(Path(norm_stats).read_text(encoding="utf-8"))
            expected_order = self.index_manifest["action_contract"]["order"]
            if stats.get("action_order") != expected_order or stats.get("action_dim") != 9:
                raise ValueError(f"Unexpected action stats: {norm_stats}")
            if stats.get("sampling_mode") != self.sampling_mode:
                raise ValueError(
                    "Action stats sampling mode mismatch: "
                    f"stats={stats.get('sampling_mode')!r} dataset={self.sampling_mode!r}"
                )
            if stats.get("image_preprocess") != self.image_preprocess:
                raise ValueError(
                    "Action stats image preprocessing mismatch: "
                    f"stats={stats.get('image_preprocess')!r} dataset={self.image_preprocess!r}"
                )
            if float(stats.get("missing_coord")) != self.MISSING_COORD:
                raise ValueError("Action stats use a different missing-coordinate sentinel")
            q01 = torch.tensor(stats["q01"], dtype=torch.float32)
            q99 = torch.tensor(stats["q99"], dtype=torch.float32)
            if q01.shape != (9,) or q99.shape != (9,) or not torch.all(q99 > q01):
                raise ValueError(f"Invalid q01/q99 action stats: {norm_stats}")
            self.action_scale = 2.0 / (q99 - q01)
            self.action_offset = -1.0 - self.action_scale * q01

        self.prompts = json.loads((self.index_dir / "target_prompts.json").read_text(encoding="utf-8"))
        self.cache_dirs = tuple(Path(value) for value in text_embedding_cache_dirs)
        if not self.cache_dirs:
            raise ValueError("At least one T5 cache directory is required")
        self._arrays: dict[str, np.ndarray] | None = None
        self._path_file = None
        self._path_mmap = None
        self._episode_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._text_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._open_index()
        if validate_all_text_caches:
            self._validate_all_text_caches()

    def _open_index(self) -> None:
        names = (
            "episode_source", "episode_frame_counts", "episode_target_ids", "episode_path_offsets",
            "joint_target_ids", "joint_target_obs_offsets", "joint_obs_episode_offsets",
            "joint_episode_ids", "canonical_target_ids", "canonical_target_obs_offsets",
            "canonical_obs_episode_offsets", "canonical_episode_ids",
        )
        if self.sampling_mode == "natural_episode":
            names = names[:4]
        self._arrays = {
            name: np.load(self.index_dir / f"{name}.npy", mmap_mode="r") for name in names
        }
        self._path_file = (self.index_dir / "episode_paths.bin").open("rb")
        self._path_mmap = mmap.mmap(self._path_file.fileno(), 0, access=mmap.ACCESS_READ)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = None
        state["_path_file"] = None
        state["_path_mmap"] = None
        state["_episode_cache"] = OrderedDict()
        state["_text_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open_index()

    def __len__(self) -> int:
        return self.schedule_windows

    def _episode_parent(self, episode_id: int) -> Path:
        assert self._arrays is not None and self._path_mmap is not None
        offsets = self._arrays["episode_path_offsets"]
        begin, end = int(offsets[episode_id]), int(offsets[episode_id + 1])
        path = Path(self._path_mmap[begin:end].decode("utf-8"))
        return path if path.is_absolute() else self.index_dir / path

    def _pick_episode(self, source: str, value: int) -> int:
        assert self._arrays is not None
        target_ids = self._arrays[f"{source}_target_ids"]
        target_obs = self._arrays[f"{source}_target_obs_offsets"]
        obs_episode = self._arrays[f"{source}_obs_episode_offsets"]
        episode_ids = self._arrays[f"{source}_episode_ids"]
        target_position = int(value % len(target_ids))
        obs_begin = int(target_obs[target_position])
        obs_end = int(target_obs[target_position + 1])
        value = _mix_uint64(value ^ 0xD1B54A32D192ED03)
        observation = obs_begin + int(value % (obs_end - obs_begin))
        episode_begin = int(obs_episode[observation])
        episode_end = int(obs_episode[observation + 1])
        value = _mix_uint64(value ^ 0x94D049BB133111EB)
        return int(episode_ids[episode_begin + int(value % (episode_end - episode_begin))])

    @staticmethod
    def _terminal_start(commands: np.ndarray) -> int:
        active = np.any(np.abs(commands) > 1.0e-7, axis=1)
        active_indices = np.flatnonzero(active)
        return 0 if not len(active_indices) else int(active_indices[-1] + 1)

    @staticmethod
    def _initial_yaw_end(commands: np.ndarray, terminal_start: int) -> int:
        end = 0
        for vx, vyaw in commands[:terminal_start]:
            if abs(float(vx)) <= 1.0e-7 and abs(float(vyaw)) > 1.0e-7:
                end += 1
            else:
                break
        return end

    def _load_episode(self, episode_id: int) -> dict[str, Any]:
        cached = self._episode_cache.get(episode_id)
        if cached is not None:
            self._episode_cache.move_to_end(episode_id)
            return cached
        assert self._arrays is not None
        parent = self._episode_parent(episode_id)
        expected_count = int(self._arrays["episode_frame_counts"][episode_id])
        source_id = int(self._arrays["episode_source"][episode_id])
        annotations = []
        with (parent / "frame_annotations.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                annotations.append(json.loads(line))
        if len(annotations) != expected_count:
            raise ValueError(f"Annotation count mismatch: {parent}")
        if source_id == 0:
            commands = np.asarray(np.load(parent / "actions_vx_vyaw.npy"), dtype=np.float32)
            if commands.shape != (expected_count, 2):
                raise ValueError(f"Joint action shape mismatch: {parent}")
        else:
            stored = np.asarray([[row["vx"], row["vyaw"]] for row in annotations], dtype=np.float32)
            commands = np.zeros_like(stored)
            commands[:-1] = stored[1:]
        bbox = np.asarray([row["target_bbox_xyxy"] for row in annotations], dtype=np.float32)
        bbox_visible = np.asarray([row["target_visible"] for row in annotations], dtype=np.bool_)
        navigable = np.asarray([row["op_pixel_xy"] for row in annotations], dtype=np.float32)
        navigable_visible = np.asarray([row["op_visible"] for row in annotations], dtype=np.bool_)
        terminal_start = self._terminal_start(commands)
        value = {
            "parent": parent,
            "video": parent / self.index_manifest.get("video_filename", "ranger_fp_h122_pitch22p5_566x320.mp4"),
            "commands": commands,
            "bbox": bbox,
            "bbox_visible": bbox_visible,
            "navigable": navigable,
            "navigable_visible": navigable_visible,
            "terminal_start": terminal_start,
            "initial_yaw_end": self._initial_yaw_end(commands, terminal_start),
            "target_id": int(self._arrays["episode_target_ids"][episode_id]),
            "source_id": source_id,
        }
        self._episode_cache[episode_id] = value
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return value

    def _stage(self, block_hash: int) -> int:
        return int(np.searchsorted(STAGE_CUMULATIVE, block_hash % 100, side="right"))

    def _source_for_stage(self, stage: int, value: int) -> str:
        if stage <= 2:
            return "joint"
        if stage == 3:
            return "canonical"
        return "joint" if value % 10 < 8 else "canonical"

    @staticmethod
    def _natural_frame_count(
        commands: np.ndarray,
        terminal_start: int,
        exclude_all_zero_starts: bool,
    ) -> int:
        if not exclude_all_zero_starts:
            return len(commands)
        # A start before terminal_start can still supervise the transition into
        # the stop tail. A start at terminal_start is already all-zero.
        return int(terminal_start) if int(terminal_start) > 0 else 0

    def _eligible_frames(self, payload: dict[str, Any], stage: int) -> np.ndarray:
        commands = payload["commands"]
        terminal = int(payload["terminal_start"])
        active_last_start = terminal - self.HORIZON
        initial_end = int(payload["initial_yaw_end"])
        if stage == 0:
            end = min(initial_end - 1, active_last_start)
            return np.arange(0, end + 1, dtype=np.int64) if end >= 0 else np.empty(0, np.int64)
        if stage in (1, 2, 3):
            begin = initial_end if stage in (1, 2) else 0
            if active_last_start < begin:
                return np.empty(0, np.int64)
            candidates = np.arange(begin, active_last_start + 1, dtype=np.int64)
            if stage == 3:
                return candidates
            angular_prefix = np.concatenate(
                (np.zeros(1, dtype=np.float64), np.cumsum(commands[:, 1], dtype=np.float64))
            )
            heading = np.abs(
                angular_prefix[candidates + self.HORIZON] - angular_prefix[candidates]
            ) / self.SOURCE_HZ
            return candidates[heading >= 0.10] if stage == 1 else candidates[heading < 0.10]
        if stage == 4:
            begin, end = max(0, terminal - self.HORIZON + 1), terminal - 1
            return (
                np.arange(begin, end + 1, dtype=np.int64)
                if end >= begin else np.empty(0, np.int64)
            )
        begin, end = terminal, len(commands) - 1
        return (
            np.arange(begin, end + 1, dtype=np.int64)
            if end >= begin else np.empty(0, np.int64)
        )

    def _sample_metadata(self, index: int) -> tuple[int, int, int, dict[str, Any]]:
        if self.sampling_mode == "natural_episode":
            assert self._arrays is not None
            episode_count = int(len(self._arrays["episode_source"]))
            value = _mix_uint64(int(index) ^ self.sampling_seed)
            for attempt in range(64):
                candidate = _mix_uint64(value + attempt * 0x9E3779B97F4A7C15)
                episode_id = int(candidate % episode_count)
                payload = self._load_episode(episode_id)
                frame_count = self._natural_frame_count(
                    payload["commands"],
                    payload["terminal_start"],
                    self.exclude_all_zero_starts,
                )
                if frame_count <= 0:
                    continue
                frame_value = _mix_uint64(candidate ^ 0xA0761D6478BD642F)
                frame_index = int(frame_value % frame_count)
                return episode_id, frame_index, -1, payload
            raise RuntimeError("Unable to sample a non-empty natural episode")

        block, within = divmod(int(index), self.MICROBLOCK)
        block_hash = _mix_uint64(block ^ self.sampling_seed)
        stage = self._stage(block_hash)
        for attempt in range(64):
            value = _mix_uint64(block_hash + attempt * 0x9E3779B97F4A7C15)
            source = self._source_for_stage(stage, value >> 32)
            episode_id = self._pick_episode(source, value)
            payload = self._load_episode(episode_id)
            eligible = self._eligible_frames(payload, stage)
            if len(eligible):
                frame_value = _mix_uint64(value ^ 0xA0761D6478BD642F)
                frame_index = int(eligible[(int(frame_value % len(eligible)) + within) % len(eligible)])
                return episode_id, frame_index, stage, payload
        raise RuntimeError(f"Unable to sample an eligible {STAGE_NAMES[stage]} window")

    def _instruction(self, target_id: int, index: int) -> tuple[str, int]:
        target_prompt = self.prompts[target_id]
        instructions = target_prompt["instructions"]
        _, within = divmod(int(index), self.MICROBLOCK)
        if within % 2 == 0:
            return str(target_prompt["point_description"]), -1
        choice = int(_mix_uint64(index ^ self.sampling_seed) % 5)
        return str(instructions[choice]), choice

    def _cache_path(self, task: str) -> Path:
        prompt = PROMPT_TEMPLATE.format(task=task)
        name = hashlib.sha256(prompt.encode("utf-8")).hexdigest() + ".t5_len128.wan22ti2v5b.pt"
        path = next((root / name for root in self.cache_dirs if (root / name).is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing T5 cache for {task!r}")
        return path

    def _validate_all_text_caches(self) -> None:
        missing = []
        for row in self.prompts:
            tasks = (row["point_description"], *row["instructions"])
            for task in tasks:
                try:
                    self._cache_path(str(task))
                except FileNotFoundError:
                    missing.append(str(task))
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} prompt slots; first={missing[:8]}")

    def _context(self, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._text_cache.get(task)
        if cached is not None:
            self._text_cache.move_to_end(task)
            return cached
        payload = torch.load(self._cache_path(task), map_location="cpu", weights_only=True)
        context = payload["context"].contiguous().clone()
        mask = payload["mask"].to(dtype=torch.bool).contiguous()
        if context.ndim != 2 or context.shape[0] != self.context_len or mask.shape != (self.context_len,):
            raise ValueError(f"Invalid T5 cache for {task!r}: {context.shape}/{mask.shape}")
        context[~mask] = 0
        value = (context, torch.ones_like(mask))
        self._text_cache[task] = value
        while len(self._text_cache) > 64:
            self._text_cache.popitem(last=False)
        return value

    def _canvas_bbox(self, bbox: np.ndarray, visible: np.ndarray, missing: float) -> np.ndarray:
        result = np.full(bbox.shape, missing, dtype=np.float32)
        valid = visible & np.isfinite(bbox).all(axis=1)
        valid &= (bbox[:, 2] > bbox[:, 0]) & (bbox[:, 3] > bbox[:, 1])
        valid &= (bbox[:, 0] >= 0) & (bbox[:, 1] >= 0)
        valid &= (bbox[:, 2] <= self.SOURCE_WH[0]) & (bbox[:, 3] <= self.SOURCE_WH[1])
        result[valid, 0] = (bbox[valid, 0] * self.RESIZE_WH[0] / self.SOURCE_WH[0] + self.PAD_LTRB[0]) / self.CANVAS_WH[0]
        result[valid, 2] = (bbox[valid, 2] * self.RESIZE_WH[0] / self.SOURCE_WH[0] + self.PAD_LTRB[0]) / self.CANVAS_WH[0]
        result[valid, 1] = bbox[valid, 1] * self.RESIZE_WH[1] / self.SOURCE_WH[1] / self.CANVAS_WH[1]
        result[valid, 3] = bbox[valid, 3] * self.RESIZE_WH[1] / self.SOURCE_WH[1] / self.CANVAS_WH[1]
        return result

    def _canvas_point(self, point: np.ndarray, visible: np.ndarray) -> np.ndarray:
        result = np.full(point.shape, self.MISSING_COORD, dtype=np.float32)
        valid = visible & np.isfinite(point).all(axis=1)
        valid &= (point[:, 0] >= 0) & (point[:, 0] <= self.SOURCE_WH[0])
        valid &= (point[:, 1] >= 0) & (point[:, 1] <= self.SOURCE_WH[1])
        result[valid, 0] = (point[valid, 0] * self.RESIZE_WH[0] / self.SOURCE_WH[0] + self.PAD_LTRB[0]) / self.CANVAS_WH[0]
        result[valid, 1] = point[valid, 1] * self.RESIZE_WH[1] / self.SOURCE_WH[1] / self.CANVAS_WH[1]
        return result

    def _decode(self, payload: dict[str, Any], frame_index: int) -> torch.Tensor:
        count = len(payload["commands"])
        indices = [min(frame_index + offset, count - 1) for offset in self.VIDEO_OFFSETS]
        frames = decode_video_frames(
            str(payload["video"]),
            timestamps=[value / self.SOURCE_HZ for value in indices],
            tolerance_s=0.034,
            backend=self.video_backend,
        )
        if tuple(frames.shape[1:]) != (3, self.SOURCE_WH[1], self.SOURCE_WH[0]):
            raise ValueError(f"Decoded shape mismatch: {payload['video']} {tuple(frames.shape)}")
        frames_for_canvas = frames.float()
        if self.RESIZE_WH != self.SOURCE_WH:
            frames_for_canvas = F.interpolate(
                frames_for_canvas,
                size=(self.RESIZE_WH[1], self.RESIZE_WH[0]),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        canvas = torch.zeros((len(frames), 3, self.CANVAS_WH[1], self.CANVAS_WH[0]))
        left = self.PAD_LTRB[0]
        canvas[:, :, :, left : left + self.RESIZE_WH[0]] = frames_for_canvas
        return canvas.mul_(2).sub_(1)

    def _raw_action(self, payload: dict[str, Any], frame_index: int) -> np.ndarray:
        count = len(payload["commands"])
        future = frame_index + 1 + np.arange(self.HORIZON, dtype=np.int64)
        commands = np.zeros((self.HORIZON, 2), dtype=np.float32)
        command_indices = frame_index + np.arange(self.HORIZON, dtype=np.int64)
        command_real = command_indices < count
        commands[command_real] = payload["commands"][command_indices[command_real]]
        clipped = np.minimum(future, count - 1)
        action_raw = np.empty((self.HORIZON, 9), dtype=np.float32)
        action_raw[:, :3] = integrate_relative_se2(commands, 1.0 / self.SOURCE_HZ)
        action_raw[:, 3:7] = self._canvas_bbox(
            payload["bbox"][clipped], payload["bbox_visible"][clipped], self.MISSING_COORD
        )
        action_raw[:, 7:9] = self._canvas_point(
            payload["navigable"][clipped], payload["navigable_visible"][clipped]
        )
        return action_raw

    def __getitem__(self, index: int) -> dict[str, Any]:
        failures = []
        for retry in range(self.decode_retries):
            candidate = (int(index) + retry * 104729) % len(self)
            episode_id, frame_index, stage, payload = self._sample_metadata(candidate)
            try:
                video = self._decode(payload, frame_index)
                break
            except (RuntimeError, OSError, ValueError) as error:
                failures.append(f"{payload['video']}:{type(error).__name__}:{error}")
        else:
            raise RuntimeError("Unable to decode fallback samples: " + " | ".join(failures))

        count = len(payload["commands"])
        action_raw = self._raw_action(payload, frame_index)
        action = torch.from_numpy(action_raw).mul(self.action_scale).add(self.action_offset)

        video_indices = np.minimum(
            frame_index + np.asarray(self.VIDEO_OFFSETS, dtype=np.int64), count - 1
        )
        video_visible = payload["bbox_visible"][video_indices].copy()
        video_bbox = self._canvas_bbox(payload["bbox"][video_indices], video_visible, 0.0)
        task, instruction_index = self._instruction(payload["target_id"], candidate)
        context, context_mask = self._context(task)
        return {
            "video": video.permute(1, 0, 2, 3).contiguous(),
            "action": action,
            "action_is_pad": torch.zeros(self.HORIZON, dtype=torch.bool),
            "image_is_pad": torch.zeros(len(self.VIDEO_OFFSETS), dtype=torch.bool),
            "video_bbox": torch.from_numpy(video_bbox),
            "video_bbox_visible": torch.from_numpy(video_visible),
            "context": context,
            "context_mask": context_mask,
            "prompt": PROMPT_TEMPLATE.format(task=task),
            "episode_index": episode_id,
            "frame_index": frame_index,
            "family_id": int(payload["source_id"]),
            "stratum_id": stage,
            "instruction_index": instruction_index,
        }
