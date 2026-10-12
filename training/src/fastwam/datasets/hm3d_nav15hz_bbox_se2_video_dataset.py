"""15 Hz HM3D navigation with cumulative relative SE(2) targets."""

from __future__ import annotations

import hashlib
import json
import mmap
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from fastwam.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames


PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"


def integrate_relative_se2(commands: np.ndarray, dt: float) -> np.ndarray:
    """Integrate body-frame ``[vx, vyaw]`` into poses relative to frame t."""
    commands = np.asarray(commands, dtype=np.float32)
    if commands.ndim != 2 or commands.shape[1] != 2:
        raise ValueError(f"Expected commands [T,2], got {commands.shape}")
    vx = commands[:, 0]
    angular = commands[:, 1]
    delta_heading = angular * float(dt)
    turning = np.abs(angular) > 1.0e-7
    local_x = vx * float(dt)
    local_y = np.zeros_like(vx)
    local_x[turning] = vx[turning] * np.sin(delta_heading[turning]) / angular[turning]
    local_y[turning] = vx[turning] * (1.0 - np.cos(delta_heading[turning])) / angular[turning]
    heading_before = np.concatenate(
        (np.zeros(1, dtype=np.float32), np.cumsum(delta_heading[:-1], dtype=np.float32))
    )
    cosine, sine = np.cos(heading_before), np.sin(heading_before)
    step_x = cosine * local_x - sine * local_y
    step_y = sine * local_x + cosine * local_y
    return np.stack(
        (
            np.cumsum(step_x, dtype=np.float32),
            np.cumsum(step_y, dtype=np.float32),
            np.cumsum(delta_heading, dtype=np.float32),
        ),
        axis=-1,
    ).astype(np.float32)


class HM3DNav15HzBBoxSE2VideoDataset(Dataset):
    """Single-view H32 FastWAM data with explicit virtual terminal tails."""

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        image_size: tuple[int, int] | list[int] = (576, 320),
        context_len: int = 128,
        action_horizon: int = 32,
        source_hz: float = 15.0,
        video_offsets: tuple[int, ...] | list[int] = (0, 8, 16, 24, 32),
        text_embedding_cache_dirs: tuple[str, ...] | list[str] = (),
        video_backend: str = "torchcodec",
        pretrained_norm_stats: str | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.manifest = json.loads((self.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        if split not in self.manifest["splits"]:
            raise ValueError(f"Unknown split {split!r}")
        split_payload = self.manifest["splits"][split]
        self.storage_version = str(self.manifest.get("storage_version", "inline_v1"))
        self._partition_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        if self.storage_version == "partitioned_mmap_v2":
            self.episodes = None
            self.source_root = Path(self.manifest["source_root"])
            self.video_name = str(self.manifest["source_contract"]["main_video"])
            self.tasks = tuple(str(task) for task in self.manifest["tasks"])
            self.partitions = sorted(self.manifest["partitions"], key=lambda item: int(item["global_start"]))
            self.partition_starts = np.asarray(
                [int(item["global_start"]) for item in self.partitions], dtype=np.int64
            )
            self.partition_ends = self.partition_starts + np.asarray(
                [int(item["episode_count"]) for item in self.partitions], dtype=np.int64
            )
            if len(self.partition_starts) == 0 or not np.array_equal(
                self.partition_starts[1:], self.partition_ends[:-1]
            ):
                raise ValueError("Partitioned episode index must be non-empty and globally contiguous")
            if int(self.partition_ends[-1]) != int(self.manifest["episode_count"]):
                raise ValueError("Partitioned episode count does not match manifest")
            self.window_episode = np.load(self.dataset_dir / split_payload["episode_path"], mmap_mode="r")
            self.window_frame = np.load(self.dataset_dir / split_payload["frame_path"], mmap_mode="r")
            self.window_family = np.load(self.dataset_dir / split_payload["family_path"], mmap_mode="r")
            self.window_stratum = np.load(self.dataset_dir / split_payload["stratum_path"], mmap_mode="r")
        else:
            self.episodes: list[dict[str, Any]] | None = split_payload["episodes"]
            windows = np.load(self.dataset_dir / split_payload["windows_path"], mmap_mode="r")
            self.window_episode = windows["episode_index"]
            self.window_frame = windows["frame_index"]
            self.window_family = windows["family_id"]
            self.window_stratum = windows["stratum_id"] if "stratum_id" in windows else None
        if not (len(self.window_episode) == len(self.window_frame) == len(self.window_family)):
            raise ValueError("Window arrays must have matching lengths")

        self.action_horizon = int(action_horizon)
        self.source_hz = float(source_hz)
        self.video_offsets = tuple(int(value) for value in video_offsets)
        if self.source_hz != 15.0 or self.action_horizon != 32:
            raise ValueError("This dataset contract is fixed to 15 Hz/H32")
        if self.video_offsets != (0, 8, 16, 24, 32):
            raise ValueError("H32 video offsets must be [0,8,16,24,32]")

        contract = self.manifest["source_contract"]
        self.source_width, self.source_height = map(int, contract["main_video_wh"])
        self.width, self.height = map(int, image_size)
        expected_canvas = tuple(map(int, contract["canvas_wh"]))
        if (self.source_width, self.source_height) != (566, 320):
            raise ValueError(f"Expected exact 566x320 sources, got {self.source_width}x{self.source_height}")
        if (self.width, self.height) != expected_canvas or (self.width, self.height) != (576, 320):
            raise ValueError(f"Expected exact 576x320 canvas, got {self.width}x{self.height}")
        self.left_pad = int(contract["pad_ltrb"][0])
        self.top_pad = int(contract["pad_ltrb"][1])
        right_pad = int(contract["pad_ltrb"][2])
        bottom_pad = int(contract["pad_ltrb"][3])
        if self.left_pad + self.source_width + right_pad != self.width:
            raise ValueError("Horizontal source/padding contract does not match canvas")
        if self.top_pad + self.source_height + bottom_pad != self.height:
            raise ValueError("Vertical source/padding contract does not match canvas")

        stats_path = Path(pretrained_norm_stats or self.dataset_dir / "stats_train_q01q99.json")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats.get("action_order") != ["x_body_m", "y_body_m", "heading_rad"]:
            raise ValueError(f"Unexpected action order in {stats_path}")
        q01 = torch.tensor(stats["relative_se2_q01"], dtype=torch.float32)
        q99 = torch.tensor(stats["relative_se2_q99"], dtype=torch.float32)
        if tuple(q01.shape) != (3,) or tuple(q99.shape) != (3,) or not torch.all(q99 > q01):
            raise ValueError(f"Invalid SE(2) q01/q99 in {stats_path}: {q01}/{q99}")
        self.motion_scale = 2.0 / (q99 - q01)
        self.motion_offset = -1.0 - self.motion_scale * q01

        if self.storage_version == "partitioned_mmap_v2":
            self.actions = None
            self.bbox_xyxy = None
            self.bbox_visible = None
        else:
            self.actions = np.load(self.dataset_dir / self.manifest["actions_path"], mmap_mode="r")
            self.bbox_xyxy = np.load(self.dataset_dir / self.manifest["bbox_xyxy_path"], mmap_mode="r")
            self.bbox_visible = np.load(self.dataset_dir / self.manifest["bbox_visible_path"], mmap_mode="r")
            if self.actions.ndim != 2 or self.actions.shape[1] != 2:
                raise ValueError(f"Expected packed actions [N,2], got {self.actions.shape}")
            if self.bbox_xyxy.shape != (len(self.actions), 4) or self.bbox_visible.shape != (len(self.actions),):
                raise ValueError("Packed bbox arrays do not align with packed actions")

        self.context_len = int(context_len)
        self.video_backend = str(video_backend)
        self.cache_dirs = [Path(path) for path in text_embedding_cache_dirs]
        if not self.cache_dirs:
            raise ValueError("At least one T5 cache directory is required")
        self._text_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._validate_all_prompt_caches()
        if self.storage_version != "partitioned_mmap_v2":
            self.family_window_indices = {
                int(family_id): np.flatnonzero(self.window_family == family_id).astype(np.int64)
                for family_id in np.unique(self.window_family)
            }

    def __len__(self) -> int:
        return int(len(self.window_episode))

    def _cache_path(self, task: str) -> Path:
        prompt = PROMPT_TEMPLATE.format(task=task)
        name = hashlib.sha256(prompt.encode("utf-8")).hexdigest() + f".t5_len{self.context_len}.wan22ti2v5b.pt"
        path = next((root / name for root in self.cache_dirs if (root / name).is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing T5 cache for task {task!r}; searched {self.cache_dirs}")
        return path

    def _validate_all_prompt_caches(self) -> None:
        episodes = self.episodes or ()
        tasks = self.manifest.get("tasks") or sorted({episode["task"] for episode in episodes})
        missing = []
        for task in tasks:
            try:
                self._cache_path(str(task))
            except FileNotFoundError:
                missing.append(str(task))
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)}/{len(tasks)} T5 caches; first={missing[:8]}")

    def _context(self, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = PROMPT_TEMPLATE.format(task=task)
        cached = self._text_cache.get(prompt)
        if cached is not None:
            self._text_cache.move_to_end(prompt)
            return cached
        payload = torch.load(self._cache_path(task), map_location="cpu", weights_only=True)
        context = payload["context"].contiguous().clone()
        mask = payload["mask"].to(dtype=torch.bool).contiguous()
        if context.ndim != 2 or context.shape[0] != self.context_len or mask.shape != (self.context_len,):
            raise ValueError(f"Invalid T5 cache shape for {task!r}: {context.shape}/{mask.shape}")
        if not torch.isfinite(context.float()).all():
            raise ValueError(f"Non-finite T5 context for {task!r}")
        context[~mask] = 0.0
        value = (context, torch.ones_like(mask))
        self._text_cache[prompt] = value
        while len(self._text_cache) > 64:
            self._text_cache.popitem(last=False)
        return value

    def _open_partition(self, part_id: int) -> dict[str, Any]:
        cached = self._partition_cache.get(part_id)
        if cached is not None:
            self._partition_cache.move_to_end(part_id)
            return cached
        part_dir = self.dataset_dir / "partitions" / f"part_{int(self.partitions[part_id]['id']):03d}"
        path_file = (part_dir / "paths.bin").open("rb")
        value = {
            "actions": np.load(part_dir / "actions.npy", mmap_mode="r"),
            "bbox": np.load(part_dir / "bbox_xyxy.npy", mmap_mode="r"),
            "visible": np.load(part_dir / "bbox_visible.npy", mmap_mode="r"),
            "frame_offsets": np.load(part_dir / "frame_offsets.npy", mmap_mode="r"),
            "frame_counts": np.load(part_dir / "frame_counts.npy", mmap_mode="r"),
            "task_ids": np.load(part_dir / "task_ids.npy", mmap_mode="r"),
            "path_offsets": np.load(part_dir / "path_offsets.npy", mmap_mode="r"),
            "path_file": path_file,
            "path_mmap": mmap.mmap(path_file.fileno(), 0, access=mmap.ACCESS_READ),
        }
        self._partition_cache[part_id] = value
        while len(self._partition_cache) > 8:
            _, old = self._partition_cache.popitem(last=False)
            old["path_mmap"].close()
            old["path_file"].close()
        return value

    def _partition_episode(self, episode_index: int) -> tuple[int, int]:
        part_id = int(np.searchsorted(self.partition_ends, episode_index, side="right"))
        if part_id >= len(self.partitions):
            raise IndexError(f"Episode index outside partitions: {episode_index}")
        return part_id, int(episode_index - self.partition_starts[part_id])

    def _episode_data(
        self, episode_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str, str]:
        if self.storage_version == "partitioned_mmap_v2":
            part_id, local_id = self._partition_episode(episode_index)
            part = self._open_partition(part_id)
            offset = int(part["frame_offsets"][local_id])
            count = int(part["frame_counts"][local_id])
            path_begin = int(part["path_offsets"][local_id])
            path_end = int(part["path_offsets"][local_id + 1])
            relative_dir = part["path_mmap"][path_begin:path_end].decode("utf-8")
            task = self.tasks[int(part["task_ids"][local_id])]
            return (
                part["actions"][offset : offset + count],
                part["bbox"][offset : offset + count],
                part["visible"][offset : offset + count],
                count,
                str(self.source_root / relative_dir / self.video_name),
                task,
            )
        assert self.episodes is not None and self.actions is not None
        assert self.bbox_xyxy is not None and self.bbox_visible is not None
        episode = self.episodes[episode_index]
        offset, count = int(episode["packed_offset"]), int(episode["frame_count"])
        return (
            self.actions[offset : offset + count],
            self.bbox_xyxy[offset : offset + count],
            self.bbox_visible[offset : offset + count],
            count,
            str(episode["video_path"]),
            str(episode["task"]),
        )

    def _future_actions(self, raw_actions: np.ndarray, frame_index: int) -> np.ndarray:
        indices = frame_index + 1 + np.arange(self.action_horizon, dtype=np.int64)
        commands = np.zeros((self.action_horizon, 2), dtype=np.float32)
        valid = indices < len(raw_actions)
        commands[valid] = np.asarray(raw_actions[indices[valid]], dtype=np.float32)
        return integrate_relative_se2(commands, dt=1.0 / self.source_hz)

    def _future_bbox(
        self, raw_bbox: np.ndarray, raw_visible: np.ndarray, frame_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = np.minimum(frame_index + np.asarray(self.video_offsets), len(raw_bbox) - 1)
        bbox_px = np.asarray(raw_bbox[indices], dtype=np.float32).copy()
        visible = np.asarray(raw_visible[indices], dtype=np.bool_).copy()
        finite = np.isfinite(bbox_px).all(axis=1)
        in_source = (
            (bbox_px[:, 0] >= 0) & (bbox_px[:, 1] >= 0)
            & (bbox_px[:, 2] <= self.source_width) & (bbox_px[:, 3] <= self.source_height)
        )
        geometry = (bbox_px[:, 2] > bbox_px[:, 0]) & (bbox_px[:, 3] > bbox_px[:, 1])
        visible &= finite & in_source & geometry
        bbox_px[:, [0, 2]] += self.left_pad
        bbox_px[:, [1, 3]] += self.top_pad
        bbox_px[:, [0, 2]] /= self.width
        bbox_px[:, [1, 3]] /= self.height
        bbox_px = np.clip(bbox_px, 0.0, 1.0)
        bbox_px[~visible] = 0.0
        return torch.from_numpy(bbox_px), torch.from_numpy(visible)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index = int(self.window_episode[index])
        frame_index = int(self.window_frame[index])
        raw_actions, raw_bbox, raw_visible, frame_count, video_path, task = self._episode_data(episode_index)
        if not (0 <= frame_index < len(raw_actions)):
            raise IndexError(f"Invalid frame {frame_index} for episode {episode_index}")
        action_raw = self._future_actions(raw_actions, frame_index)
        action = torch.from_numpy(action_raw).mul(self.motion_scale).add(self.motion_offset)
        source_indices = [min(frame_index + offset, len(raw_actions) - 1) for offset in self.video_offsets]
        frames = decode_video_frames(
            video_path,
            timestamps=[frame / self.source_hz for frame in source_indices],
            tolerance_s=0.034,
            backend=self.video_backend,
        )
        if tuple(frames.shape[1:]) != (3, self.source_height, self.source_width):
            raise ValueError(f"Decoded shape mismatch for {video_path}: {tuple(frames.shape)}")
        canvas = torch.zeros((len(frames), 3, self.height, self.width), dtype=torch.float32)
        canvas[:, :, self.top_pad : self.top_pad + self.source_height, self.left_pad : self.left_pad + self.source_width] = frames.float()
        video = canvas.mul_(2.0).sub_(1.0)
        video_bbox, video_bbox_visible = self._future_bbox(raw_bbox, raw_visible, frame_index)
        context, context_mask = self._context(task)
        return {
            "video": video.permute(1, 0, 2, 3).contiguous(),
            "action": action,
            "action_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
            "image_is_pad": torch.zeros(len(self.video_offsets), dtype=torch.bool),
            "video_bbox": video_bbox,
            "video_bbox_visible": video_bbox_visible,
            "context": context,
            "context_mask": context_mask,
            "prompt": PROMPT_TEMPLATE.format(task=task),
            "episode_index": episode_index,
            "frame_index": frame_index,
            "family_id": int(self.window_family[index]),
            "stratum_id": int(self.window_stratum[index]) if self.window_stratum is not None else -1,
        }
