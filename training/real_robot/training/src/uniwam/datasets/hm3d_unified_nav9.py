"""HM3D Joint8 navigation adapter for the mixed FastWAM H32 contract.

The source index is 15 Hz and contains one RGB video per trajectory.  This
adapter exposes a 30 Hz-compatible action horizon without inventing frames:
each source command is held for two virtual ticks and video is decoded only at
source frames ``start + [0, 2, ..., 16]``.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms.functional as transforms_F
from torch.utils.data import Dataset

from uniwam.datasets.lerobot.latent_cache import ShardedBFloat16LatentCache
from uniwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from uniwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from uniwam.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames
from uniwam.vision import build_matched_fastwam_mosaic


HORIZON = 32
SOURCE_HZ = 15.0
VIRTUAL_HZ = 30.0
SOURCE_WIDTH = 566
SOURCE_HEIGHT = 320
CANVAS_WIDTH = 384
CANVAS_HEIGHT = 320
VIDEO_VIRTUAL_OFFSETS = np.arange(0, HORIZON + 1, 4, dtype=np.int64)
VIDEO_SOURCE_OFFSETS = VIDEO_VIRTUAL_OFFSETS // 2
NAV_ACTION_DIM = 9
MISSING_AUX_VALUE = 0.0
NAV_ORDER = (
    "delta_x",
    "delta_y",
    "delta_yaw",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "op_x",
    "op_y",
)


class HM3DVideoDecodeError(RuntimeError):
    """A recoverable source-video decode failure."""


def _integrate_relative_se2(commands: np.ndarray, dt: float) -> np.ndarray:
    """Integrate body-frame [vx, vyaw] commands from a zero pose."""
    values = np.asarray(commands, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected commands [T,2], got {values.shape}.")
    vx = values[:, 0]
    angular = values[:, 1]
    theta = angular * float(dt)
    small = np.abs(theta) < 1.0e-7
    safe = np.where(small, 1.0, angular)
    local_x = np.where(small, vx * float(dt), vx * np.sin(theta) / safe)
    local_y = np.where(small, 0.0, vx * (1.0 - np.cos(theta)) / safe)
    heading_before = np.concatenate(
        (np.zeros(1, dtype=np.float32), np.cumsum(theta[:-1], dtype=np.float32))
    )
    cosine = np.cos(heading_before)
    sine = np.sin(heading_before)
    step_x = cosine * local_x - sine * local_y
    step_y = sine * local_x + cosine * local_y
    return np.stack(
        (
            np.cumsum(step_x, dtype=np.float32),
            np.cumsum(step_y, dtype=np.float32),
            np.cumsum(theta, dtype=np.float32),
        ),
        axis=-1,
    ).astype(np.float32)


def _read_jsonl_annotations(path: Path, expected_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            expected_index = len(rows)
            if int(row.get("idx", -1)) != expected_index:
                raise ValueError(
                    f"Annotation index mismatch at {path}:{line_number}: "
                    f"expected {expected_index}, got {row.get('idx')}"
                )
            rows.append(row)
    if len(rows) != int(expected_count):
        raise ValueError(
            f"Annotation count mismatch for {path}: expected {expected_count}, got {len(rows)}"
        )
    return rows


def _bool_field(row: dict[str, Any], name: str) -> bool:
    value = row.get(name)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    raise ValueError(f"Annotation field {name!r} must be boolean, got {value!r}.")


def _valid_bbox(value: Any, visible: bool) -> tuple[np.ndarray, bool]:
    bbox = np.asarray(value, dtype=np.float32)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        raise ValueError(f"Invalid target_bbox_xyxy shape/value: {value!r}")
    if not visible:
        return bbox, False
    if (
        bbox[0] < 0.0
        or bbox[1] < 0.0
        or bbox[2] > SOURCE_WIDTH
        or bbox[3] > SOURCE_HEIGHT
        or bbox[2] <= bbox[0]
        or bbox[3] <= bbox[1]
    ):
        raise ValueError(f"Visible bbox is outside the 566x320 source canvas: {bbox.tolist()}")
    return bbox, True


def _valid_point(value: Any, visible: bool) -> tuple[np.ndarray, bool]:
    point = np.asarray(value, dtype=np.float32)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"Invalid op_pixel_xy shape/value: {value!r}")
    if not visible:
        return point, False
    if not (0.0 <= float(point[0]) <= SOURCE_WIDTH and 0.0 <= float(point[1]) <= SOURCE_HEIGHT):
        raise ValueError(f"Visible operating point is outside the source canvas: {point.tolist()}")
    return point, True


class HM3DUnifiedNav9Dataset(Dataset):
    """Expose selected unified-index trajectories as a routed nav branch."""

    def __init__(
        self,
        index_dir: str,
        norm_stats: str,
        text_embedding_cache_dir: str | None,
        context_len: int = 128,
        instruction_prefix: str | None = None,
        video_backend: str = "torchcodec",
        decode_tolerance_s: float = 0.04,
        decode_max_retries: int = 8,
        episode_cache_size: int = 8,
        precomputed_latent_root: str | None = None,
        precomputed_latent_source_name: str | None = None,
        precomputed_latent_index_remap_root: str | None = None,
        precomputed_latent_temporal_frames: int | None = None,
    ) -> None:
        self.index_dir = Path(index_dir).expanduser().resolve()
        manifest_path = self.index_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != "fastwam_hm3d_unified_nav9_index_v1":
            raise ValueError(f"Unexpected HM3D index schema: {manifest_path}")
        if self.manifest.get("status") != "complete":
            raise ValueError(f"HM3D index is not complete: {manifest_path}")
        if self.manifest.get("partial_build"):
            raise ValueError("A partial HM3D index cannot be used for training.")

        self.context_len = int(context_len)
        self.video_backend = str(video_backend)
        self.decode_tolerance_s = float(decode_tolerance_s)
        self.decode_max_retries = int(decode_max_retries)
        self.episode_cache_size = int(episode_cache_size)
        if (
            self.context_len <= 0
            or self.decode_tolerance_s <= 0.0
            or self.decode_max_retries < 0
            or self.episode_cache_size <= 0
        ):
            raise ValueError(
                "context_len, decode_tolerance_s, and episode_cache_size must be positive; "
                "decode_max_retries must be non-negative"
            )
        self._decode_warning_episodes: set[int] = set()

        time_contract = self.manifest.get("time_contract", {})
        if float(time_contract.get("source_hz", -1.0)) != SOURCE_HZ:
            raise ValueError("HM3D source rate is not 15 Hz")
        if float(time_contract.get("virtual_hz", -1.0)) != VIRTUAL_HZ:
            raise ValueError("HM3D virtual rate is not 30 Hz")
        if int(time_contract.get("action_horizon", -1)) != HORIZON:
            raise ValueError("HM3D action horizon is not H32")
        if tuple(time_contract.get("video_virtual_offsets", ())) != tuple(VIDEO_VIRTUAL_OFFSETS.tolist()):
            raise ValueError("HM3D video virtual offsets do not match the H32 contract")
        action_contract = self.manifest.get("action_contract", {})
        if action_contract.get("navigation_aux_alignment") != "next_observation":
            raise ValueError("HM3D navigation auxiliary alignment is not next_observation")

        episodes_path = self.index_dir / self.manifest["files"]["episodes"]
        episodes = pq.read_table(episodes_path)
        required_episode_columns = {
            "episode_index",
            "video_path",
            "frame_annotations_path",
            "action_path",
            "action_format",
            "frame_count",
            "task",
        }
        missing = required_episode_columns - set(episodes.column_names)
        if missing:
            raise ValueError(f"HM3D episode table is missing {sorted(missing)}")
        self.episode_video = np.asarray(episodes["video_path"].to_pylist(), dtype=object)
        self.episode_annotations = np.asarray(
            episodes["frame_annotations_path"].to_pylist(), dtype=object
        )
        self.episode_action = np.asarray(episodes["action_path"].to_pylist(), dtype=object)
        self.episode_action_format = np.asarray(episodes["action_format"].to_pylist(), dtype=object)
        self.episode_frame_count = np.asarray(episodes["frame_count"].to_numpy(), dtype=np.int64)
        self.episode_task = np.asarray(episodes["task"].to_pylist(), dtype=object)
        if not (
            len(self.episode_video)
            == len(self.episode_annotations)
            == len(self.episode_action)
            == len(self.episode_action_format)
            == len(self.episode_frame_count)
            == len(self.episode_task)
            > 0
        ):
            raise ValueError("HM3D episode table arrays are empty or misaligned")
        if not np.array_equal(
            np.asarray(episodes["episode_index"].to_numpy(), dtype=np.int64),
            np.arange(len(self.episode_frame_count), dtype=np.int64),
        ):
            raise ValueError("HM3D episode indices are not contiguous")

        window_path = self.index_dir / self.manifest["files"]["window_index"]
        windows = pq.read_table(window_path)
        required_window_columns = {"window_index", "episode_index", "start_frame", "horizon", "sample_type", "sampling_weight"}
        missing = required_window_columns - set(windows.column_names)
        if missing:
            raise ValueError(f"HM3D window table is missing {sorted(missing)}")
        self.window_episode_index = np.asarray(windows["episode_index"].to_numpy(), dtype=np.int64)
        self.window_start_frame = np.asarray(windows["start_frame"].to_numpy(), dtype=np.int64)
        self.window_horizon = np.asarray(windows["horizon"].to_numpy(), dtype=np.int64)
        self.window_sample_type = np.asarray(windows["sample_type"].to_pylist(), dtype=object)
        self.window_sample_weight = np.asarray(windows["sampling_weight"].to_numpy(), dtype=np.float64)
        if not np.array_equal(
            np.asarray(windows["window_index"].to_numpy(), dtype=np.int64),
            np.arange(windows.num_rows, dtype=np.int64),
        ):
            raise ValueError("HM3D window indices are not contiguous")
        if windows.num_rows == 0 or not np.all(self.window_horizon == HORIZON):
            raise ValueError("HM3D window table is empty or not H32")
        if (
            np.any(self.window_episode_index < 0)
            or np.any(self.window_episode_index >= len(self.episode_frame_count))
            or np.any(self.window_start_frame < 0)
            or np.any(self.window_start_frame >= self.episode_frame_count[self.window_episode_index])
            or not np.isfinite(self.window_sample_weight).all()
            or np.any(self.window_sample_weight <= 0.0)
        ):
            raise ValueError("HM3D window metadata contains invalid values")

        stats = load_dataset_stats_from_json(str(Path(norm_stats).expanduser()))
        try:
            q01 = torch.as_tensor(stats["action"]["nav"]["global_q01"], dtype=torch.float32)
            q99 = torch.as_tensor(stats["action"]["nav"]["global_q99"], dtype=torch.float32)
        except KeyError as error:
            raise ValueError("Mixed stats must contain action.nav global_q01/global_q99") from error
        if q01.shape != (3,) or q99.shape != (3,) or not torch.all(q99 > q01):
            raise ValueError(f"Invalid 3D navigation stats in {norm_stats}")
        self.motion_scale = 2.0 / (q99 - q01)
        self.motion_offset = -1.0 - self.motion_scale * q01

        self.instruction_prefix = None if instruction_prefix is None else str(instruction_prefix).strip()
        if self.instruction_prefix == "":
            self.instruction_prefix = None
        if self.instruction_prefix is not None and not self.instruction_prefix.isascii():
            raise ValueError("instruction_prefix must be ASCII")
        self.text_embedding_cache_dir = (
            None if text_embedding_cache_dir is None
            else Path(text_embedding_cache_dir).expanduser()
        )
        if self.text_embedding_cache_dir is not None and not self.text_embedding_cache_dir.is_dir():
            raise FileNotFoundError(self.text_embedding_cache_dir)
        self._text_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._episode_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

        self.latent_cache = None
        if precomputed_latent_temporal_frames is not None and int(precomputed_latent_temporal_frames) != 3:
            raise ValueError(
                "HM3D latent temporal contract is fixed to 3, got "
                f"{precomputed_latent_temporal_frames}."
            )
        if precomputed_latent_root is not None:
            self.latent_cache = ShardedBFloat16LatentCache(
                precomputed_latent_root,
                precomputed_latent_source_name or self.index_dir.name,
                ("nav",),
                len(self.window_episode_index),
                index_remap_root=precomputed_latent_index_remap_root,
                source_window_index_path=window_path,
            )

        self.nav_action_dim = NAV_ACTION_DIM
        # HM3D is nav-only.  Keep the shared schema's robot manipulation
        # metadata truthful even though this branch never emits manip actions.
        self.manip_action_dim = 20
        self.proprio_dim = 20
        self.action_horizon = HORIZON
        self.num_frames = HORIZON + 1
        self.video_sample_indices = VIDEO_VIRTUAL_OFFSETS.tolist()
        self.available_branches = ("nav",)
        self.source_name = self.index_dir.name

    def __len__(self) -> int:
        return int(self.window_episode_index.size)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_text_cache"] = OrderedDict()
        state["_episode_cache"] = OrderedDict()
        return state

    def _load_episode(self, episode_index: int) -> dict[str, Any]:
        episode_index = int(episode_index)
        cached = self._episode_cache.get(episode_index)
        if cached is not None:
            self._episode_cache.move_to_end(episode_index)
            return cached
        count = int(self.episode_frame_count[episode_index])
        annotation_rows = _read_jsonl_annotations(
            Path(str(self.episode_annotations[episode_index])), count
        )
        if self.episode_action_format[episode_index] == "npy_float32_vx_vyaw":
            commands = np.asarray(
                np.load(str(self.episode_action[episode_index]), mmap_mode="r"),
                dtype=np.float32,
            )
            if commands.shape != (count, 2) or not np.isfinite(commands).all():
                raise ValueError(f"Invalid HM3D Joint action array for episode {episode_index}")
        elif self.episode_action_format[episode_index] == "frame_annotations_jsonl_float_vx_vyaw":
            # Canonical annotations store the velocity attached to the arrival
            # frame.  Shift once at this boundary so command[t] advances the
            # observation at t to the target at t+1.
            stored = np.asarray(
                [[float(row["vx"]), float(row["vyaw"])] for row in annotation_rows],
                dtype=np.float32,
            )
            commands = np.zeros_like(stored)
            if count > 1:
                commands[:-1] = stored[1:]
        else:
            raise ValueError(
                f"Unsupported action format {self.episode_action_format[episode_index]!r}"
            )

        bbox = np.empty((count, 4), dtype=np.float32)
        bbox_visible = np.empty(count, dtype=np.bool_)
        point = np.empty((count, 2), dtype=np.float32)
        point_visible = np.empty(count, dtype=np.bool_)
        for index, row in enumerate(annotation_rows):
            bbox[index], bbox_visible[index] = _valid_bbox(
                row["target_bbox_xyxy"], _bool_field(row, "target_visible")
            )
            point[index], point_visible[index] = _valid_point(
                row["op_pixel_xy"], _bool_field(row, "op_visible")
            )
        payload = {
            "commands": commands,
            "bbox": bbox,
            "bbox_visible": bbox_visible,
            "point": point,
            "point_visible": point_visible,
            "video": Path(str(self.episode_video[episode_index])),
            "task": str(self.episode_task[episode_index]),
        }
        self._episode_cache[episode_index] = payload
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return payload

    @staticmethod
    def _source_bbox(bbox: np.ndarray, visible: np.ndarray) -> np.ndarray:
        result = np.zeros_like(np.asarray(bbox, dtype=np.float32))
        valid = np.asarray(visible, dtype=bool)
        # Auxiliary coordinates use normalized coordinates in the source main
        # camera, matching the existing AgileX bbox contract.  They are not
        # normalized against the 384x320 mosaic: the main-panel resize is an
        # affine aspect-preserving transform, so source-normalized coordinates
        # remain the stable cross-camera representation.
        if np.any(valid):
            values = np.asarray(bbox, dtype=np.float32)
            result[valid, 0] = values[valid, 0] / SOURCE_WIDTH
            result[valid, 2] = values[valid, 2] / SOURCE_WIDTH
            result[valid, 1] = values[valid, 1] / SOURCE_HEIGHT
            result[valid, 3] = values[valid, 3] / SOURCE_HEIGHT
        return result

    @staticmethod
    def _source_point(point: np.ndarray, visible: np.ndarray) -> np.ndarray:
        result = np.zeros_like(np.asarray(point, dtype=np.float32))
        valid = np.asarray(visible, dtype=bool)
        if np.any(valid):
            values = np.asarray(point, dtype=np.float32)
            result[valid, 0] = values[valid, 0] / SOURCE_WIDTH
            result[valid, 1] = values[valid, 1] / SOURCE_HEIGHT
        return result

    @staticmethod
    def _aspect_fit_canvas(frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"Expected decoded frames [T,3,H,W], got {tuple(frames.shape)}")
        height, width = int(frames.shape[-2]), int(frames.shape[-1])
        if (width, height) != (SOURCE_WIDTH, SOURCE_HEIGHT):
            raise ValueError(
                f"HM3D video must decode as {SOURCE_WIDTH}x{SOURCE_HEIGHT}, got {width}x{height}"
            )
        # First fit into the canonical 424x240 camera canvas, then use the
        # exact matched 384x320 three-panel layout.  No image is synthesized
        # at an intermediate time; only spatial resizing is performed.
        camera_height, camera_width = 240, 424
        scale = min(camera_height / height, camera_width / width)
        resized_height = int(round(height * scale))
        resized_width = int(round(width * scale))
        unit_frames = frames.float()
        if float(unit_frames.max().item()) > 1.5:
            unit_frames = unit_frames / 255.0
        resized = transforms_F.resize(
            unit_frames,
            [resized_height, resized_width],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        camera = torch.full(
            (frames.shape[0], 3, camera_height, camera_width), 0.5, dtype=resized.dtype
        )
        top = (camera_height - resized_height) // 2
        left = (camera_width - resized_width) // 2
        camera[:, :, top : top + resized_height, left : left + resized_width] = resized
        gray = torch.full_like(camera, 0.5)
        mosaic = build_matched_fastwam_mosaic(camera, gray, gray)
        return mosaic.mul(2.0).sub(1.0)

    def _decode_video(self, payload: dict[str, Any], start_frame: int) -> torch.Tensor:
        count = int(payload["commands"].shape[0])
        source_indices = [
            min(int(start_frame) + int(offset), count - 1)
            for offset in VIDEO_SOURCE_OFFSETS.tolist()
        ]
        try:
            frames = decode_video_frames(
                str(payload["video"]),
                timestamps=[index / SOURCE_HZ for index in source_indices],
                tolerance_s=self.decode_tolerance_s,
                backend=self.video_backend,
            )
        except Exception as exc:
            raise HM3DVideoDecodeError(
                f"Failed to decode HM3D video {payload['video']}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if int(frames.shape[0]) != len(source_indices):
            raise ValueError(f"Decoded frame count mismatch for {payload['video']}: {tuple(frames.shape)}")
        return self._aspect_fit_canvas(frames).permute(1, 0, 2, 3).contiguous()

    def _raw_action(self, payload: dict[str, Any], start_frame: int) -> tuple[np.ndarray, np.ndarray]:
        count = int(payload["commands"].shape[0])
        virtual_command_indices = int(start_frame) + np.arange(HORIZON, dtype=np.int64) // 2
        commands = np.zeros((HORIZON, 2), dtype=np.float32)
        valid_command = virtual_command_indices < count
        if np.any(valid_command):
            commands[valid_command] = payload["commands"][virtual_command_indices[valid_command]]
        motion = _integrate_relative_se2(commands, 1.0 / VIRTUAL_HZ)

        target_indices = np.minimum(
            int(start_frame) + np.arange(HORIZON, dtype=np.int64) // 2 + 1,
            count - 1,
        )
        bbox = self._source_bbox(
            payload["bbox"][target_indices], payload["bbox_visible"][target_indices]
        )
        point = self._source_point(
            payload["point"][target_indices], payload["point_visible"][target_indices]
        )
        raw = np.concatenate((motion, bbox, point), axis=-1).astype(np.float32)
        feature_mask = np.ones((HORIZON, NAV_ACTION_DIM), dtype=np.bool_)
        feature_mask[:, 3:7] = np.repeat(
            payload["bbox_visible"][target_indices, None], 4, axis=1
        )
        feature_mask[:, 7:9] = np.repeat(
            payload["point_visible"][target_indices, None], 2, axis=1
        )
        return raw, feature_mask

    def _normalize_action(self, raw: np.ndarray, feature_mask: np.ndarray) -> torch.Tensor:
        result = torch.from_numpy(raw.copy())
        result[:, :3] = result[:, :3] * self.motion_scale + self.motion_offset
        # Auxiliary pixel coordinates have a fixed, source-independent mapping;
        # missing values stay zero and are excluded by the feature mask.
        valid = torch.from_numpy(np.asarray(feature_mask[:, 3:9], dtype=np.bool_))
        aux = torch.zeros_like(result[:, 3:9])
        aux[valid] = result[:, 3:9][valid] * 2.0 - 1.0
        result[:, 3:9] = aux
        return result

    def _context(self, task: str) -> tuple[torch.Tensor, torch.Tensor, str]:
        if self.text_embedding_cache_dir is None:
            raise RuntimeError(
                "HM3D text_embedding_cache_dir is disabled; this instance is only valid "
                "for VAE video precomputation."
            )
        full_task = task if self.instruction_prefix is None else f"{self.instruction_prefix} {task}"
        prompt = DEFAULT_PROMPT.format(task=full_task)
        cached = self._text_cache.get(prompt)
        if cached is None:
            name = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            path = self.text_embedding_cache_dir / (
                f"{name}.t5_len{self.context_len}.wan22ti2v5b.pt"
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing HM3D prompt cache: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            context = payload["context"].contiguous().clone()
            mask = payload["mask"].to(dtype=torch.bool).contiguous()
            if context.ndim != 2 or context.shape[0] != self.context_len:
                raise ValueError(f"Invalid prompt context shape in {path}: {tuple(context.shape)}")
            if mask.shape != (self.context_len,):
                raise ValueError(f"Invalid prompt mask shape in {path}: {tuple(mask.shape)}")
            if not torch.isfinite(context.float()).all():
                raise ValueError(f"Non-finite prompt context in {path}")
            context[~mask] = 0.0
            cached = (context, torch.ones_like(mask))
            self._text_cache[prompt] = cached
            while len(self._text_cache) > 128:
                self._text_cache.popitem(last=False)
        return cached[0], cached[1], prompt

    def get_branch_video(self, index: int, branch: str) -> torch.Tensor:
        if str(branch) != "nav":
            raise ValueError("HM3DUnifiedNav9Dataset only exposes the nav branch")
        episode = int(self.window_episode_index[int(index)])
        start = int(self.window_start_frame[int(index)])
        return self._decode_video(self._load_episode(episode), start)

    def get_branch(self, index: int, branch: str) -> dict[str, Any]:
        if str(branch) != "nav":
            raise ValueError("HM3DUnifiedNav9Dataset only exposes the nav branch")
        index = int(index)
        episode = int(self.window_episode_index[index])
        start = int(self.window_start_frame[index])
        payload = self._load_episode(episode)
        raw_action, feature_mask = self._raw_action(payload, start)
        action = self._normalize_action(raw_action, feature_mask)
        context, context_mask, prompt = self._context(str(payload["task"]))
        result: dict[str, Any] = {
            "nav_action": action,
            "nav_action_feature_mask": torch.from_numpy(feature_mask),
            "nav_action_is_pad": torch.zeros(HORIZON, dtype=torch.bool),
            "nav_image_is_pad": torch.zeros(len(VIDEO_SOURCE_OFFSETS), dtype=torch.bool),
            "nav_branch_valid": torch.tensor(True, dtype=torch.bool),
            "nav_loss_valid": torch.tensor(True, dtype=torch.bool),
            "nav_progress_valid": torch.tensor(False, dtype=torch.bool),
            "proprio": torch.zeros((1, 20), dtype=torch.float32),
            "proprio_valid": torch.tensor(False, dtype=torch.bool),
            "context": context,
            "context_mask": context_mask,
            "prompt": prompt,
            "sample_type": str(self.window_sample_type[index]),
            "episode_index": torch.tensor(episode, dtype=torch.long),
            "frame_index": torch.tensor(start, dtype=torch.long),
            "window_index": torch.tensor(index, dtype=torch.long),
        }
        if self.latent_cache is not None:
            result["nav_latents"] = self.latent_cache.get("nav", index)
        else:
            result["nav_video"] = self.get_branch_video(index, "nav")
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        requested_index = int(index)
        candidate_index = requested_index
        last_error: HM3DVideoDecodeError | None = None
        for attempt in range(self.decode_max_retries + 1):
            try:
                return self.get_branch(candidate_index, "nav")
            except HM3DVideoDecodeError as exc:
                last_error = exc
                episode = int(self.window_episode_index[candidate_index])
                if episode not in self._decode_warning_episodes:
                    warnings.warn(
                        f"Skipping undecodable HM3D episode={episode}: {exc}",
                        RuntimeWarning,
                    )
                    self._decode_warning_episodes.add(episode)
                candidate_index = (
                    requested_index + (attempt + 1) * 104729
                ) % len(self)
        assert last_error is not None
        raise last_error
