import hashlib
import json
import os
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms.functional as transforms_F
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from uniwam.utils import misc
from uniwam.utils.logging_config import get_logger
from uniwam.geometry.se2 import relative_se2
from uniwam.vision import (
    build_aspect_padded_fastwam_mosaic,
    build_matched_fastwam_mosaic,
    build_nav_high_fastwam_mosaic,
    build_robotwin_fastwam_mosaic,
)

from ..dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from .base_lerobot_dataset import BaseLerobotDataset
from .latent_cache import ShardedBFloat16LatentCache
from .quaternion import relative_quaternion_np, relative_quaternion_torch
from .rot6d import (
    fixed_frame_relative_pose_rot6d_np,
    fixed_frame_relative_pose_rot6d_torch,
    relative_pose_rot6d_np,
    relative_pose_rot6d_torch,
)
from .robot_video_dataset import DEFAULT_PROMPT
from .utils.normalizer import LinearNormalizer, load_dataset_stats_from_json, save_dataset_stats_to_json

logger = get_logger(__name__)

BBOX_ACTION_DIM = 8
BBOX_INVALID_VALUE = -1.0
EEF_XY_ACTION_DIM = 6
EEF_XY_INVALID_VALUE = 0.0
NAV_AUX_ACTION_DIM = 6
NAV_AUX_INVALID_VALUE = 0.0


class _BboxActionIndex:
    """Memory-mapped frame-aligned bbox labels built from LeRobot annotations."""

    def __init__(self, root: str | Path, source_name: str) -> None:
        source_root = Path(root) / source_name
        manifest_path = source_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing bbox sidecar manifest: {manifest_path}")
        import json

        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("status") != "complete" or manifest.get("source_name") != source_name:
            raise ValueError(f"Invalid bbox sidecar manifest: {manifest_path}")
        if manifest.get("bbox_order") != [
            "left_xmin", "left_ymin", "left_xmax", "left_ymax",
            "right_xmin", "right_ymin", "right_xmax", "right_ymax",
        ]:
            raise ValueError(f"Unexpected bbox slot order in {manifest_path}")
        if manifest.get("coordinate_frame") != "cam_manip_high_original_424x240":
            raise ValueError(
                "Manipulation bbox labels must be in the cam_manip_high 424x240 frame: "
                f"got {manifest.get('coordinate_frame')!r}"
            )
        self.bbox = np.load(source_root / "bbox_xyxy_int16.npy", mmap_mode="r")
        self.available = np.load(source_root / "annotation_available.npy", mmap_mode="r")
        expected_frames = int(manifest["total_frames"])
        if self.bbox.shape != (expected_frames, BBOX_ACTION_DIM):
            raise ValueError(f"BBox sidecar shape mismatch: {self.bbox.shape}")
        if self.available.shape != (expected_frames,):
            raise ValueError(f"BBox availability shape mismatch: {self.available.shape}")

    def window(
        self, start_index: int, episode_stop_index: int, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = np.minimum(
            int(start_index) + np.arange(int(horizon), dtype=np.int64),
            int(episode_stop_index) - 1,
        )
        if start_index < 0 or episode_stop_index > self.bbox.shape[0] or start_index >= episode_stop_index:
            raise IndexError(
                f"BBox window start/episode-stop [{start_index},{episode_stop_index}) "
                f"is outside {self.bbox.shape[0]} frames"
            )
        raw = np.asarray(self.bbox[indices], dtype=np.float32).copy()
        available = np.asarray(self.available[indices], dtype=bool).copy()
        scale = np.asarray([424.0, 240.0, 424.0, 240.0], dtype=np.float32)
        per_arm_valid = []
        for arm_slice in (slice(0, 4), slice(4, 8)):
            arm = raw[:, arm_slice]
            valid = np.any(arm >= 0.0, axis=-1)
            if not np.all(arm[~valid] == BBOX_INVALID_VALUE):
                raise ValueError("Invalid arm bbox must use the -1 sentinel in all four slots")
            if np.any(valid & ~available):
                raise ValueError("BBox coordinates are valid on a frame marked annotation-unavailable")
            arm[valid] /= scale
            per_arm_valid.append(valid & available)
        # Sidecars store valid pixels in [0, 1].  All action channels in the
        # diffusion model use the same [-1, 1] range; invalid values are zero
        # and are excluded by the returned feature mask.
        raw[available] = raw[available] * 2.0 - 1.0
        raw[~np.repeat(np.stack(per_arm_valid, axis=1), 4, axis=1)] = NAV_AUX_INVALID_VALUE
        values = torch.from_numpy(raw)
        feature_mask = torch.from_numpy(
            np.repeat(np.stack(per_arm_valid, axis=1), 4, axis=1)
        )
        return values, feature_mask


class _EefImageActionIndex:
    """Memory-mapped frame-aligned left/right EEF points in the main image."""

    EEF_XY_ORDER = [
        "left_x", "left_y", "right_x", "right_y",
        "left_visible", "right_visible",
    ]
    VALIDITY_MODES = {
        "geometric_in_fov": "geometric_in_fov.npy",
        "visible": "track_visible.npy",
    }

    def __init__(self, root: str | Path, source_name: str, validity: str) -> None:
        source_root = Path(root) / source_name
        manifest_path = source_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing EEF-XY sidecar manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "complete"
            or manifest.get("schema_version") != "fastwam_eef_xy_sidecar_v1"
            or manifest.get("source_name") != source_name
        ):
            raise ValueError(f"Invalid EEF-XY sidecar manifest: {manifest_path}")
        if manifest.get("eef_xy_order") != self.EEF_XY_ORDER:
            raise ValueError(f"Unexpected EEF-XY slot order in {manifest_path}")
        if manifest.get("coordinate_frame") != (
            "fastwam_manipulation_mosaic_384x320_xy_and_visibility"
        ):
            raise ValueError(f"Unexpected EEF-XY coordinate frame in {manifest_path}")
        self.validity = str(validity).strip().lower()
        if self.validity not in self.VALIDITY_MODES:
            raise ValueError(
                f"EEF-XY validity must be one of {sorted(self.VALIDITY_MODES)}, "
                f"got {validity!r}."
            )
        expected_frames = int(manifest["total_frames"])
        self.values = np.load(source_root / "eef_xy_visible_normalized.npy", mmap_mode="r")
        self.feature_mask = np.load(source_root / "feature_mask.npy", mmap_mode="r")
        if self.values.shape != (expected_frames, EEF_XY_ACTION_DIM):
            raise ValueError(f"EEF-XY sidecar shape mismatch: {self.values.shape}")
        if self.feature_mask.shape != (expected_frames, EEF_XY_ACTION_DIM):
            raise ValueError(f"EEF auxiliary mask shape mismatch: {self.feature_mask.shape}")

    def window(
        self,
        start_index: int,
        episode_stop_index: int,
        horizon: int,
        action_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(start_index) + int(action_offset)
        stop = int(episode_stop_index)
        if start < 0 or stop > self.values.shape[0] or start >= stop:
            raise IndexError(
                f"EEF-XY window [{start},{stop}) is outside {self.values.shape[0]} frames"
            )
        indices = start + np.arange(int(horizon), dtype=np.int64)
        if int(indices[-1]) >= stop:
            raise IndexError(
                f"EEF-XY H{horizon} window at {start} crosses episode stop {stop}"
            )
        values = np.asarray(self.values[indices], dtype=np.float32).copy()
        feature_mask = np.asarray(self.feature_mask[indices], dtype=bool).copy()
        if not np.isfinite(values).all():
            raise ValueError("EEF-XY sidecar contains non-finite normalized values")
        if np.any(feature_mask & ((values < -1.0001) | (values > 1.0001))):
            raise ValueError("Valid EEF auxiliary values must be in [-1,1]")
        # XY missing values are zero; visibility missing values use -1.
        values[:, :4] = np.where(feature_mask[:, :4], values[:, :4], EEF_XY_INVALID_VALUE)
        values[:, 4:] = np.where(feature_mask[:, 4:], values[:, 4:], -1.0)
        return torch.from_numpy(values), torch.from_numpy(feature_mask)


class _NavAuxActionIndex:
    """Frame-aligned navigation bbox/operating-point labels.

    The six stored values are bbox xyxy followed by operating-point xy.  The
    sidecar uses [0, 1] image coordinates for valid features and a separate
    feature mask; this reader maps valid features to [-1, 1] and leaves missing
    features at zero.  Labels are attached to the observation after the row's
    navigation command (the next-observation convention).
    """

    AUX_ORDER = [
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "op_x", "op_y",
    ]
    ALIGNMENT = "next_observation"

    def __init__(self, root: str | Path, source_name: str) -> None:
        source_root = Path(root) / source_name
        manifest_path = source_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing navigation aux sidecar manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("source_name") != source_name:
            raise ValueError(f"Invalid navigation aux manifest: {manifest_path}")
        if manifest.get("aux_order") != self.AUX_ORDER:
            raise ValueError(f"Unexpected navigation aux order in {manifest_path}")
        if manifest.get("alignment") != self.ALIGNMENT:
            raise ValueError(
                "Navigation auxiliary labels must be aligned to the observation "
                f"after the row command ({self.ALIGNMENT!r}): {manifest_path}"
            )
        if manifest.get("coordinate_frame") != "cam_nav_original_424x240":
            raise ValueError(
                "Navigation auxiliary labels must be in the cam_nav 424x240 frame: "
                f"got {manifest.get('coordinate_frame')!r}"
            )
        self.values = np.load(source_root / "nav_aux_xyxy_op.npy", mmap_mode="r")
        self.feature_mask = np.load(source_root / "feature_mask.npy", mmap_mode="r")
        expected_frames = int(manifest["total_frames"])
        if self.values.shape != (expected_frames, NAV_AUX_ACTION_DIM):
            raise ValueError(f"Navigation aux value shape mismatch: {self.values.shape}")
        if self.feature_mask.shape != (expected_frames, NAV_AUX_ACTION_DIM):
            raise ValueError(f"Navigation aux mask shape mismatch: {self.feature_mask.shape}")

    def window(
        self, start_index: int, episode_stop_index: int, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if start_index < 0 or episode_stop_index > self.values.shape[0] or start_index >= episode_stop_index:
            raise IndexError(
                f"Navigation aux window [{start_index},{episode_stop_index}) is outside "
                f"{self.values.shape[0]} frames"
            )
        # action.nav[t] is the cumulative base pose after command t.  The
        # matching visual annotation is therefore the next observation frame.
        indices = np.minimum(
            int(start_index) + 1 + np.arange(int(horizon), dtype=np.int64),
            int(episode_stop_index) - 1,
        )
        values = np.asarray(self.values[indices], dtype=np.float32).copy()
        mask = np.asarray(self.feature_mask[indices], dtype=bool).copy()
        if not np.isfinite(values).all():
            raise ValueError("Navigation aux sidecar contains non-finite values")
        if np.any(mask & ((values < 0.0) | (values > 1.0))):
            raise ValueError("Valid navigation aux sidecar coordinates must be in [0,1]")
        values[mask] = values[mask] * 2.0 - 1.0
        values[~mask] = NAV_AUX_INVALID_VALUE
        return torch.from_numpy(values), torch.from_numpy(mask)


class _RunningStats:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.count = 0
        self.sum = np.zeros((self.dim,), dtype=np.float64)
        self.sumsq = np.zeros((self.dim,), dtype=np.float64)
        self.min = np.full((self.dim,), np.inf, dtype=np.float64)
        self.max = np.full((self.dim,), -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1, self.dim)
        if arr.shape[0] == 0:
            return
        self.count += int(arr.shape[0])
        self.sum += arr.sum(axis=0)
        self.sumsq += np.square(arr).sum(axis=0)
        self.min = np.minimum(self.min, arr.min(axis=0))
        self.max = np.maximum(self.max, arr.max(axis=0))

    def as_fastwam_stats(self) -> dict[str, torch.Tensor]:
        if self.count <= 0:
            raise ValueError("Cannot finalize empty stats.")
        mean = self.sum / float(self.count)
        var = np.maximum(self.sumsq / float(self.count) - np.square(mean), 0.0)
        std = np.sqrt(var)
        return {
            "global_min": torch.from_numpy(self.min.astype(np.float32)),
            "global_max": torch.from_numpy(self.max.astype(np.float32)),
            "global_mean": torch.from_numpy(mean.astype(np.float32)),
            "global_std": torch.from_numpy(std.astype(np.float32)),
        }


def _wrap_rad_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + torch.pi, 2.0 * torch.pi) - torch.pi


def _wrap_rad_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _center_pad_no_resize(
    image: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
    fill: float = 0.5,
) -> torch.Tensor:
    """Pad trailing H/W dimensions without changing source image geometry."""
    height, width = int(image.shape[-2]), int(image.shape[-1])
    if height > target_height or width > target_width:
        raise ValueError(
            "Native image is larger than its no-resize canvas: "
            f"source={height}x{width}, target={target_height}x{target_width}."
        )
    pad_h = target_height - height
    pad_w = target_width - width
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return transforms_F.pad(image, [left, top, right, bottom], fill=fill)


def _resize_aspect_fit_edge_pad(
    image: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Resize without distortion, then replicate edges to the target canvas."""
    height, width = int(image.shape[-2]), int(image.shape[-1])
    scale = min(float(target_height) / height, float(target_width) / width)
    resized_height = max(1, min(target_height, int(round(height * scale))))
    resized_width = max(1, min(target_width, int(round(width * scale))))
    image = transforms_F.resize(
        image,
        size=[resized_height, resized_width],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    pad_h = target_height - resized_height
    pad_w = target_width - resized_width
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return transforms_F.pad(image, [left, top, right, bottom], padding_mode="edge")


def _manip_abs_to_chunk_delta_torch(
    manip_abs: torch.Tensor,
    current_state: torch.Tensor,
    *,
    relative_frame: str = "start_eef",
) -> torch.Tensor:
    if manip_abs.shape[-1] == 20:
        if current_state.shape[-1] == 23:
            left_start, right_start = 3, 13
        elif current_state.shape[-1] == 20:
            left_start, right_start = 0, 10
        else:
            raise ValueError(
                "6D-rotation state must be 23D [base SE2 + arms] or 20D "
                f"[arms only], got {current_state.shape[-1]}."
            )
        if relative_frame not in {"start_eef", "fixed_reference"}:
            raise ValueError(f"Unsupported manipulation relative frame: {relative_frame!r}.")
        relative_pose = (
            fixed_frame_relative_pose_rot6d_torch
            if relative_frame == "fixed_reference"
            else relative_pose_rot6d_torch
        )
        out = manip_abs.clone()
        left_xyz, left_rot6d = relative_pose(
            current_state[left_start : left_start + 3],
            current_state[left_start + 3 : left_start + 9],
            manip_abs[:, 0:3],
            manip_abs[:, 3:9],
        )
        right_xyz, right_rot6d = relative_pose(
            current_state[right_start : right_start + 3],
            current_state[right_start + 3 : right_start + 9],
            manip_abs[:, 10:13],
            manip_abs[:, 13:19],
        )
        out[:, 0:3] = left_xyz
        out[:, 3:9] = left_rot6d
        out[:, 10:13] = right_xyz
        out[:, 13:19] = right_rot6d
        # Grippers are absolute aperture targets. The target data stores no progress slot.
        return out
    if relative_frame != "start_eef":
        raise ValueError(
            "fixed_reference manipulation deltas currently require 20D Rot6D actions, "
            f"got {manip_abs.shape[-1]}D."
        )
    if manip_abs.shape[-1] in (16, 18):
        if current_state.shape[-1] != 19:
            raise ValueError(f"Quaternion state must be 19D, got {current_state.shape[-1]}.")
        out = manip_abs.clone()
        out[:, 0:3] -= current_state[3:6]
        out[:, 3:7] = relative_quaternion_torch(current_state[6:10], manip_abs[:, 3:7])
        right_start = 9 if manip_abs.shape[-1] == 18 else 8
        out[:, right_start : right_start + 3] -= current_state[11:14]
        out[:, right_start + 3 : right_start + 7] = relative_quaternion_torch(
            current_state[14:18],
            manip_abs[:, right_start + 3 : right_start + 7],
        )
        return out
    if manip_abs.shape[-1] not in (14, 15):
        raise ValueError(
            f"`manip_abs` last dim must be 14, 15, 16, 18, or 20, got {manip_abs.shape[-1]}."
        )
    if current_state.shape[-1] != 16:
        raise ValueError(f"Euler state must be 16D, got {current_state.shape[-1]}.")
    out = manip_abs.clone()
    base = current_state[2:16].reshape(1, 14)
    out[:, 0:6] = manip_abs[:, 0:6] - base[:, 0:6]
    out[:, 3:6] = _wrap_rad_torch(out[:, 3:6])
    out[:, 6:7] = manip_abs[:, 6:7]
    out[:, 7:13] = manip_abs[:, 7:13] - base[:, 7:13]
    out[:, 10:13] = _wrap_rad_torch(out[:, 10:13])
    # Grippers, and optional progress slot at dim 14, stay absolute.
    out[:, 13:] = manip_abs[:, 13:]
    return out


def _manip_abs_to_chunk_delta_np(
    manip_abs: np.ndarray,
    current_state: np.ndarray,
    *,
    relative_frame: str = "start_eef",
) -> np.ndarray:
    if manip_abs.shape[-1] == 20:
        if current_state.shape[-1] == 23:
            left_start, right_start = 3, 13
        elif current_state.shape[-1] == 20:
            left_start, right_start = 0, 10
        else:
            raise ValueError(
                "6D-rotation state must be 23D [base SE2 + arms] or 20D "
                f"[arms only], got {current_state.shape[-1]}."
            )
        if relative_frame not in {"start_eef", "fixed_reference"}:
            raise ValueError(f"Unsupported manipulation relative frame: {relative_frame!r}.")
        relative_pose = (
            fixed_frame_relative_pose_rot6d_np
            if relative_frame == "fixed_reference"
            else relative_pose_rot6d_np
        )
        out = np.asarray(manip_abs, dtype=np.float32).copy()
        state = np.asarray(current_state, dtype=np.float32)
        base = state.reshape(state.shape[0], 1, state.shape[-1])
        left_xyz, left_rot6d = relative_pose(
            base[:, :, left_start : left_start + 3],
            base[:, :, left_start + 3 : left_start + 9],
            manip_abs[:, :, 0:3],
            manip_abs[:, :, 3:9],
        )
        right_xyz, right_rot6d = relative_pose(
            base[:, :, right_start : right_start + 3],
            base[:, :, right_start + 3 : right_start + 9],
            manip_abs[:, :, 10:13],
            manip_abs[:, :, 13:19],
        )
        out[:, :, 0:3] = left_xyz
        out[:, :, 3:9] = left_rot6d
        out[:, :, 10:13] = right_xyz
        out[:, :, 13:19] = right_rot6d
        return out
    if relative_frame != "start_eef":
        raise ValueError(
            "fixed_reference manipulation deltas currently require 20D Rot6D actions, "
            f"got {manip_abs.shape[-1]}D."
        )
    if manip_abs.shape[-1] in (16, 18):
        if current_state.shape[-1] != 19:
            raise ValueError(f"Quaternion state must be 19D, got {current_state.shape[-1]}.")
        out = np.asarray(manip_abs, dtype=np.float32).copy()
        base = np.asarray(current_state, dtype=np.float32).reshape(-1, 1, 19)
        out[:, :, 0:3] -= base[:, :, 3:6]
        out[:, :, 3:7] = relative_quaternion_np(base[:, :, 6:10], manip_abs[:, :, 3:7])
        right_start = 9 if manip_abs.shape[-1] == 18 else 8
        out[:, :, right_start : right_start + 3] -= base[:, :, 11:14]
        out[:, :, right_start + 3 : right_start + 7] = relative_quaternion_np(
            base[:, :, 14:18],
            manip_abs[:, :, right_start + 3 : right_start + 7],
        )
        return out
    if manip_abs.shape[-1] not in (14, 15):
        raise ValueError(
            f"`manip_abs` last dim must be 14, 15, 16, 18, or 20, got {manip_abs.shape[-1]}."
        )
    out = np.asarray(manip_abs, dtype=np.float32).copy()
    state = np.asarray(current_state, dtype=np.float32)
    if state.shape[-1] != 16:
        raise ValueError(f"Euler state must be 16D, got {state.shape[-1]}.")
    base = state[:, 2:16].reshape(-1, 1, 14)
    out[:, :, 0:6] -= base[:, :, 0:6]
    out[:, :, 3:6] = _wrap_rad_np(out[:, :, 3:6])
    out[:, :, 6:7] = manip_abs[:, :, 6:7]
    out[:, :, 7:13] -= base[:, :, 7:13]
    out[:, :, 10:13] = _wrap_rad_np(out[:, :, 10:13])
    # Grippers, and optional progress slot at dim 14, stay absolute.
    out[:, :, 13:] = manip_abs[:, :, 13:]
    return out


def _read_episode_arrays(
    dataset_root: Path,
    episode_index: int,
    available_branches: tuple[str, ...],
    action_alignment: str = "current",
    state_column: str = "observation.state",
    action_columns: Optional[dict[str, str]] = None,
    nav_reference_state_column: str = "observation.state",
    manip_action_source: str = "recorded_action",
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    action_columns = action_columns or {
        branch: f"action.{branch}" for branch in available_branches
    }
    missing_action_columns = set(available_branches) - set(action_columns)
    if missing_action_columns:
        raise ValueError(f"Missing action columns for {sorted(missing_action_columns)}.")
    columns = [state_column] + [action_columns[branch] for branch in available_branches]
    if "nav" in available_branches and nav_reference_state_column not in columns:
        columns.append(nav_reference_state_column)
    table = pq.read_table(
        dataset_root / f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet",
        columns=columns,
    )
    state = np.asarray(table[state_column].to_pylist(), dtype=np.float32)
    nav_reference_state = (
        np.asarray(table[nav_reference_state_column].to_pylist(), dtype=np.float32)
        if "nav" in available_branches
        else state
    )
    if "nav" in available_branches and nav_reference_state.shape[-1] != 23:
        raise ValueError(
            "Navigation reference state must be 23D [base SE2 + dual arms], "
            f"got {nav_reference_state.shape} from {nav_reference_state_column!r}."
        )
    if "nav" not in available_branches and state.shape[-1] == 23:
        # A manipulation-only source semantically has a stationary base. Some
        # recorders retain near-zero command chatter; do not integrate that
        # noise into shared-future-state conditioning.
        state = state.copy()
        state[:, :3] = 0.0
    actions = {
        branch: np.asarray(table[action_columns[branch]].to_pylist(), dtype=np.float32)
        for branch in available_branches
    }
    if action_alignment not in {"current", "next"}:
        raise ValueError(
            f"action_alignment must be 'current' or 'next', got {action_alignment!r}."
        )
    if action_alignment == "next":
        # Legacy converted sources stored action[t] = state[t+1]. Normalize them
        # at the dataset boundary so all downstream code sees action[t] = state[t].
        for branch, action in actions.items():
            if action.shape[0] != state.shape[0]:
                raise ValueError(f"State/action length mismatch for {branch}: {state.shape}/{action.shape}")
            if action.shape[0] > 1:
                # The first legacy action is the target at t=1.  Reindex it one
                # frame later; the missing t=0 target is the observed state.
                current = np.empty_like(action)
                if branch == "manip":
                    current[0] = state[0, 3:] if state.shape[-1] == 23 else state[0]
                else:
                    current[0] = action[0]
                current[1:] = action[:-1]
                actions[branch] = current
    if manip_action_source not in {"recorded_action", "future_state"}:
        raise ValueError(
            "manip_action_source must be 'recorded_action' or 'future_state', "
            f"got {manip_action_source!r}."
        )
    if manip_action_source == "future_state":
        if action_alignment != "current":
            raise ValueError("future_state manipulation labels require action_alignment='current'.")
        if "manip" not in available_branches:
            raise ValueError("future_state manipulation labels require the manip branch.")
        arm_state = state[:, 3:] if state.shape[-1] == 23 else state
        if arm_state.shape != actions["manip"].shape:
            raise ValueError(
                "Future-state manipulation labels require state/action shape equality, "
                f"got state={arm_state.shape}, action={actions['manip'].shape}."
            )
        actions["manip"] = arm_state.copy()
    return state, actions, nav_reference_state


def _state_abs_to_snapshot_relative_torch(
    state_abs: torch.Tensor, snapshot_state: torch.Tensor
) -> torch.Tensor:
    """Express absolute 20D/23D robot states in the observation-snapshot frame."""
    if (
        state_abs.ndim != 2
        or state_abs.shape[-1] not in (20, 23)
        or snapshot_state.shape != (state_abs.shape[-1],)
    ):
        raise ValueError(
            f"Expected matching 20D or 23D state tensors, got "
            f"{tuple(state_abs.shape)} and {tuple(snapshot_state.shape)}."
        )
    result = state_abs.clone()
    left_start, right_start = (3, 13) if state_abs.shape[-1] == 23 else (0, 10)
    if state_abs.shape[-1] == 23:
        result[:, :3] = relative_se2(snapshot_state[:3], state_abs[:, :3])
    left_xyz, left_rot6d = relative_pose_rot6d_torch(
        snapshot_state[left_start : left_start + 3],
        snapshot_state[left_start + 3 : left_start + 9],
        state_abs[:, left_start : left_start + 3],
        state_abs[:, left_start + 3 : left_start + 9],
    )
    right_xyz, right_rot6d = relative_pose_rot6d_torch(
        snapshot_state[right_start : right_start + 3],
        snapshot_state[right_start + 3 : right_start + 9],
        state_abs[:, right_start : right_start + 3],
        state_abs[:, right_start + 3 : right_start + 9],
    )
    result[:, left_start : left_start + 3] = left_xyz
    result[:, left_start + 3 : left_start + 9] = left_rot6d
    result[:, right_start : right_start + 3] = right_xyz
    result[:, right_start + 3 : right_start + 9] = right_rot6d
    return result


def _nav_abs_to_snapshot_relative_torch(
    nav_abs: torch.Tensor, snapshot_state: torch.Tensor
) -> torch.Tensor:
    if nav_abs.ndim != 2 or nav_abs.shape[-1] != 3:
        raise ValueError(f"Expected nav_abs [T,3], got {tuple(nav_abs.shape)}.")
    return relative_se2(snapshot_state[:3], nav_abs)


class DualExpertRobotVideoDataset(torch.utils.data.Dataset):
    """FastWAM dataset for shared-video, nav-action, and manip-action experts.

    The dataset length is the materialized `meta/window_index.parquet`, so repeated
    pure manipulation and transition windows are applied at data loading time.
    """

    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames: int = 33,
        video_size=None,
        nav_video_size=None,
        manip_video_size=None,
        camera_key=None,
        text_embedding_cache_dir=None,
        context_len: int = 128,
        pretrained_norm_stats=None,
        val_set_proportion: float = 0.0,
        is_training_set: bool = False,
        global_sample_stride: int = 1,
        action_video_freq_ratio: int = 4,
        video_backend: Optional[str] = None,
        concat_multi_camera: str = "robotwin",
        override_instruction: Optional[str] = None,
        nav_action_dim: int = 2,
        manip_action_dim: int = 14,
        proprio_dim: int = 16,
        use_stepwise_action_norm: bool = False,
        norm_default_mode: str = "z-score",
        norm_exception_mode=None,
        supervise_all_action_losses: bool = False,
        nav_image_mode: str = "resize",
        nav_action_stride: int = 1,
        sampling_weights_path: Optional[str] = None,
        max_padding_retry: int = 3,
        available_branches: str = "both",
        random_fallback_on_error: bool = True,
        shared_observation_max_delay_steps: int = 0,
        shared_observation_delay_offsets: Optional[list[int]] = None,
        episode_cache_size: int = 2,
        precomputed_latent_root: Optional[str] = None,
        precomputed_latent_index_remap_root: Optional[str] = None,
        precomputed_latent_source_name: Optional[str] = None,
        precomputed_latent_temporal_frames: Optional[int] = None,
        precomputed_latent_allow_missing: bool = False,
        action_horizon: Optional[int] = None,
        window_index_path: Optional[str] = None,
        drop_base_state: bool = False,
        proprio_context_steps: int = 1,
        robot_manip_action_dim: Optional[int] = None,
        manip_bbox_index_root: Optional[str] = None,
        manip_bbox_source_name: Optional[str] = None,
        append_missing_bbox_slots: bool = False,
        manip_eef_xy_index_root: Optional[str] = None,
        manip_eef_xy_source_name: Optional[str] = None,
        manip_eef_xy_validity: str = "geometric_in_fov",
        append_missing_eef_xy_slots: bool = False,
        nav_aux_index_root: Optional[str] = None,
        nav_aux_source_name: Optional[str] = None,
        append_missing_nav_aux_slots: bool = False,
        robot_nav_action_dim: Optional[int] = None,
        instruction_variants_path: Optional[str] = None,
        simple_instruction_probability: float = 0.0,
        force_simple_episode_indices_path: Optional[str] = None,
        action_alignment: str = "current",
        state_column: str = "observation.state",
        manip_action_column: str = "action.manip",
        nav_action_column: str = "action.nav",
        nav_reference_state_column: str = "observation.state",
        instruction_prefix: Optional[str] = None,
        instruction_prefix_by_branch: Optional[dict[str, str]] = None,
        manip_action_source: str = "recorded_action",
        manip_action_offset: int = 0,
        manip_relative_frame: str = "start_eef",
    ):
        if video_size is None:
            video_size = [384, 320]
        if len(dataset_dirs) != 1:
            raise ValueError("DualExpertRobotVideoDataset currently expects exactly one dataset_dir.")
        if abs(float(val_set_proportion)) > 1e-9:
            raise ValueError("DualExpertRobotVideoDataset currently requires val_set_proportion=0.0.")
        if camera_key is not None:
            raise ValueError("DualExpertRobotVideoDataset expects all configured cameras; camera_key is unsupported.")

        self.dataset_root = Path(str(dataset_dirs[0]))
        self.shape_meta = (
            OmegaConf.to_container(shape_meta, resolve=True)
            if OmegaConf.is_config(shape_meta)
            else shape_meta
        )
        if not isinstance(self.shape_meta, dict):
            raise TypeError(f"shape_meta must resolve to a dict, got {type(self.shape_meta)}.")
        self.num_frames = int(num_frames)
        self.video_horizon = self.num_frames - 1
        self.action_horizon = (
            self.video_horizon if action_horizon is None else int(action_horizon)
        )
        self.window_index_path = (
            self.dataset_root / "meta/window_index.parquet"
            if window_index_path is None
            else Path(str(window_index_path))
        )
        self.drop_base_state = bool(drop_base_state)
        self.action_alignment = str(action_alignment).strip().lower()
        if self.action_alignment not in {"current", "next"}:
            raise ValueError(
                "action_alignment must be 'current' or 'next', "
                f"got {action_alignment!r}."
            )
        self.state_column = str(state_column)
        self.action_columns = {
            "manip": str(manip_action_column),
            "nav": str(nav_action_column),
        }
        self.nav_reference_state_column = str(nav_reference_state_column)
        self.manip_action_source = str(manip_action_source).strip().lower()
        if self.manip_action_source not in {"recorded_action", "future_state"}:
            raise ValueError(
                "manip_action_source must be 'recorded_action' or 'future_state', "
                f"got {manip_action_source!r}."
            )
        self.manip_action_offset = int(manip_action_offset)
        if self.manip_action_offset < 0:
            raise ValueError(
                f"manip_action_offset must be non-negative, got {manip_action_offset}."
            )
        self.manip_relative_frame = str(manip_relative_frame).strip().lower()
        if self.manip_relative_frame not in {"start_eef", "fixed_reference"}:
            raise ValueError(
                "manip_relative_frame must be start_eef or fixed_reference, "
                f"got {manip_relative_frame!r}."
            )
        # Auxiliary EEF-XY/visibility channels are appended after the 20D
        # robot pose.  The fixed-reference geometry contract applies to the
        # robot pose width, not the final model action width.
        configured_robot_dim = int(
            manip_action_dim if robot_manip_action_dim is None else robot_manip_action_dim
        )
        if self.manip_relative_frame == "fixed_reference" and configured_robot_dim != 20:
            raise ValueError(
                "fixed_reference manipulation deltas require robot_manip_action_dim=20, "
                f"got robot_manip_action_dim={configured_robot_dim}."
            )
        self.instruction_prefix = (
            None if instruction_prefix is None else str(instruction_prefix).strip()
        )
        if self.instruction_prefix is not None:
            if not self.instruction_prefix or not self.instruction_prefix.isascii():
                raise ValueError("instruction_prefix must be non-empty ASCII when provided.")
        self.instruction_prefix_by_branch = {
            str(branch): str(prefix).strip()
            for branch, prefix in (instruction_prefix_by_branch or {}).items()
        }
        if set(self.instruction_prefix_by_branch) - {"manip", "nav"}:
            raise ValueError("instruction_prefix_by_branch only supports manip/nav keys.")
        if self.instruction_prefix is not None and self.instruction_prefix_by_branch:
            raise ValueError(
                "instruction_prefix and instruction_prefix_by_branch are mutually exclusive."
            )
        for branch, prefix in self.instruction_prefix_by_branch.items():
            if not prefix or not prefix.isascii():
                raise ValueError(
                    f"instruction_prefix_by_branch[{branch!r}] must be non-empty ASCII."
                )
        state_meta = self.shape_meta.get("state", [])
        if len(state_meta) != 1 or state_meta[0].get("key") != "default":
            raise ValueError(
                "DualExpertRobotVideoDataset requires exactly one default state entry."
            )
        state_meta[0]["lerobot_key"] = self.state_column
        for action_meta in self.shape_meta.get("action", []):
            branch = str(action_meta["key"])
            if branch not in self.action_columns:
                raise ValueError(f"Unsupported logical action key {branch!r}.")
            action_meta["lerobot_key"] = self.action_columns[branch]
        self.proprio_context_steps = int(proprio_context_steps)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.video_backend = None if video_backend is None else str(video_backend).strip().lower()
        if self.video_backend not in (None, "torchcodec", "pyav", "video_reader"):
            raise ValueError(
                "video_backend must be one of None, torchcodec, pyav, or video_reader; "
                f"got {video_backend!r}."
            )
        self.video_size = list(video_size)
        self.nav_video_size = list(nav_video_size or self.video_size)
        self.manip_video_size = list(manip_video_size or self.video_size)
        for name, size in (
            ("nav_video_size", self.nav_video_size),
            ("manip_video_size", self.manip_video_size),
        ):
            if len(size) != 2 or int(size[0]) % 32 != 0 or int(size[1]) % 32 != 0:
                raise ValueError(
                    f"{name} must be [H,W] with both dimensions divisible by 32 "
                    f"for VAE-downsampled 2x2 DiT patches, got {size}."
                )
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = int(context_len)
        self._text_context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.simple_instruction_probability = float(simple_instruction_probability)
        if not 0.0 <= self.simple_instruction_probability <= 1.0:
            raise ValueError(
                "simple_instruction_probability must be in [0,1], got "
                f"{self.simple_instruction_probability}."
            )
        self.instruction_variants: dict[str, str] = {}
        if instruction_variants_path is not None:
            variants_path = Path(str(instruction_variants_path))
            if not variants_path.is_file():
                raise FileNotFoundError(f"Missing instruction variants: {variants_path}")
            for line_number, line in enumerate(
                variants_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                record = json.loads(line)
                detailed = str(record["detailed_prompt"]).strip()
                simple = str(record["simple_prompt"]).strip()
                if not detailed or not simple or not detailed.isascii() or not simple.isascii():
                    raise ValueError(
                        f"Instruction variants must be non-empty ASCII at "
                        f"{variants_path}:{line_number}."
                    )
                previous = self.instruction_variants.setdefault(detailed, simple)
                if previous != simple:
                    raise ValueError(
                        f"Ambiguous simple prompts for detailed instruction {detailed!r}."
                    )
        elif self.simple_instruction_probability != 0.0:
            raise ValueError(
                "simple_instruction_probability requires instruction_variants_path."
            )
        self.force_simple_episode_indices: set[int] = set()
        if force_simple_episode_indices_path is not None:
            force_path = Path(str(force_simple_episode_indices_path))
            if not force_path.is_file():
                raise FileNotFoundError(f"Missing force-simple episode index: {force_path}")
            for line_number, line in enumerate(
                force_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("force_simple") is not True:
                    raise ValueError(
                        f"Every force-simple row must set force_simple=true at "
                        f"{force_path}:{line_number}."
                    )
                self.force_simple_episode_indices.add(int(record["episode_index"]))
        self.nav_action_dim = int(nav_action_dim)
        self.robot_nav_action_dim = int(
            self.nav_action_dim if robot_nav_action_dim is None else robot_nav_action_dim
        )
        self.manip_action_dim = int(manip_action_dim)
        self.robot_manip_action_dim = int(
            self.manip_action_dim if robot_manip_action_dim is None else robot_manip_action_dim
        )
        self.proprio_dim = int(proprio_dim)
        self.manip_bbox_index = None
        self.manip_eef_xy_index = None
        self.nav_aux_index = None
        self.append_missing_bbox_slots = bool(append_missing_bbox_slots)
        self.append_missing_eef_xy_slots = bool(append_missing_eef_xy_slots)
        self.append_missing_nav_aux_slots = bool(append_missing_nav_aux_slots)
        if manip_bbox_index_root is not None and manip_eef_xy_index_root is not None:
            raise ValueError("Manipulation bbox and EEF-XY sidecars are mutually exclusive.")
        if self.append_missing_bbox_slots and self.append_missing_eef_xy_slots:
            raise ValueError("Only one missing manipulation auxiliary suffix may be selected.")
        if nav_aux_index_root is not None:
            if self.nav_action_dim != 3 or self.robot_nav_action_dim != 3:
                raise ValueError(
                    "Navigation auxiliary slots require raw/model robot navigation action_dim=3; "
                    f"got nav_action_dim={self.nav_action_dim}, robot_nav_action_dim={self.robot_nav_action_dim}."
                )
            self.nav_aux_index = _NavAuxActionIndex(
                nav_aux_index_root, nav_aux_source_name or self.dataset_root.name
            )
        if self.append_missing_nav_aux_slots:
            if self.nav_aux_index is not None:
                raise ValueError(
                    "append_missing_nav_aux_slots and nav_aux_index_root are mutually exclusive."
                )
            if self.nav_action_dim != 3 or self.robot_nav_action_dim != 3:
                raise ValueError("Missing navigation auxiliary slots require a 9D output.")
        if manip_bbox_index_root is not None:
            if self.manip_action_dim != self.robot_manip_action_dim + BBOX_ACTION_DIM:
                raise ValueError(
                    "BBox action output must append exactly eight slots: "
                    f"manip_action_dim={self.manip_action_dim}, "
                    f"robot_manip_action_dim={self.robot_manip_action_dim}."
                )
            if shared_observation_delay_offsets is not None or int(shared_observation_max_delay_steps) > 0:
                raise ValueError("BBox action labels are not implemented for shared-observation training")
            self.manip_bbox_index = _BboxActionIndex(
                manip_bbox_index_root, manip_bbox_source_name or self.dataset_root.name
            )
        if self.append_missing_bbox_slots:
            if self.manip_bbox_index is not None:
                raise ValueError(
                    "append_missing_bbox_slots and manip_bbox_index_root are mutually exclusive."
                )
            if self.manip_action_dim != self.robot_manip_action_dim + BBOX_ACTION_DIM:
                raise ValueError(
                    "Missing BBox slots require an 8D model suffix: "
                    f"manip_action_dim={self.manip_action_dim}, "
                    f"robot_manip_action_dim={self.robot_manip_action_dim}."
                )
        if manip_eef_xy_index_root is not None:
            if self.manip_action_dim != self.robot_manip_action_dim + EEF_XY_ACTION_DIM:
                raise ValueError(
                    "EEF-XY output must append exactly four slots: "
                    f"manip_action_dim={self.manip_action_dim}, "
                    f"robot_manip_action_dim={self.robot_manip_action_dim}."
                )
            if shared_observation_delay_offsets is not None or int(shared_observation_max_delay_steps) > 0:
                raise ValueError("EEF-XY labels are not implemented for shared-observation training")
            self.manip_eef_xy_index = _EefImageActionIndex(
                manip_eef_xy_index_root,
                manip_eef_xy_source_name or self.dataset_root.name,
                manip_eef_xy_validity,
            )
        if self.append_missing_eef_xy_slots:
            if self.manip_eef_xy_index is not None:
                raise ValueError(
                    "append_missing_eef_xy_slots and manip_eef_xy_index_root are mutually exclusive."
                )
            if self.manip_action_dim != self.robot_manip_action_dim + EEF_XY_ACTION_DIM:
                raise ValueError(
                    "Missing EEF-XY slots require a 4D model suffix: "
                    f"manip_action_dim={self.manip_action_dim}, "
                    f"robot_manip_action_dim={self.robot_manip_action_dim}."
                )
        if available_branches == "both":
            self.available_branches = ("manip", "nav")
        else:
            self.available_branches = tuple(
                part.strip() for part in str(available_branches).split(",") if part.strip()
            )
        if not self.available_branches or set(self.available_branches) - {"manip", "nav"}:
            raise ValueError(
                "available_branches must be 'both' or a comma-separated subset of manip/nav, "
                f"got {available_branches!r}."
            )
        self.supervise_all_action_losses = bool(supervise_all_action_losses)
        self.nav_image_mode = str(nav_image_mode).strip().lower()
        self.nav_action_stride = int(nav_action_stride)
        if self.nav_image_mode not in {
            "resize",
            "native_pad",
            "nav_high_mosaic",
            "matched_wrist_mosaic",
        }:
            raise ValueError(
                f"Unsupported nav_image_mode={nav_image_mode!r}; expected "
                "'resize', 'native_pad', 'nav_high_mosaic', or "
                "'matched_wrist_mosaic'."
            )
        self.max_padding_retry = int(max_padding_retry)
        self.random_fallback_on_error = bool(random_fallback_on_error)
        self.shared_observation_max_delay_steps = int(shared_observation_max_delay_steps)
        if shared_observation_delay_offsets is None:
            delay_offsets = tuple(range(self.shared_observation_max_delay_steps + 1))
            self.shared_observation_enabled = self.shared_observation_max_delay_steps > 0
        else:
            delay_offsets = tuple(int(value) for value in shared_observation_delay_offsets)
            self.shared_observation_enabled = True
        if not delay_offsets:
            raise ValueError("shared_observation_delay_offsets must not be empty.")
        if delay_offsets[0] != 0 or tuple(sorted(set(delay_offsets))) != delay_offsets:
            raise ValueError(
                "shared_observation_delay_offsets must be sorted, unique, non-negative, "
                f"and start at zero; got {delay_offsets}."
            )
        self.shared_observation_delay_offsets = delay_offsets
        self.episode_cache_size = int(episode_cache_size)
        self.precomputed_latent_root = precomputed_latent_root
        self.precomputed_latent_index_remap_root = precomputed_latent_index_remap_root
        self.precomputed_latent_source_name = precomputed_latent_source_name
        self.precomputed_latent_allow_missing = bool(precomputed_latent_allow_missing)
        self.precomputed_latent_temporal_frames = (
            None
            if precomputed_latent_temporal_frames is None
            else int(precomputed_latent_temporal_frames)
        )
        if (
            self.precomputed_latent_temporal_frames is not None
            and self.precomputed_latent_temporal_frames <= 0
        ):
            raise ValueError("precomputed_latent_temporal_frames must be positive.")
        if self.shared_observation_max_delay_steps < 0:
            raise ValueError("shared_observation_max_delay_steps must be non-negative.")
        if self.episode_cache_size <= 0:
            raise ValueError("episode_cache_size must be positive.")
        self._episode_array_cache: OrderedDict[
            int, tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]
        ] = OrderedDict()
        self.is_training_set = bool(is_training_set)
        if self.nav_action_dim not in (2, 3, 4):
            raise ValueError(f"Expected nav_action_dim=2, 3, or 4, got {self.nav_action_dim}.")
        if self.robot_nav_action_dim not in (2, 3, 4):
            raise ValueError(f"Unsupported robot_nav_action_dim={self.robot_nav_action_dim}.")
        if self.manip_action_dim not in (14, 15, 16, 18, 20, 24, 26, 28):
            raise ValueError(
                f"Expected manip_action_dim=14, 15, 16, 18, 20, 24, 26, or 28, got {self.manip_action_dim}."
            )
        if self.robot_manip_action_dim not in (14, 15, 16, 18, 20):
            raise ValueError(f"Unsupported robot_manip_action_dim={self.robot_manip_action_dim}.")
        if self.proprio_dim not in (16, 19, 20, 23):
            raise ValueError(f"Expected proprio_dim=16, 19, 20, or 23, got {self.proprio_dim}.")
        if self.nav_action_stride <= 0 or self.action_horizon % self.nav_action_stride != 0:
            raise ValueError(
                f"nav_action_stride must divide horizon {self.action_horizon}, got {self.nav_action_stride}."
            )

        if self.video_horizon <= 0 or self.action_horizon <= 0:
            raise ValueError(
                f"video_horizon/action_horizon must be positive, got "
                f"{self.video_horizon}/{self.action_horizon}."
            )
        if self.video_horizon % self.action_video_freq_ratio != 0:
            raise ValueError("num_frames-1 must be divisible by action_video_freq_ratio.")
        if (self.video_horizon // self.action_video_freq_ratio) % 4 != 0:
            raise ValueError("video transitions must be divisible by 4 for Wan tokenization.")
        if self.proprio_context_steps <= 0:
            raise ValueError("proprio_context_steps must be positive.")
        self.video_sample_indices = list(range(0, self.num_frames, self.action_video_freq_ratio))

        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=self.shape_meta,
            obs_size=self.num_frames,
            action_size=self.action_horizon,
            val_set_proportion=0.0,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            image_obs_indices=self.video_sample_indices,
            video_backend=self.video_backend,
        )
        self.lerobot_dataset._set_return_images(True)
        self._episode_lengths = {
            int(index): int(to - start)
            for index, (start, to) in enumerate(
                zip(
                    self.lerobot_dataset.episode_data_index["from"].tolist(),
                    self.lerobot_dataset.episode_data_index["to"].tolist(),
                    strict=True,
                )
            )
        }
        self._source_to_episode_index = {}
        episode_offset = 0
        for dataset in self.lerobot_dataset.multi_dataset._datasets:
            for index, record in dataset.meta.episodes.items():
                self._source_to_episode_index[int(record.get("source_episode_index", index))] = episode_offset + int(index)
            episode_offset += len(dataset.episodes)
        if self.instruction_variants:
            tasks_path = self.dataset_root / "meta" / "tasks.jsonl"
            task_rows = [
                json.loads(line)
                for line in tasks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            missing_tasks = sorted(
                {
                    str(record["task"])
                    for record in task_rows
                    if str(record["task"]) not in self.instruction_variants
                }
            )
            if missing_tasks:
                raise ValueError(
                    f"Instruction variants are missing {len(missing_tasks)} dataset tasks; "
                    f"first={missing_tasks[0]!r}."
                )
        episode_count = len(self.lerobot_dataset.episode_data_index["from"])
        invalid_force_simple = sorted(
            index
            for index in self.force_simple_episode_indices
            if index < 0 or index >= episode_count
        )
        if invalid_force_simple:
            raise ValueError(
                "force-simple episode indices are outside the copied dataset: "
                f"{invalid_force_simple[:8]} (episode_count={episode_count})."
            )


        window_path = self.window_index_path
        if not window_path.is_file():
            raise FileNotFoundError(f"Missing window index: {window_path}")
        window_table = pq.read_table(window_path)
        self.window_episode_index = np.asarray(window_table["episode_index"].to_numpy(), dtype=np.int64)
        self.window_source_episode_index = np.asarray(
            window_table["source_episode_index"].to_numpy(), dtype=np.int64
        ) if "source_episode_index" in window_table.column_names else self.window_episode_index.copy()
        self.window_start_frame = np.asarray(window_table["start_frame"].to_numpy(), dtype=np.int64)
        self.window_horizon = np.asarray(window_table["horizon"].to_numpy(), dtype=np.int64)
        default_sample_type = "pure_manip" if self.available_branches == ("manip",) else "pure_nav"
        self.window_sample_type = (
            np.asarray(window_table["sample_type"].to_pylist(), dtype=object)
            if "sample_type" in window_table.column_names
            else np.full(window_table.num_rows, default_sample_type, dtype=object)
        )
        self.window_nav_loss_valid = (
            np.asarray(window_table["nav_loss_valid"].to_numpy(), dtype=bool)
            if "nav_loss_valid" in window_table.column_names
            else np.full(window_table.num_rows, "nav" in self.available_branches, dtype=bool)
        )
        self.window_manip_loss_valid = (
            np.asarray(window_table["manip_loss_valid"].to_numpy(), dtype=bool)
            if "manip_loss_valid" in window_table.column_names
            else np.full(window_table.num_rows, "manip" in self.available_branches, dtype=bool)
        )
        if self.supervise_all_action_losses:
            if "nav" in self.available_branches:
                self.window_nav_loss_valid = np.ones_like(self.window_nav_loss_valid, dtype=bool)
            if "manip" in self.available_branches:
                self.window_manip_loss_valid = np.ones_like(self.window_manip_loss_valid, dtype=bool)
        if not np.all(self.window_horizon == self.action_horizon):
            raise ValueError(
                f"Window horizon mismatch: expected {self.action_horizon}, "
                f"got unique={sorted(set(self.window_horizon.tolist()))[:8]}"
            )

        self.window_sample_weight = np.ones(len(self.window_start_frame), dtype=np.float64)
        if sampling_weights_path is not None:
            weights_path = Path(str(sampling_weights_path))
            if not weights_path.is_absolute():
                weights_path = self.dataset_root / weights_path
            if not weights_path.is_file():
                raise FileNotFoundError(f"Missing sampling weights: {weights_path}")
            weights_table = pq.read_table(weights_path)
            required = {"window_index", "episode_index", "start_frame", "sampling_weight"}
            missing = required - set(weights_table.column_names)
            if missing:
                raise ValueError(f"Sampling weights missing columns: {sorted(missing)}")
            if weights_table.num_rows != len(self.window_start_frame):
                raise ValueError(
                    f"Sampling weight row mismatch: {weights_table.num_rows} vs {len(self.window_start_frame)}."
                )
            weight_index = weights_table["window_index"].to_numpy(zero_copy_only=False)
            weight_episode = weights_table["episode_index"].to_numpy(zero_copy_only=False)
            weight_start = weights_table["start_frame"].to_numpy(zero_copy_only=False)
            if not np.array_equal(weight_index, np.arange(len(self.window_start_frame))):
                raise ValueError("Sampling weight window_index is not contiguous and aligned.")
            if not np.array_equal(weight_episode, self.window_episode_index):
                raise ValueError("Sampling weight episode_index does not match window_index.parquet.")
            if not np.array_equal(weight_start, self.window_start_frame):
                raise ValueError("Sampling weight start_frame does not match window_index.parquet.")
            weights = weights_table["sampling_weight"].to_numpy(zero_copy_only=False).astype(np.float64)
            if not np.isfinite(weights).all() or np.any(weights <= 0.0):
                raise ValueError("Sampling weights must all be finite and positive.")
            self.window_sample_weight = weights
            logger.info(
                "Loaded aligned sampling weights from %s: min=%.3f mean=%.3f max=%.3f",
                weights_path,
                float(weights.min()),
                float(weights.mean()),
                float(weights.max()),
            )

        self.latent_cache = None
        if self.precomputed_latent_root is not None:
            self.latent_cache = ShardedBFloat16LatentCache(
                self.precomputed_latent_root,
                self.precomputed_latent_source_name or self.dataset_root.name,
                self.available_branches,
                len(self.window_start_frame),
                index_remap_root=self.precomputed_latent_index_remap_root,
                source_window_index_path=self.window_index_path,
            )
            # A partial legacy cache can cover most current windows. Keep image
            # loading enabled only when an uncached window may need raw fallback.
            self.lerobot_dataset._set_return_images(
                self.precomputed_latent_allow_missing
            )

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(args={"img_w": self.video_size[1], "img_h": self.video_size[0]})
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

        stats = self._load_or_compute_stats(
            pretrained_norm_stats=pretrained_norm_stats,
            use_stepwise_action_norm=use_stepwise_action_norm,
            norm_default_mode=norm_default_mode,
            norm_exception_mode=norm_exception_mode,
        )
        self.normalizer = LinearNormalizer(
            use_stepwise_action_norm=use_stepwise_action_norm,
            shape_meta=self.shape_meta,
            default_mode=norm_default_mode,
            exception_mode=norm_exception_mode,
            stats=stats,
        )

    def __len__(self):
        return int(self.window_start_frame.shape[0])

    def _load_or_compute_stats(
        self,
        *,
        pretrained_norm_stats,
        use_stepwise_action_norm: bool,
        norm_default_mode: str,
        norm_exception_mode,
    ):
        if pretrained_norm_stats:
            logger.info("Using dual-expert dataset stats: %s", pretrained_norm_stats)
            return load_dataset_stats_from_json(pretrained_norm_stats)

        partial = PartialState()
        if partial.is_main_process:
            logger.info("Computing dual-expert window-weighted dataset stats from %s", self.dataset_root)
            stats = self._compute_window_weighted_stats()
            save_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_dataset_stats_to_json(stats, save_path)
            logger.info("Saved dual-expert dataset stats to %s", save_path)
        else:
            stats = None

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            obj_list = [stats]
            torch.distributed.broadcast_object_list(obj_list, src=0)
            stats = obj_list[0]
        if stats is None:
            raise RuntimeError("Dataset stats were not computed or broadcast.")
        return stats

    def _compute_window_weighted_stats(self):
        stats_state = _RunningStats(self.proprio_dim)
        action_stats = {}
        if "nav" in self.available_branches:
            action_stats["nav"] = _RunningStats(self.nav_action_dim)
        if "manip" in self.available_branches:
            action_stats["manip"] = _RunningStats(self.robot_manip_action_dim)
        offsets = np.arange(self.action_horizon, dtype=np.int64)

        for episode_index in sorted(set(self.window_episode_index.tolist())):
            mask = self.window_episode_index == episode_index
            starts = self.window_start_frame[mask]
            if starts.shape[0] == 0:
                continue
            state, actions, nav_reference_state = _read_episode_arrays(
                self.dataset_root,
                int(episode_index),
                self.available_branches,
                self.action_alignment,
                self.state_column,
                self.action_columns,
                self.nav_reference_state_column,
                self.manip_action_source,
            )
            indices = starts[:, None] + offsets[None, :]
            state_context = state[starts]
            if self.drop_base_state:
                if state_context.shape[-1] != 23:
                    raise ValueError(
                        "drop_base_state requires raw 23D [base SE2, dual-arm rot6D] state."
                    )
                state_context = state_context[:, 3:]
            elif state_context.shape[-1] != self.proprio_dim:
                raise ValueError(
                    f"Raw state dimension {state_context.shape[-1]} does not match "
                    f"proprio_dim={self.proprio_dim}."
                )
            stats_state.update(state_context.reshape(-1, self.proprio_dim))
            if "nav" in actions:
                # action.nav is stored as an episode-frame SE(2) target.  The
                # model deliberately does not receive the episode-global base
                # pose, so every window must use its observation frame as the
                # navigation origin, exactly as online head prefixes do.
                nav_abs = torch.from_numpy(actions["nav"][indices]).float()
                snapshot_base = torch.from_numpy(nav_reference_state[starts, :3]).float()
                nav_win = relative_se2(snapshot_base[:, None, :], nav_abs).numpy()
                action_stats["nav"].update(
                    nav_win[:, :: self.nav_action_stride].reshape(-1, self.nav_action_dim)
                )
            if "manip" in actions:
                manip_abs_win = actions["manip"][indices]
                if self.manip_action_offset:
                    manip_indices = indices + self.manip_action_offset
                    if int(manip_indices.max()) >= len(actions["manip"]):
                        raise ValueError(
                            "Manipulation action offset exceeds an episode boundary: "
                            f"episode={episode_index}, offset={self.manip_action_offset}."
                        )
                    manip_abs_win = actions["manip"][manip_indices]
                manip_delta = _manip_abs_to_chunk_delta_np(
                    manip_abs_win,
                    state[starts],
                    relative_frame=self.manip_relative_frame,
                )
                action_stats["manip"].update(
                    manip_delta.reshape(-1, self.robot_manip_action_dim)
                )

        return {
            "action": {key: value.as_fastwam_stats() for key, value in action_stats.items()},
            "state": {
                "default": stats_state.as_fastwam_stats(),
            },
        }

    def _base_frame_index(self, episode_index: int, start_frame: int) -> int:
        ep_from = int(self.lerobot_dataset.episode_data_index["from"][episode_index].item())
        return ep_from + int(start_frame)

    def _source_episode_for_window(self, window_index: int) -> int:
        idx = int(window_index)
        logical = int(self.window_episode_index[idx])
        source = int(self.window_source_episode_index[idx])
        candidates = [logical, self._source_to_episode_index.get(source), source]
        for candidate in candidates:
            if candidate is None:
                continue
            length = self._episode_lengths.get(int(candidate))
            if length is not None and int(self.window_start_frame[idx]) + self.action_horizon <= length:
                return int(candidate)
        raise IndexError(
            f"Window {idx} has no episode mapping for logical={logical}, source={source}, "
            f"start={int(self.window_start_frame[idx])}, horizon={self.action_horizon}."
        )

    def _cached_episode_arrays(self, episode_index: int):
        cached = self._episode_array_cache.pop(episode_index, None)
        if cached is None:
            cached = _read_episode_arrays(
                self.dataset_root,
                episode_index,
                self.available_branches,
                self.action_alignment,
                self.state_column,
                self.action_columns,
                self.nav_reference_state_column,
                self.manip_action_source,
            )
        self._episode_array_cache[episode_index] = cached
        while len(self._episode_array_cache) > self.episode_cache_size:
            self._episode_array_cache.popitem(last=False)
        return cached

    def _shared_observation_targets(
        self,
        *,
        episode_index: int,
        start_frame: int,
        selected_branch: Optional[str],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        state_np, actions_np, nav_reference_state_np = self._cached_episode_arrays(episode_index)
        length = int(state_np.shape[0])
        offsets = torch.tensor(self.shared_observation_delay_offsets, dtype=torch.long)
        snapshot = torch.from_numpy(state_np[start_frame]).float()
        state_indices = torch.clamp(start_frame + offsets, max=length - 1)
        future_state = _state_abs_to_snapshot_relative_torch(
            torch.from_numpy(state_np[state_indices.numpy()]).float(), snapshot
        )
        actions: dict[str, torch.Tensor] = {}
        pads: dict[str, torch.Tensor] = {}
        steps = torch.arange(self.action_horizon, dtype=torch.long)
        for branch in self.available_branches:
            if selected_branch not in (None, branch):
                continue
            indices = start_frame + offsets[:, None] + steps[None, :]
            if branch == "manip":
                indices = indices + self.manip_action_offset
            pad = indices >= length
            clipped = torch.clamp(indices, max=length - 1)
            absolute = torch.from_numpy(actions_np[branch][clipped.numpy()]).float()
            if branch == "manip":
                flat = _manip_abs_to_chunk_delta_torch(
                    absolute.reshape(-1, self.robot_manip_action_dim),
                    snapshot,
                    relative_frame=self.manip_relative_frame,
                )
                actions[branch] = flat.reshape(
                    offsets.numel(), self.action_horizon, self.robot_manip_action_dim
                )
            else:
                nav_snapshot = torch.from_numpy(nav_reference_state_np[start_frame]).float()
                flat = _nav_abs_to_snapshot_relative_torch(
                    absolute.reshape(-1, 3), nav_snapshot
                )
                actions[branch] = flat.reshape(offsets.numel(), self.action_horizon, 3)
            pads[branch] = pad
        return future_state, actions, pads, offsets

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cached = self._text_context_cache.get(prompt)
        if cached is not None:
            return cached
        os.makedirs(self.text_embedding_cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(
            self.text_embedding_cache_dir,
            f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py for this config first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2 or context.shape[0] != self.context_len:
            raise ValueError(f"Cached context shape mismatch in {cache_path}: {tuple(context.shape)}")
        if context_mask.ndim != 1 or context_mask.shape[0] != self.context_len:
            raise ValueError(f"Cached mask shape mismatch in {cache_path}: {tuple(context_mask.shape)}")
        # Match upstream FastWAM exactly: zero padded T5 embeddings, then keep
        # the fixed 128-position context visible to cross-attention.
        context = context.masked_fill(~context_mask[:, None], 0.0).contiguous()
        context_mask = torch.ones_like(context_mask).contiguous()
        self._text_context_cache[prompt] = (context, context_mask)
        return context, context_mask

    def _select_instruction_task(self, task: str, episode_index: int) -> tuple[str, str]:
        if self.override_instruction is not None:
            return str(self.override_instruction), "override"
        if not self.instruction_variants:
            return task, "detailed"
        simple = self.instruction_variants.get(task)
        if simple is None:
            raise KeyError(f"No simple instruction variant for task {task!r}.")
        force_simple = int(episode_index) in self.force_simple_episode_indices
        use_simple = force_simple or (
            self.simple_instruction_probability > 0.0
            and np.random.random() < self.simple_instruction_probability
        )
        if use_simple:
            return simple, "forced_simple" if force_simple else "simple"
        return task, "detailed"

    def _build_video(
        self,
        images: dict[str, torch.Tensor],
        image_is_pad: torch.Tensor,
        selected_branch: Optional[str] = None,
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor]:
        if selected_branch not in (None, "manip", "nav"):
            raise ValueError(f"selected_branch must be manip/nav/None, got {selected_branch!r}.")
        processed = []
        for meta in self.shape_meta["images"]:
            key = meta["key"]
            image = images[key]
            if image.ndim != 4:
                raise ValueError(f"Expected image `{key}` shape [T,C,H,W], got {tuple(image.shape)}")
            processed.append(image.float() / 255.0)
        video = torch.stack(processed, dim=0)
        if video.shape[1] == self.num_frames:
            video = video[:, self.video_sample_indices]
            image_is_pad = image_is_pad[self.video_sample_indices]
        elif video.shape[1] != len(self.video_sample_indices):
            raise ValueError(
                f"Expected {self.num_frames} dense or {len(self.video_sample_indices)} "
                f"preselected image frames, got {video.shape[1]}."
            )

        num_cameras, t_video, c, h, w = video.shape
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(f"`concat_multi_camera='robotwin'` requires 3 cameras, got {num_cameras}.")
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_top, bottom], dim=-2)
        elif self.concat_multi_camera == "fastwam_384x320":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera=fastwam_384x320` requires 3 cameras, got {num_cameras}."
                )
            video = build_matched_fastwam_mosaic(video[0], video[1], video[2])
        elif self.concat_multi_camera == "fastwam_384x320_aspect_pad":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera=fastwam_384x320_aspect_pad` requires 3 cameras, got {num_cameras}."
                )
            video = build_aspect_padded_fastwam_mosaic(video[0], video[1], video[2])
        elif self.concat_multi_camera == "parallel_nav_manip":
            if num_cameras != 4:
                raise ValueError(
                    f"`concat_multi_camera=parallel_nav_manip` requires 4 cameras, got {num_cameras}."
                )
            nav_panel = None
            manip_panel = None
            if selected_branch != "manip":
                if self.nav_image_mode == "matched_wrist_mosaic":
                    nav_panel = build_matched_fastwam_mosaic(
                        video[0], video[2], video[3]
                    )
                elif self.nav_image_mode == "nav_high_mosaic":
                    nav_panel = build_nav_high_fastwam_mosaic(video[0], video[1])
                elif self.nav_image_mode == "native_pad":
                    nav_panel = _center_pad_no_resize(
                        video[0],
                        target_height=int(self.nav_video_size[0]),
                        target_width=int(self.nav_video_size[1]),
                    )
                else:
                    nav_panel = transforms_F.resize(
                        video[0],
                        size=[384, 320],
                        interpolation=transforms_F.InterpolationMode.BILINEAR,
                        antialias=True,
                    )
            if selected_branch != "nav" and self.nav_image_mode == "matched_wrist_mosaic":
                manip_panel = build_matched_fastwam_mosaic(
                    video[1], video[2], video[3]
                )
            elif selected_branch != "nav":
                manip_panel = build_robotwin_fastwam_mosaic(video[1], video[2], video[3])
                if self.nav_image_mode == "native_pad":
                    manip_panel = _center_pad_no_resize(
                        manip_panel,
                        target_height=int(self.manip_video_size[0]),
                        target_width=int(self.manip_video_size[1]),
                    )
            expected_nav_size = tuple(map(int, self.nav_video_size))
            expected_manip_size = tuple(map(int, self.manip_video_size))
            if nav_panel is not None and nav_panel.shape[-2:] != expected_nav_size:
                raise ValueError(
                    f"Navigation panel shape {tuple(nav_panel.shape[-2:])} does not match "
                    f"nav_video_size={expected_nav_size}."
                )
            if manip_panel is not None and manip_panel.shape[-2:] != expected_manip_size:
                raise ValueError(
                    f"Manipulation panel shape {tuple(manip_panel.shape[-2:])} does not match "
                    f"manip_video_size={expected_manip_size}."
                )
            payload = {}
            if nav_panel is not None:
                payload["nav_video"] = self.normalize_transform(nav_panel).permute(1, 0, 2, 3)
            if manip_panel is not None:
                payload["manip_video"] = self.normalize_transform(manip_panel).permute(1, 0, 2, 3)
            primary_branch = selected_branch or "manip"
            payload["video"] = payload[f"{primary_branch}_video"]
            return payload, image_is_pad
        elif self.concat_multi_camera == "dual_canvas_nav_manip":
            if num_cameras != 4:
                raise ValueError(
                    f"`concat_multi_camera=dual_canvas_nav_manip` requires 4 cameras, got {num_cameras}."
                )
            nav_panel = transforms_F.resize(
                video[0],
                size=[384, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            manip_top = transforms_F.resize(
                video[1],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            manip_left = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            manip_right = transforms_F.resize(
                video[3],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            manip_bottom = torch.cat([manip_left, manip_right], dim=-1)
            manip_panel = torch.cat([manip_top, manip_bottom], dim=-2)
            video = torch.cat([nav_panel, manip_panel], dim=-1)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(f"Invalid concat_multi_camera: {self.concat_multi_camera}")
        else:
            video = video.squeeze(0)

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        video = video.permute(1, 0, 2, 3)
        return video, image_is_pad

    def _get(self, idx: int, selected_branch: Optional[str] = None):
        if selected_branch is not None and selected_branch not in self.available_branches:
            raise ValueError(
                f"Branch {selected_branch!r} is unavailable; available={self.available_branches}."
            )
        episode_index = int(self.window_episode_index[idx])
        source_episode_index = self._source_episode_for_window(idx)
        start_frame = int(self.window_start_frame[idx])
        base_idx = self._base_frame_index(source_episode_index, start_frame)
        cached_branch = selected_branch
        if cached_branch is None and len(self.available_branches) == 1:
            cached_branch = self.available_branches[0]
        use_cached_latent = (
            self.latent_cache is not None
            and cached_branch is not None
            and self.latent_cache.has(cached_branch, idx)
        )
        # Toggle the underlying LeRobot video path per sample. Cache hits need
        # state/action only; raw fallback samples are the only ones that decode.
        if self.latent_cache is not None and self.precomputed_latent_allow_missing:
            self.lerobot_dataset._set_return_images(not use_cached_latent)
        sample = self.lerobot_dataset[base_idx]

        got_episode = int(sample["episode_index"].item())
        got_frame = int(sample["frame_index"].item())
        if got_episode != source_episode_index or got_frame != start_frame:
            raise RuntimeError(
                "LeRobot sample mismatch for window row: "
                f"expected episode/frame=({source_episode_index},{start_frame}), "
                f"got ({got_episode},{got_frame})."
            )

        cached_latent = None
        video = None
        extra_videos = None
        image_is_pad = sample["image_is_pad"].bool()
        if image_is_pad.numel() == self.num_frames:
            image_is_pad = image_is_pad[self.video_sample_indices]
        elif image_is_pad.numel() != len(self.video_sample_indices):
            raise ValueError(
                f"Unexpected image padding length {image_is_pad.numel()}; expected "
                f"{self.num_frames} or {len(self.video_sample_indices)}."
            )
        if use_cached_latent:
            if cached_branch is None:
                raise ValueError("Cached paired datasets must be accessed through get_branch().")
            cached_latent = self.latent_cache.get(cached_branch, idx)
            if self.precomputed_latent_temporal_frames is not None:
                keep = self.precomputed_latent_temporal_frames
                if cached_latent.ndim != 4 or cached_latent.shape[1] < keep:
                    raise ValueError(
                        "Cached latent cannot satisfy requested temporal crop: "
                        f"shape={tuple(cached_latent.shape)}, keep={keep}."
                    )
                cached_latent = cached_latent[:, :keep].contiguous()
        else:
            if self.latent_cache is not None and not self.precomputed_latent_allow_missing:
                raise KeyError(
                    f"No cached latent for {self.precomputed_latent_source_name or self.dataset_root.name}/"
                    f"{cached_branch}/{idx}; raw fallback is disabled."
                )
            video_payload, image_is_pad = self._build_video(
                sample["images"], sample["image_is_pad"].bool(), selected_branch
            )
            extra_videos = video_payload if isinstance(video_payload, dict) else None
            video = video_payload["video"] if isinstance(video_payload, dict) else video_payload
        state_raw = sample["state"]["default"].float()
        shared_pads: dict[str, torch.Tensor] = {}
        delay_offsets = None
        if self.shared_observation_enabled:
            proprio, actions, shared_pads, delay_offsets = self._shared_observation_targets(
                episode_index=source_episode_index,
                start_frame=start_frame,
                selected_branch=selected_branch,
            )
        else:
            if self.drop_base_state:
                if state_raw.shape[-1] != 23:
                    raise ValueError(
                        "drop_base_state requires raw 23D [base SE2, dual-arm rot6D] state."
                    )
                proprio = state_raw[: self.proprio_context_steps, 3:].float()
                # Padding applies to time steps, not state features. Projecting
                # 23D state to arm-only 20D must leave this [T] mask unchanged.
                proprio_is_pad = sample["state_is_pad"][: self.proprio_context_steps].bool()
            else:
                if state_raw.shape[-1] != self.proprio_dim:
                    raise ValueError(
                        f"Raw state dimension {state_raw.shape[-1]} does not match "
                        f"proprio_dim={self.proprio_dim}."
                    )
                proprio = state_raw[: self.proprio_context_steps].float()
                proprio_is_pad = sample["state_is_pad"][: self.proprio_context_steps].bool()
            actions = {}
            if "nav" in self.available_branches and selected_branch in (None, "nav"):
                # Stored labels are in the episode frame.  Rebase them to this
                # window's first observation so state can remain arm-only and
                # head-prefix commands share the exact same SE(2) convention.
                _, _, nav_reference_state = self._cached_episode_arrays(source_episode_index)
                actions["nav"] = _nav_abs_to_snapshot_relative_torch(
                    sample["action"]["nav"].float(),
                    torch.from_numpy(nav_reference_state[start_frame]).float(),
                )
            if "manip" in self.available_branches and selected_branch in (None, "manip"):
                if self.manip_action_source == "future_state" or self.manip_action_offset:
                    _, episode_actions, _ = self._cached_episode_arrays(source_episode_index)
                    indices = np.minimum(
                        start_frame
                        + self.manip_action_offset
                        + np.arange(self.action_horizon, dtype=np.int64),
                        len(episode_actions["manip"]) - 1,
                    )
                    if start_frame + self.manip_action_offset + self.action_horizon > len(episode_actions["manip"]):
                        raise ValueError(
                            "Manipulation action offset exceeds an episode boundary: "
                            f"episode={episode_index}, start={start_frame}, "
                            f"offset={self.manip_action_offset}, horizon={self.action_horizon}."
                        )
                    manip_abs = torch.from_numpy(episode_actions["manip"][indices]).float()
                else:
                    manip_abs = sample["action"]["manip"].float()
                actions["manip"] = _manip_abs_to_chunk_delta_torch(
                    manip_abs,
                    state_raw[0],
                    relative_frame=self.manip_relative_frame,
                )

        norm_batch = {"action": actions, "state": {"default": proprio}}
        norm_batch = self.normalizer.forward(norm_batch)
        actions = norm_batch["action"]
        proprio = norm_batch["state"]["default"]
        manip_feature_mask = None
        nav_feature_mask = None
        if "manip" in actions and self.manip_bbox_index is not None:
            bbox, bbox_feature_mask = self.manip_bbox_index.window(
                base_idx,
                int(self.lerobot_dataset.episode_data_index["to"][episode_index].item()),
                self.action_horizon,
            )
            robot_action = actions["manip"]
            if robot_action.shape != (self.action_horizon, self.robot_manip_action_dim):
                raise ValueError(f"Unexpected normalized robot action shape: {robot_action.shape}")
            actions["manip"] = torch.cat((robot_action, bbox), dim=-1)
            manip_feature_mask = torch.cat(
                (torch.ones_like(robot_action, dtype=torch.bool), bbox_feature_mask), dim=-1
            )
        elif "manip" in actions and self.append_missing_bbox_slots:
            robot_action = actions["manip"]
            if robot_action.shape != (self.action_horizon, self.robot_manip_action_dim):
                raise ValueError(f"Unexpected normalized robot action shape: {robot_action.shape}")
            missing_bbox = torch.full(
                (self.action_horizon, BBOX_ACTION_DIM),
                NAV_AUX_INVALID_VALUE,
                dtype=robot_action.dtype,
                device=robot_action.device,
            )
            actions["manip"] = torch.cat((robot_action, missing_bbox), dim=-1)
            manip_feature_mask = torch.cat(
                (
                    torch.ones_like(robot_action, dtype=torch.bool),
                    torch.zeros_like(missing_bbox, dtype=torch.bool),
                ),
                dim=-1,
            )
        elif "manip" in actions and self.manip_eef_xy_index is not None:
            eef_xy, eef_xy_feature_mask = self.manip_eef_xy_index.window(
                base_idx,
                int(self.lerobot_dataset.episode_data_index["to"][episode_index].item()),
                self.action_horizon,
                self.manip_action_offset,
            )
            robot_action = actions["manip"]
            if robot_action.shape != (self.action_horizon, self.robot_manip_action_dim):
                raise ValueError(f"Unexpected normalized robot action shape: {robot_action.shape}")
            actions["manip"] = torch.cat((robot_action, eef_xy), dim=-1)
            manip_feature_mask = torch.cat(
                (torch.ones_like(robot_action, dtype=torch.bool), eef_xy_feature_mask), dim=-1
            )
        elif "manip" in actions and self.append_missing_eef_xy_slots:
            robot_action = actions["manip"]
            if robot_action.shape != (self.action_horizon, self.robot_manip_action_dim):
                raise ValueError(f"Unexpected normalized robot action shape: {robot_action.shape}")
            missing_eef_xy = torch.full(
                (self.action_horizon, EEF_XY_ACTION_DIM),
                EEF_XY_INVALID_VALUE,
                dtype=robot_action.dtype,
                device=robot_action.device,
            )
            actions["manip"] = torch.cat((robot_action, missing_eef_xy), dim=-1)
            manip_feature_mask = torch.cat(
                (
                    torch.ones_like(robot_action, dtype=torch.bool),
                    torch.zeros_like(missing_eef_xy, dtype=torch.bool),
                ),
                dim=-1,
            )
        if "nav" in actions and self.nav_aux_index is not None:
            nav_aux, nav_aux_mask = self.nav_aux_index.window(
                base_idx,
                int(self.lerobot_dataset.episode_data_index["to"][episode_index].item()),
                self.action_horizon,
            )
            robot_action = actions["nav"]
            if robot_action.shape != (self.action_horizon, self.robot_nav_action_dim):
                raise ValueError(
                    f"Unexpected normalized robot navigation action shape: {robot_action.shape}"
                )
            actions["nav"] = torch.cat((robot_action, nav_aux), dim=-1)
            nav_feature_mask = torch.cat(
                (torch.ones_like(robot_action, dtype=torch.bool), nav_aux_mask), dim=-1
            )
        elif "nav" in actions and self.append_missing_nav_aux_slots:
            robot_action = actions["nav"]
            if robot_action.shape != (self.action_horizon, self.robot_nav_action_dim):
                raise ValueError(
                    f"Unexpected normalized robot navigation action shape: {robot_action.shape}"
                )
            missing_aux = torch.full(
                (self.action_horizon, NAV_AUX_ACTION_DIM),
                NAV_AUX_INVALID_VALUE,
                dtype=robot_action.dtype,
                device=robot_action.device,
            )
            actions["nav"] = torch.cat((robot_action, missing_aux), dim=-1)
            nav_feature_mask = torch.cat(
                (
                    torch.ones_like(robot_action, dtype=torch.bool),
                    torch.zeros_like(missing_aux, dtype=torch.bool),
                ),
                dim=-1,
            )
        if "nav" in actions:
            actions["nav"] = actions["nav"][:: self.nav_action_stride]

        task = str(sample["task"])
        task, prompt_variant = self._select_instruction_task(task, episode_index)
        instruction_prefix = self.instruction_prefix
        if self.instruction_prefix_by_branch:
            prompt_branch = selected_branch
            if prompt_branch is None and len(self.available_branches) == 1:
                prompt_branch = self.available_branches[0]
            if prompt_branch is None:
                raise ValueError(
                    "Branch-specific instruction prefixes require get_branch(index, branch)."
                )
            instruction_prefix = self.instruction_prefix_by_branch.get(prompt_branch)
            if instruction_prefix is None:
                raise ValueError(f"Missing instruction prefix for branch {prompt_branch!r}.")
        if instruction_prefix is not None:
            task = f"{instruction_prefix} {task}"
        instruction = DEFAULT_PROMPT.format(task=task)
        context, context_mask = self._get_cached_text_context(instruction)

        result = {
            "proprio": proprio,
            "prompt": instruction,
            "prompt_variant": prompt_variant,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"].bool(),
            "proprio_is_pad": (
                sample["state_is_pad"][: self.proprio_context_steps].bool()
                if self.shared_observation_enabled
                else proprio_is_pad
            ),
            "sample_type": str(self.window_sample_type[idx]),
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
            "frame_index": torch.tensor(start_frame, dtype=torch.long),
            "window_index": torch.tensor(idx, dtype=torch.long),
        }
        if video is not None:
            result["video"] = video
        if delay_offsets is not None:
            result["delay_offsets"] = delay_offsets
            result["offset_mask"] = torch.ones_like(delay_offsets, dtype=torch.bool)
            result["num_offsets"] = torch.tensor(delay_offsets.numel(), dtype=torch.long)
        for branch in ("manip", "nav"):
            active = branch in self.available_branches and selected_branch in (None, branch)
            result[f"{branch}_branch_valid"] = torch.tensor(active, dtype=torch.bool)
            if not active:
                continue
            action = actions[branch]
            result[f"{branch}_action"] = action
            result[f"{branch}_loss_valid"] = torch.tensor(
                bool(
                    self.window_manip_loss_valid[idx]
                    if branch == "manip"
                    else self.window_nav_loss_valid[idx]
                ),
                dtype=torch.bool,
            )
            result[f"{branch}_progress_valid"] = torch.tensor(
                self.manip_action_dim == 18 if branch == "manip" else self.nav_action_dim == 4,
                dtype=torch.bool,
            )
            result[f"{branch}_action_feature_mask"] = (
                manip_feature_mask
                if branch == "manip" and manip_feature_mask is not None
                else nav_feature_mask
                if branch == "nav" and nav_feature_mask is not None
                else torch.ones_like(action, dtype=torch.bool)
            )
            result[f"{branch}_image_is_pad"] = image_is_pad
            if branch == "nav":
                result["nav_action_is_pad"] = (
                    shared_pads[branch]
                    if delay_offsets is not None
                    else sample["action_is_pad"].bool()[:: self.nav_action_stride]
                )
            elif delay_offsets is not None:
                result["action_is_pad"] = shared_pads[branch]
            if cached_latent is not None:
                result[f"{branch}_latents"] = cached_latent
            else:
                branch_video = video
                if extra_videos is not None:
                    branch_video = extra_videos[f"{branch}_video"]
                result[f"{branch}_video"] = branch_video
        return result

    def __getitem__(self, idx: int):
        try:
            return self._get(int(idx))
        except Exception as exc:
            if not self.random_fallback_on_error:
                raise
            print(f"Error processing dual-expert sample idx {idx}: {exc}. Returning a random sample instead.")
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            return self._get(int(random_idx))

    def get_branch(self, idx: int, branch: str):
        """Load one routed stream without materializing the inactive mosaic/action."""
        try:
            return self._get(int(idx), selected_branch=str(branch))
        except Exception:
            if not self.random_fallback_on_error:
                raise
            random_idx = np.random.randint(len(self))
            return self._get(int(random_idx), selected_branch=str(branch))

    def get_branch_video(self, idx: int, branch: str) -> torch.Tensor:
        """Build exactly the routed training video without action/text preprocessing."""
        branch = str(branch)
        if branch not in self.available_branches:
            raise ValueError(
                f"Branch {branch!r} is unavailable; available={self.available_branches}."
            )
        idx = int(idx)
        episode_index = int(self.window_episode_index[idx])
        source_episode_index = self._source_episode_for_window(idx)
        start_frame = int(self.window_start_frame[idx])
        sample = self.lerobot_dataset[self._base_frame_index(source_episode_index, start_frame)]
        got_episode = int(sample["episode_index"].item())
        got_frame = int(sample["frame_index"].item())
        if (got_episode, got_frame) != (source_episode_index, start_frame):
            raise RuntimeError(
                "LeRobot sample mismatch while building latent cache: "
                f"expected ({source_episode_index},{start_frame}), got ({got_episode},{got_frame})."
            )
        payload, _ = self._build_video(
            sample["images"], sample["image_is_pad"].bool(), branch
        )
        video = payload[f"{branch}_video"] if isinstance(payload, dict) else payload
        if tuple(video.shape) != (3, len(self.video_sample_indices), 320, 384):
            raise ValueError(
                "Cached video contract mismatch: expected "
                f"(3,{len(self.video_sample_indices)},320,384), got {tuple(video.shape)}."
            )
        return video
