from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import ConcatDataset, Dataset


BRANCHES = ("manip", "nav")


def _scalar_bool(value: Any, name: str) -> bool:
    tensor = torch.as_tensor(value, dtype=torch.bool)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be scalar, got shape {tuple(tensor.shape)}.")
    return bool(tensor.item())


def _stack(samples: Sequence[dict[str, Any]], key: str, *, required: bool) -> torch.Tensor | None:
    values = [sample.get(key) for sample in samples]
    present = [value is not None for value in values]
    if not any(present):
        if required:
            raise KeyError(f"Every mixed-stream sample requires {key!r}.")
        return None
    if not all(present):
        raise ValueError(f"{key!r} must be present in either every sample or no samples.")
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError(f"{key!r} values must all be torch.Tensor instances.")
    try:
        return torch.stack(values, dim=0)
    except RuntimeError as exc:
        shapes = [tuple(value.shape) for value in values]
        raise ValueError(f"Cannot stack {key!r}; sample shapes are {shapes}.") from exc


def _adapt_manip_feature(
    value: torch.Tensor,
    target_dim: int,
    *,
    fill_value: float | bool,
) -> torch.Tensor:
    source_dim = int(value.shape[-1])
    if source_dim == target_dim:
        return value
    if source_dim == 20 and target_dim == 24:
        fill = torch.full(
            (*value.shape[:-1], 4),
            fill_value,
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat((value, fill), dim=-1)
    if source_dim == 20 and target_dim == 26:
        fill = torch.full(
            (*value.shape[:-1], 6),
            fill_value,
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat((value, fill), dim=-1)
    if source_dim == 20 and target_dim == 28:
        fill = torch.full(
            (*value.shape[:-1], 8),
            fill_value,
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat((value, fill), dim=-1)
    if source_dim == 16 and target_dim == 18:
        fill = torch.full(
            (*value.shape[:-1], 1),
            fill_value,
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat((value[..., :8], fill, value[..., 8:], fill.clone()), dim=-1)
    raise ValueError(
        f"Cannot adapt manipulation feature dimension {source_dim} to {target_dim}. "
        "Supported adaptations are 20D -> 24D EEF-XY suffix, 20D -> 26D EEF-XY "
        "plus visibility suffix, 20D -> 28D bbox "
        "suffix, and quaternion 16D -> 18D masked progress."
    )


def _adapt_nav_feature(
    value: torch.Tensor,
    target_dim: int,
    *,
    fill_value: float | bool,
) -> torch.Tensor:
    source_dim = int(value.shape[-1])
    if source_dim == target_dim:
        return value
    if source_dim == 3 and target_dim == 9:
        fill = torch.full(
            (*value.shape[:-1], 6),
            fill_value,
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat((value, fill), dim=-1)
    if source_dim == 3 and target_dim == 4:
        fill = torch.full(
            (*value.shape[:-1], 1),
            fill_value,
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat((value, fill), dim=-1)
    raise ValueError(f"Cannot adapt navigation feature dimension {source_dim} to {target_dim}.")


def _adapt_action(branch: str, value: torch.Tensor, target_dim: int) -> torch.Tensor:
    if value.ndim not in (2, 3):
        raise ValueError(f"{branch}_action must be [T,D] or [N,T,D], got {tuple(value.shape)}.")
    if branch == "manip":
        return _adapt_manip_feature(value, target_dim, fill_value=0.0)
    return _adapt_nav_feature(value, target_dim, fill_value=0.0)


def _feature_mask(
    sample: dict[str, Any],
    branch: str,
    source_action: torch.Tensor,
    target_dim: int,
) -> torch.Tensor:
    key = f"{branch}_action_feature_mask"
    mask = sample.get(key)
    if mask is None:
        mask = torch.ones_like(source_action, dtype=torch.bool)
    else:
        mask = torch.as_tensor(mask, dtype=torch.bool)
        if tuple(mask.shape) != tuple(source_action.shape):
            raise ValueError(
                f"{key} must match source action {tuple(source_action.shape)}, got {tuple(mask.shape)}."
            )
    if branch == "manip":
        mask = _adapt_manip_feature(mask, target_dim, fill_value=False)
        progress_slots = (8, 17) if target_dim == 18 else ()
    else:
        mask = _adapt_nav_feature(mask, target_dim, fill_value=False)
        progress_slots = (3,) if target_dim == 4 else ()
    progress_valid = sample.get(f"{branch}_progress_valid")
    if progress_valid is not None and not _scalar_bool(
        progress_valid, f"{branch}_progress_valid"
    ):
        mask = mask.clone()
        mask[..., list(progress_slots)] = False
    return mask


class MixedStreamCollator:
    """Pack only active manipulation/navigation branches in a heterogeneous batch."""

    def __init__(
        self,
        *,
        manip_action_dim: int,
        nav_action_dim: int,
        manip_horizon: int,
        nav_horizon: int,
    ):
        self.action_dims = {"manip": int(manip_action_dim), "nav": int(nav_action_dim)}
        self.horizons = {"manip": int(manip_horizon), "nav": int(nav_horizon)}

    @staticmethod
    def _is_active(sample: dict[str, Any], branch: str) -> bool:
        key = f"{branch}_branch_valid"
        if key in sample:
            return _scalar_bool(sample[key], key)
        has_visual = (
            f"{branch}_video" in sample or f"{branch}_latents" in sample
        )
        return has_visual and f"{branch}_action" in sample

    def _pack_branch(
        self,
        samples: Sequence[dict[str, Any]],
        branch: str,
        active_indices: list[int],
        result: dict[str, Any],
    ) -> None:
        if not active_indices:
            return
        target_dim = self.action_dims[branch]
        horizon = self.horizons[branch]
        videos = []
        latents = []
        video_positions = []
        latent_positions = []
        actions = []
        feature_masks = []
        image_pad = []
        action_pad = []
        loss_valid = []
        for branch_position, index in enumerate(active_indices):
            sample = samples[index]
            video_key = f"{branch}_video"
            action_key = f"{branch}_action"
            latent_key = f"{branch}_latents"
            has_video = video_key in sample
            has_latent = latent_key in sample
            if has_video == has_latent or action_key not in sample:
                raise KeyError(
                    f"Active {branch} sample {index} requires exactly one of "
                    f"{video_key!r}/{latent_key!r}, plus {action_key!r}."
                )
            visual = sample[video_key] if has_video else sample[latent_key]
            source_action = sample[action_key]
            if not isinstance(visual, torch.Tensor) or not isinstance(source_action, torch.Tensor):
                raise TypeError(f"Active {branch} visual/action must be tensors.")
            action = _adapt_action(branch, source_action, target_dim)
            expected_tail = (horizon, target_dim)
            if tuple(action.shape[-2:]) != expected_tail:
                raise ValueError(
                    f"{branch}_action must end in {expected_tail}, got {tuple(action.shape)}."
                )
            if has_video:
                videos.append(visual)
                video_positions.append(branch_position)
            else:
                latents.append(visual)
                latent_positions.append(branch_position)
            actions.append(action)
            feature_masks.append(_feature_mask(sample, branch, source_action, target_dim))

            image_pad_value = sample.get(
                f"{branch}_image_is_pad", sample.get("image_is_pad")
            )
            if image_pad_value is None:
                num_frames = int(visual.shape[1]) if has_video else (int(visual.shape[1]) - 1) * 4 + 1
                image_pad_value = torch.zeros(num_frames, dtype=torch.bool)
            image_pad.append(torch.as_tensor(image_pad_value, dtype=torch.bool))

            action_pad_key = (
                "manip_action_is_pad" if branch == "manip" else "nav_action_is_pad"
            )
            action_pad_value = sample.get(action_pad_key)
            if action_pad_value is None and branch == "manip":
                action_pad_value = sample.get("action_is_pad")
            if action_pad_value is None:
                action_pad_value = torch.zeros(action.shape[:-1], dtype=torch.bool)
            action_pad_value = torch.as_tensor(action_pad_value, dtype=torch.bool)
            if tuple(action_pad_value.shape) != tuple(action.shape[:-1]):
                raise ValueError(
                    f"{action_pad_key} must match action time axes {tuple(action.shape[:-1])}, "
                    f"got {tuple(action_pad_value.shape)}."
                )
            action_pad.append(action_pad_value)
            loss_valid.append(
                _scalar_bool(sample.get(f"{branch}_loss_valid", True), f"{branch}_loss_valid")
            )

        result[f"{branch}_owner_indices"] = torch.tensor(active_indices, dtype=torch.long)
        if videos:
            result[f"{branch}_video"] = torch.stack(videos, dim=0)
            result[f"{branch}_video_positions"] = torch.tensor(
                video_positions, dtype=torch.long
            )
        if latents:
            result[f"{branch}_latents"] = torch.stack(latents, dim=0)
            result[f"{branch}_latent_positions"] = torch.tensor(
                latent_positions, dtype=torch.long
            )
        result[f"{branch}_action"] = torch.stack(actions, dim=0)
        result[f"{branch}_action_feature_mask"] = torch.stack(feature_masks, dim=0)
        result[f"{branch}_image_is_pad"] = torch.stack(image_pad, dim=0)
        pad_key = "action_is_pad" if branch == "manip" else "nav_action_is_pad"
        result[pad_key] = torch.stack(action_pad, dim=0)
        result[f"{branch}_loss_valid"] = torch.tensor(loss_valid, dtype=torch.bool)

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty mixed-stream batch.")
        result: dict[str, Any] = {
            "context": _stack(samples, "context", required=True),
            "context_mask": _stack(samples, "context_mask", required=True),
            "batch_size": len(samples),
        }
        proprio = _stack(samples, "proprio", required=False)
        if proprio is not None:
            result["proprio"] = proprio
            # A nav-only source has no arm state.  It still supplies a fixed
            # zero-shaped placeholder so heterogeneous batches can be stacked;
            # the model masks that token with this per-sample flag.
            proprio_valid = []
            for sample in samples:
                value = sample.get("proprio_valid", True)
                if not _scalar_bool(value, "proprio_valid"):
                    proprio_valid.append(False)
                else:
                    proprio_valid.append(True)
            result["proprio_valid"] = torch.tensor(proprio_valid, dtype=torch.bool)
        for key in ("delay_offsets", "offset_mask", "num_offsets"):
            value = _stack(samples, key, required=False)
            if value is not None:
                result[key] = value
        result["prompt"] = [sample.get("prompt", "") for sample in samples]

        for key in (
            "episode_index",
            "frame_index",
            "window_index",
            "source_window_index",
        ):
            value = _stack(samples, key, required=False)
            if value is not None:
                result[key] = value
        if all("sample_type" in sample for sample in samples):
            result["sample_type"] = [str(sample["sample_type"]) for sample in samples]

        any_active = False
        for branch in BRANCHES:
            branch_valid = torch.tensor(
                [self._is_active(sample, branch) for sample in samples], dtype=torch.bool
            )
            result[f"{branch}_branch_valid"] = branch_valid
            active_indices = torch.nonzero(branch_valid, as_tuple=False).flatten().tolist()
            any_active |= bool(active_indices)
            self._pack_branch(samples, branch, active_indices, result)
        if not any_active:
            raise ValueError("Every batch must contain at least one active stream.")
        return result

    @staticmethod
    def _pad_inactive_branches(result: dict[str, Any], batch_size: int) -> None:
        """Keep both packed branches present with masked zero placeholders."""
        template_latent = next(
            (result[key] for key in ("manip_latents", "nav_latents") if key in result), None
        )
        template_video = next(
            (result[key] for key in ("manip_video", "nav_video") if key in result), None
        )
        if template_latent is None and template_video is None:
            raise ValueError("Cannot pad mixed-stream branches without a visual template.")
        for branch in BRANCHES:
            owner_key = f"{branch}_owner_indices"
            active = torch.as_tensor(result.get(owner_key, torch.empty(0, dtype=torch.long)), dtype=torch.long)
            active_count = int(active.numel())
            if active_count == batch_size:
                continue
            if active_count:
                visual_key = f"{branch}_latents" if f"{branch}_latents" in result else f"{branch}_video"
                visual = result[visual_key]
            elif template_latent is not None:
                visual_key, visual = f"{branch}_latents", template_latent
            else:
                visual_key, visual = f"{branch}_video", template_video
            full_visual = torch.zeros((batch_size, *visual.shape[1:]), dtype=visual.dtype)
            if active_count:
                full_visual[active] = visual
            result[visual_key] = full_visual
            if visual_key.endswith("latents"):
                result[f"{branch}_latent_positions"] = torch.arange(batch_size, dtype=torch.long)
            result.pop(f"{branch}_video_positions", None)
            result[f"{branch}_owner_indices"] = torch.arange(batch_size, dtype=torch.long)
            result[f"{branch}_branch_valid"] = torch.ones(batch_size, dtype=torch.bool)
            action_key = f"{branch}_action"
            if action_key in result:
                action = result[action_key]
                full = torch.zeros((batch_size, *action.shape[1:]), dtype=action.dtype)
                if active_count:
                    full[active] = action
                result[action_key] = full
            else:
                result[action_key] = torch.zeros((batch_size, 32, 26 if branch == "manip" else 3), dtype=torch.float32)
            mask_key = f"{branch}_action_feature_mask"
            if mask_key in result:
                mask = result[mask_key]
                full = torch.zeros((batch_size, *mask.shape[1:]), dtype=torch.bool)
                if active_count:
                    full[active] = mask
                result[mask_key] = full
            else:
                result[mask_key] = torch.zeros((batch_size, 32, 26 if branch == "manip" else 3), dtype=torch.bool)
            pad_key = "action_is_pad" if branch == "manip" else "nav_action_is_pad"
            if pad_key in result:
                pad = result[pad_key]
                full = torch.ones((batch_size, *pad.shape[1:]), dtype=torch.bool)
                if active_count:
                    full[active] = pad
                result[pad_key] = full
            else:
                result[pad_key] = torch.ones((batch_size, 32), dtype=torch.bool)
            valid = result.get(f"{branch}_loss_valid")
            full_valid = torch.zeros(batch_size, dtype=torch.bool)
            if active_count and valid is not None:
                full_valid[active] = valid
            result[f"{branch}_loss_valid"] = full_valid
            image_key = f"{branch}_image_is_pad"
            if image_key in result:
                image_pad = result[image_key]
                full = torch.ones((batch_size, *image_pad.shape[1:]), dtype=torch.bool)
                if active_count:
                    full[active] = image_pad
                result[image_key] = full
            else:
                frame_count = 9 if branch == "nav" else 33
                result[image_key] = torch.ones((batch_size, frame_count), dtype=torch.bool)


class BranchAnnotatedDataset(Dataset):
    """Adapt an existing dataset to explicit mixed-stream branch semantics."""

    def __init__(
        self,
        dataset: Dataset,
        branches: str | Sequence[str] = "both",
        manip_progress_valid: bool = True,
        nav_progress_valid: bool = True,
        manip_video_key: str = "manip_video",
        nav_video_key: str = "nav_video",
        manip_action_key: str = "manip_action",
        nav_action_key: str = "nav_action",
    ):
        self.dataset = dataset
        if isinstance(branches, str):
            branches = BRANCHES if branches == "both" else (branches,)
        self.branches = tuple(str(branch) for branch in branches)
        unknown = set(self.branches) - set(BRANCHES)
        if unknown or not self.branches:
            raise ValueError(f"branches must contain manip/nav, got {self.branches}.")
        self.progress_valid = {
            "manip": bool(manip_progress_valid),
            "nav": bool(nav_progress_valid),
        }
        self.window_branch = np.full(len(self.dataset), self.branches[0], dtype=object) if len(self.branches) == 1 else getattr(self.dataset, "window_branch", None)
        self.video_keys = {"manip": manip_video_key, "nav": nav_video_key}
        self.action_keys = {"manip": manip_action_key, "nav": nav_action_key}
        latent_cache = getattr(self.dataset, "latent_cache", None)
        if latent_cache is not None and not getattr(
            self.dataset, "precomputed_latent_allow_missing", False
        ):
            all_indices = np.arange(len(self.dataset), dtype=np.int64)
            for branch in self.branches:
                latent_cache.validate_required_indices(branch, all_indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = dict(self.dataset[index])
        if self.branches == ("manip",) and result.get("sample_type") in (None, "unrouted"):
            result["sample_type"] = "pure_manip"
        elif self.branches == ("nav",) and result.get("sample_type") in (None, "unrouted"):
            result["sample_type"] = "pure_nav"
        if "source_window_index" not in result:
            result["source_window_index"] = torch.tensor(index, dtype=torch.long)
        for branch in BRANCHES:
            active = branch in self.branches
            result[f"{branch}_branch_valid"] = torch.tensor(active, dtype=torch.bool)
            if active:
                video_key = self.video_keys[branch]
                action_key = self.action_keys[branch]
                latent_key = f"{branch}_latents"
                if (video_key not in result and latent_key not in result) or action_key not in result:
                    raise KeyError(
                        f"Dataset sample {index} lacks configured {branch} keys "
                        f"{video_key!r}/{action_key!r}."
                    )
                if latent_key not in result:
                    result[f"{branch}_video"] = result[video_key]
                result[f"{branch}_action"] = result[action_key]
                result[f"{branch}_progress_valid"] = torch.tensor(
                    self.progress_valid[branch], dtype=torch.bool
                )
            else:
                for suffix in (
                    "video",
                    "latents",
                    "action",
                    "action_feature_mask",
                    "image_is_pad",
                    "action_is_pad",
                    "loss_valid",
                ):
                    result.pop(f"{branch}_{suffix}", None)
        # A duplicate generic video can add hundreds of MB to prefetched batches.
        result.pop("video", None)
        return result

    def __getattr__(self, name: str) -> Any:
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


class PhaseRoutedDataset(Dataset):
    """Expose a virtual phase index while activating exactly one stream per item."""

    def __init__(
        self,
        dataset: Dataset,
        phase_index_path: str,
        manip_progress_valid: bool = False,
        nav_progress_valid: bool = False,
    ):
        self.dataset = dataset
        table = pq.read_table(phase_index_path)
        required = {
            "source_window_index",
            "episode_index",
            "start_frame",
            "horizon",
            "branch",
            "sample_kind",
            "sampling_weight",
        }
        missing = required - set(table.column_names)
        if missing:
            raise ValueError(f"Phase index is missing columns: {sorted(missing)}")
        self.source_window_index = np.asarray(
            table["source_window_index"].to_numpy(), dtype=np.int64
        )
        self.window_episode_index = np.asarray(
            table["episode_index"].to_numpy(), dtype=np.int64
        )
        self.window_start_frame = np.asarray(
            table["start_frame"].to_numpy(), dtype=np.int64
        )
        self.window_horizon = np.asarray(table["horizon"].to_numpy(), dtype=np.int64)
        self.window_branch = np.asarray(table["branch"].to_pylist(), dtype=object)
        self.window_sample_type = np.asarray(
            table["sample_kind"].to_pylist(), dtype=object
        )
        self.window_sample_weight = np.asarray(
            table["sampling_weight"].to_numpy(), dtype=np.float64
        )
        self.progress_valid = {
            "manip": bool(manip_progress_valid),
            "nav": bool(nav_progress_valid),
        }
        self._validate_index()
        latent_cache = getattr(self.dataset, "latent_cache", None)
        if latent_cache is not None and not getattr(
            self.dataset, "precomputed_latent_allow_missing", False
        ):
            for branch in BRANCHES:
                required = self.source_window_index[self.window_branch == branch]
                if required.size:
                    latent_cache.validate_required_indices(branch, required)

    def _validate_index(self) -> None:
        size = int(self.source_window_index.size)
        for name in (
            "window_episode_index",
            "window_start_frame",
            "window_horizon",
            "window_branch",
            "window_sample_type",
            "window_sample_weight",
        ):
            value = getattr(self, name)
            if value.shape != (size,):
                raise ValueError(f"{name} must have shape ({size},), got {value.shape}.")
        if size == 0:
            raise ValueError("Phase index must not be empty.")
        if self.source_window_index.min() < 0 or self.source_window_index.max() >= len(self.dataset):
            raise ValueError("Phase index source_window_index is out of dataset bounds.")
        if set(self.window_branch.tolist()) - set(BRANCHES):
            raise ValueError("Phase index branch values must be manip or nav.")
        if not np.isfinite(self.window_sample_weight).all() or np.any(
            self.window_sample_weight <= 0.0
        ):
            raise ValueError("Phase sampling weights must be finite and positive.")
        source_episode = np.asarray(self.dataset.window_episode_index, dtype=np.int64)[
            self.source_window_index
        ]
        source_start = np.asarray(self.dataset.window_start_frame, dtype=np.int64)[
            self.source_window_index
        ]
        if not np.array_equal(source_episode, self.window_episode_index):
            raise ValueError("Phase index episode_index does not match source windows.")
        if not np.array_equal(source_start, self.window_start_frame):
            raise ValueError("Phase index start_frame does not match source windows.")
        action_horizon = getattr(self.dataset, "action_horizon", None)
        if action_horizon is not None and not np.all(self.window_horizon == int(action_horizon)):
            raise ValueError(
                "Phase index horizon does not match the child action horizon: "
                f"phase={sorted(set(self.window_horizon.tolist()))[:8]}, "
                f"child={int(action_horizon)}."
            )

    def __len__(self) -> int:
        return int(self.source_window_index.size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = int(self.source_window_index[index])
        branch = str(self.window_branch[index])
        get_branch = getattr(self.dataset, "get_branch", None)
        if not callable(get_branch):
            raise TypeError(
                "PhaseRoutedDataset requires a child dataset with get_branch(index, branch)."
            )
        result = dict(get_branch(source_index, branch))
        other = "nav" if branch == "manip" else "manip"
        result[f"{branch}_branch_valid"] = torch.tensor(True, dtype=torch.bool)
        result[f"{branch}_progress_valid"] = torch.tensor(
            self.progress_valid[branch], dtype=torch.bool
        )
        result[f"{other}_branch_valid"] = torch.tensor(False, dtype=torch.bool)
        for suffix in (
            "video",
            "latents",
            "action",
            "action_feature_mask",
            "image_is_pad",
            "action_is_pad",
            "loss_valid",
            "progress_valid",
        ):
            result.pop(f"{other}_{suffix}", None)
        result.pop("video", None)
        result["sample_type"] = str(self.window_sample_type[index])
        result["window_index"] = torch.tensor(index, dtype=torch.long)
        result["source_window_index"] = torch.tensor(source_index, dtype=torch.long)
        return result

    def __getattr__(self, name: str) -> Any:
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


class MixedStreamConcatDataset(ConcatDataset):
    """Concat heterogeneous branch datasets while preserving sampler metadata."""

    def __init__(
        self,
        datasets: Sequence[Dataset | Any],
        source_sampling_weights: Sequence[float] | None = None,
        included_source_indices: Sequence[int] | None = None,
        uniform_sampling: bool = False,
    ):
        if not datasets:
            raise ValueError("MixedStreamConcatDataset requires at least one dataset.")
        datasets = list(datasets)
        if included_source_indices is not None:
            source_indices = tuple(int(index) for index in included_source_indices)
            if not source_indices or len(set(source_indices)) != len(source_indices):
                raise ValueError("included_source_indices must be non-empty and unique.")
            if min(source_indices) < 0 or max(source_indices) >= len(datasets):
                raise IndexError(
                    f"included_source_indices={source_indices} outside {len(datasets)} sources."
                )
            datasets = [datasets[index] for index in source_indices]
            if source_sampling_weights is not None:
                source_sampling_weights = [
                    source_sampling_weights[index] for index in source_indices
                ]
        resolved_datasets = []
        for dataset in datasets:
            if not isinstance(dataset, Dataset):
                # With Hydra `_recursive_: false`, select the requested sources
                # before constructing expensive child datasets.
                from hydra.utils import instantiate

                dataset = instantiate(dataset)
            if not isinstance(dataset, Dataset):
                raise TypeError(
                    "MixedStreamConcatDataset children must resolve to Dataset, "
                    f"got {type(dataset)}."
                )
            resolved_datasets.append(dataset)
        super().__init__(resolved_datasets)
        self._concat_sampler_metadata()
        if uniform_sampling:
            self.window_sample_weight = np.ones(len(self), dtype=np.float64)
        else:
            self._apply_source_sampling_weights(source_sampling_weights)
        self.uniform_sampling = bool(uniform_sampling)
        self.included_source_indices = included_source_indices

    def _apply_source_sampling_weights(
        self,
        source_sampling_weights: Sequence[float] | None,
    ) -> None:
        if source_sampling_weights is None:
            return
        source_weights = np.asarray(source_sampling_weights, dtype=np.float64)
        if source_weights.shape != (len(self.datasets),):
            raise ValueError(
                "source_sampling_weights must have one value per child dataset, "
                f"got shape {source_weights.shape} for {len(self.datasets)} datasets."
            )
        if not np.isfinite(source_weights).all() or np.any(source_weights <= 0.0):
            raise ValueError("source_sampling_weights must all be finite and strictly positive.")

        combined = []
        source_ids = []
        for source_id, (dataset, source_weight) in enumerate(
            zip(self.datasets, source_weights, strict=True)
        ):
            internal = getattr(dataset, "window_sample_weight", None)
            if internal is None:
                internal = np.ones(len(dataset), dtype=np.float64)
            internal = np.asarray(internal, dtype=np.float64)
            if internal.shape != (len(dataset),):
                raise ValueError("window_sample_weight metadata length mismatch.")
            if not np.isfinite(internal).all() or np.any(internal <= 0.0):
                raise ValueError("Child window_sample_weight values must be finite and positive.")
            combined.append(internal / internal.sum() * float(source_weight))
            source_ids.append(np.full(len(dataset), source_id, dtype=np.int64))

        self.window_sample_weight = np.concatenate(combined)
        self.window_source_id = np.concatenate(source_ids)
        self.source_sampling_weights = source_weights
        self.source_sampling_probabilities = source_weights / source_weights.sum()

    def _concat_optional(self, name: str, dtype: Any) -> np.ndarray | None:
        values = []
        for dataset in self.datasets:
            value = getattr(dataset, name, None)
            if value is None:
                return None
            value = np.asarray(value, dtype=dtype)
            if value.shape != (len(dataset),):
                raise ValueError(f"{name} metadata must match each child dataset length.")
            values.append(value)
        return np.concatenate(values)

    def _concat_sampler_metadata(self) -> None:
        for name, dtype in (
            ("window_sample_weight", np.float64),
            ("window_sample_type", object),
            ("window_branch", object),
            ("window_start_frame", np.int64),
        ):
            value = self._concat_optional(name, dtype)
            if value is not None:
                setattr(self, name, value)

        episodes = []
        episode_offset = 0
        for dataset in self.datasets:
            value = getattr(dataset, "window_episode_index", None)
            if value is None:
                return
            value = np.asarray(value, dtype=np.int64)
            if value.shape != (len(dataset),):
                raise ValueError("window_episode_index metadata length mismatch.")
            if value.size:
                normalized = value - int(value.min()) + episode_offset
                episode_offset = int(normalized.max()) + 1
            else:
                normalized = value
            episodes.append(normalized)
        self.window_episode_index = np.concatenate(episodes)
