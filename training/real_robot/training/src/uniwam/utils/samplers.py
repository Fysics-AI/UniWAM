from math import ceil
from typing import Iterator, Sequence, Sized

import numpy as np
import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        mode: str = "random",
        local_window: int = 0,
        balanced_groups: Sequence[str] | None = None,
        min_frame_gap: int = 0,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.mode = str(mode)
        self.local_window = int(local_window)
        self.balanced_groups = tuple(
            str(group) for group in (balanced_groups or ("pure_nav", "transition", "pure_manip"))
        )
        self.min_frame_gap = int(min_frame_gap)
        if self.min_frame_gap < 0:
            raise ValueError(f"min_frame_gap must be non-negative, got {self.min_frame_gap}.")
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def _balanced_random_indices(self, generator: torch.Generator) -> list[int]:
        sample_types = getattr(self.dataset, "window_sample_type", None)
        if sample_types is None:
            raise ValueError(
                "sampler_mode=balanced_random requires dataset.window_sample_type."
            )
        sample_types = np.asarray(sample_types, dtype=object)
        dataset_len = len(self.dataset)
        if sample_types.shape != (dataset_len,):
            raise ValueError(
                "dataset.window_sample_type length mismatch: "
                f"expected {dataset_len}, got {sample_types.shape}."
            )
        if not self.balanced_groups:
            raise ValueError("balanced_groups must contain at least one sample type.")

        pools: dict[str, torch.Tensor] = {}
        for group in self.balanced_groups:
            pool = torch.from_numpy(np.flatnonzero(sample_types == group).astype(np.int64))
            if pool.numel() == 0:
                raise ValueError(f"No samples found for balanced group {group!r}.")
            pools[group] = pool

        episode_index = getattr(self.dataset, "window_episode_index", None)
        start_frame = getattr(self.dataset, "window_start_frame", None)
        if self.min_frame_gap > 0:
            if episode_index is None or start_frame is None:
                raise ValueError(
                    "min_frame_gap > 0 requires dataset.window_episode_index and "
                    "dataset.window_start_frame."
                )
            episode_index = np.asarray(episode_index, dtype=np.int64)
            start_frame = np.asarray(start_frame, dtype=np.int64)
            if episode_index.shape != (dataset_len,) or start_frame.shape != (dataset_len,):
                raise ValueError("Window episode/frame metadata length mismatch.")

        orders: dict[str, torch.Tensor] = {}
        cursors = {group: 0 for group in self.balanced_groups}

        def draw(group: str) -> int:
            pool = pools[group]
            cursor = cursors[group]
            if group not in orders or cursor >= pool.numel():
                orders[group] = pool[torch.randperm(pool.numel(), generator=generator)]
                cursor = 0
            index = int(orders[group][cursor].item())
            cursors[group] = cursor + 1
            return index

        def conflicts(
            index: int,
            selected_indices: set[int],
            selected_frames: dict[int, list[int]],
        ) -> bool:
            if index in selected_indices:
                return True
            if self.min_frame_gap <= 0:
                return False
            assert episode_index is not None and start_frame is not None
            episode = int(episode_index[index])
            frame = int(start_frame[index])
            return any(
                abs(other_frame - frame) < self.min_frame_gap
                for other_frame in selected_frames.get(episode, ())
            )

        indices: list[int] = []
        global_batch_size = self.batch_size * self.num_processes
        num_batches = ceil(dataset_len / global_batch_size)
        num_groups = len(self.balanced_groups)
        for batch_idx in range(num_batches):
            batch_size = min(global_batch_size, dataset_len - len(indices))
            base_count, remainder = divmod(batch_size, num_groups)
            group_schedule: list[str] = []
            for group_offset, group in enumerate(self.balanced_groups):
                extra = int((group_offset - batch_idx) % num_groups < remainder)
                group_schedule.extend([group] * (base_count + extra))
            schedule_order = torch.randperm(len(group_schedule), generator=generator).tolist()
            group_schedule = [group_schedule[i] for i in schedule_order]

            batch: list[int] = []
            batch_indices: set[int] = set()
            batch_frames: dict[int, list[int]] = {}
            for group in group_schedule:
                candidate = draw(group)
                max_attempts = max(64, global_batch_size * 8)
                for _ in range(max_attempts):
                    if not conflicts(candidate, batch_indices, batch_frames):
                        break
                    candidate = draw(group)
                if conflicts(candidate, batch_indices, batch_frames):
                    raise RuntimeError(
                        "Unable to construct a non-overlapping balanced batch. "
                        f"group={group!r}, global_batch_size={global_batch_size}, "
                        f"min_frame_gap={self.min_frame_gap}."
                    )
                batch.append(candidate)
                batch_indices.add(candidate)
                if self.min_frame_gap > 0:
                    assert episode_index is not None and start_frame is not None
                    batch_frames.setdefault(int(episode_index[candidate]), []).append(
                        int(start_frame[candidate])
                    )
            indices.extend(batch)
        return indices

    def _weighted_random_indices(self, generator: torch.Generator) -> list[int]:
        raw_weights = getattr(self.dataset, "window_sample_weight", None)
        if raw_weights is None:
            raise ValueError(
                "sampler_mode=weighted_random requires dataset.window_sample_weight."
            )
        weights_np = np.asarray(raw_weights, dtype=np.float64)
        dataset_len = len(self.dataset)
        if weights_np.shape != (dataset_len,):
            raise ValueError(
                f"dataset.window_sample_weight must have shape ({dataset_len},), got {weights_np.shape}."
            )
        if not np.isfinite(weights_np).all() or np.any(weights_np <= 0.0):
            raise ValueError("All weighted-random sampling weights must be finite and positive.")
        weights = torch.from_numpy(weights_np)

        episode_index = getattr(self.dataset, "window_episode_index", None)
        start_frame = getattr(self.dataset, "window_start_frame", None)
        if self.min_frame_gap > 0:
            if episode_index is None or start_frame is None:
                raise ValueError(
                    "min_frame_gap > 0 requires window_episode_index and window_start_frame."
                )
            episode_index = np.asarray(episode_index, dtype=np.int64)
            start_frame = np.asarray(start_frame, dtype=np.int64)

        def conflicts(
            index: int,
            selected_indices: set[int],
            selected_frames: dict[int, list[int]],
        ) -> bool:
            if index in selected_indices:
                return True
            if self.min_frame_gap <= 0:
                return False
            assert episode_index is not None and start_frame is not None
            episode = int(episode_index[index])
            frame = int(start_frame[index])
            return any(
                abs(other_frame - frame) < self.min_frame_gap
                for other_frame in selected_frames.get(episode, ())
            )

        # Generate a large weighted candidate stream once. Replacement is intentional:
        # event windows must appear more often, rather than merely earlier in an epoch.
        candidate_count = max(dataset_len * 2, dataset_len + 4096)
        candidates = torch.multinomial(
            weights,
            num_samples=candidate_count,
            replacement=True,
            generator=generator,
        ).tolist()
        candidate_cursor = 0

        def next_candidate() -> int:
            nonlocal candidates, candidate_cursor
            if candidate_cursor >= len(candidates):
                candidates = torch.multinomial(
                    weights,
                    num_samples=max(dataset_len, 4096),
                    replacement=True,
                    generator=generator,
                ).tolist()
                candidate_cursor = 0
            result = int(candidates[candidate_cursor])
            candidate_cursor += 1
            return result

        def balance_local_streams(batch: list[int]) -> None:
            """Distribute nav/manip samples evenly across local rank batches."""
            if self.num_processes <= 1 or len(batch) < self.num_processes:
                return
            branches = np.asarray(getattr(self.dataset, "window_branch", ()), dtype=object)
            sample_types = np.asarray(getattr(self.dataset, "window_sample_type", ()), dtype=object)
            if branches.shape != (len(self.dataset),) and sample_types.shape != (len(self.dataset),):
                return
            def is_nav(index: int) -> bool:
                if branches.shape == (len(self.dataset),):
                    return str(branches[index]) == "nav"
                return str(sample_types[index]).startswith("nav_")
            parts = [(offset, batch[offset : offset + self.batch_size]) for offset in range(0, len(batch), self.batch_size)]
            counts = [sum(is_nav(index) for index in part) for _, part in parts]
            total_nav = sum(counts)
            if not total_nav or len(parts) < 2:
                return
            target = [total_nav // len(parts) + int(i < total_nav % len(parts)) for i in range(len(parts))]
            while True:
                receiver = min(range(len(counts)), key=lambda i: counts[i] - target[i])
                donor = max(range(len(counts)), key=lambda i: counts[i] - target[i])
                if counts[receiver] >= target[receiver] or counts[donor] <= target[donor]:
                    return
                ro, rpart = parts[receiver]
                do, dpart = parts[donor]
                dp = next(i for i, value in enumerate(dpart) if is_nav(value))
                rp = next(i for i, value in enumerate(rpart) if not is_nav(value))
                batch[do + dp], batch[ro + rp] = batch[ro + rp], batch[do + dp]
                dpart[dp], rpart[rp] = rpart[rp], dpart[dp]
                counts[donor] -= 1
                counts[receiver] += 1

        indices: list[int] = []
        global_batch_size = self.batch_size * self.num_processes
        while len(indices) < dataset_len:
            target_batch_size = min(global_batch_size, dataset_len - len(indices))
            batch: list[int] = []
            batch_indices: set[int] = set()
            batch_frames: dict[int, list[int]] = {}
            for _ in range(target_batch_size):
                candidate = next_candidate()
                max_attempts = max(128, global_batch_size * 16)
                for _ in range(max_attempts):
                    if not conflicts(candidate, batch_indices, batch_frames):
                        break
                    candidate = next_candidate()
                if conflicts(candidate, batch_indices, batch_frames):
                    raise RuntimeError(
                        "Unable to construct a non-overlapping weighted batch: "
                        f"global_batch_size={global_batch_size}, "
                        f"min_frame_gap={self.min_frame_gap}."
                    )
                batch.append(candidate)
                batch_indices.add(candidate)
                if self.min_frame_gap > 0:
                    assert episode_index is not None and start_frame is not None
                    batch_frames.setdefault(int(episode_index[candidate]), []).append(
                        int(start_frame[candidate])
                    )
            balance_local_streams(batch)
            indices.extend(batch)
        return indices

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        dataset_len = len(self.dataset)
        if self.mode in ("random", "shuffle"):
            indices = torch.randperm(dataset_len, generator=g).tolist()
        elif self.mode in ("balanced_random", "stratified_random"):
            indices = self._balanced_random_indices(g)
        elif self.mode in ("weighted_random", "event_weighted_random"):
            indices = self._weighted_random_indices(g)
        elif self.mode in ("local_window", "window"):
            window = max(int(self.local_window), self.batch_size * self.num_processes)
            starts = list(range(0, dataset_len, window))
            order = torch.randperm(len(starts), generator=g).tolist()
            indices = []
            for order_idx in order:
                start = starts[order_idx]
                end = min(start + window, dataset_len)
                indices.extend(range(start, end))
        else:
            raise ValueError(f"Unsupported sampler mode: {self.mode}")
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)
