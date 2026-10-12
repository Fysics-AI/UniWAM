from typing import Iterator, Sized

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
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.mode = str(mode)
        self.local_window = int(local_window)
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

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        dataset_len = len(self.dataset)
        if self.mode in ("random", "shuffle"):
            indices = torch.randperm(dataset_len, generator=g).tolist()
        elif self.mode in ("local_window", "window"):
            window = max(int(self.local_window), self.batch_size * self.num_processes)
            num_windows = (dataset_len + window - 1) // window
            order = torch.randperm(num_windows, generator=g).tolist()
            sample_offset = 0
            if self.epoch == 0 and self.resume_batch_offset > 0:
                sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes

            def iter_local_windows() -> Iterator[int]:
                skipped = 0
                for order_idx in order:
                    start = order_idx * window
                    end = min(start + window, dataset_len)
                    if skipped + (end - start) <= sample_offset:
                        skipped += end - start
                        continue
                    begin = start + max(0, sample_offset - skipped)
                    skipped += end - start
                    yield from range(begin, end)

            return iter_local_windows()
        else:
            raise ValueError(f"Unsupported sampler mode: {self.mode}")
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)
