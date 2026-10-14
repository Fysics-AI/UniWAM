"""Capacity-aware stream balancing, sampling and resume regression tests."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uniwam.utils.samplers import ResumableEpochSampler


class Windows:
    def __init__(self, count, nav_count):
        self.window_branch = np.array(
            ["nav"] * nav_count + ["manip"] * (count - nav_count), dtype=object,
        )
        self.window_sample_weight = np.ones(count)
        self.window_episode_index = np.zeros(count, dtype=np.int64)
        self.window_start_frame = np.arange(count, dtype=np.int64) * 10

    def __len__(self):
        return len(self.window_sample_weight)


class WeightedSamplerTest(unittest.TestCase):
    def test_partial_batch_all_nav_and_high_nav_reproductions(self):
        for nav_count in (14, 12):
            with self.subTest(nav_count=nav_count):
                sampler = ResumableEpochSampler(
                    Windows(14, nav_count), seed=0, batch_size=4, num_processes=2,
                    mode="weighted_random",
                )
                indices = list(sampler)
                self.assertEqual(len(indices), 14)
                for start in (0, 8):
                    batch = indices[start:start + 8]
                    self.assertEqual(len(set(batch)), len(batch))

    def test_all_tail_sizes_and_nav_counts_preserve_batch_membership(self):
        configurations = 0
        for processes in (2, 3, 4):
            for batch_size in (2, 4):
                global_batch = batch_size * processes
                for length in range(global_batch + 1, global_batch * 2 + 1):
                    for nav_count in range(length + 1):
                        dataset = Windows(length, nav_count)
                        sampler = ResumableEpochSampler(
                            dataset, seed=0, batch_size=batch_size,
                            num_processes=processes, mode="weighted_random", min_frame_gap=10,
                        )
                        candidates = torch.arange(length).repeat(300)
                        with patch("uniwam.utils.samplers.torch.multinomial", return_value=candidates):
                            indices = list(sampler)
                        self.assertEqual(len(indices), length)
                        for start in range(0, length, global_batch):
                            batch = indices[start:start + global_batch]
                            self.assertEqual(sorted(batch), list(range(start, min(start + global_batch, length))))
                            if len(batch) < processes:
                                continue
                            parts = [batch[i:i + batch_size] for i in range(0, len(batch), batch_size)]
                            counts = [sum(dataset.window_branch[index] == "nav" for index in part) for part in parts]
                            for i, count in enumerate(counts):
                                if count + 1 < max(counts):
                                    self.assertEqual(count, len(parts[i]))
                        configurations += 1
        self.assertGreater(configurations, 600)

    def test_sample_type_fallback_and_missing_branch_metadata(self):
        original = Windows(14, 12)
        dataset = SimpleNamespace(
            window_sample_weight=original.window_sample_weight,
            window_sample_type=np.array(["nav_pure"] * 12 + ["manip_pure"] * 2),
        )
        class TypedWindows:
            window_sample_weight = dataset.window_sample_weight
            window_sample_type = dataset.window_sample_type
            def __len__(self):
                return 14
        sampler = ResumableEpochSampler(
            TypedWindows(), seed=0, batch_size=4, num_processes=2, mode="weighted_random",
        )
        self.assertEqual(len(list(sampler)), 14)

    def test_seeds_epochs_and_resume_offsets_are_deterministic(self):
        sampler = ResumableEpochSampler(
            Windows(38, 35), seed=12, batch_size=4, num_processes=2,
            mode="weighted_random",
        )
        original = list(sampler)
        self.assertEqual(list(sampler), original)
        sampler.set_resume_batch_offset(2)
        self.assertEqual(list(sampler), original[16:])
        sampler.clear_resume_batch_offset()
        sampler.set_epoch(1)
        next_epoch = list(sampler)
        self.assertEqual(list(sampler), next_epoch)
        self.assertNotEqual(original, next_epoch)
        sampler.set_epoch(0)
        sampler.set_epoch_offset(1)
        self.assertEqual(list(sampler), next_epoch)

    def test_random_mode_matches_original_permutation(self):
        sampler = ResumableEpochSampler(
            Windows(14, 12), seed=5, batch_size=4, num_processes=2, mode="random",
        )
        expected = torch.randperm(14, generator=torch.Generator().manual_seed(5)).tolist()
        self.assertEqual(list(sampler), expected)


if __name__ == "__main__":
    unittest.main()
