"""Regression coverage for training cleanup, startup reads and checkpoint cadence."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import packaging.version
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uniwam import runtime
from uniwam.datasets.lerobot.lerobot import lerobot_dataset
from uniwam.trainer import Wan22Trainer


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_loss(self, sample):
        return self.weight.square(), {}

    def save_checkpoint(self, path, optimizer, step):
        torch.save({"step": step, "model": self.state_dict()}, path)


class TrainingLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="uniwam_lifecycle_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _runtime_config(self):
        return OmegaConf.create({
            "output_dir": str(self.root),
            "seed": 0,
            "mixed_precision": "no",
            "model": {},
            "data": {},
        })

    def _run_runtime(self, trainer):
        with patch.object(runtime, "instantiate", return_value=object()), \
             patch.object(runtime, "build_datasets", return_value=(object(), None)), \
             patch.object(runtime, "Wan22Trainer", return_value=trainer), \
             patch.object(runtime, "set_global_seed"), \
             patch.object(runtime, "setup_logging"), \
             patch.object(runtime.misc, "register_work_dir"):
            runtime.run_training(self._runtime_config())

    def test_runtime_marks_normal_cleanup(self):
        trainer = Mock()
        self._run_runtime(trainer)
        trainer.train.assert_called_once_with()
        trainer.close.assert_called_once_with()

    def test_runtime_marks_failures_and_preserves_original_exception(self):
        for error in (RuntimeError("rank failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                trainer = Mock()
                trainer.train.side_effect = error
                with self.assertRaises(type(error)) as raised:
                    self._run_runtime(trainer)
                self.assertIs(raised.exception, error)
                trainer.close.assert_called_once_with(failed=True)

    def test_secondary_cleanup_error_does_not_replace_training_error(self):
        error = RuntimeError("original training error")
        trainer = Mock()
        trainer.train.side_effect = error
        trainer.close.side_effect = RuntimeError("cleanup error")
        with patch.object(runtime.logger, "exception"), \
             self.assertRaises(RuntimeError) as raised:
            self._run_runtime(trainer)
        self.assertIs(raised.exception, error)

    def _closing_trainer(self):
        trainer = Wan22Trainer.__new__(Wan22Trainer)
        trainer._finish_wandb = Mock()
        trainer.train_loader = SimpleNamespace(
            _iterator=SimpleNamespace(_shutdown_workers=Mock()),
        )
        trainer.accelerator = Mock()
        return trainer

    def test_failed_cleanup_stops_workers_without_collectives_or_group_teardown(self):
        trainer = self._closing_trainer()
        shutdown = trainer.train_loader._iterator._shutdown_workers
        trainer.close(failed=True)
        trainer._finish_wandb.assert_called_once_with()
        shutdown.assert_called_once_with()
        self.assertIsNone(trainer.train_loader._iterator)
        trainer.accelerator.wait_for_everyone.assert_not_called()
        trainer.accelerator.end_training.assert_not_called()

    def test_failed_local_cleanup_still_attempts_remaining_resources(self):
        trainer = self._closing_trainer()
        trainer._finish_wandb.side_effect = RuntimeError("tracking cleanup failed")
        shutdown = trainer.train_loader._iterator._shutdown_workers
        shutdown.side_effect = RuntimeError("worker cleanup failed")
        with patch("uniwam.trainer.logger.exception") as log:
            trainer.close(failed=True)
        shutdown.assert_called_once_with()
        self.assertIsNone(trainer.train_loader._iterator)
        self.assertEqual(log.call_count, 2)
        trainer.accelerator.wait_for_everyone.assert_not_called()
        trainer.accelerator.end_training.assert_not_called()

    def test_normal_cleanup_retains_synchronization_and_teardown(self):
        trainer = self._closing_trainer()
        trainer.close()
        trainer.accelerator.wait_for_everyone.assert_called_once_with()
        trainer.accelerator.end_training.assert_called_once_with()

    @unittest.skipUnless(torch.distributed.is_gloo_available(), "Gloo is unavailable")
    def test_one_rank_failure_exits_before_distributed_timeout(self):
        started = time.monotonic()
        process = subprocess.run(
            [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
             "--nnodes=1", "--nproc-per-node=2", str(Path(__file__).resolve()),
             "--distributed-failure-probe", str(self.root)],
            capture_output=True, text=True, timeout=45,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                 "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("FAILED_RANK_CLEANUP_RETURNED", process.stdout)
        self.assertIn("injected single-rank training failure", process.stderr)
        self.assertLess(time.monotonic() - started, 45)

    def _toy_trainer(self, max_steps, save_every, save_state=True):
        trainer = Wan22Trainer.__new__(Wan22Trainer)
        trainer.model = ToyPolicy()
        trainer.max_steps = max_steps
        trainer.global_step = 0
        trainer.epoch = 0
        trainer.batch_in_epoch = 0
        trainer.max_grad_norm = 1.0
        trainer.log_every = 0
        trainer.eval_every = 0
        trainer.val_dataset = None
        trainer.save_every = save_every
        trainer.save_training_state = save_state
        trainer.weights_dir = str(self.root / "weights")
        trainer.state_dir = str(self.root / "state")
        Path(trainer.weights_dir).mkdir()
        Path(trainer.state_dir).mkdir()
        trainer.train_loader = [{} for _ in range(max_steps)]
        trainer.train_sampler = Mock()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        trainer.scheduler = Mock()
        trainer._set_dit_only_train_mode = Mock()
        trainer._assert_finite_loss = Mock()
        trainer._gradient_guard_active = Mock(return_value=False)
        trainer._deepspeed_optimizer_guard_installed = False
        trainer.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            unwrap_model=lambda model: model,
            accumulate=lambda model: contextlib.nullcontext(),
            autocast=contextlib.nullcontext,
            backward=lambda loss: loss.backward(),
            sync_gradients=True,
            clip_grad_norm_=torch.nn.utils.clip_grad_norm_,
            optimizer_step_was_skipped=False,
            gather=lambda tensor: tensor,
            reduce=lambda tensor, reduction: tensor,
            is_main_process=True,
            wait_for_everyone=Mock(),
            save_state=Mock(),
        )
        trainer.save_checkpoint = Mock(wraps=trainer.save_checkpoint)
        trainer.model.save_checkpoint = Mock(wraps=trainer.model.save_checkpoint)
        return trainer

    def _train_and_check(self, max_steps, save_every, steps, save_state=True, stop=None):
        trainer = self._toy_trainer(max_steps, save_every, save_state)
        environment = {"FASTWAM_PROFILE_TRAIN": "0"}
        environment["FASTWAM_DIAGNOSTIC_STOP_STEP"] = str(stop or max_steps)
        with patch.dict(os.environ, environment), patch("uniwam.trainer.logger.info"):
            trainer.train()
        self.assertEqual(trainer.global_step, stop or max_steps)
        self.assertEqual(trainer.save_checkpoint.call_count, len(steps))
        self.assertEqual(trainer.model.save_checkpoint.call_count, len(steps))
        self.assertEqual(
            [call.kwargs["step"] for call in trainer.model.save_checkpoint.call_args_list],
            steps,
        )
        self.assertEqual(trainer.accelerator.save_state.call_count, len(steps) if save_state else 0)
        for step in steps:
            tag = f"step_{step:06d}"
            checkpoint = torch.load(self.root / "weights" / f"{tag}.pt", weights_only=True)
            self.assertEqual(checkpoint["step"], step)
            if save_state:
                self.assertTrue((self.root / "state" / tag / "trainer_state.json").is_file())

    def test_final_and_periodic_overlap_save_weights_and_state_once(self):
        self._train_and_check(4, 2, [2, 4])

    def test_final_step_outside_periodic_cadence_is_saved(self):
        self._train_and_check(3, 2, [2, 3])

    def test_disabled_periodic_saving_still_saves_final_step(self):
        self._train_and_check(3, 0, [3])

    def test_weights_only_final_overlap_is_saved_once(self):
        self._train_and_check(4, 2, [2, 4], save_state=False)

    def test_diagnostic_stop_keeps_periodic_cadence_without_final_save(self):
        self._train_and_check(4, 2, [2], stop=3)

    def test_dataset_startup_does_not_materialize_timestamp_or_episode_columns(self):
        class NoColumnReads:
            def __getitem__(self, key):
                raise AssertionError(f"Unexpected full-column read during startup: {key}")

        metadata = SimpleNamespace(
            episodes={0: {"episode_index": 0, "length": 5}},
            fps=30,
            _version=packaging.version.parse("v2.0"),
        )
        table = NoColumnReads()
        with patch.object(lerobot_dataset, "LeRobotDatasetMetadata", return_value=metadata), \
             patch.object(lerobot_dataset.LeRobotDataset, "get_episodes_file_paths", return_value=[]), \
             patch.object(lerobot_dataset.LeRobotDataset, "load_hf_dataset", return_value=table):
            dataset = lerobot_dataset.LeRobotDataset(
                "fixture", root=self.root, video_backend="pyav",
                delta_timestamps={"action": [0.0, 1.0 / 30]},
            )
        self.assertIs(dataset.hf_dataset, table)
        self.assertEqual(dataset.delta_indices, {"action": [0, 1]})
        self.assertEqual(dataset.episode_data_index["from"].tolist(), [0])
        self.assertEqual(dataset.episode_data_index["to"].tolist(), [5])


def _distributed_failure_probe():
    import threading
    from datetime import timedelta

    torch.distributed.init_process_group("gloo", timeout=timedelta(seconds=90))
    rank = torch.distributed.get_rank()
    torch.distributed.barrier()
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.wandb_run = None
    trainer.train_loader = None
    trainer.accelerator = SimpleNamespace(
        wait_for_everyone=torch.distributed.barrier,
        end_training=torch.distributed.destroy_process_group,
    )
    if rank == 0:
        trainer.train = Mock(side_effect=RuntimeError("injected single-rank training failure"))
    else:
        trainer.train = lambda: threading.Event().wait(90)
    case = TrainingLifecycleTest()
    case.root = Path(sys.argv[2]) / f"rank_{rank}"
    case.root.mkdir()
    try:
        case._run_runtime(trainer)
    except RuntimeError:
        print("FAILED_RANK_CLEANUP_RETURNED", flush=True)
        raise


if __name__ == "__main__":
    if "--distributed-failure-probe" in sys.argv:
        _distributed_failure_probe()
    else:
        unittest.main()
