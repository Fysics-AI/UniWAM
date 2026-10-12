"""Customer single-source data and train/deploy contract fixtures."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
sys.path[:0] = [str(TRAINING / "data_pipeline"), str(TRAINING / "src"),
                str(ROOT / "deployment/cloud"), str(ROOT / "deployment/common"),
                str(ROOT / "deployment/scripts")]

from prepare_customer_piper_h32 import prepare  # noqa: E402
from prepare_customer_piper_views import camera_state  # noqa: E402
from prepare_agilex_camera_views import legacy_camera_state  # noqa: E402
from make_customer_piper_robot_config import generate  # noqa: E402
from uniwam_piper_cloud.policy_rot6d import (  # noqa: E402
    AsyncPrefixContract, load_runtime_cfg, validate_checkpoint_contract,
)


def _arm(length: int) -> list[list[float]]:
    pose = [0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.05]
    return [pose + pose for _ in range(length)]


class CustomerPiperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="uniwam_customer_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "customer_source"
        (self.data / "meta").mkdir(parents=True)
        episode = self.data / "data/chunk-000/episode_000000.parquet"
        episode.parent.mkdir(parents=True)
        raw = _arm(40)
        for pose in raw:
            pose[10] -= 0.3
        pq.write_table(pa.table({
            "episode_index": [0] * 40,
            "frame_index": list(range(40)),
            "observation.state": pa.array(raw, type=pa.list_(pa.float32(), 20)),
            "action.manip": pa.array(raw, type=pa.list_(pa.float32(), 20)),
            "observation.state.camera_dual_arm": pa.array(_arm(40), type=pa.list_(pa.float32(), 20)),
            "action.manip.camera_dual_arm": pa.array(_arm(40), type=pa.list_(pa.float32(), 20)),
        }), episode)
        left = np.eye(4).tolist()
        right = np.eye(4).tolist()
        right[0][3] = 0.3
        self.calibration = self.root / "calibration.json"
        self.calibration.write_text(json.dumps({
            "camera_from_left_base": left, "camera_from_right_base": right,
        }))
        (self.data / "meta/info.json").write_text(json.dumps({
            "features": {name: {} for name in (
                "observation.state.camera_dual_arm", "action.manip.camera_dual_arm",
                "observation.images.cam_manip_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            )},
            "fastwam_camera_frame_conversion": {
                "camera_from_left_base": left, "camera_from_right_base": right,
                "rotation_serialization": "row_major_rot6d_v1",
                "input_frame": "base",
                "input_state_dim": 20,
            },
        }))
        (self.data / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 40}) + "\n")
        (self.data / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "Move the red block."}) + "\n")

    def _raw_source(self, name: str, state: list[list[float]], action: list[list[float]]) -> Path:
        raw = self.root / name
        (raw / "meta").mkdir(parents=True)
        (raw / "videos").mkdir()
        for filename in ("tasks.jsonl", "episodes.jsonl"):
            (raw / "meta" / filename).write_text((self.data / "meta" / filename).read_text())
        info = json.loads((self.data / "meta/info.json").read_text())
        info["features"]["observation.state"] = {"dtype": "float32", "shape": [len(state[0])]}
        info["features"]["action.manip"] = {"dtype": "float32", "shape": [20]}
        (raw / "meta/info.json").write_text(json.dumps(info))
        source_file = raw / "data/chunk-000/episode_000000.parquet"
        source_file.parent.mkdir(parents=True)
        pq.write_table(pa.table({
            "episode_index": [0] * 40,
            "frame_index": list(range(40)),
            "observation.state": pa.array(state, type=pa.list_(pa.float32(), len(state[0]))),
            "action.manip": pa.array(action, type=pa.list_(pa.float32(), 20)),
        }), source_file)
        return raw

    def _convert_cli(self, source: Path, output: Path, *extra: str) -> None:
        subprocess.run([
            sys.executable, str(TRAINING / "data_pipeline/prepare_customer_piper_views.py"),
            "--source-root", str(source), "--output-root", str(output),
            "--calibration", str(self.calibration), *extra,
        ], check=True, capture_output=True, text=True)

    def test_h32_windows_and_calibrated_robot_config(self) -> None:
        output = self.root / "window.parquet"
        result = prepare(self.data, self.calibration, output)
        self.assertEqual(result["windows"], 8)
        self.assertEqual(result["model_state_dim"], 20)
        table = pq.read_table(output)
        self.assertEqual(table["start_frame"].to_pylist(), list(range(8)))
        self.assertEqual(table["manip_loss_valid"].to_pylist(), [True] * 8)
        robot = self.root / "robot.yaml"
        generate(ROOT / "deployment/configs/uniwam_robot_client_camera_frame_h32.yaml",
                 self.data, self.calibration, robot)
        cfg = OmegaConf.load(robot)
        self.assertEqual(cfg.camera_frame.camera_from_right_base[0][3], 0.3)
        self.assertTrue(cfg.control.dry_run)
        self.assertFalse(cfg.base.enabled)
        self.assertEqual(cfg.prefix_steps, 12)
        self.assertEqual(cfg.min_prefix_steps, 6)
        self.assertTrue(cfg.adaptive_prefix.enabled)
        self.assertTrue(cfg.replan_while_queue_nonempty)
        self.assertEqual(cfg.chunk_low_watermark, 13)

    def test_raw_to_camera_to_h32_and_rotation_convention(self) -> None:
        source_table = pq.read_table(self.data / "data/chunk-000/episode_000000.parquet")
        raw_state = source_table["observation.state"].to_pylist()
        raw_action = source_table["action.manip"].to_pylist()
        raw = self._raw_source("raw", raw_state, raw_action)
        calibration = json.loads(self.calibration.read_text())
        left = np.asarray(calibration["camera_from_left_base"])
        right = np.asarray(calibration["camera_from_right_base"])
        pose = np.asarray(_arm(1), dtype=np.float32)
        self.assertTrue(np.allclose(camera_state(pose, left, right)[0, 3:9], [1, 0, 0, 0, 1, 0]))
        self.assertFalse(np.allclose(legacy_camera_state(pose, left, right)[0, 3:9], [1, 0, 0, 0, 1, 0]))
        prepared = self.root / "prepared"
        self._convert_cli(raw, prepared)
        result = prepare(prepared, self.calibration, self.root / "prepared_h32.parquet")
        self.assertEqual(result["windows"], 8)

    def test_camera_frame_20d_and_legacy_23d_inputs(self) -> None:
        source_table = pq.read_table(self.data / "data/chunk-000/episode_000000.parquet")
        calibration = json.loads(self.calibration.read_text())
        left = np.asarray(calibration["camera_from_left_base"])
        right = np.asarray(calibration["camera_from_right_base"])

        camera_state_values = source_table["observation.state.camera_dual_arm"].to_pylist()
        camera_action_values = source_table["action.manip.camera_dual_arm"].to_pylist()
        camera_raw = self._raw_source("camera_raw", camera_state_values, camera_action_values)
        camera_prepared = self.root / "camera_prepared"
        self._convert_cli(camera_raw, camera_prepared, "--input-frame", "camera")
        camera_result = prepare(camera_prepared, self.calibration, self.root / "camera_h32.parquet")
        self.assertEqual(camera_result["input_frame"], "camera")
        self.assertEqual(camera_result["model_state_dim"], 20)
        copied = pq.read_table(camera_prepared / "data/chunk-000/episode_000000.parquet")
        self.assertEqual(copied["observation.state.camera_dual_arm"].to_pylist(), camera_state_values)

        arm_state_values = source_table["observation.state"].to_pylist()
        base_state_values = [[0.0, 0.0, 0.0] + row for row in arm_state_values]
        base_raw = self._raw_source("base3_raw", base_state_values, source_table["action.manip"].to_pylist())
        base_prepared = self.root / "base3_prepared"
        self._convert_cli(base_raw, base_prepared, "--raw-state-layout", "base3_arm20")
        base_result = prepare(base_prepared, self.calibration, self.root / "base3_h32.parquet")
        self.assertEqual(base_result["model_state_dim"], 20)

    def test_rejects_wrong_calibration_and_action_units(self) -> None:
        other = json.loads(self.calibration.read_text())
        other["camera_from_left_base"][0][3] = 0.1
        wrong = self.root / "wrong.json"
        wrong.write_text(json.dumps(other))
        with self.assertRaisesRegex(ValueError, "differs"):
            prepare(self.data, wrong, self.root / "window.parquet")
        episode = self.data / "data/chunk-000/episode_000000.parquet"
        table = pq.read_table(episode)
        bad = _arm(40)
        bad[0][9] = 1.0
        column = table.schema.get_field_index("action.manip.camera_dual_arm")
        pq.write_table(table.set_column(column, table.schema.field(column),
                                        pa.array(bad, type=pa.list_(pa.float32(), 20))), episode)
        with self.assertRaisesRegex(ValueError, "grippers must be in meters"):
            prepare(self.data, self.calibration, self.root / "window.parquet")

    def test_customer_checkpoint_contract_and_mismatches(self) -> None:
        environment = {
            "UNIWAM_TRAINING_ROOT": str(TRAINING),
            "UNIWAM_CUSTOM_DATA": str(self.data),
            "UNIWAM_CUSTOM_WINDOW_INDEX": str(self.root / "window.parquet"),
            "UNIWAM_CUSTOM_LATENT_ROOT": str(self.root / "latents"),
            "UNIWAM_CUSTOM_SOURCE_NAME": self.data.name,
            "UNIWAM_TEXT_CACHE_DIR": str(self.root / "text"),
            "UNIWAM_PARENT_CHECKPOINT": str(self.root / "parent.pt"),
            "DIFFSYNTH_MODEL_BASE_PATH": str(self.root / "weights"),
            "UNIWAM_CLOUD_TASK": "uniwam_customer_piper_manip_sft",
            "UNIWAM_CLOUD_SOURCE_INDEX": "0",
            "UNIWAM_CLOUD_AUX_WEIGHT": "0.0",
        }
        with patch.dict(os.environ, environment):
            runtime = load_runtime_cfg(TRAINING, "uniwam_customer_piper_manip_sft")
            self.assertEqual(int(runtime.model.proprio_dim), 20)
            self.assertTrue(runtime.data.train.datasets[0].dataset.append_missing_eef_xy_slots)
            self.assertTrue(runtime.model.disable_manip_aux_training)
            self.assertEqual(float(runtime.model.loss.lambda_manip_aux_action), 0.0)
            cfg = OmegaConf.load(ROOT / "deployment/configs/uniwam_cloud_parent_async.yaml")
            cfg.checkpoint_min_bytes = 0
            checkpoint = self.root / "run/checkpoints/weights/step_000001.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fixture")
            OmegaConf.save(runtime, self.root / "run/config.yaml", resolve=True)
            contract = AsyncPrefixContract.from_deploy_cfg(cfg)
            self.assertEqual(validate_checkpoint_contract(cfg, runtime, checkpoint, contract),
                             self.root / "run/config.yaml")
            cfg.instruction_prefix_by_mode.manip_only = "Different prefix."
            with self.assertRaisesRegex(ValueError, "manip_instruction_prefix"):
                validate_checkpoint_contract(cfg, runtime, checkpoint, contract)
            cfg.instruction_prefix_by_mode.manip_only = runtime.data.train.datasets[0].dataset.instruction_prefix
            cfg.dataset_stats = str(self.root / "wrong_stats.json")
            payload = json.loads((TRAINING / "data_indices/camera_frame_piper_agx_only_q01q99.json").read_text())
            payload["state"]["default"]["global_q99"][0] += 0.1
            Path(cfg.dataset_stats).write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "dataset_stats"):
                validate_checkpoint_contract(cfg, runtime, checkpoint, contract)

    def test_six_source_cloud_contracts_still_match(self) -> None:
        environment = {
            "UNIWAM_TRAINING_ROOT": str(TRAINING),
            "UNIWAM_EEF_SIDECAR_ROOT": str(self.root / "sidecars"),
            "UNIWAM_FRANKA_DATA": str(self.root / "franka"),
            "UNIWAM_CUP_TRAY_DATA": str(self.root / "cup"),
            "UNIWAM_MOVE_WHITE_DATA": str(self.root / "white"),
            "UNIWAM_COLOR_DATA": str(self.root / "color"),
            "UNIWAM_ORDERED_COLOR_DATA": str(self.root / "ordered"),
            "UNIWAM_7000_DATA": str(self.root / "7000"),
            "UNIWAM_FRANKA_LATENT_ROOT": str(self.root / "franka_latents"),
            "UNIWAM_MOBILE_LATENT_ROOT": str(self.root / "mobile_latents"),
            "UNIWAM_ORDERED_LATENT_ROOT": str(self.root / "ordered_latents"),
            "UNIWAM_7000_LATENT_ROOT": str(self.root / "7000_latents"),
            "UNIWAM_TEXT_CACHE_DIR": str(self.root / "text"),
            "UNIWAM_ORDERED_PROMPT_VARIANTS": str(self.root / "ordered_prompts.jsonl"),
            "UNIWAM_7000_PROMPT_VARIANTS": str(self.root / "7000_prompts.jsonl"),
            "DIFFSYNTH_MODEL_BASE_PATH": str(self.root / "weights"),
        }
        with patch.dict(os.environ, environment):
            runtime = load_runtime_cfg(
                TRAINING, "uniwam_camera_frame_six_source_manip26_embodiment_stats_200k"
            )
            checkpoint = self.root / "parent/checkpoints/weights/step_200000.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fixture")
            OmegaConf.save(runtime, self.root / "parent/config.yaml", resolve=True)
            for name in (
                "uniwam_source1_paired.yaml",
                "uniwam_source3_color_manip_only.yaml",
                "uniwam_source4_ordered_color_manip_only.yaml",
            ):
                with self.subTest(config=name):
                    cfg = OmegaConf.load(ROOT / "deployment/configs" / name)
                    cfg.checkpoint_min_bytes = 0
                    contract = AsyncPrefixContract.from_deploy_cfg(cfg)
                    self.assertEqual(
                        validate_checkpoint_contract(cfg, runtime, checkpoint, contract),
                        self.root / "parent/config.yaml",
                    )


if __name__ == "__main__":
    unittest.main()
