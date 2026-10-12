"""Guard the exact six-source parent recipe and mobile Rot6D conversion."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_agilex_camera_views import camera_state, legacy_camera_state  # noqa: E402
from preflight_parent_reproduction import (  # noqa: E402
    DATASET_NAMES, check_recipe, check_recomputed_stats,
)


class ParentReproductionTest(unittest.TestCase):
    def test_corrected_camera_rotation_is_row_major(self) -> None:
        raw = np.asarray([[0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.05] * 2], dtype=np.float32)
        left = np.eye(4)
        left[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        right = np.eye(4)
        corrected = camera_state(raw, left, right)
        np.testing.assert_allclose(corrected[0, 3:9], [0, -1, 0, 1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(corrected[0, 13:19], [1, 0, 0, 0, 1, 0], atol=1e-6)
        legacy = legacy_camera_state(raw, left, right)
        self.assertFalse(np.allclose(legacy[0, 3:9], corrected[0, 3:9]))

    def test_parent_recipe_rejects_newer_mobile_source(self) -> None:
        names = DATASET_NAMES
        environment = {
            "UNIWAM_TRAINING_ROOT": str(ROOT),
            "UNIWAM_EEF_SIDECAR_ROOT": "/data/sidecars",
            "UNIWAM_FRANKA_DATA": "/data/" + names[0],
            "UNIWAM_CUP_TRAY_DATA": "/data/" + names[1],
            "UNIWAM_MOVE_WHITE_DATA": "/data/" + names[2],
            "UNIWAM_COLOR_DATA": "/data/" + names[3],
            "UNIWAM_ORDERED_COLOR_DATA": "/data/" + names[4],
            "UNIWAM_7000_DATA": "/data/" + names[5],
            "UNIWAM_FRANKA_LATENT_ROOT": "/data/franka_latents",
            "UNIWAM_MOBILE_LATENT_ROOT": "/data/mobile_latents",
            "UNIWAM_ORDERED_LATENT_ROOT": "/data/ordered_latents",
            "UNIWAM_7000_LATENT_ROOT": "/data/7000_latents",
            "UNIWAM_TEXT_CACHE_DIR": "/data/text",
            "UNIWAM_ORDERED_PROMPT_VARIANTS": "/data/ordered_prompts.jsonl",
            "UNIWAM_7000_PROMPT_VARIANTS": "/data/7000_prompts.jsonl",
            "DIFFSYNTH_MODEL_BASE_PATH": "/data/weights",
        }
        reference = json.loads((ROOT / "data_pipeline/parent_200k_artifact_reference.json").read_text())
        with patch.dict(os.environ, environment):
            with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
                cfg = compose(config_name="train", overrides=[
                    "task=uniwam_camera_frame_six_source_manip26_embodiment_stats_200k"
                ])
            OmegaConf.resolve(cfg)
            selected = check_recipe(cfg, reference)
            self.assertEqual(len(selected), 6)
            with tempfile.TemporaryDirectory(prefix="uniwam_parent_stats_test_") as temporary:
                path = Path(temporary) / "recomputed.json"
                stats = json.loads((ROOT / "data_indices/camera_frame_piper_agx_only_q01q99.json").read_text())
                stats["audit"] = [
                    {"root": str(row["dataset_root"]),
                     "window_index": str(row["phase_index"] or row["window_index"])}
                    for row in selected[1:]
                ]
                path.write_text(json.dumps(stats))
                check_recomputed_stats(path, selected)
                stats["state"]["default"]["global_mean"][0] += 0.01
                path.write_text(json.dumps(stats))
                with self.assertRaisesRegex(ValueError, "Piper state stats differ"):
                    check_recomputed_stats(path, selected)
            cfg.data.train.datasets[1].dataset.dataset_dirs = [
                "/data/agx_cup_tray_mobile_rot6d_vlash_se2_h48_phase_split_v3_action_current_v1_camera_frame"
            ]
            with self.assertRaisesRegex(ValueError, "expected the parent dataset"):
                check_recipe(cfg, reference)
            cfg.data.train.datasets[1].dataset.dataset_dirs = ["/data/" + names[1]]
            cfg.data.train.source_sampling_weights = [3, 1, 1, 1, 1, 8]
            with self.assertRaisesRegex(ValueError, "Source sampling differs"):
                check_recipe(cfg, reference)


if __name__ == "__main__":
    unittest.main()
