"""Small end-to-end fixture for the six-source index preparation entrypoint."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


TRAINING = Path(__file__).resolve().parents[1]
SOURCES = (
    "agx_cup_tray_mobile_rot6d_vlash_se2_h48_phase_split_v3",
    "agx_move_white_box_mobile_rot6d_vlash_se2_h48_phase_split_v2",
    "agx_color_blocks_mobile_rot6d_vlash_se2_h48_phase_split_v3",
)


class IndexPreparationTest(unittest.TestCase):
    def test_manifest_cli_builds_windows_phase_and_remap(self) -> None:
        with tempfile.TemporaryDirectory(prefix="uniwam_index_test_") as temporary:
            root = Path(temporary)
            franka = root / "franka"
            (franka / "meta").mkdir(parents=True)
            (franka / "meta/episodes.jsonl").write_text(
                json.dumps({"episode_index": 0, "length": 40}) + "\n"
            )
            cache = root / "cache"
            base_remap = root / "base_remap"
            settings = {}
            for index, name in enumerate(SOURCES):
                old = root / name / "old"
                camera = root / name / "camera"
                for dataset in (old, camera):
                    (dataset / "meta").mkdir(parents=True)
                    (dataset / "meta/episodes.jsonl").write_text(
                        json.dumps({"episode_index": 0, "source_episode_index": 0, "length": 40}) + "\n"
                    )
                    video = dataset / "videos/chunk-000/observation.images.cam_manip_high/episode_000000.mp4"
                    video.parent.mkdir(parents=True)
                    video.write_bytes(b"same-video-contents")
                (camera / "meta/info.json").write_text(json.dumps({"failed_episode_labels": {}}))
                window = root / name / "upstream_window.parquet"
                phase = root / name / "upstream_phase.parquet"
                pq.write_table(pa.table({
                    "episode_index": [0, 0, 0],
                    "source_episode_index": [0, 0, 0],
                    "start_frame": [0, 1, 2],
                    "horizon": [32, 32, 32],
                }), window)
                branches = ("manip",) if index == 2 else ("manip", "nav")
                pq.write_table(pa.table({
                    "source_window_index": list(range(len(branches))),
                    "episode_index": [0] * len(branches),
                    "start_frame": list(range(len(branches))),
                    "horizon": [32] * len(branches),
                    "branch": list(branches),
                }), phase)
                for branch in branches:
                    for rank in range(8):
                        folder = cache / name / branch
                        folder.mkdir(parents=True, exist_ok=True)
                        np.save(folder / f"rank_{rank:02d}_source_indices.npy", np.arange(3) if rank == 0 else np.empty(0, np.int64))
                    if index == 1:
                        folder = base_remap / name / branch
                        folder.mkdir(parents=True, exist_ok=True)
                        np.save(folder / "rank_map.npy", np.zeros(3, dtype=np.int16))
                        np.save(folder / "row_map.npy", np.arange(3, dtype=np.int32))
                settings[name] = {
                    "old_root": str(old), "camera_root": str(camera),
                    "window_index": str(window), "phase_index": str(phase),
                    "branches": list(branches),
                    "base_remap_root": str(base_remap) if index == 1 else None,
                }
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "sources": [{"dataset_root": str(franka)}],
                "index_preparation": {"latent_cache_root": str(cache), "sources": settings},
            }))
            output = root / "prepared"
            subprocess.run(
                [sys.executable, str(TRAINING / "data_pipeline/prepare_camera_frame_cross_embodiment_h32.py"),
                 "--manifest", str(manifest), "--output-root", str(output)],
                check=True, capture_output=True, text=True,
            )
            for name in SOURCES:
                self.assertEqual(pq.read_metadata(output / name / "window_index_h32.parquet").num_rows, 3)
                self.assertEqual(
                    pq.read_metadata(output / name / "phase_branch_index_h32.parquet").num_rows,
                    1 if name == SOURCES[2] else 2,
                )
                self.assertEqual(np.load(output / "latent_remap" / name / "manip/rank_map.npy").tolist(), [0, 0, 0])
            self.assertEqual(pq.read_metadata(output / "franka_cup_pyramid/window_index_h32_stride4.parquet").num_rows, 2)


if __name__ == "__main__":
    unittest.main()
