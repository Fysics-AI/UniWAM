"""Exercise cold-start contention, interrupted builds and cache recovery."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uniwam.datasets.lerobot.lerobot import parquet_cache as cache


def _worker(source, cache_root, node, start, results):
    with patch.dict(os.environ, {"HF_DATASETS_CACHE": cache_root, "GROUP_RANK": "0"}):
        with patch.object(cache.socket, "gethostname", return_value=node):
            original = cache.datasets.load_dataset

            def load(*args, **kwargs):
                directory = Path(kwargs["cache_dir"])
                if not (directory / cache._READY_FILE).exists():
                    with (directory.parent / "builds.log").open("a") as output:
                        output.write("build\n")
                return original(*args, **kwargs)

            try:
                start.wait(30)
                with patch.object(cache.datasets, "load_dataset", side_effect=load):
                    dataset = cache.load_parquet_dataset(data_dir=source)
                results.put((node, dataset["frame_index"], dataset.cache_files[0]["filename"]))
            except Exception as error:
                results.put(("error", repr(error), ""))


def _interrupted_worker(source, cache_root):
    with patch.dict(os.environ, {"HF_DATASETS_CACHE": cache_root, "GROUP_RANK": "0"}):
        with patch.object(cache.socket, "gethostname", return_value="node-a"):
            def interrupt(*args, **kwargs):
                directory = Path(kwargs["cache_dir"])
                (directory / "partial.arrow").write_bytes(b"interrupted")
                (directory / "builder.incomplete").mkdir()
                os._exit(19)

            with patch.object(cache.datasets, "load_dataset", side_effect=interrupt):
                cache.load_parquet_dataset(data_dir=source)


class ParquetCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="uniwam_arrow_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source" / "data"
        self.source.mkdir(parents=True)
        self.episode = self.source / "episode_000000.parquet"
        pq.write_table(pa.table({
            "frame_index": [0, 1, 2],
            "action": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        }), self.episode)
        self.cache_root = self.root / "shared-cache"
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {
            "HF_HOME": str(self.root / "shared-hf-home"),
            "HF_DATASETS_CACHE": str(self.cache_root),
            "GROUP_RANK": "0",
        }).start()
        patch.object(cache.socket, "gethostname", return_value="node-a").start()

    def _load(self):
        return cache.load_parquet_dataset(data_dir=self.source)

    def test_shared_environment_is_scoped_and_warm_cache_reused(self):
        expected = pq.read_table(self.episode).to_pydict()
        first = self._load()
        second = self._load()
        self.assertEqual(first.to_dict(), expected)
        self.assertEqual(second.to_dict(), expected)
        self.assertEqual(first.cache_files, second.cache_files)
        self.assertTrue(Path(first.cache_files[0]["filename"]).is_relative_to(self.cache_root))
        self.assertFalse(list(self.cache_root.rglob("quarantine")))

    def test_hf_home_only_and_direct_python_loading(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_DATASETS_CACHE")
            with patch.object(cache.datasets.config, "HF_DATASETS_CACHE",
                              str(self.root / "shared-hf-home/datasets")):
                dataset = self._load()
        self.assertEqual(dataset.num_rows, 3)
        self.assertIn("uniwam_parquet", dataset.cache_files[0]["filename"])

    def test_two_nodes_and_same_node_workers_start_together(self):
        context = mp.get_context("spawn")
        start = context.Barrier(6)
        results = context.Queue()
        processes = [
            context.Process(target=_worker, args=(
                str(self.source), str(self.cache_root), node, start, results,
            ))
            for node in ("node-a", "node-a", "node-a", "node-b", "node-b", "node-b")
        ]
        for process in processes:
            process.start()
        try:
            records = [results.get(timeout=60) for _ in processes]
            for process in processes:
                process.join(30)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(5)
            results.close()
        filenames = {}
        for node, frames, filename in records:
            self.assertNotEqual(node, "error", frames)
            self.assertEqual(frames, [0, 1, 2])
            filenames.setdefault(node, set()).add(filename)
        self.assertEqual(len(filenames["node-a"]), 1)
        self.assertEqual(len(filenames["node-b"]), 1)
        self.assertNotEqual(filenames["node-a"], filenames["node-b"])
        counters = list(self.cache_root.rglob("builds.log"))
        self.assertEqual(len(counters), 2)
        for counter in counters:
            self.assertEqual(counter.read_text(), "build\n")

    def test_killed_builder_releases_lock_and_preserves_partial_cache(self):
        process = mp.get_context("spawn").Process(
            target=_interrupted_worker,
            args=(str(self.source), str(self.cache_root)),
        )
        process.start()
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join(5)
        self.assertEqual(process.exitcode, 19)
        dataset = self._load()
        self.assertEqual(dataset.num_rows, 3)
        preserved = list(self.cache_root.glob("uniwam_parquet/*/quarantine/*/partial.arrow"))
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0].read_bytes(), b"interrupted")

    def test_truncated_arrow_is_rebuilt(self):
        dataset = self._load()
        filename = Path(dataset.cache_files[0]["filename"])
        del dataset
        with filename.open("r+b") as output:
            output.truncate(12)
        rebuilt = self._load()
        self.assertEqual(rebuilt.num_rows, 3)
        self.assertEqual(rebuilt["frame_index"], [0, 1, 2])
        self.assertEqual(len(list(self.cache_root.glob("uniwam_parquet/*/quarantine/*"))), 1)

    def test_same_size_arrow_corruption_is_rebuilt_after_load_failure(self):
        dataset = self._load()
        filename = Path(dataset.cache_files[0]["filename"])
        del dataset
        stat = filename.stat()
        with filename.open("r+b") as output:
            output.write(b"\x00" * 16)
        os.utime(filename, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(filename.stat().st_size, stat.st_size)
        rebuilt = self._load()
        self.assertEqual(rebuilt["frame_index"], [0, 1, 2])
        self.assertEqual(len(list(self.cache_root.glob("uniwam_parquet/*/quarantine/*"))), 1)

    def test_corrupt_metadata_is_rebuilt(self):
        dataset = self._load()
        filename = Path(dataset.cache_files[0]["filename"]).parent / "dataset_info.json"
        del dataset
        stat = filename.stat()
        filename.write_bytes(b"x" * stat.st_size)
        os.utime(filename, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        rebuilt = self._load()
        self.assertEqual(rebuilt.num_rows, 3)

    def test_selected_episodes_and_modified_sources_keep_correct_rows(self):
        second = self.source / "episode_000001.parquet"
        pq.write_table(pa.table({"frame_index": [9], "action": [[0.7, 0.8]]}), second)
        selected = cache.load_parquet_dataset(data_files=[second])
        self.assertEqual(selected["frame_index"], [9])
        all_rows = self._load()
        self.assertEqual(all_rows["frame_index"], [0, 1, 2, 9])
        pq.write_table(pa.table({"frame_index": [8], "action": [[0.7, 0.8]]}), second)
        updated = self._load()
        self.assertEqual(updated["frame_index"], [0, 1, 2, 8])
        self.assertNotEqual(all_rows.cache_files, updated.cache_files)

    def test_bad_source_is_not_hidden_or_retried(self):
        self.episode.write_bytes(b"not a parquet file")
        with patch.object(cache.datasets, "load_dataset", wraps=cache.datasets.load_dataset) as load:
            with self.assertRaises(Exception):
                self._load()
        self.assertEqual(load.call_count, 1)
        self.assertEqual(self.episode.read_bytes(), b"not a parquet file")
        self.assertFalse(list(self.cache_root.rglob(cache._READY_FILE)))

    def test_invalid_source_selection_fails_before_creating_dataset_cache(self):
        with self.assertRaises(FileNotFoundError):
            cache.load_parquet_dataset(data_dir=self.root / "missing")
        with self.assertRaises(ValueError):
            cache.load_parquet_dataset(data_files=[])
        self.assertFalse(list(self.cache_root.rglob(cache._READY_FILE)))


if __name__ == "__main__":
    unittest.main()
