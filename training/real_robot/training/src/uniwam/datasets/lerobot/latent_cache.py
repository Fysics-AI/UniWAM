from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ShardedBFloat16LatentCache:
    """Read direct-indexed BF16 latent rows from rank-sharded raw arrays."""

    def __init__(
        self,
        root: str | Path,
        source_name: str,
        branches: tuple[str, ...],
        source_window_count: int,
        index_remap_root: str | Path | None = None,
        source_window_index_path: str | Path | None = None,
    ):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            raise RuntimeError(
                f"Latent cache is not complete: status={manifest.get('status')!r}."
            )
        if manifest.get("storage_dtype") != "bfloat16-raw-uint16":
            raise ValueError("Unsupported latent-cache storage dtype.")
        self.latent_shape = tuple(int(value) for value in manifest["latent_shape"])
        if self.latent_shape not in {(48, 3, 20, 24), (48, 4, 20, 24)}:
            raise ValueError(f"Unexpected latent shape {self.latent_shape}.")
        self.world_size = int(manifest["world_size"])
        sources = {source["name"]: source for source in manifest["sources"]}
        if source_name not in sources:
            raise KeyError(f"Source {source_name!r} is absent from latent-cache manifest.")
        source_manifest = sources[source_name]
        manifest_branch_counts = {
            str(branch): int(count)
            for branch, count in source_manifest["branches"].items()
        }
        self.index_remap_root = None if index_remap_root is None else Path(index_remap_root)
        expected_window_sha256 = source_manifest.get("window_index_sha256")
        if expected_window_sha256 is not None:
            if source_window_index_path is None:
                raise ValueError(
                    f"Cache {source_name!r} is bound to a window index, but no index path was provided."
                )
            actual_window_sha256 = _sha256(Path(source_window_index_path))
            if self.index_remap_root is None:
                if actual_window_sha256 != expected_window_sha256:
                    raise ValueError(
                        f"Latent cache/window index SHA256 mismatch for {source_name!r}: "
                        f"cache={expected_window_sha256}, current={actual_window_sha256}."
                    )
            else:
                remap_report_path = self.index_remap_root / "remap_report.json"
                if not remap_report_path.is_file():
                    raise FileNotFoundError(remap_report_path)
                remap_report = json.loads(remap_report_path.read_text())
                remap_source = remap_report.get("sources", {}).get(source_name)
                if remap_report.get("status") != "PASS" or remap_source is None:
                    raise ValueError(f"Invalid latent remap report for {source_name!r}.")
                if remap_source.get("old_window_index_sha256") != expected_window_sha256:
                    raise ValueError(
                        f"Latent remap old-index SHA256 does not match cache manifest for "
                        f"{source_name!r}."
                    )
                if remap_source.get("current_window_index_sha256") != actual_window_sha256:
                    raise ValueError(
                        f"Latent remap current-index SHA256 does not match configured index for "
                        f"{source_name!r}."
                    )
        manifest_branches = set(manifest_branch_counts)
        if set(branches) - manifest_branches:
            raise KeyError(
                f"Cache source {source_name!r} lacks branches {set(branches) - manifest_branches}."
            )
        self.source_name = source_name
        self._locations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._counts: dict[tuple[str, int], int] = {}
        self._memmaps: dict[tuple[str, int], np.memmap] = {}
        for branch in branches:
            if self.index_remap_root is not None:
                remap_dir = self.index_remap_root / source_name / branch
                rank_map_path = remap_dir / "rank_map.npy"
                row_map_path = remap_dir / "row_map.npy"
                if not rank_map_path.is_file() or not row_map_path.is_file():
                    raise FileNotFoundError(f"Missing latent cache index remap for {source_name}/{branch}.")
                rank_map = np.asarray(np.load(rank_map_path), dtype=np.int16)
                row_map = np.asarray(np.load(row_map_path), dtype=np.int32)
                if rank_map.shape != (source_window_count,) or row_map.shape != (source_window_count,):
                    raise ValueError(f"Latent cache remap shape mismatch for {source_name}/{branch}.")
            else:
                rank_map = np.full(source_window_count, -1, dtype=np.int16)
                row_map = np.full(source_window_count, -1, dtype=np.int32)
            branch_root = self.root / source_name / branch
            for rank in range(self.world_size):
                indices_path = branch_root / f"rank_{rank:02d}_source_indices.npy"
                data_path = branch_root / f"rank_{rank:02d}.bf16.bin"
                progress_path = branch_root / f"rank_{rank:02d}.progress.json"
                if not indices_path.is_file() or not data_path.is_file() or not progress_path.is_file():
                    raise FileNotFoundError(
                        f"Incomplete latent shard files for {source_name}/{branch}/rank_{rank:02d}."
                    )
                progress = json.loads(progress_path.read_text())
                if not progress.get("complete", False):
                    raise RuntimeError(
                        f"Incomplete latent shard {source_name}/{branch}/rank_{rank:02d}."
                    )
                indices = np.load(indices_path, mmap_mode="r")
                count = int(indices.size)
                expected_bytes = count * int(np.prod(self.latent_shape)) * 2
                if data_path.stat().st_size != expected_bytes:
                    raise ValueError(f"Latent shard size mismatch: {data_path}.")
                if count and self.index_remap_root is None:
                    if int(indices.min()) < 0 or int(indices.max()) >= source_window_count:
                        raise ValueError(f"Source indices out of bounds in {indices_path}.")
                    if np.any(rank_map[indices] >= 0):
                        raise ValueError(f"Duplicate cache source indices in {source_name}/{branch}.")
                    rank_map[indices] = rank
                    row_map[indices] = np.arange(count, dtype=np.int32)
                self._counts[(branch, rank)] = count
            actual_count = sum(
                self._counts[(branch, rank)] for rank in range(self.world_size)
            )
            expected_count = manifest_branch_counts[branch]
            if actual_count != expected_count:
                raise ValueError(
                    f"Latent shard count mismatch for {source_name}/{branch}: "
                    f"manifest={expected_count}, shards={actual_count}."
                )
            mapped = rank_map >= 0
            if self.index_remap_root is None and int(mapped.sum()) != expected_count:
                raise ValueError(
                    f"Latent index coverage mismatch for {source_name}/{branch}: "
                    f"manifest={expected_count}, mapped={int(mapped.sum())}."
                )
            if self.index_remap_root is not None and int(mapped.sum()) > expected_count:
                raise ValueError(
                    f"Latent remap maps more rows than exist in the cache for "
                    f"{source_name}/{branch}: cache={expected_count}, mapped={int(mapped.sum())}."
                )
            if np.any(row_map[mapped] < 0):
                raise ValueError(
                    f"Negative mapped cache row in {source_name}/{branch}."
                )
            if np.any(rank_map[mapped] >= self.world_size):
                raise ValueError(
                    f"Mapped cache rank exceeds world size in {source_name}/{branch}."
                )
            for rank in range(self.world_size):
                rank_rows = row_map[rank_map == rank]
                if rank_rows.size and int(rank_rows.max()) >= self._counts[(branch, rank)]:
                    raise ValueError(
                        f"Mapped cache row exceeds shard size in "
                        f"{source_name}/{branch}/rank_{rank:02d}."
                    )
            self._locations[branch] = (rank_map, row_map)

    def validate_required_indices(self, branch: str, source_indices: np.ndarray) -> None:
        """Fail at dataset construction if any routed training key lacks a latent."""
        branch = str(branch)
        if branch not in self._locations:
            raise KeyError(f"Unknown latent-cache branch {branch!r} for {self.source_name}.")
        required = np.asarray(source_indices, dtype=np.int64)
        if required.ndim != 1:
            raise ValueError(f"Required latent indices must be one-dimensional, got {required.shape}.")
        if required.size == 0:
            raise ValueError(f"Required latent indices are empty for {self.source_name}/{branch}.")
        rank_map, row_map = self._locations[branch]
        if int(required.min()) < 0 or int(required.max()) >= rank_map.size:
            raise IndexError(
                f"Required latent index is outside [0,{rank_map.size}) for "
                f"{self.source_name}/{branch}."
            )
        missing = required[(rank_map[required] < 0) | (row_map[required] < 0)]
        if missing.size:
            preview = missing[:8].tolist()
            raise ValueError(
                f"Latent cache misses {missing.size} routed keys for "
                f"{self.source_name}/{branch}; first={preview}."
            )

    def has(self, branch: str, source_window_index: int) -> bool:
        """Return whether one source window has a complete cached latent."""
        branch = str(branch)
        if branch not in self._locations:
            return False
        source_window_index = int(source_window_index)
        rank_map, row_map = self._locations[branch]
        if not 0 <= source_window_index < rank_map.size:
            return False
        return bool(
            rank_map[source_window_index] >= 0
            and row_map[source_window_index] >= 0
        )

    def coverage(self, branch: str, source_indices: np.ndarray) -> dict[str, int]:
        """Summarize cache coverage without requiring every index to be present."""
        branch = str(branch)
        if branch not in self._locations:
            raise KeyError(f"Unknown latent-cache branch {branch!r} for {self.source_name}.")
        required = np.asarray(source_indices, dtype=np.int64).reshape(-1)
        if required.size == 0:
            return {"required": 0, "cached": 0, "missing": 0}
        rank_map, row_map = self._locations[branch]
        if int(required.min()) < 0 or int(required.max()) >= rank_map.size:
            raise IndexError(
                f"Required latent index is outside [0,{rank_map.size}) for "
                f"{self.source_name}/{branch}."
            )
        cached = (rank_map[required] >= 0) & (row_map[required] >= 0)
        cached_count = int(np.count_nonzero(cached))
        return {
            "required": int(required.size),
            "cached": cached_count,
            "missing": int(required.size - cached_count),
        }

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_memmaps"] = {}
        return state

    def get(self, branch: str, source_window_index: int) -> torch.Tensor:
        branch = str(branch)
        source_window_index = int(source_window_index)
        rank_map, row_map = self._locations[branch]
        if not 0 <= source_window_index < rank_map.size:
            raise IndexError(source_window_index)
        rank = int(rank_map[source_window_index])
        row = int(row_map[source_window_index])
        if rank < 0 or row < 0:
            raise KeyError(
                f"No cached latent for {self.source_name}/{branch}/{source_window_index}."
            )
        key = (branch, rank)
        data = self._memmaps.get(key)
        if data is None:
            data = np.memmap(
                self.root / self.source_name / branch / f"rank_{rank:02d}.bf16.bin",
                mode="r",
                dtype=np.uint16,
                shape=(self._counts[key], *self.latent_shape),
            )
            self._memmaps[key] = data
        bits = np.array(data[row], copy=True)
        return torch.from_numpy(bits).view(torch.bfloat16)
