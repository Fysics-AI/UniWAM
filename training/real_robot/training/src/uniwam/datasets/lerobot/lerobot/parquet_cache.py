"""Serialize local Parquet cache preparation and recover interrupted builds."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Sequence

import datasets
import pyarrow as pa
from filelock import FileLock

logger = logging.getLogger(__name__)
_READY_FILE = ".uniwam-ready.json"
_CACHE_VERSION = 1


def _node_namespace() -> str:
    node_rank = os.environ.get("GROUP_RANK") or os.environ.get("NODE_RANK")
    if node_rank is None:
        rank = os.environ.get("RANK")
        local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
        if rank is not None and local_world_size is not None:
            node_rank = str(int(rank) // int(local_world_size))
    identity = [socket.gethostname(), node_rank]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]


def _source_key(data_dir: str | Path | None, data_files: Sequence[str | Path] | None) -> str:
    if (data_dir is None) == (data_files is None):
        raise ValueError("Provide exactly one of data_dir and data_files.")
    if data_dir is not None:
        directory = Path(data_dir).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files = sorted(directory.rglob("*.parquet"))
        source = {"data_dir": str(directory)}
    else:
        files = [Path(path).expanduser().resolve() for path in data_files]
        source = {"data_files": [str(path) for path in files]}
    if not files:
        raise ValueError("No Parquet episode files were selected.")
    entries = []
    for path in files:
        stat = path.stat()
        entries.append([str(path), stat.st_size, stat.st_mtime_ns])
    payload = [source, entries, datasets.__version__, _CACHE_VERSION]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cache_is_ready(cache_dir: Path) -> bool:
    try:
        marker = json.loads((cache_dir / _READY_FILE).read_text())
        if marker["version"] != _CACHE_VERSION or not marker["files"]:
            return False
        for entry in marker["files"]:
            path = (cache_dir / entry["path"]).resolve()
            if not path.is_relative_to(cache_dir.resolve()):
                return False
            stat = path.stat()
            if stat.st_size != entry["size"] or stat.st_mtime_ns != entry["mtime_ns"]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _quarantine(cache_dir: Path) -> None:
    if not cache_dir.exists():
        return
    quarantine_root = cache_dir.parent / "quarantine"
    quarantine_root.mkdir(exist_ok=True)
    target = quarantine_root / f"{cache_dir.name}.{time.time_ns()}.{uuid.uuid4().hex[:8]}"
    cache_dir.rename(target)
    logger.warning("Moved incomplete or damaged Arrow cache to %s", target)


def _has_invalid_cache(cache_dir: Path) -> bool:
    try:
        for path in cache_dir.rglob("dataset_info.json"):
            json.loads(path.read_text())
    except (OSError, ValueError):
        return True
    files = list(cache_dir.rglob("*.arrow"))
    if not files:
        return True
    for path in files:
        try:
            with pa.memory_map(str(path), "r") as source:
                with pa.ipc.open_stream(source) as reader:
                    for batch in reader:
                        batch.validate(full=True)
        except (OSError, ValueError, pa.ArrowException):
            return True
    return False


def _write_ready(cache_dir: Path, dataset: datasets.Dataset) -> None:
    paths = {Path(entry["filename"]).resolve() for entry in dataset.cache_files}
    paths.update(path.resolve() for path in cache_dir.rglob("dataset_info.json"))
    if not dataset.cache_files:
        raise RuntimeError("Parquet loading returned no persistent Arrow cache files.")
    entries = []
    for path in sorted(paths):
        stat = path.stat()
        entries.append({
            "path": str(path.relative_to(cache_dir.resolve())),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    temporary = cache_dir / f"{_READY_FILE}.{uuid.uuid4().hex}.tmp"
    with temporary.open("w") as output:
        json.dump({"version": _CACHE_VERSION, "files": entries}, output)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, cache_dir / _READY_FILE)


def load_parquet_dataset(
    *,
    data_dir: str | Path | None = None,
    data_files: Sequence[str | Path] | None = None,
) -> datasets.Dataset:
    base = Path(os.environ.get("HF_DATASETS_CACHE") or datasets.config.HF_DATASETS_CACHE)
    node_root = base.expanduser().resolve() / "uniwam_parquet" / _node_namespace()
    node_root.mkdir(parents=True, exist_ok=True)
    source_key = _source_key(data_dir, data_files)
    cache_dir = node_root / source_key
    lock_dir = node_root / "locks"
    lock_dir.mkdir(exist_ok=True)
    kwargs = {"data_dir": str(data_dir)} if data_dir is not None else {
        "data_files": [str(path) for path in data_files],
    }

    # Keep the guard outside the cache so recovery never replaces a held lock.
    # It covers both builder preparation and the subsequent Arrow reader.
    with FileLock(str(lock_dir / f"{source_key}.lock")):
        ready = _cache_is_ready(cache_dir)
        if cache_dir.exists() and not ready:
            _quarantine(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        try:
            dataset = datasets.load_dataset(
                "parquet", split="train", cache_dir=str(cache_dir), **kwargs,
            )
        except Exception:
            # HF versions wrap Arrow failures differently. Rebuild only when
            # an existing cache is demonstrably damaged; source errors escape.
            if not ready or (_cache_is_ready(cache_dir) and not _has_invalid_cache(cache_dir)):
                raise
            _quarantine(cache_dir)
            cache_dir.mkdir()
            dataset = datasets.load_dataset(
                "parquet", split="train", cache_dir=str(cache_dir), **kwargs,
            )
        _write_ready(cache_dir, dataset)
        return dataset
