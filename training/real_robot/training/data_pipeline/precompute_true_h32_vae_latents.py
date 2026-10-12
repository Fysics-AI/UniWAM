#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Dataset

from uniwam.models.wan22.helpers.loader import _load_registered_model
from uniwam.vision import (
    build_aspect_padded_fastwam_mosaic,
    build_matched_fastwam_mosaic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK = "uniwam_camera_frame_six_source_manip26_embodiment_stats_200k"
VAE_PATH = Path(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "/path/to/model-checkpoints")) / (
    "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
)
LATENT_SHAPE = (48, 3, 20, 24)
VIDEO_SHAPE = (3, 9, 320, 384)


def atomic_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(payload)
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        if error.errno != errno.ENOSPC or not path.is_file():
            raise
        path.write_text(payload)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase_keys(path: str) -> dict[str, np.ndarray]:
    table = pq.read_table(path, columns=["source_window_index", "branch"])
    source = np.asarray(table["source_window_index"].to_numpy(), dtype=np.int64)
    branch = np.asarray(table["branch"].to_pylist(), dtype=object)
    result = {}
    for name in sorted(set(branch.tolist())):
        values = source[branch == name]
        if values.size != np.unique(values).size:
            raise ValueError(f"Duplicate ({name}, source_window_index) keys in {path}.")
        result[str(name)] = np.sort(values)
    return result


def source_specs(cfg) -> list[dict]:
    specs = []
    for source_id, source_cfg in enumerate(cfg.data.train.datasets):
        dataset_cfg = source_cfg.dataset
        dataset_root = Path(str(dataset_cfg.dataset_dirs[0]))
        window_path = Path(
            str(dataset_cfg.get("window_index_path", dataset_root / "meta/window_index.parquet"))
        )
        if "phase_index_path" in source_cfg:
            keys = phase_keys(str(source_cfg.phase_index_path))
        else:
            branches = source_cfg.get("branches", "manip")
            if isinstance(branches, str):
                branches = ("manip", "nav") if branches == "both" else (branches,)
            # Plain/BranchAnnotated sources may override the dataset's original
            # window table (white-box does this for true H32).  Cache keys must
            # follow that exact training table, not dataset_root/meta.
            count = pq.read_metadata(window_path).num_rows
            keys = {str(branch): np.arange(count, dtype=np.int64) for branch in branches}
        specs.append(
            {
                "source_id": source_id,
                "name": dataset_root.name,
                "dataset_root": str(dataset_root),
                "dataset_cfg": dataset_cfg,
                "window_index_path": str(window_path),
                "window_index_sha256": sha256(window_path),
                "keys": keys,
            }
        )
    return specs


class VideoKeys(Dataset):
    def __init__(self, dataset, branch: str, indices: np.ndarray):
        self.dataset = dataset
        self.branch = str(branch)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, row: int) -> tuple[int, torch.Tensor]:
        source_index = int(self.indices[int(row)])
        return source_index, self.dataset.get_branch_video(source_index, self.branch)


def encode_shard(
    *,
    vae,
    source: dict,
    branch: str,
    all_indices: np.ndarray,
    output_root: Path,
    rank: int,
    world_size: int,
    workers: int,
    batch_size: int,
    prefetch_factor: int,
) -> None:
    indices = np.asarray(all_indices[rank::world_size], dtype=np.int64)
    branch_root = output_root / source["name"] / branch
    branch_root.mkdir(parents=True, exist_ok=True)
    index_path = branch_root / f"rank_{rank:02d}_source_indices.npy"
    data_path = branch_root / f"rank_{rank:02d}.bf16.bin"
    progress_path = branch_root / f"rank_{rank:02d}.progress.json"
    np.save(index_path, indices)

    expected_bytes = int(indices.size * np.prod(LATENT_SHAPE) * 2)
    mode = "r+" if data_path.is_file() and data_path.stat().st_size == expected_bytes else "w+"
    data = np.memmap(data_path, mode=mode, dtype=np.uint16, shape=(indices.size, *LATENT_SHAPE))
    completed = 0
    if mode == "r+" and progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        completed = int(progress.get("completed", 0))
    if not 0 <= completed <= indices.size:
        raise ValueError(f"Invalid progress {completed}/{indices.size} in {progress_path}.")
    if completed == indices.size:
        print(f"[rank {rank}] {source['name']}/{branch}: already complete", flush=True)
        return

    dataset = instantiate(source["dataset_cfg"])
    loader = DataLoader(
        VideoKeys(dataset, branch, indices[completed:]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=prefetch_factor if workers > 0 else None,
        pin_memory=True,
    )
    started = time.time()
    last_report = started
    next_offset = completed
    for source_indices, videos in loader:
        source_indices = torch.as_tensor(source_indices, dtype=torch.long)
        batch_count = int(source_indices.numel())
        expected_indices = torch.from_numpy(
            np.asarray(indices[next_offset : next_offset + batch_count], dtype=np.int64)
        )
        if not torch.equal(source_indices, expected_indices):
            raise RuntimeError("DataLoader changed deterministic source-index order.")
        if tuple(videos.shape[1:]) != VIDEO_SHAPE:
            raise ValueError(
                f"Video shape mismatch: {tuple(videos.shape[1:])} != {VIDEO_SHAPE}."
            )
        with torch.inference_mode():
            # WanVideoVAE.encode loops over the batch in Python. single_encode
            # runs the same frozen causal encoder as one real GPU batch.
            latent = vae.single_encode(
                videos.to("cuda", dtype=torch.bfloat16, non_blocking=True),
                device="cuda",
            )
        if tuple(latent.shape) != (batch_count, *LATENT_SHAPE) or latent.dtype != torch.bfloat16:
            raise ValueError(
                f"Latent contract mismatch: shape={tuple(latent.shape)} dtype={latent.dtype}."
            )
        data[next_offset : next_offset + batch_count] = (
            latent.contiguous().cpu().view(torch.uint16).numpy()
        )
        next_offset += batch_count
        done = next_offset
        now = time.time()
        if done % 256 < batch_count or done == indices.size:
            data.flush()
            atomic_json(
                progress_path,
                {
                    "completed": done,
                    "count": int(indices.size),
                    "last_source_index": int(source_indices[-1]),
                    "updated_unix": now,
                },
            )
        if now - last_report >= 30.0:
            rate = (done - completed) / max(now - started, 1.0e-6)
            eta = (indices.size - done) / max(rate, 1.0e-6)
            print(
                f"[rank {rank}] {source['name']}/{branch} "
                f"{done}/{indices.size} rate={rate:.2f}/s eta_h={eta / 3600:.2f}",
                flush=True,
            )
            last_report = now
    data.flush()
    atomic_json(
        progress_path,
        {
            "completed": int(indices.size),
            "count": int(indices.size),
            "complete": True,
            "updated_unix": time.time(),
        },
    )


def decode_episode_videos(dataset, episode_index: int) -> dict[str, torch.Tensor]:
    """Decode each episode camera once; adjacent H32 windows then share pixels."""
    lerobot = dataset.lerobot_dataset.multi_dataset._datasets[0]
    result = {}
    for image_meta in dataset.shape_meta["images"]:
        key = str(image_meta["key"])
        video_key = str(image_meta["lerobot_key"])
        _, height, width = map(int, image_meta["raw_shape"])
        video_path = lerobot.root / lerobot.meta.get_video_file_path(episode_index, video_key)
        raw_path = Path(
            f"/dev/shm/fastwam_h32_{os.getpid()}_{episode_index}_{key}.rgb"
        )
        try:
            with raw_path.open("wb") as raw_handle:
                process = subprocess.run(
                    [
                        "ffmpeg", "-v", "error", "-threads", "2", "-i", str(video_path),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
                    ],
                    check=False,
                    stdout=raw_handle,
                    stderr=subprocess.PIPE,
                )
            if process.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg failed for {video_path}: "
                    f"{process.stderr.decode(errors='replace')}"
                )
            frame_bytes = height * width * 3
            raw_bytes = raw_path.stat().st_size
            if raw_bytes <= 0 or raw_bytes % frame_bytes:
                raise ValueError(
                    f"Invalid rawvideo byte count for {video_path}: {raw_bytes}"
                )
            frame_count = raw_bytes // frame_bytes
            array = np.memmap(
                raw_path,
                mode="r+",
                dtype=np.uint8,
                shape=(frame_count, height, width, 3),
            )
            result[key] = torch.from_numpy(array).permute(0, 3, 1, 2)
        finally:
            raw_path.unlink(missing_ok=True)
    return result


def build_episode_video_batch(
    dataset,
    episode_frames: dict[str, torch.Tensor],
    start_frames: np.ndarray,
    branch: str,
    device: str = "cpu",
) -> torch.Tensor:
    offsets = torch.tensor(dataset.video_sample_indices, dtype=torch.long)
    starts = torch.from_numpy(np.asarray(start_frames, dtype=np.int64))
    frame_indices = starts[:, None] + offsets[None, :]
    batch_count, temporal_count = frame_indices.shape

    def select(key: str) -> torch.Tensor:
        frames = episode_frames[key]
        if int(frame_indices.max()) >= frames.shape[0]:
            raise IndexError(
                f"H32 video index {int(frame_indices.max())} exceeds {key} length {frames.shape[0]}."
            )
        selected = frames[frame_indices].to(device=device, non_blocking=True)
        selected = selected.float().div_(255.0)
        return selected.reshape(batch_count * temporal_count, *selected.shape[-3:])

    if dataset.concat_multi_camera == "parallel_nav_manip":
        if dataset.nav_image_mode != "matched_wrist_mosaic":
            raise ValueError(
                "Episodewise H32 precompute requires matched_wrist_mosaic for mobile data."
            )
        main_key = "cam_nav" if branch == "nav" else "cam_manip_high"
        panel = build_matched_fastwam_mosaic(
            select(main_key), select("cam_left_wrist"), select("cam_right_wrist")
        )
    elif dataset.concat_multi_camera == "fastwam_384x320_aspect_pad" and branch == "manip":
        keys = [str(meta["key"]) for meta in dataset.shape_meta["images"]]
        if len(keys) != 3:
            raise ValueError(f"Expected three manipulation cameras, got {keys}.")
        panel = build_aspect_padded_fastwam_mosaic(
            select(keys[0]), select(keys[1]), select(keys[2])
        )
    elif dataset.concat_multi_camera == "fastwam_384x320" and branch == "manip":
        keys = [str(meta["key"]) for meta in dataset.shape_meta["images"]]
        if len(keys) != 3:
            raise ValueError(f"Expected three manipulation cameras, got {keys}.")
        panel = build_matched_fastwam_mosaic(
            select(keys[0]), select(keys[1]), select(keys[2])
        )
    else:
        raise ValueError(
            f"Unsupported episodewise mosaic: {dataset.concat_multi_camera}/{branch}."
        )
    panel = panel.reshape(batch_count, temporal_count, *panel.shape[-3:])
    video = dataset.normalize_transform(panel).permute(0, 2, 1, 3, 4).contiguous()
    if tuple(video.shape[1:]) != VIDEO_SHAPE:
        raise ValueError(f"Episodewise video shape mismatch: {tuple(video.shape)}.")
    return video


def assign_episodes(
    episode_indices: np.ndarray,
    branch_indices: dict[str, np.ndarray],
    world_size: int,
) -> dict[int, int]:
    """Greedily balance whole episodes without decoding one MP4 on several ranks."""
    work = {}
    for indices in branch_indices.values():
        episodes, counts = np.unique(episode_indices[indices], return_counts=True)
        for episode, count in zip(episodes.tolist(), counts.tolist(), strict=True):
            work[int(episode)] = work.get(int(episode), 0) + int(count)
    loads = [0] * world_size
    owner = {}
    for episode, count in sorted(work.items(), key=lambda item: (-item[1], item[0])):
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        owner[episode] = rank
        loads[rank] += count
    return owner


def encode_source_episodewise(
    *,
    vae,
    source: dict,
    output_root: Path,
    rank: int,
    world_size: int,
    batch_size: int,
    checkpoint_episode_interval: int,
    episode_prefetch: int,
) -> None:
    dataset = instantiate(source["dataset_cfg"])
    episode_by_window = np.asarray(dataset.window_episode_index, dtype=np.int64)
    start_by_window = np.asarray(dataset.window_start_frame, dtype=np.int64)
    owner = assign_episodes(episode_by_window, source["keys"], world_size)
    owned_episodes = sorted(episode for episode, value in owner.items() if value == rank)

    states = {}
    for branch, all_indices in source["keys"].items():
        all_indices = np.asarray(all_indices, dtype=np.int64)
        owned = all_indices[
            np.fromiter(
                (owner[int(episode_by_window[index])] == rank for index in all_indices),
                dtype=bool,
                count=all_indices.size,
            )
        ]
        branch_root = output_root / source["name"] / branch
        branch_root.mkdir(parents=True, exist_ok=True)
        index_path = branch_root / f"rank_{rank:02d}_source_indices.npy"
        data_path = branch_root / f"rank_{rank:02d}.bf16.bin"
        progress_path = branch_root / f"rank_{rank:02d}.progress.json"
        np.save(index_path, owned)
        expected_bytes = int(owned.size * np.prod(LATENT_SHAPE) * 2)
        mode = "r+" if data_path.is_file() and data_path.stat().st_size == expected_bytes else "w+"
        data = np.memmap(data_path, mode=mode, dtype=np.uint16, shape=(owned.size, *LATENT_SHAPE))
        completed_episodes = set()
        if mode == "r+" and progress_path.is_file():
            progress = json.loads(progress_path.read_text())
            completed_episodes = {int(value) for value in progress.get("completed_episodes", [])}
        row_lookup = np.full(episode_by_window.size, -1, dtype=np.int64)
        row_lookup[owned] = np.arange(owned.size, dtype=np.int64)
        states[branch] = {
            "indices": owned,
            "data": data,
            "progress_path": progress_path,
            "completed_episodes": completed_episodes,
            "row_lookup": row_lookup,
        }

    started = time.time()
    initial_done = sum(
        sum(
            np.count_nonzero(episode_by_window[state["indices"]] == episode)
            for episode in state["completed_episodes"]
        )
        for state in states.values()
    )
    done = int(initial_done)
    last_report = started
    episode_jobs = []
    for episode in owned_episodes:
        pending = {}
        for branch, state in states.items():
            if episode in state["completed_episodes"]:
                continue
            mask = episode_by_window[state["indices"]] == episode
            if np.any(mask):
                pending[branch] = state["indices"][mask]
        if not pending:
            continue
        episode_jobs.append((episode, pending))

    if episode_jobs:
        with ThreadPoolExecutor(
            max_workers=episode_prefetch, thread_name_prefix="episode-decode"
        ) as pool:
            decode_futures = {}
            next_job_index = 0
            while next_job_index < min(episode_prefetch, len(episode_jobs)):
                decode_futures[next_job_index] = pool.submit(
                    decode_episode_videos, dataset, episode_jobs[next_job_index][0]
                )
                next_job_index += 1
            for job_index, (episode, pending) in enumerate(episode_jobs):
                episode_frames = decode_futures.pop(job_index).result()
                while (
                    next_job_index < len(episode_jobs)
                    and len(decode_futures) < episode_prefetch
                ):
                    decode_futures[next_job_index] = pool.submit(
                        decode_episode_videos, dataset, episode_jobs[next_job_index][0]
                    )
                    next_job_index += 1
                for branch, source_indices in pending.items():
                    state = states[branch]
                    for offset in range(0, source_indices.size, batch_size):
                        batch_indices = source_indices[offset : offset + batch_size]
                        videos = build_episode_video_batch(
                            dataset,
                            episode_frames,
                            start_by_window[batch_indices],
                            branch,
                            device="cuda",
                        )
                        count = int(videos.shape[0])
                        if count < batch_size:
                            videos = torch.cat(
                                [
                                    videos,
                                    videos[-1:].expand(batch_size - count, -1, -1, -1, -1),
                                ],
                                dim=0,
                            )
                        with torch.inference_mode():
                            latent = vae.single_encode(
                                videos.to(dtype=torch.bfloat16),
                                device="cuda",
                            )[:count]
                        rows = state["row_lookup"][batch_indices]
                        if np.any(rows < 0):
                            raise RuntimeError("Episodewise cache row lookup failed.")
                        state["data"][rows] = (
                            latent.contiguous().cpu().view(torch.uint16).numpy()
                        )
                        done += count
                    state["completed_episodes"].add(episode)
                del episode_frames
                should_checkpoint = (
                    (job_index + 1) % checkpoint_episode_interval == 0
                    or job_index + 1 == len(episode_jobs)
                )
                if should_checkpoint:
                    for state in states.values():
                        state["data"].flush()
                        atomic_json(
                            state["progress_path"],
                            {
                                "completed": int(
                                    sum(
                                        np.count_nonzero(
                                            episode_by_window[state["indices"]]
                                            == completed_episode
                                        )
                                        for completed_episode in state["completed_episodes"]
                                    )
                                ),
                                "count": int(state["indices"].size),
                                "completed_episodes": sorted(
                                    state["completed_episodes"]
                                ),
                                "updated_unix": time.time(),
                            },
                        )
                now = time.time()
                if now - last_report >= 30.0:
                    rate = (done - initial_done) / max(now - started, 1.0e-6)
                    print(
                        f"[rank {rank}] {source['name']} episode={episode} "
                        f"rows={done} rate={rate:.2f}/s",
                        flush=True,
                    )
                    last_report = now

    for state in states.values():
        state["data"].flush()
        atomic_json(
            state["progress_path"],
            {
                "completed": int(state["indices"].size),
                "count": int(state["indices"].size),
                "completed_episodes": sorted(state["completed_episodes"]),
                "complete": True,
                "updated_unix": time.time(),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--source-index", action="append", type=int)
    parser.add_argument("--episodewise-ffmpeg", action="store_true")
    parser.add_argument("--checkpoint-episode-interval", type=int, default=8)
    parser.add_argument("--episode-prefetch", type=int, default=1)
    parser.add_argument("--logical-rank-offset", type=int, default=0)
    parser.add_argument("--logical-world-size", type=int)
    parser.add_argument("--defer-manifest-finalize", action="store_true")
    args = parser.parse_args()
    if not VAE_PATH.is_file():
        raise FileNotFoundError(
            f"Wan2.2 VAE weights missing: {VAE_PATH}. Set DIFFSYNTH_MODEL_BASE_PATH."
        )
    if args.workers < 0:
        raise ValueError("--workers must be non-negative.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive.")
    if args.checkpoint_episode_interval <= 0:
        raise ValueError("--checkpoint-episode-interval must be positive.")
    if args.episode_prefetch <= 0:
        raise ValueError("--episode-prefetch must be positive.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # Ranks own disjoint cache shards. Gloo is sufficient for the few control
    # barriers and avoids unnecessary NCCL peer setup with co-resident services.
    dist.init_process_group("gloo")
    local_rank_in_group = dist.get_rank()
    local_world_size = dist.get_world_size()
    world_size = args.logical_world_size or local_world_size
    rank = args.logical_rank_offset + local_rank_in_group
    if world_size <= 0:
        raise ValueError("--logical-world-size must be positive.")
    if args.logical_rank_offset < 0 or rank >= world_size:
        raise ValueError(
            f"Invalid logical rank range: offset={args.logical_rank_offset}, "
            f"local_world_size={local_world_size}, logical_world_size={world_size}."
        )
    if not args.defer_manifest_finalize and (
        args.logical_rank_offset != 0 or local_world_size != world_size
    ):
        raise ValueError(
            "Partial logical-rank jobs must use --defer-manifest-finalize."
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[f"task={args.task}"])
    specs = source_specs(cfg)
    if args.source_index is not None:
        requested = tuple(args.source_index)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("--source-index values must be non-empty and unique.")
        if min(requested) < 0 or max(requested) >= len(specs):
            raise IndexError(f"--source-index outside configured sources: {requested}")
        specs = [specs[index] for index in requested]
    for source in specs:
        # Precomputation must always decode pixels, even when the training
        # configuration points at a cache that this job is about to create.
        source["dataset_cfg"]["precomputed_latent_root"] = None
        if "precomputed_latent_index_remap_root" in source["dataset_cfg"]:
            source["dataset_cfg"]["precomputed_latent_index_remap_root"] = None

    if rank == 0:
        manifest = {
            "version": 1,
            "status": "building",
            "task": args.task,
            "world_size": world_size,
            "logical_sharding": True,
            "coordinator_local_world_size": local_world_size,
            "batch_size_per_rank": args.batch_size,
            "workers_per_rank": args.workers,
            "prefetch_factor": args.prefetch_factor,
            "episodewise_ffmpeg": args.episodewise_ffmpeg,
            "checkpoint_episode_interval": args.checkpoint_episode_interval,
            "episode_prefetch": args.episode_prefetch,
            "video_shape": list(VIDEO_SHAPE),
            "latent_shape": list(LATENT_SHAPE),
            "storage_dtype": "bfloat16-raw-uint16",
            "vae_path": str(VAE_PATH),
            "vae_sha256": sha256(VAE_PATH),
            "sources": [
                {
                    "source_id": source["source_id"],
                    "name": source["name"],
                    "dataset_root": source["dataset_root"],
                    "window_index_path": source["window_index_path"],
                    "window_index_sha256": source["window_index_sha256"],
                    "branches": {
                        branch: int(indices.size)
                        for branch, indices in source["keys"].items()
                    },
                }
                for source in specs
            ],
        }
        atomic_json(args.output_root / "manifest.json", manifest)
    dist.barrier()

    vae = _load_registered_model(
        str(VAE_PATH),
        "wan_video_vae",
        torch_dtype=torch.bfloat16,
        device="cuda",
    ).eval()
    for source in specs:
        if args.episodewise_ffmpeg:
            encode_source_episodewise(
                vae=vae,
                source=source,
                output_root=args.output_root,
                rank=rank,
                world_size=world_size,
                batch_size=args.batch_size,
                checkpoint_episode_interval=args.checkpoint_episode_interval,
                episode_prefetch=args.episode_prefetch,
            )
        else:
            for branch, indices in source["keys"].items():
                encode_shard(
                    vae=vae,
                    source=source,
                    branch=branch,
                    all_indices=indices,
                    output_root=args.output_root,
                    rank=rank,
                    world_size=world_size,
                    workers=args.workers,
                    batch_size=args.batch_size,
                    prefetch_factor=args.prefetch_factor,
                )
        dist.barrier()

    if rank == 0 and not args.defer_manifest_finalize:
        manifest_path = args.output_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "complete"
        manifest["completed_unix"] = time.time()
        atomic_json(manifest_path, manifest)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
