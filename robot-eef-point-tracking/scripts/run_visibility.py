#!/usr/bin/env python3
"""Run DINOv2 visibility gating across GPU shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def video_path(data_root: Path, prototype: dict) -> Path:
    if "video" in prototype:
        video = Path(prototype["video"])
        return video if video.is_absolute() else data_root / video
    episode = int(prototype["episode"])
    return (
        data_root
        / prototype["dataset"]
        / "videos"
        / f"chunk-{episode // 1000:03d}"
        / prototype["video_key"]
        / f"episode_{episode:06d}.mp4"
    )


def main() -> int:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/lerobot"))
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("outputs/flange_geometry/geometry_manifest.jsonl"),
    )
    parser.add_argument("--geometry-root", type=Path, default=Path("outputs/flange_geometry"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/eef_tracks"))
    parser.add_argument("--dinov2-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prototypes", type=Path, default=Path("configs/dino_prototypes.json"))
    parser.add_argument("--devices", default="0", help="Comma-separated GPU indices, e.g. 0,1,2,3")
    parser.add_argument("--workers-per-device", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-episodes-per-shard", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    manifest = args.manifest.resolve()
    geometry_root = args.geometry_root.resolve()
    output_root = args.output_root.resolve()
    dino_repo = args.dinov2_repo.resolve()
    checkpoint = args.checkpoint.resolve()
    prototypes_path = args.prototypes.resolve()
    for required in (manifest, dino_repo, checkpoint, prototypes_path):
        if not required.exists():
            parser.error(f"required input does not exist: {required}")
    if not geometry_root.is_dir() or not manifest.is_relative_to(geometry_root):
        parser.error("--manifest must be located inside the existing --geometry-root")
    if args.workers_per_device < 1:
        parser.error("--workers-per-device must be positive")

    prototype_config = json.loads(prototypes_path.read_text(encoding="utf-8"))
    positives = prototype_config["positive"]
    negatives = prototype_config["negative"]
    for item in positives + negatives:
        path = video_path(data_root, item)
        if not path.is_file():
            parser.error(f"prototype video does not exist: {path}")

    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        parser.error("--devices must name at least one GPU")
    shard_count = len(devices) * args.workers_per_device
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[subprocess.Popen, object, Path]] = []
    try:
        for shard in range(shard_count):
            device = devices[shard % len(devices)]
            command = [
                sys.executable,
                str(package_root / "src/dino_eef_visibility_batch.py"),
                "--manifest", str(manifest),
                "--output-root", str(output_root),
                "--dinov2-repo", str(dino_repo),
                "--checkpoint", str(checkpoint),
                "--batch-size", str(args.batch_size),
                "--stride", "5",
                "--feature-radius", "0",
                "--enter-threshold", "0.20",
                "--exit-threshold", "0.10",
                "--shard-id", str(shard),
                "--num-shards", str(shard_count),
                "--events", str(logs / f"shard_{shard}.events.jsonl"),
            ]
            for item in positives:
                command += ["--prototype", str(video_path(data_root, item)), str(item["frame"]), str(item["x"]), str(item["y"])]
            for item in negatives:
                command += ["--negative-prototype", str(video_path(data_root, item)), str(item["frame"]), str(item["x"]), str(item["y"])]
            if args.max_episodes_per_shard is not None:
                command += ["--max-episodes", str(args.max_episodes_per_shard)]
            if args.overwrite:
                command.append("--overwrite")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            env["PYTHONPATH"] = os.pathsep.join(
                [str(dino_repo), env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
            env.setdefault("EEF_DINO_CPU_THREADS", "4")
            env.setdefault("OMP_NUM_THREADS", env["EEF_DINO_CPU_THREADS"])
            env.setdefault("MKL_NUM_THREADS", env["EEF_DINO_CPU_THREADS"])
            env.setdefault("OPENBLAS_NUM_THREADS", env["EEF_DINO_CPU_THREADS"])
            log_path = logs / f"shard_{shard}.log"
            log_file = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(command, cwd=package_root, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            processes.append((process, log_file, log_path))
            print(f"started shard {shard}/{shard_count} on GPU {device}: {log_path}", flush=True)
        statuses = [process.wait() for process, _, _ in processes]
    except KeyboardInterrupt:
        for process, _, _ in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        statuses = [process.wait() for process, _, _ in processes]
        return 130
    finally:
        for _, log_file, _ in processes:
            log_file.close()
    failed = [(index, code, path) for index, (code, (_, _, path)) in enumerate(zip(statuses, processes)) if code]
    if failed:
        for shard, code, path in failed:
            print(f"shard {shard} failed with exit={code}; inspect {path}", file=sys.stderr)
        return 1
    print(f"All {shard_count} visibility shards completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
