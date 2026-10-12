#!/usr/bin/env python3
"""Launch target-balanced evaluator shards against existing policy servers."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=Path, required=True)
parser.add_argument("--benchmark", type=Path, default=Path("evaluation/benchmark_15k.json"))
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--policy-host", default="127.0.0.1")
parser.add_argument("--policy-ports", default="18144,18145,18146,18147")
parser.add_argument("--habitat-gpus", default="0,1,2,3")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

ports = [int(value) for value in args.policy_ports.split(",")]
gpus = [int(value) for value in args.habitat_gpus.split(",")]
if len(ports) != len(gpus) or not ports:
    raise SystemExit("policy-ports and habitat-gpus must have the same nonzero length")

root = Path(__file__).resolve().parents[1]
evaluator = root / "scripts/source_snapshot/framework_final/hm3d_closedloop_batch.py"
data_root = args.data_root.expanduser().resolve()
benchmark = args.benchmark if args.benchmark.is_absolute() else data_root / args.benchmark
if not benchmark.is_file():
    raise SystemExit(f"Missing benchmark: {benchmark}")

for shard, (port, gpu) in enumerate(zip(ports, gpus)):
    command = [
        sys.executable, str(evaluator),
        "--data-root", str(data_root),
        "--benchmark", str(benchmark),
        "--policy-host", args.policy_host,
        "--policy-port", str(port),
        "--output-dir", str(args.output_dir / f"shard{shard:02d}"),
        "--batch-size", str(args.batch_size),
        "--policy-retries", "3", "--policy-one-shot",
        "--source-prefetch-workers", "0",
        "--target-shard-count", str(len(ports)),
        "--target-shard-index", str(shard),
        "--variant-mode", "all", "--resume",
        "--near-zero-ratio", "0.80",
        "--near-zero-position-m", "0.05",
        "--near-zero-heading-rad", "0.05",
        "--near-zero-consecutive", "3",
        "--success-position-m", "0.50",
        "--success-heading-deg", "15.0",
        "--action-horizon", "32",
        "--camera-width", "566", "--camera-height", "320",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"CUDA_VISIBLE_DEVICES={gpu}", " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.Popen(command, env=env, cwd=root, start_new_session=True)
