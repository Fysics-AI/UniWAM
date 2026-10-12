#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train.sh <nproc_per_node> [Hydra overrides...]}"
shift
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export UNIWAM_COMPONENTS_ROOT="${UNIWAM_COMPONENTS_ROOT:-$(realpath ../../UniWAM-weights/components/wan2.2)}"
LAUNCH_ARGS=(--num_processes "$NPROC_PER_NODE" --num_machines 1 --machine_rank 0 --mixed_precision bf16 --dynamo_backend no)
if (( NPROC_PER_NODE > 1 )); then
  LAUNCH_ARGS+=(--multi_gpu)
fi
accelerate launch "${LAUNCH_ARGS[@]}" scripts/train.py \
  --config-name task/uniwam_base_hm3d "$@"
