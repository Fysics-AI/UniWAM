#!/usr/bin/env bash
set -euo pipefail
NPROC_PER_NODE="${1:?Usage: train_robot.sh <GPUs per node> [training arguments]}"
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
torchrun --nnodes="${NNODES:-1}" --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank="${NODE_RANK:-0}" --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" "$SCRIPT_DIR/train_robot.py" "$@"
