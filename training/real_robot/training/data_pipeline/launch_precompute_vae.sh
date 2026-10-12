#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK:?Set TASK to the Hydra task whose dataset will be encoded}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the VAE latent cache directory}"
: "${DIFFSYNTH_MODEL_BASE_PATH:?Set DIFFSYNTH_MODEL_BASE_PATH to the external model weights}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
WORKERS="${WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
MASTER_PORT="${MASTER_PORT:-29871}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
exec torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
  data_pipeline/precompute_true_h32_vae_latents.py \
  --task "$TASK" --output-root "$OUTPUT_ROOT" \
  --workers "$WORKERS" --batch-size "$BATCH_SIZE" \
  --prefetch-factor "$PREFETCH_FACTOR" --episodewise-ffmpeg
