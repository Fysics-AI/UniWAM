#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT for the external VAE latent cache}"
WORKERS="${WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
CHECKPOINT_EPISODE_INTERVAL="${CHECKPOINT_EPISODE_INTERVAL:-8}"
MASTER_PORT="${MASTER_PORT:-29871}"
LOG_DIR="${PROJECT_ROOT}/logs/precompute"
RUN_ID="${RUN_ID:-true_h32_vae_8gpu_$(date +%Y%m%d_%H%M%S)}"
TASK="${TASK:-uniwam_camera_frame_six_source_manip26_embodiment_stats_200k}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:?Set DIFFSYNTH_MODEL_BASE_PATH to the external Wan/ActionDiT checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export FASTWAM_TORCHCODEC_DECODER_CACHE_SIZE="${FASTWAM_TORCHCODEC_DECODER_CACHE_SIZE:-16}"

torchrun --standalone --nproc_per_node=8 --master_port="${MASTER_PORT}" \
  data_pipeline/precompute_true_h32_vae_latents.py \
  --task "${TASK}" \
  --output-root "${OUTPUT_ROOT}" \
  --workers "${WORKERS}" \
  --batch-size "${BATCH_SIZE}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --checkpoint-episode-interval "${CHECKPOINT_EPISODE_INTERVAL}" \
  --episodewise-ffmpeg \
  2>&1 | tee -a "${LOG_DIR}/${RUN_ID}.log"
