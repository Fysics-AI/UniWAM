#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${NODE_RANK:?Set NODE_RANK=0 or 1 on the two hosts}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 host address}"
: "${DIFFSYNTH_MODEL_BASE_PATH:?Set DIFFSYNTH_MODEL_BASE_PATH to local model weights}"

export NNODES=2
export NODE_RANK MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29500}"
export UNIWAM_TRAINING_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_ENABLED="${WANDB_ENABLED:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
TASK="${TASK:-uniwam_camera_frame_six_source_manip26_embodiment_stats_200k}"
if [[ "$TASK" == uniwam_six_source_retrain_200k ]]; then
  echo "The 2026-09-28 retrain task used different mobile windows and malformed camera-frame Rot6D; it is not the six-source parent reproduction." >&2
  exit 2
fi
if [[ "$TASK" == uniwam_camera_frame_six_source_manip26_embodiment_stats_200k ]]; then
  : "${UNIWAM_PARENT_RECOMPUTED_STATS:?Recompute Piper stats from the selected six-source data before launching the parent reproduction}"
  python "$ROOT/scripts/preflight_parent_reproduction.py" \
    --recomputed-stats "$UNIWAM_PARENT_RECOMPUTED_STATS"
fi

resume_args=()
if [[ -n "${RESUME_STATE:-}" ]]; then
  resume_args+=("resume=${RESUME_STATE}")
elif [[ -n "${RESUME_WEIGHTS:-}" ]]; then
  resume_args+=("resume=${RESUME_WEIGHTS}" "resume_step=${RESUME_STEP:-0}")
fi

exec bash "$ROOT/scripts/train_zero1.sh" 8 \
  --config-name train "task=${TASK}" \
  batch_size=32 gradient_accumulation_steps=1 max_steps=200000 \
  model.compile_training_denoise=true model.compile_action_infer=true model.compile_vae_infer=true \
  "${resume_args[@]}" "$@"
