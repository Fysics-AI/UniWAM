#!/usr/bin/env bash
set -euo pipefail

: "${TASK_PROMPT:?Set TASK_PROMPT to the exact inference task sentence}"
: "${UNIWAM_TRAINING_ROOT:?Set UNIWAM_TRAINING_ROOT to the repository training directory}"
: "${UNIWAM_TEXT_CACHE_DIR:?Set UNIWAM_TEXT_CACHE_DIR to the text embedding cache}"
: "${DIFFSYNTH_MODEL_BASE_PATH:?Set DIFFSYNTH_MODEL_BASE_PATH to model weights}"

DEPLOY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${DEPLOY_ROOT}/configs/uniwam_cloud_parent_async.yaml}"
export PYTHONPATH="${UNIWAM_TRAINING_ROOT}/src:${UNIWAM_TRAINING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" "${DEPLOY_ROOT}/scripts/precompute_inference_prompt.py" \
  --config "$CONFIG" --task-prompt "$TASK_PROMPT" \
  --robot-type "${ROBOT_TYPE:-piper}" --output "$UNIWAM_TEXT_CACHE_DIR"
