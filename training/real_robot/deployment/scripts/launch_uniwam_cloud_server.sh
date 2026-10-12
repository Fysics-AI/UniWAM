#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT}/configs/uniwam_cloud_parent_async.yaml}"
PROJECT_ROOT="${UNIWAM_TRAINING_ROOT:?Set UNIWAM_TRAINING_ROOT to the release training directory}"
RUN_ROOT="${PROJECT_ROOT}/runs/uniwam_camera_frame_six_source_manip26_embodiment_stats_200k"
if [[ -z "${CHECKPOINT:-}" ]]; then
  CHECKPOINT="$(find "${RUN_ROOT}" -mindepth 4 -maxdepth 4 -type f -path '*/checkpoints/weights/step_*.pt' -size +12000000000c -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {sub(/^[^ ]+ /, ""); print}')"
fi
[[ -n "${CHECKPOINT:-}" && -f "${CHECKPOINT}" ]] || { echo "Set CHECKPOINT to a complete manip26 checkpoint." >&2; exit 2; }
[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
if [[ -n "${CONDA_SH:-}" && -f "$CONDA_SH" ]]; then
  source "$CONDA_SH"
  conda activate "${CONDA_ENV:-fastwam}"
fi
export PYTHONPATH="${ROOT}/cloud:${ROOT}/common:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:?set DIFFSYNTH_MODEL_BASE_PATH to the local Wan/ActionDiT checkpoint directory}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
THREADS="${FASTWAM_CPU_THREADS:-8}"
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" OPENBLAS_NUM_THREADS="${THREADS}" NUMEXPR_NUM_THREADS="${THREADS}"
echo "[UniWAM-cloud] checkpoint=${CHECKPOINT} physical_cuda=${CUDA_VISIBLE_DEVICES} logical_device=cuda:0"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" -m uniwam_piper_cloud.server_rot6d --config "${CONFIG}" checkpoint="${CHECKPOINT}" "$@"
