#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export UNIWAM_TRAINING_ROOT="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${DIFFSYNTH_MODEL_BASE_PATH:-}" ]]; then
  echo "Set DIFFSYNTH_MODEL_BASE_PATH to the local Wan/ActionDiT checkpoint directory." >&2
  exit 2
fi
export DIFFSYNTH_MODEL_BASE_PATH
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
unset TRANSFORMERS_CACHE
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

# Keep explicit launcher values authoritative while still loading credentials
# and unspecified defaults from the shared environment file.
CALLER_WANDB_ENABLED="${WANDB_ENABLED-}"
CALLER_WANDB_MODE="${WANDB_MODE-}"
CALLER_WANDB_PROJECT="${WANDB_PROJECT-}"
CALLER_WANDB_NAME="${WANDB_NAME-}"
CALLER_WANDB_ENTITY="${WANDB_ENTITY-}"
CALLER_WANDB_WORKSPACE="${WANDB_WORKSPACE-}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-${PROJECT_ROOT}/scripts/env/uniwam_wandb.env}"
if [ -f "${WANDB_ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${WANDB_ENV_FILE}"
fi
[[ -n "${CALLER_WANDB_ENABLED}" ]] && WANDB_ENABLED="${CALLER_WANDB_ENABLED}"
[[ -n "${CALLER_WANDB_MODE}" ]] && WANDB_MODE="${CALLER_WANDB_MODE}"
[[ -n "${CALLER_WANDB_PROJECT}" ]] && WANDB_PROJECT="${CALLER_WANDB_PROJECT}"
[[ -n "${CALLER_WANDB_NAME}" ]] && WANDB_NAME="${CALLER_WANDB_NAME}"
[[ -n "${CALLER_WANDB_ENTITY}" ]] && WANDB_ENTITY="${CALLER_WANDB_ENTITY}"
[[ -n "${CALLER_WANDB_WORKSPACE}" ]] && WANDB_WORKSPACE="${CALLER_WANDB_WORKSPACE}"
export WANDB_ENABLED="${WANDB_ENABLED:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_PROJECT="${WANDB_PROJECT:-UniWAM}"
export WANDB_NAME="${WANDB_NAME-}"
export WANDB_ENTITY="${WANDB_ENTITY-}"
export WANDB_WORKSPACE="${WANDB_WORKSPACE-}"
export WANDB_BASE_DIR="${WANDB_BASE_DIR:-${PROJECT_ROOT}/wandb}"
export WANDB_DIR="${WANDB_BASE_DIR}/runs"
export WANDB_CACHE_DIR="${WANDB_BASE_DIR}/cache"
export WANDB_DATA_DIR="${WANDB_BASE_DIR}/data"
export WANDB_ARTIFACT_DIR="${WANDB_BASE_DIR}/artifacts"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${WANDB_ARTIFACT_DIR}"

NPROC_PER_NODE="${1:?Usage: bash scripts/train_zero1.sh <nproc_per_node> [hydra_overrides...]}"
shift

EXTRA_ARGS=("$@")
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-configs/accelerate_configs/accelerate_zero1_ds.yaml}"

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

if ! is_integer "${NPROC_PER_NODE}" || ! is_integer "${NUM_MACHINES}" || ! is_integer "${MACHINE_RANK}"; then
  echo "Error: NPROC_PER_NODE (${NPROC_PER_NODE}), NUM_MACHINES (${NUM_MACHINES}), and MACHINE_RANK (${MACHINE_RANK}) must be integers." >&2
  exit 1
fi

TOTAL_NUM_PROCESSES="$((NPROC_PER_NODE * NUM_MACHINES))"
extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

TASK_BASENAME="train"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      ;;
  esac
done

if [[ -z "${RUN_ID:-}" ]]; then
  if (( NUM_MACHINES <= 1 )); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    RUN_ID_SYNC_TIMEOUT="${RUN_ID_SYNC_TIMEOUT:-180}"
    RUN_ID_SYNC_PORT="${RUN_ID_SYNC_PORT:-$((MAIN_PROCESS_PORT + 11))}"

    export RUN_ID_SYNC_HOST="${MAIN_PROCESS_IP}"
    export RUN_ID_SYNC_PORT
    export RUN_ID_SYNC_TIMEOUT
    export RUN_ID_SYNC_MACHINE_RANK="${MACHINE_RANK}"
    export RUN_ID_SYNC_NUM_MACHINES="${NUM_MACHINES}"
    export RUN_ID_SYNC_TASK_BASENAME="${TASK_BASENAME}"

    RUN_ID="$(
      python - <<'PY'
import datetime
import os
from datetime import timedelta

import torch.distributed as dist

host = os.environ["RUN_ID_SYNC_HOST"]
port = int(os.environ["RUN_ID_SYNC_PORT"])
timeout_s = int(os.environ["RUN_ID_SYNC_TIMEOUT"])
machine_rank = int(os.environ["RUN_ID_SYNC_MACHINE_RANK"])
num_machines = int(os.environ["RUN_ID_SYNC_NUM_MACHINES"])
task_basename = os.environ.get("RUN_ID_SYNC_TASK_BASENAME", "train")

store = dist.TCPStore(
    host_name=host,
    port=port,
    world_size=num_machines,
    is_master=(machine_rank == 0),
    timeout=timedelta(seconds=timeout_s),
)
key = f"run_id::{task_basename}"
if machine_rank == 0:
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    store.set(key, run_id)
run_id = store.get(key).decode("utf-8")
print(run_id)
PY
    )"

    echo "[run_id_sync] mode=tcpstore host=${RUN_ID_SYNC_HOST} port=${RUN_ID_SYNC_PORT} timeout_s=${RUN_ID_SYNC_TIMEOUT} run_id=${RUN_ID}"
  fi
fi

echo "[launch] nproc_per_node=${NPROC_PER_NODE} total_num_processes=${TOTAL_NUM_PROCESSES} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} main=${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT} run_id=${RUN_ID}"

accelerate launch \
  --config_file "${ACCELERATE_CONFIG_FILE}" \
  --num_processes "${TOTAL_NUM_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --same_network \
  --deepspeed_multinode_launcher standard \
  scripts/train.py \
  "output_dir=./runs/${TASK_BASENAME}/${RUN_ID}" \
  "wandb.enabled=${WANDB_ENABLED}" \
  "wandb.mode=${WANDB_MODE}" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=${WANDB_NAME:-${TASK_BASENAME}_${RUN_ID}}" \
  "${EXTRA_ARGS[@]}"
