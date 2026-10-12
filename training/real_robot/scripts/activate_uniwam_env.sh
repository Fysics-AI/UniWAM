#!/usr/bin/env bash
# Configure a UniWAM workspace for training and deployment.
#
# Edit the values in the PATHS block once, then run:
#   source scripts/activate_uniwam_env.sh
#
# This file must be sourced because a child shell cannot export variables into
# the caller's shell. It intentionally contains placeholders, not repository-
# specific paths or credentials.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Usage: source scripts/activate_uniwam_env.sh" >&2
  exit 2
fi

UNIWAM_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------- PATHS -----------------------------------
export REPO="${REPO:-$UNIWAM_REPO_ROOT}"
export UNIWAM_TRAINING_ROOT="${UNIWAM_TRAINING_ROOT:-$REPO/training}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/absolute/path/to/Wan-and-ActionDiT-checkpoints}"
export UNIWAM_INITIALIZATION_CHECKPOINT="${UNIWAM_INITIALIZATION_CHECKPOINT:-/absolute/path/to/initialization-checkpoint.pt}"
# The training config currently reads this stable variable name. Edit the
# initialization-checkpoint path above; this alias keeps the Hydra contract.
export UNIWAM_PARENT_CHECKPOINT="$UNIWAM_INITIALIZATION_CHECKPOINT"
export UNIWAM_TEXT_CACHE_DIR="${UNIWAM_TEXT_CACHE_DIR:-/absolute/path/to/text-embedding-cache}"
export UNIWAM_CUSTOM_DATA="${UNIWAM_CUSTOM_DATA:-/absolute/path/to/customer_camera_frame_dataset}"
export UNIWAM_CUSTOM_SOURCE_NAME="${UNIWAM_CUSTOM_SOURCE_NAME:-$(basename -- "$UNIWAM_CUSTOM_DATA")}"
export UNIWAM_CUSTOM_WINDOW_INDEX="${UNIWAM_CUSTOM_WINDOW_INDEX:-/absolute/path/to/customer_h32_window_index.parquet}"
export UNIWAM_CUSTOM_LATENT_ROOT="${UNIWAM_CUSTOM_LATENT_ROOT:-/absolute/path/to/customer_vae_latents}"
export UNIWAM_PIPER_URDF="${UNIWAM_PIPER_URDF:-/absolute/path/to/robot_description.urdf}"
export UNIWAM_CLOUD_TASK="${UNIWAM_CLOUD_TASK:-uniwam_customer_piper_manip_sft}"
export UNIWAM_CLOUD_SOURCE_INDEX="${UNIWAM_CLOUD_SOURCE_INDEX:-0}"
export UNIWAM_CLOUD_AUX_WEIGHT="${UNIWAM_CLOUD_AUX_WEIGHT:-0.0}"

# Optional: enable only when the selected task/config requires a separate
# EEF-XY sidecar. Leave it unset for a task without point-tracking labels.
export UNIWAM_EEF_SIDECAR_ROOT="${UNIWAM_EEF_SIDECAR_ROOT:-}"

# ----------------------------- CHECKS ---------------------------------
_uniwam_placeholder() {
  case "$1" in
    /absolute/path/to/*) return 0 ;;
    *) return 1 ;;
  esac
}

_uniwam_require_dir() {
  local name="$1" path="$2"
  if _uniwam_placeholder "$path" || [[ ! -d "$path" || ! -r "$path" ]]; then
    echo "UniWAM: set $name to a readable directory (currently '$path')." >&2
    return 1
  fi
}

_uniwam_require_file() {
  local name="$1" path="$2"
  if _uniwam_placeholder "$path" || [[ ! -f "$path" || ! -r "$path" ]]; then
    echo "UniWAM: set $name to a readable file (currently '$path')." >&2
    return 1
  fi
}

_uniwam_require_dir UNIWAM_TRAINING_ROOT "$UNIWAM_TRAINING_ROOT" || return 1
_uniwam_require_dir DIFFSYNTH_MODEL_BASE_PATH "$DIFFSYNTH_MODEL_BASE_PATH" || return 1
_uniwam_require_file UNIWAM_INITIALIZATION_CHECKPOINT "$UNIWAM_INITIALIZATION_CHECKPOINT" || return 1
# The dataset, window index, text cache, and VAE cache are outputs of the
# preparation steps and need not exist when this file is first sourced.
for _uniwam_output_name in UNIWAM_TEXT_CACHE_DIR UNIWAM_CUSTOM_DATA UNIWAM_CUSTOM_WINDOW_INDEX UNIWAM_CUSTOM_LATENT_ROOT; do
  _uniwam_output_path="${!_uniwam_output_name}"
  if _uniwam_placeholder "$_uniwam_output_path"; then
    echo "UniWAM: replace the placeholder for $_uniwam_output_name." >&2
    return 1
  fi
done
if [[ -n "$UNIWAM_EEF_SIDECAR_ROOT" ]]; then
  _uniwam_require_dir UNIWAM_EEF_SIDECAR_ROOT "$UNIWAM_EEF_SIDECAR_ROOT" || return 1
fi

export PYTHONPATH="$UNIWAM_TRAINING_ROOT/src:$UNIWAM_TRAINING_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export WANDB_ENABLED="${WANDB_ENABLED:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

echo "UniWAM environment loaded: $UNIWAM_CUSTOM_SOURCE_NAME"
