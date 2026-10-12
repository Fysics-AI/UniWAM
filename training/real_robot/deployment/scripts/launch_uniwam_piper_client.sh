#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${DEPLOY_ROOT}/configs/uniwam_robot_client_camera_frame_h32.yaml}"
source_setup() {
  local path="$1"
  if [[ -n "${path}" && -f "${path}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${path}"
    set -u
  fi
}
source_setup "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
source_setup "${PIPER_WS_SETUP:-$HOME/piper_ros/install/setup.bash}"
source_setup "${AGILEX_WS_SETUP:-$HOME/agilex_ws/install/setup.bash}"
export PYTHONPATH="${DEPLOY_ROOT}/robot:${DEPLOY_ROOT}/common${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
DEFAULT_PYTHON=/usr/bin/python3
[[ -x "${DEPLOY_ROOT}/.robot_py310/bin/python" ]] && DEFAULT_PYTHON="${DEPLOY_ROOT}/.robot_py310/bin/python"
if ! "${PYTHON_BIN:-${DEFAULT_PYTHON}}" -c 'import numpy, omegaconf, PIL, msgpack' >/dev/null 2>&1; then
  echo "Robot Python dependencies are missing from ${PYTHON_BIN:-${DEFAULT_PYTHON}}." >&2
  echo "Run: bash scripts/bootstrap_robot_python_deps.sh" >&2
  exit 2
fi
exec "${PYTHON_BIN:-${DEFAULT_PYTHON}}" -m uniwam_piper_robot.ros2_client_async_prefix12_rot6d --config "${CONFIG}" "$@"
