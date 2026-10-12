#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

source_setup() {
  local setup_path="$1"
  if [ -n "${setup_path}" ] && [ -f "${setup_path}" ]; then
    set +u
    # shellcheck disable=SC1090
    source "${setup_path}"
    set -u
  fi
}

source_setup "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
source_setup "${AGILEX_WS_SETUP:-$HOME/agilex_ws/install/setup.bash}"

if [ "${STOP_RANGER_BASE_NODE:-true}" = "true" ]; then
  pkill -f ranger_base_node 2>/dev/null || true
fi
if [ -f "${CAN_UP_SCRIPT:-$HOME/xw/00_can_up.sh}" ]; then
  bash "${CAN_UP_SCRIPT:-$HOME/xw/00_can_up.sh}"
fi

export PYTHONPATH="${DEPLOY_ROOT}/robot:${PYTHONPATH:-}"

CLAMP_ARGS=()
if [ "${BASE_CAN_DISABLE_COMMAND_CLAMP:-false}" = "true" ]; then
  CLAMP_ARGS+=(--disable-command-clamp)
fi

exec /usr/bin/python3 -m uniwam_piper_robot.direct_can_cmdvel_bridge \
  --can "${BASE_CAN_IFACE:-can0}" \
  --topic "${BASE_CMD_VEL_TOPIC:-/xw/cmd_vel_direct_can}" \
  --rate "${BASE_CAN_RATE:-30}" \
  --max-vx "${BASE_CAN_MAX_VX:-0.16}" \
  --max-wz "${BASE_CAN_MAX_WZ:-0.24}" \
  --timeout "${BASE_CAN_TIMEOUT:-0.25}" \
  --mode-refresh "${BASE_CAN_MODE_REFRESH:-1.0}" \
  "${CLAMP_ARGS[@]}" \
  "$@"
