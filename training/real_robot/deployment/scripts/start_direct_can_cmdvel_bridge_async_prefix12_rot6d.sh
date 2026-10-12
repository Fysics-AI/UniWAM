#!/usr/bin/env bash
set -euo pipefail

# The edge policy and cloud service deliberately leave model output unclamped.
# Keep the bundled vx/wz CAN bridge consistent unless the caller explicitly
# requests its historical clamp behavior.
DEPLOY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export BASE_CAN_DISABLE_COMMAND_CLAMP="${BASE_CAN_DISABLE_COMMAND_CLAMP:-true}"
exec "${DEPLOY_ROOT}/scripts/start_direct_can_cmdvel_bridge.sh" "$@"
