#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON=/usr/bin/python3
[[ -x "${ROOT}/.robot_py310/bin/python" ]] && DEFAULT_PYTHON="${ROOT}/.robot_py310/bin/python"
exec "${PYTHON_BIN:-${DEFAULT_PYTHON}}" "${ROOT}/scripts/visualize_manip26_eefxy_visible_session.py" "$@"
