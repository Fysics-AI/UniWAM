#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/polymetis/bin/python}"
export PYTHONPATH="${ROOT}/direct:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Polymetis Python: ${PYTHON_BIN}" >&2; exit 3; }
exec "${PYTHON_BIN}" "${ROOT}/direct/uniwam_franka_direct_client.py" "$@"
