#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON_BIN:-/usr/bin/python3}" "${ROOT}/scripts/visualize_manip26_eefxy_session.py" "$@"
