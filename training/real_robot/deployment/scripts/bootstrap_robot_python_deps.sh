#!/usr/bin/env bash
set -euo pipefail

# The ROS executor must run under Python 3.10, matching ROS 2 Humble. Keep its
# pure-Python dependencies in a venv that can still see the system ROS paths.
DEPLOY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-/usr/bin/python3}"
VENV_DIR="${ROBOT_PYTHON_VENV:-${DEPLOY_ROOT}/.robot_py310}"

if [ ! -x "${SYSTEM_PYTHON}" ]; then
  printf 'System Python is unavailable: %s\n' "${SYSTEM_PYTHON}" >&2
  exit 2
fi

"${SYSTEM_PYTHON}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Expected ROS Humble Python 3.10, got {sys.version.split()[0]}")
PY

if ! "${SYSTEM_PYTHON}" -m ensurepip --version >/dev/null 2>&1; then
  printf 'Missing Python venv support. Install it once with:\n  sudo apt-get install -y python3.10-venv\n' >&2
  exit 2
fi

if [ -d "${VENV_DIR}" ] && [ ! -x "${VENV_DIR}/bin/pip" ]; then
  printf 'Found an incomplete venv at %s. Remove it, then rerun this script.\n' "${VENV_DIR}" >&2
  exit 2
fi

"${SYSTEM_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
# The launcher disables the user site for deterministic ROS imports. Do not
# let pip satisfy dependencies from ~/.local; install private copies in venv.
PYTHONNOUSERSITE=1 "${VENV_DIR}/bin/python" -m pip install \
  --disable-pip-version-check --ignore-installed --no-user \
  'omegaconf==2.3.0' \
  'Pillow>=9,<12' \
  'msgpack>=1,<2'

PYTHONNOUSERSITE=1 "${VENV_DIR}/bin/python" - <<'PY'
import msgpack
import numpy
import omegaconf
import PIL
print(
    "ROBOT_PYTHON_DEPS_OK "
    f"omegaconf={omegaconf.__version__} pillow={PIL.__version__} "
    f"msgpack={msgpack.version} numpy={numpy.__version__}"
)
PY

printf 'Use robot Python: %s\n' "${VENV_DIR}/bin/python"
