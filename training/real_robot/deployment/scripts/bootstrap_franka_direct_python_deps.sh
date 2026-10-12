#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/polymetis/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || {
  echo "Missing Polymetis Python: ${PYTHON_BIN}" >&2
  exit 2
}

PYTHONNOUSERSITE=1 "${PYTHON_BIN}" - <<'PY'
required = {
    "numpy": "numpy",
    "PIL": "Pillow",
    "scipy": "scipy",
    "msgpack": "msgpack",
    "pyrealsense2": "pyrealsense2",
    "polymetis": "polymetis",
}
missing = []
for module, package in required.items():
    try:
        __import__(module)
    except Exception as exc:
        missing.append(f"{package} ({type(exc).__name__}: {exc})")
if missing:
    raise SystemExit("Missing direct-runtime dependencies: " + "; ".join(missing))
print("FRANKA_DIRECT_DEPS_OK")
PY

echo "Use this runtime: ${PYTHON_BIN}"
