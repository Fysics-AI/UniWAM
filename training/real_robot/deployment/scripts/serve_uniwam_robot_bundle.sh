#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${BUNDLE_HTTP_PORT:-18096}"
mkdir -p "${ROOT}/logs"
FILE="${BUNDLE_FILE:-uniwam_camera_frame_h32_robot_bundle.tar.gz}"
[[ -f "${ROOT}/artifacts/${FILE}" ]] || {
  echo "Missing bundle: ${ROOT}/artifacts/${FILE}" >&2
  echo "Available bundles:" >&2
  find "${ROOT}/artifacts" -maxdepth 1 -type f -name '*.tar.gz' -printf '%f\n' | sort >&2
  exit 2
}
echo "Serving ${ROOT}/artifacts/${FILE} on 0.0.0.0:${PORT}" >&2
exec python3 -m http.server "${PORT}" --bind 0.0.0.0 --directory "${ROOT}/artifacts"
