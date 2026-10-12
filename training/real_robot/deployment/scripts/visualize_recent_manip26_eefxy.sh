#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RECORDS="${1:-${HOME}/uniwam_manip26_eefxy_visible_records}"
FPS="${FPS:-30}"
exec bash "${ROOT}/scripts/visualize_manip26_eefxy_session.sh" --records "${RECORDS}" --fps "${FPS}"
