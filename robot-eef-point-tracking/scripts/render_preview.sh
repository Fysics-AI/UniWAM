#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <main-view-video.mp4> <episode-tracks.npz> <output.mp4> [stride]" >&2
  exit 2
fi
VIDEO="$1"
TRACKS="$2"
OUTPUT="$3"
STRIDE="${4:-2}"
"${PYTHON:-python3}" src/render_preview.py --video "$VIDEO" --tracks "$TRACKS" \
  --output "$OUTPUT" --stride "$STRIDE" --show-j6-reference
