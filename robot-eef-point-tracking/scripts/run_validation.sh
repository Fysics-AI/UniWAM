#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"
PYTHON="${PYTHON:-python3}"
GEOMETRY_ROOT="${EEF_FLANGE_GEOMETRY_ROOT:-$PACKAGE_ROOT/outputs/flange_geometry}"
FINAL_ROOT="${EEF_FINAL_ROOT:-$PACKAGE_ROOT/outputs/eef_tracks}"

"$PYTHON" src/validate_labels.py \
  --manifest "$GEOMETRY_ROOT/geometry_manifest.jsonl" \
  --output-root "$FINAL_ROOT" \
  --report "$FINAL_ROOT/full_validation.json" \
  --workers "${VALIDATION_WORKERS:-16}" \
  --verify-sha256 \
  --require-complete
