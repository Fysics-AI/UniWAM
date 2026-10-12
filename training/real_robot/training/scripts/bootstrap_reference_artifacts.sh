#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVES="$ROOT/reference_archives"
ARTIFACT_GROUPS=(indices retrain_indices latent_remap retrain_latent_remap)

all_present=true
for group in "${ARTIFACT_GROUPS[@]}"; do
  [[ -d "$ROOT/reference/$group" ]] || all_present=false
done
if "$all_present"; then
  (cd "$ROOT" && sha256sum -c "$ARCHIVES/FILE_SHA256SUMS" >/dev/null)
  echo "UNIWAM_REFERENCE_ARTIFACTS_OK"
  exit 0
fi

for group in "${ARTIFACT_GROUPS[@]}"; do
  if [[ -e "$ROOT/reference/$group" ]]; then
    echo "Partial reference extraction detected at $ROOT/reference/$group" >&2
    exit 1
  fi
done

temporary="$(mktemp -d)"
trap 'rm -f "$temporary"/*.tar.gz; rmdir "$temporary"' EXIT
for group in "${ARTIFACT_GROUPS[@]}"; do
  parts=("$ARCHIVES/$group".part[0-9][0-9])
  [[ -f "${parts[0]}" ]] || { echo "Missing archive parts for $group" >&2; exit 1; }
  cat "${parts[@]}" > "$temporary/$group.tar.gz"
done
(cd "$temporary" && sha256sum -c "$ARCHIVES/SHA256SUMS" >/dev/null)
mkdir -p "$ROOT/reference"
for group in "${ARTIFACT_GROUPS[@]}"; do
  tar -xzf "$temporary/$group.tar.gz" -C "$ROOT/reference"
done
(cd "$ROOT" && sha256sum -c "$ARCHIVES/FILE_SHA256SUMS" >/dev/null)
echo "UNIWAM_REFERENCE_ARTIFACTS_OK"
