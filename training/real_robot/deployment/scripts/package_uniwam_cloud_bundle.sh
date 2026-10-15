#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-${ROOT}/artifacts/uniwam_cloud_bundle.tar.gz}"
STAGE="$(mktemp -d)"
trap 'rm -rf -- "${STAGE}"' EXIT
NAME="uniwam_cloud_bundle"
mkdir -p "${STAGE}/${NAME}/configs" "${STAGE}/${NAME}/scripts"
cp -a "${ROOT}/cloud" "${ROOT}/common" "${STAGE}/${NAME}/"
cp "${ROOT}/configs/uniwam_cloud_customer_async.yaml" "${STAGE}/${NAME}/configs/"
cp "${ROOT}/scripts/launch_uniwam_cloud_server.sh" \
  "${ROOT}/scripts/precompute_inference_prompt.py" \
  "${ROOT}/scripts/precompute_inference_prompt_async_prefix12_rot6d.sh" \
  "${STAGE}/${NAME}/scripts/"
cp "${ROOT}/docs/UNIWAM_DEPLOYMENT.md" "${STAGE}/${NAME}/README.md"
mkdir -p "$(dirname -- "${OUT}")"
tar -czf "${OUT}" -C "${STAGE}" "${NAME}"
(cd "$(dirname -- "${OUT}")" && sha256sum "$(basename -- "${OUT}")" > "$(basename -- "${OUT}").sha256")
echo "${OUT}"
