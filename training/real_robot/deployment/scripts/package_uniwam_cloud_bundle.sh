#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-${ROOT}/artifacts/uniwam_camera_frame_200k_cloud_bundle.tar.gz}"
STAGE="$(mktemp -d)"
NAME="uniwam_camera_frame_200k_cloud_bundle"
mkdir -p "${STAGE}/${NAME}/configs" "${STAGE}/${NAME}/scripts"
cp -a "${ROOT}/cloud" "${ROOT}/common" "${STAGE}/${NAME}/"
cp "${ROOT}/configs/uniwam_source1_paired.yaml" \
  "${ROOT}/configs/uniwam_source3_color_manip_only.yaml" \
  "${ROOT}/configs/uniwam_source4_ordered_color_manip_only.yaml" \
  "${STAGE}/${NAME}/configs/"
cp "${ROOT}/scripts/launch_uniwam_cloud_server.sh" \
  "${ROOT}/scripts/precompute_inference_prompt_async_prefix12_rot6d.sh" \
  "${STAGE}/${NAME}/scripts/"
cp "${ROOT}/README.md" "${STAGE}/${NAME}/README.md"
mkdir -p "$(dirname -- "${OUT}")"
tar -czf "${OUT}" -C "${STAGE}" "${NAME}"
(cd "$(dirname -- "${OUT}")" && sha256sum "$(basename -- "${OUT}")" > "$(basename -- "${OUT}").sha256")
echo "${OUT}"
