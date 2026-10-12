#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-${ROOT}/artifacts/uniwam_camera_frame_h32_robot_bundle.tar.gz}"
mkdir -p "$(dirname -- "${OUT}")"
NAME="uniwam_camera_frame_h32_robot_bundle"
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
mkdir -p "${STAGE}/${NAME}"
cp -a "${ROOT}/robot" "${ROOT}/direct" "${ROOT}/common" "${ROOT}/vendor" "${STAGE}/${NAME}/"
mkdir -p "${STAGE}/${NAME}/configs" "${STAGE}/${NAME}/scripts" "${STAGE}/${NAME}/docs"
cp "${ROOT}/configs/uniwam_robot_client_camera_frame_h32.yaml" "${STAGE}/${NAME}/configs/"
cp "${ROOT}/scripts/launch_uniwam_piper_client.sh" \
  "${ROOT}/scripts/launch_uniwam_franka_client.sh" \
  "${ROOT}/scripts/bootstrap_robot_python_deps.sh" \
  "${ROOT}/scripts/bootstrap_franka_direct_python_deps.sh" \
  "${ROOT}/scripts/visualize_manip26_eefxy_session.py" \
  "${ROOT}/scripts/visualize_manip26_eefxy_session.sh" \
  "${ROOT}/scripts/visualize_recent_manip26_eefxy.sh" \
  "${ROOT}/scripts/start_direct_can_cmdvel_bridge.sh" \
  "${ROOT}/scripts/start_direct_can_cmdvel_bridge_async_prefix12_rot6d.sh" \
  "${STAGE}/${NAME}/scripts/"
cp "${ROOT}/docs/UNIWAM_DEPLOYMENT.md" "${STAGE}/${NAME}/docs/"
tar --exclude='*/__pycache__' --exclude='*.pyc' -czf "${OUT}" -C "${STAGE}" "${NAME}"
(cd "$(dirname -- "${OUT}")" && sha256sum "$(basename -- "${OUT}")" > "$(basename -- "${OUT}").sha256")
echo "${OUT}"
