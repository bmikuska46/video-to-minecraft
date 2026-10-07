#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE="${1:-brick-facade-strafe}"
OUTPUT_NAME="${2:-${FIXTURE}}"
VIDEO="${ROOT_DIR}/fixtures/captures/${FIXTURE}/video.mp4"
OUTPUT="${ROOT_DIR}/artifacts/reconstructions/${OUTPUT_NAME}"
LOCK_FILE="${ROOT_DIR}/infra/containers/reconstruction-image.lock"

if [[ ! -f "${VIDEO}" ]]; then
  printf 'Fixture video is missing: %s\nRun: python3 scripts/generate_capture_fixtures.py\n' "${VIDEO}" >&2
  exit 2
fi

IMAGE_ID="$(sed -n 's/^image_id=//p' "${LOCK_FILE}")"
if [[ -z "${IMAGE_ID}" || "${IMAGE_ID}" == "UNBUILT" ]]; then
  printf 'Reconstruction image is not built. Run: scripts/build_reconstruction_image.sh\n' >&2
  exit 2
fi

if [[ -e "${OUTPUT}" ]]; then
  printf 'Output already exists; choose a new output name: %s\n' "${OUTPUT}" >&2
  exit 2
fi

mkdir -p "$(dirname -- "${OUTPUT}")"
GPU_ARGS=(--gpus all)
if [[ "${VTM_DOCKER_GPU_MODE:-toolkit}" == "devices" ]]; then
  CUDA_DRIVER="$(readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1)"
  PTX_DRIVER="$(readlink -f /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1)"
  GPU_ARGS=(
    --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm
    --volume "${CUDA_DRIVER}:/host-nvidia/libcuda.so.1:ro"
    --volume "${PTX_DRIVER}:/host-nvidia/libnvidia-ptxjitcompiler.so.1:ro"
    --env LD_LIBRARY_PATH=/host-nvidia:/usr/local/cuda/lib64
  )
fi
docker run --rm "${GPU_ARGS[@]}" \
  --user "$(id -u):$(id -g)" \
  --volume "${ROOT_DIR}:/workspace" \
  --workdir /workspace \
  "${IMAGE_ID}" \
  python3 services/reconstruction/reconstruct_fixture.py \
    "/workspace/fixtures/captures/${FIXTURE}/video.mp4" \
    "/workspace/artifacts/reconstructions/${OUTPUT_NAME}"
