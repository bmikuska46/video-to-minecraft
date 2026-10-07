#!/usr/bin/env bash
# Turn a video into a Minecraft world.zip inside the pipeline-worker image.
#
#   scripts/video_to_world.sh VIDEO OUTPUT_DIR [options]
#
# Options are passed to scripts/video_to_world.py; see --help. The image comes
# from `docker compose -f infra/compose.yaml build pipeline-worker` (override
# with VTM_PIPELINE_IMAGE). The Paper JAR is taken from VTM_PAPER_JAR_PATH or
# infra/paper/.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${VTM_PIPELINE_IMAGE:-video-to-minecraft-pipeline-worker:latest}"
PAPER_JAR_NAME="paper-26.2-build.112-stable.jar"
PAPER_JAR="${VTM_PAPER_JAR_PATH:-${ROOT_DIR}/infra/paper/${PAPER_JAR_NAME}}"
CACHE_DIR="${XDG_CACHE_HOME:-${HOME}/.cache}/video-to-minecraft/paper-cache"

if [[ $# -lt 2 || "$1" == -* ]]; then
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    exec docker run --rm --volume "${ROOT_DIR}:/workspace:ro" "${IMAGE}" \
      python3 /workspace/scripts/video_to_world.py --help
  fi
  printf 'usage: %s VIDEO OUTPUT_DIR [options]   (--help for all options)\n' "$0" >&2
  exit 2
fi
VIDEO="$(realpath -- "$1")"
OUTPUT="$(realpath -m -- "$2")"
shift 2

if [[ ! -f "${VIDEO}" ]]; then
  printf 'Video does not exist: %s\n' "${VIDEO}" >&2
  exit 2
fi
if [[ ! -f "${PAPER_JAR}" ]]; then
  printf 'Paper JAR not found: %s\nDownload %s from https://papermc.io/downloads/paper\nand put it there, or set VTM_PAPER_JAR_PATH.\n' \
    "${PAPER_JAR}" "${PAPER_JAR_NAME}" >&2
  exit 2
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  printf 'Image %s is missing. Build it with:\n  docker compose -f infra/compose.yaml build pipeline-worker\n' "${IMAGE}" >&2
  exit 2
fi
mkdir -p -- "${OUTPUT}" "${CACHE_DIR}"
# The reconstruction probes the container format, so keep the extension.
NAME="$(basename -- "${VIDEO}")"
EXTENSION="mp4"
[[ "${NAME}" == *.* ]] && EXTENSION="${NAME##*.}"
INPUT="/input/video.${EXTENSION}"

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

# --mount with CSV quoting copes with spaces, colons and commas in host paths.
bind() { printf 'type=bind,"source=%s",target=%s%s' "${1//\"/\"\"}" "$2" "${3:+,readonly}"; }

# --init forwards Ctrl+C to the pipeline.
exec docker run --rm --init "${GPU_ARGS[@]}" \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp --env PYTHONUNBUFFERED=1 --env "VTM_SOURCE_VIDEO=${VIDEO}" \
  --mount "$(bind "${ROOT_DIR}" /workspace ro)" \
  --mount "$(bind "${VIDEO}" "${INPUT}" ro)" \
  --mount "$(bind "${OUTPUT}" /output)" \
  --mount "$(bind "${PAPER_JAR}" "/opt/paper/${PAPER_JAR_NAME}" ro)" \
  --mount "$(bind "${CACHE_DIR}" /cache/paper)" \
  --workdir /workspace \
  "${IMAGE}" \
  python3 /workspace/scripts/video_to_world.py "${INPUT}" /output \
    --paper-cache-dir /cache/paper --world-name "${NAME%.*}" "$@"
