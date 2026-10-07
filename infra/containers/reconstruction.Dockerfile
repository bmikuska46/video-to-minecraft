# syntax=docker/dockerfile:1.7

ARG CUDA_DEVEL_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04@sha256:e542739fcaa4f45da5add8c4cf5769783a61628b2518304a5bbe4ace468b8c8f
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.9.1-runtime-ubuntu24.04@sha256:1199283a7fd5fc7dcb472ea4c6b6d8eb378ef8c0b9aae5e731491c01a219ca43

FROM ${CUDA_DEVEL_IMAGE} AS builder

ARG COLMAP_COMMIT=9c23f6942fe69962e06030905e77067c8673382f
ARG CUDA_ARCHITECTURES=89
ARG BUILD_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates ccache cmake git ninja-build \
        libblas-dev liblapack-dev \
        libboost-graph-dev libboost-program-options-dev libboost-system-dev \
        libcgal-dev libceres-dev libcurl4-openssl-dev libeigen3-dev \
        libglew-dev libgoogle-glog-dev libgmock-dev libgtest-dev libmetis-dev libmkl-full-dev \
        libopenimageio-dev openimageio-tools libqt6opengl6-dev libqt6openglwidgets6 \
        libsqlite3-dev libssl-dev qt6-base-dev && \
    rm -rf /var/lib/apt/lists/* && mkdir -p /usr/include/opencv4

RUN git clone --filter=blob:none https://github.com/colmap/colmap.git /src/colmap && \
    cd /src/colmap && git checkout --detach "${COLMAP_COMMIT}" && \
    test "$(git rev-parse HEAD)" = "${COLMAP_COMMIT}"

RUN cmake -S /src/colmap -B /src/colmap/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DCMAKE_INSTALL_PREFIX=/opt/colmap \
        -DGUI_ENABLED=OFF \
        -DTESTS_ENABLED=OFF && \
    cmake --build /src/colmap/build --parallel "${BUILD_JOBS}" && \
    cmake --install /src/colmap/build

FROM maven:3.9.11-eclipse-temurin-25 AS worldgen-build
WORKDIR /src
COPY pom.xml ./
COPY packages/contracts/java packages/contracts/java
COPY services/worldgen services/worldgen
RUN mvn -B -pl services/worldgen -am package -DskipTests

FROM eclipse-temurin:25-jdk-noble AS java-runtime

FROM ${CUDA_RUNTIME_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg python3 python3-pip python3-venv \
        libboost-program-options1.83.0 libc6 libceres4t64 libcurl4t64 \
        libgcc-s1 libglew2.2 libgl1 libgoogle-glog0v6t64 libmetis5 \
        libmkl-core libmkl-intel-lp64 libmkl-intel-thread libmkl-locale libmkl-sequential \
        libomp5 libopenimageio2.4t64 libopengl0 libssl3t64 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/colmap/ /usr/local/
COPY --from=java-runtime /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk \
    PATH=/opt/venv/bin:/opt/java/openjdk/bin:${PATH} \
    PYTHONPATH=/workspace/services/reconstruction
# Depth-completion stack and pinned weights (see depth_completion.py) live in their
# own layers, before the service code, so code edits do not re-download ~7 GB.
RUN python3 -m venv --system-site-packages /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir \
        "torch>=2.14,<2.15" "torchvision>=0.29,<0.30" "transformers>=5.17,<6"
ENV HF_HOME=/opt/models
RUN /opt/venv/bin/python -c "from huggingface_hub import snapshot_download; \
snapshot_download('depth-anything/Depth-Anything-V2-Large-hf', revision='7581137eff8d4e94f6e796d3baea0e9fa79b22d2', \
allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors'])"
COPY services/api /build/api
COPY services/reconstruction /build/reconstruction
RUN /opt/venv/bin/pip install --no-cache-dir /build/api /build/reconstruction && \
    rm -rf /build
# Workers must never fetch models at run time.
ENV HF_HUB_OFFLINE=1
COPY services/worldgen/worldgen_runner.py /opt/worldgen/worldgen_runner.py
COPY services/worldgen/level_nbt.py /opt/worldgen/level_nbt.py
COPY services/worldgen/server-template /opt/worldgen/server-template
COPY --from=worldgen-build /src/services/worldgen/target/worldgen-plugin-0.1.0.jar /opt/worldgen/worldgen-plugin.jar
LABEL org.opencontainers.image.source="https://github.com/colmap/colmap" \
      org.opencontainers.image.version="4.0.4" \
      org.opencontainers.image.revision="9c23f6942fe69962e06030905e77067c8673382f"
WORKDIR /workspace
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
    CMD python3 -c "import json; assert json.load(open('/tmp/vtm-worker-ready.json'))['ready'] is True"
ENTRYPOINT []
CMD ["celery", "-A", "app.pipeline:celery_app", "worker", "--loglevel=INFO", "--concurrency=1"]
