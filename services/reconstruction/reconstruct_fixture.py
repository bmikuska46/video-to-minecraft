#!/usr/bin/env python3
"""Run one video fixture through FFmpeg and COLMAP without the API service."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# "object": one thing orbited in a short arc, later cropped with a box. "scene": a
# room or several objects, walked along for longer and kept uncut, so it needs a
# longer clip and more frames to keep neighbouring views overlapping.
SCAN_TYPES = ("object", "scene")
MAX_DURATION_SECONDS = {"object": 60.25, "scene": 300.25}
MIN_DURATION_SECONDS = 1.0
MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024
PIPELINE_VERSION = "reconstruction-fixture-v3"
MIN_REGISTERED_FRAMES = 15
MIN_REGISTRATION_RATIO = 0.60
MAX_MEDIAN_REPROJECTION_ERROR_PIXELS = 2.0
MIN_BASELINE_TO_SCENE_DIAGONAL = 0.02
MAX_TRAJECTORY_STEP_RATIO = 8.0
# Second-smallest eigenvalue of the mean right-axis outer product; below it the
# camera barely turned and the right axes cannot pin down up on their own.
MIN_RIGHT_VECTOR_SPREAD = 0.02
# At most this share of registered frames may be dropped as pose outliers before a
# trajectory break is treated as real.
MAX_TRAJECTORY_OUTLIER_FRACTION = 0.1
CANDIDATE_FRAME_RATE = 8.0
MIN_SELECTED_FRAMES = 30
MAX_SELECTED_FRAMES = {"object": 240, "scene": 600}
# Phones commonly record HEVC, 10-bit HDR or slightly overlong clips. Frame
# extraction decodes them directly and keeps only this much of the clip.
NORMALIZED_MAX_DURATION_SECONDS = {"object": 60.0, "scene": 300.0}
# Dense stereo tuned for weakly textured objects (smooth plastic, dark rugs). At
# 960 px an 11-sample window with step 2 spans ~38 px of the full-resolution
# frame; on a white air purifier this raised valid depth on its smooth faces from
# 1% to 39% and roughly halved PatchMatch time versus COLMAP's defaults.
DENSE_MAX_IMAGE_SIZE = 960
PATCH_MATCH_WINDOW_RADIUS = 9
PATCH_MATCH_WINDOW_STEP = 2
PATCH_MATCH_FILTER_MIN_NCC = 0.05
# COLMAP samples a fixed number of source views per pixel, so run time scales with
# iterations x views, not with the source list. Three iterations over the 10
# best-overlapping sources halved PatchMatch time (754 s -> 401 s on 60 frames)
# while keeping 37-39% vs 39% valid depth on the weakly textured test object.
PATCH_MATCH_NUM_ITERATIONS = 3
PATCH_MATCH_SOURCE_IMAGES = 10
# "detailed": PatchMatch + monocular hole filling. "fast": no PatchMatch; every
# frame's depth is monocular, aligned to its SfM points (~13 min -> ~1.5 min on an
# RTX 4060), with softer edges and no measured (observed) points.
PROCESSING_MODES = ("detailed", "fast")
# Room scans are dominated by plain painted walls and ceilings. COLMAP's default
# SIFT peak threshold (0.0067) finds almost no features on them, so frames facing
# a wall fail to register and the reconstruction keeps only part of the room (159
# of 180 frames and roughly half of the reference bedroom). A lower threshold and
# relaxed absolute-pose thresholds let those frames register.
SCENE_SIFT_PEAK_THRESHOLD = 0.002
# Exhaustive matching compares every pair of frames, so its cost grows with the
# square of the frame count: 163 s for the 250 frames of the reference room walk.
# Video frames only overlap with nearby frames, so longer captures match each
# frame against its SEQUENTIAL_OVERLAP neighbours plus frames 2, 4, 8, ... away
# (13 s on the same room, with as many frames registered). Short object orbits
# keep exhaustive matching, which is cheap there and links the arc's two ends.
EXHAUSTIVE_MATCHING_MAX_FRAMES = 100
SEQUENTIAL_OVERLAP = 20
# Global bundle adjustment after every 40% (not 10%) growth of the model: on the
# reference room's 250 frames the incremental mapper took 55-61 s instead of 89 s,
# with the same 240 frames and ~39,080 points, mean reprojection error 0.879 vs
# 0.875 px. (Mapper.ba_global_ignore_redundant_points3D was faster still, but on
# one of two runs of the same video it kept only 14k of the 39k points, starving
# fast mode's depth alignment of SfM anchors.) COLMAP's global mapper (GLOMAP)
# with one bundle adjustment round took 50-58 s instead of the incremental
# mapper's 54 s plus 13-23 s of SPARSE_EXTENSION, but its models had 32k points
# and pose outliers (235-239 usable frames instead of 240-249), so depth aligned
# in only 181-189 of the frames instead of 220-227; with its default three
# rounds it took 78 s. On the 60-frame barn orbit it took 82-93 s against 50 s.
# Its per-point error field is in normalized image units (~0.0008 for 0.74 px).
MAPPER_SPEED_OPTIONS = (
    "--Mapper.ba_global_frames_ratio", "1.4",
    "--Mapper.ba_global_points_ratio", "1.4",
)
SCENE_MAPPER_OPTIONS = (
    "--Mapper.abs_pose_min_num_inliers", "15",
    "--Mapper.abs_pose_min_inlier_ratio", "0.15",
    "--Mapper.max_reg_trials", "5",
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def settings_hash(settings: dict[str, Any]) -> str:
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class StageManifest:
    """Atomically persist enough state to audit or diagnose one reconstruction."""

    def __init__(self, path: Path, settings: dict[str, Any]) -> None:
        self.path = path
        self.active_stage: dict[str, Any] | None = None
        self.active_started: float | None = None
        self.data: dict[str, Any] = {
            "schemaVersion": 1,
            "pipelineVersion": PIPELINE_VERSION,
            "settingsHash": settings_hash(settings),
            "settings": settings,
            "status": "RUNNING",
            "startedAt": utc_timestamp(),
            "completedAt": None,
            "source": {},
            "metrics": {},
            "stages": [],
            "failure": None,
        }
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.path)

    def start(self, name: str) -> None:
        if self.active_stage is not None:
            raise RuntimeError(f"stage {self.active_stage['name']} is still running")
        self.active_stage = {
            "name": name,
            "status": "RUNNING",
            "startedAt": utc_timestamp(),
            "completedAt": None,
            "durationMs": None,
            "metrics": {},
            "artifacts": [],
        }
        self.active_started = time.monotonic()
        self.data["stages"].append(self.active_stage)
        self.write()

    def complete(
        self,
        *,
        metrics: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
    ) -> None:
        if self.active_stage is None or self.active_started is None:
            raise RuntimeError("no stage is running")
        self.active_stage.update(
            status="COMPLETED",
            completedAt=utc_timestamp(),
            durationMs=round((time.monotonic() - self.active_started) * 1000),
            metrics=metrics or {},
            artifacts=artifacts or [],
        )
        self.active_stage = None
        self.active_started = None
        self.write()

    def fail(self, error: Exception) -> None:
        failure = {"type": type(error).__name__, "message": str(error)}
        if self.active_stage is not None and self.active_started is not None:
            self.active_stage.update(
                status="FAILED",
                completedAt=utc_timestamp(),
                durationMs=round((time.monotonic() - self.active_started) * 1000),
                failure=failure,
            )
        self.active_stage = None
        self.active_started = None
        self.data.update(status="FAILED", completedAt=utc_timestamp(), failure=failure)
        self.write()

    def finish(self, *, source: dict[str, Any], metrics: dict[str, Any]) -> None:
        if self.active_stage is not None:
            raise RuntimeError(f"stage {self.active_stage['name']} is still running")
        self.data.update(
            status="COMPLETED",
            completedAt=utc_timestamp(),
            source=source,
            metrics=metrics,
        )
        self.write()


def run(arguments: list[str], log_path: Path, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    started = time.monotonic()
    printable = " ".join(arguments)
    print(f"+ {printable}", flush=True)
    result = subprocess.run(arguments, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout or "")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}) after {elapsed:.1f}s; see {log_path}")
    if not capture and result.stdout:
        print(result.stdout, end="")
    return result


def start_preloaded(arguments: list[str], log_path: Path) -> subprocess.Popen[str]:
    """Start a stage that loads its model now and waits for "start" on stdin (``--wait-for-start``)."""
    print(f"+ {' '.join(arguments)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        return subprocess.Popen(arguments, text=True, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)


def finish_preloaded(process: subprocess.Popen[str], log_path: Path) -> None:
    """Let a preloaded stage run, wait for it, and fail like run() does."""
    started = time.monotonic()
    try:
        process.stdin.write("start\n")
        process.stdin.close()
    except BrokenPipeError:
        pass  # It exited early; its return code and log say why.
    returncode = process.wait()
    elapsed = time.monotonic() - started
    if returncode:
        raise RuntimeError(f"command failed ({returncode}) after {elapsed:.1f}s; see {log_path}")
    print(log_path.read_text(), end="")


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(
    video: Path, output: Path, *, max_video_bytes: int = MAX_VIDEO_BYTES, scan_type: str = "object"
) -> dict:
    byte_size = video.stat().st_size
    if byte_size > max_video_bytes:
        raise ValueError(f"video is {byte_size} bytes; limit is {max_video_bytes} bytes")
    result = run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,format_name:stream=index,codec_name,codec_type,width,height,avg_frame_rate",
            "-of", "json", str(video),
        ],
        output / "logs" / "ffprobe.log",
        capture=True,
    )
    data = json.loads(result.stdout)
    format_names = set(data.get("format", {}).get("format_name", "").split(","))
    if not format_names.intersection({"mov", "mp4"}):
        raise ValueError(f"expected an MP4 container, found {','.join(sorted(format_names)) or 'unknown'}")
    streams = [stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"]
    if len(streams) != 1:
        raise ValueError(f"expected exactly one video stream, found {len(streams)}")
    duration = float(data["format"]["duration"])
    max_duration = MAX_DURATION_SECONDS[scan_type]
    if not MIN_DURATION_SECONDS <= duration <= max_duration:
        raise ValueError(f"duration {duration:.3f}s is outside {MIN_DURATION_SECONDS}-{max_duration}s")
    stream = streams[0]
    if stream.get("codec_name") != "h264":
        raise ValueError(f"expected H.264 video, found {stream.get('codec_name') or 'unknown'}")
    if int(stream.get("width", 0)) <= 0 or int(stream.get("height", 0)) <= 0:
        raise ValueError("video dimensions are invalid")
    return {
        "byteSize": byte_size,
        "durationSeconds": duration,
        "formatNames": sorted(format_names),
        "stream": stream,
        "sha256": checksum(video),
    }


def inspect_source(
    video: Path, output: Path, *, max_video_bytes: int = MAX_VIDEO_BYTES, scan_type: str = "object"
) -> dict[str, Any]:
    """Probe a source video that frame extraction will decode directly.

    Phone captures (HEVC, 10-bit HDR, rotated, slightly overlong) used to be
    re-encoded to H.264 first. On a 62 s, 120 fps room clip that took 17 s and
    decoded every frame twice. Extraction now reads any codec FFmpeg decodes,
    applies the display rotation and trims to ``usableDurationSeconds``.
    """
    byte_size = video.stat().st_size
    if byte_size > max_video_bytes:
        raise ValueError(f"video is {byte_size} bytes; limit is {max_video_bytes} bytes")
    result = run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,format_name:stream=index,codec_name,codec_type,width,height,pix_fmt,"
            "avg_frame_rate:stream_side_data=rotation",
            "-of", "json", str(video),
        ],
        output / "logs" / "ffprobe-source.log",
        capture=True,
    )
    data = json.loads(result.stdout)
    streams = [stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"]
    if not streams:
        raise ValueError("source has no video stream")
    duration = float(data.get("format", {}).get("duration") or 0.0)
    if duration < MIN_DURATION_SECONDS:
        raise ValueError(f"duration {duration:.3f}s is shorter than {MIN_DURATION_SECONDS}s")
    stream = streams[0]
    width, height = int(stream.get("width", 0)), int(stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("video dimensions are invalid")
    rotation = next((int(float(item["rotation"])) for item in stream.get("side_data_list", [])
                     if "rotation" in item), 0)
    if abs(rotation) % 180 == 90:
        width, height = height, width
    usable = min(duration, NORMALIZED_MAX_DURATION_SECONDS[scan_type])
    return {
        "byteSize": byte_size,
        "durationSeconds": duration,
        "usableDurationSeconds": usable,
        "trimmed": usable < duration,
        "formatNames": sorted(data.get("format", {}).get("format_name", "").split(",")),
        "stream": stream,
        "displayWidth": width,
        "displayHeight": height,
        "sha256": checksum(video),
    }


def limit_patch_match_sources(config: Path, count: int) -> None:
    """Rewrite image_undistorter's ``__auto__, N`` source lists to ``__auto__, count``."""
    lines = config.read_text().splitlines()
    rewritten = [f"__auto__, {count}" if re.fullmatch(r"__auto__,\s*\d+", line.strip()) else line
                 for line in lines]
    if rewritten == lines and not any(line.strip() == f"__auto__, {count}" for line in lines):
        raise ValueError(f"{config} has no automatic source image lists to limit")
    config.write_text("\n".join(rewritten) + "\n")


def read_pgm(path: Path) -> tuple[int, int, bytes]:
    """Read an 8-bit binary PGM produced by FFmpeg."""
    data = path.read_bytes()
    tokens: list[bytes] = []
    cursor = 0
    while len(tokens) < 4:
        while cursor < len(data) and chr(data[cursor]).isspace():
            cursor += 1
        if cursor < len(data) and data[cursor] == ord("#"):
            cursor = data.find(b"\n", cursor) + 1
            continue
        end = cursor
        while end < len(data) and not chr(data[end]).isspace():
            end += 1
        tokens.append(data[cursor:end])
        cursor = end
    if tokens[0] != b"P5" or tokens[3] != b"255":
        raise ValueError(f"unsupported PGM thumbnail: {path}")
    if cursor >= len(data) or not chr(data[cursor]).isspace():
        raise ValueError(f"malformed PGM header: {path}")
    cursor += 1
    if data[cursor - 1:cursor] == b"\r" and data[cursor:cursor + 1] == b"\n":
        cursor += 1
    width, height = int(tokens[1]), int(tokens[2])
    pixels = data[cursor:cursor + width * height]
    if len(pixels) != width * height:
        raise ValueError(f"truncated PGM thumbnail: {path}")
    return width, height, pixels


def read_pgm_array(path: Path) -> np.ndarray:
    width, height, pixels = read_pgm(path)
    return np.frombuffer(pixels, np.uint8).reshape(height, width)


def frame_quality(path: Path) -> dict[str, float | bool]:
    values = read_pgm_array(path).astype(np.int64)
    mean = float(values.mean())
    clipped = float(((values <= 5) | (values >= 250)).mean())
    # Variance of a four-neighbour Laplacian, normalized for mean luminance.
    laplacian = (
        4 * values[1:-1, 1:-1] - values[1:-1, :-2] - values[1:-1, 2:]
        - values[:-2, 1:-1] - values[2:, 1:-1]
    )
    variance = max(0.0, float(laplacian.astype(np.float64).var())) if laplacian.size else 0.0
    sharpness = variance / max(mean, 16.0)
    usable = 18.0 <= mean <= 238.0 and clipped <= 0.35 and sharpness >= 0.08
    return {"brightness": mean, "clippedRatio": clipped, "sharpness": sharpness, "usable": usable}


def frame_difference(left: Path, right: Path) -> float:
    left_values = read_pgm_array(left)
    right_values = read_pgm_array(right)
    if left_values.shape != right_values.shape:
        raise ValueError("thumbnail dimensions differ")
    difference = np.abs(left_values.astype(np.int16) - right_values.astype(np.int16))
    return float(difference.sum()) / (255 * left_values.size)


def parse_showinfo_timestamps(log: str) -> list[float]:
    timestamps = []
    for line in log.splitlines():
        match = re.search(r"\bpts_time:([0-9.eE+-]+)", line)
        if match:
            timestamps.append(float(match.group(1)))
    return timestamps


def extraction_command(video: Path, candidates: Path, thumbnails: Path, duration: float, use_gpu: bool) -> list[str]:
    """One decode of the source writes both candidate JPEGs and their scoring thumbnails.

    ``-hwaccel cuda`` decodes on the GPU where NVDEC supports the codec and falls
    back to software decoding otherwise; frames return to system memory, so the
    display rotation is still applied automatically.
    """
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-nostdin",
        *(["-hwaccel", "cuda"] if use_gpu else []),
        "-t", f"{duration}", "-i", str(video),
        "-filter_complex",
        f"[0:v:0]fps={CANDIDATE_FRAME_RATE},showinfo,split=2[full][thumb];"
        "[thumb]scale=160:-2,format=gray[gray]",
        "-map", "[full]", "-q:v", "2", "-fps_mode", "passthrough", str(candidates / "candidate-%04d.jpg"),
        "-map", "[gray]", "-fps_mode", "passthrough", str(thumbnails / "candidate-%04d.pgm"),
    ]


def select_keyframes(
    candidates: list[Path], thumbnails: list[Path], timestamps: list[float], target: int,
    max_frames: int = MAX_SELECTED_FRAMES["object"],
) -> tuple[list[int], list[dict[str, Any]]]:
    if not (len(candidates) == len(thumbnails) == len(timestamps)):
        raise ValueError("candidate images, thumbnails and timestamps are not aligned")
    scored: list[dict[str, Any]] = []
    for index, thumbnail in enumerate(thumbnails):
        metrics = frame_quality(thumbnail)
        difference = 1.0 if index == 0 else frame_difference(thumbnails[index - 1], thumbnail)
        scored.append({"candidateIndex": index, "timestampSeconds": timestamps[index],
                       "differenceFromPrevious": difference, **metrics})
    eligible = [index for index, item in enumerate(scored)
                if item["usable"] and (item["differenceFromPrevious"] >= 0.006 or index == 0)]
    if len(eligible) < 15:
        raise ValueError(f"only {len(eligible)} sharp, exposed, non-duplicate candidate frames")
    target = min(target, len(eligible), max_frames)
    # Pick the sharpest frame from each temporal bin to retain baseline and overlap.
    selected: list[int] = []
    for bin_index in range(target):
        start = math.floor(bin_index * len(eligible) / target)
        end = math.floor((bin_index + 1) * len(eligible) / target)
        choices = eligible[start:max(start + 1, end)]
        selected.append(max(choices, key=lambda index: float(scored[index]["sharpness"])))
    selected = sorted(set(selected))
    selected_set = set(selected)
    for item in scored:
        item["selected"] = item["candidateIndex"] in selected_set
    return selected, scored


GLOG_PREFIX = re.compile(r"^[IWEF]\d{8}\s+[\d:.]+\s+\d+\s+[^\]\s]+:\d+\]\s*")


def largest_sparse_model(sparse: Path) -> Path:
    """Return the COLMAP model with the most registered images.

    The mapper may split a capture into several models, and its multi-threaded
    numbering is not stable: on one 62 s room capture ``sparse/0`` held 159 of 180
    frames in one run and a fragment of under 15 frames in the next. ``images.bin`` starts
    with its registered-image count, so no model needs to be parsed.
    """
    counts = {}
    if (sparse / "images.bin").is_file():
        return sparse
    for model in sorted(path for path in sparse.iterdir() if path.is_dir()):
        images = model / "images.bin"
        if images.is_file() and images.stat().st_size >= 8:
            with images.open("rb") as source:
                counts[model] = struct.unpack("<Q", source.read(8))[0]
    if not counts:
        raise RuntimeError("COLMAP produced no sparse model")
    return max(counts, key=lambda model: counts[model])


def sparse_image_count(model: Path) -> int:
    """Registered images of a binary COLMAP model (``images.bin`` starts with the count)."""
    images = model / "images.bin"
    if model.is_dir() and images.is_file() and images.stat().st_size >= 8:
        # COLMAP writes models either directly or in numbered subdirectories.
        with images.open("rb") as source:
            return struct.unpack("<Q", source.read(8))[0]
    return 0


def colmap_command(executable: str, command: str, *arguments: str) -> list[str]:
    return [executable, command, *arguments]


def parse_model_analyzer(output: str, selected_frames: int) -> dict[str, Any]:
    labels = {
        "Cameras": ("cameras", int),
        "Images": ("registeredFrames", int),
        "Registered images": ("registeredFrames", int),
        "Points": ("points3D", int),
        "Observations": ("observations", int),
        "Mean track length": ("meanTrackLength", float),
        "Mean observations per image": ("meanObservationsPerImage", float),
        "Mean reprojection error": ("meanReprojectionErrorPixels", float),
    }
    metrics: dict[str, Any] = {"selectedFrames": selected_frames}
    for line in output.splitlines():
        # COLMAP 4 logs through glog: "I20260927 23:50:55.345 504 model.cc:448] Points: 1964".
        line = GLOG_PREFIX.sub("", line)
        if ":" not in line:
            continue
        label, raw_value = (part.strip() for part in line.split(":", 1))
        definition = labels.get(label)
        if definition is None:
            continue
        key, converter = definition
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", raw_value.replace(",", ""))
        if match:
            metrics[key] = converter(match.group())

    required = ("registeredFrames", "points3D", "meanReprojectionErrorPixels")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"COLMAP model analyzer omitted required metrics: {', '.join(missing)}")
    metrics["registrationRatio"] = metrics["registeredFrames"] / selected_frames if selected_frames else 0.0
    metrics["qualityDecision"] = quality_decision(metrics)
    return metrics


def read_median_reprojection_error(points3d_path: Path) -> float:
    errors = []
    with points3d_path.open() as source:
        for line in source:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 8:
                raise ValueError(f"invalid COLMAP point record in {points3d_path}")
            errors.append(float(fields[7]))
    if not errors:
        raise ValueError(f"COLMAP model has no 3D point errors: {points3d_path}")
    return statistics.median(errors)


def quaternion_rotation(qw: float, qx: float, qy: float, qz: float) -> list[list[float]]:
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def estimate_up_vector(rights: list[list[float]], ups: list[list[float]]) -> tuple[list[float], str]:
    """Estimate world up from registered cameras, assuming the phone is held without roll.

    Every camera's right (image x) axis is then horizontal, so up is the direction
    perpendicular to all of them: the least-significant eigenvector of sum(r r^T).
    Averaging the image-up axes instead is biased by pitch; on a room scan looking
    down at the floor by ~43 degrees that average tilted the floor by ~22 degrees.
    When the camera never turned (a straight strafe), the right axes are parallel
    and only fix up to within a plane, so the mean image-up is projected into it.
    """
    right = np.asarray(rights, dtype=float)
    mean_up = np.asarray(ups, dtype=float).mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(right.T @ right / len(right))
    if eigenvalues[1] >= MIN_RIGHT_VECTOR_SPREAD:
        up = eigenvectors[:, 0]
        method = "cameraRightVectors"
    else:
        axis = eigenvectors[:, 2]
        up = mean_up - (mean_up @ axis) * axis
        method = "cameraUpOrthogonalToRight"
    length = float(np.linalg.norm(up))
    if length < 1e-9:
        return [0.0, 1.0, 0.0], "default"
    up = up / length
    if up @ mean_up < 0:
        up = -up
    return [float(value) for value in up], method


def frame_index(name: str) -> int | None:
    """Selected-frame number from ``frame-0042.jpg``, or None for other names."""
    match = re.fullmatch(r"frame-(\d+)\.\w+", name)
    return int(match.group(1)) if match else None


def trajectory_step_ratio(names: list[str], centers: list[list[float]]) -> float:
    """Largest camera step between consecutive frames as a multiple of the median.

    Frames that failed to register leave gaps in the sequence. A step across such
    a gap covers several frame intervals of ordinary motion, so it is measured per
    interval; otherwise a 12-frame gap after a quick turn in a room walk reads as a
    6.7x jump and a slightly longer one fails the gate.
    """
    indices = [frame_index(name) for name in names]
    steps = [
        math.dist(left, right) / max(1, (later - earlier) if earlier is not None and later is not None else 1)
        for left, right, earlier, later in zip(centers, centers[1:], indices, indices[1:])
    ]
    nonzero_steps = [step for step in steps if step > 1e-9]
    median_step = statistics.median(nonzero_steps) if nonzero_steps else 0.0
    return max(steps) / median_step if median_step else math.inf


def trajectory_outlier_frames(names: list[str], centers: list[list[float]]) -> list[str] | None:
    """Frames whose poses spike off the camera path, or None if that cannot explain a break.

    COLMAP occasionally registers a weakly matched frame far from where it was
    taken (645x the median step on one room capture, and straight back on the next
    frame). Such frames are removed greedily, always the endpoint of the worst step
    whose removal leaves the smoothest path, up to MAX_TRAJECTORY_OUTLIER_FRACTION.
    A real break between two walks survives that budget and still fails the gate.
    """
    kept = list(zip(names, centers))
    removed: list[str] = []
    budget = int(len(kept) * MAX_TRAJECTORY_OUTLIER_FRACTION)

    def ratio(cameras: list[tuple[str, list[float]]]) -> float:
        return trajectory_step_ratio([name for name, _ in cameras], [center for _, center in cameras])

    while len(kept) > 2 and ratio(kept) > MAX_TRAJECTORY_STEP_RATIO:
        if len(removed) >= budget:
            return None
        indices = [frame_index(name) for name, _ in kept]
        steps = [math.dist(left[1], right[1]) / max(1, (b - a) if a is not None and b is not None else 1)
                 for left, right, a, b in zip(kept, kept[1:], indices, indices[1:])]
        worst = max(range(len(steps)), key=steps.__getitem__)
        candidate = min((worst, worst + 1), key=lambda index: ratio(kept[:index] + kept[index + 1:]))
        removed.append(kept.pop(candidate)[0])
    return removed if ratio(kept) <= MAX_TRAJECTORY_STEP_RATIO else None


def read_camera_poses(images_path: Path) -> list[tuple[str, list[float], list[float], list[float]]]:
    """(image name, center, world up, world right) per registered image, sorted by name."""
    cameras: list[tuple[str, list[float], list[float], list[float]]] = []
    lines = []
    for raw_line in images_path.read_text().splitlines():
        fields = raw_line.split()
        if len(fields) == 10 and fields[0].isdigit() and fields[8].isdigit():
            lines.append(raw_line.strip())
    for line in lines:
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"invalid COLMAP image record in {images_path}")
        qw, qx, qy, qz = map(float, fields[1:5])
        translation = list(map(float, fields[5:8]))
        rotation = quaternion_rotation(qw, qx, qy, qz)
        center = [-sum(rotation[row][column] * translation[row] for row in range(3))
                  for column in range(3)]
        # COLMAP camera +Y points down, hence world-up is -R^T * camera-Y.
        up = [-rotation[1][column] for column in range(3)]
        right = [rotation[0][column] for column in range(3)]
        cameras.append((fields[9], center, up, right))
    if len(cameras) < 2:
        raise ValueError("sparse model has fewer than two camera poses")
    return sorted(cameras, key=lambda item: item[0])


def sparse_geometry_metrics(images_path: Path, points_path: Path) -> dict[str, Any]:
    """Measure camera baseline/continuity and infer up from registered camera rolls."""
    cameras = read_camera_poses(images_path)
    point_coordinates = []
    for line in points_path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            fields = line.split()
            point_coordinates.append(tuple(map(float, fields[1:4])))
    if not point_coordinates:
        raise ValueError("sparse model contains no points")
    scene_min = [min(point[axis] for point in point_coordinates) for axis in range(3)]
    scene_max = [max(point[axis] for point in point_coordinates) for axis in range(3)]
    scene_diagonal = math.sqrt(sum((scene_max[axis] - scene_min[axis]) ** 2 for axis in range(3)))
    centers = [camera[1] for camera in cameras]
    baseline = max(math.dist(left, right) for left in centers for right in centers)
    step_ratio = trajectory_step_ratio([camera[0] for camera in cameras], centers)
    estimated_up, up_method = estimate_up_vector([camera[3] for camera in cameras],
                                                 [camera[2] for camera in cameras])
    return {
        "cameraBaseline": baseline,
        "sceneDiagonal": scene_diagonal,
        "baselineToSceneDiagonal": baseline / scene_diagonal if scene_diagonal else 0.0,
        "maximumTrajectoryStepRatio": step_ratio,
        "trajectoryDisconnected": step_ratio > MAX_TRAJECTORY_STEP_RATIO,
        "estimatedUpVector": estimated_up,
        "upVectorMethod": up_method,
    }


def quality_decision(metrics: dict[str, Any]) -> dict[str, Any]:
    failures = []
    if metrics["registeredFrames"] < MIN_REGISTERED_FRAMES:
        failures.append("TOO_FEW_REGISTERED_FRAMES")
    if metrics["registrationRatio"] < MIN_REGISTRATION_RATIO:
        failures.append("LOW_REGISTRATION_RATIO")
    reprojection_error = metrics.get(
        "medianReprojectionErrorPixels", metrics["meanReprojectionErrorPixels"]
    )
    if reprojection_error > MAX_MEDIAN_REPROJECTION_ERROR_PIXELS:
        failures.append("HIGH_REPROJECTION_ERROR")
    if metrics.get("baselineToSceneDiagonal", 1.0) < MIN_BASELINE_TO_SCENE_DIAGONAL:
        failures.append("INSUFFICIENT_BASELINE")
    if metrics.get("trajectoryDisconnected", False):
        failures.append("DISCONNECTED_CAMERA_TRAJECTORY")
    return {
        "usable": not failures,
        "reasons": failures,
        "thresholds": {
            "minimumRegisteredFrames": MIN_REGISTERED_FRAMES,
            "minimumRegistrationRatio": MIN_REGISTRATION_RATIO,
            "maximumMedianReprojectionErrorPixels": MAX_MEDIAN_REPROJECTION_ERROR_PIXELS,
            "minimumBaselineToSceneDiagonal": MIN_BASELINE_TO_SCENE_DIAGONAL,
            "maximumTrajectoryStepRatio": MAX_TRAJECTORY_STEP_RATIO,
        },
    }


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as source:
        if source.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        for _ in range(10_000):
            line = source.readline()
            if not line:
                break
            match = re.fullmatch(rb"element\s+vertex\s+(\d+)\s*", line)
            if match:
                return int(match.group(1))
            if line.strip() == b"end_header":
                break
    raise ValueError(f"PLY header has no vertex count: {path}")


def reconstruct(
    video: Path,
    output: Path,
    colmap: str,
    use_gpu: bool,
    frame_rate: float,
    up_vector: list[float] | None = None,
    depth_completion: bool = True,
    mode: str = "detailed",
    scan_type: str = "object",
    preload_depth: bool = True,
) -> None:
    if mode not in PROCESSING_MODES:
        raise ValueError(f"unknown processing mode {mode!r}; expected one of {', '.join(PROCESSING_MODES)}")
    if scan_type not in SCAN_TYPES:
        raise ValueError(f"unknown scan type {scan_type!r}; expected one of {', '.join(SCAN_TYPES)}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}; pass a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    settings = {
        "candidateFrameRate": CANDIDATE_FRAME_RATE,
        "selectedFrameRate": frame_rate,
        "useGpuForFeatures": use_gpu,
        "cameraModel": "SIMPLE_RADIAL",
        "singleCamera": True,
        "matcher": f"exhaustive<={EXHAUSTIVE_MATCHING_MAX_FRAMES} frames, else sequential",
        "sequentialOverlap": SEQUENTIAL_OVERLAP,
        "geometricConsistency": True,
        "alignmentSource": "recordedGravity" if up_vector else "registeredCameraPoses",
        "denseMaxImageSize": DENSE_MAX_IMAGE_SIZE,
        "patchMatchWindowRadius": PATCH_MATCH_WINDOW_RADIUS,
        "patchMatchWindowStep": PATCH_MATCH_WINDOW_STEP,
        "patchMatchFilterMinNcc": PATCH_MATCH_FILTER_MIN_NCC,
        "patchMatchNumIterations": PATCH_MATCH_NUM_ITERATIONS,
        "patchMatchSourceImages": PATCH_MATCH_SOURCE_IMAGES,
        "depthCompletion": depth_completion and mode == "detailed",
        "processingMode": mode,
        "scanType": scan_type,
        "maxSelectedFrames": MAX_SELECTED_FRAMES[scan_type],
        "sceneCompletion": scan_type == "scene",
        "backprojectedCloud": mode == "fast" and scan_type == "scene",
        # Depth Anything's shorter input side; depth_completion.py picks it per mode.
        "monocularInputSize": 392 if mode == "fast" else 518,
    }
    settings["mapperOptions"] = list(MAPPER_SPEED_OPTIONS)
    if scan_type == "scene":
        settings["siftPeakThreshold"] = SCENE_SIFT_PEAK_THRESHOLD
        settings["mapperOptions"] += list(SCENE_MAPPER_OPTIONS)
    if mode == "fast":
        for key in ("patchMatchWindowRadius", "patchMatchWindowStep", "patchMatchFilterMinNcc",
                    "patchMatchNumIterations", "patchMatchSourceImages"):
            settings.pop(key)
    manifest = StageManifest(output / "manifest.json", settings)
    preloaded: subprocess.Popen[str] | None = None
    try:
        manifest.start("VALIDATION")
        metadata = inspect_source(video, output, scan_type=scan_type)
        (output / "input.json").write_text(json.dumps(metadata, indent=2) + "\n")
        source_metadata = {
            "sha256": metadata["sha256"],
            "durationSeconds": metadata["usableDurationSeconds"],
            "sourceDurationSeconds": metadata["durationSeconds"],
            "stream": metadata["stream"],
        }
        manifest.data["source"] = source_metadata
        manifest.complete(
            metrics={
                "durationSeconds": metadata["usableDurationSeconds"],
                "sourceDurationSeconds": metadata["durationSeconds"],
                "trimmed": metadata["trimmed"],
                "width": metadata["displayWidth"],
                "height": metadata["displayHeight"],
                "codec": metadata["stream"].get("codec_name"),
                "pixelFormat": metadata["stream"].get("pix_fmt"),
            },
            artifacts=["input.json", "logs/ffprobe-source.log"],
        )

        candidates = output / "candidates"
        thumbnails = output / "thumbnails"
        images = output / "images"
        sparse = output / "sparse"
        dense = output / "dense"
        database = output / "database.db"
        candidates.mkdir()
        thumbnails.mkdir()
        images.mkdir()
        sparse.mkdir()

        manifest.start("EXTRACTING_FRAMES")
        run(extraction_command(video, candidates, thumbnails, metadata["usableDurationSeconds"], use_gpu),
            output / "logs" / "extract-frames.log")
        candidate_paths = sorted(candidates.glob("*.jpg"))
        thumbnail_paths = sorted(thumbnails.glob("*.pgm"))
        timestamps = parse_showinfo_timestamps((output / "logs" / "extract-frames.log").read_text())
        if not len(candidate_paths) == len(thumbnail_paths) == len(timestamps):
            raise ValueError(
                f"FFmpeg created {len(candidate_paths)} candidates and {len(thumbnail_paths)} thumbnails "
                f"but reported {len(timestamps)} timestamps"
            )
        target = max(MIN_SELECTED_FRAMES, round(metadata["usableDurationSeconds"] * frame_rate))
        selected_indices, frame_scores = select_keyframes(
            candidate_paths, thumbnail_paths, timestamps, target, MAX_SELECTED_FRAMES[scan_type]
        )
        timestamp_map = []
        for output_index, candidate_index in enumerate(selected_indices, start=1):
            destination = images / f"frame-{output_index:04d}.jpg"
            shutil.copyfile(candidate_paths[candidate_index], destination)
            timestamp_map.append({
                "image": destination.name,
                "candidate": candidate_paths[candidate_index].name,
                "candidateIndex": candidate_index,
                "timestampSeconds": timestamps[candidate_index],
            })
        image_count = len(selected_indices)
        selection_payload = {
            "schemaVersion": 1,
            "candidateFrameRate": CANDIDATE_FRAME_RATE,
            "candidateFrames": len(candidate_paths),
            "selectedFrames": image_count,
            "timestampMap": timestamp_map,
            "scores": frame_scores,
        }
        (output / "frame-selection.json").write_text(json.dumps(selection_payload, indent=2) + "\n")
        manifest.complete(
            metrics={"candidateFrames": len(candidate_paths), "selectedFrames": image_count,
                     "rejectedFrames": len(candidate_paths) - image_count},
            artifacts=["frame-selection.json", "logs/extract-frames.log", "logs/score-thumbnails.log"],
        )

        gpu = "1" if use_gpu else "0"
        scene_extraction = (["--SiftExtraction.peak_threshold", str(SCENE_SIFT_PEAK_THRESHOLD)]
                            if scan_type == "scene" else [])
        mapper_options = [*MAPPER_SPEED_OPTIONS, *(SCENE_MAPPER_OPTIONS if scan_type == "scene" else ())]
        if image_count <= EXHAUSTIVE_MATCHING_MAX_FRAMES:
            matching = colmap_command(
                colmap, "exhaustive_matcher", "--database_path", str(database),
                "--FeatureMatching.use_gpu", gpu,
            )
        else:
            matching = colmap_command(
                colmap, "sequential_matcher", "--database_path", str(database),
                "--FeatureMatching.use_gpu", gpu,
                "--SequentialMatching.overlap", str(SEQUENTIAL_OVERLAP),
                "--SequentialMatching.quadratic_overlap", "1",
            )
        stages = [
            (
                "FEATURE_EXTRACTION",
                colmap_command(
                    colmap, "feature_extractor", "--database_path", str(database),
                    "--image_path", str(images), "--ImageReader.single_camera", "1",
                    "--ImageReader.camera_model", "SIMPLE_RADIAL", "--FeatureExtraction.use_gpu", gpu,
                    *scene_extraction,
                ),
            ),
            ("FEATURE_MATCHING", matching),
            (
                "SPARSE_RECONSTRUCTION",
                colmap_command(
                    colmap, "mapper", "--database_path", str(database),
                    "--image_path", str(images), "--output_path", str(sparse), *mapper_options,
                ),
            ),
        ]
        for name, command in stages:
            manifest.start(name)
            log_name = name.lower().replace("_", "-") + ".log"
            run(command, output / "logs" / log_name)
            stage_metrics = {"matcher": command[1]} if name == "FEATURE_MATCHING" else None
            manifest.complete(metrics=stage_metrics, artifacts=[f"logs/{log_name}"])

        model = largest_sparse_model(sparse)
        if scan_type == "scene" and sparse_image_count(model) < image_count:
            # Continue mapping from the main model: frames that failed to join it
            # (often a separate fragment) get another registration attempt with
            # the full model's points to match against.
            manifest.start("SPARSE_EXTENSION")
            extended = output / "sparse-extended"
            extended.mkdir()
            run(
                colmap_command(
                    colmap, "mapper", "--database_path", str(database), "--image_path", str(images),
                    "--input_path", str(model), "--output_path", str(extended), *mapper_options,
                ),
                output / "logs" / "sparse-extension.log",
            )
            before = sparse_image_count(model)
            candidate = largest_sparse_model(extended) if any(extended.iterdir()) else None
            after = sparse_image_count(candidate) if candidate else before
            if candidate is not None and after > before:
                model = candidate
            manifest.complete(metrics={"registeredBefore": before, "registeredAfter": max(before, after)},
                              artifacts=["logs/sparse-extension.log"])

        manifest.start("RECONSTRUCTION_METRICS")
        metrics_model = output / "metrics-model"

        def analyse(model: Path) -> dict[str, Any]:
            shutil.rmtree(metrics_model, ignore_errors=True)
            metrics_model.mkdir()
            run(
                colmap_command(
                    colmap, "model_converter", "--input_path", str(model),
                    "--output_path", str(metrics_model), "--output_type", "TXT",
                ),
                output / "logs" / "model-converter.log",
                capture=True,
            )
            analysis = run(
                colmap_command(colmap, "model_analyzer", "--path", str(model)),
                output / "logs" / "model-analyzer.log",
                capture=True,
            )
            metrics = parse_model_analyzer(analysis.stdout, image_count)
            metrics["medianReprojectionErrorPixels"] = read_median_reprojection_error(
                metrics_model / "points3D.txt"
            )
            metrics.update(sparse_geometry_metrics(
                metrics_model / "images.txt", metrics_model / "points3D.txt"
            ))
            metrics["qualityDecision"] = quality_decision(metrics)
            return metrics

        sparse_metrics = analyse(model)
        if sparse_metrics["trajectoryDisconnected"]:
            poses = read_camera_poses(metrics_model / "images.txt")
            outliers = trajectory_outlier_frames([pose[0] for pose in poses], [pose[1] for pose in poses])
            if outliers:
                names_path = output / "trajectory-outliers.txt"
                names_path.write_text("\n".join(outliers) + "\n")
                trimmed = output / "sparse-trimmed"
                trimmed.mkdir()
                run(
                    colmap_command(
                        colmap, "image_deleter", "--input_path", str(model),
                        "--output_path", str(trimmed), "--image_names_path", str(names_path),
                    ),
                    output / "logs" / "image-deleter.log",
                )
                model = trimmed
                sparse_metrics = analyse(model)
                sparse_metrics["removedTrajectoryOutlierFrames"] = outliers
        manifest.complete(
            metrics=sparse_metrics,
            artifacts=["logs/model-analyzer.log", "logs/model-converter.log", "metrics-model/"],
        )
        if not sparse_metrics["qualityDecision"]["usable"]:
            reasons = ", ".join(sparse_metrics["qualityDecision"]["reasons"])
            raise RuntimeError(f"sparse reconstruction failed quality gate: {reasons}")

        dense.mkdir()
        dense_stages = [
            (
                "UNDISTORTION",
                colmap_command(
                    colmap, "image_undistorter", "--image_path", str(images),
                    "--input_path", str(model), "--output_path", str(dense), "--output_type", "COLMAP",
                    # Fast mode never reads more than the depth-map resolution, so
                    # full-size undistorted frames are only extra encoding and I/O.
                    *(["--max_image_size", str(DENSE_MAX_IMAGE_SIZE)] if mode == "fast" else []),
                ),
            ),
        ]
        # Fast room scans skip COLMAP stereo fusion: scene completion rebuilds the
        # surfaces from the depth maps, and the fused cloud only supplied bounds,
        # wall and floor directions, which back-projected depth gives as well.
        backprojected_cloud = mode == "fast" and scan_type == "scene"

        def fusion(fused: Path) -> list[str]:
            return colmap_command(
                colmap, "stereo_fusion", "--workspace_path", str(dense),
                "--workspace_format", "COLMAP", "--input_type", "geometric",
                # Must match PatchMatch, or fusion samples depth maps at the wrong scale.
                "--StereoFusion.max_image_size", str(DENSE_MAX_IMAGE_SIZE),
                "--output_path", str(fused),
            )

        completion_metrics_path = output / "depth-completion.json"
        if mode == "fast":
            dense_stages.append((
                "MONOCULAR_DEPTH",
                [sys.executable, str(Path(__file__).with_name("depth_completion.py")), str(dense),
                 "--sparse-anchors", "--max-image-size", str(DENSE_MAX_IMAGE_SIZE),
                 "--metrics", str(completion_metrics_path),
                 *(["--cloud", str(output / "fused.ply")] if backprojected_cloud else [])],
            ))
        else:
            dense_stages.append(
                (
                    "DENSE_RECONSTRUCTION",
                    colmap_command(
                        colmap, "patch_match_stereo", "--workspace_path", str(dense),
                        "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "1",
                        "--PatchMatchStereo.max_image_size", str(DENSE_MAX_IMAGE_SIZE),
                        "--PatchMatchStereo.window_radius", str(PATCH_MATCH_WINDOW_RADIUS),
                        "--PatchMatchStereo.window_step", str(PATCH_MATCH_WINDOW_STEP),
                        "--PatchMatchStereo.filter_min_ncc", str(PATCH_MATCH_FILTER_MIN_NCC),
                        "--PatchMatchStereo.num_iterations", str(PATCH_MATCH_NUM_ITERATIONS),
                    ),
                )
            )
        if depth_completion and mode == "detailed":
            # Fuse the observed maps first: that cloud is the provenance reference
            # that later tags every point as observed or monocular-only.
            dense_stages.append(("OBSERVED_FUSION", fusion(output / "fused-observed.ply")))
            dense_stages.append((
                "DEPTH_COMPLETION",
                [sys.executable, str(Path(__file__).with_name("depth_completion.py")), str(dense),
                 "--metrics", str(completion_metrics_path)],
            ))
        if not backprojected_cloud:
            dense_stages.append(("FUSION", fusion(output / "fused.ply")))
        if mode == "fast" and preload_depth:
            # Python, PyTorch and Depth Anything take ~3.7 s to load, which can
            # happen while COLMAP undistorts the frames (2.7 s on the barn, 10 s on
            # the room) instead of after it.
            depth_command = next(command for name, command in dense_stages if name == "MONOCULAR_DEPTH")
            preloaded = start_preloaded([*depth_command, "--wait-for-start"], output / "logs" / "monocular-depth.log")
        for name, command in dense_stages:
            manifest.start(name)
            log_name = name.lower().replace("_", "-") + ".log"
            if name == "MONOCULAR_DEPTH" and preloaded is not None:
                finish_preloaded(preloaded, output / "logs" / log_name)
                preloaded = None
            else:
                run(command, output / "logs" / log_name)
            stage_metrics = None
            artifacts = [f"logs/{log_name}"]
            if name == "UNDISTORTION" and mode == "detailed":
                limit_patch_match_sources(dense / "stereo" / "patch-match.cfg", PATCH_MATCH_SOURCE_IMAGES)
                stage_metrics = {"patchMatchSourceImages": PATCH_MATCH_SOURCE_IMAGES}
            elif name in {"FUSION", "OBSERVED_FUSION"}:
                fused_name = "fused.ply" if name == "FUSION" else "fused-observed.ply"
                stage_metrics = {"fusedPoints": ply_vertex_count(output / fused_name)}
                artifacts.append(fused_name)
            elif name in {"DEPTH_COMPLETION", "MONOCULAR_DEPTH"}:
                completion = json.loads(completion_metrics_path.read_text())
                stage_metrics = {key: value for key, value in completion.items() if key != "perFrame"}
                artifacts.append(completion_metrics_path.name)
                if backprojected_cloud:
                    stage_metrics["fusedPoints"] = ply_vertex_count(output / "fused.ply")
                    artifacts.append("fused.ply")
            manifest.complete(metrics=stage_metrics, artifacts=artifacts)

        manifest.start("FILTERING_AND_ALIGNMENT")
        filter_metrics_path = output / "filter-metrics.json"
        filter_command = [
            sys.executable, str(Path(__file__).with_name("point_cloud.py")),
            str(output / "fused.ply"), str(output / "canonical.ply"),
            str(output / "preview.ply"), "--metrics", str(filter_metrics_path),
        ]
        if mode == "fast":
            filter_command.append("--all-inferred")
        elif depth_completion:
            filter_command.extend(["--observed-reference", str(output / "fused-observed.ply")])
        alignment_up = up_vector or sparse_metrics.get("estimatedUpVector")
        if alignment_up:
            filter_command.extend(["--up", *(str(value) for value in alignment_up)])
        run(filter_command, output / "logs" / "filtering-and-alignment.log")
        filter_metrics = json.loads(filter_metrics_path.read_text())
        manifest.complete(
            metrics=filter_metrics,
            artifacts=["canonical.ply", "preview.ply", "filter-metrics.json",
                       "logs/filtering-and-alignment.log"],
        )

        completion: dict[str, Any] | None = None
        if scan_type == "scene":
            manifest.start("SCENE_COMPLETION")
            measured = output / "measured.ply"
            (output / "canonical.ply").replace(measured)
            completion_path = output / "scene-completion.json"
            run(
                [sys.executable, str(Path(__file__).with_name("scene_completion.py")),
                 str(measured), str(dense), str(filter_metrics_path),
                 str(output / "canonical.ply"), str(output / "preview.ply"),
                 "--metrics", str(completion_path)],
                output / "logs" / "scene-completion.log",
            )
            completion = json.loads(completion_path.read_text())
            manifest.complete(
                metrics={key: value for key, value in completion.items()
                         if key not in {"cameraCenters", "refinement"}},
                artifacts=["canonical.ply", "preview.ply", "measured.ply", "scene-completion.json",
                           "logs/scene-completion.log"],
            )

        metrics = {
            **sparse_metrics,
            "extractedFrames": image_count,
            "fusedPoints": ply_vertex_count(output / "fused.ply"),
            "filteredPoints": filter_metrics["filteredPoints"],
            "previewPoints": filter_metrics["previewPoints"],
            "bounds": filter_metrics["bounds"],
        }
        if completion is not None:
            metrics.update(
                completedPoints=completion["outputPoints"],
                synthesizedPoints=completion["synthesizedPoints"],
                previewPoints=completion["previewPoints"],
                bounds=completion["bounds"],
            )
        for key in ("observedPoints", "inferredPoints"):
            if key in filter_metrics:
                metrics[key] = filter_metrics[key]
        manifest.finish(
            source=source_metadata,
            metrics=metrics,
        )
        print(f"reconstruction complete: {output / 'canonical.ply'}")
    except Exception as error:
        if preloaded is not None and preloaded.poll() is None:
            preloaded.kill()
            preloaded.wait()
        manifest.fail(error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--cpu", action="store_true", help="disable GPU feature extraction and matching")
    parser.add_argument(
        "--frame-rate", type=float, default=4.0,
        help="target selected-frame density, 1-6 FPS (candidates are always decoded at 8 FPS)",
    )
    parser.add_argument(
        "--up-vector", nargs=3, type=float, metavar=("X", "Y", "Z"),
        help="recorded gravity-derived up vector in reconstruction coordinates",
    )
    parser.add_argument(
        "--mode", choices=PROCESSING_MODES, default="detailed",
        help="detailed: PatchMatch + hole filling (~13 min); fast: monocular depth on SfM points (~1.5 min, softer)",
    )
    parser.add_argument(
        "--scan-type", choices=SCAN_TYPES, default="object",
        help="object: orbit of one thing (60 s, 240 frames); scene: room or several objects (300 s, 600 frames)",
    )
    parser.add_argument(
        "--no-depth-preload", action="store_true",
        help="fast mode: load the depth network after undistortion instead of during it",
    )
    parser.add_argument(
        "--no-depth-completion", action="store_true",
        help="fuse observed multi-view stereo depth only; skip monocular hole filling",
    )
    args = parser.parse_args()
    if not 1.0 <= args.frame_rate <= 6.0:
        parser.error("--frame-rate must be between 1 and 6")
    video = args.video.resolve()
    output = args.output.resolve()
    if not video.is_file():
        parser.error(f"video does not exist: {video}")
    for executable in ("ffmpeg", "ffprobe", args.colmap):
        if not shutil.which(executable):
            parser.error(f"required executable not found: {executable}")
    try:
        reconstruct(video, output, args.colmap, not args.cpu, args.frame_rate, args.up_vector,
                    depth_completion=not args.no_depth_completion, mode=args.mode,
                    scan_type=args.scan_type, preload_depth=not args.no_depth_preload)
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
