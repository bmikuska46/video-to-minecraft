#!/usr/bin/env python3
"""Fetch a real, openly licensed walk-around video and cut it into a capture fixture.

The source is the Tanks and Temples "Barn" video (CC BY 4.0): a handheld 4K walk
around a single-storey park building. Only the MP4 index and the byte range of
the chosen 15-second segment are downloaded (about 200 MB rather than 4.4 GB).
The downloaded segment bytes are verified against a pinned SHA-256 before the
clip is downscaled to a phone-like 1920x1080 H.264 capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_ROOT = ROOT / "fixtures" / "captures"
DEFAULT_CACHE = Path.home() / ".cache" / "video-to-minecraft" / "tanks-and-temples"

SLUG = "barn-gable-arc-real"
SOURCE_NAME = "Barn.mp4"
SOURCE_DRIVE_ID = "0B-ePgl6HF260ZlBZcHFrTHFLdGM"
SOURCE_URL = (
    "https://drive.usercontent.google.com/download"
    f"?id={SOURCE_DRIVE_ID}&export=download&confirm=t"
)
SOURCE_BYTES = 4_747_838_231
SEGMENT_START_SECONDS = 110.0
SEGMENT_DURATION_SECONDS = 15.0
# SHA-256 of the source bytes spanning every packet FFmpeg reads for the segment.
SOURCE_SEGMENT_SHA256 = "71baf89f9d00ed1833f4ee4d8538e21b4870cf7333673974f28700cfb9f2117a"
WIDTH, HEIGHT = 1920, 1080
RANGE_PIECE_BYTES = 16 * 1024 * 1024
ATTRIBUTION = (
    "Knapitsch, Park, Zhou and Koltun, 'Tanks and Temples: Benchmarking "
    "Large-Scale Scene Reconstruction', ACM Transactions on Graphics 36(4), 2017. "
    "https://www.tanksandtemples.org/"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_range(start: int, end: int, attempts: int = 8) -> bytes:
    """Return bytes start..end inclusive; Google Drive intermittently serves an HTML quota page."""
    expected = end - start + 1
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            SOURCE_URL, headers={"Range": f"bytes={start}-{end}", "User-Agent": "video-to-minecraft-fixtures/1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            if response.status == 206 and len(payload) == expected:
                return payload
            reason = f"HTTP {response.status}, {len(payload)} of {expected} bytes"
        except (urllib.error.URLError, TimeoutError) as error:
            reason = str(error)
        print(f"range {start}-{end} attempt {attempt} failed: {reason}", flush=True)
        time.sleep(15 * attempt)
    raise RuntimeError(f"could not download bytes {start}-{end} of {SOURCE_NAME}")


def moov_offset(head: bytes) -> int:
    """Locate the trailing moov box from the leading ftyp and 64-bit mdat headers."""
    offset = 0
    while offset + 8 <= len(head):
        size, kind = struct.unpack(">I4s", head[offset:offset + 8])
        if kind == b"moov":
            return offset
        if size == 1:
            size = struct.unpack(">Q", head[offset + 8:offset + 16])[0]
        if size < 8:
            break
        offset += size
        if offset >= len(head):
            return offset
    raise RuntimeError("could not locate the MP4 moov box")


def segment_byte_range(sparse: Path) -> tuple[int, int]:
    """Byte span of every packet FFmpeg reads for the segment, from the index alone."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-read_intervals", f"{SEGMENT_START_SECONDS}%+{SEGMENT_DURATION_SECONDS + 1}",
            "-show_entries", "packet=pos,size", "-of", "json", str(sparse),
        ],
        text=True, capture_output=True, check=True,
    )
    spans = [(int(packet["pos"]), int(packet["size"])) for packet in json.loads(result.stdout)["packets"]]
    if not spans:
        raise RuntimeError("ffprobe found no packets in the requested segment")
    return min(pos for pos, _ in spans), max(pos + size for pos, size in spans)


def prepare_sparse_source(cache: Path, local_source: Path | None) -> tuple[Path, str]:
    """Materialize the index and segment bytes in a sparse copy of the source MP4."""
    cache.mkdir(parents=True, exist_ok=True)
    sparse = cache / f"{SOURCE_NAME}.sparse"
    progress_path = cache / f"{SOURCE_NAME}.ranges.json"
    fetched = json.loads(progress_path.read_text()) if progress_path.exists() and sparse.exists() else []

    def read(start: int, end: int) -> bytes:
        if local_source:
            with local_source.open("rb") as source:
                source.seek(start)
                return source.read(end - start + 1)
        return fetch_range(start, end)

    def ensure(start: int, end: int) -> None:
        for piece_start in range(start, end + 1, RANGE_PIECE_BYTES):
            piece_end = min(piece_start + RANGE_PIECE_BYTES - 1, end)
            if [piece_start, piece_end] in fetched:
                continue
            payload = read(piece_start, piece_end)
            with sparse.open("r+b") as destination:
                destination.seek(piece_start)
                destination.write(payload)
            fetched.append([piece_start, piece_end])
            progress_path.write_text(json.dumps(fetched))
            print(f"fetched {piece_end + 1 - start:,}/{end + 1 - start:,} bytes", flush=True)

    if not sparse.exists():
        with sparse.open("wb") as destination:
            destination.truncate(SOURCE_BYTES)
    ensure(0, 63)
    with sparse.open("rb") as source:
        index_start = moov_offset(source.read(64))
    ensure(index_start, SOURCE_BYTES - 1)
    segment_start, segment_end = segment_byte_range(sparse)
    ensure(segment_start, segment_end - 1)

    digest = hashlib.sha256()
    with sparse.open("rb") as source:
        source.seek(segment_start)
        remaining = segment_end - segment_start
        while remaining:
            chunk = source.read(min(remaining, 1024 * 1024))
            digest.update(chunk)
            remaining -= len(chunk)
    return sparse, digest.hexdigest()


def cut_clip(source: Path, video: Path) -> None:
    # Only the MP4 index and segment bytes exist locally, so FFmpeg's stream probe
    # decodes zero-filled packets and logs errors; the output is verified instead.
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(SEGMENT_START_SECONDS), "-i", str(source), "-t", str(SEGMENT_DURATION_SECONDS),
            "-vf", f"scale={WIDTH}:{HEIGHT}:flags=lanczos", "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p", "-an", "-map_metadata", "-1",
            "-fflags", "+bitexact", "-movflags", "+faststart", str(video),
        ],
        capture_output=True, check=True,
    )
    decode = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-i", str(video), "-f", "null", "-"],
        text=True, capture_output=True, check=True,
    )
    if decode.stderr.strip():
        raise RuntimeError(f"generated clip does not decode cleanly:\n{decode.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--source", type=Path, help=f"use a local copy of the full {SOURCE_NAME}")
    parser.add_argument("--force", action="store_true", help="replace an existing fixture video")
    args = parser.parse_args()
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            raise SystemExit(f"{executable} is required")
    destination = CAPTURE_ROOT / SLUG
    video = destination / "video.mp4"
    if video.exists() and not args.force:
        print(f"skip {SLUG}: {video} already exists")
        return

    sparse, segment_sha256 = prepare_sparse_source(args.cache, args.source)
    if segment_sha256 != SOURCE_SEGMENT_SHA256:
        raise SystemExit(f"source segment checksum mismatch: {segment_sha256} != {SOURCE_SEGMENT_SHA256}")
    destination.mkdir(parents=True, exist_ok=True)
    cut_clip(sparse, video)

    frame_rate = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate",
         "-of", "csv=p=0", str(video)],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    numerator, denominator = map(int, frame_rate.split("/"))
    checksum = sha256(video)
    metadata = {
        "schemaVersion": 1,
        "monotonicRecordingStartNs": 0,
        "wallClockTimestamp": "2017-01-01T00:00:00Z",
        "orientation": "landscape-left",
        "cameraFormat": f"tanks-and-temples-3840x2160-downscaled-{WIDTH}x{HEIGHT}",
        "width": WIDTH,
        "height": HEIGHT,
        "fps": round(numerator / denominator, 3),
        "codec": "h264",
        "physicalLens": None,
        "appVersion": "fetch-real-capture-v1",
        "nativeScannerModuleVersion": "none",
    }
    fixture_metadata = {
        "fixtureType": "real",
        "scene": "park_building_gable_end",
        "durationMs": round(SEGMENT_DURATION_SECONDS * 1000),
        "cameraPath": {
            "kind": "handheld-arc",
            "description": (
                "Handheld walk around the corner of a single-storey building, ending "
                "front-on to its gable end (door, lattice panel, garage doors)."
            ),
        },
        "source": {
            "dataset": "Tanks and Temples",
            "file": SOURCE_NAME,
            "url": SOURCE_URL,
            "bytes": SOURCE_BYTES,
            "segmentStartSeconds": SEGMENT_START_SECONDS,
            "segmentDurationSeconds": SEGMENT_DURATION_SECONDS,
            "segmentSha256": segment_sha256,
            "license": "CC BY 4.0",
            "licenseUrl": "https://www.tanksandtemples.org/license/",
            "attribution": ATTRIBUTION,
            "changes": f"Trimmed to the stated segment, downscaled from 3840x2160 to {WIDTH}x{HEIGHT}, "
                       "re-encoded as H.264 CRF 18 with metadata removed.",
        },
        "videoSha256": checksum,
    }
    (destination / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (destination / "fixture.json").write_text(json.dumps(fixture_metadata, indent=2) + "\n")
    (destination / "sha256.txt").write_text(f"{checksum}  video.mp4\n")
    print(f"generated {video} ({video.stat().st_size / 1024 / 1024:.1f} MiB); segment sha256 {segment_sha256}")


if __name__ == "__main__":
    main()
