#!/usr/bin/env python3
"""Generate deterministic synthetic mobile-capture fixtures.

The renderer deliberately uses simple, feature-rich planar geometry rather than
photorealism. Its purpose is to provide stable, redistributable inputs for the
standalone pipeline while physical phone captures are collected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_ROOT = ROOT / "fixtures" / "captures"
WIDTH, HEIGHT, FPS, DURATION = 1280, 720, 30, 15


@dataclass(frozen=True)
class Fixture:
    slug: str
    scene: str
    path_description: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    target: tuple[float, float, float]
    path: str


FIXTURES = (
    Fixture(
        "brick-facade-strafe",
        "brick_facade_with_open_windows",
        "Walk 3.2 m left-to-right, about 5.5 m from the wall, with the facade centered.",
        (-1.6, 1.65, 5.5),
        (1.6, 1.65, 5.5),
        (0.0, 1.7, 0.0),
        "linear",
    ),
    Fixture(
        "building-corner-arc",
        "textured_building_corner",
        "Walk a roughly 70-degree, 4.4 m arc around the corner while aiming at the corner.",
        (-3.4, 1.7, 4.0),
        (4.0, 1.7, 3.4),
        (0.0, 1.7, 0.0),
        "arc",
    ),
    Fixture(
        "painted-facade-shallow",
        "low_texture_painted_facade",
        "Walk 2.0 m left-to-right, about 6.5 m from the wall; intentionally low texture.",
        (-1.0, 1.6, 6.5),
        (1.0, 1.6, 6.5),
        (0.0, 1.7, 0.0),
        "linear",
    ),
)


def project(point: tuple[float, float, float], camera: np.ndarray, target: np.ndarray) -> tuple[float, float, float] | None:
    forward = target - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = np.asarray(point) - camera
    depth = float(relative @ forward)
    if depth <= 0.1:
        return None
    focal = 0.92 * WIDTH
    return WIDTH / 2 + focal * float(relative @ right) / depth, HEIGHT / 2 - focal * float(relative @ up) / depth, depth


def polygon(draw: ImageDraw.ImageDraw, points, camera, target, fill, outline=None, width=1):
    projected = [project(p, camera, target) for p in points]
    if any(p is None for p in projected):
        return
    xy = [(round(p[0]), round(p[1])) for p in projected]
    draw.polygon(xy, fill=fill)
    if outline:
        draw.line(xy + [xy[0]], fill=outline, width=width, joint="curve")


def camera_at(fixture: Fixture, t: float) -> np.ndarray:
    if fixture.path == "arc":
        angle = math.radians(130 - 80 * t)
        radius = 5.25
        return np.array([radius * math.cos(angle), 1.7 + 0.03 * math.sin(t * math.pi), radius * math.sin(angle)])
    eased = 0.5 - 0.5 * math.cos(math.pi * t)
    return np.asarray(fixture.start) * (1 - eased) + np.asarray(fixture.end) * eased


def draw_ground(draw, camera, target):
    polygon(draw, [(-8, 0, -3), (8, 0, -3), (8, 0, 10), (-8, 0, 10)], camera, target, "#70865d")
    for x in np.arange(-7.5, 8, 0.5):
        polygon(draw, [(x, 0.002, -2.5), (x + 0.025, 0.002, -2.5), (x + 0.025, 0.002, 8), (x, 0.002, 8)], camera, target, "#637954")


def draw_brick_facade(draw, camera, target, low_texture=False):
    wall = "#c9b69d" if low_texture else "#a6573f"
    polygon(draw, [(-3.6, 0, 0), (3.6, 0, 0), (3.6, 4.2, 0), (-3.6, 4.2, 0)], camera, target, wall)
    if not low_texture:
        for row, y in enumerate(np.arange(0.18, 4.2, 0.28)):
            offset = 0.23 if row % 2 else 0.0
            polygon(draw, [(-3.6, y, -0.012), (3.6, y, -0.012), (3.6, y + 0.018, -0.012), (-3.6, y + 0.018, -0.012)], camera, target, "#e1b6a2")
            for x in np.arange(-3.6 + offset, 3.6, 0.46):
                polygon(draw, [(x, y - 0.26, -0.014), (x + 0.014, y - 0.26, -0.014), (x + 0.014, y, -0.014), (x, y, -0.014)], camera, target, "#7f3e32")
    for cx in (-1.8, 1.35):
        polygon(draw, [(cx - 0.62, 1.0, -0.04), (cx + 0.62, 1.0, -0.04), (cx + 0.62, 2.65, -0.04), (cx - 0.62, 2.65, -0.04)], camera, target, "#17252c", "#eee4d1", 6)
        polygon(draw, [(cx - 0.04, 1.0, -0.06), (cx + 0.04, 1.0, -0.06), (cx + 0.04, 2.65, -0.06), (cx - 0.04, 2.65, -0.06)], camera, target, "#eee4d1")
    polygon(draw, [(-0.48, 0, -0.05), (0.48, 0, -0.05), (0.48, 2.05, -0.05), (-0.48, 2.05, -0.05)], camera, target, "#3f3024", "#ead8bc", 5)
    if low_texture:
        for x, y, color in [(-2.9, 3.55, "#7595a8"), (2.7, 3.4, "#dbc65c"), (-2.8, 0.45, "#b9a587")]:
            polygon(draw, [(x - .09, y - .09, -.03), (x + .09, y - .09, -.03), (x + .09, y + .09, -.03), (x - .09, y + .09, -.03)], camera, target, color)


def draw_corner(draw, camera, target):
    polygon(draw, [(-3.5, 0, 0), (0, 0, 0), (0, 4.3, 0), (-3.5, 4.3, 0)], camera, target, "#bd704d")
    polygon(draw, [(0, 0, 0), (0, 0, -3.5), (0, 4.3, -3.5), (0, 4.3, 0)], camera, target, "#9c5941")
    for i in range(1, 12):
        y = i * .35
        polygon(draw, [(-3.5, y, -.012), (0, y, -.012), (0, y + .018, -.012), (-3.5, y + .018, -.012)], camera, target, "#e2a182")
        polygon(draw, [(-.012, y, 0), (-.012, y, -3.5), (-.012, y + .018, -3.5), (-.012, y + .018, 0)], camera, target, "#d18a6e")
    for wall in ("front", "side"):
        for index in range(3):
            a = -3.1 + index * 1.05
            if wall == "front":
                pts = [(a, 1.1, -.04), (a + .65, 1.1, -.04), (a + .65, 2.45, -.04), (a, 2.45, -.04)]
            else:
                pts = [(-.04, 1.1, a), (-.04, 1.1, a + .65), (-.04, 2.45, a + .65), (-.04, 2.45, a)]
            polygon(draw, pts, camera, target, "#20343d", "#e9d7bd", 5)


def render_frame(fixture: Fixture, frame: int) -> Image.Image:
    t = frame / (FPS * DURATION - 1)
    camera = camera_at(fixture, t)
    target = np.asarray(fixture.target)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#a8c9e7")
    draw = ImageDraw.Draw(image)
    draw_ground(draw, camera, target)
    if fixture.scene == "textured_building_corner":
        draw_corner(draw, camera, target)
    else:
        draw_brick_facade(draw, camera, target, fixture.scene.startswith("low_texture"))
    return image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(fixture: Fixture, force: bool) -> None:
    destination = CAPTURE_ROOT / fixture.slug
    video = destination / "video.mp4"
    if video.exists() and not force:
        print(f"skip {fixture.slug}: {video} already exists")
        return
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{fixture.slug}-") as temporary:
        frames = Path(temporary)
        for index in range(FPS * DURATION):
            render_frame(fixture, index).save(frames / f"frame-{index:04d}.png", compress_level=1)
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(FPS), "-i", str(frames / "frame-%04d.png"),
                "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(video),
            ],
            check=True,
        )
    checksum = sha256(video)
    metadata = {
        "schemaVersion": 1,
        "monotonicRecordingStartNs": 0,
        "wallClockTimestamp": "2026-08-18T00:00:00Z",
        "orientation": "landscape-left",
        "cameraFormat": "synthetic-pinhole-1280x720",
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "codec": "h264",
        "physicalLens": "synthetic-pinhole",
        "appVersion": "fixture-generator-v1",
        "nativeScannerModuleVersion": "fixture-generator-v1",
    }
    fixture_metadata = {
        "fixtureType": "synthetic",
        "scene": fixture.scene,
        "durationMs": DURATION * 1000,
        "coordinateSystem": "+Y up, right-handed, metres",
        "cameraPath": {
            "kind": fixture.path,
            "start": fixture.start,
            "end": fixture.end,
            "lookAt": fixture.target,
            "description": fixture.path_description,
        },
        "videoSha256": checksum,
    }
    (destination / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (destination / "fixture.json").write_text(json.dumps(fixture_metadata, indent=2) + "\n")
    (destination / "sha256.txt").write_text(f"{checksum}  video.mp4\n")
    print(f"generated {video} ({video.stat().st_size / 1024 / 1024:.1f} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace existing generated videos")
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required")
    for fixture in FIXTURES:
        generate(fixture, args.force)


if __name__ == "__main__":
    main()
