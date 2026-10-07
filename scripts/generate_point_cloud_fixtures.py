#!/usr/bin/env python3
"""Generate deterministic, observed-surface-only point-cloud fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "fixtures" / "point-clouds"
SPACING = 0.1


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float
    nx: float
    ny: float
    nz: float
    red: int
    green: int
    blue: int
    confidence: float
    support_count: int
    source_view_id: int


def values(start: float, end: float, spacing: float = SPACING) -> list[float]:
    """Return an inclusive decimal grid without accumulating float error."""
    steps = round((end - start) / spacing)
    return [round(start + index * spacing, 6) for index in range(steps + 1)]


def surface(
    origin: tuple[float, float, float],
    u: tuple[float, float, float],
    v: tuple[float, float, float],
    u_extent: float,
    v_extent: float,
    normal: tuple[float, float, float],
    color: tuple[int, int, int],
    *,
    view_id: int,
    include: Callable[[float, float], bool] = lambda _u, _v: True,
    confidence: float = 0.95,
    support_count: int = 4,
) -> list[Point]:
    points = []
    for u_value in values(0.0, u_extent):
        for v_value in values(0.0, v_extent):
            if not include(u_value, v_value):
                continue
            position = tuple(
                round(origin[axis] + u_value * u[axis] + v_value * v[axis], 6)
                for axis in range(3)
            )
            points.append(
                Point(
                    *position,
                    *normal,
                    *color,
                    confidence,
                    support_count,
                    view_id,
                )
            )
    return points


def wall_with_window() -> tuple[list[Point], dict]:
    def outside_window(u: float, v: float) -> bool:
        x = -3.0 + u
        return not (-0.8 < x < 0.8 and 1.0 < v < 2.8)

    points = surface(
        (-3.0, 0.0, 0.0), (1, 0, 0), (0, 1, 0), 6.0, 4.0,
        (0, 0, 1), (166, 88, 65), view_id=10, include=outside_window,
    )
    return points, {
        "description": "One observed facade with an open window; no back or side surfaces.",
        "emptyRegions": [
            {"name": "window-opening", "min": [-0.7, 1.1, -0.02], "max": [0.7, 2.7, 0.02]},
            {"name": "unobserved-back", "min": [-2.8, 0.2, -1.0], "max": [2.8, 3.8, -0.2]},
        ],
    }


def building_corner() -> tuple[list[Point], dict]:
    front = surface(
        (-3.0, 0.0, 0.0), (1, 0, 0), (0, 1, 0), 3.0, 4.0,
        (0, 0, 1), (178, 91, 65), view_id=20,
    )
    side = surface(
        (0.0, 0.0, 0.0), (0, 0, -1), (0, 1, 0), 3.0, 4.0,
        (1, 0, 0), (145, 72, 56), view_id=21,
    )
    return front + side, {
        "description": "Two perpendicular observed walls meeting at a vertical corner.",
        "emptyRegions": [
            {"name": "unobserved-interior", "min": [-2.8, 0.2, -2.8], "max": [-0.2, 3.8, -0.2]}
        ],
    }


def half_cube() -> tuple[list[Point], dict]:
    # Only the left, bottom and front faces were observed. The opposing faces
    # deliberately do not exist in this fixture.
    left = surface(
        (0.0, 0.0, 0.0), (0, 1, 0), (0, 0, 1), 2.0, 2.0,
        (-1, 0, 0), (94, 129, 172), view_id=30,
    )
    bottom = surface(
        (0.0, 0.0, 0.0), (1, 0, 0), (0, 0, 1), 2.0, 2.0,
        (0, -1, 0), (82, 117, 158), view_id=31,
    )
    front = surface(
        (0.0, 0.0, 0.0), (1, 0, 0), (0, 1, 0), 2.0, 2.0,
        (0, 0, -1), (108, 143, 184), view_id=32,
    )
    return left + bottom + front, {
        "description": "Three observed faces of a cube; the other three sides remain open.",
        "emptyRegions": [
            {"name": "missing-right", "min": [1.98, 0.2, 0.2], "max": [2.02, 1.8, 1.8]},
            {"name": "missing-top", "min": [0.2, 1.98, 0.2], "max": [1.8, 2.02, 1.8]},
            {"name": "missing-back", "min": [0.2, 0.2, 1.98], "max": [1.8, 1.8, 2.02]},
            {"name": "open-interior", "min": [0.2, 0.2, 0.2], "max": [1.8, 1.8, 1.8]},
        ],
    }


def disconnected_railing() -> tuple[list[Point], dict]:
    points: list[Point] = []
    # Thin, mutually disconnected facade details. These should survive merely
    # because they are small components with strong observation support.
    for index, x in enumerate((-1.5, -0.5, 0.5, 1.5)):
        points += surface(
            (x - 0.05, 0.5, 0.0), (1, 0, 0), (0, 1, 0), 0.1, 2.5,
            (0, 0, 1), (55, 61, 68), view_id=40 + index,
            confidence=0.98, support_count=6,
        )
    for index, y in enumerate((0.5, 1.75, 3.0)):
        points += surface(
            (-1.55, y - 0.05, 0.0), (1, 0, 0), (0, 1, 0), 3.1, 0.1,
            (0, 0, 1), (55, 61, 68), view_id=50 + index,
            confidence=0.98, support_count=6,
        )
    return points, {
        "description": "Strongly supported thin railing details disconnected from all other geometry.",
        "expectedDetachedComponents": 1,
        "emptyRegions": [
            {"name": "space-behind-railing", "min": [-1.4, 0.65, -0.8], "max": [1.4, 2.85, -0.2]}
        ],
    }


def facade_with_ground() -> tuple[list[Point], dict]:
    facade = surface(
        (-2.0, 0.0, 0.0), (1, 0, 0), (0, 1, 0), 4.0, 3.5,
        (0, 0, 1), (194, 155, 116), view_id=60,
    )
    ground = surface(
        (-4.0, 0.0, -2.0), (1, 0, 0), (0, 0, 1), 8.0, 6.0,
        (0, 1, 0), (91, 120, 73), view_id=61, confidence=0.88, support_count=3,
    )
    adjacent_wall = surface(
        (2.5, 0.0, 0.5), (1, 0, 0), (0, 1, 0), 1.0, 2.5,
        (0, 0, 1), (151, 125, 105), view_id=62, confidence=0.82, support_count=3,
    )
    return facade + ground + adjacent_wall, {
        "description": "A facade plus observed ground and adjacent structure that the crop must remove.",
        "recommendedCrop": {"min": [-2.05, -0.05, -0.05], "max": [2.05, 3.55, 0.05]},
        "emptyRegions": [
            {"name": "unobserved-behind-facade", "min": [-1.8, 0.2, -1.0], "max": [1.8, 3.3, -0.2]}
        ],
    }


FIXTURES: dict[str, Callable[[], tuple[list[Point], dict]]] = {
    "wall-with-window": wall_with_window,
    "building-corner": building_corner,
    "half-cube": half_cube,
    "disconnected-railing": disconnected_railing,
    "facade-with-ground": facade_with_ground,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_ply(path: Path, points: Iterable[Point]) -> int:
    records = list(points)
    header = [
        "ply",
        "format ascii 1.0",
        "comment synthetic observed-surface fixture; generated, not reconstructed",
        f"element vertex {len(records)}",
        "property float x", "property float y", "property float z",
        "property float nx", "property float ny", "property float nz",
        "property uchar red", "property uchar green", "property uchar blue",
        "property float confidence",
        "property uint support_count",
        "property uint source_view_id",
        "end_header",
    ]
    rows = [
        f"{p.x:.6f} {p.y:.6f} {p.z:.6f} {p.nx:.1f} {p.ny:.1f} {p.nz:.1f} "
        f"{p.red} {p.green} {p.blue} {p.confidence:.3f} {p.support_count} {p.source_view_id}"
        for p in records
    ]
    path.write_text("\n".join(header + rows) + "\n")
    return len(records)


def generate(slug: str, factory: Callable[[], tuple[list[Point], dict]]) -> None:
    destination = FIXTURE_ROOT / slug
    destination.mkdir(parents=True, exist_ok=True)
    points, assertions = factory()
    ply = destination / "cloud.ply"
    point_count = write_ply(ply, points)
    coordinates = [(point.x, point.y, point.z) for point in points]
    metadata = {
        "schemaVersion": 1,
        "fixtureType": "synthetic-open-surface-point-cloud",
        "coordinateSystem": "+Y up, right-handed",
        "units": "metres",
        "sampleSpacing": SPACING,
        "pointCount": point_count,
        "bounds": {
            "min": [min(row[axis] for row in coordinates) for axis in range(3)],
            "max": [max(row[axis] for row in coordinates) for axis in range(3)],
        },
        "properties": [
            "position", "normal", "sRGB color", "confidence", "support_count", "source_view_id"
        ],
        "assertions": assertions,
        "plySha256": sha256(ply),
    }
    (destination / "fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (destination / "sha256.txt").write_text(f"{metadata['plySha256']}  cloud.ply\n")
    print(f"generated {slug}: {point_count} points")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", nargs="*", choices=FIXTURES, help="fixture names (default: all)")
    args = parser.parse_args()
    selected = args.fixtures or list(FIXTURES)
    for slug in selected:
        generate(slug, FIXTURES[slug])


if __name__ == "__main__":
    main()
