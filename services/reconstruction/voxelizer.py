"""Support-preserving surface voxelization for canonical reconstruction PLYs.

Only cells supported by observed points can be emitted. There is deliberately
no flood fill, hull, dilation, closing, watertight conversion, or interior fill.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    from .point_cloud import read_ply
except ImportError:  # Direct script execution.
    from point_cloud import read_ply


Cell = tuple[int, int, int]
IDENTITY_4X4 = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class VoxelizationError(ValueError):
    """Input cannot produce an auditable surface-voxel result."""


@dataclass(frozen=True)
class PaletteEntry:
    id: int
    block_state: str
    srgb: tuple[int, int, int]
    lab: tuple[float, float, float]
    texture_noise_penalty: float


@dataclass(frozen=True)
class Palette:
    version: str
    minecraft_version: str
    entries: tuple[PaletteEntry, ...]


@dataclass(frozen=True)
class VoxelizationConfig:
    minimum_supporting_points: int = 1
    minimum_distinct_views: int = 1
    minimum_confidence: float = 0.0
    allowed_surface_radius_voxels: float = math.sqrt(3.0) / 2.0 + 1e-9
    splat_radius_voxels: float = 0.0
    isolated_cell_minimum_points: int = 0
    flat_surface_y: int = 0


@dataclass(frozen=True)
class Voxel:
    x: int
    y: int
    z: int
    source_cell: Cell
    palette_id: int
    block_state: str
    supporting_points: int
    distinct_supporting_frames: int
    nearest_observed_point_distance: float
    confidence: float
    color_srgb: tuple[int, int, int]
    normal: tuple[float, float, float]


@dataclass(frozen=True)
class VoxelizationResult:
    voxel_size: float
    voxels: tuple[Voxel, ...]
    known_free_cells: frozenset[Cell]
    transformed_crop_min: tuple[float, float, float]
    transformed_crop_max: tuple[float, float, float]
    translation: Cell
    input_points: int
    cropped_points: int
    rejected_cells: int


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64) / 255.0
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _linear_rgb_to_lab(linear: np.ndarray) -> np.ndarray:
    rgb = np.asarray(linear, dtype=np.float64)
    xyz = rgb @ np.array([
        [0.4124564, 0.2126729, 0.0193339],
        [0.3575761, 0.7151522, 0.1191920],
        [0.1804375, 0.0721750, 0.9503041],
    ])
    xyz /= np.array([0.95047, 1.0, 1.08883])
    delta = 6.0 / 29.0
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4.0 / 29.0)
    return np.stack((116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])), axis=-1)


def srgb_to_lab(srgb: Sequence[float] | np.ndarray) -> np.ndarray:
    return _linear_rgb_to_lab(_srgb_to_linear(np.asarray(srgb)))


def load_palette(path: Path) -> Palette:
    payload = json.loads(path.read_text())
    if payload.get("schemaVersion") != 1 or not payload.get("entries"):
        raise VoxelizationError("palette must be a non-empty schema version 1 document")
    entries: list[PaletteEntry] = []
    seen_ids: set[int] = set()
    seen_states: set[str] = set()
    for raw in payload["entries"]:
        identifier = int(raw["id"])
        state = str(raw["blockState"])
        color = tuple(int(value) for value in raw["srgb"])
        penalty = float(raw.get("textureNoisePenalty", 0.0))
        if identifier <= 0 or identifier in seen_ids:
            raise VoxelizationError("palette IDs must be positive and unique")
        if not state.startswith("minecraft:") or state in seen_states:
            raise VoxelizationError("palette block states must be unique minecraft IDs")
        if len(color) != 3 or not all(0 <= value <= 255 for value in color) or penalty < 0:
            raise VoxelizationError("palette colors and noise penalties are invalid")
        seen_ids.add(identifier)
        seen_states.add(state)
        entries.append(PaletteEntry(identifier, state, color,
                                    tuple(float(v) for v in srgb_to_lab(color)), penalty))
    return Palette(str(payload["paletteVersion"]), str(payload["minecraftVersion"]), tuple(entries))


def closest_palette_entry(srgb: Sequence[float], palette: Palette) -> PaletteEntry:
    lab = srgb_to_lab(srgb)
    scores = [float(np.linalg.norm(lab - entry.lab)) + entry.texture_noise_penalty
              for entry in palette.entries]
    return palette.entries[int(np.argmin(scores))]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.empty(values.shape[1], dtype=np.float64)
    for axis in range(values.shape[1]):
        order = np.argsort(values[:, axis], kind="stable")
        ordered_weights = weights[order]
        threshold = ordered_weights.sum() / 2.0
        result[axis] = values[order[np.searchsorted(np.cumsum(ordered_weights), threshold)], axis]
    return result


def _validate_inputs(transform: Sequence[float], crop_min: Sequence[float],
                     crop_max: Sequence[float], axis: str, blocks: int,
                     config: VoxelizationConfig) -> None:
    if len(transform) != 16 or not np.isfinite(transform).all():
        raise VoxelizationError("transform must contain 16 finite values")
    matrix = np.asarray(transform, dtype=float).reshape(4, 4)
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise VoxelizationError("transform must be affine")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise VoxelizationError("transform must be a proper rigid transform")
    if len(crop_min) != 3 or len(crop_max) != 3 or not np.isfinite(crop_min).all() or not np.isfinite(crop_max).all():
        raise VoxelizationError("crop bounds must contain three finite values")
    if any(lower >= upper for lower, upper in zip(crop_min, crop_max, strict=True)):
        raise VoxelizationError("crop bounds must have positive extent")
    if axis not in AXIS_INDEX or blocks <= 0:
        raise VoxelizationError("scale axis and block count are invalid")
    if (config.minimum_supporting_points <= 0 or config.minimum_distinct_views <= 0
            or not 0 <= config.minimum_confidence <= 1
            or config.allowed_surface_radius_voxels <= 0
            or not 0 <= config.splat_radius_voxels <= 0.5
            or config.isolated_cell_minimum_points < 0):
        raise VoxelizationError("voxelization thresholds are invalid")


def _ray_cells(origin: np.ndarray, endpoint: np.ndarray, grid_origin: np.ndarray,
               voxel_size: float) -> set[Cell]:
    """Conservatively sample a visibility segment; exclude its observed endpoint."""
    distance_voxels = float(np.linalg.norm(endpoint - origin) / voxel_size)
    steps = max(1, math.ceil(distance_voxels * 2.0))
    samples = origin + np.arange(steps, dtype=float)[:, None] / steps * (endpoint - origin)
    indices = np.floor((samples - grid_origin) / voxel_size).astype(np.int64)
    return {tuple(int(value) for value in row) for row in indices}


def _candidate_cells(point: np.ndarray, primary: Cell, grid_origin: np.ndarray,
                     voxel_size: float, splat_radius_voxels: float) -> Iterable[Cell]:
    yield primary
    if splat_radius_voxels <= 0:
        return
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                candidate = (primary[0] + dx, primary[1] + dy, primary[2] + dz)
                if candidate == primary:
                    continue
                center = grid_origin + (np.asarray(candidate, dtype=float) + 0.5) * voxel_size
                if np.linalg.norm(point - center) <= splat_radius_voxels * voxel_size:
                    yield candidate


def voxelize_points(
    points: np.ndarray,
    *,
    transform: Sequence[float] = IDENTITY_4X4,
    crop_min: Sequence[float],
    crop_max: Sequence[float],
    scale_axis: str,
    blocks: int,
    palette: Palette,
    config: VoxelizationConfig = VoxelizationConfig(),
    camera_origins: Mapping[int, Sequence[float]] | None = None,
    known_free_cells: Iterable[Cell] = (),
) -> VoxelizationResult:
    """Transform, crop, aggregate, support-filter, color, and place surface cells."""
    _validate_inputs(transform, crop_min, crop_max, scale_axis, blocks, config)
    names = set(points.dtype.names or ())
    required = {"x", "y", "z", "red", "green", "blue"}
    if not required <= names:
        raise VoxelizationError(f"point cloud is missing properties: {sorted(required - names)}")
    xyz = np.column_stack([points[name] for name in ("x", "y", "z")]).astype(np.float64)
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    transformed = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    lower = np.asarray(crop_min, dtype=np.float64)
    upper = np.asarray(crop_max, dtype=np.float64)
    finite = np.isfinite(transformed).all(axis=1)
    inside = finite & (transformed >= lower).all(axis=1) & (transformed <= upper).all(axis=1)
    selected = np.flatnonzero(inside)
    if not len(selected):
        raise VoxelizationError("crop contains no observed points")
    extent = upper - lower
    voxel_size = float(extent[AXIS_INDEX[scale_axis]] / blocks)
    dimensions = np.maximum(1, np.ceil(extent / voxel_size - 1e-12).astype(np.int64))
    scaled = (transformed[selected] - lower) / voxel_size
    # PLY float32 roundoff must not randomly put samples intended to lie on the
    # same voxel plane on opposite sides of it.
    nearest_plane = np.rint(scaled)
    scaled = np.where(np.isclose(scaled, nearest_plane, atol=2e-6, rtol=0), nearest_plane, scaled)
    primary_array = np.floor(scaled).astype(np.int64)
    primary_array = np.minimum(primary_array, dimensions - 1)
    primary_cells = [tuple(int(value) for value in row) for row in primary_array]
    observed_cells = set(primary_cells)

    free = {tuple(map(int, cell)) for cell in known_free_cells}
    if camera_origins:
        if "source_view_id" not in names:
            raise VoxelizationError("camera rays require source_view_id point provenance")
        transformed_origins: dict[int, np.ndarray] = {}
        for view, origin in camera_origins.items():
            origin_array = np.asarray(origin, dtype=float)
            if origin_array.shape != (3,) or not np.isfinite(origin_array).all():
                raise VoxelizationError("camera origins must contain three finite values")
            transformed_origins[int(view)] = origin_array @ matrix[:3, :3].T + matrix[:3, 3]
        for point_index, point in zip(selected, transformed[selected], strict=True):
            view = int(points["source_view_id"][point_index])
            if view in transformed_origins:
                free.update(_ray_cells(transformed_origins[view], point, lower, voxel_size))
        free = {cell for cell in free
                if all(0 <= cell[axis] < dimensions[axis] for axis in range(3))}
        # A directly observed cell is not classified as free due to a crossing ray.
        free.difference_update(observed_cells)

    members: dict[Cell, set[int]] = {}
    for point_index, point, primary in zip(selected, transformed[selected], primary_cells, strict=True):
        for cell in _candidate_cells(point, primary, lower, voxel_size, config.splat_radius_voxels):
            if all(0 <= cell[axis] < dimensions[axis] for axis in range(3)) and cell not in free:
                members.setdefault(cell, set()).add(int(point_index))

    confidence_all = (np.asarray(points["confidence"], dtype=float) if "confidence" in names
                      else np.ones(len(points), dtype=float))
    view_all = (np.asarray(points["source_view_id"], dtype=np.int64)
                if "source_view_id" in names else None)
    support_all = (np.asarray(points["support_count"], dtype=np.int64)
                   if "support_count" in names else None)
    colors_all = np.column_stack([points[name] for name in ("red", "green", "blue")]).astype(float)
    normals_all = None
    if {"nx", "ny", "nz"} <= names:
        normals_all = np.column_stack([points[name] for name in ("nx", "ny", "nz")]).astype(float)
        normals_all = normals_all @ matrix[:3, :3].T
    angle_all = (np.clip(np.asarray(points["view_angle_weight"], dtype=float), 0, 1)
                 if "view_angle_weight" in names else np.ones(len(points), dtype=float))

    aggregates: dict[Cell, dict[str, object]] = {}
    rejected = 0
    allowed_distance = config.allowed_surface_radius_voxels * voxel_size
    for cell, point_indices_set in members.items():
        point_indices = np.fromiter(sorted(point_indices_set), dtype=np.int64)
        supporting_points = len(point_indices)
        if support_all is not None:
            # COLMAP-style fused records may retain only a count, not every view
            # ID. The maximum is conservative: the same views can support all
            # points in a cell, so counts must never be summed.
            distinct_views = int(support_all[point_indices].max())
        else:
            distinct_views = len(np.unique(view_all[point_indices])) if view_all is not None else 0
        confidence = float(np.average(np.clip(confidence_all[point_indices], 0, 1)))
        center = lower + (np.asarray(cell, dtype=float) + 0.5) * voxel_size
        nearest = float(np.linalg.norm(transformed[point_indices] - center, axis=1).min())
        if (supporting_points < config.minimum_supporting_points
                or ((view_all is not None or support_all is not None)
                    and distinct_views < config.minimum_distinct_views)
                or confidence < config.minimum_confidence
                or nearest > allowed_distance or cell in free):
            rejected += 1
            continue
        weights = np.maximum(np.clip(confidence_all[point_indices], 0, 1)
                             * angle_all[point_indices], np.finfo(float).eps)
        linear = _srgb_to_linear(colors_all[point_indices])
        robust_linear = _weighted_median(linear, weights)
        robust_srgb = np.where(robust_linear <= 0.0031308, robust_linear * 12.92,
                               1.055 * robust_linear ** (1 / 2.4) - 0.055)
        color = tuple(int(value) for value in np.clip(np.rint(robust_srgb * 255), 0, 255))
        normal = np.zeros(3, dtype=float)
        if normals_all is not None:
            normal = _weighted_median(normals_all[point_indices], weights)
            length = np.linalg.norm(normal)
            if length > 1e-12:
                normal /= length
        entry = closest_palette_entry(color, palette)
        aggregates[cell] = {
            "palette": entry, "points": supporting_points, "views": distinct_views,
            "nearest": nearest, "confidence": confidence, "color": color,
            "normal": tuple(float(value) for value in normal),
        }

    if config.isolated_cell_minimum_points:
        offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        isolated = [cell for cell, value in aggregates.items()
                    if int(value["points"]) < config.isolated_cell_minimum_points
                    and not any(tuple(cell[a] + offset[a] for a in range(3)) in aggregates
                                for offset in offsets)]
        for cell in isolated:
            del aggregates[cell]
        rejected += len(isolated)
    if not aggregates:
        raise VoxelizationError("no cells pass observation-support thresholds")

    source_cells = list(aggregates)
    minimum = np.min(source_cells, axis=0)
    maximum = np.max(source_cells, axis=0)
    translation = (-int(round((minimum[0] + maximum[0]) / 2)),
                   config.flat_surface_y - int(minimum[1]),
                   -int(round((minimum[2] + maximum[2]) / 2)))
    voxels: list[Voxel] = []
    for cell in sorted(source_cells, key=lambda item: (item[0] // 16, item[2] // 16, item[1], item[2], item[0])):
        value = aggregates[cell]
        entry = value["palette"]
        assert isinstance(entry, PaletteEntry)
        voxel = Voxel(cell[0] + translation[0], cell[1] + translation[1], cell[2] + translation[2],
                      cell, entry.id, entry.block_state, int(value["points"]), int(value["views"]),
                      float(value["nearest"]), float(value["confidence"]), value["color"], value["normal"])
        # The final exporter guard is intentionally repeated after all cleanup/placement.
        if (voxel.supporting_points < config.minimum_supporting_points
                or voxel.nearest_observed_point_distance > allowed_distance
                or voxel.source_cell in free):
            raise AssertionError("no-completion assertion failed")
        voxels.append(voxel)
    return VoxelizationResult(voxel_size, tuple(voxels), frozenset(free), tuple(lower), tuple(upper),
                              translation, len(points), len(selected), rejected)


def result_to_dict(result: VoxelizationResult, palette: Palette) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "paletteVersion": palette.version,
        "minecraftVersion": palette.minecraft_version,
        "voxelSize": result.voxel_size,
        "inputPoints": result.input_points,
        "croppedPoints": result.cropped_points,
        "occupiedBlocks": len(result.voxels),
        "rejectedCells": result.rejected_cells,
        "translation": list(result.translation),
        "voxels": [
            {"x": voxel.x, "y": voxel.y, "z": voxel.z, "sourceCell": list(voxel.source_cell),
             "paletteId": voxel.palette_id, "blockState": voxel.block_state,
             "supportingPoints": voxel.supporting_points,
             "distinctSupportingFrames": voxel.distinct_supporting_frames,
             "nearestObservedPointDistance": voxel.nearest_observed_point_distance,
             "confidence": voxel.confidence, "colorSrgb": list(voxel.color_srgb),
             "normal": list(voxel.normal)}
            for voxel in result.voxels
        ],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--crop-min", nargs=3, type=float, required=True)
    parser.add_argument("--crop-max", nargs=3, type=float, required=True)
    parser.add_argument("--axis", choices=AXIS_INDEX, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--transform", nargs=16, type=float, default=IDENTITY_4X4)
    parser.add_argument("--palette", type=Path, default=root / "packages/block-palette/palette-v1.json")
    parser.add_argument("--minimum-points", type=int, default=1)
    parser.add_argument("--minimum-views", type=int, default=1)
    parser.add_argument("--minimum-confidence", type=float, default=0.0)
    parser.add_argument("--splat-radius", type=float, default=0.0,
                        help="optional radius in voxels, constrained to 0..0.5")
    args = parser.parse_args()
    result = voxelize_points(read_ply(args.source), transform=args.transform,
                             crop_min=args.crop_min, crop_max=args.crop_max,
                             scale_axis=args.axis, blocks=args.blocks,
                             palette=load_palette(args.palette),
                             config=VoxelizationConfig(args.minimum_points, args.minimum_views,
                                                       args.minimum_confidence,
                                                       math.sqrt(3) / 2 + 1e-9,
                                                       args.splat_radius))
    palette = load_palette(args.palette)
    if args.output.name.endswith(".pb.zst"):
        from voxel_contract import write as write_voxel_contract
        write_voxel_contract(args.output, result, palette)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result_to_dict(result, palette),
                                          indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
