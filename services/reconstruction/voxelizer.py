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


# "vectorized" groups points by cell with one sort and aggregates every cell with
# bulk NumPy: 0.6 s instead of 11.5-11.9 s for the reference room at 100 blocks
# (1.65M points, 53-54k blocks). "loop" is the original per-point / per-cell
# implementation, kept for comparison; both give byte-identical voxels.pb.zst.
# Camera rays and splatting always use "loop".
IMPLEMENTATIONS = ("vectorized", "loop")


@dataclass(frozen=True)
class _Grid:
    names: frozenset[str]
    matrix: np.ndarray
    transformed: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    selected: np.ndarray
    voxel_size: float
    dimensions: np.ndarray
    primary: np.ndarray


def _crop_to_grid(points: np.ndarray, transform: Sequence[float], crop_min: Sequence[float],
                  crop_max: Sequence[float], scale_axis: str, blocks: int) -> _Grid:
    """Transform and crop the points and find each kept point's primary cell."""
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
    return _Grid(frozenset(names), matrix, transformed, lower, upper, selected, voxel_size,
                 dimensions, primary_array)


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
    implementation: str = "vectorized",
) -> VoxelizationResult:
    """Transform, crop, aggregate, support-filter, color, and place surface cells."""
    if implementation not in IMPLEMENTATIONS:
        raise VoxelizationError(f"unknown voxelizer implementation {implementation!r}")
    _validate_inputs(transform, crop_min, crop_max, scale_axis, blocks, config)
    grid = _crop_to_grid(points, transform, crop_min, crop_max, scale_axis, blocks)
    if implementation == "vectorized" and not camera_origins and config.splat_radius_voxels <= 0:
        return _voxelize_vectorized(points, grid, palette, config, known_free_cells)
    return _voxelize_loop(points, grid, palette, config, camera_origins, known_free_cells)


def _voxelize_loop(points: np.ndarray, grid: _Grid, palette: Palette, config: VoxelizationConfig,
                   camera_origins: Mapping[int, Sequence[float]] | None,
                   known_free_cells: Iterable[Cell]) -> VoxelizationResult:
    """The original implementation: Python sets per cell and one palette lookup per cell."""
    names, matrix, transformed, lower, upper = (grid.names, grid.matrix, grid.transformed,
                                                grid.lower, grid.upper)
    selected, voxel_size, dimensions, primary_array = (grid.selected, grid.voxel_size,
                                                       grid.dimensions, grid.primary)
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


def _cell_keys(cells: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    """One int64 per in-grid cell, ordered like the (x, y, z) tuples."""
    return (cells[:, 0] * dimensions[1] + cells[:, 1]) * dimensions[2] + cells[:, 2]


def _group_medians(values: np.ndarray, group: np.ndarray, picks: np.ndarray) -> np.ndarray:
    """Per column, the value at sorted position ``picks`` after a stable sort within groups.

    With unit weights ``_weighted_median`` picks the element at rank (k - 1) // 2
    of the cell's k points (cumulative weights 1..k reach k / 2 there, exactly).
    """
    result = np.empty((len(picks), values.shape[1]), dtype=values.dtype)
    for column in range(values.shape[1]):
        order = np.lexsort((values[:, column], group))
        result[:, column] = values[order[picks], column]
    return result


def _closest_palette_indices(colors: np.ndarray, palette: Palette) -> np.ndarray:
    """Index into palette.entries of closest_palette_entry for every row of uint8 colors.

    Scores are computed in bulk for each distinct color. Colors whose best and
    second-best scores lie within 1e-9 (where bulk and per-color rounding could
    disagree, including exact ties) are re-scored with closest_palette_entry.
    """
    packed = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
    unique, inverse = np.unique(packed, return_inverse=True)
    rgb = np.stack([unique >> 16, (unique >> 8) & 255, unique & 255], axis=1)
    entry_lab = np.array([entry.lab for entry in palette.entries], dtype=np.float64)
    penalty = np.array([entry.texture_noise_penalty for entry in palette.entries], dtype=np.float64)
    scores = np.linalg.norm(srgb_to_lab(rgb)[:, None, :] - entry_lab[None], axis=2) + penalty
    best = np.argmin(scores, axis=1)
    if len(palette.entries) > 1:
        lowest_two = np.partition(scores, 1, axis=1)[:, :2]
        for row in np.flatnonzero(lowest_two[:, 1] - lowest_two[:, 0] < 1e-9):
            entry = closest_palette_entry(tuple(int(value) for value in rgb[row]), palette)
            best[row] = palette.entries.index(entry)
    return best[inverse.reshape(-1)]


def _voxelize_vectorized(points: np.ndarray, grid: _Grid, palette: Palette, config: VoxelizationConfig,
                         known_free_cells: Iterable[Cell]) -> VoxelizationResult:
    """Same result as _voxelize_loop without camera rays or splatting, using one sort per pass."""
    names, transformed, lower, upper = grid.names, grid.transformed, grid.lower, grid.upper
    voxel_size, dimensions = grid.voxel_size, grid.dimensions
    free = {tuple(map(int, cell)) for cell in known_free_cells}
    keys = _cell_keys(grid.primary, dimensions)
    keep = np.ones(len(keys), dtype=bool)
    if free:
        free_array = np.array(sorted(free), dtype=np.int64).reshape(-1, 3)
        in_grid = ((free_array >= 0) & (free_array < dimensions)).all(axis=1)
        keep = ~np.isin(keys, _cell_keys(free_array[in_grid], dimensions))
    # A stable sort keeps each cell's points in ascending point order, as the
    # loop's sorted(member set) does.
    order = np.argsort(keys[keep], kind="stable")
    members = grid.selected[keep][order]
    sorted_keys = keys[keep][order]
    cells_of_members = grid.primary[keep][order]
    if not len(members):
        raise VoxelizationError("no cells pass observation-support thresholds")
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    counts = np.diff(np.r_[starts, len(members)])
    group = np.repeat(np.arange(len(starts)), counts)
    cells = cells_of_members[starts]

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
        normals_all = normals_all @ grid.matrix[:3, :3].T
    angle_all = (np.clip(np.asarray(points["view_angle_weight"], dtype=float), 0, 1)
                 if "view_angle_weight" in names else np.ones(len(points), dtype=float))

    if support_all is not None:
        # As in the loop: the maximum, never the sum, of per-point view counts.
        views = np.maximum.reduceat(support_all[members], starts)
    elif view_all is not None:
        member_views = view_all[members]
        by_view = np.lexsort((member_views, group))
        view_sorted, group_sorted = member_views[by_view], group[by_view]
        first = np.r_[True, (group_sorted[1:] != group_sorted[:-1]) | (view_sorted[1:] != view_sorted[:-1])]
        views = np.bincount(group_sorted[first], minlength=len(starts))
    else:
        views = np.zeros(len(starts), dtype=np.int64)
    clipped = np.clip(confidence_all[members], 0, 1)
    # Unit weights (no confidence or view-angle property, the case for every
    # pipeline PLY) make the cell averages and weighted medians exact in bulk.
    # Other weights reuse the loop's per-cell arithmetic so results stay identical.
    unit_weights = bool(np.all(clipped == 1.0) and np.all(angle_all[members] == 1.0))
    ends = np.r_[starts[1:], len(members)]
    if unit_weights:
        confidence = np.ones(len(starts), dtype=float)
    else:
        confidence = np.array([float(np.average(clipped[start:end])) for start, end in zip(starts, ends)])
    centers = lower + (cells.astype(float) + 0.5) * voxel_size
    nearest = np.minimum.reduceat(np.linalg.norm(transformed[members] - centers[group], axis=1), starts)
    allowed_distance = config.allowed_surface_radius_voxels * voxel_size
    has_views = view_all is not None or support_all is not None
    accepted = ~((counts < config.minimum_supporting_points)
                 | (has_views & (views < config.minimum_distinct_views))
                 | (confidence < config.minimum_confidence)
                 | (nearest > allowed_distance))
    rejected = int((~accepted).sum())

    linear_all = _srgb_to_linear(colors_all[members])
    if unit_weights:
        picks = starts + (counts - 1) // 2
        robust_linear = _group_medians(linear_all, group, picks)
        normal = (_group_medians(normals_all[members], group, picks) if normals_all is not None
                  else np.zeros((len(starts), 3), dtype=float))
    else:
        robust_linear = np.zeros((len(starts), 3), dtype=float)
        normal = np.zeros((len(starts), 3), dtype=float)
        weights_all = np.maximum(clipped * angle_all[members], np.finfo(float).eps)
        for cell_index in np.flatnonzero(accepted):
            start, end = starts[cell_index], ends[cell_index]
            robust_linear[cell_index] = _weighted_median(linear_all[start:end], weights_all[start:end])
            if normals_all is not None:
                normal[cell_index] = _weighted_median(normals_all[members[start:end]], weights_all[start:end])
    robust_srgb = np.where(robust_linear <= 0.0031308, robust_linear * 12.92,
                           1.055 * robust_linear ** (1 / 2.4) - 0.055)
    colors = np.clip(np.rint(robust_srgb * 255), 0, 255).astype(np.int64)
    if normals_all is not None:
        length = np.linalg.norm(normal, axis=1)
        normal = np.where((length > 1e-12)[:, None], normal / np.where(length > 1e-12, length, 1.0)[:, None],
                          normal)

    kept = np.flatnonzero(accepted)
    if config.isolated_cell_minimum_points and len(kept):
        kept_keys = _cell_keys(cells[kept], dimensions)
        has_neighbour = np.zeros(len(kept), dtype=bool)
        for offset in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            neighbour = cells[kept] + np.asarray(offset)
            in_grid = ((neighbour >= 0) & (neighbour < dimensions)).all(axis=1)
            has_neighbour |= in_grid & np.isin(_cell_keys(neighbour, dimensions), kept_keys)
        isolated = (counts[kept] < config.isolated_cell_minimum_points) & ~has_neighbour
        rejected += int(isolated.sum())
        kept = kept[~isolated]
    if not len(kept):
        raise VoxelizationError("no cells pass observation-support thresholds")

    source = cells[kept]
    minimum = np.min(source, axis=0)
    maximum = np.max(source, axis=0)
    translation = (-int(round((minimum[0] + maximum[0]) / 2)),
                   config.flat_surface_y - int(minimum[1]),
                   -int(round((minimum[2] + maximum[2]) / 2)))
    # The loop's placement order: (x // 16, z // 16, y, z, x).
    placement = kept[np.lexsort((source[:, 0], source[:, 2], source[:, 1], source[:, 2] // 16,
                                 source[:, 0] // 16))]
    entries = palette.entries
    palette_index = _closest_palette_indices(colors[placement], palette)
    voxels: list[Voxel] = []
    for cell, entry_index, points_count, view_count, distance, cell_confidence, color, cell_normal in zip(
            cells[placement].tolist(), palette_index.tolist(), counts[placement].tolist(),
            views[placement].tolist(), nearest[placement].tolist(), confidence[placement].tolist(),
            colors[placement].tolist(), normal[placement].tolist(), strict=True):
        entry = entries[entry_index]
        voxel = Voxel(cell[0] + translation[0], cell[1] + translation[1], cell[2] + translation[2],
                      tuple(cell), entry.id, entry.block_state, points_count, view_count,
                      distance, cell_confidence, tuple(color), tuple(cell_normal))
        # The final exporter guard is intentionally repeated after all cleanup/placement.
        if (voxel.supporting_points < config.minimum_supporting_points
                or voxel.nearest_observed_point_distance > allowed_distance
                or voxel.source_cell in free):
            raise AssertionError("no-completion assertion failed")
        voxels.append(voxel)
    return VoxelizationResult(voxel_size, tuple(voxels), frozenset(free), tuple(lower), tuple(upper),
                              translation, len(points), len(grid.selected), rejected)


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
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, default="vectorized",
                        help="vectorized (default) or the original per-cell loop; both give identical output")
    args = parser.parse_args()
    result = voxelize_points(read_ply(args.source), transform=args.transform,
                             crop_min=args.crop_min, crop_max=args.crop_max,
                             scale_axis=args.axis, blocks=args.blocks,
                             palette=load_palette(args.palette),
                             config=VoxelizationConfig(args.minimum_points, args.minimum_views,
                                                       args.minimum_confidence,
                                                       math.sqrt(3) / 2 + 1e-9,
                                                       args.splat_radius),
                             implementation=args.implementation)
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
