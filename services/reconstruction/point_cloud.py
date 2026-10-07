"""Conservative filtering and preview generation for COLMAP fused PLY files.

The implementation intentionally never creates points.  Filtering can only remove
points and alignment applies one rigid rotation.  When depth completion filled MVS
holes before fusion, ``--observed-reference`` (a fusion of the observed depth maps
alone) is used to add a per-point ``observed`` property: 1 for points backed by
multi-view stereo, 0 for points that exist only because of monocular completion.
``--all-inferred`` tags every point 0, for fast mode where no stereo depth exists.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np


PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
    "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def read_ply(path: Path) -> np.ndarray:
    """Read an ASCII or binary-little-endian PLY with scalar vertex properties."""
    with path.open("rb") as source:
        if source.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        fmt = None
        count = None
        properties: list[tuple[str, str]] = []
        in_vertices = False
        while True:
            raw = source.readline()
            if not raw:
                raise ValueError(f"unterminated PLY header: {path}")
            fields = raw.decode("ascii").strip().split()
            if not fields or fields[0] in {"comment", "obj_info"}:
                continue
            if fields[0] == "format":
                fmt = fields[1]
            elif fields[0] == "element":
                in_vertices = fields[1] == "vertex"
                if in_vertices:
                    count = int(fields[2])
            elif fields[0] == "property" and in_vertices:
                if fields[1] == "list":
                    raise ValueError("list-valued vertex properties are unsupported")
                if fields[1] not in PLY_TYPES:
                    raise ValueError(f"unsupported PLY property type: {fields[1]}")
                properties.append((fields[2], PLY_TYPES[fields[1]]))
            elif fields[0] == "end_header":
                break
        if count is None or not properties:
            raise ValueError(f"PLY has no vertex schema: {path}")
        dtype = np.dtype(properties)
        if fmt == "binary_little_endian":
            points = np.fromfile(source, dtype=dtype, count=count)
        elif fmt == "ascii":
            rows = [source.readline().split() for _ in range(count)]
            points = np.empty(count, dtype=dtype)
            for index, (name, _) in enumerate(properties):
                points[name] = [row[index] for row in rows]
        else:
            raise ValueError(f"unsupported PLY format: {fmt}")
    if len(points) != count:
        raise ValueError(f"PLY declares {count} vertices but contains {len(points)}")
    for coordinate in ("x", "y", "z"):
        if coordinate not in points.dtype.names:
            raise ValueError(f"PLY is missing {coordinate} coordinates")
    return points


def write_ply(path: Path, points: np.ndarray) -> None:
    """Write the complete structured vertex record as binary little endian."""
    type_names = {
        "i1": "char", "u1": "uchar", "<i2": "short", "<u2": "ushort",
        "<i4": "int", "<u4": "uint", "<f4": "float", "<f8": "double",
    }
    lines = ["ply", "format binary_little_endian 1.0", "comment video-to-minecraft observed points only",
             f"element vertex {len(points)}"]
    for name in points.dtype.names or ():
        scalar = points.dtype.fields[name][0]
        key = scalar.str.replace("|", "").replace(">", "<")
        if key not in type_names:
            raise ValueError(f"cannot write PLY property {name} with type {scalar}")
        lines.append(f"property {type_names[key]} {name}")
    lines.append("end_header")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as destination:
        destination.write(("\n".join(lines) + "\n").encode("ascii"))
        points.tofile(destination)


def rotation_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1e-9:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array([[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]],
                     [-cross[1], cross[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


# Camera-based alignment leaves the floor a degree or two off level (1.5 degrees on
# the reference room scan), which still steps a 100-block floor by ~3 blocks. The
# dominant near-horizontal plane with most of the scene above it is snapped to
# exactly level; the correction is capped so a roof or ramp cannot tip the scan.
FLOOR_MAX_CORRECTION_DEGREES = 10.0
FLOOR_SEARCH_STEP_DEGREES = 0.5
FLOOR_THICKNESS_OF_DIAGONAL = 0.005
FLOOR_MIN_SUPPORT = 0.05
FLOOR_MIN_POINTS = 500
FLOOR_MIN_FRACTION_ABOVE = 0.7
FLOOR_SAMPLE_POINTS = 50_000


def floor_leveling_rotation(xyz: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Rotation making the floor exactly level, for a cloud already roughly +Y up.

    Every tilt within the cap is tried: points are projected on the tilted normal
    and binned at floor thickness. The floor is the fullest bin with at least
    ``FLOOR_MIN_FRACTION_ABOVE`` of the scene above it; its points are then fitted
    by least squares. Returns identity when no such plane is well supported.
    """
    step = max(1, len(xyz) // FLOOR_SAMPLE_POINTS)
    sample = xyz[::step]
    low, high = np.percentile(sample, [1, 99], axis=0)
    thickness = max(float(np.linalg.norm(high - low)) * FLOOR_THICKNESS_OF_DIAGONAL, np.finfo(float).eps)
    limit = math.radians(FLOOR_MAX_CORRECTION_DEGREES)
    tilts = np.arange(-limit, limit + 1e-9, math.radians(FLOOR_SEARCH_STEP_DEGREES))
    best: tuple[int, np.ndarray, float] | None = None
    for tilt_x in tilts:
        for tilt_z in tilts:
            normal = np.array([math.tan(tilt_x), 1.0, math.tan(tilt_z)])
            normal /= np.linalg.norm(normal)
            if math.degrees(math.acos(normal[1])) > FLOOR_MAX_CORRECTION_DEGREES:
                continue
            heights = sample @ normal
            bins = np.floor((heights - heights.min()) / thickness).astype(np.int64)
            counts = np.bincount(bins)
            above = 1.0 - np.cumsum(counts) / len(sample)
            eligible = np.flatnonzero(above >= FLOOR_MIN_FRACTION_ABOVE)
            if not len(eligible):
                continue
            peak = int(eligible[np.argmax(counts[eligible])])
            if best is None or counts[peak] > best[0]:
                best = (int(counts[peak]), normal, heights.min() + (peak + 0.5) * thickness)
    details: dict[str, Any] = {"applied": False, "maxCorrectionDegrees": FLOOR_MAX_CORRECTION_DEGREES}
    if best is None or best[0] < max(FLOOR_MIN_POINTS, FLOOR_MIN_SUPPORT * len(sample)):
        return np.eye(3), details
    _, normal, height = best
    floor = sample[np.abs(sample @ normal - height) <= thickness]
    fitted = np.linalg.svd(floor - floor.mean(axis=0), full_matrices=False)[2][-1]
    fitted = fitted if fitted[1] > 0 else -fitted
    correction = math.degrees(math.acos(min(1.0, float(fitted[1]))))
    details.update(supportFraction=len(floor) / len(sample), correctionDegrees=correction)
    if correction > FLOOR_MAX_CORRECTION_DEGREES:
        return np.eye(3), details
    details["applied"] = True
    return rotation_from_to(fitted, np.array([0.0, 1.0, 0.0])), details


def conservative_radius_mask(xyz: np.ndarray, *, radius: float, minimum_neighbors: int) -> np.ndarray:
    """Remove only points whose 27 neighboring radius-sized cells are sparse.

    Counting whole adjacent cells overestimates true radius neighbors, making this
    deliberately less aggressive than an exact radius filter.
    """
    cells = np.floor(xyz / radius).astype(np.int64)
    unique, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    lookup = {tuple(cell): int(count) for cell, count in zip(unique, counts, strict=True)}
    neighborhood = np.empty(len(unique), dtype=np.int64)
    offsets = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
    for index, cell in enumerate(unique):
        neighborhood[index] = sum(lookup.get(tuple(cell + offset), 0) for offset in offsets)
    return neighborhood[inverse] >= minimum_neighbors


def conservative_outlier_mask(
    xyz: np.ndarray, *, radius: float, minimum_neighbors: int
) -> tuple[np.ndarray, str]:
    """Apply radius and very forgiving statistical filtering when SciPy is present."""
    try:
        from scipy.spatial import cKDTree  # type: ignore[import-not-found]
    except (ImportError, ValueError):
        # ValueError: SciPy compiled against an incompatible NumPy ABI.
        return conservative_radius_mask(
            xyz, radius=radius, minimum_neighbors=minimum_neighbors
        ), "voxel-radius-fallback"
    tree = cKDTree(xyz)
    radius_counts = tree.query_ball_point(xyz, radius, return_length=True, workers=-1)
    neighbor_count = min(8, len(xyz))
    distances, _ = tree.query(xyz, k=neighbor_count, workers=-1)
    mean_distance = np.asarray(distances).reshape(len(xyz), -1).mean(axis=1)
    median = float(np.median(mean_distance))
    mad = float(np.median(np.abs(mean_distance - median)))
    statistical_limit = median + 6.0 * max(mad, np.finfo(float).eps)
    return (radius_counts >= minimum_neighbors) & (mean_distance <= statistical_limit), "scipy-radius-statistical"


def observed_mask(xyz: np.ndarray, reference: np.ndarray, *, radius: float) -> np.ndarray:
    """True where a point has an observed-only fusion point within its 27 neighbouring cells."""
    cells = np.floor(xyz / radius).astype(np.int64)
    reference_cells = np.floor(reference / radius).astype(np.int64)
    origin = np.minimum(cells.min(axis=0), reference_cells.min(axis=0)) - 1
    span = np.maximum(cells.max(axis=0), reference_cells.max(axis=0)) - origin + 2

    def keys(values: np.ndarray) -> np.ndarray:
        shifted = values - origin
        return (shifted[:, 0] * span[1] + shifted[:, 1]) * span[2] + shifted[:, 2]

    occupied = np.unique(keys(reference_cells))
    near = np.zeros(len(cells), dtype=bool)
    for offset in [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]:
        near |= np.isin(keys(cells + np.array(offset)), occupied, assume_unique=False)
    return near


def add_observed_property(points: np.ndarray, flags: np.ndarray) -> np.ndarray:
    names = points.dtype.names or ()
    tagged = np.empty(len(points), dtype=np.dtype(
        [(name, points.dtype.fields[name][0]) for name in names if name != "observed"] + [("observed", "u1")]
    ))
    for name in tagged.dtype.names or ():
        if name != "observed":
            tagged[name] = points[name]
    tagged["observed"] = flags.astype(np.uint8)
    return tagged


def filter_and_align(
    source: Path,
    filtered: Path,
    preview: Path,
    *,
    up_vector: list[float] | None,
    preview_limit: int = 100_000,
    observed_reference: Path | None = None,
    all_inferred: bool = False,
) -> dict[str, Any]:
    if all_inferred and observed_reference is not None:
        raise ValueError("all_inferred and observed_reference are mutually exclusive")
    points = read_ply(source)
    original_count = len(points)
    finite = np.ones(original_count, dtype=bool)
    for name in ("x", "y", "z"):
        finite &= np.isfinite(points[name])
    points = points[finite].copy()
    xyz = np.column_stack([points[name] for name in ("x", "y", "z")]).astype(np.float64)
    bounds = np.ptp(xyz, axis=0)
    diagonal = float(np.linalg.norm(bounds))
    if not math.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("point cloud bounds are degenerate")
    # A small, scale-relative radius only drops unequivocally isolated samples.
    radius = max(diagonal * 0.002, np.finfo(float).eps)
    keep, filter_method = conservative_outlier_mask(xyz, radius=radius, minimum_neighbors=3)
    radius_filter_applied = float(keep.mean()) >= 0.5
    if not radius_filter_applied:
        # A sparse-but-legitimate scan (railings/signs in particular) must not be
        # erased merely because its sampling density differs from the calibration set.
        keep = np.ones(len(xyz), dtype=bool)
    points = points[keep].copy()
    xyz = xyz[keep]
    provenance: dict[str, Any] = {}
    if observed_reference is not None:
        reference = read_ply(observed_reference)
        reference_xyz = np.column_stack([reference[name] for name in ("x", "y", "z")]).astype(np.float64)
        reference_xyz = reference_xyz[np.isfinite(reference_xyz).all(axis=1)]
        flags = observed_mask(xyz, reference_xyz, radius=radius) if len(reference_xyz) else np.zeros(len(xyz), bool)
        points = add_observed_property(points, flags)
        provenance = {"observedPoints": int(flags.sum()), "inferredPoints": int((~flags).sum())}
    elif all_inferred:
        points = add_observed_property(points, np.zeros(len(points), dtype=bool))
        provenance = {"observedPoints": 0, "inferredPoints": len(points)}
    rotation = np.eye(3)
    alignment_source = "manual_required"
    floor_leveling: dict[str, Any] = {"applied": False}
    if up_vector is not None:
        up = np.asarray(up_vector, dtype=float)
        if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-6:
            raise ValueError("up vector must contain three finite non-zero values")
        rotation = rotation_from_to(up, np.array([0.0, 1.0, 0.0]))
        leveling, floor_leveling = floor_leveling_rotation(xyz @ rotation.T)
        rotation = leveling @ rotation
        xyz = xyz @ rotation.T
        alignment_source = "camera_or_gravity"
        for index, name in enumerate(("x", "y", "z")):
            points[name] = xyz[:, index]
        if all(name in (points.dtype.names or ()) for name in ("nx", "ny", "nz")):
            normals = np.column_stack([points[name] for name in ("nx", "ny", "nz")]) @ rotation.T
            for index, name in enumerate(("nx", "ny", "nz")):
                points[name] = normals[:, index]
    write_ply(filtered, points)
    if len(points) > preview_limit:
        # Stable spatially distributed sample; never synthesize or move a preview point.
        order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
        selection = order[np.linspace(0, len(order) - 1, preview_limit, dtype=np.int64)]
        preview_points = points[selection]
    else:
        preview_points = points
    write_ply(preview, preview_points)
    transformed = np.column_stack([points[name] for name in ("x", "y", "z")])
    return {
        "inputPoints": original_count,
        "finitePoints": int(finite.sum()),
        "filteredPoints": len(points),
        "removedPoints": original_count - len(points),
        "previewPoints": len(preview_points),
        "radius": radius,
        "minimumConservativeNeighbors": 3,
        "radiusFilterApplied": radius_filter_applied,
        "filterMethod": filter_method,
        "alignmentSource": alignment_source,
        "rotation": rotation.tolist(),
        "floorLeveling": floor_leveling,
        "bounds": {"min": transformed.min(axis=0).tolist(), "max": transformed.max(axis=0).tolist()},
        "preservedProperties": list(points.dtype.names or ()),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("filtered", type=Path)
    parser.add_argument("preview", type=Path)
    parser.add_argument("--up", nargs=3, type=float)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--observed-reference", type=Path,
                        help="fusion of observed-only depth maps; tags each point observed=1/0")
    parser.add_argument("--all-inferred", action="store_true",
                        help="tag every point observed=0 (fast mode: no multi-view stereo depth)")
    args = parser.parse_args()
    metrics = filter_and_align(args.source, args.filtered, args.preview, up_vector=args.up,
                               observed_reference=args.observed_reference, all_inferred=args.all_inferred)
    payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.metrics:
        args.metrics.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
