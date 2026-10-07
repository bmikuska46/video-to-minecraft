#!/usr/bin/env python3
"""Complete a room scan into a closed, gap-free, colored shell around observed objects.

Fusing per-frame depth with COLMAP keeps only points on which several views agree,
so weakly textured walls and ceilings come out as sparse speckle and the voxelized
room is full of holes. This stage rebuilds the surfaces volumetrically and closes
the room around its contents:

1. Every frame's depth map is fused into a truncated signed distance field (TSDF)
   on an axis-aligned grid in the canonical, floor-levelled and wall-aligned frame.
   Averaging signed distances over all views keeps surfaces that the views agree on
   and erases "ghost" surfaces seen by only a few frames, because the other views
   see through them (their space is carved free).
2. The room interior is the free space connected to the cameras. Its projection on
   the floor is the room footprint, its top per column the ceiling height (sloped
   attic ceilings included). Columns whose ceiling was never seen take a harmonic
   interpolation of the observed heights, so an unseen ceiling is not dropped onto
   the furniture.
3. The room shell is the one-voxel layer just outside that volume: floor, walls
   and ceiling. It is closed by construction and emitted whole, colored from the
   nearest fused surfaces of the same kind (floor, wall or ceiling); where no
   fused surface is near (the gaps), it is the only geometry there.
4. Surfaces outside the shell (balcony seen through a door, a neighbouring room
   through a window) are dropped, and nothing is ever added inside the room: the
   interior keeps only fused object surfaces.

Each output point carries ``synthesized`` (1 = room shell point) and
``observed`` (1 = backed by multi-view stereo, as in the input canonical cloud).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from .depth_completion import (read_camera_intrinsics, read_colmap_array, read_sparse_model,
                                   robust_affine, smooth_correction)
    from .point_cloud import read_ply, write_ply
except ImportError:  # Direct script execution.
    from depth_completion import (read_camera_intrinsics, read_colmap_array, read_sparse_model,
                                  robust_affine, smooth_correction)
    from point_cloud import read_ply, write_ply


# Voxels along the longest room axis. At 320 a 5 m room gets ~1.6 cm voxels,
# well below any practical Minecraft block size, and the grid fits in 8 GB VRAM.
GRID_RESOLUTION = 320
GRID_MARGIN_VOXELS = 4
# The grid spans the cloud's 0.2-99.8 percentile box widened by this share of its
# extent, so a sparsely sampled attic ridge is not cut off.
GRID_PADDING = 0.06
# Truncation band: at least this many voxels, growing with depth to cover the
# few-percent scale error of monocular depth.
TRUNCATION_VOXELS = 3.0
TRUNCATION_RELATIVE = 0.03
# Depth pixels whose 3x3 neighbourhood spans more than this relative range sit on
# an occlusion edge; their depth is interpolated across it and is not trusted.
EDGE_RELATIVE_JUMP = 0.06
# A fused surface needs this many views on both sides of its zero crossing.
MIN_SURFACE_WEIGHT = 3.0
# Free space needs this many views that saw past it, and at least this share of
# the views that saw it at all. Voxels behind a wall get "occluded" votes from
# every view of that wall, so a few frames whose depth overshoots the wall (or
# looks out through a skylight) cannot carve space outside the room.
MIN_FREE_VIEWS = 2
MIN_FREE_RATIO = 0.5
# Free votes stop a truncation band short of every surface (up to ~3% of the
# viewing distance). The room grows through observed space in front of surfaces
# (positive TSDF) by up to this many voxels, so its shell lies on the walls
# rather than a few voxels in front of them.
SURFACE_GROWTH_VOXELS = 10
# A shell cell counts as covered when a fused surface point is this close.
SHELL_COVER_RADIUS_VOXELS = 2.5
# Surfaces up to this far outside the shell are kept (wall thickness, noise).
OUTSIDE_TOLERANCE_VOXELS = 3.0
# Monocular depth places a wall a few percent nearer or farther in every frame,
# which smears it over many voxels. Each round renders the fused surface into
# every frame and refits that frame's depth to it (affine in inverse depth plus a
# smooth correction, as in depth_completion.py), pulling the views to consensus.
REFINEMENT_ROUNDS = 2
REFINEMENT_DOWNSAMPLE = 4
REFINEMENT_MIN_PIXELS = 400
REFINEMENT_OUTLIER_LOG_RATIO = 0.15
# Free space under a shelf or lamp near the ceiling ends at the object, which
# would hang a pillar of shell from the ceiling into the room. Ceilings have no
# narrow downward dips, so dips narrower than this (in voxels) are closed; planes,
# slopes and ridges are unchanged by a greyscale closing.
CEILING_CLOSING_VOXELS = 31
COLOR_NEIGHBORS = 12
WALL_ALIGNMENT_BINS = 360
PREVIEW_LIMIT = 100_000


class SceneCompletionError(RuntimeError):
    """The scan cannot be completed into a room."""


def wall_alignment_yaw(normals: np.ndarray) -> float:
    """Yaw (radians, about +Y) that turns the dominant wall directions onto X and Z.

    Wall normals are horizontal; their angles taken modulo 90 degrees (4x angle on
    the circle) peak at the room's orientation for rectilinear rooms.
    """
    horizontal = normals[np.abs(normals[:, 1]) < 0.3]
    if len(horizontal) < 100:
        return 0.0
    angles = np.arctan2(horizontal[:, 2], horizontal[:, 0]) * 4.0
    weights = np.hypot(horizontal[:, 0], horizontal[:, 2])
    histogram, edges = np.histogram(np.mod(angles, 2 * math.pi), bins=WALL_ALIGNMENT_BINS,
                                    range=(0, 2 * math.pi), weights=weights)
    kernel = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    smooth = np.convolve(np.concatenate([histogram[-6:], histogram, histogram[:6]]), kernel, "valid")
    peak = (edges[int(np.argmax(smooth))] + edges[1] / 2) / 4.0
    # Rotating by -peak maps the wall normal angle to 0 (mod 90 degrees).
    yaw = -peak
    return float((yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4)


def yaw_rotation(yaw: float) -> np.ndarray:
    """Rotation about +Y turning the XZ direction at angle ``-yaw`` onto +X."""
    cosine, sine = math.cos(yaw), math.sin(yaw)
    # Angle in XZ is atan2(z, x); rotate (x, z) by +yaw.
    return np.array([[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]])


def load_frames(dense: Path, alignment: np.ndarray) -> list[dict[str, Any]]:
    """Depth maps, colors and canonical-frame poses of every frame with depth."""
    intrinsics = read_camera_intrinsics(dense / "sparse")
    poses, _ = read_sparse_model(dense / "sparse")
    frames = []
    for name in sorted(poses):
        depth_path = dense / "stereo" / "depth_maps" / f"{name}.geometric.bin"
        if not depth_path.is_file():
            continue
        depth = read_colmap_array(depth_path).astype(np.float32)
        if not (depth > 0).any():
            continue
        height, width = depth.shape
        camera_width, fx, fy, cx, cy = intrinsics[name]
        scale = width / camera_width
        with Image.open(dense / "images" / name) as source:
            color = np.asarray(source.convert("RGB").resize((width, height), Image.BILINEAR))
        rotation, translation, _, _ = poses[name]
        # x_cam = R x_colmap + t and x_canonical = A x_colmap.
        rotation = rotation @ alignment.T
        frames.append({
            "name": name,
            "depth": depth,
            "color": color,
            "K": (fx * scale, fy * scale, cx * scale, cy * scale),
            "R": rotation,
            "t": translation,
            "center": -rotation.T @ translation,
        })
    if len(frames) < 3:
        raise SceneCompletionError(f"only {len(frames)} frames have depth maps")
    return frames


def edge_free_depth(depth: Any, torch: Any) -> Any:
    """Zero depth on occlusion edges, where maps interpolate between surfaces."""
    padded = depth[None, None]
    valid = padded > 0
    big = torch.where(valid, padded, torch.full_like(padded, -1.0))
    high = torch.nn.functional.max_pool2d(big, 3, 1, 1)
    small = torch.where(valid, padded, torch.full_like(padded, float("inf")))
    low = -torch.nn.functional.max_pool2d(-small, 3, 1, 1)
    edge = (high - low) > EDGE_RELATIVE_JUMP * padded
    return torch.where(edge | ~valid, torch.zeros_like(padded), padded)[0, 0]


TSDF_CHUNK = 1 << 23


def voxel_centers(origin: np.ndarray, voxel: float, shape: tuple[int, int, int], device: str) -> Any:
    """Voxel centre coordinates as an (nx, ny, nz, 3) tensor.

    Computed once and shared by every frame of every fusion pass; deriving them
    from the flat index per frame and chunk was a large share of fusion time.
    """
    import torch

    axes = [torch.arange(size, dtype=torch.float32, device=device) for size in shape]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)
    return torch.tensor(origin, dtype=torch.float32, device=device) + (grid + 0.5) * voxel


def frustum_box(frame: dict[str, Any], origin: np.ndarray, voxel: float, shape: tuple[int, int, int],
                far: float) -> tuple[np.ndarray, np.ndarray]:
    """Voxel index box [low, high) around the frame's view pyramid cut at depth ``far``.

    The pyramid is the convex hull of the camera centre and its four far corners,
    so the box of those five points holds every voxel the frame can see that near.
    """
    height, width = frame["depth"].shape
    fx, fy, cx, cy = frame["K"]
    corners = np.array([[(u - cx) / fx, (v - cy) / fy, 1.0] for u in (0, width) for v in (0, height)]) * far
    camera = np.vstack([np.zeros(3), corners])
    world = (camera - frame["t"]) @ frame["R"]
    low = np.floor((world.min(axis=0) - origin) / voxel).astype(np.int64) - 1
    high = np.ceil((world.max(axis=0) - origin) / voxel).astype(np.int64) + 2
    return np.clip(low, 0, shape), np.clip(high, 0, shape)


def fuse_tsdf(frames: list[dict[str, Any]], origin: np.ndarray, voxel: float,
              shape: tuple[int, int, int], device: str, *, centers: Any = None,
              full: bool = True) -> dict[str, np.ndarray]:
    """Projective TSDF fusion of all frames; returns tsdf, weight, color, free counts.

    Each frame only visits the voxels in the box around its view pyramid. The
    pyramid reaches the grid's far corner in full fusion, where every voxel behind
    a surface counts as occluded; ``full=False`` (the refinement passes, which only
    need the surface position) stops it a truncation band behind the frame's
    deepest pixel and skips colors and free/occluded counts. ``colorWeight`` is
    always kept because surface extraction requires a near-surface observation.
    """
    import torch

    count = int(np.prod(shape))
    centers = centers if centers is not None else voxel_centers(origin, voxel, shape, device)
    tsdf = torch.ones(count, dtype=torch.float32, device=device)
    weight = torch.zeros(count, dtype=torch.float32, device=device)
    color_weight = torch.zeros(count, dtype=torch.float32, device=device)
    if full:
        color = torch.zeros((count, 3), dtype=torch.float32, device=device)
        free = torch.zeros(count, dtype=torch.int16, device=device)
        occluded = torch.zeros(count, dtype=torch.int16, device=device)
    grid_corners = origin + np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)]) * np.array(shape) * voxel
    ny, nz = shape[1], shape[2]
    for frame in frames:
        if full:
            far = float(np.linalg.norm(grid_corners - frame["center"], axis=1).max())
        else:
            deepest = float(frame["depth"].max())
            far = deepest + max(TRUNCATION_RELATIVE * deepest, TRUNCATION_VOXELS * voxel) + voxel
        low, high = frustum_box(frame, origin, voxel, shape, far)
        if np.any(high <= low):
            continue
        depth = edge_free_depth(torch.from_numpy(frame["depth"]).to(device), torch)
        if full:
            image = torch.tensor(frame["color"], dtype=torch.float32, device=device)
        height, width = depth.shape
        fx, fy, cx, cy = frame["K"]
        rotation = torch.tensor(frame["R"], dtype=torch.float32, device=device)
        translation = torch.tensor(frame["t"], dtype=torch.float32, device=device)
        box_y, box_z = int(high[1] - low[1]), int(high[2] - low[2])
        slab = max(1, TSDF_CHUNK // (box_y * box_z))
        for x_start in range(int(low[0]), int(high[0]), slab):
            x_end = min(x_start + slab, int(high[0]))
            points = centers[x_start:x_end, low[1]:high[1], low[2]:high[2]].reshape(-1, 3)
            camera = points @ rotation.T + translation
            z = camera[:, 2]
            front = z > 1e-6
            safe_z = torch.where(front, z, torch.ones_like(z))
            u = torch.round(fx * camera[:, 0] / safe_z + cx).long()
            v = torch.round(fy * camera[:, 1] / safe_z + cy).long()
            inside = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            selected = torch.nonzero(inside).squeeze(1)
            if not len(selected):
                continue
            us, vs, zs = u[selected], v[selected], z[selected]
            observed = depth[vs, us]
            has_depth = observed > 0
            selected, us, vs, zs, observed = (value[has_depth] for value in (selected, us, vs, zs, observed))
            # Box-local index -> flat index of the whole grid.
            local_x = selected // (box_y * box_z)
            local_y = (selected // box_z) % box_y
            local_z = selected % box_z
            selected = ((local_x + x_start) * ny + local_y + int(low[1])) * nz + local_z + int(low[2])
            sdf = observed - zs
            truncation = torch.clamp(TRUNCATION_RELATIVE * zs, min=TRUNCATION_VOXELS * voxel)
            update = sdf > -truncation
            target = selected[update]
            value = torch.clamp(sdf[update] / truncation[update], max=1.0)
            old_weight = weight[target]
            tsdf[target] = (tsdf[target] * old_weight + value) / (old_weight + 1.0)
            weight[target] = old_weight + 1.0
            near = update & (sdf < truncation)
            near_target = selected[near]
            color_weight[near_target] += 1.0
            if full:
                color[near_target] += image[vs[near], us[near]]
                free[selected[sdf > truncation]] += 1
                occluded[selected[sdf < -truncation]] += 1
    volume = {
        "tsdf": tsdf.reshape(shape).cpu().numpy(),
        "weight": weight.reshape(shape).cpu().numpy(),
        "colorWeight": color_weight.reshape(shape).cpu().numpy(),
    }
    if full:
        color = color / torch.clamp(color_weight, min=1.0)[:, None]
        volume.update(
            color=color.reshape(*shape, 3).cpu().numpy(),
            free=free.reshape(shape).cpu().numpy(),
            occluded=occluded.reshape(shape).cpu().numpy(),
        )
    return volume


def render_depth(points: Any, frame: dict[str, Any], factor: int, torch: Any) -> Any:
    """Z-buffer of surface points in a frame at 1/factor resolution (0 = no point)."""
    device = points.device
    height, width = frame["depth"].shape
    low_height, low_width = -(-height // factor), -(-width // factor)
    fx, fy, cx, cy = frame["K"]
    rotation = torch.tensor(frame["R"], dtype=torch.float32, device=device)
    translation = torch.tensor(frame["t"], dtype=torch.float32, device=device)
    camera = points @ rotation.T + translation
    z = camera[:, 2]
    front = z > 1e-6
    camera, z = camera[front], z[front]
    u = torch.floor((fx * camera[:, 0] / z + cx) / factor).long()
    v = torch.floor((fy * camera[:, 1] / z + cy) / factor).long()
    inside = (u >= 0) & (u < low_width) & (v >= 0) & (v < low_height)
    buffer = torch.full((low_height * low_width,), float("inf"), device=device)
    buffer.scatter_reduce_(0, v[inside] * low_width + u[inside], z[inside], reduce="amin")
    buffer = torch.where(torch.isfinite(buffer), buffer, torch.zeros_like(buffer))
    return buffer.reshape(low_height, low_width)


def refit_depth(depth: np.ndarray, rendered: np.ndarray, factor: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Refit one frame's depth to the fused surface rendered into it.

    Returns the original map unchanged unless the refit agrees better with the
    rendered consensus than the original did.
    """
    height, width = depth.shape
    low = depth[::factor, ::factor][: rendered.shape[0], : rendered.shape[1]]
    rendered = rendered[: low.shape[0], : low.shape[1]]
    valid = (low > 0) & (rendered > 0)
    report: dict[str, Any] = {"pixels": int(valid.sum())}
    if valid.sum() < REFINEMENT_MIN_PIXELS:
        return depth, {**report, "refit": False}
    before = float(np.median(np.abs(low[valid] - rendered[valid]) / rendered[valid]))
    a, b = robust_affine(1.0 / low[valid], 1.0 / rendered[valid])
    inverse = np.where(depth > 0, a / np.maximum(depth, 1e-9) + b, 0.0)
    refit = np.where(inverse > 1e-9, 1.0 / np.maximum(inverse, 1e-9), 0.0)
    low_refit = refit[::factor, ::factor][: low.shape[0], : low.shape[1]]
    usable = valid & (low_refit > 0)
    log_ratio = np.log(np.where(usable, rendered, 1.0) / np.where(usable, low_refit, 1.0))
    # Leaks through gaps in the z-buffer show background depth; keep them out of
    # the smooth correction.
    usable &= np.abs(log_ratio) < REFINEMENT_OUTLIER_LOG_RATIO
    correction = smooth_correction(np.where(usable, log_ratio, 0.0), usable, block=8)
    correction = np.asarray(Image.fromarray(correction.astype(np.float32), mode="F").resize(
        (width, height), Image.BILINEAR))
    refit = np.where(refit > 0, refit * np.exp(correction), 0.0).astype(np.float32)
    low_refit = refit[::factor, ::factor][: low.shape[0], : low.shape[1]]
    check = valid & (low_refit > 0)
    after = float(np.median(np.abs(low_refit[check] - rendered[check]) / rendered[check])) if check.any() else math.inf
    report.update(before=before, after=after)
    if not after < before:
        return depth, {**report, "refit": False}
    return refit, {**report, "refit": True}


def refine_frames(frames: list[dict[str, Any]], surface: dict[str, np.ndarray], device: str) -> dict[str, Any]:
    """Pull every frame's depth towards the fused consensus surface, in place."""
    import torch

    from concurrent.futures import ThreadPoolExecutor

    points = torch.from_numpy(surface["xyz"].astype(np.float32)).to(device)
    rendered = [render_depth(points, frame, REFINEMENT_DOWNSAMPLE, torch).cpu().numpy() for frame in frames]
    # The refits are independent NumPy work per frame, which releases the GIL.
    with ThreadPoolExecutor(8) as pool:
        refits = list(pool.map(lambda pair: refit_depth(pair[0]["depth"], pair[1], REFINEMENT_DOWNSAMPLE),
                               zip(frames, rendered)))
    befores, afters, refitted = [], [], 0
    for frame, (depth, report) in zip(frames, refits):
        if "before" in report:
            befores.append(report["before"])
            afters.append(min(report["after"], report["before"]))
        if report["refit"]:
            frame["depth"] = depth
            refitted += 1
    return {
        "refittedFrames": refitted,
        "medianDisagreementBefore": float(np.median(befores)) if befores else None,
        "medianDisagreementAfter": float(np.median(afters)) if afters else None,
    }


def extract_surface(volume: dict[str, np.ndarray], origin: np.ndarray, voxel: float) -> dict[str, np.ndarray]:
    """Zero crossings of the TSDF between well-observed neighbouring voxels."""
    tsdf, weight, color_weight = (volume[key] for key in ("tsdf", "weight", "colorWeight"))
    color = volume.get("color")
    gradient = np.stack(np.gradient(tsdf), -1)
    points, colors, normals = [], [], []
    for axis in range(3):
        head = [slice(None)] * 3
        tail = [slice(None)] * 3
        head[axis] = slice(None, -1)
        tail[axis] = slice(1, None)
        a, b = tsdf[tuple(head)], tsdf[tuple(tail)]
        crossing = ((weight[tuple(head)] >= MIN_SURFACE_WEIGHT) & (weight[tuple(tail)] >= MIN_SURFACE_WEIGHT)
                    & ((a > 0) != (b > 0)) & (np.abs(a - b) <= 1.0)
                    & ((color_weight[tuple(head)] > 0) | (color_weight[tuple(tail)] > 0)))
        cells = np.argwhere(crossing)
        if not len(cells):
            continue
        neighbour = cells.copy()
        neighbour[:, axis] += 1
        va = a[crossing]
        vb = b[crossing]
        fraction = (va / (va - vb))[:, None]
        position = origin + (cells + 0.5 + fraction * np.eye(3)[axis]) * voxel
        if color is None:
            rgb = np.zeros((len(cells), 3))
        else:
            wa = color_weight[tuple(cells.T)][:, None]
            wb = color_weight[tuple(neighbour.T)][:, None]
            ca, cb = color[tuple(cells.T)], color[tuple(neighbour.T)]
            blend_a = (1 - fraction) * (wa > 0)
            blend_b = fraction * (wb > 0)
            rgb = (ca * blend_a + cb * blend_b) / np.maximum(blend_a + blend_b, 1e-9)
        normal = gradient[tuple(cells.T)] + gradient[tuple(neighbour.T)]
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        points.append(position)
        colors.append(rgb)
        normals.append(normal)
    if not points:
        raise SceneCompletionError("TSDF fusion produced no surface")
    return {"xyz": np.concatenate(points), "rgb": np.concatenate(colors), "normal": np.concatenate(normals)}


def harmonic_fill(values: np.ndarray, known: np.ndarray, domain: np.ndarray, iterations: int = 4000) -> np.ndarray:
    """Fill ``domain`` cells not in ``known`` by Laplace (harmonic) interpolation.

    Harmonic interpolation reproduces planes exactly where the unseen region is
    enclosed by seen cells, so a sloped ceiling seen around a gap is continued at
    its own slope. Toward a footprint edge with no seen cells it levels off.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    unknown = domain & ~known
    if not unknown.any():
        return values.copy()
    if not known.any():
        raise SceneCompletionError("no observed values to interpolate from")
    index = -np.ones(values.shape, dtype=np.int64)
    index[unknown] = np.arange(int(unknown.sum()))
    rows, columns, data = [], [], []
    rhs = np.zeros(int(unknown.sum()))
    cells = np.argwhere(unknown)
    for offset in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neighbour = cells + offset
        inside = ((neighbour >= 0) & (neighbour < values.shape)).all(axis=1)
        own = index[tuple(cells[inside].T)]
        other = neighbour[inside]
        in_domain = domain[tuple(other.T)]
        own, other = own[in_domain], other[in_domain]
        rows.append(own)
        columns.append(own)
        data.append(np.ones(len(own)))
        other_index = index[tuple(other.T)]
        free_neighbour = other_index >= 0
        rows.append(own[free_neighbour])
        columns.append(other_index[free_neighbour])
        data.append(-np.ones(int(free_neighbour.sum())))
        np.add.at(rhs, own[~free_neighbour], values[tuple(other[~free_neighbour].T)])
    matrix = sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
                               shape=(len(rhs), len(rhs)))
    # Components with no known boundary are singular; a tiny pull to the mean of
    # the known values keeps the system solvable without biasing anchored cells.
    mean = float(values[known].mean())
    matrix = matrix + sparse.identity(len(rhs)) * 1e-6
    rhs = rhs + 1e-6 * mean
    result = values.copy()
    result[unknown] = spsolve(matrix.tocsc(), rhs)
    return result


def room_volume(volume: dict[str, np.ndarray], camera_cells: np.ndarray, floor_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Boolean grid of the room's inside: footprint x [floor, ceiling(x, z)]."""
    from scipy import ndimage

    tsdf, weight = volume["tsdf"], volume["weight"]
    free_views = volume["free"].astype(np.int32)
    occluded_views = volume["occluded"].astype(np.int32)
    known_free = (free_views >= MIN_FREE_VIEWS) & (free_views >= MIN_FREE_RATIO * (free_views + occluded_views))
    labels, _ = ndimage.label(known_free)
    seeds = labels[tuple(camera_cells.T)]
    seeds = seeds[seeds > 0]
    if not len(seeds):
        raise SceneCompletionError("no camera lies in carved free space")
    # The component holding most cameras is the room; others are free space seen
    # through openings that do not connect at this resolution.
    room_label = int(np.bincount(seeds).argmax())
    interior = labels == room_label
    in_front_of_surface = (weight > 0) & (tsdf > 0)
    interior = ndimage.binary_dilation(interior, iterations=SURFACE_GROWTH_VOXELS, mask=in_front_of_surface)
    interior[:, :floor_index + 1, :] = False

    column_free = interior.sum(axis=1)
    footprint = column_free > 0
    # Fill pockets under tall furniture and remove one-cell slivers along rays.
    footprint = ndimage.binary_opening(footprint, iterations=2)
    footprint = ndimage.binary_fill_holes(ndimage.binary_closing(footprint, iterations=3))
    labels2d, count2d = ndimage.label(footprint)
    if count2d > 1:
        sizes = ndimage.sum(footprint, labels2d, range(1, count2d + 1))
        footprint = labels2d == (int(np.argmax(sizes)) + 1)

    ny = interior.shape[1]
    heights = np.arange(ny)[None, :, None]
    top_free = np.where(interior.any(axis=1), np.max(np.where(interior, heights, -1), axis=1), floor_index)
    above = np.clip(top_free + 1, 0, ny - 1)
    cap_weight = np.take_along_axis(weight, above[:, None, :], axis=1)[:, 0, :]
    cap_tsdf = np.take_along_axis(tsdf, above[:, None, :], axis=1)[:, 0, :]
    # A column's ceiling is observed when the free space ends at a fused surface
    # rather than fading into unobserved space.
    ceiling_seen = footprint & interior.any(axis=1) & (cap_weight >= MIN_SURFACE_WEIGHT) & (cap_tsdf <= 0.5)
    ceiling = harmonic_fill(top_free.astype(float), ceiling_seen, footprint)
    ceiling = np.maximum(ceiling, top_free)
    # Extend the height map past the footprint with nearest values so the
    # closing does not see the edge as a dip.
    _, nearest = ndimage.distance_transform_edt(~footprint, return_indices=True)
    extended = ceiling[tuple(nearest)]
    ceiling = np.where(footprint, ndimage.grey_closing(extended, size=CEILING_CLOSING_VOXELS), ceiling)
    ceiling_index = np.clip(np.rint(ceiling).astype(np.int64), floor_index + 1, ny - 2)

    inside = footprint[:, None, :] & (heights > floor_index) & (heights <= ceiling_index[:, None, :])
    # Keep a layer free at the grid border so the shell around the room is complete.
    inside[[0, -1], :, :] = False
    inside[:, [0, -1], :] = False
    inside[:, :, [0, -1]] = False
    details = {
        "footprintCells": int(footprint.sum()),
        "ceilingObservedFraction": float(ceiling_seen.sum() / max(1, footprint.sum())),
        "interiorFreeCells": int(interior.sum()),
    }
    return inside, details


def floor_level(surface: dict[str, np.ndarray], origin: np.ndarray, voxel: float, shape: tuple[int, int, int]) -> int:
    """Grid layer of the floor: the lowest strongly supported up-facing surface."""
    up = surface["normal"][:, 1] > 0.85
    heights = surface["xyz"][up, 1]
    if len(heights) < 100:
        raise SceneCompletionError("no floor surface found")
    layers = np.floor((heights - origin[1]) / voxel).astype(np.int64)
    counts = np.bincount(np.clip(layers, 0, shape[1] - 1), minlength=shape[1]).astype(float)
    smooth = np.convolve(counts, np.ones(3), "same")
    # The floor is the lowest layer carrying a substantial share of the
    # up-facing area; table tops and bed covers are smaller and higher.
    strong = np.flatnonzero(smooth >= 0.3 * smooth.max())
    return int(strong.min())


def complete_scene(canonical: Path, dense: Path, filter_metrics: Path, output: Path, preview: Path,
                   *, resolution: int = GRID_RESOLUTION, device: str | None = None) -> dict[str, Any]:
    import torch
    from scipy import ndimage
    from scipy.spatial import cKDTree

    started = time.monotonic()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    measured = read_ply(canonical)
    names = measured.dtype.names or ()
    measured_xyz = np.column_stack([measured[name] for name in "xyz"]).astype(np.float64)
    alignment = np.asarray(json.loads(filter_metrics.read_text())["rotation"], dtype=float)

    yaw = 0.0
    if {"nx", "ny", "nz"} <= set(names):
        yaw = wall_alignment_yaw(np.column_stack([measured[name] for name in ("nx", "ny", "nz")]))
    turn = yaw_rotation(yaw)
    measured_xyz = measured_xyz @ turn.T
    frames = load_frames(dense, turn @ alignment)
    centers = np.stack([frame["center"] for frame in frames])

    low = np.percentile(measured_xyz, 0.2, axis=0)
    high = np.percentile(measured_xyz, 99.8, axis=0)
    low = np.minimum(low, centers.min(axis=0))
    high = np.maximum(high, centers.max(axis=0))
    padding = GRID_PADDING * (high - low)
    low, high = low - padding, high + padding
    voxel = float((high - low).max() / resolution)
    origin = low - GRID_MARGIN_VOXELS * voxel
    shape = tuple(int(value) for value in np.ceil((high - low) / voxel).astype(int) + 2 * GRID_MARGIN_VOXELS)

    grid_centers = voxel_centers(origin, voxel, shape, device)
    refinement = []
    for _ in range(REFINEMENT_ROUNDS):
        volume = fuse_tsdf(frames, origin, voxel, shape, device, centers=grid_centers, full=False)
        refinement.append(refine_frames(frames, extract_surface(volume, origin, voxel), device))
    volume = fuse_tsdf(frames, origin, voxel, shape, device, centers=grid_centers)
    del grid_centers
    fused_at = time.monotonic()
    surface = extract_surface(volume, origin, voxel)
    floor_index = floor_level(surface, origin, voxel, shape)
    camera_cells = np.clip(np.floor((centers - origin) / voxel).astype(np.int64), 0, np.array(shape) - 1)
    inside, room = room_volume(volume, camera_cells, floor_index)

    # Shell: cells just outside the room volume (6-connected), i.e. floor, walls, ceiling.
    shell = ndimage.binary_dilation(inside, structure=ndimage.generate_binary_structure(3, 1)) & ~inside
    outside_far = ~ndimage.binary_dilation(inside, iterations=int(math.ceil(OUTSIDE_TOLERANCE_VOXELS)))

    surface_cells = np.clip(np.floor((surface["xyz"] - origin) / voxel).astype(np.int64), 0, np.array(shape) - 1)
    keep = ~outside_far[tuple(surface_cells.T)]
    removed_outside = int((~keep).sum())
    xyz, rgb, normal = surface["xyz"][keep], surface["rgb"][keep], surface["normal"][keep]

    # The whole shell is emitted, not only its gaps: a fused wall a voxel or two
    # off the shell does not seal the seam to it at block resolution (a 100-block
    # export of the reference room leaked through single-block seams), while the
    # shell alone is closed by construction. Where a fused surface covers the
    # shell, the shell only thickens that wall outwards, away from the room.
    shell_cells = np.argwhere(shell)
    shell_centers = origin + (shell_cells + 0.5) * voxel
    tree = cKDTree(xyz)
    distance, _ = tree.query(shell_centers, k=1, distance_upper_bound=SHELL_COVER_RADIUS_VOXELS * voxel, workers=-1)
    gaps = ~np.isfinite(distance)
    gap_cells = shell_cells
    gap_xyz = shell_centers

    # Each gap faces the room through one of its six neighbours; that direction
    # is both its normal and its kind (floor, ceiling or wall).
    gap_normal = np.zeros((len(gap_cells), 3))
    for offset in np.vstack([np.eye(3, dtype=int), -np.eye(3, dtype=int)]):
        neighbour = np.clip(gap_cells + offset, 0, np.array(shape) - 1)
        faces = inside[tuple(neighbour.T)] & ~gap_normal.any(axis=1)
        gap_normal[faces] = offset
    kind_names = ("floor", "ceiling", "wall")

    def kinds(normals: np.ndarray) -> np.ndarray:
        return np.where(normals[:, 1] > 0.7, 0, np.where(normals[:, 1] < -0.7, 1, 2))

    shell_surface = ~ndimage.binary_erosion(~outside_far, iterations=int(math.ceil(OUTSIDE_TOLERANCE_VOXELS)) + 2)
    near_shell = shell_surface[tuple(np.clip(np.floor((xyz - origin) / voxel).astype(np.int64), 0,
                                             np.array(shape) - 1).T)]
    gap_rgb = np.zeros((len(gap_xyz), 3))
    surface_kind = kinds(normal)
    gap_kind = kinds(gap_normal)
    synthesized_by_kind = {}
    for kind, kind_name in enumerate(kind_names):
        targets = gap_kind == kind
        synthesized_by_kind[kind_name] = int((targets & gaps).sum())
        if not targets.any():
            continue
        sources = near_shell & (surface_kind == kind)
        if sources.sum() < COLOR_NEIGHBORS:
            sources = near_shell if near_shell.sum() >= COLOR_NEIGHBORS else np.ones(len(xyz), bool)
        source_tree = cKDTree(xyz[sources])
        distances, indices = source_tree.query(gap_xyz[targets], k=COLOR_NEIGHBORS, workers=-1)
        weights = 1.0 / np.maximum(distances, voxel) ** 2
        gap_rgb[targets] = (rgb[sources][indices] * weights[:, :, None]).sum(1) / weights.sum(1, keepdims=True)

    observed = np.zeros(len(xyz), dtype=bool)
    if "observed" in names and measured["observed"].any():
        observed_tree = cKDTree(measured_xyz[measured["observed"] > 0])
        distance, _ = observed_tree.query(xyz, k=1, distance_upper_bound=2 * voxel, workers=-1)
        observed = np.isfinite(distance)

    total = len(xyz) + len(gap_xyz)
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("observed", "u1"), ("synthesized", "u1")])
    points = np.empty(total, dtype=dtype)
    all_xyz = np.vstack([xyz, gap_xyz])
    all_normal = np.vstack([normal, gap_normal])
    all_rgb = np.clip(np.rint(np.vstack([rgb, gap_rgb])), 0, 255)
    for index, name in enumerate("xyz"):
        points[name] = all_xyz[:, index]
    for index, name in enumerate(("nx", "ny", "nz")):
        points[name] = all_normal[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        points[name] = all_rgb[:, index]
    points["observed"] = np.concatenate([observed, np.zeros(len(gap_xyz), bool)])
    points["synthesized"] = np.concatenate([np.zeros(len(xyz), bool), np.ones(len(gap_xyz), bool)])
    write_ply(output, points)
    if total > PREVIEW_LIMIT:
        order = np.lexsort((all_xyz[:, 2], all_xyz[:, 1], all_xyz[:, 0]))
        write_ply(preview, points[order[np.linspace(0, total - 1, PREVIEW_LIMIT, dtype=np.int64)]])
    else:
        write_ply(preview, points)

    # Where the room sits in the output frame, for export verification.
    room_bounds_min = origin + np.argwhere(inside).min(axis=0) * voxel
    room_bounds_max = origin + (np.argwhere(inside).max(axis=0) + 1) * voxel
    return {
        "device": device,
        "frames": len(frames),
        "gridShape": list(shape),
        "voxelSize": voxel,
        "gridOrigin": origin.tolist(),
        "yawDegrees": math.degrees(yaw),
        "rotation": (turn @ alignment).tolist(),
        "floorY": float(origin[1] + (floor_index + 1) * voxel),
        "roomBounds": {"min": room_bounds_min.tolist(), "max": room_bounds_max.tolist()},
        "cameraCenters": centers.tolist(),
        **room,
        "measuredPoints": len(measured),
        "fusedSurfacePoints": len(surface["xyz"]),
        "removedOutsidePoints": removed_outside,
        "keptSurfacePoints": len(xyz),
        "shellCells": int(len(shell_cells)),
        "shellGapFraction": float(gaps.mean()) if len(gaps) else 0.0,
        "shellPoints": int(len(gap_xyz)),
        "synthesizedPoints": int(len(gap_xyz)),
        "gapPoints": int(gaps.sum()),
        "synthesizedByKind": synthesized_by_kind,
        "outputPoints": total,
        "previewPoints": min(total, PREVIEW_LIMIT),
        # From the float32 values written: an uncut export crops to these bounds,
        # and float64 bounds would drop shell points that rounded just outside.
        "bounds": {"min": [float(points[name].min()) for name in "xyz"],
                   "max": [float(points[name].max()) for name in "xyz"]},
        "refinement": refinement,
        "fusionSeconds": round(fused_at - started, 1),
        "durationSeconds": round(time.monotonic() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("canonical", type=Path, help="filtered, aligned cloud (canonical.ply)")
    parser.add_argument("dense", type=Path, help="COLMAP dense workspace with geometric depth maps")
    parser.add_argument("filter_metrics", type=Path, help="filter-metrics.json holding the alignment rotation")
    parser.add_argument("output", type=Path)
    parser.add_argument("preview", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--resolution", type=int, default=GRID_RESOLUTION)
    parser.add_argument("--device")
    args = parser.parse_args()
    metrics = complete_scene(args.canonical, args.dense, args.filter_metrics, args.output, args.preview,
                             resolution=args.resolution, device=args.device)
    payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.metrics:
        args.metrics.write_text(payload)
    summary = {key: value for key, value in metrics.items() if key != "cameraCenters"}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
