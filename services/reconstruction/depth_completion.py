#!/usr/bin/env python3
"""Fill holes in COLMAP geometric depth maps with monocular depth aligned to them.

PatchMatch cannot match weakly textured surfaces (smooth plastic, dark fabric), so
those pixels stay empty. For each frame, a monocular depth network (Depth Anything
V2) predicts relative inverse depth everywhere; it is fitted to the frame's observed
MVS depth (robust affine in inverse depth plus a smooth local correction) and used
only where MVS left a hole. Observed pixels are never changed.

Safeguards, in order: frames with too little observed depth to anchor the fit, or
whose fit misses held-out observed pixels by more than MAX_HELD_OUT_ERROR, are left
untouched; filled depths outside the frame's observed depth range are discarded; and
COLMAP stereo fusion afterwards keeps only points that agree across several views.

Depth and normal maps are rewritten in place, in COLMAP's binary format, so that
``colmap stereo_fusion --input_type geometric`` consumes the completed maps.

``--sparse-anchors`` is the fast mode used when PatchMatch is skipped entirely:
every frame's depth comes from the monocular network, aligned to the SfM points
triangulated in that frame, and written as new geometric maps. It is roughly 50x
faster than PatchMatch but every resulting point is inferred rather than measured,
and edges and dark glossy surfaces come out softer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


MODEL_REPOSITORY = "depth-anything/Depth-Anything-V2-Large-hf"
# Pinned snapshot; the reconstruction image downloads it at build time so workers
# run offline. The Large weights are licensed CC-BY-NC-4.0 (non-commercial).
MODEL_REVISION = "7581137eff8d4e94f6e796d3baea0e9fa79b22d2"
MIN_ANCHOR_RATIO = 0.05
MIN_ANCHOR_PIXELS = 2_000
HELD_OUT_FRACTION = 0.1
MAX_HELD_OUT_ERROR = 0.05
DEPTH_RANGE_MARGIN = 1.5
CORRECTION_BLOCK_PIXELS = 24
# Fast mode anchors each frame on its triangulated SfM keypoints (~950 per frame
# on a 60-frame phone capture) instead of PatchMatch depth.
MIN_SPARSE_ANCHORS = 100
# Shorter side of the network input. 518 is the model's training size; at 392 the
# Large model runs in half the time (60 vs 120 ms per frame on an RTX 4060) and its
# held-out error against the reference room's SfM points moved from 1.88 to 1.91%.
MODEL_INPUT_SIZE = 518
FAST_MODEL_INPUT_SIZE = 392

Predictor = Callable[[Image.Image, int, int], np.ndarray]


def read_colmap_array(path: Path) -> np.ndarray:
    """Read a COLMAP dense map (``width&height&channels&`` + column-major float32)."""
    with path.open("rb") as source:
        header = b""
        while header.count(b"&") < 3:
            byte = source.read(1)
            if not byte:
                raise ValueError(f"truncated COLMAP array header: {path}")
            header += byte
        width, height, channels = map(int, header.split(b"&")[:3])
        data = np.fromfile(source, np.float32)
    if data.size != width * height * channels:
        raise ValueError(f"{path} declares {width}x{height}x{channels} but holds {data.size} values")
    array = data.reshape((width, height, channels), order="F").transpose(1, 0, 2)
    return array[:, :, 0] if channels == 1 else array


def write_colmap_array(path: Path, array: np.ndarray) -> None:
    array = np.asarray(array, np.float32)
    if array.ndim == 2:
        array = array[:, :, None]
    height, width, channels = array.shape
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as destination:
        destination.write(f"{width}&{height}&{channels}&".encode())
        destination.write(array.transpose(1, 0, 2).tobytes(order="F"))
    temporary.replace(path)


def read_camera_intrinsics(sparse: Path) -> dict[str, tuple[int, float, float, float, float]]:
    """Map image name -> (camera width, fx, fy, cx, cy) from undistorted binary model."""
    cameras: dict[int, tuple[int, float, float, float, float]] = {}
    with (sparse / "cameras.bin").open("rb") as source:
        (count,) = struct.unpack("<Q", source.read(8))
        for _ in range(count):
            camera_id, model_id, width, _height = struct.unpack("<iiQQ", source.read(24))
            if model_id != 1:  # PINHOLE, which image_undistorter always writes
                raise ValueError(f"expected undistorted PINHOLE cameras, found model id {model_id}")
            fx, fy, cx, cy = struct.unpack("<4d", source.read(32))
            cameras[camera_id] = (width, fx, fy, cx, cy)
    intrinsics = {}
    with (sparse / "images.bin").open("rb") as source:
        (count,) = struct.unpack("<Q", source.read(8))
        for _ in range(count):
            _image_id, *_pose, camera_id = struct.unpack("<i7di", source.read(64))
            name = b""
            while (char := source.read(1)) != b"\0":
                name += char
            (points,) = struct.unpack("<Q", source.read(8))
            source.seek(points * 24, 1)
            intrinsics[name.decode()] = cameras[camera_id]
    return intrinsics


def robust_affine(x: np.ndarray, y: np.ndarray, rounds: int = 4) -> tuple[float, float]:
    """Least-squares y = a*x + b, iteratively dropping residuals beyond 3 sigma (MAD)."""
    keep = np.ones(len(x), bool)
    a = b = 0.0
    for _ in range(rounds):
        design = np.stack([x[keep], np.ones(int(keep.sum()))], 1)
        (a, b), *_ = np.linalg.lstsq(design, y[keep], rcond=None)
        residual = np.abs(a * x + b - y)
        mad = float(np.median(residual[keep])) + 1e-12
        keep = residual < 3.0 * 1.4826 * mad
    return float(a), float(b)


def _blur_rows_and_columns(grid: np.ndarray, passes: int) -> np.ndarray:
    """Repeated zero-padded [1 4 6 4 1] blur along both axes (as np.convolve "same")."""
    from scipy.ndimage import convolve1d

    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    grid = np.asarray(grid, dtype=float)
    for _ in range(passes):
        for axis in (0, 1):
            grid = convolve1d(grid, kernel, axis=axis, mode="constant", cval=0.0)
    return grid


def smooth_correction(log_ratio: np.ndarray, valid: np.ndarray, block: int = CORRECTION_BLOCK_PIXELS) -> np.ndarray:
    """Low-frequency log-scale correction interpolated from valid pixels.

    Block sums of the residual and of the valid mask are blurred identically, so
    their quotient is a normalized average that also extends into empty regions.
    """
    height, width = log_ratio.shape
    rows, columns = -(-height // block), -(-width // block)
    pad = ((0, rows * block - height), (0, columns * block - width))
    sums = np.pad(np.where(valid, log_ratio, 0.0), pad).reshape(rows, block, columns, block).sum((1, 3))
    counts = np.pad(valid.astype(float), pad).reshape(rows, block, columns, block).sum((1, 3))
    sums, counts = _blur_rows_and_columns(sums, 12), _blur_rows_and_columns(counts, 12)
    field = np.where(counts > 1e-6, sums / np.maximum(counts, 1e-6), 0.0).astype(np.float32)
    return np.asarray(Image.fromarray(field, mode="F").resize((width, height), Image.BILINEAR))


def align(disparity: np.ndarray, observed: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    """Return monocular depth fitted to observed depth, and its held-out median relative error."""
    valid = observed > 0
    rows, columns = np.nonzero(valid)
    held_out = np.random.default_rng(seed).random(len(rows)) < HELD_OUT_FRACTION
    fit = np.zeros_like(valid)
    fit[rows[~held_out], columns[~held_out]] = True
    a, b = robust_affine(disparity[fit], 1.0 / observed[fit])
    inverse = a * disparity + b
    depth = np.where(inverse > 1e-9, 1.0 / np.maximum(inverse, 1e-9), 0.0)
    ratio_valid = fit & (depth > 0)
    log_ratio = np.log(np.where(ratio_valid, observed, 1.0) / np.where(ratio_valid, depth, 1.0))
    depth = np.where(depth > 0, depth * np.exp(smooth_correction(log_ratio, ratio_valid)), 0.0)
    test_rows, test_columns = rows[held_out], columns[held_out]
    truth = observed[test_rows, test_columns]
    error = float(np.median(np.abs(depth[test_rows, test_columns] - truth) / truth)) if len(truth) else float("inf")
    return depth.astype(np.float32), error


def normals_from_depth(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Camera-frame unit normals facing the camera, as COLMAP stores them."""
    height, width = depth.shape
    v, u = np.mgrid[0:height, 0:width].astype(np.float64)
    z = depth.astype(np.float64)
    points = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)
    du = np.zeros_like(points)
    dv = np.zeros_like(points)
    du[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dv[1:-1] = points[2:] - points[:-2]
    normals = np.cross(du, dv)
    normals *= np.where(np.sum(normals * points, -1, keepdims=True) > 0, -1.0, 1.0)
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    return np.where(length > 1e-12, normals / np.maximum(length, 1e-12), 0.0).astype(np.float32)


def complete_frame(
    name: str,
    image: Image.Image,
    observed: np.ndarray,
    observed_normals: np.ndarray,
    intrinsics: tuple[int, float, float, float, float],
    predict: Predictor,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = observed.shape
    valid = observed > 0
    report: dict[str, Any] = {"image": name, "observedCoverage": float(valid.mean())}
    if valid.sum() < max(MIN_ANCHOR_PIXELS, MIN_ANCHOR_RATIO * valid.size):
        return observed, observed_normals, {**report, "completed": False, "reason": "too little observed depth"}
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    aligned, error = align(predict(image, width, height), observed, seed)
    report["heldOutMedianRelativeError"] = error
    if not error <= MAX_HELD_OUT_ERROR:
        return observed, observed_normals, {**report, "completed": False, "reason": "alignment error too high"}
    low, high = np.percentile(observed[valid], [1, 99])
    plausible = (aligned >= low / DEPTH_RANGE_MARGIN) & (aligned <= high * DEPTH_RANGE_MARGIN)
    fill = ~valid & plausible
    depth = np.where(fill, aligned, observed).astype(np.float32)
    camera_width, fx, fy, cx, cy = intrinsics
    scale = width / camera_width
    normals = normals_from_depth(depth, fx * scale, fy * scale, cx * scale, cy * scale)
    normals = np.where(fill[:, :, None], normals, observed_normals)
    report.update(completed=True, completedCoverage=float((depth > 0).mean()), filledPixels=int(fill.sum()))
    return depth, normals, report


def read_sparse_model(sparse: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dict[int, np.ndarray]]:
    """Return per-image (R, t, keypoint xy, point3D id) and point3D id -> xyz from a binary model."""
    images = {}
    with (sparse / "images.bin").open("rb") as source:
        (count,) = struct.unpack("<Q", source.read(8))
        for _ in range(count):
            _image_id, qw, qx, qy, qz, tx, ty, tz, _camera_id = struct.unpack("<i7di", source.read(64))
            name = b""
            while (char := source.read(1)) != b"\0":
                name += char
            (points,) = struct.unpack("<Q", source.read(8))
            keypoints = np.frombuffer(source.read(24 * points), dtype=[("x", "<f8"), ("y", "<f8"), ("id", "<i8")])
            rotation = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ])
            xy = np.stack([keypoints["x"], keypoints["y"]], 1)
            images[name.decode()] = (rotation, np.array([tx, ty, tz]), xy, keypoints["id"].copy())
    points3d: dict[int, np.ndarray] = {}
    with (sparse / "points3D.bin").open("rb") as source:
        (count,) = struct.unpack("<Q", source.read(8))
        for _ in range(count):
            point_id, x, y, z = struct.unpack("<Q3d", source.read(32))
            source.seek(3 + 8, 1)  # rgb, reprojection error
            (track,) = struct.unpack("<Q", source.read(8))
            source.seek(8 * track, 1)
            points3d[point_id] = np.array([x, y, z])
    return images, points3d


def sparse_depth_map(
    pose: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    points3d: dict[int, np.ndarray],
    scale: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Depth of every triangulated keypoint of one image, rasterized at map resolution."""
    rotation, translation, xy, ids = pose
    known = np.array([int(point_id) in points3d for point_id in ids], dtype=bool) if len(ids) else np.zeros(0, bool)
    depth = np.zeros((height, width), np.float32)
    if not known.any():
        return depth
    world = np.stack([points3d[int(point_id)] for point_id in ids[known]])
    z = (world @ rotation.T + translation)[:, 2]
    u = np.clip((xy[known, 0] * scale).astype(int), 0, width - 1)
    v = np.clip((xy[known, 1] * scale).astype(int), 0, height - 1)
    in_front = z > 0
    depth[v[in_front], u[in_front]] = z[in_front]
    return depth


def predict_frame_from_sparse(
    name: str,
    image: Image.Image,
    anchors: np.ndarray,
    intrinsics: tuple[int, float, float, float, float],
    predict: Predictor,
    *,
    with_normals: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Fast mode: whole-frame monocular depth aligned to the SfM points seen in the frame."""
    height, width = anchors.shape
    disparity = predict(image, width, height) if (anchors > 0).sum() >= MIN_SPARSE_ANCHORS else None
    return align_frame_to_sparse(name, disparity, anchors, intrinsics, with_normals=with_normals)


def align_frame_to_sparse(
    name: str,
    disparity: np.ndarray | None,
    anchors: np.ndarray,
    intrinsics: tuple[int, float, float, float, float],
    *,
    with_normals: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """CPU half of predict_frame_from_sparse, given the network's disparity (None: not run)."""
    height, width = anchors.shape
    valid = anchors > 0
    empty_normals = np.zeros((height, width, 3), np.float32) if with_normals else None
    empty = (np.zeros((height, width), np.float32), empty_normals)
    report: dict[str, Any] = {"image": name, "sparseAnchors": int(valid.sum())}
    if valid.sum() < MIN_SPARSE_ANCHORS or disparity is None:
        return *empty, {**report, "completed": False, "reason": "too few SfM points"}
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    aligned, error = align(disparity, anchors, seed)
    report["heldOutMedianRelativeError"] = error
    if not error <= MAX_HELD_OUT_ERROR:
        return *empty, {**report, "completed": False, "reason": "alignment error too high"}
    low, high = np.percentile(anchors[valid], [1, 99])
    depth = np.where((aligned >= low / DEPTH_RANGE_MARGIN) & (aligned <= high * DEPTH_RANGE_MARGIN), aligned, 0.0)
    depth = depth.astype(np.float32)
    normals = None
    if with_normals:
        camera_width, fx, fy, cx, cy = intrinsics
        scale = width / camera_width
        normals = normals_from_depth(depth, fx * scale, fy * scale, cx * scale, cy * scale)
        normals[depth <= 0] = 0.0
    report.update(completed=True, completedCoverage=float((depth > 0).mean()))
    return depth, normals, report


CLOUD_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])
# Back-projecting every CLOUD_STRIDE-th pixel of every map gives about as many
# points as COLMAP stereo fusion did (1.8M vs 2.1M on the reference room).
CLOUD_STRIDE = 8
# As in scene_completion.py: pixels whose 3x3 neighbourhood spans more than this
# relative depth range sit on an occlusion edge and would fly between surfaces.
CLOUD_EDGE_RELATIVE_JUMP = 0.06


def backproject_frame(
    depth: np.ndarray,
    image: Image.Image,
    pose: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    intrinsics: tuple[int, float, float, float, float],
    stride: int = CLOUD_STRIDE,
) -> np.ndarray:
    """World-space points, normals and colors of every ``stride``-th valid, non-edge pixel."""
    from scipy import ndimage

    height, width = depth.shape
    valid = depth > 0
    high = ndimage.maximum_filter(np.where(valid, depth, -np.inf), size=3)
    low = ndimage.minimum_filter(np.where(valid, depth, np.inf), size=3)
    keep = valid & (high - low <= CLOUD_EDGE_RELATIVE_JUMP * depth)
    sampled = np.zeros_like(keep)
    sampled[stride // 2::stride, stride // 2::stride] = True
    # Sampled pixels keep a valid pixel on every side, so normals need no border handling.
    keep[[0, -1], :] = False
    keep[:, [0, -1]] = False
    v, u = np.nonzero(keep & sampled)
    camera_width, fx, fy, cx, cy = intrinsics
    scale = width / camera_width
    fx, fy, cx, cy = fx * scale, fy * scale, cx * scale, cy * scale

    def unproject(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        z = depth[rows, columns].astype(np.float64)
        return np.stack([(columns - cx) / fx * z, (rows - cy) / fy * z, z], 1)

    camera = unproject(v, u)
    # Central differences as in normals_from_depth, evaluated at the samples only.
    normals = np.cross(unproject(v, u + 1) - unproject(v, u - 1), unproject(v + 1, u) - unproject(v - 1, u))
    normals *= np.where(np.sum(normals * camera, 1, keepdims=True) > 0, -1.0, 1.0)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    rotation, translation = pose[0], pose[1]
    world = (camera - translation) @ rotation
    world_normals = normals @ rotation
    colors = np.asarray(image.resize((width, height), Image.BILINEAR))[v, u]
    points = np.empty(len(v), CLOUD_DTYPE)
    for index, name in enumerate(("x", "y", "z")):
        points[name] = world[:, index]
    for index, name in enumerate(("nx", "ny", "nz")):
        points[name] = world_normals[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        points[name] = colors[:, index]
    return points


def write_cloud(path: Path, points: np.ndarray) -> None:
    """Binary PLY with the vertex layout COLMAP stereo fusion writes."""
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(points)}"]
    header += [f"property float {name}" for name in ("x", "y", "z", "nx", "ny", "nz")]
    header += [f"property uchar {name}" for name in ("red", "green", "blue")]
    header.append("end_header")
    with path.open("wb") as destination:
        destination.write(("\n".join(header) + "\n").encode("ascii"))
        points.astype(CLOUD_DTYPE, copy=False).tofile(destination)


def predict_workspace_from_sparse(
    dense: Path, predict: Predictor, max_image_size: int, *, cloud: Path | None = None,
) -> dict[str, Any]:
    """Write a geometric depth map for every frame without running PatchMatch.

    Without ``cloud`` normal maps are written too, for COLMAP stereo fusion. With
    it, fusion is skipped: the frames' depth maps are back-projected into that
    PLY instead. Room scans need it only for bounds, wall and floor directions,
    while scene_completion.py rebuilds the surfaces from the depth maps.
    """
    stereo = dense / "stereo"
    folders = ("depth_maps",) if cloud else ("depth_maps", "normal_maps")
    for folder in folders:
        (stereo / folder).mkdir(parents=True, exist_ok=True)
    intrinsics = read_camera_intrinsics(dense / "sparse")
    images, points3d = read_sparse_model(dense / "sparse")

    def load(name: str) -> tuple[Image.Image, np.ndarray]:
        with Image.open(dense / "images" / name) as source:
            image = source.convert("RGB")
        full_width, full_height = image.size
        scale = min(1.0, max_image_size / max(full_width, full_height))
        width, height = round(full_width * scale), round(full_height * scale)
        return image, sparse_depth_map(images[name], points3d, scale, width, height)

    def finish(name: str, image: Image.Image, anchors: np.ndarray, disparity: np.ndarray | None) -> tuple[dict, Any]:
        depth, normals, report = align_frame_to_sparse(name, disparity, anchors, intrinsics[name],
                                                       with_normals=cloud is None)
        write_colmap_array(stereo / "depth_maps" / f"{name}.geometric.bin", depth)
        if cloud is None:
            write_colmap_array(stereo / "normal_maps" / f"{name}.geometric.bin", normals)
            return report, None
        points = backproject_frame(depth, image, images[name], intrinsics[name]) if report["completed"] else None
        return report, points

    # The network is the bottleneck; loading the next frame and aligning, writing
    # and back-projecting the previous one run on worker threads meanwhile.
    from concurrent.futures import ThreadPoolExecutor

    names = sorted(images)
    results = []
    with ThreadPoolExecutor(2) as loader, ThreadPoolExecutor(2) as finisher:
        loading = [loader.submit(load, name) for name in names[:2]]
        for index, name in enumerate(names):
            image, anchors = loading[index].result()
            if index + 2 < len(names):
                loading.append(loader.submit(load, names[index + 2]))
            height, width = anchors.shape
            disparity = predict(image, width, height) if (anchors > 0).sum() >= MIN_SPARSE_ANCHORS else None
            results.append(finisher.submit(finish, name, image, anchors, disparity))
            loading[index] = None
        finished = [result.result() for result in results]
    frames = [report for report, _ in finished]
    clouds = [points for _, points in finished if points is not None]
    if not frames:
        raise ValueError(f"sparse model in {dense / 'sparse'} has no images")
    if cloud:
        write_cloud(cloud, np.concatenate(clouds) if clouds else np.empty(0, CLOUD_DTYPE))
    completed = [frame for frame in frames if frame["completed"]]
    if len(completed) < max(3, len(frames) // 2):
        raise RuntimeError(f"monocular depth aligned for only {len(completed)} of {len(frames)} frames")
    errors = [frame["heldOutMedianRelativeError"] for frame in frames if "heldOutMedianRelativeError" in frame]
    return {
        "mode": "sparse-anchored",
        "frames": len(frames),
        "completedFrames": len(completed),
        "skippedFrames": [
            {"image": frame["image"], "reason": frame["reason"]} for frame in frames if not frame["completed"]
        ],
        "medianSparseAnchors": int(np.median([frame["sparseAnchors"] for frame in frames])),
        "meanCompletedCoverage": float(np.mean([frame.get("completedCoverage", 0.0) for frame in frames])),
        "medianHeldOutRelativeError": float(np.median(errors)) if errors else None,
        **({"cloudPoints": int(sum(len(points) for points in clouds))} if cloud else {}),
        "perFrame": frames,
    }


def complete_workspace(dense: Path, predict: Predictor) -> dict[str, Any]:
    stereo = dense / "stereo"
    intrinsics = read_camera_intrinsics(dense / "sparse")
    frames = []
    for depth_path in sorted((stereo / "depth_maps").glob("*.geometric.bin")):
        name = depth_path.name.removesuffix(".geometric.bin")
        normal_path = stereo / "normal_maps" / depth_path.name
        observed = read_colmap_array(depth_path)
        observed_normals = read_colmap_array(normal_path)
        with Image.open(dense / "images" / name) as source:
            image = source.convert("RGB")
        depth, normals, report = complete_frame(
            name, image, observed, observed_normals, intrinsics[name], predict
        )
        if report["completed"]:
            write_colmap_array(depth_path, depth)
            write_colmap_array(normal_path, normals)
        frames.append(report)
    if not frames:
        raise ValueError(f"no geometric depth maps found in {stereo / 'depth_maps'}")
    completed = [frame for frame in frames if frame["completed"]]
    errors = [frame["heldOutMedianRelativeError"] for frame in frames if "heldOutMedianRelativeError" in frame]
    return {
        "frames": len(frames),
        "completedFrames": len(completed),
        "skippedFrames": [
            {"image": frame["image"], "reason": frame["reason"]} for frame in frames if not frame["completed"]
        ],
        "meanObservedCoverage": float(np.mean([frame["observedCoverage"] for frame in frames])),
        "meanCompletedCoverage": float(np.mean(
            [frame.get("completedCoverage", frame["observedCoverage"]) for frame in frames]
        )),
        "medianHeldOutRelativeError": float(np.median(errors)) if errors else None,
        "perFrame": frames,
    }


def load_predictor(input_size: int = MODEL_INPUT_SIZE) -> tuple[Predictor, dict[str, Any]]:
    """Depth Anything V2 via transformers; fp16 on CUDA, fp32 on CPU (slow, diagnostics only).

    ``input_size`` is the network's shorter input side (a multiple of 14).
    """
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if input_size % 14:
        raise ValueError(f"model input size {input_size} is not a multiple of 14")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoImageProcessor.from_pretrained(MODEL_REPOSITORY, revision=MODEL_REVISION,
                                                   size={"height": input_size, "width": input_size})
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_REPOSITORY, revision=MODEL_REVISION, dtype=dtype)
    model = model.to(device).eval()

    def predict(image: Image.Image, width: int, height: int) -> np.ndarray:
        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt").to(device, dtype)
            disparity = model(**inputs).predicted_depth.float()[None]
            disparity = torch.nn.functional.interpolate(
                disparity, size=(height, width), mode="bicubic", align_corners=False
            )
        return disparity[0, 0].cpu().numpy()

    return predict, {
        "model": MODEL_REPOSITORY,
        "modelRevision": MODEL_REVISION,
        "modelInputSize": input_size,
        "backend": f"torch-{torch.__version__}",
        "device": device,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dense", type=Path, help="COLMAP dense workspace (images/, sparse/, stereo/)")
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--sparse-anchors", action="store_true",
                        help="fast mode: create maps for every frame from SfM points instead of completing PatchMatch maps")
    parser.add_argument("--max-image-size", type=int, default=960,
                        help="map resolution for --sparse-anchors; must match stereo fusion")
    parser.add_argument("--model-input-size", type=int,
                        help=f"shorter network input side, a multiple of 14 (default {MODEL_INPUT_SIZE}, "
                             f"{FAST_MODEL_INPUT_SIZE} with --sparse-anchors)")
    parser.add_argument("--cloud", type=Path,
                        help="with --sparse-anchors: write back-projected points here instead of normal maps")
    args = parser.parse_args()
    if args.cloud and not args.sparse_anchors:
        parser.error("--cloud requires --sparse-anchors")
    started = time.monotonic()
    input_size = args.model_input_size or (FAST_MODEL_INPUT_SIZE if args.sparse_anchors else MODEL_INPUT_SIZE)
    predict, model_metadata = load_predictor(input_size)
    if args.sparse_anchors:
        result = predict_workspace_from_sparse(args.dense, predict, args.max_image_size, cloud=args.cloud)
    else:
        result = complete_workspace(args.dense, predict)
    metrics = {**model_metadata, **result}
    metrics["durationSeconds"] = round(time.monotonic() - started, 1)
    payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.metrics:
        args.metrics.write_text(payload)
    summary = {key: value for key, value in metrics.items() if key != "perFrame"}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
