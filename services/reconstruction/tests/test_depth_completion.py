from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "depth_completion.py"
SPEC = importlib.util.spec_from_file_location("depth_completion", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["depth_completion"] = MODULE
SPEC.loader.exec_module(MODULE)

WIDTH, HEIGHT = 160, 120
INTRINSICS = (WIDTH, 100.0, 100.0, WIDTH / 2, HEIGHT / 2)
IMAGE = Image.new("RGB", (WIDTH, HEIGHT))


def tilted_plane() -> np.ndarray:
    """Depth of a plane receding towards the top of the image, 2-4 units away."""
    rows = np.linspace(4.0, 2.0, HEIGHT)[:, None]
    return np.repeat(rows, WIDTH, axis=1).astype(np.float32)


def predictor_for(true_depth: np.ndarray, noise: float = 0.0):
    rng = np.random.default_rng(1)

    def predict(image, width, height):
        # Monocular networks return inverse depth up to an unknown scale and shift.
        disparity = 7.0 / true_depth + 0.3
        return disparity + rng.normal(0.0, noise, disparity.shape)

    return predict


class ColmapArrayTests(unittest.TestCase):
    def test_depth_and_normal_maps_round_trip_in_colmap_layout(self):
        depth = np.arange(12, dtype=np.float32).reshape(3, 4)
        normals = np.random.default_rng(0).random((3, 4, 3)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            MODULE.write_colmap_array(root / "d.bin", depth)
            MODULE.write_colmap_array(root / "n.bin", normals)
            raw = (root / "d.bin").read_bytes()
            np.testing.assert_array_equal(MODULE.read_colmap_array(root / "d.bin"), depth)
            np.testing.assert_array_equal(MODULE.read_colmap_array(root / "n.bin"), normals)
        self.assertTrue(raw.startswith(b"4&3&1&"))
        # COLMAP stores (width, height, channels) in Fortran order: x varies fastest.
        self.assertEqual(struct.unpack("<2f", raw[6:14]), (0.0, 1.0))

    def test_reads_undistorted_camera_intrinsics_per_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            sparse = Path(temporary)
            (sparse / "cameras.bin").write_bytes(
                struct.pack("<Q", 1) + struct.pack("<iiQQ", 7, 1, 1080, 1920) + struct.pack("<4d", 1.0, 2.0, 3.0, 4.0)
            )
            image = struct.pack("<i7di", 1, 1, 0, 0, 0, 0, 0, 0, 7) + b"frame-0001.jpg\0"
            image += struct.pack("<Q", 1) + struct.pack("<ddq", 5.0, 6.0, -1)
            (sparse / "images.bin").write_bytes(struct.pack("<Q", 1) + image)
            intrinsics = MODULE.read_camera_intrinsics(sparse)
        self.assertEqual(intrinsics, {"frame-0001.jpg": (1080, 1.0, 2.0, 3.0, 4.0)})


class CompletionTests(unittest.TestCase):
    def test_fills_holes_accurately_and_never_changes_observed_depth(self):
        truth = tilted_plane()
        observed = truth.copy()
        observed[30:90, 40:120] = 0  # a textureless patch PatchMatch could not match
        depth, normals, report = MODULE.complete_frame(
            "frame-0001.jpg", IMAGE, observed, np.zeros((HEIGHT, WIDTH, 3), np.float32),
            INTRINSICS, predictor_for(truth, noise=0.002),
        )
        self.assertTrue(report["completed"])
        self.assertLess(report["heldOutMedianRelativeError"], 0.01)
        np.testing.assert_array_equal(depth[observed > 0], observed[observed > 0])
        hole = observed == 0
        self.assertTrue((depth[hole] > 0).all())
        self.assertLess(np.median(np.abs(depth[hole] - truth[hole]) / truth[hole]), 0.01)
        # Filled normals face the camera; observed normals are kept as they were.
        self.assertTrue((normals[35:85, 45:115, 2] < 0).all())
        self.assertTrue((normals[observed > 0] == 0).all())

    def test_skips_frames_without_enough_observed_depth(self):
        truth = tilted_plane()
        observed = np.zeros_like(truth)
        observed[:5, :5] = truth[:5, :5]
        normals = np.zeros((HEIGHT, WIDTH, 3), np.float32)
        depth, _, report = MODULE.complete_frame(
            "frame-0001.jpg", IMAGE, observed, normals, INTRINSICS, predictor_for(truth)
        )
        self.assertFalse(report["completed"])
        np.testing.assert_array_equal(depth, observed)

    def test_skips_frames_whose_prediction_disagrees_with_observed_depth(self):
        truth = tilted_plane()
        observed = truth.copy()
        observed[30:90, 40:120] = 0
        rng = np.random.default_rng(2)
        depth, _, report = MODULE.complete_frame(
            "frame-0001.jpg", IMAGE, observed, np.zeros((HEIGHT, WIDTH, 3), np.float32), INTRINSICS,
            lambda image, width, height: rng.random((height, width)) + 0.1,
        )
        self.assertFalse(report["completed"])
        self.assertEqual(report["reason"], "alignment error too high")
        np.testing.assert_array_equal(depth, observed)

    def test_discards_filled_depth_outside_the_observed_range(self):
        truth = tilted_plane()
        observed = truth.copy()
        observed[30:90, 40:120] = 0
        far = truth.copy()
        far[50:70, 70:90] = 100.0  # e.g. a window the network believes is far away
        depth, _, report = MODULE.complete_frame(
            "frame-0001.jpg", IMAGE, observed, np.zeros((HEIGHT, WIDTH, 3), np.float32),
            INTRINSICS, predictor_for(far),
        )
        self.assertTrue(report["completed"])
        self.assertTrue((depth[50:70, 70:90] == 0).all())

    def test_fast_mode_aligns_whole_frame_to_sparse_sfm_points(self):
        truth = tilted_plane()
        anchors = np.zeros_like(truth)
        rng = np.random.default_rng(3)
        rows, columns = rng.integers(0, HEIGHT, 400), rng.integers(0, WIDTH, 400)
        anchors[rows, columns] = truth[rows, columns]
        depth, normals, report = MODULE.predict_frame_from_sparse(
            "frame-0001.jpg", IMAGE, anchors, INTRINSICS, predictor_for(truth, noise=0.002)
        )
        self.assertTrue(report["completed"])
        self.assertTrue((depth > 0).all())
        self.assertLess(np.median(np.abs(depth - truth) / truth), 0.01)
        self.assertTrue((normals[10:-10, 10:-10, 2] < 0).all())

    def test_fast_mode_writes_empty_maps_for_frames_with_too_few_sfm_points(self):
        truth = tilted_plane()
        anchors = np.zeros_like(truth)
        anchors[:3, :3] = truth[:3, :3]
        depth, normals, report = MODULE.predict_frame_from_sparse(
            "frame-0001.jpg", IMAGE, anchors, INTRINSICS, predictor_for(truth)
        )
        self.assertFalse(report["completed"])
        self.assertEqual(report["reason"], "too few SfM points")
        self.assertFalse(depth.any() or normals.any())

    def test_sparse_depth_map_projects_triangulated_keypoints_only(self):
        rotation, translation = np.eye(3), np.array([0.0, 0.0, 1.0])
        xy = np.array([[20.0, 10.0], [40.0, 30.0], [60.0, 50.0]])
        ids = np.array([7, -1, 9])  # -1: keypoint without a 3D point
        points3d = {7: np.array([0.0, 0.0, 2.0]), 9: np.array([0.0, 0.0, -5.0])}  # 9 is behind the camera
        depth = MODULE.sparse_depth_map((rotation, translation, xy, ids), points3d, 0.5, 40, 30)
        self.assertEqual(depth[5, 10], 3.0)
        self.assertEqual(int((depth > 0).sum()), 1)

    def test_fronto_parallel_plane_normals_point_at_the_camera(self):
        normals = MODULE.normals_from_depth(np.full((20, 30), 2.0, np.float32), *INTRINSICS[1:])
        np.testing.assert_allclose(normals[5:15, 5:25], np.broadcast_to([0.0, 0.0, -1.0], (10, 20, 3)), atol=1e-6)


def write_workspace(root: Path, frames: int = 7, sparse_frame: int = 5) -> np.ndarray:
    """A COLMAP dense workspace whose cameras all see tilted_plane(); one frame has too few SfM points."""
    truth = tilted_plane()
    (root / "images").mkdir(parents=True)
    (root / "sparse").mkdir()
    _, fx, fy, cx, cy = INTRINSICS
    (root / "sparse" / "cameras.bin").write_bytes(
        struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 1, WIDTH, HEIGHT) + struct.pack("<4d", fx, fy, cx, cy))
    rng = np.random.default_rng(5)
    points, images = [], []
    for frame in range(1, frames + 1):
        name = f"frame-{frame:04d}.jpg"
        Image.new("RGB", (WIDTH, HEIGHT), (frame * 20, 40, 60)).save(root / "images" / name)
        count = 10 if frame == sparse_frame else 400
        u, v = rng.integers(0, WIDTH, count), rng.integers(0, HEIGHT, count)
        keypoints = b""
        for column, row in zip(u, v):
            z = float(truth[row, column])
            points.append((len(points) + 1, ((column - cx) / fx * z, (row - cy) / fy * z, z)))
            keypoints += struct.pack("<ddq", column + 0.25, row + 0.25, len(points))
        images.append(struct.pack("<i7di", frame, 1, 0, 0, 0, 0, 0, 0, 1) + name.encode() + b"\0"
                      + struct.pack("<Q", count) + keypoints)
    (root / "sparse" / "images.bin").write_bytes(struct.pack("<Q", frames) + b"".join(images))
    (root / "sparse" / "points3D.bin").write_bytes(struct.pack("<Q", len(points)) + b"".join(
        struct.pack("<Q3d", point_id, *xyz) + bytes(3) + struct.pack("<d", 0.5) + struct.pack("<Q", 0)
        for point_id, xyz in points))
    return truth


class BatchedWorkspaceTests(unittest.TestCase):
    def test_batched_prediction_matches_frame_by_frame_prediction(self):
        outputs = {}
        for batch_size in (1, 3, 4):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                truth = write_workspace(root)
                predict = predictor_for(truth)
                calls = []

                def predict_batch(images, width, height):
                    calls.append(len(images))
                    return [predict(image, width, height) for image in images]

                metrics = MODULE.predict_workspace_from_sparse(
                    root, predict, 960, cloud=root / "fused.ply",
                    predict_batch=predict_batch if batch_size > 1 else None, batch_size=batch_size)
                maps = {path.name: path.read_bytes() for path in sorted((root / "stereo" / "depth_maps").iterdir())}
                outputs[batch_size] = (metrics, maps, (root / "fused.ply").read_bytes())
                if batch_size > 1:
                    # Six frames have enough SfM points; frame 5 is skipped, never sent to the network.
                    self.assertEqual(sum(calls), 6)
                    self.assertLessEqual(max(calls), batch_size)
        self.assertEqual(outputs[1][0]["completedFrames"], 6)
        self.assertEqual(outputs[1][0]["skippedFrames"], [{"image": "frame-0005.jpg", "reason": "too few SfM points"}])
        for batch_size in (3, 4):
            self.assertEqual(outputs[batch_size], outputs[1])


class BackprojectedCloudTests(unittest.TestCase):
    def test_plane_is_back_projected_into_world_space_with_its_normal(self):
        depth = np.full((HEIGHT, WIDTH), 2.0, np.float32)
        depth[:, :40] = 5.0  # an occlusion edge at column 40
        # Camera one unit along +x of the world origin, looking down +z.
        pose = (np.eye(3), np.array([-1.0, 0.0, 0.0]), None, None)
        image = Image.new("RGB", (WIDTH, HEIGHT), (10, 20, 30))
        points = MODULE.backproject_frame(depth, image, pose, INTRINSICS, stride=4)
        near = points[points["z"] < 3]
        self.assertGreater(len(near), 0)
        np.testing.assert_allclose(near["z"], 2.0, atol=1e-6)
        # Pixel u maps to camera x = (u - cx) / fx * z, then world x = camera x + 1.
        self.assertAlmostEqual(float(near["x"].min()), (42 - WIDTH / 2) / 100 * 2 + 1, places=5)
        np.testing.assert_allclose(np.column_stack([near[name] for name in ("nx", "ny", "nz")]),
                                   np.broadcast_to([0.0, 0.0, -1.0], (len(near), 3)), atol=1e-6)
        self.assertTrue(((points["red"] == 10) & (points["blue"] == 30)).all())
        # Samples whose 3x3 neighbourhood spans the edge are dropped.
        u = np.rint((points["x"] - 1) / points["z"] * 100 + WIDTH / 2).astype(int)
        self.assertFalse(np.isin(u, [39, 40]).any())
        self.assertTrue(np.isin([38, 42], u).all())

    def test_cloud_is_a_readable_ply(self):
        points = np.zeros(3, MODULE.CLOUD_DTYPE)
        points["x"] = [1, 2, 3]
        points["green"] = 200
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fused.ply"
            MODULE.write_cloud(path, points)
            point_cloud = importlib.util.module_from_spec(
                spec := importlib.util.spec_from_file_location("point_cloud", MODULE_PATH.with_name("point_cloud.py")))
            spec.loader.exec_module(point_cloud)
            read = point_cloud.read_ply(path)
        np.testing.assert_array_equal(read["x"], [1, 2, 3])
        np.testing.assert_array_equal(read["green"], [200, 200, 200])


if __name__ == "__main__":
    unittest.main()
