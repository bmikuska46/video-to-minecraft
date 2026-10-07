from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "point_cloud.py"
SPEC = importlib.util.spec_from_file_location("point_cloud", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PointCloudProcessingTests(unittest.TestCase):
    def test_filtering_never_creates_points_and_preserves_vertex_properties(self):
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ])
        points = np.zeros(101, dtype=dtype)
        grid = np.linspace(0.0, 0.01, 100)
        points["x"][:100] = grid
        points["y"][:100] = grid
        points["ny"] = 1
        points["red"] = np.arange(101, dtype=np.uint8)
        points["x"][100] = 100.0  # unequivocal isolated outlier
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            MODULE.write_ply(root / "source.ply", points)
            metrics = MODULE.filter_and_align(
                root / "source.ply", root / "filtered.ply", root / "preview.ply",
                up_vector=[0.0, 1.0, 0.0], preview_limit=20,
            )
            filtered = MODULE.read_ply(root / "filtered.ply")
            preview = MODULE.read_ply(root / "preview.ply")
        self.assertLessEqual(len(filtered), len(points))
        self.assertEqual(len(preview), 20)
        self.assertEqual(filtered.dtype.names, points.dtype.names)
        self.assertEqual(metrics["alignmentSource"], "camera_or_gravity")

    def test_observed_reference_tags_points_backed_by_multi_view_stereo(self):
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        rng = np.random.default_rng(0)
        observed = np.zeros(400, dtype=dtype)
        observed["x"], observed["y"] = rng.random(400), rng.random(400)
        inferred = np.zeros(400, dtype=dtype)
        inferred["x"], inferred["y"] = rng.random(400), rng.random(400)
        inferred["z"] = 0.5  # a surface the observed-only fusion never saw
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            MODULE.write_ply(root / "observed.ply", observed)
            MODULE.write_ply(root / "fused.ply", np.concatenate([observed, inferred]))
            metrics = MODULE.filter_and_align(
                root / "fused.ply", root / "filtered.ply", root / "preview.ply",
                up_vector=None, observed_reference=root / "observed.ply",
            )
            filtered = MODULE.read_ply(root / "filtered.ply")
            preview = MODULE.read_ply(root / "preview.ply")
        self.assertEqual(filtered.dtype.names[-1], "observed")
        self.assertEqual(int(filtered["observed"][np.isclose(filtered["z"], 0)].min()), 1)
        self.assertEqual(int(filtered["observed"][np.isclose(filtered["z"], 0.5)].max()), 0)
        self.assertEqual(metrics["observedPoints"] + metrics["inferredPoints"], len(filtered))
        self.assertIn("observed", preview.dtype.names)

    def test_all_inferred_tags_every_point_for_fast_mode(self):
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        points = np.zeros(200, dtype=dtype)
        points["x"], points["y"] = np.random.default_rng(1).random((2, 200))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            MODULE.write_ply(root / "fused.ply", points)
            metrics = MODULE.filter_and_align(
                root / "fused.ply", root / "filtered.ply", root / "preview.ply", up_vector=None, all_inferred=True,
            )
            filtered = MODULE.read_ply(root / "filtered.ply")
        self.assertFalse(filtered["observed"].any())
        self.assertEqual((metrics["observedPoints"], metrics["inferredPoints"]), (0, len(filtered)))

    @staticmethod
    def room(tilt_degrees: float) -> np.ndarray:
        """A 4 x 3 x 2.5 room (floor, two walls, a bed-like box) tilted about X."""
        rng = np.random.default_rng(1)
        floor = np.column_stack([rng.uniform(0, 4, 20_000), np.zeros(20_000), rng.uniform(0, 3, 20_000)])
        wall_x = np.column_stack([np.zeros(8_000), rng.uniform(0, 2.5, 8_000), rng.uniform(0, 3, 8_000)])
        wall_z = np.column_stack([rng.uniform(0, 4, 8_000), rng.uniform(0, 2.5, 8_000), np.full(8_000, 3.0)])
        bed = np.column_stack([rng.uniform(1, 3, 6_000), np.full(6_000, 0.5), rng.uniform(1, 2.5, 6_000)])
        xyz = np.vstack([floor, wall_x, wall_z, bed])
        xyz += rng.normal(0, 0.005, xyz.shape)
        angle = np.radians(tilt_degrees)
        rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        return xyz @ rotation.T

    def test_floor_leveling_snaps_a_slightly_tilted_floor_level(self):
        xyz = self.room(4.0)
        rotation, details = MODULE.floor_leveling_rotation(xyz)
        self.assertTrue(details["applied"])
        self.assertAlmostEqual(details["correctionDegrees"], 4.0, delta=0.2)
        floor_heights = (xyz[:20_000] @ rotation.T)[:, 1]
        self.assertLess(float(np.ptp(np.percentile(floor_heights, [5, 95]))), 0.03)

    def test_floor_leveling_refuses_large_corrections_and_unsupported_clouds(self):
        _, steep = MODULE.floor_leveling_rotation(self.room(25.0))
        self.assertFalse(steep["applied"])
        line = np.column_stack([np.linspace(0, 1, 300)] * 3)
        rotation, degenerate = MODULE.floor_leveling_rotation(line)
        self.assertFalse(degenerate["applied"])
        np.testing.assert_array_equal(rotation, np.eye(3))

    def test_alignment_levels_the_floor_after_the_camera_up_rotation(self):
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        xyz = self.room(3.0)
        points = np.zeros(len(xyz), dtype=dtype)
        for index, name in enumerate(("x", "y", "z")):
            points[name] = xyz[:, index]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            MODULE.write_ply(root / "source.ply", points)
            metrics = MODULE.filter_and_align(
                root / "source.ply", root / "filtered.ply", root / "preview.ply", up_vector=[0.0, 1.0, 0.0],
            )
            filtered = MODULE.read_ply(root / "filtered.ply")
        self.assertTrue(metrics["floorLeveling"]["applied"])
        self.assertEqual(len(filtered), len(points))
        rotation = np.asarray(metrics["rotation"])
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        floor_heights = np.asarray(filtered["y"][:20_000], dtype=float)
        self.assertLess(float(np.ptp(np.percentile(floor_heights, [5, 95]))), 0.03)

    def test_rotation_maps_camera_up_to_positive_y(self):
        rotation = MODULE.rotation_from_to(
            np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])
        )
        np.testing.assert_allclose(rotation @ np.array([0.0, -1.0, 0.0]), [0.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
