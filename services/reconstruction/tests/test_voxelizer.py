from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SERVICE = ROOT / "services" / "reconstruction"
sys.path.insert(0, str(SERVICE))
SPEC = importlib.util.spec_from_file_location("voxelizer", SERVICE / "voxelizer.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PALETTE_PATH = ROOT / "packages" / "block-palette" / "palette-v1.json"
FIXTURES = ROOT / "fixtures" / "point-clouds"


def points(records: list[tuple]) -> np.ndarray:
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("confidence", "<f4"), ("source_view_id", "<u4"),
    ])
    return np.asarray(records, dtype=dtype)


class VoxelizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.palette = MODULE.load_palette(PALETTE_PATH)

    def test_palette_is_versioned_unique_and_geometry_safe(self):
        payload = json.loads(PALETTE_PATH.read_text())
        self.assertEqual(self.palette.version, "geometry-safe-v1")
        self.assertEqual(len({entry.id for entry in self.palette.entries}), len(self.palette.entries))
        states = {entry.block_state for entry in self.palette.entries}
        forbidden = ("sand", "gravel", "glass", "water", "lava", "leaves", "grass",
                     "torch", "stairs", "slab")
        self.assertFalse(any(token in state for state in states for token in forbidden))
        self.assertEqual(payload["schemaVersion"], 1)

    def test_linear_rgb_median_and_palette_match(self):
        cloud = points([
            (0.01, 0.01, 0.01, 0, 0, 1, 140, 35, 35, 1.0, 1),
            (0.02, 0.02, 0.01, 0, 0, 1, 142, 33, 33, 1.0, 2),
            (0.03, 0.03, 0.01, 0, 0, 1, 145, 30, 30, 0.9, 3),
        ])
        result = MODULE.voxelize_points(
            cloud, crop_min=[0, 0, 0], crop_max=[1, 1, 1], scale_axis="x", blocks=1,
            palette=self.palette,
        )
        self.assertEqual(len(result.voxels), 1)
        voxel = result.voxels[0]
        self.assertEqual(voxel.block_state, "minecraft:red_concrete")
        self.assertEqual(voxel.supporting_points, 3)
        self.assertEqual(voxel.distinct_supporting_frames, 3)
        self.assertEqual(voxel.color_srgb, (142, 33, 33))

    def test_crop_transform_scale_and_world_placement(self):
        cloud = points([
            (0.0, 0.0, 0.0, 0, 1, 0, 125, 125, 125, 1.0, 1),
            (0.1, 0.0, 0.0, 0, 1, 0, 125, 125, 125, 1.0, 2),
            (5.0, 5.0, 5.0, 0, 1, 0, 125, 125, 125, 1.0, 3),
        ])
        transform = [1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3, 0, 0, 0, 1]
        result = MODULE.voxelize_points(
            cloud, transform=transform, crop_min=[0.9, 1.9, 2.9], crop_max=[2.9, 3.9, 4.9],
            scale_axis="x", blocks=2, palette=self.palette,
        )
        self.assertAlmostEqual(result.voxel_size, 1.0)
        self.assertEqual(result.cropped_points, 2)
        self.assertEqual(result.voxels[0].y, 0)
        self.assertEqual((result.voxels[0].x, result.voxels[0].z), (0, 0))

    def test_scale_is_exact_for_every_selected_axis(self):
        cloud = points([
            (-3.9, -1.9, -7.9, 0, 1, 0, 125, 125, 125, 1.0, 1),
            (3.9, 1.9, 7.9, 0, 1, 0, 125, 125, 125, 1.0, 2),
        ])
        crop_min = [-4.0, -2.0, -8.0]
        crop_max = [4.0, 2.0, 8.0]
        for axis, blocks, expected_voxel_size in (
            ("x", 8, 1.0), ("y", 8, 0.5), ("z", 8, 2.0)
        ):
            with self.subTest(axis=axis):
                result = MODULE.voxelize_points(
                    cloud, crop_min=crop_min, crop_max=crop_max,
                    scale_axis=axis, blocks=blocks, palette=self.palette,
                )
                self.assertEqual(result.voxel_size, expected_voxel_size)
                selected_extent = crop_max["xyz".index(axis)] - crop_min["xyz".index(axis)]
                self.assertEqual(selected_extent / result.voxel_size, blocks)

    def test_negative_coordinates_quantize_relative_to_crop_origin(self):
        cloud = points([
            (-2.9, -1.9, -0.9, 0, 1, 0, 125, 125, 125, 1.0, 1),
            (-1.1, -0.1, 0.9, 0, 1, 0, 125, 125, 125, 1.0, 2),
        ])
        result = MODULE.voxelize_points(
            cloud, crop_min=[-3, -2, -1], crop_max=[-1, 0, 1],
            scale_axis="x", blocks=2, palette=self.palette,
        )

        self.assertEqual({voxel.source_cell for voxel in result.voxels},
                         {(0, 0, 0), (1, 1, 1)})
        self.assertTrue(all(voxel.y >= 0 for voxel in result.voxels))

    def test_known_free_cell_can_never_be_emitted(self):
        cloud = points([
            (0.1, 0.1, 0.1, 0, 0, 1, 125, 125, 125, 1.0, 1),
            (0.2, 0.2, 0.2, 0, 0, 1, 125, 125, 125, 1.0, 2),
            (1.1, 0.1, 0.1, 0, 0, 1, 125, 125, 125, 1.0, 3),
            (1.2, 0.2, 0.2, 0, 0, 1, 125, 125, 125, 1.0, 4),
        ])
        result = MODULE.voxelize_points(
            cloud, crop_min=[0, 0, 0], crop_max=[2, 1, 1], scale_axis="x", blocks=2,
            palette=self.palette, known_free_cells={(0, 0, 0)},
        )
        self.assertEqual({voxel.source_cell for voxel in result.voxels}, {(1, 0, 0)})
        self.assertTrue(all(voxel.source_cell not in result.known_free_cells for voxel in result.voxels))

    def test_visibility_rays_mark_traversed_cells_but_not_observed_endpoint(self):
        cloud = points([
            (2.1, 0.1, 0.1, 0, 0, -1, 125, 125, 125, 1.0, 7),
            (2.2, 0.2, 0.2, 0, 0, -1, 125, 125, 125, 1.0, 7),
        ])
        result = MODULE.voxelize_points(
            cloud, crop_min=[0, 0, 0], crop_max=[3, 1, 1], scale_axis="x", blocks=3,
            palette=self.palette, camera_origins={7: [0.1, 0.1, 0.1]},
        )
        self.assertIn((0, 0, 0), result.known_free_cells)
        self.assertIn((1, 0, 0), result.known_free_cells)
        self.assertNotIn((2, 0, 0), result.known_free_cells)
        self.assertEqual(result.voxels[0].source_cell, (2, 0, 0))

    def test_isolated_removal_only_drops_weak_cell(self):
        cloud = points([
            (0.1, 0.1, 0.1, 0, 0, 1, 125, 125, 125, 1.0, 1),
            (0.2, 0.2, 0.2, 0, 0, 1, 125, 125, 125, 1.0, 2),
            (2.1, 0.1, 0.1, 0, 0, 1, 125, 125, 125, 1.0, 3),
            (2.2, 0.2, 0.2, 0, 0, 1, 125, 125, 125, 1.0, 4),
            (2.3, 0.3, 0.3, 0, 0, 1, 125, 125, 125, 1.0, 5),
        ])
        result = MODULE.voxelize_points(
            cloud, crop_min=[0, 0, 0], crop_max=[3, 1, 1], scale_axis="x", blocks=3,
            palette=self.palette,
            config=MODULE.VoxelizationConfig(isolated_cell_minimum_points=3),
        )
        self.assertEqual({voxel.source_cell for voxel in result.voxels}, {(2, 0, 0)})

    def test_golden_open_surfaces_remain_open(self):
        for metadata_path in sorted(FIXTURES.glob("*/fixture.json")):
            metadata = json.loads(metadata_path.read_text())
            original_lower = metadata["bounds"]["min"]
            original_upper = metadata["bounds"]["max"]
            # Give zero-thickness fixture axes a real crop extent.
            lower = [value - 0.051 if value == original_upper[i] else value
                     for i, value in enumerate(original_lower)]
            upper = [value + 0.051 if value == original_lower[i] else value
                     for i, value in enumerate(original_upper)]
            crop = metadata["assertions"].get("recommendedCrop")
            if crop:
                lower, upper = crop["min"], crop["max"]
            cloud = __import__("point_cloud").read_ply(metadata_path.with_name("cloud.ply"))
            extent = np.asarray(upper) - np.asarray(lower)
            axis_index = int(np.argmax(extent))
            axis = "xyz"[axis_index]
            blocks = max(1, round(extent[axis_index] / metadata["sampleSpacing"]))
            result = MODULE.voxelize_points(
                cloud, crop_min=lower, crop_max=upper, scale_axis=axis, blocks=blocks,
                palette=self.palette,
            )
            centers = [np.asarray(lower) + (np.asarray(voxel.source_cell) + 0.5) * result.voxel_size
                       for voxel in result.voxels]
            for region in metadata["assertions"]["emptyRegions"]:
                region_min, region_max = np.asarray(region["min"]), np.asarray(region["max"])
                self.assertFalse(any(np.all(center >= region_min) and np.all(center <= region_max)
                                     for center in centers),
                                 f"{metadata_path.parent.name}/{region['name']} was filled")

    def test_json_output_preserves_support_audit_fields(self):
        cloud = points([
            (0.1, 0.1, 0.1, 0, 0, 1, 125, 125, 125, 0.9, 1),
            (0.2, 0.2, 0.2, 0, 0, 1, 125, 125, 125, 0.8, 2),
        ])
        result = MODULE.voxelize_points(
            cloud, crop_min=[0, 0, 0], crop_max=[1, 1, 1], scale_axis="x", blocks=1,
            palette=self.palette,
        )
        payload = MODULE.result_to_dict(result, self.palette)
        voxel = payload["voxels"][0]
        self.assertEqual({"supportingPoints", "distinctSupportingFrames",
                          "nearestObservedPointDistance", "confidence"} - voxel.keys(), set())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "voxels.json"
            path.write_text(json.dumps(payload))
            self.assertEqual(json.loads(path.read_text())["occupiedBlocks"], 1)


class ImplementationEquivalenceTests(unittest.TestCase):
    """The vectorized voxelizer must reproduce the original loop exactly, field for field."""

    @classmethod
    def setUpClass(cls):
        cls.palette = MODULE.load_palette(PALETTE_PATH)

    def assert_same(self, cloud, **kwargs):
        loop = MODULE.voxelize_points(cloud, implementation="loop", palette=self.palette, **kwargs)
        vectorized = MODULE.voxelize_points(cloud, implementation="vectorized", palette=self.palette, **kwargs)
        self.assertEqual(loop, vectorized)
        from voxel_contract import encode
        self.assertEqual(encode(loop, self.palette), encode(vectorized, self.palette))
        return vectorized

    def random_cloud(self, seed: int, count: int, *, unit_weights: bool, quantized: bool) -> np.ndarray:
        rng = np.random.default_rng(seed)
        xyz = rng.uniform(-1.0, 1.0, (count, 3))
        if quantized:
            # Many points exactly on voxel planes and many equal colors: ties everywhere.
            xyz = np.round(xyz * 8) / 8
        colors = rng.integers(0, 256, (count, 3))
        if quantized:
            colors = colors // 64 * 64
        normals = rng.normal(size=(count, 3))
        confidence = np.ones(count) if unit_weights else rng.uniform(0.2, 1.0, count)
        return points([(*xyz[i], *normals[i], *colors[i], confidence[i], int(rng.integers(0, 5)))
                       for i in range(count)])

    def test_random_clouds_match_with_unit_and_varying_weights(self):
        for seed, unit_weights, quantized in ((1, True, False), (2, True, True), (3, False, False),
                                              (4, False, True)):
            cloud = self.random_cloud(seed, 4000, unit_weights=unit_weights, quantized=quantized)
            for blocks, axis in ((7, "x"), (16, "y"), (31, "z")):
                with self.subTest(seed=seed, blocks=blocks):
                    self.assert_same(cloud, crop_min=[-1, -1, -1], crop_max=[1, 1, 1],
                                     scale_axis=axis, blocks=blocks)

    def test_thresholds_known_free_cells_and_isolated_removal_match(self):
        cloud = self.random_cloud(5, 3000, unit_weights=True, quantized=False)
        free = [(0, 0, 0), (3, 4, 5), (-1, 2, 2), (40, 40, 40)]
        config = MODULE.VoxelizationConfig(minimum_supporting_points=2, minimum_distinct_views=2,
                                           isolated_cell_minimum_points=4)
        result = self.assert_same(cloud, crop_min=[-1, -1, -1], crop_max=[1, 1, 1], scale_axis="x",
                                  blocks=12, config=config, known_free_cells=free)
        self.assertGreater(result.rejected_cells, 0)
        self.assertFalse({voxel.source_cell for voxel in result.voxels} & set(free))

    def test_transform_and_support_counts_match(self):
        base = self.random_cloud(6, 2000, unit_weights=True, quantized=False)
        dtype = np.dtype([(name, base.dtype[name]) for name in base.dtype.names if name != "source_view_id"]
                         + [("support_count", "<u2")])
        cloud = np.empty(len(base), dtype)
        for name in base.dtype.names:
            if name != "source_view_id":
                cloud[name] = base[name]
        cloud["support_count"] = np.random.default_rng(7).integers(1, 9, len(base))
        quarter_turn = [0.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.0, -0.25, -1.0, 0.0, 0.0, 2.0, 0, 0, 0, 1]
        self.assert_same(cloud, transform=quarter_turn, crop_min=[-0.5, -1.25, 1.0],
                         crop_max=[1.5, 0.75, 3.0], scale_axis="z", blocks=10,
                         config=MODULE.VoxelizationConfig(minimum_distinct_views=3))

    def test_golden_fixtures_match(self):
        for cloud_path in sorted(FIXTURES.glob("*/cloud.ply")):
            cloud = __import__("point_cloud").read_ply(cloud_path)
            lower = [float(cloud[axis].min()) - 0.05 for axis in "xyz"]
            upper = [float(cloud[axis].max()) + 0.05 for axis in "xyz"]
            with self.subTest(fixture=cloud_path.parent.name):
                self.assert_same(cloud, crop_min=lower, crop_max=upper, scale_axis="x", blocks=40)


if __name__ == "__main__":
    unittest.main()
