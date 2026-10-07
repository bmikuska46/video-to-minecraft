from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "video_to_world.py"
SPEC = importlib.util.spec_from_file_location("video_to_world", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


MANIFEST = {
    "status": "COMPLETED",
    "source": {"sha256": "normalized", "original": {"sha256": "original"}},
    "settings": {"processingMode": "fast", "scanType": "scene"},
    "metrics": {"bounds": {"min": [-2.0, 0.0, -1.0], "max": [4.0, 3.0, 1.0]}},
}


def selection_args(**overrides) -> argparse.Namespace:
    values = {"rotate": (0, 0, 0), "crop_min": None, "crop_max": None, "top": None,
              "axis": "longest", "blocks": 60, "max_blocks": 1_000_000}
    return argparse.Namespace(**(values | overrides))


class RigidTransformTest(unittest.TestCase):
    def test_matches_the_app_rotation_order(self):
        # geometry.ts: Rz * Ry * Rx; a quarter turn about y maps +x to -z.
        self.assertEqual(MODULE.rigid_transform((0, 0, 0)), [float(v) for v in
                         (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)])
        turned = MODULE.rigid_transform((0, 1, 0))
        self.assertEqual((turned[0], turned[4], turned[8]), (0.0, 0.0, -1.0))
        combined = MODULE.rigid_transform((1, 1, 0))
        # Rx first: +y -> +z, then Ry: +z -> +x.
        self.assertEqual((combined[1], combined[5], combined[9]), (1.0, 0.0, 0.0))

    def test_negative_and_full_turns_wrap(self):
        self.assertEqual(MODULE.rigid_transform((0, -1, 0)), MODULE.rigid_transform((0, 3, 0)))
        self.assertEqual(MODULE.rigid_transform((4, 4, 4)), MODULE.rigid_transform((0, 0, 0)))


class ReconstructionReuseTest(unittest.TestCase):
    def test_accepts_original_or_normalized_video_hash(self):
        self.assertTrue(MODULE.reconstruction_matches(MANIFEST, "original", "fast", "scene"))
        self.assertTrue(MODULE.reconstruction_matches(MANIFEST, "normalized", "fast", "scene"))

    def test_rejects_other_video_or_settings(self):
        self.assertFalse(MODULE.reconstruction_matches(MANIFEST, "other", "fast", "scene"))
        self.assertFalse(MODULE.reconstruction_matches(MANIFEST, "original", "detailed", "scene"))
        self.assertFalse(MODULE.reconstruction_matches(MANIFEST, "original", "fast", "object"))


class ExportSelectionTest(unittest.TestCase):
    def test_defaults_to_full_bounds_along_the_longest_axis(self):
        selection = MODULE.export_selection(selection_args(), MANIFEST)
        self.assertEqual(selection["cropMin"], [-2.0, 0.0, -1.0])
        self.assertEqual(selection["cropMax"], [4.0, 3.0, 1.0])
        self.assertEqual(selection["axis"], "x")
        self.assertEqual(selection["gridDimensions"], [60, 30, 20])

    def test_rotation_moves_the_longest_axis_and_top_cuts_the_ceiling(self):
        selection = MODULE.export_selection(selection_args(rotate=(0, 1, 0), top=2.0), MANIFEST)
        self.assertEqual(selection["axis"], "z")
        self.assertEqual(selection["cropMax"][1], 2.0)
        self.assertEqual(selection["gridDimensions"], [20, 20, 60])

    def test_rejects_crops_outside_the_bounds_and_oversized_grids(self):
        with self.assertRaises(SystemExit):
            MODULE.export_selection(selection_args(top=5.0), MANIFEST)
        with self.assertRaises(SystemExit):
            MODULE.export_selection(selection_args(crop_min=[1, 1, 0], crop_max=[0, 2, 1]), MANIFEST)
        with self.assertRaises(SystemExit):
            MODULE.export_selection(selection_args(max_blocks=1000), MANIFEST)


if __name__ == "__main__":
    unittest.main()
