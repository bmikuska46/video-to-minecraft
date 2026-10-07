from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SERVICE = ROOT / "services" / "reconstruction"
sys.path.insert(0, str(SERVICE))
SPEC = importlib.util.spec_from_file_location("voxelizer", SERVICE / "voxelizer.py")
assert SPEC and SPEC.loader
VOXELIZER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VOXELIZER
SPEC.loader.exec_module(VOXELIZER)
import voxel_contract as CONTRACT


def voxel(x: int, y: int, z: int, palette_id: int):
    return VOXELIZER.Voxel(
        x, y, z, (x, y, z), palette_id, "unused", 1, 1, 0.0, 1.0,
        (0, 0, 0), (0.0, 1.0, 0.0),
    )


def fixture():
    entries = (
        VOXELIZER.PaletteEntry(2, "minecraft:white_concrete", (255, 255, 255), (0, 0, 0), 0),
        VOXELIZER.PaletteEntry(1, "minecraft:stone", (125, 125, 125), (0, 0, 0), 0),
    )
    palette = VOXELIZER.Palette("test-v1", "26.2", entries)
    result = VOXELIZER.VoxelizationResult(
        1.0,
        (voxel(0, 17, 0, 2), voxel(0, 0, 0, 1),
         voxel(-1, 0, 0, 1), voxel(0, -1, 0, 2)),
        frozenset(), (0, 0, 0), (1, 1, 1), (0, 0, 0), 4, 4, 0,
    )
    return result, palette


class VoxelContractTests(unittest.TestCase):
    def test_round_trip_is_deterministic_and_canonically_sorted(self):
        result, palette = fixture()
        first = CONTRACT.encode(result, palette)
        second = CONTRACT.encode(result, palette)
        self.assertEqual(first, second)
        message = CONTRACT.decode(first)
        self.assertEqual([(item.x, item.y, item.z) for item in message.voxels],
                         [(-1, 0, 0), (0, -1, 0), (0, 0, 0), (0, 17, 0)])
        self.assertEqual((message.bounds.min_x, message.bounds.min_y,
                          message.bounds.max_x, message.bounds.max_y), (-1, -1, 0, 17))

    def test_java_test_uses_exact_python_fixture(self):
        result, palette = fixture()
        java_test = (ROOT / "packages/contracts/java/src/test/java/dev/minecraftvideo/contracts/"
                     "VoxelWorldCompatibilityTest.java").read_text()
        encoded = base64.b64encode(CONTRACT.encode(result, palette)).decode("ascii")
        self.assertIn(f'"{encoded}"', java_test)

    def test_rejects_unknown_palette_reference(self):
        result, palette = fixture()
        broken = VOXELIZER.VoxelizationResult(
            result.voxel_size, (voxel(0, 0, 0, 99),), result.known_free_cells,
            result.transformed_crop_min, result.transformed_crop_max, result.translation,
            result.input_points, result.cropped_points, result.rejected_cells,
        )
        with self.assertRaises(CONTRACT.VoxelContractError):
            CONTRACT.encode(broken, palette)


if __name__ == "__main__":
    unittest.main()
