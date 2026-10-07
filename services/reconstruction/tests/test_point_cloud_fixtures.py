from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "fixtures" / "point-clouds"
GENERATOR = ROOT / "scripts" / "generate_point_cloud_fixtures.py"


def read_points(path: Path) -> list[tuple[float, float, float]]:
    lines = path.read_text().splitlines()
    end_header = lines.index("end_header")
    return [tuple(map(float, line.split()[:3])) for line in lines[end_header + 1 :]]


class PointCloudFixtureTests(unittest.TestCase):
    def test_all_declared_empty_regions_really_are_empty(self):
        fixture_paths = sorted(FIXTURES.glob("*/fixture.json"))
        self.assertEqual(len(fixture_paths), 5)
        for metadata_path in fixture_paths:
            metadata = json.loads(metadata_path.read_text())
            points = read_points(metadata_path.with_name("cloud.ply"))
            self.assertEqual(len(points), metadata["pointCount"])
            for region in metadata["assertions"]["emptyRegions"]:
                lower, upper = region["min"], region["max"]
                occupants = [
                    point for point in points
                    if all(lower[axis] <= point[axis] <= upper[axis] for axis in range(3))
                ]
                self.assertEqual(
                    occupants, [],
                    f"{metadata_path.parent.name}/{region['name']} contains observed samples",
                )

    def test_metadata_hashes_and_bounds_match_ply(self):
        for metadata_path in sorted(FIXTURES.glob("*/fixture.json")):
            metadata = json.loads(metadata_path.read_text())
            ply = metadata_path.with_name("cloud.ply")
            self.assertEqual(hashlib.sha256(ply.read_bytes()).hexdigest(), metadata["plySha256"])
            points = read_points(ply)
            for axis in range(3):
                self.assertEqual(min(point[axis] for point in points), metadata["bounds"]["min"][axis])
                self.assertEqual(max(point[axis] for point in points), metadata["bounds"]["max"][axis])

    def test_generation_is_deterministic(self):
        original = {
            path.relative_to(FIXTURES): path.read_bytes()
            for path in FIXTURES.glob("*/*") if path.is_file()
        }
        subprocess.run([sys.executable, str(GENERATOR)], cwd=ROOT, check=True, capture_output=True)
        regenerated = {
            path.relative_to(FIXTURES): path.read_bytes()
            for path in FIXTURES.glob("*/*") if path.is_file()
        }
        self.assertEqual(original, regenerated)

    def test_generator_can_create_one_selected_fixture(self):
        # Running outside the repository is intentionally harmless: output is
        # always resolved from the script location, never the caller's cwd.
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, str(GENERATOR), "wall-with-window"],
                cwd=temporary, check=True, capture_output=True, text=True,
            )
        self.assertIn("generated wall-with-window", result.stdout)


if __name__ == "__main__":
    unittest.main()
