"""Tests against a real walk-around video rather than the synthetic smoke fixtures.

Fetch the fixture once (about 185 MB of the 4.4 GB source is downloaded):

    python3 scripts/fetch_real_capture.py

The fixture checks run whenever the clip is present. The GPU acceptance test
reconstructs it end to end and is opt-in because it takes several minutes:

    RUN_REAL_CAPTURE_ACCEPTANCE=1 pytest services/reconstruction/tests/test_real_capture.py

Set REAL_CAPTURE_RECONSTRUCTION to an existing output directory to re-check a
previous reconstruction without re-running COLMAP.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SERVICE = ROOT / "services" / "reconstruction"
FIXTURE = ROOT / "fixtures" / "captures" / "barn-gable-arc-real"
VIDEO = FIXTURE / "video.mp4"
CAPTURE_SCHEMA = ROOT / "packages" / "contracts" / "capture.schema.json"

sys.path.insert(0, str(SERVICE))
SPEC = importlib.util.spec_from_file_location("reconstruct_fixture", SERVICE / "reconstruct_fixture.py")
assert SPEC and SPEC.loader
RECONSTRUCT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECONSTRUCT)

from point_cloud import read_ply  # noqa: E402
from voxel_contract import decode as decode_voxel_contract, write as write_voxel_contract  # noqa: E402
from voxelizer import VoxelizationConfig, load_palette, voxelize_points  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@unittest.skipUnless(VIDEO.is_file(), "real capture missing; run scripts/fetch_real_capture.py")
class RealCaptureFixtureTests(unittest.TestCase):
    def test_video_matches_recorded_checksums(self):
        recorded = (FIXTURE / "sha256.txt").read_text().split()[0]
        fixture = json.loads((FIXTURE / "fixture.json").read_text())
        actual = sha256(VIDEO)
        self.assertEqual(actual, recorded)
        self.assertEqual(actual, fixture["videoSha256"])

    def test_pipeline_validation_accepts_the_real_clip(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata = RECONSTRUCT.probe(VIDEO, Path(temporary))
        self.assertEqual(metadata["stream"]["codec_name"], "h264")
        self.assertEqual((metadata["stream"]["width"], metadata["stream"]["height"]), (1920, 1080))
        self.assertGreaterEqual(metadata["durationSeconds"], 14.9)
        self.assertLessEqual(metadata["durationSeconds"], RECONSTRUCT.MAX_DURATION_SECONDS["object"])

    def test_capture_metadata_follows_contract_and_matches_stream(self):
        schema = json.loads(CAPTURE_SCHEMA.read_text())
        capture = json.loads((FIXTURE / "capture.json").read_text())
        self.assertLessEqual(set(schema["required"]), set(capture))
        self.assertLessEqual(set(capture), set(schema["properties"]))
        self.assertEqual(capture["codec"], "h264")
        self.assertIn(capture["orientation"], schema["properties"]["orientation"]["enum"])
        stream = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(VIDEO)],
            text=True, capture_output=True, check=True,
        ).stdout)["streams"][0]
        numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
        self.assertEqual((capture["width"], capture["height"]), (stream["width"], stream["height"]))
        self.assertAlmostEqual(capture["fps"], numerator / denominator, places=2)
        self.assertEqual(int(stream["nb_read_frames"]), 450)

    def test_fixture_records_license_and_attribution(self):
        source = json.loads((FIXTURE / "fixture.json").read_text())["source"]
        self.assertEqual(source["license"], "CC BY 4.0")
        self.assertIn("Tanks and Temples", source["attribution"])
        self.assertTrue(source["changes"])
        self.assertEqual(len(source["segmentSha256"]), 64)


def up_angle_degrees(normal: np.ndarray) -> float:
    normal = normal / np.linalg.norm(normal)
    return float(np.degrees(np.arccos(abs(normal[1]))))


def plane_normal(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered / len(points))
    return vectors[:, 0]


@unittest.skipUnless(
    os.environ.get("RUN_REAL_CAPTURE_ACCEPTANCE") == "1" or os.environ.get("REAL_CAPTURE_RECONSTRUCTION"),
    "set RUN_REAL_CAPTURE_ACCEPTANCE=1 to reconstruct the real capture on the GPU",
)
@unittest.skipUnless(VIDEO.is_file(), "real capture missing; run scripts/fetch_real_capture.py")
class RealCaptureReconstructionAcceptanceTests(unittest.TestCase):
    """End-to-end reconstruction and voxelization of real footage.

    Thresholds are regression bounds with margin below what the pinned COLMAP
    4.0.4 image achieved on 2026-09-28 (60/60 frames registered, 0.45 px median
    reprojection error, 1.67M fused points, lawn plane 2.5 degrees from +Y).
    """

    output: Path
    manifest: dict
    points: np.ndarray

    @classmethod
    def setUpClass(cls):
        existing = os.environ.get("REAL_CAPTURE_RECONSTRUCTION")
        if existing:
            cls.output = Path(existing).resolve()
        else:
            name = f"{FIXTURE.name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
            cls.output = ROOT / "artifacts" / "reconstructions" / name
            subprocess.run(
                [str(ROOT / "scripts" / "reconstruct_fixture.sh"), FIXTURE.name, name],
                check=False, timeout=90 * 60,
            )
        cls.manifest = json.loads((cls.output / "manifest.json").read_text())
        canonical = cls.output / "canonical.ply"
        cls.points = read_ply(canonical) if canonical.is_file() else None

    def xyz(self) -> np.ndarray:
        return np.column_stack([self.points[axis] for axis in ("x", "y", "z")]).astype(np.float64)

    def test_every_pipeline_stage_completes(self):
        self.assertEqual(self.manifest["status"], "COMPLETED", self.manifest.get("failure"))
        self.assertEqual(
            [stage["name"] for stage in self.manifest["stages"]],
            ["VALIDATION", "EXTRACTING_FRAMES", "FEATURE_EXTRACTION", "FEATURE_MATCHING",
             "SPARSE_RECONSTRUCTION", "RECONSTRUCTION_METRICS", "UNDISTORTION",
             "DENSE_RECONSTRUCTION", "FUSION", "FILTERING_AND_ALIGNMENT"],
        )
        self.assertTrue(all(stage["status"] == "COMPLETED" for stage in self.manifest["stages"]))
        self.assertEqual(self.manifest["source"]["sha256"], sha256(VIDEO))

    def test_sparse_model_is_well_constrained(self):
        metrics = self.manifest["metrics"]
        self.assertTrue(metrics["qualityDecision"]["usable"], metrics["qualityDecision"]["reasons"])
        self.assertGreaterEqual(metrics["selectedFrames"], RECONSTRUCT.MIN_SELECTED_FRAMES)
        self.assertGreaterEqual(metrics["registrationRatio"], 0.9)
        self.assertLessEqual(metrics["medianReprojectionErrorPixels"], 1.0)
        self.assertLessEqual(metrics["maximumTrajectoryStepRatio"], 4.0)
        self.assertGreaterEqual(metrics["points3D"], 10_000)

    def test_dense_cloud_is_substantial_and_filtering_is_conservative(self):
        metrics = self.manifest["metrics"]
        self.assertGreaterEqual(metrics["fusedPoints"], 500_000)
        self.assertGreaterEqual(metrics["filteredPoints"], 0.85 * metrics["fusedPoints"])
        self.assertLessEqual(metrics["filteredPoints"], metrics["fusedPoints"])
        self.assertEqual(metrics["previewPoints"], min(100_000, metrics["filteredPoints"]))
        self.assertEqual(len(self.points), metrics["filteredPoints"])
        self.assertTrue(np.isfinite(self.xyz()).all())

    def test_canonical_cloud_is_upright_with_building_above_lawn(self):
        xyz = self.xyz()
        red, green, blue = (self.points[channel].astype(float) for channel in ("red", "green", "blue"))
        grass = (green > 1.15 * red) & (green > 1.15 * blue)
        # The lower half excludes tree canopies, leaving the lawn the camera walked across.
        lawn = grass & (xyz[:, 1] < np.median(xyz[:, 1]))
        self.assertGreater(lawn.sum(), 100_000)
        self.assertLessEqual(up_angle_degrees(plane_normal(xyz[lawn])), 8.0)
        light_wall = ~grass & (red > 150) & (np.abs(red - green) < 25) & (np.abs(green - blue) < 30)
        self.assertGreater(np.median(xyz[light_wall, 1]), np.median(xyz[lawn, 1]))

    def test_scene_voxelizes_to_observed_surface_blocks(self):
        xyz = self.xyz()
        crop_min = np.percentile(xyz, 2, axis=0)
        crop_max = np.percentile(xyz, 98, axis=0)
        palette = load_palette(ROOT / "packages" / "block-palette" / "palette-v1.json")
        result = voxelize_points(self.points, crop_min=crop_min, crop_max=crop_max, scale_axis="x",
                                 blocks=64, palette=palette, config=VoxelizationConfig())
        self.assertGreaterEqual(len(result.voxels), 3_000)

        # Observed geometry only: recompute occupied cells independently of the voxelizer.
        scaled = (xyz - crop_min) / result.voxel_size
        inside = ((xyz >= crop_min) & (xyz <= crop_max)).all(axis=1)
        dimensions = np.ceil((crop_max - crop_min) / result.voxel_size - 1e-12).astype(np.int64)
        observed = set()
        for offset in (-1e-5, 1e-5):  # tolerate the voxelizer's float32 plane snapping
            cells = np.minimum(np.floor(scaled[inside] + offset).astype(np.int64), dimensions - 1)
            observed.update(map(tuple, cells.tolist()))
        unsupported = [voxel.source_cell for voxel in result.voxels if tuple(voxel.source_cell) not in observed]
        self.assertEqual(unsupported, [])

        # Open remains open: a surface shell occupies a small fraction of its bounding box.
        cells = np.array([voxel.source_cell for voxel in result.voxels])
        box = np.prod(cells.max(axis=0) - cells.min(axis=0) + 1)
        self.assertLess(len(result.voxels) / box, 0.15)
        self.assertEqual(min(voxel.y for voxel in result.voxels), 0)

        colors = np.array([voxel.color_srgb for voxel in result.voxels], dtype=float)
        greenish = (colors[:, 1] > 1.1 * colors[:, 0]) & (colors[:, 1] > 1.1 * colors[:, 2])
        self.assertGreater(greenish.mean(), 0.1)
        self.assertGreaterEqual(len({voxel.palette_id for voxel in result.voxels}), 5)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "voxels.pb.zst"
            write_voxel_contract(path, result, palette)
            message = decode_voxel_contract(path.read_bytes())
        self.assertEqual(len(message.voxels), len(result.voxels))
