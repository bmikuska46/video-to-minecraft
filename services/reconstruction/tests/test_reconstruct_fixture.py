from __future__ import annotations

import importlib.util
import json
import math
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "reconstruct_fixture.py"
SPEC = importlib.util.spec_from_file_location("reconstruct_fixture", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProbeTests(unittest.TestCase):
    def test_probe_accepts_one_valid_video_stream(self):
        payload = {
            "format": {"duration": "15.000000", "format_name": "mov,mp4"},
            "streams": [{"codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "fixture.mp4"
            video.write_bytes(b"fixture")
            completed = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
            with patch.object(MODULE, "run", return_value=completed):
                result = MODULE.probe(video, root / "output")
        self.assertEqual(result["durationSeconds"], 15.0)
        self.assertEqual(len(result["sha256"]), 64)

    def test_probe_rejects_overlong_video(self):
        payload = {
            "format": {"duration": "60.500000", "format_name": "mov,mp4"},
            "streams": [{"codec_type": "video", "width": 1280, "height": 720}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "fixture.mp4"
            video.write_bytes(b"fixture")
            completed = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
            with patch.object(MODULE, "run", return_value=completed):
                with self.assertRaisesRegex(ValueError, "outside"):
                    MODULE.probe(video, root / "output")

    def test_scene_scans_accept_longer_video_than_object_scans(self):
        payload = {
            "format": {"duration": "120.000000", "format_name": "mov,mp4"},
            "streams": [{"codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "fixture.mp4"
            video.write_bytes(b"fixture")
            completed = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
            with patch.object(MODULE, "run", return_value=completed):
                self.assertEqual(MODULE.probe(video, root / "output", scan_type="scene")["durationSeconds"], 120.0)
                with self.assertRaisesRegex(ValueError, "outside"):
                    MODULE.probe(video, root / "output")

    def test_probe_rejects_oversized_file_before_running_ffprobe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "fixture.mp4"
            video.write_bytes(b"too large")
            with patch.object(MODULE, "run") as mocked_run:
                with self.assertRaisesRegex(ValueError, "limit is 4 bytes"):
                    MODULE.probe(video, root / "output", max_video_bytes=4)
            mocked_run.assert_not_called()

    def test_probe_rejects_non_mp4_or_non_h264_media(self):
        cases = [
            (
                {
                    "format": {"duration": "10", "format_name": "matroska,webm"},
                    "streams": [{"codec_type": "video", "codec_name": "h264", "width": 10, "height": 10}],
                },
                "MP4 container",
            ),
            (
                {
                    "format": {"duration": "10", "format_name": "mov,mp4"},
                    "streams": [{"codec_type": "video", "codec_name": "hevc", "width": 10, "height": 10}],
                },
                "H.264",
            ),
        ]
        for payload, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                video = root / "fixture.mp4"
                video.write_bytes(b"fixture")
                completed = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
                with patch.object(MODULE, "run", return_value=completed):
                    with self.assertRaisesRegex(ValueError, message):
                        MODULE.probe(video, root / "output")


class InspectSourceTests(unittest.TestCase):
    def inspect(self, payload, scan_type="object"):
        calls = []

        def fake_run(arguments, log_path, *, capture=False):
            calls.append(arguments)
            return subprocess.CompletedProcess(arguments, 0, json.dumps(payload) if capture else "", "")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "source.mp4"
            video.write_bytes(b"fixture")
            with patch.object(MODULE, "run", side_effect=fake_run):
                source = MODULE.inspect_source(video, root / "output", scan_type=scan_type)
            return source, calls

    def test_rotated_hevc_phone_capture_is_read_directly_and_trimmed(self):
        source, calls = self.inspect({
            "format": {"duration": "60.532933", "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
            "streams": [
                {"codec_type": "video", "codec_name": "hevc", "pix_fmt": "yuv420p10le", "width": 1920,
                 "height": 1080, "side_data_list": [{"rotation": -90}]},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        })
        self.assertEqual([call[0] for call in calls], ["ffprobe"])
        self.assertEqual(source["stream"]["codec_name"], "hevc")
        self.assertEqual((source["displayWidth"], source["displayHeight"]), (1080, 1920))
        self.assertEqual(source["usableDurationSeconds"], 60.0)
        self.assertTrue(source["trimmed"])
        self.assertEqual(len(source["sha256"]), 64)

    def test_scene_capture_keeps_its_longer_duration(self):
        valid = {
            "format": {"duration": "40.0", "format_name": "mov,mp4"},
            "streams": [{"codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720}],
        }
        source, _ = self.inspect(valid, "scene")
        self.assertEqual(source["usableDurationSeconds"], 40.0)
        self.assertFalse(source["trimmed"])
        source, _ = self.inspect({**valid, "format": {**valid["format"], "duration": "305.0"}}, "scene")
        self.assertEqual(source["usableDurationSeconds"], 300.0)

    def test_rejects_unusable_sources(self):
        for payload, message in (
            ({"format": {"duration": "0.4", "format_name": "mov,mp4"},
              "streams": [{"codec_type": "video", "codec_name": "hevc", "width": 10, "height": 10}]}, "shorter"),
            ({"format": {"duration": "5", "format_name": "mov,mp4"},
              "streams": [{"codec_type": "audio", "codec_name": "aac"}]}, "no video stream"),
            ({"format": {"duration": "5", "format_name": "mov,mp4"},
              "streams": [{"codec_type": "video", "codec_name": "hevc"}]}, "dimensions"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.inspect(payload)

    def test_one_decode_writes_candidates_and_thumbnails(self):
        command = MODULE.extraction_command(Path("in.mp4"), Path("c"), Path("t"), 15.0, use_gpu=True)
        self.assertEqual(command[command.index("-t") + 1], "15.0")
        self.assertLess(command.index("-t"), command.index("-i"))
        self.assertEqual(command[command.index("-hwaccel") + 1], "cuda")
        self.assertEqual(command.count("-map"), 2)
        self.assertIn("showinfo,split=2", command[command.index("-filter_complex") + 1])
        self.assertEqual(command[-1], "t/candidate-%04d.pgm")
        self.assertNotIn("-hwaccel", MODULE.extraction_command(Path("in.mp4"), Path("c"), Path("t"), 1.0, False))


class FrameScoringTests(unittest.TestCase):
    @staticmethod
    def write_pgm(path, width, height, pixels):
        path.write_bytes(f"P5\n{width} {height}\n255\n".encode() + bytes(pixels))

    def test_numpy_scores_match_the_reference_definition(self):
        import random

        width, height = 23, 17
        generator = random.Random(4)
        left = [generator.randrange(256) for _ in range(width * height)]
        right = [min(255, value + generator.randrange(3)) for value in left]
        mean = sum(left) / len(left)
        clipped = sum(value <= 5 or value >= 250 for value in left) / len(left)
        laplacians = [
            4 * left[y * width + x] - left[y * width + x - 1] - left[y * width + x + 1]
            - left[(y - 1) * width + x] - left[(y + 1) * width + x]
            for y in range(1, height - 1) for x in range(1, width - 1)
        ]
        average = sum(laplacians) / len(laplacians)
        variance = sum(value * value for value in laplacians) / len(laplacians) - average ** 2
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_pgm(root / "left.pgm", width, height, left)
            self.write_pgm(root / "right.pgm", width, height, right)
            quality = MODULE.frame_quality(root / "left.pgm")
            difference = MODULE.frame_difference(root / "left.pgm", root / "right.pgm")
        self.assertAlmostEqual(quality["brightness"], mean)
        self.assertAlmostEqual(quality["clippedRatio"], clipped)
        self.assertAlmostEqual(quality["sharpness"], variance / max(mean, 16.0))
        self.assertAlmostEqual(difference, sum(abs(a - b) for a, b in zip(left, right)) / (255 * len(left)))


class KeyframeSelectionTests(unittest.TestCase):
    def test_scene_scans_keep_more_frames_than_object_scans(self):
        count = 800
        paths = [Path(f"candidate-{index:04d}.jpg") for index in range(count)]
        timestamps = [index / 8 for index in range(count)]
        with patch.object(MODULE, "frame_quality", return_value={"usable": True, "sharpness": 1.0}), \
                patch.object(MODULE, "frame_difference", return_value=1.0):
            selected_object, _ = MODULE.select_keyframes(paths, paths, timestamps, 500)
            selected_scene, _ = MODULE.select_keyframes(
                paths, paths, timestamps, 500, MODULE.MAX_SELECTED_FRAMES["scene"]
            )
        self.assertEqual(len(selected_object), 240)
        self.assertEqual(len(selected_scene), 500)


class SparseModelSelectionTests(unittest.TestCase):
    def test_picks_the_model_with_most_registered_images_not_the_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            sparse = Path(temporary)
            for name, count in (("0", 4), ("1", 159), ("2", 11)):
                (sparse / name).mkdir()
                (sparse / name / "images.bin").write_bytes(struct.pack("<Q", count) + b"rest")
            (sparse / "3").mkdir()  # an empty model directory is ignored
            self.assertEqual(MODULE.largest_sparse_model(sparse).name, "1")

    def test_rejects_a_mapper_run_without_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "no sparse model"):
                MODULE.largest_sparse_model(Path(temporary))

    def test_accepts_a_model_written_directly_into_the_output(self):
        # The extension pass (mapper --input_path) writes its model straight into
        # the output directory instead of a numbered subdirectory.
        with tempfile.TemporaryDirectory() as temporary:
            sparse = Path(temporary)
            (sparse / "images.bin").write_bytes(struct.pack("<Q", 239) + b"rest")
            self.assertEqual(MODULE.largest_sparse_model(sparse), sparse)
            self.assertEqual(MODULE.sparse_image_count(sparse), 239)

    def test_counts_no_images_for_a_missing_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(MODULE.sparse_image_count(Path(temporary) / "missing"), 0)


class PatchMatchConfigTests(unittest.TestCase):
    def test_limits_automatic_source_lists_and_keeps_reference_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "patch-match.cfg"
            config.write_text("frame-0001.jpg\n__auto__, 20\nframe-0002.jpg\n__auto__, 20\n")
            MODULE.limit_patch_match_sources(config, 10)
            self.assertEqual(config.read_text(), "frame-0001.jpg\n__auto__, 10\nframe-0002.jpg\n__auto__, 10\n")

    def test_rejects_config_without_automatic_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "patch-match.cfg"
            config.write_text("frame-0001.jpg\nframe-0002.jpg, frame-0003.jpg\n")
            with self.assertRaisesRegex(ValueError, "no automatic source"):
                MODULE.limit_patch_match_sources(config, 10)


class MetricsTests(unittest.TestCase):
    def test_model_analyzer_metrics_include_quality_decision(self):
        output = """Cameras: 1
Images: 48
Registered images: 48
Points: 12,345
Observations: 67,890
Mean track length: 5.499
Mean observations per image: 1414.375
Mean reprojection error: 0.812px
"""
        metrics = MODULE.parse_model_analyzer(output, selected_frames=60)
        self.assertEqual(metrics["registeredFrames"], 48)
        self.assertEqual(metrics["points3D"], 12345)
        self.assertEqual(metrics["registrationRatio"], 0.8)
        self.assertTrue(metrics["qualityDecision"]["usable"])

    def test_model_analyzer_metrics_parse_colmap_4_glog_output(self):
        output = """I20260927 23:50:55.345273   504 model.cc:440] Rigs: 1
I20260927 23:50:55.345486   504 model.cc:441] Cameras: 1
I20260927 23:50:55.345497   504 model.cc:442] Frames: 49
I20260927 23:50:55.345504   504 model.cc:443] Registered frames: 49
I20260927 23:50:55.345512   504 model.cc:445] Images: 49
I20260927 23:50:55.345520   504 model.cc:446] Registered images: 49
I20260927 23:50:55.345525   504 model.cc:448] Points: 1964
I20260927 23:50:55.345532   504 model.cc:449] Observations: 13202
I20260927 23:50:55.345541   504 model.cc:451] Mean track length: 6.721996
I20260927 23:50:55.345551   504 model.cc:453] Mean observations per image: 269.428571
I20260927 23:50:55.345559   504 model.cc:456] Mean reprojection error: 0.641976px
"""
        metrics = MODULE.parse_model_analyzer(output, selected_frames=60)
        self.assertEqual(metrics["registeredFrames"], 49)
        self.assertEqual(metrics["points3D"], 1964)
        self.assertEqual(metrics["observations"], 13202)
        self.assertAlmostEqual(metrics["meanReprojectionErrorPixels"], 0.641976)

    def test_model_analyzer_metrics_report_failed_thresholds(self):
        output = """Registered images: 10
Points: 100
Mean reprojection error: 2.5px
"""
        metrics = MODULE.parse_model_analyzer(output, selected_frames=30)
        self.assertFalse(metrics["qualityDecision"]["usable"])
        self.assertEqual(
            metrics["qualityDecision"]["reasons"],
            ["TOO_FEW_REGISTERED_FRAMES", "LOW_REGISTRATION_RATIO", "HIGH_REPROJECTION_ERROR"],
        )

    def test_reads_median_reprojection_error_from_text_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            points = Path(temporary) / "points3D.txt"
            points.write_text(
                "# 3D point list\n"
                "1 0 0 0 255 0 0 0.25 1 2\n"
                "2 0 0 0 0 255 0 1.50 1 4\n"
                "3 0 0 0 0 0 255 0.75 2 6\n"
            )
            self.assertEqual(MODULE.read_median_reprojection_error(points), 0.75)

    def test_quality_decision_prefers_median_reprojection_error(self):
        metrics = {
            "registeredFrames": 30,
            "registrationRatio": 1.0,
            "meanReprojectionErrorPixels": 3.0,
            "medianReprojectionErrorPixels": 1.0,
        }
        self.assertTrue(MODULE.quality_decision(metrics)["usable"])

    def test_reads_binary_ply_vertex_count_from_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            ply = Path(temporary) / "fused.ply"
            ply.write_bytes(
                b"ply\nformat binary_little_endian 1.0\nelement vertex 4321\n"
                b"property float x\nend_header\n\x00\x01"
            )
            self.assertEqual(MODULE.ply_vertex_count(ply), 4321)

    def test_sparse_geometry_measures_baseline_and_camera_up(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images.txt").write_text(
                "# image records alternate with observations\n"
                "1 1 0 0 0 0 0 0 1 frame-0001.jpg\n\n"
                "2 1 0 0 0 -1 0 0 1 frame-0002.jpg\n0 0 -1\n"
                "3 1 0 0 0 -2 0 0 1 frame-0003.jpg\n\n"
            )
            (root / "points3D.txt").write_text(
                "1 0 0 0 255 0 0 0.2 1 0\n"
                "2 0 3 4 255 0 0 0.2 1 0\n"
            )
            metrics = MODULE.sparse_geometry_metrics(root / "images.txt", root / "points3D.txt")
        self.assertEqual(metrics["cameraBaseline"], 2.0)
        self.assertEqual(metrics["sceneDiagonal"], 5.0)
        self.assertEqual(metrics["estimatedUpVector"], [0.0, -1.0, 0.0])
        self.assertFalse(metrics["trajectoryDisconnected"])

    def trajectory(self, positions):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images.txt").write_text("".join(
                f"{index} 1 0 0 0 {-x} 0 0 1 frame-{frame:04d}.jpg\n\n"
                for index, (frame, x) in enumerate(positions, start=1)
            ))
            (root / "points3D.txt").write_text("1 0 0 0 255 0 0 0.2 1 0\n2 0 3 4 255 0 0 0.2 1 0\n")
            return MODULE.sparse_geometry_metrics(root / "images.txt", root / "points3D.txt")

    def test_unregistered_frames_do_not_look_like_a_trajectory_break(self):
        # Frames 6-15 failed to register; the camera kept moving 1 unit per frame.
        positions = [(frame, float(frame)) for frame in (*range(1, 6), *range(16, 21))]
        metrics = self.trajectory(positions)
        self.assertAlmostEqual(metrics["maximumTrajectoryStepRatio"], 1.0)
        self.assertFalse(metrics["trajectoryDisconnected"])

    def test_a_jump_between_adjacent_frames_is_still_a_trajectory_break(self):
        positions = [(frame, float(frame) + (20.0 if frame > 5 else 0.0)) for frame in range(1, 11)]
        metrics = self.trajectory(positions)
        self.assertTrue(metrics["trajectoryDisconnected"])

    def test_a_single_misregistered_frame_is_found_as_a_trajectory_outlier(self):
        names = [f"frame-{frame:04d}.jpg" for frame in range(1, 41)]
        centers = [[float(frame), 0.0, 0.0] for frame in range(1, 41)]
        centers[19] = [20.0, 600.0, 0.0]  # frame 20 registered far away, as on the room capture
        self.assertEqual(MODULE.trajectory_outlier_frames(names, centers), ["frame-0020.jpg"])
        self.assertEqual(MODULE.trajectory_outlier_frames(names[:10], [[float(i), 0.0, 0.0] for i in range(10)]), [])

    def test_a_real_break_between_two_walks_is_not_explained_away(self):
        names = [f"frame-{frame:04d}.jpg" for frame in range(1, 41)]
        centers = [[float(frame) + (500.0 if frame > 20 else 0.0), 0.0, 0.0] for frame in range(1, 41)]
        self.assertIsNone(MODULE.trajectory_outlier_frames(names, centers))

    def test_up_vector_ignores_camera_pitch_when_the_camera_turns(self):
        rights, ups = [], []
        pitch = math.radians(43)  # looking down at a floor, as on the reference room scan
        for yaw in [math.radians(value) for value in range(0, 360, 15)]:
            # Right-handed camera frame: x right, y down (image), z forward.
            right = [-math.cos(yaw), 0.0, math.sin(yaw)]
            forward = [math.sin(yaw) * math.cos(pitch), -math.sin(pitch), math.cos(yaw) * math.cos(pitch)]
            down = [forward[1] * right[2] - forward[2] * right[1],
                    forward[2] * right[0] - forward[0] * right[2],
                    forward[0] * right[1] - forward[1] * right[0]]
            rights.append(right)
            ups.append([-value for value in down])
        up, method = MODULE.estimate_up_vector(rights, ups)
        self.assertEqual(method, "cameraRightVectors")
        for actual, expected in zip(up, [0.0, 1.0, 0.0]):
            self.assertAlmostEqual(actual, expected, places=9)
        mean_up_y = sum(value[1] for value in ups) / len(ups)
        self.assertAlmostEqual(mean_up_y, math.cos(pitch), places=9)  # the old estimate was 43 degrees off

    def test_up_vector_falls_back_when_the_camera_never_turns(self):
        pitch = math.radians(20)
        rights = [[1.0, 0.0, 0.0]] * 5
        ups = [[0.0, math.cos(pitch), math.sin(pitch)]] * 5
        up, method = MODULE.estimate_up_vector(rights, ups)
        self.assertEqual(method, "cameraUpOrthogonalToRight")
        for actual, expected in zip(up, ups[0]):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_quality_gate_rejects_insufficient_baseline_and_split_trajectory(self):
        metrics = {
            "registeredFrames": 30,
            "registrationRatio": 1.0,
            "meanReprojectionErrorPixels": 0.5,
            "baselineToSceneDiagonal": 0.01,
            "trajectoryDisconnected": True,
        }
        self.assertEqual(
            MODULE.quality_decision(metrics)["reasons"],
            ["INSUFFICIENT_BASELINE", "DISCONNECTED_CAMERA_TRAJECTORY"],
        )


class KeyframeTests(unittest.TestCase):
    def test_pgm_reader_does_not_consume_whitespace_valued_first_pixel(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.pgm"
            path.write_bytes(b"P5\n2 2\n255\n" + bytes([10, 20, 30, 40]))
            self.assertEqual(MODULE.read_pgm(path), (2, 2, bytes([10, 20, 30, 40])))

    def test_showinfo_timestamp_parser_preserves_fractional_times(self):
        log = "n: 0 pts: 0 pts_time:0\nn: 1 pts: 1 pts_time:0.125 duration:1\n"
        self.assertEqual(MODULE.parse_showinfo_timestamps(log), [0.0, 0.125])


class ManifestTests(unittest.TestCase):
    def test_manifest_records_completed_stage_and_settings_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            manifest = MODULE.StageManifest(path, {"frameRate": 4.0})
            manifest.start("VALIDATION")
            manifest.complete(metrics={"durationSeconds": 15.0}, artifacts=["input.json"])
            manifest.finish(source={"sha256": "a" * 64}, metrics={"selectedFrames": 60})

            payload = json.loads(path.read_text())
            self.assertEqual(payload["status"], "COMPLETED")
            self.assertEqual(len(payload["settingsHash"]), 64)
            self.assertEqual(payload["stages"][0]["status"], "COMPLETED")
            self.assertEqual(payload["stages"][0]["metrics"]["durationSeconds"], 15.0)

    def test_manifest_preserves_failed_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            manifest = MODULE.StageManifest(path, {})
            manifest.start("SPARSE_RECONSTRUCTION")
            manifest.fail(RuntimeError("no sparse model"))

            payload = json.loads(path.read_text())
            self.assertEqual(payload["status"], "FAILED")
            self.assertEqual(payload["stages"][0]["status"], "FAILED")
            self.assertEqual(payload["failure"]["type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
