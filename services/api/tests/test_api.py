from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import Export, PipelineStageMetric, Reconstruction, Scan
from app.states import ExportStatus, ScanStatus
from app.storage import ObjectNotFoundError, SignedUrl, StoredObject


SHA256 = "a" * 64
CAPTURE_METADATA = {
    "schemaVersion": 1,
    "monotonicRecordingStartNs": 123456789,
    "wallClockTimestamp": "2026-08-18T10:00:00+02:00",
    "orientation": "landscape-left",
    "cameraFormat": "1080p-30-wide",
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "codec": "h264",
    "physicalLens": "wide-angle-camera",
    "appVersion": "0.1.0",
    "nativeScannerModuleVersion": "0.1.0",
}


class FakeStorage:
    def __init__(self) -> None:
        self.objects: dict[str, StoredObject] = {}
        self.contents: dict[str, bytes] = {}

    def sign_put(self, key, *, content_type, byte_size, sha256):
        return SignedUrl(
            f"https://storage.test/upload/{key}",
            datetime.now(timezone.utc) + timedelta(minutes=15),
            {"content-type": content_type, "x-amz-meta-sha256": sha256 or ""},
        )

    def sign_get(self, key):
        return SignedUrl(
            f"https://storage.test/download/{key}",
            datetime.now(timezone.utc) + timedelta(minutes=15),
            {},
        )

    def head(self, key):
        try:
            return self.objects[key]
        except KeyError as error:
            raise ObjectNotFoundError(key) from error

    def read_bytes(self, key):
        try:
            return self.contents[key]
        except KeyError as error:
            raise ObjectNotFoundError(key) from error


class FakeDispatcher:
    def __init__(self) -> None:
        self.reconstruction_ids: list[str] = []
        self.export_ids: list[str] = []

    def enqueue_reconstruction(self, scan_id):
        self.reconstruction_ids.append(scan_id)

    def enqueue_export(self, export_id):
        self.export_ids.append(export_id)


class APITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        database = Path(self.temporary.name) / "api.sqlite3"
        self.settings = Settings(
            database_url=f"sqlite:///{database}",
            max_video_bytes=1_000,
            retention_days=3,
            max_target_blocks=500,
        )
        self.storage = FakeStorage()
        self.dispatcher = FakeDispatcher()
        self.app = create_app(
            settings=self.settings,
            storage=self.storage,
            dispatcher=self.dispatcher,
        )
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def create_scan(self) -> str:
        response = self.client.post(
            "/v1/scans",
            json={
                "devicePlatform": "ios",
                "deviceModel": "iPhone fixture",
                "captureDurationMs": 15_000,
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    def prepare_upload(self, scan_id: str, *, byte_size: int = 100) -> str:
        response = self.client.post(
            f"/v1/scans/{scan_id}/upload-url",
            json={
                "kind": "video",
                "contentType": "video/mp4",
                "byteSize": byte_size,
                "sha256": SHA256,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        key = response.json()["objectKey"]
        self.storage.objects[key] = StoredObject(byte_size=byte_size, sha256=SHA256)
        metadata_key = f"scans/{scan_id}/source/capture.json"
        metadata_bytes = json.dumps(CAPTURE_METADATA).encode()
        self.storage.objects[metadata_key] = StoredObject(
            byte_size=len(metadata_bytes), sha256="b" * 64
        )
        self.storage.contents[metadata_key] = metadata_bytes
        return key

    def complete_upload(self, scan_id: str, key: str, *, byte_size: int = 100):
        return self.client.post(
            f"/v1/scans/{scan_id}/upload-complete",
            json={
                "videoObjectKey": key,
                "metadataObjectKey": f"scans/{scan_id}/source/capture.json",
                "sha256": SHA256,
                "byteSize": byte_size,
            },
        )

    def seed_preview(self, scan_id: str) -> None:
        with self.app.state.session_factory() as session:
            scan = session.get(Scan, scan_id)
            scan.status = ScanStatus.PREVIEW_READY
            scan.progress = 1.0
            session.add(
                Reconstruction(
                    scan_id=scan_id,
                    pipeline_version="fixture-v1",
                    settings_hash="b" * 64,
                    registered_frame_count=48,
                    selected_frame_count=60,
                    median_reprojection_error=0.8,
                    confidence_decision="usable",
                    warnings=["GLASS_REGION_LOW_CONFIDENCE"],
                    preview_glb_key=f"scans/{scan_id}/preview/preview.glb",
                    canonical_ply_key=f"scans/{scan_id}/reconstruction/cloud.ply",
                    original_bounds={"min": [0.0, 0.0, 0.0], "max": [2.0, 1.0, 0.5]},
                )
            )
            session.commit()

    def test_processing_mode_defaults_to_detailed_and_accepts_fast(self):
        scan_id = self.create_scan()
        self.assertEqual(self.client.get(f"/v1/scans/{scan_id}").json()["processingMode"], "detailed")
        payload = {"devicePlatform": "android", "deviceModel": "Pixel fixture", "captureDurationMs": 9_000}
        fast = self.client.post("/v1/scans", json={**payload, "processingMode": "fast"})
        self.assertEqual(fast.status_code, 201, fast.text)
        self.assertEqual(self.client.get(f"/v1/scans/{fast.json()['id']}").json()["processingMode"], "fast")
        invalid = self.client.post("/v1/scans", json={**payload, "processingMode": "ultra"})
        self.assertEqual(invalid.status_code, 422)

    def test_scene_scans_allow_longer_captures_and_report_their_type(self):
        scan_id = self.create_scan()
        self.assertEqual(self.client.get(f"/v1/scans/{scan_id}").json()["scanType"], "object")
        payload = {"devicePlatform": "android", "deviceModel": "Pixel fixture", "captureDurationMs": 90_000}
        too_long_object = self.client.post("/v1/scans", json=payload)
        self.assertEqual(too_long_object.status_code, 422)
        self.assertEqual(too_long_object.json()["detail"]["code"], "CAPTURE_TOO_LONG")
        scene = self.client.post("/v1/scans", json={**payload, "scanType": "scene"})
        self.assertEqual(scene.status_code, 201, scene.text)
        self.assertEqual(scene.json()["uploadConstraints"]["maxCaptureDurationMs"], 300_250)
        self.assertEqual(self.client.get(f"/v1/scans/{scene.json()['id']}").json()["scanType"], "scene")
        too_long_scene = self.client.post("/v1/scans", json={**payload, "scanType": "scene", "captureDurationMs": 300_251})
        self.assertEqual(too_long_scene.status_code, 422)
        self.assertEqual(self.client.post("/v1/scans", json={**payload, "scanType": "house"}).status_code, 422)

    def test_create_scan_returns_constraints_and_rejects_long_capture(self):
        scan_id = self.create_scan()
        response = self.client.get(f"/v1/scans/{scan_id}")
        self.assertEqual(response.json()["status"], "CREATED")
        self.assertEqual(response.json()["captureDurationMs"], 15_000)

        rejected = self.client.post(
            "/v1/scans",
            json={
                "devicePlatform": "android",
                "deviceModel": "Pixel fixture",
                "captureDurationMs": 60_251,
            },
        )
        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(rejected.json()["detail"]["code"], "CAPTURE_TOO_LONG")

    def test_health_checks_liveness_and_dependencies(self):
        self.assertEqual(self.client.get("/health/live").json(), {"status": "ok"})
        response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["checks"],
            {"postgresql": "ok", "redis": "ok", "storage": "ok"},
        )
        self.storage.ready = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
        unavailable = self.client.get("/health/ready")
        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(unavailable.json()["checks"]["storage"], "unavailable")

    def test_upload_uses_canonical_key_verifies_object_and_queues_once_per_call(self):
        scan_id = self.create_scan()
        key = self.prepare_upload(scan_id)
        self.assertEqual(key, f"scans/{scan_id}/source/video.mp4")

        completed = self.complete_upload(scan_id, key)
        self.assertEqual(completed.status_code, 200, completed.text)
        self.assertEqual(completed.json()["status"], "QUEUED")
        self.assertEqual(self.dispatcher.reconstruction_ids, [scan_id])

        replay = self.complete_upload(scan_id, key)
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(replay.headers["Idempotent-Replay"], "true")
        self.assertEqual(self.dispatcher.reconstruction_ids, [scan_id, scan_id])
        with self.app.state.session_factory() as session:
            metrics = session.query(PipelineStageMetric).all()
            self.assertEqual(len(metrics), 1)
            self.assertEqual(metrics[0].metrics, {"uploadBytes": 100})

    def test_upload_completion_rejects_missing_or_mismatched_object(self):
        scan_id = self.create_scan()
        key = self.prepare_upload(scan_id)
        self.storage.objects[key] = StoredObject(byte_size=100, sha256="b" * 64)
        response = self.complete_upload(scan_id, key)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["code"], "CHECKSUM_MISMATCH")

        wrong_key = self.complete_upload(scan_id, "scans/other/source/video.mp4")
        self.assertEqual(wrong_key.status_code, 422)
        self.assertEqual(wrong_key.json()["detail"]["code"], "INVALID_OBJECT_KEY")

    def test_upload_completion_requires_metadata_sidecar(self):
        scan_id = self.create_scan()
        key = self.prepare_upload(scan_id)
        metadata_key = f"scans/{scan_id}/source/capture.json"
        del self.storage.objects[metadata_key]
        del self.storage.contents[metadata_key]
        response = self.client.post(
            f"/v1/scans/{scan_id}/upload-complete",
            json={
                "videoObjectKey": key,
                "metadataObjectKey": metadata_key,
                "sha256": SHA256,
                "byteSize": 100,
            },
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["code"], "METADATA_UPLOAD_NOT_FOUND")

    def test_upload_urls_cover_all_canonical_capture_sidecars(self):
        scan_id = self.create_scan()
        cases = [
            ("metadata", "application/json", "capture.json"),
            ("imu", "application/zstd", "imu.jsonl.zst"),
            ("ar_frames", "application/octet-stream", "ar-frames.pb.zst"),
        ]
        for kind, content_type, filename in cases:
            response = self.client.post(
                f"/v1/scans/{scan_id}/upload-url",
                json={
                    "kind": kind,
                    "contentType": content_type,
                    "byteSize": 100,
                    "sha256": "b" * 64,
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["objectKey"], f"scans/{scan_id}/source/{filename}")

    def test_upload_completion_rejects_invalid_capture_metadata(self):
        scan_id = self.create_scan()
        key = self.prepare_upload(scan_id)
        metadata_key = f"scans/{scan_id}/source/capture.json"
        self.storage.contents[metadata_key] = b'{"schemaVersion": 2}'
        response = self.complete_upload(scan_id, key)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["code"], "INVALID_CAPTURE_METADATA")

    def test_preview_crop_export_and_download_contract(self):
        scan_id = self.create_scan()
        self.seed_preview(scan_id)

        preview = self.client.get(f"/v1/scans/{scan_id}/preview")
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertEqual(preview.json()["registeredFrames"], 48)
        self.assertIn("/preview/preview.glb", preview.json()["previewUrl"])
        self.assertEqual(preview.json()["scanType"], "object")

        transform = [1.0, 0.0, 0.0, 0.0,
                     0.0, 1.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0,
                     0.0, 0.0, 0.0, 1.0]
        crop = {"min": [0.1, 0.0, 0.0], "max": [1.7, 0.9, 0.3]}
        updated = self.client.patch(
            f"/v1/scans/{scan_id}/reconstruction",
            json={"transform": transform, "crop": crop},
        )
        self.assertEqual(updated.status_code, 200, updated.text)

        estimate = self.client.post(
            f"/v1/scans/{scan_id}/exports/estimate",
            json={"scale": {"axis": "x", "blocks": 120}, "palette": "geometry-safe-v1"},
        )
        self.assertEqual(estimate.status_code, 200, estimate.text)
        self.assertAlmostEqual(estimate.json()["voxelSize"], 1.6 / 120)
        self.assertEqual(estimate.json()["voxelDimensions"], [120, 68, 23])
        self.assertEqual(estimate.json()["blockCountEstimate"], 187_680)
        self.assertTrue(estimate.json()["estimateIsUpperBound"])
        self.assertFalse(estimate.json()["requiresConfirmation"])

        created = self.client.post(
            f"/v1/scans/{scan_id}/exports",
            json={"scale": {"axis": "x", "blocks": 120}, "palette": "geometry-safe-v1"},
        )
        self.assertEqual(created.status_code, 202, created.text)
        payload = created.json()
        self.assertAlmostEqual(payload["voxelSize"], 1.6 / 120)
        self.assertEqual(self.dispatcher.export_ids, [payload["id"]])

        pending = self.client.get(f"/v1/exports/{payload['id']}/download-url")
        self.assertEqual(pending.status_code, 409)
        with self.app.state.session_factory() as session:
            export = session.get(Export, payload["id"])
            export.status = ExportStatus.READY
            export.world_zip_key = f"exports/{export.id}/world.zip"
            session.commit()
        download = self.client.get(f"/v1/exports/{payload['id']}/download-url")
        self.assertEqual(download.status_code, 200, download.text)
        self.assertTrue(download.json()["url"].endswith("/world.zip"))

    def test_crop_bounds_and_export_dimension_are_validated(self):
        scan_id = self.create_scan()
        self.seed_preview(scan_id)
        invalid = self.client.patch(
            f"/v1/scans/{scan_id}/reconstruction",
            json={"transform": [1.0] * 16, "crop": {"min": [1, 0, 0], "max": [0, 1, 1]}},
        )
        self.assertEqual(invalid.status_code, 422)

        reflection = [-1.0, 0.0, 0.0, 2.0,
                      0.0, 1.0, 0.0, 0.0,
                      0.0, 0.0, 1.0, 0.0,
                      0.0, 0.0, 0.0, 1.0]
        invalid_transform = self.client.patch(
            f"/v1/scans/{scan_id}/reconstruction",
            json={"transform": reflection, "crop": {"min": [0, 0, 0], "max": [2, 1, 0.5]}},
        )
        self.assertEqual(invalid_transform.status_code, 422)
        self.assertEqual(
            invalid_transform.json()["detail"]["code"], "INVALID_RECONSTRUCTION_SELECTION"
        )

        outside = self.client.patch(
            f"/v1/scans/{scan_id}/reconstruction",
            json={
                "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                "crop": {"min": [0, 0, 0], "max": [2.1, 1, 0.5]},
            },
        )
        self.assertEqual(outside.status_code, 422)

    def test_rotated_reconstruction_bounds_are_used_for_crop_validation(self):
        scan_id = self.create_scan()
        self.seed_preview(scan_id)
        rotate_z_90 = [0.0, -1.0, 0.0, 0.0,
                       1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0,
                       0.0, 0.0, 0.0, 1.0]
        accepted = self.client.patch(
            f"/v1/scans/{scan_id}/reconstruction",
            json={
                "transform": rotate_z_90,
                "crop": {"min": [-1.0, 0.0, 0.0], "max": [0.0, 2.0, 0.5]},
            },
        )
        self.assertEqual(accepted.status_code, 200, accepted.text)

    def test_large_export_requires_confirmation_and_hard_limit_is_rejected(self):
        scan_id = self.create_scan()
        self.seed_preview(scan_id)
        transform = [1.0, 0.0, 0.0, 0.0,
                     0.0, 1.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0,
                     0.0, 0.0, 0.0, 1.0]
        saved = self.client.patch(
            f"/v1/scans/{scan_id}/reconstruction",
            json={
                "transform": transform,
                "crop": {"min": [0.1, 0.0, 0.0], "max": [1.7, 0.9, 0.3]},
            },
        )
        self.assertEqual(saved.status_code, 200, saved.text)

        self.settings.block_count_warning_threshold = 100_000
        estimate = self.client.post(
            f"/v1/scans/{scan_id}/exports/estimate",
            json={"scale": {"axis": "x", "blocks": 120}},
        )
        self.assertEqual(estimate.status_code, 200, estimate.text)
        self.assertTrue(estimate.json()["requiresConfirmation"])

        unconfirmed = self.client.post(
            f"/v1/scans/{scan_id}/exports",
            json={"scale": {"axis": "x", "blocks": 120}},
        )
        self.assertEqual(unconfirmed.status_code, 409)
        self.assertEqual(
            unconfirmed.json()["detail"]["code"], "EXPORT_CONFIRMATION_REQUIRED"
        )
        confirmed = self.client.post(
            f"/v1/scans/{scan_id}/exports",
            json={
                "scale": {"axis": "x", "blocks": 120},
                "confirmLargeExport": True,
            },
        )
        self.assertEqual(confirmed.status_code, 202, confirmed.text)

        self.settings.block_count_hard_limit = 150_000
        rejected = self.client.post(
            f"/v1/scans/{scan_id}/exports/estimate",
            json={"scale": {"axis": "x", "blocks": 120}},
        )
        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(rejected.json()["detail"]["code"], "OUTPUT_TOO_LARGE")


if __name__ == "__main__":
    unittest.main()
