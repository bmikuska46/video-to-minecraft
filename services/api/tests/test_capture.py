from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from pydantic import ValidationError

from app.capture import CaptureMetadata


VALID_CAPTURE = {
    "schemaVersion": 1,
    "monotonicRecordingStartNs": 4_000_000_000,
    "wallClockTimestamp": "2026-08-18T10:00:00.123456+02:00",
    "orientation": "landscape-left",
    "cameraFormat": "1920x1080-30-wide",
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "codec": "h264",
    "physicalLens": "wide-angle-camera",
    "appVersion": "0.1.0",
    "nativeScannerModuleVersion": "0.1.0",
}


class CaptureMetadataTests(unittest.TestCase):
    def test_schema_round_trips_using_canonical_camel_case_names(self) -> None:
        metadata = CaptureMetadata.model_validate(VALID_CAPTURE)

        self.assertEqual(metadata.width, 1920)
        self.assertEqual(
            metadata.model_dump(mode="json", by_alias=True, exclude_none=True),
            VALID_CAPTURE,
        )

    def test_schema_rejects_unknown_fields_naive_time_and_wrong_codec(self) -> None:
        invalid_documents = [
            {**VALID_CAPTURE, "unexpected": True},
            {**VALID_CAPTURE, "wallClockTimestamp": "2026-08-18T10:00:00"},
            {**VALID_CAPTURE, "codec": "hevc"},
        ]
        for document in invalid_documents:
            with self.subTest(document=document), self.assertRaises(ValidationError):
                CaptureMetadata.model_validate(document)

    def test_monotonic_timestamp_converts_without_timezone_or_float_loss(self) -> None:
        metadata = CaptureMetadata.model_validate(VALID_CAPTURE)
        converted = metadata.wall_clock_for_monotonic_ns(5_234_567_890)

        self.assertEqual(
            converted,
            datetime.fromisoformat(VALID_CAPTURE["wallClockTimestamp"])
            + timedelta(microseconds=1_234_568),
        )
        self.assertEqual(converted.utcoffset(), timedelta(hours=2))

    def test_timestamp_before_recording_start_is_rejected(self) -> None:
        metadata = CaptureMetadata.model_validate(VALID_CAPTURE)
        with self.assertRaisesRegex(ValueError, "precedes"):
            metadata.wall_clock_for_monotonic_ns(3_999_999_999)


if __name__ == "__main__":
    unittest.main()
