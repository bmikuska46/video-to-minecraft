from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .schemas import to_camel


class CaptureOrientation(StrEnum):
    PORTRAIT = "portrait"
    PORTRAIT_UPSIDE_DOWN = "portrait-upside-down"
    LANDSCAPE_LEFT = "landscape-left"
    LANDSCAPE_RIGHT = "landscape-right"


class CaptureMetadata(BaseModel):
    """Version 1 of the immutable capture.json sidecar."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")

    schema_version: Literal[1]
    monotonic_recording_start_ns: Annotated[int, Field(ge=0)]
    wall_clock_timestamp: datetime
    orientation: CaptureOrientation
    camera_format: Annotated[str, Field(min_length=1, max_length=160)]
    width: Annotated[int, Field(gt=0, le=16_384)]
    height: Annotated[int, Field(gt=0, le=16_384)]
    fps: Annotated[float, Field(gt=0, le=240)]
    codec: Literal["h264"]
    physical_lens: Annotated[str, Field(min_length=1, max_length=160)] | None = None
    app_version: Annotated[str, Field(min_length=1, max_length=80)]
    native_scanner_module_version: Annotated[str, Field(min_length=1, max_length=80)]

    @field_validator("wall_clock_timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("wallClockTimestamp must include a UTC offset")
        return value

    def wall_clock_for_monotonic_ns(self, timestamp_ns: int) -> datetime:
        """Convert a monotonic capture timestamp onto the sidecar wall clock.

        Capture streams use monotonic nanoseconds so clock corrections cannot
        reorder samples. The wall clock is only reconstructed at API/reporting
        boundaries. ``timedelta`` has microsecond resolution, so sub-microsecond
        values are rounded to the nearest microsecond instead of silently
        accumulating floating-point error.
        """
        if timestamp_ns < self.monotonic_recording_start_ns:
            raise ValueError("timestamp precedes monotonic recording start")
        delta_ns = timestamp_ns - self.monotonic_recording_start_ns
        delta_us, remainder_ns = divmod(delta_ns, 1_000)
        if remainder_ns >= 500:
            delta_us += 1
        return self.wall_clock_timestamp + timedelta(microseconds=delta_us)
