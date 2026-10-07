from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .states import ExportStatus, ScanStatus


def to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class APIModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, from_attributes=True)


class DevicePlatform(StrEnum):
    IOS = "ios"
    ANDROID = "android"


class ProcessingMode(StrEnum):
    """detailed: multi-view stereo + hole filling (~8 min); fast: monocular only (~2 min, softer)."""

    DETAILED = "detailed"
    FAST = "fast"


class ScanType(StrEnum):
    """object: one thing, orbited and cropped with a box; scene: a room or several objects, kept uncut."""

    OBJECT = "object"
    SCENE = "scene"


class UploadKind(StrEnum):
    VIDEO = "video"
    METADATA = "metadata"
    IMU = "imu"
    AR_FRAMES = "ar_frames"


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Vector3 = tuple[float, float, float]


class CreateScanRequest(APIModel):
    device_platform: DevicePlatform
    device_model: Annotated[str, Field(min_length=1, max_length=160)]
    capture_duration_ms: Annotated[int, Field(gt=0)]
    processing_mode: ProcessingMode = ProcessingMode.DETAILED
    scan_type: ScanType = ScanType.OBJECT


class UploadConstraints(APIModel):
    max_video_bytes: int
    max_capture_duration_ms: int
    allowed_video_content_types: list[str]


class CreateScanResponse(APIModel):
    id: UUID
    status: ScanStatus
    upload_constraints: UploadConstraints
    retention_deadline: datetime


class UploadUrlRequest(APIModel):
    kind: UploadKind = UploadKind.VIDEO
    content_type: str
    byte_size: Annotated[int, Field(gt=0)]
    sha256: Sha256

    @model_validator(mode="after")
    def require_video_fields(self):
        expected_types = {
            UploadKind.VIDEO: {"video/mp4", "application/mp4"},
            UploadKind.METADATA: {"application/json"},
            UploadKind.IMU: {"application/zstd", "application/octet-stream"},
            UploadKind.AR_FRAMES: {"application/zstd", "application/octet-stream"},
        }
        if self.content_type not in expected_types[self.kind]:
            allowed = ", ".join(sorted(expected_types[self.kind]))
            raise ValueError(f"contentType for {self.kind.value} must be one of: {allowed}")
        if self.kind == UploadKind.VIDEO:
            return self
        return self


class UploadUrlResponse(APIModel):
    method: Literal["PUT"] = "PUT"
    url: str
    object_key: str
    expires_at: datetime
    required_headers: dict[str, str]


class UploadCompleteRequest(APIModel):
    video_object_key: str
    metadata_object_key: str
    imu_object_key: str | None = None
    ar_frames_object_key: str | None = None
    sha256: Sha256
    byte_size: Annotated[int, Field(gt=0)]


class ScanResponse(APIModel):
    id: UUID
    status: ScanStatus
    failure_code: str | None
    progress: float
    video_object_key: str | None
    metadata_object_key: str | None
    imu_object_key: str | None
    ar_frames_object_key: str | None
    sha256: str | None
    byte_size: int | None
    device_platform: str
    device_model: str
    capture_duration_ms: int
    processing_mode: str
    scan_type: str
    created_at: datetime
    updated_at: datetime
    retention_deadline: datetime


class Bounds(APIModel):
    min: Vector3
    max: Vector3

    @model_validator(mode="after")
    def ordered(self):
        if any(lower >= upper for lower, upper in zip(self.min, self.max, strict=True)):
            raise ValueError("every bounds max value must be greater than min")
        return self


class ReconstructionUpdate(APIModel):
    transform: Annotated[list[float], Field(min_length=16, max_length=16)]
    crop: Bounds


class ReconstructionResponse(APIModel):
    scan_id: UUID
    transform: list[float]
    crop: Bounds


class PreviewResponse(APIModel):
    preview_url: str
    bounds: Bounds
    units: Literal["reconstruction_units"] = "reconstruction_units"
    scan_type: str
    registered_frames: int
    selected_frames: int
    quality: str
    warnings: list[str]
    expires_at: datetime


class ScaleRequest(APIModel):
    axis: Literal["x", "y", "z"]
    blocks: Annotated[int, Field(gt=0)]


class CreateExportRequest(APIModel):
    scale: ScaleRequest
    palette: Annotated[str, Field(min_length=1, max_length=80)] = "geometry-safe-v1"
    confirm_large_export: bool = False


class ExportEstimateResponse(APIModel):
    voxel_size: float
    voxel_dimensions: tuple[int, int, int]
    block_count_estimate: int
    estimate_is_upper_bound: Literal[True] = True
    warning_threshold: int
    hard_limit: int
    requires_confirmation: bool


class ExportResponse(APIModel):
    id: UUID
    scan_id: UUID
    status: ExportStatus
    failure_code: str | None
    selected_axis: str
    target_block_dimension: int
    voxel_size: float | None
    occupied_block_count: int | None
    palette_version: str
    minecraft_version: str
    created_at: datetime
    updated_at: datetime


class DownloadUrlResponse(APIModel):
    url: str
    expires_at: datetime


class ErrorResponse(APIModel):
    code: str
    message: str
