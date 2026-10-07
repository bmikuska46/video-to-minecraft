from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base
from .states import ExportStatus, ScanStatus


def uuid_string() -> str:
    return str(uuid.uuid4())


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    status: Mapped[ScanStatus] = mapped_column(String(32), default=ScanStatus.CREATED)
    failure_code: Mapped[str | None] = mapped_column(String(64))
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    video_object_key: Mapped[str | None] = mapped_column(Text)
    metadata_object_key: Mapped[str | None] = mapped_column(Text)
    imu_object_key: Mapped[str | None] = mapped_column(Text)
    ar_frames_object_key: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(64))
    byte_size: Mapped[int | None] = mapped_column(BigInteger)
    device_platform: Mapped[str] = mapped_column(String(16))
    device_model: Mapped[str] = mapped_column(String(160))
    capture_duration_ms: Mapped[int] = mapped_column(Integer)
    processing_mode: Mapped[str] = mapped_column(String(16), default="detailed", server_default="detailed")
    scan_type: Mapped[str] = mapped_column(String(16), default="object", server_default="object")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    retention_deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    reconstruction: Mapped[Reconstruction | None] = relationship(
        back_populates="scan", cascade="all, delete-orphan", uselist=False
    )
    exports: Mapped[list[Export]] = relationship(back_populates="scan", cascade="all, delete-orphan")


class Reconstruction(Base):
    __tablename__ = "reconstructions"

    scan_id: Mapped[str] = mapped_column(ForeignKey("scans.id", ondelete="CASCADE"), primary_key=True)
    pipeline_version: Mapped[str | None] = mapped_column(String(80))
    settings_hash: Mapped[str | None] = mapped_column(String(64))
    registered_frame_count: Mapped[int | None] = mapped_column(Integer)
    selected_frame_count: Mapped[int | None] = mapped_column(Integer)
    median_reprojection_error: Mapped[float | None] = mapped_column(Float)
    track_statistics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    confidence_decision: Mapped[str | None] = mapped_column(String(32))
    warnings: Mapped[list[str] | None] = mapped_column(JSON)
    canonical_ply_key: Mapped[str | None] = mapped_column(Text)
    preview_glb_key: Mapped[str | None] = mapped_column(Text)
    original_bounds: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    transform: Mapped[list[float] | None] = mapped_column(JSON)
    crop: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    scan: Mapped[Scan] = relationship(back_populates="reconstruction")


class Export(Base):
    __tablename__ = "exports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    scan_id: Mapped[str] = mapped_column(ForeignKey("scans.id", ondelete="CASCADE"), index=True)
    status: Mapped[ExportStatus] = mapped_column(String(32), default=ExportStatus.QUEUED)
    failure_code: Mapped[str | None] = mapped_column(String(64))
    selected_axis: Mapped[str] = mapped_column(String(1))
    target_block_dimension: Mapped[int] = mapped_column(Integer)
    voxel_size: Mapped[float | None] = mapped_column(Float)
    occupied_block_count: Mapped[int | None] = mapped_column(Integer)
    palette_version: Mapped[str] = mapped_column(String(80))
    minecraft_version: Mapped[str] = mapped_column(String(32))
    voxel_artifact_key: Mapped[str | None] = mapped_column(Text)
    world_zip_key: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    scan: Mapped[Scan] = relationship(back_populates="exports")


class PipelineStageMetric(Base):
    """Append-only operational record for one worker stage attempt."""

    __tablename__ = "pipeline_stage_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    job_kind: Mapped[str] = mapped_column(String(24), index=True)
    job_id: Mapped[str] = mapped_column(String(36), index=True)
    scan_id: Mapped[str] = mapped_column(String(36), index=True)
    stage: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int] = mapped_column(BigInteger)
    failure_reason: Mapped[str | None] = mapped_column(String(160))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
