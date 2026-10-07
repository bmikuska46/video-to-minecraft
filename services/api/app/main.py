from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Response, status
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy.orm import Session, selectinload

from .capture import CaptureMetadata
from .config import Settings, get_settings, max_capture_duration_ms
from .database import Base, add_missing_columns, make_engine, make_session_factory, session_dependency
from .export_contract import (
    ContractError,
    estimate_blocks,
    transformed_bounds,
    validate_crop_within_bounds,
    validate_rigid_transform,
)
from .jobs import CeleryJobDispatcher, JobDispatcher
from .models import Export, PipelineStageMetric, Reconstruction, Scan
from .observability import event
from .schemas import (
    CreateExportRequest,
    CreateScanRequest,
    CreateScanResponse,
    DownloadUrlResponse,
    ExportEstimateResponse,
    ExportResponse,
    PreviewResponse,
    ReconstructionResponse,
    ReconstructionUpdate,
    ScanResponse,
    UploadCompleteRequest,
    UploadConstraints,
    UploadKind,
    UploadUrlRequest,
    UploadUrlResponse,
)
from .states import ExportStatus, ScanStatus, require_scan_transition
from .storage import ObjectNotFoundError, ObjectStorage, S3ObjectStorage


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def problem(http_status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=http_status, detail={"code": code, "message": message})


def expected_key(scan_id: str, kind: UploadKind) -> str:
    filename = {
        UploadKind.VIDEO: "video.mp4",
        UploadKind.METADATA: "capture.json",
        UploadKind.IMU: "imu.jsonl.zst",
        UploadKind.AR_FRAMES: "ar-frames.pb.zst",
    }[kind]
    return f"scans/{scan_id}/source/{filename}"


def upload_limit(settings: Settings, kind: UploadKind) -> int:
    return {
        UploadKind.VIDEO: settings.max_video_bytes,
        UploadKind.METADATA: settings.max_metadata_bytes,
        UploadKind.IMU: settings.max_imu_bytes,
        UploadKind.AR_FRAMES: settings.max_ar_frames_bytes,
    }[kind]


def locked_scan(session: Session, scan_id: UUID) -> Scan:
    record = session.scalar(
        select(Scan).where(Scan.id == str(scan_id)).with_for_update()
    )
    if record is None:
        raise problem(status.HTTP_404_NOT_FOUND, "SCAN_NOT_FOUND", "scan does not exist")
    return record


def create_app(
    *,
    settings: Settings | None = None,
    storage: ObjectStorage | None = None,
    dispatcher: JobDispatcher | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    engine = make_engine(settings)
    session_factory = make_session_factory(engine)
    storage = storage or S3ObjectStorage(settings)
    dispatcher = dispatcher or CeleryJobDispatcher(settings)
    get_session = session_dependency(session_factory)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        Base.metadata.create_all(engine)
        add_missing_columns(engine)
        yield
        engine.dispose()

    app = FastAPI(title="Video-to-Minecraft API", version="1.0.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.storage = storage
    app.state.dispatcher = dispatcher

    @app.get("/health/live", tags=["health"])
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    def ready(response: Response) -> dict[str, object]:
        checks: dict[str, str] = {}
        dependencies = {
            "postgresql": lambda: _database_ready(engine),
            "redis": getattr(dispatcher, "ready", lambda: None),
            "storage": getattr(storage, "ready", lambda: None),
        }
        for name, check in dependencies.items():
            try:
                check()
                checks[name] = "ok"
            except Exception as error:
                checks[name] = "unavailable"
                event("health.dependency.failed", dependency=name, errorType=type(error).__name__)
        healthy = all(value == "ok" for value in checks.values())
        if not healthy:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "ok" if healthy else "unavailable", "checks": checks}

    @app.post(
        "/v1/scans",
        response_model=CreateScanResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["scans"],
    )
    def create_scan(payload: CreateScanRequest, session: Session = Depends(get_session)):
        duration_limit = max_capture_duration_ms(settings, payload.scan_type.value)
        if payload.capture_duration_ms > duration_limit:
            raise problem(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "CAPTURE_TOO_LONG",
                f"{payload.scan_type.value} capture exceeds {duration_limit} ms",
            )
        now = utcnow()
        record = Scan(
            status=ScanStatus.CREATED,
            progress=0.0,
            device_platform=payload.device_platform.value,
            device_model=payload.device_model,
            capture_duration_ms=payload.capture_duration_ms,
            processing_mode=payload.processing_mode.value,
            scan_type=payload.scan_type.value,
            created_at=now,
            updated_at=now,
            retention_deadline=now + timedelta(days=settings.retention_days),
        )
        session.add(record)
        session.commit()
        return CreateScanResponse(
            id=record.id,
            status=record.status,
            upload_constraints=UploadConstraints(
                max_video_bytes=settings.max_video_bytes,
                max_capture_duration_ms=duration_limit,
                allowed_video_content_types=["video/mp4", "application/mp4"],
            ),
            retention_deadline=record.retention_deadline,
        )

    @app.post(
        "/v1/scans/{scan_id}/upload-url",
        response_model=UploadUrlResponse,
        tags=["scans"],
    )
    def create_upload_url(
        scan_id: UUID,
        payload: UploadUrlRequest,
        session: Session = Depends(get_session),
    ):
        record = locked_scan(session, scan_id)
        current = ScanStatus(record.status)
        if current not in {ScanStatus.CREATED, ScanStatus.UPLOADING}:
            raise problem(
                status.HTTP_409_CONFLICT,
                "INVALID_SCAN_STATE",
                f"uploads are not accepted while scan is {current}",
            )
        limit = upload_limit(settings, payload.kind)
        if payload.byte_size > limit:
            raise problem(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "UPLOAD_TOO_LARGE",
                f"upload exceeds {limit} bytes",
            )
        if current == ScanStatus.CREATED:
            require_scan_transition(current, ScanStatus.UPLOADING)
            record.status = ScanStatus.UPLOADING
            record.progress = 0.02
            record.updated_at = utcnow()
            session.commit()
        key = expected_key(record.id, payload.kind)
        signed = storage.sign_put(
            key,
            content_type=payload.content_type,
            byte_size=payload.byte_size,
            sha256=payload.sha256,
        )
        return UploadUrlResponse(
            url=signed.url,
            object_key=key,
            expires_at=signed.expires_at,
            required_headers=signed.required_headers,
        )

    @app.post(
        "/v1/scans/{scan_id}/upload-complete",
        response_model=ScanResponse,
        tags=["scans"],
    )
    def upload_complete(
        scan_id: UUID,
        payload: UploadCompleteRequest,
        response: Response,
        session: Session = Depends(get_session),
    ):
        record = locked_scan(session, scan_id)
        canonical_video_key = expected_key(record.id, UploadKind.VIDEO)
        canonical_metadata_key = expected_key(record.id, UploadKind.METADATA)
        canonical_imu_key = expected_key(record.id, UploadKind.IMU)
        canonical_ar_frames_key = expected_key(record.id, UploadKind.AR_FRAMES)
        supplied_keys = {
            "videoObjectKey": (payload.video_object_key, canonical_video_key, False),
            "metadataObjectKey": (payload.metadata_object_key, canonical_metadata_key, False),
            "imuObjectKey": (payload.imu_object_key, canonical_imu_key, True),
            "arFramesObjectKey": (payload.ar_frames_object_key, canonical_ar_frames_key, True),
        }
        for field, (supplied, canonical, optional) in supplied_keys.items():
            if (supplied is None and not optional) or (supplied is not None and supplied != canonical):
                raise problem(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    "INVALID_OBJECT_KEY",
                    f"{field} is not the immutable key assigned to this scan",
                )
        if payload.byte_size > settings.max_video_bytes:
            raise problem(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "UPLOAD_TOO_LARGE", "video is too large")
        try:
            uploaded = storage.head(canonical_video_key)
        except ObjectNotFoundError as error:
            raise problem(
                status.HTTP_409_CONFLICT,
                "UPLOAD_NOT_FOUND",
                "the uploaded video is not present in object storage",
            ) from error
        if uploaded.byte_size != payload.byte_size:
            raise problem(status.HTTP_409_CONFLICT, "BYTE_SIZE_MISMATCH", "uploaded byte size does not match")
        if uploaded.sha256 != payload.sha256:
            raise problem(status.HTTP_409_CONFLICT, "CHECKSUM_MISMATCH", "uploaded checksum does not match")
        sidecars = [
            (UploadKind.METADATA, canonical_metadata_key, payload.metadata_object_key),
            (UploadKind.IMU, canonical_imu_key, payload.imu_object_key),
            (UploadKind.AR_FRAMES, canonical_ar_frames_key, payload.ar_frames_object_key),
        ]
        for kind, key, supplied in sidecars:
            if supplied is None:
                continue
            try:
                sidecar = storage.head(key)
            except ObjectNotFoundError as error:
                raise problem(
                    status.HTTP_409_CONFLICT,
                    f"{kind.value.upper()}_UPLOAD_NOT_FOUND",
                    f"the {kind.value} sidecar is not present in object storage",
                ) from error
            limit = upload_limit(settings, kind)
            if sidecar.byte_size > limit:
                raise problem(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    f"{kind.value.upper()}_UPLOAD_TOO_LARGE",
                    f"{kind.value} sidecar exceeds {limit} bytes",
                )

        try:
            CaptureMetadata.model_validate_json(storage.read_bytes(canonical_metadata_key))
        except ObjectNotFoundError as error:
            raise problem(
                status.HTTP_409_CONFLICT,
                "METADATA_UPLOAD_NOT_FOUND",
                "the capture metadata is not present in object storage",
            ) from error
        except (ValidationError, ValueError, json.JSONDecodeError) as error:
            raise problem(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "INVALID_CAPTURE_METADATA",
                "capture.json does not match capture schema version 1",
            ) from error

        current = ScanStatus(record.status)
        already_queued = current == ScanStatus.QUEUED
        if current in {ScanStatus.UPLOADED, ScanStatus.QUEUED}:
            if (
                record.video_object_key != payload.video_object_key
                or record.metadata_object_key != payload.metadata_object_key
                or record.imu_object_key != payload.imu_object_key
                or record.ar_frames_object_key != payload.ar_frames_object_key
                or record.sha256 != payload.sha256
                or record.byte_size != payload.byte_size
            ):
                raise problem(
                    status.HTTP_409_CONFLICT,
                    "UPLOAD_ALREADY_COMPLETED",
                    "scan was completed with different upload metadata",
                )
            if current == ScanStatus.UPLOADED:
                require_scan_transition(current, ScanStatus.QUEUED)
                record.status = ScanStatus.QUEUED
                record.progress = 0.08
                record.updated_at = utcnow()
                session.commit()
        elif current != ScanStatus.UPLOADING:
            raise problem(
                status.HTTP_409_CONFLICT,
                "INVALID_SCAN_STATE",
                f"upload cannot be completed while scan is {current}",
            )
        else:
            upload_started_at = record.updated_at
            require_scan_transition(current, ScanStatus.UPLOADED)
            record.status = ScanStatus.UPLOADED
            record.progress = 0.05
            record.video_object_key = payload.video_object_key
            record.metadata_object_key = payload.metadata_object_key
            record.imu_object_key = payload.imu_object_key
            record.ar_frames_object_key = payload.ar_frames_object_key
            record.sha256 = payload.sha256
            record.byte_size = payload.byte_size
            record.updated_at = utcnow()
            session.flush()
            require_scan_transition(ScanStatus.UPLOADED, ScanStatus.QUEUED)
            record.status = ScanStatus.QUEUED
            record.progress = 0.08
            completed_at = utcnow()
            duration_end = (
                completed_at.replace(tzinfo=None)
                if upload_started_at.tzinfo is None
                else completed_at
            )
            session.add(
                PipelineStageMetric(
                    job_kind="upload",
                    job_id=record.id,
                    scan_id=record.id,
                    stage="UPLOAD",
                    status="COMPLETED",
                    started_at=upload_started_at,
                    completed_at=completed_at,
                    duration_ms=max(
                        0,
                        round((duration_end - upload_started_at).total_seconds() * 1000),
                    ),
                    metrics={"uploadBytes": payload.byte_size},
                )
            )
            session.commit()
            event(
                "upload.completed",
                jobKind="upload",
                jobId=record.id,
                scanId=record.id,
                uploadBytes=payload.byte_size,
            )

        try:
            dispatcher.enqueue_reconstruction(record.id)
        except Exception as error:
            raise problem(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "QUEUE_UNAVAILABLE",
                "upload is saved but reconstruction could not be queued; retry this request",
            ) from error
        if already_queued:
            response.headers["Idempotent-Replay"] = "true"
        return record

    @app.get("/v1/scans/{scan_id}", response_model=ScanResponse, tags=["scans"])
    def get_scan(scan_id: UUID, session: Session = Depends(get_session)):
        record = session.get(Scan, str(scan_id))
        if record is None:
            raise problem(status.HTTP_404_NOT_FOUND, "SCAN_NOT_FOUND", "scan does not exist")
        return record

    @app.get("/v1/scans/{scan_id}/preview", response_model=PreviewResponse, tags=["scans"])
    def get_preview(scan_id: UUID, session: Session = Depends(get_session)):
        record = session.scalar(
            select(Scan)
            .where(Scan.id == str(scan_id))
            .options(selectinload(Scan.reconstruction))
        )
        if record is None:
            raise problem(status.HTTP_404_NOT_FOUND, "SCAN_NOT_FOUND", "scan does not exist")
        reconstruction = record.reconstruction
        if ScanStatus(record.status) != ScanStatus.PREVIEW_READY or reconstruction is None:
            raise problem(status.HTTP_409_CONFLICT, "PREVIEW_NOT_READY", "preview is not ready")
        if not reconstruction.preview_glb_key or not reconstruction.original_bounds:
            raise problem(status.HTTP_500_INTERNAL_SERVER_ERROR, "PREVIEW_INCOMPLETE", "preview record is incomplete")
        signed = storage.sign_get(reconstruction.preview_glb_key)
        return PreviewResponse(
            preview_url=signed.url,
            bounds=reconstruction.original_bounds,
            scan_type=record.scan_type,
            registered_frames=reconstruction.registered_frame_count or 0,
            selected_frames=reconstruction.selected_frame_count or 0,
            quality=reconstruction.confidence_decision or "unknown",
            warnings=reconstruction.warnings or [],
            expires_at=signed.expires_at,
        )

    @app.patch(
        "/v1/scans/{scan_id}/reconstruction",
        response_model=ReconstructionResponse,
        tags=["scans"],
    )
    def update_reconstruction(
        scan_id: UUID,
        payload: ReconstructionUpdate,
        session: Session = Depends(get_session),
    ):
        record = locked_scan(session, scan_id)
        if ScanStatus(record.status) != ScanStatus.PREVIEW_READY:
            raise problem(status.HTTP_409_CONFLICT, "PREVIEW_NOT_READY", "reconstruction is not editable yet")
        reconstruction = session.get(Reconstruction, record.id)
        if reconstruction is None:
            raise problem(status.HTTP_500_INTERNAL_SERVER_ERROR, "RECONSTRUCTION_MISSING", "reconstruction record is missing")
        if not reconstruction.original_bounds:
            raise problem(status.HTTP_500_INTERNAL_SERVER_ERROR, "RECONSTRUCTION_BOUNDS_MISSING", "reconstruction bounds are missing")
        try:
            validate_rigid_transform(payload.transform)
            available_min, available_max = transformed_bounds(
                reconstruction.original_bounds["min"],
                reconstruction.original_bounds["max"],
                payload.transform,
            )
            validate_crop_within_bounds(
                payload.crop.min, payload.crop.max, available_min, available_max
            )
        except (ContractError, KeyError, TypeError) as error:
            raise problem(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "INVALID_RECONSTRUCTION_SELECTION",
                str(error),
            ) from error
        reconstruction.transform = payload.transform
        reconstruction.crop = payload.crop.model_dump()
        record.updated_at = utcnow()
        session.commit()
        return ReconstructionResponse(
            scan_id=record.id,
            transform=reconstruction.transform,
            crop=reconstruction.crop,
        )

    def export_preflight(record: Scan, payload: CreateExportRequest, session: Session):
        if ScanStatus(record.status) != ScanStatus.PREVIEW_READY:
            raise problem(status.HTTP_409_CONFLICT, "PREVIEW_NOT_READY", "scan cannot be exported yet")
        reconstruction = session.get(Reconstruction, record.id)
        if reconstruction is None or not reconstruction.crop or not reconstruction.transform:
            raise problem(
                status.HTTP_409_CONFLICT,
                "CROP_REQUIRED",
                "save orientation and crop before creating an export",
            )
        if payload.scale.blocks > settings.max_target_blocks:
            raise problem(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "TARGET_DIMENSION_TOO_LARGE",
                f"target dimension exceeds {settings.max_target_blocks} blocks",
            )
        crop = reconstruction.crop
        estimate = estimate_blocks(
            crop["min"], crop["max"], axis=payload.scale.axis, blocks=payload.scale.blocks
        )
        if estimate.block_count_upper_bound > settings.block_count_hard_limit:
            raise problem(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "OUTPUT_TOO_LARGE",
                f"estimated block count exceeds hard limit {settings.block_count_hard_limit}",
            )
        return estimate

    @app.post(
        "/v1/scans/{scan_id}/exports/estimate",
        response_model=ExportEstimateResponse,
        tags=["exports"],
    )
    def estimate_export(
        scan_id: UUID,
        payload: CreateExportRequest,
        session: Session = Depends(get_session),
    ):
        record = locked_scan(session, scan_id)
        estimate = export_preflight(record, payload, session)
        return ExportEstimateResponse(
            voxel_size=estimate.voxel_size,
            voxel_dimensions=estimate.voxel_dimensions,
            block_count_estimate=estimate.block_count_upper_bound,
            warning_threshold=settings.block_count_warning_threshold,
            hard_limit=settings.block_count_hard_limit,
            requires_confirmation=(
                estimate.block_count_upper_bound > settings.block_count_warning_threshold
            ),
        )

    @app.post(
        "/v1/scans/{scan_id}/exports",
        response_model=ExportResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["exports"],
    )
    def create_export(
        scan_id: UUID,
        payload: CreateExportRequest,
        session: Session = Depends(get_session),
    ):
        record = locked_scan(session, scan_id)
        estimate = export_preflight(record, payload, session)
        if (
            estimate.block_count_upper_bound > settings.block_count_warning_threshold
            and not payload.confirm_large_export
        ):
            raise problem(
                status.HTTP_409_CONFLICT,
                "EXPORT_CONFIRMATION_REQUIRED",
                "estimated block count exceeds warning threshold; explicitly confirm the export",
            )
        now = utcnow()
        export = Export(
            scan_id=record.id,
            status=ExportStatus.QUEUED,
            selected_axis=payload.scale.axis,
            target_block_dimension=payload.scale.blocks,
            voxel_size=estimate.voxel_size,
            palette_version=payload.palette,
            minecraft_version=settings.minecraft_version,
            created_at=now,
            updated_at=now,
        )
        session.add(export)
        session.commit()
        try:
            dispatcher.enqueue_export(export.id)
        except Exception as error:
            raise problem(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "QUEUE_UNAVAILABLE",
                "export is saved but could not be queued",
            ) from error
        return export

    @app.get("/v1/exports/{export_id}", response_model=ExportResponse, tags=["exports"])
    def get_export(export_id: UUID, session: Session = Depends(get_session)):
        export = session.get(Export, str(export_id))
        if export is None:
            raise problem(status.HTTP_404_NOT_FOUND, "EXPORT_NOT_FOUND", "export does not exist")
        return export

    @app.get(
        "/v1/exports/{export_id}/download-url",
        response_model=DownloadUrlResponse,
        tags=["exports"],
    )
    def get_download_url(export_id: UUID, session: Session = Depends(get_session)):
        export = session.get(Export, str(export_id))
        if export is None:
            raise problem(status.HTTP_404_NOT_FOUND, "EXPORT_NOT_FOUND", "export does not exist")
        if ExportStatus(export.status) != ExportStatus.READY or not export.world_zip_key:
            raise problem(status.HTTP_409_CONFLICT, "EXPORT_NOT_READY", "export is not ready")
        signed = storage.sign_get(export.world_zip_key)
        return DownloadUrlResponse(url=signed.url, expires_at=signed.expires_at)

    return app


def _database_ready(engine) -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))


app = create_app()
