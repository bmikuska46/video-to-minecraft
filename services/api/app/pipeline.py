"""Celery entry points that connect storage, reconstruction, voxelization, and Paper."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import tempfile
from collections import Counter

from celery import Celery
from celery.exceptions import WorkerShutdown
from celery.signals import worker_ready
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .config import Settings, get_settings
from .database import make_engine, make_session_factory
from .models import Export, PipelineStageMetric, Reconstruction, Scan
from .observability import StageRecorder, event
from .states import ExportStatus, ScanStatus, require_export_transition, require_scan_transition
from .storage import ObjectStorage, S3ObjectStorage
from .worker_diagnostics import startup_self_test, write_health


SCAN_PROGRESS = {
    ScanStatus.EXTRACTING_FRAMES: 0.12,
    ScanStatus.SPARSE_RECONSTRUCTION: 0.28,
    ScanStatus.DENSE_RECONSTRUCTION: 0.52,
    ScanStatus.FILTERING: 0.84,
    ScanStatus.PREVIEW_READY: 1.0,
}
SCAN_PIPELINE = list(SCAN_PROGRESS)
EXPORT_PROGRESS = [
    ExportStatus.QUEUED,
    ExportStatus.VOXELIZING,
    ExportStatus.GENERATING_WORLD,
    ExportStatus.READY,
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log_path: Path, *, environment: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=False)
    if result.returncode != 0:
        tail = log_path.read_text(errors="replace")[-2000:]
        raise RuntimeError(f"command failed with code {result.returncode}: {tail}")


def _scan_failure_code(error: Exception) -> str:
    message = str(error).lower()
    if "sharp" in message or "candidate frames" in message:
        return "TOO_FEW_SHARP_FRAMES"
    if "baseline" in message or "parallax" in message:
        return "INSUFFICIENT_PARALLAX"
    if "register" in message or "sparse" in message:
        return "CAMERA_REGISTRATION_FAILED"
    if "dense" in message or "patch_match" in message or "stereo" in message:
        return "DENSE_RECONSTRUCTION_FAILED"
    return "RECONSTRUCTION_FAILED"


class Pipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        storage: ObjectStorage | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.engine = make_engine(self.settings)
        self.sessions = make_session_factory(self.engine)
        self.storage = storage or S3ObjectStorage(self.settings)

    def _persist_stage(self, **values) -> None:
        with self.sessions() as session:
            started_at = values.pop("started_at")
            duration_ms = values.pop("duration_ms")
            session.add(
                PipelineStageMetric(
                    **values,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    duration_ms=duration_ms,
                )
            )
            session.commit()

    def _recorder(self, job_kind: str, job_id: str, scan_id: str) -> StageRecorder:
        return StageRecorder(
            self._persist_stage, job_kind=job_kind, job_id=job_id, scan_id=scan_id
        )

    @staticmethod
    def _elapsed_ms(started_at: datetime) -> int:
        now = datetime.now(timezone.utc)
        if started_at.tzinfo is None:
            now = now.replace(tzinfo=None)
        return max(0, round((now - started_at).total_seconds() * 1000))

    def _record_terminal(
        self, *, job_kind: str, job_id: str, scan_id: str,
        started_at: datetime, status: str, failure_reason: str | None = None,
        metrics: dict | None = None,
    ) -> None:
        self._persist_stage(
            job_kind=job_kind,
            job_id=job_id,
            scan_id=scan_id,
            stage="END_TO_END",
            status=status,
            started_at=started_at,
            duration_ms=self._elapsed_ms(started_at),
            failure_reason=failure_reason,
            metrics=metrics or {},
        )

    def _import_reconstruction_stages(self, scan_id: str, manifest: dict) -> None:
        for stage in manifest.get("stages", []):
            started = stage.get("startedAt")
            try:
                started_at = datetime.fromisoformat(started.replace("Z", "+00:00"))
            except (AttributeError, ValueError):
                started_at = datetime.now(timezone.utc)
            self._persist_stage(
                job_kind="reconstruction",
                job_id=scan_id,
                scan_id=scan_id,
                stage=str(stage.get("name", "UNKNOWN")),
                status=str(stage.get("status", "UNKNOWN")),
                started_at=started_at,
                duration_ms=int(stage.get("durationMs") or 0),
                failure_reason=(stage.get("failure") or {}).get("type"),
                metrics=stage.get("metrics") or {},
            )

    def _advance_scan(self, scan_id: str, target: ScanStatus) -> None:
        with self.sessions() as session:
            scan = session.get(Scan, scan_id)
            if scan is None:
                raise ValueError(f"scan does not exist: {scan_id}")
            current = ScanStatus(scan.status)
            if current == target:
                return
            current_index = SCAN_PIPELINE.index(current) if current in SCAN_PIPELINE else -1
            target_index = SCAN_PIPELINE.index(target)
            for next_status in SCAN_PIPELINE[current_index + 1:target_index + 1]:
                require_scan_transition(ScanStatus(scan.status), next_status)
                scan.status = next_status
                scan.progress = SCAN_PROGRESS[next_status]
                scan.updated_at = datetime.now(timezone.utc)
            session.commit()

    def process_scan(self, scan_id: str) -> None:
        with self.sessions() as session:
            scan = session.get(Scan, scan_id)
            if scan is None:
                raise ValueError(f"scan does not exist: {scan_id}")
            if ScanStatus(scan.status) == ScanStatus.PREVIEW_READY:
                return
            if ScanStatus(scan.status) == ScanStatus.FAILED:
                raise ValueError("failed scans require an explicit retry workflow")
            video_key = scan.video_object_key
            processing_mode = scan.processing_mode or "detailed"
            scan_type = scan.scan_type or "object"
            job_started_at = scan.created_at
            if not video_key:
                raise ValueError("queued scan has no source video")
        work_root = Path(self.settings.pipeline_work_root)
        work_root.mkdir(parents=True, exist_ok=True)
        recorder = self._recorder("reconstruction", scan_id, scan_id)
        try:
            self._advance_scan(scan_id, ScanStatus.EXTRACTING_FRAMES)
            with tempfile.TemporaryDirectory(prefix=f"scan-{scan_id}-", dir=work_root) as temporary:
                work = Path(temporary)
                video = work / "video.mp4"
                output = work / "reconstruction"
                with recorder.stage("SOURCE_DOWNLOAD") as stage_metrics:
                    self.storage.download_file(video_key, video)
                    stage_metrics["uploadBytes"] = video.stat().st_size
                with recorder.stage("RECONSTRUCTION") as stage_metrics:
                    run(
                        [sys.executable, self.settings.reconstruction_script, str(video), str(output),
                         "--mode", processing_mode, "--scan-type", scan_type],
                        work / "reconstruction-worker.log",
                    )
                    manifest = json.loads((output / "manifest.json").read_text())
                    metrics = manifest.get("metrics", {})
                    if manifest.get("status") != "COMPLETED":
                        raise RuntimeError("reconstruction manifest did not succeed")
                    for key in (
                        "extractedFrames", "selectedFrames", "rejectedFrames",
                        "registeredFrames", "registrationRatio", "points3D",
                        "meanTrackLength", "medianReprojectionErrorPixels",
                        "fusedPoints", "filteredPoints", "observedPoints", "inferredPoints",
                    ):
                        if key in metrics:
                            stage_metrics[key] = metrics[key]
                self._import_reconstruction_stages(scan_id, manifest)
                self._advance_scan(scan_id, ScanStatus.SPARSE_RECONSTRUCTION)
                self._advance_scan(scan_id, ScanStatus.DENSE_RECONSTRUCTION)
                self._advance_scan(scan_id, ScanStatus.FILTERING)
                preview_glb = output / "preview.glb"
                with recorder.stage("PREVIEW_GENERATION") as stage_metrics:
                    run(
                        [sys.executable, self.settings.preview_glb_script,
                         str(output / "preview.ply"), str(preview_glb)],
                        work / "preview-glb.log",
                    )
                    stage_metrics["artifactBytes"] = preview_glb.stat().st_size
                prefix = f"scans/{scan_id}/reconstruction/{manifest['settingsHash']}"
                artifacts = {
                    "canonical.ply": "application/octet-stream",
                    "preview.glb": "model/gltf-binary",
                    "manifest.json": "application/json",
                }
                with recorder.stage("ARTIFACT_UPLOAD") as stage_metrics:
                    artifact_bytes = 0
                    for filename, content_type in artifacts.items():
                        source = output / filename
                        artifact_bytes += source.stat().st_size
                        self.storage.upload_file(
                            f"{prefix}/{filename}", source, content_type=content_type, sha256=sha256_file(source)
                        )
                    stage_metrics["artifactBytes"] = artifact_bytes
                with self.sessions() as session:
                    scan = session.get(Scan, scan_id)
                    reconstruction = session.get(Reconstruction, scan_id) or Reconstruction(scan_id=scan_id)
                    reconstruction.pipeline_version = manifest.get(
                        "pipelineVersion", self.settings.pipeline_version
                    )
                    reconstruction.settings_hash = manifest["settingsHash"]
                    reconstruction.selected_frame_count = metrics.get("selectedFrames") or metrics.get("extractedFrames")
                    reconstruction.registered_frame_count = metrics.get("registeredFrames")
                    reconstruction.median_reprojection_error = metrics.get("medianReprojectionErrorPixels")
                    reconstruction.track_statistics = {
                        key: metrics[key] for key in ("points3D", "meanTrackLength", "registrationRatio") if key in metrics
                    }
                    reconstruction.confidence_decision = "accepted"
                    reconstruction.warnings = []
                    reconstruction.canonical_ply_key = f"{prefix}/canonical.ply"
                    reconstruction.preview_glb_key = f"{prefix}/preview.glb"
                    reconstruction.original_bounds = metrics["bounds"]
                    session.add(reconstruction)
                    session.commit()
                self._advance_scan(scan_id, ScanStatus.PREVIEW_READY)
                self._record_terminal(
                    job_kind="reconstruction", job_id=scan_id, scan_id=scan_id,
                    started_at=job_started_at, status="COMPLETED",
                    metrics={"artifactBytes": artifact_bytes},
                )
                event("reconstruction.completed", jobId=scan_id, scanId=scan_id)
        except Exception as error:
            with self.sessions() as session:
                scan = session.get(Scan, scan_id)
                if scan is not None and ScanStatus(scan.status) not in {ScanStatus.FAILED, ScanStatus.PREVIEW_READY}:
                    require_scan_transition(ScanStatus(scan.status), ScanStatus.FAILED)
                    scan.status = ScanStatus.FAILED
                    scan.failure_code = _scan_failure_code(error)
                    session.commit()
            self._record_terminal(
                job_kind="reconstruction", job_id=scan_id, scan_id=scan_id,
                started_at=job_started_at, status="FAILED",
                failure_reason=_scan_failure_code(error),
            )
            event("reconstruction.failed", jobId=scan_id, scanId=scan_id,
                  failureReason=_scan_failure_code(error))
            raise

    def process_export(self, export_id: str) -> None:
        with self.sessions() as session:
            export = session.scalar(
                select(Export).where(Export.id == export_id).options(selectinload(Export.scan).selectinload(Scan.reconstruction))
            )
            if export is None:
                raise ValueError(f"export does not exist: {export_id}")
            if ExportStatus(export.status) == ExportStatus.READY:
                return
            if ExportStatus(export.status) == ExportStatus.FAILED:
                raise ValueError("failed exports require an explicit retry workflow")
            reconstruction = export.scan.reconstruction
            if reconstruction is None or not reconstruction.canonical_ply_key or not reconstruction.crop or not reconstruction.transform:
                raise ValueError("export is missing its saved reconstruction selection")
            job = {
                "scan_id": export.scan_id,
                "canonical_key": reconstruction.canonical_ply_key,
                "crop": reconstruction.crop,
                "transform": reconstruction.transform,
                "axis": export.selected_axis,
                "blocks": export.target_block_dimension,
                "minecraft_version": export.minecraft_version,
            }
            job_started_at = export.created_at
        work_root = Path(self.settings.pipeline_work_root)
        work_root.mkdir(parents=True, exist_ok=True)
        recorder = self._recorder("export", export_id, job["scan_id"])
        scan_id = recorder.scan_id
        try:
            with self.sessions() as session:
                export = session.get(Export, export_id)
                if ExportStatus(export.status) == ExportStatus.QUEUED:
                    require_export_transition(ExportStatus.QUEUED, ExportStatus.VOXELIZING)
                    export.status = ExportStatus.VOXELIZING
                    export.updated_at = datetime.now(timezone.utc)
                session.commit()
            with tempfile.TemporaryDirectory(prefix=f"export-{export_id}-", dir=work_root) as temporary:
                work = Path(temporary)
                cloud = work / "canonical.ply"
                voxels = work / "voxels.pb.zst"
                with recorder.stage("VOXELIZING") as stage_metrics:
                    self.storage.download_file(job["canonical_key"], cloud)
                    run(
                        [sys.executable, self.settings.voxelizer_script, str(cloud), str(voxels),
                         "--crop-min", *(str(value) for value in job["crop"]["min"]),
                         "--crop-max", *(str(value) for value in job["crop"]["max"]),
                         "--axis", job["axis"], "--blocks", str(job["blocks"]),
                         "--transform", *(str(value) for value in job["transform"]),
                         "--palette", self.settings.palette_path],
                        work / "voxelizer.log",
                    )
                    sys.path.insert(0, str(Path(self.settings.voxelizer_script).parent))
                    from voxel_contract import decode  # type: ignore[import-not-found]
                    voxel_world = decode(voxels.read_bytes())
                    occupied = len(voxel_world.voxels)
                    palette_states = {
                        entry.id: entry.block_state for entry in voxel_world.palette
                    }
                    stage_metrics["occupiedVoxels"] = occupied
                    stage_metrics["paletteDistribution"] = dict(sorted(Counter(
                        palette_states.get(voxel.palette_id, f"unknown:{voxel.palette_id}")
                        for voxel in voxel_world.voxels
                    ).items()))
                    stage_metrics["artifactBytes"] = voxels.stat().st_size
                voxel_digest = sha256_file(voxels)
                voxel_key = f"exports/{export_id}/voxels.pb.zst"
                self.storage.upload_file(voxel_key, voxels, content_type="application/zstd", sha256=voxel_digest)
                with self.sessions() as session:
                    export = session.get(Export, export_id)
                    if ExportStatus(export.status) == ExportStatus.VOXELIZING:
                        require_export_transition(ExportStatus.VOXELIZING, ExportStatus.GENERATING_WORLD)
                        export.status = ExportStatus.GENERATING_WORLD
                    elif ExportStatus(export.status) != ExportStatus.GENERATING_WORLD:
                        raise ValueError(f"export cannot generate a world while {export.status}")
                    export.voxel_artifact_key = voxel_key
                    export.occupied_block_count = occupied
                    export.updated_at = datetime.now(timezone.utc)
                    session.commit()
                payload = {
                    "schemaVersion": 1, "jobId": export_id, "minecraftVersion": job["minecraft_version"],
                    "voxelSha256": voxel_digest, "worldName": "minecraft-video-world", "platformY": 64,
                    "platformMargin": 8, "platformBlockState": "minecraft:smooth_stone", "batchSize": 20000,
                    "maxBlockCount": self.settings.block_count_hard_limit, "maxPlatformBlocks": 20_000_000,
                }
                canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
                signature = hmac.new(self.settings.worldgen_manifest_key.encode(), canonical, hashlib.sha256).hexdigest()
                manifest_path = work / "job-manifest.json"
                manifest_path.write_text(json.dumps({"payload": payload, "signature": f"hmac-sha256:{signature}"}, indent=2) + "\n")
                world_zip = work / "world.zip"
                environment = os.environ.copy()
                environment["VTM_WORLDGEN_MANIFEST_KEY"] = self.settings.worldgen_manifest_key
                progress_path = work / "worldgen-progress.json"
                with recorder.stage("GENERATING_WORLD") as stage_metrics:
                    run(
                        [sys.executable, self.settings.worldgen_runner, "generate",
                         "--paper-jar", self.settings.worldgen_paper_jar,
                         "--paper-cache-dir", self.settings.worldgen_paper_cache_dir,
                         "--plugin-jar", self.settings.worldgen_plugin_jar,
                         "--template", self.settings.worldgen_template,
                         "--manifest", str(manifest_path), "--voxels", str(voxels),
                         "--work-root", str(work / "worldgen"), "--progress", str(progress_path),
                         "--output-zip", str(world_zip)],
                        work / "worldgen-worker.log", environment=environment,
                    )
                    if progress_path.is_file():
                        progress = json.loads(progress_path.read_text())
                        for key in ("paperGenerationDurationMs", "paperValidationDurationMs"):
                            if key in progress:
                                stage_metrics[key] = progress[key]
                    stage_metrics["artifactBytes"] = world_zip.stat().st_size
                world_key = f"exports/{export_id}/world.zip"
                self.storage.upload_file(world_key, world_zip, content_type="application/zip", sha256=sha256_file(world_zip))
                with self.sessions() as session:
                    export = session.get(Export, export_id)
                    require_export_transition(ExportStatus(export.status), ExportStatus.READY)
                    export.status = ExportStatus.READY
                    export.world_zip_key = world_key
                    export.updated_at = datetime.now(timezone.utc)
                    session.commit()
                self._record_terminal(
                    job_kind="export", job_id=export_id, scan_id=scan_id,
                    started_at=job_started_at, status="COMPLETED",
                    metrics={
                        "occupiedVoxels": occupied,
                        "artifactBytes": world_zip.stat().st_size,
                    },
                )
                event("export.completed", jobId=export_id, scanId=scan_id,
                      occupiedVoxels=occupied, artifactBytes=world_zip.stat().st_size)
        except Exception:
            with self.sessions() as session:
                export = session.get(Export, export_id)
                if export is not None and ExportStatus(export.status) not in {ExportStatus.READY, ExportStatus.FAILED}:
                    failed_from = ExportStatus(export.status)
                    require_export_transition(failed_from, ExportStatus.FAILED)
                    export.status = ExportStatus.FAILED
                    export.failure_code = "WORLD_GENERATION_FAILED" if failed_from == ExportStatus.GENERATING_WORLD else "VOXELIZATION_FAILED"
                    export.updated_at = datetime.now(timezone.utc)
                    session.commit()
                    failure_code = export.failure_code
                else:
                    failure_code = "EXPORT_FAILED"
            self._record_terminal(
                job_kind="export", job_id=export_id, scan_id=scan_id,
                started_at=job_started_at, status="FAILED", failure_reason=failure_code,
            )
            event("export.failed", jobId=export_id, scanId=scan_id,
                  failureReason=failure_code)
            raise


settings = get_settings()
celery_app = Celery("video-to-minecraft-pipeline", broker=settings.celery_broker_url)


@worker_ready.connect
def report_worker_ready(**_: object) -> None:
    health_path = Path(os.environ.get("VTM_WORKER_HEALTH_PATH", "/tmp/vtm-worker-ready.json"))
    try:
        diagnostics = startup_self_test(gpu=os.environ.get("VTM_WORKER_ROLE", "pipeline") != "export")
        write_health(health_path, ready=True, diagnostics=diagnostics)
        event("worker.ready", **diagnostics)
    except Exception as error:
        write_health(health_path, ready=False, failure=type(error).__name__)
        event("worker.self_test.failed", failureReason=type(error).__name__)
        raise WorkerShutdown(1) from error


@celery_app.task(name="reconstruction.process_scan")
def process_scan(scan_id: str) -> None:
    Pipeline().process_scan(scan_id)


@celery_app.task(name="worldgen.process_export")
def process_export(export_id: str) -> None:
    Pipeline().process_export(export_id)
