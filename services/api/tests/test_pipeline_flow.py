from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import textwrap
import zipfile

from app.config import Settings
from app.database import Base
from app.models import Export, PipelineStageMetric, Reconstruction, Scan
from app.pipeline import Pipeline
from app.states import ExportStatus, ScanStatus
from app.storage import ObjectNotFoundError, StoredObject


ROOT = Path(__file__).resolve().parents[3]


class FileObjectStorage:
    """Small object-store double that preserves the same immutable key boundary."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        path = self.root.joinpath(*key.split("/"))
        if not path.is_relative_to(self.root) or ".." in Path(key).parts:
            raise ValueError("invalid object key")
        return path

    def put_bytes(self, key: str, payload: bytes) -> None:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    def download_file(self, key: str, destination: Path) -> None:
        source = self._path(key)
        if not source.is_file():
            raise ObjectNotFoundError(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    def upload_file(self, key: str, source: Path, *, content_type: str, sha256: str) -> None:
        del content_type
        payload = source.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == sha256
        self.put_bytes(key, payload)

    def head(self, key: str) -> StoredObject:
        path = self._path(key)
        if not path.is_file():
            raise ObjectNotFoundError(key)
        payload = path.read_bytes()
        return StoredObject(len(payload), hashlib.sha256(payload).hexdigest())

    def read_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise ObjectNotFoundError(key)
        return path.read_bytes()

    def sign_put(self, *args, **kwargs):  # pragma: no cover - worker never signs URLs
        raise NotImplementedError

    def sign_get(self, *args, **kwargs):  # pragma: no cover - worker never signs URLs
        raise NotImplementedError


def write_executable(path: Path, source: str) -> None:
    path.write_text(textwrap.dedent(source))
    path.chmod(0o755)


def ply_payload() -> bytes:
    vertices = [
        (0.0, 0.0, 0.0, 170, 92, 55),
        (0.0, 1.0, 0.0, 175, 96, 57),
        (1.0, 0.0, 0.0, 168, 90, 54),
        (1.0, 1.0, 0.0, 172, 94, 56),
    ]
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    rows = "".join(" ".join(map(str, vertex)) + "\n" for vertex in vertices)
    return (header + rows).encode()


def test_complete_worker_flow_produces_downloadable_world() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        database = root / "pipeline.sqlite3"
        objects = FileObjectStorage(root / "objects")
        reconstruction_stub = root / "reconstruct.py"
        worldgen_stub = root / "worldgen.py"
        cloud_literal = repr(ply_payload())
        write_executable(
            reconstruction_stub,
            f"""
            import json
            from pathlib import Path
            import sys

            output = Path(sys.argv[2])
            output.mkdir(parents=True)
            Path({str(root / "reconstruction-argv.json")!r}).write_text(json.dumps(sys.argv[1:]))
            cloud = {cloud_literal}
            (output / "canonical.ply").write_bytes(cloud)
            (output / "preview.ply").write_bytes(cloud)
            (output / "manifest.json").write_text(json.dumps({{
                "pipelineVersion": "integration-v1",
                "settingsHash": "{'b' * 64}",
                "status": "COMPLETED",
                "metrics": {{
                    "selectedFrames": 30,
                    "registeredFrames": 28,
                    "medianReprojectionErrorPixels": 0.7,
                    "points3D": 4,
                    "meanTrackLength": 3.2,
                    "registrationRatio": 28 / 30,
                    "bounds": {{"min": [0, 0, 0], "max": [1, 1, 0]}},
                }},
            }}))
            """,
        )
        write_executable(
            worldgen_stub,
            f"""
            import json
            from pathlib import Path
            import sys
            import zipfile

            Path({str(root / "worldgen-argv.json")!r}).write_text(json.dumps(sys.argv[1:]))
            output = Path(sys.argv[sys.argv.index("--output-zip") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(output, "w") as archive:
                archive.writestr("level.dat", b"integration-fixture")
            """,
        )
        settings = Settings(
            database_url=f"sqlite:///{database}",
            pipeline_work_root=str(root / "work"),
            reconstruction_script=str(reconstruction_stub),
            preview_glb_script=str(ROOT / "services/reconstruction/preview_glb.py"),
            voxelizer_script=str(ROOT / "services/reconstruction/voxelizer.py"),
            palette_path=str(ROOT / "packages/block-palette/palette-v1.json"),
            worldgen_runner=str(worldgen_stub),
            worldgen_paper_jar=str(root / "unused-paper.jar"),
            worldgen_plugin_jar=str(root / "unused-plugin.jar"),
            worldgen_template=str(root / "unused-template"),
        )
        pipeline = Pipeline(settings, storage=objects)
        Base.metadata.create_all(pipeline.engine)
        now = datetime.now(timezone.utc)
        video_key = "scans/integration/source/video.mp4"
        objects.put_bytes(video_key, b"fixture-video")
        with pipeline.sessions() as session:
            scan = Scan(
                id="integration",
                status=ScanStatus.QUEUED,
                progress=0.08,
                video_object_key=video_key,
                device_platform="ios",
                device_model="fixture",
                capture_duration_ms=15_000,
                created_at=now,
                updated_at=now,
                retention_deadline=now + timedelta(days=7),
            )
            session.add(scan)
            session.commit()

        pipeline.process_scan("integration")
        assert json.loads((root / "reconstruction-argv.json").read_text())[2:] == [
            "--mode", "detailed", "--scan-type", "object",
        ]
        with pipeline.sessions() as session:
            scan = session.get(Scan, "integration")
            assert scan is not None
            assert ScanStatus(scan.status) == ScanStatus.PREVIEW_READY
            reconstruction = session.get(Reconstruction, "integration")
            assert reconstruction is not None
            assert reconstruction.pipeline_version == "integration-v1"
            assert reconstruction.preview_glb_key is not None
            assert objects.head(reconstruction.preview_glb_key).byte_size > 0
            stage_names = {
                metric.stage
                for metric in session.query(PipelineStageMetric)
                .filter(PipelineStageMetric.job_kind == "reconstruction")
                .all()
            }
            assert {"RECONSTRUCTION", "PREVIEW_GENERATION", "ARTIFACT_UPLOAD", "END_TO_END"} <= stage_names
            reconstruction.transform = [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
            ]
            reconstruction.crop = {"min": [0, 0, -0.1], "max": [1, 1, 0.1]}
            export = Export(
                id="integration-export",
                scan_id=scan.id,
                status=ExportStatus.QUEUED,
                selected_axis="x",
                target_block_dimension=10,
                voxel_size=0.1,
                palette_version="geometry-safe-v1",
                minecraft_version="26.2",
                created_at=now,
                updated_at=now,
            )
            session.add(export)
            session.commit()

        pipeline.process_export("integration-export")
        with pipeline.sessions() as session:
            export = session.get(Export, "integration-export")
            assert export is not None
            assert ExportStatus(export.status) == ExportStatus.READY
            assert export.occupied_block_count == 4
            assert export.voxel_artifact_key is not None
            assert export.world_zip_key is not None
            export_metrics = (
                session.query(PipelineStageMetric)
                .filter(PipelineStageMetric.job_kind == "export")
                .all()
            )
            assert {"VOXELIZING", "GENERATING_WORLD", "END_TO_END"} <= {
                metric.stage for metric in export_metrics
            }
            voxel_metric = next(metric for metric in export_metrics if metric.stage == "VOXELIZING")
            assert voxel_metric.metrics["occupiedVoxels"] == 4
            assert sum(voxel_metric.metrics["paletteDistribution"].values()) == 4
            world_zip = objects._path(export.world_zip_key)
            with zipfile.ZipFile(world_zip) as archive:
                assert archive.namelist() == ["level.dat"]
            # The default writer needs no Paper server, JAR or plugin.
            worldgen_argv = json.loads((root / "worldgen-argv.json").read_text())
            assert worldgen_argv[:3] == ["generate", "--writer", "direct"]
            assert "--paper-jar" not in worldgen_argv
            world_metric = next(metric for metric in export_metrics if metric.stage == "GENERATING_WORLD")
            assert world_metric.metrics["writer"] == "direct"

        # Both terminal jobs are idempotent and do not duplicate artifacts.
        pipeline.process_scan("integration")
        pipeline.process_export("integration-export")
