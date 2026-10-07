from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VTM_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://vtm:vtm@postgres:5432/vtm"
    s3_endpoint_url: str | None = "http://minio:9000"
    # Optional phone/browser-reachable endpoint used only when signing URLs.
    # Workers continue to use s3_endpoint_url on the private container network.
    s3_public_endpoint_url: str | None = None
    s3_region: str = "us-east-1"
    s3_bucket: str = "video-to-minecraft"
    s3_access_key_id: str = "minioadmin"
    s3_secret_access_key: str = "minioadmin"
    signed_url_ttl_seconds: int = Field(default=900, ge=60, le=3600)
    max_video_bytes: int = Field(default=2 * 1024 * 1024 * 1024, ge=1)
    max_metadata_bytes: int = Field(default=1024 * 1024, ge=1)
    max_imu_bytes: int = Field(default=64 * 1024 * 1024, ge=1)
    max_ar_frames_bytes: int = Field(default=64 * 1024 * 1024, ge=1)
    max_capture_duration_ms: int = Field(default=60_250, ge=1)
    # Rooms and multi-object scenes need a longer walk to cover every wall.
    max_scene_capture_duration_ms: int = Field(default=300_250, ge=1)
    retention_days: int = Field(default=7, ge=1)
    celery_broker_url: str = "redis://redis:6379/0"
    celery_task_name: str = "reconstruction.process_scan"
    celery_export_task_name: str = "worldgen.process_export"
    # Exports are CPU and Java work; a separate queue and worker keep them from
    # waiting behind GPU reconstructions on the single pipeline worker.
    celery_export_queue: str = "export"
    minecraft_version: str = "26.2"
    max_target_blocks: int = Field(default=2048, ge=1)
    block_count_warning_threshold: int = Field(default=1_000_000, ge=1)
    block_count_hard_limit: int = Field(default=5_000_000, ge=1)
    pipeline_work_root: str = "/work/pipeline"
    reconstruction_script: str = "/workspace/services/reconstruction/reconstruct_fixture.py"
    preview_glb_script: str = "/workspace/services/reconstruction/preview_glb.py"
    voxelizer_script: str = "/workspace/services/reconstruction/voxelizer.py"
    palette_path: str = "/workspace/packages/block-palette/palette-v1.json"
    worldgen_runner: str = "/opt/worldgen/worldgen_runner.py"
    worldgen_paper_jar: str = "/opt/paper/paper-26.2-build.112-stable.jar"
    # On the persistent pipeline volume, so Paper is downloaded and patched once.
    worldgen_paper_cache_dir: str = "/work/pipeline/paper-cache"
    worldgen_plugin_jar: str = "/opt/worldgen/worldgen-plugin.jar"
    worldgen_template: str = "/opt/worldgen/server-template"
    worldgen_manifest_key: str = Field(default="development-only-change-this-key-32-bytes", min_length=32)
    pipeline_version: str = "rgb-colmap-4.0.4-v1"

    @model_validator(mode="after")
    def validate_block_limits(self):
        if self.block_count_warning_threshold > self.block_count_hard_limit:
            raise ValueError("block count warning threshold cannot exceed the hard limit")
        return self


def max_capture_duration_ms(settings: Settings, scan_type: str) -> int:
    return settings.max_scene_capture_duration_ms if scan_type == "scene" else settings.max_capture_duration_ms


@lru_cache
def get_settings() -> Settings:
    return Settings()
