"""Small, dependency-free observability primitives shared by API workers."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import time
from collections.abc import Iterator
from typing import Any, Callable


LOGGER = logging.getLogger("minecraft_video")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def event(name: str, **fields: Any) -> None:
    """Emit one flat JSON event suitable for container log collection."""
    payload = {"timestamp": utc_timestamp(), "event": name, **fields}
    LOGGER.info(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))


class StageRecorder:
    """Record a stage in the database and emit matching start/finish log events."""

    def __init__(
        self,
        persist: Callable[..., None],
        *,
        job_kind: str,
        job_id: str,
        scan_id: str,
    ) -> None:
        self.persist = persist
        self.job_kind = job_kind
        self.job_id = job_id
        self.scan_id = scan_id

    @contextmanager
    def stage(self, name: str) -> Iterator[dict[str, Any]]:
        started_at = datetime.now(timezone.utc)
        started = time.monotonic()
        metrics: dict[str, Any] = {}
        event(
            "pipeline.stage.started",
            jobKind=self.job_kind,
            jobId=self.job_id,
            scanId=self.scan_id,
            stage=name,
        )
        try:
            yield metrics
        except Exception as error:
            duration_ms = round((time.monotonic() - started) * 1000)
            self.persist(
                job_kind=self.job_kind,
                job_id=self.job_id,
                scan_id=self.scan_id,
                stage=name,
                status="FAILED",
                started_at=started_at,
                duration_ms=duration_ms,
                failure_reason=type(error).__name__,
                metrics=metrics,
            )
            event(
                "pipeline.stage.finished",
                jobKind=self.job_kind,
                jobId=self.job_id,
                scanId=self.scan_id,
                stage=name,
                status="FAILED",
                durationMs=duration_ms,
                failureReason=type(error).__name__,
                **metrics,
            )
            raise
        else:
            duration_ms = round((time.monotonic() - started) * 1000)
            self.persist(
                job_kind=self.job_kind,
                job_id=self.job_id,
                scan_id=self.scan_id,
                stage=name,
                status="COMPLETED",
                started_at=started_at,
                duration_ms=duration_ms,
                failure_reason=None,
                metrics=metrics,
            )
            event(
                "pipeline.stage.finished",
                jobKind=self.job_kind,
                jobId=self.job_id,
                scanId=self.scan_id,
                stage=name,
                status="COMPLETED",
                durationMs=duration_ms,
                **metrics,
            )
