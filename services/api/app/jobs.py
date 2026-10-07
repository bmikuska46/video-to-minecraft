from __future__ import annotations

from typing import Protocol

from celery import Celery

from .config import Settings


class JobDispatcher(Protocol):
    def ready(self) -> None: ...

    def enqueue_reconstruction(self, scan_id: str) -> None: ...

    def enqueue_export(self, export_id: str) -> None: ...


class CeleryJobDispatcher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.app = Celery("video-to-minecraft-api", broker=settings.celery_broker_url)

    def ready(self) -> None:
        connection = self.app.connection_for_read()
        try:
            connection.ensure_connection(max_retries=0, timeout=2)
        finally:
            connection.release()

    def enqueue_reconstruction(self, scan_id: str) -> None:
        self.app.send_task(
            self.settings.celery_task_name,
            args=[scan_id],
            task_id=f"reconstruct-{scan_id}",
        )

    def enqueue_export(self, export_id: str) -> None:
        self.app.send_task(
            self.settings.celery_export_task_name,
            args=[export_id],
            task_id=f"export-{export_id}",
            queue=self.settings.celery_export_queue,
        )
