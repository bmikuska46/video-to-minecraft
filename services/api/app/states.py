from __future__ import annotations

from enum import StrEnum


class ScanStatus(StrEnum):
    CREATED = "CREATED"
    UPLOADING = "UPLOADING"
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    EXTRACTING_FRAMES = "EXTRACTING_FRAMES"
    SPARSE_RECONSTRUCTION = "SPARSE_RECONSTRUCTION"
    DENSE_RECONSTRUCTION = "DENSE_RECONSTRUCTION"
    FILTERING = "FILTERING"
    PREVIEW_READY = "PREVIEW_READY"
    FAILED = "FAILED"


class ExportStatus(StrEnum):
    QUEUED = "QUEUED"
    VOXELIZING = "VOXELIZING"
    GENERATING_WORLD = "GENERATING_WORLD"
    READY = "READY"
    FAILED = "FAILED"


SCAN_TRANSITIONS: dict[ScanStatus, frozenset[ScanStatus]] = {
    ScanStatus.CREATED: frozenset({ScanStatus.UPLOADING, ScanStatus.FAILED}),
    ScanStatus.UPLOADING: frozenset({ScanStatus.UPLOADED, ScanStatus.FAILED}),
    ScanStatus.UPLOADED: frozenset({ScanStatus.QUEUED, ScanStatus.FAILED}),
    ScanStatus.QUEUED: frozenset({ScanStatus.EXTRACTING_FRAMES, ScanStatus.FAILED}),
    ScanStatus.EXTRACTING_FRAMES: frozenset(
        {ScanStatus.SPARSE_RECONSTRUCTION, ScanStatus.FAILED}
    ),
    ScanStatus.SPARSE_RECONSTRUCTION: frozenset(
        {ScanStatus.DENSE_RECONSTRUCTION, ScanStatus.FAILED}
    ),
    ScanStatus.DENSE_RECONSTRUCTION: frozenset({ScanStatus.FILTERING, ScanStatus.FAILED}),
    ScanStatus.FILTERING: frozenset({ScanStatus.PREVIEW_READY, ScanStatus.FAILED}),
    ScanStatus.PREVIEW_READY: frozenset(),
    ScanStatus.FAILED: frozenset(),
}

EXPORT_TRANSITIONS: dict[ExportStatus, frozenset[ExportStatus]] = {
    ExportStatus.QUEUED: frozenset({ExportStatus.VOXELIZING, ExportStatus.FAILED}),
    ExportStatus.VOXELIZING: frozenset({ExportStatus.GENERATING_WORLD, ExportStatus.FAILED}),
    ExportStatus.GENERATING_WORLD: frozenset({ExportStatus.READY, ExportStatus.FAILED}),
    ExportStatus.READY: frozenset(),
    ExportStatus.FAILED: frozenset(),
}


class InvalidStateTransition(ValueError):
    pass


def require_scan_transition(current: ScanStatus, target: ScanStatus) -> None:
    if target not in SCAN_TRANSITIONS[current]:
        raise InvalidStateTransition(f"cannot transition scan from {current} to {target}")


def require_export_transition(current: ExportStatus, target: ExportStatus) -> None:
    if target not in EXPORT_TRANSITIONS[current]:
        raise InvalidStateTransition(f"cannot transition export from {current} to {target}")

