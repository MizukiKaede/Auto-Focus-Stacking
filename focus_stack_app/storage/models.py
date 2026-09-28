"""Typed records used by the SQLite repository and worker boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ImageStatus(_ValueEnum):
    DISCOVERED = "DISCOVERED"
    CLASSIFYING = "CLASSIFYING"
    CLASSIFIED = "CLASSIFIED"
    ANALYZING = "ANALYZING_FOCUS"
    SELECTED = "SELECTED"
    REJECTED = "REJECTED"
    ARCHIVING = "ARCHIVING"
    ARCHIVED = "ARCHIVED"
    FAILED = "FAILED"


class JobStage(_ValueEnum):
    CLASSIFICATION = "CLASSIFICATION"
    FOCUS_ANALYSIS = "FOCUS_ANALYSIS"
    ARCHIVE = "ARCHIVE"
    ALIGNMENT = "ALIGNMENT"
    FUSION = "FUSION"
    EXPORT = "EXPORT"


class JobStatus(_ValueEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


def _value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


@dataclass(slots=True)
class ImageRecord:
    """Metadata for one source image.

    The scanner populates this record from ``stat``, JPEG headers and EXIF;
    no decoded pixel buffer is stored here.  Paths are strings at the storage
    boundary to avoid platform-specific SQLite adapters.
    """

    original_path: str
    id: int | None = None
    current_path: str | None = None
    filename: str | None = None
    stem: str | None = None
    extension: str | None = None
    file_size: int = 0
    mtime: float = 0.0
    capture_time: str | None = None
    width: int | None = None
    height: int | None = None
    camera: str | None = None
    lens: str | None = None
    sequence_index: int | None = None
    group_id: int | None = None
    status: str = ImageStatus.DISCOVERED.value
    archive_mode: str | None = None
    # ``pending_path`` is written before a move/copy/link starts.  Keeping it
    # separate from ``current_path`` means a crash cannot make an uncommitted
    # destination look like the live image location.
    pending_path: str | None = None
    archive_operation_id: str | None = None
    archive_error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        path = Path(self.original_path)
        if self.current_path is None:
            self.current_path = self.original_path
        if self.filename is None:
            self.filename = path.name
        if self.stem is None:
            self.stem = path.stem
        if self.extension is None:
            self.extension = path.suffix.lower()

    @classmethod
    def from_path(cls, path: str | Path, **metadata: Any) -> "ImageRecord":
        """Construct a lightweight record from a path plus optional fields."""

        return cls(original_path=str(path), **metadata)

    @property
    def path(self) -> Path:
        """Current path, useful to worker code after archiving."""

        return Path(self.current_path or self.original_path)

    @property
    def is_archived(self) -> bool:
        return self.status == ImageStatus.ARCHIVED.value

    @property
    def pending_destination(self) -> Path | None:
        """Destination reserved by an in-flight archive, if any."""

        return Path(self.pending_path) if self.pending_path else None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "original_path": self.original_path,
            "current_path": self.current_path or self.original_path,
            "filename": self.filename,
            "stem": self.stem,
            "extension": self.extension,
            "file_size": int(self.file_size),
            "mtime": float(self.mtime),
            "capture_time": self.capture_time,
            "width": self.width,
            "height": self.height,
            "camera": self.camera,
            "lens": self.lens,
            "sequence_index": self.sequence_index,
            "group_id": self.group_id,
            "status": _value(self.status),
            "archive_mode": self.archive_mode,
            "pending_path": self.pending_path,
            "archive_operation_id": self.archive_operation_id,
            "archive_error": self.archive_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ImageRecord":
        names = {
            "id",
            "original_path",
            "current_path",
            "filename",
            "stem",
            "extension",
            "file_size",
            "mtime",
            "capture_time",
            "width",
            "height",
            "camera",
            "lens",
            "sequence_index",
            "group_id",
            "status",
            "archive_mode",
            "pending_path",
            "archive_operation_id",
            "archive_error",
            "created_at",
            "updated_at",
        }
        available = set(row.keys())
        return cls(**{name: row[name] for name in names if name in available})


@dataclass(slots=True)
class GroupRecord:
    """A logical contiguous scene group; grouping does not create folders."""

    id: int | None = None
    first_image_id: int | None = None
    start_index: int | None = None
    end_index: int | None = None
    image_count: int = 0
    selected_count: int = 0
    confidence: float | None = None
    coverage: float | None = None
    status: str = "DISCOVERED"
    output_path: str | None = None
    preview_reference: str | None = None
    alignment_order: Any = None
    alignment_order_confidence: float | None = None
    alignment_order_fallback_used: bool = False
    pairwise_analysis_summary: Any = None
    requested_backend: str | None = None
    actual_backend: str | None = None
    alignment_level: int | None = None
    alignment_status: str | None = None
    crop_ratio: float | None = None
    diagnostics: Any = None
    created_at: str | None = None
    updated_at: str | None = None

    @property
    def is_single(self) -> bool:
        return self.image_count <= 1

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "first_image_id": self.first_image_id,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "image_count": self.image_count,
            "selected_count": self.selected_count,
            "confidence": self.confidence,
            "coverage": self.coverage,
            "status": _value(self.status),
            "output_path": self.output_path,
            "preview_reference": self.preview_reference,
            "alignment_order": self.alignment_order,
            "alignment_order_confidence": self.alignment_order_confidence,
            "alignment_order_fallback_used": bool(self.alignment_order_fallback_used),
            "pairwise_analysis_summary": self.pairwise_analysis_summary,
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
            "alignment_level": self.alignment_level,
            "alignment_status": self.alignment_status,
            "crop_ratio": self.crop_ratio,
            "diagnostics": self.diagnostics,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "GroupRecord":
        names = {
            "id",
            "first_image_id",
            "start_index",
            "end_index",
            "image_count",
            "selected_count",
            "confidence",
            "coverage",
            "status",
            "output_path",
            "preview_reference", "alignment_order", "alignment_order_confidence",
            "alignment_order_fallback_used", "pairwise_analysis_summary",
            "requested_backend", "actual_backend", "alignment_level",
            "alignment_status", "crop_ratio", "diagnostics",
            "created_at",
            "updated_at",
        }
        available = set(row.keys())
        values = {name: row[name] for name in names if name in available}
        for name in ("alignment_order", "pairwise_analysis_summary", "diagnostics"):
            if name in values and isinstance(values[name], str):
                try:
                    import json
                    values[name] = json.loads(values[name])
                except (TypeError, ValueError):
                    pass
        values["alignment_order_fallback_used"] = bool(values.get("alignment_order_fallback_used", False))
        return cls(**values)


@dataclass(slots=True)
class AnalysisRecord:
    """Cached per-image analysis values.

    ``transform`` is JSON-serializable data (typically a matrix/list), but is
    kept as a Python value at this boundary for algorithm compatibility.
    """

    image_id: int
    scene_hash: str | None = None
    scene_score: float | None = None
    sharpness_score: float | None = None
    focus_map_path: str | None = None
    transform: Any = None
    selected: bool = False
    selection_reason: str | None = None
    coverage_gain: float | None = None
    quality_score: float | None = None
    updated_at: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "scene_hash": self.scene_hash,
            "scene_score": self.scene_score,
            "sharpness_score": self.sharpness_score,
            "focus_map_path": self.focus_map_path,
            "transform": self.transform,
            "selected": bool(self.selected),
            "selection_reason": self.selection_reason,
            "coverage_gain": self.coverage_gain,
            "quality_score": self.quality_score,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "AnalysisRecord":
        names = {
            "image_id",
            "scene_hash",
            "scene_score",
            "sharpness_score",
            "focus_map_path",
            "transform",
            "selected",
            "selection_reason",
            "coverage_gain",
            "quality_score",
            "updated_at",
        }
        available = set(row.keys())
        values = {name: row[name] for name in names if name in available}
        values["selected"] = bool(values.get("selected", False))
        return cls(**values)


@dataclass(slots=True)
class JobRecord:
    """Durable job state used for restart/recovery."""

    id: int | None = None
    group_id: int | None = None
    stage: str = JobStage.CLASSIFICATION.value
    progress: float = 0.0
    status: str = JobStatus.PENDING.value
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    attempt: int = 0
    created_at: str | None = None
    updated_at: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "group_id": self.group_id,
            "stage": _value(self.stage),
            "progress": float(self.progress),
            "status": _value(self.status),
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "attempt": int(self.attempt),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "JobRecord":
        names = {
            "id",
            "group_id",
            "stage",
            "progress",
            "status",
            "error",
            "started_at",
            "finished_at",
            "attempt",
            "created_at",
            "updated_at",
        }
        available = set(row.keys())
        return cls(**{name: row[name] for name in names if name in available})


@dataclass(slots=True)
class ArchiveOperationRecord:
    """Durable journal entry for one flat-file archive operation.

    ``phase`` is deliberately a string because the file archiver has to remain
    usable with lightweight repositories and older projects.  A pending row is
    committed before the filesystem mutation; recovery can therefore compare
    the recorded destination against the source metadata without guessing from
    a basename alone.
    """

    operation_id: str
    source_path: str
    destination_path: str
    mode: str
    image_id: int | None = None
    phase: str = "pending"
    status: str = "pending"
    error: str | None = None
    bytes_transferred: int = 0
    source_size: int | None = None
    source_mtime: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @property
    def source(self) -> Path:
        return Path(self.source_path)

    @property
    def destination(self) -> Path:
        return Path(self.destination_path)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "source_path": self.source_path,
            "destination_path": self.destination_path,
            "mode": _value(self.mode),
            "image_id": self.image_id,
            "phase": _value(self.phase),
            "status": _value(self.status),
            "error": self.error,
            "bytes_transferred": int(self.bytes_transferred),
            "source_size": self.source_size,
            "source_mtime": self.source_mtime,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    as_dict = to_mapping

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ArchiveOperationRecord":
        names = {
            "operation_id",
            "source_path",
            "destination_path",
            "mode",
            "image_id",
            "phase",
            "status",
            "error",
            "bytes_transferred",
            "source_size",
            "source_mtime",
            "started_at",
            "finished_at",
            "created_at",
            "updated_at",
        }
        available = set(row.keys())
        return cls(**{name: row[name] for name in names if name in available})


__all__ = [
    "ImageStatus",
    "JobStage",
    "JobStatus",
    "ImageRecord",
    "GroupRecord",
    "AnalysisRecord",
    "JobRecord",
    "ArchiveOperationRecord",
    "ImageMetadata",
    "Group",
    "Analysis",
    "Job",
    "ArchiveOperation",
]

# Short aliases used by small scripts and prior prototypes.
ImageMetadata = ImageRecord
Group = GroupRecord
Analysis = AnalysisRecord
Job = JobRecord
ArchiveOperation = ArchiveOperationRecord

