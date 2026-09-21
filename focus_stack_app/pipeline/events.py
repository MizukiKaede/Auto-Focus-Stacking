"""Thread-safe progress event records shared by workers and the UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any


class PipelineStage(str, Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    ANALYSIS = "analysis"
    ARCHIVE = "archive"
    ALIGNMENT = "alignment"
    FUSION = "fusion"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass(frozen=True)
class PipelineEvent:
    """Immutable snapshot emitted from a worker.

    ``completed``/``total`` refer to the current stage.  ``analysis_*`` and
    ``merge_*`` remain available for a two-bar UI in parallel mode.
    """

    stage: PipelineStage | str
    current_file: str = ""
    current_group: int | str | None = None
    completed: int = 0
    total: int = 0
    groups_found: int = 0
    groups_finished: int = 0
    errors: int = 0
    message: str = ""
    analysis_completed: int = 0
    analysis_total: int = 0
    merge_completed: int = 0
    merge_total: int = 0
    timestamp: float = field(default_factory=time.time)
    # Number of merge jobs consumed, including archive-only classification
    # failures.  ``None`` keeps compatibility with older producers; callers
    # can fall back to groups_finished/merge_completed.
    merge_finished: int | None = None

    @property
    def progress(self) -> float:
        return self.completed / self.total if self.total else 0.0

    @property
    def overall_progress(self) -> float:
        total = self.analysis_total + self.merge_total
        # ``merge_finished`` counts every consumed merge job, including an
        # archive-only result or a failed group.  Older producers did not
        # expose it, so retain the historical fallback for compatibility.
        merge_done = self.merge_completed if self.merge_finished is None else self.merge_finished
        done = self.analysis_completed + merge_done
        return done / total if total else self.progress

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value if isinstance(self.stage, PipelineStage) else str(self.stage),
            "current_file": self.current_file,
            "current_group": self.current_group,
            "completed": self.completed,
            "total": self.total,
            "groups_found": self.groups_found,
            "groups_finished": self.groups_finished,
            "errors": self.errors,
            "message": self.message,
            "analysis_completed": self.analysis_completed,
            "analysis_total": self.analysis_total,
            "merge_completed": self.merge_completed,
            "merge_total": self.merge_total,
            "merge_finished": self.merge_finished,
            "timestamp": self.timestamp,
        }


__all__ = ["PipelineEvent", "PipelineStage"]
