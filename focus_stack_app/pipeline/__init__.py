"""Bounded producer/consumer processing pipeline."""

from .events import PipelineEvent, PipelineStage
from .job_queue import BoundedJobQueue, QueueClosed
from .analysis_worker import AnalysisJob, AnalysisWorker
from .merge_worker import MergeJob, MergeResult, MergeWorker, StackMergeService
from .coordinator import PipelineConfig, PipelineCoordinator, PipelineSummary, Coordinator
from .memory_guard import MemoryGuard
from .application_controller import (
    ApplicationController,
    ApplicationOptions,
    ControllerOptions,
    PipelineOptions,
    ControllerSummary,
    ControllerReport,
    ControllerError,
    DiskSpaceError,
    ControllerCancelled,
    cleanup_stale_temp_dirs,
)

__all__ = [
    "AnalysisJob",
    "AnalysisWorker",
    "BoundedJobQueue",
    "Coordinator",
    "MergeJob",
    "MergeResult",
    "MergeWorker",
    "PipelineConfig",
    "PipelineCoordinator",
    "PipelineEvent",
    "PipelineStage",
    "PipelineSummary",
    "QueueClosed",
    "StackMergeService",
    "ApplicationController",
    "ApplicationOptions",
    "ControllerOptions",
    "PipelineOptions",
    "ControllerSummary",
    "ControllerReport",
    "ControllerError",
    "DiskSpaceError",
    "ControllerCancelled",
    "MemoryGuard",
    "cleanup_stale_temp_dirs",
]
