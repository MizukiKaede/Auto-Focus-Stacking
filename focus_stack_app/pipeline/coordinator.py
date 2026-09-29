"""Coordinator for bounded analysis/merge producer-consumer execution."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable

from .analysis_worker import AnalysisWorker
from .events import PipelineEvent, PipelineStage
from .job_queue import BoundedJobQueue
from .merge_worker import MergeWorker


@dataclass(frozen=True)
class PipelineConfig:
    queue_size: int = 3
    merge_workers: int = 3
    parallel: bool = True

    def __post_init__(self) -> None:
        if int(self.queue_size) < 1:
            raise ValueError("queue_size must be at least one")
        if not 1 <= int(self.merge_workers) <= 6:
            raise ValueError("merge_workers must be between 1 and 6")


@dataclass
class PipelineSummary:
    groups_total: int = 0
    analysis_completed: int = 0
    merge_completed: int = 0
    errors: int = 0
    cancelled: bool = False
    failed_analysis: list[Any] = field(default_factory=list)
    failed_merge: list[Any] = field(default_factory=list)
    results: list[Any] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    # Number of merge-queue jobs fully consumed.  This intentionally differs
    # from ``merge_completed``: an analysis failure is archive-only and must
    # not count as a successful fusion, while it still does finish the
    # group's required archive work.  Kept last for positional compatibility
    # with the original summary constructor.
    merge_finished: int | None = None

    @property
    def succeeded(self) -> int:
        return max(0, self.merge_completed - len(self.failed_merge))

    @property
    def finished(self) -> bool:
        merge_finished = self.merge_completed if self.merge_finished is None else self.merge_finished
        return self.cancelled or (
            self.analysis_completed >= self.groups_total
            and merge_finished >= self.groups_total
        )


class PipelineCoordinator:
    """Run one analysis producer and a bounded number of merge consumers.

    In parallel mode the merge worker starts before analysis and consumes jobs
    as they become available.  With ``parallel=False`` analysis is fully
    drained before the merge worker starts, while the same bounded queue and
    cancellation semantics are retained.
    """

    def __init__(
        self,
        *,
        analyzer: Callable[..., Any] | Any | None = None,
        merger: Callable[..., Any] | Any | None = None,
        config: PipelineConfig | Any | None = None,
        queue_size: int | None = None,
        merge_workers: int | None = None,
        parallel: bool | None = None,
        repository: Any | None = None,
        event_callback: Callable[[PipelineEvent], Any] | None = None,
        complete_callback: Callable[[PipelineSummary], Any] | None = None,
        manifest_writer: Any | None = None,
        manifest_path: Any | None = None,
        memory_guard: Any | None = None,
        job_state: Any | None = None,
        logger: logging.Logger | None = None,
    ):
        if config is None:
            config = PipelineConfig(
                queue_size=3 if queue_size is None else queue_size,
                merge_workers=3 if merge_workers is None else merge_workers,
                parallel=True if parallel is None else parallel,
            )
        else:
            config = PipelineConfig(
                queue_size=getattr(config, "queue_size", getattr(config, "merge_queue_size", 3)) if queue_size is None else queue_size,
                merge_workers=getattr(config, "merge_workers", getattr(config, "max_hugin_workers", 3)) if merge_workers is None else merge_workers,
                parallel=getattr(config, "parallel", getattr(config, "parallel_pipeline", True)) if parallel is None else parallel,
            )
        self.config = config
        self.analyzer = analyzer
        self.merger = merger
        self.repository = repository
        self.event_callback = event_callback
        self.complete_callback = complete_callback
        if manifest_writer is None and manifest_path is not None:
            # Keep this import lazy so headless pipeline use does not acquire
            # storage dependencies until manifest output is requested.
            from ..storage.manifest import ManifestWriter

            manifest_writer = ManifestWriter(manifest_path)
        self.manifest_writer = manifest_writer
        self.memory_guard = memory_guard
        self.job_state = job_state
        self._manifest_lock = threading.RLock()
        self.logger = logger or logging.getLogger(__name__)
        self.cancel_event = threading.Event()
        self._queue: BoundedJobQueue[Any] | None = None
        self._controller: threading.Thread | None = None
        self._state_lock = threading.RLock()
        self._done = threading.Event()
        self._groups: list[Any] = []
        self._summary: PipelineSummary | None = None
        self._analysis_worker: AnalysisWorker | None = None
        self._merge_workers: list[MergeWorker] = []
        self._started_at = 0.0

    @property
    def running(self) -> bool:
        return self._controller is not None and self._controller.is_alive()

    @property
    def summary(self) -> PipelineSummary | None:
        return self._summary

    @property
    def queue(self) -> BoundedJobQueue[Any] | None:
        return self._queue

    def start(
        self,
        groups: Iterable[Any] | None = None,
        *,
        analyzer: Callable[..., Any] | Any | None = None,
        merger: Callable[..., Any] | Any | None = None,
    ) -> "PipelineCoordinator":
        """Start asynchronously; return immediately to keep UI responsive."""

        with self._state_lock:
            if self.running:
                raise RuntimeError("pipeline is already running")
            if groups is not None:
                self._groups = list(groups)
            if not self._groups:
                self._groups = []
            if analyzer is not None:
                self.analyzer = analyzer
            if merger is not None:
                self.merger = merger
            if self.analyzer is None:
                raise ValueError("analyzer is required")
            if self.merger is None:
                raise ValueError("merger is required")
            self.cancel_event.clear()
            self._done.clear()
            self._summary = None
            self._started_at = time.monotonic()
            self._controller = threading.Thread(target=self.run, name="focus-stack-coordinator", daemon=True)
            self._controller.start()
        return self

    def run(
        self,
        groups: Iterable[Any] | None = None,
        *,
        analyzer: Callable[..., Any] | Any | None = None,
        merger: Callable[..., Any] | Any | None = None,
    ) -> PipelineSummary:
        """Execute synchronously when called directly, or by ``start``'s thread."""

        # Direct invocation is useful in CLI/tests.  Do not reset arguments
        # when this method is called by start(), which populated _groups.
        with self._state_lock:
            if groups is not None:
                self._groups = list(groups)
            if analyzer is not None:
                self.analyzer = analyzer
            if merger is not None:
                self.merger = merger
            group_values = list(self._groups)
            if self.analyzer is None or self.merger is None:
                raise ValueError("analyzer and merger are required")
            if self._started_at <= 0:
                self._started_at = time.monotonic()
        summary = PipelineSummary(groups_total=len(group_values))
        self._queue = BoundedJobQueue(self.config.queue_size, cancel_event=self.cancel_event)
        analysis_done = threading.Event()
        analysis = AnalysisWorker(
            group_values,
            self.analyzer,
            self._queue,
            cancel_event=self.cancel_event,
            event_callback=self._on_worker_event,
            repository=self.repository,
            total=len(group_values),
            logger=self.logger,
            memory_guard=self.memory_guard,
            job_state=self.job_state,
        )
        merge_workers: list[MergeWorker] = [
            MergeWorker(
                self._queue,
                self.merger,
                cancel_event=self.cancel_event,
                event_callback=self._on_worker_event,
                repository=self.repository,
                total=len(group_values),
                defer_until=analysis_done if not self.config.parallel else None,
                result_callback=self._on_merge_result,
                logger=self.logger,
                memory_guard=self.memory_guard,
                job_state=self.job_state,
            )
            for _ in range(self.config.merge_workers)
        ]
        self._analysis_worker = analysis
        self._merge_workers = merge_workers
        try:
            # Start consumers in both modes.  In serial mode they only drain
            # (and hold) jobs until ``analysis_done`` is set, so the producer
            # can never block forever on a full bounded queue.
            for worker in merge_workers:
                worker.start()
            analysis.start()
            analysis.join()
            analysis_done.set()
            # One sentinel per consumer.  During cancellation no wait is
            # necessary; the queue's cancellation-aware get wakes workers.
            self._queue.close(wait=not self.cancel_event.is_set(), consumers=len(merge_workers))
            for worker in merge_workers:
                worker.join()
            summary.analysis_completed = analysis.completed
            summary.merge_completed = sum(worker.completed for worker in merge_workers)
            summary.merge_finished = sum(worker.finished for worker in merge_workers)
            summary.errors = analysis.errors + sum(worker.errors for worker in merge_workers)
            summary.cancelled = self.cancel_event.is_set()
            summary.failed_analysis = list(analysis.failed_jobs)
            summary.failed_merge = [item for worker in merge_workers for item in worker.failed_jobs]
            summary.results = [item for worker in merge_workers for item in worker.results]
            summary.elapsed_seconds = time.monotonic() - self._started_at
            self._summary = summary
            self._ensure_manifest()
            self._on_worker_event(
                PipelineEvent(
                    stage=PipelineStage.CANCELLED if summary.cancelled else PipelineStage.COMPLETE,
                    completed=summary.merge_completed,
                    total=len(group_values),
                    groups_found=analysis.discovered,
                    groups_finished=summary.merge_finished,
                    errors=summary.errors,
                    message="处理已取消" if summary.cancelled else "处理完成",
                    analysis_completed=summary.analysis_completed,
                    analysis_total=len(group_values),
                    merge_completed=summary.merge_completed,
                    merge_total=len(group_values),
                    merge_finished=summary.merge_finished,
                )
            )
            if self.complete_callback is not None:
                try:
                    self.complete_callback(summary)
                except Exception:
                    self.logger.debug("Pipeline completion callback failed", exc_info=True)
            return summary
        finally:
            # Release deferred merge workers even if an unexpected producer or
            # shutdown exception reaches this scope.
            analysis_done.set()
            self._done.set()

    def wait(self, timeout: float | None = None) -> PipelineSummary | None:
        """Wait for asynchronous completion and return the summary."""

        if self._controller is not None and self._controller is not threading.current_thread():
            self._controller.join(timeout=timeout)
        if self._done.is_set():
            return self._summary
        return None

    join = wait

    def cancel(self) -> None:
        """Request graceful cancellation of new work and current subprocesses."""

        self.cancel_event.set()
        if self._queue is not None:
            self._queue.close(wait=False, consumers=max(1, self.config.merge_workers))

    stop = cancel

    def _on_merge_result(self, result: Any) -> None:
        """Persist one completed/failed group without affecting workers."""

        writer = self.manifest_writer
        if writer is None:
            merger = self.merger
            writer = getattr(merger, "manifest_writer", None) if merger is not None else None
        if writer is None:
            return
        try:
            with self._manifest_lock:
                update = getattr(writer, "update_result", None)
                if callable(update):
                    update(result)
                    return
                append = getattr(writer, "append_result", None)
                if callable(append):
                    append(result)
                    return
                # Minimal adapter fallback for early manifest writers.  This
                # writes a valid per-result file even when no stateful upsert
                # API is available.
                rows = getattr(writer, "rows_from_result", None)
                write_csv = getattr(writer, "write_csv", None)
                if callable(rows) and callable(write_csv):
                    write_csv(rows(result))
        except Exception:
            self.logger.debug("Manifest update failed", exc_info=True)

    def _ensure_manifest(self) -> None:
        """Create an empty/header manifest for an empty or cancelled batch."""

        writer = self.manifest_writer
        if writer is None:
            merger = self.merger
            writer = getattr(merger, "manifest_writer", None) if merger is not None else None
        if writer is None:
            return
        target = getattr(writer, "path", None)
        write_csv = getattr(writer, "write_csv", None)
        if target is None or not callable(write_csv):
            return
        try:
            target = Path(target)
            if not target.exists():
                with self._manifest_lock:
                    if not target.exists():
                        write_csv([], target)
        except Exception:
            self.logger.debug("Unable to finalize manifest", exc_info=True)

    def _on_worker_event(self, event: PipelineEvent) -> None:
        if self.event_callback is None:
            return
        analysis = self._analysis_worker
        merges = self._merge_workers
        analysis_completed = analysis.completed if analysis is not None else event.analysis_completed
        analysis_total = analysis.total if analysis is not None else event.analysis_total
        merge_completed = sum(worker.completed for worker in merges) if merges else event.merge_completed
        merge_finished = sum(worker.finished for worker in merges) if merges else getattr(
            event, "groups_finished", merge_completed
        )
        merge_total = merges[0].total if merges else event.merge_total
        combined = PipelineEvent(
            stage=event.stage,
            current_file=event.current_file,
            current_group=event.current_group,
            completed=event.completed,
            total=event.total,
            groups_found=analysis.discovered if analysis is not None else event.groups_found,
            groups_finished=merge_finished,
            errors=(analysis.errors if analysis is not None else 0) + sum(worker.errors for worker in merges),
            message=event.message,
            analysis_completed=analysis_completed,
            analysis_total=analysis_total,
            merge_completed=merge_completed,
            merge_total=merge_total,
            merge_finished=merge_finished,
        )
        try:
            self.event_callback(combined)
        except Exception:
            self.logger.debug("Pipeline progress callback failed", exc_info=True)


Coordinator = PipelineCoordinator
Pipeline = PipelineCoordinator


__all__ = ["Coordinator", "Pipeline", "PipelineConfig", "PipelineCoordinator", "PipelineSummary"]

