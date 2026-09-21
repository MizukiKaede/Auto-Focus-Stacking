"""CPU and memory budgets for bounded focus-analysis parallelism."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import threading
from typing import Any, Iterator

from .memory import MemorySnapshot, memory_snapshot


AUTO_FOCUS_ANALYSIS_WORKERS = 4
MAX_FOCUS_ANALYSIS_WORKERS = 8
FOCUS_ANALYSIS_WORKER_BYTES = 256 * 1024**2


@dataclass(frozen=True, slots=True)
class FocusAnalysisBudget:
    """Resolved image-worker and OpenCV-thread limits for one batch."""

    requested_workers: int
    workers: int
    opencv_threads: int
    logical_cpus: int
    cpu_budget: int
    memory_worker_cap: int
    available_memory_bytes: int
    reserved_memory_bytes: int
    memory_limited: bool


def resolve_focus_analysis_budget(
    requested_workers: int = 0,
    *,
    image_count: int = MAX_FOCUS_ANALYSIS_WORKERS,
    merge_workers: int = 1,
    parallel_pipeline: bool = True,
    minimum_available_bytes: int = 2 * 1024**3,
    minimum_available_fraction: float = 0.10,
    snapshot: MemorySnapshot | None = None,
    logical_cpus: int | None = None,
    current_opencv_threads: int | None = None,
) -> FocusAnalysisBudget:
    """Resolve a safe focus-analysis budget without changing global state.

    ``requested_workers=0`` is automatic.  Manual values are upper bounds:
    the group size and memory safety reserve may reduce the effective count.
    Two logical CPUs are reserved for every concurrent merge worker while the
    producer/consumer pipeline is enabled.
    """

    requested = int(requested_workers)
    if not 0 <= requested <= MAX_FOCUS_ANALYSIS_WORKERS:
        raise ValueError(
            f"focus_analysis_workers must be between 0 and {MAX_FOCUS_ANALYSIS_WORKERS}"
        )
    count = max(1, int(image_count))
    merges = max(0, int(merge_workers))
    cpus = max(1, int(logical_cpus or os.cpu_count() or 1))
    reserved_cpus = 2 * merges if parallel_pipeline else 0
    cpu_budget = max(1, cpus - reserved_cpus)

    if requested == 0:
        # Target roughly four OpenCV threads per in-flight image.  The ceiling
        # keeps a six-core analysis budget at two image workers rather than
        # needlessly serialising it.
        desired = min(
            AUTO_FOCUS_ANALYSIS_WORKERS,
            max(1, (cpu_budget + 3) // 4),
        )
    else:
        desired = min(requested, MAX_FOCUS_ANALYSIS_WORKERS)

    current = snapshot or memory_snapshot()
    reserve = max(
        max(0, int(minimum_available_bytes)),
        int(max(0.0, float(minimum_available_fraction)) * max(0, current.total_bytes)),
    )
    usable = max(0, int(current.available_bytes) - reserve)
    # Keep one worker available even when the memory guard is currently
    # holding the pipeline; it will not start until the same reserve recovers.
    memory_cap = max(1, usable // FOCUS_ANALYSIS_WORKER_BYTES)
    workers = max(1, min(count, desired, memory_cap))

    previous = max(1, int(current_opencv_threads or cpus))
    if workers <= 1:
        opencv_threads = min(previous, cpu_budget)
    else:
        opencv_threads = min(previous, 4, max(1, cpu_budget // workers))

    return FocusAnalysisBudget(
        requested_workers=requested,
        workers=workers,
        opencv_threads=max(1, opencv_threads),
        logical_cpus=cpus,
        cpu_budget=cpu_budget,
        memory_worker_cap=memory_cap,
        available_memory_bytes=max(0, int(current.available_bytes)),
        reserved_memory_bytes=reserve,
        memory_limited=memory_cap < min(count, desired),
    )


_opencv_budget_lock = threading.RLock()


@contextmanager
def opencv_thread_budget(threads: int) -> Iterator[dict[str, Any]]:
    """Temporarily apply the process-global OpenCV thread limit.

    OpenCV exposes one process-wide setting, so overlapping application
    controllers must not race to replace it.  The previous value is restored
    even when analysis, fusion, or cancellation raises.
    """

    with _opencv_budget_lock:
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError:
            yield {"previous": None, "effective": None}
            return
        previous = int(cv2.getNumThreads())
        effective = max(1, int(threads))
        if effective != previous:
            cv2.setNumThreads(effective)
        try:
            yield {"previous": previous, "effective": effective}
        finally:
            if effective != previous:
                cv2.setNumThreads(previous)


__all__ = [
    "AUTO_FOCUS_ANALYSIS_WORKERS",
    "MAX_FOCUS_ANALYSIS_WORKERS",
    "FOCUS_ANALYSIS_WORKER_BYTES",
    "FocusAnalysisBudget",
    "resolve_focus_analysis_budget",
    "opencv_thread_budget",
]
