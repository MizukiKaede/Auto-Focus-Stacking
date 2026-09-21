"""Cancellable memory-pressure gate for expensive pipeline stages.

The scanner and archive paths are intentionally not gated: they only move or
copy one source file at a time.  Analysis and Hugin/Enfuse jobs can decode
several full-size images, so a worker asks this gate before starting each new
job.  Waiting uses ``Event.wait`` rather than sleeping, which makes a stop
request responsive even when the machine remains under pressure.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from ..utils.memory import MemorySnapshot, has_memory_headroom, memory_snapshot
from .events import PipelineEvent, PipelineStage


DEFAULT_MINIMUM_BYTES = 2 * 1024**3
DEFAULT_MINIMUM_FRACTION = 0.10


class MemoryGuard:
    """Wait until the system has safe memory headroom or cancellation wins.

    ``snapshot_fn`` is injectable so the wait policy can be tested without
    allocating gigabytes.  ``event_callback`` receives a throttled diagnostic
    progress event while waiting; callback failures never prevent cancellation
    or allow a worker to start unexpectedly.
    """

    def __init__(
        self,
        *,
        minimum_bytes: int = DEFAULT_MINIMUM_BYTES,
        minimum_fraction: float = DEFAULT_MINIMUM_FRACTION,
        snapshot_fn: Callable[[], MemorySnapshot] | None = None,
        memory_snapshot_fn: Callable[[], MemorySnapshot] | None = None,
        poll_interval: float = 0.5,
        notice_interval: float = 2.0,
        event_callback: Callable[[PipelineEvent], Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if int(minimum_bytes) < 0:
            raise ValueError("minimum_bytes cannot be negative")
        if not 0 <= float(minimum_fraction) <= 1:
            raise ValueError("minimum_fraction must be between 0 and 1")
        if snapshot_fn is not None and memory_snapshot_fn is not None:
            raise ValueError("provide only one of snapshot_fn or memory_snapshot_fn")
        self.minimum_bytes = int(minimum_bytes)
        self.minimum_fraction = float(minimum_fraction)
        self.snapshot_fn = snapshot_fn or memory_snapshot_fn or memory_snapshot
        self.poll_interval = max(0.01, float(poll_interval))
        self.notice_interval = max(0.0, float(notice_interval))
        self.event_callback = event_callback
        self.logger = logger or logging.getLogger(__name__)
        self._notice_lock = threading.Lock()
        self._last_notice = 0.0

    def snapshot(self) -> MemorySnapshot:
        """Read one current memory snapshot through the injected provider."""

        return self.snapshot_fn()

    def has_headroom(
        self,
        *,
        extra_bytes: int = 0,
        snapshot: MemorySnapshot | None = None,
    ) -> bool:
        """Return whether a new expensive task may start right now."""

        current = snapshot if snapshot is not None else self.snapshot()
        return has_memory_headroom(
            minimum_bytes=self.minimum_bytes,
            minimum_fraction=self.minimum_fraction,
            extra_bytes=max(0, int(extra_bytes)),
            snapshot=current,
        )

    can_start = has_headroom

    def _notify_low_memory(
        self,
        current: MemorySnapshot | None,
        *,
        stage: PipelineStage | str,
        current_file: str = "",
        current_group: int | str | None = None,
    ) -> None:
        callback = self.event_callback
        now = time.monotonic()
        with self._notice_lock:
            if self.notice_interval and now - self._last_notice < self.notice_interval:
                return
            self._last_notice = now
        if current is None:
            detail = "内存状态不可用"
        else:
            detail = f"可用 {current.available_bytes / 1024**3:.2f} GiB"
        message = (
            f"{detail}，低于安全阈值 {self.minimum_bytes / 1024**3:.2f} GiB/"
            f"{self.minimum_fraction:.0%}；暂停新的高内存任务（可取消）"
        )
        self.logger.warning(message)
        if callback is None:
            return
        event = PipelineEvent(
            stage=stage,
            current_file=current_file,
            current_group=current_group,
            message=message,
        )
        try:
            callback(event)
        except Exception:
            self.logger.debug("Memory-pressure event callback failed", exc_info=True)

    def wait(
        self,
        cancel_event: threading.Event | None = None,
        *,
        stage: PipelineStage | str = PipelineStage.ANALYSIS,
        current_file: str = "",
        current_group: int | str | None = None,
        extra_bytes: int = 0,
    ) -> bool:
        """Block until headroom is available; return ``False`` on cancel.

        A snapshot provider error is treated conservatively as no headroom,
        because starting a large decode while memory is unknown is unsafe.
        The cancellation event remains authoritative and is checked before
        every snapshot and during the bounded wait interval.
        """

        cancelled = cancel_event or threading.Event()
        while not cancelled.is_set():
            current: MemorySnapshot | None
            try:
                current = self.snapshot()
                allowed = self.has_headroom(extra_bytes=extra_bytes, snapshot=current)
            except Exception:
                current = None
                allowed = False
                self.logger.warning("Unable to inspect memory; delaying high-memory task", exc_info=True)
            if allowed:
                return True
            self._notify_low_memory(
                current,
                stage=stage,
                current_file=current_file,
                current_group=current_group,
            )
            cancelled.wait(self.poll_interval)
        return False

    wait_for_headroom = wait
    before = wait


__all__ = [
    "DEFAULT_MINIMUM_BYTES",
    "DEFAULT_MINIMUM_FRACTION",
    "MemoryGuard",
]
