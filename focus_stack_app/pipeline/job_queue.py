"""A bounded, cancellation-aware queue for analysis -> merge jobs."""

from __future__ import annotations

from collections import deque
import queue
import threading
import time
from typing import Generic, Iterator, TypeVar


T = TypeVar("T")


class QueueClosed(RuntimeError):
    """Raised when a producer attempts to put after queue shutdown."""


_SENTINEL = object()


class BoundedJobQueue(Generic[T]):
    """Bounded queue with explicit close and optional cancellation event."""

    def __init__(self, maxsize: int = 3, *, cancel_event: threading.Event | None = None):
        if int(maxsize) < 1:
            raise ValueError("maxsize must be at least one")
        self.maxsize = int(maxsize)
        self.cancel_event = cancel_event
        self._queue: queue.Queue[object] = queue.Queue(maxsize=self.maxsize)
        self._state_lock = threading.Lock()
        self._closed = False
        self._sentinel_enqueued = False

    @property
    def closed(self) -> bool:
        with self._state_lock:
            return self._closed

    @property
    def qsize(self) -> int:
        """Approximate queued item count, useful for bounded-pipeline telemetry."""
        return self._queue.qsize()

    def put(self, item: T, *, timeout: float | None = None) -> None:
        if item is _SENTINEL:
            raise ValueError("sentinel is reserved")
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            with self._state_lock:
                if self._closed:
                    raise QueueClosed("job queue is closed")
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise QueueClosed("job queue cancelled")
            wait = 0.10
            if deadline is not None:
                wait = min(wait, max(0.0, deadline - time.monotonic()))
                if wait <= 0:
                    raise queue.Full
            try:
                self._queue.put(item, timeout=wait)
                return
            except queue.Full:
                if deadline is not None and time.monotonic() >= deadline:
                    raise

    def get(self, *, timeout: float | None = None) -> T | None:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            wait = 0.10 if timeout is None else min(0.10, max(0.0, deadline - time.monotonic()))
            if timeout is not None and wait <= 0:
                raise queue.Empty
            try:
                item = self._queue.get(timeout=wait)
            except queue.Empty:
                if timeout is not None and time.monotonic() >= deadline:
                    raise
                if self.cancel_event is not None and self.cancel_event.is_set() and self.closed:
                    raise QueueClosed("job queue cancelled")
                continue
            if item is _SENTINEL:
                self._queue.task_done()
                return None
            return item  # type: ignore[return-value]

    def task_done(self) -> None:
        self._queue.task_done()

    def join(self) -> None:
        self._queue.join()

    def close(self, *, wait: bool = True, consumers: int = 1) -> None:
        """Stop producers and enqueue one consumer sentinel.

        ``wait`` controls whether this method waits for room when all bounded
        slots are full.  The default guarantees a consumer eventually exits;
        ``wait=False`` is useful while cancelling a coordinator.
        """

        with self._state_lock:
            self._closed = True
            if self._sentinel_enqueued:
                return
            self._sentinel_enqueued = True
        count = max(1, int(consumers))
        inserted = 0
        while inserted < count:
            try:
                self._queue.put(_SENTINEL, timeout=0.10 if wait else 0.0)
                inserted += 1
            except queue.Full:
                if not wait:
                    with self._state_lock:
                        self._sentinel_enqueued = False
                    return

    shutdown = close

    def drain(self) -> list[T]:
        drained: list[T] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not _SENTINEL:
                drained.append(item)  # type: ignore[arg-type]
            self._queue.task_done()
        return drained


BoundedQueue = BoundedJobQueue


__all__ = ["BoundedJobQueue", "BoundedQueue", "QueueClosed"]

