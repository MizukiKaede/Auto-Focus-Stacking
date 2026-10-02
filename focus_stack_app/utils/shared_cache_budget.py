"""A small process-local budget for arrays retained by concurrent caches.

The owner chooses ``max_bytes`` from the machine's current memory budget and
passes the same instance to each cache that participates. Reservations cover
only retained cache arrays; callers must leave room for each worker's active
decode, warp, focus maps, and blend buffers when choosing the pool size.
"""
from __future__ import annotations

import threading


class SharedCacheBudget:
    """Atomically limit retained bytes across caches in one Python process."""

    def __init__(self, max_bytes: int):
        if int(max_bytes) < 0:
            raise ValueError("shared cache budget cannot be negative")
        self.requested_max_bytes = int(max_bytes)
        self.max_bytes = int(max_bytes)
        self.bytes_used = 0
        self.peak_bytes = 0
        self._lock = threading.Lock()

    def configure_cap(self, max_bytes: int) -> int:
        """Apply a new request without growing a pool that is currently used.

        An active pool may be tightened immediately. A larger or replacement
        cap applies once all cache reservations have been released.
        """
        requested = int(max_bytes)
        if requested < 0:
            raise ValueError("shared cache budget cannot be negative")
        with self._lock:
            self.requested_max_bytes = requested
            if self.bytes_used == 0:
                self.max_bytes = requested
            elif requested < self.max_bytes:
                self.max_bytes = requested
            return self.max_bytes

    def try_reserve(self, size: int) -> bool:
        """Reserve ``size`` bytes, returning false when the shared cap is full."""
        size = int(size)
        if size < 0:
            raise ValueError("shared cache reservation cannot be negative")
        with self._lock:
            if size > self.max_bytes - self.bytes_used:
                return False
            self.bytes_used += size
            self.peak_bytes = max(self.peak_bytes, self.bytes_used)
            return True

    def release(self, size: int) -> None:
        """Release a prior reservation; mismatched accounting fails loudly."""
        size = int(size)
        if size < 0:
            raise ValueError("shared cache release cannot be negative")
        with self._lock:
            if size > self.bytes_used:
                raise ValueError("shared cache release exceeds reserved bytes")
            self.bytes_used -= size


_process_budget: SharedCacheBudget | None = None
_process_budget_lock = threading.Lock()


def process_global_cache_budget(max_bytes: int) -> SharedCacheBudget:
    """Return the one shared cache pool for this process.

    The first caller creates the pool. Later requests configure its cap: an
    active pool can be tightened, and an idle pool can be resized. Applications
    can call this while building worker services, then pass the returned
    instance to each participating image cache.
    """
    global _process_budget
    requested = int(max_bytes)
    if requested < 0:
        raise ValueError("shared cache budget cannot be negative")
    with _process_budget_lock:
        if _process_budget is None:
            _process_budget = SharedCacheBudget(requested)
        else:
            _process_budget.configure_cap(requested)
        return _process_budget
