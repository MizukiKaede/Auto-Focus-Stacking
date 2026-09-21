"""Memory-bounded reuse between the focus and blending passes of one stack."""
from __future__ import annotations

from ..utils.memory import memory_snapshot


DEFAULT_ALIGNED_CACHE_BYTES = 1024**3


class AlignedFrameCache:
    """Retain a bounded prefix, then consume it in the second sequential pass.

    An LRU would evict the entire useful cache on a second forward scan when
    a stack exceeds the budget. Keep the first frames instead, and decode
    uncached frames again. Each fusion owns its cache and releases it on exit.
    """

    def __init__(self, loader, *, max_bytes=DEFAULT_ALIGNED_CACHE_BYTES,
                 working_bytes=0, snapshot_fn=None):
        self.loader = loader
        self.snapshot_fn = snapshot_fn or memory_snapshot
        self.frames = {}
        self.bytes_used = self.peak_bytes = self.hits = 0
        self.reserve_bytes = max(0, int(working_bytes))
        self.max_bytes = 0
        if max_bytes < 0:
            raise ValueError("aligned cache limit cannot be negative")
        if max_bytes == 0:
            return
        try:
            snapshot = self.snapshot_fn()
            self.reserve_bytes += max(2 * 1024**3, int(snapshot.total_bytes * 0.10))
            # Leave half the spare RAM for concurrent work, in addition to
            # reserving the temporary arrays needed by the image operations.
            spare = max(0, snapshot.available_bytes - self.reserve_bytes)
            self.max_bytes = min(int(max_bytes), spare // 2)
        except Exception:
            # Unknown memory availability must never enable an unbounded cache.
            pass

    def _check_memory(self):
        if not self.max_bytes:
            return
        try:
            if self.snapshot_fn().available_bytes >= self.reserve_bytes:
                return
        except Exception:
            pass
        self.clear()
        self.max_bytes = 0

    def for_focus(self, index):
        self._check_memory()
        image = self.loader(index)
        self._check_memory()
        if self.bytes_used + image.nbytes <= self.max_bytes:
            self.frames[index] = image
            self.bytes_used += image.nbytes
            self.peak_bytes = max(self.peak_bytes, self.bytes_used)
        return image

    def for_blend(self, index):
        self._check_memory()
        image = self.frames.pop(index, None)
        if image is None:
            return self.loader(index)
        self.bytes_used -= image.nbytes
        self.hits += 1
        return image

    def clear(self):
        self.frames.clear()
        self.bytes_used = 0
