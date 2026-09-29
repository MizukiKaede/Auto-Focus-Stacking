"""Memory-bounded reuse between the focus and blending passes of one stack."""
from __future__ import annotations

import os
from pathlib import Path

from ..utils.image_io import load_rgb, load_rgb_with_previews
from ..utils.memory import memory_snapshot


DEFAULT_ALIGNED_CACHE_BYTES = 1024**3
DEFAULT_ALIGNED_TIFF_CACHE_BYTES = 512 * 1024**2


class AlignedTIFFImageCache:
    """Reuse validation decodes in the Enfuse mask pass within a RAM budget."""

    def __init__(self, *, max_bytes=DEFAULT_ALIGNED_TIFF_CACHE_BYTES, snapshot_fn=None):
        if max_bytes < 0:
            raise ValueError("aligned TIFF cache limit cannot be negative")
        self.snapshot_fn = snapshot_fn or memory_snapshot
        self.frames = {}
        self.bytes_used = self.peak_bytes = 0
        self.reserve_bytes = 0
        self.max_bytes = 0
        if not max_bytes:
            return
        try:
            snapshot = self.snapshot_fn()
            self.reserve_bytes = max(2 * 1024**3, int(snapshot.total_bytes * 0.10))
            spare = max(0, snapshot.available_bytes - self.reserve_bytes)
            self.max_bytes = min(int(max_bytes), spare // 2)
        except Exception:
            # Unknown memory availability must never enable a large cache.
            pass

    @staticmethod
    def _key(path):
        return os.path.normcase(os.path.abspath(os.fspath(path)))

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

    def validation_previews(self, path, *, edges=(1280, 640)):
        self._check_memory()
        frame, previews = load_rgb_with_previews(path, edges)
        self._check_memory()
        if self.bytes_used + frame.nbytes <= self.max_bytes:
            self.frames[self._key(path)] = frame
            self.bytes_used += frame.nbytes
            self.peak_bytes = max(self.peak_bytes, self.bytes_used)
        return previews

    def load(self, path):
        self._check_memory()
        key = self._key(path)
        # The focus pass is a forward scan. Release each cached RGB frame
        # as it is consumed instead of retaining the entire prefix while
        # grayscale masks and the external Enfuse process allocate memory.
        frame = self.frames.pop(key, None)
        if frame is not None:
            self.bytes_used -= frame.nbytes
            return frame
        return load_rgb(path)

    def clear(self):
        self.frames.clear()
        self.bytes_used = 0


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
