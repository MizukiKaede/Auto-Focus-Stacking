"""Memory-bounded reuse between the focus and blending passes of one stack."""
from __future__ import annotations

import os
from pathlib import Path

from ..utils.image_io import load_rgb, load_rgb_with_previews
from ..utils.memory import memory_snapshot
from ..utils.shared_cache_budget import SharedCacheBudget


# Candidate-only larger per-group cap for the combined prefetch/cache trial.
DEFAULT_ALIGNED_CACHE_BYTES = 2 * 1024**3
DEFAULT_ALIGNED_TIFF_CACHE_BYTES = 512 * 1024**2


class AlignedTIFFImageCache:
    """Reuse validation decodes in the Enfuse mask pass within a RAM budget."""

    def __init__(self, *, max_bytes=DEFAULT_ALIGNED_TIFF_CACHE_BYTES, snapshot_fn=None,
                 shared_budget: SharedCacheBudget | None = None):
        if max_bytes < 0:
            raise ValueError("aligned TIFF cache limit cannot be negative")
        self.snapshot_fn = snapshot_fn or memory_snapshot
        self.shared_budget = shared_budget
        self.frames = {}
        self.bytes_used = self.peak_bytes = 0
        self.reserve_bytes = 0
        self.max_bytes = 0
        self.validation_decode_calls = self.load_decode_calls = 0
        self.validation_bytes = self.load_bytes = 0
        self.cache_hits = self.cache_insertions = self.cache_rejections = 0
        self.preparation_peek_calls = self.preparation_peek_hits = 0
        self.preparation_peek_misses = self.preparation_peek_decode_calls = 0
        self.preparation_peek_bytes = self.cache_invalidations = 0
        self._load_counts = {}
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

    @classmethod
    def _signature(cls, path):
        path = Path(path)
        stat = path.stat()
        modified_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))
        return cls._key(path), int(stat.st_size), int(modified_ns)

    def _discard_stale(self, signature):
        path_key = signature[0]
        for old_signature in tuple(self.frames):
            if old_signature[0] != path_key or old_signature == signature:
                continue
            frame = self.frames.pop(old_signature)
            self.bytes_used -= frame.nbytes
            self._release_shared(frame.nbytes)
            self.cache_invalidations += 1

    def _retain(self, signature, frame):
        if signature in self.frames:
            self.cache_rejections += 1
            return False
        if (self.bytes_used + frame.nbytes > self.max_bytes
                or not self._try_reserve_shared(frame.nbytes)):
            self.cache_rejections += 1
            return False
        try:
            self.frames[signature] = frame
        except BaseException:
            self._release_shared(frame.nbytes)
            raise
        self.bytes_used += frame.nbytes
        self.peak_bytes = max(self.peak_bytes, self.bytes_used)
        self.cache_insertions += 1
        return True

    def _remove(self, signature):
        frame = self.frames.pop(signature, None)
        if frame is not None:
            self.bytes_used -= frame.nbytes
            self._release_shared(frame.nbytes)
        return frame

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
        signature = self._signature(path)
        self._discard_stale(signature)
        key = signature[0]
        self.validation_decode_calls += 1
        self._load_counts[key] = self._load_counts.get(key, 0) + 1
        frame, previews = load_rgb_with_previews(path, edges)
        self.validation_bytes += int(frame.nbytes)
        self._check_memory()
        if self._signature(path) == signature:
            self._retain(signature, frame)
        else:
            self.cache_rejections += 1
            self._discard_stale(self._signature(path))
        return previews

    def peek(self, path):
        """Return a TIFF RGB frame without consuming its cached copy.

        A miss is decoded and retained when both the local and shared budgets
        allow it. The path, size, and nanosecond modification time form the
        per-run identity; stale same-path entries release their reservations.
        """
        self._check_memory()
        self.preparation_peek_calls += 1
        signature = self._signature(path)
        self._discard_stale(signature)
        frame = self.frames.get(signature)
        if frame is not None:
            self.preparation_peek_hits += 1
            return frame

        self.preparation_peek_misses += 1
        key = signature[0]
        self.preparation_peek_decode_calls += 1
        self._load_counts[key] = self._load_counts.get(key, 0) + 1
        frame = load_rgb(path)
        self.preparation_peek_bytes += int(frame.nbytes)
        self._check_memory()
        latest_signature = self._signature(path)
        if latest_signature == signature:
            self._retain(signature, frame)
        else:
            self.cache_rejections += 1
            self._discard_stale(latest_signature)
        return frame

    def load(self, path):
        self._check_memory()
        signature = self._signature(path)
        self._discard_stale(signature)
        key = signature[0]
        # The focus pass is a forward scan. Release each cached RGB frame
        # as it is consumed instead of retaining the entire prefix while
        # grayscale masks and the external Enfuse process allocate memory.
        frame = self._remove(signature)
        if frame is not None:
            self.cache_hits += 1
            return frame
        self.load_decode_calls += 1
        self._load_counts[key] = self._load_counts.get(key, 0) + 1
        frame = load_rgb(path)
        self.load_bytes += int(frame.nbytes)
        return frame

    def _try_reserve_shared(self, size):
        return self.shared_budget is None or self.shared_budget.try_reserve(size)

    def _release_shared(self, size):
        if self.shared_budget is not None:
            self.shared_budget.release(size)

    @property
    def stats(self):
        return {
            "cache": "aligned_tiff",
            "cache_limit_bytes": self.max_bytes,
            "cache_bytes_used": self.bytes_used,
            "cache_peak_bytes": self.peak_bytes,
            "validation_decode_calls": self.validation_decode_calls,
            "validation_bytes": self.validation_bytes,
            "consumer_decode_calls": self.load_decode_calls,
            "consumer_decode_bytes": self.load_bytes,
            "preparation_peek_calls": self.preparation_peek_calls,
            "preparation_peek_hits": self.preparation_peek_hits,
            "preparation_peek_misses": self.preparation_peek_misses,
            "preparation_peek_decode_calls": self.preparation_peek_decode_calls,
            "preparation_peek_decode_bytes": self.preparation_peek_bytes,
            "cache_hits": self.cache_hits,
            "cache_insertions": self.cache_insertions,
            "cache_rejections": self.cache_rejections,
            "cache_invalidations": self.cache_invalidations,
            "cache_duplicate_path_decodes": sum(max(0, count - 1)
                                                  for count in self._load_counts.values()),
            "cache_load_counts_by_path": dict(self._load_counts),
            "shared_cache_requested_limit_bytes": (self.shared_budget.requested_max_bytes
                                                    if self.shared_budget is not None else None),
            "shared_cache_limit_bytes": (self.shared_budget.max_bytes
                                         if self.shared_budget is not None else None),
            "shared_cache_bytes_used": (self.shared_budget.bytes_used
                                        if self.shared_budget is not None else None),
            "shared_cache_peak_bytes": (self.shared_budget.peak_bytes
                                        if self.shared_budget is not None else None),
        }

    def clear(self):
        self._release_shared(self.bytes_used)
        self.frames.clear()
        self.bytes_used = 0


class AlignedFrameCache:
    """Retain a bounded prefix, then consume it in the second sequential pass.

    An LRU would evict the entire useful cache on a second forward scan when
    a stack exceeds the budget. Keep the first frames instead, and decode
    uncached frames again. Each fusion owns its cache and releases it on exit.
    """

    def __init__(self, loader, *, max_bytes=DEFAULT_ALIGNED_CACHE_BYTES,
                 working_bytes=0, snapshot_fn=None,
                 shared_budget: SharedCacheBudget | None = None):
        self.loader = loader
        self.snapshot_fn = snapshot_fn or memory_snapshot
        self.shared_budget = shared_budget
        self.frames = {}
        self.bytes_used = self.peak_bytes = self.hits = 0
        self.focus_cache_hits = self.peek_cache_hits = self.peek_loader_calls = 0
        self.focus_loader_calls = self.blend_loader_calls = self.loader_calls = 0
        self.cache_insertions = self.cache_rejections = 0
        self.loaded_bytes = 0
        self._load_counts = {}
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
            snapshot = self.snapshot_fn()
            if snapshot.available_bytes >= self.reserve_bytes:
                return
        except Exception:
            # Unknown memory availability must retain the previous safe
            # behavior: release the complete cache and disable it.
            self.clear()
            self.max_bytes = 0
            return

        # Preserve the useful forward prefix. Free newest retained frames one
        # at a time, then take a fresh snapshot: dropping an ndarray reference
        # does not guarantee that the OS immediately reports that RAM free.
        # Iterating over a fixed tuple bounds this loop by the cache size at
        # entry, even though the backing dictionary is being drained.
        for index in reversed(tuple(self.frames)):
            image = self.frames.pop(index)
            size = int(image.nbytes)
            self.bytes_used -= size
            self._release_shared(size)
            del image
            try:
                snapshot = self.snapshot_fn()
            except Exception:
                # A failed recheck means headroom is unknown; clear all
                # remaining entries and preserve the existing fail-closed
                # behavior.
                self.clear()
                self.max_bytes = 0
                return
            if snapshot.available_bytes >= self.reserve_bytes:
                # Do not refill the evicted tail later, even if RAM recovers.
                self.max_bytes = min(self.max_bytes, self.bytes_used)
                return

        # The measured deficit remained after every retained frame was
        # released. Keep the cache disabled for the rest of this stack.
        self.max_bytes = 0

    def for_focus(self, index):
        self._check_memory()
        image = self.frames.get(index)
        if image is not None:
            # Auxiliary ownership/edge passes can request frames already seen
            # in the forward focus scan. These consumers only inspect RGB;
            # returning the retained array avoids another decode and warp.
            self.focus_cache_hits += 1
            return image
        image = self._load(index, "focus")
        self._check_memory()
        if (self.bytes_used + image.nbytes <= self.max_bytes
                and self._try_reserve_shared(image.nbytes)):
            try:
                self.frames[index] = image
            except BaseException:
                self._release_shared(image.nbytes)
                raise
            self.bytes_used += image.nbytes
            self.peak_bytes = max(self.peak_bytes, self.bytes_used)
            self.cache_insertions += 1
        else:
            self.cache_rejections += 1
        return image

    def peek(self, index):
        """Read an aligned frame without consuming or duplicating a cached copy."""
        self._check_memory()
        image = self.frames.get(index)
        if image is not None:
            self.peek_cache_hits += 1
            return image
        self.peek_loader_calls += 1
        return self._load(index, "peek")

    def for_blend(self, index):
        self._check_memory()
        image = self.frames.pop(index, None)
        if image is None:
            return self._load(index, "blend")
        self.bytes_used -= image.nbytes
        self._release_shared(image.nbytes)
        self.hits += 1
        return image

    def _load(self, index, phase):
        self.loader_calls += 1
        if phase == "focus":
            self.focus_loader_calls += 1
        elif phase == "blend":
            self.blend_loader_calls += 1
        self._load_counts[index] = self._load_counts.get(index, 0) + 1
        image = self.loader(index)
        self.loaded_bytes += int(image.nbytes)
        return image

    def _try_reserve_shared(self, size):
        return self.shared_budget is None or self.shared_budget.try_reserve(size)

    def _release_shared(self, size):
        if self.shared_budget is not None:
            self.shared_budget.release(size)

    @property
    def stats(self):
        return {
            "cache": "aligned_frame",
            "cache_limit_bytes": self.max_bytes,
            "cache_bytes_used": self.bytes_used,
            "cache_peak_bytes": self.peak_bytes,
            "focus_loader_calls": self.focus_loader_calls,
            "focus_cache_reuse_hits": self.focus_cache_hits,
            "peek_loader_calls": self.peek_loader_calls,
            "peek_cache_hits": self.peek_cache_hits,
            "blend_loader_calls": self.blend_loader_calls,
            "blend_cache_hits": self.hits,
            "loader_calls": self.loader_calls,
            "loaded_bytes": self.loaded_bytes,
            "cache_insertions": self.cache_insertions,
            "cache_rejections": self.cache_rejections,
            "duplicate_loader_calls": sum(max(0, count - 1)
                                           for count in self._load_counts.values()),
            "load_counts_by_frame": dict(self._load_counts),
            "shared_cache_requested_limit_bytes": (self.shared_budget.requested_max_bytes
                                                    if self.shared_budget is not None else None),
            "shared_cache_limit_bytes": (self.shared_budget.max_bytes
                                         if self.shared_budget is not None else None),
            "shared_cache_bytes_used": (self.shared_budget.bytes_used
                                        if self.shared_budget is not None else None),
            "shared_cache_peak_bytes": (self.shared_budget.peak_bytes
                                        if self.shared_budget is not None else None),
        }

    def clear(self):
        self._release_shared(self.bytes_used)
        self.frames.clear()
        self.bytes_used = 0
