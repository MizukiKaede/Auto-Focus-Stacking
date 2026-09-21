"""Private, bounded, context-local preprocessing cache for one ordering graph."""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar

_current = ContextVar("registration_preprocessing", default=None)


class PreprocessingCache:
    def __init__(self, max_bytes=128 * 1024**2):
        self.max_bytes = max_bytes
        self.bytes_used = 0
        self.entries = OrderedDict()

    def get(self, operation, source, create):
        key = (operation, id(source))
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key][1]
        value = create()
        # Include retained source arrays, descriptors and approximate keypoint
        # overhead. Overcounting shared arrays keeps the bound conservative.
        size = int(getattr(source, "nbytes", 0)) + int(getattr(value, "nbytes", 0))
        if isinstance(value, tuple):
            size += int(getattr(value[1], "nbytes", 0)) + len(value[0] or ()) * 128
        if size <= self.max_bytes:
            while self.entries and self.bytes_used + size > self.max_bytes:
                _, (_, _, removed) = self.entries.popitem(last=False)
                self.bytes_used -= removed
            self.entries[key] = (source, value, size)
            self.bytes_used += size
        return value


def cached(operation, source, create):
    cache = _current.get()
    return create() if cache is None else cache.get(operation, source, create)


@contextmanager
def preprocessing_cache():
    from ..utils.memory import memory_snapshot
    try:
        memory = memory_snapshot()
        budget = min(128 * 1024**2, max(0, memory.available_bytes - max(2 * 1024**3, int(memory.total_bytes * .1))) // 4)
    except Exception:
        budget = 0
    cache = PreprocessingCache(budget)
    token = _current.set(cache)
    try:
        yield cache
    finally:
        _current.reset(token)
        cache.entries.clear()
        cache.bytes_used = 0
