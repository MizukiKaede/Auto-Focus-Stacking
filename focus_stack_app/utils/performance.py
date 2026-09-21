"""Diagnostic-only wall timings and sampled process-tree memory usage."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import inspect
import logging
import threading
import time

_depth = ContextVar("performance_depth", default=0)
_logger = logging.getLogger("focus_stack_app.controller.performance")
_active_logger = ContextVar("performance_logger", default=None)
_group = ContextVar("performance_group", default=None)


@contextmanager
def stage(name, **details):
    started = time.perf_counter()
    depth = _depth.get()
    token = _depth.set(depth + 1)
    outcome = "ok"
    try:
        yield
    except BaseException:
        outcome = "error"
        raise
    finally:
        _depth.reset(token)
        (_active_logger.get() or _logger).info("PERF %s", json.dumps(dict(
            stage=name, seconds=time.perf_counter() - started,
            depth=depth, outcome=outcome, group_id=_group.get(), **details), ensure_ascii=False))


def timed(name):
    def decorate(fn):
        names = list(inspect.signature(fn).parameters)
        @wraps(fn)
        def wrapped(*args, **kwargs):
            logger = getattr(args[0], "logger", None) if args and names[:1] == ["self"] else None
            log_token = _active_logger.set(logger or _active_logger.get())
            group = None
            for key in ("group", "job"):
                if key in names:
                    index = names.index(key)
                    group = args[index] if len(args) > index else kwargs.get(key)
                    group = getattr(group, "group", group)
                    break
            gid = (group.get("group_id", group.get("id")) if isinstance(group, dict)
                   else getattr(group, "group_id", getattr(group, "id", None)))
            group_token = _group.set(gid if gid is not None else _group.get())
            try:
                with stage(name):
                    return fn(*args, **kwargs)
            finally:
                _group.reset(group_token)
                _active_logger.reset(log_token)
        return wrapped
    return decorate


class BatchMemory:
    """Sample summed RSS of this process and its children every 250 ms.

    This is a sampled peak, not a system peak or exact allocation counter.
    Missing psutil/vanished processes never fail a batch.
    """
    def __init__(self, logger=None):
        self.logger = logger or _logger
        self.peak = None
        self.stop_event = threading.Event()
        self.thread = None

    def sample(self):
        try:
            import psutil
            process = psutil.Process()
            total = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.Error:
                    pass
            self.peak = max(self.peak or 0, total)
        except Exception:
            pass

    def start(self):
        self.sample()
        def monitor():
            while not self.stop_event.wait(0.25):
                self.sample()
        self.thread = threading.Thread(target=monitor, name="performance-memory", daemon=True)
        self.thread.start()

    def finish(self, seconds, **details):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.sample()
        self.logger.info("PERF %s", json.dumps(dict(
            stage="batch", seconds=seconds, sampled_process_tree_peak_rss_bytes=self.peak,
            memory_sample_interval_seconds=0.25, **details)))
