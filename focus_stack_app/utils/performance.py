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
_collector = ContextVar("performance_collector", default=None)


def _json_default(value):
    return value.tolist() if hasattr(value, "tolist") else str(value)


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
        event = dict(
            stage=name, seconds=time.perf_counter() - started,
            depth=depth, outcome=outcome, group_id=_group.get(), **details)
        collector = _collector.get()
        if collector is not None:
            collector.record(event)
        (_active_logger.get() or _logger).info("PERF %s", json.dumps(event, ensure_ascii=False, default=_json_default))


class FusionProfile:
    """Per-run timings and sampled process-tree RSS/IO; never affect pixels.

    IO counters describe the process tree, not physical HDD traffic. OS caching
    and external tools prevent deriving disk wait from these counters.
    """

    def __init__(self, directory, backend, **details):
        from pathlib import Path
        self.path = Path(directory) / "fusion_profile.json"
        self.backend = backend
        self.details = details
        self.events = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.peak_rss = 0
        self.io = {}

    def record(self, event):
        with self.lock:
            self.events.append(event)

    def sample(self):
        try:
            import psutil
            parent = psutil.Process()
            rss = 0
            for process in [parent, *parent.children(recursive=True)]:
                try:
                    rss += process.memory_info().rss
                    counters = process.io_counters()
                    key = (process.pid, process.create_time())
                    previous = self.io.get(key)
                    current = (counters.read_bytes, counters.write_bytes)
                    self.io[key] = (previous[0] if previous else current, current)
                except psutil.Error:
                    pass
            self.peak_rss = max(self.peak_rss, rss)
        except Exception:
            pass

    def __enter__(self):
        self.started = time.perf_counter()
        self.token = _collector.set(self)
        self.sample()
        def monitor():
            while not self.stop.wait(0.1):
                self.sample()
        self.thread = threading.Thread(target=monitor, daemon=True, name="fusion-profile")
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join()
        self.sample()
        _collector.reset(self.token)
        try:
            import cv2
            import sys
            from .memory import memory_snapshot
            from .opencl import opencl_status
            snapshot = memory_snapshot()
            stages = {}
            for event in self.events:
                stages[event["stage"]] = stages.get(event["stage"], 0.0) + event["seconds"]
            report = dict(
                schema_version=1, backend=self.backend,
                opencl=opencl_status(),
                outcome="error" if kind else "ok", error=str(value) if kind else None,
                seconds=time.perf_counter() - self.started,
                python_executable=sys.executable,
                opencv_version=cv2.__version__, physical_memory_bytes=snapshot.total_bytes,
                available_memory_bytes_at_end=snapshot.available_bytes,
                sampled_process_tree_peak_rss_bytes=self.peak_rss,
                sample_interval_seconds=0.1,
                process_tree_read_bytes=sum(max(0, last[0] - first[0]) for first, last in self.io.values()),
                process_tree_write_bytes=sum(max(0, last[1] - first[1]) for first, last in self.io.values()),
                io_scope="process_tree_logical_counters_not_physical_disk_or_io_wait",
                stages_seconds=stages, events=self.events, **self.details,
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        except Exception:
            _logger.warning("Unable to save fusion profile %s", self.path, exc_info=True)


def profiled_fusion(fn):
    @wraps(fn)
    def wrapped(self, group, analysis, output_path, work_dir, output_config, cancel_event):
        from dataclasses import asdict, is_dataclass
        runtime = getattr(self, "runtime_config", None)
        config = asdict(runtime) if is_dataclass(runtime) else runtime if isinstance(runtime, dict) else None
        output = asdict(output_config) if is_dataclass(output_config) else None
        with FusionProfile(work_dir, self.name, runtime_config=config, output_config=output,
                           output_path=str(output_path), temporary_directory=str(work_dir)):
            with stage("total_fusion", backend=self.name):
                return fn(self, group, analysis, output_path, work_dir, output_config, cancel_event)
    return wrapped


def diagnostic(name, **details):
    event = dict(stage=name, seconds=0.0, **details)
    collector = _collector.get()
    if collector is not None:
        collector.record(event)
    (_active_logger.get() or _logger).info("DIAGNOSTIC %s", json.dumps(event, ensure_ascii=False, default=_json_default))


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
