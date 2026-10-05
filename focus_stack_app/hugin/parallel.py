"""Process-wide admission for bounded Hugin native preparation tasks."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
import os
import threading

from ..utils.memory import memory_snapshot
from ..utils.performance import diagnostic


def validate_budget(value):
    if type(value) is not int or not 1 <= value <= 12:
        raise ValueError('hugin_parallel_cpu_budget must be an integer between 1 and 12')
    return value


def _check_cancel(event):
    if event is not None and event.is_set():
        raise InterruptedError('Hugin preparation cancelled')


class SharedPreparationBudget:
    def __init__(self):
        self.condition = threading.Condition()
        self.sessions = []
        self.used = self.bytes_used = self.peak = 0

    @property
    def capacity(self):
        return min(12, max(1, os.cpu_count() or 1), *self.sessions)

    @contextmanager
    def session(self, limit):
        with self.condition:
            self.sessions.append(limit)
            self.condition.notify_all()
        try:
            yield
        finally:
            with self.condition:
                self.sessions.remove(limit)
                self.condition.notify_all()

    def acquire(self, cost, size, event):
        with self.condition:
            while True:
                _check_cancel(event)
                snapshot = memory_snapshot()
                reserve = max(2 * 1024**3, snapshot.total_bytes // 10)
                usable = max(0, snapshot.available_bytes - reserve)
                # One serial task can proceed under pressure, as in the
                # existing image-worker budget. Additional buffers cannot.
                memory_ok = not self.used or self.bytes_used + size <= usable
                if self.used + cost <= self.capacity and memory_ok:
                    self.used += cost
                    self.bytes_used += size
                    self.peak = max(self.peak, self.used)
                    return
                self.condition.wait(0.05)

    def release(self, cost, size):
        with self.condition:
            if cost > self.used or size > self.bytes_used:
                raise RuntimeError('Hugin preparation reservation released twice')
            self.used -= cost
            self.bytes_used -= size
            self.condition.notify_all()


_shared_budget = SharedPreparationBudget()


def bounded_map(fn, items, *, workers, cpu_cost=1, cpu_budget=12,
                cancel_event=None, working_bytes=0):
    """Ordered results, limited buffers, and iterator advancement by the caller.

    Reserve before decoding/preprocessing the next item. Jobs hold their
    reservation until native computation and encoding have both finished.
    Each submission owns a separate copied context for performance events.
    """
    limit = min(validate_budget(cpu_budget), max(1, os.cpu_count() or 1))
    cost = max(1, int(cpu_cost))
    if cost > limit:
        raise ValueError('OpenCV thread count exceeds Hugin CPU budget; use opencv_thread_budget at the caller')
    workers = max(1, min(int(workers), limit // cost))
    size = max(0, int(working_bytes))
    items = iter(items)
    pending = deque()
    failure = []
    progress_lock = threading.Lock()
    active = peak_active = completed = 0
    diagnostic('hugin_preparation_budget', cpu_budget=limit, workers=workers,
               cpu_cost=cost, working_bytes_per_task=size, shared_process_budget=True)

    def run(item):
        nonlocal active, peak_active, completed
        entered = False
        try:
            with progress_lock:
                if failure:
                    raise RuntimeError('Hugin preparation stopped after a task failure') from failure[0]
                active += 1
                peak_active = max(peak_active, active)
            entered = True
            _check_cancel(cancel_event)
            result = fn(item)
            _check_cancel(cancel_event)
            return result
        except BaseException as exc:
            with progress_lock:
                if not failure:
                    failure.append(exc)
            raise
        finally:
            with progress_lock:
                if entered:
                    active -= 1
                    completed += 1
            _shared_budget.release(cost, size)

    def check_failure():
        _check_cancel(cancel_event)
        with progress_lock:
            if failure:
                raise failure[0]

    @contextmanager
    def execution_report():
        try:
            yield
        finally:
            diagnostic('hugin_preparation_execution', completed_tasks=completed,
                       peak_active_tasks=peak_active, cpu_cost=cost,
                       shared_cpu_peak=_shared_budget.peak,
                       shared_cpu_peak_scope='process_lifetime')

    with _shared_budget.session(limit), execution_report():
        if workers == 1:
            while True:
                _shared_budget.acquire(cost, size, cancel_event)
                try:
                    item = next(items)
                except StopIteration:
                    _shared_budget.release(cost, size)
                    return
                except BaseException:
                    _shared_budget.release(cost, size)
                    raise
                yield run(item)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='hugin-native') as pool:
                def submit():
                    check_failure()
                    _shared_budget.acquire(cost, size, cancel_event)
                    try:
                        check_failure()
                        item = next(items)
                    except StopIteration:
                        _shared_budget.release(cost, size)
                        return False
                    except BaseException:
                        _shared_budget.release(cost, size)
                        raise
                    try:
                        future = pool.submit(copy_context().run, run, item)
                    except BaseException:
                        _shared_budget.release(cost, size)
                        raise
                    # A cancelled queued job never enters run's finally block.
                    future.add_done_callback(lambda f: _shared_budget.release(cost, size) if f.cancelled() else None)
                    pending.append(future)
                    return True

                try:
                    for _ in range(workers):
                        if not submit():
                            break
                    while pending:
                        check_failure()
                        result = pending.popleft().result()
                        check_failure()
                        yield result
                        submit()
                finally:
                    for future in pending:
                        future.cancel()


@contextmanager
def opencv_preparation_budget(cpu_budget):
    """Direct API safety; application calls already enter a bounded CV scope."""
    import cv2
    from ..utils.concurrency import opencv_thread_budget
    limit = min(validate_budget(cpu_budget), max(1, os.cpu_count() or 1))
    current = max(1, cv2.getNumThreads())
    if current <= limit:
        yield current
    else:
        # Only the calling preparation stage changes global CV settings;
        # individual pool workers never do. The prior value is restored.
        with opencv_thread_budget(limit):
            yield limit
