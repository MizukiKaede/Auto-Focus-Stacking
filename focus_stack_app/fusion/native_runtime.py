"""Shared, bounded ctypes execution; no process-global OpenMP settings."""
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Mapping
from functools import wraps
from pathlib import Path
import ctypes
import logging
import os
import operator

import cv2
import numpy as np

AUTO_PARALLEL_BACKENDS = frozenset(['fast'])  # Accepted on 00522 and 03003.
_threads = ContextVar('fusion_native_threads', default=1)
_warnings = set()
CHUNK_PIXELS = 262144


def array(value, dtype, shape=None, *, write=False, name='array'):
    if not isinstance(value, np.ndarray) or value.dtype not in tuple(np.dtype(d) for d in (dtype if isinstance(dtype, tuple) else (dtype,))):
        raise ValueError(f'{name}: incorrect ndarray dtype')
    if shape is not None and value.shape != tuple(shape):
        raise ValueError(f'{name}: expected shape {tuple(shape)}, got {value.shape}')
    if not value.flags.c_contiguous:
        raise ValueError(f'{name}: contiguous array required')
    if write and not value.flags.writeable:
        raise ValueError(f'{name}: writable array required')
    return value


def image(value, dtype=np.uint8, *, write=False, name='RGB'):
    array(value, dtype, write=write, name=name)
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise ValueError(f'{name}: nonempty HxWx3 image required')
    return value.shape[:2]


def index(value):
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError('integer source index required') from error
    if not 0 <= result <= 65535:
        raise ValueError('source index outside uint16 range')
    return result


def chunks(n):
    for start in range(0, n, CHUNK_PIXELS):
        yield slice(start, min(n, start + CHUNK_PIXELS))


def _value(config, name, default):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


@contextmanager
def native_thread_budget(config=None, backend='quality'):
    requested = int(_value(config, 'native_threads', 0))
    if requested < 0:
        raise ValueError('native_threads must be nonnegative')
    cpus = max(1, os.cpu_count() or 1)
    groups = max(1, int(_value(config, 'max_hugin_workers', 3)))
    cv_threads = max(1, cv2.getNumThreads())
    reserved = 0
    if _value(config, 'parallel_pipeline', True):
        from ..utils.concurrency import resolve_focus_analysis_budget
        analysis = resolve_focus_analysis_budget(
            int(_value(config, 'focus_analysis_workers', 0)), merge_workers=groups,
            parallel_pipeline=True, logical_cpus=cpus, current_opencv_threads=cv_threads)
        reserved = min(cpus-1, analysis.workers*analysis.opencv_threads)
    budget = max(1, min(3, (cpus-reserved)//groups, cv_threads))
    effective = min(requested, budget) if requested else (budget if backend in AUTO_PARALLEL_BACKENDS else 1)
    token = _threads.set(effective)
    try:
        yield {'requested_threads': requested, 'threads': effective, 'cpu_budget': budget,
               'analysis_reserved_threads': reserved}
    finally:
        _threads.reset(token)


def native_fusion(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        from ..utils.performance import diagnostic
        with native_thread_budget(self.runtime_config, self.name) as budget:
            diagnostic('native_thread_budget', **budget)
            try:
                return fn(self, *args, **kwargs)
            finally:
                from .statistics_native import runtime_info
                diagnostic('statistics_native_runtime', **runtime_info())
                if self.name == 'fast':
                    from .fast_cpp import runtime_info as fast_info
                    diagnostic('fast_cpp_execution', **fast_info())
    return wrapped


class NativeLibrary:
    """Load/bind atomically. Missing optional thread exports mean serial legacy ABI."""
    def __init__(self, path, signatures, *, abi_name=None, expected_abi=None, optional_abi=False):
        self.path = Path(path)
        self.dll = None
        self.functions = {}
        self.reason = None
        self.abi = None
        self.parallel = False
        self._directory = None
        self._set_threads = None
        if os.environ.get('FOCUS_STACK_FORCE_NUMPY') == '1':
            self.reason = 'NumPy explicitly selected by FOCUS_STACK_FORCE_NUMPY'
            return
        try:
            if os.name == 'nt' and hasattr(os, 'add_dll_directory') and self.path.parent.is_dir():
                self._directory = os.add_dll_directory(str(self.path.parent.resolve()))
            dll = ctypes.CDLL(str(self.path.resolve()))
            functions = {}
            for name, (args, result) in signatures.items():
                fn = getattr(dll, name)
                fn.argtypes, fn.restype = args, result
                functions[name] = fn
            if abi_name:
                check = getattr(dll, abi_name, None)
                if check is None:
                    if not optional_abi:
                        raise ValueError('native ABI export missing')
                    self.abi = 'legacy'
                else:
                    check.argtypes, check.restype = [], ctypes.c_int
                    self.abi = int(check())
                    if self.abi != expected_abi:
                        raise ValueError(f'native ABI mismatch: expected {expected_abi}, got {self.abi}')
            setter = getattr(dll, 'native_set_threads', None)
            enabled = getattr(dll, 'native_openmp_enabled', None)
            if setter is not None and enabled is not None:
                setter.argtypes, setter.restype = [ctypes.c_int], None
                enabled.argtypes, enabled.restype = [], ctypes.c_int
                self.parallel = bool(enabled())
                self._set_threads = setter
            self.dll, self.functions = dll, functions
        except (OSError, AttributeError, ValueError) as error:
            self.reason = f'{type(error).__name__}: {error}'
            if self._directory is not None:
                self._directory.close()
                self._directory = None
            key = (str(self.path), self.reason)
            if key not in _warnings:
                _warnings.add(key)
                logging.getLogger(__name__).warning('%s unavailable; using NumPy/OpenCV: %s', self.path.name, self.reason)

    @property
    def available(self):
        return self.dll is not None

    def call(self, name, *args):
        if self._set_threads is not None:
            self._set_threads(_threads.get())
        return self.functions[name](*args)

    def info(self):
        return {'library': str(self.path), 'abi': self.abi,
                'implementation': 'C++17 DLL + OpenCV' if self.available else 'NumPy + OpenCV',
                'load_error': self.reason, 'openmp': self.parallel,
                'threads': _threads.get() if self.parallel and self.available else 1,
                'parallel_pixel_threshold': CHUNK_PIXELS}
