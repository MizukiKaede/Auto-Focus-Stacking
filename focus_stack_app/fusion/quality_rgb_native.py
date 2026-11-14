"""Single-level RGB accumulation with the historical float32 operation order."""
import ctypes as C
from collections import Counter
from pathlib import Path

import numpy as np

from .native_runtime import NativeLibrary, array, image

P, N, I = C.c_void_p, C.c_int64, C.c_int
_library = NativeLibrary(
    Path(__file__).with_name('statistics_native.dll'),
    {'quality_rgb_add': ([P, P, P, N, I], None),
     'quality_rgb_finish': ([P, P, N], None)},
    abi_name='statistics_core_abi', expected_abi=1, optional_abi=True,
)
_calls = Counter()


def available():
    return _library.available


def add(rgb, weight, accumulator, *, first):
    rgb = np.ascontiguousarray(rgb)
    shape = image(rgb)
    array(weight, np.float32, shape)
    array(accumulator, np.float32, (*shape, 3), write=True)
    _library.call('quality_rgb_add', rgb.ctypes.data, weight.ctypes.data,
                  accumulator.ctypes.data, weight.size, int(first))
    _calls['add'] += 1


def finish(accumulator):
    image(accumulator, np.float32)
    out = np.empty(accumulator.shape, np.uint8)
    _library.call('quality_rgb_finish', accumulator.ctypes.data, out.ctypes.data, accumulator.size)
    _calls['finish'] += 1
    return out


def runtime_info():
    return {**_library.info(), 'calls': dict(_calls), 'calls_scope': 'process_lifetime'}

