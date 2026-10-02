"""Strict NumPy/ctypes boundary for the isolated C++ statistics candidate."""
import ctypes
from pathlib import Path
import numpy as np

_dll = ctypes.CDLL(str(Path(__file__).with_name('statistics_native.dll')))
def _array(dtype):
    return np.ctypeslib.ndpointer(dtype=dtype, flags='C_CONTIGUOUS')
_u8, _u16, _f32, _i64 = map(_array, (np.uint8, np.uint16, np.float32, np.int64))
_dll.probe_update.argtypes = [_u8, _u8, _u8, _u8, _u16, ctypes.c_int64]
_dll.flat_noise.argtypes = [_u8, _f32, _f32, _f32, ctypes.c_int64, _f32, _f32, _f32, _i64, _i64, _f32, _f32, _f32]
_dll.chroma_map.argtypes = [_u8, _u8, ctypes.c_int64]
for _name in ('probe_update', 'flat_noise', 'chroma_map'):
    getattr(_dll, _name).restype = None

def probe_update(classes, rgb, low, high, count):
    _dll.probe_update(classes, rgb, low, high, count, classes.size)

def chroma_map(rgb):
    rgb = np.ascontiguousarray(rgb)
    result = np.empty(rgb.shape[:2], np.uint8)
    _dll.chroma_map(rgb, result, result.size)
    return result

def flat_noise(stats, small, variance, gradient, response):
    bins = np.ascontiguousarray(small[::4, ::4] // 16)
    vs, gs, fs = (np.ascontiguousarray(a[::4, ::4], dtype=np.float32) for a in (variance, gradient, response))
    ff, fv, fg = (np.zeros(16, np.float32) for _ in range(3))
    _dll.flat_noise(bins, vs, gs, fs, bins.size, ff, fv, fg, stats.samples,
                    stats.noise_fallback_counts, stats.focus_floor, stats.variance_floor, stats.gradient_floor)
    return ff, fv, fg

_dll.material_classes.argtypes = [_u8, _u8, _u8, ctypes.c_int64]
_dll.masked_colour.argtypes = [_f32, _u8, _f32, ctypes.c_int64]
_dll.normalize_colour.argtypes = [_f32, _f32, ctypes.c_int64]
_dll.update_offset.argtypes = [_u8, ctypes.c_uint8, _f32, _f32, _f32, _u8, ctypes.c_int64]
_dll.apply_colour.argtypes = [_u8, _f32, _u8, _u8, ctypes.c_int64]
for _name in ('material_classes', 'masked_colour', 'normalize_colour', 'update_offset'):
    getattr(_dll, _name).restype = None
_dll.apply_colour.restype = ctypes.c_int64

def material_classes(hsv, gray):
    result = np.empty(gray.shape, np.uint8)
    _dll.material_classes(hsv, gray, result, gray.size)
    return result

def masked_colour(rgb, mask):
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    result = np.empty(rgb.shape, np.float32)
    _dll.masked_colour(rgb, mask, result, mask.size)
    return result

def normalize_colour(field, density):
    _dll.normalize_colour(field, density, density.size)

def update_offset(classes, material, delta, density, offset, confidence):
    _dll.update_offset(classes, material, delta, density, offset, confidence, classes.size)

def apply_colour(rgb, field, valid):
    rgb = np.ascontiguousarray(rgb)
    valid = np.ascontiguousarray(valid, dtype=np.uint8)
    result = np.empty(rgb.shape, np.uint8)
    changed = _dll.apply_colour(rgb, field, valid, result, valid.size)
    return result, changed
