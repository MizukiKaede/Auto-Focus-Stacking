"""Validated statistics kernels with equivalent bounded NumPy execution."""
import ctypes as C
from collections import Counter
from functools import wraps
from pathlib import Path
import numpy as np
from .native_runtime import NativeLibrary, array, image, chunks

P, N, U = C.c_void_p, C.c_int64, C.c_uint8
_signatures = {
    'probe_update': ([P]*5+[N], None),
    'flat_noise': ([P]*4+[N]+[P]*8, None),
    'chroma_map': ([P,P,N], None),
    'material_classes': ([P]*3+[N], None),
    'masked_colour': ([P]*3+[N], None),
    'normalize_colour': ([P,P,N], None),
    'update_offset': ([P,U]+[P]*4+[N], None),
    'apply_colour': ([P]*4+[N], N),
}
_library = NativeLibrary(Path(__file__).with_name('statistics_native.dll'), _signatures,
                         abi_name='statistics_core_abi', expected_abi=1, optional_abi=True)
_calls = Counter()


def runtime_info():
    return {**_library.info(), 'calls': dict(_calls), 'calls_scope': 'process_lifetime'}


def _call(name, *args):
    return _library.call(name, *(a.ctypes.data if isinstance(a, np.ndarray) else a for a in args))


def _mask(mask, shape):
    array(mask, (np.uint8, np.bool_), shape)
    return np.ascontiguousarray(mask, dtype=np.uint8)


def probe_update(classes, rgb, low, high, count):
    shape = image(rgb)
    array(classes, np.uint8, shape)
    array(low, np.uint8, (16, *shape, 3), write=True)
    array(high, np.uint8, low.shape, write=True)
    array(count, np.uint16, (16, *shape), write=True)
    if classes.size and classes.max() >= 16:
        raise ValueError('probe class outside 0..15')
    if _library.available:
        _call('probe_update', classes, rgb, low, high, count, classes.size)
        return
    n = classes.size
    cls, pixels = classes.ravel(), rgb.reshape(-1, 3)
    lo, hi, cnt = low.reshape(16*n, 3), high.reshape(16*n, 3), count.ravel()
    for part in chunks(n):
        positions = cls[part].astype(np.int64)*n + np.arange(part.start, part.stop)
        cnt[positions] += np.uint16(1)
        lo[positions] = np.minimum(lo[positions], pixels[part])
        hi[positions] = np.maximum(hi[positions], pixels[part])


def chroma_map(rgb):
    rgb = np.ascontiguousarray(rgb)
    shape = image(rgb)
    result = np.empty(shape, np.uint8)
    if _library.available:
        _call('chroma_map', rgb, result, result.size)
    else:
        source, out = rgb.reshape(-1, 3), result.ravel()
        for part in chunks(result.size):
            out[part] = source[part].max(axis=1)-source[part].min(axis=1)
    return result


def _noise(values):
    median = np.float32(np.median(values))
    mad = np.float32(np.median(np.abs(values-median)))
    return np.float32(max(float(np.finfo(np.float32).eps), float(median)+6.0*1.4826*float(mad)))


def flat_noise(stats, small, variance, gradient, response):
    array(small, np.uint8)
    if small.ndim != 2:
        raise ValueError('luma plane required')
    for values in (variance, gradient, response):
        array(values, np.float32, small.shape)
    for name in ('samples', 'noise_fallback_counts'):
        array(getattr(stats, name), np.int64, (16,), write=True, name=name)
    for name in ('focus_floor', 'variance_floor', 'gradient_floor'):
        array(getattr(stats, name), np.float32, (16,), write=True, name=name)
    bins = np.ascontiguousarray(small[::4, ::4] // 16)
    vs, gs, fs = (np.ascontiguousarray(a[::4, ::4]) for a in (variance, gradient, response))
    ff, fv, fg = (np.zeros(16, np.float32) for _ in range(3))
    if _library.available:
        _call('flat_noise', bins, vs, gs, fs, bins.size, ff, fv, fg, stats.samples,
              stats.noise_fallback_counts, stats.focus_floor, stats.variance_floor, stats.gradient_floor)
        return ff, fv, fg
    for b in range(16):
        positions = bins == b
        values = vs[positions]
        if values.size < 64:
            continue
        ordered = np.sort(values)
        pos = (values.size-1)*0.35
        k = int(pos)
        a, upper = ordered[k], ordered[k+1]
        fraction, diff = pos-k, float(np.float32(upper-a))
        threshold = (float(upper)-diff*(1.0-fraction) if fraction >= 0.5 else float(a)+diff*fraction)
        flat = positions & (vs.astype(np.float64) <= threshold)
        size = np.count_nonzero(flat)
        if size < 32:
            continue
        stats.samples[b] += size
        ff[b], fv[b], fg[b] = (_noise(a[flat]) for a in (fs, vs, gs))
    np.maximum(stats.focus_floor, ff, out=stats.focus_floor)
    np.maximum(stats.variance_floor, fv, out=stats.variance_floor)
    np.maximum(stats.gradient_floor, fg, out=stats.gradient_floor)
    observed = ff > 0
    for b in range(16):
        if observed[b]:
            continue
        neighbors = [a for a in (b-1, b+1) if 0 <= a < 16 and observed[a]]
        if neighbors:
            for out in (ff, fv, fg):
                out[b] = max(out[a] for a in neighbors)
            stats.noise_fallback_counts[b] += 1
    return ff, fv, fg


def material_classes(hsv, gray):
    shape = image(hsv, name='HSV')
    array(gray, np.uint8, shape)
    result = np.empty(shape, np.uint8)
    if _library.available:
        _call('material_classes', hsv, gray, result, gray.size)
    else:
        source, luma, out = hsv.reshape(-1, 3), gray.ravel(), result.ravel()
        for part in chunks(gray.size):
            h, g = source[part], luma[part]
            out[part] = np.where(h[:, 1] >= 50,
                1+((h[:, 0].astype(np.uint16)+15)//30)%6+6*(g < 64),
                13+(g >= 64).astype(np.uint8)+(g >= 160).astype(np.uint8))
    return result


def masked_colour(rgb, mask):
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    shape = image(rgb, np.float32)
    mask = _mask(mask, shape)
    result = np.empty_like(rgb)
    if _library.available:
        _call('masked_colour', rgb, mask, result, mask.size)
    else:
        source, out, m = rgb.reshape(-1, 3), result.reshape(-1, 3), mask.ravel()
        for part in chunks(mask.size):
            out[part] = source[part]*m[part, None].astype(np.float32)
    return result


def normalize_colour(field, density):
    shape = image(field, np.float32, write=True)
    array(density, np.float32, shape)
    if _library.available:
        _call('normalize_colour', field, density, density.size)
    else:
        out, d = field.reshape(-1, 3), density.ravel()
        for part in chunks(density.size):
            out[part] /= np.maximum(d[part, None], np.float32(1e-6))


def update_offset(classes, material, delta, density, offset, confidence):
    shape = image(delta, np.float32)
    array(classes, np.uint8, shape)
    array(density, np.float32, shape)
    array(offset, np.float32, (*shape, 3), write=True)
    array(confidence, np.uint8, shape, write=True)
    if not isinstance(material, (int, np.integer)) or not 0 <= material < 16:
        raise ValueError('material outside 0..15')
    if _library.available:
        _call('update_offset', classes, material, delta, density, offset, confidence, classes.size)
    else:
        cls, d = classes.ravel(), density.ravel()
        source, out, conf = delta.reshape(-1, 3), offset.reshape(-1, 3), confidence.ravel()
        for part in chunks(classes.size):
            use = (cls[part] == material) & (d[part] > np.float32(1e-6))
            magnitude = np.abs(source[part]).max(axis=1)
            alpha = np.clip((magnitude-np.float32(1))/np.float32(3), 0, 1)
            alpha *= np.clip((np.float32(40)-magnitude)/np.float32(20), 0, 1)
            alpha *= np.clip(d[part]/np.float32(0.2), 0, 1)
            out[part][use] = source[part][use]*alpha[use, None]
            conf[part][use] = 1


def apply_colour(rgb, field, valid):
    rgb = np.ascontiguousarray(rgb)
    shape = image(rgb)
    array(field, np.float32, (*shape, 3))
    valid = _mask(valid, shape)
    result = np.empty_like(rgb)
    if _library.available:
        changed = _call('apply_colour', rgb, field, valid, result, valid.size)
    else:
        source, f, out, v = rgb.reshape(-1, 3), field.reshape(-1, 3), result.reshape(-1, 3), valid.ravel()
        changed = 0
        for part in chunks(valid.size):
            corrected = np.rint(np.clip(source[part].astype(np.float32)+f[part], 0, 255)).astype(np.uint8)
            out[part] = np.where(v[part, None] != 0, corrected, source[part])
            changed += int(np.count_nonzero(np.any(out[part] != source[part], axis=1)))
    return result, int(changed)


def _record(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        result = fn(*args, **kwargs)
        _calls[fn.__name__] += 1
        return result
    return wrapped


for _name in _signatures:
    globals()[_name] = _record(globals()[_name])
