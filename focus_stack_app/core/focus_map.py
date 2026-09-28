"""Multi-scale local focus maps for focus-stack analysis.

The implementation combines absolute Laplacian response, Scharr gradient
magnitude and local variance at several scales.  Maps are normalised per
image, then downsampled to a compact representation (float16 by default).
Only one analysis image is touched at a time; no module-level image cache is
created.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from typing import Any, Iterable

from .registration import load_image, resolve_image_path, resize_for_analysis, warp_image
from .types import FocusMapResult


@dataclass(slots=True)
class FocusMapConfig:
    """Focus-map parameters with safe V1 defaults."""

    analysis_long_edge: int = 1600
    output_long_edge: int = 512
    output_dtype: str = "float16"  # ``float16`` or ``uint16``
    scales: tuple[float, ...] = (1.0, 2.5, 5.0)
    laplacian_weight: float = 0.50
    gradient_weight: float = 0.30
    variance_weight: float = 0.20
    local_variance_window: int = 7
    normalise_low_percentile: float = 1.0
    normalise_high_percentile: float = 99.0
    saturation_low: float = 0.01
    saturation_high: float = 0.99
    saturation_penalty: float = 0.35
    output_blur_sigma: float = 0.6


def _setting(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    value = getattr(config, name, None)
    if value is None:
        aliases = {
            "analysis_long_edge": "focus_analysis_long_edge",
            "output_long_edge": "focus_cache_long_edge",
        }
        alias = aliases.get(name)
        if alias:
            value = getattr(config, alias, None)
    if value is None:
        section = getattr(config, "analysis", None)
        if section is not None:
            value = getattr(section, name, None)
            if value is None:
                aliases = {
                    "analysis_long_edge": "focus_analysis_long_edge",
                    "output_long_edge": "focus_cache_long_edge",
                }
                alias = aliases.get(name)
                if alias:
                    value = getattr(section, alias, None)
    return default if value is None else value


def _np() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - minimal env only
        raise RuntimeError(
            "focus-map analysis requires NumPy; install the image dependencies"
        ) from exc
    return np


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - minimal env only
        raise RuntimeError(
            "focus-map analysis requires OpenCV (opencv-python)"
        ) from exc
    return cv2


def _gray_float(image: Any) -> Any:
    np = _np()
    cv2 = None
    try:
        cv2 = _cv2()
    except RuntimeError:
        pass
    source_path = resolve_image_path(image)
    if source_path is not None:
        image = load_image(source_path)
    arr = np.asarray(image)
    if arr.ndim == 3:
        if cv2 is not None:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            arr = np.mean(arr[..., :3], axis=2)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("image must be a non-empty 2-D or 3-D array")
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    if float(arr.max(initial=0.0)) > 1.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def _normalise(response: Any, low: float, high: float) -> Any:
    np = _np()
    finite = response[np.isfinite(response)]
    if finite.size == 0:
        return np.zeros_like(response, dtype=np.float32)
    lo, hi = np.percentile(finite, [float(low), float(high)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-8:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo + 1e-8:
        return np.zeros_like(response, dtype=np.float32)
    return np.clip((response - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)


def _box_blur_numpy(image: Any, window: int) -> Any:
    """Small NumPy-only box blur used if cv2 is unavailable."""

    np = _np()
    window = max(1, int(window) | 1)
    radius = window // 2
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    h, w = image.shape
    return (
        integral[window:window + h, window:window + w]
        - integral[:-window, window:window + w]
        - integral[window:window + h, :-window]
        + integral[:-window, :-window]
    ) / float(window * window)


def _resize_map(image: Any, long_edge: int) -> Any:
    np = _np()
    arr = np.asarray(image, dtype=np.float32)
    if not long_edge or int(long_edge) <= 0:
        return arr.copy()
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= int(long_edge):
        return arr.copy()
    scale = float(long_edge) / float(longest)
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    try:
        cv2 = _cv2()
    except RuntimeError:
        cv2 = None
    if cv2 is not None:
        return cv2.resize(arr, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    ys = np.minimum((np.arange(size[1]) / scale).astype(int), h - 1)
    xs = np.minimum((np.arange(size[0]) / scale).astype(int), w - 1)
    return arr[ys[:, None], xs[None, :]].copy()


def _scale_responses(gray: Any, sigma: float,
                     variance_window: int | None = None) -> tuple[Any, Any, Any]:
    np = _np()
    try:
        cv2 = _cv2()
    except RuntimeError:
        cv2 = None
    sigma = max(0.1, float(sigma))
    if cv2 is not None:
        smoothed = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        lap = np.abs(cv2.Laplacian(smoothed, cv2.CV_32F, ksize=3))
        gx = cv2.Scharr(smoothed, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(smoothed, cv2.CV_32F, 0, 1)
        gradient = cv2.magnitude(gx, gy)
        window = max(3, int(variance_window or round(sigma * 3.0)) | 1)
        mean = cv2.boxFilter(smoothed, cv2.CV_32F, (window, window), normalize=True)
        mean2 = cv2.boxFilter(smoothed * smoothed, cv2.CV_32F, (window, window), normalize=True)
        variance = np.maximum(mean2 - mean * mean, 0.0)
        return lap, gradient, variance
    # NumPy fallback: central differences and local variance.
    blurred = _box_blur_numpy(gray, max(3, int(round(sigma * 3.0)) | 1))
    lap = np.abs(
        np.roll(blurred, 1, 0) + np.roll(blurred, -1, 0)
        + np.roll(blurred, 1, 1) + np.roll(blurred, -1, 1)
        - 4.0 * blurred
    )
    gx = (np.roll(blurred, -1, 1) - np.roll(blurred, 1, 1)) * 0.5
    gy = (np.roll(blurred, -1, 0) - np.roll(blurred, 1, 0)) * 0.5
    gradient = np.sqrt(gx * gx + gy * gy)
    window = max(3, int(variance_window or round(sigma * 3.0)) | 1)
    mean = _box_blur_numpy(blurred, window)
    variance = np.maximum(_box_blur_numpy(blurred * blurred, window) - mean * mean, 0.0)
    return lap.astype(np.float32), gradient.astype(np.float32), variance.astype(np.float32)


def _compute_full_map(gray: Any, config: Any) -> Any:
    np = _np()
    low = float(_setting(config, "normalise_low_percentile", 1.0))
    high = float(_setting(config, "normalise_high_percentile", 99.0))
    lw = float(_setting(config, "laplacian_weight", 0.50))
    gw = float(_setting(config, "gradient_weight", 0.30))
    vw = float(_setting(config, "variance_weight", 0.20))
    weight_sum = max(1e-8, abs(lw) + abs(gw) + abs(vw))
    total = np.zeros_like(gray, dtype=np.float32)
    scales = tuple(_setting(config, "scales", (1.0, 2.5, 5.0)))
    if not scales:
        scales = (1.0,)
    for sigma in scales:
        lap, gradient, variance = _scale_responses(
            gray, float(sigma),
            int(_setting(config, "local_variance_window", 7)),
        )
        total += (
            lw * _normalise(lap, low, high)
            + gw * _normalise(gradient, low, high)
            + vw * _normalise(variance, low, high)
        ) / weight_sum
    total = np.clip(total / float(len(scales)), 0.0, 1.0)
    sat_low = float(_setting(config, "saturation_low", 0.01))
    sat_high = float(_setting(config, "saturation_high", 0.99))
    penalty = float(np.clip(_setting(config, "saturation_penalty", 0.35), 0.0, 1.0))
    if penalty:
        saturated = (gray <= sat_low) | (gray >= sat_high)
        total = total * np.where(saturated, 1.0 - penalty, 1.0)
    try:
        cv2 = _cv2()
    except RuntimeError:
        cv2 = None
    blur_sigma = float(_setting(config, "output_blur_sigma", 0.6))
    if cv2 is not None and blur_sigma > 0.0:
        total = cv2.GaussianBlur(total, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    return np.clip(total, 0.0, 1.0).astype(np.float32, copy=False)


def compress_focus_map(focus_map: Any, output_long_edge: int = 512,
                       dtype: str = "float16") -> Any:
    """Downsample and compact a focus map to float16 or uint16."""

    np = _np()
    arr = np.nan_to_num(np.asarray(focus_map, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("focus_map must be a non-empty 2-D array")
    arr = np.clip(arr, 0.0, 1.0)
    arr = _resize_map(arr, int(output_long_edge))
    dtype_name = str(dtype).lower().replace("numpy.", "")
    if dtype_name in {"uint16", "u16", "ushort"}:
        return np.rint(arr * 65535.0).astype(np.uint16)
    if dtype_name in {"float16", "f16", "half"}:
        return arr.astype(np.float16)
    if dtype_name in {"float32", "f32", "single"}:
        return arr.astype(np.float32)
    raise ValueError("focus-map dtype must be float16, float32 or uint16")


def decompress_focus_map(focus_map: Any) -> Any:
    """Return a float32 [0, 1] view/copy suitable for selection."""

    np = _np()
    arr = np.asarray(focus_map)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def compute_focus_map(image: Any, config: Any = None) -> Any:
    """Compute one compressed focus map from one image/analysis array.

    The source is resized to ``analysis_long_edge`` first.  The returned array
    is at most ``output_long_edge`` on its long edge and uses the configured
    compact dtype (float16 by default).
    """

    config = config or FocusMapConfig()
    edge = int(_setting(config, "analysis_long_edge", 1600))
    source = load_image(resolve_image_path(image), max_long_edge=edge) if resolve_image_path(image) is not None else resize_for_analysis(image, edge)
    gray = _gray_float(source)
    full_map = _compute_full_map(gray, config)
    return compress_focus_map(
        full_map,
        int(_setting(config, "output_long_edge", 512)),
        str(_setting(config, "output_dtype", "float16")),
    )


def compute_focus_map_result(image: Any, config: Any = None) -> FocusMapResult:
    """Compute a map plus compact summary metrics."""

    np = _np()
    config = config or FocusMapConfig()
    edge = int(_setting(config, "analysis_long_edge", 1600))
    source = load_image(resolve_image_path(image), max_long_edge=edge) if resolve_image_path(image) is not None else resize_for_analysis(image, edge)
    gray = _gray_float(source)
    full_map = _compute_full_map(gray, config)
    result_map = compress_focus_map(
        full_map,
        int(_setting(config, "output_long_edge", 512)),
        str(_setting(config, "output_dtype", "float16")),
    )
    values = decompress_focus_map(result_map)
    texture = float(np.mean(values > float(_setting(config, "focus_texture_threshold", 0.35))))
    return FocusMapResult(
        focus_map=result_map,
        sharpness_score=float(np.mean(values)),
        texture_fraction=texture,
        mean_focus=float(np.mean(values)),
        focus_std=float(np.std(values)),
        source_shape=tuple(int(v) for v in gray.shape[:2]),
    )


def analyze_focus(image: Any, config: Any = None) -> FocusMapResult:
    return compute_focus_map_result(image, config)


def build_focus_map(image: Any, config: Any = None) -> Any:
    return compute_focus_map(image, config)


def compute_registered_focus_map(image: Any, registration: Any,
                                 config: Any = None) -> Any:
    """Compute a compact map after applying low-resolution registration."""

    config = config or FocusMapConfig()
    edge = int(_setting(config, "analysis_long_edge", 1600))
    source = load_image(resolve_image_path(image), max_long_edge=edge) if resolve_image_path(image) is not None else resize_for_analysis(image, edge)
    aligned = warp_image(source, registration, output_shape=source.shape[:2])
    gray = _gray_float(aligned)
    full_map = _compute_full_map(gray, config)
    return compress_focus_map(
        full_map,
        int(_setting(config, "output_long_edge", 512)),
        str(_setting(config, "output_dtype", "float16")),
    )


def focus_map_similarity(map_a: Any, map_b: Any) -> float:
    """Compare two compact focus maps in [0, 1].

    Correlation captures the shape of clear regions while normalised absolute
    difference prevents two flat maps from looking identical merely because
    their correlation is undefined.  Inputs may be float16 or uint16.
    """

    np = _np()
    a = decompress_focus_map(map_a)
    b = decompress_focus_map(map_b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("focus maps must be 2-D")
    if a.shape != b.shape:
        b = _resize_map(b, max(a.shape))
        try:
            cv2 = _cv2()
        except RuntimeError:
            cv2 = None
        if cv2 is not None:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            ys = np.minimum((np.arange(a.shape[0]) * b.shape[0] / a.shape[0]).astype(int), b.shape[0] - 1)
            xs = np.minimum((np.arange(a.shape[1]) * b.shape[1] / a.shape[1]).astype(int), b.shape[1] - 1)
            b = b[ys[:, None], xs[None, :]]
    af = a.ravel().astype(np.float32)
    bf = b.ravel().astype(np.float32)
    af -= float(np.mean(af))
    bf -= float(np.mean(bf))
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    corr = 1.0 if denom <= 1e-8 else float(np.dot(af, bf) / denom)
    corr_sim = (corr + 1.0) * 0.5
    diff_sim = 1.0 - float(np.mean(np.abs(a - b)))
    # Compare high-focus support as a spatial-overlap signal as well.
    mask_a = _focus_support_mask(a, 70.0)
    mask_b = _focus_support_mask(b, 70.0)
    union = np.count_nonzero(mask_a | mask_b)
    overlap = 1.0 if union == 0 else float(np.count_nonzero(mask_a & mask_b) / union)
    return float(np.clip(0.45 * corr_sim + 0.30 * diff_sim + 0.25 * overlap, 0.0, 1.0))


def map_correlation(map_a: Any, map_b: Any) -> float:
    """Pearson correlation convenience wrapper."""

    np = _np()
    a = decompress_focus_map(map_a).ravel().astype(np.float32)
    b = decompress_focus_map(map_b).ravel().astype(np.float32)
    if a.size != b.size:
        raise ValueError("maps must have the same number of elements")
    if a.size == 0:
        return 0.0
    std = float(np.std(a) * np.std(b))
    if std <= 1e-8:
        return 1.0 if float(np.mean(np.abs(a - b))) <= 1e-6 else 0.0
    return float(np.clip(np.mean((a - a.mean()) * (b - b.mean())) / std, -1.0, 1.0))


def _focus_support_mask(values: Any, percentile: float = 70.0) -> Any:
    """Return a useful high-response mask, including sparse binary maps.

    A plain ``values >= percentile(values, 70)`` test turns every zero into a
    positive pixel when fewer than 30% of a map is sharp.  A small
    mean-plus-variance fallback keeps the support spatially meaningful and
    makes duplicate/transition metrics stable.
    """

    np = _np()
    values = np.asarray(values, dtype=np.float32)
    threshold = float(np.percentile(values, percentile))
    floor = float(np.min(values))
    if threshold <= floor + 1e-7:
        threshold = float(np.mean(values) + 0.25 * np.std(values))
    if threshold <= floor + 1e-7:
        return np.zeros_like(values, dtype=bool)
    return values > threshold


focus_map = compute_focus_map
FocusAnalysisConfig = FocusMapConfig
calculate_focus_map = compute_focus_map
create_focus_map = compute_focus_map

__all__ = [
    "FocusMapConfig", "FocusAnalysisConfig", "FocusMapResult", "compute_focus_map",
    "calculate_focus_map", "create_focus_map",
    "compute_registered_focus_map",
    "compute_focus_map_result", "analyze_focus", "build_focus_map",
    "focus_map", "compress_focus_map", "decompress_focus_map",
    "focus_map_similarity", "map_correlation",
]

