"""Quality scoring used to choose the best member of duplicate focus groups."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Iterator

from .focus_map import decompress_focus_map
from .registration import load_image, resolve_image_path, resize_for_analysis
from .types import ImageQuality, RegistrationResult


@dataclass(slots=True)
class QualityConfig:
    """Weights and normalisation controls for a [0, 1] quality score."""

    analysis_long_edge: int = 1600
    sharpness_weight: float = 0.45
    motion_stability_weight: float = 0.20
    exposure_weight: float = 0.15
    clipping_weight: float = 0.10
    noise_weight: float = 0.10
    focus_map_weight: float = 0.0
    clipping_threshold: float = 0.01
    saturation_low: float = 0.01
    saturation_high: float = 0.99
    noise_smooth_sigma: float = 1.2
    sharpness_scale: float = 1200.0


def _setting(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    value = getattr(config, name, None)
    if value is None and name == "analysis_long_edge":
        value = getattr(config, "focus_analysis_long_edge", None)
    if value is None:
        section = getattr(config, "analysis", None)
        if section is not None:
            value = getattr(section, name, None)
            if value is None and name == "analysis_long_edge":
                value = getattr(section, "focus_analysis_long_edge", None)
    return default if value is None else value


def _np() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - minimal env only
        raise RuntimeError("quality scoring requires NumPy") from exc
    return np


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - minimal env only
        raise RuntimeError("quality scoring requires OpenCV") from exc
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


def _blur(gray: Any, sigma: float) -> Any:
    np = _np()
    try:
        cv2 = _cv2()
    except RuntimeError:
        cv2 = None
    if cv2 is not None:
        return cv2.GaussianBlur(gray, (0, 0), sigmaX=max(0.1, float(sigma)))
    # Approximate with a three-point separable kernel if OpenCV is absent.
    return (gray + np.roll(gray, 1, 0) + np.roll(gray, -1, 0)
            + np.roll(gray, 1, 1) + np.roll(gray, -1, 1)) / 5.0


def _sharpness(gray: Any, config: Any) -> tuple[float, float]:
    np = _np()
    try:
        cv2 = _cv2()
    except RuntimeError:
        cv2 = None
    if cv2 is not None:
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    else:
        lap = (
            np.roll(gray, 1, 0) + np.roll(gray, -1, 0)
            + np.roll(gray, 1, 1) + np.roll(gray, -1, 1) - 4.0 * gray
        )
    variance = float(np.var(lap))
    scale = max(1.0, float(_setting(config, "sharpness_scale", 1200.0)))
    # A saturating log maps a broad range of cameras/images to [0, 1] while
    # preserving ordering between blurred and in-focus frames.
    sharp = float(np.clip(np.log1p(variance * scale) / np.log1p(scale), 0.0, 1.0))
    return sharp, variance


def _exposure(gray: Any, config: Any) -> tuple[float, float, float]:
    np = _np()
    low = float(_setting(config, "saturation_low", 0.01))
    high = float(_setting(config, "saturation_high", 0.99))
    clipped = float(np.mean((gray <= low) | (gray >= high)))
    median = float(np.median(gray))
    # Midtone distance is deliberately soft: product shots may be bright or
    # dark, but a nearly empty dynamic range is still undesirable.
    midtone = float(np.clip(1.0 - abs(median - 0.5) * 1.35, 0.0, 1.0))
    p_low, p_high = np.percentile(gray, [2.0, 98.0])
    dynamic = float(np.clip((p_high - p_low) / 0.70, 0.0, 1.0))
    exposure = 0.65 * midtone + 0.35 * dynamic
    return float(np.clip(exposure, 0.0, 1.0)), clipped, dynamic


def _noise_penalty(gray: Any, config: Any) -> tuple[float, float]:
    np = _np()
    smooth = _blur(gray, float(_setting(config, "noise_smooth_sigma", 1.2)))
    residual = gray - smooth
    # Estimate noise mostly on low-gradient pixels so textured/sharp detail is
    # not incorrectly penalised as sensor noise.
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    flat = (np.abs(gx) + np.abs(gy)) < 0.04
    if np.count_nonzero(flat) < max(32, gray.size // 100):
        flat = np.ones_like(gray, dtype=bool)
    sigma = float(np.std(residual[flat]))
    # 0.025 is a mild amount of JPEG/high-ISO residual; 0.10 is clearly noisy
    # in a normalised image.
    penalty = float(np.clip((sigma - 0.008) / 0.08, 0.0, 1.0))
    return penalty, sigma


def motion_stability_score(registration: Any) -> float:
    """Map registration evidence to a stability score in [0, 1]."""

    if registration is None:
        return 0.75  # no evidence is preferable to assuming motion failure
    np = _np()
    confidence = float(np.clip(getattr(registration, "confidence", 0.0), 0.0, 1.0))
    ratio = float(np.clip(getattr(registration, "inlier_ratio", 0.0), 0.0, 1.0))
    tx, ty = getattr(registration, "translation", (0.0, 0.0))
    translation = (abs(float(tx)) + abs(float(ty)))
    scale = abs(math_log_safe(float(getattr(registration, "scale", 1.0))))
    rotation = abs(float(getattr(registration, "rotation_degrees", 0.0))) / 15.0
    motion_penalty = min(1.0, translation / 80.0 + scale + rotation)
    base = 0.55 * confidence + 0.45 * ratio
    if getattr(registration, "model", "") == "identity" and not getattr(registration, "valid", False):
        # An identity fallback usually means that this frame was too
        # texture-poor for feature matching, not that the camera definitely
        # moved.  Give it a neutral floor while still ranking verified
        # inliers above it.
        base = max(base, 0.45 + 0.20 * confidence)
    return float(np_clip(0.95 * base + 0.05 * (1.0 - motion_penalty), 0.0, 1.0))


def math_log_safe(value: float) -> float:
    """Small helper kept dependency-free for the motion score."""

    import math
    if value <= 0.0:
        return 0.0
    return math.log(value)


def np_clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def score_image(image: Any, focus_map: Any = None,
                registration: Any = None, config: Any = None) -> ImageQuality:
    """Score one image without retaining its source pixels.

    ``focus_map`` can be a compressed float16/uint16 map produced by
    :func:`focus_stack_app.core.focus_map.compute_focus_map`; it contributes
    only when ``focus_map_weight`` is configured above zero.
    """

    # Also accept ``score_image(image, config)`` for callers that do not have
    # a precomputed map; a config object has distinctive weight/resolution
    # attributes whereas a map is an array-like value.
    if config is None and focus_map is not None and (
        hasattr(focus_map, "sharpness_weight")
        or hasattr(focus_map, "focus_analysis_long_edge")
    ):
        config = focus_map
        focus_map = None
    np = _np()
    config = config or QualityConfig()
    edge = int(_setting(config, "analysis_long_edge", 1600))
    source = load_image(resolve_image_path(image), max_long_edge=edge) if resolve_image_path(image) is not None else resize_for_analysis(image, edge)
    gray = _gray_float(source)
    sharp, lap_variance = _sharpness(gray, config)
    exposure, clipped, dynamic = _exposure(gray, config)
    noise, noise_sigma = _noise_penalty(gray, config)
    stability = motion_stability_score(registration)
    clip_threshold = float(_setting(config, "clipping_threshold", 0.01))
    clipping_penalty = float(np.clip(clipped / max(clip_threshold, 1e-6), 0.0, 1.0))
    focus_component = 0.0
    if focus_map is not None:
        values = decompress_focus_map(focus_map)
        focus_component = float(np.clip(np.mean(values) + 0.35 * np.percentile(values, 90), 0.0, 1.0) / 1.35)
    sw = float(_setting(config, "sharpness_weight", 0.45))
    mw = float(_setting(config, "motion_stability_weight", 0.20))
    ew = float(_setting(config, "exposure_weight", 0.15))
    cw = float(_setting(config, "clipping_weight", 0.10))
    nw = float(_setting(config, "noise_weight", 0.10))
    fw = max(0.0, float(_setting(config, "focus_map_weight", 0.0)))
    total = sw + mw + ew + cw + nw + fw
    if total <= 0.0:
        total = 1.0
    score = (sw * sharp + mw * stability + ew * exposure
             + cw * (1.0 - clipping_penalty) + nw * (1.0 - noise)
             + fw * focus_component) / total
    return ImageQuality(
        score=float(np.clip(score, 0.0, 1.0)),
        sharpness=float(sharp),
        motion_stability=float(stability),
        exposure=float(exposure),
        clipping_penalty=float(clipping_penalty),
        noise_penalty=float(noise),
        details={
            "laplacian_variance": float(lap_variance),
            "clipped_fraction": float(clipped),
            "dynamic_range": float(dynamic),
            "noise_sigma": float(noise_sigma),
            "focus_component": float(focus_component),
        },
    )


def quality_score(image: Any, focus_map: Any = None,
                  registration: Any = None, config: Any = None) -> float:
    """Return only the final scalar score for convenient sorting."""

    return score_image(image, focus_map, registration, config).score


def compute_quality(image: Any, focus_map: Any = None,
                    registration: Any = None, config: Any = None) -> ImageQuality:
    return score_image(image, focus_map, registration, config)


score_quality = score_image


def score_images(images: Iterable[Any], focus_maps: Iterable[Any] | None = None,
                 registrations: Iterable[Any] | None = None,
                 config: Any = None) -> Iterator[ImageQuality]:
    """Stream quality scores, retaining no image arrays between iterations."""

    map_iter = iter(focus_maps) if focus_maps is not None else None
    reg_iter = iter(registrations) if registrations is not None else None
    for image in images:
        fmap = next(map_iter) if map_iter is not None else None
        reg = next(reg_iter) if reg_iter is not None else None
        yield score_image(image, fmap, reg, config)


__all__ = [
    "QualityConfig", "ImageQuality", "score_image", "score_quality",
    "quality_score", "compute_quality", "score_images", "motion_stability_score",
]

