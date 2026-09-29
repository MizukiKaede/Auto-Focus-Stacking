"""The established whole-frame selection algorithm used by scene captures.

The implementation deliberately preserves the calibrated selection behaviour:
highest whole-frame focus score as the preview anchor, affine ECC alignment,
90% local-best sharp masks, 0.05% marginal-gain cutoff, and 99.95% coverage.
Names and business semantics remain generic so every fusion backend consumes
the same result.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import time
from typing import Any, Callable, Iterable, Iterator

from ..utils.performance import stage, timed

import threading

import cv2
import numpy as np

from ..utils.image_io import load_rgb
from .group_detector import subject_feature, subject_changed


@contextmanager
def _ordered_bounded_map(
    function: Callable[[Any], Any],
    values: Iterable[Any],
    *,
    workers: int,
    cancel_event: threading.Event,
) -> Iterator[Iterator[Any]]:
    """Yield results in input order with at most ``workers`` futures alive.

    Submitting the complete group at once would let completed futures retain a
    full decoded frame or focus map while an earlier item is still running.
    This sliding window keeps transient memory proportional to concurrency.
    """

    iterator = iter(values)
    if workers <= 1:
        def serial() -> Iterator[Any]:
            for value in iterator:
                if cancel_event.is_set():
                    raise RuntimeError("selection cancelled")
                yield function(value)

        yield serial()
        return

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="focus-analysis")
    pending: deque[Future[Any]] = deque()

    def submit_one() -> bool:
        if cancel_event.is_set():
            raise RuntimeError("selection cancelled")
        try:
            value = next(iterator)
        except StopIteration:
            return False
        pending.append(executor.submit(function, value))
        return True

    try:
        for _ in range(workers):
            if not submit_one():
                break

        def parallel() -> Iterator[Any]:
            while pending:
                future = pending.popleft()
                result = future.result()
                if cancel_event.is_set():
                    raise RuntimeError("selection cancelled")
                yield result
                submit_one()

        yield parallel()
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def focus_score(rgb):
    gray = (rgb if np.asarray(rgb).ndim == 2 else cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)).astype(np.float32) / 255
    gray = cv2.GaussianBlur(gray, (0, 0), 0.6)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return cv2.GaussianBlur(laplacian * laplacian, (0, 0), 2.0)


def _registration_pyramid(rgb, target_shape=None):
    gray = (rgb if np.asarray(rgb).ndim == 2 else cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)).astype(np.float32) / 255
    height, width = target_shape or rgb.shape[:2]
    half_size = (round(width * 0.5), round(height * 0.5))
    full_size = (round(width), round(height))
    return {
        0.5: cv2.resize(gray, half_size),
        1.0: cv2.resize(gray, full_size),
    }


def register_whole_frame(reference, source, *, prepared_reference=None):
    """Coarse-to-fine affine ECC using the established scene thresholds."""
    warp = np.eye(2, 3, dtype=np.float32)
    reference_pyramid = prepared_reference or _registration_pyramid(reference)
    source_pyramid = _registration_pyramid(source, target_shape=reference.shape[:2])
    correlation = 0.0
    for scale, blur in ((0.5, 9), (1.0, 7), (1.0, 3)):
        ref = reference_pyramid[scale]
        src = source_pyramid[scale]
        scaled_warp = warp.copy()
        scaled_warp[:, 2] *= scale
        correlation, scaled_warp = cv2.findTransformECC(
            ref, src, scaled_warp, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
            None, blur,
        )
        warp = scaled_warp
        warp[:, 2] /= scale
    scales = np.linalg.svd(warp[:, :2], compute_uv=False)
    if correlation < 0.90 or scales.min() < 0.8 or scales.max() > 1.25:
        raise ValueError(f"whole-frame alignment rejected: correlation={correlation:.4f}, scale={scales.tolist()}")
    return warp, float(correlation)


def warp_preview(rgb, matrix, shape):
    return cv2.warpAffine(
        rgb, np.asarray(matrix, np.float32), (shape[1], shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _estimated_rgb_cache_bytes(paths, edge):
    """Return an exact standard-loader estimate without decoding pixels."""
    try:
        from ..utils.image_io import read_image_size

        total = 0
        for path in paths:
            size = read_image_size(path)
            scale = min(1.0, float(edge) / max(size.width, size.height))
            width = max(1, round(size.width * scale))
            height = max(1, round(size.height * scale))
            total += width * height * 3
        return total
    except Exception:
        # Custom loaders and synthetic test paths do not necessarily expose
        # JPEG headers.  The first decoded frame supplies a conservative
        # same-shape estimate; inconsistent shapes are rejected below anyway.
        return None


@timed("selection")
def select_whole_frame(
    paths, *, cancel_event=None, progress=None, edge=1280, loader=load_rgb,
    frame_cache_bytes=400 * 1024**2, workers=1, requested_workers=None,
):
    """Return the calibrated selection plan with optional exact-pixel reuse."""
    cancel_event = cancel_event or threading.Event()
    paths = [str(path) for path in paths]
    effective_workers = max(1, min(len(paths) or 1, int(workers)))
    requested_workers = int(workers if requested_workers is None else requested_workers)
    qualities, shapes = [], set()
    previous_subject = None
    previous_path = None
    budget = max(0, int(frame_cache_bytes or 0))
    estimate = _estimated_rgb_cache_bytes(paths, edge) if budget else None
    cache_enabled = bool(budget and estimate is not None and estimate <= budget)
    cache_decided = estimate is not None or not budget
    frames = {}
    cache_bytes = 0
    cache_peak_bytes = 0
    decode_count = 0
    decode_lock = threading.Lock()

    def decode(path):
        nonlocal decode_count
        with decode_lock:
            decode_count += 1
        return loader(path, edge)

    def inspect_frame(value):
        index, path = value
        rgb = decode(path)
        ratio = min(1.0, 640 / max(rgb.shape[:2]))
        small = cv2.resize(rgb, (round(rgb.shape[1] * ratio), round(rgb.shape[0] * ratio)), interpolation=cv2.INTER_AREA) if ratio < 1 else rgb
        subject = subject_feature(small) if rgb.ndim == 3 else None
        quality = float(focus_score(rgb).mean())
        return index, path, rgb, subject, quality

    quality_started = time.perf_counter()
    with stage(
        "selection_quality_scan",
        requested_workers=requested_workers,
        effective_workers=effective_workers,
    ):
        with _ordered_bounded_map(
            inspect_frame,
            enumerate(paths),
            workers=effective_workers,
            cancel_event=cancel_event,
        ) as results:
            for index, path, rgb, subject, quality in results:
                if not cache_decided:
                    cache_enabled = int(getattr(rgb, "nbytes", 0)) * len(paths) <= budget
                    cache_decided = True
                if cache_enabled:
                    frame_bytes = int(getattr(rgb, "nbytes", 0))
                    if cache_bytes + frame_bytes > budget:
                        # Unexpected aspect/dtype variation invalidated the initial
                        # estimate. Drop the complete cache instead of partial reuse.
                        frames.clear()
                        cache_bytes = 0
                        cache_enabled = False
                    else:
                        frames[index] = rgb
                        cache_bytes += frame_bytes
                        cache_peak_bytes = max(cache_peak_bytes, cache_bytes)
                if subject_changed(previous_subject, subject):
                    raise ValueError(f"主体发生转面或明显位移，请拆分照片组：{previous_path} → {path}")
                previous_subject, previous_path = subject, path
                qualities.append(quality)
                shapes.add(rgb.shape)
                if not cache_enabled:
                    rgb = None
    quality_scan_seconds = time.perf_counter() - quality_started
    if len(shapes) != 1:
        raise ValueError("images in one group have inconsistent dimensions")
    reference_index = int(np.argmax(qualities))
    reference = frames[reference_index] if cache_enabled else decode(paths[reference_index])
    reference_pyramid = _registration_pyramid(reference)
    scores, matrices, correlations, errors = [], [], [], {}

    def analyze_registered_frame(value):
        index, path = value
        try:
            rgb = frames[index] if cache_enabled else decode(path)
            if index == reference_index:
                matrix, correlation = np.eye(2, 3, dtype=np.float32), 1.0
            else:
                matrix, correlation = register_whole_frame(
                    reference, rgb, prepared_reference=reference_pyramid,
                )
            aligned = warp_preview(rgb, matrix, reference.shape)
            score = focus_score(aligned)
        except (cv2.error, ValueError) as exc:
            matrix, correlation = np.eye(2, 3, dtype=np.float32), 0.0
            score = np.zeros(reference.shape[:2], np.float32)
            return index, score.astype(np.float16), matrix.tolist(), correlation, str(exc)
        return index, score.astype(np.float16), matrix.tolist(), correlation, None

    registration_started = time.perf_counter()
    with stage(
        "selection_registration",
        requested_workers=requested_workers,
        effective_workers=effective_workers,
    ):
        with _ordered_bounded_map(
            analyze_registered_frame,
            enumerate(paths),
            workers=effective_workers,
            cancel_event=cancel_event,
        ) as results:
            for index, score, matrix, correlation, error in results:
                if error is not None:
                    errors[index] = error
                scores.append(score)
                matrices.append(matrix)
                correlations.append(correlation)
                if progress:
                    progress(index + 1, len(paths))
    registration_seconds = time.perf_counter() - registration_started
    best = np.maximum.reduce(scores)
    useful = best > max(float(best.max()) * 0.002, 1e-7)
    threshold = best * 0.90
    sharp_masks = [(score >= threshold) & useful for score in scores]
    covered = np.zeros_like(useful)
    selected, gains = [], {}
    while True:
        options = [index for index in range(len(paths)) if index not in selected and index not in errors]
        if not options:
            break
        uncovered = ~covered
        counts = {index: int(np.count_nonzero(sharp_masks[index] & uncovered)) for index in options}
        candidate = max(options, key=lambda index: (counts[index], qualities[index], -index))
        gain = counts[candidate] / max(1, int(useful.sum()))
        if selected and gain < 0.0005:
            break
        selected.append(candidate)
        gains[candidate] = gain
        covered |= sharp_masks[candidate]
        if np.count_nonzero(covered) / max(1, int(useful.sum())) >= 0.9995:
            break
    if reference_index not in selected:
        selected.append(reference_index)
    selected.sort()
    coverage = float(np.count_nonzero(covered) / max(1, int(useful.sum())))
    return {
        "paths": paths,
        "selected_indices": selected,
        "reference_index": reference_index,
        "matrices": matrices,
        "correlations": correlations,
        "errors": errors,
        "preview_shape": list(reference.shape[:2]),
        "qualities": qualities,
        "coverage": coverage,
        "gains": gains,
        "decode_count": decode_count,
        "frame_cache_enabled": cache_enabled,
        "frame_cache_hits": (len(paths) + 1) if cache_enabled else 0,
        "frame_cache_peak_bytes": cache_peak_bytes,
        "frame_cache_estimated_bytes": estimate,
        "analysis_workers_requested": requested_workers,
        "analysis_workers_effective": effective_workers,
        "opencv_threads": int(cv2.getNumThreads()),
        "quality_scan_seconds": quality_scan_seconds,
        "registration_seconds": registration_seconds,
    }


__all__ = ["focus_score", "register_whole_frame", "warp_preview", "select_whole_frame"]

