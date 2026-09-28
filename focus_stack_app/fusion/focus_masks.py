"""Full-resolution focus decisions with bounded, frame-count independent RAM."""
from __future__ import annotations

from ..utils.performance import timed

import cv2
import numpy as np


def focus_response(rgb):
    """Compare fine detail on a common scale, including its defocus fringe.

    A defocused edge can win just outside the sharp silhouette, where the
    sharp image is flat. Extend nearby focus evidence by seven source pixels
    so that this fringe belongs to the sharp edge too. Do not normalise each
    frame independently: that would promote a uniformly blurry frame.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lap = cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 0.6), cv2.CV_32F, ksize=3)
    response = cv2.GaussianBlur(lap * lap, (0, 0), 4.0)
    return cv2.dilate(response, np.ones((15, 15), np.uint8))


@timed("focus_masks")
def build_focus_labels(count, load_aligned, *, cancel_event=None):
    """Read one registered RGB frame at a time and retain only winner maps."""
    if not 1 <= count <= 65535:
        raise ValueError("focus fusion requires between 1 and 65535 frames")
    best = labels = None
    for index in range(count):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("focus fusion cancelled")
        rgb = load_aligned(index)
        score = focus_response(rgb)
        if best is None:
            best = score
            labels = np.zeros(score.shape, np.uint16)
        else:
            if score.shape != best.shape:
                raise ValueError("aligned focus frames must have matching dimensions")
            better = score > best
            best[better] = score[better]
            labels[better] = index
        del rgb, score
    return labels


def focus_weight(labels, index):
    """Feather only one output pixel, without averaging away thin features."""
    return cv2.GaussianBlur((labels == index).astype(np.float32), (0, 0), 1.0)


@timed("fusion")
def blend_focus_pyramid(count, load_aligned, labels, *, cancel_event=None, levels=1):
    """Blend focus winners without adding dark/bright silhouette halos.

    The default is a convex RGB blend with a one-pixel transition. Combining
    sharp detail with a defocused frame's coarse pyramid layers can overshoot
    the brightness of every source, creating a wide dark rim on a dark
    background (or a bright rim on a light one). Do not use those layers for
    the default focus stack. Explicit multiband callers remain supported.

    Accumulate one frame at a time to keep RAM independent of frame count.
    """
    if levels < 1:
        raise ValueError("focus blend levels must be at least 1")
    accumulators = []
    for index in range(count):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("focus fusion cancelled")
        pixels = load_aligned(index).astype(np.float32)
        weight = focus_weight(labels, index)
        for level in range(levels):
            last = level == levels - 1 or min(pixels.shape[:2]) <= 2
            if not last:
                smaller = cv2.pyrDown(pixels)
                detail = pixels - cv2.pyrUp(smaller, dstsize=(pixels.shape[1], pixels.shape[0]))
            else:
                detail = pixels
            contribution = detail * weight[..., None]
            if index == 0:
                accumulators.append(contribution)
            else:
                accumulators[level] += contribution
            if last:
                break
            pixels = smaller
            weight = cv2.pyrDown(weight)
    result = accumulators.pop()
    for detail in reversed(accumulators):
        result = cv2.pyrUp(result, dstsize=(detail.shape[1], detail.shape[0])) + detail
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)

