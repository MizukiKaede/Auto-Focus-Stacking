"""Frozen pre-C++ numerical references for exact-output regression tests."""
import cv2
import numpy as np
from focus_stack_app.utils.performance import timed
from focus_stack_app.fusion.focus_masks import _ordered_prefetched_frames, focus_weight

def _nearest_edge_support(strength, edge, radius):
    """Extend the nearest physical edge, without borrowing a distant corner."""
    present = edge != 0
    if not np.any(present):
        return np.zeros_like(strength)
    density = cv2.GaussianBlur(present.astype(np.float32), (0, 0), 2.0)
    averaged = cv2.GaussianBlur(strength * present, (0, 0), 2.0)
    averaged /= np.maximum(density, 1e-06)
    distance, nearest = cv2.distanceTransformWithLabels(np.uint8(~present), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    values = np.zeros(int(nearest.max()) + 1, np.float32)
    values[nearest[present]] = averaged[present]
    weight = np.clip((radius + 1.0 - distance) / max(2.0, radius / 4.0), 0, 1)
    return values[nearest] * weight

@timed('fusion')
def blend_focus_pyramid(count, load_aligned, labels, *, cancel_event=None, levels=1, prefetch=True):
    """Blend focus winners without adding dark/bright silhouette halos.

    The default is a convex RGB blend with a one-pixel transition. Combining
    sharp detail with a defocused frame's coarse pyramid layers can overshoot
    the brightness of every source, creating a wide dark rim on a dark
    background (or a bright rim on a light one). Do not use those layers for
    the default focus stack. Explicit multiband callers remain supported.

    Accumulate one frame at a time to keep RAM independent of frame count.
    """
    if levels < 1:
        raise ValueError('focus blend levels must be at least 1')
    accumulators = []
    if prefetch:
        frame_iterator = _ordered_prefetched_frames(count, load_aligned, cancel_event)
    else:

        def synchronous_frames():
            for index in range(count):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError('focus fusion cancelled')
                yield (index, load_aligned(index))
        frame_iterator = synchronous_frames()
    try:
        for index, rgb in frame_iterator:
            if prefetch and cancel_event is not None and cancel_event.is_set():
                raise RuntimeError('focus fusion cancelled')
            pixels = rgb.astype(np.float32)
            del rgb
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
    finally:
        close = getattr(frame_iterator, 'close', None)
        if callable(close):
            close()
    result = accumulators.pop()
    for detail in reversed(accumulators):
        result = cv2.pyrUp(result, dstsize=(detail.shape[1], detail.shape[0])) + detail
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)
