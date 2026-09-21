"""Geometry checks for coloured products, independent of backdrop brightness."""
from __future__ import annotations

import cv2
import numpy as np


def colour_subject_mask(rgb):
    """Find substantial chromatic regions; ignore neutral stands and reflections.

    Blur before segmentation so focus changes and fine printed text do not
    dominate the silhouette. Uniform colour frames are not isolated subjects.
    """
    smooth = cv2.GaussianBlur(rgb, (0, 0), 2)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 65)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    minimum = max(16, int(mask.size * 0.0005))
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum]
    mask = np.isin(labels, keep).astype(np.uint8)
    fraction = float(mask.mean())
    if not 0.001 < fraction < 0.45:
        return None
    return mask


def colour_subject_mismatch(reference, current):
    """Return a reason when silhouettes differ beyond defocus tolerance.

    Distances are measured without fitting another transform: doing so would
    hide the very residual displacement this check is supposed to detect.
    Compare both directions to catch enlarged and duplicated subjects too.
    """
    if reference is None and current is None:
        return None
    if reference is None or current is None:
        return "colour subject disappeared between aligned frames"
    if reference.shape != current.shape:
        return "colour subject canvas dimensions differ"
    tolerance = max(2.0, max(reference.shape) / 320.0)
    for source, target in ((reference, current), (current, reference)):
        distances = cv2.distanceTransform(1 - target, cv2.DIST_L2, 5)
        residual = float(np.percentile(distances[source != 0], 95))
        if residual > tolerance:
            return f"colour subject residual {residual:.2f}px exceeds {tolerance:.2f}px"
    return None
