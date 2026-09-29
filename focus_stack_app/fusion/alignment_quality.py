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


def defocus_geometry_consistent(reference_rgb, current_rgb, reference_mask, current_mask):
    """Confirm a failed colour contour using geometry, without moving pixels.

    Defocus changes saturation and opens/closes white print inside a coloured
    product. It need not move the product. Use its coarse centreline for long
    subjects, then a measured residual affine fit when that is inconclusive.
    A fitted transform is only measured here; it is never applied to inputs.
    """
    if reference_mask is None or current_mask is None or reference_mask.shape != current_mask.shape:
        return False
    tolerance = max(2.0, max(reference_mask.shape) / 320.0)
    clouds = []
    for mask in (reference_mask, current_mask):
        points = cv2.findNonZero(mask)
        if points is None or len(points) < 16:
            return False
        hull = np.zeros_like(mask)
        cv2.fillConvexPoly(hull, cv2.convexHull(points), 1)
        y, x = np.where(hull != 0)
        clouds.append(np.column_stack((x, y)).astype(np.float32))
    left, right = clouds
    centre = left.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(left.T))
    if values[1] > 9 * max(values[0], 1):
        axis = vectors[:, 1]
        normal = np.array([-axis[1], axis[0]])
        la, lt = (left - centre) @ axis, (left - centre) @ normal
        ra, rt = (right - centre) @ axis, (right - centre) @ normal
        l0, l1 = np.percentile(la, (1, 99))
        r0, r1 = np.percentile(ra, (1, 99))
        length_ratio = (r1 - r0) / max(1, l1 - l0)
        boundaries = np.linspace(max(l0, r0), min(l1, r1), 25)
        distances, widths = [], []
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            l = lt[(la >= lo) & (la < hi)]
            r = rt[(ra >= lo) & (ra < hi)]
            if min(len(l), len(r)) < 16:
                continue
            ll, lh = np.percentile(l, (5, 95))
            rl, rh = np.percentile(r, (5, 95))
            distances.append(abs((ll + lh - rl - rh) / 2))
            widths.append((rh - rl) / max(1, lh - ll))
        if (len(distances) >= 20 and 0.97 <= length_ratio <= 1.03
                and abs((l0 + l1 - r0 - r1) / 2) <= 2 * tolerance
                and np.percentile(distances, 90) <= tolerance
                and 0.75 <= np.median(widths) <= 1.33):
            return True

    # The colour mask may lose an entire low-saturation tip. Check the actual
    # blurred intensities within the subject, keeping the original displacement
    # tolerance. Neutral textured backdrops must not dominate this fit.
    height, width = reference_mask.shape
    scale = min(1.0, 640.0 / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    gray = [cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), size,
                       interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            for rgb in (reference_rgb, current_rgb)]
    mask = cv2.resize(reference_mask | current_mask, size, interpolation=cv2.INTER_NEAREST)
    mask = cv2.dilate(mask, np.ones((11, 11), np.uint8)) * 255
    try:
        correlation, matrix = cv2.findTransformECC(
            gray[0], gray[1], np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), mask, 11)
    except cv2.error:
        return False
    if correlation < 0.90 or not np.isfinite(matrix).all():
        return False
    points = left[::max(1, len(left) // 2048)] * scale
    displacement = points @ matrix[:, :2].T + matrix[:, 2] - points
    return bool(np.percentile(np.linalg.norm(displacement, axis=1), 95) <= tolerance * scale)
