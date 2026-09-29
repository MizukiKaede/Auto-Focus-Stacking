"""Small, deterministic cleanup for the quality renderer's focus ownership.

The focus detector works at full output resolution.  On low-colour materials
it can still switch owners within a few pixels, making a sharp edge look like
several tiny blocks.  Smooth only those ownership islands; retain the
detector's decisions on coloured print and surfaces.
"""

from __future__ import annotations

import cv2
import numpy as np


def stabilize_neutral_labels(labels: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Remove tiny focus-owner islands without softening the source pixels.

    This changes labels, not RGB data.  The final renderer still copies the
    selected full-resolution frame at each pixel with a one-pixel seam.
    """
    if labels.shape != reference.shape[:2]:
        raise ValueError("focus labels and aligned reference have different dimensions")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("aligned reference must be RGB")
    if labels.size == 0 or int(labels.max()) > 255:
        return labels

    gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    red, green, blue = cv2.split(reference)
    chroma = cv2.subtract(cv2.max(red, cv2.max(green, blue)),
                          cv2.min(red, cv2.min(green, blue)))
    del red, green, blue
    neutral = (gray < 235) & (chroma < 65)
    spatial_median = cv2.medianBlur(labels.astype(np.uint8), 15)
    stable = labels.copy()
    stable[neutral] = spatial_median[neutral]
    return stable


__all__ = ["stabilize_neutral_labels"]

