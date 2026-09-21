"""Capture grouping with whole-frame and isolated bright-background subject checks."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from ..utils.image_io import load_rgb
from .registration import resolve_image_path
from .types import SceneGroup


@dataclass(frozen=True)
class StackGroupDetectorConfig:
    composition_change: float = 0.025
    pause_seconds: float = 20.0
    pause_change: float = 0.006
    long_pause_seconds: float = 60.0

    def __post_init__(self):
        if not np.isfinite(self.pause_seconds) or self.pause_seconds <= 0:
            raise ValueError("pause_seconds must be positive and finite")


def subject_feature(rgb):
    """Describe an isolated dark subject on a mostly bright neutral backdrop.

    Border-connected objects (backdrop edges, stands) are excluded. Other
    scenes retain the general composition detector. Compare a shared region
    later, since defocus can disconnect a reflective tip from its dark stem.
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    neutral = np.ptp(rgb.astype(np.int16), axis=2) < 45
    if np.mean((gray > 160) & neutral) < 0.55:
        return None
    mask = (cv2.GaussianBlur(gray, (0, 0), 1) < 145).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = [i for i in range(1, count)
                  if h * w * 0.001 < stats[i, 4] < h * w * 0.2
                  and stats[i, 0] > 2 and stats[i, 1] > 2
                  and stats[i, 0] + stats[i, 2] < w - 2
                  and stats[i, 1] + stats[i, 3] < h - 2]
    if not candidates:
        return None
    x, y, width, height, _ = stats[max(candidates, key=lambda i: stats[i, 4])]
    preview = cv2.GaussianBlur(rgb, (0, 0), 2).astype(np.float32) / 255
    return preview, (float(x), float(y), float(width), float(height))


def subject_changed(left, right):
    """Compare the union of subject bounds, tolerating small focus breathing."""
    if left is None or right is None:
        return False
    if left[0].shape != right[0].shape:
        return True
    x1, y1, w1, h1 = left[1]
    x2, y2, w2, h2 = right[1]
    h, w = left[0].shape[:2]
    tolerance = max(1, round(max(h, w) / 320))
    x, y = max(tolerance, int(min(x1, x2))), max(tolerance, int(min(y1, y2)))
    end_x = min(w - tolerance, int(max(x1 + w1, x2 + w2)))
    end_y = min(h - tolerance, int(max(y1 + h1, y2 + h2)))
    region = left[0][y:end_y, x:end_x]
    if not region.size:
        return False
    difference = min(float(np.abs(region - right[0][y + dy:end_y + dy, x + dx:end_x + dx]).mean())
                     for dy in (-tolerance, 0, tolerance)
                     for dx in (-tolerance, 0, tolerance))
    return difference > 0.09


def stack_preview(rgb):
    """Suppress focus-detail changes while retaining layout and colour."""
    return cv2.GaussianBlur(cv2.resize(rgb, (160, 120)), (0, 0), 3).astype(np.float32) / 255


def composition_change(left, right):
    visible = np.maximum(left.max(axis=2), right.max(axis=2)) > 0.15
    difference = np.abs(left - right)
    return float(difference[visible].mean() if visible.any() else difference.mean())


def _capture_time(record):
    value = getattr(record, "capture_time", None)
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None


class StackGroupDetector:
    """Split a capture sequence when composition, dimensions, or time change."""

    def __init__(self, config=None, loader=None):
        self.config = config or StackGroupDetectorConfig()
        self.loader = loader or load_rgb

    def iter_groups(self, records):
        current = []
        previous = previous_time = previous_shape = previous_subject = None
        start, group_id = 0, 0
        for index, record in enumerate(records):
            rgb = self.loader(resolve_image_path(record), 640)
            subject = subject_feature(rgb)
            # Preserve the original whole-frame score's two-stage resampling.
            ratio = min(1.0, 320 / max(rgb.shape[:2]))
            small = cv2.resize(rgb, (round(rgb.shape[1] * ratio), round(rgb.shape[0] * ratio)), interpolation=cv2.INTER_AREA) if ratio < 1 else rgb
            preview = stack_preview(small)
            moment = _capture_time(record)
            try:
                gap = (moment - previous_time).total_seconds() if moment and previous_time else 0
            except TypeError:
                gap = 0
            change = composition_change(previous, preview) if previous is not None else 0
            boundary = (
                rgb.shape != previous_shape
                or gap < 0
                or gap > max(self.config.long_pause_seconds, self.config.pause_seconds)
                or change > self.config.composition_change
                or subject_changed(previous_subject, subject)
                or (gap >= self.config.pause_seconds and change > self.config.pause_change)
            )
            if current and boundary:
                group_id += 1
                yield SceneGroup(group_id, current, start, index - 1, confidence=1.0)
                current = []
            if not current:
                start = index
            current.append(record)
            previous, previous_time, previous_shape = preview, moment, rgb.shape
            previous_subject = subject
        if current:
            yield SceneGroup(group_id + 1, current, start, start + len(current) - 1, confidence=1.0)


__all__ = ["StackGroupDetector", "StackGroupDetectorConfig", "stack_preview", "composition_change"]
