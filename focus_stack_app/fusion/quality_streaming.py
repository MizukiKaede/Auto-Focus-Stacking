"""Quality's whole-group reference and exact, narrow-band RGB accumulator."""
from __future__ import annotations

import cv2
import numpy as np

from .focus_masks import focus_weight
from ..utils.memory import memory_snapshot
from ..utils.performance import diagnostic, stage, timed


class MemoryReferenceFrames:
    """Retain every aligned P1 frame for a deliberately non-streaming oracle."""

    def __init__(self, loader, *, working_bytes=0):
        self.loader, self.working_bytes = loader, int(working_bytes)
        self.frames = {}
        self.bytes_used = self.peak_bytes = self.hits = 0

    def for_focus(self, index):
        if index not in self.frames:
            snapshot = memory_snapshot()
            reserve = max(2 * 1024**3, snapshot.total_bytes // 10) + self.working_bytes
            if snapshot.available_bytes < reserve:
                raise MemoryError("whole-group Quality reference exceeds available memory budget")
            image = self.loader(index)
            self.frames[index] = image
            self.bytes_used += image.nbytes
            self.peak_bytes = max(self.peak_bytes, self.bytes_used)
        return self.frames[index]

    def for_blend(self, index):
        self.hits += 1
        return self.frames[index]

    def peek(self, index):
        return self.frames[index]

    def clear(self):
        self.frames.clear()
        self.bytes_used = 0


@timed("fusion")
def blend_focus_narrow(count, load_aligned, labels, *, cancel_event=None, rows=256):
    """Copy hard owners as uint8; preserve the old float32 sum only at seams.

    focus_weight's sigma=1 float32 Gaussian has a nine-pixel support. Pixels
    whose full support belongs to one owner receive exactly its uint8 value.
    Contributions at all other pixels use the original source order and the
    original full-resolution Gaussian weights, including the image borders.
    """
    if rows < 1 or labels.ndim != 2 or labels.size == 0:
        raise ValueError("narrow fusion requires nonempty labels and positive strip height")
    if int(labels.min()) < 0 or int(labels.max()) >= count:
        raise ValueError("focus labels contain an invalid source index")
    with stage("transition_band"):
        kernel = np.ones((9, 9), np.uint8)
        band = cv2.dilate(labels, kernel) != cv2.erode(labels, kernel)
        row_counts = np.count_nonzero(band, axis=1)
        offsets = np.concatenate(([0], np.cumsum(row_counts, dtype=np.int64)))
        transition_pixels = int(offsets[-1])
        accumulator = np.zeros((transition_pixels, 3), np.float32)
        result = np.empty((*labels.shape, 3), np.uint8)
    diagnostic("quality_narrow_accumulator", transition_pixels=transition_pixels,
               image_pixels=int(labels.size), accumulator_bytes=accumulator.nbytes,
               gaussian_support=9, strip_rows=rows,
               version="quality-two-pass-v1", accumulation_order="original_selected_index")
    height = labels.shape[0]
    for index in range(count):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("focus fusion cancelled")
        rgb = load_aligned(index)
        if rgb.shape != result.shape or rgb.dtype != np.uint8:
            raise ValueError("aligned narrow-fusion frame must be matching uint8 RGB")
        with stage("focus_weight", frame=index):
            weight = focus_weight(labels, index) if transition_pixels else None
        with stage("rgb_narrow_accumulation", frame=index):
            for y in range(0, height, rows):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("focus fusion cancelled")
                end = min(height, y + rows)
                active = band[y:end]
                hard = (labels[y:end] == index) & ~active
                result[y:end][hard] = rgb[y:end][hard]
                start_offset, end_offset = offsets[y], offsets[end]
                if end_offset > start_offset:
                    contribution = rgb[y:end][active].astype(np.float32)
                    contribution *= weight[y:end][active, None]
                    accumulator[start_offset:end_offset] += contribution
        del rgb, weight
    for y in range(0, height, rows):
        end = min(height, y + rows)
        start_offset, end_offset = offsets[y], offsets[end]
        if end_offset > start_offset:
            result[y:end][band[y:end]] = np.clip(
                np.rint(accumulator[start_offset:end_offset]), 0, 255).astype(np.uint8)
    return result
