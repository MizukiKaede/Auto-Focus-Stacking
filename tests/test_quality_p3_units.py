"""Small exactness tests for the P3 Quality memory/streaming candidates.

These tests use synthetic uint8 RGB frames only.  They do not run the real
fusion pipeline or decode large photographs.
"""
from __future__ import annotations

import math
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import pytest

from focus_stack_app.fusion.focus_masks import (
    _StableBackground,
    blend_focus_pyramid,
    build_focus_labels,
)
from focus_stack_app.fusion.quality_streaming import blend_focus_narrow
import focus_stack_app.utils.image_io as image_io


def _frames(shape: tuple[int, int], count: int = 3) -> list[np.ndarray]:
    height, width = shape
    base = np.arange(height * width * 3, dtype=np.uint32).reshape(height, width, 3)
    return [((base + 29 * index) % 256).astype(np.uint8) for index in range(count)]


def _exact(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(actual, expected)
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    max_abs = int(delta.max()) if delta.size else 0
    mae = float(delta.mean()) if delta.size else 0.0
    psnr = math.inf if max_abs == 0 else 20.0 * math.log10(255.0 / math.sqrt(float(np.mean(delta.astype(np.float64) ** 2))))
    assert max_abs == 0
    assert mae == 0.0
    assert math.isinf(psnr)


def _labels_for(kind: str, shape: tuple[int, int], count: int = 3) -> np.ndarray:
    height, width = shape
    if kind == "hard":
        return np.zeros((height, width), dtype=np.uint16)
    if kind == "dense":
        yy, xx = np.indices((height, width))
        return ((yy * 2 + xx) % count).astype(np.uint16)
    if kind == "boundary":
        labels = np.zeros((height, width), dtype=np.uint16)
        labels[:, width // 2:] = 1
        return labels
    if kind == "corners":
        yy, xx = np.indices((height, width))
        return ((yy >= height // 2).astype(np.uint16) * 2 + (xx >= width // 2)).astype(np.uint16) % count
    raise AssertionError(kind)


@pytest.mark.parametrize(
    ("kind", "shape", "rows"),
    [
        ("hard", (1, 1), 1),
        ("hard", (2, 3), 1),
        ("boundary", (5, 7), 2),
        ("corners", (9, 11), 3),
        ("dense", (23, 29), 4),
    ],
)
def test_narrow_blend_is_raw_rgb_exact_to_levels1(kind, shape, rows):
    frames = _frames(shape)
    labels = _labels_for(kind, shape)
    loader = lambda index: frames[index]
    pyramid = blend_focus_pyramid(len(frames), loader, labels, levels=1)
    narrow = blend_focus_narrow(len(frames), loader, labels, rows=rows)
    _exact(narrow, pyramid)


def test_narrow_blend_cancel_stops_before_next_frame():
    frames = _frames((8, 9))
    labels = _labels_for("boundary", (8, 9))
    cancel = threading.Event()
    loaded: list[int] = []

    def loader(index):
        loaded.append(index)
        cancel.set()
        return frames[index]

    with pytest.raises(RuntimeError, match="cancelled"):
        blend_focus_narrow(3, loader, labels, cancel_event=cancel, rows=1)
    assert loaded == [0]


def test_focus_labels_reference_first_preserves_nonzero_tie_owner_and_flat_background():
    frames = [np.full((64, 64, 3), 120, dtype=np.uint8) for _ in range(3)]
    natural = build_focus_labels(
        3, lambda index: frames[index], prefetch=False,
    )
    reference_first = build_focus_labels(
        3, lambda index: frames[index], frame_order=[2, 0, 1], prefetch=False,
    )
    _exact(natural, reference_first)
    assert np.unique(reference_first).tolist() == [0]


def test_stable_background_tone_tie_uses_smallest_source_index():
    rgb = np.full((64, 64, 3), 120, dtype=np.uint8)
    gray = np.full((64, 64), 120, dtype=np.uint8)
    score = np.zeros((64, 64), dtype=np.float32)
    labels = np.full((64, 64), 2, dtype=np.uint16)
    background = _StableBackground()
    for index in (2, 0, 1):
        background.observe(rgb, gray, score, index=index)
    background.apply(labels)
    assert np.unique(labels).tolist() == [0]


def test_quality_jpeg_semaphore_limits_read_only_and_allows_decode_overlap(tmp_path, monkeypatch):
    paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    for path in paths:
        path.write_bytes(b"placeholder")

    lock = threading.Lock()
    read_active = decode_active = 0
    max_read = max_decode = 0
    barrier_broken = []
    decode_barrier = threading.Barrier(2)

    def fake_read_bytes(path: Path):
        nonlocal read_active, max_read
        with lock:
            read_active += 1
            max_read = max(max_read, read_active)
        time.sleep(0.03)
        with lock:
            read_active -= 1
        return b"placeholder"

    def fake_imdecode(encoded, flags):
        nonlocal decode_active, max_decode
        assert flags == cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        with lock:
            decode_active += 1
            max_decode = max(max_decode, decode_active)
        try:
            try:
                decode_barrier.wait(timeout=2.0)
            except threading.BrokenBarrierError:
                barrier_broken.append(True)
            time.sleep(0.02)
            return np.zeros((2, 3, 3), dtype=np.uint8)
        finally:
            with lock:
                decode_active -= 1

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    monkeypatch.setattr(cv2, "imdecode", fake_imdecode)
    monkeypatch.setattr(cv2, "cvtColor", lambda image, code: image)
    results = [None, None]
    errors = []

    def worker(slot):
        try:
            results[slot] = image_io.load_quality_rgb(paths[slot])
        except Exception as exc:  # pragma: no cover - assertion reports the cause
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=4.0)

    assert not errors
    assert all(result is not None for result in results)
    assert max_read == 1
    assert max_decode >= 2
    assert not barrier_broken
