"""Exact-output contracts for the three isolated C++ changes."""
import importlib.util
from pathlib import Path
import threading

import cv2
import numpy as np
import pytest

from focus_stack_app.core import coverage_native as coverage
from focus_stack_app.core.whole_frame_selection import _select_coverage
from focus_stack_app.fusion import focus_masks, quality_fusion, quality_rgb_native as rgb_native


def baseline_module(name, filename):
    source = Path(__file__).resolve().parent / 'fixtures/legacy_fusion_reference.py'
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline_quality = baseline_module('_baseline_quality', 'quality_fusion.py')
baseline_masks = baseline_module('_baseline_masks', 'focus_masks.py')


@pytest.mark.parametrize('pixels', [1, 7, 63, 64, 65, 129])
def test_packed_counts_tail_bits_and_boolean_fallback(monkeypatch, pixels):
    assert coverage._library.available
    rng = np.random.default_rng(pixels)
    scores = [rng.random((1, pixels)).astype(np.float16) for _ in range(5)]
    useful = rng.random((1, pixels)) > 0.2
    threshold = np.full((1, pixels), 0.4, np.float16)
    packed = coverage.CoverageMasks(scores, threshold, useful)
    with monkeypatch.context() as patch:
        patch.setattr(coverage._library, 'dll', None)
        boolean = coverage.CoverageMasks(scores, threshold, useful)
    assert packed.native and not boolean.native
    assert packed.masks.nbytes == 5 * ((pixels + 63) // 64) * 8
    for owner in [2, 0, 4]:
        assert packed.counts([0, 1, 2, 3, 4]) == boolean.counts([0, 1, 2, 3, 4])
        packed.add(owner)
        boolean.add(owner)
        assert packed.covered_count == boolean.covered_count
    with pytest.raises(ValueError):
        packed.counts([-1])
    with pytest.raises(ValueError):
        packed.add(5)


def test_selection_ties_failed_frames_reference_and_empty_detail(monkeypatch):
    scores = [np.zeros((1, 130), np.float16) for _ in range(3)]
    scores[0][:, :70] = scores[2][:, :70] = 1
    scores[1][:, 70:] = 1
    assert _select_coverage(scores, [0.9, 0.9, 1.0], {}, 0) == (
        [0, 1, 2], {2: 70 / 130, 1: 60 / 130}, 1.0)
    assert _select_coverage(scores, [1.0, 1.0, 1.0], {2: 'alignment rejected'}, 1) == (
        [0, 1], {0: 70 / 130, 1: 60 / 130}, 1.0)
    zeros = [np.zeros((1, 65), np.float16) for _ in range(3)]
    assert _select_coverage(zeros, [1.0, 1.0, 1.0], {}, 2) == ([0, 2], {0: 0.0}, 0.0)
    native = _select_coverage(scores, [0.9, 0.9, 1.0], {}, 0)
    with monkeypatch.context() as patch:
        patch.setattr(coverage._library, 'dll', None)
        assert _select_coverage(scores, [0.9, 0.9, 1.0], {}, 0) == native
    event = threading.Event()
    event.set()
    with pytest.raises(RuntimeError, match='cancelled'):
        _select_coverage(scores, [1, 1, 1], {}, 0, event)


def test_nearest_support_matches_frozen_baseline_with_strided_inputs():
    rng = np.random.default_rng(14)
    strength = rng.random((79, 166), dtype=np.float32)[:, ::2]
    for kind in ['empty', 'full', 'sparse']:
        edge = np.zeros(strength.shape, np.uint8)
        if kind == 'full':
            edge[:] = 1
        elif kind == 'sparse':
            edge[::9, ::11] = 1
        for radius in [1, 32, 128]:
            expected = baseline_quality._nearest_edge_support(strength, edge, radius)
            actual = quality_fusion._nearest_edge_support(strength, edge, radius)
            np.testing.assert_array_equal(actual, expected, strict=True)


def test_native_rgb_float32_accumulation_and_half_even_rounding():
    assert rgb_native.available()
    rng = np.random.default_rng(15)
    actual = np.empty((65, 67, 3), np.float32)
    expected = None
    for index in range(5):
        rgb = rng.integers(0, 256, (65, 134, 3), np.uint8)[:, ::2]
        weight = rng.random((65, 67), dtype=np.float32)
        contribution = rgb.astype(np.float32) * weight[..., None]
        expected = contribution if index == 0 else expected + contribution
        rgb_native.add(rgb, weight, actual, first=index == 0)
        np.testing.assert_array_equal(actual, expected, strict=True)
    np.testing.assert_array_equal(rgb_native.finish(actual),
                                  np.clip(np.rint(expected), 0, 255).astype(np.uint8), strict=True)
    halves = np.repeat(np.array([-1, 0.5, 1.5, 2.5, 253.5, 254.5, 255.5, 300], np.float32)[None, :, None], 3, axis=2)
    np.testing.assert_array_equal(rgb_native.finish(halves),
                                  np.clip(np.rint(halves), 0, 255).astype(np.uint8), strict=True)


@pytest.mark.parametrize('prefetch', [False, True])
def test_quality_blend_matches_frozen_baseline_and_missing_dll(monkeypatch, prefetch):
    rng = np.random.default_rng(16)
    frames = [rng.integers(0, 256, (65, 67, 3), np.uint8) for _ in range(4)]
    frames[0][:4] = 0
    frames[1][-4:] = 255
    labels = rng.integers(0, 3, (65, 67), np.uint16)
    expected = baseline_masks.blend_focus_pyramid(4, frames.__getitem__, labels, prefetch=prefetch)
    before = rgb_native._calls['add']
    actual = focus_masks.blend_focus_pyramid(4, frames.__getitem__, labels, prefetch=prefetch)
    assert rgb_native._calls['add'] - before == 4
    np.testing.assert_array_equal(actual, expected, strict=True)
    with monkeypatch.context() as patch:
        patch.setattr(rgb_native._library, 'dll', None)
        fallback = focus_masks.blend_focus_pyramid(4, frames.__getitem__, labels, prefetch=prefetch)
    np.testing.assert_array_equal(fallback, expected, strict=True)
    # Explicit multiband callers retain their existing numerical path.
    actual_multiband = focus_masks.blend_focus_pyramid(4, frames.__getitem__, labels, levels=2, prefetch=prefetch)
    expected_multiband = baseline_masks.blend_focus_pyramid(4, frames.__getitem__, labels, levels=2, prefetch=prefetch)
    np.testing.assert_array_equal(actual_multiband, expected_multiband, strict=True)


def test_non_uint8_library_callers_keep_the_original_blend_path():
    rng = np.random.default_rng(17)
    frames = [rng.integers(0, 256, (65, 67, 3), np.uint8) for _ in range(3)]
    frames[1] = frames[1].astype(np.float32) + np.float32(0.25)
    labels = rng.integers(0, 3, (65, 67), np.uint16)
    expected = baseline_masks.blend_focus_pyramid(3, frames.__getitem__, labels, prefetch=False)
    actual = focus_masks.blend_focus_pyramid(3, frames.__getitem__, labels, prefetch=False)
    np.testing.assert_array_equal(actual, expected, strict=True)
