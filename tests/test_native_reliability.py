"""Targeted native/NumPy contract and resource tests for the isolated candidate."""
from __future__ import annotations

from copy import deepcopy
import logging
from collections.abc import Mapping
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("FOCUS_STACK_NATIVE_TEST_SOURCE", ROOT)).resolve()
sys.path.insert(0, str(SOURCE_ROOT))

from focus_stack_app.fusion import fast_cpp, fast_numpy, native_runtime, statistics_native  # noqa: E402


def _assert_equal(name, left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        a, b = np.asarray(left), np.asarray(right)
        assert a.shape == b.shape, f"{name}: shape mismatch {a.shape} vs {b.shape}"
        assert a.dtype == b.dtype, f"{name}: dtype mismatch {a.dtype} vs {b.dtype}"
        if np.issubdtype(a.dtype, np.floating):
            close = np.isclose(a, b, atol=1e-6, rtol=1e-6, equal_nan=True)
            if not np.all(close):
                diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
                index = tuple(int(v) for v in np.unravel_index(int(np.nanargmax(diff)), diff.shape))
                pytest.fail(
                    f"{name}: float mismatch at {index}, native={a[index]!r}, numpy={b[index]!r}, "
                    f"max_abs_diff={float(np.nanmax(diff)):.9g}, atol=1e-6, rtol=1e-6"
                )
        else:
            same = a == b
            if not np.all(same):
                where = tuple(int(v) for v in np.argwhere(~same)[0])
                pytest.fail(f"{name}: mismatch at {where}, native={a[where]!r}, numpy={b[where]!r}")
        return
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        assert type(left) is type(right), f"{name}: container types differ"
        assert len(left) == len(right), f"{name}: container lengths differ"
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_equal(f"{name}[{index}]", a, b)
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        assert isinstance(left, Mapping) and isinstance(right, Mapping), f"{name}: mapping types differ"
        assert left.keys() == right.keys(), f"{name}: mapping keys differ"
        for key in left:
            _assert_equal(f"{name}.{key}", left[key], right[key])
        return
    if isinstance(left, SimpleNamespace) or isinstance(right, SimpleNamespace):
        assert isinstance(left, SimpleNamespace) and isinstance(right, SimpleNamespace), f"{name}: object types differ"
        _assert_equal(f"{name}.__dict__", vars(left), vars(right))
        return
    assert left == right, f"{name}: {left!r} != {right!r}"


def _fast_pair(monkeypatch, name, args):
    assert fast_cpp._library.available, f"fast_core unavailable: {fast_cpp.runtime_info()}"
    function = getattr(fast_cpp, name)
    native_args = deepcopy(args)
    native_result = function(*native_args)
    fallback_args = deepcopy(args)
    with monkeypatch.context() as patch:
        patch.setattr(fast_cpp._library, "dll", None)
        fallback_result = function(*fallback_args)
    _assert_equal(f"fast_cpp.{name}.return", native_result, fallback_result)
    _assert_equal(f"fast_cpp.{name}.arguments_after_call", native_args, fallback_args)


def _statistics_pair(monkeypatch, name, args):
    assert statistics_native._library.available, (
        f"statistics_native unavailable: {statistics_native.runtime_info()}"
    )
    function = getattr(statistics_native, name)
    native_args = deepcopy(args)
    native_result = function(*native_args)
    fallback_args = deepcopy(args)
    with monkeypatch.context() as patch:
        patch.setattr(statistics_native._library, "dll", None)
        fallback_result = function(*fallback_args)
    _assert_equal(f"statistics_native.{name}.return", native_result, fallback_result)
    _assert_equal(f"statistics_native.{name}.arguments_after_call", native_args, fallback_args)


def _random_arrays(seed, shape):
    rng = np.random.default_rng(seed)
    return rng


def test_selected_native_libraries_load_from_source_and_export_openmp():
    assert Path(fast_cpp.__file__).resolve().is_relative_to(SOURCE_ROOT)
    assert Path(statistics_native.__file__).resolve().is_relative_to(SOURCE_ROOT)
    fast_info = fast_cpp.runtime_info()
    stats_info = statistics_native.runtime_info()
    assert fast_info["openmp"] is True, fast_info
    assert stats_info["openmp"] is True, stats_info
    assert fast_info["threads"] == 1, fast_info
    assert stats_info["threads"] == 1, stats_info


def test_every_public_fast_kernel_matches_numpy(monkeypatch):
    rng = _random_arrays(20261004, (12, 15))

    rgb_backing = rng.integers(0, 256, size=(12, 30, 3), dtype=np.uint8)
    rgb = rgb_backing[:, ::2, :]
    rgb.flags.writeable = False
    _fast_pair(monkeypatch, "chroma", (rgb,))

    detail = rng.random((12, 15), dtype=np.float32)
    broad = rng.random((12, 15), dtype=np.float32)
    gradient = rng.random((12, 15), dtype=np.float32)
    _fast_pair(monkeypatch, "independent", (detail, broad, gradient, 0.08))

    xx, yy, xy = (rng.random((12, 15), dtype=np.float32) for _ in range(3))
    variance = rng.random((12, 15), dtype=np.float32)
    _fast_pair(monkeypatch, "focus_fields", (xx, yy, xy, gradient, variance, detail, 0.03))

    valid = rng.integers(0, 2, (12, 15), dtype=np.uint8)
    evidence = rng.integers(0, 2, (12, 15), dtype=np.uint8)
    texture = rng.integers(0, 2, (12, 15), dtype=np.uint8).astype(bool)
    score = rng.random((12, 15), dtype=np.float32)
    labels = np.zeros((12, 15), dtype=np.uint16)
    best = np.zeros((12, 15), dtype=np.float32)
    detail_best = np.zeros((12, 15), dtype=np.float32)
    detail_owner = np.zeros((12, 15), dtype=np.uint16)
    local_best = np.zeros((12, 15), dtype=np.float32)
    local_owner = np.zeros((12, 15), dtype=np.uint16)
    textured = np.zeros((12, 15), dtype=bool)
    _fast_pair(monkeypatch, "proxy_winners", (
        valid, evidence, texture, score, detail, 2, best, labels,
        detail_best, detail_owner, local_best, local_owner, textured,
    ))

    # Both mask representations are public inputs; output remains a bool writer.
    mixed_texture = rng.integers(0, 2, (12, 15), dtype=np.uint8)
    mixed_textured = np.zeros((12, 15), dtype=bool)
    _fast_pair(monkeypatch, "proxy_winners", (
        valid, evidence, mixed_texture, score, detail, 2, best, labels,
        detail_best, detail_owner, local_best, local_owner, mixed_textured,
    ))

    propagated = rng.integers(0, 4, (12, 15), dtype=np.uint16)
    propagated_detail = rng.random((12, 15), dtype=np.float32)
    active = rng.integers(0, 2, (12, 15), dtype=np.uint8)
    _fast_pair(monkeypatch, "preserve_detail", (
        labels, propagated, propagated_detail, detail_best, detail_owner, active,
    ))

    strength_backing = rng.random((12, 30), dtype=np.float32)
    strength = strength_backing[:, ::2]
    strength.flags.writeable = False
    edge_backing = rng.integers(0, 2, (12, 30), dtype=np.uint8)
    edge = edge_backing[:, ::2]
    edge.flags.writeable = False
    _fast_pair(monkeypatch, "nearest_support", (strength, edge, 7.0))

    local_best = rng.random((12, 15), dtype=np.float32)
    positions = rng.integers(0, local_best.size, (12, 15), dtype=np.int64)
    details = rng.random((12, 15), dtype=np.float32)
    chroma = rng.integers(0, 256, (12, 15), dtype=np.uint8)
    foreground = rng.integers(0, 2, (12, 15), dtype=np.uint8)
    material = rng.integers(0, 2, (12, 15), dtype=np.uint8)
    _fast_pair(monkeypatch, "neutral_filter", (positions, details, local_best, chroma, foreground, material))

    targets = rng.integers(0, 2, (12, 15), dtype=np.uint8)
    material = rng.integers(0, 2, (12, 15), dtype=np.uint8)
    update_strength = rng.random((12, 15), dtype=np.float32)
    update_best = rng.random((12, 15), dtype=np.float32)
    owner = rng.integers(0, 4, (12, 15), dtype=np.uint16)
    chroma = rng.integers(0, 256, (12, 15), dtype=np.uint8)
    _fast_pair(monkeypatch, "neutral_update", (
        targets, material, update_strength, update_best, owner, chroma, 3,
    ))

    labels = rng.integers(0, 4, (12, 15), dtype=np.uint16)
    owner = rng.integers(0, 4, (12, 15), dtype=np.uint16)
    best = rng.random((12, 15), dtype=np.float32)
    protected = rng.integers(0, 2, (12, 15), dtype=np.uint8)
    _fast_pair(monkeypatch, "neutral_apply", (labels, owner, best, chroma, protected))

    _fast_pair(monkeypatch, "contour_energy", (
        gradient, detail, rng.integers(0, 2, (12, 15), dtype=np.uint8),
    ))


def test_proxy_winners_mixed_mask_types_include_255_and_preserve_input(monkeypatch):
    assert fast_cpp._library.available, fast_cpp.runtime_info()

    def arguments():
        shape = (8, 9)
        valid = np.ones(shape, np.uint8)
        evidence = np.ones(shape, np.uint8)
        texture = np.resize(np.array([0, 1, 255], np.uint8), shape)
        original_texture = texture.copy()
        texture.flags.writeable = False
        score = np.full(shape, 0.5, np.float32)
        detail = np.full(shape, 0.75, np.float32)
        best = np.zeros(shape, np.float32)
        labels = np.zeros(shape, np.uint16)
        detail_best = np.zeros(shape, np.float32)
        detail_owner = np.zeros(shape, np.uint16)
        local_best = np.zeros(shape, np.float32)
        local_owner = np.zeros(shape, np.uint16)
        textured = np.zeros(shape, bool)
        values = (valid, evidence, texture, score, detail, 2, best, labels,
                  detail_best, detail_owner, local_best, local_owner, textured)
        return values, original_texture

    native_args, expected_texture = arguments()
    native_result = fast_cpp.proxy_winners(*native_args)
    native_values = native_args
    assert np.array_equal(native_args[2], expected_texture), "native wrapper modified its uint8 texture input"
    numpy_args, expected_texture = arguments()
    with monkeypatch.context() as patch:
        patch.setattr(fast_cpp._library, "dll", None)
        numpy_result = fast_cpp.proxy_winners(*numpy_args)
    numpy_values = numpy_args
    assert np.array_equal(numpy_args[2], expected_texture), "NumPy fallback modified its uint8 texture input"
    _assert_equal("proxy_winners mixed uint8/bool masks", native_result, numpy_result)
    _assert_equal("proxy_winners mixed input mutations", native_values, numpy_values)


def _boundary_case():
    rng = np.random.default_rng(20261005)
    height, width = 9, 11
    tile = {
        "x": 2, "y": 1, "x1": 8, "y1": 7,
        "best": rng.random((6, 6), dtype=np.float32),
        "owner": rng.integers(0, 3, (6, 6), dtype=np.uint16),
        "reference_score": np.zeros((6, 6), np.float32),
        "mask": rng.integers(0, 2, (6, 6), dtype=np.uint8),
        "rgb": np.zeros((6, 6, 3), np.uint8),
        "reference_rgb": np.zeros((6, 6, 3), np.uint8),
        "labels": rng.integers(0, 3, (6, 6), dtype=np.uint16),
    }
    valid = rng.integers(0, 2, (height, width), dtype=np.uint8)
    score = rng.random((6, 6), dtype=np.float32)
    first_better = fast_cpp.rank(tile, valid, score, 2, 0)
    source = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    fast_cpp.capture(tile, source, first_better, 0)
    second_score = rng.random((6, 6), dtype=np.float32)
    second_better = fast_cpp.rank(tile, valid, second_score, 1, 1)
    reference_source = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    fast_cpp.capture(tile, reference_source, second_better, 1)
    covered = np.zeros((height, width), np.uint8)
    counts = fast_cpp.select(tile, covered, 0)
    output = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    distance = rng.random((height, width), dtype=np.float32) * 3
    fast_cpp.blend(tile, output, covered, distance, 2.0)
    return first_better, second_better, counts, tile, covered, output


def test_fast_boundary_kernels_rank_capture_select_blend_match_numpy(monkeypatch):
    assert fast_cpp._library.available, fast_cpp.runtime_info()
    native = _boundary_case()
    with monkeypatch.context() as patch:
        patch.setattr(fast_cpp._library, "dll", None)
        numpy_result = _boundary_case()
    _assert_equal("fast boundary native/numpy", native, numpy_result)


def _render_case(threads):
    height = width = 600
    yy, xx = np.indices((height, width))
    labels = np.asarray(xx >= width // 2, dtype=np.uint16)
    band = np.zeros((height, width), np.uint8)
    band[:, width // 2 - 2:width // 2 + 2] = 1
    rng = np.random.default_rng(20261006)
    sources = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(2)]
    valid = np.ones((height, width), np.uint8)
    feather = np.full((height, width), 0.5, np.float32)
    output = sources[0].copy()
    config = SimpleNamespace(native_threads=threads, max_hugin_workers=1, parallel_pipeline=False)
    with native_runtime.native_thread_budget(config, "fast") as budget:
        with fast_cpp.RenderAccumulator(labels, band) as accumulator:
            seam_count = accumulator.seam_count
            owner0, count0 = accumulator.owner(0)
            owner1, count1 = accumulator.owner(1)
            accumulator.add(0, sources[0], valid, feather, output)
            accumulator.add(1, sources[1], valid, feather, output)
            missing = accumulator.finish(output)
    return budget, seam_count, owner0, count0, owner1, count1, output, missing


def test_large_render_accumulator_serial_and_three_threads_match_exactly():
    assert fast_cpp._library.available, fast_cpp.runtime_info()
    one = _render_case(1)
    three = _render_case(3)
    assert one[0]["threads"] == 1
    assert three[0]["threads"] == 3, three[0]
    _assert_equal("large RenderAccumulator threads=1 vs 3", one[1:], three[1:])


def _render_fallback_case(use_numpy):
    height = width = 600
    yy, xx = np.indices((height, width))
    labels = np.asarray(xx >= width // 2, dtype=np.uint16)
    band = np.zeros((height, width), np.uint8)
    band[:, width // 2 - 1:width // 2 + 1] = 1
    sources = [
        np.full((height, width, 3), (180, 20, 40), np.uint8),
        np.full((height, width, 3), (10, 80, 220), np.uint8),
    ]
    sources[0][150:170, 150:170] = 0  # A valid true-black reference patch is still covered.
    valid = [np.ones((height, width), np.uint8) for _ in range(2)]
    valid[0][350:360, 200:210] = 0  # No owner covers this interior patch.
    feather = [np.ones((height, width), np.float32) for _ in range(2)]
    feather[0][250:280, width // 2 - 1:width // 2 + 1] = 0
    feather[1][250:280, width // 2 - 1:width // 2 + 1] = 0  # Zero total seam weight.
    output = np.full((height, width, 3), 19, np.uint8)
    config = SimpleNamespace(native_threads=3, max_hugin_workers=1, parallel_pipeline=False)

    def run():
        with native_runtime.native_thread_budget(config, "fast") as budget:
            with fast_cpp.RenderAccumulator(labels, band) as accumulator:
                seam_count = accumulator.seam_count
                owners = [accumulator.owner(index) for index in (0, 1)]
                for index in (0, 1):
                    accumulator.add(index, sources[index], valid[index], feather[index], output)
                missing = accumulator.finish(output)
        return budget, seam_count, owners, missing, output

    if use_numpy:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(fast_cpp._library, "dll", None)
            return run()
    assert fast_cpp._library.available, fast_cpp.runtime_info()
    return run()


def test_large_render_accumulator_native_numpy_zero_weight_invalid_and_true_black():
    native = _render_fallback_case(False)
    numpy_result = _render_fallback_case(True)
    _assert_equal("large RenderAccumulator native vs NumPy", native[1:], numpy_result[1:])
    output = native[-1]
    assert native[3] > 0, "fixture must exercise missing-pixel accounting"
    assert np.all(output[150:170, 150:170] == 0), "valid true-black reference pixels were lost"
    assert np.all(output[255:275, 299:301] == 19), "zero-weight seam pixels must remain uncovered"
    assert np.all(output[350:360, 200:210] == 19), "invalid interior pixels must remain uncovered"


def test_statistics_kernels_all_match_numpy(monkeypatch):
    rng = np.random.default_rng(20261007)
    h, w = 13, 17

    classes = rng.integers(0, 16, (h, w), dtype=np.uint8)
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    low = np.full((16, h, w, 3), 255, np.uint8)
    high = np.zeros((16, h, w, 3), np.uint8)
    count = np.zeros((16, h, w), np.uint16)
    _statistics_pair(monkeypatch, "probe_update", (classes, rgb, low, high, count))

    backing = rng.integers(0, 256, (h, 2 * w, 3), dtype=np.uint8)
    noncontiguous_rgb = backing[:, ::2]
    noncontiguous_rgb.flags.writeable = False
    _statistics_pair(monkeypatch, "chroma_map", (noncontiguous_rgb,))

    yy, xx = np.indices((256, 256))
    small = ((xx // 16) * 16 + 4).astype(np.uint8)
    variance = rng.random((256, 256), dtype=np.float32)
    gradient = rng.random((256, 256), dtype=np.float32)
    response = rng.random((256, 256), dtype=np.float32)
    stats = SimpleNamespace(
        samples=np.zeros(16, np.int64),
        noise_fallback_counts=np.zeros(16, np.int64),
        focus_floor=np.zeros(16, np.float32),
        variance_floor=np.zeros(16, np.float32),
        gradient_floor=np.zeros(16, np.float32),
    )
    _statistics_pair(monkeypatch, "flat_noise", (stats, small, variance, gradient, response))

    hsv = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    gray = rng.integers(0, 256, (h, w), dtype=np.uint8)
    _statistics_pair(monkeypatch, "material_classes", (hsv, gray))

    rgbf = rng.random((h, w, 3), dtype=np.float32)
    mask = rng.integers(0, 2, (h, w), dtype=np.uint8)
    _statistics_pair(monkeypatch, "masked_colour", (rgbf, mask))

    field = rng.random((h, w, 3), dtype=np.float32)
    density = rng.random((h, w), dtype=np.float32)
    _statistics_pair(monkeypatch, "normalize_colour", (field, density))

    classes = rng.integers(0, 16, (h, w), dtype=np.uint8)
    material_id = 5
    delta = rng.uniform(-8, 8, (h, w, 3)).astype(np.float32)
    density = rng.random((h, w), dtype=np.float32)
    offset = np.zeros((h, w, 3), np.float32)
    confidence = np.zeros((h, w), np.uint8)
    _statistics_pair(monkeypatch, "update_offset", (classes, material_id, delta, density, offset, confidence))

    image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    correction = rng.uniform(-7, 7, (h, w, 3)).astype(np.float32)
    valid = rng.integers(0, 2, (h, w), dtype=np.uint8)
    _statistics_pair(monkeypatch, "apply_colour", (image, correction, valid))


def _flat_noise_fixture():
    small = np.full((256, 256), 200, np.uint8)  # Bin 12 has ample observations.
    sample_bins = np.full(64 * 64, 12, np.uint8)
    sample_bins[:256] = 4
    sample_bins[256:512] = 6
    sample_bins[512:575] = 7  # 63 samples: below the 64-sample observation threshold.
    grid = small[::4, ::4]
    grid[:] = (sample_bins.reshape(64, 64) * 16 + 8).astype(np.uint8)
    yy, xx = np.indices(small.shape)
    variance = (0.1 + ((yy * 3 + xx * 7) % 101).astype(np.float32) / 100.0).astype(np.float32)
    gradient = (0.2 + ((yy * 5 + xx * 11) % 73).astype(np.float32) / 80.0).astype(np.float32)
    response = (0.3 + ((yy * 13 + xx * 17) % 89).astype(np.float32) / 90.0).astype(np.float32)
    stats = SimpleNamespace(
        samples=np.zeros(16, np.int64),
        noise_fallback_counts=np.zeros(16, np.int64),
        focus_floor=np.zeros(16, np.float32),
        variance_floor=np.zeros(16, np.float32),
        gradient_floor=np.zeros(16, np.float32),
    )
    return stats, small, variance, gradient, response


def test_flat_noise_sparse_bins_neighbor_fallback_and_probe_count_wrap(monkeypatch):
    assert statistics_native._library.available, statistics_native.runtime_info()
    native_args = _flat_noise_fixture()
    native_result = statistics_native.flat_noise(*native_args)
    fallback_args = _flat_noise_fixture()
    with monkeypatch.context() as patch:
        patch.setattr(statistics_native._library, "dll", None)
        fallback_result = statistics_native.flat_noise(*fallback_args)
    _assert_equal("flat_noise sparse native vs NumPy", (native_result, native_args[0]),
                  (fallback_result, fallback_args[0]))
    stats = native_args[0]
    assert stats.samples[4] >= 32 and stats.samples[6] >= 32 and stats.samples[12] >= 32
    assert stats.samples[7] == 0, "the sparse 63-sample bin must use fallback"
    assert stats.noise_fallback_counts[5] == 1, "two-sided adjacent fallback must be counted"
    assert stats.noise_fallback_counts[7] == 1, "sparse bin must copy its observed neighbor"
    assert stats.noise_fallback_counts[8] == 0, "fallbacks do not cascade through another fallback bin"
    for values in native_result:
        assert values[5] == max(values[4], values[6])
        assert values[7] == values[6]
        assert values[8] == 0

    classes = np.zeros((1, 1), np.uint8)
    rgb = np.array([[[90, 80, 70]]], np.uint8)
    low = np.full((16, 1, 1, 3), 255, np.uint8)
    high = np.zeros_like(low)
    count = np.zeros((16, 1, 1), np.uint16)
    count[0, 0, 0] = np.iinfo(np.uint16).max
    statistics_native.probe_update(classes, rgb, low, high, count)
    assert count[0, 0, 0] == 0, "uint16 probe count must wrap from 65535 to 0"


def test_fast_rejects_invalid_inputs_and_noncontiguous_write_buffers():
    with pytest.raises(ValueError):
        fast_cpp.chroma(np.zeros((4, 5, 3), np.float32))
    with pytest.raises(ValueError):
        fast_cpp.chroma(np.zeros((4, 5), np.uint8))
    with pytest.raises(ValueError):
        fast_cpp.focus_fields(*(np.zeros((4, 5), np.float32)[:, ::2] for _ in range(6)), 0.1)
    with pytest.raises(ValueError):
        fast_cpp.RenderAccumulator(np.zeros((4, 5), np.float32), np.zeros((4, 5), np.uint8))

    labels = np.zeros((8, 9), np.uint16)
    band = np.zeros((8, 9), np.uint8)
    accumulator = fast_cpp.RenderAccumulator(labels, band)
    backing = np.zeros((8, 9, 6), np.uint8)
    noncontiguous_output = backing[:, :, ::2]
    with pytest.raises(ValueError):
        accumulator.finish(noncontiguous_output)
    accumulator.close()


def test_statistics_rejects_invalid_shapes_dtypes_and_noncontiguous_writers():
    rgb = np.zeros((5, 7, 3), np.uint8)
    classes = np.zeros((5, 7), np.uint8)
    low = np.zeros((16, 5, 7, 3), np.uint8)
    high = np.zeros_like(low)
    count = np.zeros((16, 5, 7), np.uint16)
    with pytest.raises(ValueError):
        statistics_native.probe_update(classes.astype(np.uint16), rgb, low, high, count)
    with pytest.raises(ValueError):
        statistics_native.probe_update(classes, rgb[:, ::2], low, high, count)

    backing = np.zeros((16, 5, 14, 3), np.uint8)
    noncontiguous_low = backing[:, :, ::2]
    with pytest.raises(ValueError):
        statistics_native.probe_update(classes, rgb, noncontiguous_low, high, count)
    readonly_count = count.copy()
    readonly_count.flags.writeable = False
    with pytest.raises(ValueError):
        statistics_native.probe_update(classes, rgb, low, high, readonly_count)


class _FakeFunction:
    def __init__(self, result=None):
        self.result = result
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.result


class _FakeDLL:
    def __init__(self, functions):
        self._functions = functions

    def __getattr__(self, name):
        if name not in self._functions:
            raise AttributeError(name)
        return self._functions[name]


def test_native_library_load_failures_missing_abi_and_legacy_abi(monkeypatch, tmp_path, caplog):
    missing_path = tmp_path / "not_created" / "missing.dll"

    def fail_load(path):
        raise OSError("deliberate loader failure")

    with monkeypatch.context() as patch:
        patch.setattr(native_runtime.ctypes, "CDLL", fail_load)
        with caplog.at_level(logging.WARNING):
            missing_a = native_runtime.NativeLibrary(missing_path, {"kernel": ([], None)})
            missing_b = native_runtime.NativeLibrary(missing_path, {"kernel": ([], None)})
    assert missing_a.available is False
    assert missing_b.available is False
    assert "OSError" in missing_a.reason
    messages = [record for record in caplog.records if "missing.dll unavailable" in record.getMessage()]
    assert len(messages) == 1, [record.getMessage() for record in caplog.records]

    fake_legacy = _FakeDLL({"kernel": _FakeFunction(17)})
    with monkeypatch.context() as patch:
        patch.setattr(native_runtime.ctypes, "CDLL", lambda path: fake_legacy)
        legacy = native_runtime.NativeLibrary(
            missing_path, {"kernel": ([native_runtime.ctypes.c_int], native_runtime.ctypes.c_int)},
            abi_name="optional_abi", expected_abi=2, optional_abi=True,
        )
    assert legacy.available is True
    assert legacy.abi == "legacy"
    assert legacy.call("kernel", 3) == 17

    fake_mismatch = _FakeDLL({"kernel": _FakeFunction(), "core_abi": _FakeFunction(4)})
    with monkeypatch.context() as patch:
        patch.setattr(native_runtime.ctypes, "CDLL", lambda path: fake_mismatch)
        mismatch = native_runtime.NativeLibrary(
            missing_path, {"kernel": ([], None)}, abi_name="core_abi", expected_abi=3,
        )
    assert mismatch.available is False
    assert "ABI mismatch" in mismatch.reason
    assert mismatch.functions == {}

    fake_missing_symbol = _FakeDLL({})
    with monkeypatch.context() as patch:
        patch.setattr(native_runtime.ctypes, "CDLL", lambda path: fake_missing_symbol)
        symbol_error = native_runtime.NativeLibrary(missing_path, {"absent": ([], None)})
    assert symbol_error.available is False
    assert "AttributeError" in symbol_error.reason
    assert symbol_error.functions == {}


def test_render_accumulator_cleanup_on_exception_repeated_close_and_construct_failure(monkeypatch):
    assert fast_cpp._library.available, fast_cpp.runtime_info()
    labels = np.zeros((4, 5), np.uint16)
    band = np.zeros((4, 5), np.uint8)
    destroyed = []
    handles = iter((12345, 23456))
    monkeypatch.setattr(fast_cpp, "_create", lambda *args: next(handles))
    monkeypatch.setattr(fast_cpp, "_seams", lambda handle: 0)
    monkeypatch.setattr(fast_cpp, "_destroy", lambda handle: destroyed.append(handle))

    accumulator = fast_cpp.RenderAccumulator(labels, band)
    with pytest.raises(RuntimeError, match="deliberate render failure"):
        with accumulator:
            raise RuntimeError("deliberate render failure")
    accumulator.close()
    assert destroyed == [12345], f"render handle released {len(destroyed)} times: {destroyed}"
    with pytest.raises(RuntimeError, match="closed"):
        accumulator.owner(0)

    normal = fast_cpp.RenderAccumulator(labels, band)
    with normal as entered:
        assert entered is normal
    normal.close()
    assert destroyed == [12345, 23456], f"normal with exit release mismatch: {destroyed}"

    monkeypatch.setattr(fast_cpp, "_create", lambda *args: None)
    with pytest.raises(MemoryError, match="allocation failed"):
        fast_cpp.RenderAccumulator(labels, band)
    assert destroyed == [12345, 23456], "failed handle construction must not destroy a null handle"

    monkeypatch.setattr(fast_cpp, "_create", lambda *args: 34567)
    def fail_seam_count(handle):
        assert handle == 34567
        raise RuntimeError("deliberate seam query failure")
    monkeypatch.setattr(fast_cpp, "_seams", fail_seam_count)
    with pytest.raises(RuntimeError, match="deliberate seam query failure"):
        fast_cpp.RenderAccumulator(labels, band)
    assert destroyed == [12345, 23456, 34567], f"seam query construction failure release mismatch: {destroyed}"


def test_cancelled_fast_render_releases_native_accumulator_once(monkeypatch):
    assert fast_cpp._library.available, fast_cpp.runtime_info()
    from focus_stack_app.fusion import fast_fusion

    destroyed = []
    monkeypatch.setattr(fast_cpp, "_create", lambda *args: 45678)
    monkeypatch.setattr(fast_cpp, "_seams", lambda handle: 0)
    monkeypatch.setattr(fast_cpp, "_destroy", lambda handle: destroyed.append(handle))
    event = __import__("threading").Event()
    event.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        fast_fusion.render_fast_stream(
            [Path("unused.jpg")], [0], [np.eye(3, dtype=np.float32)], [(2, 2)],
            (2, 2), 0, (2, 2), np.zeros((2, 2), np.uint16),
            encoded_sources={}, cancel_event=event,
        )
    assert destroyed == [45678], f"cancelled render handle release count={len(destroyed)}"


@pytest.mark.parametrize("failure_point", ["decode", "tone"])
def test_fast_stream_propagates_render_exceptions_and_releases_once(monkeypatch, failure_point):
    assert fast_cpp._library.available, fast_cpp.runtime_info()
    from focus_stack_app.fusion import fast_fusion

    handle = 58001 if failure_point == "decode" else 58002
    destroyed = []
    monkeypatch.setattr(fast_cpp, "_create", lambda *args: handle)
    monkeypatch.setattr(fast_cpp, "_seams", lambda _handle: 0)
    monkeypatch.setattr(fast_cpp, "_destroy", lambda value: destroyed.append(value))
    monkeypatch.setattr(fast_cpp, "_owner", lambda _handle, _index, _mask_ptr: 12)
    expected = RuntimeError(f"deliberate {failure_point} failure")
    if failure_point == "decode":
        monkeypatch.setattr(fast_fusion, "_read_rgb_image", lambda _encoded: (_ for _ in ()).throw(expected))
        tone = None
    else:
        monkeypatch.setattr(fast_fusion, "_read_rgb_image", lambda _encoded: np.full((3, 4, 3), 90, np.uint8))

        class FailingToneModel:
            def correct(self, _aligned, _index):
                raise expected

        tone = FailingToneModel()

    kwargs = dict(
        paths=[Path("fixture.jpg")], indices=[0], transforms=[np.eye(3, dtype=np.float32)],
        analysis_shapes=[(3, 4)], reference_analysis_shape=(3, 4), reference_index=0,
        full_shape=(3, 4), proxy_labels=np.zeros((3, 4), np.uint16),
        encoded_sources={0: b"encoded fixture"}, tone_model=tone,
    )
    with pytest.raises(RuntimeError) as caught:
        fast_fusion.render_fast_stream(**kwargs)
    assert caught.value is expected, f"original {failure_point} exception was not propagated"
    assert destroyed == [handle], f"{failure_point} exception released handle {len(destroyed)} times: {destroyed}"


def _publish_fixture(tmp_path):
    import build_cpp

    build = tmp_path / "build"
    destination = tmp_path / "published"
    build.mkdir()
    destination.mkdir()
    sources = {"fast_core.dll": b"new fast library", "statistics_native.dll": b"new stats library"}
    previous = {"fast_core.dll": b"old fast library intact", "statistics_native.dll": b"old stats library intact"}
    for name, payload in sources.items():
        (build / name).write_bytes(payload)
        (destination / name).write_bytes(previous[name])
    return build_cpp, build, destination, sources, previous


def test_build_publish_staging_copy_failure_keeps_both_old_libraries(monkeypatch, tmp_path):
    build_cpp, build, destination, _sources, previous = _publish_fixture(tmp_path)
    original_copy2 = build_cpp.shutil.copy2

    def fail_second_staged_copy(source, target, *args, **kwargs):
        target = Path(target)
        if target.parent.name.startswith(".native-publish-") and target.name == "statistics_native.dll":
            raise OSError("deliberate staging copy failure")
        return original_copy2(source, target, *args, **kwargs)

    monkeypatch.setattr(build_cpp.shutil, "copy2", fail_second_staged_copy)
    with pytest.raises(OSError, match="deliberate staging copy failure"):
        build_cpp.publish(build, destination)
    assert {name: (destination / name).read_bytes() for name in previous} == previous


def test_build_publish_second_replace_failure_restores_first_old_library(monkeypatch, tmp_path):
    import os as os_module
    from pathlib import Path as PathType

    build_cpp, build, destination, _sources, previous = _publish_fixture(tmp_path)
    original_iterdir = PathType.iterdir
    original_replace = os_module.replace
    ordered = [build / "fast_core.dll", build / "statistics_native.dll"]

    def deterministic_build_iteration(path):
        if path.resolve() == build.resolve():
            return iter(ordered)
        return original_iterdir(path)

    def fail_second_replace(source, target):
        if Path(source).name == "statistics_native.dll" and Path(target) == destination / "statistics_native.dll":
            raise OSError("deliberate second replace failure")
        return original_replace(source, target)

    monkeypatch.setattr(PathType, "iterdir", deterministic_build_iteration)
    monkeypatch.setattr(build_cpp.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="deliberate second replace failure"):
        build_cpp.publish(build, destination)
    assert {name: (destination / name).read_bytes() for name in previous} == previous


def test_app_config_native_thread_default_legacy_and_round_trip(tmp_path):
    from focus_stack_app.config import AppConfig, RuntimeConfig

    assert RuntimeConfig().native_threads == 0
    assert AppConfig.from_mapping({}).runtime.native_threads == 0
    legacy = AppConfig.from_mapping({"runtime": {"fusion_backend": "quality", "max_hugin_workers": 2}})
    assert legacy.runtime.native_threads == 0
    configured = legacy.with_overrides(runtime={"native_threads": 2})
    config_path = tmp_path / "round-trip" / "stack-config.json"
    configured.save(config_path)
    loaded = AppConfig.load(config_path)
    assert loaded.runtime.native_threads == 2
    assert loaded.to_mapping() == configured.to_mapping()


def test_stack_merge_service_passes_native_threads_to_quality_and_fast_backends(tmp_path):
    from focus_stack_app.config import RuntimeConfig
    from focus_stack_app.pipeline.merge_worker import StackMergeService

    for backend_name in ("quality", "fast"):
        service = StackMergeService(
            tmp_path / backend_name,
            archive_enabled=False,
            runtime_config=RuntimeConfig(native_threads=2),
            fusion_backend=backend_name,
        )
        assert service.backend.runtime_config.native_threads == 2


def test_shipped_native_thread_auto_and_forced_budgets():
    assert "fast" in native_runtime.AUTO_PARALLEL_BACKENDS
    assert "quality" not in native_runtime.AUTO_PARALLEL_BACKENDS
    base = {"native_threads": 0, "max_hugin_workers": 1, "parallel_pipeline": False,
            "focus_analysis_workers": 0}
    with native_runtime.native_thread_budget(base, "fast") as fast:
        assert 1 <= fast["threads"] <= 3
    with native_runtime.native_thread_budget(base, "quality") as quality:
        assert quality["threads"] == 1
    forced = {**base, "native_threads": 1}
    for backend_name in ("fast", "quality"):
        with native_runtime.native_thread_budget(forced, backend_name) as budget:
            assert budget["threads"] == 1
