import cv2
import numpy as np

from focus_stack_app.fusion.focus_masks import focus_response
from focus_stack_app.fusion.quality_fusion import FlatTextureStatistics, clean_tiny_labels


def _gray_rgb(value, shape=(128, 128)):
    return np.full((*shape, 3), value, dtype=np.uint8)


def _observe(stats, index, rgb, score=None):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if score is None:
        score = np.zeros(gray.shape, np.float32)
    stats.observe(index, rgb, gray, score)


def test_gate_locks_240_to_255_white_noise_to_reference():
    reference = _gray_rgb(245)
    competing = _gray_rgb(255)
    rng = np.random.default_rng(7)
    noise = rng.uniform(0.0, 1e-4, reference.shape[:2]).astype(np.float32)
    stats = FlatTextureStatistics(reference_index=0)
    _observe(stats, 0, reference)
    _observe(stats, 1, competing, noise)

    labels, protected = stats.apply(np.ones(reference.shape[:2], np.uint16))

    assert np.all(labels == 0)
    assert not np.any(protected)


def test_gate_v2_real_rgb_white_noise_does_not_protect_whole_image():
    rng = np.random.default_rng(31)
    reference = _gray_rgb(245)
    competing = np.clip(
        245.0 + rng.normal(0.0, 2.0, reference.shape), 0, 255,
    ).astype(np.uint8)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    competing_gray = cv2.cvtColor(competing, cv2.COLOR_RGB2GRAY)
    stats = FlatTextureStatistics(reference_index=0)
    _observe(stats, 0, reference, focus_response(reference, gray=reference_gray, support_radius=0))
    _observe(stats, 1, competing, focus_response(competing, gray=competing_gray, support_radius=0))

    _, protected = stats.apply(np.ones(reference.shape[:2], np.uint16))

    assert np.count_nonzero(protected) < protected.size * 0.5


def test_gate_v4_slow_white_slope_noise_keeps_fine_line_without_edge_flood():
    height = width = 256
    slope = np.linspace(235, 250, width, dtype=np.float32)[None, :]
    reference = np.repeat(np.repeat(slope, height, axis=0)[:, :, None], 3, axis=2)
    reference = reference.astype(np.uint8)
    rng = np.random.default_rng(31)
    competing = np.clip(
        reference.astype(np.int16) + rng.normal(0.0, 2.0, reference.shape),
        0, 255,
    ).astype(np.uint8)
    # A real narrow line is present only in the competing frame.  Scores are
    # measured from the actual RGB data so the gate must retain its edge band.
    competing[:, 124:132] = 0

    stats = FlatTextureStatistics(reference_index=0)
    for index, rgb in enumerate((reference, competing)):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _observe(stats, index, rgb, focus_response(rgb, gray=gray, support_radius=0))

    labels, protected = stats.apply(np.ones((height, width), np.uint16))

    # The slow paper-like slope and its sensor noise must not turn most of the
    # frame into an edge safety band, while the actual line remains protected.
    assert np.mean(protected[:, :110]) < 0.5
    assert np.mean(protected) < 0.5
    assert np.mean(protected[:, 124:132]) > 0.8
    assert np.mean(labels[:, :110] == 0) > 0.5
    assert np.all(labels[:, 124:132] == 1)


def test_gate_protects_weak_gray_edge_without_blurring_rgb():
    reference = _gray_rgb(180)
    reference[:, 63:65] = 140
    other = _gray_rgb(180)
    reference_before = reference.copy()
    other_before = other.copy()
    stats = FlatTextureStatistics(reference_index=0)
    _observe(stats, 0, reference)
    _observe(stats, 1, other)

    labels, protected = stats.apply(np.ones(reference.shape[:2], np.uint16))

    assert np.any(protected[:, 55:73])
    assert np.all(labels[:, 55:73] == 1)
    assert labels[10, 10] == 0
    np.testing.assert_array_equal(reference, reference_before)
    np.testing.assert_array_equal(other, other_before)


def test_flat_texture_statistics_caps_pixels_and_keeps_fixed_map_shapes():
    rng = np.random.default_rng(11)
    stats = FlatTextureStatistics(reference_index=0, maximum_pixels=4096)
    observed_shapes = []
    for index in range(3):
        rgb = rng.integers(0, 256, (256, 320, 3), dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        score = rng.random(gray.shape, dtype=np.float32)
        stats.observe(index, rgb, gray, score)
        observed_shapes.append(stats.max_response.shape)

    assert stats.size[0] * stats.size[1] <= 4096
    assert observed_shapes == [observed_shapes[0]] * 3
    expected = (stats.size[1], stats.size[0])
    assert stats.max_response.shape == expected
    assert stats.max_variance.shape == expected
    assert stats.max_gradient.shape == expected
    assert stats.max_chroma.shape == expected
    assert stats.samples.shape == (16,)


def test_gate_does_not_strong_lock_unknown_brightness_bin():
    reference = _gray_rgb(20)
    reference[20:28, 20:28] = 200
    other = _gray_rgb(20)
    stats = FlatTextureStatistics(reference_index=0)
    _observe(stats, 0, reference)
    _observe(stats, 1, other)
    labels, _ = stats.apply(np.ones(reference.shape[:2], np.uint16))

    assert labels[22, 22] == 1
    assert labels[100, 100] == 0


def test_clean_tiny_labels_removes_island_but_preserves_protected_edge():
    labels = np.zeros((32, 32), np.uint16)
    labels[5, 5] = 1
    labels[16, 16] = 1
    protected = np.zeros(labels.shape, bool)
    protected[16, 16] = True

    cleaned = clean_tiny_labels(labels, protected, maximum_area=8)

    assert cleaned[5, 5] == 0
    assert cleaned[16, 16] == 1
