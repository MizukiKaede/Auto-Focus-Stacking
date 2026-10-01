"""Narrow localized-v5 gate scenarios using synthetic RGB observations."""

import cv2
import numpy as np

from focus_stack_app.fusion.focus_masks import focus_response
from focus_stack_app.fusion.quality_fusion import FlatTextureStatistics


def _observe(stats, index, rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    stats.observe(index, rgb, gray, focus_response(rgb, gray=gray, support_radius=0))


def test_localized_slow_curved_white_rgb_noise_focus_does_not_flood():
    height = width = 256
    y, x = np.mgrid[:height, :width].astype(np.float32)
    radius = ((x - width / 2) ** 2 + (y - height / 2) ** 2) / (2 * (width / 2) ** 2)
    illumination = np.clip(238.0 + 10.0 * radius, 0, 255)
    reference = np.repeat(illumination[:, :, None], 3, axis=2).astype(np.uint8)
    rng = np.random.default_rng(123)
    competing = np.clip(
        reference.astype(np.float32) + rng.normal(0.0, 1.8, reference.shape),
        0,
        255,
    ).astype(np.uint8)

    stats = FlatTextureStatistics(reference_index=0, edge_mode="localized")
    _observe(stats, 0, reference)
    _observe(stats, 1, competing)
    _, protected = stats.apply(np.ones((height, width), np.uint16))

    assert np.count_nonzero(stats.focus_evidence) > 0
    assert np.mean(stats.edge_evidence) < 0.1
    assert np.mean(stats.texture_evidence) < 0.1
    assert np.mean(protected) < 0.1


def test_localized_fine_gray_line_protected_without_edge_flood():
    reference = np.full((256, 256, 3), 180, dtype=np.uint8)
    reference[:, 126:130] = 140
    competing = np.full_like(reference, 180)

    stats = FlatTextureStatistics(reference_index=0, edge_mode="localized")
    _observe(stats, 0, reference)
    _observe(stats, 1, competing)
    labels, protected = stats.apply(np.ones((256, 256), np.uint16))

    assert np.mean(protected[:, 120:136]) > 0.9
    assert np.mean(protected[:, :100]) < 0.1
    assert np.all(labels[:, 126:130] == 1)


def test_localized_multiframe_flat_gray_enclosed_interior_is_not_locked():
    rng = np.random.default_rng(42)
    reference = np.full((256, 256, 3), 245, dtype=np.uint8)
    reference[70:190, 70:190] = 160
    frames = [reference]
    for offset in (1, -1):
        frames.append(
            np.clip(
                reference.astype(np.int16) + offset + rng.normal(0.0, 0.5, reference.shape),
                0,
                255,
            ).astype(np.uint8)
        )

    stats = FlatTextureStatistics(reference_index=0, edge_mode="localized")
    for index, frame in enumerate(frames):
        _observe(stats, index, frame)
    labels, protected = stats.apply(np.ones((256, 256), np.uint16))

    interior = np.s_[90:170, 90:170]
    assert np.mean(protected[interior]) > 0.95
    assert np.all(labels[interior] == 1)
    assert np.mean(labels[:50, :50] == 0) > 0.95
