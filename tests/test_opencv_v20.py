"""Only V20's source-footprint defect and private routing contracts."""

import cv2
import numpy as np

from focus_stack_app.fusion.opencv_v20 import ValidSourceOwnership


def test_reflected_border_cannot_win_or_be_reintroduced_by_a_later_guard():
    shape = (80, 120)
    correction = ValidSourceOwnership()
    correction.register(0, np.array([[.8, 0, 12], [0, .8, 8], [0, 0, 1]]), shape)
    correction.register(1, np.eye(3), shape)
    reflected_score = np.full(shape, 100, np.float32)
    real_score = np.ones(shape, np.float32)
    correction.observe(0, reflected_score)
    correction.observe(1, real_score)
    valid = correction.valid_mask(0, shape)
    assert not np.any(reflected_score[~valid])
    assert np.all(reflected_score[valid] == 100)
    # Simulate later silhouette/ink propagation reassigning invalid sources.
    labels = np.zeros(shape, np.uint16)
    result = correction.apply(labels)
    assert np.all(result[~valid] == 1)
    assert np.all(result[valid] == 0)


def test_identity_stack_is_unchanged_and_ties_are_capture_order_stable():
    shape = (32, 48)
    outputs = []
    for order in ((8, 2), (2, 8)):
        guard = ValidSourceOwnership()
        for index in order:
            guard.register(index, np.eye(3), shape)
            score = np.ones(shape, np.float32)
            guard.observe(index, score)
            np.testing.assert_array_equal(score, np.ones(shape, np.float32))
        assert np.all(guard.owner == 2)
        labels = np.full(shape, 8, np.uint16)
        outputs.append(guard.apply(labels))
        np.testing.assert_array_equal(outputs[-1], labels)
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_transformed_padding_is_not_a_real_photograph():
    shape = (80, 120)
    source = np.full((*shape, 3), 220, np.uint8)
    source[55:, 55:85] = (190, 30, 40)
    matrix = np.array([[.8, 0, 12], [0, .8, 8], [0, 0, 1]], np.float32)
    reflected = cv2.warpPerspective(source, matrix, (shape[1], shape[0]),
                                    flags=cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_REFLECT_101)
    guard = ValidSourceOwnership()
    guard.register(0, matrix, shape)
    guard.register(1, np.eye(3), shape)
    reflected_coloured = (reflected.max(axis=2) - reflected.min(axis=2)) > 60
    invented = reflected_coloured & ~guard.valid_mask(0, shape)
    assert np.count_nonzero(invented) > 0
    guard.observe(0, np.full(shape, 10, np.float32))
    guard.observe(1, np.ones(shape, np.float32))
    result = guard.apply(np.zeros(shape, np.uint16))
    assert np.all(result[invented] == 1)


def test_low_chroma_defocus_tail_is_not_protected_as_independent_detail():
    from focus_stack_app.fusion.opencv_v20 import OpenCVBoundaryOwnership
    from focus_stack_app.fusion.quality_fusion import SurfaceBoundaryOwnership
    rng = np.random.default_rng(123)
    sharp = np.full((240, 320, 3), (210, 216, 226), np.uint8)
    sharp[40:200, 100:280] = (195, 30, 45)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 18)
    # Real sensor/JPEG grain accompanies both exposures. The low-chroma
    # defocus tail is a smooth signal comparable to the local noise ceiling.
    frames = [np.clip(x.astype(np.float32) + rng.normal(0, 2.0, x.shape), 0, 255).astype(np.uint8)
              for x in (blurred, sharp)]
    old = SurfaceBoundaryOwnership()
    new = OpenCVBoundaryOwnership()
    scores = [np.full(sharp.shape[:2], 3e-4, np.float32),
              np.full(sharp.shape[:2], 1e-4, np.float32)]
    scores[1][35:205, 95:106] = .02
    for guard in (old, new):
        for i in range(2):
            guard.observe(i, frames[i], scores[i])
    labels = np.zeros(sharp.shape[:2], np.uint16)
    old_result, result = old.apply(labels), new.apply(labels)
    tail = np.s_[85:155, 66:76]
    assert np.max(frames[0][tail].max(2)-frames[0][tail].min(2)) < 45
    assert np.all(old_result[tail] == 0)
    assert np.mean(result[tail] == 1) > .95
    np.testing.assert_array_equal(result[85:155, 130:240], old_result[85:155, 130:240])
    assert np.all(result[:, :20] == 0)


def test_cyan_tinted_gray_rim_keeps_its_independent_sharp_plane():
    from focus_stack_app.fusion.opencv_v20 import OpenCVBoundaryOwnership
    scene = np.full((240, 320, 3), 225, np.uint8)
    scene[120:220, 30:290] = (35, 205, 215)
    scene[80:120, 60:260] = (145, 195, 205)
    scene[95:112, 80:240:12] = (105, 153, 163)
    blurred = cv2.GaussianBlur(scene, (0, 0), 5)
    # The gray cap is a separate depth plane. Blur that layer without
    # importing an out-of-focus copy of the independently sharp cyan body.
    gray_layer = scene.copy()
    gray_layer[120:] = 225
    paint_plane = cv2.GaussianBlur(gray_layer, (0, 0), 5)
    paint_plane[120:] = scene[120:]
    metal_plane = scene.copy()
    metal_plane[120:] = blurred[120:]
    paint_score = np.full(scene.shape[:2], 1e-4, np.float32)
    paint_score[116:130] = .03
    metal_score = np.full(scene.shape[:2], 1e-4, np.float32)
    metal_score[76:114, 56:264] = .01
    guard = OpenCVBoundaryOwnership()
    guard.observe(0, paint_plane, paint_score)
    guard.observe(1, metal_plane, metal_score)
    labels = np.where(metal_score > paint_score, 1, 0).astype(np.uint16)
    result = guard.apply(labels)
    rim = np.s_[82:88, 80:240]
    proposed = cv2.resize(guard.owner, (320, 240), interpolation=cv2.INTER_NEAREST)
    # Exercise the actual V19 failure: its proposed owner must be the wrong
    # paint plane here, or merely retaining a metal proposal proves nothing.
    assert np.mean(proposed[rim] == 0) > .95
    # V19's neutral<45 statistics excluded this actual gray material.
    assert np.all(scene[rim].max(2)-scene[rim].min(2) > 45)
    assert np.mean(result[rim] == 1) > .95
    assert np.mean(result[130:145, 80:240] == 0) > .8


def test_invalid_reflected_texture_cannot_supply_local_detail_evidence():
    from focus_stack_app.fusion.opencv_v20 import _local_detail_statistics
    small = np.full((128, 256, 3), 180, np.uint8)
    small[32:96, 52:58] = 145
    yy, xx = np.indices((128, 128))
    small[:, 128:] = (170 + ((xx+yy) % 2) * 20)[..., None]
    valid = np.zeros(small.shape[:2], bool)
    valid[:, :120] = True
    _, evidence = _local_detail_statistics(small, valid)
    assert not np.any(evidence[~valid])
    assert np.any(evidence[40:88, 50:60])

