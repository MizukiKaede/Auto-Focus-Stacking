import cv2
import numpy as np

from focus_stack_app.fusion.focus_masks import focus_response
from focus_stack_app.fusion.opencv_v20 import OpenCVBoundaryOwnership, _local_detail_statistics
from focus_stack_app.fusion.quality_fusion import SurfaceBoundaryOwnership


def test_fractional_resize_of_fully_valid_support_keeps_detail_evidence():
    scene = np.full((251, 337, 3), 235, np.uint8)
    scene[45:215, 55:285] = (45, 80, 175)
    scene[100:180:7, 65:100] = (65, 105, 200)
    score = focus_response(scene)
    supported = OpenCVBoundaryOwnership(maximum_pixels=6000)
    supported.observe(0, scene, score, valid=np.ones(scene.shape[:2], bool))
    size = (supported.best.shape[1], supported.best.shape[0])
    small = cv2.resize(scene, size, interpolation=cv2.INTER_AREA)
    response, evidence = _local_detail_statistics(small, np.ones(small.shape[:2], bool))
    expected = np.where(evidence, response, 0)
    assert np.count_nonzero(expected) > 100
    np.testing.assert_array_equal(supported.local_detail_best, expected)


def test_coloured_interior_keeps_detail_at_an_independent_focus_plane():
    scene = np.full((240, 320, 3), 230, np.uint8)
    scene[35:205, 40:280] = (185, 45, 60)
    paint_plane = scene.copy()
    detail_plane = cv2.GaussianBlur(scene, (0, 0), 5)
    # Shallow low-contrast marks remain coloured, so the neutral-only guard
    # cannot protect them. The painted outer edge is at a different depth.
    for y in range(85, 165, 12):
        detail_plane[y:y+3, 58:90] = (155, 35, 50)
    paint_score = focus_response(paint_plane)
    detail_score = focus_response(detail_plane)
    old = SurfaceBoundaryOwnership()
    new = OpenCVBoundaryOwnership()
    for guard in (old, new):
        guard.observe(0, paint_plane, paint_score)
        guard.observe(1, detail_plane, detail_score)
    labels = np.ones(scene.shape[:2], np.uint16)
    original = old.apply(labels)
    result = new.apply(labels)
    marks = np.s_[85:165, 60:88]
    assert np.mean(original[marks] == 0) > .9
    assert np.mean(result[marks] == 1) > .9
    # The separate sharp silhouette must still own the outer fringe.
    assert np.mean(result[85:165, 35:42] == 0) > .9
