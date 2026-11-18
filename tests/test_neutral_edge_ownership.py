import cv2
import numpy as np

from focus_stack_app.fusion.frame_features import FrameFeatures
from focus_stack_app.fusion.neutral_edge_ownership import NeutralEdgeOwnership


def test_fragmented_slanted_edge_uses_one_sharp_source_and_preserves_color():
    height, width = 360, 540
    rows, columns = np.indices((height, width), dtype=np.float32)
    edge_x = 110 + 0.22 * rows
    colored = np.s_[155:192, 141:169]

    def frame(offset, transition_width):
        mix = np.clip((columns - edge_x - offset) / transition_width + 0.5, 0, 1)
        gray = (226 - 168 * mix).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        rgb[colored] = (192, 138, 38)
        return rgb

    sharp = frame(0, 1.5)
    softer_offset = frame(4, 12)
    valid = np.ones((height, width), dtype=bool)
    owner = NeutralEdgeOwnership(reference_index=0)
    for index, rgb in enumerate((sharp, softer_offset)):
        owner.observe(index, rgb, valid, FrameFeatures(rgb))

    fragmented = ((rows.astype(np.uint16) // 7) % 2)
    result = fragmented.copy()
    owner.apply(result)

    representatives = ((rows >= 52) & (rows < 115) | (rows >= 233) & (rows < 308))
    transition = representatives & (np.abs(columns - edge_x) <= 4)
    assert np.mean(fragmented[transition] == 0) < 0.6
    assert np.mean(result[transition] == 0) > 0.95
    np.testing.assert_array_equal(result[colored], fragmented[colored])
    np.testing.assert_array_equal(result[:, 280:], fragmented[:, 280:])


def test_short_bright_silhouette_bevel_requires_matching_endpoint_owners():
    bright, dark = 224, 38
    gray = np.full((90, 100), bright, np.uint8)
    cv2.fillConvexPoly(gray, np.array([(24, 36), (38, 23), (82, 23), (82, 76), (24, 76)]), dark)
    guard = NeutralEdgeOwnership(reference_index=0)
    guard.silhouette_gray = gray
    untouched = np.iinfo(np.uint16).max
    before = np.full(gray.shape, untouched, np.uint16)
    start, end = (24, 36), (38, 23)
    before[start[1], start[0]] = before[end[1], end[0]] = 0
    before[32, 31] = 1
    flat_pair = ((47, 59), (60, 49))

    result = before.copy()
    assert guard._join_silhouette(result, {0: (start, end, *flat_pair)}, untouched) == 1
    assert result[29, 31] == 0
    assert result[32, 31] == 1
    assert result[54, 54] == untouched

    result = before.copy()
    result[end[1], end[0]] = 1
    assert guard._join_silhouette(result, {0: (start,), 1: (end,)}, untouched) == 0
    assert result[29, 31] == untouched
