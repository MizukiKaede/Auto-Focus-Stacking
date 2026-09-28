import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from focus_stack_app.core.quality import QualityConfig, score_image


def test_sharp_frame_scores_above_blurred_frame():
    cv2 = pytest.importorskip("cv2")
    sharp = np.zeros((220, 220), np.uint8)
    sharp[25:-25, 25:-25] = 200
    for x in range(35, 200, 15):
        cv2.line(sharp, (x, 25), (x, 195), 30, 2)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 7)
    sharp_quality = score_image(sharp, config=QualityConfig(analysis_long_edge=220))
    blurred_quality = score_image(blurred, config=QualityConfig(analysis_long_edge=220))
    assert sharp_quality.sharpness > blurred_quality.sharpness
    assert sharp_quality.score > blurred_quality.score


