import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from focus_stack_app.core.registration import RegistrationConfig, register_images, resize_for_analysis, warp_image


def _pattern(size=420):
    image = np.zeros((size, size), dtype=np.uint8)
    rng = np.random.default_rng(7)
    image[:] = rng.integers(0, 28, image.shape, dtype=np.uint8)
    for _ in range(40):
        x, y = rng.integers(20, size - 20, 2)
        cv2.circle(image, (int(x), int(y)), int(rng.integers(3, 12)), int(rng.integers(80, 255)), -1)
    return image


def test_resize_preserves_aspect_and_caps_long_edge():
    resized = resize_for_analysis(np.zeros((200, 400), np.uint8), 160)
    assert max(resized.shape[:2]) == 160
    assert resized.shape[:2] == (80, 160)


def test_ecc_or_akaze_registration_recovers_small_translation():
    reference = _pattern()
    transform = np.float32([[1, 0, 4], [0, 1, -3]])
    moved = cv2.warpAffine(reference, transform, (reference.shape[1], reference.shape[0]))
    result = register_images(reference, moved, RegistrationConfig(analysis_long_edge=420, max_features=1000))
    assert result.valid
    assert result.inliers >= 6
    assert abs(result.translation[0] + 4) < 3.0 or abs(result.translation[0] - 4) < 3.0
    aligned = warp_image(moved, result, reference.shape)
    assert np.mean(np.abs(aligned[12:-12, 12:-12].astype(float) - reference[12:-12, 12:-12])) < 6
