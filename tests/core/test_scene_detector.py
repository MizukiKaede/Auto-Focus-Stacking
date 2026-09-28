import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from focus_stack_app.core.scene_detector import SceneConfig, SceneDetector


def _scene(seed, size=180):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 30, (size, size), dtype=np.uint8)
    for _ in range(35):
        x, y = rng.integers(12, size - 12, 2)
        cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 9)), int(rng.integers(80, 255)), -1)
    return image


def test_confirmation_window_splits_after_two_changed_frames():
    first = _scene(1)
    second = _scene(2)
    frames = [first, cv2.GaussianBlur(first, (0, 0), 2),
              second, cv2.GaussianBlur(second, (0, 0), 2)]
    groups = SceneDetector(SceneConfig(
        scene_preview_long_edge=160,
        scene_confirmation_window=2,
        scene_similarity_threshold=0.80,
        geometric_inlier_threshold=0.40,
    )).detect(frames)
    assert [group.image_count for group in groups] == [2, 2]


