import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from focus_stack_app.core.focus_map import (
    FocusMapConfig,
    compress_focus_map,
    compute_focus_map,
    decompress_focus_map,
)


def _textured_image(size=256):
    image = np.zeros((size, size), dtype=np.uint8)
    image[32:-32, 32:-32] = 35
    for x in range(24, size - 24, 18):
        image[:, x:x + 4] = 220
    for y in range(20, size - 20, 23):
        image[y:y + 4, :] = 170
    return image


def test_focus_map_is_compact_and_bounded():
    image = _textured_image()
    config = FocusMapConfig(analysis_long_edge=160, output_long_edge=48, output_dtype="float16")
    result = compute_focus_map(image, config)
    assert result.shape == (48, 48)
    assert result.dtype == np.float16
    values = decompress_focus_map(result)
    assert float(values.min()) >= 0.0
    assert float(values.max()) <= 1.0


def test_uint16_round_trip_preserves_map():
    source = np.linspace(0.0, 1.0, 80 * 40, dtype=np.float32).reshape(80, 40)
    result = compress_focus_map(source, output_long_edge=40, dtype="uint16")
    restored = decompress_focus_map(result)
    assert result.dtype == np.uint16
    assert np.max(np.abs(restored - source[::2, ::2])) < 0.03

