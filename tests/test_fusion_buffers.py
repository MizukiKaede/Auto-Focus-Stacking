"""Contracts for scoped feature sharing and lossless tone-buffer reuse."""
import cv2
import numpy as np
import pytest

from focus_stack_app.fusion.frame_features import FrameFeatures
from focus_stack_app.fusion.statistics_native import masked_colour
from focus_stack_app.fusion.surface_tone import SurfaceToneHarmonizer, _normalised_colour
from focus_stack_app.fusion import statistics_native


def test_frame_features_reuse_exact_resize_order_without_mutating_source(monkeypatch):
    rgb = np.random.default_rng(38).integers(0, 256, (193, 257, 3), np.uint8)
    source = rgb.copy()
    expected = cv2.resize(rgb, (79, 61), interpolation=cv2.INTER_AREA)
    calls = []
    resize = cv2.resize
    def counted(*args, **kwargs):
        calls.append(args[1])
        return resize(*args, **kwargs)
    monkeypatch.setattr(cv2, 'resize', counted)
    features = FrameFeatures(rgb)
    small = features.rgb((79, 61))
    np.testing.assert_array_equal(small, expected, strict=True)
    assert features.rgb((79, 61)) is small
    np.testing.assert_array_equal(features.gray((79, 61)), cv2.cvtColor(expected, cv2.COLOR_RGB2GRAY))
    np.testing.assert_array_equal(features.chroma((79, 61)), expected.max(2)-expected.min(2))
    assert features.gray((79, 61)) is features.gray((79, 61))
    assert calls == [(79, 61)]
    np.testing.assert_array_equal(rgb, source)


@pytest.mark.parametrize('native', [True, False])
def test_masked_colour_reuses_validated_destination(monkeypatch, native):
    if not native:
        monkeypatch.setattr(statistics_native._library, 'dll', None)
    rng = np.random.default_rng(39)
    rgb = rng.normal(size=(31, 43, 3)).astype(np.float32)
    mask = rng.integers(0, 2, (31, 43), np.uint8)
    out = np.empty_like(rgb)
    assert masked_colour(rgb, mask, out=out) is out
    np.testing.assert_array_equal(out, rgb*mask[..., None], strict=True)
    with pytest.raises(ValueError):
        masked_colour(rgb, mask, out=np.empty(rgb.shape, np.float64))


def test_normalised_colour_workspace_retains_exact_fields_across_materials():
    rng = np.random.default_rng(40)
    rgb = rng.normal(size=(41, 57, 3)).astype(np.float32)
    workspace = (np.empty(rgb.shape[:2], np.float32), np.empty_like(rgb),
                 np.empty(rgb.shape[:2], np.float32), np.empty_like(rgb))
    for fraction in [0.0, 0.2, 0.8, 1.0]:
        mask = np.uint8(rng.random(rgb.shape[:2]) < fraction)
        field, density = _normalised_colour(rgb, mask, 4.0)
        reused_field, reused_density = _normalised_colour(rgb, mask, 4.0, workspace=workspace)
        assert reused_field is workspace[3] and reused_density is workspace[2]
        np.testing.assert_array_equal(reused_field, field, strict=True)
        np.testing.assert_array_equal(reused_density, density, strict=True)


@pytest.mark.parametrize('top', [0, 254, 674])
def test_striped_tone_preserves_gaussian_support_at_real_and_cropped_edges(monkeypatch, top):
    import focus_stack_app.fusion.surface_tone as tone

    rng = np.random.default_rng(48)
    reference = np.full((733, 443, 3), (172, 173, 174), np.uint8)
    reference[top:top + 59] = (164, 54, 48)
    reference += rng.integers(0, 8, reference.shape, np.uint8)
    shifted = reference - rng.integers(4, 16, reference.shape, np.uint8)
    model = SurfaceToneHarmonizer(0, maximum_pixels=65_000)
    model.observe(0, reference)
    model.observe(1, shifted)
    expected = model.correct(shifted, index=1)
    assert np.count_nonzero(expected[top:top + 59] != shifted[top:top + 59])

    processed_rows = []
    normalise = tone._normalised_colour

    def record_rows(rgb, mask, sigma, *, workspace):
        processed_rows.append(rgb.shape[0])
        return normalise(rgb, mask, sigma, workspace=workspace)

    monkeypatch.setattr(tone, '_normalised_colour', record_rows)
    result = model.correct(shifted, index=1, strip_rows=256)
    np.testing.assert_array_equal(result, expected, strict=True)
    assert 0 < min(processed_rows) < model.size[1]


def test_reference_shortcut_checks_pixels_and_finishes_drift_probe(monkeypatch):
    reference = np.full((128, 192, 3), (180, 60, 60), np.uint8)
    shifted = np.full_like(reference, (169, 49, 49))
    model = SurfaceToneHarmonizer(0)
    model.observe(0, reference)
    model.observe(1, shifted)
    model._finish_probe()
    assert model.drifting_materials
    import focus_stack_app.fusion.surface_tone as tone
    normalise = tone._normalised_colour
    def unexpected(*args, **kwargs):
        raise AssertionError('reference must not run material blurs')
    with monkeypatch.context() as patch:
        patch.setattr(tone, '_normalised_colour', unexpected)
        np.testing.assert_array_equal(model.correct(reference, index=0), reference)
    # A caller passing different pixels with the reference index still gets correction.
    corrected = model.correct(shifted, index=0)
    assert np.count_nonzero(corrected != shifted)
    assert tone._normalised_colour is normalise
