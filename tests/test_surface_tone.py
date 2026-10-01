"""Small behavioral checks for the opt-in surface tone candidate."""

from __future__ import annotations

import unittest

import numpy as np

from focus_stack_app.fusion.surface_tone import SurfaceToneHarmonizer


def _red_surface(base=(180, 60, 60), *, marker=(245, 245, 245)):
    image = np.empty((128, 160, 3), dtype=np.uint8)
    image[...] = np.asarray(base, dtype=np.uint8)
    image[42:60, 70:86] = np.asarray(marker, dtype=np.uint8)
    return image


def _red_checker_surface(base=(180, 60, 60)):
    """A single red material with an internal, high-frequency checker texture."""
    image = np.empty((128, 160, 3), dtype=np.uint8)
    image[...] = np.asarray(base, dtype=np.uint8)
    yy, xx = np.indices((80, 96))
    checker = ((yy // 4 + xx // 4) % 2) == 1
    light = np.asarray(base, dtype=np.int16) + np.array((18, 18, 18), dtype=np.int16)
    patch = np.where(checker[..., None], light, np.asarray(base, dtype=np.int16))
    image[16:96, 16:112] = np.clip(patch, 0, 255).astype(np.uint8)
    return image


def _neutral_scratched_surface(base=140):
    """A neutral plane with fine grain and several narrow, curved scratches."""
    height = width = 192
    yy, xx = np.indices((height, width))
    grain = np.random.default_rng(17).integers(-2, 3, size=(height, width), dtype=np.int16)
    plane = np.full((height, width), base, dtype=np.int16) + grain
    scratches = (
        (np.abs(yy - (28 + 0.22 * xx + 2 * np.sin(xx / 13))) < 0.75)
        | (np.abs(yy - (119 - 0.16 * xx + 1.3 * np.sin(xx / 17))) < 0.9)
        | (np.abs(yy - (71 + 0.035 * xx + 1.5 * np.sin(xx / 21))) < 0.65)
    )
    plane[scratches] -= 16
    return np.repeat(np.clip(plane, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)


def _high_frequency_rms(rgb):
    gray = rgb[:, :, 0].astype(np.float32)
    center = gray[1:-1, 1:-1]
    neighbors = (gray[:-2, 1:-1] + gray[2:, 1:-1]
                 + gray[1:-1, :-2] + gray[1:-1, 2:]) / 4.0
    return float(np.sqrt(np.mean((center - neighbors) ** 2)))


def _observe(harmonizer, frames):
    for index, frame in frames:
        harmonizer.observe(index, frame)


class SurfaceToneHarmonizerTests(unittest.TestCase):
    def test_identical_frames_preserve_rgb_exactly(self):
        reference = _red_surface()
        harmonizer = SurfaceToneHarmonizer(0)
        _observe(harmonizer, [(0, reference), (1, reference.copy())])

        corrected = harmonizer.correct(reference.copy(), index=1)

        self.assertTrue(np.array_equal(corrected, reference))

    def test_material_drift_reduces_broad_offset_and_keeps_detail(self):
        reference = _red_checker_surface()
        drifted = _red_checker_surface((168, 52, 52))
        harmonizer = SurfaceToneHarmonizer(0)
        _observe(harmonizer, [(0, reference), (1, drifted)])

        corrected = harmonizer.correct(drifted.copy(), index=1)

        region = (slice(16, 96), slice(16, 112))
        reference_mean = reference[region].mean(axis=(0, 1))
        drifted_mean = drifted[region].mean(axis=(0, 1))
        corrected_mean = corrected[region].mean(axis=(0, 1))
        self.assertLess(np.abs(corrected_mean - reference_mean).mean(),
                        np.abs(drifted_mean - reference_mean).mean())

        yy, xx = np.indices((80, 96))
        light = ((yy // 4 + xx // 4) % 2) == 1
        dark = ~light
        reference_contrast = reference[region][light].mean(axis=0) - reference[region][dark].mean(axis=0)
        drifted_contrast = drifted[region][light].mean(axis=0) - drifted[region][dark].mean(axis=0)
        corrected_contrast = corrected[region][light].mean(axis=0) - corrected[region][dark].mean(axis=0)
        self.assertLessEqual(float(np.max(np.abs(corrected_contrast - drifted_contrast))), 1.0)
        self.assertLessEqual(float(np.max(np.abs(corrected_contrast - reference_contrast))), 1.0)

    def test_uniform_drift_preserves_medium_scale_print_and_gradient(self):
        height = width = 192
        yy, xx = np.indices((height, width))
        gradient = np.rint((yy - (height - 1) / 2) * 12 / (height - 1))
        print_bands = np.where((xx // 16) % 2 == 0, 10, -10)
        variation = gradient + print_bands
        reference = np.clip(
            np.array((50, 100, 200), dtype=np.int16)[None, None, :]
            + variation[:, :, None], 0, 255,
        ).astype(np.uint8)
        drifted = np.clip(reference.astype(np.int16) - 12, 0, 255).astype(np.uint8)

        harmonizer = SurfaceToneHarmonizer(0)
        _observe(harmonizer, [(0, reference), (1, drifted)])
        corrected = harmonizer.correct(drifted.copy(), index=1, materials="colour")
        self.assertTrue(harmonizer.drifting_materials)

        region = (slice(32, 160), slice(32, 160))
        light = ((xx[region] // 16) % 2) == 0
        raw_contrast = float(drifted[region][light, 0].mean()
                             - drifted[region][~light, 0].mean())
        corrected_contrast = float(corrected[region][light, 0].mean()
                                   - corrected[region][~light, 0].mean())

        self.assertGreater(raw_contrast, 15.0)
        self.assertGreaterEqual(corrected_contrast, raw_contrast * 0.8)

    def test_coloured_neutral_boundary_does_not_cross_materials(self):
        reference = np.empty((128, 160, 3), dtype=np.uint8)
        reference[:, :80] = (180, 60, 60)
        reference[:, 80:] = (180, 180, 180)
        drifted = reference.copy()
        drifted[:, :80] = (168, 52, 52)
        harmonizer = SurfaceToneHarmonizer(0)
        _observe(harmonizer, [(0, reference), (1, drifted)])

        corrected = harmonizer.correct(drifted.copy(), index=1)

        self.assertTrue(np.array_equal(corrected[:, 112], drifted[:, 112]))
        self.assertTrue(np.array_equal(corrected[:, 80], drifted[:, 80]))
        self.assertTrue(np.array_equal(corrected[:, 79], drifted[:, 79]))

    def test_neutral_owner_mosaic_removes_seams_and_keeps_scratches(self):
        reference = _neutral_scratched_surface()
        drifted = np.clip(reference.astype(np.int16) + 11, 0, 255).astype(np.uint8)
        harmonizer = SurfaceToneHarmonizer(0)
        _observe(harmonizer, [(0, reference), (1, drifted)])

        corrected_reference = harmonizer.correct(
            reference.copy(), index=0, materials="neutral")
        corrected_drifted = harmonizer.correct(
            drifted.copy(), index=1, materials="neutral")

        yy, xx = np.indices(reference.shape[:2])
        owner = ((yy // 24 + xx // 24) % 2) == 1
        raw_mosaic = np.where(owner[:, :, None], drifted, reference)
        corrected_mosaic = np.where(
            owner[:, :, None], corrected_drifted, corrected_reference)

        owner_edges = owner[:, 1:] != owner[:, :-1]
        reference_edges = (reference[:, 1:, 0].astype(np.int16)
                           - reference[:, :-1, 0].astype(np.int16))
        raw_edges = (raw_mosaic[:, 1:, 0].astype(np.int16)
                     - raw_mosaic[:, :-1, 0].astype(np.int16))
        corrected_edges = (corrected_mosaic[:, 1:, 0].astype(np.int16)
                           - corrected_mosaic[:, :-1, 0].astype(np.int16))
        raw_seam_error = float(np.abs(raw_edges - reference_edges)[owner_edges].mean())
        corrected_seam_error = float(
            np.abs(corrected_edges - reference_edges)[owner_edges].mean())
        self.assertGreater(raw_seam_error, 9.0)
        self.assertLess(corrected_seam_error, 1.0)

        interior = (slice(12, -12), slice(12, -12))
        raw_error = float(np.abs(
            raw_mosaic[interior].astype(np.int16)
            - reference[interior].astype(np.int16)).mean())
        corrected_error = float(np.abs(
            corrected_mosaic[interior].astype(np.int16)
            - reference[interior].astype(np.int16)).mean())
        self.assertLess(corrected_error, raw_error * 0.2)

        reference_detail = _high_frequency_rms(reference)
        corrected_detail = _high_frequency_rms(corrected_mosaic)
        self.assertGreaterEqual(corrected_detail, reference_detail * 0.9)
        self.assertLessEqual(corrected_detail, reference_detail * 1.1)

    def test_neutral_and_colour_stages_only_change_their_materials(self):
        reference = np.empty((128, 160, 3), dtype=np.uint8)
        reference[:, :80] = (180, 60, 60)
        reference[:, 80:] = (140, 140, 140)
        drifted = reference.copy()
        drifted[:, :80] = (169, 49, 49)
        drifted[:, 80:] = (151, 151, 151)
        harmonizer = SurfaceToneHarmonizer(0)
        _observe(harmonizer, [(0, reference), (1, drifted)])

        neutral_stage = harmonizer.correct(
            drifted.copy(), index=1, materials="neutral")
        colour_stage = harmonizer.correct(
            drifted.copy(), index=1, materials="colour")

        self.assertTrue(np.array_equal(neutral_stage[:, :80], drifted[:, :80]))
        self.assertTrue(np.array_equal(colour_stage[:, 80:], drifted[:, 80:]))
        self.assertGreater(np.count_nonzero(
            neutral_stage[:, 88:152] != drifted[:, 88:152]), 0)
        self.assertGreater(np.count_nonzero(
            colour_stage[8:120, 8:72] != drifted[8:120, 8:72]), 0)

    def test_natural_and_reference_first_observation_order_match(self):
        reference = _red_surface()
        drifted = _red_surface((168, 52, 52))

        reference_first = SurfaceToneHarmonizer(0)
        _observe(reference_first, [(0, reference), (1, drifted)])
        first_result = reference_first.correct(drifted.copy(), index=1)

        natural_order = SurfaceToneHarmonizer(0)
        _observe(natural_order, [(1, drifted), (0, reference)])
        natural_result = natural_order.correct(drifted.copy(), index=1)

        self.assertEqual(reference_first.drifting_materials, natural_order.drifting_materials)
        self.assertTrue(np.array_equal(first_result, natural_result))

    def test_large_material_difference_is_not_pulled_to_reference(self):
        reference = _red_surface((180, 100, 100))
        displaced = _red_surface((120, 40, 40))
        harmonizer = SurfaceToneHarmonizer(0)
        _observe(harmonizer, [(0, reference), (1, displaced)])

        corrected = harmonizer.correct(displaced.copy(), index=1)

        self.assertTrue(np.array_equal(corrected, displaced))

    def test_shaded_colour_keeps_its_gradient_when_purity_support_moves(self):
        yy, xx = np.indices((192, 384))
        t = xx / 383.0
        reference = np.stack((20 + 60*t, 120 + 90*t, 135 + 90*t), axis=2)
        reference += (((xx + yy) % 3) - 1)[:, :, None]
        reference = np.rint(reference).astype(np.uint8)
        drifted = (reference.astype(np.int16) - 10).astype(np.uint8)
        harmonizer = SurfaceToneHarmonizer(0)
        _observe(harmonizer, [(0, reference), (1, drifted)])
        corrected = harmonizer.correct(drifted, index=1, materials="colour")
        # Assert colour and shading where paired samples provide support.
        # The separate sparse-support test checks the intentional fade.
        region = np.s_[32:160, 40:220]
        error = corrected[region].astype(np.int16) - reference[region].astype(np.int16)
        self.assertLess(float(np.abs(error).mean()), 1.0)
        # The shadow and highlight remain distinct rather than flattening the
        # surface to a global colour or adding a second illumination slope.
        gradient = float(corrected[96, 210, 1]) - float(corrected[96, 50, 1])
        expected_gradient = float(reference[96, 210, 1]) - float(reference[96, 50, 1])
        self.assertGreater(gradient, 30)
        self.assertLess(abs(gradient - expected_gradient), 2)
        self.assertLess(int(np.max(np.abs(np.diff(error[64, :, 1])))), 3)

    def test_sparse_colour_support_fades_without_an_on_off_contour(self):
        reference = np.full((192, 384, 3), (190, 35, 45), np.uint8)
        reference[32:160, 145:245] = (185, 95, 100)
        reference[80:112, 290:320] = (240, 240, 240)
        drifted = reference.copy()
        coloured = np.max(reference, axis=2) - np.min(reference, axis=2) > 50
        drifted[coloured] = (reference[coloured].astype(np.int16) - 10).astype(np.uint8)
        harmonizer = SurfaceToneHarmonizer(0)
        _observe(harmonizer, [(0, reference), (1, drifted)])
        corrected = harmonizer.correct(drifted, index=1, materials="colour")
        added = corrected[96, 110:270, 0].astype(np.int16) - drifted[96, 110:270, 0]
        self.assertGreaterEqual(int(added.max()), 9)
        self.assertLessEqual(int(added.min()), 5)
        self.assertLessEqual(int(np.max(np.abs(np.diff(added)))), 2)
        self.assertTrue(np.array_equal(corrected[80:112, 290:320], drifted[80:112, 290:320]))


if __name__ == "__main__":
    unittest.main()
