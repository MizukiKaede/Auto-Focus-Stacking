"""Stable broad surface colour, with fine detail from the focus winners."""
from __future__ import annotations

import cv2
import numpy as np

from ..utils.performance import diagnostic, timed

SURFACE_TONE_VERSION = "material-paired-tone-v17"


def _materials(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hue = ((hsv[:, :, 0].astype(np.uint16) + 15) // 30) % 6
    coloured = hsv[:, :, 1] >= 50
    classes = (1 + hue + 6 * (gray < 64)).astype(np.uint8)
    classes[~coloured] = 13 + (gray[~coloured] >= 64) + (gray[~coloured] >= 160)
    return classes


def _normalised_colour(rgb, mask, sigma):
    density = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    field = cv2.GaussianBlur(rgb.astype(np.float32) * mask[:, :, None], (0, 0), sigma)
    field /= np.maximum(density[:, :, None], 1e-6)
    return field, density


def _pure_core(mask, saturation, scale, *, coloured):
    mask = mask.copy()
    if coloured and np.any(mask):
        purity = float(np.percentile(saturation[mask], 75)) * 0.9
        mask &= saturation >= purity
    clearance = max(1, int(np.ceil(8 * scale)))
    return cv2.erode(mask.astype(np.uint8), np.ones((2 * clearance + 1,) * 2, np.uint8))


class SurfaceToneHarmonizer:
    """One compact material reference and an order-independent drift probe.

    Only broad RGB differences change. Source detail, focus decisions and
    transforms remain intact. No image names, hand-painted ROI or colour
    constants identify a particular product.
    """

    def __init__(self, reference_index, *, maximum_pixels=3_000_000):
        self.reference_index = int(reference_index)
        self.maximum_pixels = int(maximum_pixels)
        if self.maximum_pixels < 1:
            raise ValueError("surface tone statistics need a positive pixel cap")
        self.classes = self.reference_rgb = None
        self.reference_cores = {}
        self.materials = []
        self.probe_min = self.probe_max = self.probe_count = None
        self.reference_probe = self.drifting_materials = None

    @timed("surface_tone_statistics")
    def observe(self, index, rgb):
        h, w = rgb.shape[:2]
        probe_scale = min(1.0, 256.0 / max(h, w))
        probe_size = (max(1, round(w * probe_scale)), max(1, round(h * probe_scale)))
        probe = cv2.resize(rgb, probe_size, interpolation=cv2.INTER_AREA)
        probe = cv2.GaussianBlur(probe, (0, 0), 1.0)
        probe_classes = _materials(probe)
        if self.probe_min is None:
            self.probe_min = np.full((16, *probe.shape), 255, np.uint8)
            self.probe_max = np.zeros((16, *probe.shape), np.uint8)
            self.probe_count = np.zeros((16, *probe.shape[:2]), np.uint16)
        for material in np.unique(probe_classes):
            mask = probe_classes == material
            self.probe_min[material][mask] = np.minimum(self.probe_min[material][mask], probe[mask])
            self.probe_max[material][mask] = np.maximum(self.probe_max[material][mask], probe[mask])
            self.probe_count[material][mask] += 1
        if int(index) != self.reference_index:
            return
        self.reference_probe = probe_classes.copy()
        self.scale = min(0.5, (self.maximum_pixels / (h * w)) ** 0.5)
        self.size = (max(1, int(w * self.scale)), max(1, int(h * self.scale)))
        reference = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA)
        self.reference_rgb = reference
        self.classes = _materials(reference)
        saturation = cv2.cvtColor(reference, cv2.COLOR_RGB2HSV)[:, :, 1]
        for material in np.unique(self.classes):
            core = _pure_core(self.classes == material, saturation, self.scale, coloured=material <= 12)
            if np.count_nonzero(core) < 64:
                continue
            self.reference_cores[int(material)] = core
            self.materials.append(int(material))

    def _finish_probe(self):
        if self.classes is None:
            raise ValueError("surface tone reference was not observed")
        drifting = []
        for material in self.materials:
            common = (self.reference_probe == material) & (self.probe_count[material] >= 2)
            span = self.probe_max[material].astype(np.int16) - self.probe_min[material]
            magnitude = np.max(span, axis=2)
            support = int(np.count_nonzero(common & (magnitude > 3) & (magnitude <= 40)))
            if support >= max(64, round(np.count_nonzero(common) * 0.01)):
                drifting.append(material)
        self.drifting_materials = drifting
        self.probe_min = self.probe_max = self.probe_count = self.reference_probe = None
        diagnostic("surface_tone_configuration", statistics_size=list(self.size),
                   drifting_materials=drifting, reference_index=self.reference_index,
                   routing="material_before_blend",
                   version=SURFACE_TONE_VERSION)

    @timed("surface_tone_correction")
    def correct(self, rgb, index=None, *, materials=None):
        if materials not in {None, "neutral", "colour"}:
            raise ValueError("surface tone materials must be neutral, colour or None")
        if self.drifting_materials is None:
            self._finish_probe()
        active_materials = [m for m in self.drifting_materials
                            if materials is None or (m >= 13) == (materials == "neutral")]
        if not active_materials:
            return rgb
        small = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA)
        classes = _materials(small)
        saturation = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[:, :, 1]
        offset = np.zeros((*classes.shape, 3), np.float32)
        confidence = np.zeros(classes.shape, np.uint8)
        for material in active_materials:
            mask = classes == material
            if not np.any(mask):
                continue
            core = _pure_core(mask, saturation, self.scale, coloured=material <= 12)
            if np.count_nonzero(core) < 64:
                continue
            # Compare the same physical samples in both planes. Independently
            # normalised cores move with defocus and saturation; on a shaded
            # surface their different sampling positions invent a colour shift.
            common = core & self.reference_cores[material]
            residual = self.reference_rgb.astype(np.float32) - small.astype(np.float32)
            delta, density = _normalised_colour(residual, common, max(1.0, 32 * self.scale))
            magnitude = np.max(np.abs(delta), axis=2)
            # Continuous confidence avoids threshold contours in the colour
            # field. Larger mismatches can describe a displaced feature.
            valid = mask & (density > 1e-6)
            alpha = np.clip((magnitude - 1.0) / 3.0, 0, 1)
            alpha *= np.clip((40.0 - magnitude) / 20.0, 0, 1)
            alpha *= np.clip(density / 0.2, 0, 1)
            offset[valid] = delta[valid] * alpha[valid, None]
            confidence[valid] = 1
        height, width = rgb.shape[:2]
        field = cv2.resize(offset, (width, height), interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_NEAREST) != 0
        full_classes = _materials(rgb)
        kernel = np.ones((3, 3), np.uint8)
        valid &= cv2.erode(full_classes, kernel) == cv2.dilate(full_classes, kernel)
        result = rgb.copy()
        changed = 0
        for y in range(0, height, 128):
            rows = slice(y, min(height, y + 128))
            active = valid[rows]
            if not np.any(active):
                continue
            values = np.rint(np.clip(rgb[rows].astype(np.float32) + field[rows], 0, 255)).astype(np.uint8)
            changed += int(np.count_nonzero(active & np.any(values != rgb[rows], axis=2)))
            result[rows][active] = values[active]
        diagnostic("surface_tone_frame", frame=index, corrected_pixels=changed,
                   materials=materials or "all", version=SURFACE_TONE_VERSION)
        return result


__all__ = ["SurfaceToneHarmonizer", "SURFACE_TONE_VERSION"]
