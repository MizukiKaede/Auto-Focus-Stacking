"""Stable broad surface colour, with fine detail from the focus winners."""
from __future__ import annotations

import cv2
import numpy as np
from .statistics_native import (probe_update, material_classes, masked_colour,
                                normalize_colour, update_offset, apply_colour)

from ..utils.performance import diagnostic, timed

SURFACE_TONE_VERSION = "material-paired-tone-v17"


def _materials(rgb, *, hsv=None):
    if hsv is None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return material_classes(hsv, gray)


def _normalised_colour(rgb, mask, sigma, *, workspace=None):
    if workspace is None:
        density = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
        field = cv2.GaussianBlur(masked_colour(rgb, mask), (0, 0), sigma)
    else:
        mask_float, masked, density, field = workspace
        np.copyto(mask_float, mask, casting='unsafe')
        masked_colour(rgb, mask, out=masked)
        cv2.GaussianBlur(mask_float, (0, 0), sigma, dst=density)
        cv2.GaussianBlur(masked, (0, 0), sigma, dst=field)
    normalize_colour(field, density)
    return field, density


def _pure_core(mask, saturation, scale, *, coloured, copy=True):
    if copy:
        mask = mask.copy()
    if coloured and np.any(mask):
        purity = float(np.percentile(saturation[mask], 75)) * 0.9
        mask &= saturation >= purity
    clearance = max(1, int(np.ceil(8 * scale)))
    mask_u8 = mask.view(np.uint8) if (mask.dtype == np.bool_ and mask.flags.c_contiguous) else mask.astype(np.uint8)
    return cv2.erode(mask_u8, np.ones((2 * clearance + 1,) * 2, np.uint8))


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
    def observe(self, index, rgb, *, features=None):
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
        probe_update(probe_classes, probe, self.probe_min, self.probe_max, self.probe_count)
        if int(index) != self.reference_index:
            return
        self.reference_probe = probe_classes.copy()
        self.scale = min(0.5, (self.maximum_pixels / (h * w)) ** 0.5)
        self.size = (max(1, int(w * self.scale)), max(1, int(h * self.scale)))
        reference = (cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA)
                     if features is None else features.rgb(self.size))
        self.reference_rgb = reference
        hsv = cv2.cvtColor(reference, cv2.COLOR_RGB2HSV)
        self.classes = _materials(reference, hsv=hsv)
        saturation = hsv[:, :, 1]
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

    def _offset_fields(self, rgb, index, active_materials, *, bound_rows=False):
        # Compact working arrays leave scope before full-resolution expansion.
        small = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA)
        if index == self.reference_index and np.array_equal(small, self.reference_rgb):
            return None, None
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        classes = _materials(small, hsv=hsv)
        saturation = hsv[:, :, 1]
        class_counts = np.bincount(classes.ravel(), minlength=16)
        offset = np.zeros((*classes.shape, 3), np.float32)
        confidence = np.zeros(classes.shape, np.uint8)
        residual = None
        workspace = None
        sigma = max(1.0, 32 * self.scale)
        # The automatic float32 Gaussian reaches at most ceil(4*sigma) pixels.
        # The extra zero pixel keeps reflection at a cropped edge out of support.
        radius = int(np.ceil(4.0 * sigma)) + 1
        height, width = classes.shape
        full_area = height * width
        for material in active_materials:
            if class_counts[material] < 64:
                continue
            mask = classes == material
            ref_core = self.reference_cores[material]
            if not np.any(mask & (ref_core != 0)):
                continue
            my_r = np.flatnonzero(np.any(mask, axis=1))
            mx_c = np.flatnonzero(np.any(mask, axis=0))
            core = _pure_core(mask, saturation, self.scale, coloured=material <= 12, copy=False)
            if np.count_nonzero(core) < 64:
                continue
            # Compare the same physical samples in both planes. Independently
            # normalised cores move with defocus and saturation; on a shaded
            # surface their different sampling positions invent a colour shift.
            common = core & ref_core
            if not np.any(common):
                continue
            if residual is None:
                residual = np.subtract(self.reference_rgb, small, dtype=np.float32)
            cy_r = np.flatnonzero(np.any(common, axis=1))
            cx_c = np.flatnonzero(np.any(common, axis=0))
            oy0 = max(int(my_r[0]), int(cy_r[0]) - radius)
            oy1 = min(int(my_r[-1]) + 1, int(cy_r[-1]) + radius + 1)
            ox0 = max(int(mx_c[0]), int(cx_c[0]) - radius)
            ox1 = min(int(mx_c[-1]) + 1, int(cx_c[-1]) + radius + 1)
            if oy0 >= oy1 or ox0 >= ox1:
                continue
            iy0 = max(0, oy0 - radius)
            iy1 = min(height, oy1 + radius)
            ix0 = max(0, ox0 - radius)
            ix1 = min(width, ox1 + radius)
            if (iy1 - iy0) * (ix1 - ix0) < full_area * 0.85:
                sub_res = np.ascontiguousarray(residual[iy0:iy1, ix0:ix1])
                sub_com = np.ascontiguousarray(common[iy0:iy1, ix0:ix1])
                delta, density = _normalised_colour(sub_res, sub_com, sigma, workspace=None)
                dy0, dy1 = oy0 - iy0, oy1 - iy0
                dx0, dx1 = ox0 - ix0, ox1 - ix0
                sub_cls = np.ascontiguousarray(classes[oy0:oy1, ox0:ox1])
                sub_delta = np.ascontiguousarray(delta[dy0:dy1, dx0:dx1])
                sub_dens = np.ascontiguousarray(density[dy0:dy1, dx0:dx1])
                sub_off = np.ascontiguousarray(offset[oy0:oy1, ox0:ox1])
                sub_conf = np.ascontiguousarray(confidence[oy0:oy1, ox0:ox1])
                update_offset(sub_cls, material, sub_delta, sub_dens, sub_off, sub_conf)
                offset[oy0:oy1, ox0:ox1] = sub_off
                confidence[oy0:oy1, ox0:ox1] = sub_conf
            else:
                if workspace is None:
                    workspace = (np.empty(classes.shape, np.float32), np.empty_like(residual),
                                 np.empty(classes.shape, np.float32), np.empty_like(residual))
                rows = slice(max(0, int(cy_r[0]) - radius), min(height, int(cy_r[-1]) + 1 + radius)) if bound_rows else slice(None)
                delta, density = _normalised_colour(
                    residual[rows], common[rows], sigma,
                    workspace=tuple(buffer[rows] for buffer in workspace),
                )
                update_offset(classes[rows], material, delta, density, offset[rows], confidence[rows])
        if not np.any(offset):
            return None, None
        return offset, confidence

    @timed("surface_tone_correction")
    def correct(self, rgb, index=None, *, materials=None, active_rows=None, strip_rows=None):
        if materials not in {None, "neutral", "colour"}:
            raise ValueError("surface tone materials must be neutral, colour or None")
        if self.drifting_materials is None:
            self._finish_probe()
        active_materials = [m for m in self.drifting_materials
                            if materials is None or (m >= 13) == (materials == "neutral")]
        if not active_materials:
            return rgb
        if active_rows is not None and not np.any(active_rows):
            diagnostic("surface_tone_frame", frame=index, corrected_pixels=0,
                       materials=materials or "all", version=SURFACE_TONE_VERSION)
            return rgb
        offset, confidence = self._offset_fields(
            rgb, index, active_materials, bound_rows=strip_rows is not None)
        if offset is None:
            diagnostic("surface_tone_frame", frame=index, corrected_pixels=0,
                       materials=materials or "all", version=SURFACE_TONE_VERSION)
            return rgb
        height, width = rgb.shape[:2]
        field = cv2.resize(offset, (width, height), interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_NEAREST) != 0
        kernel = np.ones((3, 3), np.uint8)
        if strip_rows is None:
            full_classes = _materials(rgb)
            valid &= cv2.erode(full_classes, kernel) == cv2.dilate(full_classes, kernel)
            del full_classes
            result, changed = apply_colour(rgb, field, valid)
        else:
            result = rgb.copy()
            changed = 0
            for y in range(0, height, strip_rows):
                end = min(height, y + strip_rows)
                active = valid[y:end]
                if not np.any(active):
                    continue
                # One extra row on either side preserves the 3x3 morphology.
                top, bottom = max(0, y - 1), min(height, end + 1)
                classes = _materials(rgb[top:bottom])
                interior = cv2.erode(classes, kernel) == cv2.dilate(classes, kernel)
                del classes
                active &= interior[y - top:end - top]
                _, count = apply_colour(rgb[y:end], field[y:end], active, out=result[y:end])
                changed += count
        diagnostic("surface_tone_frame", frame=index, corrected_pixels=changed,
                   materials=materials or "all", version=SURFACE_TONE_VERSION)
        return result


__all__ = ["SurfaceToneHarmonizer", "SURFACE_TONE_VERSION"]
