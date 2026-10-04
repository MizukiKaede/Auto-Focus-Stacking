"""Compact, paired-material colour statistics for FastFusion.

Keep the tested V17 same-coordinate residual and boundary safeguards. Only
statistics are reduced; correction remains additive before winner copying, so
we never smooth the selected high-frequency RGB texture.
"""
from __future__ import annotations

import cv2
import numpy as np

from .surface_tone import SurfaceToneHarmonizer, _pure_core
from ..utils.performance import diagnostic


class FastSurfaceTone(SurfaceToneHarmonizer):
    def __init__(self, reference_index, full_shape):
        super().__init__(reference_index, maximum_pixels=300_000)
        self.full_shape = tuple(full_shape)

    def observe_proxy(self, index, rgb):
        super().observe(index, rgb)
        if int(index) != self.reference_index:
            return
        # Parent.observe normally receives full-sized RGB. Here its image is a
        # proxy; express erosion/smoothing radii in output pixels explicitly.
        self.scale = min(self.size[0] / self.full_shape[1], self.size[1] / self.full_shape[0])
        saturation = cv2.cvtColor(self.reference_rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
        self.reference_cores = {}
        self.materials = []
        for material in np.unique(self.classes):
            core = _pure_core(self.classes == material, saturation, self.scale,
                              coloured=material <= 12)
            if np.count_nonzero(core) >= 64:
                self.reference_cores[int(material)] = core
                self.materials.append(int(material))
        diagnostic("fast_surface_tone_reference", statistics_size=list(self.size),
                   output_scale=self.scale, version="fast-paired-material-v3")
