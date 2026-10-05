"""Local sharp rim ownership confined to the exterior of Hugin subjects."""
from __future__ import annotations

import cv2
import numpy as np

from ..utils.performance import diagnostic


class ExteriorRimOwnership:
    """Rank real luminance edges independently of the coloured body.

    Filled foreground contours exclude printing and internal metallic texture.
    Each rim position keeps its own focus source; no whole-object frame wins.
    """

    def __init__(self, maximum_pixels=3_000_000, support_radius=48):
        self.maximum_pixels = maximum_pixels
        self.support_radius = support_radius
        self.best = self.confidence = self.owner = self.interior = self.detail = self.owner_detail = None

    def observe(self, index, rgb):
        from ..fusion.focus_masks import _filled_chromatic_silhouette, focus_response
        from ..fusion.fast_cpp import nearest_support as _nearest_edge_support

        height, width = rgb.shape[:2]
        self.scale = min(0.5, (self.maximum_pixels / (height * width)) ** 0.5)
        self.size = (max(1, round(width * self.scale)), max(1, round(height * self.scale)))
        small = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        chroma = small.max(axis=2) - small.min(axis=2)
        foreground = np.uint8((chroma >= 65) | (gray < 145))
        count, parts, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
        keep = np.zeros(count, np.uint8)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(64, round(gray.size * 0.001))
        interior = _filled_chromatic_silhouette(keep[parts])
        edge = cv2.morphologyEx(interior, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        radius = max(2, round(self.support_radius * self.scale))
        near = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)) != 0
        luma = gray.astype(np.float32) / 255.0
        smooth = cv2.GaussianBlur(luma, (0, 0), 0.8)
        gradient = cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
        broad = cv2.GaussianBlur(luma, (0, 0), 4.0)
        broad_gradient = cv2.magnitude(cv2.Sobel(broad, cv2.CV_32F, 1, 0), cv2.Sobel(broad, cv2.CV_32F, 0, 1))
        ridge = gradient >= cv2.dilate(gradient, np.ones((3, 3), np.uint8)) * 0.98
        # A neutral lip may lie outside the coloured contour. Only a localized
        # physical gradient can seed it; broad shadows cannot choose a source.
        seeds = near & ridge & (gradient > 0.04) & (gradient > 1.5 * broad_gradient)
        seeds &= (interior == 0) | (edge != 0)
        focus = focus_response(small, support_radius=0)
        strength = _nearest_edge_support(gradient * gradient * np.sqrt(focus), np.uint8(seeds), radius)
        confidence = _nearest_edge_support(gradient, np.uint8(seeds), radius)
        local_detail = cv2.dilate(focus, np.ones((5, 5), np.uint8))
        if self.best is None:
            self.best = strength
            self.confidence = confidence
            self.owner = np.full(gray.shape, index, np.uint16)
            self.interior = interior
            self.detail = local_detail
            self.owner_detail = local_detail.copy()
        else:
            better = strength > self.best
            self.best[better] = strength[better]
            self.confidence[better] = confidence[better]
            self.owner[better] = index
            self.owner_detail[better] = local_detail[better]
            self.interior |= interior
            np.maximum(self.detail, local_detail, out=self.detail)

    def apply(self, labels):
        if self.best is None:
            return labels
        # The union of filled contours is an absolute veto. This excludes
        # internal gold grain and white ink even in a blurred reference frame.
        external = self.interior == 0
        detail_veto = (self.detail > 0.001) & (self.owner_detail < 0.8 * self.detail)
        # Focus changes the source ranking, never the contrast units of the
        # coverage gate. Pale genuine rims still have valid gradient evidence.
        active = external & (self.confidence > 0.04) & (self.best > 0) & ~detail_veto
        size = (labels.shape[1], labels.shape[0])
        active = cv2.resize(np.uint8(active), size, interpolation=cv2.INTER_NEAREST) != 0
        owner = cv2.resize(self.owner, size, interpolation=cv2.INTER_NEAREST)
        result = labels.copy()
        changed = active & (result != owner)
        result[active] = owner[active]
        diagnostic("hugin_exterior_rim_ownership", changed_pixels=int(changed.sum()),
                   guarded_pixels=int(active.sum()), version="external-local-rim-v1")
        return result
