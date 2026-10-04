"""Veto Hugin boundary owners that erase real texture at an interior target."""
from __future__ import annotations

import cv2
import numpy as np

from ..utils.performance import diagnostic


def texture_energy(gray):
    luma = gray.astype(np.float32) / 255.0
    fine = (cv2.GaussianBlur(luma, (0, 0), 0.6)
            - cv2.GaussianBlur(luma, (0, 0), 2.4))
    energy = cv2.GaussianBlur(fine * fine, (0, 0), 2.0)
    return fine, energy


class InteriorTextureDetail:
    """Keep locally resolved, nondirectional grain on its focus winner.

    Boundary ranking can borrow support from a different part of the subject.
    Only short-scale grain at the actual target may veto that substitution.
    A coherent stroke, smooth halo, or exterior background is not eligible.
    """

    def __init__(self):
        self.best = self.energy = self.textured = self.owner = None

    def observe(self, index, rgb, gray, score):
        fine, energy = texture_energy(gray)
        gx = cv2.Sobel(fine, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(fine, cv2.CV_32F, 0, 1)
        xx = cv2.GaussianBlur(gx * gx, (0, 0), 3.0)
        xy = cv2.GaussianBlur(gx * gy, (0, 0), 3.0)
        yy = cv2.GaussianBlur(gy * gy, (0, 0), 3.0)
        coherence = np.sqrt((xx - yy) ** 2 + 4.0 * xy * xy) / np.maximum(xx + yy, 1e-8)
        textured = (energy > (6.0 / 255.0) ** 2) & (coherence < 0.6)
        if self.best is None:
            self.best = score.copy()
            self.owner = np.full(score.shape, index, np.uint16)
            self.energy = energy
            self.textured = textured
        else:
            better = score > self.best
            self.owner[better] = index
            self.energy[better] = energy[better]
            self.textured[better] = textured[better]
            np.maximum(self.best, score, out=self.best)

    def apply(self, labels, raw_labels, loader, interior):
        if self.best is None:
            return labels
        size = (labels.shape[1], labels.shape[0])
        # Stay inside the filled physical subject, including its neutral print.
        inside = cv2.erode(np.uint8(interior != 0), np.ones((3, 3), np.uint8))
        inside = cv2.resize(inside, size, interpolation=cv2.INTER_NEAREST) != 0
        # Statistics belong to the unregularized focus winner. A previous
        # texture/median decision may have chosen another source; do not use
        # the winner's energy as evidence for that different source.
        eligible = ((labels != raw_labels) & (raw_labels == self.owner)
                    & inside & self.textured)
        result = labels.copy()
        restored = 0
        for index in np.unique(labels[eligible]):
            ys, xs = np.where(eligible & (labels == index))
            # All filter support is included; source pixels are never altered.
            margin = 24
            x0, x1 = max(0, int(xs.min()) - margin), min(size[0], int(xs.max()) + margin + 1)
            y0, y1 = max(0, int(ys.min()) - margin), min(size[1], int(ys.max()) + margin + 1)
            rgb = loader(int(index))
            gray = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
            _, candidate_energy = texture_energy(gray)
            veto = candidate_energy[ys - y0, xs - x0] < 0.7 * self.energy[ys, xs]
            result[ys[veto], xs[veto]] = raw_labels[ys[veto], xs[veto]]
            restored += int(veto.sum())
        diagnostic("hugin_interior_texture_detail", restored_pixels=restored,
                   eligible_pixels=int(eligible.sum()),
                   version="interior-texture-target-v1")
        return result
