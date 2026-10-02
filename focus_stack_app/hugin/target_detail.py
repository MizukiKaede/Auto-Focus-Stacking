"""Retain a neutral exterior's original focus when Hugin regularizers regress it."""
import cv2
import numpy as np
from ..utils.performance import diagnostic


class NeutralTargetDetail:
    def __init__(self):
        self.best = self.neutral = self.coloured_interior = None
        self.local_detail = None

    def observe(self, index, rgb, gray, score):
        from ..fusion.focus_masks import _filled_chromatic_silhouette
        chroma = rgb.max(axis=2) - rgb.min(axis=2)
        neutral = cv2.erode(np.uint8((chroma < 45) & (gray > 30) & (gray < 235)),
                            np.ones((9, 9), np.uint8)) != 0
        # The focus score has a 7px max support on top of a sigma-4 energy
        # average. It can be strong on an empty neutral target merely because
        # a defocused coloured silhouette or a shadow is nearby. Such scores
        # must not undo exterior ownership or a textureless-background lock.
        # Require the raw winner to contain short-scale detail at the target
        # itself. Two 8-bit levels RMS rejects sensor grain and broad slopes;
        # the 2px extension retains the neighbourhood of real fine scratches.
        luma = gray.astype(np.float32) / 255.0
        fine = cv2.GaussianBlur(luma, (0, 0), 0.6)
        broad = cv2.GaussianBlur(luma, (0, 0), 2.4)
        energy = cv2.GaussianBlur((fine - broad) ** 2, (0, 0), 0.8)
        local_detail = cv2.dilate(energy, np.ones((5, 5), np.uint8)) > (2.0 / 255.0) ** 2
        # Large coloured silhouettes exclude internal white ink and gold grain.
        scale = min(1.0, 1024 / max(gray.shape))
        size = (max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale)))
        coloured = cv2.resize(np.uint8(chroma >= 65), size, interpolation=cv2.INTER_NEAREST)
        count, parts, stats, _ = cv2.connectedComponentsWithStats(coloured, 8)
        keep = np.zeros(count, np.uint8)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(64, round(coloured.size * 0.001))
        interior = cv2.resize(_filled_chromatic_silhouette(keep[parts]),
                              (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST) != 0
        if self.best is None:
            self.best = score.copy(); self.neutral = neutral
            self.coloured_interior = interior
            self.local_detail = local_detail
        else:
            better = score > self.best
            self.neutral[better] = neutral[better]
            self.local_detail[better] = local_detail[better]
            np.maximum(self.best, score, out=self.best)
            self.coloured_interior |= interior

    def apply(self, labels, raw_labels, loader):
        from ..fusion.focus_masks import focus_response
        eligible = (labels != raw_labels) & self.neutral & ~self.coloured_interior & (self.best > 0.0003)
        active = eligible & self.local_detail
        result = labels.copy()
        restored = 0
        for index in np.unique(labels[active]):
            selected = active & (labels == index)
            rgb = loader(int(index))
            focus = focus_response(rgb, support_radius=7)
            veto = selected & (focus < 0.7 * self.best)
            restored += int(veto.sum())
            result[veto] = raw_labels[veto]
        diagnostic("hugin_neutral_target_detail", restored_pixels=restored,
                   rejected_borrowed_detail_pixels=int((eligible & ~self.local_detail).sum()),
                   version="neutral-exterior-target-v2-local-evidence")
        return result
