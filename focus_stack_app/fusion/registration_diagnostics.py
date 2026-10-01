"""Measure full-size warp residuals without changing preview registration."""
from __future__ import annotations

import cv2
import numpy as np

from ..utils.performance import diagnostic, stage


class RegistrationResiduals:
    def __init__(self, reference, *, roi_size=256, maximum_rois=6):
        h, w = reference.shape[:2]
        side = min(roi_size, h, w)
        candidates = []
        for y in np.linspace(0, max(0, h - side), 5, dtype=int):
            for x in np.linspace(0, max(0, w - side), 5, dtype=int):
                patch = cv2.cvtColor(reference[y:y + side, x:x + side], cv2.COLOR_RGB2GRAY).astype(np.float32)
                texture = float(cv2.Laplacian(patch, cv2.CV_32F).var())
                if texture > 1.0:
                    candidates.append((texture, int(x), int(y), patch))
        self.patches = sorted(candidates, key=lambda item: item[0], reverse=True)[:maximum_rois]
        self.side = side
        self.window = cv2.createHanningWindow((side, side), cv2.CV_32F) if side > 1 else None
        diagnostic("registration_residual_rois", rois=[dict(x=x, y=y, size=side, texture=t)
                                                      for t, x, y, _ in self.patches],
                   method="full_resolution_phase_correlation_diagnostic_only")

    def measure(self, aligned, frame, matrix, source_shape):
        results = []
        if self.window is None:
            return
        inverse = np.linalg.inv(np.asarray(matrix, np.float64))
        sh, sw = source_shape
        with stage("registration_residual", frame=frame):
            for texture, x, y, reference in self.patches:
                corners = np.array([[[x, y], [x + self.side - 1, y],
                                     [x, y + self.side - 1], [x + self.side - 1, y + self.side - 1]]], np.float32)
                original = cv2.perspectiveTransform(corners, inverse)[0]
                if np.any(original < 0) or np.any(original[:, 0] >= sw) or np.any(original[:, 1] >= sh):
                    continue
                patch = cv2.cvtColor(aligned[y:y + self.side, x:x + self.side], cv2.COLOR_RGB2GRAY).astype(np.float32)
                shift, confidence = cv2.phaseCorrelate(reference.copy(), patch, self.window)
                if np.all(np.isfinite(shift)) and np.isfinite(confidence):
                    results.append(dict(x=x, y=y, dx=shift[0], dy=shift[1],
                                        pixels=float(np.hypot(*shift)), confidence=confidence, texture=texture))
        reliable = [r["pixels"] for r in results if r["confidence"] >= 0.1]
        diagnostic("registration_residual_result", frame=frame, rois=results,
                   median_reliable_residual_pixels=float(np.median(reliable)) if reliable else None,
                   caveat="defocus_and_parallax_can_reduce_phase_confidence")
