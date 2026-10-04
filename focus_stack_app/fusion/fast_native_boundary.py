"""Coherent native-resolution sources for exposed chromatic silhouettes.

RGB is copied from aligned photographs after the existing tone correction.
Only focus evidence is filtered; the output is never sharpened or deconvolved.
"""
from __future__ import annotations

import cv2
import numpy as np

from .fast_ownership import _nearest_edge_support
from ..utils.performance import diagnostic
from . import fast_cpp as cpp


class NativeBoundary:
    def __init__(self, mask, labels, reference_index, tile_size=384, radius=32):
        # Public callers may provide uint8 labels; the native ABI stores uint16.
        labels = np.ascontiguousarray(labels, dtype=np.uint16)
        self.tiles = []
        self.radius = radius
        self.reference_index = reference_index
        self.shape = labels.shape
        height, width = labels.shape
        mask = cv2.resize(mask.astype(np.uint8), (width, height),
                          interpolation=cv2.INTER_NEAREST) != 0
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                y1, x1 = min(height, y + tile_size), min(width, x + tile_size)
                target = np.ascontiguousarray(mask[y:y1, x:x1])
                if not target.any():
                    continue
                shape = target.shape
                self.tiles.append(dict(
                    x=x, y=y, x1=x1, y1=y1, mask=target,
                    labels=labels[y:y1, x:x1].copy(), best=np.zeros(shape, np.float32),
                    owner=labels[y:y1, x:x1].copy(),
                    reference_score=np.zeros(shape, np.float32),
                    reference_rgb=np.zeros((*shape, 3), np.uint8),
                    rgb=np.zeros((*shape, 3), np.uint8)))
        self.pending = []

    def rank(self, rgb, valid, index):
        self.pending = []
        height, width = valid.shape
        for tile in self.tiles:
            x, y, x1, y1 = (tile[k] for k in ('x', 'y', 'x1', 'y1'))
            # Include the complete filter and nearest-seed support at tile
            # boundaries. Tiles partition storage, not the scoring geometry.
            halo = self.radius + 40
            left, top = max(0, x - halo), max(0, y - halo)
            right, bottom = min(width, x1 + halo), min(height, y1 + halo)
            patch = rgb[top:bottom, left:right]
            chroma = cpp.chroma(patch)
            silhouette = np.uint8(chroma >= 65)
            edge = cv2.morphologyEx(silhouette, cv2.MORPH_GRADIENT,
                                    np.ones((3, 3), np.uint8)) != 0
            signal = cv2.GaussianBlur(chroma.astype(np.float32) / 255, (0, 0), 0.8)
            dx = cv2.Sobel(signal, cv2.CV_32F, 1, 0, scale=1 / 8)
            dy = cv2.Sobel(signal, cv2.CV_32F, 0, 1, scale=1 / 8)
            gradient = cv2.boxFilter(dx * dx + dy * dy, -1, (3, 3))
            lap = cv2.Laplacian(signal, cv2.CV_32F, ksize=3)
            detail = cv2.boxFilter(lap * lap, -1, (3, 3))
            real = cv2.erode(np.uint8(valid[top:bottom, left:right]),
                             np.ones((13, 13), np.uint8)) != 0
            seeds = edge & real & (gradient > 1e-3)
            # Absolute edge energy rejects noisy defocus tails. Aggregate along
            # physical contours before extending a source across a blur wing.
            coherent = cpp.contour_energy(gradient, detail, seeds)
            strength = _nearest_edge_support(coherent, seeds, self.radius)
            score = strength[y - top:y1 - top, x - left:x1 - left]
            better = cpp.rank(tile, valid, score, index, index == self.reference_index)
            self.pending.append((tile, better, index == self.reference_index))

    def capture_corrected(self, rgb):
        for tile, better, is_reference in self.pending:
            cpp.capture(tile, rgb, better, is_reference)
        self.pending.clear()

    def finish(self, output):
        covered = np.zeros(self.shape, np.uint8)
        reference_pixels = replaced_owner = 0
        for tile in self.tiles:
            # Prefer the aligned reference for near ties, avoiding tiny contour
            # displacements between equally sharp photographs. A genuinely
            # sharper source still wins; no sequence position is hard-coded.
            references, replaced = cpp.select(tile, covered, self.reference_index)
            reference_pixels += references
            replaced_owner += replaced
        # Fade only at the outer support boundary, globally across tiles.
        # Crucially, cover pixels even if their label already equals the native
        # winner: V12's seam renderer mixed neighbouring defocused sources there.
        distance = cv2.distanceTransform(covered, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        fade_pixels = max(1.0, min(16.0, self.radius / 2.0))
        for tile in self.tiles:
            cpp.blend(tile, output, covered, distance, fade_pixels)
        diagnostic('fast_native_boundary', version='continuous-boundary-cpp-v3',
                   tile_count=len(self.tiles), covered_pixels=int(np.count_nonzero(covered)),
                   replaced_label_pixels=replaced_owner, reference_near_tie_pixels=reference_pixels,
                   support_radius=self.radius, score_sigma=8.0, reference_score_ratio=0.9,
                   fade_pixels=fade_pixels, fade_curve='smoothstep', same_label_seams_restored=True,
                   source='native_chromatic_gradient_times_sqrt_detail')
        return output
