"""OpenCV V20 private source-footprint and exterior ownership corrections."""

from __future__ import annotations

import cv2
import numpy as np

from .quality_fusion import SurfaceBoundaryOwnership

OPENCV_REPAIR_VERSION = "opencv-valid-source-local-detail-v20-astra"


class ValidSourceOwnership:
    """Reject reflected warp padding as photographic focus evidence.

    The RGB warp is unchanged. Its reflection is useful filter padding, but
    cannot become a source outside the photographed footprint. Keep the best
    real source for a final check after all ownership propagation as well.
    Only compact geometry is retained per frame, not an image/mask stack.
    """

    def __init__(self):
        self.geometry = {}
        self.best = self.owner = None
        self.rejected_pixels = 0

    def register(self, index, matrix, source_shape):
        self.geometry[int(index)] = (np.asarray(matrix, np.float32).copy(),
                                     tuple(source_shape[:2]))

    def valid_mask(self, index, shape):
        matrix, source_shape = self.geometry[int(index)]
        # Linear sampling of a constant coverage image rejects partial border
        # support too. An identity reference covers its own complete canvas.
        coverage = cv2.warpPerspective(
            np.ones(source_shape, np.float32), matrix, (shape[1], shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return coverage >= 1.0

    def observe(self, index, score):
        valid = self.valid_mask(index, score.shape)
        self.rejected_pixels += int(np.count_nonzero(~valid))
        if self.best is None:
            self.best = np.full(score.shape, -1.0, np.float32)
            self.owner = np.zeros(score.shape, np.uint16)
        better = valid & ((score > self.best)
                          | ((score == self.best) & (index < self.owner)))
        self.best[better] = score[better]
        self.owner[better] = index
        # build_focus_labels calls the Quality observer before comparing this
        # score. This private instance affects only that Quality invocation.
        # Also stops shared edge guards from ranking a reflected edge itself.
        score[~valid] = 0
        return valid

    def apply(self, labels):
        from ..utils.performance import diagnostic

        if self.best is None:
            return labels
        if np.any(self.best < 0):
            raise ValueError("Quality output has pixels outside every source footprint")
        invalid = np.zeros(labels.shape, bool)
        for index in np.unique(labels):
            invalid |= (labels == index) & ~self.valid_mask(int(index), labels.shape)
        result = labels.copy()
        result[invalid] = self.owner[invalid]
        diagnostic("opencv_valid_source_ownership", version=OPENCV_REPAIR_VERSION,
                   rejected_focus_samples=self.rejected_pixels,
                   replaced_invalid_owners=int(np.count_nonzero(invalid)),
                   registered_frames=len(self.geometry))
        return result


def _local_detail_statistics(small, valid=None):
    """Local fine energy and robust noise evidence, with no colour class.

    Compare to a separate noise ceiling for each frame and brightness range.
    Establish flat support across the valid photograph before splitting it by
    brightness. A bin consisting entirely of engraved material has no noise
    support; choosing its own least-textured 35% would still measure engraving.
    Twice the robust upper ceiling rejects marginal grain fluctuations.
    """
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    luma = gray.astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(luma, (0, 0), 0.6)
    lap = cv2.Laplacian(smooth, cv2.CV_32F, ksize=3)
    response = cv2.GaussianBlur(lap * lap, (0, 0), 2.0)
    low = cv2.GaussianBlur(luma, (0, 0), 2.0)
    detail = low - cv2.GaussianBlur(low, (0, 0), 2.0)
    variance = cv2.GaussianBlur(detail * detail, (0, 0), 2.0)
    bins = gray[::4, ::4] // 16
    sampled_variance = variance[::4, ::4]
    sampled_response = response[::4, ::4]
    sampled_valid = np.ones(bins.shape, bool) if valid is None else valid[::4, ::4]
    flat_support = np.zeros(bins.shape, bool)
    if np.count_nonzero(sampled_valid) >= 64:
        variance_limit = np.percentile(sampled_variance[sampled_valid], 35)
        flat_support = sampled_valid & (sampled_variance <= variance_limit)
    noise_floor = np.zeros(16, np.float32)
    for band in range(16):
        flat = (bins == band) & flat_support
        if np.count_nonzero(flat) < 32:
            continue
        values = sampled_response[flat]
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        noise_floor[band] = max(np.finfo(np.float32).eps, median + 6 * 1.4826 * mad)
    # A smooth defocus tail can cross a brightness-bin edge although only the
    # adjoining background bin contains enough actual flat samples. Borrow
    # its conservative noise upper bound, not the tail's own structure. Use
    # original observed bins only: do not propagate through unobserved ranges.
    observed = noise_floor > 0
    for band in range(16):
        if observed[band]:
            continue
        adjacent = [b for b in (band - 1, band + 1) if 0 <= b < 16 and observed[b]]
        if adjacent:
            noise_floor[band] = max(noise_floor[b] for b in adjacent)
    ceiling = noise_floor[gray // 16]
    # Insufficient samples cannot authorise replacing a region's real detail.
    evidence = (ceiling == 0) | (response > 2.0 * ceiling)
    if valid is not None:
        evidence &= valid
    return response, evidence


class OpenCVBoundaryOwnership(SurfaceBoundaryOwnership):
    """Extend V17's exterior coverage while retaining any sharper material.

    Gray metal may have a colour cast stronger than the old neutral threshold.
    Its own spatial detail, across every source, vetoes borrowing the painted
    edge's focus plane. Noise-only defocus tails do not obtain that veto.
    All V17 interior and texture protections remain in the unchanged base.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.local_detail_best = self.local_detail_owner = None

    def observe(self, index, rgb, score=None, *, valid=None):
        super().observe(index, rgb, score)
        size = (self.best.shape[1], self.best.shape[0])
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        valid_small = (None if valid is None else
                       cv2.resize(valid.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 1.0)
        response, evidence = _local_detail_statistics(small, valid_small)
        if valid is None and score is not None:
            # The private source-footprint observer already zeros invalid
            # focus samples; reflected RGB padding must not supply evidence.
            evidence &= cv2.resize(score, size, interpolation=cv2.INTER_AREA) > 0
        detail = np.where(evidence, response, 0)
        if self.local_detail_best is None:
            self.local_detail_best = detail
            self.local_detail_owner = response.copy()
        else:
            np.maximum(self.local_detail_best, detail, out=self.local_detail_best)
            selected = self.owner == index
            self.local_detail_owner[selected] = response[selected]

    def apply(self, labels):
        from ..utils.performance import diagnostic

        result = super().apply(labels)
        if self.best is None:
            return result
        clearance = max(1, int(np.ceil(8 * self.scale)))
        interior = cv2.dilate(
            self.interior, np.ones((2 * clearance + 1,) * 2, np.uint8)) != 0
        independent_detail = ((self.local_detail_best > 0)
                              & (self.local_detail_owner < 0.7 * self.local_detail_best))
        exterior = ((self.band != 0) & ~interior & (self.confidence > 0.002)
                    & (self.best > 0) & ~independent_detail)
        size = (labels.shape[1], labels.shape[0])
        active = cv2.resize(np.uint8(exterior), size, interpolation=cv2.INTER_NEAREST) != 0
        active &= ~self.texture_protection(labels.shape)
        owner = cv2.resize(self.owner, size, interpolation=cv2.INTER_NEAREST)
        changed = active & (result != owner)
        result[active] = owner[active]
        diagnostic("opencv_exterior_local_detail_ownership", version=OPENCV_REPAIR_VERSION,
                   changed_pixels=int(np.count_nonzero(changed)),
                   guarded_pixels=int(np.count_nonzero(active)),
                   independent_detail_veto_pixels=int(np.count_nonzero(independent_detail & ~interior)),
                   support_radius=self.support_radius,
                   noise_model="per-frame-luma-bin-median-plus-six-mad",
                   missing_bin_policy="adjacent-observed-flat-support-only",
                   independent_detail_noise_multiple=2.0)
        return result

