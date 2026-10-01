"""A reversible, streaming-compatible scalar luminance correction."""
from __future__ import annotations

import cv2
import numpy as np

from ..utils.performance import diagnostic, stage


FOCUS_MEASURE_VERSION = "full-resolution-gain-v1"


class ExposureGain:
    """Estimate on valid flat overlap; apply one RGB gain before the full warp.

    A single headroom constraint protects every previously unsaturated channel.
    Saturated input channels may remain saturated; no local exposure surface or
    channel-specific correction is used.
    """

    def __init__(self, source, matrix, output_shape, reference_index, *, long_edge=1600):
        self.height, self.width = output_shape
        self.scale = min(1.0, long_edge / max(self.height, self.width))
        self.size = (max(1, round(self.width * self.scale)), max(1, round(self.height * self.scale)))
        self.reference_index = reference_index
        self.reference, self.valid = self.preview(source, matrix)
        self.reference_gray = cv2.cvtColor(self.reference, cv2.COLOR_RGB2GRAY).astype(np.float32)
        self.reference_gradient = self.gradient(self.reference_gray)
        self.gains = {reference_index: 1.0}
        diagnostic("brightness_gain", frame=reference_index, estimated_gain=1.0, applied_gain=1.0,
                   reference=True, eligible_pixels=int(np.count_nonzero(self.valid)),
                   highlight_limited=False, focus_measure_version=FOCUS_MEASURE_VERSION)

    @staticmethod
    def gradient(gray):
        return cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))

    def preview(self, source, matrix):
        sh, sw = source.shape[:2]
        small = cv2.resize(source, (max(1, round(sw * self.scale)), max(1, round(sh * self.scale))),
                           interpolation=cv2.INTER_AREA)
        to_small = np.diag([self.size[0] / self.width, self.size[1] / self.height, 1.0])
        to_source = np.diag([sw / small.shape[1], sh / small.shape[0], 1.0])
        transform = to_small @ matrix @ to_source
        aligned = cv2.warpPerspective(small, transform, self.size, flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.warpPerspective(np.ones(small.shape[:2], np.uint8), transform, self.size,
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) != 0
        # Avoid interpolation of pixels outside the true overlap.
        valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) != 0
        return aligned, valid

    def estimate(self, index, source, matrix):
        with stage("brightness_gain_estimation", frame=index):
            current, valid = self.preview(source, matrix)
            gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gradient = np.maximum(self.reference_gradient, self.gradient(gray))
            eligible = (valid & self.valid & (gray > 16) & (gray < 250)
                        & (self.reference_gray > 16) & (self.reference_gray < 250)
                        & (current.max(axis=2) < 250) & (self.reference.max(axis=2) < 250))
            if np.any(eligible):
                low_gradient = float(np.percentile(gradient[eligible], 35))
                eligible &= gradient <= low_gradient
            samples = int(np.count_nonzero(eligible))
            estimated = float(np.median(self.reference_gray[eligible] / gray[eligible])) if samples >= 256 else 1.0
            gain = float(np.clip(estimated, 0.9, 1.1))
            # Integer JPEG samples below 255 must not become newly clipped to
            # 255. Search in strips to avoid a full-size boolean/value copy.
            maximum_unsaturated = 0
            if gain > 1:
                for y in range(0, source.shape[0], 128):
                    strip = source[y:y + 128]
                    values = strip[strip < 255]
                    maximum_unsaturated = max(maximum_unsaturated, int(values.max()) if values.size else 0)
                if maximum_unsaturated:
                    gain = min(gain, 254.49 / maximum_unsaturated)
            self.gains[index] = gain
            diagnostic("brightness_gain", frame=index, estimated_gain=estimated, applied_gain=gain,
                       eligible_pixels=samples, maximum_unsaturated_channel=maximum_unsaturated,
                       highlight_limited=gain < float(np.clip(estimated, 0.9, 1.1)),
                       focus_measure_version=FOCUS_MEASURE_VERSION)
        return gain

    def apply(self, index, source, matrix):
        gain = self.gains.get(index)
        if gain is None:
            gain = self.estimate(index, source, matrix)
        if gain == 1.0:
            return source
        with stage("brightness_gain_application", frame=index):
            corrected = np.empty_like(source)
            for y in range(0, source.shape[0], 128):
                strip = source[y:y + 128].astype(np.float32)
                strip *= gain
                np.rint(strip, out=strip)
                np.clip(strip, 0, 255, out=strip)
                corrected[y:y + 128] = strip.astype(np.uint8)
            return corrected
