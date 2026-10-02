"""Small, deterministic cleanup for the quality renderer's focus ownership.

The focus detector works at full output resolution.  On low-colour materials
it can still switch owners within a few pixels, making a sharp edge look like
several tiny blocks.  Smooth only those ownership islands; retain the
detector's decisions on coloured print and surfaces.
"""

from __future__ import annotations

import cv2
import numpy as np
from .statistics_native import flat_noise, chroma_map
from ..utils.performance import timed


@timed("label_regularization")
def stabilize_neutral_labels(labels: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Remove tiny focus-owner islands without softening the source pixels.

    This changes labels, not RGB data.  The final renderer still copies the
    selected full-resolution frame at each pixel with a one-pixel seam.
    """
    if labels.shape != reference.shape[:2]:
        raise ValueError("focus labels and aligned reference have different dimensions")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("aligned reference must be RGB")
    if labels.size == 0 or int(labels.max()) > 255:
        return labels

    gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    red, green, blue = cv2.split(reference)
    chroma = cv2.subtract(cv2.max(red, cv2.max(green, blue)),
                          cv2.min(red, cv2.min(green, blue)))
    del red, green, blue
    neutral = (gray < 235) & (chroma < 65)
    spatial_median = cv2.medianBlur(labels.astype(np.uint8), 15)
    stable = labels.copy()
    stable[neutral] = spatial_median[neutral]
    return stable


FOCUS_REGULARIZATION_VERSION = "flat-texture-gate-v4"


def neutral_mask(reference, gray=None):
    """Capture the compatibility mask while streaming selected frame zero."""
    if gray is None:
        gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    chroma = reference.max(axis=2) - reference.min(axis=2)
    return (gray < 235) & (chroma < 65)


@timed("label_regularization")
def stabilize_neutral_mask(labels, neutral):
    if labels.shape != neutral.shape:
        raise ValueError("focus labels and neutral mask have different dimensions")
    if labels.size == 0 or int(labels.max()) > 255:
        return labels
    result = labels.copy()
    median = cv2.medianBlur(labels.astype(np.uint8), 15)
    result[neutral] = median[neutral]
    return result


class FlatTextureStatistics:
    """Quarter-size running statistics; full-size scores still choose owners."""

    def __init__(self, reference_index, *, maximum_pixels=12_000_000, edge_radius=16,
                 edge_mode="coherent"):
        if edge_mode not in {"coherent", "localized"}:
            raise ValueError("edge_mode must be coherent or localized")
        self.edge_mode = edge_mode
        self.reference_index = int(reference_index)
        self.maximum_pixels = int(maximum_pixels)
        self.edge_radius = int(edge_radius)
        self.size = None
        self.max_response = self.max_variance = self.max_gradient = self.max_chroma = None
        self.focus_evidence = self.texture_evidence = self.edge_evidence = None
        self.reference_gray = self.neutral = None
        self.focus_floor = np.zeros(16, np.float32)
        self.variance_floor = np.zeros(16, np.float32)
        self.gradient_floor = np.zeros(16, np.float32)
        self.samples = np.zeros(16, np.int64)
        self.noise_fallback_counts = np.zeros(16, np.int64)

    @staticmethod
    def _noise_upper(values):
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return max(np.finfo(np.float32).eps, median + 6 * 1.4826 * mad)

    def observe(self, index, rgb, gray, score):
        h, w = gray.shape
        if self.size is None:
            scale = min(0.5, (self.maximum_pixels / (h * w)) ** 0.5)
            self.scale = scale
            self.size = (max(1, int(w * scale)), max(1, int(h * scale)))
        small = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA)
        luma = small.astype(np.float32) / 255.0
        mean = cv2.GaussianBlur(luma, (0, 0), 2.0)
        contrast_mean = cv2.GaussianBlur(mean, (0, 0), 2.0)
        # Remove the local illumination slope from the texture measurement.
        # A gradual shadow/white-paper gradient is not a physical detail edge.
        detail = mean - contrast_mean
        variance = cv2.GaussianBlur(detail * detail, (0, 0), 2.0)
        np.maximum(variance, 0, out=variance)
        # A coherent, smoothed gradient protects a physical edge. A single
        # noisy pixel must not grow into a 33-pixel exclusion region.
        gradient = cv2.magnitude(cv2.Sobel(mean, cv2.CV_32F, 1, 0), cv2.Sobel(mean, cv2.CV_32F, 0, 1))
        response = cv2.resize(score, self.size, interpolation=cv2.INTER_AREA)
        small_rgb = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA)
        chroma = chroma_map(small_rgb)
        if self.max_response is None:
            self.max_response, self.max_variance, self.max_gradient = response, variance, gradient
            self.max_chroma = chroma
        else:
            np.maximum(self.max_response, response, out=self.max_response)
            np.maximum(self.max_variance, variance, out=self.max_variance)
            np.maximum(self.max_gradient, gradient, out=self.max_gradient)
            np.maximum(self.max_chroma, chroma, out=self.max_chroma)
        if index == self.reference_index:
            self.reference_gray = small.copy()
        # The compatibility rule historically uses selected frame zero,
        # independently of the geometric/exposure reference.
        if index == 0:
            r, g, b = cv2.split(rgb)
            full_chroma = cv2.max(r, cv2.max(g, b)) - cv2.min(r, cv2.min(g, b))
            self.neutral = (gray < 235) & (full_chroma < 65)
        # Sample compact maps, with a separate noise distribution per luma bin.
        frame_focus, frame_variance, frame_gradient = flat_noise(
            self, small, variance, gradient, response)
        # Compare each frame to its own luma/noise bin before taking a union.
        # Comparing every maximum to the reference luma bin confuses real
        # exposure drift with detail and can protect an entire white backdrop.
        full_bins = small // 16
        known = frame_focus[full_bins] > 0
        focus = ~known | (response > frame_focus[full_bins])
        texture = ~known | (variance > frame_variance[full_bins])
        coherent_edge = known & (gradient > frame_gradient[full_bins]) & texture
        if self.edge_mode == "localized":
            # Illumination slopes retain their gradient across scales; a
            # physical in-focus rim has a localized gradient peak. Require
            # that distinction before a weak paper slope grows a safety band.
            broad = cv2.GaussianBlur(luma, (0, 0), 8.0)
            broad_gradient = cv2.magnitude(cv2.Sobel(broad, cv2.CV_32F, 1, 0),
                                           cv2.Sobel(broad, cv2.CV_32F, 0, 1))
            localized = gradient > 2.0 * broad_gradient
            coherent_edge &= localized
            # Variance by itself includes curved lighting and JPEG grain.
            # It vetoes locking only with focus or localized edge evidence.
            texture = ~known | (texture & (focus | coherent_edge))
        coherent_edge = cv2.morphologyEx(coherent_edge.astype(np.uint8), cv2.MORPH_OPEN,
                                        np.ones((3, 3), np.uint8)) != 0
        edge = ~known | coherent_edge
        if self.focus_evidence is None:
            self.focus_evidence, self.texture_evidence, self.edge_evidence = focus, texture, edge
        else:
            self.focus_evidence |= focus
            self.texture_evidence |= texture
            self.edge_evidence |= edge

    @timed("texture_gate_regularization")
    def apply(self, labels):
        from ..utils.performance import diagnostic
        if self.reference_gray is None:
            raise ValueError("texture statistics did not observe the selected reference frame")
        bins = (self.reference_gray // 16).astype(np.uint8)
        known = self.samples[bins] > 0
        no_focus_evidence = ~self.focus_evidence
        flat = ~self.texture_evidence
        # Retain the existing chromatic-body decisions. Luma is not an upper
        # cutoff: genuine 240..255 white background participates in this gate.
        flat &= self.max_chroma < 40
        reliable_edges = self.edge_evidence
        radius = max(1, int(np.ceil(self.edge_radius * self.scale)))
        protected = cv2.dilate(reliable_edges.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8))
        interior_pixels = 0
        if self.edge_mode == "localized":
            # A flat product surface enclosed by its protected silhouette is
            # still foreground. Protect it instead of changing its exposure.
            # This classifies regions, not focus-owner component cleanup.
            count, regions = cv2.connectedComponents(np.uint8(protected == 0), 8)
            exterior = np.zeros(count, bool)
            exterior[np.unique(np.concatenate((regions[0], regions[-1],
                                                regions[:, 0], regions[:, -1])))] = True
            interior = (protected == 0) & ~exterior[regions]
            interior_pixels = int(np.count_nonzero(interior))
            protected[interior] = 1
        locked = (known & flat & no_focus_evidence & (protected == 0)).astype(np.uint8)
        size = (labels.shape[1], labels.shape[0])
        locked = cv2.resize(locked, size, interpolation=cv2.INTER_NEAREST) != 0
        protected = cv2.resize(protected, size, interpolation=cv2.INTER_NEAREST) != 0
        result = labels.copy()
        if int(labels.max()) <= 255:
            median = cv2.medianBlur(labels.astype(np.uint8), 15)
            result[self.neutral] = median[self.neutral]
        result[locked] = self.reference_index
        diagnostic("texture_gate", statistics_size=list(self.size), maximum_pixels=self.maximum_pixels,
                   locked_pixels=int(np.count_nonzero(locked)), edge_protected_pixels=int(np.count_nonzero(protected)),
                   reference_index=self.reference_index, noise_samples=self.samples,
                   focus_noise_upper=self.focus_floor, variance_noise_upper=self.variance_floor,
                   gradient_noise_upper=self.gradient_floor,
                   adjacent_noise_fallback_counts=self.noise_fallback_counts,
                   protected_interior_statistics_pixels=interior_pixels,
                   edge_mode=self.edge_mode,
                   focus_regularization_version=(FOCUS_REGULARIZATION_VERSION if self.edge_mode == "coherent"
                                                 else "flat-texture-gate-localized-v5"))
        return result, protected


@timed("tiny_label_components")
def clean_tiny_labels(labels, protected, *, maximum_area=8, cancel_event=None):
    """Optional final stage; never merge a tiny component in an edge safety band."""
    result = labels.copy()
    for owner in np.unique(labels):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("focus fusion cancelled")
        count, parts, stats, _ = cv2.connectedComponentsWithStats((labels == owner).astype(np.uint8), 8)
        for part in range(1, count):
            if stats[part, cv2.CC_STAT_AREA] > maximum_area:
                continue
            x, y, w, h = stats[part, :4]
            x0, y0 = max(0, x - 1), max(0, y - 1)
            x1, y1 = min(labels.shape[1], x + w + 1), min(labels.shape[0], y + h + 1)
            component = parts[y0:y1, x0:x1] == part
            if np.any(protected[y0:y1, x0:x1][component]):
                continue
            neighbours = labels[y0:y1, x0:x1][~component]
            neighbours = neighbours[neighbours != owner]
            if neighbours.size:
                dominant = int(np.bincount(neighbours).argmax())
                result[y0:y1, x0:x1][component] = dominant
    return result


def _nearest_edge_support(strength, edge, radius):
    """Extend the nearest physical edge, without borrowing a distant corner."""
    present = edge != 0
    if not np.any(present):
        return np.zeros_like(strength)
    density = cv2.GaussianBlur(present.astype(np.float32), (0, 0), 2.0)
    averaged = cv2.GaussianBlur(strength * present, (0, 0), 2.0)
    averaged /= np.maximum(density, 1e-6)
    distance, nearest = cv2.distanceTransformWithLabels(
        np.uint8(~present), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    values = np.zeros(int(nearest.max()) + 1, np.float32)
    values[nearest[present]] = averaged[present]
    weight = np.clip((radius + 1.0 - distance) / max(2.0, radius / 4.0), 0, 1)
    return values[nearest] * weight


class SurfaceBoundaryOwnership:
    """Keep a coloured silhouette and its defocus fringe on a sharp source."""

    def __init__(self, *, maximum_pixels=3_000_000, support_radius=64):
        self.maximum_pixels = int(maximum_pixels)
        self.support_radius = int(support_radius)
        self.best = self.confidence = self.owner = self.band = self.texture = self.interior = None
        self.owner_focus = self.neutral_focus = None

    def observe(self, index, rgb, score=None):
        from .focus_masks import _filled_chromatic_silhouette

        if score is None:
            from .focus_masks import focus_response
            score = focus_response(rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), support_radius=7)

        height, width = rgb.shape[:2]
        scale = min(0.5, (self.maximum_pixels / (height * width)) ** 0.5)
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        chroma = chroma_map(small)
        count, parts, stats, _ = cv2.connectedComponentsWithStats(np.uint8(chroma >= 65), 8)
        retained = np.zeros(count, np.uint8)
        retained[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(64, round(chroma.size * 0.001))
        silhouette = _filled_chromatic_silhouette(retained[parts])
        edge = cv2.morphologyEx(silhouette, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        # Never substitute a blurred copy of genuinely textured colour. The
        # union across focus planes also protects texture when another plane
        # is blurred enough to look flat.
        low_sigma = 2.0
        # White ink must not enter a paint pixel's texture estimate through
        # the low-pass filter. Keep its samples at least three sigma inside
        # the material, rather than protecting that colour fringe as texture.
        clearance = int(np.ceil(3 * low_sigma))
        core = cv2.erode(np.uint8(chroma >= 65), np.ones((2 * clearance + 1,) * 2, np.uint8))
        low = cv2.GaussianBlur(small.astype(np.float32), (0, 0), low_sigma)
        fine = np.mean((small.astype(np.float32) - low) ** 2, axis=2)
        density = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 3.0)
        texture = cv2.GaussianBlur(fine * core, (0, 0), 3.0)
        texture = (texture / np.maximum(density, 1e-6) > 144) & (density > 0.4)
        signal = cv2.GaussianBlur(chroma.astype(np.float32) / 255.0, (0, 0), 1.0)
        gx = cv2.Sobel(signal, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(signal, cv2.CV_32F, 0, 1)
        colour_strength = gx * gx + gy * gy
        radius = max(1, int(np.ceil(self.support_radius * scale)))
        confidence = _nearest_edge_support(colour_strength, edge, radius)
        # A darker or more saturated defocused rim may have a larger colour
        # step than a sharp one. Rank evidence at the physical edge using the
        # existing high-frequency focus score, before extending it to fringes.
        focus = cv2.resize(score, size, interpolation=cv2.INTER_AREA)
        # A nearby painted rim may focus at a different depth from neutral
        # metal or engraving. Its propagated owner must not replace sharper
        # detail at the target itself. Keep this short core local to this
        # guard; expanding it would also protect defocused printing fringes.
        luma = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        clearance = max(1, int(np.ceil(4 * scale)))
        neutral = cv2.erode(np.uint8((chroma < 45) & (luma > 30) & (luma < 235)),
                            np.ones((2 * clearance + 1,) * 2, np.uint8)) != 0
        neutral_focus = np.where(neutral, focus, 0)
        strength = _nearest_edge_support(
            colour_strength * np.sqrt(np.maximum(focus, 0)), edge, radius)
        band_radius = radius
        band = cv2.dilate(edge, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * band_radius + 1,) * 2))
        if self.best is None:
            self.best = strength
            self.confidence = confidence
            self.owner = np.full(strength.shape, index, np.uint16)
            self.band = band
            self.texture = texture
            self.interior = silhouette
            self.scale = scale
            self.owner_focus = focus.copy()
            self.neutral_focus = neutral_focus
        else:
            better = (strength > self.best) | ((strength == self.best) & (index < self.owner))
            self.best[better] = strength[better]
            self.confidence[better] = confidence[better]
            self.owner[better] = index
            self.owner_focus[better] = focus[better]
            np.maximum(self.neutral_focus, neutral_focus, out=self.neutral_focus)
            self.band |= band
            self.texture |= texture
            self.interior |= silhouette

    def apply(self, labels):
        from ..utils.performance import diagnostic

        if self.best is None:
            return labels
        texture = self.texture_protection(labels.shape)
        size = (labels.shape[1], labels.shape[0])
        clearance = max(1, int(np.ceil(8 * self.scale)))
        interior = cv2.dilate(self.interior, np.ones((2 * clearance + 1,) * 2, np.uint8)) != 0
        neutral_detail = ((self.neutral_focus > 0)
                          & (self.owner_focus < 0.7 * self.neutral_focus))
        active = cv2.resize(np.uint8((self.band != 0) & (self.confidence > 0.002)
                                    & (self.best > 0) & interior & ~neutral_detail), size,
                            interpolation=cv2.INTER_NEAREST) != 0
        active &= ~texture
        owner = cv2.resize(self.owner, size, interpolation=cv2.INTER_NEAREST)
        result = labels.copy()
        changed = active & (result != owner)
        result[active] = owner[active]
        diagnostic("surface_boundary_ownership", changed_pixels=int(np.count_nonzero(changed)),
                   guarded_pixels=int(np.count_nonzero(active)), support_radius=self.support_radius,
                   band_radius=self.support_radius,
                   neutral_detail_veto_pixels=int(np.count_nonzero(neutral_detail)),
                   version="focused-chromatic-silhouette-ownership-v4-neutral-detail")
        return result

    def texture_protection(self, shape):
        radius = max(1, int(np.ceil(self.support_radius * self.scale)))
        texture = cv2.dilate(np.uint8(self.texture), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)) != 0
        return cv2.resize(np.uint8(texture), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) != 0


class PrintedEdgeOwnership:
    """Keep sharp internal printing and its defocus fringe on the same source.

    Coloured, otherwise flat pixels can win on a blurred white letter's halo.
    This changes ownership only; it does not correct colour or blur RGB. Maps
    are compact and updated one frame at a time, with capture-index tie rules.
    """

    def __init__(self, *, support_radius=128, maximum_pixels=12_000_000):
        self.support_radius = int(support_radius)
        if not 1 <= self.support_radius <= 128:
            raise ValueError("printed-edge support radius must be between 1 and 128")
        self.maximum_pixels = int(maximum_pixels)
        if self.maximum_pixels < 1:
            raise ValueError("printed-edge statistics must have a positive pixel cap")
        self.best = self.owner = self.inside = None

    def observe(self, index, rgb, gray, score):
        from .focus_masks import _filled_chromatic_silhouette

        scale = min(0.5, (self.maximum_pixels / (rgb.shape[0] * rgb.shape[1])) ** 0.5)
        size = (max(1, int(rgb.shape[1] * scale)), max(1, int(rgb.shape[0] * scale)))
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        luma = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        chroma = chroma_map(small)
        coloured = (chroma >= 65) & (luma >= 30)
        filled = _filled_chromatic_silhouette(coloured.astype(np.uint8))
        # A clipped character can open its white island to the image border.
        # Recover it by contact with paint, without filling the exterior white
        # backdrop. Antialiased ink may retain moderate colour from the paint.
        white = np.uint8((chroma <= 80) & (luma >= 140))
        count, parts, stats, _ = cv2.connectedComponentsWithStats(white, 8)
        rim = white & (cv2.erode(white, np.ones((3, 3), np.uint8),
                                borderType=cv2.BORDER_CONSTANT, borderValue=0) == 0)
        paint_contact = cv2.dilate(np.uint8((chroma >= 90) & (luma >= 30)),
                                   np.ones((3, 3), np.uint8)) != 0
        perimeter = np.bincount(parts[rim != 0], minlength=count)
        contact = np.bincount(parts[(rim != 0) & paint_contact], minlength=count)
        supported = (contact >= 0.6 * np.maximum(perimeter, 1))
        supported[0] = False
        supported[1:] &= stats[1:, cv2.CC_STAT_AREA] >= max(8, round(32 * scale * scale))
        supported_ink = supported[parts]
        filled |= np.uint8(supported_ink)
        # A white island inside colour is printing. Keep the external object
        # rim and background outside this guard, including breathing fringes.
        clearance = max(1, int(np.ceil(8 * scale)))
        inside = cv2.erode(filled, np.ones((2 * clearance + 1,) * 2, np.uint8)) != 0
        # Moderate colour is allowed only on the paint-supported ink island.
        # A blurred gold/coloured border can become bright and low-chroma;
        # admitting it globally would seed a defocused source into the paint.
        printing = (inside & ((chroma <= 45) | supported_ink)
                    & (luma >= 140)).astype(np.uint8)
        # Reject isolated highlights by area, retaining connected thin ink
        # strokes. A 3x3 opening erased sharp fine lines while a defocused,
        # wider copy survived and could wrongly own its pale halo.
        count, parts, stats, _ = cv2.connectedComponentsWithStats(printing, 8)
        retained = np.zeros(count, np.uint8)
        retained[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(8, round(32 * scale * scale))
        printing = retained[parts]
        edge = cv2.morphologyEx(printing, cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8))
        signal = cv2.GaussianBlur(luma.astype(np.float32) / 255.0, (0, 0), 0.6)
        gx = cv2.Sobel(signal, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(signal, cv2.CV_32F, 0, 1)
        strength = gx * gx + gy * gy
        strength *= np.sqrt(np.maximum(cv2.resize(score, size, interpolation=cv2.INTER_AREA), 0))
        strength *= edge
        if self.best is None:
            self.best = strength
            self.owner = np.full(strength.shape, index, np.uint16)
            self.inside = inside
            self.scale = scale
        else:
            better = (strength > self.best) | ((strength == self.best) & (index < self.owner))
            self.best[better] = strength[better]
            self.owner[better] = index
            self.inside |= inside

    def apply(self, labels, selected_coloured=None, *, protected_texture=None):
        from ..utils.performance import diagnostic

        if self.best is None:
            return labels
        if selected_coloured is not None and labels.shape != selected_coloured.shape:
            raise ValueError("printed-edge colour mask and labels differ")
        radius = max(1, int(np.ceil(self.support_radius * self.scale)))
        local_best = cv2.dilate(self.best, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
        seeds = (self.best > 1e-8) & (self.best >= 0.5 * local_best)
        if not np.any(seeds):
            return labels
        distance, nearest = cv2.distanceTransformWithLabels(
            np.uint8(~seeds), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        owners = np.zeros(int(nearest.max()) + 1, np.uint16)
        owners[nearest[seeds]] = self.owner[seeds]
        propagated = owners[nearest]
        # Colour identifies the sharp ink seeds; it must not punch holes in
        # their final coverage. Include white ink, antialiasing and paint.
        active = (distance <= radius) & self.inside
        size = (labels.shape[1], labels.shape[0])
        active = cv2.resize(active.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) != 0
        if protected_texture is not None:
            active &= ~protected_texture
        owner = cv2.resize(propagated, size, interpolation=cv2.INTER_NEAREST)
        result = labels.copy()
        changed = active & (labels != owner)
        result[active] = owner[active]
        diagnostic("printed_edge_ownership", support_radius=self.support_radius,
                   statistics_size=[self.best.shape[1], self.best.shape[0]],
                   guarded_pixels=int(np.count_nonzero(active)),
                   changed_pixels=int(np.count_nonzero(changed)),
                   version="printed-edge-ownership-v12-supported-ink")
        return result


__all__ = ["stabilize_neutral_labels", "FlatTextureStatistics", "clean_tiny_labels",
           "PrintedEdgeOwnership", "SurfaceBoundaryOwnership"]
