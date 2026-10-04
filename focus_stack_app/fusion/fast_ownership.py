"""Private Fast guards; independent of Quality and Hugin behavior."""
import cv2
import numpy as np
from .statistics_native import chroma_map

def _nearest_edge_support(strength, edge, radius):
    """Extend the nearest physical edge, without borrowing a distant corner."""
    from .fast_cpp import nearest_support
    return nearest_support(strength, edge, radius)


class SurfaceBoundaryOwnership:
    """Keep a coloured silhouette and its defocus fringe on a sharp source."""

    def __init__(self, *, maximum_pixels=3_000_000, support_radius=64, maximum_scale=0.5):
        self.maximum_pixels = int(maximum_pixels)
        self.maximum_scale = float(maximum_scale)
        self.support_radius = int(support_radius)
        self.best = self.confidence = self.owner = self.band = self.texture = self.interior = None
        self.owner_focus = self.neutral_focus = None

    def observe(self, index, rgb, score=None):
        from .focus_masks import _filled_chromatic_silhouette

        if score is None:
            from .focus_masks import focus_response
            score = focus_response(rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), support_radius=7)

        height, width = rgb.shape[:2]
        scale = min(self.maximum_scale, (self.maximum_pixels / (height * width)) ** 0.5)
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

    def __init__(self, *, support_radius=128, maximum_pixels=12_000_000, source_scale=1.0,
                 maximum_scale=0.5):
        self.support_radius = int(support_radius)
        if not 1 <= self.support_radius <= 128:
            raise ValueError("printed-edge support radius must be between 1 and 128")
        self.maximum_pixels = int(maximum_pixels)
        if self.maximum_pixels < 1:
            raise ValueError("printed-edge statistics must have a positive pixel cap")
        self.source_scale = float(source_scale)
        if not 0 < self.source_scale <= 1:
            raise ValueError("printed-edge input scale must be within (0, 1]")
        self.maximum_scale = float(maximum_scale)
        if not 0 < self.maximum_scale <= 1:
            raise ValueError("printed-edge statistics scale must be within (0, 1]")
        self.best = self.owner = self.inside = None

    def observe(self, index, rgb, gray, score):
        from .focus_masks import _filled_chromatic_silhouette

        scale = min(self.maximum_scale, (self.maximum_pixels / (rgb.shape[0] * rgb.shape[1])) ** 0.5)
        # Fast has already reduced the photograph before calling this guard.
        # Scale the component-area floor too: otherwise a sharp small opening
        # is rejected while its larger defocused copy becomes a valid seed.
        minimum_area = max(1, round(max(8, 32 * scale * scale) * self.source_scale ** 2))
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
        supported[1:] &= stats[1:, cv2.CC_STAT_AREA] >= minimum_area
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
        retained[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum_area
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


