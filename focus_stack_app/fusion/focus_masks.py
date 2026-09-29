"""Full-resolution focus decisions with bounded, frame-count independent RAM."""
from __future__ import annotations

from ..utils.performance import timed

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from ..utils.memory import memory_snapshot


DEFAULT_FOCUS_GRAY_CACHE_BYTES = 256 * 1024**2


def _gray_cache_budget(max_bytes):
    if max_bytes <= 0:
        return 0
    try:
        snapshot = memory_snapshot()
        reserve = max(2 * 1024**3, int(snapshot.total_bytes * 0.10))
        spare = max(0, snapshot.available_bytes - reserve)
        return min(int(max_bytes), spare // 4)
    except Exception:
        return 0


def _ordered_prefetched_frames(count, load_aligned, cancel_event):
    """Load at most one upcoming frame while the current frame is scored."""
    if count <= 1:
        for index in range(count):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("focus fusion cancelled")
            yield index, load_aligned(index)
        return

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="focus-prefetch")
    future = None
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("focus fusion cancelled")
        future = executor.submit(load_aligned, 0)
        for index in range(count):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("focus fusion cancelled")
            rgb = future.result()
            future = None
            if index + 1 < count:
                future = executor.submit(load_aligned, index + 1)
            yield index, rgb
    finally:
        if future is not None:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


class SurfaceToneMonitor:
    """Detect frame-to-frame colour drift away from actual subject edges.

    Hard, single-level masks reveal this drift as islands of different colour.
    Ignore edges and their defocus fringe so a focus change alone does not
    enable multiband blending, which can otherwise introduce silhouette halos.
    The monitor retains only a 320-pixel RGB reference and a boolean mask.
    """

    def __init__(self):
        self.reference = self.flat = None
        self.needs_multiband = False

    def observe(self, rgb):
        if self.needs_multiband:
            return
        height, width = rgb.shape[:2]
        scale = min(1.0, 320.0 / max(height, width))
        small = cv2.resize(rgb, (max(1, round(width * scale)),
                                max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small.astype(np.float32), (0, 0), 1.0)
        edges = np.zeros(small.shape[:2], np.uint8)
        edges[:, 1:] |= (np.max(np.abs(small[:, 1:] - small[:, :-1]), axis=2) > 1.5)
        edges[1:] |= (np.max(np.abs(small[1:] - small[:-1]), axis=2) > 1.5)
        flat = cv2.dilate(edges, np.ones((17, 17), np.uint8)) == 0
        if self.reference is None:
            self.reference, self.flat = small, flat
            return
        common = flat & self.flat
        difference = np.max(np.abs(small - self.reference), axis=2)
        # Require substantial evidence: >3 levels in at least 5% of the
        # image, all in flat areas of both frames. Noise and tiny details do
        # not switch the renderer. Explicit Enfuse levels remain authoritative.
        self.needs_multiband = np.count_nonzero(common & (difference > 3.0)) >= max(
            64, round(common.size * 0.05))


def focus_response(rgb, gray=None, *, support_radius=7):
    """Compare fine detail on a common scale, including its defocus fringe.

    A defocused edge can win just outside the sharp silhouette, where the
    sharp image is flat. Extend nearby focus evidence by the requested
    support radius so that the fringe belongs to the sharp edge too. A
    multiband renderer needs wider ownership than the seven-pixel default,
    otherwise its coarse layers pull blurred text rims back into the result.
    Do not normalise each
    frame independently: that would promote a uniformly blurry frame.
    """
    if gray is None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32) / 255.0
    lap = cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 0.6), cv2.CV_32F, ksize=3)
    response = cv2.GaussianBlur(lap * lap, (0, 0), 4.0)
    diameter = 2 * int(support_radius) + 1
    return cv2.dilate(response, np.ones((diameter, diameter), np.uint8))


class _StableBackground:
    """Keep textureless areas on one exposure instead of chasing sensor noise.

    Retain a union of reliable detail across every focus plane. Fine detail,
    coherent low-contrast edges and chromatic subjects all veto replacement.
    Only the union and per-frame scalar statistics survive each observation.
    """

    def __init__(self):
        self.detail = None
        self.tones = []

    def observe(self, rgb, gray, score):
        # Bound the adaptive noise estimate: a fully textured photograph must
        # never classify its lowest-scoring fifth as a textureless background.
        noise = float(np.percentile(score[::8, ::8], 20))
        fine_floor = max(1e-5, min(3 * noise, 6e-4))
        smooth = cv2.GaussianBlur(gray.astype(np.float32) / 255.0, (0, 0), 2.0)
        lap = cv2.Laplacian(smooth, cv2.CV_32F, ksize=3)
        coherent = cv2.GaussianBlur(lap * lap, (0, 0), 4.0)
        coarse_noise = float(np.percentile(coherent[::8, ::8], 20))
        coarse_floor = max(2e-6, min(8 * coarse_noise, 1e-5))
        detail = (score > fine_floor) | (coherent > coarse_floor)
        del smooth, lap, coherent
        r, g, b = cv2.split(rgb)
        chroma = cv2.subtract(cv2.max(r, cv2.max(g, b)), cv2.min(r, cv2.min(g, b)))
        detail |= chroma > 40
        if self.detail is None:
            self.detail = detail.astype(np.uint8)
        else:
            self.detail |= detail
        # A central exposure preserves the original light/shadow gradient.
        # No filename, capture index or manually named result defines it.
        self.tones.append(float(gray[::8, ::8].mean()))

    def apply(self, labels, protected=None):
        if len(self.tones) < 2:
            return
        detail = cv2.dilate(self.detail, np.ones((41, 41), np.uint8))
        count, regions, stats, _ = cv2.connectedComponentsWithStats(1 - detail)
        minimum = max(1024, round(labels.size * 0.002))
        keep = np.zeros(count, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum
        flat = keep[regions]
        if protected is not None:
            flat &= ~protected
        anchor = int(np.argmin(np.abs(np.asarray(self.tones) - np.median(self.tones))))
        labels[flat] = anchor


def _filled_chromatic_silhouette(subject):
    # Holes in coloured print are lettering, not the external background.
    contours, _ = cv2.findContours(subject, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(subject)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return filled


def _textured_exterior(rgb, silhouette, scale):
    """Where real neutral backdrop texture can compete with an outer edge."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    score = focus_response(rgb, gray=gray, support_radius=0)
    noise = float(np.percentile(score[::8, ::8], 20))
    threshold = max(3e-4, min(6 * noise, 3e-3))
    # Exclude the silhouette's own gradient and thin neutral blade rims.
    clearance = max(24, round(12 * scale))
    exterior = cv2.erode(1 - silhouette, np.ones((2 * clearance + 1,) * 2, np.uint8))
    texture = ((score > threshold) & (exterior != 0)).astype(np.uint8)
    radius = max(clearance + max(2, round(12 * scale)), round(45 * scale))
    return cv2.dilate(texture, np.ones((2 * radius + 1,) * 2, np.uint8))


def _local_boundary_detail(rgb, scale):
    """Measure the nearby physical rim, rejecting unstructured texture noise.

    Smoothed intensity gradients retain a thin neutral metal rim, whereas
    squared Laplacian energy alone can still be dominated by backdrop grain.
    The short support cannot borrow focus from a distant printed feature.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (0, 0), 2.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    radius = max(2, round(12 * scale))
    return cv2.dilate(gx * gx + gy * gy, np.ones((2 * radius + 1,) * 2, np.uint8))


def _chromatic_boundary_response(rgb, subject, scale):
    """Measure a coloured boundary without using neutral backdrop texture."""
    smooth = cv2.GaussianBlur(rgb, (0, 0), 1.2)
    chroma = (smooth.max(axis=2).astype(np.float32) - smooth.min(axis=2)) / 255.0
    gx = cv2.Sobel(chroma, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(chroma, cv2.CV_32F, 0, 1)
    edge = cv2.morphologyEx(subject, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    score = (gx * gx + gy * gy) * edge
    radius = max(2, round(45 * scale))
    score = cv2.dilate(score, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    return cv2.GaussianBlur(score, (0, 0), max(1.0, 3 * scale))


@timed("focus_masks")
def build_focus_labels(
    count, load_aligned, *, cancel_event=None, protect_chromatic_edges=False,
    gray_cache_bytes=DEFAULT_FOCUS_GRAY_CACHE_BYTES, tone_monitor=None,
    focus_support_radius=7, stabilize_background=True,
):
    """Choose focused pixels and protect nearby neutral subject silhouettes.

    A background-focused frame may retain a displaced, defocused subject edge.
    For a coloured subject with a bright neutral rim, choose the frame whose
    local rim boundary is sharpest and use it just outside the silhouette.
    """
    if not 0 <= int(focus_support_radius) <= 128:
        raise ValueError("focus support radius must be between 0 and 128")
    if not 1 <= count <= 65535:
        raise ValueError("focus fusion requires between 1 and 65535 frames")
    best = labels = silhouette_union = rim_best = rim_owner = None
    colour_union = colour_best = colour_owner = colour_luma = None
    winner_luma = None
    boundary_best = boundary_owner = boundary_texture = boundary_silhouette = None
    boundary_local_best = boundary_local_selected = None
    background = _StableBackground() if stabilize_background else None
    gray_cache_budget = _gray_cache_budget(int(gray_cache_bytes)) if protect_chromatic_edges else 0
    gray_frames = {}
    gray_bytes = 0
    if protect_chromatic_edges:
        from .alignment_quality import colour_subject_mask
    for index, rgb in _ordered_prefetched_frames(count, load_aligned, cancel_event):
        if tone_monitor is not None:
            tone_monitor.observe(rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        score = focus_response(rgb, gray=gray, support_radius=focus_support_radius)
        if background is not None:
            background.observe(rgb, gray, score)
        if gray_bytes + gray.nbytes <= gray_cache_budget:
            gray_frames[index] = gray
            gray_bytes += gray.nbytes
        if best is None:
            best = score
            labels = np.zeros(score.shape, np.uint16)
            if protect_chromatic_edges:
                winner_luma = gray.copy()
        else:
            if score.shape != best.shape:
                raise ValueError("aligned focus frames must have matching dimensions")
            better = score > best
            best[better] = score[better]
            labels[better] = index
            if protect_chromatic_edges:
                winner_luma[better] = gray[better]
        if protect_chromatic_edges:
            height, width = score.shape
            scale = min(1.0, 2048.0 / max(height, width))
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
            subject = colour_subject_mask(small)
            if subject is not None:
                if silhouette_union is None:
                    silhouette_union = np.zeros(subject.shape, np.uint8)
                    rim_best = np.zeros(subject.shape, np.float32)
                    rim_owner = np.zeros(subject.shape, np.uint16)
                    colour_union = np.zeros(subject.shape, np.uint8)
                    colour_best = np.zeros(subject.shape, np.float32)
                    colour_owner = np.zeros(subject.shape, np.uint16)
                    colour_luma = np.zeros(subject.shape, np.uint8)
                colour_union |= subject
                filled = _filled_chromatic_silhouette(subject)
                local_focus = _local_boundary_detail(small, scale)
                boundary_score = _chromatic_boundary_response(small, filled, scale)
                if boundary_best is None:
                    boundary_best = np.zeros(subject.shape, np.float32)
                    boundary_owner = np.zeros(subject.shape, np.uint16)
                    boundary_texture = np.zeros(subject.shape, np.uint8)
                    boundary_silhouette = np.zeros(subject.shape, np.uint8)
                    boundary_local_best = np.zeros(subject.shape, np.float32)
                    boundary_local_selected = np.zeros(subject.shape, np.float32)
                np.maximum(boundary_local_best, local_focus, out=boundary_local_best)
                boundary_texture |= _textured_exterior(small, filled, scale)
                boundary_silhouette |= filled
                better_boundary = boundary_score > boundary_best
                boundary_best[better_boundary] = boundary_score[better_boundary]
                boundary_owner[better_boundary] = index
                boundary_local_selected[better_boundary] = local_focus[better_boundary]
                del boundary_score, better_boundary, local_focus
                # The coloured print is inset from the white blade edge.
                rim_radius = max(2, round(60 * scale))
                diameter = rim_radius * 2 + 1
                nearby = cv2.dilate(subject, cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (diameter, diameter)))
                gray_small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
                bright = np.uint8(gray_small > 120)
                white_rim = nearby & bright
                silhouette_union |= subject | white_rim
                small_score = cv2.resize(score, size, interpolation=cv2.INTER_AREA)
                colour_score = cv2.GaussianBlur(small_score * subject, (0, 0),
                                                max(2.0, 35.0 * scale))
                better = colour_score > colour_best
                colour_best[better] = colour_score[better]
                colour_owner[better] = index
                colour_luma[better] = gray_small[better]
                # Evaluate the actual neutral rim boundary. Printed texture
                # elsewhere on a long, slanted subject can be in a different
                # focus plane and must not choose this edge's owner.
                edge = white_rim - cv2.erode(white_rim, np.ones((3, 3), np.uint8))
                rim_score = small_score * edge
                rim_score = cv2.GaussianBlur(rim_score, (0, 0), max(2.0, 35.0 * scale))
                better = rim_score > rim_best
                rim_best[better] = rim_score[better]
                rim_owner[better] = index
                del rim_score, colour_score, small_score
            del small, subject
        del rgb, score
    del best
    protected = None
    if rim_best is not None:
        radius = max(24, min(75, round(max(labels.shape) * 0.012)))
        diameter = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        red = cv2.resize(colour_union, (labels.shape[1], labels.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        red_band = cv2.dilate(red, kernel)
        colour_evidence = cv2.resize(colour_best, (labels.shape[1], labels.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
        base_guard = (red == 0) & (red_band != 0) & (colour_evidence > 0)
        base_owner = cv2.resize(colour_owner, (labels.shape[1], labels.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        base_luma = cv2.resize(colour_luma, (labels.shape[1], labels.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        labels[base_guard] = base_owner[base_guard]
        winner_luma[base_guard] = base_luma[base_guard]
        owner = cv2.resize(rim_owner, (labels.shape[1], labels.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        evidence = cv2.resize(rim_best, (labels.shape[1], labels.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        silhouette = cv2.resize(silhouette_union, (labels.shape[1], labels.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        outer_band = cv2.dilate(silhouette, kernel)
        guarded = (silhouette == 0) & (outer_band != 0) & (evidence > 0)
        # The rim can be only a few full-size pixels wide. Recheck candidate
        # pixels at full resolution: a downsampled preview can miss a white
        # fragment that has shifted into the dark background.
        for index in np.unique(owner[guarded]):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("focus fusion cancelled")
            ys, xs = np.where(guarded & (owner == index))
            candidate_gray = gray_frames.get(int(index))
            if candidate_gray is None:
                candidate_rgb = load_aligned(int(index))
                candidate_gray = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2GRAY)
                del candidate_rgb
            candidate = candidate_gray[ys, xs]
            previous = winner_luma[ys, xs]
            false_foreground = (candidate > 110) & (
                candidate.astype(np.int16) - previous.astype(np.int16) > 50)
            guarded[ys[false_foreground], xs[false_foreground]] = False
        labels[guarded] = owner[guarded]
        # Neutral scratches behind the subject can outscore its defocused
        # edge in luminance, including pixels inside the colour-mask union.
        # Use chromatic sharpness on BOTH sides of that boundary. Keep the
        # override narrow (12 source pixels) to retain nearby background detail.
        external = cv2.resize(boundary_silhouette, (labels.shape[1], labels.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        boundary_band = cv2.morphologyEx(
            external, cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
        ) != 0
        boundary_evidence = cv2.resize(boundary_best, (labels.shape[1], labels.shape[0]))
        boundary_band &= boundary_evidence > 0.002
        # Only override the regular mask when external textured background
        # actually creates competition. Smooth backdrops and printed letters
        # must retain their normal fine-detail/neutral-rim focus decisions.
        boundary_band &= cv2.resize(
            boundary_texture, (labels.shape[1], labels.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) != 0
        # A wide chromatic neighbourhood can favour a sharp paint feature
        # while its adjacent neutral metal rim is out of focus. It may only
        # override a pixel when its own local luminance edge detail is close
        # to the best local detail observed across the stack.
        reliable = boundary_local_selected >= 0.8 * boundary_local_best
        boundary_band &= cv2.resize(reliable.astype(np.uint8),
                                   (labels.shape[1], labels.shape[0]),
                                   interpolation=cv2.INTER_NEAREST) != 0
        boundary_labels = cv2.resize(boundary_owner, (labels.shape[1], labels.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
        labels[boundary_band] = boundary_labels[boundary_band]
        protected = base_guard | guarded | boundary_band
    if background is not None:
        background.apply(labels, protected=protected)
    return labels


def focus_weight(labels, index):
    """Feather only one output pixel, without averaging away thin features."""
    return cv2.GaussianBlur((labels == index).astype(np.float32), (0, 0), 1.0)


@timed("fusion")
def blend_focus_pyramid(count, load_aligned, labels, *, cancel_event=None, levels=1):
    """Blend focus winners without adding dark/bright silhouette halos.

    The default is a convex RGB blend with a one-pixel transition. Combining
    sharp detail with a defocused frame's coarse pyramid layers can overshoot
    the brightness of every source, creating a wide dark rim on a dark
    background (or a bright rim on a light one). Do not use those layers for
    the default focus stack. Explicit multiband callers remain supported.

    Accumulate one frame at a time to keep RAM independent of frame count.
    """
    if levels < 1:
        raise ValueError("focus blend levels must be at least 1")
    accumulators = []
    for index in range(count):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("focus fusion cancelled")
        pixels = load_aligned(index).astype(np.float32)
        weight = focus_weight(labels, index)
        for level in range(levels):
            last = level == levels - 1 or min(pixels.shape[:2]) <= 2
            if not last:
                smaller = cv2.pyrDown(pixels)
                detail = pixels - cv2.pyrUp(smaller, dstsize=(pixels.shape[1], pixels.shape[0]))
            else:
                detail = pixels
            contribution = detail * weight[..., None]
            if index == 0:
                accumulators.append(contribution)
            else:
                accumulators[level] += contribution
            if last:
                break
            pixels = smaller
            weight = cv2.pyrDown(weight)
    result = accumulators.pop()
    for detail in reversed(accumulators):
        result = cv2.pyrUp(result, dstsize=(detail.shape[1], detail.shape[0])) + detail
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)

