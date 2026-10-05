"""Experimental proxy ownership and one full-resolution render pass.

Compressed input bytes are retained between passes: one file read, one reduced
JPEG proxy decode and at most one full RGB decode/warp per selected frame.
The native interpolation is cubic, so direct copying preserves aligned pixels, not
the pre-interpolation sensor samples. Quality claims require image inspection.
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from ..utils.performance import stage, diagnostic, timed
from . import fast_cpp as cpp

VERSION = "fast-cpp-v4-print-statistics"


def _neutral_material_region(detail, chroma, foreground):
    """Require an independently sharp neutral object, not a defocus fringe.

    Sparse neutral samples on a coloured edge or noisy backdrop do not form
    a material region. Filled connected support retains the weak rim around
    actual engraved metal, without lending ownership across an entire image.
    """
    from .focus_masks import _filled_chromatic_silhouette
    neutral = np.float32((detail > 0) & (chroma < 65) & foreground)
    density = cv2.boxFilter(neutral, -1, (11, 11))
    count, parts, stats, _ = cv2.connectedComponentsWithStats(np.uint8(density > 0.4), 8)
    retained = np.zeros(count, np.uint8)
    retained[1:] = stats[1:, cv2.CC_STAT_AREA] >= 32
    material = _filled_chromatic_silhouette(retained[parts])
    return cv2.dilate(material, np.ones((13, 13), np.uint8)) != 0


class _NeutralContourOwnership:
    """Rank weak exposed neutral edges separately from nearby coloured edges."""

    def __init__(self, shape, reference, radius):
        self.best = np.zeros(shape, np.float32)
        self.owner = np.full(shape, reference, np.uint16)
        self.chroma = np.full(shape, 255, np.uint8)
        self.radius = radius
        self.candidates = []

    def observe(self, index, chroma, gray, sharp, detail, gradient, texture, valid):
        from .focus_masks import _filled_chromatic_silhouette
        # Filled colour excludes internal white printing. Pure neutral support
        # excludes the coloured silhouette itself; it cannot borrow its rank.
        filled = _filled_chromatic_silhouette(np.uint8(chroma >= 65)) != 0
        neutral = cv2.erode(np.uint8((chroma < 65) & valid), np.ones((3, 3), np.uint8)) != 0
        # A contour, rather than every internal reflection or engraving,
        # establishes the exposed neutral surface's focus plane. Estimate its
        # backdrop from actual neutral border samples, independent of colour.
        border_gray = np.concatenate((gray[0, ::4], gray[-1, ::4], gray[::4, 0], gray[::4, -1]))
        border_valid = np.concatenate((neutral[0, ::4], neutral[-1, ::4], neutral[::4, 0], neutral[::4, -1]))
        if np.count_nonzero(border_valid) < 16:
            return
        backdrop = float(np.percentile(border_gray[border_valid], 75))
        subject = np.uint8((gray < backdrop - 8) & neutral & ~filled)
        count, parts, stats, _ = cv2.connectedComponentsWithStats(subject, 8)
        retained = np.zeros(count, np.uint8)
        retained[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(32, round(gray.size * 0.0001))
        subject = _filled_chromatic_silhouette(retained[parts])
        contour = cv2.morphologyEx(subject, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) != 0
        # The coloured-edge gradient floor omits a pale rim even when its
        # curvature is above noise. Keep a lower gradient floor for this
        # coherent neutral contour, still requiring per-frame detail evidence.
        seeds = ((gradient > 1e-6) & (detail > _detail_noise_floor(detail))
                 & neutral & ~filled & contour)
        seeds &= _independent_detail_evidence(gray.astype(np.float32) / 255, detail, gradient)
        # Normalise contrast: a bright defocus/reflection step must not outrank
        # the narrower, weaker physical rim just because its amplitude grew.
        width_score = detail / np.maximum(gradient, 1e-6)
        positions = np.flatnonzero(seeds)
        if positions.size:
            # Defer ranking until all independently focused material evidence
            # is known. Filtering only the final targets leaves false seeds
            # able to win and propagate inside an otherwise valid grey object.
            self.candidates.append((index, positions, width_score.ravel()[positions],
                                    detail.ravel()[positions],
                                    np.packbits((valid & (chroma < 65)).ravel())))

    def finalize(self, local_best, local_chroma, local_foreground, material):
        from .fast_ownership import _nearest_edge_support
        accepted = rejected = 0
        for index, positions, scores, details, target_bits in self.candidates:
            use = cpp.neutral_filter(positions, details, local_best, local_chroma,
                                     local_foreground, material)
            accepted += int(np.count_nonzero(use))
            rejected += int(use.size - np.count_nonzero(use))
            if not use.any():
                continue
            seeds = np.zeros(self.best.size, np.uint8)
            seeds[positions[use]] = 1
            strength = np.zeros(self.best.size, np.float32)
            strength[positions[use]] = scores[use]
            strength = _nearest_edge_support(strength.reshape(self.best.shape),
                                             seeds.reshape(self.best.shape) != 0, self.radius)
            targets = np.unpackbits(target_bits, count=self.best.size).reshape(self.best.shape) != 0
            cpp.neutral_update(targets, material, strength, self.best, self.owner,
                               self.chroma, index)
        self.candidates.clear()
        diagnostic('fast_neutral_seed_validation', accepted_seeds=accepted,
                   rejected_seeds=rejected, minimum_local_detail_ratio=0.7,
                   version=VERSION)

    def apply(self, labels, textured):
        result, guarded, changed = cpp.neutral_apply(labels, self.owner, self.best,
                                                    self.chroma, textured)
        diagnostic("fast_neutral_contour", guarded_pixels=guarded,
                   changed_pixels=changed, version=VERSION)
        return result


def _restore_valid_sources(labels, fallback, geometry):
    """Propagation cannot select reflected pixels beyond a real photograph."""
    result = labels.copy()
    invalid = np.zeros(labels.shape, bool)
    for local in np.unique(labels):
        source_shape, matrix = geometry[int(local)]
        valid = _warp_valid(source_shape, matrix, labels.shape)
        valid = cv2.erode(np.uint8(valid), np.ones((7, 7), np.uint8)) != 0
        invalid |= (labels == local) & ~valid
    result[invalid] = fallback[invalid]
    diagnostic("fast_final_valid_source", restored_pixels=int(np.count_nonzero(invalid)),
               fallback="best-actual-valid-proxy-source", version=VERSION)
    return result


def _check_cancel(event):
    if event is not None and event.is_set():
        raise RuntimeError("Fast fusion cancelled")


def scaled_matrix(matrix, source_from, source_to, target_from, target_to):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Fast fusion requires a finite 3x3 transform")
    sfh, sfw = source_from
    sh, sw = source_to
    tfh, tfw = target_from
    th, tw = target_to
    if min(sfh, sfw, sh, sw, tfh, tfw, th, tw) <= 0:
        raise ValueError("Coordinate dimensions must be positive")
    return np.diag([tw / tfw, th / tfh, 1.0]) @ matrix @ np.diag([sfw / sw, sfh / sh, 1.0])


def _read_gray_image(encoded, original_shape, proxy_long_edge):
    # Reduced JPEG decoding avoids constructing full-resolution proxy pixels.
    ratio = max(original_shape) / proxy_long_edge
    reduction = 8 if ratio >= 8 else 4 if ratio >= 4 else 2 if ratio >= 2 else 1
    flags = {1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2,
             4: cv2.IMREAD_REDUCED_GRAYSCALE_4, 8: cv2.IMREAD_REDUCED_GRAYSCALE_8}[reduction]
    gray = cv2.imdecode(np.frombuffer(encoded, np.uint8), flags | cv2.IMREAD_IGNORE_ORIENTATION)
    if gray is None:
        raise ValueError("Unable to decode proxy image")
    scale = min(1.0, proxy_long_edge / max(original_shape))
    size = (max(1, round(original_shape[1] * scale)), max(1, round(original_shape[0] * scale)))
    if gray.shape[::-1] != size:
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return gray


def _read_rgb_image(encoded):
    bgr = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise ValueError("Unable to decode full-resolution image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_proxy_rgb_image(encoded, original_shape, long_edge=800):
    ratio = max(original_shape) / long_edge
    reduction = 8 if ratio >= 8 else 4 if ratio >= 4 else 2 if ratio >= 2 else 1
    flags = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
             4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[reduction]
    bgr = cv2.imdecode(np.frombuffer(encoded, np.uint8), flags | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise ValueError("Unable to decode tone proxy")
    scale = min(1.0, long_edge / max(original_shape))
    size = (max(1, round(original_shape[1] * scale)), max(1, round(original_shape[0] * scale)))
    if (bgr.shape[1], bgr.shape[0]) != size:
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _warp_valid(shape, matrix, output_shape):
    return cv2.warpPerspective(np.ones(shape, np.uint8), matrix, output_shape[::-1],
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) != 0


def _detail_noise_floor(detail):
    noise = float(np.percentile(detail[::4, ::4], 20))
    return max(2e-5, min(8 * noise, 2e-3))


def _independent_detail_evidence(grayf, detail, gradient=None):
    """Distinguish local fine structure from a smooth defocus wing.

    Curvature amplitude alone is not enough: a wide blurred step also has a
    second derivative. Real fine structure either loses energy at a coarser
    scale or has rapid directional changes relative to its first derivative.
    Both tests are needed for faint engraving and isolated sharp rims.
    """
    if gradient is None:
        gx = cv2.Sobel(grayf, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
        gy = cv2.Sobel(grayf, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
        gradient = cv2.boxFilter(gx * gx + gy * gy, -1, (3, 3))
    broad = cv2.GaussianBlur(grayf, (0, 0), 1.6)
    lap = cv2.Laplacian(broad, cv2.CV_32F, ksize=3)
    broad_detail = cv2.boxFilter(lap * lap, -1, (3, 3))
    return cpp.independent(detail, broad_detail, gradient, _detail_noise_floor(detail))


def _preserve_independent_detail(labels, propagated_owner, propagated_detail,
                                 local_best, local_owner, active):
    """A nearby strong edge cannot consume a sharper target surface.

    This includes straight engraving and weak neutral rims: neither needs a
    two-direction texture response. Flat/noise-only targets retain propagation.
    Printing is applied afterwards, so defocus wings cannot reopen ink holes.
    """
    return cpp.preserve_detail(labels, propagated_owner, propagated_detail,
                               local_best, local_owner, active)


def _proxy_focus_features(grayf, *, with_gradient=False):
    """Separate broad contrast, sharp-edge evidence and two-direction texture.

    V1's first derivative can prefer a defocus wing at a location where the
    sharp photograph is flat. A second-derivative term ranks the physical
    edge before its ownership is extended to that wing. Texture is identified
    independently so a nearby stronger edge cannot consume real grain.
    """
    dx = cv2.Sobel(grayf, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    dy = cv2.Sobel(grayf, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    xx = cv2.boxFilter(dx * dx, -1, (7, 7), normalize=True)
    yy = cv2.boxFilter(dy * dy, -1, (7, 7), normalize=True)
    xy = cv2.boxFilter(dx * dy, -1, (7, 7), normalize=True)
    energy = dx * dx + dy * dy
    fine_gradient = cv2.boxFilter(energy, -1, (3, 3), normalize=True)
    mean = cv2.boxFilter(grayf, -1, (5, 5), normalize=True)
    variance = cv2.boxFilter(grayf * grayf, -1, (5, 5), normalize=True) - mean * mean
    signal = cv2.GaussianBlur(grayf, (0, 0), 0.6)
    lap = cv2.Laplacian(signal, cv2.CV_32F, ksize=3)
    detail = cv2.boxFilter(lap * lap, -1, (3, 3), normalize=True)
    base, sharp, texture = cpp.focus_fields(xx, yy, xy, fine_gradient, variance,
                                           detail, _detail_noise_floor(detail))
    result = (base, detail, sharp, texture)
    return result + (fine_gradient,) if with_gradient else result


@timed("fast_proxy_depth_map")
def compute_proxy_depth_map(paths, indices, transforms, analysis_shapes,
                            reference_analysis_shape, reference_index, full_shape,
                            *, proxy_long_edge=1600, cancel_event=None, encoded_sources=None,
                            tone_model=None, render_context=None):
    """Labels are local selected-frame indices, never original sequence indices."""
    from PIL import Image
    from io import BytesIO
    from .fast_ownership import PrintedEdgeOwnership
    from .fast_boundary_ownership import OpenCVBoundaryOwnership
    if encoded_sources is None:
        encoded_sources = {}
    scale = min(1.0, proxy_long_edge / max(full_shape))
    shape = (max(1, round(full_shape[0] * scale)), max(1, round(full_shape[1] * scale)))
    printed = PrintedEdgeOwnership(support_radius=max(1, min(128, round(128 * scale))),
                                   source_scale=scale, maximum_scale=1.0)
    boundary = OpenCVBoundaryOwnership(support_radius=max(1, round(64 * scale)), maximum_scale=1.0)
    neutral_contour = _NeutralContourOwnership(shape, reference_index, max(2, round(64 * scale)))
    source_geometry = {}
    reference_paint = None
    reference_silhouette = None
    best = np.full(shape, -1.0, np.float32)
    labels = np.full(shape, reference_index, np.uint16)
    edge_best = np.zeros(shape, np.float32)
    edge_owner = labels.copy()
    edge_target_detail = np.zeros(shape, np.float32)
    local_detail_best = np.zeros(shape, np.float32)
    local_detail_owner = labels.copy()
    local_detail_chroma = np.full(shape, 255, np.uint8)
    local_detail_foreground = np.zeros(shape, bool)
    detail_best = np.zeros(shape, np.float32)
    detail_owner = labels.copy()
    textured = np.zeros(shape, bool)
    # Radius specified in output pixels, so the same physical support survives
    # proxy-resolution changes. This extends labels, never the RGB blur kernel.
    support_radius = max(2, round(64 * scale))
    support_kernel = np.ones((2 * support_radius + 1,) * 2, np.uint8)
    reference_valid = None
    bytes_read = 0
    # Reference-first makes exact ties deterministic and preserves flat material.
    order = [reference_index] + [i for i in range(len(paths)) if i != reference_index]
    for local in order:
        _check_cancel(cancel_event)
        index, path = indices[local], Path(paths[local])
        with stage("fast_compressed_read", frame=local):
            encoded = path.read_bytes()
            encoded_sources[local] = encoded
            bytes_read += len(encoded)
        with Image.open(BytesIO(encoded)) as header:
            source_shape = (header.height, header.width)
        with stage("fast_proxy_decode", frame=local):
            gray = _read_gray_image(encoded, source_shape, proxy_long_edge)
        matrix = scaled_matrix(transforms[index], analysis_shapes[index], gray.shape,
                               reference_analysis_shape, shape)
        source_geometry[local] = (gray.shape, matrix.copy())
        with stage("fast_proxy_score", frame=local):
            aligned = cv2.warpPerspective(gray, matrix, shape[::-1], flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REFLECT_101)
            valid = _warp_valid(gray.shape, matrix, shape)
            # Keep Sobel/box neighborhoods inside the actual source footprint.
            valid = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)) != 0
            if local == reference_index:
                reference_valid = valid.copy()
            grayf = aligned.astype(np.float32) / 255.0
            score, detail, sharp, texture, gradient = _proxy_focus_features(grayf, with_gradient=True)
            local_better = cpp.proxy_winners(
                valid, _independent_detail_evidence(grayf, detail, gradient), texture,
                score, detail, local, best, labels, detail_best, detail_owner,
                local_detail_best, local_detail_owner, textured)
            # Low-chroma background noise can pass a multi-frame detail max.
            # Independent neutral material additionally needs foreground
            # contrast, as in the existing neutral contour detector. Estimate
            # each photograph's backdrop from its real border samples.
            border_gray = np.concatenate((aligned[0, ::4], aligned[-1, ::4],
                                          aligned[::4, 0], aligned[::4, -1]))
            border_valid = np.concatenate((valid[0, ::4], valid[-1, ::4],
                                           valid[::4, 0], valid[::4, -1]))
            backdrop_samples = border_gray[border_valid]
            backdrop = (float(np.percentile(backdrop_samples, 75))
                        if backdrop_samples.size >= 16 else
                        float(np.percentile(aligned[valid], 90)) if np.any(valid) else 255.0)
            local_detail_foreground[local_better] = aligned[local_better] < backdrop - 8
            sharp[~valid] = 0
            supported = cv2.dilate(sharp, support_kernel)
            edge_better = valid & (supported > edge_best)
            edge_best[edge_better] = supported[edge_better]
            edge_owner[edge_better] = local
            edge_target_detail[edge_better] = detail[edge_better]
        with stage("fast_printed_proxy", frame=local):
            colour = _read_proxy_rgb_image(encoded, source_shape, long_edge=proxy_long_edge)
            colour = cv2.warpPerspective(colour, matrix, shape[::-1], flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT_101)
            # Material evidence must come from the locally sharp source itself:
            # a defocused coloured silhouette can tint adjacent bright metal.
            colour_chroma = cpp.chroma(colour)
            local_detail_chroma[local_better] = colour_chroma[local_better]
            neutral_contour.observe(local, colour_chroma, aligned, sharp, detail, gradient, texture, valid)
            # Reuse the main project's supported-ink seeds and nearest sharp
            # seed propagation. This runs on the proxy, not full RGB frames.
            printed.observe(local, colour, aligned, np.where(valid, detail, 0))
            # Use real silhouette seeds, not a nearby letter or defocus-wing
            # texture. The existing guard measures grain away from colour
            # transitions and retains independently focused adjacent material.
            boundary.observe(local, colour, np.where(valid, detail, 0), valid=valid)
            if local == reference_index:
                from .focus_masks import _filled_chromatic_silhouette
                reference_silhouette = _filled_chromatic_silhouette(np.uint8(colour_chroma >= 65))
                # The generic ink guard erodes the whole coloured silhouette.
                # Near clipped lettering this can exclude genuine painted
                # fringe pixels along with the external background. Retain
                # conservative reference-material evidence for that narrow rim.
                small = cv2.resize(colour, printed.inside.shape[::-1], interpolation=cv2.INTER_AREA)
                chroma = small.max(axis=2).astype(np.int16) - small.min(axis=2)
                luma = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
                reference_paint = cv2.erode(np.uint8((chroma >= 90) & (luma >= 30)),
                                            np.ones((3, 3), np.uint8)) != 0
        if tone_model is not None:
            with stage("fast_tone_proxy", frame=local):
                tone_rgb = _read_proxy_rgb_image(encoded, source_shape)
                tone_scale = min(1.0, 800 / max(full_shape))
                tone_shape = (max(1, round(full_shape[0] * tone_scale)),
                              max(1, round(full_shape[1] * tone_scale)))
                tone_matrix = scaled_matrix(transforms[index], analysis_shapes[index], tone_rgb.shape[:2],
                                            reference_analysis_shape, tone_shape)
                tone_rgb = cv2.warpPerspective(tone_rgb, tone_matrix, tone_shape[::-1],
                                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                tone_model.observe_proxy(local, tone_rgb)
    if tone_model is not None:
        tone_model._finish_probe()
    _check_cancel(cancel_event)
    usable = best[best >= 0]
    noise = float(np.percentile(usable, 15)) if usable.size else 0.0
    threshold = max(1.0e-6, min(2.0e-5, 3.0 * noise))
    # Avoid propagating the background lock across a nearby physical edge.
    edge = cv2.dilate((best > threshold * 4).astype(np.uint8), np.ones((7, 7), np.uint8)) != 0
    flat = (best >= 0) & (best < threshold) & ~edge & reference_valid
    labels[flat] = reference_index
    # Protect genuine grain at its own coordinates; one-pixel expansion covers
    # a texture cell without exempting an entire printed-edge fringe.
    textured = cv2.dilate(textured.astype(np.uint8), np.ones((3, 3), np.uint8)) != 0
    active = (edge_best > 1e-6) & ~textured
    before_support = labels.copy()
    labels, independent_detail = _preserve_independent_detail(
        labels, edge_owner, edge_target_detail, local_detail_best, local_detail_owner, active)
    labels[textured] = detail_owner[textured]
    diagnostic("fast_independent_target_detail", preserved_pixels=int(np.count_nonzero(independent_detail)),
               focus_ratio=0.7, noise_floor="per-frame-proxy-detail-20-percentile",
               version=VERSION)
    before_boundary = labels
    # The inherited neutral accumulator accepts a grey/white defocus wing in
    # any frame. That can veto the sharp painted silhouette at an otherwise
    # coloured target. Fast's protection below instead requires independently
    # sharp evidence and material colour from that very source, including
    # bright metal omitted by the inherited luma ceiling. Let it replace this
    # older guard, while retaining the boundary's genuine grain protection.
    boundary.neutral_focus.fill(0)
    labels = boundary.apply(labels)
    # The silhouette guard's neutral test omits very bright surfaces. Retain
    # independently qualified detail on an adjacent low-chroma material at its
    # own focus plane. Use the same chroma threshold as the silhouette seeds;
    # coloured defocus wings still receive boundary ownership. Printing is
    # applied last so this cannot reopen pale holes beside lettering.
    adjacent_detail = ((local_detail_best > 0)
                       & (before_boundary == local_detail_owner)
                       & (local_detail_chroma < 65))
    changed_adjacent = adjacent_detail & (labels != before_boundary)
    labels[adjacent_detail] = before_boundary[adjacent_detail]
    diagnostic("fast_adjacent_material_detail",
               restored_pixels=int(np.count_nonzero(changed_adjacent)),
               material_evidence="independently-sharp-source-chroma", version=VERSION)
    neutral_material = _neutral_material_region(
        local_detail_best, local_detail_chroma, local_detail_foreground)
    neutral_contour.finalize(local_detail_best, local_detail_chroma,
                             local_detail_foreground, neutral_material)
    sharp_coloured = (local_detail_best > 0) & (local_detail_chroma >= 65)
    # A source becoming pale when defocused is not evidence of grey metal.
    # Establish the material from independently focused sources first, then
    # allow the weak neutral contour to select that material's own focal plane.
    neutral_allowed = neutral_material & ~sharp_coloured
    labels = neutral_contour.apply(labels, textured | adjacent_detail | ~neutral_allowed)
    diagnostic('fast_neutral_material_gate',
               allowed_pixels=int(np.count_nonzero(neutral_allowed)),
               material_pixels=int(np.count_nonzero(neutral_material)),
               rule='coherent_independently_sharp_neutral_foreground', version=VERSION)
    # Curved defocus wings also have a two-direction structure tensor. V4
    # incorrectly protected these small islands and punched holes in the
    # supported-ink coverage. Keep the ink/fringe source continuous; texture
    # outside its established support retains the normal detail winner above.
    clearance = max(1, int(np.ceil(8 * printed.scale)))
    near_inside = cv2.dilate(printed.inside.astype(np.uint8),
                            np.ones((2 * clearance + 1,) * 2, np.uint8)) != 0
    restored_rim = near_inside & reference_paint & ~printed.inside
    printed.inside |= restored_rim
    labels = printed.apply(labels)
    labels = _restore_valid_sources(labels, before_support, source_geometry)
    diagnostic("fast_printed_coverage", texture_veto=False,
               restored_reference_material_rim=int(np.count_nonzero(restored_rim)),
               rule="continuous_supported_ink_and_fringe", version=VERSION)
    diagnostic("fast_sharp_edge_support", support_radius_proxy=support_radius,
               support_radius_output=64, supported_pixels=int(np.count_nonzero(active)),
               changed_pixels=int(np.count_nonzero(before_support != labels)),
               protected_texture_pixels=int(np.count_nonzero(textured)),
               version=VERSION)
    diagnostic("fast_proxy_configuration", version=VERSION, proxy_shape=list(shape),
               frame_count=len(paths), compressed_bytes=bytes_read, file_reads=len(paths),
               flat_lock_pixels=int(np.count_nonzero(flat)), flat_threshold=threshold,
               uncovered_proxy_pixels=int(np.count_nonzero(best < 0)),
               scoring="Sobel energy box3/box7 + local variance box5")
    if render_context is not None:
        # Only exposed chromatic contours enter native refinement. Adjacent
        # independently focused neutral objects and supported printing keep
        # their V12 owners, including the bright rims of metal teeth.
        contour = cv2.morphologyEx(reference_silhouette, cv2.MORPH_GRADIENT,
                                   np.ones((3, 3), np.uint8))
        radius = max(1, round(32 * scale))
        target = cv2.dilate(contour, np.ones((2 * radius + 1,) * 2, np.uint8)) != 0
        protected = neutral_material
        # ``inside`` is the union of whole coloured silhouettes, NOT an ink
        # mask. Defocus expands that union across the very rim being repaired.
        # Match the actual supported-print coverage used by printed.apply.
        ink_radius = max(1, int(np.ceil(printed.support_radius * printed.scale)))
        ink_local_best = cv2.dilate(printed.best, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * ink_radius + 1,) * 2))
        ink_seeds = (printed.best > 1e-8) & (printed.best >= 0.5 * ink_local_best)
        if np.any(ink_seeds):
            ink_distance, _ = cv2.distanceTransformWithLabels(
                np.uint8(~ink_seeds), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            ink_small = (ink_distance <= ink_radius) & printed.inside
        else:
            ink_small = np.zeros(printed.inside.shape, bool)
        ink = cv2.resize(np.uint8(ink_small), shape[::-1], interpolation=cv2.INTER_NEAREST) != 0
        render_context['native_boundary_mask'] = target & ~protected & ~ink
        diagnostic('fast_native_boundary_mask', target_pixels=int(np.count_nonzero(target)),
                   neutral_excluded_pixels=int(np.count_nonzero(target & protected)),
                   printing_excluded_pixels=int(np.count_nonzero(target & ink)),
                   enabled_proxy_pixels=int(np.count_nonzero(render_context['native_boundary_mask'])),
                   printing_mask='actual-supported-ink-coverage', version=VERSION)
    return labels, shape


@timed("fast_streaming_render")
def render_fast_stream(paths, indices, transforms, analysis_shapes,
                       reference_analysis_shape, reference_index, full_shape, proxy_labels,
                       *, encoded_sources, seam_radius=2, cancel_event=None, tone_model=None, render_context=None):
    if seam_radius not in (1, 2, 3):
        raise ValueError("seam_radius must be 1, 2 or 3")
    h, w = full_shape
    if proxy_labels.min() < 0 or proxy_labels.max() >= len(paths):
        raise ValueError("Proxy label has no selected source")
    labels = cv2.resize(proxy_labels, (w, h), interpolation=cv2.INTER_NEAREST)
    native = None
    if render_context is not None and 'native_boundary_mask' in render_context:
        from .fast_native_boundary import NativeBoundary
        native = NativeBoundary(render_context['native_boundary_mask'], labels, reference_index)
    boundary = np.zeros(full_shape, np.uint8)
    dx = labels[:, 1:] != labels[:, :-1]
    dy = labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= dx
    boundary[:, :-1] |= dx
    boundary[1:, :] |= dy
    boundary[:-1, :] |= dy
    radius_kernel = np.ones((2 * seam_radius + 1, 2 * seam_radius + 1), np.uint8)
    band = cv2.dilate(boundary, radius_kernel) != 0
    with cpp.RenderAccumulator(labels, band) as accumulator:
        seam_count = accumulator.seam_count
        output = None
        decoded = 0
        fallback = 0
        order = [reference_index] + [i for i in range(len(paths)) if i != reference_index]
        for local in order:
            _check_cancel(cancel_event)
            owner, owner_count = accumulator.owner(local)
            encoded = encoded_sources.pop(local)
            if not owner_count and local != reference_index and (native is None or not native.tiles):
                continue
            index = indices[local]
            with stage("fast_full_decode", frame=local):
                source = _read_rgb_image(encoded)
            del encoded
            decoded += 1
            matrix = scaled_matrix(transforms[index], analysis_shapes[index], source.shape[:2],
                                   reference_analysis_shape, full_shape)
            with stage("fast_full_warp", frame=local):
                aligned = cv2.warpPerspective(source, matrix, (w, h), flags=cv2.INTER_CUBIC,
                                              borderMode=cv2.BORDER_REFLECT_101)
                valid = _warp_valid(source.shape[:2], matrix, full_shape)
            del source
            if native is not None:
                with stage('fast_native_boundary_rank', frame=local):
                    native.rank(aligned, valid, local)
            if tone_model is not None:
                _check_cancel(cancel_event)
                with stage("fast_paired_tone", frame=local):
                    aligned = tone_model.correct(aligned, local)
            if native is not None:
                native.capture_corrected(aligned)
            if output is None:
                output = aligned.copy()  # Explicit reference fallback, including true black pixels.
            with stage("fast_copy_and_seam", frame=local):
                feather = None
                if seam_count and owner_count:
                    feather = cv2.GaussianBlur(owner.astype(np.float32),
                                              (2 * seam_radius + 1, 2 * seam_radius + 1),
                                              sigmaX=max(0.6, seam_radius / 2))
                accumulator.add(local, aligned, valid, feather, output)
                del feather
            del aligned, valid
        _check_cancel(cancel_event)
        with stage("fast_finish_seams"):
            fallback = accumulator.finish(output)
    if native is not None:
        with stage('fast_native_boundary_finish'):
            output = native.finish(output)
    diagnostic("fast_render_configuration", seam_radius=seam_radius,
               feather_kernel=2 * seam_radius + 1, seam_pixels=seam_count,
               seam_fraction=seam_count / (h * w), reference_fallback_pixels=fallback,
               full_rgb_decodes=decoded, full_rgb_warps=decoded,
               validity_mask_warps=decoded, interpolation="INTER_CUBIC",
               tone_correction=tone_model is not None, full_resolution_focus_refinement=native is not None)
    diagnostic('fast_cpp_runtime', **cpp.runtime_info())
    return output
