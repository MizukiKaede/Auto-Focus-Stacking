"""Full-resolution focus decisions with bounded, frame-count independent RAM."""
from __future__ import annotations

from ..utils.performance import timed

import cv2
import numpy as np


def focus_response(rgb):
    """Compare fine detail on a common scale, including its defocus fringe.

    A defocused edge can win just outside the sharp silhouette, where the
    sharp image is flat. Extend nearby focus evidence by seven source pixels
    so that this fringe belongs to the sharp edge too. Do not normalise each
    frame independently: that would promote a uniformly blurry frame.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lap = cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 0.6), cv2.CV_32F, ksize=3)
    response = cv2.GaussianBlur(lap * lap, (0, 0), 4.0)
    return cv2.dilate(response, np.ones((15, 15), np.uint8))


@timed("focus_masks")
def build_focus_labels(count, load_aligned, *, cancel_event=None, protect_chromatic_edges=False):
    """Choose focused pixels and protect nearby neutral subject silhouettes.

    A background-focused frame may retain a displaced, defocused subject edge.
    For a coloured subject with a bright neutral rim, choose the frame whose
    local rim boundary is sharpest and use it just outside the silhouette.
    """
    if not 1 <= count <= 65535:
        raise ValueError("focus fusion requires between 1 and 65535 frames")
    best = labels = silhouette_union = rim_best = rim_owner = None
    colour_union = colour_best = colour_owner = colour_luma = None
    winner_luma = None
    if protect_chromatic_edges:
        from .alignment_quality import colour_subject_mask
    for index in range(count):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("focus fusion cancelled")
        rgb = load_aligned(index)
        score = focus_response(rgb)
        if best is None:
            best = score
            labels = np.zeros(score.shape, np.uint16)
            if protect_chromatic_edges:
                winner_luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            if score.shape != best.shape:
                raise ValueError("aligned focus frames must have matching dimensions")
            better = score > best
            best[better] = score[better]
            labels[better] = index
            if protect_chromatic_edges:
                gray_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                winner_luma[better] = gray_full[better]
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
            candidate = cv2.cvtColor(load_aligned(int(index)), cv2.COLOR_RGB2GRAY)[ys, xs]
            previous = winner_luma[ys, xs]
            false_foreground = (candidate > 110) & (
                candidate.astype(np.int16) - previous.astype(np.int16) > 50)
            guarded[ys[false_foreground], xs[false_foreground]] = False
        labels[guarded] = owner[guarded]
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
