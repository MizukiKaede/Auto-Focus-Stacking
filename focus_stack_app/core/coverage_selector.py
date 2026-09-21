"""Greedy maximum-coverage focus-image selection.

Selection works only on compressed focus maps.  It is deliberately independent
of capture/focus order: each candidate contributes the spatial regions in
which it is locally sharp, and a greedy marginal-gain rule stops once the
configured coverage target (or minimum gain floor) is reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .focus_map import decompress_focus_map, focus_map_similarity
from .types import CoverageSelection


@dataclass(slots=True)
class CoverageConfig:
    """Coverage and stopping parameters.

    ``mode`` accepts ``sparse``/``简洁`` (98%), ``balanced``/``平衡`` (99.5%)
    and ``maximum``/``extreme`` (99.95%).  An explicit ``coverage_target`` takes
    precedence over the preset.
    """

    mode: str = "balanced"
    coverage_target: float | None = None
    min_coverage_gain: float = 0.002
    # A pixel only counts as covered when this frame is very close to the
    # strongest response available for that pixel across the whole bracket.
    # The old 0.45 default let a broadly half-sharp frame cover an entire
    # scene, which dropped the intermediate focus planes needed by Enfuse.
    focus_threshold: float = 0.95
    support_threshold: float = 0.15
    relative_focus_threshold: bool = True
    quality_tiebreak_weight: float = 0.01
    max_selected: int | None = None
    transition_reorder: bool = True


_TARGETS = {
    "sparse": 0.98,
    "compact": 0.98,
    "minimal": 0.98,
    "精简": 0.98,
    "balanced": 0.995,
    "balance": 0.995,
    "平衡": 0.995,
    "extreme": 0.9995,
    "maximum": 0.9995,
    "极致": 0.9995,
}


def _setting(config: Any, name: str, default: Any) -> Any:
    value = getattr(config, name, default) if config is not None else default
    if config is not None and (value is None or value == default):
        section = getattr(config, "analysis", None)
        if section is not None:
            nested = getattr(section, name, None)
            if nested is not None:
                value = nested
    return default if value is None else value


def _np() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - minimal env only
        raise RuntimeError("coverage selection requires NumPy") from exc
    return np


def _target(config: Any) -> float:
    explicit = _setting(config, "coverage_target", None)
    if explicit is None:
        # A few callers use ``target_coverage``; support it without coupling
        # this module to the main config class.
        explicit = _setting(config, "target_coverage", None)
    if explicit is None:
        mode = str(_setting(config, "mode", "balanced")).strip().lower()
        explicit = _TARGETS.get(mode, 0.995)
    return float(max(0.0, min(1.0, float(explicit))))


def _resize_maps(maps: Sequence[Any]) -> list[Any]:
    np = _np()
    if not maps:
        return []
    # Keep the resident candidate set compact.  A 512px float16 map is about
    # 0.5 MiB; expanding a thousand maps to float32 would need roughly 1 GiB.
    def compact(value: Any) -> Any:
        arr = np.asarray(value)
        if arr.dtype == np.uint16:
            return (arr.astype(np.float32) / 65535.0).astype(np.float16)
        return np.clip(arr.astype(np.float16, copy=False), 0.0, 1.0)

    decoded = [compact(m) for m in maps]
    for arr in decoded:
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError("focus maps must be non-empty 2-D arrays")
    shape = decoded[0].shape
    if all(arr.shape == shape for arr in decoded[1:]):
        return decoded
    try:
        import cv2
    except ImportError:
        cv2 = None
    result = [decoded[0]]
    for arr in decoded[1:]:
        if arr.shape == shape:
            result.append(arr)
        elif cv2 is not None:
            result.append(cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_AREA).astype(np.float16))
        else:
            ys = np.minimum((np.arange(shape[0]) * arr.shape[0] / shape[0]).astype(int), arr.shape[0] - 1)
            xs = np.minimum((np.arange(shape[1]) * arr.shape[1] / shape[1]).astype(int), arr.shape[1] - 1)
            result.append(arr[ys[:, None], xs[None, :]])
    return result


def _quality_value(value: Any, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    if hasattr(value, "score"):
        value = value.score
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def coverage_fraction(coverage_map: Any, support_mask: Any | None = None,
                      focus_threshold: float = 0.95) -> float:
    """Compute the fraction of supported pixels currently above threshold."""

    np = _np()
    values = decompress_focus_map(coverage_map)
    support = np.ones_like(values, dtype=bool) if support_mask is None else np.asarray(support_mask, dtype=bool)
    denom = int(np.count_nonzero(support))
    if denom <= 0:
        return 0.0
    return float(np.count_nonzero((values >= float(focus_threshold)) & support) / denom)


def _transition_distance(map_a: Any, map_b: Any) -> float:
    return float(1.0 - focus_map_similarity(map_a, map_b))


def order_selected_by_transition(selected_indices: Sequence[int],
                                 focus_maps: Sequence[Any],
                                 quality_scores: Sequence[Any] | None = None,
                                 start_index: int | None = None) -> list[int]:
    """Greedy nearest-neighbour ordering that minimises map transitions.

    The first selected index is kept by default, preserving the group seed
    (and therefore the first-original-image convention).  Pass
    ``start_index`` to choose another known index explicitly.
    """

    indices = list(dict.fromkeys(int(i) for i in selected_indices))
    if len(indices) <= 2:
        return indices
    if start_index is not None and int(start_index) in indices:
        current = int(start_index)
    else:
        current = indices[0]
    remaining = set(indices)
    remaining.remove(current)
    ordered = [current]
    while remaining:
        best = min(
            remaining,
            key=lambda idx: (_transition_distance(focus_maps[current], focus_maps[idx]), idx),
        )
        ordered.append(best)
        remaining.remove(best)
        current = best
    return ordered


def reorder_selected(selected_indices: Sequence[int], focus_maps: Sequence[Any],
                     quality_scores: Sequence[Any] | None = None,
                     start_index: int | None = None) -> list[int]:
    return order_selected_by_transition(selected_indices, focus_maps, quality_scores, start_index)


def select_focus_images(focus_maps: Sequence[Any],
                        quality_scores: Sequence[Any] | None = None,
                        config: Any = None,
                        candidate_indices: Sequence[int] | None = None) -> CoverageSelection:
    """Select a compact set whose union of sharp regions maximises coverage.

    ``candidate_indices`` maps a deduplicated subset back to original group
    indices.  If omitted, input positions are returned.  The algorithm keeps
    only compressed maps and a single running ``coverage_map``.
    """

    if config is None and quality_scores is not None and (
        hasattr(quality_scores, "mode")
        or hasattr(quality_scores, "coverage_target")
        or hasattr(quality_scores, "min_coverage_gain")
    ):
        config = quality_scores
        quality_scores = None
    np = _np()
    maps = _resize_maps(focus_maps)
    if not maps:
        return CoverageSelection([], [], 0.0, {}, {}, None)
    config = config or CoverageConfig()
    n = len(maps)
    qualities = list(quality_scores) if quality_scores is not None else [None] * n
    if len(qualities) != n:
        raise ValueError("quality_scores length must match focus_maps")
    if candidate_indices is None:
        original_indices = list(range(n))
    else:
        original_indices = [int(i) for i in candidate_indices]
        if len(original_indices) != n:
            raise ValueError("candidate_indices length must match focus_maps")
    focus_threshold = float(_setting(config, "focus_threshold", 0.95))
    support_threshold = float(_setting(config, "support_threshold", 0.15))
    support_reference = np.maximum.reduce(maps)
    support = support_reference >= support_threshold
    # If normalisation/low-texture input produces no support, use every pixel
    # with the maximum response as the denominator rather than returning NaN.
    if not np.any(support):
        support = support_reference > 0.0
    if not np.any(support):
        support = np.ones_like(support_reference, dtype=bool)
    support_count = max(1, int(np.count_nonzero(support)))
    target = _target(config)
    min_gain = max(0.0, float(_setting(config, "min_coverage_gain", 0.002)))
    max_selected = _setting(config, "max_selected", None)
    max_selected = n if max_selected is None else max(1, int(max_selected))
    quality_weight = max(0.0, float(_setting(config, "quality_tiebreak_weight", 0.01)))

    quality_values = [
        _quality_value(qualities[i], fallback=float(np.mean(maps[i])))
        for i in range(n)
    ]
    # A map's absolute response varies with scene texture and exposure.  By
    # default compare each response with the best response available at that
    # pixel.  Requiring a near-best response is important: a defocused frame
    # still retains low-frequency edges and must not be allowed to mark those
    # pixels as covered merely because it has some response there.
    relative = bool(_setting(config, "relative_focus_threshold", True))
    scale_reference = np.maximum(support_reference, 1e-6) if relative else 1.0

    def sharp_support(focus: Any) -> Any:
        response = focus / scale_reference if relative else focus
        return (response >= focus_threshold) & support

    def covered_support(running_map: Any) -> Any:
        response = running_map / scale_reference if relative else running_map
        return (response >= focus_threshold) & support

    # Seed: highest useful support, quality as tie-breaker, original order as
    # final tie-breaker.  This avoids selecting a noisy map that has no clear
    # area merely because its global score is high.
    support_counts = [int(np.count_nonzero(sharp_support(m))) for m in maps]
    seed = max(range(n), key=lambda i: (support_counts[i], quality_values[i], -i))
    selected_local: list[int] = [seed]
    running = maps[seed].copy().astype(np.float32)
    covered = covered_support(running)
    gains: dict[int, float] = {original_indices[seed]: float(np.count_nonzero(covered) / support_count)}
    reasons: dict[int, str] = {original_indices[seed]: "coverage seed"}
    remaining = set(range(n))
    remaining.remove(seed)
    while remaining and len(selected_local) < max_selected:
        current_coverage = int(np.count_nonzero(covered)) / support_count
        if current_coverage >= target:
            break
        best_local = None
        best_gain = -1.0
        best_value = None
        for candidate in remaining:
            candidate_map = maps[candidate]
            new_pixels = sharp_support(candidate_map) & ~covered
            gain = float(np.count_nonzero(new_pixels) / support_count)
            tie_value = gain + quality_weight * quality_values[candidate] / max(1, n)
            key = (tie_value, quality_values[candidate], -candidate)
            if best_value is None or key > best_value:
                best_value = key
                best_local = candidate
                best_gain = gain
        if best_local is None:
            break
        original = original_indices[best_local]
        gains[original] = best_gain
        if best_gain < min_gain:
            reasons[original] = f"below minimum coverage gain ({best_gain:.4f})"
            for candidate in remaining:
                if candidate != best_local:
                    idx = original_indices[candidate]
                    gains.setdefault(idx, 0.0)
                    reasons.setdefault(idx, "not needed after coverage stopped")
            break
        selected_local.append(best_local)
        reasons[original] = f"coverage gain {best_gain:.4f}"
        running = np.maximum(running, maps[best_local])
        covered = covered_support(running)
        remaining.remove(best_local)
    selected_original = [original_indices[i] for i in selected_local]
    selected_set = set(selected_local)
    final_coverage = float(np.count_nonzero(covered) / support_count)
    for local in range(n):
        original = original_indices[local]
        if local not in selected_set:
            if original not in reasons:
                gains[original] = 0.0
                reasons[original] = "redundant; no material new coverage"
    if bool(_setting(config, "transition_reorder", True)) and len(selected_local) > 1:
        ordered_local = order_selected_by_transition(selected_local, maps, quality_values)
        selected_original = [original_indices[i] for i in ordered_local]
    rejected_original = [original_indices[i] for i in range(n) if i not in selected_set]
    rejected_original.sort()
    return CoverageSelection(
        selected_indices=selected_original,
        rejected_indices=rejected_original,
        coverage=final_coverage,
        gains=gains,
        reasons=reasons,
        coverage_map=running.astype(np.float16),
    )


def greedy_maximum_coverage(focus_maps: Sequence[Any],
                            quality_scores: Sequence[Any] | None = None,
                            config: Any = None,
                            candidate_indices: Sequence[int] | None = None) -> CoverageSelection:
    return select_focus_images(focus_maps, quality_scores, config, candidate_indices)


def select_by_coverage(focus_maps: Sequence[Any],
                       quality_scores: Sequence[Any] | None = None,
                       config: Any = None,
                       candidate_indices: Sequence[int] | None = None) -> CoverageSelection:
    return select_focus_images(focus_maps, quality_scores, config, candidate_indices)


MaximumCoverageConfig = CoverageConfig
maximum_coverage_selection = select_focus_images
select_images = select_focus_images


__all__ = [
    "CoverageConfig", "MaximumCoverageConfig", "CoverageSelection", "coverage_fraction",
    "select_focus_images", "select_images", "maximum_coverage_selection",
    "greedy_maximum_coverage", "select_by_coverage",
    "order_selected_by_transition", "reorder_selected",
]
