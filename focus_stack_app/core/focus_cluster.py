"""Near-identical focus detection and quality-based deduplication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .focus_map import _focus_support_mask, decompress_focus_map, focus_map_similarity
from .types import DeduplicationResult, FocusCluster


@dataclass(slots=True)
class FocusClusterConfig:
    """Thresholds for grouping maps that cover essentially the same focus."""

    similarity_threshold: float = 0.92
    normalized_difference_threshold: float = 0.16
    top_percentile: float = 70.0
    min_spatial_overlap: float = 0.60


def _setting(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    value = getattr(config, name, None)
    if value is None and name == "similarity_threshold":
        value = getattr(config, "duplicate_focus_threshold", None)
    if value is None:
        section = getattr(config, "analysis", None)
        if section is not None:
            value = getattr(section, name, None)
            if value is None and name == "similarity_threshold":
                value = getattr(section, "duplicate_focus_threshold", None)
    return default if value is None else value


def _np() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - minimal env only
        raise RuntimeError("focus clustering requires NumPy") from exc
    return np


def _quality_value(value: Any, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    if hasattr(value, "score"):
        value = value.score
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _resize_pair(a: Any, b: Any) -> tuple[Any, Any]:
    np = _np()
    a = decompress_focus_map(a)
    b = decompress_focus_map(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("focus maps must be 2-D")
    if a.shape == b.shape:
        return a, b
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        ys = np.minimum((np.arange(a.shape[0]) * b.shape[0] / a.shape[0]).astype(int), b.shape[0] - 1)
        xs = np.minimum((np.arange(a.shape[1]) * b.shape[1] / a.shape[1]).astype(int), b.shape[1] - 1)
        b = b[ys[:, None], xs[None, :]]
    return a, b


def focus_map_distance(map_a: Any, map_b: Any,
                       config: Any = None) -> float:
    """Return a duplicate-focus distance in [0, 1] (zero means identical)."""

    np = _np()
    a, b = _resize_pair(map_a, map_b)
    diff = float(np.mean(np.abs(a - b)))
    # focus_map_similarity includes correlation and top-region overlap.  Use
    # the two signals together so flat maps do not produce NaNs or false
    # negatives.
    similarity = float(focus_map_similarity(a, b))
    return float(np.clip(0.60 * (1.0 - similarity) + 0.40 * min(1.0, diff), 0.0, 1.0))


def _comparison_metrics(map_a: Any, map_b: Any, config: Any = None) -> tuple[float, float, float]:
    np = _np()
    a, b = _resize_pair(map_a, map_b)
    similarity = float(focus_map_similarity(a, b))
    difference = float(np.mean(np.abs(a - b)))
    q = float(_setting(config, "top_percentile", 70.0))
    mask_a = _focus_support_mask(a, q)
    mask_b = _focus_support_mask(b, q)
    union = int(np.count_nonzero(mask_a | mask_b))
    overlap = 1.0 if union == 0 else float(np.count_nonzero(mask_a & mask_b) / union)
    return similarity, difference, overlap


def maps_represent_same_focus(map_a: Any, map_b: Any,
                              config: Any = None) -> bool:
    """Apply similarity, difference and spatial-overlap duplicate criteria."""

    config = config or FocusClusterConfig()
    similarity, difference, overlap = _comparison_metrics(map_a, map_b, config)
    threshold = float(_setting(config, "similarity_threshold", 0.92))
    diff_threshold = float(_setting(config, "normalized_difference_threshold", 0.16))
    min_overlap = float(_setting(config, "min_spatial_overlap", 0.60))
    # Similar maps with slight exposure/noise changes pass all three.  For
    # unusually flat maps overlap is uninformative, so similarity + diff is
    # sufficient when both maps have almost no variance.
    np = _np()
    a, b = _resize_pair(map_a, map_b)
    flat = float(np.std(a) + np.std(b)) < 0.035
    if flat:
        return similarity >= threshold and difference <= diff_threshold
    return (similarity >= threshold and difference <= diff_threshold
            and overlap >= min_overlap)


def cluster_focus_maps(focus_maps: Sequence[Any],
                       quality_scores: Sequence[Any] | None = None,
                       config: Any = None) -> list[FocusCluster]:
    """Cluster focus maps incrementally without retaining full source images.

    Input maps are expected to be compressed maps (roughly 400--600 px).  A
    map is compared with one compact representative per existing cluster;
    this keeps the common case close to O(N * K) and avoids an N-by-N image
    comparison.  Cluster representatives are replaced by the best-quality
    member after assignment, so duplicate selection is deterministic.
    """

    # Permit the convenient ``cluster_focus_maps(maps, config)`` spelling in
    # addition to the explicit quality_scores/config form.
    if config is None and quality_scores is not None and (
        hasattr(quality_scores, "similarity_threshold")
        or hasattr(quality_scores, "duplicate_focus_threshold")
    ):
        config = quality_scores
        quality_scores = None
    maps = list(focus_maps)
    if not maps:
        return []
    config = config or FocusClusterConfig()
    qualities = list(quality_scores) if quality_scores is not None else [None] * len(maps)
    if len(qualities) != len(maps):
        raise ValueError("quality_scores length must match focus_maps")
    # A cluster record is [member indices, comparison representative index].
    working: list[dict[str, Any]] = []
    for index, fmap in enumerate(maps):
        best_cluster = None
        for cluster in working:
            rep_index = int(cluster["representative"])
            if maps_represent_same_focus(fmap, maps[rep_index], config):
                best_cluster = cluster
                break
        if best_cluster is None:
            working.append({"members": [index], "representative": index})
        else:
            best_cluster["members"].append(index)
            current = int(best_cluster["representative"])
            current_q = _quality_value(qualities[current], fallback=_map_quality(maps[current]))
            candidate_q = _quality_value(qualities[index], fallback=_map_quality(fmap))
            if candidate_q > current_q:
                best_cluster["representative"] = index
    result: list[FocusCluster] = []
    for cluster_id, cluster in enumerate(working):
        members = tuple(int(v) for v in cluster["members"])
        rep = int(cluster["representative"])
        rep_quality = _quality_value(qualities[rep], fallback=_map_quality(maps[rep]))
        similarity = {
            member: float(focus_map_similarity(maps[member], maps[rep]))
            for member in members
        }
        result.append(FocusCluster(
            cluster_id=cluster_id,
            members=members,
            representative_index=rep,
            representative_quality=rep_quality,
            similarity_to_representative=similarity,
        ))
    return result


def _map_quality(focus_map: Any) -> float:
    np = _np()
    values = decompress_focus_map(focus_map)
    if values.size == 0:
        return 0.0
    return float(np.clip(0.65 * np.mean(values) + 0.35 * np.percentile(values, 90), 0.0, 1.0))


def deduplicate_focus_maps(focus_maps: Sequence[Any],
                           quality_scores: Sequence[Any] | None = None,
                           config: Any = None) -> DeduplicationResult:
    """Keep the highest-quality image from each near-identical-focus cluster."""

    clusters = cluster_focus_maps(focus_maps, quality_scores, config)
    representative_indices = [cluster.representative_index for cluster in clusters]
    representatives = set(representative_indices)
    rejected: list[int] = []
    reasons: dict[int, str] = {}
    for cluster in clusters:
        for index in cluster.members:
            if index == cluster.representative_index:
                reasons[index] = "best quality for focus cluster"
            else:
                rejected.append(index)
                similarity = cluster.similarity_to_representative.get(index, 0.0)
                reasons[index] = (
                    f"redundant focus (similarity {similarity:.3f}); "
                    f"kept image {cluster.representative_index} with higher quality"
                )
    # Stable ordering is important for manifest/database consumers.
    representative_indices.sort()
    rejected.sort()
    return DeduplicationResult(
        clusters=clusters,
        representative_indices=representative_indices,
        rejected_indices=rejected,
        reasons=reasons,
    )


def select_best_per_focus(focus_maps: Sequence[Any],
                          quality_scores: Sequence[Any] | None = None,
                          config: Any = None) -> DeduplicationResult:
    return deduplicate_focus_maps(focus_maps, quality_scores, config)


def needs_focus_stack(focus_maps: Sequence[Any], config: Any = None) -> bool:
    """Whether a group contains more than one materially distinct focus."""

    return len(focus_maps) > 1 and len(cluster_focus_maps(focus_maps, config=config)) > 1


def focus_similarity(map_a: Any, map_b: Any) -> float:
    return float(focus_map_similarity(map_a, map_b))


DuplicateFocusConfig = FocusClusterConfig
cluster_focus = cluster_focus_maps


__all__ = [
    "FocusClusterConfig", "DuplicateFocusConfig", "FocusCluster", "DeduplicationResult",
    "focus_map_distance", "focus_similarity", "maps_represent_same_focus",
    "cluster_focus_maps", "cluster_focus", "deduplicate_focus_maps", "select_best_per_focus",
    "needs_focus_stack",
]

