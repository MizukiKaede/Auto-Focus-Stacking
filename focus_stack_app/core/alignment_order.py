"""Preview-reference selection and registration-friendly Hugin ordering."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Iterable, Sequence

from .registration import RegistrationResult, register_images
from .registration_cache import preprocessing_cache


ORDERING_PLAN_VERSION = "local-hubs-v2"


@dataclass(frozen=True, slots=True)
class PairwiseRegistration:
    left: int
    right: int
    valid: bool
    confidence: float
    match_quality: float
    inlier_ratio: float
    residual: float | None
    scale_difference: float
    effective_overlap: float
    transform_magnitude: float
    model: str
    message: str = ""

    @property
    def cost(self) -> float:
        residual_penalty = min(1.0, float(self.residual or 0.0) / 8.0)
        return float(max(0.0, min(2.0,
            1.0 - self.confidence
            + 0.25 * residual_penalty
            + 0.30 * self.scale_difference
            + 0.30 * (1.0 - self.effective_overlap)
            + (0.75 if not self.valid else 0.0)
        )))


def _edge_from_result(left: int, right: int, result: RegistrationResult, shape: tuple[int, int]) -> PairwiseRegistration:
    h, w = shape
    tx, ty = result.translation
    translation_fraction = math.hypot(float(tx) / max(1, w), float(ty) / max(1, h))
    scale_difference = abs(math.log(max(1e-6, float(result.scale))))
    overlap = max(0.0, min(1.0, (1.0 - min(1.0, abs(tx) / max(1, w))) * (1.0 - min(1.0, abs(ty) / max(1, h)))))
    transform_magnitude = translation_fraction + scale_difference + abs(float(result.rotation_degrees)) / 45.0
    confidence = float(result.confidence) * (0.6 + 0.4 * overlap)
    if result.reprojection_error is not None:
        confidence *= max(0.2, 1.0 - min(1.0, float(result.reprojection_error) / 10.0))
    return PairwiseRegistration(
        left, right, bool(result.valid), max(0.0, min(1.0, confidence)),
        max(0.0, min(1.0, float(result.confidence))), float(result.inlier_ratio),
        result.reprojection_error, scale_difference, overlap, transform_magnitude,
        result.model, result.message,
    )


def analyze_pairwise_registration(
    images: Sequence[Any], *, config: Any = None,
    pairs: Iterable[tuple[int, int]] | None = None,
) -> list[PairwiseRegistration]:
    """Build an undirected registration graph from compact preview arrays."""
    count = len(images)
    if count < 2:
        return []
    # Full graphs are ideal for normal brackets. Large shoots use a connected
    # sparse graph (near neighbours plus evenly-spaced hubs) to stay bounded.
    if pairs is None:
        candidate_pairs: set[tuple[int, int]] = set()
        if count <= 40:
            candidate_pairs.update((i, j) for i in range(count) for j in range(i + 1, count))
        else:
            for i in range(count):
                for j in range(i + 1, min(count, i + 5)):
                    candidate_pairs.add((i, j))
            hubs = sorted({0, count // 4, count // 2, (3 * count) // 4, count - 1})
            for hub in hubs:
                for i in range(count):
                    if i != hub:
                        candidate_pairs.add(tuple(sorted((i, hub))))
    else:
        candidate_pairs = set(pairs)
    edges: list[PairwiseRegistration] = []
    with preprocessing_cache():
        for left, right in sorted(candidate_pairs):
            try:
                result = register_images(images[left], images[right], config)
                edges.append(_edge_from_result(left, right, result, images[left].shape[:2]))
            except Exception as exc:
                edges.append(PairwiseRegistration(
                    left, right, False, 0.0, 0.0, 0.0, None, 1.0, 0.0, 2.0,
                    "failed", str(exc),
                ))
    return edges


def _ordering_pairs(count: int) -> set[tuple[int, int]]:
    """Keep local overlap evidence and sample distant links for large stacks."""
    pairs = {(i, j) for i in range(count)
             for j in range(i + 1, min(count, i + 4))}
    hubs = sorted({0, count // 2, count - 1})
    for hub in hubs:
        for i in range(0, count, 2):
            if i != hub:
                pairs.add((min(i, hub), max(i, hub)))
    for index, left in enumerate(hubs):
        for right in hubs[index + 1:]:
            pairs.add((left, right))
    return pairs


def choose_preview_reference(count: int, edges: Sequence[PairwiseRegistration]) -> tuple[int, dict[str, Any]]:
    """Choose the graph medoid with reliable, low-residual connections."""
    if count <= 1:
        return 0, {"scores": [1.0], "connected_fraction": 1.0}
    scores: list[float] = []
    for index in range(count):
        incident = [edge for edge in edges if edge.left == index or edge.right == index]
        valid = [edge for edge in incident if edge.valid]
        success = len(valid) / max(1, len(incident))
        mean_confidence = sum(edge.confidence for edge in incident) / max(1, len(incident))
        mean_overlap = sum(edge.effective_overlap for edge in incident) / max(1, len(incident))
        residuals = [edge.residual for edge in valid if edge.residual is not None]
        residual_score = 1.0 / (1.0 + (sum(residuals) / max(1, len(residuals))))
        scores.append(0.40 * success + 0.30 * mean_confidence + 0.20 * mean_overlap + 0.10 * residual_score)
    best = max(range(count), key=lambda index: (scores[index], -abs(index - (count - 1) / 2), -index))
    return best, {
        "scores": scores,
        "connected_fraction": sum(edge.valid for edge in edges) / max(1, len(edges)),
        "edge_count": len(edges),
    }


def _edge_map(edges: Sequence[PairwiseRegistration]) -> dict[tuple[int, int], PairwiseRegistration]:
    return {(min(edge.left, edge.right), max(edge.left, edge.right)): edge for edge in edges}


def _path_cost(path: Sequence[int], lookup: dict[tuple[int, int], PairwiseRegistration]) -> float:
    return sum(lookup.get((min(a, b), max(a, b)), PairwiseRegistration(a, b, False, 0, 0, 0, None, 1, 0, 2, "missing")).cost for a, b in zip(path, path[1:]))


def build_alignment_order(
    selected_images: Sequence[Any], *, config: Any = None,
    capture_order: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Order selected previews for reliable adjacent Hugin registration."""
    count = len(selected_images)
    capture = list(capture_order) if capture_order is not None else list(range(count))
    if count < 2:
        return {"alignment_order": capture, "alignment_order_confidence": 1.0,
                "alignment_order_diagnostics": {"edges": []},
                "alignment_order_fallback_used": False}
    if count <= 24:
        edges = analyze_pairwise_registration(selected_images, config=config)
    else:
        edges = analyze_pairwise_registration(
            selected_images, config=config, pairs=_ordering_pairs(count),
        )
    lookup = _edge_map(edges)
    starts = sorted({0, count // 2, count - 1})
    candidates: list[list[int]] = []
    for start in starts:
        path, remaining = [start], set(range(count)) - {start}
        while remaining:
            current = path[-1]
            nxt = min(remaining, key=lambda item: (lookup.get((min(current, item), max(current, item)), PairwiseRegistration(current, item, False, 0, 0, 0, None, 1, 0, 2, "missing")).cost, item))
            path.append(nxt)
            remaining.remove(nxt)
        candidates.extend((path, list(reversed(path))))
    best = min(candidates, key=lambda value: _path_cost(value, lookup))
    # Small 2-opt refinement; end points may change because Hugin has no
    # physical near/far focus-order requirement.
    improved = True
    while improved:
        improved = False
        for i in range(1, count - 1):
            for j in range(i + 1, count):
                candidate = best[:i] + list(reversed(best[i:j + 1])) + best[j + 1:]
                if _path_cost(candidate, lookup) + 1e-9 < _path_cost(best, lookup):
                    best, improved = candidate, True
    path_edges = [lookup.get((min(a, b), max(a, b))) for a, b in zip(best, best[1:])]
    confidence = sum(edge.confidence if edge is not None and edge.valid else 0.0 for edge in path_edges) / max(1, len(path_edges))
    fallback = confidence < 0.35
    if fallback:
        best = list(range(count))
    return {
        "alignment_order": [capture[index] for index in best],
        "alignment_order_confidence": float(confidence),
        "alignment_order_diagnostics": {
            "code": "ALIGNMENT_ORDER_UNCERTAIN" if fallback else "OK",
            "path_cost": _path_cost(best, lookup),
            "edges": [asdict(edge) for edge in edges],
        },
        "alignment_order_fallback_used": fallback,
    }


__all__ = ["PairwiseRegistration", "analyze_pairwise_registration", "choose_preview_reference", "build_alignment_order"]
