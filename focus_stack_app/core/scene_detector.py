"""Sequential, bounded-memory scene grouping for a shoot sequence.

Scene boundaries are inferred from low-frequency preview similarity and, when
available, ORB/RANSAC geometry.  Only a small number of preview-sized anchors
is retained, and a confirmation window prevents one difficult frame from
prematurely splitting an otherwise continuous scene.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from .registration import load_image, resolve_image_path, resize_for_analysis
from .types import SceneComparison, SceneFeature, SceneGroup


@dataclass(slots=True)
class SceneConfig:
    """Scene detector settings (all can be overridden by duck-typed config)."""

    scene_preview_long_edge: int = 512
    scene_similarity_threshold: float = 0.82
    geometric_inlier_threshold: float = 0.55
    geometric_min_matches: int = 8
    geometric_min_inliers: int = 6
    histogram_bins: int = 32
    hash_size: int = 8
    blur_sigma: float = 3.0
    max_anchors: int = 3
    scene_confirmation_window: int = 2
    require_geometry_for_boundary: bool = False
    orb_features: int = 800
    orb_ratio_test: float = 0.75
    ransac_reprojection_threshold: float = 3.0


def _setting(config: Any, name: str, default: Any, *aliases: str) -> Any:
    if config is None:
        return default
    names = (name,) + aliases
    for candidate in names:
        value = getattr(config, candidate, None)
        if value is not None:
            return value
    section = getattr(config, "analysis", None)
    if section is not None:
        for candidate in names:
            value = getattr(section, candidate, None)
            if value is not None:
                return value
    return default


def _np() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - minimal env only
        raise RuntimeError("scene detection requires NumPy") from exc
    return np


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - minimal env only
        raise RuntimeError("scene detection requires OpenCV") from exc
    return cv2


def _gray_u8(image: Any) -> Any:
    np = _np()
    try:
        cv2 = _cv2()
    except RuntimeError:
        cv2 = None
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if cv2 is not None else np.mean(arr[..., :3], axis=2)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("preview must be a non-empty 2-D or 3-D image")
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    if float(arr.max(initial=0.0)) <= 1.0:
        arr *= 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _resize_gray(gray: Any, shape: tuple[int, int]) -> Any:
    np = _np()
    if gray.shape == shape:
        return gray
    try:
        cv2 = _cv2()
    except RuntimeError:
        cv2 = None
    if cv2 is not None:
        return cv2.resize(gray, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    ys = np.minimum((np.arange(shape[0]) * gray.shape[0] / shape[0]).astype(int), gray.shape[0] - 1)
    xs = np.minimum((np.arange(shape[1]) * gray.shape[1] / shape[1]).astype(int), gray.shape[1] - 1)
    return gray[ys[:, None], xs[None, :]]


def _blur(gray: Any, sigma: float) -> Any:
    np = _np()
    try:
        cv2 = _cv2()
    except RuntimeError:
        cv2 = None
    if cv2 is not None:
        return cv2.GaussianBlur(gray, (0, 0), sigmaX=max(0.1, float(sigma)))
    # A compact fallback; the exact blur kernel is less important than
    # suppressing focus-specific high-frequency differences.
    value = gray.astype(np.float32)
    return (value + np.roll(value, 1, 0) + np.roll(value, -1, 0)
            + np.roll(value, 1, 1) + np.roll(value, -1, 1)) / 5.0


def _hash_bits(gray: Any, size: int) -> Any:
    np = _np()
    size = max(4, int(size))
    try:
        cv2 = _cv2()
    except RuntimeError:
        cv2 = None
    tiny = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA) if cv2 is not None else _resize_gray(gray, (size, size))
    if tiny.shape != (size, size):
        tiny = _resize_gray(tiny, (size, size))
    if cv2 is not None:
        dct_input = tiny.astype(np.float32)
        dct = cv2.dct(dct_input)
        block = dct[:size, :size]
        threshold = float(np.median(block[1:, 1:])) if block.size > 1 else float(np.mean(block))
        return (block > threshold).reshape(-1).astype(np.uint8)
    threshold = float(np.mean(tiny))
    return (tiny > threshold).reshape(-1).astype(np.uint8)


def _histogram(gray: Any, bins: int) -> Any:
    np = _np()
    hist, _ = np.histogram(gray, bins=max(4, int(bins)), range=(0, 256))
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    return hist / total if total > 0 else hist


def _load_preview(item: Any, config: Any,
                  loader: Callable[..., Any] | None = None) -> Any:
    edge = int(_setting(config, "scene_preview_long_edge", 512))
    if loader is not None:
        try:
            image = loader(item, max_long_edge=edge)
        except TypeError:
            image = loader(item)
        return resize_for_analysis(image, edge)
    source_path = resolve_image_path(item)
    if source_path is not None:
        return load_image(source_path, max_long_edge=edge)
    return resize_for_analysis(item, edge)


def extract_scene_feature(item: Any, config: Any = None,
                          loader: Callable[..., Any] | None = None) -> SceneFeature:
    """Generate one compact preview feature record."""

    config = config or SceneConfig()
    np = _np()
    image = _load_preview(item, config, loader)
    gray = _gray_u8(image)
    blurred = _blur(gray, float(_setting(config, "blur_sigma", 3.0)))
    # Keep the anchor representation small even when a custom loader ignores
    # the requested preview size.
    blurred = resize_for_analysis(blurred, int(_setting(config, "scene_preview_long_edge", 512)))
    hist = _histogram(blurred, int(_setting(config, "histogram_bins", 32)))
    bits = _hash_bits(blurred, int(_setting(config, "hash_size", 8)))
    keypoints = descriptors = None
    try:
        cv2 = _cv2()
        orb = cv2.ORB_create(nfeatures=int(_setting(config, "orb_features", 800)))
        keypoints, descriptors = orb.detectAndCompute(blurred, None)
    except (RuntimeError, Exception):
        # A preview can still be grouped by low-frequency evidence when ORB
        # is unavailable or the frame contains too little texture.
        keypoints = descriptors = None
    return SceneFeature(
        preview=blurred.astype(np.uint8, copy=True),
        histogram=hist,
        hash_bits=bits,
        descriptors=descriptors,
        keypoints=keypoints,
        source_shape=tuple(int(v) for v in gray.shape[:2]),
    )


def _low_frequency_similarity(a: SceneFeature, b: SceneFeature) -> tuple[float, float, float]:
    np = _np()
    first = a.preview.astype(np.float32)
    second = _resize_gray(b.preview, first.shape)
    af = first.reshape(-1)
    bf = second.astype(np.float32).reshape(-1)
    ac = af - float(np.mean(af))
    bc = bf - float(np.mean(bf))
    denom = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    corr = 1.0 if denom <= 1e-8 else float(np.dot(ac, bc) / denom)
    corr_similarity = (corr + 1.0) * 0.5
    mad_similarity = 1.0 - float(np.mean(np.abs(first - second)) / 255.0)
    low = float(np.clip(0.70 * corr_similarity + 0.30 * mad_similarity, 0.0, 1.0))
    hist_a = np.asarray(a.histogram, dtype=np.float32)
    hist_b = np.asarray(b.histogram, dtype=np.float32)
    if hist_a.shape != hist_b.shape:
        hist_b = np.resize(hist_b, hist_a.shape)
    hist_similarity = float(np.clip(np.minimum(hist_a, hist_b).sum(), 0.0, 1.0))
    bits_a = np.asarray(a.hash_bits).reshape(-1)
    bits_b = np.asarray(b.hash_bits).reshape(-1)
    length = min(bits_a.size, bits_b.size)
    hash_similarity = 0.0 if length == 0 else float(np.mean(bits_a[:length] == bits_b[:length]))
    return low, hist_similarity, hash_similarity


def _geometry(a: SceneFeature, b: SceneFeature, config: Any) -> tuple[int, int, float, float, float, tuple[float, float]]:
    """Return matches, inliers, ratio, scale, rotation, translation."""

    np = _np()
    if a.descriptors is None or b.descriptors is None or not a.keypoints or not b.keypoints:
        return 0, 0, 0.0, 1.0, 0.0, (0.0, 0.0)
    try:
        cv2 = _cv2()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw = matcher.knnMatch(b.descriptors, a.descriptors, k=2)
        ratio = float(_setting(config, "orb_ratio_test", 0.75))
        good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
        min_matches = int(_setting(config, "geometric_min_matches", 8))
        if len(good) < min_matches:
            return len(good), 0, 0.0, 1.0, 0.0, (0.0, 0.0)
        src = np.float32([b.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([a.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(
            src, dst, cv2.RANSAC,
            float(_setting(config, "ransac_reprojection_threshold", 3.0)),
        )
        if matrix is None or mask is None:
            return len(good), 0, 0.0, 1.0, 0.0, (0.0, 0.0)
        inliers = int(np.count_nonzero(mask))
        inlier_ratio = inliers / max(1, len(good))
        linear = matrix[:2, :2]
        scale = float((np.linalg.norm(linear[:, 0]) + np.linalg.norm(linear[:, 1])) * 0.5)
        rotation = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
        translation = (float(matrix[0, 2]), float(matrix[1, 2]))
        return len(good), inliers, float(inlier_ratio), scale, rotation, translation
    except Exception:
        return 0, 0, 0.0, 1.0, 0.0, (0.0, 0.0)


def compare_scene_features(candidate: SceneFeature, anchor: SceneFeature,
                           config: Any = None,
                           anchor_index: int | None = None) -> SceneComparison:
    """Compare a candidate preview to one scene anchor."""

    config = config or SceneConfig()
    low, hist, hashed = _low_frequency_similarity(candidate, anchor)
    matches, inliers, inlier_ratio, scale, rotation, translation = _geometry(candidate, anchor, config)
    geo_threshold = float(_setting(config, "geometric_inlier_threshold", 0.55))
    min_matches = int(_setting(config, "geometric_min_matches", 8))
    min_inliers = int(_setting(config, "geometric_min_inliers", 6))
    geo_valid = matches >= min_matches and inliers >= min_inliers and inlier_ratio >= geo_threshold
    geometric_similarity = float(np_clip(0.55 * inlier_ratio + 0.45 * min(1.0, inliers / 30.0), 0.0, 1.0))
    threshold = float(_setting(config, "scene_similarity_threshold", 0.82))
    evidence = 0.50 * low + 0.15 * hist + 0.10 * hashed + 0.25 * geometric_similarity
    # Focus changes retain low-frequency composition; geometry is strong
    # evidence, but low-frequency evidence is still allowed to carry the
    # decision where ORB has too few features.
    same = low >= threshold and (
        geo_valid
        or (not bool(_setting(config, "require_geometry_for_boundary", False))
            and hist >= max(0.60, threshold - 0.15)
            and hashed >= max(0.50, threshold - 0.20))
    )
    confidence = evidence
    if low < threshold:
        confidence *= 0.55
    if matches >= min_matches and not geo_valid:
        confidence *= 0.80
    return SceneComparison(
        low_frequency_similarity=float(low),
        histogram_similarity=float(hist),
        hash_similarity=float(hashed),
        geometric_similarity=float(geometric_similarity),
        valid_matches=int(matches),
        inliers=int(inliers),
        inlier_ratio=float(inlier_ratio),
        scale=float(scale),
        rotation_degrees=float(rotation),
        translation=translation,
        same_scene=bool(same),
        confidence=float(np_clip(confidence, 0.0, 1.0)),
        anchor_index=anchor_index,
    )


def scene_hash(value: Any, config: Any = None,
               loader: Callable[..., Any] | None = None) -> str:
    """Return a compact hexadecimal perceptual hash for one preview/source."""

    feature = value if isinstance(value, SceneFeature) else extract_scene_feature(value, config, loader)
    return feature.scene_hash


def np_clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


class SceneDetector:
    """Streaming detector retaining no more than ``max_anchors`` previews."""

    def __init__(self, config: Any = None,
                 loader: Callable[..., Any] | None = None) -> None:
        self.config = config or SceneConfig()
        self.loader = loader

    def _best_comparison(self, feature: SceneFeature,
                         anchors: Sequence[tuple[int, SceneFeature]]) -> SceneComparison:
        comparisons = [
            compare_scene_features(feature, anchor, self.config, anchor_index=index)
            for index, anchor in anchors
        ]
        # Geometric evidence wins ties; otherwise low-frequency similarity is
        # the most reliable focus-invariant signal.
        return max(comparisons, key=lambda value: (
            float(value.same_scene), value.confidence,
            value.geometric_similarity, value.low_frequency_similarity,
        ))

    def iter_groups(self, items: Iterable[Any]) -> Iterator[SceneGroup]:
        """Yield groups as soon as a confirmed boundary is observed."""

        max_anchors = max(1, int(_setting(self.config, "max_anchors", 3)))
        confirmation = max(0, int(_setting(
            self.config, "scene_confirmation_window", 2,
        )))
        current_items: list[Any] = []
        current_comparisons: list[SceneComparison] = []
        pending: list[tuple[int, Any, SceneFeature, SceneComparison]] = []
        # Keep only first/latest/best compact previews.  Group items are paths
        # or lightweight metadata references; decoded source arrays never
        # enter this list in normal pipeline use.
        first_anchor: tuple[int, SceneFeature] | None = None
        latest_anchor: tuple[int, SceneFeature] | None = None
        best_anchor: tuple[int, SceneFeature] | None = None
        best_anchor_strength = -1.0
        current_start = 0
        current_end = 0
        group_id = 0

        def anchor_strength(feature: SceneFeature) -> float:
            # Preview standard deviation is a cheap proxy for useful visual
            # structure and keeps the optional third anchor representative.
            try:
                import numpy as np
                return float(np.std(feature.preview))
            except Exception:
                return 0.0

        def anchors_for() -> list[tuple[int, SceneFeature]]:
            values = [first_anchor, latest_anchor, best_anchor]
            unique: list[tuple[int, SceneFeature]] = []
            seen: set[int] = set()
            for value in values:
                if value is not None and value[0] not in seen:
                    unique.append(value)
                    seen.add(value[0])
            return unique[:max_anchors]

        def reset_anchors(features: Sequence[tuple[int, SceneFeature]]) -> None:
            nonlocal first_anchor, latest_anchor, best_anchor, best_anchor_strength
            first_anchor = latest_anchor = best_anchor = None
            best_anchor_strength = -1.0
            for index, feature in features:
                update_anchors(index, feature)

        def update_anchors(index: int, feature: SceneFeature) -> None:
            nonlocal first_anchor, latest_anchor, best_anchor, best_anchor_strength
            if first_anchor is None:
                first_anchor = (index, feature)
            latest_anchor = (index, feature)
            strength = anchor_strength(feature)
            if best_anchor is None or strength > best_anchor_strength:
                best_anchor = (index, feature)
                best_anchor_strength = strength

        def make_group() -> SceneGroup:
            nonlocal group_id
            group_id += 1
            if current_comparisons:
                confidence = sum(c.confidence for c in current_comparisons) / len(current_comparisons)
            else:
                confidence = 0.5 if current_items else 0.0
            return SceneGroup(
                group_id=group_id,
                items=list(current_items),
                start_index=current_start,
                end_index=current_end,
                confidence=float(np_clip(confidence, 0.0, 1.0)),
                comparisons=list(current_comparisons),
            )

        for index, item in enumerate(items):
            feature = extract_scene_feature(item, self.config, self.loader)
            if not current_items:
                current_items.append(item)
                current_start = current_end = index
                reset_anchors([(index, feature)])
                continue
            anchors = anchors_for()
            comparison = self._best_comparison(feature, anchors)
            if comparison.same_scene:
                # A single anomalous frame is absorbed once a later frame
                # confirms continuity.
                for pending_index, pending_item, pending_feature, pending_comparison in pending:
                    current_items.append(pending_item)
                    current_end = pending_index
                    current_comparisons.append(pending_comparison)
                    update_anchors(pending_index, pending_feature)
                pending.clear()
                current_items.append(item)
                current_end = index
                current_comparisons.append(comparison)
                update_anchors(index, feature)
                continue
            pending.append((index, item, feature, comparison))
            if confirmation == 0 or len(pending) >= confirmation:
                # Boundary lies before the first pending frame.  Start the
                # next scene with all confirmation frames, preserving order.
                if current_items:
                    yield make_group()
                current_items = [entry[1] for entry in pending]
                current_start = pending[0][0]
                current_end = pending[-1][0]
                # The first frame of a new group has no prior in-group
                # comparison.  Comparisons among confirmation frames are
                # recomputed cheaply from their compact features.
                current_comparisons = []
                pending_features = [(entry[0], entry[2]) for entry in pending]
                reset_anchors(pending_features)
                for left, right in zip(pending_features, pending_features[1:]):
                    current_comparisons.append(compare_scene_features(
                        right[1], left[1], self.config, anchor_index=left[0],
                    ))
                pending = []
        if pending:
            # An unconfirmed tail belongs to the current scene by definition
            # of the confirmation window.
            for pending_index, pending_item, pending_feature, pending_comparison in pending:
                current_items.append(pending_item)
                current_end = pending_index
                current_comparisons.append(pending_comparison)
                update_anchors(pending_index, pending_feature)
        if current_items:
            yield make_group()

    def detect(self, items: Iterable[Any]) -> list[SceneGroup]:
        return list(self.iter_groups(items))


def detect_scenes(items: Iterable[Any], config: Any = None,
                  loader: Callable[..., Any] | None = None) -> list[SceneGroup]:
    return SceneDetector(config, loader).detect(items)


def iter_scene_groups(items: Iterable[Any], config: Any = None,
                      loader: Callable[..., Any] | None = None) -> Iterator[SceneGroup]:
    return SceneDetector(config, loader).iter_groups(items)


split_scenes = detect_scenes
SceneDetectionConfig = SceneConfig
detect_scene_groups = detect_scenes

__all__ = [
    "SceneConfig", "SceneFeature", "SceneComparison", "SceneGroup",
    "SceneDetectionConfig",
    "extract_scene_feature", "compare_scene_features", "SceneDetector",
    "scene_hash", "detect_scenes", "detect_scene_groups", "iter_scene_groups", "split_scenes",
]
