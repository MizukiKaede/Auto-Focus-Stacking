"""Small, dependency-free value types shared by the vision algorithms.

The core modules deliberately keep their public data structures plain.  In
particular, no full-resolution image is stored in any of these objects.  A
``SceneFeature`` only contains a preview-sized feature representation and a
``FocusMapResult`` contains the compressed focus map produced by the analysis
stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


@dataclass(slots=True)
class SceneFeature:
    """Compact representation used by sequential scene detection.

    ``preview`` is a small, blurred grayscale array.  Scene detectors retain
    this only for a bounded number of anchors; it is never a source image.
    ``descriptors`` and ``keypoints`` are optional OpenCV values and can be
    omitted when feature matching is unavailable.
    """

    preview: Any
    histogram: Any
    hash_bits: Any
    descriptors: Any = None
    keypoints: Any = None
    source_shape: tuple[int, int] | None = None

    @property
    def scene_hash(self) -> str:
        """Stable hexadecimal hash useful for SQLite/cache keys."""

        try:
            import numpy as np
            return np.packbits(np.asarray(self.hash_bits, dtype=np.uint8)).tobytes().hex()
        except Exception:
            return ""


@dataclass(slots=True)
class SceneComparison:
    """Evidence for one candidate-to-anchor scene comparison."""

    low_frequency_similarity: float = 0.0
    histogram_similarity: float = 0.0
    hash_similarity: float = 0.0
    geometric_similarity: float = 0.0
    valid_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    scale: float = 1.0
    rotation_degrees: float = 0.0
    translation: tuple[float, float] = (0.0, 0.0)
    same_scene: bool = False
    confidence: float = 0.0
    anchor_index: int | None = None

    @property
    def blur_similarity(self) -> float:
        """Backward-compatible name used by early prototypes."""

        return self.low_frequency_similarity

    @property
    def geometry_valid(self) -> bool:
        return self.inliers > 0 and self.inlier_ratio > 0.0


@dataclass(slots=True)
class SceneGroup:
    """A sequential group emitted by :class:`SceneDetector`."""

    group_id: int
    items: list[Any] = field(default_factory=list)
    start_index: int = 0
    end_index: int = 0
    confidence: float = 0.0
    comparisons: list[SceneComparison] = field(default_factory=list)
    excluded_reason: str | None = None

    @property
    def image_count(self) -> int:
        return len(self.items)

    @property
    def first_item(self) -> Any | None:
        return self.items[0] if self.items else None

    @property
    def images(self) -> list[Any]:
        """Alias used by pipeline code that calls group members images."""

        return self.items

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index: int) -> Any:
        return self.items[index]


@dataclass(slots=True)
class RegistrationResult:
    """Low-resolution registration result.

    ``matrix`` is a 3x3 homography mapping the input image into reference
    coordinates.  It may be ``None`` for an identity/fallback result.
    """

    matrix: Any = None
    model: str = "identity"
    valid: bool = False
    valid_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    scale: float = 1.0
    rotation_degrees: float = 0.0
    translation: tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0
    reprojection_error: float | None = None
    message: str = ""

    # Names used by a few consumers and useful for serialising the result.
    @property
    def matches(self) -> int:
        return self.valid_matches

    @property
    def transform(self) -> Any:
        return self.matrix


@dataclass(slots=True)
class FocusMapResult:
    """Compressed focus-map and summary metrics for one image."""

    focus_map: Any
    sharpness_score: float = 0.0
    texture_fraction: float = 0.0
    mean_focus: float = 0.0
    focus_std: float = 0.0
    source_shape: tuple[int, int] | None = None

    @property
    def map(self) -> Any:
        return self.focus_map


@dataclass(slots=True)
class ImageQuality:
    """Normalised quality components and final score in the [0, 1] range."""

    score: float = 0.0
    sharpness: float = 0.0
    motion_stability: float = 0.0
    exposure: float = 0.0
    clipping_penalty: float = 0.0
    noise_penalty: float = 0.0
    details: Mapping[str, float] = field(default_factory=dict)

    def __float__(self) -> float:
        return self.score


@dataclass(slots=True)
class FocusCluster:
    """A near-identical-focus cluster and its selected representative."""

    cluster_id: int
    members: tuple[int, ...]
    representative_index: int
    representative_quality: float = 0.0
    similarity_to_representative: Mapping[int, float] = field(default_factory=dict)

    @property
    def selected_index(self) -> int:
        return self.representative_index


@dataclass(slots=True)
class DeduplicationResult:
    """Output of duplicate focus clustering."""

    clusters: list[FocusCluster]
    representative_indices: list[int]
    rejected_indices: list[int]
    reasons: Mapping[int, str] = field(default_factory=dict)


@dataclass(slots=True)
class CoverageSelection:
    """Greedy maximum-coverage selection result."""

    selected_indices: list[int]
    rejected_indices: list[int]
    coverage: float
    gains: Mapping[int, float] = field(default_factory=dict)
    reasons: Mapping[int, str] = field(default_factory=dict)
    coverage_map: Any = None

    @property
    def selected(self) -> list[int]:
        return self.selected_indices

    @property
    def coverage_fraction(self) -> float:
        return self.coverage

