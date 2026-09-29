Warning: truncated output (original token count: 21908)
Total output lines: 1885

"""Production group-level analysis orchestration.

``GroupAnalyzer`` is the bridge between scene grouping and the bounded
analysis/merge pipeline.  It intentionally owns no full-resolution image
collection: one analysis image is decoded, registered against the selected
reference, converted to a compact focus map, scored, and released before the
next image is read.  Only compact maps and light metadata remain until
deduplication/coverage selection finishes.
"""

from __future__ import annotations

from ..utils.performance import timed
from ..utils.concurrency import resolve_focus_analysis_budget

from dataclasses import dataclass
import hashlib
import io
import json
import logging
import math
import os
from os import PathLike
from pathlib import Path
import threading
from typing import Any, Callable, Iterable, Mapping, Sequence

from .coverage_selector import CoverageConfig, select_focus_images
from .focus_cluster import FocusClusterConfig, deduplicate_focus_maps, focus_map_distance
from .focus_map import FocusMapConfig, compute_focus_map, decompress_focus_map
from .quality import QualityConfig, score_image
from .registration import RegistrationConfig, load_image, register_images, resolve_image_path, warp_image
from .types import ImageQuality, RegistrationResult


# Bump this whenever the implementation changes the numerical meaning of a
# cached focus map.  The source fingerprint is still kept as the first part
# of the filename, while this version/config digest forms the namespace after
# it.  Keeping the two components separate makes it possible to invalidate a
# map for a changed algorithm/config without weakening source-file reuse.
CACHE_ALGORITHM_VERSION = "focus-map-v1"

_FOCUS_MAP_CACHE_FIELDS: tuple[tuple[str, Any, tuple[str, ...]], ...] = (
    ("analysis_long_edge", 1600, ("focus_analysis_long_edge",)),
    ("output_long_edge", 512, ("focus_cache_long_edge", "focus_output_long_edge")),
    ("output_dtype", "float16", ("focus_map_dtype",)),
    ("scales", (1.0, 2.5, 5.0), ()),
    ("laplacian_weight", 0.50, ()),
    ("gradient_weight", 0.30, ()),
    ("variance_weight", 0.20, ()),
    ("local_variance_window", 7, ()),
    ("normalise_low_percentile", 1.0, ()),
    ("normalise_high_percentile", 99.0, ()),
    ("saturation_low", 0.01, ()),
    ("saturation_high", 0.99, ()),
    ("saturation_penalty", 0.35, ()),
    ("output_blur_sigma", 0.6, ()),
)


@dataclass(slots=True)
class GroupAnalyzerConfig:
    """Optional direct configuration for headless/group analysis.

    An ``AnalysisConfig`` or ``AppConfig`` from :mod:`focus_stack_app.config`
    can be passed instead; the core stages read its ``analysis`` fields via
    duck typing.  Explicit nested stage configs are useful for tests and
    advanced tuning without changing the application-wide config schema.
    """

    analysis_long_edge: int = 1600
    output_long_edge: int = 512
    output_dtype: str = "float16"
    duplicate_focus_threshold: float = 0.995
    minimum_stack_images: int = 2
    # Groups below this threshold are classified before any pixel decoding.
    minimum_stack_group_size: int = 3
    # A zero threshold disables the application-level stability gate for
    # callers that use the core analyzer as a library.
    minimum_stack_stability: float = 0.0
    coverage_target: float | None = None
    min_coverage_gain: float = 0.002
    focus_threshold: float = 0.95
    coverage_mode: str = "balanced"
    cache_focus_maps: bool = True
    selection_frame_cache_bytes: int = 400 * 1024**2
    # Standalone/library callers retain serial behavior unless they opt in.
    # The desktop AppConfig uses RuntimeConfig's automatic value instead.
    focus_analysis_workers: int = 1


class GroupAnalysisCancelled(RuntimeError):
    """Marker exception for integrations that choose fail-fast cancellation."""


def _setting(config: Any, name: str, default: Any, *aliases: str) -> Any:
    """Read a setting from direct config or nested ``config.analysis``."""

    if config is None:
        return default
    names = (name,) + aliases
    nested = config.get("analysis") if isinstance(config, Mapping) else getattr(config, "analysis", None)
    for source in (config, nested):
        if source is None:
            continue
        for candidate in names:
            if isinstance(source, Mapping):
                value = source.get(candidate)
            else:
                value = getattr(source, candidate, None)
            if value is not None:
                return value
    return default


def _group_value(group: Any, *names: str, default: Any = None) -> Any:
    if group is None:
        return default
    if isinstance(group, Mapping):
        for name in names:
            if name in group and group[name] is not None:
                return group[name]
        return default
    for name in names:
        try:
            value = getattr(group, name)
        except Exception:
            continue
        if value is not None:
            return value
    return default


def _item_value(item: Any, *names: str, default: Any = None) -> Any:
    if item is None:
        return default
    if isinstance(item, Mapping):
        for name in names:
            if name in item and item[name] is not None:
                return item[name]
        return default
    for name in names:
        try:
            value = getattr(item, name)
        except Exception:
            continue
        if value is not None:
            return value
    return default


def _group_id(group: Any) -> int | str | None:
    # ``ImageRecord.id`` identifies an image, not a logical group.  Direct
    # singleton analysis is supported, so do not accidentally persist that
    # image id as a group foreign key when ``group_id`` is unset.
    if isinstance(group, Mapping):
        if "original_path" in group and group.get("group_id") is None:
            return None
    elif (
        hasattr(group, "original_path")
        and not hasattr(group, "items")
        and not hasattr(group, "images")
        and getattr(group, "group_id", None) is None
    ):
        return None
    return _group_value(group, "group_id", "id", default=None)


def _items_for_group(group: Any, database: Any = None) -> list[Any]:
    values = _group_value(
        group, "all_images", "images", "items", "image_records", "all_paths", "image_paths", default=None,
    )
    if values is not None:
        if isinstance(values, (str, bytes, PathLike)):
            return [values]
        try:
            resolved = list(values)
            if resolved:
                return resolved
        except TypeError:
            pass
    gid = _group_id(group)
    getter = getattr(database, "get_group_images", None) if database is not None else None
    if callable(getter) and gid is not None:
        try:
            found = list(getter(int(gid)))
            if found:
                return found
        except Exception:
            pass
    # A direct ``ImageRecord`` is a useful singleton input for callers that
    # have not materialised a ``SceneGroup`` yet.  Do this last so a normal
    # group with a missing/empty collection still reports a useful error
    # rather than accidentally treating the group metadata itself as an
    # image.
    if _path_for_item(group) is not None:
        return [group]
    return []


def _path_for_item(item: Any) -> Path | None:
    value = resolve_image_path(item)
    if value is None:
        return None
    try:
        return Path(value)
    except (TypeError, ValueError):
        return None


def _identity_result() -> RegistrationResult:
    try:
        import numpy as np
        matrix = np.eye(3, dtype=np.float32)
    except ImportError:  # pragma: no cover - package requires NumPy
        matrix = None
    return RegistrationResult(
        matrix=matrix, model="identity", valid=True,
        inlier_ratio=1.0, confidence=1.0, message="reference image",
    )


def _matrix_json(result: RegistrationResult | None) -> Any:
    if result is None or result.matrix is None:
        return None
    try:
        return result.matrix.tolist()
    except AttributeError:
        return result.matrix


def _focus_array(value: Any) -> Any:
    """Accept either a compact array or a ``FocusMapResult``-like wrapper."""

    if isinstance(value, Mapping):
        return value.get("focus_map", value.get("map", value))
    for name in ("focus_map", "map"):
        candidate = getattr(value, name, None)
        if candidate is not None and candidate is not value:
            return candidate
    return value


def _serialize_focus_map(focus_map: Any) -> bytes:
    """Encode one compact map for ``DiskCache``/cache duck types."""

    import numpy as np
    stream = io.BytesIO()
    np.savez_compressed(stream, focus_map=np.asarray(_focus_array(focus_map)))
    return stream.getvalue()


def _deserialize_focus_map(data: bytes) -> Any:
    import numpy as np
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if "focus_map" in archive:
            return archive["focus_map"].copy()
        # Be liberal with older cache writers that used ``map``/``arr``.
        for key in ("map", "arr", "data"):
            if key in archive:
                return archive[key].copy()
    raise ValueError("focus-map cache does not contain a map array")


def _compact_shape(source_shape: tuple[int, int], long_edge: int) -> tuple[int, int]:
    """Return the deterministic map shape for one analysis image."""

    height, width = (max(1, int(source_shape[0])), max(1, int(source_shape[1])))
    edge = int(long_edge)
    if edge <= 0 or max(height, width) <= edge:
        return height, width
    scale = float(edge) / float(max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _stable_cache_value(value: Any) -> Any:
    """Convert a focus-map setting into deterministic JSON-compatible data.

    The application config currently contains only scalar values and a tuple
    of scales, but accepting mappings/paths here keeps the cache namespace
    useful for lightweight duck-typed configs used by integrations and tests.
    """

    if isinstance(value, Mapping):
        return {
            str(key): _stable_cache_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable_cache_value(item) for item in value]
    if isinstance(value, PathLike):
        return str(value)
    # Enum-like config values (for example a future dtype enum) should use
    # their actual value rather than a process-specific repr.
    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        return _stable_cache_value(enum_value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _focus_map_cache_namespace(config: Any) -> str:
    """Return a stable algorithm/config namespace for one raw focus map.

    A source fingerprint alone is insufficient: changing output resolution,
    quantisation, scales, response weights, saturation handling, or any
    other numerical map setting must not reuse the old bytes.  The explicit
    algorithm version gives future implementations a one-line invalidation
    switch even when the setting list is unchanged.
    """

    integer_fields = {"analysis_long_edge", "output_long_edge", "local_variance_window"}
    float_fields = {
        "laplacian_weight", "gradient_weight", "variance_weight",
        "normalise_low_percentile", "normalise_high_percentile",
        "saturation_low", "saturation_high", "saturation_penalty",
        "output_blur_sigma",
    }
    settings: dict[str, Any] = {}
    for name, default, aliases in _FOCUS_MAP_CACHE_FIELDS:
        value = _setting(config, name, default, *aliases)
        try:
            if name in integer_fields:
                value = int(value)
            elif name in float_fields:
                value = float(value)
            elif name == "output_dtype":
                value = str(getattr(value, "value", value)).lower()
            elif name == "scales":
                value = [float(item) for item in (value or ())]
        except (TypeError, ValueError):
            # Let the focus-map implementation apply its normal validation;
            # still make malformed values a distinct namespace rather than
            # silently aliasing them to a valid default.
            value = _stable_cache_value(value)
        settings[name] = _stable_cache_value(value)
    payload = {
        "algorithm_version": CACHE_ALGORITHM_VERSION,
        "focus_map": settings,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(cache: Any, path: Path, namespace: str | None = None) -> Path | None:
    """Resolve a source-fingerprint path and append a map namespace.

    ``DiskCache`` intentionally owns source fingerprinting, so we first ask
    it for the ordinary path and then add ``.<namespace>`` before the suffix.
    This preserves the source fingerprint while avoiding the old unscoped map
    and keeps all variants in the existing ``focusmaps`` directory.
    """

    if cache is None:
        return None
    base: Path | None = None
    for name in ("focus_map_path", "get_focus_map_path"):
        method = getattr(cache, name, None)
        if callable(method):
            try:
                base = Path(method(path, suffix=".npz"))
                break
            except Exception:
                try:
                    base = Path(method(path))
                    break
                except Exception:
                    continue
    if base is None:
        return None
    if not namespace:
        return base
    suffix = base.suffix or ".npz"
    return base.with_name(f"{base.stem}.{namespace}{suffix}")


def _cache_memory_get(cache: Any, key: str) -> Any:
    memory = getattr(cache, "focus_memory", None)
    getter = getattr(memory, "get", None)
    if not callable(getter):
        return None
    try:
        return getter(key)
    except Exception:
        return None


def _cache_memory_put(cache: Any, key: str, value: bytes) -> None:
    memory = getattr(cache, "focus_memory", None)
    putter = getattr(memory, "put", None)
    if callable(putter):
        try:
            putter(key, value)
        except Exception:
            pass


def _cache_memory_delete(cache: Any, key: str) -> None:
    memory = getattr(cache, "focus_memory", None)
    deleter = getattr(memory, "delete", None)
    if callable(deleter):
        try:
            deleter(key)
        except Exception:
            pass


def _read_cache_bytes(cache: Any, path: Path) -> bytes | None:
    reader = getattr(cache, "read_bytes", None)
    if callable(reader):
        try:
            payload = reader(path)
            return None if payload is None else bytes(payload)
        except Exception:
            return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def _write_cache_bytes(cache: Any, path: Path, payload: bytes) -> None:
    """Write a namespaced map, using a cache's atomic writer when available."""

    atomic_writer = getattr(cache, "_write_atomic", None)
    if callable(atomic_writer):
        lock = getattr(cache, "_lock", None)
        try:
            if lock is not None and hasattr(lock, "__enter__"):
                with lock:
                    atomic_writer(path, payload)
            else:
                atomic_writer(path, payload)
            return
        except Exception:
            pass
    writer = getattr(cache, "write_bytes", None)
    if callable(writer):
        try:
            writer(path, payload)
            return
        except Exception:
            pass
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    except OSError:
        pass


def _load_or_create_cached_map(
    cache: Any, path: Path, creator: Callable[[], Any], *,
    namespace: str | None = None,
) -> tuple[Any, Path | None]:
    """Read/write one namespaced compressed raw map through cache interfaces.

    The cache API predates algorithm namespaces and its convenience methods
    derive keys from only the source path.  Calling those methods here would
    reintroduce stale-map hits, so this adapter operates on the namespaced
    path directly while retaining the cache's bounded focus-memory LRU.
    """

    if cache is None:
        return creator(), None
    cache_path = _cache_path(cache, path, namespace)
    # A cache without a path API cannot safely express a namespace.  Compute
    # rather than falling back to the legacy source-only convenience method;
    # this is the safe behavior for minimal duck-typed test caches.
    if cache_path is None:
        return creator(), None
    memory_key = f"focus-map:{cache_path}"
    memory_payload = _cache_memory_get(cache, memory_key)
    if isinstance(memory_payload, (bytes, bytearray, memoryview)):
        try:
            return _deserialize_focus_map(bytes(memory_payload)), cache_path
        except Exception:
            _cache_memory_delete(cache, memory_key)
    payload = _read_cache_bytes(cache, cache_path)
    if payload:
        try:
            value = _deserialize_focus_map(payload)
            _cache_memory_put(cache, memory_key, payload)
            return value, cache_path
        except Exception:
            pass
    value = creator()
    encoded = _serialize_focus_map(value)
    _write_cache_bytes(cache, cache_path, encoded)
    _cache_memory_put(cache, memory_key, encoded)
    return value, cache_path


def _warp_focus_map(focus_map: Any, registration: RegistrationResult,
                    source_shape: tuple[int, int],
                    target_shape: tuple[int, int],
                    reference_shape: tuple[int, int] | None = None) -> Any:
    """Warp a compact raw map while scaling an analysis-resolution matrix."""

    import numpy as np
    # Cache/config may use uint16 to halve disk/RAM cost.  Convert that
    # representation before geometric operations; otherwise an identity map
    # would expose 0..65535 values and a warped map would be clipped to 1.
    values = decompress_focus_map(_focus_array(focus_map)).astype(np.float32, copy=False)

    def resize_values(arr: Any) -> Any:
        if arr.shape == target_shape:
            return arr
        try:
            import cv2
            return cv2.resize(arr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
        except ImportError:
            ys = np.minimum(
                (np.arange(target_shape[0]) * arr.shape[0] / target_shape[0]).astype(int),
                arr.shape[0] - 1,
            )
            xs = np.minimum(
                (np.arange(target_shape[1]) * arr.shape[1] / target_shape[1]).astype(int),
                arr.shape[1] - 1,
            )
            return arr[ys[:, None], xs[None, :]]

    matrix = getattr(registration, "matrix", None)
    if matrix is None or getattr(registration, "model", "identity") == "identity":
        return resize_values(values).astype(np.float16, copy=False)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (2, 3):
        converted = np.eye(3, dtype=np.float32)
        converted[:2] = matrix
        matrix = converted
    if matrix.shape != (3, 3):
        return resize_values(values).astype(np.float16, copy=False)
    sh, sw = source_shape
    rh, rw = reference_shape or source_shape
    th, tw = target_shape
    mh, mw = values.shape[:2]
    # H maps source analysis pixels to reference analysis pixels.  Convert it
    # to source-map -> reference-map coordinates.  The map and source images
    # may have different aspect/resolution, so both scale factors matter.
    source_map_to_analysis = np.diag([
        max(1.0, float(sw)) / max(1.0, float(mw)),
        max(1.0, float(sh)) / max(1.0, float(mh)), 1.0,
    ]).astype(np.float32)
    reference_analysis_to_map = np.diag([
        max(1.0, float(tw)) / max(1.0, float(rw)),
        max(1.0, float(th)) / max(1.0, float(rh)), 1.0,
    ]).astype(np.float32)
    map_matrix = reference_analysis_to_map @ matrix @ source_map_to_analysis
    try:
        warped = warp_image(
            values,
            RegistrationResult(
                matrix=map_matrix,
                model=registration.model,
                valid=registration.valid,
            ),
            output_shape=target_shape,
        )
        return np.clip(warped, 0.0, 1.0).astype(np.float16)
    except Exception:
        # OpenCV is optional for the rest of the analysis; if it is missing or
        # rejects a degenerate transform, retain a correctly sized best-effort
        # map instead of poisoning the next candidate with a shape mismatch.
        return resize_values(values).astype(np.float16, copy=False)


class GroupAnalyzer:
    """Analyze one :class:`SceneGroup` with bounded image memory.

    Parameters
    ----------
    config:
        ``AnalysisConfig``, ``AppConfig`` or any object exposing matching
        analysis attributes.  A ``GroupAnalyzerConfig`` is also accepted.
    database:
        Optional repository exposing ``upsert_analysis`` and
        ``set_group_status``/``set_image_group``.
    plan_cache:
        Optional repository exposing versioned ``get_cached_plan`` and
        ``upsert_cached_plan`` methods.  It is deliberately separate from
        ``database`` so ``preserve_cache=False`` can disable reuse while
        retaining normal analysis persistence.
    cache:
        Optional ``DiskCache``-compatible object.  Cached maps are raw
        image-coordinate maps and are transformed into the current group's
        reference coordinates on each run.
    """

    def __init__(self, config: Any = None, *, database: Any = None,
                 cache: Any = None, analysis_cache: Any = None,
                 plan_cache: Any = None,
                 logger: logging.Logger | None = None,
                 loader: Callable[..., Any] | None = None,
                 focus_analysis_workers: int | None = None,
                 focus_analysis_workers_requested: int | None = None,
                 registration_config: Any = None,
                 focus_map_config: Any = None,
                 quality_config: Any = None,
                 cluster_config: Any = None,
                 coverage_config: Any = None) -> None:
        self.config = config if config is not None else GroupAnalyzerConfig()
        self.database = database
        self.cache = cache if cache is not None else analysis_cache
        self.plan_cache = plan_cache
        self.logger = logger or logging.getLogger(__name__)
        self.loader = loader
        if focus_analysis_workers is not None and not 1 <= int(focus_analysis_workers) <= 8:
            raise ValueError("focus_analysis_workers must be between 1 and 8")
        if focus_analysis_workers_requested is not None and not 0 <= int(focus_analysis_workers_requested) <= 8:
            raise ValueError("focus_analysis_workers_requested must be between 0 and 8")
        self.focus_analysis_workers = None if focus_analysis_workers is None else int(focus_analysis_workers)
        self.focus_analysis_workers_requested = (
            None
            if focus_analysis_workers_requested is None
            else int(focus_analysis_workers_requested)
        )
        self.registration_config = registration_config if registration_config is not None else self.config
        self.focus_map_config = focus_map_config if focus_map_config is not None else self.config
        self.quality_config = quality_config if quality_config is not None else self.config
        self.cluster_config = cluster_config if cluster_config is not None else self.config
        self.coverage_config = coverage_config if coverage_config is not None else self.config

    def _load_analysis(self, item: Any) -> Any:
        source_path = _path_for_item(item)
        edge = int(
            _setting(self.config, "focus_analysis_long_edge", 1600, "analysis_long_edge")
        )
        if self.loader is not None:
            try:
                image = self.loader(item, max_long_edge=edge)
            except TypeError:
                image = self.loader(item)
            from .registration import resize_for_analysis
            return resize_for_analysis(image, edge)
        if source_path is not None:
            return load_image(source_path, max_long_edge=edge)
        from .registration import resize_for_analysis
        return resize_for_analysis(item, edge)

    # Public stage boundaries make the unified analysis pipeline independently
    # testable without reintroducing mode-specific analyzers.
    def analyze_pairwise_registration(self, previews: Sequence[Any]) -> list[Any]:
        from .alignment_order import analyze_pairwise_registration
        return analyze_pairwise_registration(previews, config=self._registration_config())

    def choose_preview_reference(self, previews: Sequence[Any]) -> tuple[int, dict[str, Any]]:
        from .alignment_order import choose_preview_reference
        edges = self.analyze_pairwise_registration(previews)
        return choose_preview_reference(len(previews), edges)

    def align_preview(self, reference: Any, image: Any) -> RegistrationResult:
        return register_images(reference, image, self._registration_config())

    def build_focus_maps(self, images: Sequence[Any]) -> list[Any]:
        return [compute_focus_map(image, self._focus_config()) for image in images]

    def score_frames(self, images: Sequence[Any], focus_maps: Sequence[Any],
                     registrations: Sequence[RegistrationResult]) -> list[ImageQuality]:
        return [
            score_image(image, focus_map, registration, self._quality_config())
            for image, focus_map, registration in zip(images, focus_maps, registrations)
        ]

    def remove_duplicate_focus(self, focus_maps: Sequence[Any], quality_scores: Sequence[float]) -> Any:
        return deduplicate_focus_maps(focus_maps, quality_scores, self._cluster_config())

    def select_frames(self, focus_maps: Sequence[Any], quality_scores: Sequence[float],
                      *, candidate_indices: Sequence[int] | None = None) -> Any:
        return select_focus_images(
            focus_maps, quality_scores, self._coverage_config(), candidate_indices=candidate_indices,
        )

    @timed("ordering_registration")
    def build_alignment_order(self, selected_previews: Sequence[Any],
                              *, capture_order: Sequence[int] | None = None) -> dict[str, Any]:
        from .alignment_order import build_alignment_order
        return build_alignment_order(
            selected_previews, config=self._registration_config(), capture_order=capture_order,
        )

    @staticmethod
    def evaluate_analysis(result: Mapping[str, Any]) -> str:
        if bool(result.get("cancelled")):
            return "CANCELLED"
        return "READY_FOR_MERGE" if len(result.get("selected_indices", ())) > 1 else "NO_MERGE"

    def _focus_config(self) -> Any:
        if self.focus_map_config is not self.config:
            return self.focus_map_config
        return FocusMapConfig(
            analysis_long_edge=int(_setting(self.config, "focus_analysis_long_edge", 1600, "analysis_long_edge")),
            output_long_edge=int(_setting(self.config, "focus_cache_long_edge", 512, "output_long_edge")),
            output_dtype=str(_setting(self.config, "focus_map_dtype", "float16", "output_dtype")),
        )

    def _registration_config(self) -> Any:
        if self.registration_config is not self.config:
            return self.registration_config
        return RegistrationConfig(
            analysis_long_edge=int(_setting(self.config, "focus_analysis_long_edge", 1600, "analysis_long_edge")),
        )

    def _quality_config(self) -> Any:
        if self.quality_config is not self.config:
            return self.quality_config
        return QualityConfig(
            analysis_long_edge=int(_setting(self.config, "focus_analysis_long_edge", 1600, "analysis_long_edge")),
        )

    def _cluster_config(self) -> Any:
        if self.cluster_config is not self.config:
            return self.cluster_config
        return FocusClusterConfig(
            similarity_threshold=float(_setting(self.config, "duplicate_focus_threshold", 0.995)),
        )

    def _coverage_config(self) -> Any:
        if self.coverage_config is not self.config:
            return self.coverage_config
        mode = str(_setting(self.config, "coverage_mode", "balanced", "mode"))
        target = _setting(self.config, "coverage_target", None)
        values: dict[str, Any] = {
            "mode": mode,
            "min_coverage_gain": float(_setting(self.config, "min_coverage_gain", 0.002)),
            "focus_threshold": float(_setting(self.config, "focus_threshold", 0.95)),
        }
        # Leaving the target unset preserves the sparse/balanced/extreme
        # preset selected by the caller/UI.
        if target is not None:
            values["coverage_target"] = float(target)
        return CoverageConfig(**values)

    def _minimum_stack_images(self) -> int:
        """Return the smallest selected set allowed for a confirmed scene."""

        value = _setting(
            self.config,
            "minimum_stack_images",
            1,
            "minimum_merge_images",
        )
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1

    def _minimum_stack_group_size(self) -> int:
        """Return the smallest scene cardinality eligible for fusion."""

        value = _setting(
            self.config,
            "minimum_stack_group_size",
            3,
            "minimum_merge_group_size",
        )
        try:
            return max(2, int(value))
        except (TypeError, ValueError):
            return 3

    def _minimum_stack_stability(self) -> float:
        """Return the adjacent-frame similarity required before fusion."""

        value = _setting(
            self.config,
            "minimum_stack_stability",
            0.0,
            "stack_stability_threshold",
        )
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _requested_focus_analysis_workers(self) -> int:
        if self.focus_analysis_workers_requested is not None:
            return int(self.focus_analysis_workers_requested)
        runtime = (
            self.config.get("runtime")
            if isinstance(self.config, Mapping)
            else getattr(self.config, "runtime", None)
        )
        for source in (runtime, self.config):
            if source is None:
                continue
            value = (
                source.get("focus_analysis_workers")
                if isinstance(source, Mapping)
                else getattr(source, "focus_analysis_workers", None)
            )
            if value is not None:
                return int(value)
        return 1

    def _effective_focus_analysis_workers(self, image_count: int) -> tuple[int, int]:
        requested = self._requested_focus_analysis_workers()
        if self.focus_analysis_workers is not None:
            return requested, max(1, min(int(image_count), int(self.focus_analysis_workers)))
        runtime = (
            self.config.get("runtime")
            if isinstance(self.config, Mapping)
            else getattr(self.config, "runtime", None)
        )
        budget = resolve_focus_analysis_budget(
            requested,
            image_count=image_count,
            merge_workers=int(
                getattr(runtime, "max_hugin_workers", 1)
                if runtime is not None and not isinstance(runtime, Mapping)
                else (runtime or {}).get("max_hugin_workers", 1)
            ),
            parallel_pipeline=bool(
                getattr(runtime, "parallel_pipeline", True)
                if runtime is not None and not isinstance(runtime, Mapping)
                else (runtime or {}).get("parallel_pipeline", True)
            ),
            minimum_available_bytes=int(
                getattr(runtime, "min_available_memory_bytes", 2 * 1024**3)
                if runtime is not None and not isinstance(runtime, Mapping)
                else (runtime or {}).get("min_available_memory_bytes", 2 * 1024**3)
            ),
            minimum_available_fraction=float(
                getattr(runtime, "min_available_memory_fraction", 0.10)
                if runtime is not None and not isinstance(runtime, Mapping)
                else (runtime or {}).get("min_available_memory_fraction", 0.10)
            ),
        )
        return requested, budget.workers

    @staticmethod
    def _group_minimum_scene_similarity(group: Any) -> float | None:
        """Return the weakest consecutive scene comparison, if available."""

        values: list[float] = []
        for comparison in list(_group_value(group, "comparisons", default=()) or ()):
            value = _item_value(
                comparison,
                "low_frequency_similarity",
                "blur_similarity",
                default=None,
            )
            try:
                if value is not None:
                    values.append(float(value))
            except (TypeError, ValueError):
                continue
        return min(values) if values else None

    def _persist_status(self, group: Any, status: str, **values: Any) -> None:
        repository = self.database
        gid = _group_id(group)
        if repository is None or gid is None:
            return
        setter = getattr(repository, "set_group_status", None)
        if callable(setter):
            try:
                setter(int(gid), status, **values)
                return
            except Exception:
                try:
                    setter(gid, status)
                    return
                except Exception:
                    self.logger.debug("Unable to persist group state %s", gid, exc_info=True)
        updater = getattr(repository, "update_group", None)
        if callable(updater):
            try:
                if isinstance(group, Mapping):
                    group["status"] = status
                    group.update(values)
                else:
                    for name, value in values.items():
                        if hasattr(group, name):
                            setattr(group, name, value)
                    if hasattr(group, "status"):
                        setattr(group, "status", status)
                updater(group)
            except Exception:
                self.logger.debug("Unable to persist group state", exc_info=True)

    def _persist_image(self, group: Any, item: Any, result: Mapping[str, Any]) -> None:
        repository = self.database
        image_id = _item_value(item, "id", default=None)
        gid = _group_id(group)
        if repository is None or image_id is None:
            return
        selected = bool(result.get("selected", False))
        status = "SELECTED" if selected else "REJECTED"
        set_group = getattr(repository, "set_image_group", None)
        if callable(set_group) and gid is not None:
            try:
                set_group(int(image_id), int(gid), status=status)
            except Exception:
                try:
                    set_group(image_id, gid, status=status)
                except Exception:
                    self.logger.debug("Unable to persist image group", exc_info=True)
        try:
            from ..storage.models import AnalysisRecord
            record = AnalysisRecord(
                image_id=int(image_id),
                scene_hash=result.get("scene_hash"),
                scene_score=result.get("scene_score"),
                sharpness_score=result.get("sharpness_score"),
                focus_map_path=result.get("focus_map_path"),
                transform=result.get("transform"),
                selected=selected,
                selection_reason=result.get("selection_reason"),
                coverage_gain=result.get("coverage_gain"),
                quality_score=result.get("quality_score"),
            )
        except (TypeError, ValueError):
            record = dict(result)
            record["image_id"] = image_id
        saver = getattr(repository, "upsert_analysis", getattr(repository, "save_analysis", None))
        if callable(saver):
            try:
                saver(record)
            except Exception:
                if not isinstance(record, dict):
                    try:
                        saver(record.to_mapping())
                    except Exception:
                        self.logger.debug("Unable to persist analysis record", exc_info=True)
                else:
                    self.logger.debug("Unable to persist analysis record", exc_info=True)

    def _emit_progress(self, callback: Callable[..., Any] | None, group: Any,
                       path: Path | None, index: int, total: int) -> None:
        if callback is None:
            return
        gid = _group_id(group)
        try:
            # PipelineCoordinator expects its normal immutable event object;
            # importing lazily keeps core imports headless and lightweight.
            from ..pipeline.events import PipelineEvent, PipelineStage
            callback(PipelineEvent(
                stage=PipelineStage.ANALYSIS,
                current_file=str(path or ""), current_group=gid,
                completed=index + 1, total=total,
                groups_found=1, message=f"焦点分析 {index + 1}/{total}",
                analysis_completed=index + 1, analysis_total=total,
            ))
        except ImportError:
            try:
                callback({
                    "stage": "analysis", "current_file": str(path or ""),
                    "current_group": gid, "completed": index + 1,
                    "total": total, "analysis_completed": index + 1,
                    "analysis_total": total,
                    "message": f"焦点分析 {index + 1}/{total}",
                })
            except Exception:
                self.logger.debug("Progress callback failed", exc_info=True)
        except Exception:
            self.logger.debug("Progress callback failed", exc_info=True)

    def _analyze_established_scene_selection(
        self, group: Any, items: list[Any], cancel_event: threading.Event,
        progress_callback: Callable[..., Any] | None,
    ) -> dict[str, Any]:
        """Run the calibrated whole-frame selection under the current API."""
        import cv2
    …1908 tokens truncated…n cache write failed group_id=%s", gid, exc_info=True,
                )
        assert plan is not None and order_result is not None
        selected_indices = list(plan["selected_indices"])
        selected_set = set(selected_indices)
        selected_paths = [str(paths[index]) for index in selected_indices]

        # The established ECC matrices use inverse-warp convention. Publish
        # source -> reference matrices for the generic experimental backend.
        transforms: list[Any] = []
        for matrix in plan["matrices"]:
            affine = cv2.invertAffineTransform(np.asarray(matrix, np.float32))
            homogeneous = np.eye(3, dtype=np.float32)
            homogeneous[:2] = affine
            transforms.append(homogeneous.tolist())

        alignment_order_indices = list(order_result["alignment_order"])
        alignment_order = [str(paths[index]) for index in alignment_order_indices]

        reasons: dict[int, str] = {}
        gains: dict[int, float] = {}
        per_image: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if index in plan["errors"]:
                reason = "whole-frame alignment failed: " + plan["errors"][index]
            elif index == plan["reference_index"]:
                reason = "highest whole-frame focus score; preview reference"
            elif index in selected_set:
                reason = f"adds whole-frame sharp coverage {plan['gains'].get(index, 0.0):.2%}"
            else:
                reason = "sharp regions are already covered by selected frames"
            reasons[index] = reason
            gains[index] = float(plan["gains"].get(index, 0.0))
            row = {
                "index": index,
                "image": item,
                "path": str(paths[index]),
                "selected": index in selected_set,
                "selection_reason": reason,
                "reason": reason,
                "quality_score": float(plan["qualities"][index]),
                "quality": float(plan["qualities"][index]),
                "sharpness_score": float(plan["qualities"][index]),
                "coverage_gain": gains[index],
                "gain": gains[index],
                "focus_map_path": None,
                "transform": transforms[index],
                "error": plan["errors"].get(index),
            }
            per_image.append(row)
            self._persist_image(group, item, row)

        first_item = items[0]
        first_original = _item_value(first_item, "original_path", default=paths[0])
        cancelled = cancel_event.is_set()
        if len(items) <= 1:
            merge_status = "NO_MERGE_SINGLE"
        elif len(selected_indices) <= 1:
            merge_status = "NO_MERGE_REPEATED"
        elif len(items) - len(plan["errors"]) < self._minimum_stack_group_size():
            merge_status = "NO_MERGE_TOO_SMALL"
        else:
            merge_status = "READY_FOR_MERGE"
        confidence = min((plan["correlations"][index] for index in selected_indices), default=0.0)
        preview_reference = str(paths[plan["reference_index"]])
        result = {
            "group_id": gid,
            "all_images": list(items), "items": list(items), "image_records": list(items),
            "selected_indices": selected_indices, "selected_paths": selected_paths,
            "selected_images": selected_paths,
            "first_original_image": first_item, "first_original_path": str(first_original),
            "selected_count": len(selected_indices), "image_count": len(items),
            "coverage": float(plan["coverage"]), "confidence": float(confidence),
            "needs_merge": not cancelled and merge_status == "READY_FOR_MERGE",
            "status": "CANCELLED" if cancelled else merge_status,
            "merge_status": merge_status, "cancelled": cancelled,
            "quality_scores": list(plan["qualities"]), "qualities": list(plan["qualities"]),
            "coverage_gains": gains, "gains": gains,
            "selection_reasons": reasons, "reasons": reasons,
            "per_image": per_image, "analysis_records": per_image,
            "errors": dict(plan["errors"]), "focus_map_paths": [None] * len(items),
            "capture_order": capture_order,
            "preview_reference": preview_reference,
            "preview_reference_index": int(plan["reference_index"]),
            "pairwise_analysis_summary": {"strategy": "established_whole_frame_ecc"},
            "alignment_order": alignment_order,
            "alignment_order_indices": alignment_order_indices,
            "alignment_order_confidence": order_result["alignment_order_confidence"],
            "alignment_order_diagnostics": order_result["alignment_order_diagnostics"],
            "alignment_order_fallback_used": order_result["alignment_order_fallback_used"],
            "preview_transforms": transforms,
            "analysis_shapes": [list(plan["preview_shape"])] * len(items),
            "reference_analysis_shape": list(plan["preview_shape"]),
            "selection_plan_cache_hit": cache_hit,
            "selection_decode_count": 0 if cache_hit else int(plan.get("decode_count", 0)),
            "selection_frame_cache_peak_bytes": (
                0 if cache_hit else int(plan.get("frame_cache_peak_bytes", 0))
            ),
            "analysis_workers_requested": int(
                plan.get("analysis_workers_requested", requested_workers)
            ),
            "analysis_workers_effective": int(
                plan.get("analysis_workers_effective", 0 if cache_hit else effective_workers)
            ),
            "opencv_threads": int(plan.get("opencv_threads", cv2.getNumThreads())),
            "quality_scan_seconds": float(plan.get("quality_scan_seconds", 0.0)),
            "registration_seconds": float(plan.get("registration_seconds", 0.0)),
        }
        self._persist_status(
            group, "CANCELLED" if cancelled else ("SELECTED" if result["needs_merge"] else "CLASSIFIED"),
            confidence=result["confidence"], coverage=result["coverage"],
            selected_count=len(selected_indices), image_count=len(items),
            preview_reference=preview_reference, alignment_order=alignment_order,
            alignment_order_confidence=result["alignment_order_confidence"],
            alignment_order_fallback_used=result["alignment_order_fallback_used"],
            pairwise_analysis_summary=result["pairwise_analysis_summary"],
            start_index=_group_value(group, "start_index", default=0),
            end_index=_group_value(group, "end_index", default=max(0, len(items) - 1)),
        )
        return result

    @staticmethod
    def _restore_selection_plan(
        payload: Any, image_count: int,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Validate and normalise a path-free JSON selection cache record."""

        if not isinstance(payload, Mapping):
            return None
        raw_plan = payload.get("plan")
        raw_order = payload.get("order")
        if not isinstance(raw_plan, Mapping) or not isinstance(raw_order, Mapping):
            return None
        try:
            selected = [int(value) for value in raw_plan["selected_indices"]]
            reference = int(raw_plan["reference_index"])
            matrices = [
                [[float(value) for value in row] for row in matrix]
                for matrix in raw_plan["matrices"]
            ]
            correlations = [float(value) for value in raw_plan["correlations"]]
            qualities = [float(value) for value in raw_plan["qualities"]]
            preview_shape = [int(value) for value in raw_plan["preview_shape"]]
            errors = {int(key): str(value) for key, value in dict(raw_plan["errors"]).items()}
            gains = {int(key): float(value) for key, value in dict(raw_plan["gains"]).items()}
            alignment_order = [int(value) for value in raw_order["alignment_order"]]
            coverage = float(raw_plan["coverage"])
            order_confidence = float(raw_order["alignment_order_confidence"])
            if (
                image_count < 1
                or len(matrices) != image_count
                or any(len(matrix) != 2 or any(len(row) != 3 for row in matrix) for matrix in matrices)
                or any(not math.isfinite(value) for matrix in matrices for row in matrix for value in row)
                or len(correlations) != image_count
                or any(not math.isfinite(value) for value in correlations)
                or len(qualities) != image_count
                or any(not math.isfinite(value) for value in qualities)
                or len(preview_shape) != 2
                or any(value < 1 for value in preview_shape)
                or not 0 <= reference < image_count
                or any(not 0 <= value < image_count for value in selected)
                or len(set(selected)) != len(selected)
                or reference not in selected
                or any(not 0 <= value < image_count for value in errors)
                or any(not 0 <= value < image_count for value in gains)
                or any(not math.isfinite(value) for value in gains.values())
                or not math.isfinite(coverage)
                or not 0.0 <= coverage <= 1.0
                or set(alignment_order) != set(selected)
                or len(set(alignment_order)) != len(alignment_order)
                or not math.isfinite(order_confidence)
            ):
                return None
            plan = dict(raw_plan)
            plan.update({
                "selected_indices": selected,
                "reference_index": reference,
                "matrices": matrices,
                "correlations": correlations,
                "qualities": qualities,
                "preview_shape": preview_shape,
                "errors": errors,
                "gains": gains,
                "coverage": coverage,
            })
            order = dict(raw_order)
            order["alignment_order"] = alignment_order
            order["alignment_order_confidence"] = order_confidence
            order["alignment_order_fallback_used"] = bool(
                raw_order["alignment_order_fallback_used"]
            )
            if not isinstance(order.get("alignment_order_diagnostics"), Mapping):
                return None
            return plan, order
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    def _classify_without_focus(
        self,
        group: Any,
        items: list[Any],
        *,
        reason: str,
        merge_status: str,
    ) -> dict[str, Any]:
        """Return a complete no-merge result without decoding image pixels."""

        gid = _group_id(group)
        total = len(items)
        paths: list[Path | None] = [_path_for_item(item) for item in items]
        confidence = float(_group_value(group, "confidence", default=0.0) or 0.0)
        per_image = [
            {
                "index": index,
                "image": item,
                "path": str(paths[index]) if paths[index] is not None else "",
                "selected": False,
                "selection_reason": reason,
                "reason": reason,
                "quality_score": 0.0,
                "quality": 0.0,
                "sharpness_score": 0.0,
                "coverage_gain": 0.0,
                "gain": 0.0,
                "scene_score": confidence,
                "scene_hash": _item_value(item, "scene_hash", default=None),
                "focus_map_path": None,
                "transform": None,
                "error": None,
            }
            for index, item in enumerate(items)
        ]
        for item, row in zip(items, per_image):
            self._persist_image(group, item, row)

        first_item = items[0]
        first_path = paths[0]
        first_original = _item_value(first_item, "original_path", default=first_path)
        result = {
            "group_id": gid,
            "all_images": list(items),
            "items": list(items),
            "image_records": list(items),
            "selected_indices": [],
            "selected_paths": [],
            "selected_images": [],
            "capture_order": [str(path) for path in paths if path is not None],
            "preview_reference": str(first_path) if first_path is not None else 0,
            "preview_reference_index": 0,
            "alignment_order": [],
            "alignment_order_indices": [],
            "alignment_order_confidence": 1.0,
            "alignment_order_diagnostics": {"code": "NO_MERGE"},
            "alignment_order_fallback_used": False,
            "pairwise_analysis_summary": {
                "edge_count": 0,
                "strategy": "skipped_before_focus_analysis",
            },
            "first_original_image": first_item,
            "first_original_path": str(first_original) if first_original is not None else "",
            "selected_count": 0,
            "image_count": total,
            "coverage": 0.0,
            "confidence": confidence,
            "merge_status": merge_status,
            "needs_merge": False,
            "status": merge_status,
            "cancelled": False,
            "quality_scores": [0.0] * total,
            "qualities": [0.0] * total,
            "coverage_gains": {index: 0.0 for index in range(total)},
            "gains": {index: 0.0 for index in range(total)},
            "selection_reasons": {index: reason for index in range(total)},
            "reasons": {index: reason for index in range(total)},
            "per_image": per_image,
            "analysis_records": per_image,
            "errors": {},
            "focus_map_paths": [None] * total,
            "preview_transforms": [None] * total,
            "analysis_shapes": [None] * total,
            "reference_analysis_shape": None,
            "selection_plan_cache_hit": False,
            "selection_decode_count": 0,
            "selection_frame_cache_peak_bytes": 0,
            "analysis_workers_requested": self._requested_focus_analysis_workers(),
            "analysis_workers_effective": 0,
            "opencv_threads": 0,
            "quality_scan_seconds": 0.0,
            "registration_seconds": 0.0,
            "analysis_skipped": True,
            "analysis_skip_reason": merge_status,
        }
        self.logger.info(
            "focus analysis skipped group_id=%s image_count=%s status=%s reason=%s",
            gid, total, merge_status, reason,
        )
        self._persist_status(
            group,
            "CLASSIFIED",
            confidence=confidence,
            coverage=0.0,
            selected_count=0,
            image_count=total,
            start_index=_group_value(group, "start_index", default=0),
            end_index=_group_value(group, "end_index", default=max(0, total - 1)),
        )
        return result

    @timed("selection_inclusive")
    def analyze_group(self, group: Any, *, cancel_event: threading.Event | None = None,
                      progress_callback: Callable[..., Any] | None = None,
                      progress: Callable[..., Any] | None = None) -> dict[str, Any]:
        """Analyze and select one group, returning a merge-compatible mapping."""

        cancel_event = cancel_event or threading.Event()
        if progress_callback is None:
            progress_callback = progress
        items = _items_for_group(group, self.database)
        gid = _group_id(group)
        if not items:
            raise ValueError(f"group {gid!r} contains no images")
        total = len(items)
        minimum_group_size = self._minimum_stack_group_size()
        if total < minimum_group_size:
            if total == 1:
                reason = (
                    f"scene has one image; at least {minimum_group_size} similar images "
                    "are required for fusion"
                )
                merge_status = "NO_MERGE_SINGLE"
            else:
                reason = (
                    f"scene has {total} image(s); at least {minimum_group_size} "
                    "similar images are required for fusion"
                )
                merge_status = "NO_MERGE_TOO_SMALL"
            return self._classify_without_focus(
                group, items, reason=reason, merge_status=merge_status,
            )
        minimum_similarity = self._group_minimum_scene_similarity(group)
        unstable = (
            minimum_similarity is not None
            and self._minimum_stack_stability() > 0
            and minimum_similarity < self._minimum_stack_stability()
        )
        if not unstable:
            return self._analyze_established_scene_selection(
                group, items, cancel_event, progress_callback,
            )
        self._persist_status(group, "ANALYZING_FOCUS", image_count=len(items))
        paths: list[Path | None] = [_path_for_item(item) for item in items]
        minimum_stability = self._minimum_stack_stability()
        minimum_similarity = self._group_minimum_scene_similarity(group)
        too_small = total < minimum_group_size
        too_unstable = (
            minimum_stability > 0.0
            and minimum_similarity is not None
            and minimum_similarity < minimum_stability
        )
        if too_small or too_unstable:
            if too_small:
                reason = (
                    f"scene has {total} image(s); at least {minimum_group_size} similar images are required for fusion"
                )
                merge_status = "NO_MERGE_TOO_SMALL"
            else:
                reason = (
                    "scene changes too much between adjacent frames "
                    f"(minimum similarity {minimum_similarity:.3f}; required {minimum_stability:.3f})"
                )
                merge_status = "NO_MERGE_UNSTABLE"
            return self._classify_without_focus(
                group, items, reason=reason, merge_status=merge_status,
            )
        focus_config = self._focus_config()
        focus_cache_namespace = _focus_map_cache_namespace(focus_config)
        quality_config = self._quality_config()
        maps: list[Any | None] = []
        map_paths: list[Path | None] = []
        registrations: list[RegistrationResult | None] = []
        analysis_shapes: list[tuple[int, int] | None] = [None] * total
        qualities: list[ImageQuality] = []
        image_errors: list[str | None] = [None] * total
        scene_scores: list[float | None] = [None] * total
        # Reference choice is a separate whole-group pre-registration stage.
        # It deliberately precedes focus-map construction and is not reused as
        # the later Hugin input-order decision.
        from .alignment_order import choose_preview_reference
        preview_edge = int(_setting(self.config, "scene_preview_long_edge", 512, "preview_long_edge"))
        preview_images: list[Any | None] = []
        for item in items:
            try:
                if self.loader is not None:
                    try:
                        preview_images.append(self.loader(item, max_long_edge=preview_edge))
                    except TypeError:
                        preview_images.append(self.loader(item))
                else:
                    preview_images.append(load_image(item, max_long_edge=preview_edge))
            except Exception as exc:
                self.logger.warning("Preview pre-registration load failed for %s: %s", _path_for_item(item) or item, exc)
                preview_images.append(None)
        preview_positions = [index for index, image in enumerate(preview_images) if image is not None]
        preview_edges = self.analyze_pairwise_registration(
            [preview_images[index] for index in preview_positions],
        ) if preview_positions else []
        local_reference, preview_diagnostics = choose_preview_reference(len(preview_positions), preview_edges)
        preview_reference_index = preview_positions[local_reference] if preview_positions else 0
        preview_edge_rows = []
        from dataclasses import asdict
        for edge in preview_edges:
            row = asdict(edge)
            row["left"] = preview_positions[edge.left]
            row["right"] = preview_positions[edge.right]
            preview_edge_rows.append(row)
        preview_diagnostics["edges"] = preview_edge_rows

        reference: Any = None
        reference_shape: tuple[int, int] | None = None
        reference_map_shape: tuple[int, int] | None = None
        comparisons = list(_group_value(group, "comparisons", default=()) or ())
        try:
            reference = self._load_analysis(items[preview_reference_index])
            reference_shape = tuple(int(v) for v in reference.shape[:2])
        except Exception as exc:
            self.logger.warning("Chosen preview reference could not be decoded: %s", exc)
            reference = None
        for index, item in enumerate(items):
            path = paths[index]
            if cancel_event.is_set():
                break
            self._emit_progress(progress_callback, group, path, index, total)
            source = None
            registration: RegistrationResult | None = None
            try:
                source = reference if index == preview_reference_index and reference is not None else self._load_analysis(item)
                analysis_shapes[index] = tuple(int(v) for v in source.shape[:2])
                reference_candidate = index == preview_reference_index or reference is None
                candidate_reference_shape = (
                    tuple(int(v) for v in source.shape[:2])
                    if reference_candidate else reference_shape
                )
                if index == preview_reference_index or reference is None:
                    registration = _identity_result()
                else:
                    registration = self.align_preview(reference, source)
                raw_map = None
                use_cache = bool(_setting(self.config, "cache_focus_maps", True))
                if path is not None and use_cache and self.cache is not None:
                    raw_map, map_path = _load_or_create_cached_map(
                        self.cache, path,
                        lambda source=source: compute_focus_map(source, focus_config),
                        namespace=focus_cache_namespace,
                    )
                else:
                    raw_map = compute_focus_map(source, focus_config)
                    # ``cache_focus_maps=False`` means there is no durable
                    # artifact to advertise to the manifest/database.
                    map_path = None
                raw_map = _focus_array(raw_map)
                target_map_shape = reference_map_shape
                if target_map_shape is None:
                    # Do not let a stale cache entry silently override the
                    # current output resolution.  Cache files are keyed by
                    # source fingerprint for reuse, while the requested map
                    # shape remains a run-time analysis setting.
                    target_map_shape = _compact_shape(
                        candidate_reference_shape or tuple(int(v) for v in source.shape[:2]),
                        int(_setting(focus_config, "output_long_edge", 512)),
                    )
                aligned_map = _warp_focus_map(
                    raw_map, registration, tuple(int(v) for v in source.shape[:2]), target_map_shape,
                    reference_shape=candidate_reference_shape,
                )
                quality = score_image(source, aligned_map, registration, quality_config)
                if reference_candidate:
                    # A corrupt/unsupported first frame must not become the
                    # registration anchor for every later frame.  Promote it
                    # only after map and quality stages have succeeded.
                    reference = source
                    preview_reference_index = index
                    reference_shape = candidate_reference_shape
                    reference_map_shape = target_map_shape
                maps.append(aligned_map)
                map_paths.append(map_path)
                qualities.append(quality)
                registrations.append(registration)
                if index < len(comparisons):
                    scene_scores[index + 1] = float(
                        _item_value(comparisons[index], "confidence", default=0.0) or 0.0
                    )
            except Exception as exc:
                # A corrupt frame should remain archivable and visible in the
                # manifest; continue the group with a zero evidence map.
                self.logger.warning("Focus analysis failed for %s: %s", path or item, exc)
                image_errors[index] = str(exc)
                maps.append(None)
                map_paths.append(
                    _cache_path(self.cache, path, focus_cache_namespace)
                    if path is not None and self.cache is not None else None
                )
                registrations.append(registration or _identity_result())
                qualities.append(ImageQuality(score=0.0))
            finally:
                # ``source`` is deliberately local; releasing it here keeps a
                # full analysis frame out of the next iteration.
                source = None
                # ``maps`` owns the compact aligned result.  Drop transient
                # raw/aligned aliases so a cache decode and the next frame do
                # not overlap longer than necessary.
                raw_map = None
                aligned_map = None
        if len(maps) < total:
            # Cancellation returns a partial, explicitly marked result rather
            # than silently pretending the group completed.
            for index in range(len(maps), total):
                maps.append(None)
                map_paths.append(None)
                registrations.append(_identity_result())
                qualities.append(ImageQuality(score=0.0))
                image_errors[index] = "cancelled"
        if not maps:
            raise ValueError(f"group {gid!r} contains no analyzable images")
        import numpy as np
        # Keep map dimensions consistent and fill decode failures with zero
        # evidence after the first successful map establishes the shape.
        valid_map = next((value for value in maps if value is not None), None)
        if valid_map is None:
            valid_map = np.zeros((1, 1), dtype=np.float16)
        valid_map = np.asarray(valid_map)
        maps = [valid_map.copy() if value is None else value for value in maps]
        quality_scores = [float(value.score) for value in qualities]
        # Decode failures remain in the manifest/archive set, but never win a
        # focus representative while at least one image was analyzable.  Map
        # clustering then uses local candidate positions and is translated
        # back to stable group indices for coverage/reason persistence.
        candidate_indices = [index for index in range(total) if not image_errors[index]]
        if not candidate_indices:
            candidate_indices = list(range(total))
        candidate_maps = [maps[index] for index in candidate_indices]
        candidate_qualities = [quality_scores[index] for index in candidate_indices]
        dedup_local = self.remove_duplicate_focus(candidate_maps, candidate_qualities)
        representatives = [candidate_indices[index] for index in dedup_local.representative_indices]
        if not representatives:
            representatives = [max(candidate_indices, key=lambda i: quality_scores[i])]
        coverage = self.select_frames(
            [maps[index] for index in representatives],
            [quality_scores[index] for index in representatives],
            candidate_indices=representatives,
        )
        selected_indices = list(coverage.selected_indices)
        if not selected_indices:
            selected_indices = [representatives[0]]
        # Scene grouping has already established that these frames share the
        # same composition.  Focus-map metrics are intentionally aggressive
        # about duplicate removal, but for a captured bracket they can score
        # every frame as redundant and incorrectly suppress fusion.  Keep the
        # most different remaining compact map(s) up to the configured floor.
        required_inputs = min(len(candidate_indices), self._minimum_stack_images())
        minimum_stack_additions: dict[int, float] = {}
        while len(selected_indices) < required_inputs:
            remaining = [index for index in candidate_indices if index not in selected_indices]
            if not remaining:
                break

            def difference_from_selection(index: int) -> tuple[float, float, int]:
                distance = min(
                    focus_map_distance(maps[index], maps[selected])
                    for selected in selected_indices
                )
                # Prefer the higher-quality frame only when its focus-map
                # difference is tied, then retain capture order determinism.
                return float(distance), float(quality_scores[index]), -int(index)

            extra = max(remaining, key=difference_from_selection)
            minimum_stack_additions[extra] = difference_from_selection(extra)[0]
            selected_indices.append(extra)
        # A single frame trivially covers the only available scene, even if it
        # is texture-free and its map is all zeros.
        if total == 1 or len(representatives) == 1:
            final_coverage = 1.0
        else:
            final_coverage = float(coverage.coverage)
        selected_set = set(selected_indices)
        duplicate_reasons = {
            candidate_indices[index]: reason
            for index, reason in dedup_local.reasons.items()
        }
        reasons: dict[int, str] = {}
        gains: dict[int, float] = {}
        for index in range(total):
            if image_errors[index]:
                reasons[index] = f"analysis failed: {image_errors[index]}"
                gains[index] = 0.0
            elif index in selected_set:
                if index in minimum_stack_additions:
                    reasons[index] = (
                        "retained as the most distinct frame for minimum focus-stack input count"
                    )
                    gains[index] = float(minimum_stack_additions[index])
                else:
                    reasons[index] = coverage.reasons.get(index, "coverage")
                    gains[index] = float(coverage.gains.get(index, 0.0))
            elif index in duplicate_reasons:
                reasons[index] = duplicate_reasons[index]
                gains[index] = 0.0
            else:
                reasons[index] = coverage.reasons.get(index, "redundant; no material new coverage")
                gains[index] = float(coverage.gains.get(index, 0.0))
        per_image: list[dict[str, Any]] = []
        analysis_records: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            selected = index in selected_set
            q = qualities[index]
            scene_score = scene_scores[index]
            if scene_score is None:
                scene_score = float(_group_value(group, "confidence", default=0.0) or 0.0)
            scene_hash = _item_value(
                item, "scene_hash", default=_group_value(group, "scene_hash", default=None)
            )
            row = {
                "index": index,
                "image": item,
                "path": str(paths[index]) if paths[index] is not None else "",
                "selected": selected,
                "selection_reason": reasons[index],
                "reason": reasons[index],
                "quality_score": float(q.score),
                "quality": float(q.score),
                "sharpness_score": float(q.sharpness),
                "coverage_gain": float(gains[index]),
                "gain": float(gains[index]),
                "scene_score": float(scene_score),
                "scene_hash": scene_hash,
                "focus_map_path": str(map_paths[index]) if map_paths[index] is not None else None,
                "transform": _matrix_json(registrations[index]),
                "error": image_errors[index],
            }
            per_image.append(row)
            analysis_records.append(row)
            self._persist_image(group, item, row)
        selected_paths = [str(paths[index]) for index in selected_indices if paths[index] is not None]
        selected_preview_positions = [index for index in selected_indices if preview_images[index] is not None]
        selected_preview_positions.sort(key=lambda index: (
            _item_value(items[index], "sequence_index", default=index) is None,
            _item_value(items[index], "sequence_index", default=index),
            index,
        ))
        order_result = self.build_alignment_order(
            [preview_images[index] for index in selected_preview_positions],
            capture_order=selected_preview_positions,
        ) if selected_preview_positions else {
            "alignment_order": list(selected_indices),
            "alignment_order_confidence": 0.0,
            "alignment_order_diagnostics": {"code": "ALIGNMENT_ORDER_UNCERTAIN", "edges": []},
            "alignment_order_fallback_used": True,
        }
        alignment_order_indices = list(order_result["alignment_order"])
        # Any selected image whose preview failed remains eligible and is
        # appended in SQLite capture order; Hugin can still attempt it.
        missing_order = [index for index in selected_indices if index not in alignment_order_indices]
        missing_order.sort(key=lambda index: (
            _item_value(items[index], "sequence_index", default=index) is None,
            _item_value(items[index], "sequence_index", default=index),
            index,
        ))
        alignment_order_indices.extend(missing_order)
        alignment_order_paths = [str(paths[index]) for index in alignment_order_indices if paths[index] is not None]
        capture_indices = sorted(range(total), key=lambda index: (
            _item_value(items[index], "sequence_index", default=index) is None,
            _item_value(items[index], "sequence_index", default=index),
            index,
        ))
        capture_order = [str(paths[index]) for index in capture_indices if paths[index] is not None]
        first_item = items[0]
        first_path = paths[0]
        first_original_value = _item_value(first_item, "original_path", default=first_path)
        if len(items) <= 1:
            merge_status = "NO_MERGE_SINGLE"
        elif len(selected_indices) <= 1:
            merge_status = "NO_MERGE_REPEATED"
        else:
            merge_status = "READY_FOR_MERGE"
        cancelled = cancel_event.is_set()
        result: dict[str, Any] = {
            "group_id": gid,
            "all_images": list(items),
            "items": list(items),
            "image_records": list(items),
            "selected_indices": selected_indices,
            "selected_paths": selected_paths,
            "selected_images": selected_paths,
            "capture_order": capture_order,
            "preview_reference": str(paths[preview_reference_index]) if paths[preview_reference_index] is not None else preview_reference_index,
            "preview_reference_index": preview_reference_index,
            "pairwise_analysis_summary": preview_diagnostics,
            "alignment_order": alignment_order_paths,
            "alignment_order_indices": alignment_order_indices,
            "alignment_order_confidence": order_result["alignment_order_confidence"],
            "alignment_order_diagnostics": order_result["alignment_order_diagnostics"],
            "alignment_order_fallback_used": order_result["alignment_order_fallback_used"],
            "preview_transforms": [_matrix_json(value) for value in registrations],
            "analysis_shapes": [list(value) if value is not None else None for value in analysis_shapes],
            "reference_analysis_shape": list(reference_shape) if reference_shape is not None else None,
            "first_original_image": first_item,
            "first_original_path": str(first_original_value) if first_original_value is not None else "",
            "selected_count": len(selected_indices),
            "image_count": len(items),
            "coverage": final_coverage,
            "confidence": float(_group_value(group, "confidence", default=0.0) or 0.0),
            "merge_status": merge_status,
            "needs_merge": merge_status == "READY_FOR_MERGE",
            "status": "CANCELLED" if cancelled else merge_status,
            "cancelled": cancelled,
            "quality_scores": quality_scores,
            "qualities": quality_scores,
            "coverage_gains": gains,
            "gains": gains,
            "selection_reasons": reasons,
            "reasons": reasons,
            "per_image": per_image,
            "analysis_records": analysis_records,
            "errors": {index: error for index, error in enumerate(image_errors) if error},
            "focus_map_paths": [str(path) if path is not None else None for path in map_paths],
        }
        durable_status = (
            "CANCELLED" if cancelled
            else ("SELECTED" if result["needs_merge"] else "CLASSIFIED")
        )
        self._persist_status(
            group, durable_status,
            confidence=result["confidence"], coverage=final_coverage,
            selected_count=len(selected_indices), image_count=len(items),
            preview_reference=result["preview_reference"],
            alignment_order=result["alignment_order"],
            alignment_order_confidence=result["alignment_order_confidence"],
            alignment_order_fallback_used=result["alignment_order_fallback_used"],
            pairwise_analysis_summary=result["pairwise_analysis_summary"],
            start_index=_group_value(group, "start_index", default=0),
            end_index=_group_value(group, "end_index", default=max(0, len(items) - 1)),
        )
        return result

    analyze = analyze_group
    process = analyze_group
    __call__ = analyze_group


def analyze_group(group: Any, config: Any = None, *, database: Any = None,
                  cache: Any = None, analysis_cache: Any = None,
                  cancel_event: threading.Event | None = None,
                  progress_callback: Callable[..., Any] | None = None,
                  progress: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Convenience wrapper around :class:`GroupAnalyzer`."""

    return GroupAnalyzer(
        config, database=database, cache=cache, analysis_cache=analysis_cache,
    ).analyze_group(
        group, cancel_event=cancel_event, progress_callback=progress_callback,
        progress=progress,
    )


__all__ = [
    "GroupAnalyzerConfig", "GroupAnalysisCancelled", "GroupAnalyzer",
    "CACHE_ALGORITHM_VERSION", "analyze_group",
]

