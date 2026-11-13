"""Production group-level analysis orchestration.

``GroupAnalyzer`` bridges scene grouping and the bounded analysis/merge
pipeline. Production selection uses calibrated whole-frame ECC and a bounded
preview cache. Hugin ordering is optional and cached independently of selection;
standalone stage helpers remain available for library callers.
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
from typing import Any, Callable, Mapping, Sequence

from .coverage_selector import CoverageConfig, select_focus_images
from .focus_cluster import FocusClusterConfig, deduplicate_focus_maps
from .focus_map import FocusMapConfig, compute_focus_map
from .quality import QualityConfig, score_image
from .registration import RegistrationConfig, register_images, resolve_image_path
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
    minimum_stack_group_size: int = 4
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
        Optional ``DiskCache``-compatible object retained for library callers.
        Production whole-frame selection uses ``plan_cache`` instead.
    require_alignment_order:
        Compute Hugin's registration-friendly input order. Defaults to True
        for standalone callers; the application disables it for Quality/Fast.
    """

    def __init__(self, config: Any = None, *, database: Any = None,
                 cache: Any = None, analysis_cache: Any = None,
                 plan_cache: Any = None,
                 require_alignment_order: bool = True,
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
        self.require_alignment_order = bool(require_alignment_order)
        self.logger = logger or logging.getLogger(__name__)
        self.loader = loader
        if focus_analysis_workers is not None and not 1 <= int(focus_analysis_workers) <= 10:
            raise ValueError("focus_analysis_workers must be between 1 and 10")
        if focus_analysis_workers_requested is not None and not 0 <= int(focus_analysis_workers_requested) <= 10:
            raise ValueError("focus_analysis_workers_requested must be between 0 and 10")
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
        import numpy as np
        from .plan_cache import (
            SELECTION_PLAN_VERSION, build_plan_key, json_compatible,
        )
        from .whole_frame_selection import select_whole_frame
        from .alignment_order import ORDERING_PLAN_VERSION
        from ..utils.image_io import load_rgb

        paths = [_path_for_item(item) for item in items]
        if any(path is None for path in paths):
            raise ValueError("every group image must expose a path")
        requested_workers, effective_workers = self._effective_focus_analysis_workers(len(paths))
        path_items = {os.path.normcase(os.path.abspath(str(path))): item for path, item in zip(paths, items)}

        def load_path(path: Any, edge: int) -> Any:
            item = path_items.get(os.path.normcase(os.path.abspath(str(path))), path)
            if self.loader is not None:
                try:
                    return self.loader(item, max_long_edge=edge)
                except TypeError:
                    return self.loader(item)
            return load_rgb(path, edge)

        gid = _group_id(group)
        self._persist_status(group, "ANALYZING_FOCUS", image_count=len(items))
        if cancel_event.is_set():
            raise RuntimeError("selection cancelled")

        capture_indices = sorted(range(len(items)), key=lambda index: (
            _item_value(items[index], "sequence_index", default=index) is None,
            _item_value(items[index], "sequence_index", default=index), index,
        ))
        capture_order = [str(paths[index]) for index in capture_indices]

        cache_key: str | None = None
        cache_hit = False
        plan: dict[str, Any] | None = None
        order_result: dict[str, Any] | None = None
        order_refresh_needed = False
        if self.plan_cache is not None:
            try:
                cache_key = build_plan_key(
                    "selection",
                    SELECTION_PLAN_VERSION,
                    paths,
                    {
                        "selection_edge": 1280,
                        "ordering_preview_edge": 512,
                        "registration": self._registration_config(),
                        "capture_indices": capture_indices,
                    },
                )
                payload = self.plan_cache.get_cached_plan(
                    "selection", cache_key, algorithm_version=SELECTION_PLAN_VERSION,
                )
                restored = self._restore_selection_plan(payload, len(items))
                if restored is not None:
                    plan, order_result = restored
                    cache_hit = True
                    order_refresh_needed = self.require_alignment_order and (
                        order_result is None or payload.get("ordering_version") != ORDERING_PLAN_VERSION
                    )
            except Exception:
                self.logger.warning(
                    "selection plan cache read failed group_id=%s", gid, exc_info=True,
                )
                cache_key = None

        if cache_hit:
            assert plan is not None
            plan.update({
                "analysis_workers_requested": requested_workers,
                "analysis_workers_effective": 0,
                "opencv_threads": int(cv2.getNumThreads()),
                "quality_scan_seconds": 0.0,
                "registration_seconds": 0.0,
            })
            for index, path in enumerate(paths):
                if cancel_event.is_set():
                    raise RuntimeError("selection cancelled")
                self._emit_progress(progress_callback, group, path, index, len(paths))
            ordering_preview_decodes = 0
            if order_refresh_needed:
                selected_set_for_order = set(plan["selected_indices"])
                selected_capture = [index for index in capture_indices if index in selected_set_for_order]
                selected_previews = [load_path(paths[index], 512) for index in selected_capture]
                ordering_preview_decodes = len(selected_previews)
                order_result = self.build_alignment_order(
                    selected_previews, capture_order=selected_capture,
                )
            self.logger.info(
                "selection plan cache status=hit group_id=%s key=%s decode_count=0 "
                "ordering_preview_decodes=%s frame_cache_peak_bytes=0 "
                "requested_workers=%s effective_workers=0 opencv_threads=%s",
                gid, cache_key, ordering_preview_decodes, requested_workers,
                cv2.getNumThreads(),
            )
        else:
            plan = select_whole_frame(
                paths,
                cancel_event=cancel_event,
                loader=load_path,
                frame_cache_bytes=int(_setting(
                    self.config, "selection_frame_cache_bytes", 400 * 1024**2,
                )),
                workers=effective_workers,
                requested_workers=requested_workers,
                progress=lambda current, total: self._emit_progress(
                    progress_callback, group, paths[current - 1], current - 1, total,
                ),
            )
            if self.require_alignment_order:
                selected_set_for_order = set(plan["selected_indices"])
                selected_capture = [index for index in capture_indices if index in selected_set_for_order]
                selected_previews = [load_path(paths[index], 512) for index in selected_capture]
                order_result = self.build_alignment_order(
                    selected_previews, capture_order=selected_capture,
                )
            self.logger.info(
                "selection plan cache status=miss group_id=%s key=%s decode_count=%s "
                "frame_cache_enabled=%s frame_cache_peak_bytes=%s requested_workers=%s "
                "effective_workers=%s opencv_threads=%s quality_scan_seconds=%.6f "
                "registration_seconds=%.6f",
                gid, cache_key, plan.get("decode_count", 0),
                plan.get("frame_cache_enabled", False), plan.get("frame_cache_peak_bytes", 0),
                plan.get("analysis_workers_requested", requested_workers),
                plan.get("analysis_workers_effective", effective_workers),
                plan.get("opencv_threads", cv2.getNumThreads()),
                plan.get("quality_scan_seconds", 0.0),
                plan.get("registration_seconds", 0.0),
            )
        if (not cache_hit or order_refresh_needed) and self.plan_cache is not None and cache_key is not None and not cancel_event.is_set():
            try:
                self.plan_cache.upsert_cached_plan(
                    "selection", cache_key, SELECTION_PLAN_VERSION,
                    {
                        "plan": {
                            key: json_compatible(value)
                            for key, value in plan.items()
                            if key not in {
                                "paths",
                                "analysis_workers_requested",
                                "analysis_workers_effective",
                                "opencv_threads",
                                "quality_scan_seconds",
                                "registration_seconds",
                            }
                        },
                        "order": json_compatible(order_result),
                        "ordering_version": ORDERING_PLAN_VERSION if order_result is not None else None,
                    },
                )
            except Exception:
                self.logger.warning(
                    "selection plan cache write failed group_id=%s", gid, exc_info=True,
                )
        if not self.require_alignment_order:
            order_result = {
                "alignment_order": [],
                "alignment_order_confidence": 0.0,
                "alignment_order_diagnostics": {"code": "NOT_REQUESTED", "edges": []},
                "alignment_order_fallback_used": False,
            }
        assert plan is not None and order_result is not None
        selected_indices = list(plan["selected_indices"])
        selected_set = set(selected_indices)
        selected_paths = [str(paths[index]) for index in selected_indices]

        # The established ECC matrices use inverse-warp convention. Publish
        # source-to-reference matrices for the quality backend.
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
    ) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
        """Restore selection independently of an optional Hugin ordering plan."""

        if not isinstance(payload, Mapping):
            return None
        raw_plan = payload.get("plan")
        if not isinstance(raw_plan, Mapping):
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
            coverage = float(raw_plan["coverage"])
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
            return plan, GroupAnalyzer._restore_order(payload.get("order"), selected)
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _restore_order(raw_order: Any, selected: Sequence[int]) -> dict[str, Any] | None:
        if not isinstance(raw_order, Mapping):
            return None
        try:
            alignment_order = [int(value) for value in raw_order["alignment_order"]]
            order_confidence = float(raw_order["alignment_order_confidence"])
            if (set(alignment_order) != set(selected)
                    or len(set(alignment_order)) != len(alignment_order)
                    or not math.isfinite(order_confidence)):
                return None
            order = dict(raw_order)
            order["alignment_order"] = alignment_order
            order["alignment_order_confidence"] = order_confidence
            order["alignment_order_fallback_used"] = bool(
                raw_order["alignment_order_fallback_used"]
            )
            if not isinstance(order.get("alignment_order_diagnostics"), Mapping):
                return None
            return order
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    def ensure_alignment_order(
        self, group: Any, analysis: Mapping[str, Any], *,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Complete recovered Hugin metadata without changing selection or geometry."""
        result = dict(analysis)
        if not self.require_alignment_order or not result.get("needs_merge", True):
            return result
        selected = list(result.get("selected_paths", ()))
        if len(selected) < 2:
            return result

        def key(path: Any) -> str:
            return os.path.normcase(os.path.abspath(os.fspath(path)))

        selected_keys = [key(path) for path in selected]
        ordered = list(result.get("alignment_order", ()) or ())
        if (len(ordered) == len(selected) and len(set(selected_keys)) == len(selected)
                and {key(path) for path in ordered} == set(selected_keys)):
            return result
        indices = list(result.get("selected_indices", range(len(selected))))
        if len(indices) != len(selected) or len(set(indices)) != len(indices):
            raise ValueError("recovered selected paths and indices disagree")
        capture = {key(path): index for index, path in enumerate(result.get("capture_order", selected))}
        positions = sorted(range(len(selected)), key=lambda i: capture.get(selected_keys[i], len(capture) + i))
        items = {
            key(path): item for item in _items_for_group(group, self.database)
            if (path := _path_for_item(item)) is not None
        }
        from ..utils.image_io import load_rgb
        event = cancel_event or threading.Event()
        previews = []
        for position in positions:
            if event.is_set():
                raise GroupAnalysisCancelled("ordering cancelled")
            path = selected[position]
            if self.loader is None:
                preview = load_rgb(path, 512)
            else:
                item = items.get(selected_keys[position], path)
                try:
                    preview = self.loader(item, max_long_edge=512)
                except TypeError:
                    preview = self.loader(item)
            previews.append(preview)
        order = self.build_alignment_order(previews, capture_order=[indices[i] for i in positions])
        if event.is_set():
            raise GroupAnalysisCancelled("ordering cancelled")
        by_index = dict(zip(indices, selected))
        result.update(order)
        result["alignment_order_indices"] = order["alignment_order"]
        result["alignment_order"] = [str(by_index[index]) for index in order["alignment_order"]]
        self._persist_status(
            group, "SELECTED", alignment_order=result["alignment_order"],
            alignment_order_confidence=result["alignment_order_confidence"],
            alignment_order_fallback_used=result["alignment_order_fallback_used"],
        )
        return result

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
        minimum_stability = self._minimum_stack_stability()
        if minimum_stability > 0 and minimum_similarity is not None and minimum_similarity < minimum_stability:
            self._persist_status(group, "ANALYZING_FOCUS", image_count=len(items))
            return self._classify_without_focus(
                group, items,
                reason=(
                    "scene changes too much between adjacent frames "
                    f"(minimum similarity {minimum_similarity:.3f}; required {minimum_stability:.3f})"
                ),
                merge_status="NO_MERGE_UNSTABLE",
            )
        return self._analyze_established_scene_selection(
            group, items, cancel_event, progress_callback,
        )

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
