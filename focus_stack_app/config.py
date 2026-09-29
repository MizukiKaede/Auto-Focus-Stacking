"""Central application configuration.

All tunable values used by the first version live in dataclasses in this
module.  The loader uses JSON from the standard library so opening a project
does not require optional dependencies.  Unknown keys are ignored when
loading an older/newer configuration file, while malformed values fall back
to their dataclass defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class AnalysisConfig:
    """Resolution and analysis thresholds.

    Analysis images are normally streamed by the algorithm workers.  The
    selection stage may retain one complete 1280px group only when its byte
    estimate fits the explicit limit below.
    """

    scene_preview_long_edge: int = 512
    focus_analysis_long_edge: int = 1600
    focus_cache_long_edge: int = 512
    # Focus bracketing can change enough local contrast that 0.88 breaks one
    # physical stack into several scenes.  This value is deliberately more
    # tolerant; the histogram/hash and confirmation checks in SceneDetector
    # still prevent a single weak comparison from joining unrelated shots.
    scene_similarity_threshold: float = 0.82
    geometric_inlier_threshold: float = 0.55
    # Focus maps from adjacent frames in a real stack often score above .98
    # even though each frame contributes detail at a different depth.  Only
    # collapse effectively identical maps by default.
    duplicate_focus_threshold: float = 0.995
    # A confirmed multi-frame scene is a candidate focus stack.  Retain at
    # least this many inputs so an over-confident focus-map comparison cannot
    # turn an entire captured bracket into ``NO_MERGE``.
    minimum_stack_images: int = 2
    # Short bursts are more likely to be accidental duplicates than a full
    # focus bracket.  They remain classified but never invoke fusion.
    minimum_stack_group_size: int = 3
    # Scene grouping can tolerate a brief interruption while discovering a
    # sequence.  Fusion cannot: every adjacent frame must remain this close
    # in low-frequency structure, otherwise the group is an action/change
    # sequence rather than a stable focus bracket.
    minimum_stack_stability: float = 0.98
    coverage_target: float = 0.995
    min_coverage_gain: float = 0.002
    # Coverage is relative to the best frame at each pixel.  A high threshold
    # prevents a generally soft frame from masquerading as full focus coverage.
    focus_threshold: float = 0.95
    max_preview_cache_items: int = 12
    max_focus_cache_items: int = 4
    # Reuse the exact 1280px arrays across quality/reference/ECC stages when
    # the whole group fits.  Zero preserves the historical streaming path.
    selection_frame_cache_bytes: int = 400 * 1024**2
    scene_confirmation_window: int = 2

    @property
    def preview_long_edge(self) -> int:
        """Short alias used by preview/scene workers."""

        return self.scene_preview_long_edge

    @property
    def analysis_long_edge(self) -> int:
        """Canonical V1 name for the 1600px focus-analysis resolution."""

        return self.focus_analysis_long_edge

    def __post_init__(self) -> None:
        if self.scene_preview_long_edge < 1:
            raise ValueError("scene_preview_long_edge must be positive")
        if self.focus_analysis_long_edge < 1:
            raise ValueError("focus_analysis_long_edge must be positive")
        if self.focus_cache_long_edge < 1:
            raise ValueError("focus_cache_long_edge must be positive")
        for name in (
            "scene_similarity_threshold",
            "geometric_inlier_threshold",
            "duplicate_focus_threshold",
            "coverage_target",
            "minimum_stack_stability",
            "focus_threshold",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.min_coverage_gain < 0:
            raise ValueError("min_coverage_gain cannot be negative")
        if int(self.minimum_stack_images) < 1:
            raise ValueError("minimum_stack_images must be at least one")
        if int(self.minimum_stack_group_size) < 1:
            raise ValueError("minimum_stack_group_size must be at least one")
        if self.max_preview_cache_items < 0 or self.max_focus_cache_items < 0:
            raise ValueError("cache limits cannot be negative")
        if int(self.selection_frame_cache_bytes) < 0:
            raise ValueError("selection_frame_cache_bytes cannot be negative")
        if self.scene_confirmation_window < 0:
            raise ValueError("scene_confirmation_window cannot be negative")


@dataclass(slots=True)
class OutputConfig:
    """Output encoding defaults.

    ``overwrite`` is false by design: an original JPEG must never be
    replaced by a generated result.
    """

    format: str = "jpg"
    jpeg_quality: int = 100
    jpeg_subsampling: int = 0
    output_suffix: str = "_stack"
    overwrite: bool = False

    def __post_init__(self) -> None:
        self.format = str(getattr(self.format, "value", self.format)).lower().lstrip(".")
        if self.format not in {"jpg", "jpeg", "tif", "tiff"}:
            raise ValueError("format must be jpg, jpeg, tif, or tiff")
        if not 1 <= int(self.jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if int(self.jpeg_subsampling) not in {0, 1, 2}:
            raise ValueError("jpeg_subsampling must be 0, 1, or 2")
        if not self.output_suffix:
            raise ValueError("output_suffix cannot be empty")


@dataclass(slots=True)
class RuntimeConfig:
    """Concurrency, safety and external-tool settings."""

    max_hugin_workers: int = 1
    parallel_pipeline: bool = True
    preserve_cache: bool = True
    min_available_memory_bytes: int = 2 * 1024**3
    min_available_memory_fraction: float = 0.10
    disk_safety_margin_bytes: int = 2 * 1024**3
    hugin_bin: str | None = None
    align_image_stack_path: str | None = None
    enfuse_path: str | None = None
    fusion_backend: str = "quality"
    crop_ratio_warning: float = 0.65
    crop_ratio_fail: float = 0.35
    crop_ratio_max: float = 1.75
    min_alignment_tiff_bytes: int = 128
    min_fusion_output_bytes: int = 128
    opencv_aligned_cache_bytes: int = 1024**3
    # Zero selects an adaptive CPU/memory budget.
    focus_analysis_workers: int = 0
    # Appended to preserve positional compatibility with earlier callers.
    aligned_tiff_cache_bytes: int = 512 * 1024**2

    def __post_init__(self) -> None:
        if not 1 <= int(self.max_hugin_workers) <= 2:
            raise ValueError("max_hugin_workers must be 1 or 2")
        if not 0 <= int(self.focus_analysis_workers) <= 8:
            raise ValueError("focus_analysis_workers must be between 0 and 8")
        self.focus_analysis_workers = int(self.focus_analysis_workers)
        if self.min_available_memory_bytes < 0:
            raise ValueError("min_available_memory_bytes cannot be negative")
        if not 0 <= float(self.min_available_memory_fraction) <= 1:
            raise ValueError("min_available_memory_fraction must be between 0 and 1")
        if self.disk_safety_margin_bytes < 0:
            raise ValueError("disk_safety_margin_bytes cannot be negative")
        from .fusion_modes import normalize_fusion_backend
        self.fusion_backend = normalize_fusion_backend(self.fusion_backend)
        if not 0 < float(self.crop_ratio_fail) <= float(self.crop_ratio_warning) <= 1:
            raise ValueError("crop ratio thresholds must satisfy 0 < fail <= warning <= 1")
        if float(self.crop_ratio_max) < 1:
            raise ValueError("crop_ratio_max must be at least 1")
        if int(self.min_alignment_tiff_bytes) < 1 or int(self.min_fusion_output_bytes) < 1:
            raise ValueError("output validation byte thresholds must be positive")
        if int(self.opencv_aligned_cache_bytes) < 0:
            raise ValueError("opencv_aligned_cache_bytes cannot be negative")
        if int(self.aligned_tiff_cache_bytes) < 0:
            raise ValueError("aligned_tiff_cache_bytes cannot be negative")


@dataclass(slots=True)
class CacheConfig:
    """Project-local cache layout."""

    directory_name: str = ".stack_cache"
    preview_directory: str = "previews"
    focusmap_directory: str = "focusmaps"
    temp_directory: str = "temp"
    log_directory: str = "logs"
    database_filename: str = "database.sqlite"

    def __post_init__(self) -> None:
        for name in (
            "directory_name",
            "preview_directory",
            "focusmap_directory",
            "temp_directory",
            "log_directory",
            "database_filename",
        ):
            value = str(getattr(self, name))
            if not value or Path(value).name != value:
                raise ValueError(f"{name} must be a single relative path component")


@dataclass(slots=True)
class ScannerConfig:
    """Metadata scanner options."""

    extensions: tuple[str, ...] = (".jpg", ".jpeg")
    recursive: bool = False
    include_hidden: bool = False

    def __post_init__(self) -> None:
        normalized = tuple(
            sorted({str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}" for ext in self.extensions})
        )
        if not normalized:
            raise ValueError("at least one image extension is required")
        object.__setattr__(self, "extensions", normalized)


@dataclass(slots=True)
class AppConfig:
    """Complete configuration object passed between workers."""

    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "AppConfig":
        """Build config from a nested mapping, tolerating unknown keys."""

        value = value or {}

        def make(item_type: type[Any], key: str) -> Any:
            raw = value.get(key, {})
            if not isinstance(raw, Mapping):
                raw = {}
            allowed = {f.name for f in fields(item_type)}
            kwargs = {k: v for k, v in raw.items() if k in allowed}
            try:
                return item_type(**kwargs)
            except (TypeError, ValueError):
                # A hand-edited project config should not prevent opening the
                # UI.  Keep the entire section at safe defaults and let the
                # caller optionally surface a validation warning.
                return item_type()

        return cls(
            analysis=make(AnalysisConfig, "analysis"),
            output=make(OutputConfig, "output"),
            runtime=make(RuntimeConfig, "runtime"),
            cache=make(CacheConfig, "cache"),
            scanner=make(ScannerConfig, "scanner"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, Mapping):
            raise ValueError("configuration root must be a JSON object")
        return cls.from_mapping(data)

    def to_mapping(self) -> dict[str, Any]:
        data = asdict(self)
        # JSON has no tuple type; keeping lists also makes hand editing easier.
        data["scanner"]["extensions"] = list(self.scanner.extensions)
        return data

    def save(self, path: str | Path) -> None:
        """Atomically write JSON configuration, including parent creation."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_mapping(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)

    def with_overrides(self, **sections: Mapping[str, Any]) -> "AppConfig":
        """Return a copy with selected section values replaced."""

        data = self.to_mapping()
        for section, values in sections.items():
            if section not in data or not isinstance(values, Mapping):
                raise KeyError(section)
            data[section].update(values)
        return type(self).from_mapping(data)


DEFAULT_CONFIG = AppConfig()
MAX_HUGIN_WORKERS = 1


def load_config(path: str | Path) -> AppConfig:
    return AppConfig.load(path)


def save_config(config: AppConfig, path: str | Path) -> None:
    config.save(path)


Config = AppConfig

__all__ = [
    "AnalysisConfig",
    "OutputConfig",
    "RuntimeConfig",
    "CacheConfig",
    "ScannerConfig",
    "AppConfig",
    "DEFAULT_CONFIG",
    "MAX_HUGIN_WORKERS",
    "load_config",
    "save_config",
    "Config",
]
