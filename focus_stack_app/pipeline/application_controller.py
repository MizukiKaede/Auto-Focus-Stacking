"""Application-level orchestration for the Focus Stack V1 desktop app.

The lower-level pipeline intentionally knows nothing about folders, cache
layout, or Qt.  ``ApplicationController`` is the small adapter which joins
those concerns together while keeping all scanning, decoding, persistence,
archive I/O, and fusion work on a background thread.  It is also useful from a
headless script: every expensive dependency can be injected with a test
double without importing PySide6 or decoding an image.
"""

from __future__ import annotations

from ..utils.performance import timed, BatchMemory

from dataclasses import dataclass, field
import inspect
import io
import logging
import os
from os import PathLike
from pathlib import Path
import re
import shutil
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..config import AppConfig
from ..utils.concurrency import opencv_thread_budget, resolve_focus_analysis_budget
from ..files.archiver import ArchiveMode, FileArchiver
from ..storage.cache import CachePaths, DiskCache
from ..storage.database import Database
from ..storage.manifest import ManifestWriter
from ..storage.models import GroupRecord, ImageRecord
from .coordinator import PipelineConfig, PipelineCoordinator, PipelineSummary
from .events import PipelineEvent, PipelineStage
from .job_state import JobStateStore
from .memory_guard import MemoryGuard
from .merge_worker import StackMergeService


class ControllerError(RuntimeError):
    """Base error for setup/preflight failures before the bounded pipeline."""


class DiskSpaceError(ControllerError):
    """Raised when the destination volume cannot safely hold the batch."""


class ControllerCancelled(ControllerError):
    """Internal marker used when cancellation happens before coordination."""


@dataclass(slots=True)
class ApplicationOptions:
    """User-facing settings accepted by :class:`ApplicationController`.

    Paths are deliberately kept as ``str | Path`` at this boundary so the
    same object can be built from a Qt line edit, a JSON mapping, or a
    headless test.  The controller normalises them before touching the file
    system.  Generated stacks and archived source sets deliberately use
    different directories.  ``archive_mode`` is flat by design: no group
    subdirectories are created by V1.
    """

    source_dir: str | Path
    output_dir: str | Path | None = ""
    archive_mode: ArchiveMode | str = ArchiveMode.MOVE
    # Preserve originals unless the user explicitly opts into post-fusion
    # archiving.  This makes calibration runs and first passes reversible.
    archive_enabled: bool = False
    minimum_stack_group_size: int = 4
    grouping_pause_seconds: int = 20
    fusion_backend: str = "quality"
    output_format: str = "jpg"
    parallel: bool = True
    preserve_cache: bool = False
    hugin_bin: str | Path | None = None
    align_image_stack_path: str | Path | None = None
    enfuse_path: str | Path | None = None
    queue_size: int = 3
    merge_workers: int = 3
    recursive: bool = False
    include_hidden: bool = False
    archive_dir: str | Path | None = None
    # Kept last for positional compatibility. 0 = automatic; 1..10 is a
    # memory-safe upper bound for images analysed concurrently in one group.
    focus_analysis_workers: int = 0

    def __post_init__(self) -> None:
        if self.output_dir is None or (isinstance(self.output_dir, str) and not self.output_dir.strip()):
            self.output_dir = Path(self.source_dir) / "合成"
        self.archive_mode = _coerce_archive_mode(self.archive_mode)
        if isinstance(self.archive_enabled, str):
            self.archive_enabled = self.archive_enabled.strip().casefold() in {"1", "true", "yes", "on"}
        else:
            self.archive_enabled = bool(self.archive_enabled)
        self.output_format = str(getattr(self.output_format, "value", self.output_format)).lower().lstrip(".")
        if self.output_format not in {"jpg", "jpeg", "tif", "tiff"}:
            raise ValueError("output_format must be jpg, jpeg, tif, or tiff")
        from ..fusion_modes import normalize_fusion_backend
        self.fusion_backend = normalize_fusion_backend(self.fusion_backend)
        self.minimum_stack_group_size = int(self.minimum_stack_group_size)
        if not 2 <= self.minimum_stack_group_size <= 1000:
            raise ValueError("minimum_stack_group_size must be between 2 and 1000")
        self.grouping_pause_seconds = int(self.grouping_pause_seconds)
        if not 1 <= self.grouping_pause_seconds <= 3600:
            raise ValueError("grouping_pause_seconds must be between 1 and 3600")
        self.queue_size = max(1, int(self.queue_size))
        self.merge_workers = max(1, min(6, int(self.merge_workers)))
        self.focus_analysis_workers = int(self.focus_analysis_workers)
        if not 0 <= self.focus_analysis_workers <= 10:
            raise ValueError("focus_analysis_workers must be between 0 and 10")


# Names used by early UI prototypes and external scripts.
ControllerOptions = ApplicationOptions
PipelineOptions = ApplicationOptions


@dataclass(slots=True)
class ControllerSummary:
    """Durable, UI-friendly result of one controller run."""

    source_dir: Path
    output_dir: Path
    scan_report: Any | None = None
    groups: list[Any] = field(default_factory=list)
    pipeline_summary: PipelineSummary | None = None
    errors: int = 0
    diagnostics: list[str] = field(default_factory=list)
    cancelled: bool = False
    elapsed_seconds: float = 0.0
    archive_dir: Path | None = None

    @property
    def images_total(self) -> int:
        report = self.scan_report
        if report is None:
            return 0
        try:
            return int(getattr(report, "count"))
        except (AttributeError, TypeError, ValueError):
            values = getattr(report, "images", getattr(report, "records", ()))
            try:
                return len(values)
            except TypeError:
                return 0

    @property
    def groups_total(self) -> int:
        if self.pipeline_summary is not None:
            return int(self.pipeline_summary.groups_total)
        return len(self.groups)

    @property
    def groups_found(self) -> int:
        return len(self.groups)

    @property
    def groups_finished(self) -> int:
        return self.merge_finished

    @property
    def analysis_completed(self) -> int:
        return int(self.pipeline_summary.analysis_completed) if self.pipeline_summary else 0

    @property
    def merge_completed(self) -> int:
        return int(self.pipeline_summary.merge_completed) if self.pipeline_summary else 0

    @property
    def merge_finished(self) -> int:
        if self.pipeline_summary is None:
            return 0
        value = self.pipeline_summary.merge_finished
        return int(self.pipeline_summary.merge_completed if value is None else value)

    @property
    def results(self) -> list[Any]:
        return list(self.pipeline_summary.results) if self.pipeline_summary else []

    @property
    def failed_analysis(self) -> list[Any]:
        return list(self.pipeline_summary.failed_analysis) if self.pipeline_summary else []

    @property
    def failed_merge(self) -> list[Any]:
        return list(self.pipeline_summary.failed_merge) if self.pipeline_summary else []

    @property
    def finished(self) -> bool:
        # A setup/validation failure has no coordinator summary and must not
        # look like a completed run to the UI (otherwise it would snap the
        # overall bar to 100%).  Once coordination exists, its own finished
        # predicate includes archive-only and failed-group consumption.
        return bool(self.pipeline_summary is not None and self.pipeline_summary.finished)


ControllerReport = ControllerSummary


def _coerce_archive_mode(value: ArchiveMode | str) -> ArchiveMode:
    if isinstance(value, ArchiveMode):
        return value
    raw = str(getattr(value, "value", value)).strip().lower()
    aliases = {"move": ArchiveMode.MOVE, "移动": ArchiveMode.MOVE,
               "copy": ArchiveMode.COPY, "复制": ArchiveMode.COPY,
               "hardlink": ArchiveMode.HARDLINK, "hard-link": ArchiveMode.HARDLINK,
               "link": ArchiveMode.HARDLINK, "硬链接": ArchiveMode.HARDLINK}
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ValueError(f"unsupported archive_mode: {value!r}") from exc


def _mapping_value(value: Any, *names: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        for name in names:
            if name in value and value[name] is not None:
                return value[name]
        return default
    for name in names:
        try:
            candidate = getattr(value, name)
        except Exception:
            continue
        if candidate is not None:
            return candidate
    return default


def _path_for_item(item: Any) -> Path | None:
    if isinstance(item, (str, bytes, PathLike)):
        try:
            return Path(item)
        except (TypeError, ValueError):
            return None
    value = _mapping_value(item, "current_path", "path", "original_path", "filename")
    if value is None:
        return None
    try:
        return Path(value)
    except (TypeError, ValueError):
        return None


def _items_for_group(group: Any) -> list[Any]:
    values = _mapping_value(group, "items", "images", "all_images", "image_records", "image_paths")
    if values is None:
        return []
    if isinstance(values, (str, bytes, PathLike)):
        return [values]
    try:
        return list(values)
    except TypeError:
        return []


def _set_group_items(group: Any, values: list[Any]) -> None:
    if isinstance(group, Mapping):
        # A normal dict is mutable.  Custom Mapping implementations are
        # allowed to remain unchanged; the pipeline can still use its paths.
        try:
            group["items"] = values  # type: ignore[index]
        except Exception:
            return
        return
    try:
        setattr(group, "items", values)
    except Exception:
        pass


def _set_group_id(group: Any, value: int) -> None:
    if isinstance(group, Mapping):
        try:
            group["group_id"] = value  # type: ignore[index]
        except Exception:
            pass
        return
    try:
        setattr(group, "group_id", value)
    except Exception:
        pass


def _group_id(group: Any) -> int | str | None:
    return _mapping_value(group, "group_id", "id")


def _invoke_with_optional_source(fn: Callable[..., Any], source: Path) -> Any:
    """Call scanners/detectors with either their bound or source signature."""

    try:
        parameters = list(inspect.signature(fn).parameters.values())
        required = [item for item in parameters
                    if item.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    and item.default is inspect.Parameter.empty]
        return fn(source) if required else fn()
    except (TypeError, ValueError):
        # Some C-extension/test-double callables do not expose signatures.
        try:
            return fn()
        except TypeError:
            return fn(source)


# The application supplies ``<group>_<uuid>``; Enfuser's standalone adapter
# historically used a bare UUID.  Both are safe only because cleanup is
# confined to the private cache temp root.
_TEMP_WORK_DIR_RE = re.compile(r"^(?:[^/\\]+_[0-9a-f]{32}|[0-9a-f]{32})$", re.IGNORECASE)
_TEMP_PRESERVE_MARKERS = frozenset({".active", ".keep", ".preserve", ".lock"})


def cleanup_completed_temp_dirs(
    temp_root: str | Path,
    database: Any,
    *,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Remove UUID work directories whose durable group state is ``DONE``."""

    getter = getattr(database, "get_group", None)
    root = Path(temp_root).expanduser()
    if not callable(getter) or root.is_symlink() or not root.is_dir():
        return []
    try:
        root_resolved = root.resolve(strict=True)
        children = list(root.iterdir())
    except OSError:
        return []
    removed: list[Path] = []
    log = logger or logging.getLogger(__name__)
    for child in children:
        try:
            if child.is_symlink() or not child.is_dir() or not _TEMP_WORK_DIR_RE.fullmatch(child.name):
                continue
            if any((child / marker).exists() for marker in _TEMP_PRESERVE_MARKERS):
                continue
            try:
                group_id = int(child.name.rsplit("_", 1)[0])
            except ValueError:
                continue
            group = getter(group_id)
            if str(_mapping_value(group, "status", default="")).upper() != "DONE":
                continue
            resolved = child.resolve(strict=True)
            if resolved == root_resolved or resolved.parent != root_resolved:
                continue
            shutil.rmtree(resolved)
            removed.append(resolved)
        except OSError:
            log.warning("Unable to remove completed temp directory %s", child, exc_info=True)
    return removed


def cleanup_stale_temp_dirs(
    temp_root: str | Path,
    *,
    max_age_seconds: float = 24 * 60 * 60,
    now: float | None = None,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Remove only old private merge work directories below ``temp_root``.

    Hugin/Enfuse failures intentionally leave their work directory available
    for diagnosis.  Cleanup therefore runs at startup, only accepts the
    controller's UUID-suffixed directory names, skips active/keep markers and
    recent directories, and never follows a symlink outside the cache root.
    Returning the removed paths makes the policy easy to audit and test.
    """

    if float(max_age_seconds) < 0:
        raise ValueError("max_age_seconds cannot be negative")
    root = Path(temp_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        return []
    try:
        root_resolved = root.resolve(strict=True)
    except OSError:
        return []
    timestamp = time.time() if now is None else float(now)
    threshold = float(max_age_seconds)
    removed: list[Path] = []
    log = logger or logging.getLogger(__name__)
    try:
        children = list(root.iterdir())
    except OSError:
        return []
    for child in children:
        try:
            # A symlink can point outside the private temp root; never follow
            # it, even when its apparent name matches our work-dir pattern.
            if child.is_symlink() or not child.is_dir() or not _TEMP_WORK_DIR_RE.fullmatch(child.name):
                continue
            if any((child / marker).exists() for marker in _TEMP_PRESERVE_MARKERS):
                continue
            stat = child.stat()
            age = timestamp - max(float(stat.st_mtime), float(stat.st_ctime))
            if age < threshold:
                continue
            resolved = child.resolve(strict=True)
            # ``child`` must be a direct child after resolution.  This also
            # rejects junctions and links which resolve to another directory.
            if resolved == root_resolved or resolved.parent != root_resolved:
                continue
            shutil.rmtree(resolved)
            removed.append(resolved)
        except OSError:
            log.warning("Unable to remove stale temp directory %s", child, exc_info=True)
    return removed


class _RecoveryAnalyzerAdapter:
    """Return durable analysis rows for resumed groups, else delegate."""

    def __init__(self, fallback: Any, payloads: Mapping[int, Any]) -> None:
        self.fallback = fallback
        self.payloads = dict(payloads)

    def analyze_group(self, group: Any, **kwargs: Any) -> Any:
        gid = _group_id(group)
        if gid is not None and int(gid) in self.payloads:
            return self.payloads[int(gid)]
        callback = getattr(self.fallback, "analyze_group", None)
        if not callable(callback):
            callback = getattr(self.fallback, "analyze", self.fallback)
        if not callable(callback):
            raise TypeError("analyzer must be callable or expose analyze_group()/analyze()")
        try:
            return callback(group, **kwargs)
        except TypeError:
            return callback(group)

    analyze = analyze_group
    __call__ = analyze_group


class ApplicationController:
    """Build and run the complete V1 workflow on one worker thread.

    The controller deliberately exposes simple public attributes
    (``event_callback`` and ``complete_callback``) so the optional Qt window
    can attach queued signals without importing Qt into this module.  Call
    :meth:`start` from any thread; it returns immediately.  Call :meth:`wait`
    from a CLI or test to obtain the summary.
    """

    def __init__(
        self,
        options: ApplicationOptions | Mapping[str, Any] | str | Path | None = None,
        output_dir: str | Path | None = None,
        *,
        source_dir: str | Path | None = None,
        config: AppConfig | Mapping[str, Any] | Any | None = None,
        database: Database | Any | None = None,
        cache: Any | None = None,
        scanner: Any | None = None,
        scene_detector: Any | None = None,
        analyzer: Any | None = None,
        merge_service: Any | None = None,
        coordinator: PipelineCoordinator | Any | None = None,
        memory_guard: Any | None = None,
        stale_temp_age_seconds: float = 24 * 60 * 60,
        cleanup_stale_temp_on_start: bool = True,
        event_callback: Callable[[PipelineEvent], Any] | None = None,
        complete_callback: Callable[[ControllerSummary], Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger("focus_stack_app.controller")
        self.event_callback = event_callback
        self.complete_callback = complete_callback
        self._cancel_event = threading.Event()
        self._state_lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._summary: ControllerSummary | None = None
        self._coordinator: Any | None = coordinator
        self._database: Any | None = database
        self._owns_database = database is None
        self._cache: DiskCache | Any | None = cache
        self._cache_paths: CachePaths | None = None
        self._scanner = scanner
        self._scene_detector = scene_detector
        self._analyzer = analyzer
        self._merge_service = merge_service
        self._memory_guard = memory_guard
        self._job_state: JobStateStore | None = None
        if float(stale_temp_age_seconds) < 0:
            raise ValueError("stale_temp_age_seconds cannot be negative")
        self.stale_temp_age_seconds = float(stale_temp_age_seconds)
        self.cleanup_stale_temp_on_start = bool(cleanup_stale_temp_on_start)
        self._event_errors = 0
        self._diagnostics: list[str] = []
        self._groups_found = 0
        self._started_at = 0.0
        self._log_handler: logging.Handler | None = None

        self.options = self._make_options(options, output_dir=output_dir, source_dir=source_dir)
        self.config = self._make_config(config, self.options)
        self.source_dir = Path(self.options.source_dir)
        self.output_dir = Path(self.options.output_dir)
        self.archive_dir = (
            Path(self.options.archive_dir)
            if self.options.archive_dir is not None and str(self.options.archive_dir).strip()
            else self.source_dir / "归档"
        )

    @staticmethod
    def _make_options(
        value: ApplicationOptions | Mapping[str, Any] | str | Path | None,
        *,
        output_dir: str | Path | None,
        source_dir: str | Path | None,
    ) -> ApplicationOptions:
        if isinstance(value, ApplicationOptions):
            data = {name: getattr(value, name) for name in ApplicationOptions.__dataclass_fields__}
        elif isinstance(value, Mapping):
            data = dict(value)
        elif value is not None and output_dir is not None:
            data = {"source_dir": value, "output_dir": output_dir}
        else:
            data = {}
        if source_dir is not None:
            data["source_dir"] = source_dir
        if output_dir is not None:
            data["output_dir"] = output_dir
        aliases = {
            "source": "source_dir", "input_dir": "source_dir", "input": "source_dir",
            "output": "output_dir", "destination_dir": "output_dir", "destination": "output_dir",
            "archive_dirname": "archive_dir", "archive_directory": "archive_dir",
            "archive": "archive_mode", "archive_method": "archive_mode",
            "archive_originals": "archive_enabled", "archive_after_merge": "archive_enabled",
            "format": "output_format", "hugin_path": "hugin_bin",
            "align_path": "align_image_stack_path", "align_image_stack": "align_image_stack_path",
            "enfuse": "enfuse_path", "hugin": "hugin_bin", "cache": "preserve_cache",
            "parallel_pipeline": "parallel",
            "backend": "fusion_backend", "fusion_engine": "fusion_backend",
        }
        for old, new in aliases.items():
            if old in data and new not in data:
                data[new] = data[old]
        if "source_dir" not in data:
            raise ValueError("source_dir is required")
        allowed = set(ApplicationOptions.__dataclass_fields__)
        return ApplicationOptions(**{key: item for key, item in data.items() if key in allowed})

    @staticmethod
    def _make_config(config: Any | None, options: ApplicationOptions) -> Any:
        if config is None:
            base: Any = AppConfig()
        elif isinstance(config, Mapping):
            base = AppConfig.from_mapping(config)
        else:
            base = config
        # AppConfig.with_overrides is the canonical adapter.  A custom
        # duck-typed config can still be used by tests and advanced callers.
        runtime_values: dict[str, Any] = {
            "parallel_pipeline": bool(options.parallel),
            "preserve_cache": bool(options.preserve_cache),
            "max_hugin_workers": int(options.merge_workers),
            "focus_analysis_workers": int(options.focus_analysis_workers),
            "fusion_backend": options.fusion_backend,
        }
        for key, value in (
            ("hugin_bin", options.hugin_bin),
            ("align_image_stack_path", options.align_image_stack_path),
            ("enfuse_path", options.enfuse_path),
        ):
            if value is not None and str(value).strip():
                runtime_values[key] = str(value)
        output_values = {"format": options.output_format, "jpeg_quality": 100, "jpeg_subsampling": 0}
        scanner_values = {"recursive": bool(options.recursive), "include_hidden": bool(options.include_hidden)}
        override = getattr(base, "with_overrides", None)
        if callable(override):
            try:
                return override(output=output_values, runtime=runtime_values, scanner=scanner_values,
                                analysis={"minimum_stack_group_size": options.minimum_stack_group_size})
            except (TypeError, ValueError, KeyError):
                pass
        # Preserve custom config objects while making the three explicit UI
        # choices available to the component adapters.
        return base

    @property
    def database(self) -> Any | None:
        return self._database

    @property
    def cache(self) -> Any | None:
        return self._cache

    @property
    def cache_paths(self) -> CachePaths | None:
        return self._cache_paths

    @property
    def coordinator(self) -> Any | None:
        return self._coordinator

    @property
    def memory_guard(self) -> Any | None:
        return self._memory_guard

    @property
    def job_state(self) -> JobStateStore | None:
        return self._job_state

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def summary(self) -> ControllerSummary | None:
        return self._summary

    def start(self) -> "ApplicationController":
        """Start setup, scan, analysis, and merge asynchronously."""

        with self._state_lock:
            if self.running:
                raise RuntimeError("application controller is already running")
            self._cancel_event.clear()
            self._summary = None
            self._event_errors = 0
            self._diagnostics = []
            self._groups_found = 0
            self._started_at = time.monotonic()
            self._thread = threading.Thread(target=self._run, name="focus-stack-application", daemon=True)
            self._thread.start()
        return self

    run_async = start

    def run(self, timeout: float | None = None) -> ControllerSummary:
        """Run synchronously for CLI/headless callers.

        The actual work still executes on the controller worker; this helper
        only waits for it, which makes it safe to use in a script without
        accidentally putting scanner/vision work on a Qt thread.
        """

        if not self.running:
            self.start()
        result = self.wait(timeout)
        if result is None:
            raise TimeoutError("application controller did not finish before timeout")
        return result

    run_sync = run
    process = run

    def wait(self, timeout: float | None = None) -> ControllerSummary | None:
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        return self._summary if thread is None or not thread.is_alive() else None

    join = wait

    def cancel(self) -> None:
        """Request graceful cancellation of setup and the bounded pipeline."""

        self._cancel_event.set()
        coordinator = self._coordinator
        if coordinator is not None:
            try:
                coordinator.cancel()
            except Exception:
                self.logger.debug("Unable to cancel coordinator", exc_info=True)

    stop = cancel

    def close(self) -> None:
        """Close the owned SQLite connection after workers have stopped."""

        if self.running:
            self.cancel()
            self.wait(5)
        if self._owns_database and self._database is not None:
            close = getattr(self._database, "close", None)
            if callable(close):
                close()
            self._database = None
        if self._log_handler is not None:
            try:
                self.logger.removeHandler(self._log_handler)
                self._log_handler.close()
            except Exception:
                pass
            self._log_handler = None

    def _emit(
        self,
        stage: PipelineStage | str,
        *,
        current_file: str = "",
        current_group: int | str | None = None,
        completed: int = 0,
        total: int = 0,
        groups_found: int | None = None,
        groups_finished: int = 0,
        errors: int | None = None,
        message: str = "",
        analysis_completed: int = 0,
        analysis_total: int = 0,
        merge_completed: int = 0,
        merge_total: int = 0,
        merge_finished: int | None = None,
    ) -> None:
        callback = self.event_callback
        if callback is None:
            return
        event = PipelineEvent(
            stage=stage,
            current_file=str(current_file or ""),
            current_group=current_group,
            completed=int(completed),
            total=int(total),
            groups_found=self._groups_found if groups_found is None else int(groups_found),
            groups_finished=int(groups_finished),
            errors=self._event_errors if errors is None else int(errors),
            message=str(message or ""),
            analysis_completed=int(analysis_completed),
            analysis_total=int(analysis_total),
            merge_completed=int(merge_completed),
            merge_total=int(merge_total),
            merge_finished=None if merge_finished is None else int(merge_finished),
        )
        try:
            callback(event)
        except Exception:
            self.logger.debug("Application event callback failed", exc_info=True)

    def _on_pipeline_event(self, event: PipelineEvent) -> None:
        self._event_errors = max(self._event_errors, int(getattr(event, "errors", 0) or 0))
        callback = self.event_callback
        if callback is None:
            return
        try:
            callback(event)
        except Exception:
            self.logger.debug("Pipeline event callback failed", exc_info=True)

    def _configure_logging(self) -> None:
        if self._cache_paths is None:
            return
        if self._log_handler is not None:
            return
        try:
            log_path = self._cache_paths.logs / "application.log"
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self._log_handler = handler
        except OSError:
            # Logging must never prevent an archive from starting.
            self.logger.debug("Unable to create application log", exc_info=True)

    def _validate_paths(self) -> tuple[Path, Path]:
        source = Path(self.options.source_dir).expanduser()
        output = Path(self.options.output_dir).expanduser()
        archive = (
            Path(self.options.archive_dir).expanduser()
            if self.options.archive_dir is not None and str(self.options.archive_dir).strip()
            else source / "归档"
        )
        if not source.exists() or not source.is_dir():
            raise ControllerError(f"source directory is not available: {source}")
        source_resolved = source.resolve(strict=True)
        output_resolved = output.resolve(strict=False)
        archive_resolved = archive.resolve(strict=False)
        if len({source_resolved, output_resolved, archive_resolved}) != 3:
            raise ControllerError("source, composite output, and archive directories must be different")

        def nested(child: Path, parent: Path) -> bool:
            try:
                child.relative_to(parent)
                return True
            except ValueError:
                return False

        # The UI defaults output and archive to dedicated folders inside the
        # source folder.  The production scanner explicitly excludes both,
        # including during recursive scans.  The reverse nesting remains
        # unsafe because scanning a source below either destination would mix
        # project roles and make the exclusion swallow the source itself.
        if nested(source_resolved, output_resolved):
            raise ControllerError("source directory cannot be inside composite output directory")
        if nested(source_resolved, archive_resolved):
            raise ControllerError("source directory cannot be inside archive directory")
        if nested(output_resolved, archive_resolved) or nested(archive_resolved, output_resolved):
            raise ControllerError("composite output and archive directories cannot contain each other")
        output.mkdir(parents=True, exist_ok=True)
        # Do not create or touch the archive location during a preservation
        # run.  Existing archive recovery is likewise irrelevant until the
        # user enables the opt-in archival action.
        if self.options.archive_enabled:
            archive.mkdir(parents=True, exist_ok=True)
        self.archive_dir = archive_resolved
        return source_resolved, output_resolved

    def _prepare_runtime(self, source: Path, output: Path) -> None:
        self.source_dir, self.output_dir = source, output
        cache_config = getattr(self.config, "cache", None)
        self._cache_paths = CachePaths.for_project(output, cache_config).ensure()
        self._configure_logging()
        if self._database is None:
            self._database = Database(self._cache_paths.database)
        completed_temp = cleanup_completed_temp_dirs(
            self._cache_paths.temp, self._database, logger=self.logger,
        )
        if completed_temp:
            self.logger.info(
                "Removed %d completed merge temp directories", len(completed_temp),
            )
        # Reconcile durable rows left by a previous interrupted run.  These
        # calls are best-effort because early/custom repositories may not have
        # the recovery API; they do not touch source files unless a journal
        # proves the destination exists.
        archive_recovery = getattr(self._database, "recover_archived_images", None)
        if self.options.archive_enabled and callable(archive_recovery):
            try:
                # Supplying the archive root lets legacy image rows resolve a
                # destination even when their original source was moved away.
                try:
                    archive_recovery(destination_root=self.archive_dir)
                except TypeError:
                    try:
                        archive_recovery(self.archive_dir)
                    except TypeError:
                        archive_recovery()
            except Exception:
                self.logger.warning("Recovery check recover_archived_images failed", exc_info=True)
        job_recovery = getattr(self._database, "recover_incomplete_jobs", None)
        if callable(job_recovery):
            try:
                job_recovery()
            except Exception:
                self.logger.warning("Recovery check recover_incomplete_jobs failed", exc_info=True)
        self._job_state = JobStateStore(self._database, logger=self.logger)
        if self._cache is None:
            analysis_config = getattr(self.config, "analysis", None)
            self._cache = DiskCache(
                output,
                cache_config,
                max_preview_items=int(getattr(analysis_config, "max_preview_cache_items", 12)),
                max_focus_items=int(getattr(analysis_config, "max_focus_cache_items", 4)),
            )
        if self._memory_guard is None:
            runtime = getattr(self.config, "runtime", None)
            self._memory_guard = MemoryGuard(
                minimum_bytes=int(getattr(runtime, "min_available_memory_bytes", 2 * 1024**3) or 0),
                minimum_fraction=float(getattr(runtime, "min_available_memory_fraction", 0.10) or 0),
                event_callback=self._on_pipeline_event,
                logger=self.logger,
            )
        if self.cleanup_stale_temp_on_start:
            removed = self.cleanup_stale_temp()
            if removed:
                self.logger.info("Removed %d stale merge temp directories", len(removed))

    def cleanup_stale_temp(self, *, now: float | None = None) -> list[Path]:
        """Apply the startup-only stale temporary-directory policy."""

        if self._cache_paths is None:
            return []
        return cleanup_stale_temp_dirs(
            self._cache_paths.temp,
            max_age_seconds=self.stale_temp_age_seconds,
            now=now,
            logger=self.logger,
        )

    @staticmethod
    def _recovery_image_path(database: Any, image: Any) -> Path | None:
        """Resolve one restart image through current/pending/journal paths."""

        candidates: list[Any] = []
        for name in ("current_path", "pending_path", "original_path"):
            value = _mapping_value(image, name)
            if value:
                candidates.append(value)
        image_id = _mapping_value(image, "id")
        operations = getattr(database, "list_archive_operations", None)
        if callable(operations) and image_id is not None:
            try:
                rows = list(operations(image_id=int(image_id)))
            except Exception:
                rows = []
            # The DB returns oldest first; newest destination is authoritative
            # when the image row was last updated just before a crash.
            for operation in reversed(rows):
                destination = _mapping_value(operation, "destination_path", "destination")
                if destination:
                    candidates.append(destination)
        seen: set[str] = set()
        for value in candidates:
            try:
                path = Path(value)
            except (TypeError, ValueError):
                continue
            key = str(path.resolve(strict=False)).casefold()
            if key in seen:
                continue
            seen.add(key)
            try:
                if path.is_file():
                    # MergeService consumes ImageRecord.path, so reflect the
                    # recovered live path in this in-memory record while
                    # preserving original_path for output naming/manifest.
                    try:
                        image.current_path = str(path)
                    except Exception:
                        pass
                    return path
            except OSError:
                continue
        return None

    def _recovery_group_payload(self, job: Any, group_record: Any) -> tuple[Any, Any] | None:
        """Build a merge-compatible group/result from durable DB rows."""

        if self._database is None:
            return None
        gid = _mapping_value(group_record, "id", "group_id")
        if gid is None:
            return None
        getter = getattr(self._database, "get_group_images", None)
        if not callable(getter):
            return None
        try:
            images = list(getter(int(gid)))
        except Exception:
            self.logger.warning("Unable to load images for recovery group %s", gid, exc_info=True)
            return None
        if not images:
            self._diagnostics.append(f"恢复组 {gid} 没有可用 image rows，保留为未完成任务。")
            return None
        paths: list[Path | None] = [self._recovery_image_path(self._database, image) for image in images]
        if not any(path is not None for path in paths):
            self._diagnostics.append(f"恢复组 {gid} 的 current/pending/destination 均不存在，保留为未完成任务。")
            return None

        analysis_rows: list[Any] = []
        list_analysis = getattr(self._database, "list_analysis", None)
        if callable(list_analysis):
            try:
                analysis_rows = list(list_analysis(group_id=int(gid)))
            except Exception:
                self.logger.warning("Unable to load analysis rows for recovery group %s", gid, exc_info=True)
        by_image_id = {
            int(value): row
            for row in analysis_rows
            if (value := _mapping_value(row, "image_id")) is not None
        }
        selected_indices = [
            index for index, image in enumerate(images)
            if bool(_mapping_value(by_image_id.get(_mapping_value(image, "id")), "selected", default=False))
        ]
        # A repository may expose selected image rows without exposing the
        # analysis list; use that as a compatible fallback.
        if not selected_indices:
            selected_getter = getattr(self._database, "get_selected_images", None)
            if callable(selected_getter):
                try:
                    selected_ids = {
                        int(_mapping_value(image, "id"))
                        for image in selected_getter(int(gid))
                        if _mapping_value(image, "id") is not None
                    }
                    selected_indices = [
                        index for index, image in enumerate(images)
                        if _mapping_value(image, "id") in selected_ids
                    ]
                except Exception:
                    selected_indices = []

        stage = str(_mapping_value(job, "stage", default="")).upper()
        group_status = str(_mapping_value(group_record, "status", default="") or "").upper()
        merge_stage = stage in {"ALIGNMENT", "FUSION", "EXPORT"}
        merge_status = group_status in {
            "SELECTED", "QUEUED_FOR_MERGE", "ALIGNING", "FUSING",
            "FAILED_ALIGNMENT", "FAILED_FUSION", "FAILED_HUGIN",
        }
        selected_paths = [str(paths[index]) for index in selected_indices if paths[index] is not None]
        if not selected_paths and merge_stage:
            # If a crash happened after selection rows but before a later
            # cache write, all recovered images are safer than silently
            # skipping a required stack.
            selected_indices = [index for index, path in enumerate(paths) if path is not None]
            selected_paths = [str(paths[index]) for index in selected_indices if paths[index] is not None]
        needs_merge = len(images) >= self.options.minimum_stack_group_size and len(selected_paths) > 1 and (
            merge_status or merge_stage or int(_mapping_value(group_record, "selected_count", default=0) or 0) > 1
        )
        first = images[0]
        first_path = paths[0] or Path(_mapping_value(first, "current_path", "original_path", default=""))
        def recovered_current(value: Any) -> str | None:
            if value is None:
                return None
            wanted = os.path.normcase(os.path.abspath(os.fspath(value)))
            for image, current in zip(images, paths):
                if current is None:
                    continue
                original = _mapping_value(image, "original_path", default=current)
                if os.path.normcase(os.path.abspath(os.fspath(original))) == wanted:
                    return str(current)
            basename = Path(os.fspath(value)).name.casefold()
            matches = [str(current) for current in paths if current is not None and current.name.casefold() == basename]
            return matches[0] if len(matches) == 1 else None

        stored_order = list(_mapping_value(group_record, "alignment_order", default=()) or ())
        recovered_order = [resolved for value in stored_order if (resolved := recovered_current(value)) is not None]
        selected_keys = {os.path.normcase(os.path.abspath(value)) for value in selected_paths}
        recovered_order = [value for value in recovered_order if os.path.normcase(os.path.abspath(value)) in selected_keys]
        for value in selected_paths:
            if os.path.normcase(os.path.abspath(value)) not in {
                os.path.normcase(os.path.abspath(item)) for item in recovered_order
            }:
                recovered_order.append(value)
        recovered_reference = recovered_current(_mapping_value(group_record, "preview_reference", default=None))
        analysis: dict[str, Any] = {
            "group_id": int(gid),
            "all_images": list(images),
            "items": list(images),
            "image_records": list(images),
            "selected_indices": selected_indices,
            "selected_paths": selected_paths,
            "selected_images": selected_paths,
            "capture_order": [str(path) for path in paths if path is not None],
            "alignment_order": recovered_order or selected_paths,
            "alignment_order_confidence": _mapping_value(group_record, "alignment_order_confidence", default=0.0),
            "alignment_order_fallback_used": bool(_mapping_value(group_record, "alignment_order_fallback_used", default=False)),
            "preview_reference": recovered_reference or str(first_path),
            "first_original_image": first,
            "first_original_path": str(_mapping_value(first, "original_path", default=first_path)),
            "selected_count": len(selected_indices),
            "image_count": len(images),
            "coverage": _mapping_value(group_record, "coverage", default=0.0),
            "confidence": _mapping_value(group_record, "confidence", default=0.0),
            "needs_merge": needs_merge,
            "status": "READY_FOR_MERGE" if needs_merge else "NO_MERGE_RECOVERED",
            "merge_status": "READY_FOR_MERGE" if needs_merge else "NO_MERGE_RECOVERED",
            "analysis_records": [],
        }
        for index, image in enumerate(images):
            row = by_image_id.get(_mapping_value(image, "id"))
            image_path = paths[index]
            analysis["analysis_records"].append({
                "index": index,
                "image": image,
                "path": str(image_path or ""),
                "selected": index in selected_indices,
                "selection_reason": _mapping_value(row, "selection_reason", default="coverage" if index in selected_indices else "not selected"),
                "coverage_gain": _mapping_value(row, "coverage_gain", default=0.0),
                "quality_score": _mapping_value(row, "quality_score", default=0.0),
            })
        group = {
            "group_id": int(gid),
            "id": int(gid),
            "items": list(images),
            "images": list(images),
            "image_records": list(images),
            "image_count": len(images),
            "selected_count": len(selected_indices),
            "start_index": _mapping_value(group_record, "start_index", default=0),
            "end_index": _mapping_value(group_record, "end_index", default=max(0, len(images) - 1)),
            "confidence": _mapping_value(group_record, "confidence", default=0.0),
            "coverage": _mapping_value(group_record, "coverage", default=0.0),
            "status": group_status,
            "_recovery": True,
        }
        return group, analysis

    def _load_recovery_groups(self) -> tuple[list[Any], dict[int, Any]]:
        """Load unfinished groups and cached selection payloads after restart."""

        store = self._job_state
        if store is None or not store.supported or self._database is None:
            return [], {}
        jobs = store.unfinished()
        if not jobs:
            return [], {}
        groups: list[Any] = []
        payloads: dict[int, Any] = {}
        get_group = getattr(self._database, "get_group", None)
        if not callable(get_group):
            return [], {}
        for job in jobs:
            gid = _mapping_value(job, "group_id")
            if gid is None:
                continue
            try:
                group_record = get_group(int(gid))
            except Exception:
                group_record = None
            if group_record is None:
                self._diagnostics.append(f"恢复任务 {getattr(job, 'id', gid)} 缺少 group row，保留为未完成任务。")
                continue
            built = self._recovery_group_payload(job, group_record)
            if built is None:
                continue
            group, analysis = built
            groups.append(group)
            payloads[int(gid)] = analysis
        return groups, payloads

    def _disk_check(self, required_bytes: int) -> None:
        runtime = getattr(self.config, "runtime", None)
        margin = int(getattr(runtime, "disk_safety_margin_bytes", 0) or 0)
        required = max(0, int(required_bytes)) + max(0, margin)
        try:
            usage = shutil.disk_usage(self.output_dir)
        except OSError as exc:
            raise DiskSpaceError(f"unable to inspect destination disk: {exc}") from exc
        if usage.free < required:
            available_gib = usage.free / 1024**3
            required_gib = required / 1024**3
            raise DiskSpaceError(
                f"insufficient free space at {self.output_dir}: "
                f"{available_gib:.2f} GiB available, {required_gib:.2f} GiB required"
            )

    @timed("scan")
    def _scan(self) -> tuple[Any, list[ImageRecord]]:
        scanner = self._scanner
        if scanner is None:
            from ..core.scanner import ImageScanner

            scanner_config = getattr(self.config, "scanner", None)
            scanner = ImageScanner(
                self.source_dir,
                scanner_config,
                excluded_roots=(self.output_dir, self.archive_dir),
                logger=self.logger,
            )
            self._scanner = scanner
        report: Any
        report_method = getattr(scanner, "scan_report", None)
        if callable(report_method):
            report = _invoke_with_optional_source(report_method, self.source_dir)
        else:
            scan_method = getattr(scanner, "scan", None)
            if callable(scan_method):
                report = _invoke_with_optional_source(scan_method, self.source_dir)
            elif callable(scanner):
                report = _invoke_with_optional_source(scanner, self.source_dir)
            else:
                raise ControllerError("scanner must expose scan_report(), scan(), or be callable")
        if hasattr(report, "images"):
            values = list(getattr(report, "images"))
        elif hasattr(report, "records"):
            values = list(getattr(report, "records"))
        elif isinstance(report, Mapping):
            values = list(report.get("images", report.get("records", ())))
        else:
            values = list(report or ())
        records: list[ImageRecord] = []
        for index, item in enumerate(values):
            if isinstance(item, ImageRecord):
                record = item
            else:
                path = _path_for_item(item)
                if path is None:
                    raise ControllerError(f"scanner returned an item without a path: {item!r}")
                if isinstance(item, Mapping):
                    data = dict(item)
                    data.setdefault("original_path", str(path))
                    data.pop("path", None)
                    allowed = set(ImageRecord.__dataclass_fields__)
                    record = ImageRecord(**{key: value for key, value in data.items() if key in allowed})
                else:
                    record = ImageRecord.from_path(path)
            if record.sequence_index is None:
                record.sequence_index = index
            records.append(record)
        # A lightweight report from a test double may not expose total_bytes;
        # all real scanner reports do.  The controller only needs the byte sum
        # for the safety check.
        self._emit(PipelineStage.SCANNING, completed=len(records), total=len(records), message=f"发现 {len(records)} 张 JPG")
        return report, records

    def _persist_images(self, records: list[ImageRecord]) -> None:
        if self._database is None:
            return
        insert = getattr(self._database, "insert_image", None)
        if not callable(insert):
            insert_many = getattr(self._database, "insert_images", None)
            if callable(insert_many):
                insert_many(records)
                return
            raise ControllerError("database must expose insert_image(s)")
        for index, record in enumerate(records):
            if self._cancel_event.is_set():
                raise ControllerCancelled()
            insert(record)
            self._emit(
                PipelineStage.SCANNING,
                current_file=record.path,
                completed=index + 1,
                total=len(records),
                message="保存图片记录",
            )

    def _fallback_singleton_groups(self, records: list[ImageRecord], error: Exception) -> list[Any]:
        """Keep a corrupt preview from blocking safe archive processing."""

        from ..core.types import SceneGroup

        message = (
            f"场景预览失败：{type(error).__name__}: {error}；"
            "已降级为逐张归档组，不会执行合成。"
        )
        self._event_errors += 1
        self._diagnostics.append(message)
        self._emit(PipelineStage.ERROR, errors=self._event_errors, message=message)
        groups = [
            SceneGroup(index + 1, [record], start_index=index, end_index=index, confidence=0.0)
            for index, record in enumerate(records)
        ]
        self._groups_found = len(groups)
        return groups

    @timed("grouping")
    def _scene_groups(self, records: list[ImageRecord]) -> list[Any]:
        from ..core.manual_groups import group_with_overrides
        from ..core.plan_cache import (
            GROUPING_PLAN_VERSION, build_plan_key, json_compatible,
            optional_file_fingerprint,
        )

        manual_path = self.source_dir / "stack_groups.json"
        plan_key: str | None = None
        plan_cache_enabled = bool(self.options.preserve_cache and self._database is not None)
        if plan_cache_enabled:
            try:
                paths = [_path_for_item(record) for record in records]
                if any(path is None for path in paths):
                    raise ValueError("every scanned image must expose a path")
                detector = self._scene_detector
                if detector is None:
                    from ..core.group_detector import StackGroupDetector, StackGroupDetectorConfig

                    detector_type = f"{StackGroupDetector.__module__}.{StackGroupDetector.__qualname__}"
                    detector_config = StackGroupDetectorConfig(
                        pause_seconds=self.options.grouping_pause_seconds,
                    )
                else:
                    detector_type = f"{type(detector).__module__}.{type(detector).__qualname__}"
                    detector_config = getattr(detector, "config", None)
                detector_signature = {
                    "type": detector_type,
                    "config": detector_config,
                }
                plan_key = build_plan_key(
                    "grouping",
                    GROUPING_PLAN_VERSION,
                    paths,
                    {
                        "pause_seconds": self.options.grouping_pause_seconds,
                        "detector": detector_signature,
                    },
                    extra={"manual_groups": optional_file_fingerprint(manual_path)},
                )
                payload = self._database.get_cached_plan(
                    "grouping", plan_key, algorithm_version=GROUPING_PLAN_VERSION,
                )
                cached = self._restore_grouping_plan(payload, records)
                if cached is not None:
                    self._groups_found = len(cached)
                    self.logger.info(
                        "grouping plan cache status=hit key=%s groups=%s decode_count=0",
                        plan_key, len(cached),
                    )
                    return cached
            except Exception:
                self.logger.warning("grouping plan cache read failed", exc_info=True)
                plan_key = None

        groups = group_with_overrides(
            records, manual_path, self._automatic_scene_groups,
        )
        self._groups_found = len(groups)
        self.logger.info(
            "grouping plan cache status=miss key=%s groups=%s", plan_key, len(groups),
        )
        if plan_cache_enabled and plan_key is not None and not self._cancel_event.is_set():
            try:
                positions = {id(record): index for index, record in enumerate(records)}
                payload = {
                    "groups": [
                        {
                            "group_id": int(_group_id(group) or index + 1),
                            "indices": [positions[id(item)] for item in _items_for_group(group)],
                            "start_index": int(_mapping_value(group, "start_index", default=0) or 0),
                            "end_index": int(_mapping_value(group, "end_index", default=0) or 0),
                            "confidence": float(_mapping_value(group, "confidence", default=0.0) or 0.0),
                            "excluded_reason": _mapping_value(group, "excluded_reason", default=None),
                            "comparisons": json_compatible(
                                _mapping_value(group, "comparisons", default=[]) or []
                            ),
                        }
                        for index, group in enumerate(groups)
                    ]
                }
                self._database.upsert_cached_plan(
                    "grouping", plan_key, GROUPING_PLAN_VERSION, payload,
                )
            except Exception:
                self.logger.warning("grouping plan cache write failed", exc_info=True)
        return groups

    @staticmethod
    def _restore_grouping_plan(payload: Any, records: list[ImageRecord]) -> list[Any] | None:
        if not isinstance(payload, Mapping) or not isinstance(payload.get("groups"), list):
            return None
        from ..core.types import SceneComparison, SceneGroup

        restored: list[Any] = []
        seen: list[int] = []
        try:
            for position, raw in enumerate(payload["groups"]):
                if not isinstance(raw, Mapping):
                    return None
                indices = [int(value) for value in raw["indices"]]
                if not indices or any(not 0 <= value < len(records) for value in indices):
                    return None
                comparisons = []
                for value in raw.get("comparisons", []):
                    if not isinstance(value, Mapping):
                        return None
                    comparisons.append(SceneComparison(**dict(value)))
                restored.append(SceneGroup(
                    group_id=int(raw.get("group_id", position + 1)),
                    items=[records[index] for index in indices],
                    start_index=int(raw.get("start_index", min(indices))),
                    end_index=int(raw.get("end_index", max(indices))),
                    confidence=float(raw.get("confidence", 0.0)),
                    comparisons=comparisons,
                    excluded_reason=raw.get("excluded_reason"),
                ))
                seen.extend(indices)
            if sorted(seen) != list(range(len(records))):
                return None
            return restored
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    def _automatic_scene_groups(self, records: list[ImageRecord]) -> list[Any]:
        detector = self._scene_detector
        if detector is None:
            from ..core.group_detector import StackGroupDetector, StackGroupDetectorConfig
            # Calibrated grouping uses original RGB thumbnails, not recompressed cache previews.
            detector = StackGroupDetector(StackGroupDetectorConfig(
                pause_seconds=self.options.grouping_pause_seconds,
            ))
            self._scene_detector = detector
        iterator = getattr(detector, "iter_groups", None)
        groups: list[Any] = []
        if callable(iterator):
            try:
                result: Iterable[Any] = iterator(records)
                for group in result:
                    if self._cancel_event.is_set():
                        raise ControllerCancelled()
                    groups.append(group)
                    self._groups_found = len(groups)
                    items = _items_for_group(group)
                    first = _path_for_item(items[0]) if items else None
                    self._emit(
                        PipelineStage.SCANNING,
                        current_file=first or "",
                        current_group=_group_id(group),
                        completed=len(groups),
                        total=max(len(records), len(groups)),
                        groups_found=len(groups),
                        message=f"发现第 {len(groups)} 组（{len(items)} 张）",
                    )
                return groups
            except ControllerCancelled:
                raise
            except Exception as exc:
                return self._fallback_singleton_groups(records, exc)
        detect = getattr(detector, "detect", None)
        if callable(detect):
            try:
                detected = _invoke_with_optional_source(detect, self.source_dir) if not records else detect(records)
                groups = list(detected or ())
            except Exception as exc:
                return self._fallback_singleton_groups(records, exc)
            for index, group in enumerate(groups, start=1):
                self._groups_found = index
                values = _items_for_group(group)
                first = _path_for_item(values[0]) if values else None
                self._emit(
                    PipelineStage.SCANNING,
                    current_file=first or "",
                    current_group=_group_id(group),
                    completed=index,
                    total=max(len(records), len(groups)),
                    groups_found=index,
                    message=f"发现第 {index} 组（{len(values)} 张）",
                )
            return groups
        if callable(detector):
            try:
                groups = list(detector(records) or ())
            except Exception as exc:
                return self._fallback_singleton_groups(records, exc)
            for index, group in enumerate(groups, start=1):
                self._groups_found = index
                values = _items_for_group(group)
                first = _path_for_item(values[0]) if values else None
                self._emit(
                    PipelineStage.SCANNING,
                    current_file=first or "",
                    current_group=_group_id(group),
                    completed=index,
                    total=max(len(records), len(groups)),
                    groups_found=index,
                    message=f"发现第 {index} 组（{len(values)} 张）",
                )
            return groups
        raise ControllerError("scene detector must expose iter_groups()/detect() or be callable")

    def _persist_groups(self, groups: list[Any], records: list[ImageRecord]) -> None:
        if self._database is None:
            return
        by_path: dict[str, ImageRecord] = {}
        for record in records:
            for value in (record.original_path, record.current_path, str(record.path)):
                if value:
                    by_path[str(Path(value).resolve(strict=False)).casefold()] = record
        insert = getattr(self._database, "insert_group", None)
        set_image_group = getattr(self._database, "set_image_group", None)
        if not callable(insert):
            raise ControllerError("database must expose insert_group")
        for index, group in enumerate(groups):
            values = _items_for_group(group)
            canonical: list[Any] = []
            for item in values:
                record = item if isinstance(item, ImageRecord) else by_path.get(
                    str(_path_for_item(item).resolve(strict=False)).casefold() if _path_for_item(item) else ""
                )
                canonical.append(record or item)
            _set_group_items(group, canonical)
            image_ids = [getattr(item, "id", None) for item in canonical]
            image_ids = [int(value) for value in image_ids if value is not None]
            first_image_id = image_ids[0] if image_ids else None
            group_record = GroupRecord(
                first_image_id=first_image_id,
                start_index=_mapping_value(group, "start_index", default=index),
                end_index=_mapping_value(group, "end_index", default=max(index, index + len(canonical) - 1)),
                image_count=len(canonical) or int(_mapping_value(group, "image_count", default=0) or 0),
                confidence=_mapping_value(group, "confidence", default=None),
                status="DISCOVERED",
            )
            group_id = int(insert(group_record))
            _set_group_id(group, group_id)
            if self._job_state is not None:
                self._job_state.ensure(group, stage="CLASSIFICATION")
            for item in canonical:
                image_id = getattr(item, "id", None)
                if image_id is None:
                    continue
                try:
                    item.group_id = group_id
                except Exception:
                    pass
                if callable(set_image_group):
                    set_image_group(int(image_id), group_id)

    def _encode_preview(self, image: Any) -> bytes:
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]

            encoded_ok, encoded = cv2.imencode(
                ".jpg", np.asarray(image), [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )
            if encoded_ok:
                return bytes(encoded.tobytes())
        except ImportError:
            pass
        try:
            from PIL import Image  # type: ignore[import-not-found]

            stream = io.BytesIO()
            Image.fromarray(image).save(stream, format="JPEG", quality=90, subsampling=0)
            return stream.getvalue()
        except Exception as exc:
            raise ControllerError("preview encoding requires OpenCV or Pillow + NumPy") from exc

    def _decode_preview(self, payload: bytes) -> Any:
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]

            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                return image
        except ImportError:
            pass
        try:
            from PIL import Image  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]

            with Image.open(io.BytesIO(payload)) as image:
                return np.asarray(image.convert("RGB"))
        except Exception as exc:
            raise ControllerError("cached preview decoding requires OpenCV or Pillow + NumPy") from exc

    def _preview_loader(self, item: Any, max_long_edge: int | None = None) -> Any:
        """Decode one bounded preview and persist its encoded representation."""

        path = _path_for_item(item)
        edge = int(max_long_edge or getattr(getattr(self.config, "analysis", None), "scene_preview_long_edge", 512))
        from ..core.registration import load_image

        if path is None:
            return load_image(item, max_long_edge=edge)

        def create() -> bytes:
            return self._encode_preview(load_image(path, max_long_edge=edge))

        if (
            self._cache is None
            or not bool(self.options.preserve_cache)
            or not callable(getattr(self._cache, "get_or_create_preview_bytes", None))
        ):
            return load_image(path, max_long_edge=edge)
        payload = self._cache.get_or_create_preview_bytes(path, create, suffix=".jpg")
        return self._decode_preview(payload)

    def _analysis_loader(self, item: Any, max_long_edge: int | None = None) -> Any:
        path = _path_for_item(item)
        if path is None:
            return item
        from ..utils.image_io import load_rgb

        return load_rgb(path, max_long_edge)

    def _build_analyzer(self) -> Any:
        if self._analyzer is not None:
            return self._analyzer
        from ..core.coverage_selector import CoverageConfig
        from ..core.group_analyzer import GroupAnalyzer

        analysis = getattr(self.config, "analysis", None)
        self._analyzer = GroupAnalyzer(
            self.config,
            database=self._database,
            cache=self._cache if self.options.preserve_cache else None,
            plan_cache=self._database if self.options.preserve_cache else None,
            loader=self._analysis_loader,
            coverage_config=CoverageConfig(
                mode="balanced",
                focus_threshold=float(getattr(analysis, "focus_threshold", 0.95)),
                min_coverage_gain=float(getattr(analysis, "min_coverage_gain", 0.002)),
            ),
            logger=self.logger,
        )
        return self._analyzer

    def _build_merger(self) -> Any:
        if self._merge_service is not None:
            return self._merge_service
        manifest = ManifestWriter(self.output_dir / "stack_manifest.csv")
        archiver = (
            FileArchiver(self.archive_dir, self.options.archive_mode, repository=self._database, logger=self.logger)
            if self.options.archive_enabled
            else None
        )
        runtime = getattr(self.config, "runtime", None)
        self._merge_service = StackMergeService(
            self.output_dir,
            archive_dir=self.archive_dir,
            archiver=archiver,
            archive_enabled=self.options.archive_enabled,
            output_config=getattr(self.config, "output", None),
            repository=self._database,
            cache_dir=self._cache_paths.root if self._cache_paths is not None else self.output_dir / ".stack_cache",
            runtime_config=runtime,
            hugin_bin=self.options.hugin_bin,
            align_image_stack_path=self.options.align_image_stack_path,
            enfuse_path=self.options.enfuse_path,
            manifest_writer=manifest,
            fusion_backend=self.options.fusion_backend,
            minimum_stack_group_size=self.options.minimum_stack_group_size,
            logger=self.logger,
        )
        return self._merge_service

    def _build_coordinator(self, analyzer: Any, merger: Any) -> Any:
        if self._coordinator is not None:
            coordinator = self._coordinator
            try:
                # A caller may inject a configured coordinator, or only the
                # coordinator shell.  Supplying the resolved defaults here
                # keeps both forms compatible with the production wiring.
                if getattr(coordinator, "analyzer", None) is None:
                    coordinator.analyzer = analyzer
                if getattr(coordinator, "merger", None) is None:
                    coordinator.merger = merger
                if getattr(coordinator, "memory_guard", None) is None:
                    coordinator.memory_guard = self._memory_guard
                if getattr(coordinator, "job_state", None) is None:
                    coordinator.job_state = self._job_state
                coordinator.event_callback = self._on_pipeline_event
                coordinator.complete_callback = None
            except Exception:
                pass
            return coordinator
        manifest_path = self.output_dir / "stack_manifest.csv"
        return PipelineCoordinator(
            analyzer=analyzer,
            merger=merger,
            config=PipelineConfig(
                queue_size=self.options.queue_size,
                merge_workers=self.options.merge_workers,
                parallel=self.options.parallel,
            ),
            repository=self._database,
            event_callback=self._on_pipeline_event,
            manifest_path=manifest_path,
            memory_guard=self._memory_guard,
            job_state=self._job_state,
            logger=self.logger,
        )

    def _run(self) -> None:
        started = self._started_at or time.monotonic()
        summary: ControllerSummary | None = None
        report: Any | None = None
        groups: list[Any] = []
        focus_budget: Any | None = None
        memory_profile = BatchMemory(self.logger)
        memory_profile.start()
        try:
            source, output = self._validate_paths()
            self._prepare_runtime(source, output)
            recovery_groups, recovery_payloads = self._load_recovery_groups()
            records: list[Any] = []
            if recovery_groups:
                # A MOVE can leave the source directory empty after a clean
                # archive but before fusion starts.  Rebuild the batch from DB
                # rows instead of asking the scanner to rediscover vanished
                # originals.
                seen_images: set[str] = set()
                for group in recovery_groups:
                    for item in _items_for_group(group):
                        key = str(_mapping_value(item, "id", default=_path_for_item(item) or "")).casefold()
                        if key in seen_images:
                            continue
                        seen_images.add(key)
                        records.append(item)
                report = SimpleNamespace(
                    root=str(source),
                    images=list(records),
                    records=list(records),
                    count=len(records),
                    total_bytes=sum(int(_mapping_value(item, "file_size", default=0) or 0) for item in records),
                    issues=(),
                )
                groups = recovery_groups
                self._groups_found = len(groups)
                self._emit(
                    PipelineStage.SCANNING,
                    completed=len(groups),
                    total=len(groups),
                    groups_found=len(groups),
                    message=f"恢复 {len(groups)} 个未完成组（使用归档路径）",
                )
            else:
                recovery_payloads = {}
                self._emit(PipelineStage.SCANNING, total=0, message="正在扫描 JPG…")
                report, records = self._scan()
            if self._cancel_event.is_set():
                raise ControllerCancelled()
            scan_issues = list(getattr(report, "issues", ()) or ())
            if scan_issues:
                self._event_errors += len(scan_issues)
                self._diagnostics.extend(
                    f"扫描警告：{getattr(issue, 'path', '')}：{getattr(issue, 'message', issue)}"
                    for issue in scan_issues
                )
            total_bytes = int(getattr(report, "total_bytes", 0) or sum(int(item.file_size) for item in records))
            self._disk_check(total_bytes)
            if not recovery_groups:
                self._persist_images(records)
                if self._cancel_event.is_set():
                    raise ControllerCancelled()
                groups = self._scene_groups(records)
                self._persist_groups(groups, records)
                self._groups_found = len(groups)
            if self._cancel_event.is_set():
                raise ControllerCancelled()
            runtime = getattr(self.config, "runtime", None)
            largest_group = max((len(_items_for_group(group)) for group in groups), default=1)
            try:
                import cv2  # type: ignore[import-not-found]

                current_opencv_threads = int(cv2.getNumThreads())
            except ImportError:
                current_opencv_threads = None
            focus_budget = resolve_focus_analysis_budget(
                int(getattr(runtime, "focus_analysis_workers", self.options.focus_analysis_workers) or 0),
                image_count=largest_group,
                merge_workers=self.options.merge_workers,
                parallel_pipeline=self.options.parallel,
                minimum_available_bytes=int(
                    getattr(runtime, "min_available_memory_bytes", 2 * 1024**3) or 0
                ),
                minimum_available_fraction=float(
                    getattr(runtime, "min_available_memory_fraction", 0.10) or 0
                ),
                current_opencv_threads=current_opencv_threads,
            )
            self.logger.info(
                "Focus analysis budget requested_workers=%s effective_workers=%s "
                "opencv_threads=%s logical_cpus=%s cpu_budget=%s memory_worker_cap=%s "
                "memory_limited=%s",
                focus_budget.requested_workers,
                focus_budget.workers,
                focus_budget.opencv_threads,
                focus_budget.logical_cpus,
                focus_budget.cpu_budget,
                focus_budget.memory_worker_cap,
                focus_budget.memory_limited,
            )
            with opencv_thread_budget(focus_budget.opencv_threads):
                analyzer = self._build_analyzer()
                if recovery_payloads:
                    analyzer = _RecoveryAnalyzerAdapter(analyzer, recovery_payloads)
                merger = self._build_merger()
                coordinator = self._build_coordinator(analyzer, merger)
                self._coordinator = coordinator
                pipeline_summary = coordinator.run(groups)
            for failed_group, error in list(getattr(pipeline_summary, "failed_merge", ()) or ()):
                group_label = _group_id(getattr(failed_group, "group", failed_group))
                detail = f"融合失败（组 {group_label!r}）：{type(error).__name__}: {error}"
                if detail not in self._diagnostics:
                    self._diagnostics.append(detail)
            summary = ControllerSummary(
                source_dir=self.source_dir,
                output_dir=self.output_dir,
                archive_dir=self.archive_dir,
                scan_report=report,
                groups=list(groups),
                pipeline_summary=pipeline_summary,
                # ``_event_errors`` already tracks the coordinator's live
                # error count (and any preflight diagnostics); adding the
                # final worker count would double-count the same failures.
                errors=max(self._event_errors, int(pipeline_summary.errors)),
                diagnostics=list(self._diagnostics),
                cancelled=bool(pipeline_summary.cancelled or self._cancel_event.is_set()),
                elapsed_seconds=time.monotonic() - started,
            )
            if summary.cancelled:
                self._emit(PipelineStage.CANCELLED, groups_found=len(groups), message="处理已取消")
        except ControllerCancelled:
            summary = ControllerSummary(
                source_dir=self.source_dir,
                output_dir=self.output_dir,
                archive_dir=self.archive_dir,
                scan_report=report,
                groups=list(groups),
                errors=self._event_errors,
                diagnostics=[*self._diagnostics, "处理已取消"],
                cancelled=True,
                elapsed_seconds=time.monotonic() - started,
            )
            self._emit(PipelineStage.CANCELLED, errors=self._event_errors, message="处理已取消")
        except Exception as exc:
            self.logger.exception("Application pipeline failed")
            self._event_errors += 1
            summary = ControllerSummary(
                source_dir=self.source_dir,
                output_dir=self.output_dir,
                archive_dir=self.archive_dir,
                scan_report=report,
                groups=list(groups),
                errors=self._event_errors,
                diagnostics=[*self._diagnostics, f"{type(exc).__name__}: {exc}"],
                cancelled=bool(self._cancel_event.is_set()),
                elapsed_seconds=time.monotonic() - started,
            )
            self._emit(
                PipelineStage.CANCELLED if summary.cancelled else PipelineStage.ERROR,
                errors=self._event_errors,
                message=summary.diagnostics[0],
            )
        finally:
            memory_profile.finish(
                time.monotonic() - started,
                merge_workers=self.options.merge_workers,
                focus_analysis_workers_requested=(
                    getattr(focus_budget, "requested_workers", self.options.focus_analysis_workers)
                ),
                focus_analysis_workers_effective=getattr(focus_budget, "workers", None),
                opencv_threads=getattr(focus_budget, "opencv_threads", None),
                errors=getattr(summary, "errors", None),
                cancelled=getattr(summary, "cancelled", False),
            )
            self._summary = summary
            callback = self.complete_callback
            if callback is not None and summary is not None:
                try:
                    callback(summary)
                except Exception:
                    self.logger.debug("Application completion callback failed", exc_info=True)


__all__ = [
    "ApplicationController",
    "ApplicationOptions",
    "ControllerOptions",
    "PipelineOptions",
    "ControllerSummary",
    "ControllerReport",
    "ControllerError",
    "DiskSpaceError",
    "ControllerCancelled",
    "MemoryGuard",
    "cleanup_stale_temp_dirs",
]

