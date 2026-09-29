"""Merge consumer and image fusion service."""

from __future__ import annotations

from ..utils.performance import timed

from dataclasses import dataclass, is_dataclass
import inspect
import logging
import os
from pathlib import Path
import shutil
import threading
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid

from ..files.archiver import ArchiveResult, FileArchiver
from ..hugin.align import AlignImageStack, AlignmentError
from ..hugin.enfuse import Enfuser, EnfuseError
from ..hugin.hugin_locator import HuginToolNotFound
from ..hugin.output_encoder import OutputConfig, OutputFormat, output_path_for
from ..fusion.backends import FusionBackend, HuginEnfuseBackend, QualityFusionBackend
from ..fusion_modes import normalize_fusion_backend
from .analysis_worker import AnalysisJob
from .events import PipelineEvent, PipelineStage
from .job_queue import BoundedJobQueue, QueueClosed


@dataclass
class MergeResult:
    group: Any
    status: str
    output_path: Path | None = None
    aligned_paths: tuple[Path, ...] = ()
    archive_result: ArchiveResult | None = None
    work_dir: Path | None = None
    error: Exception | None = None
    # Keep the analysis/selection evidence on the result boundary.  The
    # manifest writer consumes these fields without importing algorithm
    # models, and the defaults preserve the old positional constructor.
    analysis: Any = None
    all_paths: tuple[Path, ...] = ()
    selected_source_paths: tuple[Path, ...] = ()
    first_original: Path | None = None
    requested_backend: str = "quality"
    actual_backend: str | None = None
    fallback_used: bool = False
    alignment_level: int | None = None
    alignment_status: str | None = None
    crop_ratio: float | None = None
    diagnostics: tuple[str, ...] = ()
    actual_hugin_input_order: tuple[Path, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status in {
            "DONE",
            "SKIPPED_SINGLE",
            "SINGLE",
            "NO_MERGE",
            "ARCHIVED_ONLY",
        } and self.error is None

    @property
    def group_id(self) -> int | str | None:
        return _group_id(self.group)


def _value(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
    else:
        for name in names:
            try:
                value = getattr(obj, name)
            except Exception:
                continue
            if value is not None:
                return value
    return default


def _group_id(group: Any) -> int | str | None:
    return _value(group, "group_id", "id", default=None)


def _item_path(item: Any) -> Path | None:
    if isinstance(item, (str, os.PathLike)):
        return Path(item)
    value = _value(item, "current_path", "path", "original_path", "filename", default=None)
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(os.fspath(value))


def _objects(group: Any, analysis: Any) -> tuple[list[Any], Any]:
    """Return all image objects and the selected-image evidence object."""

    evidence = analysis if analysis is not None else group
    candidates = [analysis, group]
    all_values: Any = None
    for obj in candidates:
        all_values = _value(obj, "all_images", "images", "items", "image_records", "all_paths", "image_paths", default=None)
        if all_values is not None:
            break
    if all_values is None:
        all_values = []
    if isinstance(all_values, (str, os.PathLike)):
        all_values = [all_values]
    try:
        items = list(all_values)
    except TypeError:
        items = []
    return items, evidence


def extract_image_paths(group: Any, analysis: Any = None) -> tuple[list[Path], list[Path], Path | None]:
    """Extract ``all``, ``selected`` and first-original paths without schema coupling."""

    items, evidence = _objects(group, analysis)
    paths: list[Path] = []
    item_to_path: dict[int, Path] = {}
    for item in items:
        path = _item_path(item)
        if path is not None:
            item_to_path[len(paths)] = path
            paths.append(path)

    selected_raw: Any = None
    selected_supplied = False
    for obj in (analysis, group):
        selected_raw = _value(obj, "selected_paths", "selected_images", "selected_items", "selected", default=None)
        if selected_raw is not None:
            selected_supplied = not isinstance(selected_raw, bool)
            break
    selected: list[Path] = []
    if selected_raw is not None and not isinstance(selected_raw, bool):
        if isinstance(selected_raw, (str, os.PathLike)):
            selected_raw = [selected_raw]
        try:
            values = list(selected_raw)
        except TypeError:
            values = []
        # CoverageSelection exposes selected_indices.  Also accept a plain
        # sequence of integer indices as a convenient test-double format.
        if values and all(isinstance(value, int) for value in values) and paths:
            selected = [paths[index] for index in values if -len(paths) <= index < len(paths)]
        else:
            for value in values:
                path = _item_path(value)
                if path is None:
                    continue
                current_keys = {
                    os.path.normcase(os.path.abspath(os.fspath(current)))
                    for current in paths
                }
                if os.path.normcase(os.path.abspath(os.fspath(path))) in current_keys:
                    selected.append(path)
                    continue
                # A cached analysis may retain an original_path string while
                # the corresponding ImageRecord has already been archived and
                # now exposes only its current_path.  Resolve that evidence
                # against the current group objects before fusion.
                raw_key = os.path.normcase(os.path.abspath(os.fspath(path)))
                for item, current in zip(items, paths):
                    alternatives = [
                        _item_path(item),
                        _value(item, "original_path", default=None),
                    ]
                    if any(
                        alternative is not None
                        and os.path.normcase(os.path.abspath(os.fspath(alternative))) == raw_key
                        for alternative in alternatives
                    ):
                        selected.append(current)
                        break
                else:
                    basename_matches = [
                        current for item, current in zip(items, paths)
                        if Path(os.fspath(_value(item, "original_path", default=_item_path(item) or current))).name.casefold()
                        == path.name.casefold()
                    ]
                    if len(basename_matches) == 1:
                        selected.extend(basename_matches)
    if not selected and not selected_supplied:
        indices = _value(analysis, "selected_indices", "selected", default=None)
        if isinstance(indices, Sequence) and not isinstance(indices, (str, bytes)) and paths:
            try:
                selected = [paths[int(index)] for index in indices]
            except (TypeError, ValueError, IndexError):
                selected = []
            selected_supplied = True
    if not selected and not selected_supplied and items:
        flags = [_value(item, "selected", default=None) for item in items]
        if any(flag is not None for flag in flags):
            selected = [path for path, flag in zip(paths, flags) if bool(flag)]
            selected_supplied = True
    if not selected and not selected_supplied:
        selected = list(paths)

    # De-duplicate paths but preserve the selection order supplied by the
    # algorithm; every backend receives that same focus-transition order.
    def unique(values: Iterable[Path]) -> list[Path]:
        seen: set[str] = set()
        result: list[Path] = []
        for path in values:
            key = os.path.normcase(os.path.abspath(os.fspath(path)))
            if key not in seen:
                seen.add(key)
                result.append(path)
        return result

    paths, selected = unique(paths), unique(selected)
    first = None
    for obj in (group, analysis):
        first_value = _value(obj, "first_original_image", "first_original", "first_image", "first_item", default=None)
        if first_value is not None:
            first = _item_path(first_value)
            if first is not None:
                break
    if first is None and items:
        first = _item_path(items[0])
    if first is None and paths:
        first = paths[0]
    if first is not None and paths:
        first_key = os.path.normcase(os.path.abspath(os.fspath(first)))
        current_keys = {
            os.path.normcase(os.path.abspath(os.fspath(path))) for path in paths
        }
        if first_key not in current_keys:
            for item, current in zip(items, paths):
                original = _value(item, "original_path", default=None)
                if original is not None and os.path.normcase(os.path.abspath(os.fspath(original))) == first_key:
                    first = current
                    break
            else:
                matches = [path for path in paths if path.name.casefold() == first.name.casefold()]
                if len(matches) == 1:
                    first = matches[0]
    return paths, selected, first


def _set_group_state(repository: Any, group: Any, status: str, **values: Any) -> None:
    if repository is None:
        return
    gid = _group_id(group)
    if gid is None:
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
                logging.getLogger(__name__).debug("Unable to set group state %s", gid, exc_info=True)
    updater = getattr(repository, "update_group", None)
    if callable(updater):
        try:
            if isinstance(group, Mapping):
                group["status"] = status
                group.update(values)
            else:
                setattr(group, "status", status)
                for key, value in values.items():
                    if hasattr(group, key):
                        setattr(group, key, value)
            updater(group)
        except Exception:
            logging.getLogger(__name__).debug("Unable to persist group state %s", gid, exc_info=True)


def _invoke(callback: Callable[..., Any], job: AnalysisJob, *, cancel_event: threading.Event) -> Any:
    fn = callback
    if not callable(fn):
        fn = getattr(callback, "process", getattr(callback, "merge", None))
    if not callable(fn):
        raise TypeError("merger must be callable or expose process()/merge()")
    try:
        params = inspect.signature(fn).parameters
        accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs = {"cancel_event": cancel_event} if accepts_kwargs or "cancel_event" in params else {}
    except (TypeError, ValueError):
        kwargs = {}
    return fn(job, **kwargs)


class StackMergeService:
    """Fuse the selected frames, publish the result, then optionally archive.

    Original files are not handed to the archiver until the fused output has
    been published successfully.  This ordering is intentional: a missing
    An alignment or fusion error must leave the source folder untouched.
    """

    def __init__(
        self,
        output_dir: os.PathLike[str] | str,
        *,
        archive_dir: os.PathLike[str] | str | None = None,
        archiver: FileArchiver | None = None,
        archive_enabled: bool = True,
        aligner: AlignImageStack | Any | None = None,
        enfuser: Enfuser | Any | None = None,
        output_config: OutputConfig | Any | None = None,
        repository: Any | None = None,
        cache_dir: os.PathLike[str] | str | None = None,
        runtime_config: Any | None = None,
        runtime: Any | None = None,
        hugin_bin: os.PathLike[str] | str | None = None,
        align_image_stack_path: os.PathLike[str] | str | None = None,
        enfuse_path: os.PathLike[str] | str | None = None,
        manifest_writer: Any | None = None,
        fusion_backend: str | FusionBackend = "quality",
        minimum_stack_group_size: int = 4,
        logger: logging.Logger | None = None,
    ):
        self.minimum_stack_group_size = max(2, int(minimum_stack_group_size))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.archive_dir = Path(archive_dir) if archive_dir is not None else self.output_dir.parent / "归档"
        self.archive_enabled = bool(archive_enabled)
        self.archiver = (
            (archiver or FileArchiver(self.archive_dir, repository=repository, logger=self.logger))
            if self.archive_enabled
            else None
        )
        # RuntimeConfig is deliberately accepted as an untyped adapter.  This
        # keeps the pipeline importable when the configuration/storage agent is
        # not installed yet while still honoring the user's explicit Hugin
        # executable overrides in the real application.
        runtime = runtime_config if runtime_config is not None else runtime

        def runtime_value(*names: str) -> Any:
            if runtime is None:
                return None
            sources: list[Any] = [runtime]
            nested = runtime.get("runtime") if isinstance(runtime, Mapping) else getattr(runtime, "runtime", None)
            if nested is not None:
                sources.append(nested)
            for source in sources:
                if isinstance(source, Mapping):
                    for name in names:
                        if source.get(name) is not None:
                            return source[name]
                else:
                    for name in names:
                        value = getattr(source, name, None)
                        if value is not None:
                            return value
            return None

        hugin_root = hugin_bin if hugin_bin is not None else runtime_value("hugin_bin", "hugin_path")
        align_path = (
            align_image_stack_path
            if align_image_stack_path is not None
            else runtime_value("align_image_stack_path", "align_image_stack", "align_path")
        )
        enfuse_executable = (
            enfuse_path
            if enfuse_path is not None
            else runtime_value("enfuse_path", "enfuse", "enfuse_executable")
        )
        requested = (fusion_backend.name if isinstance(fusion_backend, FusionBackend)
                     else normalize_fusion_backend(fusion_backend))
        create_hugin_backend = not isinstance(fusion_backend, FusionBackend) and requested == "hugin_enfuse"
        self.aligner = (aligner or AlignImageStack(align_path or hugin_root, logger=self.logger)
                        if create_hugin_backend else aligner)
        self.enfuser = (enfuser or Enfuser(enfuse_executable or hugin_root, logger=self.logger)
                        if create_hugin_backend else enfuser)
        for component in (self.aligner, self.enfuser):
            if component is None:
                continue
            try:
                component.logger = self.logger
                runner = getattr(component, "runner", None)
                if runner is not None and hasattr(runner, "logger"):
                    runner.logger = self.logger
            except Exception:
                # Lightweight mock components need not expose logger fields;
                # production adapters above are constructed with this logger.
                pass
        self.output_config = output_config or OutputConfig()
        if isinstance(fusion_backend, FusionBackend):
            self.backend = fusion_backend
            self.requested_backend = fusion_backend.name
        else:
            self.requested_backend = requested
            if self.requested_backend == "quality":
                self.backend = QualityFusionBackend(aligned_cache_bytes=runtime_value("opencv_aligned_cache_bytes"))
            elif self.requested_backend == "hugin_enfuse":
                self.backend = HuginEnfuseBackend(
                    self.aligner, self.enfuser, runtime_config=runtime, logger=self.logger,
                )
            else:
                raise ValueError("fusion_backend must be quality or hugin_enfuse")
        self.repository = repository
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.output_dir / ".stack_cache"
        if manifest_writer is None:
            # A real pipeline should leave a durable manifest even when the
            # caller only supplied an output directory.  Import lazily to
            # preserve the low-dependency/headless construction path.
            from ..storage.manifest import ManifestWriter

            manifest_writer = ManifestWriter(self.output_dir / "stack_manifest.csv")
        elif isinstance(manifest_writer, (str, os.PathLike)):
            from ..storage.manifest import ManifestWriter

            manifest_writer = ManifestWriter(manifest_writer)
        self.manifest_writer = manifest_writer

    def _archive_map(self, all_objects: list[Any], all_paths: list[Path]) -> tuple[ArchiveResult, dict[str, Path]]:
        if self.archiver is None:
            raise RuntimeError("archiver is unavailable while archival is disabled")
        # Pass original objects so ImageRecord.id reaches Database.update_image_path.
        values: list[Any] = all_objects if all_objects and any(not isinstance(item, (str, os.PathLike)) for item in all_objects) else all_paths
        archive = self.archiver.archive_files(values)
        mapping: dict[str, Path] = {}
        for record in archive.records:
            if record.status in {"archived", "already_archived"}:
                mapping[os.path.normcase(os.path.abspath(record.source_path))] = Path(record.destination_path)
            elif record.status == "conflict":
                destination = Path(record.destination_path)
                source = Path(record.source_path)
                # A prior successful run may have left a matching destination;
                # accepting it makes resume idempotent without hiding a real
                # conflicting file.
                try:
                    from ..files.archiver import files_match

                    if destination.is_file() and files_match(source, destination):
                        mapping[os.path.normcase(os.path.abspath(record.source_path))] = destination
                except OSError:
                    pass
        return archive, mapping

    def _publish_manifest(self, result: MergeResult) -> None:
        writer = self.manifest_writer
        if writer is None:
            return
        try:
            update = getattr(writer, "update_result", None)
            if callable(update):
                update(result)
                return
            append = getattr(writer, "append_result", None)
            if callable(append):
                append(result)
        except Exception:
            # Manifest export is best-effort observability; it must not turn a
            # safely archived or fused group into a pipeline failure.
            self.logger.debug("Unable to update manifest for group %s", _group_id(result.group), exc_info=True)

    def _cleanup_successful_work_dir(self, work: Path) -> None:
        """Remove only the private per-group directory created by this service."""

        try:
            temp_root = (self.cache_dir / "temp").resolve()
            target = work.resolve()
            if target.parent != temp_root:
                self.logger.warning(
                    "refusing to clean unexpected fusion work directory: %s", target,
                )
                return
            shutil.rmtree(target)
            self.logger.info("cleaned successful fusion work directory: %s", target)
        except FileNotFoundError:
            return
        except OSError:
            # The final image and manifest are already durable.  A locked TIFF
            # should be visible in logs but must not downgrade a good stack.
            self.logger.warning(
                "unable to clean successful fusion work directory: %s", work, exc_info=True,
            )

    @timed("merge_inclusive")
    def process(self, job: AnalysisJob | Any, *, cancel_event: threading.Event | None = None) -> MergeResult:
        cancel_event = cancel_event or threading.Event()
        group = job.group if isinstance(job, AnalysisJob) else _value(job, "group", default=job)
        analysis = job.analysis if isinstance(job, AnalysisJob) else _value(job, "analysis", default=None)
        archive_only = bool(_value(job, "archive_only", default=False))
        analysis_error = _value(job, "error", default=None)
        all_objects, _ = _objects(group, analysis)
        all_paths, selected_paths, first = extract_image_paths(group, analysis)
        if not all_paths:
            raise ValueError(f"Group {_group_id(group)!r} contains no image paths")
        if first is None:
            first = all_paths[0]
        # Preserve the original first-image identity for output naming even
        # when a resumed ImageRecord now exposes an archived current_path (or
        # when an explicit collision policy renamed the archive destination).
        first_original_for_name = first
        for obj in (group, analysis):
            candidate = _value(
                obj,
                "first_original_image",
                "first_original_path",
                "first_original",
                "first_image",
                "first_item",
                default=None,
            )
            candidate_path = _value(
                candidate,
                "original_path",
                "current_path",
                "path",
                "filename",
                default=None,
            )
            if candidate_path is None and candidate is not None:
                candidate_path = _item_path(candidate)
            if candidate_path is not None:
                try:
                    first_original_for_name = Path(os.fspath(candidate_path))
                except TypeError:
                    self.logger.debug("Unable to resolve first original path for group %s", _group_id(group), exc_info=True)
                break
        if archive_only:
            # Classification/focus-analysis failures and successful no-merge
            # decisions leave every source in place. Only a real fused output
            # makes the complete group eligible for post-success archive.
            state = "FAILED_CLASSIFICATION" if analysis_error is not None else "CLASSIFIED"
            _set_group_state(
                self.repository,
                group,
                state,
                output_path=None,
                selected_count=0 if analysis_error is not None else len(selected_paths),
                image_count=len(all_paths),
            )
            result_status = "FAILED_CLASSIFICATION" if analysis_error is not None else "NO_MERGE"
            result = MergeResult(
                group,
                result_status,
                error=analysis_error,
                analysis=analysis,
                all_paths=tuple(all_paths),
                selected_source_paths=tuple() if analysis_error is not None else tuple(selected_paths),
                first_original=first_original_for_name,
                requested_backend=self.requested_backend,
            )
            self._publish_manifest(result)
            return result
        if len(all_paths) < self.minimum_stack_group_size or len(selected_paths) <= 1:
            _set_group_state(self.repository, group, "CLASSIFIED", output_path=None, selected_count=len(selected_paths))
            result = MergeResult(
                group,
                "NO_MERGE",
                analysis=analysis,
                all_paths=tuple(all_paths),
                selected_source_paths=tuple(selected_paths),
                first_original=first_original_for_name,
                requested_backend=self.requested_backend,
            )
            self._publish_manifest(result)
            return result
        if cancel_event.is_set():
            _set_group_state(self.repository, group, "CANCELLED")
            result = MergeResult(
                group,
                "CANCELLED",
                analysis=analysis,
                all_paths=tuple(all_paths),
                selected_source_paths=tuple(selected_paths),
                first_original=first_original_for_name,
                requested_backend=self.requested_backend,
            )
            self._publish_manifest(result)
            return result

        work = self.cache_dir / "temp" / f"{_group_id(group) or 'group'}_{uuid.uuid4().hex}"
        work.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "fusion group_id=%s original_count=%s selected_count=%s anchor=%s capture_order=%s "
            "preview_reference=%s alignment_order=%s alignment_order_confidence=%s requested_backend=%s",
            _group_id(group), len(all_paths), len(selected_paths), first_original_for_name,
            _value(analysis, "capture_order", default=all_paths),
            _value(analysis, "preview_reference", default=""),
            _value(analysis, "alignment_order", default=selected_paths),
            _value(analysis, "alignment_order_confidence", default=None), self.requested_backend,
        )
        _set_group_state(
            self.repository, group, "ALIGNING", selected_count=len(selected_paths),
            requested_backend=self.requested_backend,
            alignment_order=_value(analysis, "alignment_order", default=selected_paths),
            alignment_order_confidence=_value(analysis, "alignment_order_confidence", default=None),
            alignment_order_fallback_used=_value(analysis, "alignment_order_fallback_used", default=False),
        )
        # The backend receives the GroupAnalyzer's explicit order; output
        # naming continues to use the immutable group anchor above.
        _set_group_state(self.repository, group, "FUSING")
        fmt = getattr(self.output_config, "format", "jpg")
        fmt_value = getattr(fmt, "value", fmt)
        suffix = getattr(self.output_config, "output_suffix", "_stack")
        destination = output_path_for(first_original_for_name, self.output_dir, fmt, suffix=suffix if str(fmt_value).casefold().lstrip(".") in {"jpg", "jpeg", "tif", "tiff"} else None)
        # output_path_for applies _stack only for JPG when suffix is omitted;
        # for a shared config object, preserve the explicit suffix for JPG and
        # avoid it for TIFF as required by the V1 naming rule.
        if str(fmt_value).casefold().lstrip(".") in {"tif", "tiff"}:
            destination = output_path_for(first_original_for_name, self.output_dir, fmt)
        fusion = self.backend.fuse(
            group, analysis, destination, work, self.output_config, cancel_event,
        )
        fused_path = Path(fusion.output_path)
        if not fused_path.is_file():
            raise RuntimeError(f"Fusion reported success but output file is missing: {fused_path}")
        self.logger.info(
            "fusion result group_id=%s actual_backend=%s alignment_level=%s alignment_status=%s "
            "crop_ratio=%s fallback_used=%s actual_hugin_input_order=%s output_path=%s final_state=DONE diagnostics=%s",
            _group_id(group), fusion.actual_backend, fusion.alignment_level,
            fusion.alignment_status, fusion.crop_ratio, fusion.fallback_used,
            fusion.actual_hugin_input_order, fused_path, fusion.diagnostics,
        )

        archive = None
        if self.archive_enabled:
            # Archive every original in the scene only after the composite
            # exists. Selected paths drive fusion; all_paths is the complete
            # source set.
            _set_group_state(self.repository, group, "ARCHIVING", output_path=str(fused_path))
            archive, destinations = self._archive_map(all_objects, all_paths)
            if archive.failed or archive.conflicts and any(
                os.path.normcase(os.path.abspath(record.source_path)) not in destinations for record in archive.records
            ):
                detail = "; ".join(
                    record.error or record.source_path
                    for record in archive.records
                    if record.status in {"failed", "conflict"}
                )
                raise RuntimeError(f"Unable to archive group {_group_id(group)!r} after fusion: {detail}")

        _set_group_state(
            self.repository, group, "DONE", output_path=str(fused_path), selected_count=len(selected_paths),
            requested_backend=self.requested_backend, actual_backend=fusion.actual_backend,
            alignment_level=fusion.alignment_level, alignment_status=fusion.alignment_status,
            crop_ratio=fusion.crop_ratio, diagnostics=list(fusion.diagnostics),
        )
        result = MergeResult(
            group,
            "DONE",
            fused_path,
            fusion.aligned_paths,
            archive,
            work,
            analysis=analysis,
            all_paths=tuple(all_paths),
            selected_source_paths=tuple(selected_paths),
            first_original=first_original_for_name,
            requested_backend=self.requested_backend,
            actual_backend=fusion.actual_backend,
            fallback_used=fusion.fallback_used,
            alignment_level=fusion.alignment_level,
            alignment_status=fusion.alignment_status,
            crop_ratio=fusion.crop_ratio,
            diagnostics=fusion.diagnostics,
            actual_hugin_input_order=fusion.actual_hugin_input_order,
        )
        self._publish_manifest(result)
        self._cleanup_successful_work_dir(work)
        return result

    merge = process
    run = process


class MergeWorker(threading.Thread):
    """Consume analysis jobs and run the selected fusion backend."""

    def __init__(
        self,
        merge_queue: BoundedJobQueue[AnalysisJob],
        merger: Callable[..., Any],
        *,
        cancel_event: threading.Event | None = None,
        event_callback: Callable[[PipelineEvent], Any] | None = None,
        repository: Any | None = None,
        total: int = 0,
        # In coordinator serial mode workers start early solely to drain the
        # bounded queue.  They defer fusion until this event is set,
        # which removes the producer-join deadlock without changing the
        # requested serial processing semantics.
        defer_until: threading.Event | None = None,
        result_callback: Callable[[Any], Any] | None = None,
        logger: logging.Logger | None = None,
        memory_guard: Any | None = None,
        job_state: Any | None = None,
    ):
        super().__init__(name="focus-stack-merge", daemon=True)
        self.merge_queue = merge_queue
        self.merger = merger
        self.cancel_event = cancel_event or threading.Event()
        self.event_callback = event_callback
        self.repository = repository
        self.total = int(total)
        self.defer_until = defer_until
        self.result_callback = result_callback
        self.logger = logger or logging.getLogger(__name__)
        self.memory_guard = memory_guard
        self.job_state = job_state
        self.completed = 0
        # ``completed`` remains the number of successful merge results (and
        # historical failed fusion attempts).  ``finished`` counts every
        # consumed job, including archive-only classification failures, so a
        # coordinator can distinguish an archive-only completion from work
        # that is still waiting in the merge queue.
        self.finished = 0
        self.errors = 0
        self.results: list[Any] = []
        self.failed_jobs: list[tuple[AnalysisJob, Exception]] = []
        self._deferred_jobs: list[AnalysisJob] = []

    def _update_job(
        self,
        group: Any,
        *,
        stage: str,
        status: str,
        progress: float | None = None,
        error: Exception | str | None = None,
    ) -> None:
        updater = getattr(self.job_state, "update", None) if self.job_state is not None else None
        if not callable(updater):
            return
        try:
            updater(
                group,
                stage=stage,
                status=status,
                progress=progress,
                error=None if error is None else str(error),
            )
        except Exception:
            self.logger.debug("Unable to persist merge job state", exc_info=True)

    def _emit(self, job: AnalysisJob | None = None, *, stage: PipelineStage = PipelineStage.FUSION, message: str = "") -> None:
        if self.event_callback is None:
            return
        group = job.group if job is not None else None
        file_value = ""
        if group is not None:
            _, _, first = extract_image_paths(group, job.analysis)
            file_value = str(first or "")
        event = PipelineEvent(
            stage=stage,
            current_file=file_value,
            current_group=_group_id(group),
            completed=self.completed,
            total=self.total,
            groups_finished=self.finished,
            errors=self.errors,
            message=message,
            merge_completed=self.completed,
            merge_total=self.total,
            merge_finished=self.finished,
        )
        try:
            self.event_callback(event)
        except Exception:
            self.logger.debug("Progress callback failed", exc_info=True)

    def _notify_result(self, result: Any) -> None:
        if self.result_callback is None:
            return
        try:
            self.result_callback(result)
        except Exception:
            # Manifest/UI callbacks are observability concerns and must not
            # turn a successfully completed group into a merge failure.
            self.logger.debug("Merge result callback failed", exc_info=True)

    @staticmethod
    def _status(result: Any, default: str = "DONE") -> str:
        if isinstance(result, Mapping):
            return str(result.get("status", default))
        return str(getattr(result, "status", default))

    def _process_job(self, job: AnalysisJob) -> None:
        """Process one job; queue accounting is owned by :meth:`run`."""

        if self.cancel_event.is_set():
            _set_group_state(self.repository, job.group, "CANCELLED")
            self._update_job(job.group, stage="FUSION", status="CANCELLED")
            return
        if job.archive_only and job.error is not None:
            # Keep the classification failure visible while the archive-only
            # consumer performs its required file-safety work.  The default
            # service may transition this to its final diagnostic state after
            # archiving, but a lightweight test merger must not erase it.
            _set_group_state(self.repository, job.group, "FAILED_CLASSIFICATION")
        else:
            _set_group_state(self.repository, job.group, "QUEUED_FOR_MERGE")
        self._emit(job, message="进入合成队列")
        try:
            result = _invoke(self.merger, job, cancel_event=self.cancel_event)
            self._notify_result(result)
            # A classification failure is intentionally consumed by the
            # archive-only path so every source can be relocated, but it is
            # already represented in ``analysis.errors``.  Keep it out of the
            # successful merge-result list/counter while retaining the result
            # callback for manifests and diagnostics.
            if not (job.archive_only and job.error is not None):
                self.results.append(result)
                self.completed += 1
            self._update_job(
                job.group,
                stage="EXPORT",
                status="DONE",
                progress=1.0,
                error=job.error,
            )
            self._emit(job, message=f"合成完成：{self._status(result)}")
        except Exception as exc:
            self.errors += 1
            self.failed_jobs.append((job, exc))
            if self.cancel_event.is_set():
                failure_state = "CANCELLED"
            elif isinstance(exc, AlignmentError):
                failure_state = "FAILED_ALIGNMENT"
            elif (
                isinstance(exc, HuginToolNotFound)
                or "hugin tool" in str(exc).casefold()
                or (
                    isinstance(exc, FileNotFoundError)
                    and any(name in str(exc).casefold() for name in ("align_image_stack", "enfuse", "hugin"))
                )
            ):
                # Missing Hugin is a distinct, actionable failure.  It must
                # never be presented as a successful ``DONE`` fusion.
                failure_state = "FAILED_HUGIN"
            elif "archive" in str(exc).casefold():
                failure_state = "FAILED_ARCHIVE"
            else:
                failure_state = "FAILED_FUSION"
            _set_group_state(self.repository, job.group, failure_state)
            self._update_job(
                job.group,
                stage="ARCHIVE" if failure_state == "FAILED_ARCHIVE" else "FUSION",
                status="FAILED",
                progress=0.0,
                error=exc,
            )
            self.logger.exception("Merge failed for group %s", _group_id(job.group))
            # Keep a result-shaped diagnostic for manifest consumers even when
            # the merger failed before it could construct MergeResult.  The
            # import is intentionally local to avoid a class-level cycle.
            failure = MergeResult(
                job.group,
                failure_state,
                error=exc,
                analysis=job.analysis,
                requested_backend=str(getattr(self.merger, "requested_backend", "quality")),
                actual_backend=str(getattr(self.merger, "requested_backend", "quality")),
                fallback_used=False,
                diagnostics=(str(exc),),
            )
            self._notify_result(failure)
            self.completed += 1
            self._emit(job, stage=PipelineStage.ERROR, message=f"合成失败：{exc}")

    def _discard_deferred(self) -> None:
        while self._deferred_jobs:
            job = self._deferred_jobs.pop(0)
            _set_group_state(self.repository, job.group, "CANCELLED")
            # A deferred item was removed from the queue but its unfinished
            # task count is intentionally held until it is either processed or
            # discarded, preserving BoundedJobQueue.join() semantics.
            self.merge_queue.task_done()

    def _process_with_memory_guard(self, job: AnalysisJob) -> bool:
        """Wait before fusion; archive-only jobs stay cancellable."""

        if self.cancel_event.is_set():
            return False
        # Archive-only work performs bounded file operations and, importantly,
        # must still run after a classification failure so source safety is
        # preserved even when the machine is under memory pressure.
        if self.memory_guard is not None and not job.archive_only:
            wait = getattr(self.memory_guard, "wait", None)
            if not callable(wait):
                wait = getattr(self.memory_guard, "wait_for_headroom", None)
            if not callable(wait):
                raise TypeError("memory_guard must expose wait()/wait_for_headroom()")
            group = job.group
            try:
                allowed = wait(
                    self.cancel_event,
                    stage=PipelineStage.FUSION,
                    current_file=str(extract_image_paths(group, job.analysis)[2] or ""),
                    current_group=_group_id(group),
                )
            except TypeError:
                allowed = wait(self.cancel_event)
            if not allowed:
                _set_group_state(self.repository, group, "CANCELLED")
                self._update_job(group, stage="FUSION", status="CANCELLED")
                return False
        self._update_job(
            job.group,
            stage="EXPORT" if job.archive_only else "FUSION",
            status="RUNNING",
            progress=0.45 if job.archive_only else 0.65,
        )
        self._process_job(job)
        return True

    def run(self) -> None:
        sentinel_seen = False
        while True:
            # Once analysis has completed, process the jobs drained while the
            # producer was running before accepting the queue sentinel.
            if self._deferred_jobs and (self.defer_until is None or self.defer_until.is_set()):
                job = self._deferred_jobs.pop(0)
                processed = False
                try:
                    processed = self._process_with_memory_guard(job)
                finally:
                    if processed:
                        self.finished += 1
                    self.merge_queue.task_done()
                continue
            if sentinel_seen:
                break
            if self.cancel_event.is_set() and self._deferred_jobs:
                self._discard_deferred()

            try:
                job = self.merge_queue.get(timeout=0.10)
            except QueueClosed:
                # Cancellation closes the queue without necessarily inserting
                # sentinels.  Any deferred work is no longer safe to process.
                if self._deferred_jobs:
                    self._discard_deferred()
                break
            except Exception:
                if self.cancel_event.is_set():
                    if self._deferred_jobs:
                        self._discard_deferred()
                    break
                continue
            if job is None:
                # A manually closed queue is also an unambiguous producer-done
                # signal.  This makes the worker safe outside Coordinator.  A
                # concurrent event.set() may race this branch, so always let
                # the next iteration's deferred-job check win.
                if self._deferred_jobs and not self.cancel_event.is_set():
                    if self.defer_until is not None:
                        self.defer_until.set()
                    sentinel_seen = True
                    continue
                if self._deferred_jobs:
                    self._discard_deferred()
                break
            if self.defer_until is not None and not self.defer_until.is_set() and not self.cancel_event.is_set():
                self._deferred_jobs.append(job)
                # Keep the queue unfinished-task count until deferred work is
                # eventually processed or explicitly discarded.
                continue
            processed = False
            try:
                processed = self._process_with_memory_guard(job)
            finally:
                if processed:
                    self.finished += 1
                self.merge_queue.task_done()
        self._emit(stage=PipelineStage.CANCELLED if self.cancel_event.is_set() else PipelineStage.FUSION, message="合成线程完成")


__all__ = ["MergeJob", "MergeResult", "MergeWorker", "StackMergeService", "extract_image_paths"]


# Backward-compatible name used by callers that model both worker jobs with a
# single type.  AnalysisJob already has the required group/analysis fields.
MergeJob = AnalysisJob
