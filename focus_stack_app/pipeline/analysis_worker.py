"""Analysis producer worker for the bounded Focus Stack pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import logging
from pathlib import Path
import threading
from typing import Any, Callable, Iterable

from .events import PipelineEvent, PipelineStage
from .job_queue import BoundedJobQueue, QueueClosed


@dataclass
class AnalysisJob:
    group: Any
    analysis: Any = None
    error: Exception | None = None
    # Historical name retained for compatibility.  It now means the group
    # must bypass Hugin/Enfuse and remain in the source directory.  Archiving
    # is allowed only after a real composite has been created.
    archive_only: bool = False

    @property
    def group_id(self) -> int | str | None:
        return _group_id(self.group)


AnalysisCallable = Callable[..., Any]
EventCallback = Callable[[PipelineEvent], Any]


def _group_id(group: Any) -> int | str | None:
    if isinstance(group, dict):
        return group.get("group_id", group.get("id"))
    return getattr(group, "group_id", getattr(group, "id", None))


def _group_first_file(group: Any, result: Any = None) -> str:
    for item in (result, group):
        if item is None:
            continue
        if isinstance(item, dict):
            value = item.get("current_file", item.get("first_original_image", item.get("first_image", "")))
        else:
            value = getattr(item, "current_file", getattr(item, "first_original_image", getattr(item, "first_image", "")))
        if value:
            value = getattr(value, "path", getattr(value, "current_path", value))
            return str(value)
    return ""


def _get(value: Any, *names: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
    else:
        for name in names:
            try:
                found = getattr(value, name)
            except Exception:
                continue
            if found is not None:
                return found
    return default


def _image_count(group: Any) -> int:
    value = _get(group, "image_count", default=None)
    if value is not None:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            pass
    for name in ("items", "images", "all_images", "image_paths"):
        value = _get(group, name, default=None)
        if value is not None and not isinstance(value, (str, bytes)):
            try:
                return len(value)
            except TypeError:
                pass
    return 0


def _selected_values(result: Any) -> tuple[Any, bool]:
    """Return explicit selected evidence and whether it was supplied.

    An explicit empty selection is meaningful (a repeated single/no-merge
    scene) and must not be replaced by all group images downstream.
    """

    if result is None:
        return None, False
    for name in ("selected_paths", "selected_images", "selected_items", "selected_indices"):
        value = _get(result, name, default=None)
        if value is not None:
            return value, True
    # CoverageSelection exposes ``selected`` as a property alias.
    value = _get(result, "selected", default=None)
    if value is not None and not isinstance(value, bool):
        return value, True
    return None, False


def _selection_count(result: Any, group: Any) -> int:
    values, supplied = _selected_values(result)
    if supplied:
        if isinstance(values, (str, bytes)):
            return 1
        try:
            return max(0, len(values))
        except TypeError:
            return 0
    explicit = _get(result, "selected_count", default=None)
    if explicit is not None:
        try:
            return max(0, int(explicit))
        except (TypeError, ValueError):
            pass
    return _image_count(group)


def _needs_merge(result: Any, group: Any) -> bool:
    """Infer whether a successfully analysed group should enter Hugin."""

    explicit = _get(result, "needs_merge", "needs_focus_stack", "requires_merge", default=None)
    if explicit is not None:
        return bool(explicit)
    status = str(_get(result, "status", default="")).casefold()
    if (
        status in {"single", "no_merge", "skipped_single", "repeated_single", "repeated single", "classified"}
        or status.startswith(("single_", "no_merge_", "repeated_"))
    ):
        return False
    if status in {"selected", "merge", "needs_merge", "needs merge", "focus_stack"}:
        return True
    count = _selection_count(result, group)
    if count > 1:
        return True
    if count == 0:
        # An opaque test double or an early caller may carry no image-count
        # metadata at all.  Preserve the historical behaviour for that
        # boundary and let the merger validate its paths; a real GroupRecord
        # or SceneGroup always exposes image_count/items and can be classified
        # as archive-only deterministically.
        has_count = _get(group, "image_count", default=None) is not None
        has_images = any(_get(group, name, default=None) is not None for name in ("items", "images", "all_images", "image_paths"))
        return not (has_count or has_images)
    return False


def _analysis_group_status(result: Any, group: Any) -> str:
    existing = str(_get(group, "status", default="") or "")
    # Preserve an explicit durable state set by a resume/recovery caller.
    if existing in {"ARCHIVED", "DONE", "FAILED_ALIGNMENT", "FAILED_FUSION", "FAILED_HUGIN", "CANCELLED"}:
        return existing
    if _needs_merge(result, group):
        return "SELECTED"
    # CLASSIFIED means the scene was analysed successfully but does not need
    # Hugin (single image, repeated single scene, or no material focus gain).
    return "CLASSIFIED"


def _invoke(callback: AnalysisCallable, value: Any, *, cancel_event: threading.Event, progress_callback: EventCallback | None) -> Any:
    fn = callback
    if not callable(fn):
        fn = getattr(callback, "analyze_group", getattr(callback, "analyze", None))
    if not callable(fn):
        raise TypeError("analyzer must be callable or expose analyze_group()/analyze()")
    context = {
        "cancel_event": cancel_event,
        "progress_callback": progress_callback,
        "progress": progress_callback,
    }
    try:
        signature = inspect.signature(fn)
        parameters = signature.parameters
        accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        kwargs = context if accepts_kwargs else {key: val for key, val in context.items() if key in parameters}
    except (TypeError, ValueError):
        kwargs = {}
    return fn(value, **kwargs)


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
        except (TypeError, ValueError):
            try:
                setter(gid, status)
                return
            except Exception:
                pass
    updater = getattr(repository, "update_group", None)
    if callable(updater):
        try:
            if isinstance(group, dict):
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


class AnalysisWorker(threading.Thread):
    """Run group analysis serially and push completed jobs to merge queue."""

    def __init__(
        self,
        groups: Iterable[Any],
        analyzer: AnalysisCallable,
        merge_queue: BoundedJobQueue[AnalysisJob],
        *,
        cancel_event: threading.Event | None = None,
        event_callback: EventCallback | None = None,
        repository: Any | None = None,
        total: int | None = None,
        logger: logging.Logger | None = None,
        close_queue: bool = False,
        memory_guard: Any | None = None,
        job_state: Any | None = None,
    ):
        super().__init__(name="focus-stack-analysis", daemon=True)
        self.groups = list(groups)
        self.analyzer = analyzer
        self.merge_queue = merge_queue
        self.cancel_event = cancel_event or threading.Event()
        self.event_callback = event_callback
        self.repository = repository
        self.total = len(self.groups) if total is None else int(total)
        self.logger = logger or logging.getLogger(__name__)
        self.close_queue = close_queue
        self.memory_guard = memory_guard
        self.job_state = job_state
        self.completed = 0
        self.discovered = 0
        self.errors = 0
        self.failed_jobs: list[AnalysisJob] = []

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
            self.logger.debug("Unable to persist analysis job state", exc_info=True)

    def _emit(self, *, stage: PipelineStage = PipelineStage.ANALYSIS, group: Any = None, result: Any = None, message: str = "") -> None:
        if self.event_callback is None:
            return
        event = PipelineEvent(
            stage=stage,
            current_file=_group_first_file(group, result),
            current_group=_group_id(group) if group is not None else None,
            completed=self.completed,
            total=self.total,
            groups_found=self.discovered,
            errors=self.errors,
            message=message,
            analysis_completed=self.completed,
            analysis_total=self.total,
        )
        try:
            self.event_callback(event)
        except Exception:
            self.logger.debug("Progress callback failed", exc_info=True)

    def run(self) -> None:
        try:
            for group in self.groups:
                if self.cancel_event.is_set():
                    self._update_job(group, stage="FOCUS_ANALYSIS", status="CANCELLED")
                    break
                gid = _group_id(group)
                # A group is discovered as soon as the producer sees it,
                # independent of whether analysis later succeeds.  This keeps
                # UI ``groups_found`` and stage progress truthful.
                self.discovered += 1
                _set_group_state(self.repository, group, "CLASSIFYING")
                self._update_job(group, stage="FOCUS_ANALYSIS", status="RUNNING", progress=0.0)
                self._emit(group=group, message="分析中")
                try:
                    if self.memory_guard is not None:
                        wait = getattr(self.memory_guard, "wait", None)
                        if not callable(wait):
                            wait = getattr(self.memory_guard, "wait_for_headroom", None)
                        if not callable(wait):
                            raise TypeError("memory_guard must expose wait()/wait_for_headroom()")
                        try:
                            allowed = wait(
                                self.cancel_event,
                                stage=PipelineStage.ANALYSIS,
                                current_file=_group_first_file(group),
                                current_group=gid,
                            )
                        except TypeError:
                            # Keep simple test doubles and early adapters
                            # compatible with the one-argument protocol.
                            allowed = wait(self.cancel_event)
                        if not allowed:
                            _set_group_state(self.repository, group, "CANCELLED")
                            self._update_job(group, stage="FOCUS_ANALYSIS", status="CANCELLED")
                            break
                    result = _invoke(self.analyzer, group, cancel_event=self.cancel_event, progress_callback=self.event_callback)
                    if self.cancel_event.is_set():
                        _set_group_state(self.repository, group, "CANCELLED")
                        self._update_job(group, stage="FOCUS_ANALYSIS", status="CANCELLED")
                        break
                    job = AnalysisJob(group=group, analysis=result, archive_only=not _needs_merge(result, group))
                    _set_group_state(self.repository, group, _analysis_group_status(result, group))
                    # The analysis payload is now durable/in-memory.  A merge
                    # candidate proceeds to Hugin; a no-merge decision is
                    # finalized without touching its source files.
                    self._update_job(group, stage="FUSION", status="RUNNING", progress=0.35)
                    self.merge_queue.put(job)
                    self.completed += 1
                    self._emit(
                        group=group,
                        result=result,
                        message="分析完成，已加入合成队列" if not job.archive_only else "分析完成，无需合成，原图保持不动",
                    )
                except QueueClosed:
                    if self.cancel_event.is_set():
                        self._update_job(group, stage="FOCUS_ANALYSIS", status="CANCELLED")
                        break
                    raise
                except Exception as exc:
                    self.errors += 1
                    # Analysis failures bypass Hugin and leave every original
                    # in place.  The consumer still emits a manifest result.
                    job = AnalysisJob(group=group, error=exc, archive_only=True)
                    self.failed_jobs.append(job)
                    _set_group_state(self.repository, group, "FAILED_CLASSIFICATION")
                    self._update_job(group, stage="FUSION", status="RUNNING", progress=0.35, error=exc)
                    self.logger.exception("Analysis failed for group %s", gid)
                    queued = False
                    try:
                        self.merge_queue.put(job)
                        queued = True
                    except QueueClosed:
                        if not self.cancel_event.is_set():
                            raise
                    if not queued:
                        break
                    self.completed += 1
                    self._emit(group=group, message=f"分析失败，原图未移动：{exc}")
        finally:
            if self.close_queue:
                self.merge_queue.close(wait=not self.cancel_event.is_set())
            self._emit(
                stage=PipelineStage.CANCELLED if self.cancel_event.is_set() else PipelineStage.ANALYSIS,
                message="分析已停止" if self.cancel_event.is_set() else "分析线程完成",
            )


__all__ = ["AnalysisJob", "AnalysisWorker"]

