"""Duck-typed durable job-state adapter for restartable pipeline work.

The storage layer owns the SQLite schema.  This module only speaks its public
``JobRecord``/``create_job``/``update_job``/``list_jobs`` API and degrades to a
no-op for lightweight repositories used by headless callers.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterable

from ..storage.models import JobRecord


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


def _group_id(group: Any) -> int | None:
    if group is None:
        return None
    if isinstance(group, dict):
        value = group.get("group_id", group.get("id"))
    else:
        value = getattr(group, "group_id", getattr(group, "id", None))
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


class JobStateStore:
    """Keep one latest durable job row per logical group."""

    UNFINISHED = frozenset({"PENDING", "RUNNING", "FAILED"})

    def __init__(self, repository: Any | None, *, logger: logging.Logger | None = None) -> None:
        self.repository = repository
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._jobs: dict[int, JobRecord] = {}
        self.refresh()

    @property
    def supported(self) -> bool:
        return self.repository is not None and callable(getattr(self.repository, "list_jobs", None))

    @staticmethod
    def _status(job: Any) -> str:
        return str(_value(getattr(job, "status", "")) or "").upper()

    @staticmethod
    def _stage(job: Any) -> str:
        return str(_value(getattr(job, "stage", "")) or "").upper()

    def refresh(self) -> list[JobRecord]:
        """Reload latest rows, tolerating repositories without job support."""

        if not self.supported:
            return []
        try:
            rows = list(self.repository.list_jobs())
        except Exception:
            self.logger.warning("Unable to list durable pipeline jobs", exc_info=True)
            return []
        latest: dict[int, JobRecord] = {}
        for row in rows:
            gid = _group_id(row)
            if gid is None:
                gid_value = getattr(row, "group_id", None)
                try:
                    gid = int(gid_value) if gid_value is not None else None
                except (TypeError, ValueError):
                    gid = None
            if gid is None:
                continue
            # Database.list_jobs is ordered by id.  Keep the greatest id when
            # a minimal repository returns rows in another order.
            previous = latest.get(gid)
            previous_id = int(getattr(previous, "id", -1) or -1) if previous is not None else -1
            row_id = int(getattr(row, "id", -1) or -1)
            if previous is None or row_id >= previous_id:
                latest[gid] = row
        with self._lock:
            self._jobs = latest
            return list(latest.values())

    def job_for(self, group: Any) -> JobRecord | None:
        gid = _group_id(group)
        if gid is None:
            return None
        with self._lock:
            return self._jobs.get(gid)

    def ensure(self, group: Any, *, stage: str = "CLASSIFICATION") -> JobRecord | None:
        """Return or create a job row without duplicating an existing one."""

        gid = _group_id(group)
        if gid is None or not self.supported:
            return None
        with self._lock:
            current = self._jobs.get(gid)
            if current is not None:
                return current
            row = JobRecord(group_id=gid, stage=str(_value(stage)), status="PENDING")
            creator = next(
                (getattr(self.repository, name, None) for name in ("create_job", "add_job", "save_job")
                 if callable(getattr(self.repository, name, None))),
                None,
            )
            if creator is None:
                return None
            try:
                creator(row)
            except Exception:
                self.logger.warning("Unable to create pipeline job for group %s", gid, exc_info=True)
                return None
            self._jobs[gid] = row
            return row

    def update(
        self,
        group: Any,
        *,
        stage: str,
        status: str,
        progress: float | None = None,
        error: str | None = None,
    ) -> JobRecord | None:
        """Persist one state transition, retaining an existing error by default."""

        row = self.ensure(group, stage=stage)
        if row is None:
            return None
        status_value = str(_value(status)).upper()
        stage_value = str(_value(stage)).upper()
        with self._lock:
            if self._status(row) == "DONE" and status_value != "DONE":
                return row
            row.stage = stage_value
            row.status = status_value
            if progress is not None:
                row.progress = max(0.0, min(1.0, float(progress)))
            if error is not None:
                row.error = str(error)
            updater = getattr(self.repository, "update_job", None)
            if not callable(updater):
                updater = getattr(self.repository, "save_job", None)
            if not callable(updater):
                return row
            try:
                updater(row)
            except Exception:
                self.logger.warning("Unable to update pipeline job %s", getattr(row, "id", None), exc_info=True)
            return row

    def unfinished(self) -> list[JobRecord]:
        """Return latest PENDING/RUNNING/FAILED rows in stable group order."""

        rows = self.refresh()
        return [row for row in rows if self._status(row) in self.UNFINISHED]


__all__ = ["JobStateStore"]

