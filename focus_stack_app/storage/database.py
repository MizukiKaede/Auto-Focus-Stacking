Warning: truncated output (original token count: 18267)
Total output lines: 1659

"""SQLite persistence for projects, metadata and restartable jobs.

The repository is intentionally small and explicit rather than hiding state
behind an ORM.  Every write is transactional, foreign keys are enabled, and
the schema contains the state needed to reconcile a move after a process
crash.  SQLite stores paths and scalar metadata only; decoded image pixels
never enter this module.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from threading import RLock
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence
import uuid

from .models import AnalysisRecord, ArchiveOperationRecord, GroupRecord, ImageRecord, JobRecord


class DatabaseError(RuntimeError):
    """Raised for repository errors with a stable application-level type."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path_string(value: str | Path) -> str:
    # ``strict=False`` is important for a source path which disappeared during
    # a crash.  normpath also gives deterministic values on Windows.
    return os.path.normpath(os.path.abspath(os.fspath(value)))


def _json_value(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        # NumPy matrices and scalar arrays are common at the algorithm boundary
        # but are optional here.  Convert them without importing NumPy.
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            return json.dumps(tolist(), ensure_ascii=False, separators=(",", ":"))
        return json.dumps(str(value), ensure_ascii=False)


def _from_json(value: Any) -> Any:
    if value in (None, ""):
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        # Preserve old/custom data instead of making a project unreadable.
        return value


class Database:
    """Thread-safe SQLite repository.

    ``check_same_thread=False`` plus an RLock allows a scanner worker and a UI
    refresh worker to share a repository safely.  Callers that need multiple
    related writes should use :meth:`transaction` so a crash cannot leave a
    half-updated group.
    """

    SCHEMA_VERSION = 4

    def __init__(self, path: str | Path, *, timeout: float = 30.0, initialize: bool = True) -> None:
        self.path = Path(path)
        is_memory = str(path) == ":memory:"
        if not is_memory:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._connection = sqlite3.connect(
                ":memory:" if is_memory else str(self.path),
                timeout=timeout,
                isolation_level=None,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._lock = RLock()
            self._closed = False
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            if not is_memory:
                self._connection.execute("PRAGMA journal_mode = WAL")
                self._connection.execute("PRAGMA synchronous = NORMAL")
            if initialize:
                self.create_schema()
        except sqlite3.Error as exc:
            raise DatabaseError(f"unable to open database {path!s}: {exc}") from exc

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose the connection for read-only integrations and migrations."""

        return self._connection

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        """Run a group of operations atomically.

        ``BEGIN IMMEDIATE`` obtains the write lock before yielding, avoiding a
        partial state if two worker threads race to update the same group.
        """

        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
                yield self._connection
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def create_schema(self) -> None:
        with self._lock:
            try:
                self._connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        original_path TEXT NOT NULL UNIQUE,
                        current_path TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        stem TEXT NOT NULL,
                        extension TEXT NOT NULL,
                        file_size INTEGER NOT NULL DEFAULT 0,
                        mtime REAL NOT NULL DEFAULT 0,
                        capture_time TEXT,
                        width INTEGER,
                        height INTEGER,
                        camera TEXT,
                        lens TEXT,
                        sequence_index INTEGER,
                        group_id INTEGER,
                        status TEXT NOT NULL DEFAULT 'DISCOVERED',
                        archive_mode TEXT,
                        pending_path TEXT,
                        archive_operation_id TEXT,
                        archive_error TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE SET NULL
                    );

                    CREATE TABLE IF NOT EXISTS groups (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        first_image_id INTEGER,
                        start_index INTEGER,
                        end_index INTEGER,
                        image_count INTEGER NOT NULL DEFAULT 0,
                        selected_count INTEGER NOT NULL DEFAULT 0,
                        confidence REAL,
                        coverage REAL,
                        status TEXT NOT NULL DEFAULT 'DISCOVERED',
                        output_path TEXT,
                        preview_reference TEXT,
                        alignment_order TEXT,
                        alignment_order_confidence REAL,
                        alignment_order_fallback_used INTEGER NOT NULL DEFAULT 0,
                        pairwise_analysis_summary TEXT,
                        requested_backend TEXT,
                        actual_backend TEXT,
                        alignment_level INTEGER,
                        alignment_status TEXT,
                        crop_ratio REAL,
                        diagnostics TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(first_image_id) REFERENCES images(id) ON DELETE SET NULL
                    );

                    CREATE TABLE IF NOT EXISTS analysis (
                        image_id INTEGER PRIMARY KEY,
                        scene_hash TEXT,
                        scene_score REAL,
                        sharpness_score REAL,
                        focus_map_path TEXT,
                        transform TEXT,
                        transform_json TEXT,
                        selected INTEGER NOT NULL DEFAULT 0 CHECK(selected IN (0, 1)),
                        selection_reason TEXT,
                        coverage_gain REAL,
                        quality_score REAL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS jobs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        group_id INTEGER,
                        stage TEXT NOT NULL,
                        progress REAL NOT NULL DEFAULT 0 CHECK(progress >= 0 AND progress <= 1),
                        status TEXT NOT NULL DEFAULT 'PENDING',
                        error TEXT,
                        started_at TEXT,
                        finished_at TEXT,
                        attempt INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS archive_operations (
                        operation_id TEXT PRIMARY KEY,
                        image_id INTEGER,
                        source_path TEXT NOT NULL,
                        destination_path TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        phase TEXT NOT NULL DEFAULT 'pending',
                        status TEXT NOT NULL DEFAULT 'pending',
                        error TEXT,
                        bytes_transferred INTEGER NOT NULL DEFAULT 0,
                        source_size INTEGER,
                        source_mtime REAL,
                        started_at REAL,
                        finished_at REAL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE SET NULL
                    );

                    CREATE TABLE IF NOT EXISTS schema_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS cached_plans (
                        kind TEXT NOT NULL,
                        cache_key TEXT NOT NULL,
                        algorithm_version TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(kind, cache_key)
                    );

                    CREATE INDEX IF NOT EXISTS idx_images_sequence ON images(sequence_index);
                    CREATE INDEX IF NOT EXISTS idx_images_group ON images(group_id, sequence_index);
                    CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
                    CREATE INDEX IF NOT EXISTS idx_groups_status ON groups(status);
                    CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, stage);
                    CREATE INDEX IF NOT EXISTS idx_archive_operations_image ON archive_operations(image_id, phase);
                    CREATE INDEX IF NOT EXISTS idx_archive_operations_phase ON archive_operations(phase, updated_at);
                    CREATE INDEX IF NOT EXISTS idx_cached_plans_kind_updated ON cached_plans(kind, updated_at);
                    INSERT INTO schema_meta(key, value) VALUES ('schema_version', '1')
                        ON CONFLICT(key) DO UPDATE SET value = excluded.value;
                    """
                )
                # Columns/tables added in schema v2 are installed explicitly so
                # projects created by the v1 foundation layer remain readable.
                # SQLite has no ``ADD COLUMN IF NOT EXISTS``; inspect first and
                # issue only the missing ALTER statements.
                image_columns = {
                    row["name"]
                    for row in self._connection.execute("PRAGMA table_info(images)").fetchall()
                }
                for name, definition in (
                    ("pending_path", "TEXT"),
                    ("archive_operation_id", "TEXT"),
                    ("archive_error", "TEXT"),
                ):
                    if name not in image_columns:
                        self._connection.execute(f"ALTER TABLE images ADD COLUMN {name} {definition}")
                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS archive_operations (
                        operation_id TEXT PRIMARY KEY,
                        image_id INTEGER,
                        source_path TEXT NOT NULL,
                        destination_path TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        phase TEXT NOT NULL DEFAULT 'pending',
                        status TEXT NOT NULL DEFAULT 'pending',
                        error TEXT,
                        bytes_transferred INTEGER NOT NULL DEFAULT 0,
                        source_size INTEGER,
                        source_mtime REAL,
                        started_at REAL,
                        finished_at REAL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE SET NULL
                    )
                    """
                )
                self._connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_archive_operations_image ON archive_operations(image_id, phase)"
                )
                self._connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_archive_operations_phase ON archive_operations(phase, updated_at)"
                )
                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cached_plans (
                        kind TEXT NOT NULL,
                        cache_key TEXT NOT NULL,
                        algorithm_version TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(kind, cache_key)
                    )
                    """
                )
                self._connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cached_plans_kind_updated ON cached_plans(kind, updated_at)"
                )
                # ``transform_json`` was the name used by an early prototype;
                # retain both names so existing project files and integrations
                # using the plan's public ``transform`` field remain readable.
                columns = {
                    row["name"]
                    for row in self._connection.execute("PRAGMA table_info(analysis)").fetchall()
                }
                if "transform" not in columns:
                    self._connection.execute("ALTER TABLE analysis ADD COLUMN transform TEXT")
                if "transform_json" not in columns:
                    self._connection.execute("ALTER TABLE analysis ADD COLUMN transform_json TEXT")
                group_columns = {
                    row["name"] for row in self._connection.execute("PRAGMA table_info(groups)").fetchall()
                }
                for name, definition in (
                    ("preview_reference", "TEXT"),
                    ("alignment_order", "TEXT"),
                    ("alignment_order_confidence", "REAL"),
                    ("alignment_order_fallback_used", "INTEGER NOT NULL DEFAULT 0"),
                    ("pairwise_analysis_summary", "TEXT"),
                    ("requested_backend", "TEXT"),
                    ("actual_backend", "TEXT"),
                    ("alignment_level", "INTEGER"),
                    ("alignment_status", "TEXT"),
                    ("crop_ratio", "REAL"),
                    ("diagnostics", "TEXT"),
                ):
                    if name not in group_columns:
                        self._connection.execute(f"ALTER TABLE groups ADD COLUMN {name} {definition}")
                self._connection.execute(
                    "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(self.SCHEMA_VERSION),),
                )
                self._connection.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
            except sqlite3.Error as exc:
                raise DatabaseError(f"unable to create schema: {exc}") from exc

    # Names kept for small integrations that treat schema creation as an
    # explicit setup step.  ``Database(...)`` already initializes by default.
    initialize = create_schema
    create_tables = create_schema

    # ---- low-level helpers -------------------------------------------------

    def _fetchone(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._connection.execute(sql, params).fetchone()

    def _fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._connection.execute(sql, params).fetchall())

    def _run(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            try:
                return self._connection.execute(sql, params)
            except sqlite3.Error as exc:
                raise DatabaseError(str(exc)) from exc

    # ---- lightweight grouping/selection plans -----------------------------

    def get_cached_plan(
        self, kind: str, cache_key: str, *, algorithm_version: str | None = None,
    ) -> Any | None:
        """Return one valid JSON plan; corrupt/version-mismatched rows miss."""

        row = self._fetchone(
            "SELECT algorithm_version, payload_json FROM cached_plans WHERE kind = ? AND cache_key = ?",
            (str(kind), str(cache_key)),
        )
        if row is None or (
            algorithm_version is not None and row["algorithm_version"] != str(algorithm_version)
        ):
            return None
        try:
            return json.loads(row["payload_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            # A partial/manual edit must never make a project unreadable.
            self.invalidate_cached_plans(kind=str(kind), cache_key=str(cache_key))
            return None

    def upsert_cached_plan(
        self, kind: str, cache_key: str, algorithm_version: str, payload: Any,
    ) -> None:
        """Atomically publish one JSON-only plan."""

        try:
            encoded = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise DatabaseError(f"cached plan is not valid JSON: {exc}") from exc
        now = _now()
        with self._lock:
            try:
                self._connection.execute(
                    """
                    INSERT INTO cached_plans (
                        kind, cache_key, algorithm_version, payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(kind, cache_key) DO UPDATE SET
                        algorithm_version=excluded.algorithm_version,
                        payload_json=excluded.payload_json,
                        updated_at=excluded.updated_at
                    """,
                    (str(kind), str(cache_key), str(algorithm_version), encoded, now, now),
                )
            except sqlite3.Error as exc:
                raise DatabaseError(str(exc)) from exc

    def invalidate_cached_plans(
        self, *, kind: str | None = None, cache_key: str | None = None,
    ) -> int:
        """Delete cached plans by kind/key and return the affected row count."""

        clauses: list[str] = []
        params: list[str] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(str(kind))
        if cache_key is not None:
            clauses.append("cache_key = ?")
            params.append(str(cache_key))
        sql = "DELETE FROM cached_plans"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        cursor = self._run(sql, params)
        return max(0, int(cursor.rowcount))

    # ---- images ------------------------------------------------------------

    def insert_image(self, image: ImageRecord) -> int:
        """Insert metadata or refresh an existing source-path row.

        Re-scanning an already archived project updates stat/EXIF fields but
        deliberately preserves ``current_path``, ``group_id`` and status.
        """

        path = _path_string(image.original_path)
        now = image.updated_at or _now()
        created = image.created_at or now
        current = _path_string(image.current_path or path)
        with self._lock:
            try:
                self._connection.execute(
                    """
                    INSERT INTO images (
                        original_path, current_path, filename, stem, extension,
                        file_size, mtime, capture_time, width, height, camera,
                        lens, sequence_index, group_id, status, archive_mode,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(original_path) DO UPDATE SET
                        filename=excluded.filename,
                        stem=excluded.stem,
                        extension=excluded.extension,
                        file_size=excluded.file_size,
                        mtime=excluded.mtime,
                        capture_time=excluded.capture_time,
                        width=excluded.width,
                        height=excluded.height,
                        camera=excluded.camera,
                        lens=excluded.lens,
                        sequence_index=excluded.sequence_index,
                        updated_at=excluded.updated_at
                    """,
                    (
                        path,
                        current,
                        image.filename or Path(path).name,
                        image.stem or Path(path).stem,
                        image.extension or Path(path).suffix.lower(),
                        int(image.file_size),
                        float(image.mtime),
                        image.capture_time,
                        image.width,
                        image.height,
                        image.camera,
                        image.lens,
                        image.sequence_index,
                        image.group_id,
                        image.status,
                        image.archive_mode,
                        created,
                        now,
                    ),
                )
                row = self._connection.execute(
                    "SELECT * FROM images WHERE original_path = ?", (path,)
                ).fetchone()
                assert row is not None
                image.id = int(row[0])
                image.original_path = path
                # On a re-scan an existing row may already be archived or have
                # an in-flight destination.  Reflect the durable row back onto
                # the caller's record instead of handing it the scanner's
                # source path and accidentally attempting a second move.
                image.current_path = row["current_path"]
                image.status = row["status"]
                image.archive_mode = row["archive_mode"]
                image.pending_path = row["pending_path"] if "pending_path" in row.keys() else None
                image.archive_operation_id = row["archive_operation_id"] if "archive_operation_id" in row.keys() else None
       …6267 tokens truncated…_size is None or source_mtime is None:
            try:
                source_stat = Path(source_path).stat()
            except OSError:
                source_stat = None
            if source_stat is not None:
                if source_size is None:
                    source_size = int(source_stat.st_size)
                if source_mtime is None:
                    source_mtime = float(source_stat.st_mtime)
        started_at = self._optional_float(data.get("started_at"))
        finished_at = self._optional_float(data.get("finished_at"))
        error = data.get("error")
        error = None if error in (None, "") else str(error)
        now = _now()

        with self.transaction():
            existing = self._connection.execute(
                "SELECT * FROM archive_operations WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if existing is not None:
                if source_size is None:
                    source_size = existing["source_size"]
                if source_mtime is None:
                    source_mtime = existing["source_mtime"]
                if started_at is None:
                    started_at = existing["started_at"]
                if supplied_image_id is None:
                    supplied_image_id = existing["image_id"]
                created_at = existing["created_at"] or now
            else:
                created_at = now

            image_id = self._resolve_archive_image_id(
                supplied_image_id, source_path, destination_path
            )
            self._connection.execute(
                """
                INSERT INTO archive_operations (
                    operation_id, image_id, source_path, destination_path, mode,
                    phase, status, error, bytes_transferred, source_size,
                    source_mtime, started_at, finished_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(operation_id) DO UPDATE SET
                    image_id=COALESCE(excluded.image_id, archive_operations.image_id),
                    source_path=excluded.source_path,
                    destination_path=excluded.destination_path,
                    mode=excluded.mode,
                    phase=excluded.phase,
                    status=excluded.status,
                    error=excluded.error,
                    bytes_transferred=excluded.bytes_transferred,
                    source_size=COALESCE(excluded.source_size, archive_operations.source_size),
                    source_mtime=COALESCE(excluded.source_mtime, archive_operations.source_mtime),
                    started_at=COALESCE(excluded.started_at, archive_operations.started_at),
                    finished_at=excluded.finished_at,
                    updated_at=excluded.updated_at
                """,
                (
                    operation_id,
                    image_id,
                    source_path,
                    destination_path,
                    mode,
                    phase,
                    status,
                    error,
                    bytes_transferred,
                    source_size,
                    source_mtime,
                    started_at,
                    finished_at,
                    created_at,
                    now,
                ),
            )

            if image_id is not None:
                if phase in {"pending", "archiving", "started"}:
                    self._connection.execute(
                        """
                        UPDATE images SET pending_path=?, archive_operation_id=?,
                            archive_mode=?, status='ARCHIVING', archive_error=NULL,
                            updated_at=? WHERE id=?
                        """,
                        (destination_path, operation_id, mode, now, image_id),
                    )
                elif phase in {"archived", "already_archived", "complete", "completed"}:
                    self._connection.execute(
                        """
                        UPDATE images SET current_path=?, pending_path=NULL,
                            archive_operation_id=?, archive_mode=?, status='ARCHIVED',
                            archive_error=NULL, updated_at=? WHERE id=?
                        """,
                        (destination_path, operation_id, mode, now, image_id),
                    )
                elif phase in {"failed", "conflict", "error"}:
                    self._connection.execute(
                        """
                        UPDATE images SET pending_path=?, archive_operation_id=?,
                            archive_mode=?, status='FAILED', archive_error=?,
                            updated_at=? WHERE id=?
                        """,
                        (destination_path, operation_id, mode, error or status, now, image_id),
                    )

        return self.get_archive_operation(operation_id) or ArchiveOperationRecord(
            operation_id=operation_id,
            source_path=source_path,
            destination_path=destination_path,
            mode=mode,
            image_id=image_id,
            phase=phase,
            status=status,
            error=error,
            bytes_transferred=bytes_transferred,
            source_size=source_size,
            source_mtime=source_mtime,
            started_at=started_at,
            finished_at=finished_at,
            created_at=created_at,
            updated_at=now,
        )

    upsert_archive_operation = record_archive_operation
    save_archive_operation = record_archive_operation
    record_file_operation = record_archive_operation
    upsert_file_operation = record_archive_operation

    def get_archive_operation(self, operation_id: str) -> ArchiveOperationRecord | None:
        row = self._fetchone(
            "SELECT * FROM archive_operations WHERE operation_id=?",
            (str(operation_id),),
        )
        return ArchiveOperationRecord.from_row(row) if row is not None else None

    def list_archive_operations(
        self,
        *,
        image_id: int | None = None,
        phase: str | None = None,
        status: str | None = None,
    ) -> list[ArchiveOperationRecord]:
        conditions: list[str] = []
        params: list[Any] = []
        if image_id is not None:
            conditions.append("image_id=?")
            params.append(int(image_id))
        if phase is not None:
            conditions.append("phase=?")
            params.append(str(phase).casefold())
        if status is not None:
            conditions.append("status=?")
            params.append(str(status).casefold())
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._fetchall(
            f"SELECT * FROM archive_operations{where} ORDER BY updated_at, operation_id",
            params,
        )
        return [ArchiveOperationRecord.from_row(row) for row in rows]

    list_archive_records = list_archive_operations

    def _set_archive_operation_state(
        self,
        operation: ArchiveOperationRecord,
        *,
        phase: str,
        status: str | None = None,
        error: str | None = None,
        bytes_transferred: int | None = None,
        finished_at: float | None = None,
    ) -> None:
        self._run(
            """
            UPDATE archive_operations SET phase=?, status=?, error=?,
                bytes_transferred=COALESCE(?, bytes_transferred),
                finished_at=COALESCE(?, finished_at), updated_at=?
            WHERE operation_id=?
            """,
            (
                str(phase).casefold(),
                str(status or phase).casefold(),
                error,
                bytes_transferred,
                finished_at,
                _now(),
                operation.operation_id,
            ),
        )

    def recover_incomplete_jobs(self) -> list[JobRecord]:
        """Reset RUNNING jobs to PENDING after an unclean shutdown.

        The old error is retained as an audit hint and a new attempt can
        safely claim the job.  No filesystem operation is performed here.
        """

        with self.transaction():
            rows = self._connection.execute(
                "SELECT * FROM jobs WHERE status = 'RUNNING' ORDER BY id"
            ).fetchall()
            for row in rows:
                note = row["error"] or ""
                marker = "recovered after interrupted run"
                error = f"{note}; {marker}" if note else marker
                self._connection.execute(
                    """
                    UPDATE jobs SET status='PENDING', error=?, started_at=NULL,
                        finished_at=NULL, attempt=attempt+1, updated_at=? WHERE id=?
                    """,
                    (error, _now(), row["id"]),
                )
        return self.list_jobs(status="PENDING")

    def recover_archived_images(self, destination_root: str | Path | None = None) -> list[ImageRecord]:
        """Reconcile archive operations without touching the filesystem.

        New operations have a journal row and an ``images.pending_path`` that
        are committed before the file layer starts.  Recovery can consequently
        distinguish a completed move from an unrelated same-name file.  A
        destination is accepted when it is a regular file and its size matches
        the size captured before the operation (or the image metadata for a
        legacy row).  If both sides of a move still exist, recovery records a
        conflict and never deletes either side.

        Old v1 databases have no archive journal.  Their existing
        ``current_path``/``destination_root`` heuristic is retained as a
        compatibility fallback, with the same regular-file and size checks when
        a size is available.
        """

        candidates = self.list_images()
        by_id = {int(image.id): image for image in candidates if image.id is not None}
        recovered: list[ImageRecord] = []
        touched: set[int] = set()
        root = Path(destination_root) if destination_root is not None else None

        def append_updated(image_id: int | None) -> None:
            if image_id is None or image_id in touched:
                return
            updated = self.get_image(image_id)
            if updated is not None:
                recovered.append(updated)
                touched.add(image_id)

        def expected_size(image: ImageRecord | None, operation: ArchiveOperationRecord | None) -> int | None:
            if operation is not None and operation.source_size is not None:
                return int(operation.source_size)
            if image is not None and int(image.file_size) > 0:
                return int(image.file_size)
            if operation is not None and int(operation.bytes_transferred) > 0:
                return int(operation.bytes_transferred)
            return None

        def is_regular(path: Path) -> bool:
            try:
                return path.is_file()
            except OSError:
                return False

        def destination_matches(path: Path, expected: int | None) -> bool:
            if not is_regular(path):
                return False
            if expected is None:
                return True
            try:
                return int(path.stat().st_size) == int(expected)
            except OSError:
                return False

        def mark_archived(
            image: ImageRecord | None,
            operation: ArchiveOperationRecord | None,
            destination: Path,
        ) -> None:
            try:
                size = int(destination.stat().st_size)
            except OSError:
                size = None
            now_epoch = time.time()
            now_text = _now()
            with self.transaction():
                if image is not None and image.id is not None:
                    self._connection.execute(
                        """
                        UPDATE images SET current_path=?, pending_path=NULL,
                            archive_operation_id=?, archive_mode=?, status='ARCHIVED',
                            archive_error=NULL, updated_at=? WHERE id=?
                        """,
                        (
                            _path_string(destination),
                            operation.operation_id if operation is not None else image.archive_operation_id,
                            operation.mode if operation is not None else image.archive_mode,
                            now_text,
                            int(image.id),
                        ),
                    )
                if operation is not None:
                    self._connection.execute(
                        """
                        UPDATE archive_operations SET phase='archived', status='archived',
                            error=NULL, bytes_transferred=COALESCE(?, bytes_transferred),
                            finished_at=COALESCE(finished_at, ?), updated_at=?
                        WHERE operation_id=?
                        """,
                        (size, now_epoch, now_text, operation.operation_id),
                    )
            append_updated(image.id if image is not None else None)

        def mark_conflict(
            image: ImageRecord | None,
            operation: ArchiveOperationRecord | None,
            destination: Path,
            message: str,
        ) -> None:
            now_text = _now()
            with self.transaction():
                if image is not None and image.id is not None:
                    self._connection.execute(
                        """
                        UPDATE images SET pending_path=?, archive_operation_id=?,
                            archive_mode=?, status='FAILED', archive_error=?, updated_at=?
                        WHERE id=?
                        """,
                        (
                            _path_string(destination),
                            operation.operation_id if operation is not None else image.archive_operation_id,
                            operation.mode if operation is not None else image.archive_mode,
                            message,
                            now_text,
                            int(image.id),
                        ),
                    )
                if operation is not None:
                    self._connection.execute(
                        """
                        UPDATE archive_operations SET phase='conflict', status='conflict',
                            error=?, updated_at=? WHERE operation_id=?
                        """,
                        (message, now_text, operation.operation_id),
                    )
            append_updated(image.id if image is not None else None)

        def mark_retryable(
            image: ImageRecord | None,
            operation: ArchiveOperationRecord | None,
        ) -> None:
            if image is None or image.id is None:
                return
            with self.transaction():
                self._connection.execute(
                    """
                    UPDATE images SET status='DISCOVERED', pending_path=NULL,
                        archive_error=NULL, updated_at=? WHERE id=?
                    """,
                    (_now(), int(image.id)),
                )
            # Keep the pending operation row as an audit trail.  A subsequent
            # archive attempt gets a fresh operation id and cannot duplicate a
            # completed move because the source is still present here.
            append_updated(image.id)

        def mark_missing(
            image: ImageRecord | None,
            operation: ArchiveOperationRecord | None,
            destination: Path,
        ) -> None:
            message = "Source and destination are both missing after archive operation"
            now_text = _now()
            with self.transaction():
                if image is not None and image.id is not None:
                    self._connection.execute(
                        """
                        UPDATE images SET pending_path=?, archive_operation_id=?,
                            archive_mode=?, status='FAILED', archive_error=?, updated_at=?
                        WHERE id=?
                        """,
                        (
                            _path_string(destination),
                            operation.operation_id if operation is not None else image.archive_operation_id,
                            operation.mode if operation is not None else image.archive_mode,
                            message,
                            now_text,
                            int(image.id),
                        ),
                    )
                if operation is not None:
                    self._connection.execute(
                        """
                        UPDATE archive_operations SET phase='failed', status='failed',
                            error=?, updated_at=? WHERE operation_id=?
                        """,
                        (message, now_text, operation.operation_id),
                    )
            append_updated(image.id if image is not None else None)

        # A single image can have multiple retries.  Only the newest journal
        # row is authoritative; older rows remain available for audit queries.
        operations = self.list_archive_operations()
        latest_by_image: dict[int, ArchiveOperationRecord] = {}
        for operation in operations:
            if operation.image_id is not None:
                latest_by_image[int(operation.image_id)] = operation

        for operation in operations:
            if operation.image_id is not None and latest_by_image.get(int(operation.image_id)) is not operation:
                continue
            image = by_id.get(int(operation.image_id)) if operation.image_id is not None else None
            if image is None:
                image = self.find_image_by_path(operation.source_path)
                if image is None:
                    image = self.find_image_by_path(operation.destination_path, current=True)
            source = Path(image.original_path) if image is not None else Path(operation.source_path)
            destination = Path(operation.destination_path)
            expected = expected_size(image, operation)
            source_exists = is_regular(source)
            destination_exists = is_regular(destination)
            # A successfully committed row needs no repeated work on every
            # application start.  It is still selected for reconciliation when
            # the image row was left behind in ARCHIVING/FAILED.
            image_already_done = (
                image is not None
                and image.status == "ARCHIVED"
                and not image.pending_path
                and os.path.normcase(os.path.abspath(image.current_path or ""))
                == os.path.normcase(os.path.abspath(os.fspath(destination)))
            )
            if image_already_done and operation.phase in {"archived", "already_archived", "complete", "completed"}:
                continue
            if destination_exists and not destination_matches(destination, expected):
                mark_conflict(
                    image,
                    operation,
                    destination,
                    "Archive destination exists but does not match the recorded source size",
                )
            elif destination_exists and not source_exists:
                mark_archived(image, operation, destination)
            elif destination_exists and source_exists:
                mode = operation.mode.casefold()
                # A collision was detected before any filesystem action, so a
                # pre-existing same-size destination is not evidence that this
                # operation completed.  Leave that case for manual review;
                # failed operations, by contrast, may have completed the file
                # transfer before their final DB update failed.
                if operation.phase == "conflict":
                    mark_conflict(
                        image,
                        operation,
                        destination,
                        "Archive collision remains unresolved; source and destination both exist",
                    )
                elif mode in {"copy", "hardlink", "link"}:
                    mark_archived(image, operation, destination)
                else:
                    mark_conflict(
                        image,
                        operation,
                        destination,
                        "Both source and destination exist after an interrupted move; manual review required",
                    )
            elif source_exists:
                if operation.phase in {"pending", "archiving", "started"}:
                    mark_retryable(image, operation)
                elif image is not None and image.status == "ARCHIVING":
                    mark_conflict(
                        image,
                        operation,
                        destination,
                        "Archive operation failed before destination was created; retry is required",
                    )
            else:
                mark_missing(image, operation, destination)

        # v1 fallback: there is no operation row, so use current_path or the
        # caller-provided destination root.  Never accept a directory or a
        # known-size mismatch as a completed archive.
        for image in candidates:
            if image.id is not None and image.id in touched:
                continue
            source = Path(image.original_path)
            current = Path(image.current_path or image.original_path)
            destination: Path | None = None
            if not is_regular(source):
                if image.pending_path:
                    destination = Path(image.pending_path)
                elif current != source and is_regular(current):
                    destination = current
                elif root is not None:
                    fallback = root / (image.filename or source.name)
                    if is_regular(fallback):
                        destination = fallback
            if destination is not None:
                expected = int(image.file_size) if int(image.file_size) > 0 else None
                if not destination_matches(destination, expected):
                    mark_conflict(
                        image,
                        None,
                        destination,
                        "Legacy archive destination exists but does not match the recorded source size",
                    )
                else:
                    mark_archived(image, None, destination)
            elif image.status == "ARCHIVING" and is_regular(source):
                # The process may have died before touching the filesystem.
                # Leave the source in place and make it retryable rather than
                # presenting a permanently in-progress row to the UI.
                mark_retryable(image, None)
            elif image.status == "ARCHIVING" and not is_regular(source):
                mark_missing(image, None, Path(image.pending_path or current))
        return recovered

    recover_moves = recover_archived_images

    def find_missing_files(self) -> list[ImageRecord]:
        """Return records whose source and current paths are both absent."""

        result: list[ImageRecord] = []
        for image in self.list_images():
            source = Path(image.original_path)
            current = Path(image.current_path or image.original_path)
            if not source.exists() and not current.exists():
                result.append(image)
        return result


__all__ = ["Database", "DatabaseError"]

