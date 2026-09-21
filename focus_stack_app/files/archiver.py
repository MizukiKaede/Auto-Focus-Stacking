"""Crash-safe flat-file archiving.

The archiver is intentionally independent of the project's database schema.
It accepts a small repository/adapter object and calls whichever persistence
hooks are available.  This lets the pipeline work with both the V1 SQLite
repository and light-weight test doubles.

Every operation is recorded as ``pending`` before touching a source file and
as ``archived`` only after the destination has been flushed and the source has
been removed (for a move).  A pending operation can be inspected with
``recover`` after an interrupted process.
"""

from __future__ import annotations

from ..utils.performance import timed

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol
import uuid

from .collision import Collision, CollisionError, CollisionPolicy, resolve_destination, same_file_or_path


# A FileArchiver is normally shared by the parallel merge workers, but callers
# can also construct one archiver per worker.  In either case all workers that
# publish into the same flat directory must serialize name resolution and the
# final filesystem operation.  Keep the registry process-local and key it by a
# canonical directory path so equivalent Windows spellings share one lock.
_DESTINATION_LOCKS_GUARD = threading.Lock()
_DESTINATION_LOCKS: dict[str, threading.RLock] = {}


def _destination_key(path: os.PathLike[str] | str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def _destination_lock(path: os.PathLike[str] | str) -> threading.RLock:
    key = _destination_key(path)
    with _DESTINATION_LOCKS_GUARD:
        return _DESTINATION_LOCKS.setdefault(key, threading.RLock())


class ArchiveMode(str, Enum):
    MOVE = "move"
    COPY = "copy"
    HARDLINK = "hardlink"


ArchiveMethod = ArchiveMode


class ArchiveRepository(Protocol):
    """Optional persistence hooks used by :class:`FileArchiver`.

    Implementations may expose only a subset.  The archiver probes common
    method names at runtime to keep this protocol compatible with the storage
    agent's database implementation.
    """

    def record_archive_operation(self, operation: Mapping[str, Any]) -> Any: ...


@dataclass
class ArchiveRecord:
    source_path: str
    destination_path: str
    mode: str
    status: str = "pending"
    operation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    error: str | None = None
    bytes_transferred: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    collision_path: str | None = None
    image_id: int | None = None

    @property
    def source(self) -> Path:
        return Path(self.source_path)

    @property
    def destination(self) -> Path:
        return Path(self.destination_path)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArchiveResult:
    records: list[ArchiveRecord] = field(default_factory=list)

    @property
    def archived(self) -> list[ArchiveRecord]:
        return [r for r in self.records if r.status in {"archived", "already_archived"}]

    @property
    def conflicts(self) -> list[ArchiveRecord]:
        return [r for r in self.records if r.status == "conflict"]

    @property
    def failed(self) -> list[ArchiveRecord]:
        return [r for r in self.records if r.status == "failed"]

    @property
    def ok(self) -> bool:
        return not self.conflicts and not self.failed

    @property
    def destination_paths(self) -> list[Path]:
        return [r.destination for r in self.archived]

    def __iter__(self) -> Iterator[ArchiveRecord]:
        return iter(self.records)


class ArchiveError(RuntimeError):
    """Base class for non-collision archive failures."""


class ArchiveVerificationError(ArchiveError):
    """Raised when a copied/moved destination cannot be verified."""


ProgressCallback = Callable[[ArchiveRecord], Any]


def _call_hook(repository: Any, names: tuple[str, ...], record: ArchiveRecord, **extra: Any) -> Any:
    """Call the first compatible persistence hook, if one exists.

    Database adapters in early project revisions have used both dataclass and
    mapping arguments.  We pass a mapping first and fall back to the record
    object; errors from a real hook are deliberately allowed to propagate so a
    database failure cannot be mistaken for a completed file operation.
    """

    if repository is None:
        return None
    payload = record.as_dict()
    payload.update(extra)
    for name in names:
        hook = getattr(repository, name, None)
        if not callable(hook):
            continue
        try:
            return hook(payload)
        except TypeError as first_error:
            try:
                return hook(record, **extra)
            except TypeError:
                # Do not hide a hook's TypeError unless both supported calling
                # conventions failed.  Re-raise the original for diagnostics.
                raise first_error
    return None


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def files_match(source: Path, destination: Path) -> bool:
    """Compare two files without loading either one into memory."""

    try:
        src_stat, dst_stat = source.stat(), destination.stat()
        if src_stat.st_size != dst_stat.st_size:
            return False
        # A byte comparison is substantially cheaper than hashing a 15 MB
        # image twice in the common recovery case and remains bounded in RAM.
        with source.open("rb") as src, destination.open("rb") as dst:
            while True:
                left = src.read(1024 * 1024)
                right = dst.read(1024 * 1024)
                if left != right:
                    return False
                if not left:
                    return True
    except OSError:
        return False


class FileArchiver:
    """Archive image files into one flat directory.

    Parameters
    ----------
    destination_dir:
        One directory for all original images.  No group subdirectories are
        ever created.
    mode:
        ``move`` (default), ``copy``, or ``hardlink``.
    collision_policy:
        ``error`` (default) returns a conflict record and leaves both files
        untouched.  ``skip`` has the same filesystem behaviour and is useful
        for a resumable UI.  ``rename`` is explicit opt-in.
    strict:
        If true, a collision or per-file failure raises after its record has
        been persisted.  The default lets a batch continue with other files.
    """

    def __init__(
        self,
        destination_dir: os.PathLike[str] | str,
        mode: ArchiveMode | str = ArchiveMode.MOVE,
        *,
        repository: ArchiveRepository | Any | None = None,
        db: Any | None = None,
        collision_policy: CollisionPolicy | str = CollisionPolicy.ERROR,
        strict: bool = False,
        logger: logging.Logger | None = None,
        progress_callback: ProgressCallback | None = None,
        operation_id_factory: Callable[[], str] | None = None,
    ):
        self.destination_dir = Path(destination_dir)
        self.mode = ArchiveMode(mode)
        self.repository = repository if repository is not None else db
        self.collision_policy = CollisionPolicy(collision_policy)
        self.strict = strict
        self.logger = logger or logging.getLogger(__name__)
        self.progress_callback = progress_callback
        self.operation_id_factory = operation_id_factory or (lambda: uuid.uuid4().hex)
        self._occupied: set[Path] = set()
        # The destination-directory lock is the primary serialization point;
        # this small lock also keeps the per-instance reservation set safe for
        # integrations that inspect or use the low-level helpers directly.
        self._occupied_lock = threading.RLock()

    def _notify(self, record: ArchiveRecord) -> None:
        if self.progress_callback is not None:
            self.progress_callback(record)

    def _persist(self, record: ArchiveRecord, phase: str) -> None:
        """Persist a file operation state using the available adapter hooks."""

        _call_hook(
            self.repository,
            (
                "record_archive_operation",
                "upsert_archive_operation",
                "save_archive_operation",
                "record_file_operation",
                "upsert_file_operation",
            ),
            record,
            phase=phase,
        )

        # The built-in Database does not need a second archive journal table;
        # its image row is the durable per-file journal.  Marking ARCHIVING
        # before the filesystem operation and FAILED on an exception makes a
        # crash/restart visible even when no custom operation hook exists.
        image_id = record.image_id
        if image_id is None and self.repository is not None:
            finder = getattr(self.repository, "find_image_by_path", None)
            if callable(finder):
                try:
                    image = finder(record.source_path)
                    image_id = getattr(image, "id", None) if image is not None else None
                except Exception:
                    self.logger.debug("Unable to resolve image id for %s", record.source_path, exc_info=True)
        update_image = getattr(self.repository, "update_image_path", None) if self.repository is not None else None
        if callable(update_image) and image_id is not None and phase in {"pending", "failed", "conflict"}:
            state = "ARCHIVING" if phase == "pending" else "FAILED"
            update_image(int(image_id), record.source_path, status=state, archive_mode=record.mode)

        # Keep image-path state in sync when the storage adapter exposes the
        # common update method.  It is okay for an adapter to implement only
        # operation journaling or only image updates.
        if phase in {"archived", "already_archived"}:
            # Database.update_image_path(image_id, current_path, *, status,
            # archive_mode) is the canonical SQLite signature.  Resolve an id
            # lazily from original_path when callers did not supply one.
            image_id = record.image_id
            if image_id is None and self.repository is not None:
                finder = getattr(self.repository, "find_image_by_path", None)
                if callable(finder):
                    try:
                        image = finder(record.source_path)
                        image_id = getattr(image, "id", None) if image is not None else None
                    except Exception:
                        self.logger.debug("Unable to resolve image id for %s", record.source_path, exc_info=True)
            update = getattr(self.repository, "update_image_path", None) if self.repository is not None else None
            if callable(update) and image_id is not None:
                update(
                    int(image_id),
                    record.destination_path,
                    status="ARCHIVED",
                    archive_mode=record.mode,
                )
            else:
                _call_hook(
                    self.repository,
                    ("set_image_current_path", "mark_image_archived"),
                    record,
                    current_path=record.destination_path,
                    status="ARCHIVED",
                )

    def _new_record(self, source: Path, destination: Path) -> ArchiveRecord:
        return ArchiveRecord(
            source_path=str(source),
            destination_path=str(destination),
            mode=self.mode.value,
            operation_id=self.operation_id_factory(),
            started_at=time.time(),
        )

    def archive_file(self, source: os.PathLike[str] | str, *, image_id: int | None = None) -> ArchiveRecord:
        """Archive one source file and return its durable operation record.

        Name resolution, collision handling, and publication are kept under a
        process-wide lock for the destination directory.  This is required
        when parallel merge workers share an archiver *or* construct separate
        archivers for the same flat archive directory.
        """

        with _destination_lock(self.destination_dir):
            return self._archive_file_locked(source, image_id=image_id)

    def _archive_file_locked(
        self,
        source: os.PathLike[str] | str,
        *,
        image_id: int | None = None,
    ) -> ArchiveRecord:
        """Archive one source file and return its durable operation record."""

        src = Path(source)
        # The source must be a regular file.  Resolve only for diagnostics; do
        # not dereference a symlink into a different target for the operation.
        try:
            if not src.is_file():
                raise FileNotFoundError(f"Source file does not exist: {src}")
        except OSError as exc:
            raise FileNotFoundError(f"Cannot inspect source file {src}: {exc}") from exc

        self.destination_dir.mkdir(parents=True, exist_ok=True)
        with self._occupied_lock:
            occupied = tuple(self._occupied)
        destination, collision = resolve_destination(
            src,
            self.destination_dir,
            policy=self.collision_policy,
            occupied=occupied,
        )
        record = self._new_record(src, destination)
        record.image_id = image_id
        if collision is not None:
            record.collision_path = str(collision.destination)
            if collision.same_file:
                record.status = "already_archived"
                record.bytes_transferred = src.stat().st_size
                record.finished_at = time.time()
                self._persist(record, "already_archived")
                with self._occupied_lock:
                    self._occupied.add(destination)
                self._notify(record)
                return record
            if self.collision_policy is not CollisionPolicy.RENAME:
                record.status = "conflict"
                record.error = str(CollisionError(src, collision.destination))
                # The conflict itself is durable information; persist it even
                # though no filesystem operation is attempted.
                self._persist(record, "conflict")
                self._notify(record)
                if self.strict or self.collision_policy is CollisionPolicy.ERROR:
                    # In non-strict batch mode return the record to let the
                    # caller continue.  Strict mode is intended for one-file
                    # workflows.
                    if self.strict:
                        raise CollisionError(src, collision.destination)
                return record
            # RENAME deliberately keeps the original collision for the
            # record's audit trail but proceeds with the generated destination.

        self._persist(record, "pending")
        try:
            if self.mode is ArchiveMode.MOVE:
                self._move_no_overwrite(src, destination)
            elif self.mode is ArchiveMode.COPY:
                self._copy_no_overwrite(src, destination)
            else:
                self._hardlink_no_overwrite(src, destination)
            record.status = "archived"
            record.bytes_transferred = destination.stat().st_size
            record.finished_at = time.time()
            self._persist(record, "archived")
            with self._occupied_lock:
                self._occupied.add(destination)
            self.logger.info("Archived %s -> %s (%s)", src, destination, self.mode.value)
        except Exception as exc:
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
            record.finished_at = time.time()
            try:
                self._persist(record, "failed")
            except Exception:
                self.logger.exception("Unable to persist failed archive operation %s", record.operation_id)
            self.logger.exception("Archive failed: %s -> %s", src, destination)
            if self.strict:
                raise
        self._notify(record)
        return record

    @timed("archive")
    def archive_files(self, sources: Iterable[os.PathLike[str] | str | Any]) -> ArchiveResult:
        """Archive all files, continuing after per-file failures."""

        result = ArchiveResult()
        for source_item in sources:
            if isinstance(source_item, Mapping):
                source = source_item.get(
                    "current_path",
                    source_item.get("path", source_item.get("original_path", source_item.get("filename", source_item))),
                )
                image_id = source_item.get("id")
            else:
                source = getattr(source_item, "path", source_item)
                image_id = getattr(source_item, "id", None)
            try:
                result.records.append(self.archive_file(source, image_id=image_id))
            except Exception as exc:
                # A missing source or strict failure is still represented so a
                # caller can include it in the manifest.
                src = Path(source)
                record = self._new_record(src, self.destination_dir / src.name)
                record.status = "failed"
                record.error = f"{type(exc).__name__}: {exc}"
                record.finished_at = time.time()
                try:
                    # A source can fail before ``archive_file`` has created
                    # its pending record (for example, it disappeared between
                    # scan and processing).  Keep this batch-level failure in
                    # the same durable journal so recovery/manifest consumers
                    # do not see a silent gap.
                    self._persist(record, "failed")
                except Exception:
                    self.logger.exception("Unable to persist failed archive operation %s", record.operation_id)
                result.records.append(record)
                if self.strict:
                    raise
        return result

    # Friendly aliases used by older call sites.
    archive = archive_file
    archive_batch = archive_files

    def _reserve_destination(self, destination: Path) -> None:
        if destination.exists():
            raise FileExistsError(f"Destination appeared during archive: {destination}")

    @staticmethod
    def _flush_file(path: Path) -> None:
        with path.open("rb") as stream:
            # Reading through the descriptor ensures the temporary file is
            # complete before an atomic rename; fsync is best-effort on some
            # Windows filesystems and is therefore handled as an OSError.
            try:
                os.fsync(stream.fileno())
            except OSError:
                pass

    def _copy_to_temp(self, source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # The temporary is created in the destination directory so linking it
        # into place below is atomic even when source and destination are on
        # different volumes.
        fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent)
        os.close(fd)
        temp = Path(temp_name)
        try:
            shutil.copy2(source, temp)
            with temp.open("rb") as stream:
                try:
                    os.fsync(stream.fileno())
                except OSError:
                    pass
            return temp
        except Exception:
            temp.unlink(missing_ok=True)
            raise

    def _publish_temp_no_overwrite(self, temp: Path, destination: Path) -> None:
        """Publish *temp* at *destination* without replacing an existing file.

        ``os.replace`` is deliberately not used here: on Windows and POSIX it
        replaces a destination that appeared after an earlier existence check.
        A hard link created from a same-directory temporary gives us an atomic
        no-clobber publish.  Filesystems that do not support hard links fall
        back to an exclusive create and bounded streaming copy; that fallback
        is not fully atomic, but it still cannot overwrite an existing path.
        """

        try:
            # The temporary and destination are in the same directory, so this
            # is normally one atomic directory operation followed by cleanup
            # of the old temporary name by the caller.
            os.link(temp, destination)
            return
        except FileExistsError:
            # Do not turn a real collision into the non-atomic fallback.
            raise
        except OSError as link_error:
            self.logger.debug(
                "Hard-link publish unavailable (%s); using exclusive copy for %s",
                link_error,
                destination,
            )

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0)
        descriptor: int | None = None
        created = False
        try:
            descriptor = os.open(os.fspath(destination), flags, 0o666)
            created = True
            stream = os.fdopen(descriptor, "wb")
            descriptor = None
            with stream:
                with temp.open("rb") as source_stream:
                    shutil.copyfileobj(source_stream, stream, length=1024 * 1024)
                stream.flush()
                try:
                    os.fsync(stream.fileno())
                except OSError:
                    pass
            # copy2 already applied metadata to the temporary.  Keep that
            # behaviour for filesystems where the exclusive fallback permits
            # metadata updates, while treating unsupported metadata as a
            # non-fatal platform limitation.
            try:
                shutil.copystat(temp, destination)
            except OSError:
                self.logger.debug("Unable to copy archive metadata to %s", destination, exc_info=True)
        except Exception:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            # Only remove a path after our exclusive create succeeded.  In
            # particular, if another process created the destination between
            # the failed hard-link attempt and os.open(O_EXCL), that
            # FileExistsError must not cause us to unlink its file.
            if created:
                try:
                    destination.unlink(missing_ok=True)
                except OSError:
                    self.logger.debug("Unable to clean failed destination %s", destination, exc_info=True)
            raise

    def _copy_no_overwrite(self, source: Path, destination: Path) -> None:
        self._reserve_destination(destination)
        temp = self._copy_to_temp(source, destination)
        try:
            self._publish_temp_no_overwrite(temp, destination)
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                # Publication has already committed (or failed independently);
                # a stale private temporary must not turn a successful archive
                # into a false failure or hide the original error.
                self.logger.debug("Unable to clean archive temporary %s", temp, exc_info=True)
        if not files_match(source, destination):
            # Do not remove a source when verification fails.  The destination
            # was created by this operation, so removing it avoids leaving a
            # corrupt file that would block a later retry.
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                self.logger.debug("Unable to clean unverified destination %s", destination, exc_info=True)
            raise ArchiveVerificationError(f"Copied file failed verification: {source} -> {destination}")

    def _move_no_overwrite(self, source: Path, destination: Path) -> None:
        self._reserve_destination(destination)
        # A hard-link + unlink is an atomic no-overwrite move on the same
        # filesystem.  It also avoids os.replace's overwrite semantics on
        # POSIX.  If the files are on different volumes, copy and then unlink
        # only after a byte-for-byte verification.
        try:
            os.link(source, destination)
            try:
                if not files_match(source, destination):
                    raise ArchiveVerificationError(f"Moved file failed verification: {source} -> {destination}")
            except Exception:
                destination.unlink(missing_ok=True)
                raise
            source.unlink()
            return
        except FileExistsError:
            raise
        except OSError as link_error:
            self.logger.debug("Hard-link move unavailable (%s); using verified copy", link_error)

        self._copy_no_overwrite(source, destination)
        try:
            source.unlink()
        except Exception:
            # Keeping both copies is safer than reporting a successful move;
            # the operation remains failed and recovery can retry the unlink.
            raise

    def _hardlink_no_overwrite(self, source: Path, destination: Path) -> None:
        self._reserve_destination(destination)
        try:
            os.link(source, destination)
        except OSError as exc:
            raise ArchiveError(
                f"Unable to create hard link {destination} from {source}: {exc}. "
                "Use copy or move when source and destination are on different filesystems."
            ) from exc

    def recover(self, records: Iterable[ArchiveRecord | Mapping[str, Any]]) -> ArchiveResult:
        """Reconcile pending records after a crash.

        No file is deleted by recovery.  If both source and destination exist,
        matching bytes are treated as an interrupted copy/move and the record
        is marked archived only for copy/hardlink; a move keeps both files and
        is marked ``conflict`` because silently deleting an original would be
        unsafe.  The caller can choose to resolve that case explicitly.
        """

        result = ArchiveResult()
        for raw in records:
            record = raw if isinstance(raw, ArchiveRecord) else ArchiveRecord(**dict(raw))
            src, dst = record.source, record.destination
            src_exists, dst_exists = src.exists(), dst.exists()
            if dst_exists and not src_exists:
                record.status = "archived"
                record.finished_at = record.finished_at or time.time()
                try:
                    record.bytes_transferred = dst.stat().st_size
                    self._persist(record, "archived")
                except OSError:
                    pass
            elif src_exists and dst_exists:
                if same_file_or_path(src, dst) or files_match(src, dst):
                    if record.mode == ArchiveMode.MOVE.value:
                        record.status = "conflict"
                        record.error = "Both source and destination exist after an interrupted move; manual review required"
                        self._persist(record, "conflict")
                    else:
                        record.status = "archived"
                        record.bytes_transferred = dst.stat().st_size
                        self._persist(record, "archived")
                else:
                    record.status = "conflict"
                    record.error = "Source and destination both exist with different contents"
                    self._persist(record, "conflict")
            elif src_exists and not dst_exists:
                record.status = "pending"
                record.error = None
                self._persist(record, "pending")
            else:
                record.status = "failed"
                record.error = "Source and destination are both missing"
                self._persist(record, "failed")
            result.records.append(record)
            self._notify(record)
        return result

    recover_pending = recover


def _safe_transfer(source: os.PathLike[str] | str, destination: os.PathLike[str] | str, mode: ArchiveMode) -> Path:
    """Transfer a file to an exact path using the same no-overwrite primitives.

    These small functional helpers are useful for integrations that already
    computed a destination path; the normal group workflow should use
    :class:`FileArchiver` so its SQLite journal is populated.
    """

    src, dst = Path(source), Path(destination)
    if not src.is_file():
        raise FileNotFoundError(src)
    if same_file_or_path(src, dst):
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise CollisionError(src, dst)
    helper = FileArchiver(dst.parent, mode=mode, strict=True)
    if mode is ArchiveMode.MOVE:
        helper._move_no_overwrite(src, dst)
    elif mode is ArchiveMode.COPY:
        helper._copy_no_overwrite(src, dst)
    else:
        helper._hardlink_no_overwrite(src, dst)
    return dst


def safe_move(source: os.PathLike[str] | str, destination: os.PathLike[str] | str) -> Path:
    return _safe_transfer(source, destination, ArchiveMode.MOVE)


def safe_copy(source: os.PathLike[str] | str, destination: os.PathLike[str] | str) -> Path:
    return _safe_transfer(source, destination, ArchiveMode.COPY)


def safe_hardlink(source: os.PathLike[str] | str, destination: os.PathLike[str] | str) -> Path:
    return _safe_transfer(source, destination, ArchiveMode.HARDLINK)


__all__ = [
    "ArchiveError",
    "ArchiveMode",
    "ArchiveMethod",
    "ArchiveRecord",
    "ArchiveRepository",
    "ArchiveResult",
    "ArchiveVerificationError",
    "CollisionError",
    "CollisionPolicy",
    "FileArchiver",
    "files_match",
    "safe_copy",
    "safe_hardlink",
    "safe_move",
]
