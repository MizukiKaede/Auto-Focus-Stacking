"""Collision handling for flat archive and output directories.

The application deliberately keeps all source images in one directory.  A
collision is therefore a normal, recoverable condition (for example when a
batch is resumed), but it must never result in an implicit overwrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import os
import re
from typing import Iterable


class CollisionPolicy(str, Enum):
    """How a caller wants to report a destination collision.

    ``ERROR`` is the safe default.  ``SKIP`` is useful when resuming a batch;
    it still returns a conflict record and never overwrites a file.
    ``RENAME`` is opt-in and produces a deterministic non-conflicting name.
    """

    ERROR = "error"
    SKIP = "skip"
    RENAME = "rename"


class CollisionError(FileExistsError):
    """Raised when a destination already exists and overwrite is forbidden."""

    def __init__(self, source: os.PathLike[str] | str, destination: os.PathLike[str] | str):
        self.source = Path(source)
        self.destination = Path(destination)
        super().__init__(f"Destination already exists: {self.destination} (source: {self.source})")


@dataclass(frozen=True)
class Collision:
    source: Path
    destination: Path
    exists: bool
    same_file: bool = False


def same_file_or_path(source: os.PathLike[str] | str, destination: os.PathLike[str] | str) -> bool:
    """Return whether *source* and *destination* identify the same file.

    ``Path.samefile`` is preferred because it catches aliases and hard links,
    while the resolved-path fallback works when one side does not exist.
    """

    src = Path(source)
    dst = Path(destination)
    try:
        if src.exists() and dst.exists() and os.path.samefile(src, dst):
            return True
    except (OSError, ValueError):
        pass
    try:
        return os.fspath(src.resolve(strict=False)).casefold() == os.fspath(dst.resolve(strict=False)).casefold()
    except OSError:
        return os.path.abspath(os.fspath(src)).casefold() == os.path.abspath(os.fspath(dst)).casefold()


def _next_name(path: Path, occupied: Iterable[Path] = ()) -> Path:
    """Return ``path`` with a `` (n)`` suffix that does not exist.

    The implementation intentionally does not create a reservation.  Callers
    still perform an exclusive/atomic operation and must handle a race by
    reporting a collision.
    """

    occupied_keys = {os.path.normcase(os.path.abspath(os.fspath(p))) for p in occupied}
    if not path.exists() and os.path.normcase(os.path.abspath(os.fspath(path))) not in occupied_keys:
        return path
    stem, suffix = path.stem, path.suffix
    # Keep the generated name readable for camera names that already contain a
    # collision suffix.
    match = re.match(r"^(.*) \((\d+)\)$", stem)
    base = match.group(1) if match else stem
    n = int(match.group(2)) + 1 if match else 1
    while True:
        candidate = path.with_name(f"{base} ({n}){suffix}")
        key = os.path.normcase(os.path.abspath(os.fspath(candidate)))
        if not candidate.exists() and key not in occupied_keys:
            return candidate
        n += 1


def resolve_destination(
    source: os.PathLike[str] | str,
    target_dir: os.PathLike[str] | str,
    *,
    policy: CollisionPolicy | str = CollisionPolicy.ERROR,
    occupied: Iterable[Path] = (),
) -> tuple[Path, Collision | None]:
    """Resolve a flat destination without ever selecting an existing file.

    The returned collision is non-``None`` when the basename already exists.
    Under ``RENAME`` the returned path is the generated path while the
    collision object retains the original conflicting path.
    """

    src = Path(source)
    dst = Path(target_dir) / src.name
    policy = CollisionPolicy(policy)
    if same_file_or_path(src, dst):
        return dst, Collision(src, dst, exists=dst.exists(), same_file=True)
    if not dst.exists():
        return dst, None
    collision = Collision(src, dst, exists=True, same_file=False)
    if policy is CollisionPolicy.RENAME:
        return _next_name(dst, occupied), collision
    return dst, collision


__all__ = [
    "Collision",
    "CollisionError",
    "CollisionPolicy",
    "resolve_destination",
    "same_file_or_path",
]

