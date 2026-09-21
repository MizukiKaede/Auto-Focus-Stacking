"""Project-local disk cache with bounded in-memory preview storage."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from threading import RLock
from typing import Any, Callable, Generic, TypeVar

from ..config import CacheConfig
from ..utils.image_io import image_fingerprint


T = TypeVar("T")
@dataclass(frozen=True, slots=True)
class CachePaths:
    """Stable cache directory layout for one project/output root."""

    root: Path
    previews: Path
    focusmaps: Path
    temp: Path
    logs: Path
    database: Path

    @classmethod
    def for_project(cls, project_root: str | Path, config: CacheConfig | None = None) -> "CachePaths":
        config = config or CacheConfig()
        base = Path(project_root)
        root = base if base.name == config.directory_name else base / config.directory_name
        return cls(
            root=root,
            previews=root / config.preview_directory,
            focusmaps=root / config.focusmap_directory,
            temp=root / config.temp_directory,
            logs=root / config.log_directory,
            database=root / config.database_filename,
        )

    def ensure(self) -> "CachePaths":
        for directory in (self.root, self.previews, self.focusmaps, self.temp, self.logs):
            directory.mkdir(parents=True, exist_ok=True)
        return self


class PreviewCache(Generic[T]):
    """Small LRU cache; it evicts before the configured bound is exceeded."""

    def __init__(self, max_items: int = 12) -> None:
        if max_items < 0:
            raise ValueError("max_items cannot be negative")
        self.max_items = max_items
        self._items: OrderedDict[str, T] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str, default: T | None = None) -> T | None:
        with self._lock:
            if key not in self._items:
                return default
            value = self._items.pop(key)
            self._items[key] = value
            return value

    def put(self, key: str, value: T) -> None:
        with self._lock:
            if self.max_items == 0:
                return
            self._items.pop(key, None)
            self._items[key] = value
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._items.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._items.keys())


class DiskCache:
    """Disk-backed preview/focus cache keyed by source stat fingerprint.

    A cache path includes ``size + mtime_ns`` through
    :func:`image_fingerprint`; edits naturally produce a new file and leave
    an old entry available for optional cleanup.  Writes use a sibling temp
    file and ``replace`` so a crash cannot leave a partially written image.
    """

    def __init__(
        self,
        project_root: str | Path,
        config: CacheConfig | None = None,
        *,
        max_preview_items: int = 12,
        max_focus_items: int = 4,
    ) -> None:
        self.paths = CachePaths.for_project(project_root, config).ensure()
        self.preview_memory: PreviewCache[Any] = PreviewCache(max_preview_items)
        self.focus_memory: PreviewCache[Any] = PreviewCache(max_focus_items)
        self._lock = RLock()

    @staticmethod
    def key_for(image_path: str | Path) -> str:
        return image_fingerprint(image_path)

    cache_key = key_for

    @staticmethod
    def _check_suffix(suffix: str) -> str:
        suffix = str(suffix)
        if not suffix.startswith(".") or Path(suffix).name != suffix or len(suffix) > 10:
            raise ValueError("cache suffix must be a short extension such as .jpg or .npz")
        return suffix.lower()

    def preview_path(self, image_path: str | Path, *, suffix: str = ".jpg") -> Path:
        suffix = self._check_suffix(suffix)
        return self.paths.previews / f"{self.key_for(image_path)}{suffix}"

    def focus_map_path(self, image_path: str | Path, *, suffix: str = ".npz") -> Path:
        suffix = self._check_suffix(suffix)
        return self.paths.focusmaps / f"{self.key_for(image_path)}{suffix}"

    # Clear aliases used by worker code.
    get_preview_path = preview_path
    get_focus_map_path = focus_map_path

    def _write_atomic(self, target: Path, data: bytes) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            # NamedTemporaryFile is closed before replace, which works on
            # Windows where an open target cannot be atomically replaced.
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
            return target
        finally:
            if temporary is not None and temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass

    def write_preview(self, image_path: str | Path, data: bytes, *, suffix: str = ".jpg") -> Path:
        target = self.preview_path(image_path, suffix=suffix)
        with self._lock:
            return self._write_atomic(target, bytes(data))

    def write_focus_map(self, image_path: str | Path, data: bytes, *, suffix: str = ".npz") -> Path:
        target = self.focus_map_path(image_path, suffix=suffix)
        with self._lock:
            return self._write_atomic(target, bytes(data))

    def read_bytes(self, path: str | Path) -> bytes | None:
        try:
            return Path(path).read_bytes()
        except OSError:
            return None

    def read_preview(self, image_path: str | Path, *, suffix: str = ".jpg") -> bytes | None:
        return self.read_bytes(self.preview_path(image_path, suffix=suffix))

    def read_focus_map(self, image_path: str | Path, *, suffix: str = ".npz") -> bytes | None:
        return self.read_bytes(self.focus_map_path(image_path, suffix=suffix))

    def get_or_create_preview(
        self,
        image_path: str | Path,
        creator: Callable[[], T],
        *,
        suffix: str = ".jpg",
    ) -> T:
        """Return the bounded-memory preview or create one for this process.

        ``creator`` may return any decoded/encoded object; use
        :meth:`get_or_create_preview_bytes` when persistence is desired.  This
        keeps the cache independent of Pillow/OpenCV while still offering a
        common rolling memory API.
        """

        key = self.key_for(image_path)
        memory_value = self.preview_memory.get(key)
        if memory_value is not None:
            return memory_value
        value = creator()
        self.preview_memory.put(key, value)
        return value

    def get_or_create_preview_bytes(
        self,
        image_path: str | Path,
        creator: Callable[[], bytes],
        *,
        suffix: str = ".jpg",
    ) -> bytes:
        """Load a persisted encoded preview or atomically generate it."""

        key = self.key_for(image_path)
        memory_value = self.preview_memory.get(key)
        if isinstance(memory_value, bytes):
            return memory_value
        cached = self.read_preview(image_path, suffix=suffix)
        if cached is not None:
            self.preview_memory.put(key, cached)
            return cached
        value = bytes(creator())
        self.write_preview(image_path, value, suffix=suffix)
        self.preview_memory.put(key, value)
        return value

    def get_or_create_focus_map_bytes(
        self,
        image_path: str | Path,
        creator: Callable[[], bytes],
        *,
        suffix: str = ".npz",
    ) -> bytes:
        """Load a persisted focus-map blob or create it once."""

        key = self.key_for(image_path)
        memory_value = self.focus_memory.get(key)
        if isinstance(memory_value, bytes):
            return memory_value
        cached = self.read_focus_map(image_path, suffix=suffix)
        if cached is not None:
            self.focus_memory.put(key, cached)
            return cached
        value = bytes(creator())
        self.write_focus_map(image_path, value, suffix=suffix)
        self.focus_memory.put(key, value)
        return value

    def invalidate(self, image_path: str | Path) -> int:
        """Drop current preview/focus entries; stale files are left intact."""

        key = self.key_for(image_path)
        self.preview_memory.delete(key)
        self.focus_memory.delete(key)
        removed = 0
        for directory in (self.paths.previews, self.paths.focusmaps):
            for candidate in directory.glob(f"{key}.*"):
                try:
                    candidate.unlink()
                except OSError:
                    continue
                removed += 1
        return removed

    def prune(self, *, keep_keys: set[str] | None = None) -> int:
        """Remove stale cache files, optionally retaining a set of keys."""

        keep_keys = keep_keys or set()
        removed = 0
        for directory in (self.paths.previews, self.paths.focusmaps):
            for candidate in directory.iterdir():
                if not candidate.is_file() or candidate.stem in keep_keys:
                    continue
                try:
                    candidate.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed


# Alternate names for integrations.
ProjectCache = DiskCache
RollingCache = PreviewCache

__all__ = ["CachePaths", "PreviewCache", "RollingCache", "DiskCache", "ProjectCache"]
