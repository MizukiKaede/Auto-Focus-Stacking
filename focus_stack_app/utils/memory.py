"""Memory and disk-space guards used before expensive worker stages."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import shutil
from typing import Any


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    total_bytes: int
    available_bytes: int
    used_bytes: int

    @property
    def available_fraction(self) -> float:
        return self.available_bytes / self.total_bytes if self.total_bytes else 0.0

    @property
    def used_fraction(self) -> float:
        return self.used_bytes / self.total_bytes if self.total_bytes else 0.0


def memory_snapshot() -> MemorySnapshot:
    """Return system memory, using psutil or a Windows API fallback."""

    try:
        import psutil  # type: ignore[import-not-found]

        virtual = psutil.virtual_memory()
        return MemorySnapshot(int(virtual.total), int(virtual.available), int(virtual.used))
    except ImportError:
        pass

    if os.name == "nt":
        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            total = int(status.ullTotalPhys)
            available = int(status.ullAvailPhys)
            return MemorySnapshot(total, available, max(total - available, 0))

    # POSIX fallback; this also makes the helper useful in CI without psutil.
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        total = int(page_size * pages)
        available = int(page_size * available_pages)
        return MemorySnapshot(total, available, max(total - available, 0))
    except (AttributeError, OSError, ValueError):
        return MemorySnapshot(0, 0, 0)


def get_memory_snapshot() -> MemorySnapshot:
    return memory_snapshot()


def memory_available_bytes() -> int:
    return memory_snapshot().available_bytes


get_available_memory = memory_available_bytes


def has_memory_headroom(
    *,
    minimum_bytes: int = 2 * 1024**3,
    minimum_fraction: float = 0.10,
    extra_bytes: int = 0,
    snapshot: MemorySnapshot | None = None,
) -> bool:
    """Whether a new decode/job may start without crossing safety limits."""

    if minimum_bytes < 0 or extra_bytes < 0 or not 0 <= minimum_fraction <= 1:
        raise ValueError("memory limits must be non-negative and fraction must be between 0 and 1")
    current = snapshot or memory_snapshot()
    available_after = max(0, current.available_bytes - extra_bytes)
    return available_after >= minimum_bytes and (
        current.total_bytes <= 0 or available_after / current.total_bytes >= minimum_fraction
    )


def can_start_memory_intensive_task(**kwargs: Any) -> bool:
    return has_memory_headroom(**kwargs)


@dataclass(frozen=True, slots=True)
class DiskSpace:
    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    required_bytes: int = 0
    safety_margin_bytes: int = 0

    @property
    def sufficient(self) -> bool:
        return self.free_bytes >= self.required_bytes + self.safety_margin_bytes


def disk_space(
    path: str | Path,
    *,
    required_bytes: int = 0,
    safety_margin_bytes: int = 0,
) -> DiskSpace:
    """Inspect the volume containing ``path`` without writing anything."""

    if required_bytes < 0 or safety_margin_bytes < 0:
        raise ValueError("disk requirements cannot be negative")
    path = Path(path)
    probe = path if path.exists() else path.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError:
        # Let callers handle a clear zero-capacity result rather than
        # accidentally assuming unlimited space.
        return DiskSpace(str(path), 0, 0, 0, required_bytes, safety_margin_bytes)
    return DiskSpace(
        str(path),
        int(usage.total),
        int(usage.used),
        int(usage.free),
        int(required_bytes),
        int(safety_margin_bytes),
    )


def check_disk_space(
    path: str | Path,
    required_bytes: int = 0,
    *,
    safety_margin_bytes: int = 0,
) -> DiskSpace:
    return disk_space(path, required_bytes=required_bytes, safety_margin_bytes=safety_margin_bytes)


def is_disk_space_sufficient(
    path: str | Path,
    required_bytes: int = 0,
    *,
    safety_margin_bytes: int = 0,
) -> bool:
    return check_disk_space(
        path, required_bytes, safety_margin_bytes=safety_margin_bytes
    ).sufficient


__all__ = [
    "MemorySnapshot",
    "memory_snapshot",
    "get_memory_snapshot",
    "memory_available_bytes",
    "get_available_memory",
    "has_memory_headroom",
    "can_start_memory_intensive_task",
    "DiskSpace",
    "disk_space",
    "check_disk_space",
    "is_disk_space_sufficient",
]

