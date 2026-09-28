"""Streaming, metadata-only JPEG directory scanner."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging as std_logging
import os
from pathlib import Path
from typing import Iterable, Iterator

from ..config import ScannerConfig
from ..storage.models import ImageRecord
from ..utils.exif import ExifMetadata, read_exif
from ..utils.image_io import ImageIOError, read_image_size
from ..utils.natural_sort import natural_sort_key


class ScanError(RuntimeError):
    """Raised when the selected root cannot be enumerated."""


@dataclass(slots=True)
class ScanIssue:
    path: str
    message: str


@dataclass(slots=True)
class ScanReport:
    root: str
    images: list[ImageRecord] = field(default_factory=list)
    issues: list[ScanIssue] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.images)

    @property
    def total_bytes(self) -> int:
        return sum(image.file_size for image in self.images)


class ImageScanner:
    """Discover JPG/JPEG files while retaining only bounded metadata.

    ``scan`` stores one lightweight record per file, not decoded pixel data.
    ``iter_scan`` yields that already ordered list one record at a time so a
    caller can persist incrementally.  A complete list of paths/records is
    still needed for deterministic capture-time ordering and is tiny compared
    with image contents (roughly a few hundred bytes per file).
    """

    def __init__(
        self,
        root: str | Path,
        config: ScannerConfig | None = None,
        *,
        excluded_roots: Iterable[str | Path] = (),
        logger: std_logging.Logger | None = None,
    ) -> None:
        self.root = Path(root)
        self.config = config or ScannerConfig()
        self.excluded_roots = tuple(Path(path).resolve(strict=False) for path in excluded_roots)
        self.logger = logger or std_logging.getLogger(__name__)
        self.last_report: ScanReport | None = None

    def _is_excluded(self, path: Path) -> bool:
        resolved = path.resolve(strict=False)
        return any(resolved == root or root in resolved.parents for root in self.excluded_roots)

    def _iter_paths(self) -> Iterator[Path]:
        if not self.root.exists() or not self.root.is_dir():
            raise ScanError(f"scan root is not a directory: {self.root}")
        if self.config.recursive:
            for base, dirs, names in os.walk(self.root):
                # Never inspect the project's own cache when users select its
                # parent directory recursively.
                dirs[:] = [
                    name
                    for name in dirs
                    if name != ".stack_cache"
                    and (self.config.include_hidden or not name.startswith("."))
                    and not self._is_excluded(Path(base) / name)
                ]
                for name in names:
                    path = Path(base) / name
                    if self._accept(path):
                        yield path
        else:
            try:
                entries = self.root.iterdir()
            except OSError as exc:
                raise ScanError(f"unable to enumerate {self.root}: {exc}") from exc
            for path in entries:
                if path.is_file() and self._accept(path):
                    yield path

    def _accept(self, path: Path) -> bool:
        if not self.config.include_hidden and path.name.startswith("."):
            return False
        return path.suffix.lower() in self.config.extensions

    def _read_record(self, path: Path, issues: list[ScanIssue]) -> ImageRecord | None:
        try:
            stat = path.stat()
        except OSError as exc:
            issues.append(ScanIssue(str(path), f"stat failed: {exc}"))
            self.logger.warning("Unable to stat image %s: %s", path, exc)
            return None

        # Each helper reads headers only.  Failure to parse one corrupt image
        # should not discard the rest of a batch.
        size = None
        try:
            size = read_image_size(path)
        except ImageIOError as exc:
            issues.append(ScanIssue(str(path), str(exc)))
            self.logger.warning("Unable to read dimensions for %s: %s", path, exc)

        try:
            exif: ExifMetadata = read_exif(path)
        except Exception as exc:  # defensive boundary for vendor EXIF
            exif = ExifMetadata()
            issues.append(ScanIssue(str(path), f"EXIF read failed: {exc}"))
            self.logger.warning("Unable to read EXIF for %s: %s", path, exc)

        resolved = os.path.normpath(os.path.abspath(os.fspath(path)))
        return ImageRecord(
            original_path=resolved,
            current_path=resolved,
            filename=path.name,
            stem=path.stem,
            extension=path.suffix.lower(),
            file_size=int(stat.st_size),
            mtime=float(stat.st_mtime),
            capture_time=exif.capture_time,
            width=size.width if size else None,
            height=size.height if size else None,
            camera=exif.camera,
            lens=exif.lens,
        )

    @staticmethod
    def _sort_key(image: ImageRecord) -> tuple[object, ...]:
        # Prefer capture date whenever available.  Missing dates are grouped
        # after dated files and naturally sorted by filename/path.
        if image.capture_time:
            return (0, image.capture_time, natural_sort_key(image.filename or ""), natural_sort_key(image.original_path))
        return (1, natural_sort_key(image.filename or ""), natural_sort_key(image.original_path))

    def scan_report(self) -> ScanReport:
        issues: list[ScanIssue] = []
        records: list[ImageRecord] = []
        for path in self._iter_paths():
            record = self._read_record(path, issues)
            if record is not None:
                records.append(record)
        records.sort(key=self._sort_key)
        for index, record in enumerate(records):
            record.sequence_index = index
        report = ScanReport(root=os.path.normpath(os.path.abspath(os.fspath(self.root))), images=records, issues=issues)
        self.last_report = report
        self.logger.info(
            "Scanned %d JPEG files from %s (%d bytes, %d issues)",
            report.count,
            report.root,
            report.total_bytes,
            len(report.issues),
        )
        return report

    def scan(self) -> list[ImageRecord]:
        return self.scan_report().images

    def iter_scan(self) -> Iterator[ImageRecord]:
        yield from self.scan()


def scan_directory(
    root: str | Path,
    *,
    config: ScannerConfig | None = None,
    recursive: bool | None = None,
    include_hidden: bool | None = None,
) -> list[ImageRecord]:
    """Convenience metadata scan returning records in capture/natural order."""

    if recursive is not None or include_hidden is not None:
        base = config or ScannerConfig()
        config = ScannerConfig(
            extensions=base.extensions,
            recursive=base.recursive if recursive is None else recursive,
            include_hidden=base.include_hidden if include_hidden is None else include_hidden,
        )
    return ImageScanner(root, config).scan()


def iter_scan_directory(
    root: str | Path,
    *,
    config: ScannerConfig | None = None,
    recursive: bool | None = None,
    include_hidden: bool | None = None,
) -> Iterator[ImageRecord]:
    scanner = ImageScanner(root, config)
    if recursive is not None or include_hidden is not None:
        base = config or ScannerConfig()
        scanner.config = ScannerConfig(
            extensions=base.extensions,
            recursive=base.recursive if recursive is None else recursive,
            include_hidden=base.include_hidden if include_hidden is None else include_hidden,
        )
    yield from scanner.iter_scan()


# Backward-compatible aliases that keep callers concise.
scan_folder = scan_directory
scan_images = scan_directory
Scanner = ImageScanner
ImageMetadata = ImageRecord
ScanResult = ScanReport

__all__ = [
    "ScanError",
    "ScanIssue",
    "ScanReport",
    "ImageScanner",
    "Scanner",
    "scan_directory",
    "scan_folder",
    "scan_images",
    "iter_scan_directory",
    "ImageMetadata",
    "ScanResult",
]

