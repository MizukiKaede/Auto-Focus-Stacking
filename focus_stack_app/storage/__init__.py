"""Public storage interfaces.

The package-level imports intentionally contain only standard-library-backed
modules.  In particular, importing :mod:`focus_stack_app.storage` must not
require Pillow, OpenCV, PySide6, or an external Hugin installation.  Keeping
these re-exports here gives workers and small scripts one stable import path
while the implementation modules remain independently importable.
"""

from . import cache, manifest, models
from .cache import CachePaths, DiskCache, PreviewCache, ProjectCache, RollingCache
from .database import Database, DatabaseError
from .manifest import MANIFEST_FIELDS, ManifestRow, ManifestWriter, write_manifest, write_stack_manifest
from .models import (
    Analysis,
    AnalysisRecord,
    ArchiveOperation,
    ArchiveOperationRecord,
    Group,
    GroupRecord,
    ImageMetadata,
    ImageRecord,
    ImageStatus,
    Job,
    JobRecord,
    JobStage,
    JobStatus,
)

# Older worker prototypes called the disk-backed focus/preview store an
# ``AnalysisCache``.  Keep that spelling available without duplicating or
# wrapping the cache implementation.
AnalysisCache = DiskCache

__all__ = [
    "cache",
    "manifest",
    "models",
    "Database",
    "DatabaseError",
    "ImageStatus",
    "JobStage",
    "JobStatus",
    "ImageRecord",
    "ImageMetadata",
    "GroupRecord",
    "Group",
    "AnalysisRecord",
    "Analysis",
    "JobRecord",
    "Job",
    "ArchiveOperationRecord",
    "ArchiveOperation",
    "CachePaths",
    "PreviewCache",
    "RollingCache",
    "DiskCache",
    "ProjectCache",
    "AnalysisCache",
    "MANIFEST_FIELDS",
    "ManifestRow",
    "ManifestWriter",
    "write_manifest",
    "write_stack_manifest",
]

