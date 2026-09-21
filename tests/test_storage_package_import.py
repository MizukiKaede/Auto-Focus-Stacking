"""Package-level storage imports stay safe without optional dependencies."""

from __future__ import annotations

import unittest


class StoragePackageImportTests(unittest.TestCase):
    def test_public_storage_api_is_available_from_package(self) -> None:
        import focus_stack_app.storage as storage

        from focus_stack_app.storage import (
            AnalysisCache,
            AnalysisRecord,
            ArchiveOperationRecord,
            CachePaths,
            Database,
            DatabaseError,
            DiskCache,
            ImageRecord,
            ManifestWriter,
            PreviewCache,
            RollingCache,
        )

        self.assertIs(storage.AnalysisCache, DiskCache)
        self.assertIs(AnalysisCache, DiskCache)
        self.assertIs(storage.models.ImageRecord, ImageRecord)
        self.assertIs(storage.cache.DiskCache, DiskCache)
        self.assertIs(storage.manifest.ManifestWriter, ManifestWriter)
        for public_type in (
            AnalysisRecord,
            ArchiveOperationRecord,
            CachePaths,
            Database,
            DatabaseError,
            ImageRecord,
            ManifestWriter,
            PreviewCache,
            RollingCache,
        ):
            self.assertIn(public_type.__name__, storage.__all__)


if __name__ == "__main__":
    unittest.main()
