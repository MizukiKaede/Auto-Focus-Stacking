from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from focus_stack_app.config import ScannerConfig
from focus_stack_app.core.scanner import ImageScanner
from focus_stack_app.utils.exif import normalize_capture_time
from focus_stack_app.utils.image_io import read_jpeg_size


def _minimal_jpeg(width: int = 200, height: int = 100) -> bytes:
    # Valid enough for JPEG header/metadata tests; no entropy-coded image is
    # needed because the scanner must never decode pixels.
    sof_payload = bytes([8]) + height.to_bytes(2, "big") + width.to_bytes(2, "big") + bytes([1, 1, 0x11, 0])
    return b"\xff\xd8\xff\xc0" + (len(sof_payload) + 2).to_bytes(2, "big") + sof_payload + b"\xff\xd9"


class ImageIOScannerTests(unittest.TestCase):
    def test_jpeg_dimensions_are_header_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.JPG"
            path.write_bytes(_minimal_jpeg(3776, 2832))
            self.assertEqual(read_jpeg_size(path).width, 3776)
            self.assertEqual(read_jpeg_size(path).height, 2832)

    def test_scan_natural_order_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "DSC10.JPG").write_bytes(_minimal_jpeg())
            (root / "DSC2.JPEG").write_bytes(_minimal_jpeg())
            (root / "DSC1.jpg").write_bytes(_minimal_jpeg())
            (root / "ignore.png").write_bytes(b"not an image")
            images = ImageScanner(root).scan()
            self.assertEqual([item.filename for item in images], ["DSC1.jpg", "DSC2.JPEG", "DSC10.JPG"])
            self.assertEqual([item.sequence_index for item in images], [0, 1, 2])
            self.assertTrue(all(item.width == 200 and item.height == 100 for item in images))
            self.assertTrue(all(item.file_size > 0 for item in images))

    def test_capture_time_normalizes_for_sorting(self) -> None:
        self.assertEqual(normalize_capture_time("2026:09:07 14:36:49"), "2026-09-07 14:36:49")

    def test_recursive_scan_skips_output_and_archive_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "合成"
            archive = root / "归档"
            nested = root / "拍摄批次"
            output.mkdir()
            archive.mkdir()
            nested.mkdir()
            (root / "source.JPG").write_bytes(_minimal_jpeg())
            (nested / "nested.JPG").write_bytes(_minimal_jpeg())
            (output / "stack.jpg").write_bytes(_minimal_jpeg())
            (archive / "archived.JPG").write_bytes(_minimal_jpeg())

            images = ImageScanner(
                root,
                ScannerConfig(recursive=True),
                excluded_roots=(output, archive),
            ).scan()

            self.assertEqual([item.filename for item in images], ["nested.JPG", "source.JPG"])


class PhotoSmokeTests(unittest.TestCase):
    def test_repository_photo_sample_is_metadata_only(self) -> None:
        root = Path(__file__).resolve().parents[1] / "photo"
        if not root.is_dir():
            self.skipTest("photo sample is not present")
        report = ImageScanner(root).scan_report()
        self.assertGreaterEqual(report.count, 1)
        self.assertTrue(all(item.width and item.height for item in report.images))
        self.assertEqual(report.total_bytes, sum(Path(item.original_path).stat().st_size for item in report.images))
        self.assertGreater(report.total_bytes, 0)


if __name__ == "__main__":
    unittest.main()
