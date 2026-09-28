from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from focus_stack_app.storage.cache import DiskCache, PreviewCache
from focus_stack_app.utils.memory import MemorySnapshot, disk_space, has_memory_headroom


class CacheAndResourceTests(unittest.TestCase):
    def test_lru_is_bounded(self) -> None:
        cache: PreviewCache[bytes] = PreviewCache(2)
        cache.put("a", b"a")
        cache.put("b", b"b")
        cache.get("a")
        cache.put("c", b"c")
        self.assertEqual(cache.keys(), ("a", "c"))
        self.assertIsNone(cache.get("b"))

    def test_disk_cache_fingerprint_write_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "source.JPG"
            image.write_bytes(b"source")
            cache = DiskCache(directory, max_preview_items=1)
            path = cache.write_preview(image, b"preview")
            self.assertTrue(path.is_file())
            self.assertEqual(cache.read_preview(image), b"preview")
            self.assertTrue(str(path).startswith(str(Path(directory) / ".stack_cache" / "previews")))

    def test_memory_guard_and_disk_check(self) -> None:
        snapshot = MemorySnapshot(total_bytes=10_000, available_bytes=5_000, used_bytes=5_000)
        self.assertTrue(has_memory_headroom(minimum_bytes=1_000, minimum_fraction=0.1, snapshot=snapshot))
        self.assertFalse(has_memory_headroom(minimum_bytes=6_000, minimum_fraction=0.1, snapshot=snapshot))
        with tempfile.TemporaryDirectory() as directory:
            check = disk_space(directory, required_bytes=1)
            self.assertTrue(check.total_bytes >= check.free_bytes)
            self.assertTrue(check.required_bytes == 1)


if __name__ == "__main__":
    unittest.main()

