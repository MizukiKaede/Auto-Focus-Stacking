"""Bounded concurrency regressions for flat-file archive publication."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from threading import Barrier, BrokenBarrierError, Lock
import unittest

from focus_stack_app.files.archiver import ArchiveMode, CollisionPolicy, FileArchiver


class _EntryBarrierArchiver(FileArchiver):
    """Release two callers together before entering the public archive path."""

    def __init__(self, *args, entry_barrier: Barrier, **kwargs):
        super().__init__(*args, **kwargs)
        self._entry_barrier = entry_barrier

    def archive_file(self, source, *, image_id=None):
        self._entry_barrier.wait(timeout=5)
        return super().archive_file(source, image_id=image_id)


class _ReserveBarrierArchiver(FileArchiver):
    """Coordinate the old two existence checks to expose replace races."""

    def __init__(self, *args, reserve_barriers: tuple[Barrier, Barrier], **kwargs):
        super().__init__(*args, **kwargs)
        self._reserve_barriers = reserve_barriers
        self._reserve_calls = 0
        self._reserve_calls_lock = Lock()

    def _reserve_destination(self, destination: Path) -> None:
        with self._reserve_calls_lock:
            call = self._reserve_calls
            self._reserve_calls += 1
        # The current implementation reserves once.  The two barriers also
        # coordinate the former second check, so this test fails against an
        # implementation that checks then calls os.replace from two threads.
        if call < len(self._reserve_barriers) * 2:
            phase = call // 2
            try:
                self._reserve_barriers[phase].wait(timeout=5)
            except BrokenBarrierError:
                # A fixed implementation may no longer perform phase 1.
                pass
        super()._reserve_destination(destination)


def _run_pair(first, second):
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="archive-test") as pool:
        futures = [pool.submit(first), pool.submit(second)]
        results = []
        for future in futures:
            results.append(future.result(timeout=10))
        return results


class ArchiveConcurrencyTests(unittest.TestCase):
    def _sources(self, root: Path) -> tuple[Path, Path, dict[str, bytes]]:
        first = root / "source-a" / "same.JPG"
        second = root / "source-b" / "same.JPG"
        first.parent.mkdir()
        second.parent.mkdir()
        first_bytes = b"source-a-contents\n" * 4096
        second_bytes = b"source-b-contents\n" * 4096
        first.write_bytes(first_bytes)
        second.write_bytes(second_bytes)
        return first, second, {str(first): first_bytes, str(second): second_bytes}

    def test_same_name_parallel_archive_never_overwrites_for_each_mode(self):
        """ERROR collision policy leaves one destination and one source intact."""

        for mode in (ArchiveMode.COPY, ArchiveMode.MOVE, ArchiveMode.HARDLINK):
            with self.subTest(mode=mode.value), tempfile.TemporaryDirectory() as raw_root:
                root = Path(raw_root)
                first, second, contents = self._sources(root)
                archive_dir = root / "archive"
                barrier = Barrier(2)
                archivers = [
                    _EntryBarrierArchiver(
                        archive_dir,
                        mode=mode,
                        collision_policy=CollisionPolicy.ERROR,
                        entry_barrier=barrier,
                    )
                    for _ in range(2)
                ]

                records = _run_pair(
                    lambda: archivers[0].archive_file(first),
                    lambda: archivers[1].archive_file(second),
                )

                self.assertEqual(sorted(record.status for record in records), ["archived", "conflict"])
                archived = next(record for record in records if record.status == "archived")
                conflict = next(record for record in records if record.status == "conflict")
                destination = archive_dir / "same.JPG"
                self.assertTrue(destination.is_file())
                self.assertEqual(destination.read_bytes(), contents[archived.source_path])
                self.assertEqual([path.name for path in archive_dir.iterdir()], ["same.JPG"])
                self.assertTrue(Path(conflict.source_path).exists())
                if mode is ArchiveMode.MOVE:
                    self.assertFalse(Path(archived.source_path).exists())
                else:
                    self.assertTrue(Path(archived.source_path).exists())

    def test_parallel_rename_policy_allocates_distinct_flat_names(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            first, second, contents = self._sources(root)
            archive_dir = root / "archive"
            barrier = Barrier(2)
            archivers = [
                _EntryBarrierArchiver(
                    archive_dir,
                    mode=ArchiveMode.COPY,
                    collision_policy=CollisionPolicy.RENAME,
                    entry_barrier=barrier,
                )
                for _ in range(2)
            ]

            records = _run_pair(
                lambda: archivers[0].archive_file(first),
                lambda: archivers[1].archive_file(second),
            )

            self.assertEqual([record.status for record in records], ["archived", "archived"])
            self.assertEqual(
                sorted(path.name for path in archive_dir.iterdir()),
                ["same (1).JPG", "same.JPG"],
            )
            for record in records:
                self.assertEqual((archive_dir / Path(record.destination_path).name).read_bytes(), contents[record.source_path])
                self.assertTrue(Path(record.source_path).exists())

    def test_copy_commit_is_exclusive_when_low_level_calls_race(self):
        """The final publish cannot let os.replace clobber the first copy."""

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            first, second, contents = self._sources(root)
            archive_dir = root / "archive"
            destination = archive_dir / "same.JPG"
            archiver = _ReserveBarrierArchiver(
                archive_dir,
                mode=ArchiveMode.COPY,
                reserve_barriers=(Barrier(2), Barrier(2)),
            )

            def publish(source: Path):
                archiver._copy_no_overwrite(source, destination)
                return "ok"

            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="archive-commit-test") as pool:
                futures = [pool.submit(publish, first), pool.submit(publish, second)]
                outcomes = []
                for future in futures:
                    try:
                        outcomes.append(future.result(timeout=10))
                    except BaseException as exc:  # keep both worker outcomes for the assertion
                        outcomes.append(exc)

            self.assertEqual(sum(outcome == "ok" for outcome in outcomes), 1)
            errors = [outcome for outcome in outcomes if outcome != "ok"]
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], FileExistsError)
            self.assertIn(destination.read_bytes(), contents.values())
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertEqual(list(archive_dir.glob("*.partial")), [ ])


if __name__ == "__main__":
    unittest.main()
