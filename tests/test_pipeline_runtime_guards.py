"""Regression tests for runtime memory, temp and completion safety."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from focus_stack_app.hugin.hugin_locator import HuginToolNotFound
from focus_stack_app.pipeline.application_controller import (
    cleanup_completed_temp_dirs, cleanup_stale_temp_dirs,
)
from focus_stack_app.pipeline.coordinator import PipelineCoordinator, PipelineConfig, PipelineSummary
from focus_stack_app.pipeline.memory_guard import MemoryGuard
from focus_stack_app.utils.memory import MemorySnapshot


class RuntimeGuardTests(unittest.TestCase):
    def test_memory_guard_waits_then_allows(self):
        low = MemorySnapshot(total_bytes=1_000, available_bytes=50, used_bytes=950)
        high = MemorySnapshot(total_bytes=1_000, available_bytes=200, used_bytes=800)
        snapshots = iter((low, high))
        events = []
        guard = MemoryGuard(
            minimum_bytes=100,
            minimum_fraction=0.10,
            snapshot_fn=lambda: next(snapshots),
            poll_interval=0.001,
            notice_interval=0,
            event_callback=events.append,
        )
        self.assertTrue(guard.wait(threading.Event(), current_group=7))
        self.assertEqual(len(events), 1)
        self.assertIn("暂停", events[0].message)

    def test_memory_guard_is_cancellable_while_pressure_persists(self):
        low = MemorySnapshot(total_bytes=1_000, available_bytes=50, used_bytes=950)
        cancel = threading.Event()
        result: list[bool] = []
        guard = MemoryGuard(
            minimum_bytes=100,
            minimum_fraction=0.10,
            snapshot_fn=lambda: low,
            poll_interval=0.001,
            notice_interval=60,
        )
        thread = threading.Thread(target=lambda: result.append(guard.wait(cancel)))
        thread.start()
        time.sleep(0.02)
        cancel.set()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [False])

    def test_stale_temp_cleanup_is_scoped_and_preserves_recent_or_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / ("group_" + "a" * 32)
            recent = root / ("group_" + "b" * 32)
            marked = root / ("group_" + "c" * 32)
            unsafe = root / "not-a-merge-work-dir"
            for child in (old, recent, marked, unsafe):
                child.mkdir()
                (child / "payload.tmp").write_bytes(b"x")
            (marked / ".keep").touch()
            reference = time.time() + 7200
            # Keep ``recent`` recent relative to the injected clock while
            # ``old`` remains older than the one-hour retention threshold.
            os.utime(recent, (reference, reference))
            removed = cleanup_stale_temp_dirs(root, max_age_seconds=3600, now=reference)
            self.assertEqual(removed, [old.resolve()])
            self.assertFalse(old.exists())
            self.assertTrue(recent.exists())
            self.assertTrue(marked.exists())
            self.assertTrue(unsafe.exists())

    def test_completed_temp_cleanup_preserves_failed_cancelled_and_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            done = root / ("1_" + "a" * 32)
            failed = root / ("2_" + "b" * 32)
            cancelled = root / ("3_" + "c" * 32)
            marked = root / ("4_" + "d" * 32)
            for child in (done, failed, cancelled, marked):
                child.mkdir()
                (child / "aligned.tif").write_bytes(b"tiff")
            (marked / ".keep").touch()

            class Repository:
                statuses = {1: "DONE", 2: "FAILED", 3: "CANCELLED", 4: "DONE"}

                def get_group(self, group_id):
                    return {"id": group_id, "status": self.statuses[group_id]}

            removed = cleanup_completed_temp_dirs(root, Repository())
            self.assertEqual(removed, [done.resolve()])
            self.assertFalse(done.exists())
            self.assertTrue(failed.exists())
            self.assertTrue(cancelled.exists())
            self.assertTrue(marked.exists())

    def test_pipeline_finished_waits_for_archive_only_consumer(self):
        summary = PipelineSummary(groups_total=2, analysis_completed=2, merge_completed=1, merge_finished=1)
        self.assertFalse(summary.finished)
        summary.merge_finished = 2
        self.assertTrue(summary.finished)

    def test_analysis_failure_archive_only_counts_as_finished_but_not_success(self):
        groups = [{"id": 1, "images": ["one.jpg", "two.jpg"]}]
        coordinator = PipelineCoordinator(
            config=PipelineConfig(queue_size=1, parallel=False),
            analyzer=lambda group: (_ for _ in ()).throw(ValueError("bad analysis")),
            merger=lambda job: job.group,
        )
        summary = coordinator.run(groups)
        self.assertTrue(summary.finished)
        self.assertEqual(summary.merge_finished, 1)
        self.assertEqual(summary.merge_completed, 0)

    def test_missing_hugin_is_explicit_failed_result(self):
        published = []

        class Writer:
            def update_result(self, result):
                published.append(result)

        def missing_hugin(_job):
            raise HuginToolNotFound("enfuse")

        summary = PipelineCoordinator(
            analyzer=lambda group: {"needs_merge": True},
            merger=missing_hugin,
            manifest_writer=Writer(),
        ).run([{"id": 5}])
        self.assertEqual(summary.errors, 1)
        self.assertEqual(summary.failed_merge[0][1].__class__, HuginToolNotFound)
        self.assertEqual(published[0].status, "FAILED_HUGIN")
        self.assertFalse(published[0].ok)


if __name__ == "__main__":
    unittest.main()

