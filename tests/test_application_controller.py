"""Headless end-to-end checks for the application/controller wiring."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from focus_stack_app.app import default_controller_factory
from focus_stack_app.files.archiver import ArchiveMode, FileArchiver
from focus_stack_app.pipeline.application_controller import (
    ApplicationController,
    ApplicationOptions,
)
from focus_stack_app.pipeline.merge_worker import StackMergeService
from focus_stack_app.pipeline.events import PipelineStage
from focus_stack_app.core.scanner import ScanReport
from focus_stack_app.core.types import SceneGroup
from focus_stack_app.storage.models import ImageRecord


class _Scanner:
    def __init__(self, records):
        self.records = records

    def scan_report(self):
        return ScanReport(root=str(Path(self.records[0].path).parent), images=list(self.records))


class _Detector:
    def detect(self, records):
        return [SceneGroup(99, list(records), start_index=0, end_index=len(records) - 1, confidence=0.9)]


class _BrokenDetector:
    def detect(self, records):
        raise ValueError("preview decode failed")


class _Analyzer:
    def analyze_group(self, group, **kwargs):
        values = list(group.items)
        return {
            "group_id": group.group_id,
            "all_images": values,
            "selected_paths": [values[0].path],
            "selected_indices": [0],
            "first_original_image": values[0],
            "selected_count": 1,
            "image_count": len(values),
            "status": "NO_MERGE_SINGLE",
            "needs_merge": False,
        }


class _MergeAnalyzer:
    def analyze_group(self, group, **kwargs):
        values = list(group.items)
        return {
            "group_id": group.group_id,
            "all_images": values,
            "selected_paths": [value.path for value in values],
            "selected_indices": list(range(len(values))),
            "first_original_image": values[0],
            "selected_count": len(values),
            "image_count": len(values),
            "status": "READY_FOR_MERGE",
            "needs_merge": True,
        }


class _MustNotRun:
    def align(self, *args, **kwargs):
        raise AssertionError("Hugin alignment must not run for needs_merge=False")

    def fuse(self, *args, **kwargs):
        raise AssertionError("Enfuse must not run for needs_merge=False")


class _RepositoryProxy:
    """Resolve the controller's DB lazily for an injected archiver."""

    def __init__(self, controller):
        self.controller = controller

    def __getattr__(self, name):
        return getattr(self.controller.database, name)


class ApplicationControllerTests(unittest.TestCase):
    def _make_controller(self, root: Path, *, count: int = 1):
        source = root / "source"
        output = root / "output"
        source.mkdir()
        records = []
        for index in range(count):
            path = source / f"IMG{index}.JPG"
            path.write_bytes(f"image-{index}".encode())
            records.append(ImageRecord.from_path(path, file_size=path.stat().st_size))
        events = []
        options = ApplicationOptions(
            source_dir=source,
            output_dir=output,
            archive_mode=ArchiveMode.COPY,
            preserve_cache=True,
            align_image_stack_path=root / "missing-align.exe",
            enfuse_path=root / "missing-enfuse.exe",
        )
        controller = ApplicationController(
            options,
            scanner=_Scanner(records),
            scene_detector=_Detector(),
            analyzer=_Analyzer(),
            event_callback=events.append,
        )
        archiver = FileArchiver(
            output,
            ArchiveMode.COPY,
            repository=_RepositoryProxy(controller),
        )
        controller._merge_service = StackMergeService(
            output,
            archiver=archiver,
            aligner=_MustNotRun(),
            enfuser=_MustNotRun(),
        )
        return controller, source, output, events, records

    def test_headless_controller_leaves_single_group_in_source_without_hugin(self):
        with tempfile.TemporaryDirectory() as directory:
            controller, source, output, events, records = self._make_controller(Path(directory))
            try:
                controller.start()
                summary = controller.wait(5)
                self.assertIsNotNone(summary)
                assert summary is not None
                self.assertEqual(summary.groups_total, 1)
                self.assertEqual(summary.merge_completed, 1)
                self.assertEqual(summary.results[0].status, "NO_MERGE")
                self.assertTrue(all(path.is_file() for path in source.glob("*.JPG")))
                self.assertFalse((output / records[0].filename).exists())
                self.assertTrue((output / "stack_manifest.csv").is_file())
                self.assertTrue((output / ".stack_cache" / "database.sqlite").is_file())
                images = controller.database.list_images()
                self.assertEqual(len(images), 1)
                self.assertEqual(images[0].group_id, 1)
                self.assertFalse(any(event.stage == PipelineStage.ERROR for event in events))
            finally:
                controller.close()

    def test_default_mode_no_merge_does_not_require_hugin(self):
        with tempfile.TemporaryDirectory() as directory:
            controller, source, output, events, _ = self._make_controller(Path(directory), count=2)
            try:
                controller.start()
                summary = controller.wait(5)
                self.assertIsNotNone(summary)
                assert summary is not None
                self.assertEqual(summary.results[0].status, "NO_MERGE")
                # The default whole-frame mode does not need external tools.
                self.assertEqual(summary.errors, 0)
                self.assertFalse(any("Hugin" in event.message for event in events))
                self.assertEqual(list(output.glob("IMG*.JPG")), [])
                self.assertTrue(all(path.is_file() for path in source.glob("IMG*.JPG")))
                self.assertEqual(
                    len((output / "stack_manifest.csv").read_text(encoding="utf-8-sig").splitlines()),
                    3,
                )
            finally:
                controller.close()

    def test_missing_hugin_leaves_sources_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            records = []
            for index in range(4):
                path = source / f"STACK{index}.JPG"
                path.write_bytes(f"image-{index}".encode())
                records.append(ImageRecord.from_path(path, file_size=path.stat().st_size))
            controller = ApplicationController(
                ApplicationOptions(
                    source_dir=source,
                    output_dir=output,
                    fusion_backend="hugin_enfuse",
                    align_image_stack_path=root / "missing-align.exe",
                    enfuse_path=root / "missing-enfuse.exe",
                ),
                scanner=_Scanner(records),
                scene_detector=_Detector(),
                analyzer=_MergeAnalyzer(),
            )
            try:
                summary = controller.run(timeout=5)
                self.assertTrue(summary.failed_merge)
                self.assertTrue(all((source / record.filename).is_file() for record in records))
                self.assertTrue(all(not (output / record.filename).exists() for record in records))
                self.assertEqual(list((root / "归档").glob("*.JPG")), [])
                self.assertEqual(
                    len((output / "stack_manifest.csv").read_text(encoding="utf-8-sig").splitlines()),
                    5,
                )
                self.assertTrue(any("Hugin" in message for message in summary.diagnostics))
            finally:
                controller.close()

    def test_default_factory_is_headless_constructible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = default_controller_factory(source_dir=root, output_dir=root / "out")
            self.assertIsInstance(controller, ApplicationController)
            self.assertEqual(controller.options.output_format, "jpg")
            self.assertFalse(controller.options.archive_enabled)
            self.assertEqual(controller.options.grouping_pause_seconds, 20)
            self.assertEqual(controller.archive_dir, root / "归档")
            controller.close()

    def test_default_output_and_archive_subdirectories_are_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            controller = default_controller_factory(source_dir=source, output_dir=source / "合成")
            try:
                source_path, output_path = controller._validate_paths()
                self.assertEqual(source_path, source.resolve())
                self.assertEqual(output_path, (source / "合成").resolve())
                self.assertEqual(controller.archive_dir, (source / "归档").resolve())
            finally:
                controller.close()

    def test_grouping_pause_setting_reaches_automatic_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = default_controller_factory(source_dir=root, output_dir=root / 'out',
                                                    grouping_pause_seconds=12)
            try:
                self.assertEqual(controller._automatic_scene_groups([]), [])
                self.assertEqual(controller._scene_detector.config.pause_seconds, 12)
            finally:
                controller.close()

    def test_grouping_plan_cache_hits_and_order_change_invalidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            records = []
            for index in range(3):
                path = source / f"IMG{index}.JPG"
                path.write_bytes(f"pixels-{index}".encode())
                records.append(ImageRecord.from_path(path, sequence_index=index))

            class CountingDetector:
                def __init__(self):
                    self.calls = 0

                def detect(self, values):
                    self.calls += 1
                    return [SceneGroup(1, list(values), start_index=0,
                                       end_index=len(values) - 1, confidence=0.9)]

            detector = CountingDetector()
            controller = ApplicationController(
                ApplicationOptions(source_dir=source, output_dir=output, preserve_cache=True),
                scene_detector=detector,
            )
            try:
                source_path, output_path = controller._validate_paths()
                controller._prepare_runtime(source_path, output_path)
                first = controller._scene_groups(records)
                second = controller._scene_groups(records)
                reordered = controller._scene_groups(list(reversed(records)))
                controller.options.grouping_pause_seconds += 1
                controller._scene_groups(records)
                manual = source / "stack_groups.json"
                manual.write_text(
                    '{"groups":[["IMG0.JPG","IMG2.JPG"]]}', encoding="utf-8",
                )
                manual_first = controller._scene_groups(records)
                manual_second = controller._scene_groups(records)
                self.assertEqual(detector.calls, 4)
                self.assertEqual(len(manual_first), 2)
                self.assertEqual(
                    [[item.path for item in group.items] for group in manual_first],
                    [[item.path for item in group.items] for group in manual_second],
                )
                self.assertEqual([item.path for item in first[0].items],
                                 [item.path for item in second[0].items])
                self.assertEqual(reordered[0].items[0].path, records[-1].path)
            finally:
                controller.close()

    def test_preserve_cache_false_disables_grouping_plan_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            path = source / "IMG0.JPG"
            path.write_bytes(b"pixels")
            records = [ImageRecord.from_path(path, sequence_index=0)]

            class CountingDetector:
                def __init__(self):
                    self.calls = 0

                def detect(self, values):
                    self.calls += 1
                    return [SceneGroup(1, list(values), start_index=0, end_index=0)]

            detector = CountingDetector()
            controller = ApplicationController(
                ApplicationOptions(source_dir=source, output_dir=output, preserve_cache=False),
                scene_detector=detector,
            )
            try:
                source_path, output_path = controller._validate_paths()
                controller._prepare_runtime(source_path, output_path)
                controller._scene_groups(records)
                controller._scene_groups(records)
                self.assertEqual(detector.calls, 2)
                self.assertIsNone(controller._build_analyzer().plan_cache)
            finally:
                controller.close()

    def test_invalid_grouping_pause_is_rejected(self):
        for value in (0, -1, 3601):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ApplicationOptions(source_dir='source', output_dir='out', grouping_pause_seconds=value)

    def test_controller_keeps_composite_and_archive_directories_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "photo"
            output = root / "合成"
            archive = root / "归档"
            source.mkdir()
            controller = default_controller_factory(
                source_dir=source,
                output_dir=output,
                archive_dir=archive,
            )
            try:
                controller._validate_paths()
                merger = controller._build_merger()
                self.assertEqual(merger.output_dir.resolve(), output.resolve())
                self.assertFalse(merger.archive_enabled)
                self.assertIsNone(merger.archiver)
                self.assertEqual(merger.archive_dir.resolve(), archive.resolve())
            finally:
                controller.close()

    def test_scene_preview_failure_falls_back_to_archive_only_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            controller, source, output, events, _ = self._make_controller(Path(directory), count=2)
            controller._scene_detector = _BrokenDetector()
            try:
                summary = controller.run(timeout=5)
                self.assertEqual(summary.groups_total, 2)
                self.assertEqual(summary.merge_completed, 2)
                self.assertEqual(len(summary.diagnostics), 1)
                self.assertEqual(list(output.glob("IMG*.JPG")), [])
                self.assertTrue(all(path.is_file() for path in source.glob("IMG*.JPG")))
                self.assertTrue(any(event.stage == PipelineStage.ERROR for event in events))
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()

