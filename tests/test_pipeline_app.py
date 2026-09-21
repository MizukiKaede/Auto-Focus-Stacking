"""Headless smoke tests for archive, mock Hugin and bounded pipeline layers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest

from focus_stack_app.files.archiver import ArchiveMode, FileArchiver, safe_copy
from focus_stack_app.hugin.align import AlignConfig, AlignImageStack
from focus_stack_app.hugin.hugin_locator import HuginLocator, HuginToolNotFound
from focus_stack_app.hugin.output_encoder import OutputConfig, OutputFormat, output_path_for
from focus_stack_app.config import RuntimeConfig
from focus_stack_app.pipeline.coordinator import PipelineConfig, PipelineCoordinator
from focus_stack_app.pipeline.events import PipelineStage
from focus_stack_app.pipeline.merge_worker import StackMergeService
from focus_stack_app.pipeline.analysis_worker import AnalysisJob
from focus_stack_app.storage.manifest import ManifestWriter
from focus_stack_app.ui.progress_panel import ProgressSnapshot


class _MockCommandRunner:
    def __init__(self, *, returncode: int = 0):
        self.returncode = returncode
        self.commands: list[list[str]] = []

    def run(self, command, **kwargs):
        self.commands.append(list(command))
        work = Path(kwargs["cwd"])
        prefix = Path(command[command.index("-a") + 1])
        prefix = prefix if prefix.is_absolute() else work / prefix
        if self.returncode == 0:
            (prefix.parent / f"{prefix.name}0000.tif").write_bytes(b"aligned")
            (prefix.parent / f"{prefix.name}0001.tif").write_bytes(b"aligned")
        from focus_stack_app.hugin.process import CommandResult

        return CommandResult(tuple(command), self.returncode, stderr="mock failure" if self.returncode else "")


class ArchiveAndHuginTests(unittest.TestCase):
    def test_flat_archive_preserves_copy_and_conflict(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source" / "IMG0001.JPG"
            source.parent.mkdir()
            source.write_bytes(b"jpeg bytes")
            result = FileArchiver(root / "archive", ArchiveMode.COPY).archive_file(source)
            self.assertEqual(result.status, "archived")
            self.assertTrue(source.exists())
            self.assertTrue((root / "archive" / source.name).exists())
            conflict = FileArchiver(root / "archive", ArchiveMode.COPY).archive_file(source)
            self.assertEqual(conflict.status, "conflict")
            self.assertEqual((root / "archive" / source.name).read_bytes(), b"jpeg bytes")

    def test_batch_missing_source_is_persisted_as_failed_operation(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)

            class Repository:
                def __init__(self):
                    self.events = []

                def record_archive_operation(self, payload):
                    self.events.append(payload)

            repository = Repository()
            missing = root / "missing.JPG"
            result = FileArchiver(root / "archive", repository=repository).archive_files([missing])
            self.assertEqual(len(result.failed), 1)
            self.assertTrue(any(event.get("phase") == "failed" for event in repository.events))

    def test_mock_align_builds_focus_stack_command_and_missing_tool_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            inputs = [root / f"a{i}.jpg" for i in range(2)]
            for path in inputs:
                path.write_bytes(b"input")
            runner = _MockCommandRunner()
            aligner = AlignImageStack(executable=root / "align_image_stack.exe", runner=runner, config=AlignConfig(timeout_seconds=2))
            result = aligner.align(inputs, work_dir=root / "tmp")
            self.assertTrue(result.ok)
            self.assertIn("-m", runner.commands[0])
            self.assertEqual(len(result.aligned_paths), 2)
        with self.assertRaises(HuginToolNotFound) as raised:
            HuginLocator(environ={}).require("enfuse")
        self.assertIn("enfuse", str(raised.exception))

    def test_output_name_and_manifest_are_stable(self):
        self.assertEqual(output_path_for("DSC08421.JPG", "out"), Path("out/DSC08421_stack.jpg"))
        self.assertEqual(output_path_for("DSC08421.JPG", "out", OutputFormat.TIFF), Path("out/DSC08421.tif"))
        with tempfile.TemporaryDirectory() as root:
            path = ManifestWriter(Path(root) / "stack_manifest.csv").write_csv(
                [{"group_id": 1, "first_image": "a.JPG", "image": "a.JPG", "selected": True, "status": "DONE"}]
            )
            text = path.read_text(encoding="utf-8-sig")
            self.assertIn("group_id,first_image", text)
            self.assertIn("true", text)

    def test_runtime_hugin_paths_are_propagated_to_default_service(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            align_path = root / "align_image_stack.exe"
            enfuse_path = root / "enfuse.exe"
            runtime = RuntimeConfig(
                hugin_bin=str(root),
                align_image_stack_path=str(align_path),
                enfuse_path=str(enfuse_path),
            )
            service = StackMergeService(root / "output", runtime_config=runtime)
            self.assertEqual(service.aligner.executable, align_path)
            self.assertEqual(service.enfuser.executable, enfuse_path)

    def test_successful_stack_is_written_before_complete_group_is_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "photo"
            output_dir = root / "合成"
            archive_dir = root / "归档"
            source_dir.mkdir()
            sources = [source_dir / f"IMG{i}.JPG" for i in range(4)]
            for source in sources:
                source.write_bytes(source.name.encode())

            sequence = []

            class Aligner:
                def align(self, paths, *, work_dir, **kwargs):
                    values = tuple(Path(path) for path in paths)
                    self_paths_exist = all(path.is_file() for path in values)
                    if not self_paths_exist:
                        raise AssertionError("sources must exist while alignment runs")
                    sequence.append("align")
                    aligned = []
                    for index, _ in enumerate(values):
                        path = Path(work_dir) / f"aligned-{index}.tif"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b"aligned")
                        aligned.append(path)
                    return SimpleNamespace(aligned_paths=tuple(aligned))

            class Enfuser:
                def fuse(self, paths, output_path, **kwargs):
                    self_output = Path(output_path)
                    if not all(source.is_file() for source in sources):
                        raise AssertionError("sources must exist while fusion runs")
                    sequence.append("fuse")
                    self_output.parent.mkdir(parents=True, exist_ok=True)
                    self_output.write_bytes(b"stack")
                    return SimpleNamespace(output_path=self_output)

            class TrackingArchiver(FileArchiver):
                def archive_files(self, values):
                    if not (output_dir / "IMG0_stack.jpg").is_file():
                        raise AssertionError("composite must exist before archiving")
                    sequence.append("archive")
                    return super().archive_files(values)

            service = StackMergeService(
                output_dir,
                archive_dir=archive_dir,
                archiver=TrackingArchiver(archive_dir, ArchiveMode.MOVE),
                aligner=Aligner(),
                enfuser=Enfuser(),
            )
            group = {"id": 7, "images": [{"path": source} for source in sources]}
            analysis = {
                "all_images": group["images"],
                "selected_paths": [sources[0], sources[2]],
                "first_original_image": group["images"][0],
                "needs_merge": True,
            }

            result = service.process(AnalysisJob(group=group, analysis=analysis))

            self.assertEqual(result.status, "DONE")
            self.assertIsNotNone(result.work_dir)
            self.assertFalse(result.work_dir.exists())
            self.assertEqual(sequence, ["align", "fuse", "archive"])
            self.assertEqual(
                [path.name for path in output_dir.glob("*.jpg")],
                ["IMG0_stack.jpg"],
            )
            self.assertTrue(all(not source.exists() for source in sources))
            self.assertEqual(
                sorted(path.name for path in archive_dir.glob("*.JPG")),
                sorted(path.name for path in sources),
            )

    def test_successful_stack_preserves_sources_when_archival_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "photo"
            output_dir = root / "合成"
            archive_dir = root / "归档"
            source_dir.mkdir()
            sources = [source_dir / f"IMG{i}.JPG" for i in range(4)]
            for source in sources:
                source.write_bytes(source.name.encode())

            class Aligner:
                def align(self, paths, *, work_dir, **kwargs):
                    aligned = []
                    for index, _ in enumerate(paths):
                        path = Path(work_dir) / f"aligned-{index}.tif"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b"aligned")
                        aligned.append(path)
                    return SimpleNamespace(aligned_paths=tuple(aligned))

            class Enfuser:
                def fuse(self, paths, output_path, **kwargs):
                    output = Path(output_path)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"stack")
                    return SimpleNamespace(output_path=output)

            class MustNotArchive:
                def archive_files(self, values):
                    raise AssertionError("archiving must be skipped when disabled")

            service = StackMergeService(
                output_dir,
                archive_dir=archive_dir,
                archiver=MustNotArchive(),
                archive_enabled=False,
                aligner=Aligner(),
                enfuser=Enfuser(),
            )
            group = {"id": 8, "images": [{"path": source} for source in sources]}
            analysis = {
                "all_images": group["images"],
                "selected_paths": sources,
                "first_original_image": group["images"][0],
                "needs_merge": True,
            }

            result = service.process(AnalysisJob(group=group, analysis=analysis))

            self.assertEqual(result.status, "DONE")
            self.assertIsNotNone(result.work_dir)
            self.assertFalse(result.work_dir.exists())
            self.assertIsNone(result.archive_result)
            self.assertTrue(all(source.is_file() for source in sources))
            self.assertFalse(archive_dir.exists())
            self.assertTrue((output_dir / "IMG0_stack.jpg").is_file())

    def test_failed_stack_retains_private_work_directory_for_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "photo"
            output_dir = root / "output"
            source_dir.mkdir()
            sources = [source_dir / f"IMG{i}.JPG" for i in range(4)]
            for source in sources:
                source.write_bytes(b"source")

            class FailingAligner:
                def align(self, paths, *, work_dir, **kwargs):
                    work = Path(work_dir)
                    work.mkdir(parents=True, exist_ok=True)
                    (work / "alignment-diagnostic.txt").write_text("failed", encoding="utf-8")
                    raise RuntimeError("alignment failed")

            class MustNotFuse:
                def fuse(self, *args, **kwargs):
                    raise AssertionError("fusion must not run after alignment failure")

            service = StackMergeService(
                output_dir,
                archive_enabled=False,
                aligner=FailingAligner(),
                enfuser=MustNotFuse(),
            )
            group = {"id": 9, "images": [{"path": source} for source in sources]}
            analysis = {
                "all_images": group["images"],
                "selected_paths": sources,
                "first_original_image": group["images"][0],
                "needs_merge": True,
            }

            with self.assertRaisesRegex(RuntimeError, "alignment failed"):
                service.process(AnalysisJob(group=group, analysis=analysis))

            work_dirs = list((output_dir / ".stack_cache" / "temp").iterdir())
            self.assertEqual(len(work_dirs), 1)
            self.assertTrue(list(work_dirs[0].rglob("alignment-diagnostic.txt")))

    def test_fusion_failure_leaves_source_and_archive_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "photo"
            source_dir.mkdir()
            sources = [source_dir / f"IMG{i}.JPG" for i in range(4)]
            for source in sources:
                source.write_bytes(source.name.encode())

            class Aligner:
                def align(self, paths, *, work_dir, **kwargs):
                    return SimpleNamespace(aligned_paths=tuple(Path(path) for path in paths))

            class BrokenEnfuser:
                def fuse(self, *args, **kwargs):
                    raise RuntimeError("fusion failed")

            output_dir = root / "合成"
            archive_dir = root / "归档"
            service = StackMergeService(
                output_dir,
                archive_dir=archive_dir,
                archiver=FileArchiver(archive_dir, ArchiveMode.MOVE),
                aligner=Aligner(),
                enfuser=BrokenEnfuser(),
            )
            group = {"id": 9, "images": [{"path": source} for source in sources]}
            analysis = {
                "all_images": group["images"],
                "selected_paths": list(sources),
                "first_original_image": group["images"][0],
                "needs_merge": True,
            }

            with self.assertRaisesRegex(RuntimeError, "fusion failed"):
                service.process(AnalysisJob(group=group, analysis=analysis))

            self.assertTrue(all(source.is_file() for source in sources))
            self.assertEqual(list(archive_dir.glob("*.JPG")), [])
            self.assertEqual(list(output_dir.glob("*_stack.jpg")), [])

    def test_manifest_result_has_one_row_per_archived_source(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            sources = [root / f"IMG{i}.JPG" for i in range(3)]
            for source in sources:
                source.write_bytes(source.name.encode())

            class Record:
                def __init__(self, source):
                    self.source_path = str(source)
                    self.destination_path = str(root / "archive" / source.name)
                    self.status = "archived"

            class Archive:
                records = [Record(source) for source in sources]

            group = {"id": 7, "images": [{"path": source} for source in sources]}
            analysis = {
                "selected_indices": [0, 2],
                "reasons": {0: "coverage", 2: "coverage"},
                "gains": {0: 0.75, 2: 0.22},
                "coverage": 0.97,
            }
            from focus_stack_app.pipeline.merge_worker import MergeResult

            result = MergeResult(
                group,
                "DONE",
                root / "archive" / "IMG0_stack.jpg",
                archive_result=Archive(),
                analysis=analysis,
                all_paths=tuple(sources),
                selected_source_paths=(sources[0], sources[2]),
                first_original=sources[0],
            )
            writer = ManifestWriter(root / "stack_manifest.csv")
            writer.update_result(result)
            rows = (root / "stack_manifest.csv").read_text(encoding="utf-8-sig").splitlines()
            self.assertEqual(len(rows), 4)  # header + all three image records
            self.assertEqual(sum(",true," in row for row in rows[1:]), 2)

    def test_analysis_failure_leaves_all_sources_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source_dir = root / "source"
            source_dir.mkdir()
            sources = [source_dir / f"IMG{i}.JPG" for i in range(3)]
            for source in sources:
                source.write_bytes(source.name.encode())

            class MustNotRun:
                def align(self, *args, **kwargs):
                    raise AssertionError("alignment must not run after analysis failure")

                def fuse(self, *args, **kwargs):
                    raise AssertionError("fusion must not run after analysis failure")

            group = {"id": 8, "images": [{"path": source} for source in sources]}
            output_dir = root / "output"
            archive_dir = root / "archive"
            service = StackMergeService(
                output_dir,
                archive_dir=archive_dir,
                aligner=MustNotRun(),
                enfuser=MustNotRun(),
            )
            manifest = ManifestWriter(root / "stack_manifest.csv")
            coordinator = PipelineCoordinator(
                config=PipelineConfig(queue_size=2, parallel=False),
                analyzer=lambda group: (_ for _ in ()).throw(ValueError("classification failed")),
                merger=service,
                manifest_writer=manifest,
            )
            summary = coordinator.run([group])
            self.assertEqual(summary.errors, 1)
            self.assertEqual(summary.merge_completed, 0)
            self.assertTrue(all(source.exists() for source in sources))
            self.assertEqual(list(archive_dir.glob("*.JPG")), [])
            self.assertEqual(list(output_dir.glob("*_stack.jpg")), [])
            text = (root / "stack_manifest.csv").read_text(encoding="utf-8-sig")
            self.assertEqual(len(text.splitlines()), 4)
            self.assertIn("FAILED_CLASSIFICATION", text)


class PipelineSmokeTests(unittest.TestCase):
    def test_serial_bounded_queue_does_not_deadlock(self):
        """Serial mode must drain a queue of eight groups with capacity two."""

        groups = [{"id": i} for i in range(8)]
        coordinator = PipelineCoordinator(
            config=PipelineConfig(queue_size=2, parallel=False),
            analyzer=lambda group: group,
            merger=lambda job: job.group["id"],
        )
        coordinator.start(groups)
        summary = coordinator.wait(5)
        if summary is None:
            # Keep a regression failure bounded even if a future queue change
            # reintroduces the producer-join deadlock.
            coordinator.cancel()
            coordinator.wait(2)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary.analysis_completed, 8)
        self.assertEqual(summary.merge_completed, 8)
        self.assertFalse(summary.cancelled)

    def test_analysis_statuses_and_groups_found_reflect_selection(self):
        groups = [
            {"id": 1, "images": ["a", "b"]},
            {"id": 2, "images": ["a"]},
            {"id": 3, "images": ["a", "b"]},
        ]
        states = []
        jobs = []

        class Repository:
            def set_group_status(self, group_id, status, **values):
                states.append((group_id, status))

        def analyze(group):
            if group["id"] == 1:
                return {"status": "NO_MERGE_REPEATED", "selected_indices": []}
            if group["id"] == 2:
                return {"status": "NO_MERGE_SINGLE", "selected_indices": [0]}
            return {"status": "READY_FOR_MERGE", "needs_merge": True, "selected_indices": [0, 1]}

        def merge(job):
            jobs.append(job)
            return job.group["id"]

        events = []
        summary = PipelineCoordinator(
            config=PipelineConfig(queue_size=2, parallel=True),
            analyzer=analyze,
            merger=merge,
            repository=Repository(),
            event_callback=events.append,
        ).run(groups)
        self.assertEqual(summary.analysis_completed, 3)
        self.assertEqual(summary.merge_completed, 3)
        self.assertEqual([job.archive_only for job in jobs], [True, True, False])
        self.assertIn((1, "CLASSIFIED"), states)
        self.assertIn((2, "CLASSIFIED"), states)
        self.assertIn((3, "SELECTED"), states)
        self.assertEqual(events[-1].groups_found, 3)

    def test_bounded_pipeline_continues_after_one_failure(self):
        events = []
        groups = [{"id": i} for i in range(5)]

        def analyze(group):
            if group["id"] == 2:
                raise ValueError("bad group")
            return group

        coordinator = PipelineCoordinator(
            config=PipelineConfig(queue_size=2, parallel=True),
            analyzer=analyze,
            merger=lambda job: job.group["id"],
            event_callback=events.append,
        )
        summary = coordinator.run(groups)
        self.assertEqual(summary.analysis_completed, 5)
        self.assertEqual(summary.merge_completed, 4)
        self.assertEqual(summary.errors, 1)
        self.assertEqual(summary.results, [0, 1, 3, 4])
        self.assertEqual(events[-1].stage, PipelineStage.COMPLETE)

    def test_cancel_does_not_leave_controller_running(self):
        groups = [{"id": i} for i in range(20)]

        def analyze(group, cancel_event=None):
            time.sleep(0.01)
            return group

        coordinator = PipelineCoordinator(analyzer=analyze, merger=lambda job: job.group)
        coordinator.start(groups)
        time.sleep(0.03)
        coordinator.cancel()
        summary = coordinator.wait(5)
        self.assertIsNotNone(summary)
        self.assertTrue(summary.cancelled)
        self.assertFalse(coordinator.running)

    def test_progress_snapshot_is_headless_ui_smoke(self):
        snapshot = ProgressSnapshot.from_event(
            {
                "stage": "fusion",
                "analysis_completed": 3,
                "analysis_total": 4,
                "merge_completed": 1,
                "merge_total": 4,
                "current_file": "IMG0003.JPG",
                "groups_found": 4,
            }
        )
        self.assertEqual(snapshot.analysis, 75)
        self.assertEqual(snapshot.merge, 25)
        self.assertEqual(snapshot.overall, 50)
        self.assertEqual(snapshot.current_file, "IMG0003.JPG")


if __name__ == "__main__":
    unittest.main()
