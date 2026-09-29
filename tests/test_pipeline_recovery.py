"""Crash/restart smoke tests for the durable application pipeline."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from focus_stack_app.config import AppConfig, OutputConfig, RuntimeConfig
from focus_stack_app.files.archiver import ArchiveMode, FileArchiver
from focus_stack_app.hugin.align import AlignImageStack
from focus_stack_app.hugin.enfuse import Enfuser, EnfuseConfig
from focus_stack_app.hugin.process import CommandResult
from focus_stack_app.pipeline.application_controller import ApplicationController, ApplicationOptions
from focus_stack_app.pipeline.merge_worker import StackMergeService
from focus_stack_app.pipeline.events import PipelineEvent, PipelineStage
from focus_stack_app.storage.database import Database
from focus_stack_app.storage.models import GroupRecord, ImageRecord, JobRecord
from focus_stack_app.ui.progress_panel import ProgressSnapshot


class _NoScan:
    def scan_report(self):
        raise AssertionError("restart must rebuild unfinished groups from the database")


class _NoAnalyze:
    def analyze_group(self, group, **kwargs):
        raise AssertionError("durable analysis rows should be used on restart")


class _RepositoryProxy:
    def __init__(self, controller: ApplicationController):
        self.controller = controller

    def __getattr__(self, name):
        return getattr(self.controller.database, name)


class _FakeAligner:
    def __init__(self):
        self.calls: list[tuple[Path, ...]] = []

    def align(self, paths, *, work_dir, cancel_event=None, **kwargs):
        values = tuple(Path(path) for path in paths)
        self.calls.append(values)
        return SimpleNamespace(aligned_paths=values)


class _FakeEnfuser:
    def __init__(self):
        self.calls: list[Path] = []

    def fuse(self, paths, output_path, *, work_dir, cancel_event=None, **kwargs):
        destination = Path(output_path)
        self.calls.append(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake fused jpeg")
        return SimpleNamespace(output_path=destination)


class _LoggingCommandRunner:
    """Deterministic command runner which emits the same files as Hugin."""

    def run(self, command, **kwargs):
        argv = list(command)
        work = Path(kwargs["cwd"])
        if "-a" in argv:
            prefix = Path(argv[argv.index("-a") + 1])
            prefix = prefix if prefix.is_absolute() else work / prefix
            (prefix.parent / f"{prefix.name}0000.tif").write_bytes(b"aligned")
            (prefix.parent / f"{prefix.name}0001.tif").write_bytes(b"aligned")
        if "-o" in argv:
            output = Path(argv[argv.index("-o") + 1])
            output = output if output.is_absolute() else work / output
            output.write_bytes(b"enfused")
        return CommandResult(tuple(str(value) for value in argv), 0, stdout="mock ok")


class PipelineRecoveryTests(unittest.TestCase):
    def test_move_done_group_analyzed_before_crash_resumes_hugin_from_archive(self):
        """A MOVE-completed source can resume after a crash before alignment."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            originals = []
            for index in range(4):
                path = source / f"STACK{index}.JPG"
                path.write_bytes((f"jpeg-{index}-" + "x" * 32).encode())
                originals.append(path)

            database_path = output / ".stack_cache" / "database.sqlite"
            database = Database(database_path)
            images = [
                ImageRecord.from_path(path, file_size=path.stat().st_size, sequence_index=index)
                for index, path in enumerate(originals)
            ]
            database.insert_images(images)
            group = GroupRecord(
                first_image_id=images[0].id,
                start_index=0,
                end_index=3,
                image_count=4,
                selected_count=4,
                status="SELECTED",
            )
            database.insert_group(group)
            assert group.id is not None
            for image in images:
                database.set_image_group(image.id or 0, group.id)

            # The archive completed and its journal rows are durable.  The
            # analysis/selection also completed, but Hugin has not started.
            archive = FileArchiver(output, ArchiveMode.MOVE, repository=database)
            archive_result = archive.archive_files(images)
            self.assertTrue(archive_result.ok)
            self.assertTrue(all(not path.exists() for path in originals))
            database.set_group_selection(group.id, [image.id or 0 for image in images])
            database.set_group_status(group.id, "SELECTED", selected_count=4)
            job = JobRecord(group_id=group.id, stage="FUSION", status="RUNNING", progress=0.65)
            database.create_job(job)
            database.close()

            options = ApplicationOptions(
                source_dir=source,
                output_dir=output,
                archive_mode=ArchiveMode.MOVE,
                parallel=False,
                queue_size=2,
                merge_workers=1,
            )
            config = AppConfig(
                output=OutputConfig(format="jpg", jpeg_quality=100, jpeg_subsampling=0),
                runtime=RuntimeConfig(
                    disk_safety_margin_bytes=0,
                    min_available_memory_bytes=0,
                    min_available_memory_fraction=0,
                ),
            )
            controller = ApplicationController(
                options,
                config=config,
                scanner=_NoScan(),
                analyzer=_NoAnalyze(),
            )
            proxy = _RepositoryProxy(controller)
            aligner = _FakeAligner()
            enfuser = _FakeEnfuser()
            controller._merge_service = StackMergeService(
                output,
                archiver=FileArchiver(output, ArchiveMode.MOVE, repository=proxy),
                aligner=aligner,
                enfuser=enfuser,
                fusion_backend="hugin_enfuse",
                repository=proxy,
            )
            try:
                summary = controller.run(timeout=5)
                self.assertFalse(summary.cancelled)
                self.assertEqual(summary.groups_total, 1)
                self.assertEqual(
                    summary.merge_finished,
                    1,
                    f"diagnostics={summary.diagnostics!r}, results={summary.results!r}, "
                    f"pipeline={summary.pipeline_summary!r}",
                )
                self.assertEqual(summary.results[0].status, "DONE")
                self.assertEqual(len(aligner.calls), 1)
                self.assertEqual(len(enfuser.calls), 1)
                self.assertTrue((output / "STACK0_stack.jpg").is_file())
                self.assertTrue(all(not path.exists() for path in originals))

                jobs = controller.database.list_jobs(group_id=group.id)
                self.assertEqual(len(jobs), 1, "resume must not duplicate a durable job")
                self.assertEqual(jobs[0].status, "DONE")
                self.assertEqual(jobs[0].stage, "EXPORT")
                recovered = controller.database.get_group(group.id)
                assert recovered is not None
                self.assertEqual(recovered.status, "DONE")
            finally:
                controller.close()

    def test_done_job_is_not_reprocessed_on_next_start(self):
        """A cleanly completed job is not reconstructed from durable state."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            # An empty source is the expected state after the preceding MOVE.
            database = Database(output / ".stack_cache" / "database.sqlite")
            group = GroupRecord(image_count=2, selected_count=2, status="DONE")
            # A DONE row without an active source is sufficient to exercise
            # the job filter; no groups should be rebuilt or sent to Hugin.
            database.insert_group(group)
            assert group.id is not None
            database.create_job(JobRecord(group_id=group.id, stage="EXPORT", status="DONE", progress=1.0))
            database.close()

            class EmptyScan:
                def scan_report(self):
                    return SimpleNamespace(root=str(source), images=[], records=[], count=0, total_bytes=0, issues=())

            class EmptyDetector:
                def detect(self, records):
                    return []

            controller = ApplicationController(
                ApplicationOptions(source_dir=source, output_dir=output),
                scanner=EmptyScan(),
                scene_detector=EmptyDetector(),
                analyzer=_NoAnalyze(),
                config=AppConfig(
                    runtime=RuntimeConfig(
                        disk_safety_margin_bytes=0,
                        min_available_memory_bytes=0,
                        min_available_memory_fraction=0,
                    )
                ),
            )
            try:
                summary = controller.run(timeout=5)
                self.assertFalse(summary.cancelled)
                self.assertEqual(summary.groups_total, 0)
                self.assertEqual(summary.merge_finished, 0)
                self.assertEqual(len(controller.database.list_jobs(status="DONE")), 1)
            finally:
                controller.close()

    def test_application_logger_is_shared_by_hugin_adapters(self):
        """Commands and return codes from both adapters reach application.log."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            (source / "input.JPG").write_bytes(b"input")
            controller = ApplicationController(
                ApplicationOptions(source_dir=source, output_dir=output),
                config=AppConfig(
                    output=OutputConfig(format="jpg", jpeg_quality=100, jpeg_subsampling=0),
                    runtime=RuntimeConfig(
                        disk_safety_margin_bytes=0,
                        min_available_memory_bytes=0,
                        min_available_memory_fraction=0,
                    ),
                ),
            )
            try:
                source_path, output_path = controller._validate_paths()
                controller._prepare_runtime(source_path, output_path)
                aligner = AlignImageStack(
                    executable=root / "align_image_stack.exe",
                    runner=_LoggingCommandRunner(),
                )
                enfuser = Enfuser(
                    executable=root / "enfuse.exe",
                    runner=_LoggingCommandRunner(),
                    # This logger test uses byte-only TIFF stand-ins.
                    config=EnfuseConfig(full_resolution_focus_masks=False),
                )
                service = StackMergeService(
                    output,
                    aligner=aligner,
                    enfuser=enfuser,
                    fusion_backend="hugin_enfuse",
                    logger=controller.logger,
                )
                self.assertIs(service.aligner.logger, controller.logger)
                self.assertIs(service.enfuser.logger, controller.logger)
                inputs = [source / "a.jpg", source / "b.jpg"]
                for path in inputs:
                    path.write_bytes(b"input")
                alignment = service.aligner.align(inputs, work_dir=output / "work")
                service.enfuser.fuse(
                    alignment.aligned_paths,
                    output / "result.tif",
                    work_dir=output / "work",
                    output_config=OutputConfig(format="tiff"),
                )
                log_text = (output / ".stack_cache" / "logs" / "application.log").read_text(encoding="utf-8")
                self.assertIn("align_image_stack command:", log_text)
                self.assertIn("enfuse command:", log_text)
                self.assertGreaterEqual(log_text.count("rc=0"), 2)
            finally:
                controller.close()

    def test_overall_progress_uses_merge_finished_for_archive_only_jobs(self):
        event = PipelineEvent(
            stage=PipelineStage.COMPLETE,
            analysis_completed=2,
            analysis_total=2,
            merge_completed=0,
            merge_total=2,
            merge_finished=2,
        )
        self.assertEqual(event.overall_progress, 1.0)
        snapshot = ProgressSnapshot.from_event(event)
        self.assertEqual(snapshot.overall, 100)
        self.assertEqual(snapshot.merge, 0)
        self.assertEqual(snapshot.merge_finished, 2)


if __name__ == "__main__":
    unittest.main()

