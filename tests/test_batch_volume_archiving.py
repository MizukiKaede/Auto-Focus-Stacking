from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from focus_stack_app.files.archiver import ArchiveError, FileArchiver
from focus_stack_app.fusion.backends import FusionBackend, FusionResult
from focus_stack_app.pipeline.analysis_worker import AnalysisJob
from focus_stack_app.pipeline.application_controller import ApplicationController, ApplicationOptions, DiskSpaceError
from focus_stack_app.pipeline.coordinator import PipelineCoordinator
from focus_stack_app.pipeline.merge_worker import StackMergeService
from focus_stack_app.storage.models import ImageRecord


class RecordingFusion(FusionBackend):
    name = "quality"

    def __init__(self):
        self.groups = []

    def fuse(self, group, analysis, output_path, work_dir, output_config, cancel_event):
        self.groups.append(group["group_id"])
        output_path.write_bytes(b"test composite")
        return FusionResult(output_path, self.name)


@pytest.fixture
def archive_setup(tmp_path, monkeypatch):
    remote = tmp_path / "other_volume"
    remote.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    fusion = RecordingFusion()
    merger = StackMergeService(
        tmp_path / "output", archiver=FileArchiver(archive, mode="hardlink"),
        fusion_backend=fusion, minimum_stack_group_size=2, publish_manifest=False,
    )
    original = Path.lstat

    def lstat(path):
        result = original(path)
        if path.parent == remote:
            return SimpleNamespace(st_mode=result.st_mode, st_dev=result.st_dev + 1)
        return result

    monkeypatch.setattr(Path, "lstat", lstat)
    return remote, archive, merger, fusion


def make_group(parent, group_id):
    items = []
    for index in range(2):
        path = parent / f"photo_{group_id}_{index}.jpg"
        path.write_bytes(f"fixture {group_id} {index}".encode())
        items.append(ImageRecord.from_path(path))
    return {"group_id": group_id, "items": items}


def test_cross_volume_hardlink_rejects_only_the_group_that_needs_a_new_link(archive_setup):
    remote, archive, merger, fusion = archive_setup
    cross_volume = make_group(remote, 1)
    no_merge = make_group(remote, 2)
    same_volume = make_group(remote.parent, 3)

    def analyze(group):
        needs_merge = group is not no_merge
        items = group["items"] if needs_merge else group["items"][:1]
        return {"selected_paths": [item.path for item in items], "needs_merge": needs_merge}

    summary = PipelineCoordinator(analyzer=analyze, merger=merger, parallel=False, merge_workers=1).run(
        [cross_volume, no_merge, same_volume],
    )
    assert summary.finished and summary.errors == 1
    [(failed_job, error)] = summary.failed_merge
    assert failed_job.group is cross_volume and isinstance(error, ArchiveError)
    assert str(remote) in str(error) and str(archive) in str(error) and "复制或移动" in str(error)
    assert [(result.group_id, result.status) for result in summary.results] == [(2, "NO_MERGE"), (3, "DONE")]
    assert fusion.groups == [3]
    assert len(list(merger.output_dir.glob("*.jpg"))) == 1
    assert all(item.path.is_file() for group in (cross_volume, no_merge) for item in group["items"])
    assert {path.name for path in archive.iterdir()} == {item.path.name for item in same_volume["items"]}
    assert all(item.path.samefile(archive / item.path.name) for item in same_volume["items"])


@pytest.mark.parametrize("recovery_state", ["finished", "already_archived", "existing_copy", "still_pending"])
def test_cross_volume_hardlink_resume_preserves_published_result(archive_setup, recovery_state):
    remote, archive, merger, fusion = archive_setup
    group = make_group(remote, 1)
    saved = merger.output_dir / "saved.jpg"
    saved.write_bytes(b"published before recovery")
    group.update(_recovery=True, _published_output=saved,
                 status="DONE" if recovery_state == "finished" else "FAILED_ARCHIVE")
    if recovery_state in {"already_archived", "existing_copy"}:
        for item in group["items"]:
            destination = archive / item.path.name
            destination.write_bytes(item.path.read_bytes())
            if recovery_state == "already_archived":
                item.current_path = str(destination)
    job = AnalysisJob(group, {"selected_indices": []}, archive_only=True)

    if recovery_state == "still_pending":
        with pytest.raises(ArchiveError, match="硬链接归档"):
            merger.process(job)
        assert not list(archive.iterdir())
    else:
        result = merger.process(job)
        assert result.status == "DONE" and result.output_path == saved
    assert fusion.groups == []
    assert saved.read_bytes() == b"published before recovery"
    assert all(Path(item.original_path).is_file() for item in group["items"])


def test_disk_error_reports_exact_shortfall_at_rounded_threshold(tmp_path, monkeypatch):
    controller = ApplicationController(ApplicationOptions(source_dir=tmp_path, output_dir=tmp_path / "output"))
    required = 65_536
    threshold = required + controller.config.runtime.disk_safety_margin_bytes
    monkeypatch.setattr(shutil, "disk_usage", lambda _target: SimpleNamespace(free=threshold))
    controller._disk_check(required)
    monkeypatch.setattr(shutil, "disk_usage", lambda _target: SimpleNamespace(free=threshold - 1))
    with pytest.raises(DiskSpaceError, match="还差 1 字节"):
        controller._disk_check(required)
