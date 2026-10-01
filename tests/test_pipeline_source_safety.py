"""Targeted source-preservation checks for archive-only merge jobs.

These cases deliberately use real temporary JPEG files.  An archive-only
decision must publish its classification result while leaving every source
file in the input directory; only a successful fused output may reach the
archiver.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from focus_stack_app.files.archiver import ArchiveMode, FileArchiver
from focus_stack_app.pipeline.analysis_worker import AnalysisJob
from focus_stack_app.pipeline.merge_worker import StackMergeService


class _MustNotRun:
    def align(self, *args, **kwargs):
        raise AssertionError("alignment must not run for an archive-only job")

    def fuse(self, *args, **kwargs):
        raise AssertionError("fusion must not run for an archive-only job")


@pytest.mark.parametrize("case", ["NO_MERGE", "FAILED_CLASSIFICATION"])
def test_unmerged_archive_only_job_preserves_real_sources(tmp_path: Path, case: str) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    archive_dir = tmp_path / "archive"
    source_dir.mkdir()
    sources = []
    for index in range(3):
        path = source_dir / f"DSC{index:04d}.JPG"
        Image.new("RGB", (8, 8), (20 + index * 30, 40, 60)).save(path, format="JPEG", quality=95)
        sources.append(path)
    original_bytes = {path: path.read_bytes() for path in sources}
    group = {
        "id": 901 if case == "NO_MERGE" else 902,
        "images": [{"path": path} for path in sources],
    }

    service = StackMergeService(
        output_dir,
        archive_dir=archive_dir,
        archiver=FileArchiver(archive_dir, ArchiveMode.MOVE),
        aligner=_MustNotRun(),
        enfuser=_MustNotRun(),
    )
    if case == "NO_MERGE":
        analysis = {
            "all_images": group["images"],
            "selected_paths": [sources[0]],
            "first_original_image": group["images"][0],
            "needs_merge": False,
        }
        job = AnalysisJob(group=group, analysis=analysis, archive_only=True)
        expected_status = "NO_MERGE"
    else:
        job = AnalysisJob(
            group=group,
            error=ValueError("classification failed"),
            archive_only=True,
        )
        expected_status = "FAILED_CLASSIFICATION"

    result = service.process(job)

    assert result.status == expected_status
    assert all(path.is_file() for path in sources)
    assert {path: path.read_bytes() for path in sources} == original_bytes
    assert not list(output_dir.glob("*_stack.*"))
    archived_entries = list(archive_dir.rglob("*") ) if archive_dir.exists() else []
    assert archived_entries == []
