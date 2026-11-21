import csv
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter
import pytest

from focus_stack_app.config import AppConfig, RuntimeConfig
from focus_stack_app.pipeline.application_controller import ApplicationController, ApplicationOptions
from focus_stack_app.storage.database import Database


def _make_sources(root):
    rng = np.random.default_rng(1234)
    coarse = Image.fromarray(rng.integers(15, 245, (20, 24, 3), dtype=np.uint8))
    coarse = coarse.resize((144, 120), Image.Resampling.BILINEAR)
    detail = Image.fromarray(rng.integers(20, 240, (120, 144, 3), dtype=np.uint8))
    sharp = Image.blend(coarse, detail, 0.20)
    blurred = sharp.filter(ImageFilter.GaussianBlur(1.5))
    for source, groups in (("A", 2), ("B", 1)):
        (root / source).mkdir()
        for group in range(1, groups + 1):
            for index, box in enumerate(((0, 0, 72, 120), (72, 0, 144, 120))):
                frame = blurred.copy()
                frame.paste(sharp.crop(box), box)
                # Two exposures per burst, with a gap long enough to split A.
                stamp = f"2026:10:01 10:{group:02d}:{index:02d}"
                exif = Image.Exif()
                exif[306] = stamp
                exif[34665] = {36867: stamp}
                frame.save(root / source / f"{source}{group}_{index}.jpg", exif=exif, quality=98)


def _run(root, source):
    options = ApplicationOptions(
        source_dir=root / source,
        output_dir=root / "output",
        archive_dir=root / f"archive-{source}",
        archive_enabled=True,
        minimum_stack_group_size=2,
        focus_analysis_workers=1,
        merge_workers=1,
        parallel=False,
    )
    config = AppConfig(runtime=RuntimeConfig(
        disk_safety_margin_bytes=0, min_available_memory_bytes=0, min_available_memory_fraction=0,
    ))
    controller = ApplicationController(options, config=config)
    try:
        return controller.run(timeout=30)
    finally:
        controller.close()


def _interrupt_second_manifest(root, interruption):
    stage = Database.stage_manifest_group

    def stage_group(db, source, group, rows):
        if Path(rows[0]["first_image"]).name == "A2_0.jpg":
            if interruption == "process exit":
                os._exit(67)
            raise OSError("injected manifest journal write error")
        return stage(db, source, group, rows)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Database, "stage_manifest_group", stage_group)
        return _run(root, "A")


def _manifest_rows(output):
    with (output / "stack_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _products(output):
    return {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in output.glob("*.jpg")}


@pytest.mark.parametrize("interruption", ["process exit", "journal write error"])
def test_shared_output_resume_after_product_is_published(tmp_path, interruption):
    _make_sources(tmp_path)
    output = tmp_path / "output"
    database = output / ".stack_cache" / "database.sqlite"
    if interruption == "process exit":
        # Exit the child from the real merge callback, before A2 can be staged.
        child = subprocess.run(
            [sys.executable, "-m", "tests.test_shared_output_recovery", str(tmp_path)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=45,
        )
        assert child.returncode == 67, (child.stdout, child.stderr)
    else:
        failed = _interrupt_second_manifest(tmp_path, interruption)
        assert failed.errors == 1, failed.diagnostics
        assert any("injected manifest journal write error" in item for item in failed.diagnostics)
        assert [result.status for result in failed.results] == ["DONE", "DONE"]

    published_a = _products(output)
    assert set(published_a) == {"A1_0_stack.jpg", "A2_0_stack.jpg"}
    assert not list((tmp_path / "A").glob("*.jpg"))
    archived_a = sorted((tmp_path / "archive-A").glob("*.jpg"))
    assert [path.name for path in archived_a] == ["A1_0.jpg", "A1_1.jpg", "A2_0.jpg", "A2_1.jpg"]
    source_a = str((tmp_path / "A").resolve())
    with Database(database) as db:
        a_jobs = db.list_jobs()
        assert len(a_jobs) == 2 and all(job.source_root == source_a for job in a_jobs)
        assert a_jobs[0].status == "DONE"
        assert a_jobs[1].status == ("RUNNING" if interruption == "process exit" else "FAILED")
        assert all(db.get_group(job.group_id).status == "DONE" for job in a_jobs)
        first_group_rows = db.pending_manifest_rows(source_a)
        assert [Path(row["image"]).name for row in first_group_rows] == ["A1_0.jpg", "A1_1.jpg"]
        a_archive_op_ids = [op.operation_id for op in db.list_archive_operations()]

    b = _run(tmp_path, "B")
    assert b.errors == 0 and b.groups_total == 1, b.diagnostics
    assert [result.status for result in b.results] == ["DONE"]
    assert [Path(row["image"]) for row in _manifest_rows(output)] == [
        tmp_path / "B" / "B1_0.jpg", tmp_path / "B" / "B1_1.jpg",
    ]
    published_b = _products(output)
    assert {name: published_b[name] for name in published_a} == published_a
    with Database(database) as db:
        assert db.list_jobs()[:2] == a_jobs
        assert db.pending_manifest_rows(source_a) == first_group_rows
        b_job = db.list_jobs()[2]
        assert Path(b_job.source_root) == tmp_path / "B" and b_job.status == "DONE"
        assert Path(db.get_group(b_job.group_id).output_path) == output / "B1_0_stack.jpg"
        assert not db.pending_manifest_rows(str((tmp_path / "B").resolve()))

    resumed = _run(tmp_path, "A")
    assert resumed.errors == 0 and resumed.groups_total == 1, resumed.diagnostics
    assert [(result.group_id, result.status, result.output_path) for result in resumed.results] == [
        (a_jobs[1].group_id, "DONE", output / "A2_0_stack.jpg"),
    ]
    assert _products(output) == published_b
    assert sorted((tmp_path / "archive-A").glob("*.jpg")) == archived_a
    rows = _manifest_rows(output)
    assert [(row["group_id"], Path(row["image"]), Path(row["merge_output"])) for row in rows] == [
        (str(job.group_id), tmp_path / "A" / f"A{group}_{index}.jpg", output / f"A{group}_0_stack.jpg")
        for group, job in enumerate(a_jobs, 1) for index in range(2)
    ]
    assert all(row["status"] == "DONE" and row["selected"] == "true" for row in rows)
    assert rows[:2] == [
        {name: "" if value is None else str(value) for name, value in row.items()}
        for row in first_group_rows
    ]
    with Database(database) as db:
        jobs = db.list_jobs()
        assert [job.id for job in jobs] == [job.id for job in (*a_jobs, b_job)]
        assert jobs[0] == a_jobs[0] and jobs[2] == b_job
        assert jobs[1].status == "DONE" and jobs[1].stage == "EXPORT"
        assert len(db.list_groups()) == 3
        assert not db.pending_manifest_rows(source_a)
        assert [
            op.operation_id for op in db.list_archive_operations() if Path(op.source_path).parent == tmp_path / "A"
        ] == a_archive_op_ids


if __name__ == "__main__":
    _interrupt_second_manifest(Path(sys.argv[1]), "process exit")
