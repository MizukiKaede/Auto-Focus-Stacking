import csv
import json
import struct
import subprocess
import sys
import threading

import numpy as np
from PIL import Image, ImageCms
import pytest

from focus_stack_app.core.group_analyzer import GroupAnalyzer
from focus_stack_app.core.group_detector import StackGroupDetector
from focus_stack_app.core.manual_groups import group_with_overrides
from focus_stack_app.core.types import SceneGroup
from focus_stack_app.fusion.backends import QualityFusionBackend
from focus_stack_app.config import OutputConfig
from focus_stack_app.pipeline.application_controller import ApplicationController
from focus_stack_app.pipeline.job_state import JobStateStore
from focus_stack_app.storage.database import Database
from focus_stack_app.storage.manifest import ManifestWriter
from focus_stack_app.storage.models import ImageRecord
from focus_stack_app.utils.exif import _read_scalar
from focus_stack_app.utils.image_io import read_jpeg_size


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("kind,code", [(3, "H"), (4, "I"), (9, "i")])
def test_multivalue_exif(endian, kind, code):
    values = [-7, 4] if kind == 9 else [3, 7]
    assert _read_scalar(struct.pack(endian + "2" + code, *values), 0, kind, 2, endian) == values


def test_late_jpeg_dimensions(tmp_path):
    path = tmp_path / "large_header.jpg"
    Image.new("RGB", (57, 31)).save(path)
    data = path.read_bytes()
    segment = bytes.fromhex("ffe2ffff") + bytes(65533)
    path.write_bytes(data[:2] + segment * 2 + data[2:])
    assert (read_jpeg_size(path).width, read_jpeg_size(path).height) == (57, 31)


def test_unreadable_frame_splits_only_its_neighbours(tmp_path):
    items = []
    for i in range(9):
        path = tmp_path / f"{i}.jpg"
        Image.new("RGB", (80, 60), "grey").save(path)
        items.append(ImageRecord.from_path(path))
    items[4].path.write_bytes(b"bad jpeg")
    groups = list(StackGroupDetector().iter_groups(items))
    assert [g.image_count for g in groups] == [4, 1, 4]
    assert str(items[4].path) in groups[1].excluded_reason
    with pytest.raises(ValueError, match="预览读取失败"):
        GroupAnalyzer().analyze_group(groups[1])
    Image.new("RGB", (80, 60), "grey").save(items[4].path)
    assert [g.image_count for g in StackGroupDetector().iter_groups(items)] == [9]


def test_recursive_manual_groups(tmp_path):
    items = [ImageRecord.from_path(tmp_path / folder / "1.jpg") for folder in ("A", "B")]
    config = tmp_path / "stack_groups.json"
    config.write_text(json.dumps({"groups": [["1.jpg"]]}))
    with pytest.raises(ValueError, match="A/1.jpg.*B/1.jpg"):
        group_with_overrides(items, config, lambda records: [])
    config.write_text(json.dumps({"groups": [["A/1.jpg"], ["B/1.jpg"]]}))
    groups = group_with_overrides(items, config, lambda records: [])
    assert [g.items for g in groups] == [[items[0]], [items[1]]]
    config.write_text(json.dumps({"groups": [["missing.jpg"]]}))
    with pytest.raises(ValueError, match="missing.jpg"):
        group_with_overrides(items, config, lambda records: [])


def test_group_transaction_and_source_scoped_recovery(tmp_path, monkeypatch):
    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir()
    b.mkdir()
    records = []
    for i in range(2):
        path = a / f"{i}.jpg"
        Image.new("RGB", (40, 30)).save(path)
        records.append(ImageRecord.from_path(path))
    with Database(tmp_path / "jobs.db") as db:
        ctl = ApplicationController(source_dir=a, database=db)
        ctl._job_state = JobStateStore(db)
        db.insert_images(records)
        group = SceneGroup(1, records)
        with monkeypatch.context() as patch:
            def fail(*args):
                raise OSError("injected association failure")
            patch.setattr(db, "set_image_group", fail)
            with pytest.raises(OSError):
                ctl._persist_groups([group], records)
        assert not db.list_jobs() and not db.list_groups()
        assert all(item.group_id is None for item in records)
        ctl._persist_groups([group], records)
        with monkeypatch.context() as patch:
            original = db.upsert_analysis
            def fail_second(row):
                if row.image_id == records[1].id:
                    raise OSError("injected analysis failure")
                original(row)
            patch.setattr(db, "upsert_analysis", fail_second)
            with pytest.raises(OSError):
                GroupAnalyzer(database=db).analyze_group(group)
        assert not db.list_analysis()
        assert db.list_jobs()[0].source_root == str(a)
        assert len(ctl._load_recovery_groups()[0]) == 1
        ctl.source_dir = b
        assert ctl._load_recovery_groups() == ([], {})


def test_manifest_replays_abrupt_process_exit(tmp_path):
    dbpath = tmp_path / "journal.sqlite"
    manifest = tmp_path / "stack_manifest.csv"
    code = """
import os,sys
from focus_stack_app.storage.database import Database
from focus_stack_app.storage.manifest import ManifestWriter
w = ManifestWriter(sys.argv[2], journal=Database(sys.argv[1]), source_root='A', flush_interval_seconds=60)
w.update_result(dict(group_id=1, all_paths=['a.jpg'], status='DONE'))
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", code, str(dbpath), str(manifest)], check=True)
    assert not manifest.exists()
    with Database(dbpath) as db:
        w = ManifestWriter(manifest, journal=db, source_root="A", flush_interval_seconds=60)
        w.recover(resume=True)
        w.update_result(dict(group_id=1, all_paths=["a.jpg"], status="DONE"))
        w.update_result(dict(group_id=2, all_paths=["b.jpg"], status="FAILED"))
        w.flush(finished=True)
        assert not db.pending_manifest_rows("A")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        assert [(r["group_id"], r["image"]) for r in csv.DictReader(stream)] == [("1", "a.jpg"), ("2", "b.jpg")]


@pytest.mark.parametrize("fmt", ["jpg", "tiff"])
@pytest.mark.parametrize("orientation", [None, 1, 6])
def test_quality_capture_metadata(tmp_path, fmt, orientation):
    image = Image.new("RGB", (96, 64), (25, 90, 170))
    image.paste((200, 25, 10), (0, 0, 40, 24))
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    kwargs = {}
    if orientation is not None:
        tags = Image.Exif()
        tags[272] = "Test Camera"
        tags[274] = orientation
        tags[34665] = {36867: "2026:11:21 10:12:14", 42036: "Test Lens"}
        kwargs = {"exif": tags, "icc_profile": profile}
    paths = [tmp_path / f"{i}.jpg" for i in range(2)]
    for path in paths:
        image.save(path, **kwargs)
    analysis = dict(selected_paths=paths, selected_indices=[0, 1], preview_reference=paths[0],
                    preview_reference_index=0, preview_transforms=[np.eye(3).tolist()] * 2,
                    analysis_shapes=[[64, 96]] * 2, reference_analysis_shape=[64, 96])
    output = tmp_path / f"stack.{fmt}"
    QualityFusionBackend().fuse({}, analysis, output, tmp_path / "work", OutputConfig(format=fmt), threading.Event())
    with Image.open(output) as result:
        result.load()
        assert result.size == ((64, 96) if orientation == 6 else (96, 64))
        mark = (56, 10) if orientation == 6 else (10, 8)
        assert result.getpixel(mark)[0] > 150
        from PySide6.QtGui import QImageReader
        reader = QImageReader(str(output))
        reader.setAutoTransform(True)
        displayed = reader.read()
        assert (displayed.width(), displayed.height()) == result.size
        assert displayed.pixelColor(*mark).red() > 150
        if orientation:
            assert result.info["icc_profile"] == profile
            assert result.getexif()[272] == "Test Camera"
            assert result.getexif().get(274, 1) == 1
            details = result.getexif().get_ifd(34665)
            assert details[42036] == "Test Lens" and details[36867] == "2026:11:21 10:12:14"


@pytest.mark.parametrize("mode,same_volume,archive_bytes", [
    ("copy", False, 500), ("copy", True, 500), ("move", True, 0),
    ("move", False, 500), ("hardlink", True, 0),
])
def test_volume_disk_estimates(tmp_path, monkeypatch, mode, same_volume, archive_bytes):
    from pathlib import Path
    from types import SimpleNamespace
    from focus_stack_app.pipeline.application_controller import ApplicationOptions

    out = tmp_path / "out"
    arc = tmp_path / "archive"
    out.mkdir()
    arc.mkdir()
    ctl = ApplicationController(ApplicationOptions(source_dir=tmp_path, output_dir=out, archive_dir=arc,
                                                  archive_enabled=True, archive_mode=mode, merge_workers=1))
    files = []
    for i in range(4):
        path = tmp_path / f"{i}.jpg"
        path.touch()
        rec = ImageRecord.from_path(path)
        rec.file_size, rec.width, rec.height = 100, 40, 30
        files.append(rec)
    stat = Path.stat
    def disk_stat(path, *args, **kwargs):
        real = stat(path, *args, **kwargs)
        return SimpleNamespace(st_dev=20 if path == arc and not same_volume else 10, st_mode=real.st_mode)
    monkeypatch.setattr(Path, "stat", disk_stat)
    checks = {}
    monkeypatch.setattr(ctl, "_disk_check", lambda required, target: checks.update({target: required}))
    ctl._quality_disk_check([SceneGroup(1, files)])
    assert checks[out] == 65536 + 1200 + (archive_bytes if same_volume else 0)
    assert checks.get(arc, 0) == (0 if same_volume else archive_bytes)


def test_bad_jpeg_manifest_then_repaired_regroup(tmp_path):
    from pathlib import Path
    from focus_stack_app.pipeline.application_controller import ApplicationOptions
    source = tmp_path / "photos"
    source.mkdir()
    options = ApplicationOptions(source_dir=source, output_dir=tmp_path/"out", archive_enabled=False,
                                 focus_analysis_workers=1, parallel=False)
    # Missing EXIF on the damaged exposure must not move it after both bursts.
    exif = Image.Exif()
    exif[306] = "2026:10:01 10:00:00"
    exif[34665] = {36867: "2026:10:01 10:00:00"}
    pixels = np.random.default_rng(2).integers(30, 225, (160, 200, 3), dtype=np.uint8)
    good = Image.fromarray(pixels)
    for i in range(9):
        good.save(source/f"{i}.jpg", exif=exif)
    bad = source/"4.jpg"
    bad.write_bytes(b"bad jpeg")
    ctl = ApplicationController(options)
    try:
        failed = ctl.run(timeout=30)
        assert [g.image_count for g in failed.groups] == [4, 1, 4]
        with (tmp_path/"out/stack_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 9
        errors = [r for r in rows if r["status"] == "FAILED_CLASSIFICATION"]
        assert len(errors) == 1 and Path(errors[0]["image"]) == bad
        assert "预览读取失败" in errors[0]["diagnostics"]
        assert len(failed.results) == 2
    finally:
        ctl.close()
    good.save(bad, exif=exif)
    ctl = ApplicationController(options)
    try:
        repaired = ctl.run(timeout=30)
        assert [g.image_count for g in repaired.groups] == [9] and not repaired.errors
    finally:
        ctl.close()


def test_journal_normalizes_real_output_and_retains_failed_flush(tmp_path, monkeypatch):
    output = tmp_path/"stack.jpg"
    path = tmp_path/"manifest.csv"
    with Database(tmp_path/"db") as db:
        writer = ManifestWriter(path, journal=db, source_root="photos", flush_interval_seconds=60)
        writer.update_result(dict(group_id=1, all_paths=[str(output)], output_path=output, status="DONE"))
        with monkeypatch.context() as patch:
            def fail(*args):
                raise PermissionError("locked CSV")
            patch.setattr(writer, "_write_rows", fail)
            with pytest.raises(PermissionError, match="locked CSV"):
                writer.flush(finished=True)
        assert db.pending_manifest_rows("photos")[0]["merge_output"] == str(output)
        writer.flush(finished=True)
        assert path.is_file() and not db.pending_manifest_rows("photos")


def test_rgb_thumbnail_matches_eager_decode(tmp_path):
    from focus_stack_app.utils.image_io import load_rgb
    path = tmp_path/"noise.jpg"
    pixels = np.random.default_rng(7).integers(0, 256, (1300, 2900, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, quality=92)
    with Image.open(path) as source:
        expected = source.convert("RGB")
    expected.thumbnail((640, 640))
    assert np.array_equal(load_rgb(path, 640), np.asarray(expected))


def test_recovery_republishes_committed_output_without_fusing(tmp_path):
    from focus_stack_app.pipeline.application_controller import ApplicationOptions
    from focus_stack_app.storage.models import AnalysisRecord

    source, out = tmp_path/"A", tmp_path/"output"
    source.mkdir()
    out.mkdir()
    opts = ApplicationOptions(source_dir=source, output_dir=out, archive_enabled=False, parallel=False)
    images = []
    for i in range(4):
        path = source/f"{i}.jpg"
        Image.new("RGB", (96, 64), "green").save(path)
        images.append(ImageRecord.from_path(path))
    product = out/"already_published.jpg"
    Image.new("RGB", (96, 64), "blue").save(product)
    timestamp = product.stat().st_mtime_ns
    with Database(tmp_path/"recovery.db") as db:
        ctl = ApplicationController(opts, database=db)
        db.insert_images(images)
        group = SceneGroup(1, images)
        ctl._persist_groups([group], images)
        for image in images:
            db.upsert_analysis(AnalysisRecord(image_id=image.id, selected=True, quality_score=0.9))
        db.set_group_status(group.group_id, "DONE", output_path=str(product), actual_backend="quality",
                            selected_count=4)
        summary = ctl.run(timeout=30)
        assert summary.errors == 0 and summary.results[0].output_path == product
        assert product.stat().st_mtime_ns == timestamp
        assert list(out.glob("*.jpg")) == [product]
        assert db.list_jobs()[0].status == "DONE"
        with (out/"stack_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4 and all(r["quality_score"] == "0.9" and r["status"] == "DONE" for r in rows)


def test_empty_batch_replaces_previous_source_manifest(tmp_path):
    csv_path = tmp_path/"stack_manifest.csv"
    ManifestWriter(csv_path).update_result(dict(group_id=7, all_paths=["other.jpg"], status="DONE"))
    with Database(tmp_path/"db") as db:
        w = ManifestWriter(csv_path, journal=db, source_root="new source", flush_interval_seconds=2)
        w.recover(resume=False)
        w.flush(finished=True)
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        assert "image" in reader.fieldnames and list(reader) == []
