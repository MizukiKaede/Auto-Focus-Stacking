import threading

from focus_stack_app.core.scanner import ImageScanner
from focus_stack_app.pipeline.job_queue import BoundedJobQueue
from focus_stack_app.storage.database import Database
from focus_stack_app.storage.manifest import ManifestWriter


def _minimal_jpeg(width=32, height=24):
    sof = bytes([8]) + height.to_bytes(2, "big") + width.to_bytes(2, "big") + bytes([1, 1, 0x11, 0])
    return b"\xff\xd8\xff\xc0" + (len(sof) + 2).to_bytes(2, "big") + sof + b"\xff\xd9"


def test_1000_jpg_metadata_database_manifest_and_bounded_queue(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    payload = _minimal_jpeg()
    for index in range(1000):
        (source / f"IMG{index:04d}.JPG").write_bytes(payload)

    records = ImageScanner(source).scan()
    assert len(records) == 1000
    assert [record.sequence_index for record in records] == list(range(1000))

    with Database(tmp_path / "state.sqlite") as database:
        database.insert_images(records)
        assert len(database.list_images()) == 1000

    queue = BoundedJobQueue(maxsize=3)
    consumed = []

    def consume():
        while True:
            item = queue.get()
            if item is None:
                return
            consumed.append(item)
            queue.task_done()

    worker = threading.Thread(target=consume)
    worker.start()
    for index in range(1000):
        queue.put(index)
        assert queue.qsize <= 3
    queue.close()
    worker.join(5)
    assert consumed == list(range(1000))

    manifest = ManifestWriter(tmp_path / "stack_manifest.csv")
    manifest.write_results([{
        "group_id": 1,
        "all_images": records,
        "selected_paths": [records[0].path, records[-1].path],
        "capture_order": [str(record.path) for record in records],
        "alignment_order": [str(records[-1].path), str(records[0].path)],
        "status": "READY_FOR_MERGE",
    }])
    assert len((tmp_path / "stack_manifest.csv").read_text(encoding="utf-8-sig").splitlines()) == 1001

