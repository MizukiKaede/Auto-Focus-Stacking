from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

from focus_stack_app.storage.database import Database
from focus_stack_app.storage.models import AnalysisRecord, GroupRecord, ImageRecord, JobRecord


class DatabaseTests(unittest.TestCase):
    def test_image_group_analysis_and_selected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / ".stack_cache" / "database.sqlite"
            with Database(db_path) as database:
                first = ImageRecord(str(Path(directory) / "DSC1.JPG"), sequence_index=0, width=100, height=80)
                second = ImageRecord(str(Path(directory) / "DSC2.JPG"), sequence_index=1, width=100, height=80)
                database.insert_images([first, second])
                group = GroupRecord(first_image_id=first.id, start_index=0, end_index=1, image_count=2)
                database.insert_group(group)
                database.set_image_group(first.id or 0, group.id)
                database.set_image_group(second.id or 0, group.id)
                database.upsert_analysis(AnalysisRecord(first.id or 0, transform=[[1, 0], [0, 1]], selected=True))
                database.upsert_analysis(AnalysisRecord(second.id or 0, selected=False))
                selected = database.get_selected_images(group.id or 0)
                self.assertEqual([item.filename for item in selected], ["DSC1.JPG"])
                self.assertEqual(database.get_analysis(first.id or 0).transform, [[1, 0], [0, 1]])

    def test_archive_and_job_recovery_are_read_check_then_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.JPG"
            destination = root / "out" / source.name
            destination.parent.mkdir()
            destination.write_bytes(b"image")
            with Database(root / "db.sqlite") as database:
                image = ImageRecord(str(source))
                database.insert_image(image)
                # Simulate a crash after the filesystem move but before the
                # SQLite current_path/status update.
                database.connection.execute(
                    "UPDATE images SET current_path=?, status='ARCHIVING' WHERE id=?",
                    (str(destination), image.id),
                )
                recovered = database.recover_archived_images()
                self.assertEqual(len(recovered), 1)
                self.assertEqual(recovered[0].status, "ARCHIVED")
                self.assertEqual(Path(recovered[0].current_path or ""), destination)

                job = JobRecord(group_id=None, stage="FUSION", status="RUNNING")
                database.create_job(job)
                pending = database.recover_incomplete_jobs()
                self.assertTrue(any(item.id == job.id and item.status == "PENDING" for item in pending))

    def test_archive_journal_recovers_move_after_filesystem_before_db_update(self) -> None:
        """A pending destination is enough to reconcile a move crash safely."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.JPG"
            destination = root / "archive" / source.name
            source.write_bytes(b"jpeg bytes")
            db_path = root / "db.sqlite"

            with Database(db_path) as database:
                image = ImageRecord(str(source), file_size=source.stat().st_size)
                database.insert_image(image)
                database.record_archive_operation(
                    {
                        "operation_id": "move-crash-1",
                        "image_id": image.id,
                        "source_path": str(source),
                        "destination_path": str(destination),
                        "mode": "move",
                        "phase": "pending",
                        "status": "pending",
                    }
                )
                pending = database.get_image(image.id or 0)
                self.assertEqual(pending.status, "ARCHIVING")
                self.assertEqual(Path(pending.pending_path or ""), destination)

                # Simulate the filesystem move and a process crash before the
                # normal post-move image update can run.
                destination.parent.mkdir()
                source.replace(destination)

            with Database(db_path) as database:
                recovered = database.recover_archived_images()
                image = database.get_image(image.id or 0)
                operation = database.get_archive_operation("move-crash-1")
                self.assertEqual(len(recovered), 1)
                self.assertEqual(image.status, "ARCHIVED")
                self.assertEqual(Path(image.current_path or ""), destination)
                self.assertIsNone(image.pending_path)
                self.assertEqual(operation.phase, "archived")
                self.assertEqual(operation.source_size, len(b"jpeg bytes"))

    def test_archive_recovery_rejects_wrong_destination_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.JPG"
            destination = root / "archive" / source.name
            source.write_bytes(b"jpeg bytes")
            destination.parent.mkdir()
            db_path = root / "db.sqlite"

            with Database(db_path) as database:
                image = ImageRecord(str(source), file_size=source.stat().st_size)
                database.insert_image(image)
                database.record_archive_operation(
                    {
                        "operation_id": "move-conflict-1",
                        "image_id": image.id,
                        "source_path": str(source),
                        "destination_path": str(destination),
                        "mode": "move",
                        "phase": "pending",
                        "status": "pending",
                    }
                )
                source.unlink()
                destination.write_bytes(b"not the source")
                database.recover_archived_images()
                image = database.get_image(image.id or 0)
                operation = database.get_archive_operation("move-conflict-1")
                self.assertEqual(image.status, "FAILED")
                self.assertEqual(Path(image.current_path or ""), source)
                self.assertEqual(Path(image.pending_path or ""), destination)
                self.assertEqual(operation.phase, "conflict")

    def test_v1_schema_migrates_archive_columns_and_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "legacy.sqlite"
            connection = sqlite3.connect(db_path)
            connection.executescript(
                """
                CREATE TABLE images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_path TEXT NOT NULL UNIQUE,
                    current_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    stem TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    mtime REAL NOT NULL DEFAULT 0,
                    capture_time TEXT,
                    width INTEGER,
                    height INTEGER,
                    camera TEXT,
                    lens TEXT,
                    sequence_index INTEGER,
                    group_id INTEGER,
                    status TEXT NOT NULL DEFAULT 'DISCOVERED',
                    archive_mode TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE analysis (
                    image_id INTEGER PRIMARY KEY,
                    scene_hash TEXT,
                    scene_score REAL,
                    sharpness_score REAL,
                    focus_map_path TEXT,
                    transform_json TEXT,
                    selected INTEGER NOT NULL DEFAULT 0,
                    selection_reason TEXT,
                    coverage_gain REAL,
                    quality_score REAL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO schema_meta(key, value) VALUES ('schema_version', '1');
                """
            )
            connection.close()

            with Database(db_path) as database:
                image_columns = {
                    row["name"]
                    for row in database.connection.execute("PRAGMA table_info(images)")
                }
                self.assertTrue({"pending_path", "archive_operation_id", "archive_error"}.issubset(image_columns))
                self.assertEqual(database.connection.execute("PRAGMA user_version").fetchone()[0], 4)
                group_columns = {
                    row["name"] for row in database.connection.execute("PRAGMA table_info(groups)")
                }
                self.assertTrue({"preview_reference", "alignment_order", "requested_backend", "crop_ratio"}.issubset(group_columns))
                self.assertEqual(
                    len(list(database.connection.execute("PRAGMA table_info(archive_operations)"))),
                    15,
                )
                self.assertTrue(
                    list(database.connection.execute("PRAGMA table_info(cached_plans)"))
                )

    def test_cached_plans_round_trip_version_and_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with Database(Path(directory) / "db.sqlite") as database:
                database.upsert_cached_plan("selection", "key", "v1", {"selected": [1, 3]})
                self.assertEqual(
                    database.get_cached_plan("selection", "key", algorithm_version="v1"),
                    {"selected": [1, 3]},
                )
                self.assertIsNone(
                    database.get_cached_plan("selection", "key", algorithm_version="v2")
                )
                database.connection.execute(
                    "UPDATE cached_plans SET payload_json = ? WHERE kind = ? AND cache_key = ?",
                    ("{broken", "selection", "key"),
                )
                self.assertIsNone(database.get_cached_plan("selection", "key"))
                self.assertEqual(
                    database.connection.execute("SELECT COUNT(*) FROM cached_plans").fetchone()[0],
                    0,
                )

    def test_transaction_rolls_back_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with Database(Path(directory) / "db.sqlite") as database:
                with self.assertRaises(RuntimeError):
                    with database.transaction():
                        database.insert_image(ImageRecord(str(Path(directory) / "one.JPG")))
                        raise RuntimeError("abort")
                self.assertEqual(database.list_images(), [])


if __name__ == "__main__":
    unittest.main()
