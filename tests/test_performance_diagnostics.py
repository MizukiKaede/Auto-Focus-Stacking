import json
import logging
import pytest

from focus_stack_app.utils.performance import BatchMemory, stage, timed


def records(caplog):
    return [json.loads(r.message[5:]) for r in caplog.records if r.message.startswith("PERF ")]


def test_nested_timing_and_errors_preserve_result(caplog):
    caplog.set_level(logging.INFO, logger="focus_stack_app.controller.performance")
    @timed("outer")
    def calculate():
        with stage("inner"):
            raise ValueError("original error")
    with pytest.raises(ValueError, match="original error"):
        calculate()
    rows = records(caplog)
    assert [r["stage"] for r in rows] == ["inner", "outer"]
    assert [r["depth"] for r in rows] == [1, 0]
    assert all(r["seconds"] >= 0 and r["outcome"] == "error" for r in rows)


def test_memory_sampler_finishes_and_reports_unavailable(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="focus_stack_app.controller.performance")
    memory = BatchMemory()
    monkeypatch.setattr(memory, "sample", lambda: None)
    memory.start()
    memory.finish(2.5, merge_workers=2)
    assert not memory.thread.is_alive()
    assert records(caplog)[0]["sampled_process_tree_peak_rss_bytes"] is None


def test_stage_details_reach_project_logger_and_restore_context(tmp_path):
    path = tmp_path / "application.log"
    logger = logging.getLogger("test.project.performance")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path, encoding="utf-8")
    logger.addHandler(handler)
    class Worker:
        @timed("selection_inclusive")
        def analyze(self, group):
            with stage("selection"):
                return 42
    worker = Worker()
    worker.logger = logger
    try:
        assert worker.analyze({"id": 9}) == 42
        memory = BatchMemory(logger)
        memory.finish(1.0)
    finally:
        logger.removeHandler(handler)
        handler.close()
    rows = [json.loads(line.split("PERF ", 1)[1]) for line in path.read_text().splitlines()]
    assert [row["stage"] for row in rows] == ["selection", "selection_inclusive", "batch"]
    assert all(row["group_id"] == 9 for row in rows[:2])
    from focus_stack_app.utils.performance import _active_logger, _group
    assert _active_logger.get() is None and _group.get() is None


def test_merge_workers_ui_reaches_controller(tmp_path):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from focus_stack_app.ui.main_window import MainWindow
    from focus_stack_app.app import default_controller_factory
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.source_edit.setText(str(tmp_path))
        window.output_edit.setText(str(tmp_path / "out"))
        assert window.merge_workers_spin.value() == 1
        window.merge_workers_spin.setValue(2)
        window.focus_analysis_workers_spin.setValue(6)
        controller = default_controller_factory(**window._settings())
        try:
            assert controller.options.merge_workers == 2
            assert controller.options.focus_analysis_workers == 6
            assert controller.config.runtime.max_hugin_workers == 2
            assert controller.config.runtime.focus_analysis_workers == 6
        finally:
            controller.close()
    finally:
        window.close()
        app.processEvents()
