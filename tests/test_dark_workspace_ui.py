"""UI lifecycle regressions using a controller that never touches photographs."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox
from focus_stack_app.ui.main_window import MainWindow
from focus_stack_app.pipeline.coordinator import PipelineSummary
from focus_stack_app.pipeline.events import PipelineEvent, PipelineStage


class FakeController:
    running = False

    def start(self):
        self.running = True

    def cancel(self):
        self.running = False

    def close(self):
        self.running = False

    def wait(self, timeout):
        return None


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: None)
    received = []

    def factory(**settings):
        received.append(settings)
        return FakeController()

    window = MainWindow(controller_factory=factory)
    window.source_edit.setText(str(tmp_path))
    window.output_edit.setText(str(tmp_path / "合成"))
    window.show()
    from PySide6.QtTest import QTest
    for _ in range(200):
        if window.start_button.isEnabled():
            break
        QTest.qWait(10)
    assert window.start_button.isEnabled()
    yield app, window, received
    window.close()
    app.processEvents()


def test_disclosure_and_restart_settings(workspace):
    app, window, received = workspace
    assert not window.advanced_content.isVisible()
    assert not window.archive_content.isVisible()
    window.archive_check.setChecked(True)
    window.advanced_toggle.click()
    assert window.archive_content.isVisible()
    assert window.advanced_content.isVisible()
    window.start_processing()
    assert not window.settings_content.isEnabled()
    assert not window.start_button.isEnabled()
    assert window.stop_button.isEnabled()
    assert received[-1]["archive_enabled"] is True
    window._on_complete(PipelineSummary())
    assert window.settings_content.isEnabled()
    window.grouping_pause_spin.setValue(31)
    window.focus_analysis_workers_spin.setValue(6)
    window.start_processing()
    assert received[-1]["grouping_pause_seconds"] == 31
    assert received[-1]["focus_analysis_workers"] == 6
    assert window.progress_panel.snapshot.overall == 0
    assert window.group_list.table.rowCount() == 0


def test_stopping_keeps_status_and_cancel_preserves_progress(workspace):
    _, window, _ = workspace
    window.start_processing()
    window.stop_processing()
    window._on_progress(PipelineEvent(PipelineStage.FUSION, current_file="photo.jpg", analysis_total=4, analysis_completed=2, merge_total=4, merge_completed=1))
    assert window.run_status.text() == "正在安全停止…"
    assert window.statusBar().currentMessage() == "正在安全停止…"
    progress = window.progress_panel.overall_bar.value()
    window._on_complete(PipelineSummary(groups_total=4, cancelled=True))
    assert window.progress_panel.overall_bar.value() == progress
    assert window.settings_content.isEnabled()
    assert window.run_status.text() == "已取消"


@pytest.mark.parametrize("errors", [0, 1])
def test_completion_results_and_empty_next_run(workspace, errors):
    _, window, _ = workspace
    window.start_processing()
    window._on_complete(PipelineSummary(
        groups_total=1, analysis_completed=1, merge_completed=1, errors=errors,
        results=[{"group_id": 1, "status": "FAILED" if errors else "DONE", "output_path": "C:/photos/合成/example.jpg"}],
    ))
    assert window.group_list.table.isVisible()
    assert window.group_list.table.item(0, 4).text() == ("合成失败" if errors else "已完成")
    assert window.progress_panel.overall_bar.value() == 100
    window.start_processing()
    assert not window.group_list.table.isVisible()
    assert window.group_list.empty_label.text() == "处理结束后显示结果"
    window._on_complete(PipelineSummary())
    assert window.group_list.empty_label.text() == "本次没有可显示的结果"


def test_long_paths_and_diagnostics_do_not_expand_window(workspace):
    app, window, _ = workspace
    window.resize(960, 640)
    path = "C:/" + "非常长的照片目录/" * 40 + "photo.jpg"
    window._on_progress(PipelineEvent(PipelineStage.ANALYSIS, current_file=path, message="诊断信息" * 500))
    window.progress_panel.details_toggle.click()
    window.advanced_toggle.click()
    window.archive_check.setChecked(True)
    app.processEvents()
    assert window.width() == 960
    assert window.height() == 640
    assert window.progress_panel.file_label.toolTip() == path
    assert window.progress_panel.details_text.height() == 88
    assert window.settings_scroll.verticalScrollBar().maximum() > 0
    assert window.start_button.isVisible()
    assert window.group_list.height() > 100


def test_start_failure_keeps_settings_available(workspace):
    _, window, _ = workspace

    def fail(**kwargs):
        raise RuntimeError("Cannot start")

    window.controller_factory = fail
    window.start_processing()
    assert window.settings_content.isEnabled()
    assert window.start_button.isEnabled()
    assert not window.stop_button.isEnabled()


def test_bulk_results_filter_once_and_preserve_selection(workspace, monkeypatch):
    _, window, _ = workspace
    view = window.group_list
    calls = []
    original = view.table.setRowHidden

    def track(row, hidden):
        calls.append(row)
        original(row, hidden)

    monkeypatch.setattr(view.table, "setRowHidden", track)
    view.update_results({"group_id": i, "status": "DONE"} for i in range(500))
    assert calls == list(range(500))
    view.search_box.setText("#499")
    assert view.table.currentRow() == 499
    assert not view.table.isRowHidden(499)


def test_run_and_stopping_reject_directory_drop(workspace, tmp_path):
    from PySide6.QtCore import QMimeData, QUrl
    from types import SimpleNamespace

    _, window, _ = workspace
    other = tmp_path / "B"
    other.mkdir()
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(other))])
    decisions = []
    event = SimpleNamespace(mimeData=lambda: mime, ignore=lambda: decisions.append(False),
                            acceptProposedAction=lambda: decisions.append(True))
    destination = window.output_directory
    window.start_processing()
    window.dropEvent(event)
    window.stop_processing()
    window.dropEvent(event)
    assert decisions == [False, False]
    assert window.source_directory == tmp_path
    window._on_complete(PipelineSummary(cancelled=True))
    window.dropEvent(event)
    assert decisions[-1] and window.source_directory == other
    assert window.group_list._output_dir == destination


def test_async_preview_and_compact_actions(workspace, tmp_path):
    from PIL import Image
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    app, window, _ = workspace
    view = window.group_list
    paths = [tmp_path / "red.jpg", tmp_path / "blue.jpg"]
    for path, color in zip(paths, ("red", "blue")):
        Image.new("RGB", (1600, 1200), color).save(path)
    view.update_results([dict(group_id=i, output_path=str(p), status="DONE")
                         for i, p in enumerate(paths)])
    view.table.selectRow(1)
    for _ in range(250):
        if view._current_pixmap is not None:
            break
        QTest.qWait(10)
    assert view._current_pixmap.toImage().pixelColor(0, 0).blue() > 240
    assert view._current_pixmap.height() < 1200
    assert "1600 × 1200" in view.preview_meta.text()
    for _ in range(2):
        for width, height in [(1180, 780), (960, 640)]:
            window.resize(width, height)
            app.processEvents()
            assert view.more_btn.isVisible() == (width == 960)
            if width == 960:
                assert not view.zoom_btn.visibleRegion().isEmpty()
                assert not view.more_btn.visibleRegion().isEmpty()
                assert [a.text() for a in view.more_btn.menu().actions()] == ["系统打开", "定位文件"]
                assert view.more_btn.focusPolicy() & Qt.TabFocus
            assert view.zoom_btn.isVisible()
        window.theme_toggle_btn.click()


def test_obsolete_folder_statistics_cannot_enable_or_replace_current_folder(workspace, tmp_path, monkeypatch):
    import threading
    from PySide6.QtTest import QTest
    from focus_stack_app.ui import main_window

    _, window, _ = workspace
    older, newest = tmp_path/"older", tmp_path/"newest"
    older.mkdir()
    newest.mkdir()
    entered, release = threading.Event(), threading.Event()
    def scan(path, *args, cancelled):
        if path == older:
            entered.set()
            release.wait(timeout=5)
        return path.name, False, 1
    monkeypatch.setattr(main_window, "_folder_summary", scan)
    window.source_edit.setText(str(older))
    for _ in range(100):
        if entered.is_set():
            break
        QTest.qWait(10)
    assert entered.is_set()
    window.source_edit.setText(str(newest))
    assert not window.start_button.isEnabled()
    release.set()
    for _ in range(100):
        if window.start_button.isEnabled():
            break
        QTest.qWait(10)
    assert window.folder_stats_label.text() == "newest"
    window.source_edit.clear()
    assert window.group_list.empty_browse_btn.isDefault() and not window.start_button.isEnabled()
