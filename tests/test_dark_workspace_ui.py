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
def workspace(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: None)
    received = []

    def factory(**settings):
        received.append(settings)
        return FakeController()

    window = MainWindow(controller_factory=factory)
    window.source_edit.setText("C:/photos")
    window.output_edit.setText("C:/photos/合成")
    window.show()
    app.processEvents()
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

