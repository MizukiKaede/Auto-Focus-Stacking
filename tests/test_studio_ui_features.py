"""Focused tests for the redesigned Focus Stack Studio UI features."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

import focus_stack_app.ui.main_window as main_window_mod
from focus_stack_app.pipeline.events import PipelineEvent, PipelineStage
from focus_stack_app.ui import ImageLightboxDialog, MainWindow
from focus_stack_app.ui.group_list import GroupListWidget
from focus_stack_app.ui.progress_panel import ProgressPanel


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _make_sample_image(path: Path, width: int = 320, height: int = 240) -> Path:
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("#233042"))
    painter = QPainter(pixmap)
    painter.setBrush(QColor("#3B82F6"))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(60, 40, 200, 160)
    painter.end()
    pixmap.save(str(path), "PNG")
    return path


def test_theme_toggle_switches_light_and_dark():
    _app()
    window = MainWindow()
    assert window._theme_name == "light"
    assert window.theme_toggle_btn.text() == "暗房模式"

    window.theme_toggle_btn.click()
    assert window._theme_name == "dark"
    assert window.theme_toggle_btn.text() == "浅色工作室"
    assert window.progress_panel._tokens.name == "dark"
    assert window.group_list._tokens.name == "dark"

    window.theme_toggle_btn.click()
    assert window._theme_name == "light"
    assert window.theme_toggle_btn.text() == "暗房模式"
    window.close()


def test_scan_options_and_hugin_settings_reach_controller(tmp_path):
    _app()
    captured: list[dict] = []

    class DummyController:
        running = False

        def start(self, **kwargs):
            return None

        def cancel(self):
            return None

        def wait(self, timeout=None):
            return True

        def close(self):
            return None

    def factory(**settings):
        captured.append(dict(settings))
        return DummyController()

    source_dir = tmp_path / "photos"
    source_dir.mkdir()
    hugin_dir = tmp_path / "hugin_bin"
    hugin_dir.mkdir()

    window = MainWindow(controller_factory=factory)
    window.source_edit.setText(str(source_dir))
    window.recursive_check.setChecked(True)
    window.include_hidden_check.setChecked(True)
    idx = window.backend_combo.findData("hugin_enfuse")
    assert idx >= 0
    window.backend_combo.setCurrentIndex(idx)
    window.hugin_bin_edit.setText(str(hugin_dir))

    window.start_processing()
    assert captured
    sent = captured[-1]
    assert sent["recursive"] is True
    assert sent["include_hidden"] is True
    assert sent["fusion_backend"] == "hugin_enfuse"
    assert sent["hugin_bin"] == str(hugin_dir)
    window.close()


def _wait_until(app, predicate, timeout: float = 3.0) -> None:
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    app.processEvents()


def test_source_folder_inspection_reports_jpg_non_jpg_and_manual_groups(tmp_path):
    app = _app()
    window = MainWindow()

    nested = tmp_path / "sub_batch"
    nested.mkdir()
    (nested / "IMG_001.JPG").write_bytes(b"fakejpg1")
    (nested / "IMG_002.JPG").write_bytes(b"fakejpg2")
    (tmp_path / "RAW_001.NEF").write_bytes(b"fakeraw")

    window.source_edit.setText(str(tmp_path))
    _wait_until(app, lambda: "非 JPG" in window.folder_stats_label.text())
    assert window.manual_groups_label.isHidden()
    assert "非 JPG" in window.folder_stats_label.text()

    # Enabling recursive reveals the nested JPGs
    window.recursive_check.setChecked(True)
    _wait_until(app, lambda: "发现 2 张 JPG" in window.folder_stats_label.text())
    assert "发现 2 张 JPG" in window.folder_stats_label.text()
    assert "1 个非 JPG" in window.folder_stats_label.text()

    # Manual grouping file detection
    (tmp_path / "stack_groups.json").write_text('{"groups": []}', encoding="utf-8")
    window._inspect_source_folder(str(tmp_path))
    _wait_until(app, lambda: not window.manual_groups_label.isHidden())
    assert not window.manual_groups_label.isHidden()
    assert "stack_groups.json" in window.manual_groups_label.text()
    window.close()


def test_group_list_inspector_details_filter_and_slice_toggle(tmp_path):
    app = _app()
    widget = GroupListWidget()
    widget.resize(860, 560)
    widget.show()
    app.processEvents()

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_img = _make_sample_image(out_dir / "stack_001.png")
    (out_dir / "stack_manifest.csv").write_text("group_id,status\n1,DONE\n", encoding="utf-8")
    widget.set_output_directory(out_dir)

    group_ok = SimpleNamespace(
        group_id=1,
        image_count=5,
        source_paths=[tmp_path / f"IMG_{i:04d}.JPG" for i in range(1, 6)],
        output_path=out_img,
    )
    analysis_ok = SimpleNamespace(
        image_count=5,
        selected_count=3,
        coverage_ratio=0.875,
        selected_paths=[tmp_path / f"IMG_{i:04d}.JPG" for i in (2, 3, 4)],
        excluded_reason="",
    )
    archive_ok = SimpleNamespace(
        records=[SimpleNamespace(status="archived") for _ in range(5)],
        failed=[],
    )
    result_ok = SimpleNamespace(
        group=group_ok,
        analysis=analysis_ok,
        status="DONE",
        output_path=out_img,
        requested_backend="quality",
        actual_backend="hugin",
        fallback_used=True,
        crop_ratio=0.923,
        alignment_status="aligned",
        archive_result=archive_ok,
        error=None,
        diagnostics=["align_image_stack converged in 1.2s"],
    )

    group_err = SimpleNamespace(
        group_id=2,
        image_count=4,
        source_paths=[tmp_path / f"IMG_00{i}.JPG" for i in range(6, 10)],
        output_path=None,
    )
    analysis_err = SimpleNamespace(
        image_count=4,
        selected_count=2,
        coverage_ratio=0.25,
        excluded_reason="有效清晰切片覆盖率不足",
    )
    result_err = SimpleNamespace(
        group=group_err,
        analysis=analysis_err,
        status="FAILED",
        output_path=None,
        error=RuntimeError("enfuse returned exit code 1"),
        diagnostics=["missing highlight frames"],
    )

    widget.update_results([result_ok, result_err])
    app.processEvents()

    assert widget.count_badge.text() == "2 组"
    assert "1 已合成" in widget.breakdown_label.text()
    assert "1 失败" in widget.breakdown_label.text()
    assert widget.open_manifest_btn.isEnabled()

    # Row 0 selected automatically
    assert widget.info_group_id.text() == "组 #1"
    assert "回退至 Hugin" in widget.info_backend_pill.text()
    assert "有效画幅 92.3%" in widget.align_meta_label.text()
    assert "已处理 5 张" in widget.info_archive.text()
    _wait_until(app, lambda: widget.zoom_btn.isEnabled())
    assert widget.zoom_btn.isEnabled()

    # Slice checklist toggle
    assert widget.slices_toggle.isVisible()
    widget.slices_toggle.setChecked(True)
    slice_text = widget.slices_box.toPlainText()
    assert "✓ [选中]  IMG_0002.JPG" in slice_text
    assert "· [未选]  IMG_0001.JPG" in slice_text

    # Select row 1 (FAILED) and confirm diagnostics banner displays error + reason + diagnostics
    widget.table.selectRow(1)
    app.processEvents()
    assert not widget.reason_frame.isHidden()
    reason_text = widget.info_reason.text()
    assert "排除原因：有效清晰切片覆盖率不足" in reason_text
    assert "enfuse returned exit code 1" in reason_text
    assert "missing highlight frames" in reason_text
    assert not widget.zoom_btn.isEnabled()

    # Filter chips & search box
    btn_failed = next(b for b in widget.filter_buttons if b.property("filter_id") == "failed")
    btn_all = next(b for b in widget.filter_buttons if b.property("filter_id") == "all")
    btn_failed.click()
    assert widget.table.isRowHidden(0)
    assert not widget.table.isRowHidden(1)
    btn_all.click()
    widget.search_box.setText("IMG_0001")
    assert not widget.table.isRowHidden(0)
    assert widget.table.isRowHidden(1)
    widget.close()


def test_image_lightbox_zoom_and_fit(tmp_path):
    app = _app()
    img_path = _make_sample_image(tmp_path / "preview.png", width=640, height=480)
    dialog = ImageLightboxDialog(img_path, title="组 #1 · 校对")
    dialog.resize(800, 600)
    dialog.show()
    app.processEvents()

    dialog.zoom_actual_size()
    assert dialog.zoom_badge.text() == "100%"

    dialog._step_zoom(1.5)
    assert dialog.zoom_badge.text() == "150%"

    dialog.fit_to_window()
    assert dialog._fit_mode is True or int(dialog.zoom_badge.text().rstrip("%")) > 0
    dialog.close()


def test_settings_persistence_roundtrip_with_restore_settings(tmp_path, monkeypatch):
    _app()
    ini_path = tmp_path / "studio_test.ini"
    monkeypatch.setattr(
        main_window_mod,
        "QSettings",
        lambda org, app_name: QSettings(str(ini_path), QSettings.IniFormat),
    )

    hugin_dir = tmp_path / "hugin"
    hugin_dir.mkdir()

    w1 = MainWindow(restore_settings=True)
    w1.theme_toggle_btn.click()  # light -> dark
    w1.recursive_check.setChecked(True)
    w1.include_hidden_check.setChecked(True)
    w1.minimum_group_spin.setValue(6)
    w1.hugin_bin_edit.setText(str(hugin_dir))
    w1.close()

    w2 = MainWindow(restore_settings=True)
    assert w2._theme_name == "dark"
    assert w2.recursive_check.isChecked() is True
    assert w2.include_hidden_check.isChecked() is True
    assert w2.minimum_group_spin.value() == 6
    assert w2.hugin_bin_edit.text() == str(hugin_dir)
    w2.close()


def test_progress_panel_stepper_diagnostics_and_eta(tmp_path):
    app = _app()
    panel = ProgressPanel()
    panel.resize(780, 220)
    panel.show()

    log_file = tmp_path / "application.log"
    log_file.write_text("line 1\n", encoding="utf-8")
    panel.set_log_path(log_file)
    assert panel.open_log_btn.isEnabled()

    panel.reset()
    panel._run_started_mono -= 10.0  # simulate 10s elapsed
    event = PipelineEvent(
        stage=PipelineStage.FUSION,
        current_file=str(tmp_path / "IMG_0010.JPG"),
        current_group=1,
        completed=5,
        total=10,
        groups_found=2,
        groups_finished=1,
        errors=0,
        message="正在高质量对齐融合第 1 组",
        analysis_completed=10,
        analysis_total=10,
        merge_completed=1,
        merge_total=2,
        merge_finished=1,
    )
    panel.update_event(event)
    app.processEvents()

    assert panel.stage_label.text() == "正在进行全分辨率景深融合…"
    assert "预计剩余" in panel.eta_label.text()

    panel.details_toggle.setChecked(True)
    panel._copy_diagnostics()
    assert "正在高质量对齐融合第 1 组" in QApplication.clipboard().text()
    panel.close()


def test_phase_aligned_timer_and_anti_glare_light_palette(monkeypatch):
    _app()
    from focus_stack_app.ui.theme import LIGHT_THEME

    # Light theme avoids harsh pure-#FFFFFF large panel surfaces
    assert LIGHT_THEME.bg_surface.upper() != "#FFFFFF"
    assert LIGHT_THEME.bg_canvas.upper() != "#FFFFFF"
    assert LIGHT_THEME.bg_input.upper() != "#FFFFFF"

    window = MainWindow()
    # Running stopwatch uses floor semantics so x.5s ticks never skip ahead early
    assert window._format_hms(1.49) == "00:00:01"
    assert window._format_hms(1.51) == "00:00:01"
    assert window._format_hms(1.99) == "00:00:01"
    assert window._format_hms(1.99, round_final=True) == "00:00:02"

    # Simulated phase-aligned wakeup computes next delay to the upcoming whole-second boundary
    now = [100.0]
    monkeypatch.setattr("focus_stack_app.ui.main_window.time.monotonic", lambda: now[0])
    window._start_timestamp = 100.0
    window._last_displayed_second = 0

    now[0] = 101.02
    window._update_timer()
    assert window.timer_label.text() == "00:00:01"
    remaining_ms = window._elapsed_timer.interval()
    assert 960 <= remaining_ms <= 1010

    now[0] = 102.30  # simulate momentary UI stall: timer reschedules shorter delay to re-align to 3.0s
    window._update_timer()
    assert window.timer_label.text() == "00:00:02"
    assert 680 <= window._elapsed_timer.interval() <= 730
    window._elapsed_timer.stop()
    window.close()
