import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest

pytest.importorskip('PySide6')
from PySide6.QtWidgets import QApplication, QFileDialog
from focus_stack_app.ui.main_window import MainWindow
from focus_stack_app.app import default_controller_factory


def test_pause_control_default_and_factory_settings(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.source_edit.setText(str(tmp_path))
        window.output_edit.setText(str(tmp_path / 'out'))
        assert window.grouping_pause_spin.value() == 20
        window.grouping_pause_spin.setValue(12)
        controller = default_controller_factory(**window._settings())
        try:
            controller._automatic_scene_groups([])
            assert controller._scene_detector.config.pause_seconds == 12
        finally:
            controller.close()
    finally:
        window.close()
        app.processEvents()


def test_source_picker_defaults_output_folders_inside_source(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    selected = iter((tmp_path / 'first', tmp_path / 'second'))
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args, **kwargs: str(next(selected)))
    try:
        window.choose_source()
        assert window.output_directory == tmp_path / 'first' / '合成'
        assert window.archive_directory == tmp_path / 'first' / '归档'

        window.choose_source()
        assert window.output_directory == tmp_path / 'second' / '合成'
        assert window.archive_directory == tmp_path / 'second' / '归档'
    finally:
        window.close()
        app.processEvents()


def test_experimental_picker_uses_hugin_backend(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.source_edit.setText(str(tmp_path))
        window.output_edit.setText(str(tmp_path / 'out'))
        assert window.backend_combo.currentData() == 'quality'
        index = window.backend_combo.findData('hugin_enfuse')
        assert index >= 0
        assert '实验模式' in window.backend_combo.itemText(index)
        window.backend_combo.setCurrentIndex(index)
        controller = default_controller_factory(**window._settings())
        try:
            assert controller.options.fusion_backend == 'hugin_enfuse'
        finally:
            controller.close()
    finally:
        window.close()
        app.processEvents()
