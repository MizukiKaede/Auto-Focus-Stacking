"""PySide6 main window for the Focus Stack V1 workflow.

The window only collects options and starts an :class:`ApplicationController`.
Scanner, OpenCV, SQLite writes, archive I/O, and Hugin execution are all
performed by its worker thread; Qt receives immutable events through a queued
signal bridge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from ..pipeline.application_controller import ApplicationController, ApplicationOptions
from ..pipeline.coordinator import PipelineCoordinator, PipelineSummary
from ..pipeline.events import PipelineEvent


def default_controller_factory(**settings: Any) -> ApplicationController:
    """Build the production headless controller used by a bare window.

    Keeping this factory outside the Qt import block makes it available to
    command-line and headless smoke tests as well.
    """

    # Let the controller's mapping adapter accept both the UI's canonical
    # keys and the short aliases used by embedding/headless callers.
    return ApplicationController(settings)


try:  # PySide6 remains optional for headless tests and package imports.
    from PySide6.QtCore import QObject, Qt, Signal, Slot
    from PySide6.QtWidgets import (
        QFrame,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    from .group_list import GroupListWidget
    from .progress_panel import ProgressPanel
    from .theme import apply_theme

    class _SignalBridge(QObject):
        event = Signal(object)
        completed = Signal(object)

    class MainWindow(QMainWindow):
        """Dark two-column workspace for the desktop processing workflow."""

        def __init__(
            self,
            *,
            controller: ApplicationController | None = None,
            controller_factory: Callable[..., ApplicationController] | None = None,
            coordinator: PipelineCoordinator | None = None,
            coordinator_factory: Callable[..., PipelineCoordinator] | None = None,
            parent: QWidget | None = None,
        ):
            super().__init__(parent)
            self.setWindowTitle("Focus Stack Assistant")
            self.resize(1180, 780)
            self.setMinimumSize(960, 640)
            self._stopping = False
            apply_theme(self)
            self.controller = controller
            self._controller_was_injected = controller is not None
            # An explicitly supplied legacy coordinator factory keeps its old
            # contract.  A completely bare window, however, gets the real
            # application controller by default.
            self.controller_factory = (
                controller_factory
                if controller_factory is not None
                else (default_controller_factory if controller is None and coordinator_factory is None else None)
            )
            self.coordinator = coordinator
            self.coordinator_factory = coordinator_factory
            self._groups: list[Any] = []
            self._bridge = _SignalBridge(self)
            self._bridge.event.connect(self._on_progress)
            self._bridge.completed.connect(self._on_complete)
            self._build_ui()

        def _build_ui(self) -> None:
            root = QWidget(self)
            root.setObjectName("workspace")
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)
            layout.setContentsMargins(24, 20, 24, 8)
            layout.setSpacing(20)
            heading = QHBoxLayout()
            title = QLabel("Focus Stack  /  景深合成")
            title.setProperty("role", "title")
            heading.addWidget(title)
            heading.addStretch()
            self.run_status = QLabel("就绪")
            self.run_status.setProperty("role", "badge")
            heading.addWidget(self.run_status)
            layout.addLayout(heading)
            body = QHBoxLayout()
            body.setSpacing(20)
            layout.addLayout(body, 1)
            sidebar = QFrame()
            sidebar.setObjectName("panel")
            sidebar.setFixedWidth(340)
            left = QVBoxLayout(sidebar)
            left.setContentsMargins(0, 0, 0, 16)
            self.settings_scroll = QScrollArea()
            self.settings_scroll.setWidgetResizable(True)
            self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.settings_content = QWidget()
            self.settings_content.setObjectName("settingsContent")
            settings_layout = QVBoxLayout(self.settings_content)
            settings_layout.setContentsMargins(16, 16, 16, 16)
            settings_layout.setSpacing(20)
            self.settings_scroll.setWidget(self.settings_content)
            left.addWidget(self.settings_scroll, 1)
            body.addWidget(sidebar)
            right = QVBoxLayout()
            right.setSpacing(16)
            body.addLayout(right, 1)

            def section(title, parent_layout):
                container = QWidget()
                section_layout = QVBoxLayout(container)
                section_layout.setContentsMargins(0, 0, 0, 0)
                section_layout.setSpacing(12)
                label = QLabel(title)
                label.setProperty("role", "section")
                section_layout.addWidget(label)
                form = QFormLayout()
                form.setRowWrapPolicy(QFormLayout.WrapAllRows)
                form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
                form.setVerticalSpacing(8)
                section_layout.addLayout(form)
                parent_layout.addWidget(container)
                return form

            files_form = section("文件位置", settings_layout)
            basic_form = section("合成设置", settings_layout)
            self.advanced_toggle = QPushButton("高级设置 · 展开")
            self.advanced_toggle.setCheckable(True)
            settings_layout.addWidget(self.advanced_toggle)
            self.advanced_content = QWidget()
            advanced_layout = QVBoxLayout(self.advanced_content)
            advanced_layout.setContentsMargins(0, 0, 0, 0)
            advanced_form = QFormLayout()
            advanced_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
            advanced_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            advanced_form.setVerticalSpacing(8)
            advanced_layout.addLayout(advanced_form)
            settings_layout.addWidget(self.advanced_content)
            self.advanced_content.hide()
            self.advanced_toggle.toggled.connect(self.advanced_content.setVisible)
            self.advanced_toggle.toggled.connect(
                lambda checked: self.advanced_toggle.setText("高级设置 · 收起" if checked else "高级设置 · 展开")
            )
            settings_layout.addStretch()

            self.source_edit = QLineEdit()
            self.source_edit.setObjectName("sourceDirectory")
            source_button = QPushButton("选择…")
            source_button.setFixedWidth(68)
            source_button.clicked.connect(self.choose_source)
            source_row = QHBoxLayout()
            source_row.addWidget(self.source_edit)
            source_row.addWidget(source_button)
            files_form.addRow("源文件夹", source_row)

            self.output_edit = QLineEdit()
            self.output_edit.setObjectName("outputDirectory")
            self.output_edit.setPlaceholderText("默认：原图片文件夹内的“合成”文件夹")
            output_button = QPushButton("选择…")
            output_button.setFixedWidth(68)
            output_button.clicked.connect(self.choose_output)
            output_row = QHBoxLayout()
            output_row.addWidget(self.output_edit)
            output_row.addWidget(output_button)
            files_form.addRow("合成结果目录", output_row)

            self.archive_edit = QLineEdit()
            self.archive_edit.setObjectName("archiveDirectory")
            self.archive_edit.setPlaceholderText("默认：原图片文件夹内的“归档”文件夹")
            archive_button = QPushButton("选择…")
            archive_button.setFixedWidth(68)
            archive_button.clicked.connect(self.choose_archive)
            archive_row = QHBoxLayout()
            archive_row.addWidget(self.archive_edit)
            archive_row.addWidget(archive_button)
            self.archive_content = QWidget()
            archive_form = QFormLayout(self.archive_content)
            archive_form.setContentsMargins(0, 0, 0, 0)
            archive_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
            archive_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            archive_form.addRow("原图归档目录", archive_row)

            self.archive_combo = QComboBox()
            self.archive_combo.setObjectName("archiveMode")
            self.archive_combo.addItem("移动", "move")
            self.archive_combo.addItem("复制", "copy")
            self.archive_combo.addItem("硬链接", "hardlink")
            archive_form.addRow("归档方式", self.archive_combo)

            self.archive_check = QCheckBox("合成成功后归档原图")
            self.archive_check.setObjectName("archiveEnabled")
            self.archive_check.setChecked(False)
            self.archive_check.setToolTip("默认关闭：合成完成后仍保留原图，便于核对分组和结果。")
            files_form.addRow(self.archive_check)
            files_form.addRow(self.archive_content)
            self.archive_content.hide()
            self.archive_check.toggled.connect(self.archive_content.setVisible)

            self.backend_combo = QComboBox()
            self.backend_combo.setObjectName("fusionBackend")
            self.backend_combo.addItem("高质量合成（推荐）", "hugin_enfuse")
            self.backend_combo.addItem("快速合成（实验）", "opencv")
            self.backend_combo.setToolTip("高质量：Hugin + Enfuse；快速：OpenCV。所有引擎使用相同的自动分组、人工覆盖与智能选片。标准引擎失败时不会静默切换。")
            self.backend_combo.setCurrentIndex(0)
            advanced_form.addRow("合成引擎", self.backend_combo)

            self.minimum_group_spin = QSpinBox()
            self.minimum_group_spin.setObjectName("minimumStackGroupSize")
            self.minimum_group_spin.setRange(2, 1000)
            self.minimum_group_spin.setValue(3)
            self.minimum_group_spin.setSuffix(" 张")
            self.minimum_group_spin.setToolTip("同组至少达到此数量的可用相似照片才合成；实际选片仍按清晰度覆盖率决定。")
            basic_form.addRow("最少相似照片数", self.minimum_group_spin)
            self.grouping_pause_spin = QSpinBox()
            self.grouping_pause_spin.setObjectName("groupingPauseSeconds")
            self.grouping_pause_spin.setRange(1, 3600)
            self.grouping_pause_spin.setValue(20)
            self.grouping_pause_spin.setSuffix(" 秒")
            self.grouping_pause_spin.setToolTip("相邻照片间隔达到此值时，结合画面变化分组；主体明显转面或移动仍会独立分组。超过较大值（60 秒或此设置）时强制分组。修改后下次开始生效。")
            basic_form.addRow("分组拍摄间隔", self.grouping_pause_spin)

            self.format_combo = QComboBox()
            self.format_combo.setObjectName("outputFormat")
            self.format_combo.addItem("JPG（100%质量）", "jpg")
            self.format_combo.addItem("TIFF", "tiff")
            basic_form.addRow("输出格式", self.format_combo)

            self.merge_workers_spin = QSpinBox()
            self.merge_workers_spin.setObjectName("mergeWorkers")
            self.merge_workers_spin.setRange(1, 2)
            self.merge_workers_spin.setValue(1)
            self.merge_workers_spin.setToolTip("同时处理的合成组数；2 组会增加内存和磁盘负载。")
            advanced_form.addRow("同时合成组数", self.merge_workers_spin)

            self.focus_analysis_workers_spin = QSpinBox()
            self.focus_analysis_workers_spin.setObjectName("focusAnalysisWorkers")
            self.focus_analysis_workers_spin.setRange(0, 8)
            self.focus_analysis_workers_spin.setSpecialValueText("自动")
            self.focus_analysis_workers_spin.setValue(0)
            self.focus_analysis_workers_spin.setSuffix(" 张")
            self.focus_analysis_workers_spin.setToolTip(
                "同时进行焦点分析的最大图片数；自动会根据 CPU、合成线程和可用内存选择，"
                "手动值仍受内存安全保护。"
            )
            advanced_form.addRow("同时焦点分析", self.focus_analysis_workers_spin)

            self.parallel_check = QCheckBox("分类与合成并行")
            self.parallel_check.setChecked(True)
            self.parallel_check.setObjectName("parallelPipeline")
            self.cache_check = QCheckBox("保留分析缓存")
            self.cache_check.setChecked(True)
            self.cache_check.setObjectName("preserveCache")
            checks = QVBoxLayout()
            checks.addWidget(self.parallel_check)
            checks.addWidget(self.cache_check)
            checks.addStretch(1)
            advanced_form.addRow("选项", checks)

            # Optional executable overrides are plain text fields so the UI
            # remains small and works with both a Hugin bin directory and
            # separately installed tools.  Empty values use PATH discovery.
            self.hugin_edit = QLineEdit()
            self.hugin_edit.setObjectName("huginBin")
            self.hugin_edit.setPlaceholderText("可选：Hugin bin 目录")
            advanced_form.addRow("Hugin bin", self.hugin_edit)
            self.align_edit = QLineEdit()
            self.align_edit.setObjectName("alignImageStackPath")
            self.align_edit.setPlaceholderText("可选：align_image_stack.exe")
            advanced_form.addRow("align_image_stack", self.align_edit)
            self.enfuse_edit = QLineEdit()
            self.enfuse_edit.setObjectName("enfusePath")
            self.enfuse_edit.setPlaceholderText("可选：enfuse.exe")
            advanced_form.addRow("enfuse", self.enfuse_edit)
            for edit in (self.source_edit, self.output_edit, self.archive_edit,
                         self.hugin_edit, self.align_edit, self.enfuse_edit):
                edit.setMinimumWidth(0)
                edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
                edit.textChanged.connect(edit.setToolTip)
            self.source_edit.setPlaceholderText("选择照片所在文件夹")
            self.output_edit.setPlaceholderText("源文件夹内的“合成”文件夹")
            self.archive_edit.setPlaceholderText("源文件夹内的“归档”文件夹")

            buttons = QHBoxLayout()
            self.start_button = QPushButton("开始处理")
            self.start_button.setObjectName("startButton")
            self.start_button.clicked.connect(self.start_processing)
            self.stop_button = QPushButton("停止")
            self.stop_button.setObjectName("stopButton")
            self.stop_button.setEnabled(False)
            self.stop_button.clicked.connect(self.stop_processing)
            buttons.setContentsMargins(16, 0, 16, 0)
            buttons.addWidget(self.start_button, 1)
            buttons.addWidget(self.stop_button)
            left.addLayout(buttons)
            self.progress_panel = ProgressPanel(self)
            right.addWidget(self.progress_panel)
            self.group_list = GroupListWidget(self)
            right.addWidget(self.group_list, 1)
            self.statusBar().showMessage("就绪")

        @property
        def source_directory(self) -> Path:
            return Path(self.source_edit.text().strip())

        @property
        def output_directory(self) -> Path:
            return Path(self.output_edit.text().strip())

        @property
        def archive_directory(self) -> Path:
            value = self.archive_edit.text().strip()
            return Path(value) if value else self.source_directory / "归档"

        def choose_source(self) -> None:
            value = QFileDialog.getExistingDirectory(self, "选择源文件夹", self.source_edit.text())
            if value:
                previous_source = self.source_edit.text().strip()
                output_value = self.output_edit.text().strip()
                archive_value = self.archive_edit.text().strip()
                output_uses_default = not output_value or (
                    bool(previous_source) and Path(output_value) == Path(previous_source) / "合成"
                )
                archive_uses_default = not archive_value or (
                    bool(previous_source) and Path(archive_value) == Path(previous_source) / "归档"
                )
                self.source_edit.setText(value)
                source = Path(value)
                if output_uses_default:
                    self.output_edit.setText(str(source / "合成"))
                if archive_uses_default:
                    self.archive_edit.setText(str(source / "归档"))

        def choose_output(self) -> None:
            value = QFileDialog.getExistingDirectory(self, "选择合成结果目录", self.output_edit.text())
            if value:
                self.output_edit.setText(value)

        def choose_archive(self) -> None:
            value = QFileDialog.getExistingDirectory(self, "选择原图归档目录", self.archive_edit.text())
            if value:
                self.archive_edit.setText(value)

        def set_groups(self, groups: Iterable[Any]) -> None:
            # Group metadata is lightweight; no image pixels are retained.
            self._groups = list(groups)

        def _settings(self) -> dict[str, Any]:
            return {
                "source_dir": self.source_directory,
                "output_dir": self.output_directory,
                "archive_dir": self.archive_directory,
                "archive_mode": self.archive_combo.currentData(),
                "archive_enabled": self.archive_check.isChecked(),
                "fusion_backend": self.backend_combo.currentData(),
                "minimum_stack_group_size": self.minimum_group_spin.value(),
                "grouping_pause_seconds": self.grouping_pause_spin.value(),
                "output_format": self.format_combo.currentData(),
                "parallel": self.parallel_check.isChecked(),
                "merge_workers": self.merge_workers_spin.value(),
                "focus_analysis_workers": self.focus_analysis_workers_spin.value(),
                "preserve_cache": self.cache_check.isChecked(),
                "hugin_bin": self.hugin_edit.text().strip() or None,
                "align_image_stack_path": self.align_edit.text().strip() or None,
                "enfuse_path": self.enfuse_edit.text().strip() or None,
            }

        @staticmethod
        def _retire_worker(worker: Any) -> None:
            """Close a previous controller/coordinator before rebuilding."""

            if worker is None:
                return
            try:
                if bool(getattr(worker, "running", False)):
                    cancel = getattr(worker, "cancel", getattr(worker, "stop", None))
                    if callable(cancel):
                        cancel()
                    wait = getattr(worker, "wait", getattr(worker, "join", None))
                    if callable(wait):
                        wait(5)
                close = getattr(worker, "close", None)
                if callable(close):
                    close()
            except Exception:
                # A stale worker must not prevent a fresh UI run from being
                # constructed; its own cancellation path remains best effort.
                pass

        @Slot()
        def start_processing(self) -> None:
            if not self.source_edit.text().strip() or not self.output_edit.text().strip():
                QMessageBox.warning(self, "缺少目录", "请先选择源文件夹和输出目录。")
                return
            try:
                # A completed/cancelled controller retains its cancellation
                # Event and worker-owned DB/cache state.  Rebuild on every
                # factory-backed start so current UI edits apply and no stale
                # cancellation leaks into the next batch.
                if self.controller_factory is not None:
                    self._retire_worker(self.controller)
                    self.controller = self.controller_factory(**self._settings())
                elif self.coordinator_factory is not None:
                    self._retire_worker(self.coordinator)
                    self.coordinator = self.coordinator_factory(**self._settings())
                elif self.controller is None and self.coordinator is None:
                    raise RuntimeError("未配置 ApplicationController 工厂")
                if self.controller is not None:
                    self.controller.event_callback = self._bridge.event.emit
                    self.controller.complete_callback = self._bridge.completed.emit
                    self.controller.start()
                else:
                    # Backward-compatible injection point for the lower-level
                    # coordinator used by older scripts/tests.
                    self.coordinator.event_callback = self._bridge.event.emit
                    self.coordinator.complete_callback = self._bridge.completed.emit
                    self.coordinator.start(self._groups)
            except Exception as exc:
                self.run_status.setText("无法开始")
                self.statusBar().showMessage("无法开始处理，请检查设置")
                QMessageBox.critical(self, "无法开始", str(exc))
                if self.controller_factory is not None:
                    self._retire_worker(self.controller)
                    self.controller = None
                if self.coordinator_factory is not None:
                    self._retire_worker(self.coordinator)
                    self.coordinator = None
                return
            self._stopping = False
            self.settings_content.setEnabled(False)
            self.progress_panel.reset()
            self.group_list.set_empty_message("处理结束后显示结果")
            self.start_button.setEnabled(False)
            self.start_button.setText("正在处理…")
            self.stop_button.setEnabled(True)
            self.run_status.setText("处理中")
            self.statusBar().showMessage("处理中…")

        @Slot()
        def stop_processing(self) -> None:
            if self.controller is not None:
                self.controller.cancel()
            elif self.coordinator is not None:
                self.coordinator.cancel()
            self._stopping = True
            self.stop_button.setEnabled(False)
            self.run_status.setText("正在安全停止…")
            self.statusBar().showMessage("正在安全停止…")

        @Slot(object)
        def _on_progress(self, event: PipelineEvent) -> None:
            self.progress_panel.update_event(event)
            if not self._stopping and getattr(event, "current_file", ""):
                message = str(event.message or event.current_file)
                self.statusBar().setToolTip(message)
                self.statusBar().showMessage(self.fontMetrics().elidedText(
                    message.splitlines()[0], Qt.ElideMiddle, max(100, self.statusBar().width() - 32)
                ))

        @Slot(object)
        def _on_complete(self, summary: PipelineSummary) -> None:
            self._stopping = False
            self.settings_content.setEnabled(True)
            self.start_button.setEnabled(True)
            self.start_button.setText("开始处理")
            self.stop_button.setEnabled(False)
            complete = getattr(self.progress_panel, "complete", None)
            if callable(complete):
                complete(summary)
            self.group_list.update_results(summary.results)
            if not summary.results:
                self.group_list.set_empty_message("已取消，本次没有结果" if summary.cancelled else "本次没有可显示的结果")
            if summary.cancelled:
                self.run_status.setText("已取消")
                self.statusBar().showMessage("已取消")
            elif int(getattr(summary, "errors", 0) or 0) > 0:
                count = int(getattr(summary, "errors", 0) or 0)
                self.run_status.setText(f"处理结束 · {count} 个错误")
                self.statusBar().showMessage(f"处理结束：{count} 个错误，失败组原图未移动")
                diagnostics = list(getattr(summary, "diagnostics", ()) or ())
                detail = "\n\n".join(str(item) for item in diagnostics) if diagnostics else "部分图片组未能完成合成；失败组原图未移动。"
                self.progress_panel.set_message(detail)
                QMessageBox.warning(self, "处理未全部完成", "部分图片组未能完成合成；失败组原图未移动。请展开诊断详情查看原因。")
            else:
                self.run_status.setText("处理完成")
                self.statusBar().showMessage("处理完成")

        def closeEvent(self, event: Any) -> None:
            if self.controller is not None and self.controller.running:
                self.controller.cancel()
                self.controller.wait(5)
            elif self.coordinator is not None and self.coordinator.running:
                self.coordinator.cancel()
                self.coordinator.wait(5)
            if self.controller is not None:
                self.controller.close()
            event.accept()

except ImportError:  # pragma: no cover - this environment has no PySide6

    class MainWindow:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError("PySide6 is required to create the desktop UI")


__all__ = ["MainWindow", "default_controller_factory"]

