"""PySide6 main window for the Focus Stack Studio workflow.

The window collects options, provides drag-and-drop import, folder inspection,
and starts an :class:`ApplicationController`. Scanner, OpenCV, SQLite writes,
archive I/O, and fusion are all performed by its worker thread; Qt receives
immutable events through a queued signal bridge.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Iterable

from ..pipeline.application_controller import ApplicationController, ApplicationOptions
from ..pipeline.coordinator import PipelineCoordinator, PipelineSummary
from ..pipeline.events import PipelineEvent


def default_controller_factory(**settings: Any) -> ApplicationController:
    """Build the production headless controller used by a bare window."""
    return ApplicationController(settings)


try:  # PySide6 remains optional for headless tests and package imports.
    from PySide6.QtCore import QObject, QSize, Qt, QTimer, QUrl, Signal, Slot
    from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent, QIcon, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
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
        """Modern Studio workspace for desktop focus stacking."""

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
            self.setWindowTitle("Focus Stack Assistant - 景深合成工作台")
            self.resize(1180, 780)
            self.setMinimumSize(960, 640)
            self.setAcceptDrops(True)
            self._stopping = False
            self._elapsed_timer = QTimer(self)
            self._elapsed_timer.setInterval(1000)
            self._elapsed_timer.timeout.connect(self._update_timer)
            self._start_timestamp: float | None = None

            apply_theme(self)
            self.controller = controller
            self._controller_was_injected = controller is not None
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
            layout.setContentsMargins(20, 16, 20, 8)
            layout.setSpacing(14)

            # --- Top Header Bar ---
            heading = QHBoxLayout()
            heading.setSpacing(12)

            logo_title_col = QVBoxLayout()
            logo_title_col.setSpacing(2)

            brand_row = QHBoxLayout()
            brand_row.setSpacing(8)
            title = QLabel("Focus Stack Studio / 景深合成工作台")
            title.setProperty("role", "brand")
            brand_row.addWidget(title)

            sub_title = QLabel("全自动镜头焦点堆栈 · 智能场景识别与多频多分辨率深度融合")
            sub_title.setProperty("role", "muted")

            logo_title_col.addLayout(brand_row)
            logo_title_col.addWidget(sub_title)
            heading.addLayout(logo_title_col)

            heading.addStretch()

            # Stopwatch Timer Pill
            timer_widget = QWidget()
            timer_layout = QHBoxLayout(timer_widget)
            timer_layout.setContentsMargins(10, 3, 10, 3)
            timer_layout.setSpacing(6)
            timer_widget.setStyleSheet(
                "background: #111E2E; border: 1px solid #1E3A5F; border-radius: 12px;"
            )
            timer_icon = QLabel()
            icon_path = Path(__file__).parent / "icons" / "clock.svg"
            if icon_path.exists():
                timer_icon.setPixmap(QPixmap(str(icon_path)).scaled(13, 13, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.timer_label = QLabel("00:00:00")
            self.timer_label.setStyleSheet(
                "color: #38BDF8; font-family: 'Consolas', 'Segoe UI Mono', monospace; font-size: 12px; font-weight: 600;"
            )
            timer_layout.addWidget(timer_icon)
            timer_layout.addWidget(self.timer_label)
            heading.addWidget(timer_widget)

            # Status Badge Pill
            self.run_status = QLabel("就绪")
            self.run_status.setProperty("role", "badge")
            heading.addWidget(self.run_status)
            layout.addLayout(heading)

            # --- Main Body (Sidebar + Right Work Area) ---
            body = QHBoxLayout()
            body.setSpacing(16)
            layout.addLayout(body, 1)

            # Left Sidebar
            sidebar = QFrame()
            sidebar.setObjectName("panel")
            sidebar.setFixedWidth(340)
            left = QVBoxLayout(sidebar)
            left.setContentsMargins(0, 0, 0, 14)

            self.settings_scroll = QScrollArea()
            self.settings_scroll.setWidgetResizable(True)
            self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.settings_content = QWidget()
            self.settings_content.setObjectName("settingsContent")
            settings_layout = QVBoxLayout(self.settings_content)
            settings_layout.setContentsMargins(16, 16, 16, 16)
            settings_layout.setSpacing(16)
            self.settings_scroll.setWidget(self.settings_content)
            left.addWidget(self.settings_scroll, 1)
            body.addWidget(sidebar)

            # Right Layout
            right = QVBoxLayout()
            right.setSpacing(14)
            body.addLayout(right, 1)

            def section(title_text, parent_layout):
                container = QWidget()
                section_layout = QVBoxLayout(container)
                section_layout.setContentsMargins(0, 0, 0, 0)
                section_layout.setSpacing(10)
                label = QLabel(title_text)
                label.setProperty("role", "section")
                section_layout.addWidget(label)
                form = QFormLayout()
                form.setRowWrapPolicy(QFormLayout.WrapAllRows)
                form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
                form.setVerticalSpacing(8)
                section_layout.addLayout(form)
                parent_layout.addWidget(container)
                return form

            # --- Section 1: File Locations & Drop Zone ---
            files_form = section("文件位置与导入", settings_layout)

            # Source directory
            self.source_edit = QLineEdit()
            self.source_edit.setObjectName("sourceDirectory")
            self.source_edit.setPlaceholderText("选择或直接拖拽照片文件夹至此")

            source_btn = QPushButton("选择…")
            source_btn.setFixedWidth(64)
            source_btn.clicked.connect(self.choose_source)

            source_open_btn = QPushButton("打开")
            source_open_btn.setFixedWidth(52)
            source_open_btn.setToolTip("在系统文件管理器中打开源文件夹")
            source_open_btn.clicked.connect(self._open_source_folder)

            source_row = QHBoxLayout()
            source_row.setSpacing(6)
            source_row.addWidget(self.source_edit)
            source_row.addWidget(source_btn)
            source_row.addWidget(source_open_btn)
            files_form.addRow("原照片文件夹", source_row)

            # Instant Folder Inspection Stats Pill
            self.folder_stats_label = QLabel("可将照片文件夹直接拖入本窗口")
            self.folder_stats_label.setProperty("role", "muted")
            self.folder_stats_label.setStyleSheet("color: #60A5FA; font-size: 11px;")
            files_form.addRow(self.folder_stats_label)

            # Output directory
            self.output_edit = QLineEdit()
            self.output_edit.setObjectName("outputDirectory")
            self.output_edit.setPlaceholderText("默认：原图片文件夹内的“合成”文件夹")

            output_btn = QPushButton("选择…")
            output_btn.setFixedWidth(64)
            output_btn.clicked.connect(self.choose_output)

            output_open_btn = QPushButton("打开")
            output_open_btn.setFixedWidth(52)
            output_open_btn.setToolTip("在系统文件管理器中打开合成结果文件夹")
            output_open_btn.clicked.connect(self._open_output_folder)

            output_row = QHBoxLayout()
            output_row.setSpacing(6)
            output_row.addWidget(self.output_edit)
            output_row.addWidget(output_btn)
            output_row.addWidget(output_open_btn)
            files_form.addRow("合成结果目录", output_row)

            # Archive options
            self.archive_edit = QLineEdit()
            self.archive_edit.setObjectName("archiveDirectory")
            self.archive_edit.setPlaceholderText("默认：原图片文件夹内的“归档”文件夹")
            archive_btn = QPushButton("选择…")
            archive_btn.setFixedWidth(64)
            archive_btn.clicked.connect(self.choose_archive)
            archive_row = QHBoxLayout()
            archive_row.setSpacing(6)
            archive_row.addWidget(self.archive_edit)
            archive_row.addWidget(archive_btn)

            self.archive_content = QWidget()
            archive_form = QFormLayout(self.archive_content)
            archive_form.setContentsMargins(0, 4, 0, 0)
            archive_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
            archive_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            archive_form.addRow("原图归档目录", archive_row)

            self.archive_combo = QComboBox()
            self.archive_combo.setObjectName("archiveMode")
            self.archive_combo.addItem("移动（推荐整理模式）", "move")
            self.archive_combo.addItem("复制（保留源文件）", "copy")
            self.archive_combo.addItem("硬链接（极速且省磁盘空间）", "hardlink")
            archive_form.addRow("归档方式", self.archive_combo)

            self.archive_check = QCheckBox("合成成功后自动归档原图")
            self.archive_check.setObjectName("archiveEnabled")
            self.archive_check.setChecked(False)
            self.archive_check.setToolTip("默认关闭：合成完成后保留原图不变，便于人工校对与二次调整。")
            files_form.addRow(self.archive_check)
            files_form.addRow(self.archive_content)
            self.archive_content.hide()
            self.archive_check.toggled.connect(self.archive_content.setVisible)

            # --- Section 2: Stacking Parameters ---
            basic_form = section("合成与分组设置", settings_layout)

            self.minimum_group_spin = QSpinBox()
            self.minimum_group_spin.setObjectName("minimumStackGroupSize")
            self.minimum_group_spin.setRange(2, 1000)
            self.minimum_group_spin.setValue(4)
            self.minimum_group_spin.setSuffix(" 张")
            self.minimum_group_spin.setToolTip("同组至少达到此数量的可用相似照片才合成；实际选片仍按清晰度覆盖率决定。")
            basic_form.addRow("最少相似照片数", self.minimum_group_spin)

            self.grouping_pause_spin = QSpinBox()
            self.grouping_pause_spin.setObjectName("groupingPauseSeconds")
            self.grouping_pause_spin.setRange(1, 3600)
            self.grouping_pause_spin.setValue(20)
            self.grouping_pause_spin.setSuffix(" 秒")
            self.grouping_pause_spin.setToolTip("相邻照片间隔达到此值时，结合画面变化分组；超过较大值时强制切分组。")
            basic_form.addRow("分组拍摄间隔", self.grouping_pause_spin)

            self.format_combo = QComboBox()
            self.format_combo.setObjectName("outputFormat")
            self.format_combo.addItem("JPG（100% 极高品质）", "jpg")
            self.format_combo.addItem("TIFF（无损专业打印）", "tiff")
            basic_form.addRow("输出格式", self.format_combo)

            # --- Section 3: Advanced & Performance ---
            self.advanced_toggle = QPushButton("高级与性能设置 · 展开")
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
                lambda checked: self.advanced_toggle.setText("高级与性能设置 · 收起" if checked else "高级与性能设置 · 展开")
            )

            self.backend_combo = QComboBox()
            self.backend_combo.setObjectName("fusionBackend")
            self.backend_combo.addItem("高质量合成引擎（全分辨率内存融合·推荐）", "quality")
            self.backend_combo.addItem("实验模式（Hugin 对齐 / Enfuse 合成）", "hugin_enfuse")
            self.backend_combo.setToolTip("高质量：全分辨率内存对齐与边缘一致性融合；实验：Hugin 对齐及 Enfuse 合成。")
            self.backend_combo.setCurrentIndex(0)
            advanced_form.addRow("合成引擎", self.backend_combo)

            self.merge_workers_spin = QSpinBox()
            self.merge_workers_spin.setObjectName("mergeWorkers")
            self.merge_workers_spin.setRange(1, 6)
            self.merge_workers_spin.setValue(3)
            self.merge_workers_spin.setToolTip("同时处理的合成组数；组数越多，占用的内存和磁盘带宽越高。")
            advanced_form.addRow("并发合成组数", self.merge_workers_spin)

            self.focus_analysis_workers_spin = QSpinBox()
            self.focus_analysis_workers_spin.setObjectName("focusAnalysisWorkers")
            self.focus_analysis_workers_spin.setRange(0, 10)
            self.focus_analysis_workers_spin.setSpecialValueText("自动推荐")
            self.focus_analysis_workers_spin.setValue(0)
            self.focus_analysis_workers_spin.setSuffix(" 张")
            self.focus_analysis_workers_spin.setToolTip("同时进行焦点分析的最大图片数；自动会根据 CPU 核心数与可用内存配置。")
            advanced_form.addRow("并发焦点分析", self.focus_analysis_workers_spin)

            self.parallel_check = QCheckBox("分类与合成流水线并行")
            self.parallel_check.setChecked(True)
            self.parallel_check.setObjectName("parallelPipeline")

            self.cache_check = QCheckBox("保留焦点分析缓存")
            self.cache_check.setChecked(False)
            self.cache_check.setObjectName("preserveCache")

            checks = QVBoxLayout()
            checks.setSpacing(6)
            checks.addWidget(self.parallel_check)
            checks.addWidget(self.cache_check)
            advanced_form.addRow("优化选项", checks)

            settings_layout.addStretch()

            for edit in (self.source_edit, self.output_edit, self.archive_edit):
                edit.setMinimumWidth(0)
                edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
                edit.textChanged.connect(edit.setToolTip)
            self.source_edit.textChanged.connect(self._inspect_source_folder)

            # Action Buttons
            buttons = QHBoxLayout()
            buttons.setSpacing(10)
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

            # --- Right Panels ---
            self.progress_panel = ProgressPanel(self)
            right.addWidget(self.progress_panel)

            self.group_list = GroupListWidget(self)
            right.addWidget(self.group_list, 1)

            self.statusBar().showMessage("系统就绪，请选择照片文件夹")

        def _inspect_source_folder(self, text: str):
            path_str = text.strip()
            if not path_str:
                self.folder_stats_label.setText("可将照片文件夹直接拖入本窗口")
                return
            path = Path(path_str)
            if not path.is_dir():
                self.folder_stats_label.setText("目录不存在")
                return

            # Lightweight scan of top-level or immediate photos
            photo_exts = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".arw", ".cr2", ".cr3", ".nef", ".dng", ".orf", ".rw2"}
            try:
                count = 0
                total_bytes = 0
                for entry in os.scandir(path):
                    if entry.is_file():
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in photo_exts:
                            count += 1
                            total_bytes += entry.stat().st_size
                if count > 0:
                    size_mb = total_bytes / (1024 * 1024)
                    self.folder_stats_label.setText(f"📸 发现 {count} 张照片 · 约 {size_mb:.1f} MB")
                else:
                    self.folder_stats_label.setText("未在顶层检测到常见格式照片")
            except Exception:
                self.folder_stats_label.setText("就绪")

        def _open_source_folder(self):
            path_str = self.source_edit.text().strip()
            if path_str and Path(path_str).is_dir():
                QDesktopServices.openUrl(QUrl.fromLocalFile(path_str))

        def _open_output_folder(self):
            out_dir = self.output_directory
            if out_dir.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))

        # Drag and Drop handlers
        def dragEnterEvent(self, event: QDragEnterEvent) -> None:
            if event.mimeData().hasUrls():
                for url in event.mimeData().urls():
                    if Path(url.toLocalFile()).is_dir():
                        event.acceptProposedAction()
                        return
            event.ignore()

        def dropEvent(self, event: QDropEvent) -> None:
            for url in event.mimeData().urls():
                local_path = url.toLocalFile()
                if Path(local_path).is_dir():
                    self._set_source_path(local_path)
                    event.acceptProposedAction()
                    return

        def _set_source_path(self, value: str):
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

        @property
        def source_directory(self) -> Path:
            return Path(self.source_edit.text().strip())

        @property
        def output_directory(self) -> Path:
            value = self.output_edit.text().strip()
            return Path(value) if value else self.source_directory / "合成"

        @property
        def archive_directory(self) -> Path:
            value = self.archive_edit.text().strip()
            return Path(value) if value else self.source_directory / "归档"

        def choose_source(self) -> None:
            value = QFileDialog.getExistingDirectory(self, "选择源文件夹", self.source_edit.text())
            if value:
                self._set_source_path(value)

        def choose_output(self) -> None:
            value = QFileDialog.getExistingDirectory(self, "选择合成结果目录", self.output_edit.text())
            if value:
                self.output_edit.setText(value)

        def choose_archive(self) -> None:
            value = QFileDialog.getExistingDirectory(self, "选择原图归档目录", self.archive_edit.text())
            if value:
                self.archive_edit.setText(value)

        def set_groups(self, groups: Iterable[Any]) -> None:
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
            }

        @staticmethod
        def _retire_worker(worker: Any) -> None:
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
                pass

        def _update_timer(self):
            if self._start_timestamp is not None:
                elapsed = int(time.time() - self._start_timestamp)
                mins, secs = divmod(elapsed, 60)
                hours, mins = divmod(mins, 60)
                self.timer_label.setText(f"{hours:02d}:{mins:02d}:{secs:02d}")

        @Slot()
        def start_processing(self) -> None:
            if not self.source_edit.text().strip():
                QMessageBox.warning(self, "缺少目录", "请先选择源文件夹。")
                return
            try:
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
            self._start_timestamp = time.time()
            self._elapsed_timer.start()
            self.settings_content.setEnabled(False)
            self.progress_panel.reset()
            self.group_list.set_empty_message("处理结束后显示结果")
            self.start_button.setEnabled(False)
            self.start_button.setText("正在处理…")
            self.stop_button.setEnabled(True)
            self.run_status.setText("处理中")
            self.statusBar().showMessage("流水线处理中…")

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
                self.statusBar().showMessage(
                    self.fontMetrics().elidedText(
                        message.splitlines()[0], Qt.ElideMiddle, max(100, self.statusBar().width() - 32)
                    )
                )

        @Slot(object)
        def _on_complete(self, summary: PipelineSummary) -> None:
            self._stopping = False
            self._elapsed_timer.stop()
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
                self.statusBar().showMessage("景深合成处理完成")

        def closeEvent(self, event: Any) -> None:
            self._elapsed_timer.stop()
            if self.controller is not None and self.controller.running:
                self.controller.cancel()
                self.controller.wait(5)
            elif self.coordinator is not None and self.coordinator.running:
                self.coordinator.cancel()
                self.coordinator.wait(5)
            if self.controller is not None:
                self.controller.close()
            event.accept()

except ImportError:  # pragma: no cover

    class MainWindow:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError("PySide6 is required to create the desktop UI")


__all__ = ["MainWindow", "default_controller_factory"]

