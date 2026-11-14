"""PySide6 main window for the Focus Stack Studio daylight & darkroom workbench."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable

from ..pipeline.application_controller import ApplicationController
from ..pipeline.coordinator import PipelineCoordinator, PipelineSummary
from ..pipeline.events import PipelineEvent


def default_controller_factory(**settings: Any) -> ApplicationController:
    """Build the production headless controller used by a bare window."""
    return ApplicationController(settings)


try:  # PySide6 remains optional for headless tests and package imports.
    from PySide6.QtCore import QObject, QSettings, QSize, Qt, QTimer, QUrl, Signal, Slot
    from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDragLeaveEvent, QDropEvent
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
    from .theme import ThemeTokens, apply_theme, themed_icon, themed_pixmap

    _SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}
    _OTHER_PHOTO_EXTENSIONS = {".tif", ".tiff", ".png", ".arw", ".cr2", ".cr3", ".nef", ".dng", ".orf", ".rw2"}

    class _SignalBridge(QObject):
        event = Signal(object)
        completed = Signal(object)

    class MainWindow(QMainWindow):
        """Daylight-first creative studio workbench for batch focus stacking."""

        def __init__(
            self,
            *,
            controller: ApplicationController | None = None,
            controller_factory: Callable[..., ApplicationController] | None = None,
            coordinator: PipelineCoordinator | None = None,
            coordinator_factory: Callable[..., PipelineCoordinator] | None = None,
            restore_settings: bool = False,
            parent: QWidget | None = None,
        ):
            super().__init__(parent)
            self.setWindowTitle("Focus Stack Studio - 景深合成工作台")
            self.resize(1180, 780)
            self.setMinimumSize(960, 640)
            self.setAcceptDrops(True)

            self._restore_settings = bool(restore_settings)
            self._theme_name = "light"
            self._tokens: ThemeTokens = apply_theme(self, self._theme_name)

            self._stopping = False
            self._elapsed_timer = QTimer(self)
            self._elapsed_timer.setSingleShot(True)
            self._elapsed_timer.setTimerType(Qt.PreciseTimer)
            self._elapsed_timer.timeout.connect(self._update_timer)
            self._start_timestamp: float | None = None
            self._last_displayed_second: int = -1
            self._hugin_probed = False

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
            self._apply_theme_icons()
            self._set_run_status("就绪", tone="idle")

            if self._restore_settings:
                self._load_settings()

        def _build_ui(self) -> None:
            root = QWidget(self)
            root.setObjectName("workspace")
            self.setCentralWidget(root)
            outer = QVBoxLayout(root)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)

            # --- Top Studio Header ---
            header = QWidget()
            header.setObjectName("studioHeader")
            header.setFixedHeight(52)
            heading = QHBoxLayout(header)
            heading.setContentsMargins(18, 0, 18, 0)
            heading.setSpacing(10)

            self._brand_icon = QLabel()
            heading.addWidget(self._brand_icon)

            title = QLabel("Focus Stack Studio")
            title.setProperty("role", "brand")
            heading.addWidget(title)

            sub_title = QLabel("景深合成工作台")
            sub_title.setProperty("role", "muted")
            heading.addWidget(sub_title)

            heading.addStretch()

            # Stopwatch Timer
            timer_wrap = QWidget()
            timer_wrap.setFixedHeight(28)
            timer_layout = QHBoxLayout(timer_wrap)
            timer_layout.setContentsMargins(0, 0, 0, 0)
            timer_layout.setSpacing(6)
            self._clock_icon = QLabel()
            self.timer_label = QLabel("00:00:00")
            self.timer_label.setProperty("role", "timer")
            self.timer_label.setFixedHeight(28)
            self.timer_label.setAlignment(Qt.AlignCenter)
            timer_layout.addWidget(self._clock_icon)
            timer_layout.addWidget(self.timer_label)
            heading.addWidget(timer_wrap, 0, Qt.AlignVCenter)

            # Status Badge Pill
            self.run_status = QLabel("就绪")
            self.run_status.setProperty("role", "badge")
            self.run_status.setFixedHeight(28)
            self.run_status.setAlignment(Qt.AlignCenter)
            heading.addWidget(self.run_status, 0, Qt.AlignVCenter)

            # Light / Darkroom Theme Switcher
            self.theme_toggle_btn = QPushButton("暗房模式")
            self.theme_toggle_btn.setObjectName("themeToggleBtn")
            self.theme_toggle_btn.setFixedHeight(28)
            self.theme_toggle_btn.setToolTip("在浅色工作室与低光暗房界面之间切换")
            self.theme_toggle_btn.clicked.connect(self._toggle_theme)
            heading.addWidget(self.theme_toggle_btn, 0, Qt.AlignVCenter)

            outer.addWidget(header)

            # --- Main Architectural Split: Left Control Rail + Right Proofing Stage ---
            body = QHBoxLayout()
            body.setContentsMargins(0, 0, 0, 0)
            body.setSpacing(0)
            outer.addLayout(body, 1)

            # Left Control Rail
            sidebar = QFrame()
            sidebar.setObjectName("controlRail")
            sidebar.setFixedWidth(328)
            left = QVBoxLayout(sidebar)
            left.setContentsMargins(0, 0, 0, 0)
            left.setSpacing(0)

            self.settings_scroll = QScrollArea()
            self.settings_scroll.setObjectName("settingsScroll")
            self.settings_scroll.setWidgetResizable(True)
            self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.settings_content = QWidget()
            self.settings_content.setObjectName("settingsContent")
            settings_layout = QVBoxLayout(self.settings_content)
            settings_layout.setContentsMargins(14, 14, 14, 14)
            settings_layout.setSpacing(12)
            self.settings_scroll.setWidget(self.settings_content)
            left.addWidget(self.settings_scroll, 1)

            def add_divider() -> None:
                div = QFrame()
                div.setObjectName("railDivider")
                settings_layout.addWidget(div)

            def section(title_text: str) -> QFormLayout:
                container = QWidget()
                section_layout = QVBoxLayout(container)
                section_layout.setContentsMargins(0, 0, 0, 0)
                section_layout.setSpacing(8)
                label = QLabel(title_text)
                label.setProperty("role", "section")
                section_layout.addWidget(label)
                form = QFormLayout()
                form.setContentsMargins(0, 0, 0, 0)
                form.setRowWrapPolicy(QFormLayout.WrapAllRows)
                form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
                form.setVerticalSpacing(8)
                section_layout.addLayout(form)
                settings_layout.addWidget(container)
                return form

            # --- Section 1: Directories & Import ---
            files_form = section("目录与输入")

            self.source_edit = QLineEdit()
            self.source_edit.setObjectName("sourceDirectory")
            self.source_edit.setPlaceholderText("选择或拖入照片文件夹…")

            source_btn = QPushButton("选择…")
            source_btn.setFixedWidth(58)
            source_btn.clicked.connect(self.choose_source)

            source_open_btn = QPushButton("打开")
            source_open_btn.setFixedWidth(50)
            source_open_btn.setToolTip("在系统文件管理器中打开源文件夹")
            source_open_btn.clicked.connect(self._open_source_folder)

            source_row = QHBoxLayout()
            source_row.setSpacing(6)
            source_row.addWidget(self.source_edit)
            source_row.addWidget(source_btn)
            source_row.addWidget(source_open_btn)
            files_form.addRow("原照片文件夹", source_row)

            # Folder inspection summary row (SVG icon + concise summary + manual group badge)
            inspect_row = QWidget()
            inspect_layout = QHBoxLayout(inspect_row)
            inspect_layout.setContentsMargins(0, 0, 0, 0)
            inspect_layout.setSpacing(6)
            self._inspect_icon = QLabel()
            self.folder_stats_label = QLabel("支持将照片文件夹直接拖入窗口")
            self.folder_stats_label.setProperty("role", "hint")
            self.folder_stats_label.setWordWrap(True)
            inspect_layout.addWidget(self._inspect_icon, 0, Qt.AlignTop)
            inspect_layout.addWidget(self.folder_stats_label, 1)
            files_form.addRow(inspect_row)

            self.manual_groups_label = QLabel("")
            self.manual_groups_label.setProperty("role", "hint")
            self.manual_groups_label.setWordWrap(True)
            self.manual_groups_label.hide()
            files_form.addRow(self.manual_groups_label)

            self.recursive_check = QCheckBox("包含子文件夹（递归扫描）")
            self.recursive_check.setObjectName("recursiveScan")
            self.recursive_check.setChecked(False)
            self.recursive_check.setToolTip("勾选后将子目录内的 JPG 照片也一并纳入排序与自动分组")
            self.recursive_check.toggled.connect(lambda _: self._inspect_source_folder(self.source_edit.text()))
            files_form.addRow(self.recursive_check)

            # Output directory
            self.output_edit = QLineEdit()
            self.output_edit.setObjectName("outputDirectory")
            self.output_edit.setPlaceholderText("默认：源文件夹内的“合成”文件夹")

            output_btn = QPushButton("选择…")
            output_btn.setFixedWidth(58)
            output_btn.clicked.connect(self.choose_output)

            output_open_btn = QPushButton("打开")
            output_open_btn.setFixedWidth(50)
            output_open_btn.setToolTip("在系统文件管理器中打开合成结果文件夹")
            output_open_btn.clicked.connect(self._open_output_folder)

            output_row = QHBoxLayout()
            output_row.setSpacing(6)
            output_row.addWidget(self.output_edit)
            output_row.addWidget(output_btn)
            output_row.addWidget(output_open_btn)
            files_form.addRow("合成结果目录", output_row)

            # Archive disclosure
            self.archive_check = QCheckBox("合成成功后自动归档原图")
            self.archive_check.setObjectName("archiveEnabled")
            self.archive_check.setChecked(False)
            self.archive_check.setToolTip("默认关闭：保留原图不变，方便校对和二次调整")
            files_form.addRow(self.archive_check)

            self.archive_content = QFrame()
            self.archive_content.setObjectName("archiveDrawer")
            archive_form = QFormLayout(self.archive_content)
            archive_form.setContentsMargins(6, 2, 0, 2)
            archive_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
            archive_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            archive_form.setVerticalSpacing(6)

            self.archive_edit = QLineEdit()
            self.archive_edit.setObjectName("archiveDirectory")
            self.archive_edit.setPlaceholderText("默认：源文件夹内的“归档”文件夹")
            archive_btn = QPushButton("选择…")
            archive_btn.setFixedWidth(58)
            archive_btn.clicked.connect(self.choose_archive)
            archive_row = QHBoxLayout()
            archive_row.setSpacing(6)
            archive_row.addWidget(self.archive_edit)
            archive_row.addWidget(archive_btn)
            archive_form.addRow("原图归档目录", archive_row)

            self.archive_combo = QComboBox()
            self.archive_combo.setObjectName("archiveMode")
            self.archive_combo.addItem("移动（推荐整理模式）", "move")
            self.archive_combo.addItem("复制（保留源文件）", "copy")
            self.archive_combo.addItem("硬链接（极速且省磁盘空间）", "hardlink")
            archive_form.addRow("归档方式", self.archive_combo)

            files_form.addRow(self.archive_content)
            self.archive_content.hide()
            self.archive_check.toggled.connect(self.archive_content.setVisible)

            add_divider()

            # --- Section 2: Stacking Settings ---
            basic_form = section("合成与分组设置")

            self.minimum_group_spin = QSpinBox()
            self.minimum_group_spin.setObjectName("minimumStackGroupSize")
            self.minimum_group_spin.setRange(2, 1000)
            self.minimum_group_spin.setValue(4)
            self.minimum_group_spin.setSuffix(" 张")
            self.minimum_group_spin.setToolTip("同组至少达到此数量的可用相似照片才执行融合")
            basic_form.addRow("最少相似照片数", self.minimum_group_spin)

            self.grouping_pause_spin = QSpinBox()
            self.grouping_pause_spin.setObjectName("groupingPauseSeconds")
            self.grouping_pause_spin.setRange(1, 3600)
            self.grouping_pause_spin.setValue(20)
            self.grouping_pause_spin.setSuffix(" 秒")
            self.grouping_pause_spin.setToolTip("相邻照片间隔达到此值时结合画面变化分组；超过较大阈值时强制切分")
            basic_form.addRow("分组拍摄间隔", self.grouping_pause_spin)

            self.format_combo = QComboBox()
            self.format_combo.setObjectName("outputFormat")
            self.format_combo.addItem("JPG（100% 极高品质）", "jpg")
            self.format_combo.addItem("TIFF（无损专业打印）", "tiff")
            basic_form.addRow("输出格式", self.format_combo)

            add_divider()

            # --- Section 3: Advanced & Performance ---
            self.advanced_toggle = QPushButton("高级与性能设置 · 展开")
            self.advanced_toggle.setObjectName("disclosureBtn")
            self.advanced_toggle.setCheckable(True)
            settings_layout.addWidget(self.advanced_toggle)

            self.advanced_content = QWidget()
            advanced_layout = QVBoxLayout(self.advanced_content)
            advanced_layout.setContentsMargins(0, 4, 0, 0)
            advanced_layout.setSpacing(8)
            advanced_form = QFormLayout()
            advanced_form.setContentsMargins(0, 0, 0, 0)
            advanced_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
            advanced_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            advanced_form.setVerticalSpacing(8)
            advanced_layout.addLayout(advanced_form)
            settings_layout.addWidget(self.advanced_content)
            self.advanced_content.hide()
            self.advanced_toggle.toggled.connect(self._on_advanced_toggled)

            self.backend_combo = QComboBox()
            self.backend_combo.setObjectName("fusionBackend")
            self.backend_combo.addItem("高质量合成引擎（全分辨率内存融合·推荐）", "quality")
            self.backend_combo.addItem("Fast 快速合成（C++）", "fast")
            self.backend_combo.addItem("实验模式（Hugin 对齐 / Enfuse 合成）", "hugin_enfuse")
            self.backend_combo.setToolTip("高质量：全分辨率对齐与边缘一致性融合；Fast：快速焦点融合；实验：Hugin/Enfuse")
            self.backend_combo.setCurrentIndex(0)
            self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
            advanced_form.addRow("合成引擎", self.backend_combo)

            # Optional Hugin binary locator / override
            self.hugin_container = QWidget()
            hugin_layout = QVBoxLayout(self.hugin_container)
            hugin_layout.setContentsMargins(0, 0, 0, 0)
            hugin_layout.setSpacing(4)
            self.hugin_status_label = QLabel("仅实验模式需要 Hugin 工具")
            self.hugin_status_label.setProperty("role", "hint")
            self.hugin_status_label.setWordWrap(True)
            hugin_layout.addWidget(self.hugin_status_label)

            hugin_row = QHBoxLayout()
            hugin_row.setSpacing(6)
            self.hugin_bin_edit = QLineEdit()
            self.hugin_bin_edit.setObjectName("huginBinDirectory")
            self.hugin_bin_edit.setPlaceholderText("自动探测内置或系统 Hugin/bin")
            hugin_browse = QPushButton("选择…")
            hugin_browse.setFixedWidth(58)
            hugin_browse.clicked.connect(self.choose_hugin_bin)
            hugin_row.addWidget(self.hugin_bin_edit)
            hugin_row.addWidget(hugin_browse)
            hugin_layout.addLayout(hugin_row)
            advanced_form.addRow("Hugin 工具目录（可选）", self.hugin_container)

            self.merge_workers_spin = QSpinBox()
            self.merge_workers_spin.setObjectName("mergeWorkers")
            self.merge_workers_spin.setRange(1, 6)
            self.merge_workers_spin.setValue(3)
            self.merge_workers_spin.setToolTip("同时处理的合成组数；组数越多，占用内存与磁盘带宽越高")
            advanced_form.addRow("并发合成组数", self.merge_workers_spin)

            self.focus_analysis_workers_spin = QSpinBox()
            self.focus_analysis_workers_spin.setObjectName("focusAnalysisWorkers")
            self.focus_analysis_workers_spin.setRange(0, 10)
            self.focus_analysis_workers_spin.setSpecialValueText("自动推荐")
            self.focus_analysis_workers_spin.setValue(0)
            self.focus_analysis_workers_spin.setSuffix(" 张")
            self.focus_analysis_workers_spin.setToolTip("同时进行焦点分析的最大图片数；0 表示按 CPU 与可用内存自动分配")
            advanced_form.addRow("并发焦点分析", self.focus_analysis_workers_spin)

            self.parallel_check = QCheckBox("分类与合成流水线并行")
            self.parallel_check.setChecked(True)
            self.parallel_check.setObjectName("parallelPipeline")

            self.cache_check = QCheckBox("保留焦点分析缓存")
            self.cache_check.setChecked(False)
            self.cache_check.setObjectName("preserveCache")

            self.include_hidden_check = QCheckBox("扫描隐藏文件/目录")
            self.include_hidden_check.setChecked(False)
            self.include_hidden_check.setObjectName("includeHidden")
            self.include_hidden_check.toggled.connect(lambda _: self._inspect_source_folder(self.source_edit.text()))

            self.notify_check = QCheckBox("处理完成后发出提示音")
            self.notify_check.setChecked(True)
            self.notify_check.setObjectName("completionNotify")

            checks = QVBoxLayout()
            checks.setSpacing(4)
            checks.addWidget(self.parallel_check)
            checks.addWidget(self.cache_check)
            checks.addWidget(self.include_hidden_check)
            checks.addWidget(self.notify_check)
            advanced_form.addRow("运行选项", checks)

            self.reset_defaults_btn = QPushButton("恢复默认参数")
            self.reset_defaults_btn.setObjectName("ghostBtn")
            self.reset_defaults_btn.clicked.connect(self._reset_defaults)
            advanced_layout.addWidget(self.reset_defaults_btn, 0, Qt.AlignLeft)

            settings_layout.addStretch()

            for edit in (self.source_edit, self.output_edit, self.archive_edit, self.hugin_bin_edit):
                edit.setMinimumWidth(0)
                edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
                edit.textChanged.connect(edit.setToolTip)
            for combo in (self.format_combo, self.archive_combo, self.backend_combo):
                combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
                combo.setMinimumContentsLength(8)
                combo.setMinimumWidth(0)
                combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            for spin in (
                self.minimum_group_spin,
                self.grouping_pause_spin,
                self.merge_workers_spin,
                self.focus_analysis_workers_spin,
            ):
                spin.setMinimumWidth(0)
                spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.source_edit.textChanged.connect(self._inspect_source_folder)
            self.output_edit.textChanged.connect(lambda _: self._sync_output_dir())
            self.hugin_bin_edit.textChanged.connect(lambda _: self._probe_hugin_status())

            # Pinned Action Footer on Left Rail
            rail_footer = QWidget()
            rail_footer.setObjectName("railFooter")
            buttons = QHBoxLayout(rail_footer)
            buttons.setContentsMargins(16, 12, 16, 12)
            buttons.setSpacing(8)

            self.start_button = QPushButton("开始处理")
            self.start_button.setObjectName("startButton")
            self.start_button.clicked.connect(self.start_processing)

            self.stop_button = QPushButton("停止")
            self.stop_button.setObjectName("stopButton")
            self.stop_button.setEnabled(False)
            self.stop_button.clicked.connect(self.stop_processing)

            buttons.addWidget(self.start_button, 1)
            buttons.addWidget(self.stop_button)
            left.addWidget(rail_footer)

            body.addWidget(sidebar)

            # --- Right Proofing Workspace ---
            right_host = QWidget()
            right = QVBoxLayout(right_host)
            right.setContentsMargins(16, 14, 16, 14)
            right.setSpacing(12)

            self.progress_panel = ProgressPanel(self, tokens=self._tokens)
            right.addWidget(self.progress_panel)

            self.group_list = GroupListWidget(self, tokens=self._tokens)
            self.group_list.browse_requested.connect(self.choose_source)
            right.addWidget(self.group_list, 1)

            body.addWidget(right_host, 1)

            self.statusBar().showMessage("系统就绪，请选择或拖入照片文件夹")

        def _apply_theme_icons(self) -> None:
            t = self._tokens
            self._brand_icon.setPixmap(themed_pixmap("layers.svg", t.accent, 20))
            self._clock_icon.setPixmap(themed_pixmap("clock.svg", t.text_muted, 14))
            self._inspect_icon.setPixmap(themed_pixmap("image.svg", t.accent, 14))
            self.start_button.setIcon(themed_icon("play.svg", "#FFFFFF", 14))
            self.stop_button.setIcon(themed_icon("stop.svg", t.danger if self.stop_button.isEnabled() else t.text_disabled, 13))
            self.reset_defaults_btn.setIcon(themed_icon("refresh.svg", t.text_secondary, 13))
            if self._theme_name == "dark":
                self.theme_toggle_btn.setText("浅色工作室")
                self.theme_toggle_btn.setIcon(themed_icon("sun.svg", t.text_secondary, 14))
            else:
                self.theme_toggle_btn.setText("暗房模式")
                self.theme_toggle_btn.setIcon(themed_icon("moon.svg", t.text_secondary, 14))

        def _toggle_theme(self) -> None:
            self._theme_name = "dark" if self._theme_name == "light" else "light"
            self._tokens = apply_theme(self, self._theme_name)
            self._apply_theme_icons()
            self.progress_panel.apply_theme_tokens(self._tokens)
            self.group_list.apply_theme_tokens(self._tokens)
            self._refresh_run_status_style()
            self._save_settings()

        def _set_run_status(self, text: str, *, tone: str) -> None:
            self._run_status_tone = tone
            self.run_status.setText(text)
            self._refresh_run_status_style()

        def _refresh_run_status_style(self) -> None:
            t = self._tokens
            tone = getattr(self, "_run_status_tone", "idle")
            styles = {
                "idle": (t.accent_soft, t.accent_text, t.accent_soft_border),
                "running": (t.accent_soft, t.accent_text, t.accent_soft_border),
                "stopping": (t.warning_soft, t.warning, t.warning_border),
                "cancelled": (t.warning_soft, t.warning, t.warning_border),
                "success": (t.success_soft, t.success, t.success_border),
                "error": (t.danger_soft, t.danger, t.danger_border),
            }
            bg, fg, bd = styles.get(tone, styles["idle"])
            self.run_status.setStyleSheet(
                f"background: {bg}; color: {fg}; border: 1px solid {bd}; "
                "padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 12px;"
            )

        def _on_advanced_toggled(self, checked: bool) -> None:
            self.advanced_content.setVisible(checked)
            self.advanced_toggle.setText("高级与性能设置 · 收起" if checked else "高级与性能设置 · 展开")
            if checked and not self._hugin_probed:
                self._probe_hugin_status()

        def _on_backend_changed(self) -> None:
            if self.backend_combo.currentData() == "hugin_enfuse":
                self._probe_hugin_status()

        def _probe_hugin_status(self) -> None:
            self._hugin_probed = True
            t = self._tokens
            try:
                from ..hugin.hugin_locator import HuginLocator

                explicit = self.hugin_bin_edit.text().strip() or None
                install = HuginLocator(explicit_path=explicit).find_installation(require=False)
                if install.available:
                    root = str(install.root or (install.align_image_stack.parent if install.align_image_stack else ""))
                    self.hugin_status_label.setText(f"已检测到 Hugin：{root}")
                    self.hugin_status_label.setStyleSheet(f"color: {t.success}; font-size: 11px;")
                else:
                    self.hugin_status_label.setText("未检测到 Hugin/Enfuse（若需使用实验模式请安装或指定 bin 目录）")
                    self.hugin_status_label.setStyleSheet(f"color: {t.warning}; font-size: 11px;")
            except Exception:
                self.hugin_status_label.setText("Hugin 工具检测不可用")
                self.hugin_status_label.setStyleSheet(f"color: {t.text_muted}; font-size: 11px;")

        def _sync_output_dir(self) -> None:
            if self.source_edit.text().strip() or self.output_edit.text().strip():
                self.group_list.set_output_directory(self.output_directory)
            else:
                self.group_list.set_output_directory(None)

        def _inspect_source_folder(self, text: str) -> None:
            t = self._tokens
            path_str = text.strip()
            self._sync_output_dir()
            if not path_str:
                self.folder_stats_label.setText("支持将照片文件夹直接拖入窗口")
                self.folder_stats_label.setStyleSheet(f"color: {t.text_muted}; font-size: 11px;")
                self.manual_groups_label.hide()
                return
            path = Path(path_str)
            if not path.is_dir():
                self.folder_stats_label.setText("目录不存在或无法访问")
                self.folder_stats_label.setStyleSheet(f"color: {t.danger}; font-size: 11px;")
                self.manual_groups_label.hide()
                return

            manual_file = path / "stack_groups.json"
            if manual_file.is_file():
                self.manual_groups_label.setText("已检测到手动分组文件 stack_groups.json")
                self.manual_groups_label.setStyleSheet(f"color: {t.accent_text}; font-size: 11px; font-weight: 600;")
                self.manual_groups_label.show()
            else:
                self.manual_groups_label.hide()

            recursive = self.recursive_check.isChecked()
            include_hidden = self.include_hidden_check.isChecked()
            excluded_dirs = {".stack_cache"}
            for d in (self.output_directory, self.archive_directory):
                try:
                    excluded_dirs.add(str(d.resolve(strict=False)).casefold())
                except Exception:
                    pass

            jpg_count = 0
            other_photo_count = 0
            total_bytes = 0
            scanned_cap = 15000
            seen_entries = 0

            try:
                if recursive:
                    for base, dirs, files in os.walk(path):
                        dirs[:] = [
                            d for d in dirs
                            if d != ".stack_cache"
                            and (include_hidden or not d.startswith("."))
                            and str((Path(base) / d).resolve(strict=False)).casefold() not in excluded_dirs
                        ]
                        for name in files:
                            seen_entries += 1
                            if seen_entries > scanned_cap:
                                break
                            if not include_hidden and name.startswith("."):
                                continue
                            ext = os.path.splitext(name)[1].lower()
                            if ext in _SUPPORTED_EXTENSIONS:
                                jpg_count += 1
                                try:
                                    total_bytes += (Path(base) / name).stat().st_size
                                except OSError:
                                    pass
                            elif ext in _OTHER_PHOTO_EXTENSIONS:
                                other_photo_count += 1
                        if seen_entries > scanned_cap:
                            break
                else:
                    for entry in os.scandir(path):
                        if entry.is_file():
                            if not include_hidden and entry.name.startswith("."):
                                continue
                            ext = os.path.splitext(entry.name)[1].lower()
                            if ext in _SUPPORTED_EXTENSIONS:
                                jpg_count += 1
                                try:
                                    total_bytes += entry.stat().st_size
                                except OSError:
                                    pass
                            elif ext in _OTHER_PHOTO_EXTENSIONS:
                                other_photo_count += 1

                if jpg_count > 0:
                    size_mb = total_bytes / (1024 * 1024)
                    msg = f"发现 {jpg_count} 张 JPG · 约 {size_mb:.1f} MB"
                    if other_photo_count > 0:
                        msg += f"（另 {other_photo_count} 个非 JPG）"
                    if seen_entries > scanned_cap:
                        msg += " · 已达预检上限"
                    self.folder_stats_label.setText(msg)
                    self.folder_stats_label.setStyleSheet(f"color: {t.accent_text}; font-size: 11px; font-weight: 600;")
                elif other_photo_count > 0:
                    self.folder_stats_label.setText(f"检测到 {other_photo_count} 个非 JPG 文件（当前仅扫描 JPG/JPEG）")
                    self.folder_stats_label.setStyleSheet(f"color: {t.warning}; font-size: 11px;")
                else:
                    hint = "未检测到 JPG 照片" + ("" if recursive else "（可尝试勾选“包含子文件夹”）")
                    self.folder_stats_label.setText(hint)
                    self.folder_stats_label.setStyleSheet(f"color: {t.text_muted}; font-size: 11px;")
            except Exception:
                self.folder_stats_label.setText("目录已就绪")
                self.folder_stats_label.setStyleSheet(f"color: {t.text_muted}; font-size: 11px;")

        def _open_source_folder(self) -> None:
            path_str = self.source_edit.text().strip()
            if path_str and Path(path_str).is_dir():
                QDesktopServices.openUrl(QUrl.fromLocalFile(path_str))

        def _open_output_folder(self) -> None:
            out_dir = self.output_directory
            if out_dir.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))

        def dragEnterEvent(self, event: QDragEnterEvent) -> None:
            if event.mimeData().hasUrls():
                for url in event.mimeData().urls():
                    if Path(url.toLocalFile()).is_dir():
                        self.group_list.set_drag_active(True)
                        event.acceptProposedAction()
                        return
            event.ignore()

        def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
            self.group_list.set_drag_active(False)
            super().dragLeaveEvent(event)

        def dropEvent(self, event: QDropEvent) -> None:
            self.group_list.set_drag_active(False)
            for url in event.mimeData().urls():
                local_path = url.toLocalFile()
                if Path(local_path).is_dir():
                    self._set_source_path(local_path)
                    event.acceptProposedAction()
                    return

        def _set_source_path(self, value: str) -> None:
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

        def choose_hugin_bin(self) -> None:
            value = QFileDialog.getExistingDirectory(self, "选择 Hugin bin 目录", self.hugin_bin_edit.text())
            if value:
                self.hugin_bin_edit.setText(value)

        def set_groups(self, groups: Iterable[Any]) -> None:
            self._groups = list(groups)

        def _reset_defaults(self) -> None:
            self.recursive_check.setChecked(False)
            self.include_hidden_check.setChecked(False)
            self.archive_check.setChecked(False)
            self.archive_combo.setCurrentIndex(0)
            self.minimum_group_spin.setValue(4)
            self.grouping_pause_spin.setValue(20)
            self.format_combo.setCurrentIndex(0)
            self.backend_combo.setCurrentIndex(0)
            self.hugin_bin_edit.clear()
            self.merge_workers_spin.setValue(3)
            self.focus_analysis_workers_spin.setValue(0)
            self.parallel_check.setChecked(True)
            self.cache_check.setChecked(False)
            self.notify_check.setChecked(True)
            self._save_settings()
            self.statusBar().showMessage("已恢复推荐默认参数")

        def _settings(self) -> dict[str, Any]:
            settings: dict[str, Any] = {
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
                "recursive": self.recursive_check.isChecked(),
                "include_hidden": self.include_hidden_check.isChecked(),
            }
            hugin_bin = self.hugin_bin_edit.text().strip()
            if hugin_bin:
                settings["hugin_bin"] = hugin_bin
            return settings

        def _load_settings(self) -> None:
            store = QSettings("AutoStack", "FocusStackStudio")
            saved_theme = str(store.value("ui/theme", "light"))
            if saved_theme in {"light", "dark"} and saved_theme != self._theme_name:
                self._theme_name = saved_theme
                self._tokens = apply_theme(self, self._theme_name)
                self._apply_theme_icons()
                self.progress_panel.apply_theme_tokens(self._tokens)
                self.group_list.apply_theme_tokens(self._tokens)
                self._refresh_run_status_style()

            geom = store.value("ui/geometry")
            if geom is not None:
                self.restoreGeometry(geom)

            last_source = str(store.value("paths/last_source", "") or "")
            if last_source and Path(last_source).is_dir():
                self._set_source_path(last_source)

            hugin_bin = str(store.value("paths/hugin_bin", "") or "")
            if hugin_bin:
                self.hugin_bin_edit.setText(hugin_bin)

            self.recursive_check.setChecked(self._as_bool(store.value("scan/recursive", False)))
            self.include_hidden_check.setChecked(self._as_bool(store.value("scan/include_hidden", False)))
            self.minimum_group_spin.setValue(int(store.value("stack/min_group_size", 4)))
            self.grouping_pause_spin.setValue(int(store.value("stack/pause_seconds", 20)))

            fmt = str(store.value("stack/output_format", "jpg"))
            idx = self.format_combo.findData(fmt)
            if idx >= 0:
                self.format_combo.setCurrentIndex(idx)

            backend = str(store.value("runtime/fusion_backend", "quality"))
            idx = self.backend_combo.findData(backend)
            if idx >= 0:
                self.backend_combo.setCurrentIndex(idx)

            arc_mode = str(store.value("archive/mode", "move"))
            idx = self.archive_combo.findData(arc_mode)
            if idx >= 0:
                self.archive_combo.setCurrentIndex(idx)

            self.merge_workers_spin.setValue(int(store.value("runtime/merge_workers", 3)))
            self.focus_analysis_workers_spin.setValue(int(store.value("runtime/focus_analysis_workers", 0)))
            self.parallel_check.setChecked(self._as_bool(store.value("runtime/parallel", True)))
            self.cache_check.setChecked(self._as_bool(store.value("runtime/preserve_cache", False)))
            self.notify_check.setChecked(self._as_bool(store.value("ui/notify_on_complete", True)))

        def _save_settings(self) -> None:
            if not self._restore_settings:
                return
            store = QSettings("AutoStack", "FocusStackStudio")
            store.setValue("ui/theme", self._theme_name)
            store.setValue("ui/geometry", self.saveGeometry())
            if self.source_edit.text().strip():
                store.setValue("paths/last_source", self.source_edit.text().strip())
            store.setValue("paths/hugin_bin", self.hugin_bin_edit.text().strip())
            store.setValue("scan/recursive", self.recursive_check.isChecked())
            store.setValue("scan/include_hidden", self.include_hidden_check.isChecked())
            store.setValue("stack/min_group_size", self.minimum_group_spin.value())
            store.setValue("stack/pause_seconds", self.grouping_pause_spin.value())
            store.setValue("stack/output_format", self.format_combo.currentData())
            store.setValue("runtime/fusion_backend", self.backend_combo.currentData())
            store.setValue("archive/mode", self.archive_combo.currentData())
            store.setValue("runtime/merge_workers", self.merge_workers_spin.value())
            store.setValue("runtime/focus_analysis_workers", self.focus_analysis_workers_spin.value())
            store.setValue("runtime/parallel", self.parallel_check.isChecked())
            store.setValue("runtime/preserve_cache", self.cache_check.isChecked())
            store.setValue("ui/notify_on_complete", self.notify_check.isChecked())

        @staticmethod
        def _as_bool(val: Any) -> bool:
            if isinstance(val, bool):
                return val
            return str(val).strip().lower() in {"1", "true", "yes", "on"}

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

        def _format_hms(self, elapsed_seconds: float, *, round_final: bool = False) -> str:
            val = max(0.0, float(elapsed_seconds))
            total = int(round(val)) if round_final else int(val)
            mins, secs = divmod(total, 60)
            hours, mins = divmod(mins, 60)
            return f"{hours:02d}:{mins:02d}:{secs:02d}"

        def _update_timer(self) -> None:
            if self._start_timestamp is None:
                return
            elapsed = max(0.0, time.monotonic() - self._start_timestamp)
            whole = int(elapsed)
            if whole != self._last_displayed_second:
                self._last_displayed_second = whole
                self.timer_label.setText(self._format_hms(whole))
            # Align the next wakeup right after the next whole-second boundary
            next_delay_ms = max(25, int((whole + 1.0 - elapsed) * 1000.0) + 10)
            self._elapsed_timer.start(next_delay_ms)

        @Slot()
        def start_processing(self) -> None:
            if not self.source_edit.text().strip():
                QMessageBox.warning(self, "缺少目录", "请先选择原照片文件夹。")
                return
            self._save_settings()
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
                self._set_run_status("无法开始", tone="error")
                self.statusBar().showMessage("无法开始处理，请检查目录和参数设置")
                QMessageBox.critical(self, "无法开始", str(exc))
                if self.controller_factory is not None:
                    self._retire_worker(self.controller)
                    self.controller = None
                if self.coordinator_factory is not None:
                    self._retire_worker(self.coordinator)
                    self.coordinator = None
                return

            self._stopping = False
            self._start_timestamp = time.monotonic()
            self._last_displayed_second = 0
            self.timer_label.setText("00:00:00")
            self._elapsed_timer.start(1005)
            self.settings_content.setEnabled(False)
            self.progress_panel.reset()
            self.progress_panel.set_log_path(self.output_directory / ".stack_cache" / "logs" / "application.log")
            self.group_list.set_output_directory(self.output_directory)
            self.group_list.set_empty_message("处理结束后显示结果")
            self.start_button.setEnabled(False)
            self.start_button.setText("正在处理…")
            self.stop_button.setEnabled(True)
            self.stop_button.setIcon(themed_icon("stop.svg", self._tokens.danger, 13))
            self._set_run_status("处理中", tone="running")
            self.statusBar().showMessage("流水线处理中…")

        @Slot()
        def stop_processing(self) -> None:
            if self.controller is not None:
                self.controller.cancel()
            elif self.coordinator is not None:
                self.coordinator.cancel()
            self._stopping = True
            self.stop_button.setEnabled(False)
            self.stop_button.setIcon(themed_icon("stop.svg", self._tokens.text_disabled, 13))
            self._set_run_status("正在安全停止…", tone="stopping")
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

            elapsed = float(getattr(summary, "elapsed_seconds", 0.0) or 0.0)
            if elapsed <= 0.0 and self._start_timestamp is not None:
                elapsed = max(0.0, time.monotonic() - self._start_timestamp)
            self._start_timestamp = None
            self.timer_label.setText(self._format_hms(elapsed, round_final=True))

            self.settings_content.setEnabled(True)
            self.start_button.setEnabled(True)
            self.start_button.setText("开始处理")
            self.stop_button.setEnabled(False)
            self.stop_button.setIcon(themed_icon("stop.svg", self._tokens.text_disabled, 13))

            complete = getattr(self.progress_panel, "complete", None)
            if callable(complete):
                complete(summary)

            self.group_list.set_output_directory(self.output_directory)
            self.group_list.update_results(summary.results)
            if not summary.results:
                self.group_list.set_empty_message("已取消，本次没有结果" if summary.cancelled else "本次没有可显示的结果")

            if summary.cancelled:
                self._set_run_status("已取消", tone="cancelled")
                self.statusBar().showMessage("已取消")
            elif int(getattr(summary, "errors", 0) or 0) > 0:
                count = int(getattr(summary, "errors", 0) or 0)
                self._set_run_status(f"处理结束 · {count} 个错误", tone="error")
                self.statusBar().showMessage(f"处理结束：{count} 个错误，失败组原图未移动（耗时 {elapsed:.1f}s）")
                diagnostics = list(getattr(summary, "diagnostics", ()) or ())
                detail = "\n\n".join(str(item) for item in diagnostics) if diagnostics else "部分图片组未能完成合成；失败组原图未移动。"
                self.progress_panel.set_message(detail)
                QMessageBox.warning(self, "处理未全部完成", "部分图片组未能完成合成；失败组原图未移动。请展开诊断日志查看原因。")
            else:
                self._set_run_status("处理完成", tone="success")
                total_groups = len(getattr(summary, "results", ()) or ())
                self.statusBar().showMessage(f"景深合成处理完成 · 共 {total_groups} 组 · 耗时 {elapsed:.1f} 秒")
                if self.notify_check.isChecked():
                    QApplication.alert(self, 3000)

        def closeEvent(self, event: Any) -> None:
            self._elapsed_timer.stop()
            self._save_settings()
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
