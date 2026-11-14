"""Proofing workbench with searchable results table, scalable preview, and 1:1 lightbox."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

_COLUMNS = ("组", "原图", "精选", "覆盖率", "状态", "输出")


def _get(value: Any, *names: str, default: Any = "") -> Any:
    if isinstance(value, dict):
        for name in names:
            if name in value and value[name] is not None:
                return value[name]
    else:
        for name in names:
            if hasattr(value, name):
                attr = getattr(value, name)
                if attr is not None:
                    return attr
    return default


def _item_filename(item: Any) -> str:
    if not item:
        return ""
    for attr in ("filename", "name", "path", "current_path", "source_path", "original_path"):
        if hasattr(item, attr) and getattr(item, attr):
            return Path(str(getattr(item, attr))).name
    if isinstance(item, (str, os.PathLike)):
        return Path(os.fspath(item)).name
    return ""


def _extract_image_names(result: Any) -> list[str]:
    """Extract ordered list of original photo filenames for this result."""
    names: list[str] = []

    def add_item(item: Any) -> None:
        name = _item_filename(item)
        if name:
            names.append(name)

    all_paths = _get(result, "all_paths", default=None)
    if all_paths and isinstance(all_paths, (list, tuple)):
        for p in all_paths:
            add_item(p)
        if names:
            return names

    group = _get(result, "group", default=None)
    if group is not None:
        for attr in ("items", "all_images", "images", "image_records", "source_paths"):
            items = _get(group, attr, default=None)
            if items and isinstance(items, (list, tuple)):
                for it in items:
                    add_item(it)
                if names:
                    return names

    analysis = _get(result, "analysis", default=None)
    if analysis is not None:
        for attr in ("all_images", "images", "items", "all_paths"):
            items = _get(analysis, attr, default=None)
            if items and isinstance(items, (list, tuple)):
                for it in items:
                    add_item(it)
                if names:
                    return names

    archive = _get(result, "archive_result", default=None)
    if archive is not None:
        records = getattr(archive, "records", ()) or ()
        for r in records:
            add_item(r)
        if names:
            return names

    for attr in ("images", "items", "photo_paths"):
        val = _get(result, attr, default=None)
        if val and isinstance(val, (list, tuple)):
            for it in val:
                add_item(it)
            if names:
                return names

    first = _get(result, "first_original", default=_get(group, "first_original", default=None))
    if first:
        add_item(first)

    return names


def _extract_selected_names(result: Any, analysis: Any) -> list[str]:
    """Extract filenames of the frames selected for fusion."""
    selected: list[str] = []
    for source in (
        _get(result, "selected_source_paths", default=None),
        _get(analysis, "selected_paths", "selected_images", default=None),
    ):
        if source and isinstance(source, (list, tuple)):
            for item in source:
                name = _item_filename(item)
                if name:
                    selected.append(name)
            if selected:
                return selected
    return selected


def _backend_label(backend: str | None) -> str:
    mapping = {
        "quality": "高质量引擎",
        "fast": "Fast C++",
        "hugin_enfuse": "Hugin / Enfuse",
        "hugin": "Hugin / Enfuse",
    }
    key = str(backend or "").strip().lower()
    return mapping.get(key, str(backend or ""))


try:
    from PySide6.QtCore import Qt, QUrl, Signal
    from PySide6.QtGui import QAction, QColor, QDesktopServices, QPixmap
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QFrame,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    from .image_viewer import ImageLightboxDialog
    from .theme import ThemeTokens, get_theme_tokens, themed_icon, themed_pixmap

    class GroupListWidget(QFrame):
        """Results workbench with searchable table, live inspector, and lightbox zoom."""

        browse_requested = Signal()

        def __init__(self, parent: QWidget | None = None, *, tokens: ThemeTokens | None = None):
            super().__init__(parent)
            self.setObjectName("resultsSurface")
            self._tokens = tokens or get_theme_tokens("light")
            self._all_results: list[Any] = []
            self._result_images: list[list[str]] = []
            self._result_ranges: list[str] = []
            self._filter_status = "all"
            self._current_output_path = ""
            self._current_pixmap: QPixmap | None = None
            self._output_dir: Path | None = None

            main_layout = QVBoxLayout(self)
            main_layout.setContentsMargins(16, 12, 16, 12)
            main_layout.setSpacing(10)

            # --- Two-Row Responsive Toolbar (zero overlap down to 960x640) ---
            header_row = QHBoxLayout()
            header_row.setSpacing(8)

            title = QLabel("合成结果")
            title.setProperty("role", "section")
            header_row.addWidget(title)

            self.count_badge = QLabel("0 组")
            self.count_badge.setProperty("role", "countPill")
            header_row.addWidget(self.count_badge)

            self.breakdown_label = QLabel("")
            self.breakdown_label.setProperty("role", "hint")
            header_row.addWidget(self.breakdown_label)

            header_row.addStretch(1)

            # Batch Quick Open Actions
            self.open_output_btn = QPushButton("输出目录")
            self.open_output_btn.setObjectName("ghostBtn")
            self.open_output_btn.setToolTip("打开合成结果文件夹")
            self.open_output_btn.setEnabled(False)
            self.open_output_btn.clicked.connect(self._open_output_folder)
            header_row.addWidget(self.open_output_btn)

            self.open_manifest_btn = QPushButton("清单 CSV")
            self.open_manifest_btn.setObjectName("ghostBtn")
            self.open_manifest_btn.setToolTip("打开 stack_manifest.csv 处理清单")
            self.open_manifest_btn.setEnabled(False)
            self.open_manifest_btn.clicked.connect(self._open_manifest_csv)
            header_row.addWidget(self.open_manifest_btn)

            main_layout.addLayout(header_row)

            filter_bar = QHBoxLayout()
            filter_bar.setSpacing(8)

            # Search Box with native leading SVG icon
            self.search_box = QLineEdit()
            self.search_box.setPlaceholderText("搜索原图名 (如 0015) 或组号…")
            self.search_box.setToolTip("输入组号、原图文件名（如 0015 或 DSC0015）或输出文件名，快速定位对应图组")
            self.search_box.setClearButtonEnabled(True)
            self.search_box.setMinimumWidth(140)
            self.search_box.setFixedHeight(28)
            self._search_action = QAction(self.search_box)
            self.search_box.addAction(self._search_action, QLineEdit.LeadingPosition)
            self.search_box.textChanged.connect(self._apply_filter)
            filter_bar.addWidget(self.search_box, 1)

            # Segmented Filter Chips
            filter_row = QHBoxLayout()
            filter_row.setSpacing(4)
            self.filter_buttons: list[QPushButton] = []
            for filter_id, filter_label in (
                ("all", "全部"),
                ("done", "已完成"),
                ("skipped", "无需合成"),
                ("failed", "失败"),
            ):
                btn = QPushButton(filter_label)
                btn.setProperty("class", "segBtn")
                btn.setProperty("filter_id", filter_id)
                btn.setCheckable(True)
                btn.setChecked(filter_id == "all")
                btn.clicked.connect(lambda checked, fid=filter_id: self._set_filter(fid))
                filter_row.addWidget(btn)
                self.filter_buttons.append(btn)
            filter_bar.addLayout(filter_row)

            main_layout.addLayout(filter_bar)

            # --- Central Splitter: Left Results Table + Right Specimen Inspector ---
            self.splitter = QSplitter(Qt.Horizontal)
            self.splitter.setChildrenCollapsible(False)
            main_layout.addWidget(self.splitter, 1)

            # Left: Clean Table
            self.table = QTableWidget(0, len(_COLUMNS))
            self.table.setMinimumWidth(260)
            self.table.setHorizontalHeaderLabels(list(_COLUMNS))
            self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            header = self.table.horizontalHeader()
            header.setStretchLastSection(True)
            header.setSectionResizeMode(QHeaderView.Interactive)
            for col, width in enumerate((36, 44, 44, 68, 84)):
                self.table.setColumnWidth(col, width)
            self.table.verticalHeader().hide()
            self.table.verticalHeader().setDefaultSectionSize(36)
            self.table.setShowGrid(False)
            self.table.setWordWrap(False)
            self.table.setAlternatingRowColors(True)
            self.table.setTextElideMode(Qt.ElideMiddle)
            self.table.itemSelectionChanged.connect(self._on_selection_changed)
            self.table.cellDoubleClicked.connect(lambda r, c: self._open_lightbox())
            self.splitter.addWidget(self.table)

            # Right: Scroll-safe Visual Inspector
            self.inspector = QFrame()
            self.inspector.setObjectName("inspectorPane")
            self.inspector.setMinimumWidth(250)
            self.inspector.setMaximumWidth(460)
            inspector_outer = QVBoxLayout(self.inspector)
            inspector_outer.setContentsMargins(0, 0, 0, 0)

            insp_scroll = QScrollArea()
            insp_scroll.setObjectName("inspectorScroll")
            insp_scroll.setWidgetResizable(True)
            insp_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            insp_body = QWidget()
            insp_body.setObjectName("inspectorScrollContent")
            inspector_layout = QVBoxLayout(insp_body)
            inspector_layout.setContentsMargins(14, 12, 14, 14)
            inspector_layout.setSpacing(8)
            insp_scroll.setWidget(insp_body)
            inspector_outer.addWidget(insp_scroll)

            # Inspector header
            insp_header = QHBoxLayout()
            insp_header.setSpacing(6)
            self.info_group_id = QLabel("组 —")
            self.info_group_id.setStyleSheet("font-weight: 700; font-size: 13px;")
            insp_header.addWidget(self.info_group_id)

            self.info_status_pill = QLabel("—")
            self.info_status_pill.setProperty("role", "countPill")
            insp_header.addWidget(self.info_status_pill)

            self.info_backend_pill = QLabel("")
            self.info_backend_pill.setProperty("role", "hint")
            self.info_backend_pill.hide()
            insp_header.addWidget(self.info_backend_pill)
            insp_header.addStretch()
            inspector_layout.addLayout(insp_header)

            # Preview Stage (bounded height so metadata and actions stay visible above fold)
            self.preview_stage = QFrame()
            self.preview_stage.setObjectName("previewStage")
            stage_layout = QVBoxLayout(self.preview_stage)
            stage_layout.setContentsMargins(6, 6, 6, 6)
            stage_layout.setSpacing(4)

            self.preview_image = QLabel()
            self.preview_image.setAlignment(Qt.AlignCenter)
            self.preview_image.setWordWrap(True)
            self.preview_image.setProperty("role", "muted")
            self.preview_image.setFixedHeight(168)
            self.preview_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            self.preview_image.setCursor(Qt.PointingHandCursor)
            self.preview_image.setText("点击左侧列表查看合成预览")
            self.preview_image.mouseDoubleClickEvent = lambda e: self._open_lightbox()  # type: ignore[method-assign]
            stage_layout.addWidget(self.preview_image)

            self.preview_meta = QLabel("")
            self.preview_meta.setAlignment(Qt.AlignCenter)
            self.preview_meta.setProperty("role", "hint")
            self.preview_meta.hide()
            stage_layout.addWidget(self.preview_meta)
            inspector_layout.addWidget(self.preview_stage)

            # Inspector action buttons (hidden when no output image exists so compact widths stay clean)
            self._action_row_widget = QWidget()
            action_row = QHBoxLayout(self._action_row_widget)
            action_row.setContentsMargins(0, 0, 0, 0)
            action_row.setSpacing(6)
            self.zoom_btn = QPushButton("放大校对")
            self.zoom_btn.setEnabled(False)
            self.zoom_btn.setToolTip("在内置大图看图器中 1:1 检查细节清晰度")
            self.zoom_btn.clicked.connect(self._open_lightbox)

            self.open_file_btn = QPushButton("系统打开")
            self.open_file_btn.setEnabled(False)
            self.open_file_btn.setToolTip("使用系统默认看图软件打开")
            self.open_file_btn.clicked.connect(self._open_current_image)

            self.reveal_file_btn = QPushButton("定位文件")
            self.reveal_file_btn.setEnabled(False)
            self.reveal_file_btn.setToolTip("在资源管理器中选中此文件")
            self.reveal_file_btn.clicked.connect(self._reveal_current_image)

            action_row.addWidget(self.zoom_btn, 1)
            action_row.addWidget(self.open_file_btn)
            action_row.addWidget(self.reveal_file_btn)
            inspector_layout.addWidget(self._action_row_widget)

            # Error / Exclusion / Diagnostics Box (placed right below actions for visibility)
            self.reason_frame = QFrame()
            self.reason_frame.setObjectName("diagBanner")
            reason_layout = QVBoxLayout(self.reason_frame)
            reason_layout.setContentsMargins(10, 8, 10, 8)
            reason_layout.setSpacing(4)
            self.info_reason = QLabel()
            self.info_reason.setWordWrap(True)
            self.info_reason.setTextInteractionFlags(Qt.TextSelectableByMouse)
            reason_layout.addWidget(self.info_reason)
            self.reason_frame.hide()
            inspector_layout.addWidget(self.reason_frame)

            # Coverage bar + alignment/crop info
            coverage_box = QVBoxLayout()
            coverage_box.setSpacing(4)
            cov_head = QHBoxLayout()
            self.coverage_text = QLabel("清晰覆盖率：—")
            self.coverage_text.setProperty("role", "muted")
            cov_head.addWidget(self.coverage_text)
            cov_head.addStretch()
            self.align_meta_label = QLabel("")
            self.align_meta_label.setProperty("role", "hint")
            self.align_meta_label.hide()
            cov_head.addWidget(self.align_meta_label)
            coverage_box.addLayout(cov_head)

            self.coverage_bar = QProgressBar()
            self.coverage_bar.setRange(0, 100)
            self.coverage_bar.setValue(0)
            self.coverage_bar.setFixedHeight(5)
            self.coverage_bar.setTextVisible(False)
            coverage_box.addWidget(self.coverage_bar)
            inspector_layout.addLayout(coverage_box)

            # Metadata lines
            self.info_range = QLabel("原图范围：—")
            self.info_range.setWordWrap(True)
            inspector_layout.addWidget(self.info_range)

            photos_row = QHBoxLayout()
            photos_row.setSpacing(6)
            self.info_photos = QLabel("精选切片：—")
            self.info_photos.setProperty("role", "muted")
            photos_row.addWidget(self.info_photos, 1)
            self.slices_toggle = QPushButton("查看切片")
            self.slices_toggle.setObjectName("ghostBtn")
            self.slices_toggle.setCheckable(True)
            self.slices_toggle.hide()
            self.slices_toggle.toggled.connect(self._toggle_slices_list)
            photos_row.addWidget(self.slices_toggle)
            inspector_layout.addLayout(photos_row)

            self.slices_box = QTextEdit()
            self.slices_box.setReadOnly(True)
            self.slices_box.setFixedHeight(84)
            self.slices_box.hide()
            inspector_layout.addWidget(self.slices_box)

            self.info_status = QLabel("状态：—")
            self.info_status.hide()  # Kept for backwards attribute compatibility; status is shown in header pill

            self.info_archive = QLabel("")
            self.info_archive.setProperty("role", "hint")
            self.info_archive.hide()
            inspector_layout.addWidget(self.info_archive)

            inspector_layout.addStretch()

            self.splitter.addWidget(self.inspector)
            self.splitter.setStretchFactor(0, 3)
            self.splitter.setStretchFactor(1, 2)
            self.splitter.setSizes([450, 320])

            # --- Empty / Drop-Zone State ---
            self.empty_container = QWidget()
            empty_outer = QVBoxLayout(self.empty_container)
            empty_outer.setContentsMargins(24, 24, 24, 24)
            empty_outer.addStretch()

            self.empty_drop_card = QFrame()
            self.empty_drop_card.setObjectName("emptyDropCard")
            self.empty_drop_card.setMaximumWidth(520)
            card_layout = QVBoxLayout(self.empty_drop_card)
            card_layout.setContentsMargins(28, 26, 28, 26)
            card_layout.setSpacing(10)
            card_layout.setAlignment(Qt.AlignCenter)

            self.empty_icon_label = QLabel()
            self.empty_icon_label.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(self.empty_icon_label)

            self.empty_label = QLabel("选择照片文件夹，开始景深合成")
            self.empty_label.setAlignment(Qt.AlignCenter)
            self.empty_label.setWordWrap(True)
            self.empty_label.setStyleSheet("font-size: 14px; font-weight: 600;")
            card_layout.addWidget(self.empty_label)

            self.empty_hint = QLabel("支持直接将照片文件夹拖入本窗口 · 自动分组、清晰覆盖选片与全分辨率对齐融合")
            self.empty_hint.setProperty("role", "muted")
            self.empty_hint.setAlignment(Qt.AlignCenter)
            self.empty_hint.setWordWrap(True)
            card_layout.addWidget(self.empty_hint)

            self.empty_browse_btn = QPushButton("选择照片文件夹…")
            self.empty_browse_btn.setFixedWidth(180)
            self.empty_browse_btn.clicked.connect(self.browse_requested.emit)
            card_layout.addWidget(self.empty_browse_btn, 0, Qt.AlignCenter)

            empty_outer.addWidget(self.empty_drop_card, 0, Qt.AlignCenter)
            empty_outer.addStretch()
            main_layout.addWidget(self.empty_container, 1)

            # Initial visibility
            self.table.hide()
            self.inspector.hide()
            self.splitter.hide()

            self.apply_theme_tokens(self._tokens)

        def apply_theme_tokens(self, tokens: ThemeTokens) -> None:
            self._tokens = tokens
            self._search_action.setIcon(themed_icon("search.svg", tokens.text_muted, 14))
            self.open_output_btn.setIcon(themed_icon("folder.svg", tokens.text_secondary, 13))
            self.open_manifest_btn.setIcon(themed_icon("file-text.svg", tokens.text_secondary, 13))
            self.zoom_btn.setIcon(themed_icon("zoom-in.svg", tokens.text_primary, 14))
            self.open_file_btn.setIcon(themed_icon("external-link.svg", tokens.text_secondary, 13))
            self.reveal_file_btn.setIcon(themed_icon("folder.svg", tokens.text_secondary, 13))
            self.empty_icon_label.setPixmap(themed_pixmap("folder.svg", tokens.accent, 32))
            self.preview_image.setStyleSheet(f"color: {tokens.text_muted}; background: transparent; border: none;")
            self._refresh_table_status_colors()

        def set_drag_active(self, active: bool) -> None:
            self.empty_drop_card.setProperty("dragActive", bool(active))
            self.empty_drop_card.style().unpolish(self.empty_drop_card)
            self.empty_drop_card.style().polish(self.empty_drop_card)

        def set_output_directory(self, output_dir: Path | str | None) -> None:
            self._output_dir = Path(output_dir) if output_dir and str(output_dir).strip() else None
            self._refresh_batch_actions()

        def _refresh_batch_actions(self) -> None:
            out_exists = bool(self._output_dir and self._output_dir.is_dir())
            csv_exists = bool(self._output_dir and (self._output_dir / "stack_manifest.csv").is_file())
            self.open_output_btn.setEnabled(out_exists)
            self.open_manifest_btn.setEnabled(csv_exists)

        def _open_output_folder(self) -> None:
            if self._output_dir and self._output_dir.is_dir():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._output_dir)))

        def _open_manifest_csv(self) -> None:
            if self._output_dir:
                csv_path = self._output_dir / "stack_manifest.csv"
                if csv_path.is_file():
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(csv_path)))

        def set_empty_message(self, message: str) -> None:
            self.clear()
            self.empty_label.setText(message)
            is_idle = "选择照片文件夹" in message
            self.empty_hint.setVisible(is_idle)
            self.empty_browse_btn.setVisible(is_idle)

        def clear(self) -> None:
            self._all_results.clear()
            self._result_images.clear()
            self._result_ranges.clear()
            self.table.setRowCount(0)
            self.count_badge.setText("0 组")
            self.breakdown_label.setText("")
            self.table.hide()
            self.inspector.hide()
            self.splitter.hide()
            self.empty_container.show()
            self._reset_inspector()

        def _reset_inspector(self) -> None:
            self._current_output_path = ""
            self._current_pixmap = None
            self.preview_stage.show()
            self._action_row_widget.show()
            self.preview_image.setFixedHeight(168)
            self.preview_image.clear()
            self.preview_image.setText("点击左侧列表查看合成预览")
            self.preview_meta.hide()
            self.zoom_btn.setEnabled(False)
            self.open_file_btn.setEnabled(False)
            self.reveal_file_btn.setEnabled(False)
            self.info_group_id.setText("组 —")
            self.info_status_pill.setText("—")
            self.info_backend_pill.hide()
            self.info_range.setText("原图范围：—")
            self.info_range.setToolTip("")
            self.info_photos.setText("精选切片：—")
            self.slices_toggle.hide()
            self.slices_toggle.setChecked(False)
            self.slices_box.hide()
            self.info_status.setText("状态：—")
            self.coverage_text.setText("清晰覆盖率：—")
            self.coverage_bar.setValue(0)
            self.align_meta_label.hide()
            self.info_archive.hide()
            self.reason_frame.hide()

        def _toggle_slices_list(self, checked: bool) -> None:
            self.slices_box.setVisible(checked)
            self.slices_toggle.setText("收起切片" if checked else "查看切片")

        def _set_filter(self, filter_id: str) -> None:
            self._filter_status = filter_id
            for btn in self.filter_buttons:
                btn.setChecked(btn.property("filter_id") == filter_id)
            self._apply_filter()

        def _apply_filter(self) -> None:
            search_text = self.search_box.text().strip().lower()
            fid = self._filter_status

            visible_count = 0
            done_count = 0
            skip_count = 0
            fail_count = 0

            for row in range(self.table.rowCount()):
                gid_item = self.table.item(row, 0)
                status_item = self.table.item(row, 4)
                output_item = self.table.item(row, 5)

                gid = gid_item.text() if gid_item else ""
                status = status_item.text() if status_item else ""
                raw_status = str(status_item.data(Qt.UserRole) or "") if status_item else ""
                output_full = str(output_item.data(Qt.UserRole) or "") if output_item else ""
                output_name = output_item.text() if output_item else ""
                images = self._result_images[row] if row < len(self._result_images) else []
                range_str = self._result_ranges[row] if row < len(self._result_ranges) else ""

                is_done = raw_status == "DONE" or "已完成" in status
                is_fail = (
                    "FAILED" in raw_status
                    or "ERROR" in raw_status
                    or any(k in status for k in ("失败", "错误", "异常", "缺少"))
                )

                if is_done:
                    done_count += 1
                elif is_fail:
                    fail_count += 1
                else:
                    skip_count += 1

                text_match = True
                if search_text:
                    gid_match = (
                        search_text == gid
                        or search_text == f"#{gid}"
                        or f"组 #{gid}" in search_text
                        or search_text in gid.lower()
                    )
                    output_match = search_text in output_name.lower() or search_text in output_full.lower()
                    status_match = search_text in status.lower()
                    range_match = search_text in range_str.lower()
                    photo_match = any(search_text in img.lower() for img in images)
                    text_match = gid_match or output_match or status_match or range_match or photo_match

                cat_match = True
                if fid == "done":
                    cat_match = is_done
                elif fid == "skipped":
                    cat_match = not is_done and not is_fail
                elif fid == "failed":
                    cat_match = is_fail

                hide_row = not (text_match and cat_match)
                self.table.setRowHidden(row, hide_row)
                if not hide_row:
                    visible_count += 1

            total_rows = self.table.rowCount()
            self.count_badge.setText(
                f"{visible_count} / {total_rows} 组" if visible_count != total_rows else f"{total_rows} 组"
            )
            parts = []
            if done_count:
                parts.append(f"{done_count} 已合成")
            if skip_count:
                parts.append(f"{skip_count} 跳过")
            if fail_count:
                parts.append(f"{fail_count} 失败")
            self.breakdown_label.setText(" · ".join(parts))

            if visible_count > 0:
                selected_rows = self.table.selectionModel().selectedRows()
                if not selected_rows or self.table.isRowHidden(selected_rows[0].row()):
                    for r in range(total_rows):
                        if not self.table.isRowHidden(r):
                            self.table.selectRow(r)
                            break
            else:
                self._reset_inspector()

        def _status_color(self, raw_status: str) -> str:
            t = self._tokens
            if "FAILED" in raw_status or "ERROR" in raw_status:
                return t.danger
            if raw_status == "DONE":
                return t.success
            if raw_status == "CANCELLED":
                return t.warning
            return t.text_muted

        def _refresh_table_status_colors(self) -> None:
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 4)
                if item is not None:
                    raw = str(item.data(Qt.UserRole) or "")
                    item.setForeground(QColor(self._status_color(raw)))

        def add_result(self, result: Any) -> None:
            self._all_results.append(result)

            images = _extract_image_names(result)
            self._result_images.append(images)
            if images:
                range_text = images[0] if len(images) == 1 else f"{images[0]} – {images[-1]} (共 {len(images)} 张)"
            else:
                range_text = "—"
            self._result_ranges.append(range_text)

            self.empty_container.hide()
            self.splitter.show()
            self.table.show()
            self.inspector.show()

            group = _get(result, "group", default=result)
            analysis = _get(result, "analysis", default=None) or group
            row = self.table.rowCount()
            self.table.insertRow(row)

            gid = _get(result, "group_id", default=_get(group, "group_id", "id", default=""))
            all_count = _get(analysis, "image_count", default=_get(group, "image_count", default=""))
            archive = _get(result, "archive_result", default=None)
            if archive is not None:
                all_count = len(getattr(archive, "records", ())) or all_count
            if not all_count and images:
                all_count = len(images)

            selected = _get(analysis, "selected_count", default="")
            if selected == "" and _extract_selected_names(result, analysis):
                selected = len(_extract_selected_names(result, analysis))
            output = _get(result, "output_path", default=_get(group, "output_path", default=""))
            output_str = str(output) if output else ""
            output_display = Path(output_str).name if output_str else "—"

            status = _get(result, "status", default=_get(group, "status", default=""))
            raw_status = str(getattr(status, "value", status))
            status_map = {
                "DONE": "已完成",
                "FAILED": "合成失败",
                "FAILED_ALIGNMENT": "对齐失败",
                "FAILED_HUGIN": "缺少 Hugin 工具",
                "FAILED_ARCHIVE": "归档失败",
                "FAILED_FUSION": "合成失败",
                "FAILED_CLASSIFICATION": "分析失败",
                "CANCELLED": "已取消",
                "RUNNING": "处理中",
                "SKIPPED_SINGLE": "单张（未合成）",
                "NO_MERGE_SINGLE": "单张（未合成）",
                "NO_MERGE_REPEATED": "重复（未合成）",
                "NO_MERGE": "无需合成",
                "ARCHIVED_ONLY": "仅归档",
                "SKIPPED": "已跳过",
                "CLASSIFIED": "已分类",
                "SELECTED": "已选片",
                "READY_FOR_MERGE": "待合成",
            }
            status_text = status_map.get(raw_status, raw_status)
            reason = _get(analysis, "excluded_reason", default=None)
            if reason and not ("FAILED" in raw_status or "ERROR" in raw_status or raw_status == "DONE"):
                status_text = "已排除：" + str(reason)

            coverage = _get(analysis, "coverage", "group_coverage", "coverage_ratio", default="")
            if isinstance(coverage, (int, float)):
                coverage_str = f"{float(coverage):.1%}"
            else:
                coverage_str = str(coverage)

            values = [gid, all_count, selected, coverage_str, status_text, output_display]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value if value is not None else ""))
                if col == 5:
                    item.setToolTip(output_str or "—")
                    item.setData(Qt.UserRole, output_str)
                else:
                    item.setToolTip(item.text())
                if col < 4:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if col == 4:
                    item.setData(Qt.UserRole, raw_status)
                    item.setForeground(QColor(self._status_color(raw_status)))
                self.table.setItem(row, col, item)

            self._apply_filter()
            self._refresh_batch_actions()

            if self.table.rowCount() == 1:
                self.table.selectRow(0)

        def update_results(self, results: Iterable[Any]) -> None:
            self.clear()
            for result in results:
                self.add_result(result)
            self._refresh_batch_actions()

        def _render_preview_pixmap(self) -> None:
            if self._current_pixmap is None or self._current_pixmap.isNull():
                return
            w = max(120, self.preview_image.width() - 6)
            h = max(72, self.preview_image.height() - 6)
            scaled = self._current_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_image.setPixmap(scaled)

        def resizeEvent(self, event: Any) -> None:
            super().resizeEvent(event)
            self._render_preview_pixmap()

        def _on_selection_changed(self) -> None:
            selected_rows = self.table.selectionModel().selectedRows()
            if not selected_rows:
                self._reset_inspector()
                return
            row = selected_rows[0].row()
            if row < 0 or row >= len(self._all_results):
                return
            result = self._all_results[row]
            group = _get(result, "group", default=result)
            analysis = _get(result, "analysis", default=None) or group

            gid = _get(result, "group_id", default=_get(group, "group_id", "id", default=""))
            all_count = _get(analysis, "image_count", default=_get(group, "image_count", default="—"))
            selected_names = _extract_selected_names(result, analysis)
            selected_count = _get(analysis, "selected_count", default=(len(selected_names) if selected_names else "—"))
            output_item = self.table.item(row, 5)
            output_path = str(output_item.data(Qt.UserRole) or "") if output_item else ""
            if not output_path:
                raw_out = _get(result, "output_path", default=_get(group, "output_path", default=""))
                output_path = str(raw_out) if raw_out else ""
            self._current_output_path = output_path

            status_text = self.table.item(row, 4).text() if self.table.item(row, 4) else "—"
            raw_status = str(self.table.item(row, 4).data(Qt.UserRole) or "") if self.table.item(row, 4) else ""
            coverage_text = self.table.item(row, 3).text() if self.table.item(row, 3) else ""

            range_str = self._result_ranges[row] if row < len(self._result_ranges) else "—"
            images = self._result_images[row] if row < len(self._result_images) else []

            t = self._tokens
            self.info_group_id.setText(f"组 #{gid}")
            self.info_status_pill.setText(status_text)
            if "FAILED" in raw_status or "ERROR" in raw_status:
                pill_style = f"background: {t.danger_soft}; color: {t.danger}; border: 1px solid {t.danger_border};"
            elif raw_status == "DONE":
                pill_style = f"background: {t.success_soft}; color: {t.success}; border: 1px solid {t.success_border};"
            else:
                pill_style = f"background: {t.bg_rail}; color: {t.text_muted}; border: 1px solid {t.border};"
            self.info_status_pill.setStyleSheet(
                f"{pill_style} padding: 2px 8px; border-radius: 5px; font-size: 11px; font-weight: 600;"
            )

            # Backend / fallback badge
            actual_backend = _get(result, "actual_backend", default="")
            requested_backend = _get(result, "requested_backend", default="")
            fallback_used = bool(_get(result, "fallback_used", default=False))
            backend_name = _backend_label(actual_backend or requested_backend)
            if backend_name:
                if fallback_used:
                    self.info_backend_pill.setText(f"回退至 {backend_name}")
                    self.info_backend_pill.setStyleSheet(
                        f"background: {t.warning_soft}; color: {t.warning}; border: 1px solid {t.warning_border}; "
                        "padding: 2px 7px; border-radius: 5px; font-size: 11px; font-weight: 600;"
                    )
                else:
                    self.info_backend_pill.setText(backend_name)
                    self.info_backend_pill.setStyleSheet(
                        f"color: {t.text_muted}; font-size: 11px;"
                    )
                self.info_backend_pill.show()
            else:
                self.info_backend_pill.hide()

            self.info_range.setText(f"原图范围：{range_str}")
            if images:
                preview_list = "\n".join(images[:40])
                if len(images) > 40:
                    preview_list += f"\n… 等共 {len(images)} 张照片"
                self.info_range.setToolTip(f"本组照片：\n{preview_list}")
            else:
                self.info_range.setToolTip(range_str)

            self.info_photos.setText(f"精选切片：{all_count} 张原图 → 精选 {selected_count} 张")
            self.info_status.setText(f"当前状态：{status_text}")

            # Slice list toggle
            if images or selected_names:
                sel_set = set(selected_names)
                lines = []
                source_list = images if images else selected_names
                for name in source_list:
                    marker = "✓ [选中]" if name in sel_set else "· [未选]"
                    lines.append(f"{marker}  {name}")
                self.slices_box.setPlainText("\n".join(lines))
                self.slices_toggle.show()
            else:
                self.slices_toggle.hide()
                self.slices_toggle.setChecked(False)
                self.slices_box.hide()

            # Coverage & crop / alignment metadata
            try:
                cov_val = float(coverage_text.replace("%", "").strip())
                self.coverage_bar.setValue(max(0, min(100, int(round(cov_val)))))
                self.coverage_text.setText(f"清晰覆盖率：{coverage_text}")
            except Exception:
                self.coverage_bar.setValue(0)
                self.coverage_text.setText(f"清晰覆盖率：{coverage_text or '—'}")

            align_parts: list[str] = []
            crop_ratio = _get(result, "crop_ratio", default=None)
            if isinstance(crop_ratio, (int, float)):
                align_parts.append(f"有效画幅 {float(crop_ratio):.1%}")
            align_status = _get(result, "alignment_status", default="")
            if align_status:
                align_parts.append(f"对齐 {align_status}")
            if align_parts:
                self.align_meta_label.setText(" · ".join(align_parts))
                self.align_meta_label.show()
            else:
                self.align_meta_label.hide()

            # Archive summary
            archive = _get(result, "archive_result", default=None)
            if archive is not None:
                records = getattr(archive, "records", ()) or ()
                failed_arc = getattr(archive, "failed", ()) or ()
                archived_n = sum(1 for r in records if getattr(r, "status", "") in {"archived", "already_archived"})
                if archived_n or failed_arc:
                    text = f"归档：已处理 {archived_n} 张"
                    if failed_arc:
                        text += f"（{len(failed_arc)} 张失败）"
                    self.info_archive.setText(text)
                    self.info_archive.show()
                else:
                    self.info_archive.hide()
            else:
                self.info_archive.hide()

            # Collect full error / exclusion / diagnostic reasons
            diag_lines: list[str] = []
            reason = _get(analysis, "excluded_reason", default="")
            if reason:
                diag_lines.append(f"排除原因：{reason}")
            error_obj = _get(result, "error", default=None)
            if error_obj:
                diag_lines.append(f"错误详情：{error_obj}")
            result_diags = _get(result, "diagnostics", default=())
            if isinstance(result_diags, (list, tuple)):
                for d in result_diags:
                    if d and str(d) not in diag_lines:
                        diag_lines.append(str(d))
            if diag_lines:
                is_error = bool(error_obj or "FAILED" in raw_status)
                self.reason_frame.setObjectName("diagBanner" if is_error else "warnBanner")
                self.reason_frame.style().unpolish(self.reason_frame)
                self.reason_frame.style().polish(self.reason_frame)
                self.info_reason.setStyleSheet(f"color: {t.danger if is_error else t.warning}; font-size: 12px;")
                self.info_reason.setText("\n".join(diag_lines))
                self.reason_frame.show()
            else:
                self.reason_frame.hide()

            # Load output thumbnail preview
            if self._current_output_path and os.path.exists(self._current_output_path):
                pixmap = QPixmap(self._current_output_path)
                if not pixmap.isNull():
                    self.preview_stage.show()
                    self._action_row_widget.show()
                    self.preview_image.setFixedHeight(168)
                    self._current_pixmap = pixmap
                    self._render_preview_pixmap()
                    try:
                        size_mb = Path(self._current_output_path).stat().st_size / (1024 * 1024)
                        self.preview_meta.setText(f"{pixmap.width()} × {pixmap.height()} px · {size_mb:.2f} MB · 双击放大")
                    except OSError:
                        self.preview_meta.setText(f"{pixmap.width()} × {pixmap.height()} px · 双击放大")
                    self.preview_meta.show()
                    self.zoom_btn.setEnabled(True)
                    self.open_file_btn.setEnabled(True)
                    self.reveal_file_btn.setEnabled(True)
                else:
                    self.preview_stage.show()
                    self._action_row_widget.show()
                    self.preview_image.setFixedHeight(78)
                    self._current_pixmap = None
                    self.preview_image.clear()
                    self.preview_image.setText("无法解码图片预览")
                    self.preview_meta.hide()
                    self.zoom_btn.setEnabled(False)
                    self.open_file_btn.setEnabled(True)
                    self.reveal_file_btn.setEnabled(True)
            else:
                self._current_pixmap = None
                self.preview_image.clear()
                self.preview_meta.hide()
                self.zoom_btn.setEnabled(False)
                self.open_file_btn.setEnabled(False)
                self.reveal_file_btn.setEnabled(False)
                self._action_row_widget.hide()
                if diag_lines:
                    # Focus the inspector directly on the failure / exclusion reason
                    self.preview_stage.hide()
                else:
                    self.preview_stage.show()
                    self.preview_image.setFixedHeight(78)
                    self.preview_image.setText("暂无合成图片（未完成或无需合成）")

        def _open_lightbox(self) -> None:
            if self._current_output_path and os.path.exists(self._current_output_path):
                dlg = ImageLightboxDialog(self._current_output_path, tokens=self._tokens, parent=self.window())
                dlg.exec()

        def _open_current_image(self) -> None:
            if self._current_output_path and os.path.exists(self._current_output_path):
                QDesktopServices.openUrl(QUrl.fromLocalFile(self._current_output_path))

        def _reveal_current_image(self) -> None:
            if not self._current_output_path:
                return
            path = Path(self._current_output_path).resolve()
            if sys.platform == "win32":
                if path.exists():
                    subprocess.run(["explorer", f"/select,{path}"], check=False)
                elif path.parent.exists():
                    subprocess.run(["explorer", str(path.parent)], check=False)
            else:
                target = str(path if path.exists() else path.parent)
                QDesktopServices.openUrl(QUrl.fromLocalFile(target))

except ImportError:  # pragma: no cover

    class GroupListWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError("PySide6 is required to create the desktop UI")


__all__ = ["GroupListWidget"]
