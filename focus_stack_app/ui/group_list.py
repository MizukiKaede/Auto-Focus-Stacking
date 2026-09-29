"""Compact results table with interactive master-detail image preview and inspector."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


_COLUMNS = ("组", "原图", "选中", "覆盖率", "状态", "输出")


def _get(value: Any, *names: str, default: Any = "") -> Any:
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
    else:
        for name in names:
            if hasattr(value, name):
                return getattr(value, name)
    return default


def _extract_image_names(result: Any) -> list[str]:
    """Extract ordered list of original photo filenames for this result."""
    names: list[str] = []

    def add_item(item: Any) -> None:
        if not item:
            return
        if hasattr(item, "filename") and item.filename:
            names.append(Path(str(item.filename)).name)
        elif hasattr(item, "name") and item.name:
            names.append(Path(str(item.name)).name)
        elif hasattr(item, "path") and item.path:
            names.append(Path(str(item.path)).name)
        elif hasattr(item, "source_path") and item.source_path:
            names.append(Path(str(item.source_path)).name)
        elif isinstance(item, (str, os.PathLike)):
            names.append(Path(os.fspath(item)).name)

    # 1. Direct all_paths on result
    all_paths = _get(result, "all_paths", default=None)
    if all_paths and isinstance(all_paths, (list, tuple)):
        for p in all_paths:
            add_item(p)
        if names:
            return names

    # 2. Group items
    group = _get(result, "group", default=None)
    if group is not None:
        for attr in ("items", "all_images", "images", "image_records"):
            items = _get(group, attr, default=None)
            if items and isinstance(items, (list, tuple)):
                for it in items:
                    add_item(it)
                if names:
                    return names

    # 3. Analysis items
    analysis = _get(result, "analysis", default=None)
    if analysis is not None:
        for attr in ("all_images", "images", "items", "all_paths"):
            items = _get(analysis, attr, default=None)
            if items and isinstance(items, (list, tuple)):
                for it in items:
                    add_item(it)
                if names:
                    return names

    # 4. Archive result records
    archive = _get(result, "archive_result", default=None)
    if archive is not None:
        records = getattr(archive, "records", ()) or ()
        for r in records:
            add_item(r)
        if names:
            return names

    # 5. Direct images attribute on result/dict
    for attr in ("images", "items", "photo_paths"):
        val = _get(result, attr, default=None)
        if val and isinstance(val, (list, tuple)):
            for it in val:
                add_item(it)
            if names:
                return names

    # 6. Fallback: first_original
    first = _get(result, "first_original", default=_get(group, "first_original", default=None))
    if first:
        add_item(first)

    return names


try:
    from PySide6.QtCore import Qt, QUrl, Slot
    from PySide6.QtGui import QColor, QDesktopServices, QPixmap
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QFrame,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    from .theme import ElidedLabel

    class GroupListWidget(QFrame):
        """Results studio with searchable group table and real-time inspector."""

        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)
            self.setObjectName("panel")
            self._all_results: list[Any] = []
            self._result_images: list[list[str]] = []
            self._result_ranges: list[str] = []
            self._filter_status = "all"
            self._current_output_path = ""

            main_layout = QVBoxLayout(self)
            main_layout.setContentsMargins(18, 14, 18, 14)
            main_layout.setSpacing(10)

            # --- Header Toolbar ---
            toolbar = QHBoxLayout()
            toolbar.setSpacing(10)
            title = QLabel("合成结果")
            title.setProperty("role", "section")
            toolbar.addWidget(title)

            self.count_badge = QLabel("0 组")
            self.count_badge.setProperty("role", "badge")
            self.count_badge.setStyleSheet(
                "background: #141822; color: #94A3B8; border: 1px solid #232B3A; padding: 2px 8px;"
            )
            toolbar.addWidget(self.count_badge)
            toolbar.addStretch()

            # Search Box (supports fuzzy matching original photo names)
            self.search_box = QLineEdit()
            self.search_box.setPlaceholderText("🔍 搜索原图名 (如 0015) 或组号…")
            self.search_box.setToolTip("输入组号、原图文件名（如 0015 或 DSC0015）或输出路径，快速定位对应图组")
            self.search_box.setFixedWidth(210)
            self.search_box.setFixedHeight(28)
            self.search_box.textChanged.connect(self._apply_filter)
            toolbar.addWidget(self.search_box)

            # Filter Buttons: 全部 / 已完成 / 无需合成 / 失败
            self.filter_buttons: list[QPushButton] = []
            for filter_id, filter_label in (
                ("all", "全部"),
                ("done", "已完成"),
                ("skipped", "无需合成"),
                ("failed", "失败"),
            ):
                btn = QPushButton(filter_label)
                btn.setProperty("filter_id", filter_id)
                btn.setCheckable(True)
                btn.setChecked(filter_id == "all")
                btn.setFixedHeight(28)
                btn.setStyleSheet(
                    "QPushButton { font-size: 11px; padding: 0 8px; min-height: 26px; } "
                    "QPushButton:checked { background: #2563EB; color: #FFFFFF; border-color: #3B82F6; }"
                )
                btn.clicked.connect(lambda checked, fid=filter_id: self._set_filter(fid))
                toolbar.addWidget(btn)
                self.filter_buttons.append(btn)

            main_layout.addLayout(toolbar)

            # --- Central Splitter: Left Table + Right Detail Inspector ---
            self.splitter = QSplitter(Qt.Horizontal)
            self.splitter.setChildrenCollapsible(False)
            main_layout.addWidget(self.splitter, 1)

            # Left: Table
            self.table = QTableWidget(0, len(_COLUMNS))
            self.table.setHorizontalHeaderLabels(list(_COLUMNS))
            self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
            for col, width in enumerate((56, 56, 56, 76, 120)):
                self.table.setColumnWidth(col, width)
            self.table.verticalHeader().hide()
            self.table.verticalHeader().setDefaultSectionSize(36)
            self.table.setShowGrid(False)
            self.table.setWordWrap(False)
            self.table.setAlternatingRowColors(True)
            self.table.setTextElideMode(Qt.ElideMiddle)
            self.table.itemSelectionChanged.connect(self._on_selection_changed)
            self.table.cellDoubleClicked.connect(lambda r, c: self._open_current_image())
            self.splitter.addWidget(self.table)

            # Right: Inspector Panel
            self.inspector = QFrame()
            self.inspector.setObjectName("innerCard")
            self.inspector.setMinimumWidth(260)
            self.inspector.setMaximumWidth(340)
            inspector_layout = QVBoxLayout(self.inspector)
            inspector_layout.setContentsMargins(12, 10, 12, 10)
            inspector_layout.setSpacing(6)

            insp_header = QHBoxLayout()
            insp_header.setSpacing(6)
            insp_title = QLabel("图组检查器")
            insp_title.setProperty("role", "section")
            insp_header.addWidget(insp_title)
            insp_header.addStretch()

            self.open_file_btn = QPushButton("打开大图")
            self.open_file_btn.setEnabled(False)
            self.open_file_btn.setFixedHeight(24)
            self.open_file_btn.setStyleSheet("font-size: 11px; padding: 0 6px;")
            self.open_file_btn.setToolTip("用系统默认看图工具全屏查看合成结果")
            self.open_file_btn.clicked.connect(self._open_current_image)

            self.reveal_file_btn = QPushButton("定位文件")
            self.reveal_file_btn.setEnabled(False)
            self.reveal_file_btn.setFixedHeight(24)
            self.reveal_file_btn.setStyleSheet("font-size: 11px; padding: 0 6px;")
            self.reveal_file_btn.setToolTip("在资源管理器中高亮选中此文件")
            self.reveal_file_btn.clicked.connect(self._reveal_current_image)

            insp_header.addWidget(self.open_file_btn)
            insp_header.addWidget(self.reveal_file_btn)
            inspector_layout.addLayout(insp_header)

            # Thumbnail preview box
            self.preview_image = QLabel()
            self.preview_image.setAlignment(Qt.AlignCenter)
            self.preview_image.setFixedHeight(105)
            self.preview_image.setStyleSheet(
                "background: #0D1117; border: 1px dashed #2A3345; border-radius: 6px; color: #64748B; padding: 2px;"
            )
            self.preview_image.setText("暂无图片预览\n点击列表项查看")
            inspector_layout.addWidget(self.preview_image)

            # Detail info lines
            self.info_group_id = QLabel("图组：—")
            self.info_group_id.setStyleSheet("font-weight: 600; color: #F1F5F9; font-size: 12px;")

            # Original Photo Range (e.g. DSC0000 - DSC0015)
            self.info_range = QLabel("原图范围：—")
            self.info_range.setStyleSheet("color: #93C5FD; font-size: 12px; font-weight: 500;")
            self.info_range.setWordWrap(True)

            self.info_photos = QLabel("精选切片：—")
            self.info_photos.setProperty("role", "muted")

            self.info_status = QLabel("状态：—")
            self.info_status.setStyleSheet("font-size: 12px;")

            # Coverage Bar
            coverage_box = QVBoxLayout()
            coverage_box.setSpacing(3)
            self.coverage_text = QLabel("清晰覆盖率：—")
            self.coverage_text.setProperty("role", "muted")
            self.coverage_bar = QProgressBar()
            self.coverage_bar.setRange(0, 100)
            self.coverage_bar.setValue(0)
            self.coverage_bar.setFixedHeight(5)
            self.coverage_bar.setTextVisible(False)
            coverage_box.addWidget(self.coverage_text)
            coverage_box.addWidget(self.coverage_bar)

            self.info_reason = QLabel()
            self.info_reason.setProperty("role", "muted")
            self.info_reason.setStyleSheet("color: #F87171; font-size: 11px;")
            self.info_reason.setWordWrap(True)
            self.info_reason.hide()

            inspector_layout.addWidget(self.info_group_id)
            inspector_layout.addWidget(self.info_range)
            inspector_layout.addWidget(self.info_photos)
            inspector_layout.addWidget(self.info_status)
            inspector_layout.addLayout(coverage_box)
            inspector_layout.addWidget(self.info_reason)
            inspector_layout.addStretch()

            self.splitter.addWidget(self.inspector)
            self.splitter.setStretchFactor(0, 3)
            self.splitter.setStretchFactor(1, 2)

            # --- Empty State ---
            self.empty_label = QLabel("选择照片文件夹，开始景深合成")
            self.empty_label.setProperty("role", "muted")
            self.empty_label.setAlignment(Qt.AlignCenter)
            self.empty_label.setWordWrap(True)
            main_layout.addWidget(self.empty_label, 1)

            # Initial visibility
            self.table.hide()
            self.inspector.hide()
            self.splitter.hide()

        def set_empty_message(self, message: str) -> None:
            self.clear()
            self.empty_label.setText(message)

        def clear(self) -> None:
            self._all_results.clear()
            self._result_images.clear()
            self._result_ranges.clear()
            self.table.setRowCount(0)
            self.count_badge.setText("0 组")
            self.table.hide()
            self.inspector.hide()
            self.splitter.hide()
            self.empty_label.show()
            self._reset_inspector()

        def _reset_inspector(self):
            self._current_output_path = ""
            self.preview_image.clear()
            self.preview_image.setText("暂无图片预览\n点击列表项查看")
            self.open_file_btn.setEnabled(False)
            self.reveal_file_btn.setEnabled(False)
            self.info_group_id.setText("图组：—")
            self.info_range.setText("原图范围：—")
            self.info_range.setToolTip("")
            self.info_photos.setText("精选切片：—")
            self.info_status.setText("状态：—")
            self.coverage_text.setText("清晰覆盖率：—")
            self.coverage_bar.setValue(0)
            self.info_reason.hide()

        def _set_filter(self, filter_id: str):
            self._filter_status = filter_id
            for btn in self.filter_buttons:
                btn.setChecked(btn.property("filter_id") == filter_id)
            self._apply_filter()

        def _apply_filter(self):
            search_text = self.search_box.text().strip().lower()
            fid = self._filter_status

            visible_count = 0
            for row in range(self.table.rowCount()):
                gid_item = self.table.item(row, 0)
                status_item = self.table.item(row, 4)
                output_item = self.table.item(row, 5)

                gid = gid_item.text() if gid_item else ""
                status = status_item.text() if status_item else ""
                output = output_item.text() if output_item else ""
                images = self._result_images[row] if row < len(self._result_images) else []
                range_str = self._result_ranges[row] if row < len(self._result_ranges) else ""

                # Text filter: matches group id, output path, status, range string, or ANY photo filename
                text_match = True
                if search_text:
                    gid_match = (
                        search_text == gid
                        or search_text == f"#{gid}"
                        or f"组 #{gid}" in search_text
                        or search_text in gid.lower()
                    )
                    output_match = search_text in output.lower()
                    status_match = search_text in status.lower()
                    range_match = search_text in range_str.lower()
                    # Fuzzy match against any photo name in this group (e.g. '0015' matches 'DSC0015.JPG')
                    photo_match = any(search_text in img.lower() for img in images)

                    text_match = gid_match or output_match or status_match or range_match or photo_match

                # Category filter: 全部 / 已完成 / 无需合成 / 失败
                cat_match = True
                if fid == "done":
                    cat_match = "已完成" in status
                elif fid == "skipped":
                    cat_match = any(k in status for k in ("无需", "跳过", "未合成", "排除", "单张", "重复"))
                elif fid == "failed":
                    cat_match = any(k in status for k in ("失败", "错误", "异常"))

                hide_row = not (text_match and cat_match)
                self.table.setRowHidden(row, hide_row)
                if not hide_row:
                    visible_count += 1

            self.count_badge.setText(
                f"{visible_count} 组" if visible_count != self.table.rowCount() else f"{self.table.rowCount()} 组"
            )

            # Auto-select the first visible row so inspector stays up-to-date
            if visible_count > 0:
                selected_rows = self.table.selectionModel().selectedRows()
                if not selected_rows or self.table.isRowHidden(selected_rows[0].row()):
                    for r in range(self.table.rowCount()):
                        if not self.table.isRowHidden(r):
                            self.table.selectRow(r)
                            break
            else:
                self._reset_inspector()

        def add_result(self, result: Any) -> None:
            self._all_results.append(result)

            # Extract image filenames and compute range text
            images = _extract_image_names(result)
            self._result_images.append(images)
            if images:
                if len(images) == 1:
                    range_text = images[0]
                else:
                    range_text = f"{images[0]} - {images[-1]} (共 {len(images)} 张)"
            else:
                range_text = "—"
            self._result_ranges.append(range_text)

            self.empty_label.hide()
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
            output = _get(result, "output_path", default=_get(group, "output_path", default=""))
            status = _get(result, "status", default=_get(group, "status", default=""))
            raw_status = str(getattr(status, "value", status))
            status_map = {
                "DONE": "已完成",
                "FAILED": "合成失败",
                "FAILED_CLASSIFICATION": "分析失败",
                "CANCELLED": "已取消",
                "RUNNING": "处理中",
                "SKIPPED_SINGLE": "单张（未合成）",
                "NO_MERGE_SINGLE": "单张（未合成）",
                "NO_MERGE_REPEATED": "重复（未合成）",
                "NO_MERGE": "无需合成",
                "SKIPPED": "已跳过",
                "CLASSIFIED": "已分类",
                "SELECTED": "已选片",
                "READY_FOR_MERGE": "待合成",
            }
            status_text = status_map.get(raw_status, raw_status)
            reason = _get(analysis, "excluded_reason", default=None)
            if reason:
                status_text = "已排除：" + str(reason)
            elif status_text == "SKIPPED_SINGLE":
                status_text = "单张（未合成）"

            coverage = _get(analysis, "coverage", "group_coverage", default="")
            coverage_num = 0.0
            if isinstance(coverage, (int, float)):
                coverage_num = float(coverage)
                coverage_str = f"{coverage_num:.1%}"
            else:
                coverage_str = str(coverage)

            values = [gid, all_count, selected, coverage_str, status_text, output]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value if value is not None else ""))
                item.setToolTip(item.text())
                if col < 4:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if col == 4:
                    color = "#F87171" if "FAILED" in raw_status else "#34D399" if raw_status == "DONE" else "#94A3B8"
                    item.setForeground(QColor(color))
                self.table.setItem(row, col, item)

            self.count_badge.setText(f"{self.table.rowCount()} 组")
            self._apply_filter()

            # Automatically select first row if none selected
            if self.table.rowCount() == 1:
                self.table.selectRow(0)

        def update_results(self, results: Iterable[Any]) -> None:
            self.clear()
            for result in results:
                self.add_result(result)

        def _on_selection_changed(self):
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
            selected_count = _get(analysis, "selected_count", default="—")
            output_path = _get(result, "output_path", default=_get(group, "output_path", default=""))
            self._current_output_path = str(output_path) if output_path else ""

            status_text = self.table.item(row, 4).text() if self.table.item(row, 4) else "—"
            coverage_text = self.table.item(row, 3).text() if self.table.item(row, 3) else ""

            range_str = self._result_ranges[row] if row < len(self._result_ranges) else "—"
            images = self._result_images[row] if row < len(self._result_images) else []

            self.info_group_id.setText(f"图组标识：组 #{gid}")
            self.info_range.setText(f"原图范围：{range_str}")
            if images:
                preview_list = "\n".join(images[:40])
                if len(images) > 40:
                    preview_list += f"\n... 等共 {len(images)} 张照片"
                self.info_range.setToolTip(f"本组包含的照片：\n{preview_list}")
            else:
                self.info_range.setToolTip(range_str)

            self.info_photos.setText(f"精选切片：{all_count} 张原图 ➔ 精选 {selected_count} 张合成")
            self.info_status.setText(f"当前状态：{status_text}")

            # Coverage
            try:
                cov_val = float(coverage_text.replace("%", "").strip())
                self.coverage_bar.setValue(int(cov_val))
                self.coverage_text.setText(f"清晰覆盖率：{coverage_text}")
            except Exception:
                self.coverage_bar.setValue(0)
                self.coverage_text.setText(f"清晰覆盖率：{coverage_text or '—'}")

            reason = _get(analysis, "excluded_reason", default="")
            if reason:
                self.info_reason.setText(f"排除原因：{reason}")
                self.info_reason.show()
            else:
                self.info_reason.hide()

            # Load thumbnail preview if output file exists
            if self._current_output_path and os.path.exists(self._current_output_path):
                pixmap = QPixmap(self._current_output_path)
                if not pixmap.isNull():
                    scaled_pix = pixmap.scaled(
                        max(100, self.preview_image.width() - 8),
                        100,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                    self.preview_image.setPixmap(scaled_pix)
                    self.open_file_btn.setEnabled(True)
                    self.reveal_file_btn.setEnabled(True)
                else:
                    self.preview_image.setText("无法解码图片预览")
                    self.open_file_btn.setEnabled(True)
                    self.reveal_file_btn.setEnabled(True)
            else:
                self.preview_image.setText("暂无合成图片结果\n（未完成或无需合成）")
                self.open_file_btn.setEnabled(False)
                self.reveal_file_btn.setEnabled(False)

        def _open_current_image(self):
            if self._current_output_path and os.path.exists(self._current_output_path):
                QDesktopServices.openUrl(QUrl.fromLocalFile(self._current_output_path))

        def _reveal_current_image(self):
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
