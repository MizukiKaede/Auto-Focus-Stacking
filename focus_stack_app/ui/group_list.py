"""Compact results table for group status and coverage."""

from __future__ import annotations

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


try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QAbstractItemView, QFrame, QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

    class GroupListWidget(QFrame):
        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)
            self.setObjectName("panel")
            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)
            title = QLabel("合成结果")
            title.setProperty("role", "section")
            layout.addWidget(title)
            self.table = QTableWidget(0, len(_COLUMNS))
            self.table.setHorizontalHeaderLabels(list(_COLUMNS))
            self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
            for col, width in enumerate((56, 56, 56, 76, 130)):
                self.table.setColumnWidth(col, width)
            self.table.verticalHeader().hide()
            self.table.verticalHeader().setDefaultSectionSize(38)
            self.table.setShowGrid(False)
            self.table.setWordWrap(False)
            self.table.setAlternatingRowColors(True)
            self.table.setTextElideMode(Qt.ElideMiddle)
            layout.addWidget(self.table, 1)
            self.empty_label = QLabel("选择照片文件夹，开始景深合成")
            self.empty_label.setProperty("role", "muted")
            self.empty_label.setAlignment(Qt.AlignCenter)
            self.empty_label.setWordWrap(True)
            layout.addWidget(self.empty_label, 1)
            self.table.hide()

        def set_empty_message(self, message: str) -> None:
            self.clear()
            self.empty_label.setText(message)

        def clear(self) -> None:
            self.table.setRowCount(0)
            self.table.hide()
            self.empty_label.show()

        def add_result(self, result: Any) -> None:
            self.empty_label.hide()
            self.table.show()
            group = _get(result, "group", default=result)
            analysis = _get(result, "analysis", default=None) or group
            row = self.table.rowCount()
            self.table.insertRow(row)
            gid = _get(result, "group_id", default=_get(group, "group_id", "id", default=""))
            all_count = _get(analysis, "image_count", default=_get(group, "image_count", default=""))
            archive = _get(result, "archive_result", default=None)
            if archive is not None:
                all_count = len(getattr(archive, "records", ())) or all_count
            selected = _get(analysis, "selected_count", default="")
            output = _get(result, "output_path", default=_get(group, "output_path", default=""))
            status = _get(result, "status", default=_get(group, "status", default=""))
            raw_status = str(getattr(status, "value", status))
            status = {
                "DONE": "已完成", "FAILED": "合成失败", "FAILED_CLASSIFICATION": "分析失败",
                "CANCELLED": "已取消", "RUNNING": "处理中", "SKIPPED_SINGLE": "单张（未合成）",
                "NO_MERGE_SINGLE": "单张（未合成）", "NO_MERGE_REPEATED": "重复（未合成）",
                "NO_MERGE": "无需合成", "SKIPPED": "已跳过", "CLASSIFIED": "已分类",
                "SELECTED": "已选片", "READY_FOR_MERGE": "待合成",
            }.get(raw_status, raw_status)
            if _get(analysis, "excluded_reason", default=None):
                status = "已排除：" + str(_get(analysis, "excluded_reason"))
            elif status == "SKIPPED_SINGLE":
                status = "单张（未合成）"
            coverage = _get(analysis, "coverage", "group_coverage", default="")
            if isinstance(coverage, (int, float)):
                coverage = f"{coverage:.1%}"
            values = [gid, all_count, selected, coverage, status, output]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value if value is not None else ""))
                item.setToolTip(item.text())
                if col < 4:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if col == 4:
                    color = "#F28B82" if "FAILED" in raw_status else "#81C995" if raw_status == "DONE" else "#A8AFBA"
                    item.setForeground(QColor(color))
                self.table.setItem(row, col, item)

        def update_results(self, results: Iterable[Any]) -> None:
            self.clear()
            for result in results:
                self.add_result(result)

except ImportError:  # pragma: no cover - headless environment

    class GroupListWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError("PySide6 is required to create the desktop UI")


__all__ = ["GroupListWidget"]

