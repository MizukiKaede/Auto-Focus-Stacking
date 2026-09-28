"""Responsive progress panel; all work is performed by pipeline workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..pipeline.events import PipelineEvent, PipelineStage


@dataclass
class ProgressSnapshot:
    stage: str = "idle"
    current_file: str = ""
    current_group: str = ""
    overall: int = 0
    analysis: int = 0
    merge: int = 0
    analysis_completed: int = 0
    analysis_total: int = 0
    merge_completed: int = 0
    merge_total: int = 0
    merge_finished: int = 0
    groups_found: int = 0
    groups_finished: int = 0
    errors: int = 0
    message: str = ""

    @classmethod
    def from_event(cls, event: PipelineEvent | Any) -> "ProgressSnapshot":
        def value(name: str, default: Any = "") -> Any:
            if isinstance(event, dict):
                return event.get(name, default)
            return getattr(event, name, default)

        total = int(value("total", 0) or 0)
        completed = int(value("completed", 0) or 0)
        analysis_total = int(value("analysis_total", total) or 0)
        merge_total = int(value("merge_total", total) or 0)
        analysis_completed = int(value("analysis_completed", completed) or 0)
        merge_completed = int(value("merge_completed", completed) or 0)
        raw_merge_finished = value("merge_finished", None)
        if raw_merge_finished is None:
            raw_merge_finished = value("groups_finished", merge_completed)
        merge_finished = int(raw_merge_finished or 0)
        overall_total = analysis_total + merge_total
        # Archive-only jobs (including analysis failures) are not successful
        # fusions, so ``merge_completed`` intentionally stays lower.  Overall
        # progress reflects consumed merge work via ``merge_finished``.
        overall_done = analysis_completed + merge_finished
        overall = int(round(100 * (overall_done / overall_total if overall_total else (completed / total if total else 0))))
        stage = value("stage", "idle")
        stage = stage.value if isinstance(stage, PipelineStage) else str(stage)
        return cls(
            stage=stage,
            current_file=str(value("current_file", "") or ""),
            current_group=str(value("current_group", "") or ""),
            overall=max(0, min(100, overall)),
            analysis=max(0, min(100, int(round(100 * (analysis_completed / analysis_total if analysis_total else 0))))),
            merge=max(0, min(100, int(round(100 * (merge_completed / merge_total if merge_total else 0))))),
            analysis_completed=analysis_completed,
            analysis_total=analysis_total,
            merge_completed=merge_completed,
            merge_total=merge_total,
            merge_finished=merge_finished,
            groups_found=int(value("groups_found", 0) or 0),
            groups_finished=int(value("groups_finished", 0) or 0),
            errors=int(value("errors", 0) or 0),
            message=str(value("message", "") or ""),
        )


try:  # PySide6 is optional for headless tests and CLI use.
    from PySide6.QtCore import Slot
    from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QTextEdit, QVBoxLayout, QWidget
    from .theme import ElidedLabel

    STAGE_LABELS = {
        "idle": "空闲", "scanning": "扫描照片", "analysis": "分析选片",
        "archive": "归档原图", "alignment": "对齐照片", "fusion": "合成照片",
        "complete": "处理完成", "cancelled": "已取消", "error": "处理出错",
    }

    class ProgressPanel(QFrame):
        """Compact progress overview with bounded diagnostic details."""

        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)
            self.setObjectName("panel")
            self.snapshot = ProgressSnapshot()
            layout = QVBoxLayout(self)
            layout.setContentsMargins(20, 16, 20, 16)
            layout.setSpacing(12)
            heading = QHBoxLayout()
            title = QLabel("处理概览")
            title.setProperty("role", "section")
            self.stage_label = QLabel("空闲")
            self.stage_label.setProperty("role", "muted")
            heading.addWidget(title)
            heading.addStretch()
            heading.addWidget(self.stage_label)
            layout.addLayout(heading)
            metrics = QHBoxLayout()
            self.metric_labels = []
            for text in ("总体进度", "已发现组数", "已完成组数", "错误"):
                column = QVBoxLayout()
                value = QLabel("0%" if not self.metric_labels else "0")
                value.setProperty("role", "metric")
                label = QLabel(text)
                label.setProperty("role", "muted")
                column.addWidget(value)
                column.addWidget(label)
                metrics.addLayout(column, 1)
                self.metric_labels.append(value)
            self.metric_labels[-1].setStyleSheet("color: #F28B82;")
            layout.addLayout(metrics)
            self.overall_bar = QProgressBar()
            self.analysis_bar = QProgressBar()
            self.merge_bar = QProgressBar()
            for bar in (self.overall_bar, self.analysis_bar, self.merge_bar):
                bar.setRange(0, 100)
                bar.setValue(0)
                bar.setTextVisible(False)
                bar.setFixedHeight(6)
            layout.addWidget(self.overall_bar)
            stages = QHBoxLayout()
            self.analysis_count = QLabel("分析  0 / 0")
            self.merge_count = QLabel("合成  0 / 0")
            for label, bar in ((self.analysis_count, self.analysis_bar), (self.merge_count, self.merge_bar)):
                column = QVBoxLayout()
                label.setProperty("role", "muted")
                column.addWidget(label)
                column.addWidget(bar)
                stages.addLayout(column, 1)
            stages.setSpacing(24)
            layout.addLayout(stages)
            current = QHBoxLayout()
            self.file_label = ElidedLabel("等待开始")
            self.group_label = ElidedLabel("—")
            self.group_label.setMaximumWidth(150)
            current.addWidget(self.file_label, 1)
            current.addWidget(self.group_label)
            layout.addLayout(current)
            # Compatibility labels remain available without duplicating the overview.
            self.count_label = QLabel(self)
            self.stats_label = QLabel(self)
            self.count_label.hide()
            self.stats_label.hide()
            self.details_toggle = QPushButton("诊断详情 · 展开")
            self.details_toggle.setCheckable(True)
            self.details_toggle.setEnabled(False)
            self.details_toggle.toggled.connect(self._toggle_details)
            layout.addWidget(self.details_toggle)
            self.message_label = QLabel(self)
            self.message_label.hide()
            self.details_text = QTextEdit()
            self.details_text.setReadOnly(True)
            self.details_text.setFixedHeight(88)
            self.details_text.hide()
            layout.addWidget(self.details_text)

        def _toggle_details(self, checked):
            self.details_text.setVisible(checked)
            self.details_toggle.setText("诊断详情 · 收起" if checked else "诊断详情 · 展开")

        def set_message(self, message):
            self.message_label.setText(message)
            self.details_text.setPlainText(message)
            self.details_toggle.setEnabled(bool(message))
            if not message:
                self.details_toggle.setChecked(False)

        def reset(self):
            self.details_toggle.setChecked(False)
            self.update_event({"stage": "scanning"})

        @Slot(object)
        def update_event(self, event: PipelineEvent | Any) -> None:
            snapshot = ProgressSnapshot.from_event(event)
            self.snapshot = snapshot
            self.overall_bar.setValue(snapshot.overall)
            self.analysis_bar.setValue(snapshot.analysis)
            self.merge_bar.setValue(snapshot.merge)
            self.stage_label.setText(STAGE_LABELS.get(snapshot.stage, snapshot.stage))
            self.file_label.setText(snapshot.current_file or "—")
            self.group_label.setText(f"组 {snapshot.current_group}" if snapshot.current_group else "—")
            self.analysis_count.setText(f"分析  {snapshot.analysis_completed} / {snapshot.analysis_total}")
            self.merge_count.setText(f"合成  {snapshot.merge_completed} / {snapshot.merge_total}")
            self.count_label.setText(
                f"分析：{snapshot.analysis_completed} / {snapshot.analysis_total}    "
                f"合成：{snapshot.merge_completed} / {snapshot.merge_total}"
            )
            self.stats_label.setText(
                f"已发现：{snapshot.groups_found}    已完成：{snapshot.groups_finished}    错误：{snapshot.errors}"
            )
            for label, value in zip(self.metric_labels, (f"{snapshot.overall}%", snapshot.groups_found, snapshot.groups_finished, snapshot.errors)):
                label.setText(str(value))
            self.set_message(snapshot.message)

        def complete(self, summary: Any) -> None:
            """Keep original overall completion semantics, including cancellation."""
            cancelled = bool(getattr(summary, "cancelled", False))
            errors = int(getattr(summary, "errors", 0) or 0)
            self.stage_label.setText("已取消" if cancelled else "处理结束 · 有错误" if errors else "处理完成")
            if bool(getattr(summary, "finished", False)) and not cancelled:
                self.snapshot.overall = 100
                self.overall_bar.setValue(100)
                self.metric_labels[0].setText("100%")
            self.metric_labels[-1].setText(str(errors))
            diagnostics = list(getattr(summary, "diagnostics", ()) or ())
            if diagnostics:
                self.set_message("\n\n".join(str(item) for item in diagnostics))

except ImportError:  # pragma: no cover - exercised in this environment

    class ProgressPanel:  # type: ignore[no-redef]
        """Import-safe placeholder when PySide6 is not installed."""

        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError("PySide6 is required to create the desktop UI")


__all__ = ["ProgressPanel", "ProgressSnapshot"]

