"""Responsive modern progress panel; all work is performed by pipeline workers."""

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
    from PySide6.QtCore import Qt, Slot
    from PySide6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    from .theme import ElidedLabel

    STAGE_LABELS = {
        "idle": "空闲",
        "scanning": "正在扫描照片与自动分组…",
        "analysis": "正在进行全图焦点清晰度分析…",
        "archive": "正在执行原图安全归档…",
        "alignment": "正在进行亚像素高精度对齐…",
        "fusion": "正在进行边缘一致性多频融合…",
        "complete": "全部处理完成",
        "cancelled": "处理已安全终止",
        "error": "处理遇到错误",
    }

    PIPELINE_STEPS = ["扫描分组", "焦点分析", "图像对齐", "景深融合", "完成归档"]

    class ProgressPanel(QFrame):
        """Modern studio progress cockpit with 5-stage stepper and real-time metrics."""

        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)
            self.setObjectName("panel")
            self.snapshot = ProgressSnapshot()
            layout = QVBoxLayout(self)
            layout.setContentsMargins(18, 14, 18, 14)
            layout.setSpacing(12)

            # --- Header ---
            heading = QHBoxLayout()
            heading.setSpacing(8)
            title = QLabel("处理概览")
            title.setProperty("role", "section")
            self.stage_label = QLabel("空闲")
            self.stage_label.setProperty("role", "muted")
            heading.addWidget(title)
            heading.addStretch()
            heading.addWidget(self.stage_label)
            layout.addLayout(heading)

            # --- 5-Stage Visual Stepper ---
            self.stepper_widget = QWidget()
            stepper_layout = QHBoxLayout(self.stepper_widget)
            stepper_layout.setContentsMargins(0, 0, 0, 0)
            stepper_layout.setSpacing(6)
            self.step_labels: list[QLabel] = []
            for i, name in enumerate(PIPELINE_STEPS):
                step_box = QLabel(f"{i + 1}. {name}")
                step_box.setAlignment(Qt.AlignCenter)
                step_box.setStyleSheet(
                    "background: #141821; color: #64748B; border: 1px solid #232B3A; "
                    "border-radius: 6px; padding: 4px 6px; font-size: 11px; font-weight: 500;"
                )
                stepper_layout.addWidget(step_box, 1)
                self.step_labels.append(step_box)
            layout.addWidget(self.stepper_widget)

            # --- 4 Metrics Cards ---
            metrics_layout = QHBoxLayout()
            metrics_layout.setSpacing(10)
            self.metric_labels: list[QLabel] = []
            metric_configs = [
                ("总体进度", "0%"),
                ("已发现组数", "0"),
                ("已完成组数", "0"),
                ("错误", "0"),
            ]
            for title_text, default_val in metric_configs:
                card = QFrame()
                card.setObjectName("innerCard")
                card_v = QVBoxLayout(card)
                card_v.setContentsMargins(12, 8, 12, 8)
                card_v.setSpacing(2)

                val_lbl = QLabel(default_val)
                val_lbl.setProperty("role", "metric")
                val_lbl.setAlignment(Qt.AlignCenter)

                sub_lbl = QLabel(title_text)
                sub_lbl.setProperty("role", "muted")
                sub_lbl.setAlignment(Qt.AlignCenter)

                card_v.addWidget(val_lbl)
                card_v.addWidget(sub_lbl)
                metrics_layout.addWidget(card, 1)
                self.metric_labels.append(val_lbl)

            self.metric_labels[-1].setStyleSheet("color: #F87171;")  # Red for errors
            layout.addLayout(metrics_layout)

            # --- Master Overall Progress Bar ---
            self.overall_bar = QProgressBar()
            self.overall_bar.setObjectName("overallBar")
            self.overall_bar.setRange(0, 100)
            self.overall_bar.setValue(0)
            self.overall_bar.setTextVisible(False)
            self.overall_bar.setFixedHeight(8)
            layout.addWidget(self.overall_bar)

            # --- Sub-bars (Analysis & Merge) ---
            stages = QHBoxLayout()
            stages.setSpacing(16)
            self.analysis_bar = QProgressBar()
            self.merge_bar = QProgressBar()
            self.analysis_count = QLabel("分析  0 / 0")
            self.merge_count = QLabel("合成  0 / 0")

            for count_lbl, bar in ((self.analysis_count, self.analysis_bar), (self.merge_count, self.merge_bar)):
                column = QVBoxLayout()
                column.setSpacing(4)
                count_lbl.setProperty("role", "muted")
                bar.setRange(0, 100)
                bar.setValue(0)
                bar.setTextVisible(False)
                bar.setFixedHeight(5)
                column.addWidget(count_lbl)
                column.addWidget(bar)
                stages.addLayout(column, 1)

            layout.addLayout(stages)

            # --- Live Task Status Card ---
            status_box = QFrame()
            status_box.setObjectName("innerCard")
            status_layout = QHBoxLayout(status_box)
            status_layout.setContentsMargins(12, 8, 12, 8)
            status_layout.setSpacing(8)

            self.file_label = ElidedLabel("等待开始")
            self.group_label = ElidedLabel("—")
            self.group_label.setMaximumWidth(120)
            self.group_label.setStyleSheet("color: #60A5FA; font-weight: 600;")

            status_layout.addWidget(self.file_label, 1)
            status_layout.addWidget(self.group_label)
            layout.addWidget(status_box)

            # Compatibility labels kept in DOM for headless tests
            self.count_label = QLabel(self)
            self.stats_label = QLabel(self)
            self.count_label.hide()
            self.stats_label.hide()

            # --- Diagnostic Details ---
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

        def _update_stepper(self, stage: str):
            """Highlight active step in the 5-stage pipeline."""
            step_idx = -1
            if stage == "scanning":
                step_idx = 0
            elif stage == "analysis":
                step_idx = 1
            elif stage == "alignment":
                step_idx = 2
            elif stage == "fusion":
                step_idx = 3
            elif stage in ("archive", "complete"):
                step_idx = 4

            for i, label in enumerate(self.step_labels):
                if step_idx < 0:
                    label.setStyleSheet(
                        "background: #141821; color: #64748B; border: 1px solid #232B3A; "
                        "border-radius: 6px; padding: 4px 6px; font-size: 11px; font-weight: 500;"
                    )
                elif i < step_idx or (stage == "complete" and i <= step_idx):
                    label.setStyleSheet(
                        "background: #142823; color: #34D399; border: 1px solid #059669; "
                        "border-radius: 6px; padding: 4px 6px; font-size: 11px; font-weight: 600;"
                    )
                elif i == step_idx:
                    label.setStyleSheet(
                        "background: #172A46; color: #38BDF8; border: 1px solid #2563EB; "
                        "border-radius: 6px; padding: 4px 6px; font-size: 11px; font-weight: 600;"
                    )
                else:
                    label.setStyleSheet(
                        "background: #141821; color: #475569; border: 1px solid #202633; "
                        "border-radius: 6px; padding: 4px 6px; font-size: 11px; font-weight: 500;"
                    )

        def _toggle_details(self, checked: bool):
            self.details_text.setVisible(checked)
            self.details_toggle.setText("诊断详情 · 收起" if checked else "诊断详情 · 展开")

        def set_message(self, message: str):
            self.message_label.setText(message)
            self.details_text.setPlainText(message)
            self.details_toggle.setEnabled(bool(message))
            if not message:
                self.details_toggle.setChecked(False)

        def reset(self):
            self.details_toggle.setChecked(False)
            self._update_stepper("scanning")
            self.update_event({"stage": "scanning"})

        @Slot(object)
        def update_event(self, event: PipelineEvent | Any) -> None:
            snapshot = ProgressSnapshot.from_event(event)
            self.snapshot = snapshot
            self.overall_bar.setValue(snapshot.overall)
            self.analysis_bar.setValue(snapshot.analysis)
            self.merge_bar.setValue(snapshot.merge)
            self.stage_label.setText(STAGE_LABELS.get(snapshot.stage, snapshot.stage))
            self._update_stepper(snapshot.stage)

            self.file_label.setText(snapshot.current_file or "—")
            self.group_label.setText(f"组 #{snapshot.current_group}" if snapshot.current_group else "—")
            self.analysis_count.setText(f"焦点分析  {snapshot.analysis_completed} / {snapshot.analysis_total}")
            self.merge_count.setText(f"图像合成  {snapshot.merge_completed} / {snapshot.merge_total}")
            self.count_label.setText(
                f"分析：{snapshot.analysis_completed} / {snapshot.analysis_total}    "
                f"合成：{snapshot.merge_completed} / {snapshot.merge_total}"
            )
            self.stats_label.setText(
                f"已发现：{snapshot.groups_found}    已完成：{snapshot.groups_finished}    错误：{snapshot.errors}"
            )
            for label, value in zip(
                self.metric_labels,
                (f"{snapshot.overall}%", snapshot.groups_found, snapshot.groups_finished, snapshot.errors),
            ):
                label.setText(str(value))
            self.set_message(snapshot.message)

        def complete(self, summary: Any) -> None:
            """Keep original overall completion semantics, including cancellation."""
            cancelled = bool(getattr(summary, "cancelled", False))
            errors = int(getattr(summary, "errors", 0) or 0)
            self.stage_label.setText("已取消" if cancelled else "处理结束 · 有错误" if errors else "处理完成")
            self._update_stepper("cancelled" if cancelled else "complete")
            if bool(getattr(summary, "finished", False)) and not cancelled:
                self.snapshot.overall = 100
                self.overall_bar.setValue(100)
                self.metric_labels[0].setText("100%")
            self.metric_labels[-1].setText(str(errors))
            diagnostics = list(getattr(summary, "diagnostics", ()) or ())
            if diagnostics:
                self.set_message("\n\n".join(str(item) for item in diagnostics))

except ImportError:  # pragma: no cover

    class ProgressPanel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError("PySide6 is required to create the desktop UI")


__all__ = ["ProgressPanel", "ProgressSnapshot"]
