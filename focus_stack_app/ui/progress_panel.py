"""Clean, compact studio progress cockpit with live ETA and expandable diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
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


def _format_duration(seconds: float) -> str:
    secs = max(0, int(round(seconds)))
    if secs < 60:
        return f"{secs} 秒"
    mins, rem = divmod(secs, 60)
    if mins < 60:
        return f"{mins} 分 {rem:02d} 秒"
    hours, mins = divmod(mins, 60)
    return f"{hours} 时 {mins:02d} 分"


try:  # PySide6 is optional for headless tests and CLI use.
    from PySide6.QtCore import Qt, QUrl, Slot
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QHBoxLayout,
        QLabel,
        QProgressBar,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    from .theme import ElidedLabel, ThemeTokens, get_theme_tokens, themed_icon

    STAGE_LABELS = {
        "idle": "等待导入照片文件夹",
        "scanning": "正在扫描照片与自动分组…",
        "analysis": "正在分析焦点清晰度与选片…",
        "archive": "正在执行原图安全归档…",
        "alignment": "正在进行亚像素图像对齐…",
        "fusion": "正在进行全分辨率景深融合…",
        "complete": "全部处理完成",
        "cancelled": "处理已安全终止",
        "error": "处理遇到错误",
    }

    PIPELINE_STEPS = ["扫描分组", "焦点分析", "图像对齐", "景深融合", "完成归档"]

    class ProgressPanel(QFrame):
        """Compact, daylight-clean progress ribbon with inline stepper and diagnostics drawer."""

        def __init__(self, parent: QWidget | None = None, *, tokens: ThemeTokens | None = None):
            super().__init__(parent)
            self.setObjectName("progressStrip")
            self._tokens = tokens or get_theme_tokens("light")
            self.snapshot = ProgressSnapshot()
            self._run_started_mono: float | None = None
            self._log_path: Path | None = None
            self._diagnostic_history: list[str] = []
            self._cached_err_key: tuple[str, bool] | None = None
            self._cached_dot_key: tuple[str, str, int] | None = None
            self._cached_stepper_key: tuple[str, str] | None = None
            self._cached_message: str | None = None

            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 12, 16, 12)
            layout.setSpacing(10)

            # --- Row 1: Stage Title + Compact KPI Badges + Details Toggle ---
            top_row = QHBoxLayout()
            top_row.setSpacing(10)

            self.stage_dot = QLabel()
            self.stage_dot.setFixedSize(8, 8)
            top_row.addWidget(self.stage_dot)

            self.stage_label = QLabel(STAGE_LABELS["idle"])
            self.stage_label.setStyleSheet("font-weight: 600; font-size: 13px;")
            top_row.addWidget(self.stage_label)

            self.eta_label = QLabel("")
            self.eta_label.setProperty("role", "hint")
            self.eta_label.hide()
            top_row.addWidget(self.eta_label)

            top_row.addStretch()

            # Compact inline metrics (kept as self.metric_labels for test/UI compatibility)
            self.metric_labels: list[QLabel] = []
            self._metric_containers: list[tuple[QLabel, QLabel]] = []
            for caption, default_val in (
                ("总体", "0%"),
                ("已发现", "0"),
                ("已完成", "0"),
                ("错误", "0"),
            ):
                cap_lbl = QLabel(caption)
                cap_lbl.setProperty("role", "hint")
                val_lbl = QLabel(default_val)
                val_lbl.setProperty("role", "mono")
                top_row.addWidget(cap_lbl)
                top_row.addWidget(val_lbl)
                self.metric_labels.append(val_lbl)
                self._metric_containers.append((cap_lbl, val_lbl))

            self.details_toggle = QPushButton("诊断日志")
            self.details_toggle.setObjectName("ghostBtn")
            self.details_toggle.setCheckable(True)
            self.details_toggle.setEnabled(False)
            self.details_toggle.toggled.connect(self._toggle_details)
            top_row.addWidget(self.details_toggle)

            layout.addLayout(top_row)

            # --- Row 2: Slim 5-Stage Inline Stepper ---
            self.stepper_widget = QWidget()
            stepper_layout = QHBoxLayout(self.stepper_widget)
            stepper_layout.setContentsMargins(0, 0, 0, 0)
            stepper_layout.setSpacing(6)
            self.step_labels: list[QLabel] = []
            for idx, name in enumerate(PIPELINE_STEPS):
                step_box = QLabel(f"{idx + 1}. {name}")
                step_box.setAlignment(Qt.AlignCenter)
                stepper_layout.addWidget(step_box, 1)
                self.step_labels.append(step_box)
            layout.addWidget(self.stepper_widget)

            # --- Row 3: Master Bar + Dual Sub-Bars ---
            self.overall_bar = QProgressBar()
            self.overall_bar.setObjectName("overallBar")
            self.overall_bar.setRange(0, 100)
            self.overall_bar.setValue(0)
            self.overall_bar.setTextVisible(False)
            self.overall_bar.setFixedHeight(6)
            layout.addWidget(self.overall_bar)

            sub_row = QHBoxLayout()
            sub_row.setSpacing(16)

            self.analysis_bar = QProgressBar()
            self.merge_bar = QProgressBar()
            self.analysis_count = QLabel("焦点分析  0 / 0")
            self.merge_count = QLabel("图像合成  0 / 0")

            for count_lbl, bar in ((self.analysis_count, self.analysis_bar), (self.merge_count, self.merge_bar)):
                col = QHBoxLayout()
                col.setSpacing(8)
                count_lbl.setProperty("role", "hint")
                count_lbl.setMinimumWidth(116)
                bar.setRange(0, 100)
                bar.setValue(0)
                bar.setTextVisible(False)
                bar.setFixedHeight(4)
                col.addWidget(count_lbl)
                col.addWidget(bar, 1)
                sub_row.addLayout(col, 1)

            layout.addLayout(sub_row)

            # --- Row 4: Live File & Current Group ---
            live_row = QHBoxLayout()
            live_row.setSpacing(8)

            self.file_label = ElidedLabel("就绪：等待开始任务")
            self.group_label = ElidedLabel("—")
            self.group_label.setMaximumWidth(120)
            self.group_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            live_row.addWidget(self.file_label, 1)
            live_row.addWidget(self.group_label)
            layout.addLayout(live_row)

            # Hidden compatibility labels kept for headless tests
            self.count_label = QLabel(self)
            self.stats_label = QLabel(self)
            self.message_label = QLabel(self)
            self.count_label.hide()
            self.stats_label.hide()
            self.message_label.hide()

            # --- Row 5: Expandable Diagnostics & Log Drawer ---
            self._diag_actions = QWidget()
            diag_bar = QHBoxLayout(self._diag_actions)
            diag_bar.setContentsMargins(0, 2, 0, 0)
            diag_bar.setSpacing(8)
            diag_hint = QLabel("运行诊断与日志输出")
            diag_hint.setProperty("role", "hint")
            diag_bar.addWidget(diag_hint)
            diag_bar.addStretch()

            self.copy_diag_btn = QPushButton("复制诊断")
            self.copy_diag_btn.setObjectName("ghostBtn")
            self.copy_diag_btn.clicked.connect(self._copy_diagnostics)
            diag_bar.addWidget(self.copy_diag_btn)

            self.open_log_btn = QPushButton("打开日志文件")
            self.open_log_btn.setObjectName("ghostBtn")
            self.open_log_btn.setEnabled(False)
            self.open_log_btn.clicked.connect(self._open_log_file)
            diag_bar.addWidget(self.open_log_btn)

            self._diag_actions.hide()
            layout.addWidget(self._diag_actions)

            self.details_text = QTextEdit()
            self.details_text.setReadOnly(True)
            self.details_text.setFixedHeight(88)
            self.details_text.hide()
            layout.addWidget(self.details_text)

            self.apply_theme_tokens(self._tokens)

        def apply_theme_tokens(self, tokens: ThemeTokens) -> None:
            self._tokens = tokens
            self.file_label.set_text_color(tokens.text_secondary)
            self.group_label.set_text_color(tokens.accent_text)
            self.details_toggle.setIcon(themed_icon("file-text.svg", tokens.text_secondary, 13))
            self.copy_diag_btn.setIcon(themed_icon("copy.svg", tokens.text_secondary, 13))
            self.open_log_btn.setIcon(themed_icon("external-link.svg", tokens.text_secondary, 13))
            self.metric_labels[0].setStyleSheet(
                f"font-family: 'Consolas', monospace; font-size: 12px; font-weight: 700; color: {tokens.accent_text};"
            )
            self.metric_labels[1].setStyleSheet(
                f"font-family: 'Consolas', monospace; font-size: 12px; font-weight: 600; color: {tokens.text_primary};"
            )
            self.metric_labels[2].setStyleSheet(
                f"font-family: 'Consolas', monospace; font-size: 12px; font-weight: 600; color: {tokens.success};"
            )
            self._cached_err_key = None
            self._cached_dot_key = None
            self._cached_stepper_key = None
            self._refresh_error_color()
            self._update_stage_dot(self.snapshot.stage, self.snapshot.errors)
            self._update_stepper(self.snapshot.stage)

        def _refresh_error_color(self) -> None:
            t = self._tokens
            err_val = int(self.snapshot.errors or 0)
            key = (t.name, err_val > 0)
            if key == self._cached_err_key:
                return
            self._cached_err_key = key
            color = t.danger if err_val > 0 else t.text_muted
            self.metric_labels[-1].setStyleSheet(
                f"font-family: 'Consolas', monospace; font-size: 12px; font-weight: 700; color: {color};"
            )

        def _update_stage_dot(self, stage: str, errors: int = 0) -> None:
            t = self._tokens
            err_bucket = 1 if errors > 0 else 0
            key = (t.name, stage, err_bucket)
            if key == self._cached_dot_key:
                return
            self._cached_dot_key = key
            if stage in ("error",) or (stage == "complete" and errors > 0):
                dot_color = t.danger
            elif stage == "cancelled":
                dot_color = t.warning
            elif stage == "complete":
                dot_color = t.success
            elif stage in ("scanning", "analysis", "alignment", "fusion", "archive"):
                dot_color = t.accent
            else:
                dot_color = t.border_strong
            self.stage_dot.setStyleSheet(f"background: {dot_color}; border-radius: 4px;")

        def _update_stepper(self, stage: str) -> None:
            t = self._tokens
            key = (t.name, stage)
            if key == self._cached_stepper_key:
                return
            self._cached_stepper_key = key
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

            for idx, label in enumerate(self.step_labels):
                if step_idx < 0:
                    label.setStyleSheet(
                        f"background: {t.bg_subsurface}; color: {t.text_muted}; border: 1px solid {t.border}; "
                        "border-radius: 5px; padding: 3px 6px; font-size: 11px; font-weight: 500;"
                    )
                elif idx < step_idx or (stage == "complete" and idx <= step_idx):
                    label.setStyleSheet(
                        f"background: {t.success_soft}; color: {t.success}; border: 1px solid {t.success_border}; "
                        "border-radius: 5px; padding: 3px 6px; font-size: 11px; font-weight: 600;"
                    )
                elif idx == step_idx:
                    label.setStyleSheet(
                        f"background: {t.accent_soft}; color: {t.accent_text}; border: 1px solid {t.accent_soft_border}; "
                        "border-radius: 5px; padding: 3px 6px; font-size: 11px; font-weight: 700;"
                    )
                else:
                    label.setStyleSheet(
                        f"background: {t.bg_subsurface}; color: {t.text_disabled}; border: 1px solid {t.border}; "
                        "border-radius: 5px; padding: 3px 6px; font-size: 11px; font-weight: 500;"
                    )

        def _toggle_details(self, checked: bool) -> None:
            self._diag_actions.setVisible(checked)
            self.details_text.setVisible(checked)
            self.details_toggle.setText("诊断日志 · 收起" if checked else "诊断日志")

        def set_log_path(self, path: Path | str | None) -> None:
            self._log_path = Path(path) if path else None
            exists = bool(self._log_path and self._log_path.is_file())
            self.open_log_btn.setEnabled(exists)
            if exists:
                self.details_toggle.setEnabled(True)

        def _copy_diagnostics(self) -> None:
            text = self.details_text.toPlainText().strip()
            if text:
                clipboard = QApplication.clipboard()
                if clipboard is not None:
                    clipboard.setText(text)
                    self.copy_diag_btn.setText("已复制")

        def _open_log_file(self) -> None:
            if self._log_path and self._log_path.is_file():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._log_path)))

        def set_message(self, message: str) -> None:
            if message == self._cached_message:
                return
            self._cached_message = message
            self.message_label.setText(message)
            self.details_text.setPlainText(message)
            self.copy_diag_btn.setText("复制诊断")
            has_log = bool(self._log_path and self._log_path.is_file())
            self.details_toggle.setEnabled(bool(message) or has_log)
            if not message and not has_log:
                self.details_toggle.setChecked(False)

        def reset(self) -> None:
            self._run_started_mono = time.monotonic()
            self._diagnostic_history.clear()
            self._cached_message = None
            self.eta_label.hide()
            self.eta_label.setText("")
            self.details_toggle.setChecked(False)
            self.overall_bar.setObjectName("overallBar")
            self.overall_bar.style().polish(self.overall_bar)
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
            self._update_stage_dot(snapshot.stage, snapshot.errors)
            self._update_stepper(snapshot.stage)

            # Live ETA estimate once active progress moves past 5%
            if self._run_started_mono is not None and 5 <= snapshot.overall < 100 and snapshot.stage not in ("complete", "cancelled", "error"):
                elapsed = time.monotonic() - self._run_started_mono
                if elapsed >= 2.0:
                    remaining = elapsed * (100 - snapshot.overall) / snapshot.overall
                    self.eta_label.setText(f"预计剩余约 {_format_duration(remaining)}")
                    self.eta_label.show()
            elif snapshot.stage in ("complete", "cancelled", "error", "idle"):
                self.eta_label.hide()

            self.file_label.setText(snapshot.current_file or ("等待开始任务" if snapshot.stage == "idle" else "—"))
            self.group_label.setText(f"组 #{snapshot.current_group}" if snapshot.current_group else "—")

            self.analysis_count.setText(f"焦点分析  {snapshot.analysis_completed} / {snapshot.analysis_total}")
            if snapshot.merge_finished > snapshot.merge_completed:
                self.merge_count.setText(
                    f"图像合成  {snapshot.merge_finished} / {snapshot.merge_total}（成功 {snapshot.merge_completed}）"
                )
            else:
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
            self._refresh_error_color()
            self.set_message(snapshot.message)

        def complete(self, summary: Any) -> None:
            """Finalize progress readout while keeping cancellation progress intact."""
            cancelled = bool(getattr(summary, "cancelled", False))
            errors = int(getattr(summary, "errors", 0) or 0)
            self.eta_label.hide()
            self.stage_label.setText("已取消" if cancelled else "处理结束 · 有错误" if errors else "全部处理完成")
            self._update_stage_dot("cancelled" if cancelled else "complete", errors)
            self._update_stepper("cancelled" if cancelled else "complete")
            if bool(getattr(summary, "finished", False)) and not cancelled:
                self.snapshot.overall = 100
                self.overall_bar.setValue(100)
                self.metric_labels[0].setText("100%")
                if not errors:
                    self.overall_bar.setObjectName("successBar")
                    self.overall_bar.style().polish(self.overall_bar)
            self.snapshot.errors = errors
            self.metric_labels[-1].setText(str(errors))
            self._refresh_error_color()

            output_dir = getattr(summary, "output_dir", None)
            if output_dir:
                log_candidate = Path(output_dir) / ".stack_cache" / "logs" / "application.log"
                if log_candidate.is_file():
                    self.set_log_path(log_candidate)

            diagnostics = list(getattr(summary, "diagnostics", ()) or ())
            if diagnostics:
                self.set_message("\n\n".join(str(item) for item in diagnostics))

except ImportError:  # pragma: no cover

    class ProgressPanel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError("PySide6 is required to create the desktop UI")


__all__ = ["ProgressPanel", "ProgressSnapshot"]
