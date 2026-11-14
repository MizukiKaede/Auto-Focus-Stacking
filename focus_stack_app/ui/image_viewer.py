"""Built-in high-resolution lightbox viewer with 1:1 pixel inspection and pan/zoom."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any

try:
    from PySide6.QtCore import Qt, QUrl
    from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPixmap
    from PySide6.QtWidgets import (
        QDialog,
        QGraphicsPixmapItem,
        QGraphicsScene,
        QGraphicsView,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
    from .theme import ThemeTokens, get_theme_tokens, themed_icon

    class _ZoomGraphicsView(QGraphicsView):
        """Interactive graphics view supporting wheel zoom and double-click 1:1 toggle."""

        def __init__(self, scene: QGraphicsScene, on_zoom_changed: Any, on_double_click: Any, parent: QWidget | None = None):
            super().__init__(scene, parent)
            self._on_zoom_changed = on_zoom_changed
            self._on_double_click = on_double_click
            self.setRenderHint(QPainter.Antialiasing, True)
            self.setRenderHint(QPainter.SmoothPixmapTransform, True)
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
            self.setFrameShape(QGraphicsView.NoFrame)

        def wheelEvent(self, event: Any) -> None:
            delta = event.angleDelta().y()
            if delta == 0:
                return
            factor = 1.18 if delta > 0 else (1.0 / 1.18)
            current = self.transform().m11()
            new_scale = current * factor
            if 0.05 <= new_scale <= 16.0:
                self.scale(factor, factor)
                if callable(self._on_zoom_changed):
                    self._on_zoom_changed(self.transform().m11())

        def mouseDoubleClickEvent(self, event: Any) -> None:
            if callable(self._on_double_click):
                self._on_double_click()
            super().mouseDoubleClickEvent(event)

    class ImageLightboxDialog(QDialog):
        """Modal studio lightbox for inspecting full-resolution composite results."""

        def __init__(
            self,
            image_path: str | Path,
            *,
            title: str | None = None,
            tokens: ThemeTokens | None = None,
            parent: QWidget | None = None,
        ):
            super().__init__(parent)
            self._path = Path(image_path)
            self._tokens = tokens or get_theme_tokens("light")
            self._pixmap = QPixmap(str(self._path))
            self._fit_mode = True

            self.setWindowTitle(title or f"大图校对 — {self._path.name}")
            self.resize(1040, 720)
            self.setMinimumSize(680, 480)
            self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)

            # Top toolbar
            toolbar_host = QWidget()
            toolbar_host.setStyleSheet(
                f"background: {self._tokens.bg_surface}; border-bottom: 1px solid {self._tokens.border};"
            )
            bar = QHBoxLayout(toolbar_host)
            bar.setContentsMargins(16, 10, 16, 10)
            bar.setSpacing(8)

            name_lbl = QLabel(self._path.name)
            name_lbl.setStyleSheet(f"font-weight: 700; font-size: 13px; color: {self._tokens.text_primary}; border: none;")
            bar.addWidget(name_lbl)

            meta_parts: list[str] = []
            if not self._pixmap.isNull():
                meta_parts.append(f"{self._pixmap.width()} × {self._pixmap.height()} px")
            try:
                size_mb = self._path.stat().st_size / (1024 * 1024)
                meta_parts.append(f"{size_mb:.2f} MB")
            except OSError:
                pass
            meta_lbl = QLabel(" · ".join(meta_parts) if meta_parts else "")
            meta_lbl.setStyleSheet(
                f"font-family: 'Consolas', monospace; font-size: 12px; color: {self._tokens.text_muted}; border: none;"
            )
            bar.addWidget(meta_lbl)
            bar.addStretch()

            self.zoom_badge = QLabel("适应")
            self.zoom_badge.setStyleSheet(
                f"font-family: 'Consolas', monospace; font-size: 12px; font-weight: 600; "
                f"color: {self._tokens.accent_text}; background: {self._tokens.accent_soft}; "
                f"border: 1px solid {self._tokens.accent_soft_border}; border-radius: 5px; padding: 2px 8px;"
            )
            bar.addWidget(self.zoom_badge)

            fit_btn = QPushButton("适应窗口")
            fit_btn.clicked.connect(self.fit_to_window)
            one_to_one_btn = QPushButton("1:1 像素")
            one_to_one_btn.clicked.connect(self.zoom_actual_size)
            zoom_out_btn = QPushButton("−")
            zoom_out_btn.setFixedWidth(32)
            zoom_out_btn.clicked.connect(lambda: self._step_zoom(1.0 / 1.25))
            zoom_in_btn = QPushButton("+")
            zoom_in_btn.setFixedWidth(32)
            zoom_in_btn.clicked.connect(lambda: self._step_zoom(1.25))

            open_ext_btn = QPushButton("系统打开")
            open_ext_btn.setIcon(themed_icon("external-link.svg", self._tokens.text_secondary, 14))
            open_ext_btn.clicked.connect(self._open_external)

            reveal_btn = QPushButton("定位文件")
            reveal_btn.setIcon(themed_icon("folder.svg", self._tokens.text_secondary, 14))
            reveal_btn.clicked.connect(self._reveal_file)

            for btn in (fit_btn, one_to_one_btn, zoom_out_btn, zoom_in_btn, open_ext_btn, reveal_btn):
                bar.addWidget(btn)

            root.addWidget(toolbar_host)

            # Canvas
            self._scene = QGraphicsScene(self)
            self._scene.setBackgroundBrush(QColor(self._tokens.bg_mat))
            self._view = _ZoomGraphicsView(self._scene, self._update_zoom_label, self._toggle_fit_or_actual, self)
            if not self._pixmap.isNull():
                self._item = QGraphicsPixmapItem(self._pixmap)
                self._item.setTransformationMode(Qt.SmoothTransformation)
                self._scene.addItem(self._item)
                self._scene.setSceneRect(self._pixmap.rect())
            else:
                self._item = None
                err = QLabel(f"无法读取图像文件：\n{self._path}")
                err.setAlignment(Qt.AlignCenter)
                err.setStyleSheet(f"color: {self._tokens.text_muted}; padding: 40px;")
                root.addWidget(err, 1)
                return

            root.addWidget(self._view, 1)

            hint_bar = QLabel("提示：滚轮缩放 · 左键拖拽平移 · 双击切换适应窗口 / 1:1 原像素 · Esc 关闭")
            hint_bar.setAlignment(Qt.AlignCenter)
            hint_bar.setStyleSheet(
                f"background: {self._tokens.bg_surface}; color: {self._tokens.text_muted}; "
                f"border-top: 1px solid {self._tokens.border}; padding: 6px 12px; font-size: 11px;"
            )
            root.addWidget(hint_bar)

        def showEvent(self, event: Any) -> None:
            super().showEvent(event)
            if self._fit_mode:
                self.fit_to_window()

        def resizeEvent(self, event: Any) -> None:
            super().resizeEvent(event)
            if self._fit_mode and self._item is not None:
                self.fit_to_window()

        def fit_to_window(self) -> None:
            if self._item is None:
                return
            self._fit_mode = True
            self._view.resetTransform()
            self._view.fitInView(self._item, Qt.KeepAspectRatio)
            self._item.setTransformationMode(Qt.SmoothTransformation)
            self._update_zoom_label(self._view.transform().m11())

        def zoom_actual_size(self) -> None:
            if self._item is None:
                return
            self._fit_mode = False
            self._view.resetTransform()
            self._item.setTransformationMode(Qt.FastTransformation)
            self._update_zoom_label(1.0)

        def _toggle_fit_or_actual(self) -> None:
            if self._fit_mode:
                self.zoom_actual_size()
            else:
                self.fit_to_window()

        def _step_zoom(self, factor: float) -> None:
            if self._item is None:
                return
            self._fit_mode = False
            current = self._view.transform().m11()
            if 0.05 <= current * factor <= 16.0:
                self._view.scale(factor, factor)
                scale = self._view.transform().m11()
                self._item.setTransformationMode(Qt.FastTransformation if scale >= 1.5 else Qt.SmoothTransformation)
                self._update_zoom_label(scale)

        def _update_zoom_label(self, scale: float) -> None:
            self._fit_mode = False if abs(scale - 1.0) < 0.01 else self._fit_mode
            pct = int(round(scale * 100))
            self.zoom_badge.setText(f"{pct}%")

        def _open_external(self) -> None:
            if self._path.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._path)))

        def _reveal_file(self) -> None:
            resolved = self._path.resolve()
            if sys.platform == "win32":
                if resolved.exists():
                    subprocess.run(["explorer", f"/select,{resolved}"], check=False)
                elif resolved.parent.exists():
                    subprocess.run(["explorer", str(resolved.parent)], check=False)
            else:
                target = str(resolved if resolved.exists() else resolved.parent)
                QDesktopServices.openUrl(QUrl.fromLocalFile(target))

except ImportError:  # pragma: no cover
    class ImageLightboxDialog:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError("PySide6 is required to create the desktop UI")


__all__ = ["ImageLightboxDialog"]
