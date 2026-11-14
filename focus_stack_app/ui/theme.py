"""Modern Daylight Studio & Darkroom palette tokens, QSS, and SVG icon helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPixmap
    from PySide6.QtWidgets import QApplication, QLabel, QSizePolicy, QStyleFactory
except ImportError:  # pragma: no cover - headless package imports remain safe
    Qt = None  # type: ignore[assignment]


ICONS_DIR = Path(__file__).parent / "icons"


@dataclass(frozen=True, slots=True)
class ThemeTokens:
    name: str
    bg_canvas: str
    bg_surface: str
    bg_subsurface: str
    bg_rail: str
    bg_mat: str
    bg_input: str
    bg_input_hover: str
    bg_input_disabled: str
    bg_button: str
    bg_button_hover: str
    bg_button_pressed: str
    bg_button_disabled: str
    border: str
    border_strong: str
    text_primary: str
    text_secondary: str
    text_muted: str
    text_disabled: str
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_soft: str
    accent_soft_border: str
    accent_text: str
    success: str
    success_soft: str
    success_border: str
    warning: str
    warning_soft: str
    warning_border: str
    danger: str
    danger_soft: str
    danger_border: str
    table_alt: str
    table_hover: str
    table_selected_bg: str
    table_selected_fg: str
    header_bg: str
    scrollbar_handle: str
    scrollbar_hover: str
    progress_track: str


LIGHT_THEME = ThemeTokens(
    name="light",
    bg_canvas="#DDE1E7",
    bg_surface="#E8EBEF",
    bg_subsurface="#DFE3E9",
    bg_rail="#D4D9E2",
    bg_mat="#CFD4DD",
    bg_input="#EFEFF2",
    bg_input_hover="#F4F5F7",
    bg_input_disabled="#D9DDE4",
    bg_button="#EBEDF1",
    bg_button_hover="#F4F5F8",
    bg_button_pressed="#D8DDE6",
    bg_button_disabled="#DDE1E7",
    border="#C6CCD8",
    border_strong="#B0B8C7",
    text_primary="#181D26",
    text_secondary="#363E4E",
    text_muted="#555F71",
    text_disabled="#8D96A6",
    accent="#2563EB",
    accent_hover="#1D4ED8",
    accent_pressed="#1E40AF",
    accent_soft="#D9E4F7",
    accent_soft_border="#A6C0EE",
    accent_text="#1E40AF",
    success="#047857",
    success_soft="#D4EBE2",
    success_border="#9CD0BE",
    warning="#B45309",
    warning_soft="#EFE1C6",
    warning_border="#D8BC88",
    danger="#B91C1C",
    danger_soft="#F1DADA",
    danger_border="#DFA6A6",
    table_alt="#E2E6EB",
    table_hover="#D9E0EB",
    table_selected_bg="#CFDEF5",
    table_selected_fg="#181D26",
    header_bg="#DFE3EA",
    scrollbar_handle="#AEB7C5",
    scrollbar_hover="#8E99AA",
    progress_track="#CDD3DE",
)

DARK_THEME = ThemeTokens(
    name="dark",
    bg_canvas="#101216",
    bg_surface="#171A21",
    bg_subsurface="#13161C",
    bg_rail="#14171D",
    bg_mat="#0C0E12",
    bg_input="#12151B",
    bg_input_hover="#1A1F28",
    bg_input_disabled="#121419",
    bg_button="#1D222C",
    bg_button_hover="#252C38",
    bg_button_pressed="#181C24",
    bg_button_disabled="#161920",
    border="#252B36",
    border_strong="#363F50",
    text_primary="#F1F5F9",
    text_secondary="#CBD5E1",
    text_muted="#8E9BAE",
    text_disabled="#525F73",
    accent="#3B82F6",
    accent_hover="#2563EB",
    accent_pressed="#1D4ED8",
    accent_soft="#17253D",
    accent_soft_border="#264373",
    accent_text="#93C5FD",
    success="#34D399",
    success_soft="#102922",
    success_border="#1B4D3E",
    warning="#FBBF24",
    warning_soft="#2A2011",
    warning_border="#594017",
    danger="#F87171",
    danger_soft="#2B1619",
    danger_border="#5C262C",
    table_alt="#191D25",
    table_hover="#1F2530",
    table_selected_bg="#1E304D",
    table_selected_fg="#FFFFFF",
    header_bg="#14181F",
    scrollbar_handle="#333C4D",
    scrollbar_hover="#4A5568",
    progress_track="#232934",
)


def get_theme_tokens(theme_name: str = "light") -> ThemeTokens:
    return DARK_THEME if str(theme_name).strip().lower() == "dark" else LIGHT_THEME


def themed_pixmap(name: str, color: str, size: int = 16) -> "QPixmap":
    path = ICONS_DIR / name
    if not path.is_file():
        return QPixmap()
    dpr = 2
    base = QIcon(str(path)).pixmap(size * dpr, size * dpr)
    if base.isNull():
        return QPixmap()
    tinted = QPixmap(base.size())
    tinted.fill(Qt.transparent)
    painter = QPainter(tinted)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.drawPixmap(0, 0, base)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(tinted.rect(), QColor(color))
    painter.end()
    tinted.setDevicePixelRatio(dpr)
    return tinted


def themed_icon(name: str, color: str, size: int = 16) -> "QIcon":
    pix = themed_pixmap(name, color, size=size)
    return QIcon(pix) if not pix.isNull() else QIcon()


def build_stylesheet(tokens: ThemeTokens, assets_dir: str | None = None) -> str:
    assets = assets_dir or ICONS_DIR.as_posix()
    t = tokens
    return f"""
QMainWindow, QWidget#workspace {{
    background: {t.bg_canvas};
    color: {t.text_primary};
}}

QWidget {{
    color: {t.text_primary};
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif;
    font-size: 13px;
}}

QDialog, QMessageBox {{
    background: {t.bg_surface};
    color: {t.text_primary};
}}

QMessageBox QLabel {{
    color: {t.text_primary};
    background: transparent;
}}

QWidget#studioHeader {{
    background: {t.bg_surface};
    border-bottom: 1px solid {t.border};
}}

QFrame#controlRail {{
    background: {t.bg_rail};
    border-right: 1px solid {t.border};
}}

QScrollArea#settingsScroll, QWidget#settingsContent, QScrollArea#inspectorScroll, QWidget#inspectorScrollContent {{
    background: transparent;
    border: none;
}}

QWidget#railFooter {{
    background: {t.bg_rail};
    border-top: 1px solid {t.border};
}}

QFrame#railDivider {{
    background: {t.border};
    max-height: 1px;
    min-height: 1px;
    border: none;
}}

QFrame#archiveDrawer {{
    border-left: 2px solid {t.border_strong};
    padding-left: 8px;
}}

QFrame#progressStrip, QFrame#resultsSurface {{
    background: {t.bg_surface};
    border: 1px solid {t.border};
    border-radius: 10px;
}}

QFrame#inspectorPane {{
    background: {t.bg_subsurface};
    border: 1px solid {t.border};
    border-radius: 8px;
}}

QFrame#emptyDropCard {{
    background: {t.bg_subsurface};
    border: 1.5px dashed {t.border_strong};
    border-radius: 12px;
}}

QFrame#emptyDropCard[dragActive="true"] {{
    background: {t.accent_soft};
    border: 1.5px dashed {t.accent};
}}

QFrame#previewStage {{
    background: {t.bg_mat};
    border: 1px solid {t.border};
    border-radius: 8px;
}}

QFrame#diagBanner {{
    background: {t.danger_soft};
    border: 1px solid {t.danger_border};
    border-radius: 6px;
}}

QFrame#warnBanner {{
    background: {t.warning_soft};
    border: 1px solid {t.warning_border};
    border-radius: 6px;
}}

QLabel {{
    background: transparent;
}}

QLabel[role="brand"] {{
    font-size: 15px;
    font-weight: 700;
    color: {t.text_primary};
}}

QLabel[role="section"] {{
    font-size: 12px;
    font-weight: 700;
    color: {t.text_secondary};
    letter-spacing: 0.3px;
}}

QLabel[role="muted"] {{
    color: {t.text_muted};
    font-size: 12px;
}}

QLabel[role="hint"] {{
    color: {t.text_muted};
    font-size: 11px;
}}

QLabel[role="mono"] {{
    font-family: "Consolas", "Cascadia Mono", "Segoe UI Mono", monospace;
    font-size: 12px;
    color: {t.text_secondary};
}}

QLabel[role="timer"] {{
    font-family: "Consolas", "Cascadia Mono", "Segoe UI Mono", monospace;
    font-size: 12px;
    font-weight: 600;
    color: {t.text_secondary};
    background: {t.bg_subsurface};
    border: 1px solid {t.border};
    border-radius: 6px;
    padding: 3px 10px;
}}

QLabel[role="badge"] {{
    color: {t.accent_text};
    background: {t.accent_soft};
    border: 1px solid {t.accent_soft_border};
    padding: 3px 10px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 12px;
}}

QLabel[role="countPill"] {{
    color: {t.text_secondary};
    background: {t.bg_rail};
    border: 1px solid {t.border};
    padding: 2px 8px;
    border-radius: 10px;
    font-family: "Consolas", "Cascadia Mono", monospace;
    font-size: 11px;
    font-weight: 600;
}}

QLineEdit, QSpinBox, QComboBox {{
    background: {t.bg_input};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    border-radius: 6px;
    min-height: 30px;
    padding: 0 9px;
    selection-background-color: {t.accent};
    selection-color: #FFFFFF;
}}

QLineEdit:hover, QSpinBox:hover, QComboBox:hover {{
    background: {t.bg_input_hover};
    border-color: {t.accent};
}}

QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
    background: {t.bg_input};
    border: 1.5px solid {t.accent};
}}

QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{
    color: {t.text_disabled};
    border-color: {t.border};
    background: {t.bg_input_disabled};
}}

QComboBox QAbstractItemView {{
    background: {t.bg_surface};
    border: 1px solid {t.border_strong};
    color: {t.text_primary};
    selection-background-color: {t.accent};
    selection-color: #FFFFFF;
    outline: none;
    padding: 3px;
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox::down-arrow {{
    image: url({assets}/chevron-down.svg);
    width: 11px;
    height: 11px;
}}

QSpinBox::up-button, QSpinBox::down-button {{
    subcontrol-origin: border;
    width: 20px;
    border: none;
    background: transparent;
}}

QSpinBox::up-button {{
    subcontrol-position: top right;
}}

QSpinBox::down-button {{
    subcontrol-position: bottom right;
}}

QSpinBox::up-arrow {{
    image: url({assets}/chevron-up.svg);
    width: 10px;
    height: 10px;
}}

QSpinBox::down-arrow {{
    image: url({assets}/chevron-down.svg);
    width: 10px;
    height: 10px;
}}

QPushButton {{
    background: {t.bg_button};
    color: {t.text_primary};
    border: 1px solid {t.border_strong};
    border-radius: 6px;
    min-height: 30px;
    padding: 0 11px;
    font-weight: 500;
}}

QPushButton:hover {{
    background: {t.bg_button_hover};
    border-color: {t.accent};
}}

QPushButton:pressed {{
    background: {t.bg_button_pressed};
}}

QPushButton:focus {{
    border-color: {t.accent};
}}

QPushButton:disabled {{
    color: {t.text_disabled};
    background: {t.bg_button_disabled};
    border-color: {t.border};
}}

QPushButton#startButton {{
    background: {t.accent};
    color: #FFFFFF;
    border: 1px solid {t.accent};
    font-weight: 600;
    min-height: 36px;
    border-radius: 7px;
    font-size: 13px;
}}

QPushButton#startButton:hover {{
    background: {t.accent_hover};
    border-color: {t.accent_hover};
}}

QPushButton#startButton:pressed {{
    background: {t.accent_pressed};
}}

QPushButton#startButton:disabled {{
    background: {t.bg_button_disabled};
    color: {t.text_disabled};
    border-color: {t.border};
}}

QPushButton#stopButton {{
    background: {t.bg_button};
    border: 1px solid {t.border_strong};
    color: {t.danger};
    font-weight: 600;
    min-height: 36px;
    border-radius: 7px;
}}

QPushButton#stopButton:hover {{
    background: {t.danger_soft};
    border-color: {t.danger};
}}

QPushButton#stopButton:disabled {{
    color: {t.text_disabled};
    background: {t.bg_button_disabled};
    border-color: {t.border};
}}

QPushButton#disclosureBtn {{
    background: transparent;
    border: 1px solid {t.border};
    border-radius: 6px;
    color: {t.text_secondary};
    font-weight: 600;
    text-align: left;
    padding: 0 10px;
    min-height: 32px;
}}

QPushButton#disclosureBtn:hover {{
    background: {t.bg_button_hover};
    border-color: {t.border_strong};
    color: {t.text_primary};
}}

QPushButton#ghostBtn, QPushButton#themeToggleBtn {{
    background: transparent;
    border: 1px solid {t.border};
    border-radius: 6px;
    color: {t.text_secondary};
    min-height: 28px;
    padding: 0 9px;
    font-size: 12px;
}}

QPushButton#ghostBtn:hover, QPushButton#themeToggleBtn:hover {{
    background: {t.bg_button_hover};
    border-color: {t.border_strong};
    color: {t.text_primary};
}}

QPushButton#ghostBtn:disabled {{
    color: {t.text_disabled};
    border-color: transparent;
    background: transparent;
}}

QPushButton.segBtn {{
    background: {t.bg_subsurface};
    border: 1px solid {t.border};
    color: {t.text_secondary};
    border-radius: 6px;
    min-height: 27px;
    padding: 0 10px;
    font-size: 12px;
    font-weight: 500;
}}

QPushButton.segBtn:hover {{
    background: {t.bg_button_hover};
    color: {t.text_primary};
}}

QPushButton.segBtn:checked {{
    background: {t.accent};
    color: #FFFFFF;
    border-color: {t.accent};
    font-weight: 600;
}}

QCheckBox {{
    spacing: 8px;
    min-height: 24px;
    color: {t.text_primary};
}}

QCheckBox:disabled {{
    color: {t.text_disabled};
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1.5px solid {t.border_strong};
    border-radius: 4px;
    background: {t.bg_input};
}}

QCheckBox::indicator:hover {{
    border-color: {t.accent};
}}

QCheckBox::indicator:checked {{
    background: {t.accent};
    border-color: {t.accent};
    image: url({assets}/check.svg);
}}

QCheckBox::indicator:disabled {{
    border-color: {t.border};
    background: {t.bg_input_disabled};
}}

QProgressBar {{
    background: {t.progress_track};
    border: none;
    border-radius: 3px;
}}

QProgressBar::chunk {{
    background: {t.accent};
    border-radius: 3px;
}}

QProgressBar#successBar::chunk {{
    background: {t.success};
    border-radius: 3px;
}}

QTableWidget {{
    background: {t.bg_surface};
    alternate-background-color: {t.table_alt};
    border: 1px solid {t.border};
    border-radius: 8px;
    selection-background-color: {t.table_selected_bg};
    selection-color: {t.table_selected_fg};
    gridline-color: transparent;
    outline: none;
}}

QTableWidget::item {{
    padding: 0 6px;
    border: none;
    border-bottom: 1px solid {t.border};
}}

QTableWidget::item:hover {{
    background: {t.table_hover};
}}

QTableWidget::item:selected {{
    background: {t.table_selected_bg};
    color: {t.table_selected_fg};
}}

QHeaderView::section {{
    background: {t.header_bg};
    color: {t.text_muted};
    padding: 7px 8px;
    border: none;
    border-bottom: 1px solid {t.border};
    font-weight: 600;
    font-size: 12px;
}}

QTextEdit, QPlainTextEdit {{
    background: {t.bg_subsurface};
    border: 1px solid {t.border};
    border-radius: 6px;
    padding: 8px;
    font-family: "Consolas", "Cascadia Mono", monospace;
    font-size: 12px;
    color: {t.text_secondary};
}}

QScrollBar:vertical {{
    background: transparent;
    width: 6px;
    margin: 2px 0;
}}

QScrollBar::handle:vertical {{
    background: {t.scrollbar_handle};
    border-radius: 3px;
    min-height: 28px;
}}

QScrollBar::handle:vertical:hover {{
    background: {t.scrollbar_hover};
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 6px;
    margin: 0 2px;
}}

QScrollBar::handle:horizontal {{
    background: {t.scrollbar_handle};
    border-radius: 3px;
    min-width: 28px;
}}

QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {{
    height: 0;
    width: 0;
    background: transparent;
}}

QSplitter::handle {{
    background: transparent;
    width: 8px;
}}

QSplitter::handle:hover {{
    background: {t.border};
    border-radius: 2px;
}}

QStatusBar {{
    color: {t.text_muted};
    background: {t.bg_surface};
    font-size: 12px;
    border-top: 1px solid {t.border};
    padding: 0 12px;
}}

QToolTip {{
    color: {t.text_primary};
    background: {t.bg_surface};
    border: 1px solid {t.border_strong};
    padding: 6px 8px;
    border-radius: 6px;
}}
"""


def apply_theme(window: Any, theme_name: str = "light") -> ThemeTokens:
    tokens = get_theme_tokens(theme_name)
    app = QApplication.instance()
    if app is not None:
        current_style = app.style()
        if current_style is None or current_style.objectName().casefold() != "fusion":
            app.setStyle("Fusion")
    font = QFont("Microsoft YaHei UI")
    font.setFamilies([
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "Segoe UI",
        "PingFang SC",
        "Noto Sans CJK SC",
        "sans-serif",
    ])
    font.setPixelSize(13)
    window.setFont(font)

    palette = QPalette()
    for role, color in {
        QPalette.Window: tokens.bg_canvas,
        QPalette.WindowText: tokens.text_primary,
        QPalette.Base: tokens.bg_surface,
        QPalette.AlternateBase: tokens.table_alt,
        QPalette.Text: tokens.text_primary,
        QPalette.Button: tokens.bg_button,
        QPalette.ButtonText: tokens.text_primary,
        QPalette.Highlight: tokens.accent,
        QPalette.HighlightedText: "#FFFFFF",
        QPalette.ToolTipBase: tokens.bg_surface,
        QPalette.ToolTipText: tokens.text_primary,
        QPalette.PlaceholderText: tokens.text_muted,
    }.items():
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(tokens.text_disabled))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(tokens.text_disabled))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(tokens.text_disabled))
    window.setPalette(palette)
    app = QApplication.instance()
    if app is not None:
        app.setPalette(palette)
    window.setStyleSheet(build_stylesheet(tokens))
    return tokens


if Qt is not None:
    class ElidedLabel(QLabel):
        """Keep full label text accessible via tooltip while painting a single shortened line."""

        def __init__(self, text: str = "", parent: Any = None):
            super().__init__(text, parent)
            self.setMinimumWidth(0)
            self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            self.setToolTip(text)
            self._explicit_color: str | None = None

        def set_text_color(self, color: str | None) -> None:
            self._explicit_color = color
            self.update()

        def setText(self, text: str) -> None:
            super().setText(text)
            self.setToolTip(text)

        def paintEvent(self, event: Any) -> None:
            painter = QPainter(self)
            if self._explicit_color:
                painter.setPen(QColor(self._explicit_color))
            else:
                painter.setPen(self.palette().color(QPalette.WindowText))
            text = self.fontMetrics().elidedText(self.text(), Qt.ElideMiddle, self.contentsRect().width())
            painter.drawText(self.contentsRect(), Qt.AlignVCenter | Qt.AlignLeft, text)
else:  # pragma: no cover
    class ElidedLabel:  # type: ignore[no-redef]
        pass


__all__ = [
    "DARK_THEME",
    "ICONS_DIR",
    "LIGHT_THEME",
    "ElidedLabel",
    "ThemeTokens",
    "apply_theme",
    "build_stylesheet",
    "get_theme_tokens",
    "themed_icon",
    "themed_pixmap",
]
