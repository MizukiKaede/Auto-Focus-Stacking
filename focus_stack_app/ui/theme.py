"""Shared dark palette and compact desktop controls."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QLabel, QSizePolicy, QStyleFactory


def apply_theme(window):
    window.setStyle(QStyleFactory.create("Fusion"))
    font = QFont("Microsoft YaHei UI")
    font.setFamilies(["Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI"])
    font.setPixelSize(14)
    window.setFont(font)
    palette = QPalette()
    for role, color in {
        QPalette.Window: "#191B1F", QPalette.WindowText: "#E8EAED",
        QPalette.Base: "#1C1F24", QPalette.AlternateBase: "#262A30",
        QPalette.Text: "#E8EAED", QPalette.Button: "#23262B",
        QPalette.ButtonText: "#E8EAED", QPalette.Highlight: "#365780",
        QPalette.HighlightedText: "#FFFFFF", QPalette.ToolTipBase: "#23262B",
        QPalette.ToolTipText: "#E8EAED", QPalette.PlaceholderText: "#A8AFBA",
    }.items():
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#777F8B"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#777F8B"))
    window.setPalette(palette)
    window.setStyleSheet(STYLE.replace("@ASSETS@", (Path(__file__).parent / "icons").as_posix()))


class ElidedLabel(QLabel):
    """Keep full label text accessible while painting a single shortened line."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setToolTip(text)

    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter
        painter = QPainter(self)
        painter.setPen(self.palette().color(QPalette.WindowText))
        text = self.fontMetrics().elidedText(self.text(), Qt.ElideMiddle, self.contentsRect().width())
        painter.drawText(self.contentsRect(), Qt.AlignVCenter | Qt.AlignLeft, text)


STYLE = """
QMainWindow, QWidget#workspace { background: #191B1F; }
QWidget { color: #E8EAED; font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI"; font-size: 14px; }
QFrame#panel, QWidget#settingsContent { background: #23262B; border-radius: 8px; }
QLabel { background: transparent; }
QLabel[role="title"] { font-size: 20px; font-weight: 600; }
QLabel[role="section"] { font-size: 15px; font-weight: 600; }
QLabel[role="muted"] { color: #A8AFBA; }
QLabel[role="badge"] { color: #669CFF; background: #28364A; padding: 6px 14px; border-radius: 6px; }
QLabel[role="metric"] { font-size: 24px; font-weight: 600; }
QLineEdit, QSpinBox, QComboBox { background: #1C1F24; border: 1px solid #383D45; border-radius: 6px; min-height: 34px; padding: 0 8px; selection-background-color: #365780; }
QLineEdit:hover, QSpinBox:hover, QComboBox:hover { border-color: #596373; }
QLineEdit:focus, QSpinBox:focus, QComboBox:focus { border-color: #669CFF; }
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled { color: #777F8B; border-color: #30343A; }
QComboBox QAbstractItemView { background: #23262B; color: #E8EAED; selection-background-color: #365780; }
QComboBox::drop-down { border: none; width: 28px; }
QComboBox::down-arrow { image: url(@ASSETS@/chevron-down.svg); width: 12px; height: 12px; }
QSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 24px; border: none; }
QSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 24px; border: none; }
QSpinBox::up-arrow { image: url(@ASSETS@/chevron-up.svg); width: 10px; height: 10px; }
QSpinBox::down-arrow { image: url(@ASSETS@/chevron-down.svg); width: 10px; height: 10px; }
QPushButton { background: #2C3139; border: 1px solid #383D45; border-radius: 6px; min-height: 34px; padding: 0 12px; }
QPushButton:hover { background: #363E49; border-color: #596373; }
QPushButton:pressed { background: #242B34; }
QPushButton:focus { border-color: #669CFF; }
QPushButton:disabled { color: #777F8B; background: #25292F; border-color: #30343A; }
QPushButton#startButton { background: #669CFF; color: #101B2E; border-color: #669CFF; font-weight: 600; }
QPushButton#startButton:hover { background: #83AEFF; }
QPushButton#startButton:pressed { background: #5185DF; }
QPushButton#startButton:disabled { background: #344661; color: #8995A8; border-color: #344661; }
QCheckBox { spacing: 8px; min-height: 28px; }
QCheckBox:disabled { color: #777F8B; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #596373; border-radius: 3px; background: #1C1F24; }
QCheckBox::indicator:checked { background: #669CFF; border-color: #669CFF; image: url(@ASSETS@/check.svg); }
QCheckBox::indicator:hover { border-color: #669CFF; }
QCheckBox::indicator:disabled { border-color: #383D45; background: #303844; }
QScrollArea { background: #23262B; border: none; border-radius: 8px; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #454D59; border-radius: 4px; min-height: 28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QProgressBar { background: #343A44; border: none; border-radius: 3px; }
QProgressBar::chunk { background: #669CFF; border-radius: 3px; }
QTableWidget { background: #23262B; alternate-background-color: #262A30; border: none; selection-background-color: #304668; }
QTableWidget::item { padding: 0 8px; border: none; }
QHeaderView::section { background: #292D34; color: #A8AFBA; padding: 10px 8px; border: none; }
QTextEdit { background: #1C1F24; border: 1px solid #383D45; border-radius: 6px; padding: 8px; }
QStatusBar { color: #A8AFBA; background: #191B1F; }
QToolTip { color: #E8EAED; background: #23262B; border: 1px solid #383D45; padding: 6px; }
"""
