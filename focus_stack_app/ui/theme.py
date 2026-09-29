"""Studio dark palette and modern creative desktop controls."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QLabel, QSizePolicy, QStyleFactory


def apply_theme(window):
    window.setStyle(QStyleFactory.create("Fusion"))
    font = QFont("Microsoft YaHei UI")
    font.setFamilies([
        "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI",
        "PingFang SC", "Noto Sans CJK SC", "sans-serif"
    ])
    font.setPixelSize(13)
    window.setFont(font)

    palette = QPalette()
    for role, color in {
        QPalette.Window: "#0F1218",
        QPalette.WindowText: "#F1F5F9",
        QPalette.Base: "#131720",
        QPalette.AlternateBase: "#181D26",
        QPalette.Text: "#F1F5F9",
        QPalette.Button: "#1E2430",
        QPalette.ButtonText: "#F1F5F9",
        QPalette.Highlight: "#2563EB",
        QPalette.HighlightedText: "#FFFFFF",
        QPalette.ToolTipBase: "#1E2430",
        QPalette.ToolTipText: "#F1F5F9",
        QPalette.PlaceholderText: "#64748B",
    }.items():
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#475569"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#475569"))
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
QMainWindow, QWidget#workspace {
    background: #0F1218;
}

QWidget {
    color: #F1F5F9;
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", "PingFang SC", sans-serif;
    font-size: 13px;
}

QFrame#panel, QWidget#settingsContent, QFrame#card {
    background: #181D26;
    border: 1px solid #262E3D;
    border-radius: 10px;
}

QFrame#innerCard {
    background: #141821;
    border: 1px solid #232B3A;
    border-radius: 8px;
}

QLabel {
    background: transparent;
}

QLabel[role="brand"] {
    font-size: 18px;
    font-weight: 700;
    color: #FFFFFF;
}

QLabel[role="title"] {
    font-size: 18px;
    font-weight: 600;
    color: #F8FAFC;
}

QLabel[role="section"] {
    font-size: 13px;
    font-weight: 600;
    color: #CBD5E1;
    letter-spacing: 0.5px;
}

QLabel[role="muted"] {
    color: #8E9AA8;
    font-size: 12px;
}

QLabel[role="badge"] {
    color: #60A5FA;
    background: #1A273D;
    border: 1px solid #2563EB;
    padding: 4px 14px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 12px;
}

QLabel[role="timer"] {
    color: #38BDF8;
    background: #111E2E;
    border: 1px solid #1E3A5F;
    padding: 3px 12px;
    border-radius: 12px;
    font-family: "Consolas", "Segoe UI Mono", monospace;
    font-weight: 600;
    font-size: 12px;
}

QLabel[role="metric"] {
    font-size: 22px;
    font-weight: 700;
    color: #38BDF8;
}

QLineEdit, QSpinBox, QComboBox {
    background: #121620;
    border: 1px solid #273040;
    border-radius: 6px;
    min-height: 32px;
    padding: 0 10px;
    selection-background-color: #2563EB;
}

QLineEdit:hover, QSpinBox:hover, QComboBox:hover {
    border-color: #3B82F6;
}

QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
    border-color: #60A5FA;
    background: #161B27;
}

QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {
    color: #475569;
    border-color: #1E2430;
    background: #11141C;
}

QComboBox QAbstractItemView {
    background: #1A202C;
    border: 1px solid #2D3748;
    color: #F1F5F9;
    selection-background-color: #2563EB;
    outline: none;
}

QComboBox::drop-down {
    border: none;
    width: 26px;
}

QComboBox::down-arrow {
    image: url(@ASSETS@/chevron-down.svg);
    width: 12px;
    height: 12px;
}

QSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border: none;
}

QSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 22px;
    border: none;
}

QSpinBox::up-arrow {
    image: url(@ASSETS@/chevron-up.svg);
    width: 10px;
    height: 10px;
}

QSpinBox::down-arrow {
    image: url(@ASSETS@/chevron-down.svg);
    width: 10px;
    height: 10px;
}

QPushButton {
    background: #1E2533;
    border: 1px solid #2E394E;
    border-radius: 6px;
    min-height: 32px;
    padding: 0 12px;
    font-weight: 500;
}

QPushButton:hover {
    background: #283245;
    border-color: #3B82F6;
    color: #FFFFFF;
}

QPushButton:pressed {
    background: #19202D;
}

QPushButton:focus {
    border-color: #3B82F6;
}

QPushButton:disabled {
    color: #475569;
    background: #151A23;
    border-color: #1F2633;
}

QPushButton#startButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563EB, stop:1 #3B82F6);
    color: #FFFFFF;
    border: 1px solid #3B82F6;
    font-weight: 600;
    min-height: 36px;
    border-radius: 8px;
    font-size: 14px;
}

QPushButton#startButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1D4ED8, stop:1 #2563EB);
}

QPushButton#startButton:pressed {
    background: #1E40AF;
}

QPushButton#startButton:disabled {
    background: #192333;
    color: #4B5A72;
    border-color: #1F2C3F;
}

QPushButton#stopButton {
    background: #1F242E;
    border: 1px solid #3A3F4D;
    color: #F87171;
    font-weight: 500;
}

QPushButton#stopButton:hover {
    background: #3B1D22;
    border-color: #EF4444;
    color: #FCA5A5;
}

QPushButton#stopButton:disabled {
    color: #553A3D;
    border-color: #252830;
    background: #181B22;
}

QPushButton.actionBtn {
    min-height: 28px;
    padding: 0 10px;
    font-size: 12px;
    background: #1A212E;
    border: 1px solid #2C3649;
}

QPushButton.actionBtn:hover {
    background: #2563EB;
    border-color: #3B82F6;
    color: #FFFFFF;
}

QCheckBox {
    spacing: 8px;
    min-height: 26px;
}

QCheckBox:disabled {
    color: #475569;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #3B4659;
    border-radius: 4px;
    background: #121620;
}

QCheckBox::indicator:checked {
    background: #2563EB;
    border-color: #3B82F6;
    image: url(@ASSETS@/check.svg);
}

QCheckBox::indicator:hover {
    border-color: #3B82F6;
}

QCheckBox::indicator:disabled {
    border-color: #232B3A;
    background: #151A24;
}

QScrollArea {
    background: #181D26;
    border: none;
    border-radius: 10px;
}

QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #334155;
    border-radius: 3px;
    min-height: 26px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
}

QProgressBar {
    background: #1E2533;
    border: none;
    border-radius: 4px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563EB, stop:1 #38BDF8);
    border-radius: 4px;
}

QTableWidget {
    background: #141822;
    alternate-background-color: #181D28;
    border: 1px solid #232B3A;
    border-radius: 8px;
    selection-background-color: #1E3A5F;
    gridline-color: transparent;
}

QTableWidget::item {
    padding: 0 8px;
    border: none;
}

QTableWidget::item:selected {
    background: #1E3A5F;
    color: #FFFFFF;
}

QHeaderView::section {
    background: #1B212D;
    color: #94A3B8;
    padding: 8px 8px;
    border: none;
    border-bottom: 1px solid #262E3D;
    font-weight: 600;
    font-size: 12px;
}

QTextEdit {
    background: #121620;
    border: 1px solid #262E3D;
    border-radius: 6px;
    padding: 8px;
    font-family: "Consolas", monospace;
    font-size: 12px;
    color: #CBD5E1;
}

QStatusBar {
    color: #8E9AA8;
    background: #0F1218;
    font-size: 12px;
    border-top: 1px solid #1E2430;
}

QToolTip {
    color: #F1F5F9;
    background: #1E2430;
    border: 1px solid #334155;
    padding: 6px;
    border-radius: 4px;
}
"""

