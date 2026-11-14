"""Render the existing layers SVG into a Windows multi-resolution icon.

Run with the project's Python environment: python tools/generate_app_icon.py
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer


ICON_DIR = Path(__file__).resolve().parents[1] / "focus_stack_app" / "ui" / "icons"
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication([])
    renderer = QSvgRenderer(str(ICON_DIR / "layers.svg"))
    if not renderer.isValid():
        raise ValueError("Could not load layers.svg")

    frames = []
    for size in SIZES:
        # Render from the vector at each target size, with antialiased edges.
        edge = size * 4
        image = QImage(edge, edge, QImage.Format_RGBA8888)
        image.fill(0)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        renderer.render(painter, QRectF(0, 0, edge, edge))
        painter.end()
        frame = Image.frombytes("RGBA", (edge, edge), bytes(image.constBits()))
        frames.append(frame.resize((size, size), Image.Resampling.LANCZOS))

    target = ICON_DIR / "layers.ico"
    frames[-1].save(
        target,
        format="ICO",
        sizes=[(size, size) for size in SIZES],
        append_images=frames[:-1],
    )
    print(f"Created {target} ({', '.join(map(str, SIZES))} px)")
    app.quit()


if __name__ == "__main__":
    main()
