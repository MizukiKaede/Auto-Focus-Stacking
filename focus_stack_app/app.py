"""Application entry point for the Focus Stack desktop UI."""

from __future__ import annotations

import sys
from typing import Any, Sequence

from .pipeline.application_controller import ApplicationController, ApplicationOptions


def default_controller_factory(**settings: Any) -> ApplicationController:
    """Return the production controller for the default desktop window."""

    return ApplicationController(settings)


def create_main_window(**kwargs: Any) -> Any:
    """Create a configured ``MainWindow`` without starting Qt's event loop."""

    from .ui.main_window import MainWindow

    return MainWindow(**kwargs)


def create_application(argv: Sequence[str] | None = None) -> tuple[Any, Any]:
    """Create ``QApplication`` and the main window lazily.

    Importing the package on a headless machine remains safe; PySide6 is only
    required when this function is actually called.
    """

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise RuntimeError("PySide6 is required to run Focus Stack Assistant") from exc
    app = QApplication.instance() or QApplication(list(argv if argv is not None else sys.argv))
    # MainWindow's default factory is intentionally explicit here as well:
    # this documents that a bare launch is a working application, not a
    # coordinator-only demo, and makes the dependency easy to replace in a
    # test or embedding application.
    window = create_main_window(controller_factory=default_controller_factory)
    return app, window


def main(argv: Sequence[str] | None = None) -> int:
    app, window = create_application(argv)
    window.show()
    return int(app.exec())


run = main


__all__ = ["ApplicationController", "ApplicationOptions", "default_controller_factory",
           "create_main_window", "create_application", "main", "run"]
