"""Application logging setup with a project-local rotating log file."""

from __future__ import annotations

from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import sys


DEFAULT_LOGGER_NAME = "focus_stack_app"
DEFAULT_LOG_FILENAME = "application.log"


def get_log_path(cache_root: str | Path, *, filename: str = DEFAULT_LOG_FILENAME) -> Path:
    """Resolve the log path from either a cache root or its ``logs`` folder."""

    root = Path(cache_root)
    if root.name.lower() != "logs":
        root = root / "logs"
    return root / filename


def configure_logging(
    cache_root: str | Path,
    *,
    level: int = logging.INFO,
    logger_name: str = DEFAULT_LOGGER_NAME,
    filename: str = DEFAULT_LOG_FILENAME,
    console: bool = True,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """Configure an idempotent logger and return it.

    The file handler uses UTF-8 and is safe for Windows paths.  Calling this
    repeatedly (for example after opening another project) replaces stale file
    handlers without duplicating every message.
    """

    if not filename or Path(filename).name != filename:
        raise ValueError("filename must be a single file name")
    log_path = get_log_path(cache_root, filename=filename)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    expected = str(log_path.resolve(strict=False)).casefold()
    for handler in list(logger.handlers):
        if getattr(handler, "_focus_stack_file", False):
            if getattr(handler, "baseFilename", "").casefold() != expected:
                logger.removeHandler(handler)
                handler.close()

    if not any(
        getattr(handler, "_focus_stack_file", False)
        and getattr(handler, "baseFilename", "").casefold() == expected
        for handler in logger.handlers
    ):
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,
        )
        file_handler._focus_stack_file = True  # type: ignore[attr-defined]
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s")
        )
        logger.addHandler(file_handler)

    if console and not any(getattr(handler, "_focus_stack_console", False) for handler in logger.handlers):
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler._focus_stack_console = True  # type: ignore[attr-defined]
        console_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(console_handler)

    return logger


def setup_logging(*args, **kwargs) -> logging.Logger:
    """Compatibility alias for :func:`configure_logging`."""

    return configure_logging(*args, **kwargs)


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or DEFAULT_LOGGER_NAME)


__all__ = ["configure_logging", "setup_logging", "get_logger", "get_log_path"]
