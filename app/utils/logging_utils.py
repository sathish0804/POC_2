from __future__ import annotations

import logging
import os
from typing import Optional

from loguru import logger


class _InterceptHandler(logging.Handler):
    """Redirect standard logging records to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except Exception:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(log_path: Optional[str] = None, verbose: bool = False) -> str:
    """
    Configure unified logging across all modules.

    - Routes Python stdlib logging (e.g., FastAPI, Gunicorn, libraries) to Loguru
    - Writes human-readable console logs and rotates a file log
    - Defaults to output/app.log; override with LOG_PATH env var or argument
    - Rotation: 10 MB, Retention: 7 days
    - Level: DEBUG when verbose=True else INFO
    """
    level = "DEBUG" if verbose else "INFO"

    # Allow overriding via environment if not explicitly provided
    if not log_path:
        log_path = os.getenv("LOG_PATH")

    if not log_path:
        # Default to /.../output/app.log at repo root
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_dir = os.path.join(repo_root, "output")
        os.makedirs(default_dir, exist_ok=True)
        log_path = os.path.join(default_dir, "app.log")
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Bridge stdlib logging to Loguru
    logging.root.handlers = [
        _InterceptHandler(),
    ]
    logging.root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for noisy in ("gunicorn", "gunicorn.error", "gunicorn.access", "werkzeug"):
        lg = logging.getLogger(noisy)
        lg.handlers = []
        lg.propagate = True

    # Remove previous sinks to avoid duplicate logs when called multiple times
    logger.remove()

    # Console sink
    logger.add(lambda msg: print(msg, end=""), level=level, enqueue=True)

    # File sink
    logger.add(
        log_path,
        level=level,
        rotation="10 MB",
        retention="7 days",
        backtrace=False,
        diagnose=False,
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
    )

    logger.debug(f"Logging configured. Path={log_path}, level={level}")
    return log_path


