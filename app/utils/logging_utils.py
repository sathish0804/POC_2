from __future__ import annotations

import os
from typing import Optional

from loguru import logger


def configure_logging(log_path: Optional[str] = None, verbose: bool = False) -> str:
    """
    Configure Loguru to write logs to a rotating file. Returns the log file path.

    - If log_path is None or empty, defaults to output/app.log under the project root
    - Rotation: 10 MB, Retention: 7 days, Compression: none
    - Level: DEBUG when verbose=True else INFO
    """
    level = "DEBUG" if verbose else "INFO"

    if not log_path:
        # Default to /.../output/app.log at repo root
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_dir = os.path.join(repo_root, "output")
        os.makedirs(default_dir, exist_ok=True)
        log_path = os.path.join(default_dir, "app.log")
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

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


