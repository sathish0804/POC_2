from flask import Flask
from app.utils.logging_utils import configure_logging


def init_extensions(app: Flask) -> None:
    # Support overriding log path via environment or app config
    log_path = app.config.get("LOG_PATH")
    configure_logging(log_path=log_path, verbose=bool(app.config.get("DEBUG", False)))


