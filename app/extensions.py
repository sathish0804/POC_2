from flask import Flask
from app.utils.logging_utils import configure_logging


def init_extensions(app: Flask) -> None:
    configure_logging(verbose=bool(app.config.get("DEBUG", False)))


