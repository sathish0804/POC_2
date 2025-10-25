from fastapi import FastAPI
from app.utils.logging_utils import configure_logging


def init_extensions(app: FastAPI) -> None:
    # Support overriding log path via environment or app config
    log_path = getattr(app, 'config', {}).get("LOG_PATH")
    debug = getattr(app, 'debug', False)
    configure_logging(log_path=log_path, verbose=debug)


