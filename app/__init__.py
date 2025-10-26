from loguru import logger

# This package no longer provides a Flask application. See `app/fastapi_app.py` and `asgi.py`.

logger.debug(f"[{__name__}] package loaded")


