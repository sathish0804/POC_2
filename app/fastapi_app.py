from __future__ import annotations

import os
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings
from app.utils.logging_utils import configure_logging
from app.controllers import api_router
from app.controllers.health_controller import router as health_router
from app.controllers.jobs_controller import router as jobs_router


def _get_cors_origin() -> str:
    origin = os.getenv("FRONTEND_ORIGIN", "*")
    return origin


def create_app(config_name: Optional[str] = None) -> FastAPI:
    app = FastAPI()

    # Configure logging
    log_dir = settings.log_dir or "output"
    import os as _os
    _os.makedirs(log_dir, exist_ok=True)
    log_path = _os.path.join(log_dir, "app.log")
    verbose = (settings.environment or "production").lower() != "production"
    try:
        configure_logging(log_path=log_path, verbose=verbose)
    except Exception:
        pass

    # CORS
    origin = settings.frontend_origin or "*"
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin] if origin != "*" else ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges"],
    )

    # Routers
    api_router.include_router(health_router)
    api_router.include_router(jobs_router)
    app.include_router(api_router)

    logger.debug(f"[{__name__}] FastAPI app created with routers")
    return app


# ASGI application default (development)
app = create_app(os.getenv("APP_ENV", None))


