from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings
from app.utils.logging_utils import configure_logging
from app.controllers import api_router
from app.controllers.health_controller import router as health_router
from app.controllers.jobs_controller import router as jobs_router


def create_app() -> FastAPI:
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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Routers
    api_router.include_router(health_router)
    api_router.include_router(jobs_router)
    app.include_router(api_router)

    logger.debug(f"[{__name__}] FastAPI app created with routers")
    return app


# ASGI application default (development)
app = create_app()


