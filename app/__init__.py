from typing import Optional
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from .config import get_config
from .extensions import init_extensions
from .controllers.health_controller import router as health_router
from .controllers.ui_controller import router as ui_router


def _tune_runtime_threads() -> None:
    """Constrain library-internal threading to avoid oversubscription under multiprocessing."""
    try:
        import torch
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
        torch.set_num_interop_threads(int(os.getenv("TORCH_NUM_INTEROP_THREADS", "1")))
    except Exception:
        pass
    try:
        import cv2
        cv2.setNumThreads(int(os.getenv("OPENCV_NUM_THREADS", "1")))
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass
    except Exception:
        pass


def create_app(config_name: Optional[str] = None) -> FastAPI:
    _tune_runtime_threads()
    
    # Get configuration
    config = get_config(config_name)
    
    # Create FastAPI app
    app = FastAPI(
        title="CCTV Activity Detection API",
        description="AI-powered CCTV activity detection and analysis",
        version="1.0.0",
        debug=config.DEBUG if hasattr(config, 'DEBUG') else False
    )
    
    # Initialize extensions
    init_extensions(app)
    
    # CORS configuration
    cors_origin = os.getenv("FRONTEND_ORIGIN", "*")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[cors_origin] if cors_origin != "*" else ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "Range"],
    )
    
    # Static files and templates
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
    
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    templates = Jinja2Templates(directory=template_dir)
    app.state.templates = templates
    
    # Include routers
    app.include_router(health_router, prefix="/health", tags=["Health"])
    app.include_router(ui_router, tags=["UI"])
    
    return app


# Create the FastAPI app instance
app = create_app()

# Module import log
logger.debug(f"[{__name__}] module loaded")


