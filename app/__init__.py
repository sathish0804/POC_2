from typing import Optional
import os
from flask import Flask
from loguru import logger

from .config import get_config
from .extensions import init_extensions
from .controllers.health_controller import health_bp
from .controllers.ui_controller import ui_bp


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


def create_app(config_name: Optional[str] = None) -> Flask:
    _tune_runtime_threads()
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))
    app = Flask(__name__, instance_relative_config=False, template_folder=template_dir)
    app.config.from_object(get_config(config_name))
    init_extensions(app)

    app.register_blueprint(health_bp, url_prefix="/health")
    app.register_blueprint(ui_bp)
    return app


# Module import log
logger.debug(f"[{__name__}] module loaded")


