from typing import Optional
import os
from flask import Flask
from loguru import logger

from .config import get_config
from .extensions import init_extensions
from .controllers.health_controller import health_bp
from .controllers.ui_controller import ui_bp


def create_app(config_name: Optional[str] = None) -> Flask:
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))
    app = Flask(__name__, instance_relative_config=False, template_folder=template_dir)
    app.config.from_object(get_config(config_name))
    init_extensions(app)

    app.register_blueprint(health_bp, url_prefix="/health")
    app.register_blueprint(ui_bp)
    return app


# Module import log
logger.debug(f"[{__name__}] module loaded")


