from typing import Optional
import os
from flask import Flask, jsonify, request
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

    # CORS: allow configurable origins for external UI
    def _get_cors_origin() -> str:
        # Set FRONTEND_ORIGIN env (e.g., https://ui.example.com) or allow all in development
        origin = os.getenv("FRONTEND_ORIGIN", "*")
        return origin

    @app.after_request
    def add_cors_headers(resp):
        origin = _get_cors_origin()
        resp.headers.add("Access-Control-Allow-Origin", origin)
        resp.headers.add("Vary", "Origin")
        resp.headers.add("Access-Control-Allow-Credentials", "true")
        resp.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization, Range")
        resp.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        # If preflight, return empty 204 with headers
        if request.method == "OPTIONS":
            resp.status_code = 204
            resp.data = b""
        return resp

    # JSON error handlers (API-friendly)
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "bad_request", "message": str(getattr(e, 'description', e))}), 400

    @app.errorhandler(404)
    def not_found(e):
        # Let HTML pages render normally; only JSONify /api/* paths
        if request.path.startswith("/api/"):
            return jsonify({"error": "not_found", "path": request.path}), 404
        return e

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "payload_too_large"}), 413

    @app.errorhandler(500)
    def internal_error(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "internal_server_error"}), 500
        return e

    app.register_blueprint(health_bp, url_prefix="/health")
    app.register_blueprint(ui_bp)
    return app


# Module import log
logger.debug(f"[{__name__}] module loaded")


