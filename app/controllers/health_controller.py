from flask import Blueprint, jsonify
from loguru import logger


health_bp = Blueprint("health", __name__)


@health_bp.get("/")
def health():
    return jsonify({"status": "ok"}), 200


# Module import log
logger.debug(f"[{__name__}] module loaded")


