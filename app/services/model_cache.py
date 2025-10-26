import threading
from typing import Dict, Tuple, Optional
from loguru import logger
from app.services.yolo_service import YoloService
from app.services.mediapipe_service import MediaPipeService
from app.services.antenna_refiner import AntennaRefiner
from app.utils.ocr_utils import OcrUtils
from app.config import settings

_lock = threading.Lock()
_yolo_cache: Dict[Tuple[str, float, float], YoloService] = {}
_mediapipe_singleton: Optional[MediaPipeService] = None
_ocr_singleton: Optional[OcrUtils] = None
_antenna_cache: Dict[Tuple[str, bool], AntennaRefiner] = {}


def get_yolo_service(weights_path: str, conf: float, iou: float) -> YoloService:
    key = (str(weights_path), float(conf), float(iou))
    with _lock:
        svc = _yolo_cache.get(key)
        if svc is None:
            svc = YoloService(weights_path, conf=conf, iou=iou)
            _yolo_cache[key] = svc
            logger.debug(f"[model_cache] YOLO loaded: {key}")
        return svc


def get_mediapipe_service() -> MediaPipeService:
    global _mediapipe_singleton
    with _lock:
        if _mediapipe_singleton is None:
            _mediapipe_singleton = MediaPipeService()
            logger.debug("[model_cache] MediaPipe initialized")
        return _mediapipe_singleton


def get_ocr_utils(enabled: bool) -> Optional[OcrUtils]:
    if not enabled:
        return None
    global _ocr_singleton
    with _lock:
        if _ocr_singleton is None:
            _ocr_singleton = OcrUtils()
            logger.debug("[model_cache] OCR initialized")
        return _ocr_singleton


def get_antenna_refiner(yolo_weights: str, use_heuristic: bool = True) -> AntennaRefiner:
    key = (str(yolo_weights), bool(use_heuristic))
    with _lock:
        svc = _antenna_cache.get(key)
        if svc is None:
            svc = AntennaRefiner(yolo_weights=yolo_weights, use_heuristic=use_heuristic)
            _antenna_cache[key] = svc
            logger.debug(f"[model_cache] AntennaRefiner initialized: {key}")
        return svc


def preload_models(weights_path: str) -> None:
    try:
        get_yolo_service(weights_path, conf=0.25, iou=0.45)
    except Exception:
        pass
    try:
        get_mediapipe_service()
    except Exception:
        pass
    try:
        if bool(settings.preload_ocr):
            get_ocr_utils(True)
    except Exception:
        pass


