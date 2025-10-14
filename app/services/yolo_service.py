from typing import List, Tuple, Optional
import numpy as np
from ultralytics import YOLO
from loguru import logger


class YoloService:
    """CPU-only YOLOv8 inference wrapper."""

    def __init__(self, weights_path: str, conf: float = 0.25, iou: float = 0.45):
        self.model = YOLO(weights_path)
        self.model.to('cpu')
        self.conf = conf
        self.iou = iou
        logger.debug(f"[YoloService] model loaded from {weights_path}, conf={conf}, iou={iou}")

    def detect(self, image_bgr: np.ndarray) -> List[Tuple[int, float, Tuple[float, float, float, float]]]:
        """Run detection and return list of (class_id, score, (x1,y1,x2,y2))."""
        results = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device='cpu'
        )
        detections: List[Tuple[int, float, Tuple[float, float, float, float]]] = []
        if not results:
            return detections
        r0 = results[0]
        if r0.boxes is None:
            return detections
        for b in r0.boxes:
            cls_id = int(b.cls.item()) if b.cls is not None else -1
            score = float(b.conf.item()) if b.conf is not None else 0.0
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            detections.append((cls_id, score, (x1, y1, x2, y2)))
        return detections

    def class_name(self, class_id: int) -> Optional[str]:
        try:
            return self.model.names.get(class_id)
        except Exception:
            return None


# Module import log
logger.debug(f"[{__name__}] module loaded")
