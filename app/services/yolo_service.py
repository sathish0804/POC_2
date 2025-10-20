from typing import List, Tuple
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
        # Allow configurable inference resolution for speed/accuracy tradeoff on CPU
        try:
            self.imgsz =  640
        except Exception:
            self.imgsz = 640
        logger.debug(f"[YoloService] model loaded from {weights_path}, conf={conf}, iou={iou}")

    def detect(self, image_bgr: np.ndarray) -> List[Tuple[int, float, Tuple[float, float, float, float]]]:
        """Run detection and return list of (class_id, score, (x1,y1,x2,y2))."""
        results = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device='cpu',
            imgsz=self.imgsz,
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

    def detect_batch(self, images_bgr: List[np.ndarray]) -> List[List[Tuple[int, float, Tuple[float, float, float, float]]]]:
        """Run detection on a batch of images and return list of per-image detections.

        Each element in the returned list corresponds to an input image and is a list of
        (class_id, score, (x1,y1,x2,y2)).
        """
        if not images_bgr:
            return []
        results = self.model.predict(
            source=images_bgr,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device='cpu',
            imgsz=self.imgsz,
        )
        batched: List[List[Tuple[int, float, Tuple[float, float, float, float]]]] = []
        for r in results:
            dets: List[Tuple[int, float, Tuple[float, float, float, float]]] = []
            if getattr(r, "boxes", None) is not None:
                for b in r.boxes:
                    cls_id = int(b.cls.item()) if b.cls is not None else -1
                    score = float(b.conf.item()) if b.conf is not None else 0.0
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    dets.append((cls_id, score, (x1, y1, x2, y2)))
            batched.append(dets)
        return batched


# Module import log
logger.debug(f"[{__name__}] module loaded")
