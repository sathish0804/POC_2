from typing import Optional, Dict, Any, List, Tuple
import os
import math
import numpy as np
import cv2
from loguru import logger

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


class AntennaRefiner:
    def __init__(
        self,
        yolo_weights: Optional[str] = None,
        conf: float = 0.25,
        imgsz: int = 320,
        use_heuristic: bool = True,
        roi_up_frac: float = 1.2,
        roi_side_frac: float = 0.30,
        angle_deg: float = 20.0,
        min_len_frac: float = 0.20,
        anchor_tol_frac: float = 0.12,
        center_band_frac: float = 0.35,
        required_up_frac: float = 0.20,
        omnidirectional: bool = True,
        side_margin_frac: float = 0.30,
    ):
        self.model = None
        self.conf = conf
        self.imgsz = imgsz
        self.use_heuristic = use_heuristic
        self.roi_up_frac = float(max(0.0, roi_up_frac))
        self.roi_side_frac = float(max(0.0, roi_side_frac))
        self.angle_deg = float(max(0.0, angle_deg))
        self.min_len_frac = float(max(0.0, min_len_frac))
        self.anchor_tol_frac = float(max(0.0, anchor_tol_frac))
        self.center_band_frac = float(max(0.0, center_band_frac))
        self.required_up_frac = float(max(0.0, required_up_frac))
        self.omnidirectional = bool(omnidirectional)
        self.side_margin_frac = float(max(0.0, side_margin_frac))
        if yolo_weights and os.path.isfile(yolo_weights) and YOLO is not None:
            try:
                self.model = YOLO(yolo_weights)
                self.model.overrides["conf"] = conf
                logger.debug("[AntennaRefiner] YOLO model initialized for antenna detection")
            except Exception:
                self.model = None

    @staticmethod
    def _clip(val: int, lo: int, hi: int) -> int:
        return int(max(lo, min(hi, val)))

    def _roi_for_antenna(self, device_bbox: List[float], frame_shape: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        H, W, _ = frame_shape
        x1, y1, x2, y2 = map(int, device_bbox)
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        extend_up = int(self.roi_up_frac * h)
        extend_side = int(self.roi_side_frac * w)
        rx1 = self._clip(x1 - extend_side, 0, W - 1)
        ry1 = self._clip(y1 - extend_up, 0, H - 1)
        rx2 = self._clip(x2 + extend_side, 0, W - 1)
        ry2 = self._clip(y2, 0, H - 1)
        drx1 = x1 - rx1
        dry1 = y1 - ry1
        drx2 = x2 - rx1
        dry2 = y2 - ry1
        return (rx1, ry1, rx2, ry2), (drx1, dry1, drx2, dry2)

    @staticmethod
    def _is_vertical(dx: int, dy: int, max_deg: float = 8.0) -> bool:
        if dy == 0:
            return False
        return abs(dx) <= math.tan(math.radians(max_deg)) * abs(dy)

    def _heuristic_detect(self, roi_bgr: np.ndarray, device_roi_box: Tuple[int, int, int, int]) -> Tuple[bool, Dict[str, Any]]:
        h, w, _ = roi_bgr.shape
        if h < 20 or w < 20:
            return False, {"reason": "roi_too_small"}
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
        min_len = max(10, int(self.min_len_frac * h))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=min_len, maxLineGap=10)
        if lines is None:
            return False, {"reason": "no_lines"}
        x1d, y1d, x2d, y2d = device_roi_box
        device_top_y = y1d
        side_margin_px = int(self.side_margin_frac * max(x2d - x1d, y2d - y1d))
        anchor_tol_px = max(4, int(self.anchor_tol_frac * h))
        required_up = int(self.required_up_frac * h)
        for l in lines[:, 0, :]:
            x1l, y1l, x2l, y2l = map(int, l.tolist())
            dx = x2l - x1l
            dy = y2l - y1l
            length = math.hypot(dx, dy)
            if length < min_len:
                continue
            def accept_top():
                if not self._is_vertical(dx, dy, max_deg=self.angle_deg):
                    return False
                bottom_y = max(y1l, y2l)
                top_y = min(y1l, y2l)
                bottom_x = x1l if y1l >= y2l else x2l
                anchored = (
                    abs(bottom_y - device_top_y) <= anchor_tol_px
                    and (x1d - side_margin_px) <= bottom_x <= (x2d + side_margin_px)
                )
                extends_up_enough = (device_top_y - top_y) >= required_up
                return anchored and extends_up_enough
            ok = accept_top()
            if ok:
                return True, {"rule": "hough_line_top", "length": float(length)}
        return False, {"reason": "no_vertical_line_touching_device"}

    def has_antenna(self, frame_bgr: np.ndarray, device_bbox: List[float]) -> Tuple[bool, Dict[str, Any]]:
        (rx1, ry1, rx2, ry2), device_roi_box = self._roi_for_antenna(device_bbox, frame_bgr.shape)
        roi_bgr = frame_bgr[ry1:ry2, rx1:rx2]
        if self.model is not None and roi_bgr.size > 0:
            try:
                res = self.model.predict(source=roi_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
                boxes = res.boxes
                names = res.names if hasattr(res, "names") else {}
                if boxes is not None and len(boxes) > 0:
                    clss = boxes.cls.cpu().numpy().astype(int)
                    for ci in clss:
                        name = str(names.get(int(ci), str(ci))).lower()
                        if "antenna" in name:
                            return True, {"rule": "yolo_antenna", "roi": [rx1, ry1, rx2, ry2]}
            except Exception:
                pass
        if self.use_heuristic and roi_bgr.size > 0:
            ok, ev = self._heuristic_detect(roi_bgr, device_roi_box)
            if ok:
                ev.update({"roi": [rx1, ry1, rx2, ry2]})
                return True, ev
        return False, {"reason": "no_antenna_detected", "roi": [rx1, ry1, rx2, ry2]}


# Module import log
logger.debug(f"[{__name__}] module loaded")
