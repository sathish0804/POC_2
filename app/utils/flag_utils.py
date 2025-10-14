from typing import List, Tuple
import cv2
import numpy as np
from loguru import logger


BBox = Tuple[int, int, int, int]


def _morph(mask: np.ndarray, open_ks: int, close_ks: int) -> np.ndarray:
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)
    return closed


def detect_green_flags(
    image_bgr: np.ndarray,
    *,
    min_area_frac: float = 0.002,
    min_saturation: int = 60,
    min_value: int = 60,
    hue_lo: int = 35,
    hue_hi: int = 90,
    open_kernel: int = 5,
    close_kernel: int = 7,
) -> List[BBox]:
    """Return list of green flag candidate boxes (x1,y1,x2,y2) in full-frame coords.

    This is a simple HSV segmentation tuned for bright-green flags typically used in
    signal exchange. Final selection is filtered by area and basic contour quality.
    """
    H, W = image_bgr.shape[0], image_bgr.shape[1]
    frame_area = float(H * W)
    min_area = max(1.0, float(min_area_frac) * frame_area)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([int(hue_lo), int(min_saturation), int(min_value)], dtype=np.uint8)
    upper = np.array([int(hue_hi), 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    mask = _morph(mask, int(open_kernel), int(close_kernel))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[BBox] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = float(w * h)
        if area < min_area:
            continue
        # Reject extremely thin shapes; flags are usually not needle-like.
        ar = (h / max(1.0, float(w)))
        if ar < 0.25 or ar > 4.0:
            continue
        cnt_area = float(cv2.contourArea(cnt))
        solidity = cnt_area / max(1.0, area)
        if solidity < 0.25:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))

    return boxes


# Module import log
logger.debug(f"[{__name__}] module loaded")
