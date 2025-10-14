from typing import List, Tuple
import cv2
import numpy as np
from loguru import logger


BBox = Tuple[int, int, int, int]


def detect_window_regions(
    image_bgr: np.ndarray,
    *,
    brightness_thresh: int = 180,
    min_area_frac: float = 0.05,
    max_area_frac: float = 0.50,
    close_kernel: int = 9,
) -> List[BBox]:
    """Heuristic detector for bright window regions.

    - Assumes windows are relatively bright vs cabin interior and roughly rectangular.
    - Returns candidate boxes sorted by area descending.
    """
    H, W = image_bgr.shape[:2]
    area = float(H * W)
    min_area = max(1.0, min_area_frac * area)
    max_area = max(1.0, max_area_frac * area)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, int(brightness_thresh), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[BBox] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        a = float(w * h)
        if a < min_area or a > max_area:
            continue
        # Prefer boxes adjacent to right side (typical window position)
        if x > (W * 0.2):
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes


# Module import log
logger.debug(f"[{__name__}] module loaded")
