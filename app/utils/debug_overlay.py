from typing import List, Tuple
import os
import cv2
from loguru import logger


def _draw_text_block(img, lines: List[str], origin: Tuple[int, int] = (10, 30)) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (255, 255, 255)
    thickness = 1
    line_h = 18
    # Background rectangle
    if lines:
        max_w = 0
        for ln in lines:
            size, _ = cv2.getTextSize(ln, font, scale, thickness)
            max_w = max(max_w, size[0])
        pad = 8
        rect_w = max_w + pad * 2
        rect_h = line_h * len(lines) + pad * 2
        overlay = img.copy()
        cv2.rectangle(overlay, (x - pad, y - line_h - pad), (x - pad + rect_w, y - line_h - pad + rect_h), (0, 0, 0), -1)
        alpha = 0.45
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    # Text lines
    ty = y
    for ln in lines:
        cv2.putText(img, ln, (x, ty), font, scale, color, thickness, cv2.LINE_AA)
        ty += line_h


def save_debug_overlay(frame_bgr, lines: List[str], tag: str, out_dir: str, subdir: str = "debug") -> str:
    os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)
    img = frame_bgr.copy()
    _draw_text_block(img, lines, origin=(10, 30))
    out_path = os.path.join(out_dir, subdir, f"{tag}_debug.jpg")
    cv2.imwrite(out_path, img)
    logger.debug(f"[debug_overlay] saved {out_path}")
    return out_path


