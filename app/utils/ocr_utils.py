from typing import Optional, Tuple
import re
from loguru import logger

try:
    import easyocr  # type: ignore
except Exception:  # pragma: no cover
    easyocr = None

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None


DATE_RE = re.compile(r"(20\d{2})[-/](\d{2})[-/](\d{2})")
TIME_RE = re.compile(r"(\d{2}):(\d{2})(?::(\d{2}))?")


class OcrUtils:
    """OCR helpers for date/time overlays.

    Prefer EasyOCR; fallback to Tesseract if not available.
    """

    def __init__(self):
        self.reader = None
        if easyocr is not None:
            try:
                self.reader = easyocr.Reader(['en'], gpu=False)
                logger.debug("[OcrUtils] EasyOCR initialized")
            except Exception:
                self.reader = None

    def extract_text(self, image_bgr) -> str:
        if self.reader is not None:
            try:
                results = self.reader.readtext(image_bgr)
                return " ".join([r[1] for r in results])
            except Exception:
                pass
        if pytesseract is not None:
            try:
                txt = pytesseract.image_to_string(image_bgr)
                return txt
            except Exception:
                pass
        return ""

    def extract_date_time(self, image_bgr) -> Tuple[str, str]:
        text = self.extract_text(image_bgr)
        date_match = DATE_RE.search(text)
        time_match = TIME_RE.search(text)
        date_str = date_match.group(0) if date_match else ""
        time_str = time_match.group(0) if time_match else ""
        return date_str, time_str


# Module import log
logger.debug(f"[{__name__}] module loaded")
