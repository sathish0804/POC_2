from typing import Generator, Tuple
import cv2
import os
from loguru import logger


def sample_video_frames(video_path: str, sample_fps: int) -> Generator[Tuple[int, float, any], None, None]:
    """Yield frames at fixed intervals (1 frame per second default).

    Returns tuples: (frame_index, timestamp_sec, frame_bgr)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native_fps / max(1, sample_fps))))
    logger.debug(f"[video_utils] open {video_path}, fps={native_fps:.2f}, step={step}")

    frame_idx = 0
    sampled_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            timestamp = frame_idx / native_fps
            yield sampled_idx, timestamp, frame
            sampled_idx += 1
        frame_idx += 1
    cap.release()
    logger.debug(f"[video_utils] completed sampling for {video_path}")


def get_video_filename(video_path: str) -> str:
    return os.path.basename(video_path)


def get_video_duration_str(video_path: str) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps <= 0:
        return ""
    seconds = int(total / fps)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_expected_sampled_frames(video_path: str, sample_fps: int) -> int:
    """Estimate number of frames that will be yielded by sample_video_frames.

    Uses native fps and total frames to compute the sampling step and total samples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if native_fps <= 0 or total_frames <= 0:
        return 0
    step = max(1, int(round(native_fps / max(1, sample_fps))))
    # Number of frames we would read that satisfy frame_idx % step == 0 in [0, total_frames)
    # This is ceil(total_frames / step) but since index starts at 0, it's ((total_frames - 1) // step) + 1
    return ((total_frames - 1) // step) + 1 if total_frames > 0 else 0


# Module import log
logger.debug(f"[{__name__}] module loaded")
