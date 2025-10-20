from typing import Generator, Tuple
import cv2
import os
from loguru import logger


def sample_video_frames(video_path: str, sample_fps: float) -> Generator[Tuple[int, float, any], None, None]:
    """Yield frames at fixed intervals (default 1 FPS).

    Returns tuples: (sample_index, timestamp_sec, frame_bgr)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native_fps / max(1e-6, float(sample_fps)))))
    logger.debug(f"[video_utils] open {video_path}, fps={native_fps:.2f}, step={step}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sampled_idx = 0
    for frame_idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / native_fps
        yield sampled_idx, timestamp, frame
        sampled_idx += 1
    cap.release()
    logger.debug(f"[video_utils] completed sampling for {video_path}")


def get_video_filename(video_path: str) -> str:
    """Return basename of the provided video path."""
    return os.path.basename(video_path)


def get_video_duration_str(video_path: str) -> str:
    """Return HH:MM:SS duration string computed from FPS and total frames; empty on failure."""
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


def get_expected_sampled_frames(video_path: str, sample_fps: float) -> int:
    """Estimate number of samples yielded by `sample_video_frames` for the full video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if native_fps <= 0 or total_frames <= 0:
        return 0
    step = max(1, int(round(native_fps / max(1e-6, float(sample_fps)))))
    # Number of frames we would read that satisfy frame_idx % step == 0 in [0, total_frames)
    # This is ceil(total_frames / step) but since index starts at 0, it's ((total_frames - 1) // step) + 1
    return ((total_frames - 1) // step) + 1 if total_frames > 0 else 0


def get_expected_sampled_frames_in_range(video_path: str, sample_fps: float, start_frame: int, end_frame: int) -> int:
    """Estimate number of sampled frames within [start_frame, end_frame) using same sampling logic."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native_fps / max(1e-6, float(sample_fps)))))
    cap.release()
    lo = max(0, int(start_frame))
    hi = max(0, int(end_frame))
    if hi <= lo:
        return 0
    first = ((lo + step - 1) // step) * step
    if first >= hi:
        return 0
    return ((hi - 1 - first) // step) + 1


# Module import log
logger.debug(f"[{__name__}] module loaded")
