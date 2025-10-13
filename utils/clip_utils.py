from typing import Optional
import os
import shutil
import subprocess
import cv2


def extract_clip(video_path: str, center_ts: float, out_dir: str, tag: str, duration_s: float = 4.0) -> Optional[str]:
    """Extract a short MP4 clip around center_ts. Returns filename or None."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        start = max(0.0, float(center_ts) - float(duration_s) / 2.0)
    except Exception:
        start = 0.0
    out_name = f"{tag}_activity.mp4"
    out_path = os.path.join(out_dir, out_name)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            cmd = [
                ffmpeg, "-y", "-ss", f"{start:.3f}", "-i", video_path,
                "-t", f"{duration_s:.3f}", "-an",
                "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                "-movflags", "+faststart", out_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.isfile(out_path):
                return out_name
        except Exception:
            pass

    # Fallback to OpenCV writer if ffmpeg is unavailable or failed
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start_frame = max(0, int(start * fps))
        end_frame = min(max(0, total - 1), int((start + duration_s) * fps))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frame_idx += 1
        writer.release()
        cap.release()
        return out_name if os.path.isfile(out_path) else None
    except Exception:
        return None


