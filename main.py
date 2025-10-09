import argparse
import os
from datetime import datetime

from controllers.pipeline import ActivityPipeline
from persistence.events_writer import write_events_to_json


def ensure_dummy_video(video_path: str, duration_seconds: int = 5, fps: int = 15) -> None:
    """Create a small dummy video if input path does not exist (for demo purposes)."""
    import cv2
    import numpy as np

    if os.path.exists(video_path):
        return

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    width, height = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    start = datetime(2025, 10, 3, 10, 0, 0)
    for _ in range(duration_seconds * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        text = (start).strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out.write(frame)
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCTV video activity detection for train loco pilots (CPU-only)")
    parser.add_argument("--video", type=str, default="/Users/satishvanga/Documents/Vanga/POC_2/example_data/sample_loco.mp4", help="Path to input video")
    parser.add_argument("--model", type=str, default="yolov8l.pt", help="YOLOv8 model weights (m or l)")
    parser.add_argument("--fps", type=int, default=1, help="Sampling FPS")
    parser.add_argument("--out", type=str, default="/Users/satishvanga/Documents/Vanga/POC_2/output/events.json", help="Where to write events JSON")
    parser.add_argument("--trip_id", type=str, default="DEMO-TRIP-001")
    parser.add_argument("--crew_name", type=str, default="Demo Crew")
    parser.add_argument("--crew_id", type=str, default="C-001")
    parser.add_argument("--crew_role", type=int, default=1)
    parser.add_argument("--disable_ocr", action="store_true", help="Disable OCR to speed up first run or avoid downloads")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose progress logging")
    parser.add_argument("--max_frames", type=int, default=0, help="Process only first N sampled frames (0=all)")
    # Sleep thresholds
    parser.add_argument("--sleep_eye_thresh", type=float, default=0.18, help="Eye openness threshold (<= means closed)")
    parser.add_argument("--sleep_headdown_deg", type=float, default=100.0, help="Head-down angle threshold (deg)")
    parser.add_argument("--sleep_micro_max_min", type=float, default=15.0, help="Max minutes for micro-sleep classification")
    parser.add_argument("--sleep_min_duration", type=float, default=10.0, help="Minimum seconds to consider a sleep episode")
    args = parser.parse_args()

    ensure_dummy_video(args.video)

    pipeline = ActivityPipeline(
        trip_id=args.trip_id,
        crew_name=args.crew_name,
        crew_id=args.crew_id,
        crew_role=args.crew_role,
        yolo_weights=args.model,
        sample_fps=args.fps,
        enable_ocr=not args.disable_ocr,
        verbose=args.verbose,
        max_frames=args.max_frames,
        sleep_eye_thresh=args.sleep_eye_thresh,
        sleep_headdown_deg=args.sleep_headdown_deg,
        sleep_micro_max_min=args.sleep_micro_max_min,
        sleep_min_duration=args.sleep_min_duration,
    )
    events = pipeline.process_video(args.video)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_events_to_json(events, args.out)
    print(f"Wrote {len(events)} events to {args.out}")
