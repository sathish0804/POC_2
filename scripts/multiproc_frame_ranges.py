import argparse
import multiprocessing as mp
import sys
import time
from typing import List, Tuple

import cv2


def split_frame_ranges(total_frames: int, num_processes: int) -> List[Tuple[int, int]]:
    """
    Evenly partition [0, total_frames) into num_processes contiguous ranges.
    Distributes any remainder to the first 'remainder' ranges.
    """
    if num_processes <= 0:
        return [(0, total_frames)]

    base = total_frames // num_processes
    remainder = total_frames % num_processes

    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(num_processes):
        count = base + (1 if i < remainder else 0)
        end = start + count
        ranges.append((start, end))
        start = end
    return ranges


def count_total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Fallback: if metadata is unavailable, count by reading (rare; slower)
    if total <= 0:
        total = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            total += 1

    cap.release()
    return total


def worker_function(name: str, video_path: str, start_frame: int, end_frame: int) -> None:
    print(f"[{name}] starting: frames [{start_frame}, {end_frame})")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{name}] ERROR: cannot open {video_path}")
        return

    # Position the capture to the start of this worker's assigned range
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    processed = 0
    current = start_frame
    while current < end_frame:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # Replace this section with your per-frame processing if needed.
        # For now, it just reads frames to measure IO/decoding throughput.

        processed += 1
        current += 1

    cap.release()
    print(f"[{name}] finished: processed {processed} frames [{start_frame}, {end_frame})")


def run_multiproc(video_path: str, num_processes: int) -> None:
    total_frames = count_total_frames(video_path)
    if total_frames <= 0:
        print("ERROR: Unable to determine total frames or video has zero frames.")
        sys.exit(1)
    ranges = split_frame_ranges(total_frames, num_processes)

    procs: List[mp.Process] = []
    t0 = time.perf_counter()
    for i, (start, end) in enumerate(ranges, start=1):
        if start >= end:
            print(f"[Worker-{i}] skipped: empty range [{start}, {end})")
            continue
        p = mp.Process(target=worker_function, args=(f"Worker-{i}", video_path, start, end))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    elapsed = time.perf_counter() - t0
    print(
        f"All workers finished in {elapsed:.2f} seconds. "
        f"Total frames: {total_frames}, processes used: {len(procs)}/{num_processes}"
    )


def cli_main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Multiprocess frame range reader")
    parser.add_argument("-v", "--video", required=True, help="Path to input video")
    parser.add_argument("-p", "--processes", type=int, default=10, help="Number of processes (default: 10)")
    args = parser.parse_args(argv)
    mp.set_start_method("spawn", force=True)  # macOS-safe start method
    run_multiproc(args.video, max(1, args.processes))


