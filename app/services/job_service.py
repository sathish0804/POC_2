from __future__ import annotations

import json
import math
import os
import shutil
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from loguru import logger

from app.boot import get_pool
from app.config import settings
from app.utils.path_utils import resource_path


# In-memory job registry; lightweight state is also persisted to disk
JOBS: Dict[str, Dict[str, Any]] = {}


# Persist lightweight job state to survive worker restarts
STATE_DIR = Path(__file__).resolve().parents[2] / "output" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)


def state_path(job_id: str) -> Path:
    return STATE_DIR / f"{job_id}.json"


def persist_state(job_id: str, state: Dict[str, Any]) -> None:
    target_path = state_path(job_id)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "processed": int(state.get("processed", 0)),
                "total": int(state.get("total", 0)),
                "done": bool(state.get("done", False)),
                "error": state.get("error"),
                "trip_id": state.get("trip_id"),
                "asset_root": state.get("asset_root"),
                "host_url": "https://celebxmedia.info",  # Persist host URL for external API calls
            },
            f,
        )
    tmp_path.replace(target_path)


def load_state(job_id: str) -> Dict[str, Any] | None:
    path = state_path(job_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def ffprobe_duration_seconds(video_path: str) -> float:
    """Fast duration probe via ffprobe; returns seconds or 0.0 on failure."""
    try:
        import subprocess
        out = subprocess.check_output([
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path,
        ], stderr=subprocess.STDOUT, text=True)
        data = json.loads(out)
        dur = float(data.get("format", {}).get("duration", 0.0) or 0.0)
        return max(0.0, dur)
    except Exception:
        return 0.0


def count_total_frames(video_path: str) -> int:
    """Count total frames using metadata, fall back to scan if needed."""
    try:
        import cv2
    except Exception:
        return 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        total = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            total += 1
    cap.release()
    return total


def split_frame_ranges(total_frames: int, num_processes: int) -> List[Tuple[int, int]]:
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


def split_frame_ranges_by_seconds(video_path: str, chunk_seconds: float) -> List[Tuple[int, int]]:
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
    except Exception:
        fps = 30.0
        total_frames = count_total_frames(video_path)

    if total_frames <= 0:
        return []

    try:
        sec = float(chunk_seconds)
    except Exception:
        sec = 6.0
    sec = max(1.0, sec)

    frames_per_chunk = max(1, int(round(fps * sec)))
    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < total_frames:
        end = min(total_frames, start + frames_per_chunk)
        ranges.append((start, end))
        start = end
    return ranges


def expected_sampled_in_range(video_path: str, sample_fps: float, start_frame: int, end_frame: int) -> int:
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(native_fps / max(1e-6, float(sample_fps)))))
        cap.release()
    except Exception:
        step = max(1, int(round(30.0 / max(1e-6, float(sample_fps)))))

    lo = max(0, int(start_frame))
    hi = max(0, int(end_frame))
    if hi <= lo:
        return 0
    first = ((lo + step - 1) // step) * step
    if first >= hi:
        return 0
    return ((hi - 1 - first) // step) + 1


def worker_run_range(vpath: str, start_f: int, end_f: int, pipeline_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    import os as _os
    try:
        import torch as _torch
        _torch.set_num_threads(int(_os.getenv("TORCH_NUM_THREADS", "1")))
        _torch.set_num_interop_threads(int(_os.getenv("TORCH_NUM_INTEROP_THREADS", "1")))
    except Exception:
        pass
    try:
        import cv2 as _cv2
        _cv2.setNumThreads(int(_os.getenv("OPENCV_NUM_THREADS", "1")))
        try:
            _cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass
    except Exception:
        pass
    from app.services.pipeline_service import ActivityPipeline
    pipe = ActivityPipeline(**pipeline_cfg)
    try:
        evts = pipe.process_video_range(vpath, start_f, end_f, progress_cb=None)
    except Exception:
        evts = []
    return [e.model_dump() if hasattr(e, "model_dump") else e for e in (evts or [])]


def run_job(jid: str) -> None:
    state = JOBS.get(jid)
    logger.info(f"[jobs] run_job start jid={jid}, has_state={bool(state)}")
    try:
        video_path = state["video_path"]
        import multiprocessing as mp
        num_processes = max(1, min(int(settings.pool_procs or 6), (mp.cpu_count() or 1)))

        # Prefer expected sampled frames for progress denominator
        try:
            from app.utils.video_utils import get_expected_sampled_frames
            total_sampled = int(get_expected_sampled_frames(video_path, float(settings.sample_fps or 0.5)))
        except Exception:
            total_sampled = count_total_frames(video_path)
        state["total"] = int(total_sampled)
        persist_state(jid, state)

        total_frames = count_total_frames(video_path)
        if total_frames <= 0:
            raise RuntimeError("Video has zero frames or cannot be read")

        # Ensure a valid working directory for libs that rely on cwd
        try:
            cur = os.getcwd()
            if not os.path.isdir(cur):
                raise FileNotFoundError
        except Exception:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            try:
                os.chdir(repo_root)
                logger.info(f"[jobs] run_job: reset cwd to {repo_root}")
            except Exception as cwd_err:
                logger.warning(f"[jobs] run_job: failed to reset cwd: {cwd_err}")
            try:
                os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", os.path.join(repo_root, "output", ".ultralytics"))
            except Exception:
                pass

        from app.services.pipeline_service import ActivityPipeline
        pipeline_cfg = dict(
            trip_id=state["trip_id"],
            crew_name="demo",
            crew_id="1",
            crew_role=1,
            yolo_weights=resource_path("yolo11s.pt"),
            sample_fps=float(settings.sample_fps or 0.5),
            enable_ocr=False,
            verbose=False,
            max_frames=0,
            use_advanced_sleep=True,
            sleep_min_duration=10.0,
            sleep_micro_max_min=0.25,
            save_debug_overlays=0,
            sleep_cfg_short_window_s=4.0,
            sleep_cfg_mid_window_s=30.0,
            sleep_cfg_long_window_s=120.0,
            sleep_cfg_smoothing_alpha=0.5,
            sleep_cfg_eye_closed_run_s=2.0,
            sleep_cfg_perclos_drowsy_thresh=0.35,
            sleep_cfg_perclos_sleep_thresh=0.75,
            sleep_cfg_head_pitch_down_deg=20.0,
            sleep_cfg_head_neutral_deg=12.0,
            sleep_cfg_hold_transition_s=1.0,
            sleep_cfg_recovery_hold_s=2.0,
            sleep_cfg_open_prob_closed_thresh=0.50,
            sleep_cfg_no_eye_head_down_deg=30.0,
        )

        try:
            chunk_seconds = float(settings.chunk_seconds or 6.0)
        except Exception:
            chunk_seconds = 6.0

        ranges = split_frame_ranges_by_seconds(video_path, chunk_seconds)
        if not ranges:
            ranges = split_frame_ranges(total_frames, num_processes)

        per_range_expected = [
            expected_sampled_in_range(video_path, float(pipeline_cfg.get("sample_fps", 1.0)), s, e)
            for (s, e) in ranges
        ]
        total_expected = sum(per_range_expected)
        state["total"] = int(total_expected or state.get("total", 0))
        persist_state(jid, state)

        t0 = time.perf_counter()
        all_events: List[Dict[str, Any]] = []
        completed_expected = 0
        pool = get_pool(max_workers=num_processes)
        futures = []
        for idx, (start_f, end_f) in enumerate(ranges):
            if start_f >= end_f:
                continue
            futures.append((idx, per_range_expected[idx], pool.submit(worker_run_range, video_path, start_f, end_f, pipeline_cfg)))

        for idx, expected_cnt, fut in futures:
            try:
                part = fut.result()
                if isinstance(part, list):
                    all_events.extend(part)
                else:
                    all_events.append(part)
            except Exception:
                part = []
            finally:
                completed_expected += int(expected_cnt)
                state["processed"] = int(completed_expected)
                persist_state(jid, state)

        # Persist output assets to repo output/<timestamp>
        try:
            tmp_output_dir = os.path.join(os.path.dirname(state["video_path"]), "output")
            if os.path.isdir(tmp_output_dir):
                from datetime import datetime
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                dest_root = os.path.join(repo_root, "output", stamp)
                os.makedirs(dest_root, exist_ok=True)
                for entry in os.listdir(tmp_output_dir):
                    src_path = os.path.join(tmp_output_dir, entry)
                    dst_path = os.path.join(dest_root, entry)
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_path, dst_path)
                state["asset_root"] = dest_root
            else:
                state["asset_root"] = None
        except Exception as persist_err:
            logger.warning(f"[jobs] Failed to persist activity assets: {persist_err}")
            state["asset_root"] = None

        # Persist detected events to asset_root for cross-worker results access
        try:
            if state.get("asset_root"):
                events_path = os.path.join(state["asset_root"], "events.json")
                with open(events_path, "w", encoding="utf-8") as f:
                    json.dump(all_events or [], f, ensure_ascii=False, indent=2)
        except Exception as _e:
            logger.warning(f"[jobs] Failed to write events.json: {_e}")

        # Post results to external API (non-blocking, errors don't fail the job)
        try:
            from app.services.external_api_service import post_cvvr_results
            trip_id = state.get("trip_id") or ""
            host_url = state.get("host_url")  # Get host URL from state
            if trip_id:
                api_result = post_cvvr_results(
                    trip_id=trip_id,
                    events=all_events,
                    job_id=jid,
                    host_url=host_url  # Pass host_url to construct fileUrl
                )
                # Store API result in state for potential debugging
                state["external_api_result"] = api_result
                if not api_result.get("success"):
                    logger.warning(
                        f"[jobs] External API call failed for job {jid}: {api_result.get('error')}"
                    )
        except Exception as api_err:
            logger.warning(f"[jobs] Failed to post results to external API for job {jid}: {api_err}")
            # Don't fail the job if external API fails
            state["external_api_error"] = str(api_err)

        elapsed = time.perf_counter() - t0
        state["processed"] = int(completed_expected)
        state["done"] = True
        state["events"] = all_events
        persist_state(jid, state)
        logger.info(f"[jobs] Parallel pipeline completed in {elapsed:.2f}s (sampled_frames={state['processed']}/{state['total']}, native_frames={total_frames}, events={len(all_events)})")
    except Exception as e:
        logger.exception(f"[jobs] run_job crashed jid={jid}: {e}")
        if state is None:
            state = {"processed": 0, "total": 0}
            JOBS[jid] = state
        state["error"] = str(e)
        state["done"] = True
        persist_state(jid, state)
    finally:
        try:
            shutil.rmtree(state.get("tmp_dir", ""))
        except Exception:
            pass


