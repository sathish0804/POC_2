from __future__ import annotations

import os
import shutil
import tempfile
import uuid
import threading
from typing import Dict, Any, List, Tuple
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, Response
import mimetypes
from loguru import logger
import json
from pathlib import Path
import subprocess
import math
import time
import multiprocessing as mp
import concurrent.futures as cf
import cv2
from app.utils.video_utils import get_expected_sampled_frames


ui_bp = Blueprint("ui", __name__)
JOBS: Dict[str, Dict[str, Any]] = {}

# Persist lightweight job state to survive worker restarts
STATE_DIR = Path(__file__).resolve().parents[2] / "output" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)


def _state_path(job_id: str) -> Path:
    return STATE_DIR / f"{job_id}.json"


def _persist_state(job_id: str, state: Dict[str, Any]) -> None:
    # Ensure parent directory exists at write-time (defensive in case import-time mkdir didn't run)
    target_path = _state_path(job_id)
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
            },
            f,
        )
    tmp_path.replace(target_path)


def _load_state(job_id: str) -> Dict[str, Any] | None:
    path = _state_path(job_id)
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _ffprobe_duration_seconds(video_path: str) -> float:
    """Fast duration probe via ffprobe; returns seconds or 0.0 on failure."""
    try:
        # Using json output keeps parsing simple and robust
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


def _count_total_frames(video_path: str) -> int:
    """Count total frames using metadata, fall back to scan if needed."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    logger.info(f"[Flask] _count_total_frames: {video_path}, total={total}")
    if total <= 0:
        total = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            total += 1
    cap.release()
    logger.info(f"[Flask] _count_total_frames: {video_path}, total={total}")
    return total


def _split_frame_ranges(total_frames: int, num_processes: int) -> List[Tuple[int, int]]:
    """Evenly partition [0, total_frames) into num_processes contiguous ranges."""
    if num_processes <= 0:
        return [(0, total_frames)]
    logger.info(f"[Flask] _split_frame_ranges:, total_frames={total_frames}, num_processes={num_processes}")
    base = total_frames // num_processes
    remainder = total_frames % num_processes
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(num_processes):
        count = base + (1 if i < remainder else 0)
        end = start + count
        ranges.append((start, end))
        start = end
    logger.info(f"[Flask] _split_frame_ranges: , ranges={ranges}")
    return ranges


def _split_frame_ranges_by_seconds(video_path: str, chunk_seconds: float) -> List[Tuple[int, int]]:
    """
    Partition [0, total_frames) into contiguous ranges sized by chunk_seconds.
    Produces many small ranges to enable better load balancing across the pool.
    """
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
        total_frames = _count_total_frames(video_path)

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

def _expected_sampled_in_range(video_path: str, sample_fps: int, start_frame: int, end_frame: int) -> int:
    """Compute expected sampled frames in [start_frame, end_frame) using same logic as video_utils."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native_fps / max(1, sample_fps))))
    cap.release()
    lo = max(0, int(start_frame))
    hi = max(0, int(end_frame))
    if hi <= lo:
        return 0
    first = ((lo + step - 1) // step) * step
    if first >= hi:
        return 0
    return ((hi - 1 - first) // step) + 1


## Removed legacy frame-only worker; real pipeline worker is now a pure function for ProcessPool


def worker_run_range(vpath: str, start_f: int, end_f: int, pipeline_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pure, pickle-safe worker: initialize pipeline lazily and return serialized events for range."""
    import os
    try:
        import torch
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
        torch.set_num_interop_threads(int(os.getenv("TORCH_NUM_INTEROP_THREADS", "1")))
    except Exception:
        pass
    try:
        import cv2
        cv2.setNumThreads(int(os.getenv("OPENCV_NUM_THREADS", "1")))
        try:
            cv2.ocl.setUseOpenCL(False)
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

@ui_bp.get("/")
def index():
    return render_template("index.html")


@ui_bp.post("/start")
def start():
    trip_id = request.form.get("tripId", "").strip()
    file = request.files.get("cvvrFile")

    if not trip_id:
        flash("tripId is required", "error")
        return redirect(url_for("ui.index"))
    if file is None or not getattr(file, "filename", None):
        flash("cvvrFile is required", "error")
        return redirect(url_for("ui.index"))

    tmp_dir = tempfile.mkdtemp(prefix="upload_")
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    video_path = os.path.join(tmp_dir, f"input{suffix}")
    file.save(video_path)
    logger.info(f"[Flask] Uploaded video saved to {video_path}")

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "trip_id": trip_id,
        "tmp_dir": tmp_dir,
        "video_path": video_path,
        "processed": 0,
        "total": 0,
        "done": False,
        "error": None,
        "events": None,
        "asset_root": None,
    }
    try:
        # Prefill total for immediate UI feedback
        pref_total = int(get_expected_sampled_frames(video_path, 1))
        logger.info(f"[Flask] start: pref_total={pref_total}")
        if pref_total <= 0:
            dur = _ffprobe_duration_seconds(video_path)
            pref_total = int(math.ceil(dur)) if dur > 0 else 0
        JOBS[job_id]["total"] = max(0, pref_total)
    except Exception:
        JOBS[job_id]["total"] = 0
    _persist_state(job_id, JOBS[job_id])

    def _run_job(jid: str) -> None:
        state = JOBS.get(jid)
        logger.info(f"[Flask] _run_job start jid={jid}, has_state={bool(state)}")
        try:
            video_path = state["video_path"]
            num_processes = max(1, min(int(os.getenv("POOL_PROCS", "6")), (mp.cpu_count() or 1)))
            # Use expected sampled frames for progress denominator
            try:
                total_sampled = int(get_expected_sampled_frames(video_path, 1))
            except Exception:
                total_sampled = _count_total_frames(video_path)
            state["total"] = int(total_sampled)
            _persist_state(jid, state)
            total_frames = _count_total_frames(video_path)
            if total_frames <= 0:
                raise RuntimeError("Video has zero frames or cannot be read")

            # Ensure a valid working directory because some libs call Path.cwd() on import
            try:
                cur = os.getcwd()
                if not os.path.isdir(cur):
                    raise FileNotFoundError
            except Exception:
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                try:
                    os.chdir(repo_root)
                    logger.info(f"[Flask] _run_job: reset cwd to {repo_root}")
                except Exception as cwd_err:
                    logger.warning(f"[Flask] _run_job: failed to reset cwd: {cwd_err}")
                try:
                    os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", os.path.join(repo_root, "output", ".ultralytics"))
                except Exception:
                    pass
            # Initialize pipeline once to reuse config across processes
            from app.services.pipeline_service import ActivityPipeline
            pipeline_cfg = dict(
                trip_id=state["trip_id"],
                crew_name="demo",
                crew_id="1",
                crew_role=1,
                yolo_weights="yolov8n.pt",
                sample_fps=1,
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
                sleep_cfg_eye_closed_run_s=2.2,
                sleep_cfg_perclos_drowsy_thresh=0.35,
                sleep_cfg_perclos_sleep_thresh=0.75,
                sleep_cfg_head_pitch_down_deg=20.0,
                sleep_cfg_head_neutral_deg=12.0,
                sleep_cfg_hold_transition_s=1.0,
                sleep_cfg_recovery_hold_s=2.0,
                sleep_cfg_open_prob_closed_thresh=0.45,
                sleep_cfg_no_eye_head_down_deg=32.0,
            )

            chunk_seconds_env = os.getenv("CHUNK_SECONDS", "").strip()
            if chunk_seconds_env:
                try:
                    chunk_seconds = float(chunk_seconds_env)
                except Exception:
                    chunk_seconds = 6.0
            else:
                chunk_seconds = 6.0

            ranges = _split_frame_ranges_by_seconds(video_path, chunk_seconds)
            if not ranges:
                ranges = _split_frame_ranges(total_frames, num_processes)
            # Estimate expected sampled frames per range for progress aggregation
            per_range_expected = [
                _expected_sampled_in_range(video_path, int(pipeline_cfg.get("sample_fps", 1)), s, e)
                for (s, e) in ranges
            ]
            total_expected = sum(per_range_expected)
            state["total"] = int(total_expected or state.get("total", 0))
            _persist_state(jid, state)

            t0 = time.perf_counter()
            all_events: List[Dict[str, Any]] = []
            completed_expected = 0
            with cf.ProcessPoolExecutor(max_workers=num_processes, mp_context=mp.get_context("spawn")) as pool:
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
                        _persist_state(jid, state)

            # Persist output assets to repo output/<timestamp> like original flow
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
                logger.warning(f"[Flask] Failed to persist activity assets: {persist_err}")
                state["asset_root"] = None

            # Persist detected events to asset_root for cross-worker results access
            try:
                if state.get("asset_root"):
                    events_path = os.path.join(state["asset_root"], "events.json")
                    with open(events_path, "w", encoding="utf-8") as f:
                        json.dump(all_events or [], f, ensure_ascii=False, indent=2)
            except Exception as _e:
                logger.warning(f"[Flask] Failed to write events.json: {_e}")

            elapsed = time.perf_counter() - t0
            state["processed"] = int(completed_expected)
            state["done"] = True
            state["events"] = all_events
            _persist_state(jid, state)
            logger.info(f"[Flask] Parallel pipeline completed in {elapsed:.2f}s: {state['processed']}/{state['total']} (events={len(all_events)})")
        except Exception as e:
            logger.exception(f"[Flask] _run_job crashed jid={jid}: {e}")
            if state is None:
                state = {"processed": 0, "total": 0}
                JOBS[jid] = state
            state["error"] = str(e)
            state["done"] = True
            _persist_state(jid, state)
        finally:
            try:
                shutil.rmtree(state.get("tmp_dir", ""))
            except Exception:
                pass

    t = threading.Thread(target=_run_job, args=(job_id,), daemon=False)
    t.start()
    logger.info(f"[Flask] Started background job thread for {job_id}, ident={t.ident}")
    return redirect(url_for("ui.job", job_id=job_id))


@ui_bp.get("/job/<job_id>")
def job(job_id: str):
    # Allow page to load even if in-memory state is missing; progress polling relies on persisted state
    return render_template("job.html", job_id=job_id)


@ui_bp.get("/progress/<job_id>")
def progress(job_id: str):
    persisted = _load_state(job_id)
    if not persisted:
        time.sleep(0.05)
        persisted = _load_state(job_id)
        if not persisted:
            return jsonify({
                "processed": 0,
                "total": 0,
                "done": False,
                "error": None,
                "notFound": True,
            }), 200

    return jsonify({
        "processed": int(persisted.get("processed", 0)),
        "total": int(persisted.get("total", 0)),
        "done": bool(persisted.get("done", False)),
        "error": persisted.get("error"),
    })


@ui_bp.get("/results/<job_id>")
def results(job_id: str):
    # Try in-memory first; fall back to persisted state so this works across workers
    state = JOBS.get(job_id)
    if not state:
        persisted = _load_state(job_id)
        if not persisted:
            flash("Invalid job id", "error")
            return redirect(url_for("ui.index"))
        state = persisted
    if state.get("error"):
        flash(f"Processing failed: {state['error']}", "error")
        return redirect(url_for("ui.index"))
    events = state.get("events") or []
    trip_id = state.get("trip_id") or ""

    # Fallback: load events from persisted asset_root if in-memory is missing
    if (not events) and state.get("asset_root"):
        try:
            events_path = os.path.join(state["asset_root"], "events.json")
            with open(events_path, "r", encoding="utf-8") as f:
                events = json.load(f) or []
        except Exception:
            events = []

    try:
        page = max(1, int(request.args.get("page", 1)))
    except Exception:
        page = 1
    try:
        page_size = int(request.args.get("page_size", 25))
    except Exception:
        page_size = 25
    page_size = max(1, min(100, page_size))

    total = len(events)
    total_pages = max(1, (total + page_size - 1) // page_size)
    if page > total_pages:
        page = total_pages
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    paged_events = events[start:end]

    return render_template(
        "results.html",
        events=paged_events,
        trip_id=trip_id,
        job_id=job_id,
        page=page,
        page_size=page_size,
        total=total,
        start=start,
        end=end,
        total_pages=total_pages,
    )


@ui_bp.get("/media/<job_id>/<path:filename>")
def media(job_id: str, filename: str):
    # Support cross-worker access by falling back to persisted state
    state = JOBS.get(job_id) or _load_state(job_id)
    if not state:
        return jsonify({"error": "not_found"}), 404
    root = state.get("asset_root")
    if not root:
        return jsonify({"error": "no_assets"}), 404

    # Resolve and validate path
    abs_root = os.path.abspath(root)
    file_path = os.path.abspath(os.path.join(abs_root, filename))
    try:
        # commonpath raises on different drives; guard with try
        if os.path.commonpath([abs_root, file_path]) != abs_root:
            return jsonify({"error": "file_missing"}), 404
    except Exception:
        return jsonify({"error": "file_missing"}), 404
    if not os.path.isfile(file_path):
        return jsonify({"error": "file_missing"}), 404

    mime, _ = mimetypes.guess_type(file_path)
    # Fallback for common case where mimetypes may not detect MP4
    if file_path.lower().endswith(".mp4"):
        mime = "video/mp4"
    if mime == "video/mp4":
        # Support HTTP Range for MP4 streaming
        range_header = request.headers.get("Range")
        file_size = os.path.getsize(file_path)
        if range_header:
            try:
                start_str, end_str = range_header.replace("bytes=", "").split("-")
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
                start = max(0, start)
                end = min(file_size - 1, end)
                length = end - start + 1
                with open(file_path, "rb") as f:
                    f.seek(start)
                    data = f.read(length)
                resp = Response(data, 206, mimetype=mime, direct_passthrough=True)
                resp.headers.add("Content-Range", f"bytes {start}-{end}/{file_size}")
                resp.headers.add("Accept-Ranges", "bytes")
                resp.headers.add("Content-Length", str(length))
                return resp
            except Exception:
                pass  # fall back to full file
        with open(file_path, "rb") as f:
            data = f.read()
        resp = Response(data, 200, mimetype=mime, direct_passthrough=True)
        resp.headers.add("Accept-Ranges", "bytes")
        resp.headers.add("Content-Length", str(file_size))
        return resp

    # Images and others
    return send_from_directory(abs_root, os.path.relpath(file_path, abs_root))


