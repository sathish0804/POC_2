from __future__ import annotations

import os
import shutil
import tempfile
import uuid
import threading
from typing import Dict, Any, List, Tuple
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Response as FastAPIResponse
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import mimetypes
from loguru import logger
import json
from pathlib import Path
import subprocess
import math
import time
import multiprocessing as mp
import concurrent.futures as cf
from app.boot import get_pool
import cv2
from app.utils.video_utils import get_expected_sampled_frames
from app.models.job_models import JobResponse, JobProgress, JobStatus, JobResults

router = APIRouter()
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
    logger.info(f"[FastAPI] _count_total_frames: {video_path}, total={total}")
    if total <= 0:
        total = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            total += 1
    cap.release()
    logger.info(f"[FastAPI] _count_total_frames: {video_path}, total={total}")
    return total


def _split_frame_ranges(total_frames: int, num_processes: int) -> List[Tuple[int, int]]:
    """Evenly partition [0, total_frames) into num_processes contiguous ranges."""
    if num_processes <= 0:
        return [(0, total_frames)]
    logger.info(f"[FastAPI] _split_frame_ranges:, total_frames={total_frames}, num_processes={num_processes}")
    base = total_frames // num_processes
    remainder = total_frames % num_processes
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(num_processes):
        count = base + (1 if i < remainder else 0)
        end = start + count
        ranges.append((start, end))
        start = end
    logger.info(f"[FastAPI] _split_frame_ranges: , ranges={ranges}")
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

def _expected_sampled_in_range(video_path: str, sample_fps: float, start_frame: int, end_frame: int) -> int:
    """Compute expected sampled frames in [start_frame, end_frame) using same logic as video_utils."""
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


def _run_job(jid: str) -> None:
    state = JOBS.get(jid)
    logger.info(f"[FastAPI] _run_job start jid={jid}, has_state={bool(state)}")
    try:
        video_path = state["video_path"]
        num_processes = max(1, min(int(os.getenv("POOL_PROCS", "6")), (mp.cpu_count() or 1)))
        # Use expected sampled frames for progress denominator
        try:
            total_sampled = int(get_expected_sampled_frames(video_path, float(os.getenv("SAMPLE_FPS", "0.5"))))
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
                logger.info(f"[FastAPI] _run_job: reset cwd to {repo_root}")
            except Exception as cwd_err:
                logger.warning(f"[FastAPI] _run_job: failed to reset cwd: {cwd_err}")
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
            yolo_weights="yolo11s.pt",
            sample_fps=float(os.getenv("SAMPLE_FPS", "0.5")),
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

        logger.info(f"[FastAPI] Job {jid} config: yolo_weights={pipeline_cfg['yolo_weights']}, sample_fps={pipeline_cfg['sample_fps']}, native_frames={total_frames}")

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
            _expected_sampled_in_range(video_path, float(pipeline_cfg.get("sample_fps", 1.0)), s, e)
            for (s, e) in ranges
        ]
        total_expected = sum(per_range_expected)
        state["total"] = int(total_expected or state.get("total", 0))
        _persist_state(jid, state)

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
            logger.warning(f"[FastAPI] Failed to persist activity assets: {persist_err}")
            state["asset_root"] = None

        # Persist detected events to asset_root for cross-worker results access
        try:
            if state.get("asset_root"):
                events_path = os.path.join(state["asset_root"], "events.json")
                with open(events_path, "w", encoding="utf-8") as f:
                    json.dump(all_events or [], f, ensure_ascii=False, indent=2)
        except Exception as _e:
            logger.warning(f"[FastAPI] Failed to write events.json: {_e}")

        elapsed = time.perf_counter() - t0
        state["processed"] = int(completed_expected)
        state["done"] = True
        state["events"] = all_events
        _persist_state(jid, state)
        logger.info(f"[FastAPI] Parallel pipeline completed in {elapsed:.2f}s (yolo_weights={pipeline_cfg['yolo_weights']}, sample_fps={pipeline_cfg['sample_fps']}, sampled_frames={state['processed']}/{state['total']}, native_frames={total_frames}, events={len(all_events)})")
    except Exception as e:
        logger.exception(f"[FastAPI] _run_job crashed jid={jid}: {e}")
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

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the landing page to upload a CVVR file and Trip ID."""
    templates = request.app.state.templates
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/start", response_class=HTMLResponse)
async def start(
    request: Request,
    trip_id: str = Form(...),
    cvvr_file: UploadFile = File(...)
):
    """Create a processing job for the uploaded video and start a background worker."""
    if not trip_id.strip():
        raise HTTPException(status_code=400, detail="tripId is required")
    
    if not cvvr_file.filename:
        raise HTTPException(status_code=400, detail="cvvrFile is required")

    tmp_dir = tempfile.mkdtemp(prefix="upload_")
    suffix = os.path.splitext(cvvr_file.filename)[1] or ".mp4"
    video_path = os.path.join(tmp_dir, f"input{suffix}")
    
    # Save uploaded file
    with open(video_path, "wb") as buffer:
        content = await cvvr_file.read()
        buffer.write(content)
    
    logger.info(f"[FastAPI] Uploaded video saved to {video_path}")

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
        pref_total = int(get_expected_sampled_frames(video_path, float(os.getenv("SAMPLE_FPS", "0.5"))))
        logger.info(f"[FastAPI] start: pref_total={pref_total}")
        if pref_total <= 0:
            dur = _ffprobe_duration_seconds(video_path)
            pref_total = int(math.ceil(dur)) if dur > 0 else 0
        JOBS[job_id]["total"] = max(0, pref_total)
    except Exception:
        JOBS[job_id]["total"] = 0
    _persist_state(job_id, JOBS[job_id])

    t = threading.Thread(target=_run_job, args=(job_id,), daemon=False)
    t.start()
    logger.info(f"[FastAPI] Started background job thread for {job_id}, ident={t.ident}")
    
    # Redirect to job page
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/job/{job_id}", status_code=302)


@router.get("/job/{job_id}", response_class=HTMLResponse)
async def job(request: Request, job_id: str):
    """Render job page which polls progress for the given job id."""
    # Allow page to load even if in-memory state is missing; progress polling relies on persisted state
    templates = request.app.state.templates
    return templates.TemplateResponse("job.html", {"request": request, "job_id": job_id})


@router.get("/progress/{job_id}", response_model=JobProgress)
async def progress(job_id: str):
    """Return JSON progress for a job from the persisted lightweight state."""
    persisted = _load_state(job_id)
    if not persisted:
        time.sleep(0.05)
        persisted = _load_state(job_id)
        if not persisted:
            return JobProgress(
                processed=0,
                total=0,
                done=False,
                error=None,
                not_found=True
            )

    return JobProgress(
        processed=int(persisted.get("processed", 0)),
        total=int(persisted.get("total", 0)),
        done=bool(persisted.get("done", False)),
        error=persisted.get("error")
    )


@router.get("/results/{job_id}", response_class=HTMLResponse)
async def results(request: Request, job_id: str, page: int = 1, page_size: int = 25):
    """Render paginated results for detected events of the given job."""
    # Try in-memory first; fall back to persisted state so this works across workers
    state = JOBS.get(job_id)
    if not state:
        persisted = _load_state(job_id)
        if not persisted:
            raise HTTPException(status_code=404, detail="Invalid job id")
        state = persisted
    if state.get("error"):
        raise HTTPException(status_code=500, detail=f"Processing failed: {state['error']}")
    
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

    page = max(1, page)
    page_size = max(1, min(100, page_size))

    total = len(events)
    total_pages = max(1, (total + page_size - 1) // page_size)
    if page > total_pages:
        page = total_pages
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    paged_events = events[start:end]

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "events": paged_events,
            "trip_id": trip_id,
            "job_id": job_id,
            "page": page,
            "page_size": page_size,
            "total": total,
            "start": start,
            "end": end,
            "total_pages": total_pages,
        }
    )


@router.get("/media/{job_id}/{filename:path}")
async def media(request: Request, job_id: str, filename: str):
    """Serve media files from the job's asset root; supports HTTP range for MP4 streaming."""
    # Support cross-worker access by falling back to persisted state
    state = JOBS.get(job_id) or _load_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="not_found")
    root = state.get("asset_root")
    if not root:
        raise HTTPException(status_code=404, detail="no_assets")

    # Resolve and validate path
    abs_root = os.path.abspath(root)
    file_path = os.path.abspath(os.path.join(abs_root, filename))
    try:
        # commonpath raises on different drives; guard with try
        if os.path.commonpath([abs_root, file_path]) != abs_root:
            raise HTTPException(status_code=404, detail="file_missing")
    except Exception:
        raise HTTPException(status_code=404, detail="file_missing")
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="file_missing")

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
                
                def iter_file():
                    with open(file_path, "rb") as f:
                        f.seek(start)
                        yield f.read(length)
                
                headers = {
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(length)
                }
                return StreamingResponse(iter_file(), status_code=206, media_type=mime, headers=headers)
            except Exception:
                pass  # fall back to full file
        
        def iter_file():
            with open(file_path, "rb") as f:
                yield from f
        
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size)
        }
        return StreamingResponse(iter_file(), media_type=mime, headers=headers)

    # Images and others
    return FileResponse(file_path)


# --------------------- JSON API endpoints for external UI ---------------------

@router.post("/api/jobs", response_model=JobResponse)
async def api_create_job(
    request: Request,
    trip_id: str = Form(...),
    cvvr_file: UploadFile = File(...)
):
    """Create a job via API and start background processing; returns endpoints for polling and results."""
    if not trip_id.strip():
        raise HTTPException(status_code=400, detail="tripId is required")
    if not cvvr_file.filename:
        raise HTTPException(status_code=400, detail="cvvrFile is required")

    tmp_dir = tempfile.mkdtemp(prefix="upload_")
    suffix = os.path.splitext(cvvr_file.filename)[1] or ".mp4"
    video_path = os.path.join(tmp_dir, f"input{suffix}")
    
    # Save uploaded file
    with open(video_path, "wb") as buffer:
        content = await cvvr_file.read()
        buffer.write(content)
    
    logger.info(f"[API] Uploaded video saved to {video_path}")

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
        pref_total = int(get_expected_sampled_frames(video_path, 1))
        if pref_total <= 0:
            dur = _ffprobe_duration_seconds(video_path)
            pref_total = int(math.ceil(dur)) if dur > 0 else 0
        JOBS[job_id]["total"] = max(0, pref_total)
    except Exception:
        JOBS[job_id]["total"] = 0
    _persist_state(job_id, JOBS[job_id])

    t = threading.Thread(target=_run_job, args=(job_id,), daemon=False)
    t.start()
    logger.info(f"[API] Started background job thread for {job_id}, ident={t.ident}")

    base_url = str(request.base_url).rstrip("/")
    return JobResponse(
        job_id=job_id,
        status_url=f"{base_url}/api/jobs/{job_id}",
        progress_url=f"{base_url}/api/jobs/{job_id}/progress",
        results_url=f"{base_url}/api/jobs/{job_id}/results",
        media_url_prefix=f"{base_url}/api/jobs/{job_id}/media"
    )


@router.get("/api/jobs/{job_id}", response_model=JobStatus)
async def api_get_job(job_id: str):
    """Return job status summary as JSON (processed/total/percent/done/error)."""
    state = JOBS.get(job_id) or _load_state(job_id) or {
        "processed": 0, "total": 0, "done": False, "error": None, "trip_id": None
    }
    processed = int(state.get("processed", 0))
    total = max(0, int(state.get("total", 0)))
    done = bool(state.get("done", False))
    error = state.get("error")
    percent = (processed / total * 100.0) if total > 0 else 0.0
    return JobStatus(
        job_id=job_id,
        processed=processed,
        total=total,
        percent=round(percent, 2),
        done=done,
        error=error
    )


@router.get("/api/jobs/{job_id}/progress", response_model=JobProgress)
async def api_progress(job_id: str):
    """Return job progress JSON for API clients."""
    persisted = _load_state(job_id)
    if not persisted:
        time.sleep(0.05)
        persisted = _load_state(job_id)
        if not persisted:
            return JobProgress(
                processed=0,
                total=0,
                done=False,
                error=None,
                not_found=True
            )

    return JobProgress(
        processed=int(persisted.get("processed", 0)),
        total=int(persisted.get("total", 0)),
        done=bool(persisted.get("done", False)),
        error=persisted.get("error")
    )


@router.get("/api/jobs/{job_id}/results", response_model=JobResults)
async def api_results(request: Request, job_id: str, page: int = 1, page_size: int = 25):
    """Return paginated events as JSON with media URLs for the given job id."""
    state = JOBS.get(job_id) or _load_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="invalid_job")
    if state.get("error"):
        raise HTTPException(status_code=500, detail=state["error"])

    events = state.get("events") or []
    trip_id = state.get("trip_id") or ""

    if (not events) and state.get("asset_root"):
        try:
            events_path = os.path.join(state["asset_root"], "events.json")
            with open(events_path, "r", encoding="utf-8") as f:
                events = json.load(f) or []
        except Exception:
            events = []

    page = max(1, page)
    page_size = max(1, min(100, page_size))

    total = len(events)
    total_pages = max(1, (total + page_size - 1) // page_size)
    if page > total_pages:
        page = total_pages
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    paged_events = events[start:end]

    base_url = str(request.base_url).rstrip("/")
    media_prefix = f"{base_url}/api/jobs/{job_id}/media"
    events_with_urls = []
    for e in paged_events:
        try:
            e_copy = dict(e)
        except Exception:
            # Fallback: if event is a pydantic model, try model_dump
            try:
                e_copy = e.model_dump()
            except Exception:
                e_copy = e
        img = e_copy.get("activityImage")
        clip = e_copy.get("activityClip")
        if img:
            e_copy["activityImageUrl"] = f"{media_prefix}/{img}"
        if clip:
            e_copy["activityClipUrl"] = f"{media_prefix}/{clip}"
        events_with_urls.append(e_copy)

    return JobResults(
        job_id=job_id,
        trip_id=trip_id,
        events=events_with_urls,
        page=page,
        page_size=page_size,
        total=total,
        start=start,
        end=end,
        total_pages=total_pages
    )


@router.get("/api/jobs/{job_id}/media/{filename:path}")
async def api_media(request: Request, job_id: str, filename: str):
    """Serve media file for the given job id via API."""
    return await media(request, job_id, filename)

