from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any, Dict, List
from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from loguru import logger

from app.services.job_service import JOBS, persist_state, load_state, ffprobe_duration_seconds, run_job
from app.config import settings


router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("")
async def create_job(request: Request, tripId: str = Form(...), cvvrFile: UploadFile = File(...)) -> Dict[str, Any]:
    trip_id = (tripId or "").strip()
    if not trip_id:
        raise HTTPException(status_code=400, detail="tripId is required")
    if (cvvrFile is None) or (not getattr(cvvrFile, "filename", None)):
        raise HTTPException(status_code=400, detail="cvvrFile is required")

    try:
        tmp_dir = tempfile.mkdtemp(prefix="upload_")
        suffix = os.path.splitext(cvvrFile.filename or "")[1] or ".mp4"
        video_path = os.path.join(tmp_dir, f"input{suffix}")
        contents = await cvvrFile.read()
        with open(video_path, "wb") as f:
            f.write(contents)
        logger.info(f"[API] Uploaded video saved to {video_path}")
    finally:
        try:
            await cvvrFile.close()
        except Exception:
            pass

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
        from app.utils.video_utils import get_expected_sampled_frames
        pref_total = int(get_expected_sampled_frames(video_path, float(settings.sample_fps or 0.5)))
        if pref_total <= 0:
            dur = ffprobe_duration_seconds(video_path)
            pref_total = int(dur) if dur > 0 else 0
    except Exception:
        dur = ffprobe_duration_seconds(video_path)
        pref_total = int(dur) if dur > 0 else 0
    JOBS[job_id]["total"] = max(0, pref_total)
    persist_state(job_id, JOBS[job_id])

    import threading
    t = threading.Thread(target=run_job, args=(job_id,), daemon=False)
    t.start()
    logger.info(f"[API] Started background job thread for {job_id}, ident={t.ident}")

    try:
        host_url = str(request.url).split(request.url.path)[0]
    except Exception:
        host_url = str(request.base_url).rstrip("/")

    return {
        "job_id": job_id,
        "status_url": f"{host_url}/api/jobs/{job_id}",
        "progress_url": f"{host_url}/api/jobs/{job_id}/progress",
        "results_url": f"{host_url}/api/jobs/{job_id}/results",
    }


@router.get("/server-videos")
async def list_server_videos() -> Dict[str, Any]:
    logger.info(f"[API] Listing server videos from {settings.video_input_dir}")
    base_dir = (settings.video_input_dir ).strip()
    allowed_exts = {".mp4", ".mov", ".mkv", ".avi"}
    videos: list[str] = []
    logger.info(f"[API] Found {len(videos)} videos in {base_dir}")
    try:
        if base_dir and os.path.isdir(base_dir):
            videos = sorted(
                f for f in os.listdir(base_dir)
                if os.path.splitext(f)[1].lower() in allowed_exts
            )
    except Exception:
        videos = []
    logger.info(f"[API] Returning {len(videos)} videos")
    return {"videos": videos}


@router.post("/start")
async def start_server_video(tripId: str = Form(...), videoName: str = Form(...)) -> Dict[str, Any]:
    trip_id = (tripId or "").strip()
    video_name = (videoName or "").strip()
    if not trip_id:
        raise HTTPException(status_code=400, detail="tripId is required")
    if not video_name:
        raise HTTPException(status_code=400, detail="videoName is required")

    base_dir = str(settings.video_input_dir or "").strip()
    if not base_dir or not os.path.isdir(base_dir):
        raise HTTPException(status_code=400, detail="VIDEO_INPUT_DIR is not configured or does not exist")

    allowed_exts = {".mp4", ".mov", ".mkv", ".avi"}
    if os.path.splitext(video_name)[1].lower() not in allowed_exts:
        raise HTTPException(status_code=400, detail="Unsupported video type")

    full_path = os.path.realpath(os.path.join(base_dir, video_name))
    base_real = os.path.realpath(base_dir)
    if not full_path.startswith(base_real + os.sep):
        raise HTTPException(status_code=400, detail="Invalid video selection")
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="Selected video not found")

    tmp_dir = tempfile.mkdtemp(prefix="job_")
    logger.info(f"[API] Selected server video at {full_path}")

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "trip_id": trip_id,
        "tmp_dir": tmp_dir,
        "video_path": full_path,
        "processed": 0,
        "total": 0,
        "done": False,
        "error": None,
        "events": None,
        "asset_root": None,
    }
    try:
        from app.utils.video_utils import get_expected_sampled_frames
        pref_total = int(get_expected_sampled_frames(full_path, float(settings.sample_fps or 0.5)))
        if pref_total <= 0:
            dur = ffprobe_duration_seconds(full_path)
            pref_total = int(dur) if dur > 0 else 0
        JOBS[job_id]["total"] = max(0, pref_total)
    except Exception:
        JOBS[job_id]["total"] = 0
    persist_state(job_id, JOBS[job_id])

    import threading
    t = threading.Thread(target=run_job, args=(job_id,), daemon=False)
    t.start()
    logger.info(f"[API] Started background job thread for {job_id}, ident={t.ident}")

    return {"job_id": job_id}


@router.get("/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    state = JOBS.get(job_id) or load_state(job_id) or {
        "processed": 0, "total": 0, "done": False, "error": None, "trip_id": None
    }
    processed = int(state.get("processed", 0))
    total = max(0, int(state.get("total", 0)))
    done = bool(state.get("done", False))
    error = state.get("error")
    percent = (processed / total * 100.0) if total > 0 else 0.0
    return {
        "job_id": job_id,
        "processed": processed,
        "total": total,
        "percent": round(percent, 2),
        "done": done,
        "error": error,
    }


@router.get("/{job_id}/progress")
async def progress(job_id: str) -> Dict[str, Any]:
    persisted = load_state(job_id)
    if not persisted:
        import time as _time
        _time.sleep(0.05)
        persisted = load_state(job_id)
        if not persisted:
            return {"processed": 0, "total": 0, "done": False, "error": None, "notFound": True}
    return {
        "processed": int(persisted.get("processed", 0)),
        "total": int(persisted.get("total", 0)),
        "done": bool(persisted.get("done", False)),
        "error": persisted.get("error"),
    }


@router.get("/{job_id}/results")
async def results(job_id: str, request: Request, page: int = 1, page_size: int = 25) -> Dict[str, Any]:
    state = JOBS.get(job_id) or load_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="invalid_job")
    if state.get("error"):
        raise HTTPException(status_code=500, detail=str(state["error"]))

    events = state.get("events") or []
    trip_id = state.get("trip_id") or ""

    if (not events) and state.get("asset_root"):
        try:
            import json
            events_path = os.path.join(state["asset_root"], "events.json")
            with open(events_path, "r", encoding="utf-8") as f:
                events = json.load(f) or []
        except Exception:
            events = []

    page = max(1, int(page or 1))
    page_size = max(1, min(100, int(page_size or 25)))
    total = len(events)
    total_pages = max(1, (total + page_size - 1) // page_size)
    if page > total_pages:
        page = total_pages
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    paged_events = events[start:end]

    try:
        host_url = str(request.url).split(request.url.path)[0]
    except Exception:
        host_url = str(request.base_url).rstrip("/")
    media_prefix = f"{host_url}/api/jobs/{job_id}/media"

    events_with_urls: List[Dict[str, Any]] = []
    for e in paged_events:
        try:
            e_copy = dict(e)
        except Exception:
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

    return {
        "job_id": job_id,
        "trip_id": trip_id,
        "events": events_with_urls,
        "page": page,
        "page_size": page_size,
        "total": total,
        "start": start,
        "end": end,
        "total_pages": total_pages,
    }


@router.get("/{job_id}/media/{filename:path}")
async def media(job_id: str, filename: str, request: Request) -> Response:
    state = JOBS.get(job_id) or load_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="not_found")
    root = state.get("asset_root")
    if not root:
        raise HTTPException(status_code=404, detail="no_assets")

    abs_root = os.path.abspath(root)
    file_path = os.path.abspath(os.path.join(abs_root, filename))
    try:
        if os.path.commonpath([abs_root, file_path]) != abs_root:
            raise HTTPException(status_code=404, detail="file_missing")
    except Exception:
        raise HTTPException(status_code=404, detail="file_missing")
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="file_missing")

    import mimetypes as _m
    mime, _ = _m.guess_type(file_path)
    if file_path.lower().endswith(".mp4"):
        mime = "video/mp4"

    range_header = request.headers.get("range") or request.headers.get("Range")
    file_size = os.path.getsize(file_path)
    if mime == "video/mp4" and range_header:
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
            resp = Response(content=data, status_code=206, media_type=mime)
            resp.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            resp.headers["Accept-Ranges"] = "bytes"
            resp.headers["Content-Length"] = str(length)
            return resp
        except Exception:
            pass

    with open(file_path, "rb") as f:
        data = f.read()
    resp = Response(content=data, status_code=200, media_type=mime or "application/octet-stream")
    if mime == "video/mp4":
        resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Content-Length"] = str(file_size)
    return resp


