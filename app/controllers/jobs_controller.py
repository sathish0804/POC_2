from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
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
        
        # Stream the file in chunks instead of loading it all into memory
        CHUNK_SIZE = 1024 * 1024  # 1MB chunks
        total_size = 0
        
        # Create job ID early for upload progress tracking
        job_id = uuid.uuid4().hex
        
        # Get host URL for constructing media URLs
        try:
            host_url = str(request.url).split(request.url.path)[0]
        except Exception:
            host_url = str(request.base_url).rstrip("/")
        
        JOBS[job_id] = {
            "trip_id": trip_id,
            "tmp_dir": tmp_dir,
            "video_path": video_path,
            "processed": 0,
            "total": 0,
            "upload_progress": 0,  # Track upload progress
            "upload_total": 0,     # Total upload size (if known)
            "done": False,
            "error": None,
            "events": None,
            "asset_root": None,
            "host_url": host_url,  # Store host URL for external API calls
        }
        persist_state(job_id, JOBS[job_id])
        
        # Try to get content length if available
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                JOBS[job_id]["upload_total"] = int(content_length)
            except (ValueError, TypeError):
                pass
        
        with open(video_path, "wb") as f:
            while True:
                chunk = await cvvrFile.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                total_size += len(chunk)
                
                # Update upload progress
                JOBS[job_id]["upload_progress"] = total_size
                # Only persist state occasionally to avoid excessive I/O
                if total_size % (10 * CHUNK_SIZE) == 0:  # Every 10MB
                    persist_state(job_id, JOBS[job_id])
                
        logger.info(f"[API] Uploaded video saved to {video_path} (size: {total_size / (1024 * 1024):.2f} MB)")
        
        # Mark upload as complete
        JOBS[job_id]["upload_progress"] = total_size
    finally:
        try:
            await cvvrFile.close()
        except Exception:
            pass

    # Job ID already created during upload
    # Just update with any additional fields if needed
    JOBS[job_id].update({
        "upload_complete": True  # Mark upload as fully complete
    })

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

    # Get host URL from state (already stored during job creation)
    host_url = JOBS[job_id].get("host_url", "")
    if not host_url:
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
async def start_server_video(request: Request, tripId: str = Form(...), videoName: str = Form(...)) -> Dict[str, Any]:
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

    # Get host URL for constructing media URLs
    try:
        host_url = str(request.url).split(request.url.path)[0]
    except Exception:
        host_url = str(request.base_url).rstrip("/")

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
        "host_url": host_url,  # Store host URL for external API calls
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
    
    # For upload progress, check in-memory state as it's not persisted
    state = JOBS.get(job_id, {})
    upload_progress = int(state.get("upload_progress", 0))
    upload_total = int(state.get("upload_total", 0))
    upload_complete = bool(state.get("upload_complete", False))
    
    # Calculate upload percentage if we have a total
    upload_percent = 0
    if upload_total > 0 and upload_progress > 0:
        upload_percent = min(100, round((upload_progress / upload_total) * 100, 1))
    elif upload_complete:
        upload_percent = 100
    
    return {
        "processed": int(persisted.get("processed", 0)),
        "total": int(persisted.get("total", 0)),
        "done": bool(persisted.get("done", False)),
        "error": persisted.get("error"),
        "upload_progress": upload_progress,
        "upload_total": upload_total,
        "upload_complete": upload_complete,
        "upload_percent": upload_percent,
    }


def _transform_seconds_to_iso_time(seconds_str: str, base_date: Optional[str] = None, base_time: Optional[str] = None) -> str:
    """Convert seconds timestamp to ISO format datetime string."""
    try:
        base_dt = None
        
        # Try to parse base date and time if available
        if base_date and base_time:
            try:
                # Try common date formats
                date_formats = [
                    "%Y-%m-%d %H:%M:%S",
                    "%d-%m-%Y %H:%M:%S",
                    "%Y/%m/%d %H:%M:%S",
                    "%d/%m/%Y %H:%M:%S",
                ]
                for fmt in date_formats:
                    try:
                        base_dt = datetime.strptime(f"{base_date} {base_time}", fmt)
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
        
        # If no base datetime, use current time
        if base_dt is None:
            base_dt = datetime.now()
        
        # Add seconds offset
        seconds = float(seconds_str)
        result_dt = base_dt + timedelta(seconds=seconds)
        return result_dt.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        # Fallback: use current time
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _format_duration(duration_str: str) -> str:
    """Format duration string to HH:MM:SS format."""
    try:
        # Try parsing as HH:MM:SS first
        if ":" in duration_str:
            parts = duration_str.split(":")
            if len(parts) == 3:
                return duration_str
        
        # Try parsing as seconds
        seconds = float(duration_str)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    except Exception:
        # If already formatted or invalid, return as is or default
        return duration_str if duration_str else "00:00:00"


def _event_to_violation(
    event: Dict[str, Any],
    trip_id: str,
    host_url: str,
    job_id: str
) -> Dict[str, Any]:
    """
    Convert a single event to a violation object.
    
    Args:
        event: Single activity event dictionary
        trip_id: The trip ID
        host_url: Base URL for constructing media URLs
        job_id: Job ID for constructing media URLs
        
    Returns:
        Single violation object matching the required format
    """
    # Get object type
    object_type = event.get("objectType", "")
    
    # Get start and end times
    start_ts = event.get("activityStartTime", "")
    end_ts = event.get("activityEndTime") or start_ts
    base_date = event.get("date", "")
    base_time = event.get("time", "")
    
    if start_ts:
        start_time = _transform_seconds_to_iso_time(
            str(start_ts),
            base_date,
            base_time
        )
    else:
        start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    
    if end_ts:
        end_time = _transform_seconds_to_iso_time(
            str(end_ts),
            base_date,
            base_time
        )
    else:
        end_time = start_time
    
    # Get description
    description = event.get("des", "") or "Activity violation detected"
    
    # Get activity type
    activity_type = event.get("activityType", 1)
    
    # Get filename
    filename = event.get("filename", "")
    if not filename and event.get("fileUrl"):
        filename = os.path.basename(event.get("fileUrl", ""))
    
    # Get file duration
    file_duration = event.get("fileDuration", "00:00:00")
    file_duration = _format_duration(file_duration)
    
    # Get crew name
    crew_name = event.get("crewName", "")
    
    # Get activity clip URL as fileUrl
    media_prefix = f"{host_url}/api/jobs/{job_id}/media"
    file_url = ""
    clip = event.get("activityClip")
    if clip:
        file_url = f"{media_prefix}/{clip}"
    
    # Build payload
    payload = {
        "tripId": trip_id,
        "type": activity_type,
        "startTime": start_time,
        "endTime": end_time,
        "remarks": "Violation detected during trip processing",
        "reason": "Automated detection",
        "description": description,
        "objectTypes": object_type,
        "fileName": filename,
        "fileDuration": file_duration,
        "crewName": crew_name,
        "fileType": 2,  # Default file type (2 = video)
        "fileUrl": file_url,
        "createdDate": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "createdBy": "system",
        "status": 1,  # Default status (1 = active/complete)
    }
    
    return payload


@router.get("/{job_id}/results")
async def results(job_id: str, request: Request) -> List[Dict[str, Any]]:
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

    try:
        host_url = str(request.url).split(request.url.path)[0]
    except Exception:
        host_url = str(request.base_url).rstrip("/")

    # Convert each event to a violation object (one violation per event)
    if not events:
        return []

    raw_violations = []
    for event in events:
        violation = _event_to_violation(
            event=event,
            trip_id=trip_id,
            host_url=host_url,
            job_id=job_id
        )
        raw_violations.append(violation)

    # Deduplicate identical violations (same type/object/start/end/file)
    dedup_map: Dict[tuple, Dict[str, Any]] = {}
    for v in raw_violations:
        key = (
            v.get("type"),
            v.get("objectTypes"),
            v.get("startTime"),
            v.get("endTime"),
            v.get("fileUrl"),
        )
        if key not in dedup_map:
            dedup_map[key] = v
        # else ignore exact duplicate

    violations = list(dedup_map.values())
    # Sort violations by startTime (earliest first)
    violations.sort(key=lambda v: v.get("startTime", ""))

    return violations


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


