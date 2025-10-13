from __future__ import annotations

import os
import shutil
import tempfile
import uuid
import threading
from typing import Dict, Any, List

from flask import Blueprint, current_app, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, Response
import mimetypes
from app.models.activity_event import ActivityEvent
from loguru import logger


ui_bp = Blueprint("ui", __name__)
JOBS: Dict[str, Dict[str, Any]] = {}


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

    def _run_job(jid: str) -> None:
        state = JOBS.get(jid)
        if not state:
            return
        try:
            # Lazily import heavy pipeline to avoid expensive module imports during app startup
            from app.services.pipeline_service import ActivityPipeline
            pipeline = ActivityPipeline(
                trip_id=state["trip_id"],
                crew_name="demo",
                crew_id="1",
                crew_role=1,
                yolo_weights="yolov8l.pt",
                sample_fps=1,
                enable_ocr=False,
                verbose=False,
                max_frames=0,
                use_advanced_sleep=True,
                sleep_min_duration=10.0,
                sleep_micro_max_min=0.25,
                save_debug_overlays=True,
            )

            def _progress(ev: Dict[str, Any]):
                state["processed"] = int(ev.get("processed", state.get("processed", 0)))
                state["total"] = int(ev.get("total", state.get("total", 0)))

            events: List[ActivityEvent] = pipeline.process_video(state["video_path"], progress_cb=_progress)

            try:
                tmp_output_dir = os.path.join(os.path.dirname(state["video_path"]), "output")
                if os.path.isdir(tmp_output_dir):
                    from datetime import datetime
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Compute repo root from file path to avoid needing app context in thread
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
                    logger.info(f"[Flask] Persisted activity assets to {dest_root}")
                    state["asset_root"] = dest_root
                else:
                    state["asset_root"] = None
            except Exception as persist_err:
                logger.warning(f"[Flask] Failed to persist activity assets: {persist_err}")
                state["asset_root"] = None

            logger.info(f"[Flask] Found {len(events)} events")
            state["events"] = [e.model_dump() for e in events]
            state["done"] = True
        except Exception as e:
            state["error"] = str(e)
            state["done"] = True
        finally:
            try:
                shutil.rmtree(state.get("tmp_dir", ""))
            except Exception:
                pass

    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return redirect(url_for("ui.job", job_id=job_id))


@ui_bp.get("/job/<job_id>")
def job(job_id: str):
    if job_id not in JOBS:
        flash("Invalid job id", "error")
        return redirect(url_for("ui.index"))
    return render_template("job.html", job_id=job_id)


@ui_bp.get("/progress/<job_id>")
def progress(job_id: str):
    state = JOBS.get(job_id)
    if not state:
        return jsonify({"error": "not_found"}), 404
    return jsonify({
        "processed": int(state.get("processed", 0)),
        "total": int(state.get("total", 0)),
        "done": bool(state.get("done", False)),
        "error": state.get("error"),
    })


@ui_bp.get("/results/<job_id>")
def results(job_id: str):
    state = JOBS.get(job_id)
    if not state:
        flash("Invalid job id", "error")
        return redirect(url_for("ui.index"))
    if state.get("error"):
        flash(f"Processing failed: {state['error']}", "error")
        return redirect(url_for("ui.index"))
    events = state.get("events") or []
    trip_id = state.get("trip_id") or ""

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
    state = JOBS.get(job_id)
    if not state:
        return jsonify({"error": "not_found"}), 404
    root = state.get("asset_root")
    if not root:
        return jsonify({"error": "no_assets"}), 404

    # Resolve and validate path
    abs_root = os.path.abspath(root)
    file_path = os.path.abspath(os.path.join(abs_root, filename))
    if not file_path.startswith(abs_root + os.sep) and file_path != abs_root:
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


