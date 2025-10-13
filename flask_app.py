from __future__ import annotations

import os
import shutil
import tempfile
from typing import List, Dict, Any
import threading
import uuid

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory

from controllers.pipeline import ActivityPipeline
from models.activity_event import ActivityEvent
from utils.logging_utils import configure_logging
from loguru import logger


JOBS: Dict[str, Dict[str, Any]] = {}


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "change-me")

    # Configure logging once per app startup
    log_path = configure_logging(verbose=False)
    logger.info(f"Flask UI startup. Logs at: {log_path}")

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/start")
    def start():
        trip_id = request.form.get("tripId", "").strip()
        file = request.files.get("cvvrFile")

        if not trip_id:
            flash("tripId is required", "error")
            return redirect(url_for("index"))
        if file is None or not getattr(file, "filename", None):
            flash("cvvrFile is required", "error")
            return redirect(url_for("index"))

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
        }

        def _run_job(jid: str) -> None:
            state = JOBS.get(jid)
            if not state:
                return
            try:
                pipeline = ActivityPipeline(
                    trip_id=state["trip_id"],
                    crew_name="demo",
                    crew_id="1",
                    crew_role=1,
                    yolo_weights="yolo11s.pt",
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
                    if ev.get("done"):
                        state["done"] = True

                events: List[ActivityEvent] = pipeline.process_video(state["video_path"], progress_cb=_progress)

                # Optional: persist output images similar to FastAPI handler
                try:
                    tmp_output_dir = os.path.join(os.path.dirname(state["video_path"]), "output")
                    if os.path.isdir(tmp_output_dir):
                        from datetime import datetime
                        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        dest_root = os.path.join(
                            "/Users/satishvanga/Documents/Vanga/POC_2/example_data", "output", stamp
                        )
                        os.makedirs(dest_root, exist_ok=True)
                        for entry in os.listdir(tmp_output_dir):
                            src_path = os.path.join(tmp_output_dir, entry)
                            dst_path = os.path.join(dest_root, entry)
                            if os.path.isdir(src_path):
                                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                            else:
                                shutil.copy2(src_path, dst_path)
                        logger.info(f"[Flask] Persisted activity images to {dest_root}")
                        state["asset_root"] = dest_root
                    else:
                        state["asset_root"] = None
                except Exception as persist_err:
                    logger.warning(f"[Flask] Failed to persist activity images: {persist_err}")
                    state["asset_root"] = None

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

        t = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
        t.start()

        return redirect(url_for("job", job_id=job_id))

    @app.get("/job/<job_id>")
    def job(job_id: str):
        if job_id not in JOBS:
            flash("Invalid job id", "error")
            return redirect(url_for("index"))
        return render_template("job.html", job_id=job_id)

    @app.get("/assets/<path:filename>")
    def assets(filename: str):
        root = os.path.join(os.path.dirname(__file__), "src")
        file_path = os.path.join(root, filename)
        if not os.path.isfile(file_path):
            return jsonify({"error": "not_found"}), 404
        return send_from_directory(root, filename)

    @app.get("/progress/<job_id>")
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

    @app.get("/results/<job_id>")
    def results(job_id: str):
        state = JOBS.get(job_id)
        if not state:
            flash("Invalid job id", "error")
            return redirect(url_for("index"))
        if state.get("error"):
            flash(f"Processing failed: {state['error']}", "error")
            return redirect(url_for("index"))
        events = state.get("events") or []
        trip_id = state.get("trip_id") or ""

        # Pagination params with sane bounds
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

    @app.get("/media/<job_id>/<path:filename>")
    def media(job_id: str, filename: str):
        state = JOBS.get(job_id)
        if not state:
            return jsonify({"error": "not_found"}), 404
        root = state.get("asset_root")
        if not root:
            return jsonify({"error": "no_assets"}), 404
        file_path = os.path.join(root, filename)
        if not os.path.isfile(file_path):
            return jsonify({"error": "file_missing"}), 404
        return send_from_directory(root, filename)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)


