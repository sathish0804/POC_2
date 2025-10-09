from typing import List, Optional
import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from controllers.pipeline import ActivityPipeline
from models.activity_event import ActivityEvent
from utils.logging_utils import configure_logging
from loguru import logger


app = FastAPI(title="Activity Detection API", version="1.0.0")


@app.on_event("startup")
def _on_startup():
    # Configure logging with default INFO level to file
    log_path = configure_logging(verbose=False)
    logger.info(f"API startup. Logs at: {log_path}")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


def _build_pipeline(
    trip_id: str,
    crew_name: str,
    crew_id: str,
    crew_role: int,
    *,
    yolo_weights: str = "yolov8l.pt",
    sample_fps: int = 1,
    enable_ocr: bool = True,
    verbose: bool = False,
    max_frames: int = 0,
):
    logger.debug(
        "Creating ActivityPipeline: trip_id={trip_id}, crew_name={crew_name}, crew_id={crew_id}, crew_role={crew_role}, weights={yolo_weights}, fps={sample_fps}, ocr={enable_ocr}, verbose={verbose}, max_frames={max_frames}",
        trip_id=trip_id,
        crew_name=crew_name,
        crew_id=crew_id,
        crew_role=crew_role,
        yolo_weights=yolo_weights,
        sample_fps=sample_fps,
        enable_ocr=enable_ocr,
        verbose=verbose,
        max_frames=max_frames,
    )
    return ActivityPipeline(
        trip_id=trip_id,
        crew_name=crew_name,
        crew_id=crew_id,
        crew_role=crew_role,
        yolo_weights=yolo_weights,
        sample_fps=sample_fps,
        enable_ocr=enable_ocr,
        verbose=verbose,
        max_frames=max_frames,
    )


@app.post("/process")
async def process_upload(
    tripId: str = Form(...),
    cvvrFile: UploadFile = File(...),
):
    if not cvvrFile.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Save to a temp file on disk for OpenCV/ffmpeg access
    tmp_dir = tempfile.mkdtemp(prefix="upload_")
    try:
        suffix = os.path.splitext(cvvrFile.filename)[1] or ".mp4"
        video_path = os.path.join(tmp_dir, f"input{suffix}")
        with open(video_path, "wb") as f:
            shutil.copyfileobj(cvvrFile.file, f)
        logger.info(f"Uploaded video saved to {video_path}")

        # Defaults as requested: demo crew and fixed processing params
        demo_crew_name = "demo"
        demo_crew_id = "1"
        demo_crew_role = 1
        default_model = "yolo11s.pt"
        default_fps = 1
        default_enable_ocr = False  # disable_ocr=False -> enable_ocr=True
        default_verbose = False
        default_max_frames = 0  # empty => process all frames

        pipeline = _build_pipeline(
            trip_id=tripId,
            crew_name=demo_crew_name,
            crew_id=demo_crew_id,
            crew_role=int(demo_crew_role),
            yolo_weights=default_model,
            sample_fps=int(default_fps),
            enable_ocr=bool(default_enable_ocr),
            verbose=bool(default_verbose),
            max_frames=int(default_max_frames),
        )
        logger.info("Starting video processing")
        events = pipeline.process_video(video_path)
        logger.info(f"Processing complete. Found {len(events)} events")
        # Return list of events matching output/events.json schema
        # Persist any generated activity images from the temp output dir
        try:
            tmp_output_dir = os.path.join(os.path.dirname(video_path), "output")
            if os.path.isdir(tmp_output_dir):
                # Persist under example_data/output/<timestamp>/
                from datetime import datetime
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_root = os.path.join(
                    "/Users/satishvanga/Documents/Vanga/POC_2/example_data", "output", stamp
                )
                os.makedirs(dest_root, exist_ok=True)
                # Copy tree (only images and small artifacts). If exists, merge content.
                for entry in os.listdir(tmp_output_dir):
                    src_path = os.path.join(tmp_output_dir, entry)
                    dst_path = os.path.join(dest_root, entry)
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_path, dst_path)
                logger.info(f"Persisted activity images to {dest_root}")
            else:
                logger.debug("No temp output directory found; nothing to persist")
        except Exception as persist_err:
            logger.warning(f"Failed to persist activity images: {persist_err}")
        return JSONResponse([e.model_dump() for e in events])
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

