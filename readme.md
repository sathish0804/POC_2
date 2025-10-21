# CCTV Activity Detection (CPU-Only)

Python system to detect loco pilot activities from CCTV, running fully on CPU. It combines YOLOv8 detections, MediaPipe landmarks, OCR, and temporal logic to emit structured events and annotated media.

## Core Business Logic
- Detects and reports:
  - Micro sleep and sleep states using eye metrics (EAR), head pose, PERCLOS-like statistics, and a sliding-window decision machine.
  - Using cell phone vs walkie-talkie (antenna refinement) via YOLO + hand/face landmarks + heuristics.
  - Writing while moving, gated by presence of a writing surface (book or OCR text) near lap.
  - Packing events via repeated hand–bag interactions and IoU-over-time.
  - Signal exchange with flag using green-flag detection and hand overlap near window region.
  - More than two people (group) using de-duplicated person boxes.
- Output is a list of `ActivityEvent` records with timestamps, evidence, and media references.

## Architecture
- Flask Web App:
  - Blueprints:
    - `health` (`/health/`): health probe.
    - `ui`:
      - HTML UI: `GET /`, `POST /start`, `GET /job/<id>`, `GET /results/<id>`, `GET /media/<id>/<path>`.
      - JSON API: `POST /api/jobs`, `GET /api/jobs/<id>`, `GET /api/jobs/<id>/progress`, `GET /api/jobs/<id>/results`, `GET /api/jobs/<id>/media/<path>`.
  - CORS headers added post-request; JSON error handlers for `/api/*`.
  - Logging via Loguru with rotation to `output/app.log`.
- Processing Model:
  - Background job per upload:
    - Video saved to temp dir, a thread orchestrates parallel processing across a shared `ProcessPoolExecutor`.
    - The video is split into ~6s chunks; each worker runs pipeline range processing and returns events.
    - Progress is tracked by expected sampled frames per range at `sample_fps`.
  - Pipeline (`ActivityPipeline`):
    - Services from `model_cache`: `YoloService` (CPU), `MediaPipeService`, `OcrUtils`, `AntennaRefiner`.
    - Preprocessing: ROI include/exclude masks, CLAHE, gamma, simple background gating to suppress static background.
    - Detection flow per sampled frame:
      1) YOLO detects persons and objects; persons are filtered by confidence, area, and NMS IoU.
      2) MediaPipe face/hands/pose landmarks collected and mapped to frame coords.
      3) Heuristics:
         - Phone: area/aspect checks; hand–phone overlap; suppress glare-only artifacts; landmark-only inference with head-down/hand-height suppression; antenna refiner to reclassify walkie-talkie (suppressed from logging).
         - Flag: green flag detection + overlap with hands inside window region.
         - Sleep: basic tracker OR advanced `SleepDecisionMachine` with sliding windows and hysteresis; emits micro-sleep and sleep.
         - Writing: requires a lap-surface (book class or OCR text >= min chars) to emit.
         - Packing: hand–bag interactions in torso band with temporal motion.
         - Group: >2 distinct high-confidence persons after IoU merging.
      4) Annotated image and short clip are generated for events.
      5) Events are mapped to business `activityType` and human description.
    - Range mode supports YOLO micro-batching (`YOLO_BATCH`) and respects `max_frames`.

## Data Model
`app/models/activity_event.py`:
```json
{
  "tripId": "",
  "activityType": 0,
  "des": "",
  "objectType": "cell phone | sleep | ...",
  "fileUrl": "",
  "fileDuration": "HH:MM:SS",
  "activityStartTime": "seconds.string",
  "activityEndTime": "seconds.string | null",
  "crewName": "",
  "crewId": "",
  "crewRole": 1,
  "date": "",
  "time": "",
  "filename": "",
  "peopleCount": 1,
  "evidence": { "rule": "..." },
  "activityImage": "frame_..._activity.jpg",
  "activityClip": "frame_... .mp4"
}
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r /Users/satishvanga/Documents/Vanga/POC_2/requirements.txt
```

## Run (Web UI + API)
- Dev server:
```bash
python /Users/satishvanga/Documents/Vanga/POC_2/run.py
# opens on http://localhost:5000
```
- Health:
```bash
curl -s http://localhost:5000/health/
```
- UI:
  - Visit `http://localhost:5000/` to upload a video and view progress/results.
- API:
```bash
# create job via multipart
curl -s -X POST http://localhost:5000/api/jobs \
  -F tripId=DEMO-TRIP-001 \
  -F cvvrFile=@/Users/satishvanga/Documents/Vanga/POC_2/example_data/video_cfr.mp4

# get job status
curl -s http://localhost:5000/api/jobs/<job_id>

# stream media (supports Range for MP4)
curl -I http://localhost:5000/api/jobs/<job_id>/media/<relative_path_from_asset_root>
```

## Run (CLI, optional)
If you have a separate entry like `main.py`, you can still run ad-hoc processing. For parallel IO benchmarks:
```bash
python /Users/satishvanga/Documents/Vanga/POC_2/run.py multiproc --video /path/to/video.mp4 --processes 8
```

## Configuration
- Environment knobs:
  - Pooling: `POOL_PROCS` (default min(cpu, 6))
  - Sampling: `SAMPLE_FPS` (default 0.5)
  - Chunk size: `CHUNK_SECONDS` (default 6)
  - YOLO: `YOLO_WEIGHTS_PRELOAD`, `YOLO_CONF` (default 0.25), `YOLO_IOU` (0.45), `YOLO_BATCH` (1+)
  - Threads: `TORCH_NUM_THREADS`, `TORCH_NUM_INTEROP_THREADS`, `OPENCV_NUM_THREADS`
  - Preprocessing: `PREPROC_GAMMA`, `ROI_INCLUDE_POLY`, `ROI_EXCLUDE_POLY`, `BG_ALPHA`, `BG_THRESH`
  - Logging: `LOG_PATH` (default `output/app.log`); `FRONTEND_ORIGIN` for CORS
  - OCR: `PRELOAD_OCR` (0/1), and pipeline `enable_ocr` flag
- Advanced sleep:
  - Enable via `use_advanced_sleep=True` in pipeline config.
  - Tunables: short/mid/long windows, smoothing alpha, closed-run threshold, PERCLOS thresholds, head-pitch degrees, recovery holds.

## Outputs
- Events saved in job asset root under `output/<timestamp>/events.json`.
- Annotated images `*_activity.jpg` and short clips near the event timestamp.
- Log file at `/Users/satishvanga/Documents/Vanga/POC_2/output/app.log`.

## Tech Stack
- Flask 3 for web/API; Loguru for logging.
- Ultralytics YOLOv8 (CPU), MediaPipe Face/Hands/Pose.
- EasyOCR/Tesseract (optional) for surface/text gating.
- OpenCV for IO, preprocessing, overlays, and clips.
- Multiprocessing with `spawn` and a shared `ProcessPoolExecutor`.

## Notes and Tuning
- Everything is CPU-only; lower `imgsz` or `YOLO_CONF` to adjust performance/recall.
- OCR downloads on first use; set `enable_ocr=False` for speed if not required.
- Person filtering uses minimum area and IoU de-duplication to reduce spurious extra persons.
- Walkie-talkie detections are suppressed from logging (misuse of phone is the target).