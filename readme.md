# CCTV Activity Detection (CPU-Only)

Python pipeline for detecting loco pilot activities using YOLOv8 (Ultralytics), MediaPipe (Face/Hands/Pose), OCR (EasyOCR/Tesseract), and OpenCV.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r /Users/satishvanga/Documents/Vanga/POC_2/requirements.txt
```

## Run (CLI)

```bash
python /Users/satishvanga/Documents/Vanga/POC_2/main.py \
  --video /Users/satishvanga/Documents/Vanga/POC_2/example_data/video_cfr.mp4 \
  --model yolov8m.pt \
  --fps 1 \
  --out /Users/satishvanga/Documents/Vanga/POC_2/output/events.json \
  --disable_ocr  # optional to skip OCR downloads on first run
```

- First run may download YOLO weights and OCR models.
- The script writes JSON events to `--out`.

## Run (API Server)

Start the server:

```bash
python /Users/satishvanga/Documents/Vanga/POC_2/api_main.py
```

```bash
uvicorn controllers.api:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl -s http://localhost:8000/healthz
```

Process a local upload (multipart form):

```bash
curl -s -X POST http://localhost:8000/process \
  -F tripId=DEMO-TRIP-001 \
  -F crewName="Demo Crew" \
  -F crewId=C-001 \
  -F crewRole=1 \
  -F cvvrFile=@/Users/satishvanga/Documents/Vanga/POC_2/example_data/video_cfr.mp4 \
  -F model=yolov8l.pt -F fps=1 -F disable_ocr=false -F verbose=false -F max_frames=0
```

Process by URL with JSON payload:

```bash
curl -s -X POST http://localhost:8000/process-by-url \
  -H 'Content-Type: application/json' \
  -d '{
        "tripId": "DEMO-TRIP-001",
        "cvvrFileUrl": "/Users/satishvanga/Documents/Vanga/POC_2/example_data/video_cfr.mp4",
        "crews": [{"crewName": "Demo Crew", "crewId": "C-001", "crewRole": 1}]
      }'
```

Both endpoints return a JSON array of events using the same schema as `output/events.json`.

## Logging

- Logs are written to `/Users/satishvanga/Documents/Vanga/POC_2/output/app.log` (rotates at ~10MB, keeps 7 days).
- API startup configures logging automatically. To enable verbose logs for a request, set `verbose=true` (form field) on `/process`.

## Activities
1. Micro sleep (eyes closed ≥2s with low motion) [advanced: sliding windows]
2. Sleeping (high PERCLOS with stillness and head-down) [advanced]
3. Using cell phone (hand-phone IoU > threshold)
4. Writing while moving (hand motion near pen/notebook)
5. Packing (hand overlaps bag frequently)
6. Calling signals (arm extension gesture repeated)
7. Signal exchange with flag (flag + hand interaction)

## JSON Schema

```json
{
  "tripId": "",
  "activityType": 1,
  "des": "",
  "fileUrl": "",
  "fileDuration": "",
  "activityStartTime": "",
  "crewName": "",
  "crewId": "",
  "crewRole": 1,
  "date": "",
  "time": "",
  "filename": "",
  "peopleCount": 1
}
```

## Notes
- Class IDs for phone/bag/flag/pen are placeholders; adapt to your YOLO model `names`.
- Temporal thresholds assume ~1 FPS sampling; they auto-adjust using timestamps but tune as needed.
- Everything runs on CPU; for speed, consider disabling OCR with `--disable_ocr`.

## Advanced sleep detection (optional)

- Enable via `ActivityPipeline(use_advanced_sleep=True)`.
- Computes EAR and a coarse eye-open probability, smooths with EWMA, and applies a sliding-window, hysteresis decision machine:
  - Microsleep: continuous closure ≥ 2s with low head motion (short window)
  - Drowsy: PERCLOS(mid window) > ~0.4
  - Sleep: PERCLOS(mid window) > ~0.8 with stillness and head-down
  - Recovery: sustained eye-open and neutral head pose
- If landmarks are unreliable, logic degrades gracefully and avoids false alarms (may emit no sleep state).