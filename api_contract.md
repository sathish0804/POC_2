## Activity Detection API Contract (v1.0.0)

This document describes the public HTTP API exposed by the Activity Detection service implemented in `controllers/api.py`.

### Base URL

- Local development: `http://localhost:8000`

### Health

- Method: GET
- Path: `/healthz`
- Description: Liveness probe.
- Request: No body
- Responses:
  - 200 OK: `{ "status": "ok" }`

### Process Video

- Method: POST
- Path: `/process`
- Description: Accepts a video file and returns detected activity events. The server persists annotated frames to `example_data/output/<timestamp>/` for inspection.
- Request Content-Type: `multipart/form-data`
- Request Fields:
  - `tripId` (form field, string, required): Identifier for the trip/session.
  - `cvvrFile` (file, required): Video file to analyze. Common extensions: `.mp4`.

- Behavior Notes:
  - Uses YOLO weights `yolo11s.pt` with sampling at 1 fps.
  - OCR is disabled by default in this endpoint.
  - Processes entire video unless internal limits are configured.
  - Temporary upload is deleted after processing; derived artifacts are copied to `example_data/output/<timestamp>/` if present.

- Responses:
  - 200 OK: JSON array of `ActivityEvent` objects (can be empty if no events found).
  - 400 Bad Request: When file is missing.
  - 5xx: Unexpected server errors.

#### ActivityEvent Schema

Pydantic model defined in `models/activity_event.py`.

```start:finish:models/activity_event.py
class ActivityEvent(BaseModel):
    tripId: str
    activityType: int
    des: str
    objectType: Optional[str]
    fileUrl: str
    fileDuration: str
    activityStartTime: str
    crewName: str
    crewId: str
    crewRole: int
    date: str
    time: str
    filename: str
    peopleCount: Optional[int]
    evidence: Optional[Dict[str, Any]]
```

Field details:
- `activityType` codes (mapping derives from internal heuristics):
  - 1: Micro sleep episode
  - 2: Sleeping episode
  - 3: Using cell phone
  - 4: Writing while moving
  - 5: Packing
  - 7: Signal exchange with flag
  - 8: More than two people detected
  - 0: Unknown/other (fallback)

#### Request Example

```bash
curl -X POST \
  -F "tripId=DemoTrip" \
  -F "cvvrFile=@/path/to/video.mp4" \
  http://localhost:8000/process
```

#### Success Response Example (200)

```json
[
  {
    "tripId": "DemoTrip",
    "activityType": 3,
    "des": "Using cell phone",
    "objectType": "cell phone",
    "fileUrl": "/tmp/upload_xxx/input.mp4",
    "fileDuration": "00:10:05",
    "activityStartTime": "20.16",
    "crewName": "demo",
    "crewId": "1",
    "crewRole": 1,
    "date": "",
    "time": "",
    "filename": "input.mp4",
    "peopleCount": 1,
    "evidence": { "rule": "hand_object_intersection" }
  }
]
```

#### Error Response Examples

- 400 Bad Request (no file uploaded)

```json
{ "detail": "No file uploaded" }
```

### Notes for Integrators

- The endpoint may return multiple events for different timestamps and persons; clients should aggregate or filter as needed.
- `fileUrl` points to the temporary processing path and is informational; do not rely on its persistence.
- Annotated frames, if generated, are copied to `example_data/output/<timestamp>/` on the server.


