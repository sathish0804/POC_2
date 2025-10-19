API Contract (External UI)

Base URL
- Local: http://localhost:5000
- Server: http://103.195.244.67:8000/

CORS
- Allowed Origin: set via env `FRONTEND_ORIGIN` (default `*` in development)
- Methods: GET, POST, OPTIONS
- Headers: Content-Type, Authorization, Range

Health
- GET /health/
  - 200 OK { "status": "ok" }

Create Job
- POST /api/jobs
  - Content-Type: multipart/form-data
  - Fields:
    - tripId: string (required)
    - cvvrFile: file (video/*, required)
  - 201 Created
  - Response:
    - job_id: string
    - status_url: string
    - progress_url: string
    - results_url: string
    - media_url_prefix: string

Get Job Summary
- GET /api/jobs/{job_id}
  - 200 OK
  - Response:
    - job_id: string
    - processed: number
    - total: number
    - percent: number
    - done: boolean
    - error: string | null

Poll Progress
- GET /api/jobs/{job_id}/progress
  - 200 OK
  - Response:
    - processed: number
    - total: number
    - done: boolean
    - error: string | null
    - notFound: boolean (optional)

Fetch Results (paginated)
- GET /api/jobs/{job_id}/results?page={n}&page_size={m}
  - 200 OK
  - Response:
    - job_id: string
    - trip_id: string
    - events: array of objects
      - activityImage: string | null
      - activityClip: string | null
      - activityImageUrl: string | null  (absolute URL)
      - activityClipUrl: string | null   (absolute URL)
    - page: number
    - page_size: number
    - total: number
    - start: number
    - end: number
    - total_pages: number

Fetch Media
- GET /api/jobs/{job_id}/media/{filename}
  - 200 OK (binary)
  - Notes: Supports HTTP Range for mp4; sets Accept-Ranges and Content-Range when partial.

Errors (JSON)
- 400 { "error": "bad_request", "message": string }
- 404 { "error": "not_found", "path": string } (for /api/* only)
- 413 { "error": "payload_too_large" }
- 500 { "error": "internal_server_error" } (for /api/* only)

Examples
```bash
curl -X POST http://localhost:5000/api/jobs \
  -F tripId=TRIP-001 \
  -F cvvrFile=@/path/to/video.mp4

curl http://localhost:5000/api/jobs/{job_id}

curl http://localhost:5000/api/jobs/{job_id}/progress

curl "http://localhost:5000/api/jobs/{job_id}/results?page=1&page_size=25"

curl -L "http://localhost:5000/api/jobs/{job_id}/media/r0_frame_000010_10.00s_activity.jpg" -o frame.jpg
```


