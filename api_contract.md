API Contract (External UI)

Base URL
- Local: http://localhost:8000
- Server: http://103.195.244.67:8000/

CORS
- Allowed Origin: set via env `FRONTEND_ORIGIN` (default `*` in development)
- Methods: GET, POST, OPTIONS
- Headers: Content-Type, Authorization, Range

Health
- GET /health
  - 200 OK { "status": "ok" }

Create Job (upload)
- POST /api/jobs
  - Content-Type: multipart/form-data
  - Fields:
    - tripId: string (required)
    - cvvrFile: file (mp4/mov/mkv/avi, required)
  - 201 Created
  - Response:
    - job_id: string
    - status_url: string
    - progress_url: string
    - results_url: string

List Server Videos
- GET /api/jobs/server-videos
  - 200 OK
  - Response:
    - videos: string[]

Create Job (server file)
- POST /api/jobs/start
  - Content-Type: application/x-www-form-urlencoded
  - Body:
    - tripId: string (required)
    - videoName: string (required; must exist under configured video_input_dir)
  - 200 OK
  - Response:
    - job_id: string

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
      - activityImage: string | null      (relative filename)
      - activityClip: string | null       (relative filename)
    - page: number
    - page_size: number
    - total: number
    - start: number
    - end: number
    - total_pages: number

Errors (JSON)
- 400 { "error": "bad_request", "message": string }
- 404 { "error": "not_found", "path": string } (for /api/* only)
- 413 { "error": "payload_too_large" }
- 500 { "error": "internal_server_error" } (for /api/* only)

Examples
```bash
curl -X POST http://localhost:8000/api/jobs \
  -F tripId=TRIP-001 \
  -F cvvrFile=@/path/to/video.mp4

curl http://localhost:8000/api/jobs/{job_id}

curl http://localhost:8000/api/jobs/{job_id}/progress

curl "http://localhost:8000/api/jobs/{job_id}/results?page=1&page_size=25"

# server-side flow
curl http://localhost:8000/api/jobs/server-videos
curl -X POST http://localhost:8000/api/jobs/start -d "tripId=TRIP-001&videoName=Cabin 27 min Video.mp4"
```


