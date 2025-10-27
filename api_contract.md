API Contract (External UI)

Base URL
- Server: http://103.195.244.67:8000/

CORS
- Allowed Origin: set via env `FRONTEND_ORIGIN` (default `*`; supports comma-separated list)
- Methods: GET, POST, OPTIONS
- Headers: * (all headers; includes `Range`)
- Expose Headers: Content-Range, Accept-Ranges

Health
- GET /health
  - 200 OK { "status": "ok" }

Create Job (upload)
- POST /api/jobs
  - Content-Type: multipart/form-data
  - Fields:
    - tripId: string (required)
    - cvvrFile: file (mp4/mov/mkv/avi, required)
  - 200 OK
  - Response:
    - job_id: string
    - status_url: string
    - progress_url: string
    - results_url: string
  - Errors:
    - 400 { "detail": "tripId is required" }
    - 400 { "detail": "cvvrFile is required" }
    - 500 { "detail": string }

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
  - Errors:
    - 400 { "detail": "VIDEO_INPUT_DIR is not configured or does not exist" }
    - 400 { "detail": "Unsupported video type" }
    - 400 { "detail": "Invalid video selection" }
    - 404 { "detail": "Selected video not found" }

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
      - activityImageUrl: string | null   (absolute URL to image)
      - activityClipUrl: string | null    (absolute URL to mp4; streamable)
    - page: number
    - page_size: number
    - total: number
    - start: number
    - end: number
    - total_pages: number
  - Errors:
    - 404 { "detail": "invalid_job" }
    - 500 { "detail": string }

Fetch Media (images and clips)
- GET /api/jobs/{job_id}/media/{filename}
  - Description: Serves images and mp4 clips generated for a job.
  - Requests:
    - Headers (optional): Range: bytes={start}-{end} (mp4 partial requests)
  - Responses:
    - 200 OK (full content)
    - 206 Partial Content (mp4 with Range)
      - Headers: Accept-Ranges: bytes, Content-Range: "bytes {start}-{end}/{size}", Content-Length
    - 404 { "detail": "not_found" | "no_assets" | "file_missing" }
  - Notes:
    - MIME detection is automatic; mp4 served as video/mp4.

Errors
- Error responses use FastAPI default shape: { "detail": string }
- Common statuses:
  - 400: bad request (e.g., missing fields, invalid selection)
  - 404: not found (e.g., invalid_job, file_missing)
  - 500: internal_server_error or job error message

Examples
```bash
curl -X POST http://103.195.244.67:8000/api/jobs \
  -F tripId=TRIP-001 \
  -F cvvrFile=@/path/to/video.mp4

curl http://103.195.244.67:8000/api/jobs/{job_id}

curl http://103.195.244.67:8000/api/jobs/{job_id}/progress

curl "http://103.195.244.67:8000/api/jobs/{job_id}/results?page=1&page_size=25"

# server-side flow
curl http://103.195.244.67:8000/api/jobs/server-videos
curl -X POST http://103.195.244.67:8000/api/jobs/start -d "tripId=TRIP-001&videoName=Cabin 27 min Video.mp4"

# media (range request for mp4)
curl -i -H "Range: bytes=0-1023" http://103.195.244.67:8000/api/jobs/{job_id}/media/path/to/file.mp4
```


