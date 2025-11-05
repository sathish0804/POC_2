from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import requests
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor

from desktop.state import AppState


ProgressCallback = Callable[[int, int], None]


class ApiClient:
    def __init__(self, state: Optional[AppState] = None) -> None:
        self.state = state or AppState.load()

    @property
    def base_url(self) -> str:
        return (self.state.server_url or "").rstrip("/")

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        token = (self.state.api_token or '').strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def health(self, timeout: int = 10) -> Dict[str, Any]:
        url = f"{self.base_url}/health"
        resp = requests.get(url, headers=self._headers(), timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def create_job(self, trip_id: str, video_path: str, progress_cb: Optional[ProgressCallback] = None, timeout: int = 600) -> Dict[str, Any]:
        url = f"{self.base_url}/api/jobs"

        if not os.path.isfile(video_path):
            raise FileNotFoundError(video_path)

        filename = os.path.basename(video_path)

        with open(video_path, "rb") as f:
            encoder = MultipartEncoder(fields={
                "tripId": trip_id,
                "cvvrFile": (filename, f, "video/mp4"),
            })

            total = encoder.len

            def _monitor_cb(m: MultipartEncoderMonitor) -> None:
                if progress_cb:
                    progress_cb(m.bytes_read, total)

            monitor = MultipartEncoderMonitor(encoder, _monitor_cb)
            headers = {"Content-Type": monitor.content_type}
            headers.update(self._headers())
            resp = requests.post(url, data=monitor, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()

    def poll_progress(self, job_id: str, timeout: int = 10) -> Dict[str, Any]:
        url = f"{self.base_url}/api/jobs/{job_id}/progress"
        resp = requests.get(url, headers=self._headers(), timeout=timeout)
        resp.raise_for_status()
        return resp.json()


