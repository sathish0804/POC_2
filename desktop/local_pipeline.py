from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional


def _resource_path(rel_path: str) -> str:
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return os.path.join(base, rel_path)
    # fallback to repo root
    here = os.path.abspath(os.path.dirname(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    return os.path.join(root, rel_path)


def process_video_locally(
    video_path: str,
    trip_id: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    sample_fps: float = 0.5,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Run the activity pipeline locally and return events as dictionaries.

    progress_cb receives (processed, total) integers.
    """
    # Lazily import heavy deps so this module can be imported fast
    from app.services.pipeline_service import ActivityPipeline

    yolo_path_candidate = _resource_path("yolo11s.pt")
    yolo_weights = yolo_path_candidate if os.path.exists(yolo_path_candidate) else "yolo11s.pt"

    pipe = ActivityPipeline(
        trip_id=trip_id,
        crew_name="local",
        crew_id="local",
        crew_role=1,
        yolo_weights=yolo_weights,
        sample_fps=sample_fps,
        enable_ocr=False,
        verbose=verbose,
        max_frames=0,
        use_advanced_sleep=True,
        sleep_min_duration=10.0,
        sleep_micro_max_min=0.25,
    )

    def _progress_adapter(d: Dict[str, Any]) -> None:
        if progress_cb:
            try:
                progress_cb(int(d.get("processed", 0)), int(d.get("total", 0)))
            except Exception:
                pass

    events = pipe.process_video(video_path, progress_cb=_progress_adapter)
    # Normalize to plain dicts
    out: List[Dict[str, Any]] = []
    for e in events or []:
        if hasattr(e, "model_dump"):
            out.append(e.model_dump())
        elif isinstance(e, dict):
            out.append(e)
    return out


