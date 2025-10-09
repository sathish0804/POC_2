import json
import os
from typing import List

from models.activity_event import ActivityEvent


def write_events_to_json(events: List[ActivityEvent], out_path: str) -> None:
    """Serialize a list of ActivityEvent models to a JSON file.

    The file will contain an array of JSON objects matching the required schema.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = [e.model_dump() for e in events]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
