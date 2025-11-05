from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests


def _transform_seconds_to_iso_time(seconds_str: str, base_date: Optional[str] = None, base_time: Optional[str] = None) -> str:
    try:
        base_dt = None
        if base_date and base_time:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
                try:
                    base_dt = datetime.strptime(f"{base_date} {base_time}", fmt)
                    break
                except ValueError:
                    continue
        if base_dt is None:
            base_dt = datetime.now()
        seconds = float(seconds_str)
        result_dt = base_dt + timedelta(seconds=seconds)
        return result_dt.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _format_duration(duration_str: str) -> str:
    try:
        if ":" in duration_str:
            parts = duration_str.split(":")
            if len(parts) == 3:
                return duration_str
        seconds = float(duration_str)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    except Exception:
        return duration_str if duration_str else "00:00:00"


def _event_to_violation(event: Dict[str, Any], trip_id: str) -> Dict[str, Any]:
    object_type = event.get("objectType", "")
    start_ts = event.get("activityStartTime", "")
    end_ts = event.get("activityEndTime") or start_ts
    base_date = event.get("date", "")
    base_time = event.get("time", "")

    if start_ts:
        start_time = _transform_seconds_to_iso_time(str(start_ts), base_date, base_time)
    else:
        start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    if end_ts:
        end_time = _transform_seconds_to_iso_time(str(end_ts), base_date, base_time)
    else:
        end_time = start_time

    description = event.get("des", "") or "Activity violation detected"
    activity_type = event.get("activityType", 1)

    import os
    filename = event.get("filename", "")
    if not filename and event.get("fileUrl"):
        filename = os.path.basename(event.get("fileUrl", ""))

    file_duration = _format_duration(event.get("fileDuration", "00:00:00"))
    crew_name = event.get("crewName", "")

    # Local mode: fileUrl is left blank (no hosted media)
    file_url = ""

    return {
        "tripId": trip_id,
        "type": activity_type,
        "startTime": start_time,
        "endTime": end_time,
        "remarks": "Violation detected during trip processing",
        "reason": "Automated detection",
        "description": description,
        "objectTypes": object_type,
        "fileName": filename,
        "fileDuration": file_duration,
        "crewName": crew_name,
        "fileType": 2,
        "fileUrl": file_url,
        "createdDate": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "createdBy": "desktop",
        "status": 1,
    }


def post_results(
    trip_id: str,
    events: List[Dict[str, Any]],
    endpoint_url: str,
    token: Optional[str] = None,
    timeout: int = 30,
    no_events_url: Optional[str] = None,
) -> Dict[str, Any]:
    if not endpoint_url:
        return {"success": False, "error": "No endpoint URL provided"}

    # Transform + dedup
    violations = [_event_to_violation(e, trip_id=trip_id) for e in (events or [])]
    dedup_map: Dict[tuple, Dict[str, Any]] = {}
    for v in violations:
        key = (v.get("type"), v.get("objectTypes"), v.get("startTime"), v.get("endTime"), v.get("fileUrl"))
        if key not in dedup_map:
            dedup_map[key] = v
    unique_violations = list(dedup_map.values())
    unique_violations.sort(key=lambda v: v.get("startTime", ""))

    # Handle no-events flow if configured
    if not unique_violations and (no_events_url or ""):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            resp = requests.post(no_events_url, json={"tripId": trip_id}, headers=headers, timeout=timeout)
            try:
                data = resp.json()
            except json.JSONDecodeError:
                data = {"text": resp.text}
            if resp.status_code in (200, 201):
                return {"success": True, "status_code": resp.status_code, "response": data, "no_events": True}
            return {"success": False, "status_code": resp.status_code, "error": data, "no_events": True}
        except requests.exceptions.Timeout:
            return {"success": False, "error": f"Request timeout after {timeout}s", "no_events": True}
        except requests.exceptions.ConnectionError as e:
            return {"success": False, "error": f"Connection error: {e}", "no_events": True}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {e}", "no_events": True}

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.post(endpoint_url, json=unique_violations, headers=headers, timeout=timeout)
        try:
            data = resp.json()
        except json.JSONDecodeError:
            data = {"text": resp.text}
        if resp.status_code in (200, 201):
            return {"success": True, "status_code": resp.status_code, "response": data}
        return {"success": False, "status_code": resp.status_code, "error": data}
    except requests.exceptions.Timeout:
        return {"success": False, "error": f"Request timeout after {timeout}s"}
    except requests.exceptions.ConnectionError as e:
        return {"success": False, "error": f"Connection error: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}


