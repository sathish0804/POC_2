from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from loguru import logger

import requests

from app.config import settings


def _transform_seconds_to_iso_time(seconds_str: str, base_date: Optional[str] = None, base_time: Optional[str] = None) -> str:
    """Convert seconds timestamp to ISO format datetime string."""
    try:
        base_dt = None
        
        # Try to parse base date and time if available
        if base_date and base_time:
            try:
                # Try common date formats
                date_formats = [
                    "%Y-%m-%d %H:%M:%S",
                    "%d-%m-%Y %H:%M:%S",
                    "%Y/%m/%d %H:%M:%S",
                    "%d/%m/%Y %H:%M:%S",
                ]
                for fmt in date_formats:
                    try:
                        base_dt = datetime.strptime(f"{base_date} {base_time}", fmt)
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
        
        # If no base datetime, use current time
        if base_dt is None:
            base_dt = datetime.now()
        
        # Add seconds offset
        seconds = float(seconds_str)
        result_dt = base_dt + timedelta(seconds=seconds)
        return result_dt.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        # Fallback: use current time
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _format_duration(duration_str: str) -> str:
    """Format duration string to HH:MM:SS format."""
    try:
        # Try parsing as HH:MM:SS first
        if ":" in duration_str:
            parts = duration_str.split(":")
            if len(parts) == 3:
                return duration_str
        
        # Try parsing as seconds
        seconds = float(duration_str)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    except Exception:
        # If already formatted or invalid, return as is or default
        return duration_str if duration_str else "00:00:00"


def _event_to_violation(
    event: Dict[str, Any],
    trip_id: str,
    host_url: Optional[str] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert a single event to a violation object.
    
    Args:
        event: Single activity event dictionary
        trip_id: The trip ID
        host_url: Base URL for constructing media URLs (optional)
        job_id: Job ID for constructing media URLs (optional)
        
    Returns:
        Single violation object matching the required format
    """
    # Get object type
    object_type = event.get("objectType", "")
    
    # Get start and end times
    start_ts = event.get("activityStartTime", "")
    end_ts = event.get("activityEndTime") or start_ts
    base_date = event.get("date", "")
    base_time = event.get("time", "")
    
    if start_ts:
        start_time = _transform_seconds_to_iso_time(
            str(start_ts),
            base_date,
            base_time
        )
    else:
        start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    
    if end_ts:
        end_time = _transform_seconds_to_iso_time(
            str(end_ts),
            base_date,
            base_time
        )
    else:
        end_time = start_time
    
    # Get description
    description = event.get("des", "") or "Activity violation detected"
    
    # Get activity type
    activity_type = event.get("activityType", 1)
    
    # Get filename
    import os
    filename = event.get("filename", "")
    if not filename and event.get("fileUrl"):
        filename = os.path.basename(event.get("fileUrl", ""))
    
    # Get file duration
    file_duration = event.get("fileDuration", "00:00:00")
    file_duration = _format_duration(file_duration)
    
    # Get crew name
    crew_name = event.get("crewName", "")
    
    # Get activity clip URL as fileUrl (if host_url and job_id provided)
    file_url = ""
    if host_url and job_id:
        media_prefix = f"{host_url}/api/jobs/{job_id}/media"
        clip = event.get("activityClip")
        if clip:
            file_url = f"{media_prefix}/{clip}"
    
    # Build payload
    payload = {
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
        "fileType": 2,  # Default file type (2 = video)
        "fileUrl": file_url,
        "createdDate": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "createdBy": "system",
        "status": 1,  # Default status (1 = active/complete)
    }
    
    return payload


def post_cvvr_results(
    trip_id: str,
    events: List[Dict[str, Any]],
    job_id: Optional[str] = None,
    host_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Post CVVR trip violations to the external API.
    
    Args:
        trip_id: The trip ID associated with the events
        events: List of activity events to post
        job_id: Optional job ID for logging and URL construction
        host_url: Optional host URL for constructing media URLs
        
    Returns:
        Dict containing the response from the API or error information
        
    Raises:
        Exception: If the API call fails critically (connection errors, etc.)
    """
    if not settings.cvvr_api_url:
        logger.warning(f"[external_api] CVVR_API_URL not configured, skipping POST for trip_id={trip_id}")
        return {"success": False, "error": "API URL not configured"}
    
    if not events:
        logger.info(f"[external_api] No events to post for trip_id={trip_id}, skipping")
        return {"success": True, "message": "No events to post"}
    
    url = settings.cvvr_api_url
    timeout = settings.cvvr_api_timeout or 30
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
    }
    
    # Add authentication token if provided
    if settings.cvvr_api_token:
        headers["Authorization"] = f"Bearer {settings.cvvr_api_token}"
        # Alternative: if API uses a different auth header format:
        # headers["X-API-Key"] = settings.cvvr_api_token
    
    # Convert each event to a violation object (one violation per event)
    if not events:
        # Return early if no events
        return {"success": True, "message": "No events to post"}
    
    violations = []
    for event in events:
        violation = _event_to_violation(
            event=event,
            trip_id=trip_id,
            host_url=host_url,
            job_id=job_id
        )
        violations.append(violation)
    
    # Sort violations by startTime (earliest first)
    violations.sort(key=lambda v: v.get("startTime", ""))
    
    try:
        logger.info(
            f"[external_api] Posting {len(violations)} violation(s) to {url} "
            f"for trip_id={trip_id} (job_id={job_id})"
        )
        
        # Send payload as array of violation objects
        response = requests.post(
            url,
            json=violations,
            headers=headers,
            timeout=timeout
        )
        
        # Log response details
        logger.info(
            f"[external_api] Response status={response.status_code} "
            f"for trip_id={trip_id} (job_id={job_id})"
        )
        
        # Try to parse JSON response
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            response_data = {"text": response.text}
        
        # Check if request was successful
        if response.status_code in (200, 201):
            logger.success(
                f"[external_api] Successfully posted {len(events)} events "
                f"for trip_id={trip_id} (job_id={job_id})"
            )
            return {
                "success": True,
                "status_code": response.status_code,
                "response": response_data,
            }
        else:
            logger.warning(
                f"[external_api] API returned status {response.status_code} "
                f"for trip_id={trip_id} (job_id={job_id}): {response_data}"
            )
            return {
                "success": False,
                "status_code": response.status_code,
                "error": response_data,
            }
            
    except requests.exceptions.Timeout:
        logger.error(
            f"[external_api] Timeout after {timeout}s while posting to {url} "
            f"for trip_id={trip_id} (job_id={job_id})"
        )
        return {
            "success": False,
            "error": f"Request timeout after {timeout}s",
        }
    except requests.exceptions.ConnectionError as e:
        logger.error(
            f"[external_api] Connection error while posting to {url} "
            f"for trip_id={trip_id} (job_id={job_id}): {e}"
        )
        return {
            "success": False,
            "error": f"Connection error: {str(e)}",
        }
    except Exception as e:
        logger.exception(
            f"[external_api] Unexpected error while posting to {url} "
            f"for trip_id={trip_id} (job_id={job_id}): {e}"
        )
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
        }


# Module import log
logger.debug(f"[{__name__}] module loaded")

