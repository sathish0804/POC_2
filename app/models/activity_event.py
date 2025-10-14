from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from loguru import logger


class ActivityEvent(BaseModel):
    """Schema for activity detection output event."""
    tripId: str = Field(default="")
    activityType: int = Field(default=0)
    des: str = Field(default="")
    # Optional: specific object/activity label (e.g., "cell phone", "sleep")
    objectType: Optional[str] = Field(default=None)
    fileUrl: str = Field(default="")
    fileDuration: str = Field(default="")
    activityStartTime: str = Field(default="")
    crewName: str = Field(default="")
    crewId: str = Field(default="")
    crewRole: int = Field(default=1)
    date: str = Field(default="")
    time: str = Field(default="")
    filename: str = Field(default="")
    peopleCount: Optional[int] = Field(default=None)
    # Optional diagnostics for evaluation/sweeps: rule/evidence used to infer the event
    evidence: Optional[Dict[str, Any]] = Field(default=None)
    # Optional: filename of annotated activity image for this frame
    activityImage: Optional[str] = Field(default=None)
    # Optional: short MP4 clip around the event timestamp (for sleep/micro-sleep)
    activityClip: Optional[str] = Field(default=None)


# Module import log
logger.debug(f"[{__name__}] module loaded")
