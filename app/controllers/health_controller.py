from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import platform
import psutil
import time
from pydantic import BaseModel
from typing import Dict, Any, Optional
import sys
from app.config import settings


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    timestamp: float
    system_info: Dict[str, Any]
    services: Dict[str, Dict[str, Any]]


router = APIRouter()


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health() -> JSONResponse:
    """
    Health check endpoint that returns system health information.
    """
    # Basic system information
    system_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage_percent": psutil.disk_usage('/').percent
    }
    
    # Service checks - add more services as needed
    services = {
        "api": {
            "status": "healthy",
            "uptime_seconds": time.time() - psutil.boot_time()
        }
    }
    
    # Construct the response
    health_data = {
        "status": "ok",
        "version": "1.0.0",  # Replace with actual version
        "environment": settings.environment,
        "timestamp": time.time(),
        "system_info": system_info,
        "services": services
    }
    
    return JSONResponse(content=health_data)


