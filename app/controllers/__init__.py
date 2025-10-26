from fastapi import APIRouter


api_router = APIRouter()

from loguru import logger

# Package marker for controllers

# Module import log
logger.debug(f"[{__name__}] package loaded")

