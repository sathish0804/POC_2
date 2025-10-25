from fastapi import APIRouter
from loguru import logger

router = APIRouter()


@router.get("/")
async def health():
    return {"status": "ok"}


# Module import log
logger.debug(f"[{__name__}] module loaded")


