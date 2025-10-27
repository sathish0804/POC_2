from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pydantic_settings import BaseSettings
from pydantic import Field


# Load .env once at import time
load_dotenv()


def _default_video_input_dir() -> str:
    # repo_root/app/config.py → repo_root/example_data
    return str((Path(__file__).resolve().parents[1] / "example_data").as_posix())


class Settings(BaseSettings):
    """Configuration settings for the application (only relevant to this project)."""

    # app environment + logging
    environment: str = Field(default="production", env="ENVIRONMENT")  # development|staging|production
    log_dir: str = Field(default="output", env="LOG_DIR")
    frontend_origin: str = Field(default="*", env="FRONTEND_ORIGIN")

    # inputs
    video_input_dir: str = Field(default_factory=_default_video_input_dir)

    # performance controls (used by pool/threads and pipeline)
    pool_procs: int = Field(default=6, env="POOL_PROCS")
    sample_fps: float = Field(default=0.5, env="SAMPLE_FPS")
    chunk_seconds: float = Field(default=6.0, env="CHUNK_SECONDS")

    torch_num_threads: int = Field(default=1, env="TORCH_NUM_THREADS")
    torch_num_interop_threads: int = Field(default=1, env="TORCH_NUM_INTEROP_THREADS")
    opencv_num_threads: int = Field(default=1, env="OPENCV_NUM_THREADS")

    # YOLO/pipeline knobs
    yolo_weights_preload: str = Field(default="yolo11s.pt", env="YOLO_WEIGHTS_PRELOAD")
    yolo_conf: float = Field(default=0.25, env="YOLO_CONF")
    yolo_iou: float = Field(default=0.45, env="YOLO_IOU")
    yolo_batch: int = Field(default=1, env="YOLO_BATCH")

    # Optimization flags
    enable_mediapipe_gate: bool = Field(default=True, env="ENABLE_MEDIAPIPE_GATE")
    downscale_for_inference: bool = Field(default=True, env="DOWNSCALE_FOR_INFERENCE")
    inference_max_side: int = Field(default=960, env="INFERENCE_MAX_SIDE")
    motion_gate_enable: bool = Field(default=True, env="MOTION_GATE_ENABLE")
    motion_gate_thresh: float = Field(default=0.006, env="MOTION_GATE_THRESH")  # fraction of changed pixels
    motion_gate_pixel_diff: int = Field(default=12, env="MOTION_GATE_PIXEL_DIFF")  # diff threshold (0-255)
    clip_enable: bool = Field(default=True, env="CLIP_ENABLE")
    clip_top_n: int = Field(default=0, env="CLIP_TOP_N")  # 0 = unlimited

    roi_include_poly: str = Field(default="", env="ROI_INCLUDE_POLY")
    roi_exclude_poly: str = Field(default="", env="ROI_EXCLUDE_POLY")
    preproc_gamma: float = Field(default=0.9, env="PREPROC_GAMMA")
    bg_alpha: float = Field(default=0.05, env="BG_ALPHA")
    bg_thresh: int = Field(default=18, env="BG_THRESH")
    preload_ocr: bool = Field(default=False, env="PRELOAD_OCR")

    class Config:
        env_file = ".env"
        extra = "ignore"


# Singleton settings instance
settings = Settings()

logger.debug(f"[{__name__}] settings loaded for environment={settings.environment}")

