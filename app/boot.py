import os
import atexit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
from loguru import logger
from app.config import settings

_POOL: Optional[ProcessPoolExecutor] = None


def _init_worker() -> None:
    try:
        import torch
        torch.set_num_threads(int(settings.torch_num_threads))
        torch.set_num_interop_threads(int(settings.torch_num_interop_threads))
    except Exception:
        pass
    try:
        import cv2
        cv2.setNumThreads(int(settings.opencv_num_threads))
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass
    except Exception:
        pass
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", os.path.join(repo_root, "output", ".ultralytics"))
    except Exception:
        pass
    try:
        weights = (settings.yolo_weights_preload or "").strip()
        # Resolve path inside PyInstaller bundle or repo
        try:
            from app.utils.path_utils import resource_path as _res
            if weights:
                weights = _res(weights)
        except Exception:
            pass
        if weights:
            from app.services.model_cache import preload_models
            preload_models(weights_path=weights)
            logger.info(f"[boot] Preloaded models in worker (weights={weights})")
    except Exception as e:
        logger.warning(f"[boot] Worker preload skipped: {e}")


def get_pool(max_workers: Optional[int] = None) -> ProcessPoolExecutor:
    global _POOL
    if _POOL is not None:
        return _POOL
    if max_workers is None:
        env = int(settings.pool_procs or 6)
        cpu = (mp.cpu_count() or 1)
        max_workers = max(1, min(env, cpu))
    ctx = mp.get_context("spawn")
    _POOL = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=_init_worker)
    logger.info(f"[boot] Created shared ProcessPoolExecutor(max_workers={max_workers})")
    return _POOL


def _shutdown_pool() -> None:
    global _POOL
    try:
        if _POOL is not None:
            _POOL.shutdown(wait=False, cancel_futures=True)
            _POOL = None
            logger.info("[boot] Shut down shared ProcessPoolExecutor")
    except Exception:
        pass


atexit.register(_shutdown_pool)


