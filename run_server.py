import os
import sys


def _configure_env() -> None:
    # Ensure YOLO preload path resolves inside bundled executable
    try:
        from app.utils.path_utils import resource_path
        weights = resource_path(os.getenv("YOLO_WEIGHTS_PRELOAD", "yolo11s.pt"))
        os.environ.setdefault("YOLO_WEIGHTS_PRELOAD", weights)
    except Exception:
        pass
    # Ultralytics cache/config directory to a writable folder
    try:
        base_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.getcwd()
        out_dir = os.path.join(base_dir, "output", ".ultralytics")
        os.makedirs(out_dir, exist_ok=True)
        os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", out_dir)
    except Exception:
        pass


def main() -> None:
    _configure_env()
    import uvicorn
    # Import the ASGI app explicitly so PyInstaller includes it
    from asgi import app as asgi_app

    host = os.getenv("HOST", "0.0.0.0")
    try:
        port = int(os.getenv("PORT", "8000"))
    except Exception:
        port = 8000
    try:
        workers = int(os.getenv("WORKERS", "1"))
    except Exception:
        workers = 1

    # Import after env configuration so pydantic settings see env
    uvicorn.run(asgi_app, host=host, port=port, workers=workers, reload=False)


if __name__ == "__main__":
    main()


