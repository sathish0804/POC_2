import multiprocessing
workers = max(1, multiprocessing.cpu_count() // 2)
threads = 1
preload_app = True
timeout = 600
graceful_timeout = 30
max_requests = 2000
max_requests_jitter = 200
bind = "0.0.0.0:8000"

raw_env = [
    "YOLO_WEIGHTS_PRELOAD=yolo11s.pt",
    "PRELOAD_OCR=1",
]
