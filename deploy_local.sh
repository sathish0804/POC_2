#!/usr/bin/env bash
set -euo pipefail

SERVER="103.195.244.67"
USER="root"
PASS='Login@123@@@'
APP_DIR="/opt/poc2"

# Package and stream code to remote (avoid temp file and macOS tar attrs)
tar czf - --exclude '.git' --exclude '__pycache__' --exclude 'output' -C "$(pwd)" . \
  | sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no ${USER}@${SERVER} "mkdir -p ${APP_DIR} && tar xzf - -C ${APP_DIR}"

sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no ${USER}@${SERVER} bash -s <<'REMOTE_EOF'
set -euo pipefail
APP_DIR="/opt/poc2"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-venv python3-dev build-essential \
    libgl1-mesa-glx libglib2.0-0 libjpeg-dev zlib1g-dev \
    tesseract-ocr ffmpeg
elif command -v yum >/dev/null 2>&1; then
  yum install -y python3 python3-venv python3-devel gcc gcc-c++ \
    glib2 glib2-devel tesseract ffmpeg
fi

mkdir -p "$APP_DIR"

python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip setuptools wheel

# Nuke any OpenCV variants first
"$APP_DIR/venv/bin/pip" uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true

# Split requirements into base vs ml
grep -viE '^(torch|torchvision|tensorflow|jax|jaxlib|keras)(==.*)?$' "$APP_DIR/requirements.txt" > "$APP_DIR/requirements.base.txt"
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.base.txt"

# Force headless OpenCV to avoid X11/GL issues
"$APP_DIR/venv/bin/pip" install opencv-contrib-python-headless==4.11.0.86

# Install CPU-only PyTorch/torchvision via official CPU index, matching pinned versions
TORCH_VER=$(awk -F'==' 'tolower($1)=="torch" {print $2}' "$APP_DIR/requirements.txt" | tail -n1)
TV_VER=$(awk -F'==' 'tolower($1)=="torchvision" {print $2}' "$APP_DIR/requirements.txt" | tail -n1)
CPU_IDX="--index-url https://download.pytorch.org/whl/cpu"
if [ -n "${TORCH_VER:-}" ] && [ -n "${TV_VER:-}" ]; then
  "$APP_DIR/venv/bin/pip" install $CPU_IDX torch=="$TORCH_VER" torchvision=="$TV_VER"
else
  "$APP_DIR/venv/bin/pip" install $CPU_IDX torch torchvision
fi

# (Optional) Re-add mediapipe separately if it wasn't in base
# "$APP_DIR/venv/bin/pip" install mediapipe==0.10.18

# Quick smoke test to fail fast with a readable error
cat > "$APP_DIR/_import_check.py" <<'PY'
import cv2, numpy as np
print("cv2 ok:", cv2.__version__)
import torch, torchvision
print("torch ok:", torch.__version__, "cuda?", torch.cuda.is_available())
try:
    import mediapipe as mp
    print("mediapipe ok:", mp.__version__)
except Exception as e:
    print("mediapipe optional import failed:", repr(e))
PY
"$APP_DIR/venv/bin/python" "$APP_DIR/_import_check.py"

# Derive pool size from CPU count (can be overridden below)
POOL_PROCS=$( (command -v nproc >/dev/null 2>&1 && nproc) || (getconf _NPROCESSORS_ONLN) || echo 4 )

# Systemd unit (values templated)
cat >/etc/systemd/system/poc2.service <<UNIT
[Unit]
Description=POC_2 Gunicorn Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/poc2
Environment=PYTHONUNBUFFERED=1
Environment=FLASK_ENV=production
Environment=CUDA_VISIBLE_DEVICES=
Environment=OMP_NUM_THREADS=1
Environment=OPENBLAS_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
Environment=NUMEXPR_NUM_THREADS=1
Environment=OPENCV_NUM_THREADS=1
Environment=TORCH_NUM_THREADS=1
Environment=TORCH_NUM_INTEROP_THREADS=1
# Hard-set recommended knobs for 12-core server; adjust as needed
Environment=POOL_PROCS=12
Environment=CHUNK_SECONDS=6
Environment=YOLO_BATCH=3
Environment=SAVE_DEBUG_OVERLAYS=0
ExecStart=/opt/poc2/venv/bin/gunicorn -c gunicorn.conf.py wsgi:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now poc2

# Check bind on :8000
if command -v ss >/dev/null 2>&1; then
  ss -ltnp | grep ':8000' || (journalctl -u poc2 --no-pager -n 100; exit 1)
else
  netstat -ltnp | grep ':8000' || (journalctl -u poc2 --no-pager -n 100; exit 1)
fi
REMOTE_EOF
