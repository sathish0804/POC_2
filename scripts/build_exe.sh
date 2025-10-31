#!/usr/bin/env bash
set -euo pipefail

# Build a onefile executable for the FastAPI server using PyInstaller

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

APP_NAME="cvvr_server"

echo "[build] Cleaning previous builds..."
rm -rf build dist "$APP_NAME.spec" || true

echo "[build] Running PyInstaller..."
pyinstaller \
  --noconfirm \
  --clean \
  --name "$APP_NAME" \
  --onefile \
  --console \
  --collect-all mediapipe \
  --collect-all ultralytics \
  --collect-all torch \
  --collect-all torchvision \
  --collect-binaries cv2 \
  --add-data "yolo11s.pt:yolo11s.pt" \
  --add-data "app/static:app/static" \
  run_server.py

echo "[build] Done. Executable: dist/$APP_NAME"


