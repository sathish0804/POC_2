#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/.."

# Ensure dependencies
python -m pip install -r desktop/requirements.txt >/dev/null

# Build
pyinstaller \
  --name CVVR-Uploader \
  --windowed \
  --noconfirm \
  --clean \
  --add-data=desktop:desktop \
  desktop/app.py

echo "Built app in dist/CVVR-Uploader.app"

# Create a compressed DMG for distribution
APP_PATH="dist/CVVR-Uploader.app"
DMG_PATH="dist/CVVR-Uploader.dmg"
if [ -d "$APP_PATH" ]; then
  echo "Creating DMG at $DMG_PATH"
  hdiutil create -volname "CVVR Uploader" -srcfolder "$APP_PATH" -ov -format UDZO "$DMG_PATH" >/dev/null
  echo "DMG created: $DMG_PATH"
else
  echo "App bundle not found at $APP_PATH" 1>&2
  exit 1
fi


