Param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location ..

python -m pip install -r desktop/requirements.txt | Out-Null

pyinstaller `
  --name CVVR-Uploader `
  --windowed `
  --noconfirm `
  --clean `
  --add-data "yolo11s.pt;." `
  --collect-all mediapipe `
  --collect-data mediapipe `
  --collect-binaries mediapipe `
  --hidden-import=mediapipe.python._framework_bindings `
  --hidden-import=_ssl `
  --hidden-import=_hashlib `
  --collect-data certifi `
  desktop/main.py

Pop-Location
Pop-Location

Write-Host "Built app in dist/CVVR-Uploader" -ForegroundColor Green


