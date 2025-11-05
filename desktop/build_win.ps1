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
  --add-data=desktop;desktop `
  desktop/app.py

Pop-Location
Pop-Location

Write-Host "Built app in dist/CVVR-Uploader" -ForegroundColor Green


