Param(
    [string]$Version = "1.0.0"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir '..')
Set-Location $RepoRoot

Write-Host "[build] Python environment info" -ForegroundColor Cyan
python --version
pip --version

Write-Host "[build] Installing dependencies" -ForegroundColor Cyan
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install pyinstaller

Write-Host "[build] Building PyInstaller one-folder executable" -ForegroundColor Cyan
pyinstaller cvvr_server.spec

$DistDir = Join-Path $RepoRoot 'dist/cvvr_server'
if (-not (Test-Path $DistDir)) {
    throw "PyInstaller output not found at $DistDir"
}

Write-Host "[build] PyInstaller output: $DistDir" -ForegroundColor Green

$IssPath = Join-Path $RepoRoot 'installer/cvvr_server.iss'
if (-not (Test-Path $IssPath)) {
    throw "Installer script not found at $IssPath"
}

function Get-Iscc {
    try { return (Get-Command iscc.exe -ErrorAction Stop).Source } catch { return $null }
}

$Iscc = Get-Iscc
if ($Iscc) {
    Write-Host "[installer] Compiling Inno Setup: $IssPath" -ForegroundColor Cyan
    & $Iscc $IssPath | Write-Output
    $OutputDir = Join-Path $RepoRoot 'installer/output'
    if (Test-Path $OutputDir) {
        $Installer = Get-ChildItem -Path $OutputDir -Filter '*.exe' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($Installer) {
            Write-Host "[installer] Created: $($Installer.FullName)" -ForegroundColor Green
        } else {
            Write-Warning "[installer] No installer .exe found in $OutputDir"
        }
    } else {
        Write-Warning "[installer] Output directory not found: $OutputDir"
    }
} else {
    Write-Warning "[installer] Inno Setup (iscc.exe) not found in PATH. Skipping installer build. Install from https://jrsoftware.org/isinfo.php or via Chocolatey: 'choco install innosetup -y'"
}

Write-Host "[done] Build steps complete" -ForegroundColor Green


