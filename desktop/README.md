## CVVR Desktop Uploader (macOS + Windows)

### Features
- Login page: enter Server URL and optional API token; checks `/health`.
- Upload page: select `tripId` and local video file; shows upload progress; starts server processing and returns to upload screen.

### Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r desktop/requirements.txt
python desktop/app.py
```

### Build (PyInstaller)
Install PyInstaller once:
```bash
pip install pyinstaller
```

macOS build:
```bash
bash desktop/build_mac.sh
```

Windows build (PowerShell):
```powershell
desktop\\build_win.ps1
```

Artifacts will be under `dist/`.

### Build on GitHub (Windows .exe)
A GitHub Actions workflow builds the Windows app and uploads it as an artifact.

Steps:
- Push your changes to `main` (or trigger manually in Actions via "Run workflow").
- Open GitHub → Actions → "Build Windows Desktop App" → select the latest run.
- Download artifact `CVVR-Uploader-Windows` → it contains the built `.exe` folder.
- Zip and send that folder to the user; they can run `CVVR-Uploader.exe`.

### Notes
- The app posts to your existing backend `POST /api/jobs` and relies on server-side processing.
- API token is optional; if your server enforces auth, it is sent as `Authorization: Bearer <token>`.
- Videos can be large; upload progress is shown and the job is started once upload completes.

