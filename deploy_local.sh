#!/usr/bin/env bash
set -euo pipefail

SERVER="103.195.244.67"
USER="root"
PASS='Login@123@@@'
APP_DIR="/opt/poc2"

# 1) Prereq for non-interactive SSH
if ! command -v sshpass >/dev/null 2>&1; then
  brew install hudochenkov/sshpass/sshpass
fi

# 2) Package and upload code (keep yolov8l.pt; skip caches/outputs)
tar czf /tmp/poc2.tgz --exclude '.git' --exclude '__pycache__' --exclude 'output' -C "$(pwd)" .

sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no ${USER}@${SERVER} "mkdir -p ${APP_DIR}"
sshpass -p "$PASS" scp -o StrictHostKeyChecking=no /tmp/poc2.tgz ${USER}@${SERVER}:/tmp/poc2.tgz

# 3) Remote install, venv, requirements, systemd, start
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no ${USER}@${SERVER} bash -s <<'REMOTE_EOF'
set -euo pipefail
APP_DIR="/opt/poc2"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-dev build-essential libgl1 libglib2.0-0 tesseract-ocr ffmpeg
elif command -v yum >/dev/null 2>&1; then
  yum install -y python3 python3-venv python3-devel gcc gcc-c++ glib2 glib2-devel tesseract ffmpeg
fi

mkdir -p "$APP_DIR"
tar xzf /tmp/poc2.tgz -C "$APP_DIR"

python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip setuptools wheel
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt"

cat >/etc/systemd/system/poc2.service <<'UNIT'
[Unit]
Description=POC_2 Gunicorn Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/poc2
Environment=PYTHONUNBUFFERED=1
Environment=FLASK_ENV=production
ExecStart=/opt/poc2/venv/bin/gunicorn -c gunicorn.conf.py wsgi:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now poc2

# Quick health check on :8000
if command -v ss >/dev/null 2>&1; then
  ss -ltnp | grep ':8000' || (journalctl -u poc2 --no-pager -n 100; exit 1)
else
  netstat -ltnp | grep ':8000' || (journalctl -u poc2 --no-pager -n 100; exit 1)
fi
REMOTE_EOF

echo "Deployed. Try: curl http://${SERVER}:8000/health || open http://${SERVER}:8000/"
