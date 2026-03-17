#!/usr/bin/env bash
# CropNet API — production install
set -euo pipefail

DIR=/opt/cropnet
REPO=$(cd "$(dirname "$0")/.." && pwd)

echo "==> Installing CropNet API to $DIR"
mkdir -p "$DIR"/{logs,models/general}

# Copy API code
cp -r "$REPO/api" "$DIR/"

# Python venv
python3 -m venv "$DIR/venv"
"$DIR/venv/bin/pip" install --upgrade pip -q
"$DIR/venv/bin/pip" install -r "$DIR/api/requirements.txt" -q
echo "==> Deps installed"

# .env
if [ ! -f "$DIR/.env" ]; then
    echo "CROPNET_API_KEY=$(openssl rand -hex 32)" > "$DIR/.env"
    chmod 600 "$DIR/.env"
    echo "==> .env created"
fi

chown -R www-data:www-data "$DIR"

# systemd
cp "$REPO/scripts/cropnet.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable cropnet
systemctl restart cropnet
echo "==> Service started"

echo "✅ CropNet installed. Health: curl http://127.0.0.1:8001/health"
