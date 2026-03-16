#!/usr/bin/env bash
# CropNet API — production install script
# Usage: sudo bash scripts/install.sh
set -euo pipefail

INSTALL_DIR=/opt/cropnet
MODELS_DIR=$INSTALL_DIR/models
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)

echo "==> Installing CropNet API to $INSTALL_DIR"

# 1. Create dir structure
mkdir -p "$INSTALL_DIR"/{logs,models}
cp -r "$REPO_DIR/app" "$INSTALL_DIR/"

# 2. Python venv
python3 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip --quiet
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/app/requirements.txt" --quiet
echo "==> Python deps installed"

# 3. Copy models (expects cropnet_v2.onnx + cropnet_v2.onnx.data + labels.json)
if [ -d "$REPO_DIR/models" ] && [ -f "$REPO_DIR/models/cropnet_v2.onnx" ]; then
    cp "$REPO_DIR/models/"* "$MODELS_DIR/"
    echo "==> Models copied"
else
    echo "⚠️  No models found at $REPO_DIR/models — copy manually:"
    echo "    cp /path/to/cropnet_v2.onnx $MODELS_DIR/"
    echo "    cp /path/to/cropnet_v2.onnx.data $MODELS_DIR/"
    echo "    cp /path/to/labels.json $MODELS_DIR/"
fi

# 4. Create .env file (if not exists)
if [ ! -f "$INSTALL_DIR/.env" ]; then
    echo "CROPNET_API_KEY=$(openssl rand -hex 32)" > "$INSTALL_DIR/.env"
    chmod 600 "$INSTALL_DIR/.env"
    echo "==> .env created with random API key:"
    cat "$INSTALL_DIR/.env"
fi

# 5. Set permissions
chown -R www-data:www-data "$INSTALL_DIR"

# 6. Install systemd service
cp "$REPO_DIR/scripts/cropnet.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable cropnet
systemctl restart cropnet
echo "==> Service started"

# 7. Install nginx config
cp "$REPO_DIR/nginx/cropnet.conf" /etc/nginx/sites-available/cropnet
if [ ! -L /etc/nginx/sites-enabled/cropnet ]; then
    ln -s /etc/nginx/sites-available/cropnet /etc/nginx/sites-enabled/cropnet
fi
nginx -t && systemctl reload nginx
echo "==> Nginx configured"

echo ""
echo "✅ CropNet API installed!"
echo "   Health: http://127.0.0.1:8001/health"
echo "   Docs:   https://api.cropnet.fellah.tn/docs"
echo "   API key: $(grep CROPNET_API_KEY $INSTALL_DIR/.env)"
