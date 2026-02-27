#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HOME/code/python/venvs/denv"
NGINX_CONF="ai.neodafer.com"
SERVICE_FILE="ai.service"

# Activate venv and install dependencies
source "$VENV_DIR/bin/activate"
# uv pip install -r "$SCRIPT_DIR/environment/requirements.txt"

# Nginx: symlink config to sites-available, then enable via sites-enabled
sudo ln -sf "$SCRIPT_DIR/$NGINX_CONF" "/etc/nginx/sites-available/$NGINX_CONF"
sudo ln -sf "/etc/nginx/sites-available/$NGINX_CONF" "/etc/nginx/sites-enabled/$NGINX_CONF"
sudo nginx -t && sudo systemctl reload nginx

# TLS: obtain cert if not already present (requires DNS pointing to this server)
if ! sudo certbot certificates 2>/dev/null | grep -q "ai.neodafer.com"; then
    sudo certbot --nginx -d ai.neodafer.com --non-interactive --agree-tos -m dafer@neodafer.com
else
    echo "Certificate for ai.neodafer.com already exists, skipping certbot."
fi

# Systemd: symlink service file and enable
sudo ln -sf "$SCRIPT_DIR/$SERVICE_FILE" "/etc/systemd/system/$SERVICE_FILE"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_FILE"
sudo systemctl start "$SERVICE_FILE"

echo "Setup complete for dafer-ai"
