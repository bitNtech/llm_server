#!/usr/bin/env bash
# Install the server on this VM: a systemd unit that starts on every boot and
# restarts if it exits, and a Caddy site that serves it with TLS on
# LLM_PUBLIC_DOMAIN (from .env).
#
#   ./run.sh install && sudo deploy/vm/install.sh
#
# Idempotent: rerun after changing LLM_PUBLIC_DOMAIN or PORT in .env.

set -euo pipefail
umask 022  # caddy runs as its own user and must read what this writes
[[ $EUID -eq 0 ]] || { echo "install.sh: run with sudo" >&2; exit 1; }

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_USER="${RUN_USER:-${SUDO_USER:-$(stat -c %U "$APP_DIR")}}"
[[ -x "$APP_DIR/.venv/bin/python" ]] || { echo "install.sh: no venv; run ./run.sh install first" >&2; exit 1; }

env_value() { sed -n "s/^$1=\([^ #]*\).*/\1/p" "$APP_DIR/.env" 2>/dev/null | tail -n1; }
DOMAIN="$(env_value LLM_PUBLIC_DOMAIN)"
PORT="$(env_value PORT)"
PORT="${PORT:-8000}"
[[ -n "$DOMAIN" ]] || { echo "install.sh: set LLM_PUBLIC_DOMAIN in .env" >&2; exit 1; }

command -v caddy >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq caddy; }
install -d /etc/caddy/sites
grep -q '^import /etc/caddy/sites/\*\.caddy' /etc/caddy/Caddyfile 2>/dev/null \
    || echo 'import /etc/caddy/sites/*.caddy' >> /etc/caddy/Caddyfile

sed -e "s|@APP_DIR@|$APP_DIR|g" -e "s|@USER@|$RUN_USER|g" \
    "$APP_DIR/deploy/vm/aica-llm-server.service" > /etc/systemd/system/aica-llm-server.service
sed -e "s|@DOMAIN@|$DOMAIN|g" -e "s|@PORT@|$PORT|g" \
    "$APP_DIR/deploy/vm/Caddyfile" > /etc/caddy/sites/aica-llm.caddy
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null

systemctl daemon-reload
systemctl enable aica-llm-server.service caddy.service
systemctl reload-or-restart caddy.service

echo "install.sh: installed for $RUN_USER in $APP_DIR, serving https://$DOMAIN -> 127.0.0.1:$PORT"
echo "  start now:   sudo systemctl restart aica-llm-server"
echo "  logs:        journalctl -u aica-llm-server -f"
