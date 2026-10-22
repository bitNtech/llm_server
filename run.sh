#!/usr/bin/env bash
# AICA LLM server - lifecycle helper (the same on a laptop and on a VM; containers run `python -m app`).
#
#   ./run.sh                 install if needed, validate configuration, run the server in the foreground
#   ./run.sh dev             same, with a local Firestore emulator when FIRESTORE_EMULATOR_HOST is unset
#   ./run.sh install [--dev] create .venv and install dependencies (--dev adds test tooling)
#   ./run.sh check           validate configuration and Google credentials, print the effective setup
#   ./run.sh test [args]     unit + integration tests (Firestore emulator + fake Vertex; no cloud access)
#   ./run.sh test-live       live tests against real Vertex AI (needs credentials; costs a few cents)
#   ./run.sh emulator [port] run the Firestore emulator in the foreground
#
# Configuration comes from the environment and .env (see .env.example). SIGTERM/SIGINT drain the
# server gracefully (SHUTDOWN_GRACE_S): in-flight answers finish, WebSockets are told to reconnect.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
cd "$ROOT"

log() { printf '%s [run.sh] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

install() {
  local dev="${1:-}"
  if [[ ! -x "$PY" ]]; then
    if command -v uv >/dev/null 2>&1; then
      uv venv -q -p 3.12 "$VENV"
    else
      command -v python3 >/dev/null || die "python3 (>= 3.11) or uv is required"
      python3 -m venv "$VENV"
    fi
  fi
  local req="requirements.txt"
  [[ "$dev" == "--dev" ]] && req="requirements-dev.txt"
  local stamp="$VENV/.installed-$(sha256sum "$req" requirements.txt constraints.txt | sha256sum | cut -c1-16)"
  if [[ ! -f "$stamp" ]]; then
    log "installing dependencies ($req)"
    if command -v uv >/dev/null 2>&1; then
      VIRTUAL_ENV="$VENV" uv pip install -q -r "$req" -c constraints.txt
    else
      "$PY" -m pip install -q --upgrade pip
      "$PY" -m pip install -q -r "$req" -c constraints.txt
    fi
    touch "$stamp"
  fi
}

check() {
  "$PY" - <<'PY'
import sys
from pydantic import ValidationError
from app.config import Settings
try:
    s = Settings()
except ValidationError as exc:
    print(f"configuration error:\n{exc}", file=sys.stderr)
    sys.exit(2)
from app.gcp.credentials import load_credentials
try:
    creds, project = load_credentials(s.gcp_credentials_json.get_secret_value())
    cred_info = f"{type(creds).__name__} (project from credentials: {project or '-'})"
except Exception as exc:  # noqa: BLE001
    cred_info = f"NOT AVAILABLE: {exc}"
fs_cred_info = "same as Vertex"
if s.firestore_credentials_json.get_secret_value():
    try:
        fs_creds, fs_project = load_credentials(s.firestore_credentials_json.get_secret_value())
        fs_cred_info = f"{type(fs_creds).__name__} (project from credentials: {fs_project or '-'})"
        s.firestore_project_id = s.firestore_project_id or fs_project or ""
    except Exception as exc:  # noqa: BLE001
        fs_cred_info = f"NOT AVAILABLE: {exc}"
import os
print(f"env              {s.app_env}  (instance {s.instance_id})")
print(f"listen           {s.host}:{s.port}")
print(f"gcp project      {s.gcp_project_id or '(from credentials)'}")
print(f"credentials      {cred_info}")
print(f"vertex model     {s.vertex_model}  fallbacks={s.fallback_model_list}  locations={s.location_list}")
print(f"thinking         {s.vertex_reasoning_effort or 'model default'} (after tool results: "
      f"{s.vertex_tool_followup_reasoning_effort or 'unchanged'})")
print(f"firestore        database={s.firestore_database} project={s.firestore_project or '(gcp project)'} "
      f"emulator={os.environ.get('FIRESTORE_EMULATOR_HOST') or 'no'}")
print(f"firestore creds  {fs_cred_info}")
print(f"admin keys       {len(s.admin_key_list)}   secrets key: {'set' if s.encryption_keys else 'EPHEMERAL'}"
      f"   realtime token secret: {'set' if s.realtime_token_secret.get_secret_value() else 'EPHEMERAL'}")
print(f"tracing          {s.otel_exporter_otlp_endpoint or 'off'}   metrics: {'on' if s.metrics_enabled else 'off'}")
PY
}

emulator_pid=""
cleanup() { [[ -n "$emulator_pid" ]] && kill "$emulator_pid" 2>/dev/null || true; }

start() {
  install
  check
  log "starting server"
  exec "$PY" -m app
}

dev() {
  install --dev
  if [[ -z "${FIRESTORE_EMULATOR_HOST:-}" ]]; then
    local port="${FIRESTORE_EMULATOR_PORT:-8686}"
    "$ROOT/scripts/firestore-emulator.sh" start "$port" >/dev/null 2>&1 &
    emulator_pid=$!
    trap cleanup EXIT
    export FIRESTORE_EMULATOR_HOST="127.0.0.1:$port"
    for _ in $(seq 1 60); do curl -fs "http://$FIRESTORE_EMULATOR_HOST/" >/dev/null 2>&1 && break; sleep 0.5; done
    log "Firestore emulator on $FIRESTORE_EMULATOR_HOST"
  fi
  check
  "$PY" -m app
}

case "${1:-start}" in
  start) start ;;
  dev) dev ;;
  install) shift; install "${1:-}" ;;
  check) install; check ;;
  test) shift; install --dev; "$ROOT/scripts/firestore-emulator.sh" install; exec "$PY" -m pytest "$@" ;;
  test-live) shift; install --dev
    export LIVE_VERTEX=1 LIVE_GCP_PROJECT="${LIVE_GCP_PROJECT:-${GCP_PROJECT_ID:-}}"
    [[ -n "$LIVE_GCP_PROJECT" ]] || die "set LIVE_GCP_PROJECT (or GCP_PROJECT_ID) to the Vertex project"
    exec "$PY" -m pytest -m live tests/live -s "$@" ;;
  emulator) shift; exec "$ROOT/scripts/firestore-emulator.sh" start "$@" ;;
  *) die "unknown command '$1' (start | dev | install | check | test | test-live | emulator)" ;;
esac
