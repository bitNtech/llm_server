#!/usr/bin/env bash
# =============================================================================
# run.sh - lifecycle manager for the vLLM + FastAPI inference server.
#
#   ./run.sh [start]     validate -> install (idempotent) -> start engine -> start gateway,
#                        supervise both in the foreground; SIGINT/SIGTERM = graceful shutdown
#   ./run.sh install     create virtualenvs, install dependencies + the right vLLM build,
#                        download the model (no services started)
#   ./run.sh check       hardware + configuration report (alias: doctor)
#   ./run.sh test [args] run the test suite (no GPU or model needed)
#   ./run.sh stop        gracefully stop a running instance (uses run/llm-server.pid)
#   ./run.sh status      show whether an instance is running and ready
#   ./run.sh engine-cmd  print the `vllm serve` command that would be executed
#
# Environment knobs for this script (everything else lives in .env, see .env.example):
#   ENV_FILE=path        configuration file (default: ./.env; created from .env.example in dev)
#   SKIP_INSTALL=1       never install/upgrade; fail if something is missing (immutable hosts)
#   PYTHON_VERSION=3.12  interpreter for both virtualenvs (uv downloads it if needed)
#   UV_AUTO_INSTALL=0    do not bootstrap `uv`; fall back to python3 -m venv + pip
# =============================================================================
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

GW_VENV="$ROOT/.venv"
ENGINE_VENV="$ROOT/.venv-vllm"
RUN_DIR="$ROOT/run"
PIDFILE="$RUN_DIR/llm-server.pid"
ENGINE_LOG="$RUN_DIR/engine.log"
# vLLM's API server can stay alive (unhealthy) after its engine core dies during startup
ENGINE_FATAL='EngineCore failed to start|Engine core initialization failed|CUDA out of memory|No CUDA GPUs are available'
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
UV=""
ENGINE_PID=""
GATEWAY_PID=""

# ----------------------------------------------------------------------------- helpers
log() { printf '%s [run.sh] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }
warn() { log "WARNING: $*"; }
die() {
  log "ERROR: $*"
  # never leave a (setsid-detached) engine or gateway behind
  if [[ -n "${ENGINE_PID:-}${GATEWAY_PID:-}" ]] && ((!${SHUTTING_DOWN:-0})); then shutdown 1; fi
  exit 1
}
have() { command -v "$1" >/dev/null 2>&1; }
truthy() { [[ "${1,,}" =~ ^(1|true|yes|on)$ ]]; }

port_in_use() { (exec 3<>"/dev/tcp/$1/$2") 2>/dev/null; }

# Parse a dotenv file without executing it. Real environment variables take precedence.
load_env() {
  local file="${ENV_FILE:-$ROOT/.env}"
  if [[ ! -f "$file" ]]; then
    if [[ "${APP_ENV:-development}" != "production" && -z "${ENV_FILE:-}" && -f "$ROOT/.env.example" ]]; then
      cp "$ROOT/.env.example" "$ROOT/.env"
      log "created .env from .env.example (review it before deploying)"
    else
      log "no env file at $file: using process environment and defaults"
      return 0
    fi
  fi
  export LLM_SERVER_ENV_FILE="$file"
  local line key val
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    line="${line#export }"
    if [[ ! "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      warn "ignoring malformed line in $file: ${line:0:40}"
      continue
    fi
    key="${BASH_REMATCH[1]}"
    val="${BASH_REMATCH[2]}"
    if [[ "$val" =~ ^\"(.*)\"[[:space:]]*(#.*)?$ ]]; then
      val="${BASH_REMATCH[1]}"
    elif [[ "$val" =~ ^\'(.*)\'[[:space:]]*(#.*)?$ ]]; then
      val="${BASH_REMATCH[1]}"
    else
      val="${val%%[[:space:]]#*}"
      val="${val%"${val##*[![:space:]]}"}"
    fi
    [[ -n "${!key+x}" ]] && continue
    export "$key=$val"
  done <"$file"
}

preflight() {
  [[ "$(uname -s)" == "Linux" ]] || warn "only Linux is supported for production; $(uname -s) is best-effort"
  for cmd in curl setsid sha256sum; do have "$cmd" || die "'$cmd' is required (curl: apt-get/dnf install curl; setsid, sha256sum: util-linux, coreutils)"; done
  local soft hard
  soft="$(ulimit -Sn)"
  hard="$(ulimit -Hn)"
  if [[ "$soft" != "unlimited" && "$soft" -lt 65536 ]]; then
    if [[ "$hard" == "unlimited" || "$hard" -ge 65536 ]]; then ulimit -Sn 65536 2>/dev/null || true; else ulimit -Sn "$hard" 2>/dev/null || true; fi
  fi
}

# ------------------------------------------------------------------- python tooling
ensure_uv() {
  if have uv; then UV="$(command -v uv)"; return; fi
  if [[ -x "$HOME/.local/bin/uv" ]]; then UV="$HOME/.local/bin/uv"; return; fi
  if truthy "${UV_AUTO_INSTALL:-true}" && ! truthy "${SKIP_INSTALL:-0}"; then
    log "installing uv (fast Python package manager) into ~/.local/bin"
    if curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null 2>&1; then
      UV="$HOME/.local/bin/uv"
      return
    fi
    warn "could not install uv; falling back to python -m venv + pip"
  fi
  UV=""
}

find_python() {
  local candidate
  for candidate in "python$PYTHON_VERSION" python3.13 python3.12 python3.11 python3; do
    if have "$candidate" && "$candidate" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] < (3, 15) else 1)' 2>/dev/null; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

HAS_HEADERS='import os, sys, sysconfig; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)'

# make_venv DIR [headers]: "headers" = the interpreter must ship Python.h, because vLLM JIT-compiles
# C/C++ at startup (torch inductor on CPU, Triton launchers on GPU).
make_venv() {
  local dir="$1" need_headers="${2:-}"
  if [[ -x "$dir/bin/python" ]] && "$dir/bin/python" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] < (3, 15) else 1)' 2>/dev/null; then
    if [[ -z "$need_headers" ]] || "$dir/bin/python" -c "$HAS_HEADERS" 2>/dev/null; then
      return 0
    fi
    truthy "${SKIP_INSTALL:-0}" && die "$dir lacks Python development headers (Python.h) and SKIP_INSTALL is set"
    log "recreating $dir: its interpreter has no Python.h (needed for vLLM's JIT compilation)"
    rm -f "$dir/.llm-stamp"
  fi
  truthy "${SKIP_INSTALL:-0}" && die "virtualenv $dir is missing and SKIP_INSTALL is set"
  rm -rf "$dir"
  if [[ -n "$UV" ]]; then
    # uv's managed CPython builds include headers; system interpreters often do not (python3-dev)
    "$UV" venv --quiet --seed --managed-python --python "$PYTHON_VERSION" "$dir" || die "uv could not create $dir"
  else
    local py
    py="$(find_python)" || die "Python 3.11-3.14 is required (or let run.sh install uv: UV_AUTO_INSTALL=1)"
    "$py" -m venv "$dir" || die "python -m venv failed (Debian/Ubuntu: apt-get install python3-venv)"
    "$dir/bin/python" -m pip install --quiet --upgrade pip
  fi
  if [[ -n "$need_headers" ]] && ! "$dir/bin/python" -c "$HAS_HEADERS" 2>/dev/null; then
    die "Python.h not found for $("$dir/bin/python" -V): install python$PYTHON_VERSION-dev (apt) / python3-devel (dnf), or allow uv"
  fi
}

pip_install() {
  local dir="$1"
  shift
  if [[ -n "$UV" ]]; then
    VIRTUAL_ENV="$dir" "$UV" pip install --quiet --python "$dir/bin/python" "$@"
  else
    "$dir/bin/python" -m pip install --quiet "$@"
  fi
}

# --------------------------------------------------------------------- gateway venv
setup_gateway() {
  local with_dev="${1:-auto}" reqs stamp
  if [[ "$with_dev" == auto ]]; then
    [[ "${APP_ENV:-development}" == production ]] && with_dev=no || with_dev=yes
  fi
  [[ "$with_dev" == yes ]] && reqs="$ROOT/requirements-dev.txt" || reqs="$ROOT/requirements.txt"
  make_venv "$GW_VENV"
  GW_PY="$GW_VENV/bin/python"
  stamp="$(cat "$ROOT/requirements.txt" "$reqs" | sha256sum | cut -c1-16)-$with_dev"
  if [[ "$(cat "$GW_VENV/.llm-stamp" 2>/dev/null)" != "$stamp" ]]; then
    truthy "${SKIP_INSTALL:-0}" && die "gateway dependencies are out of date and SKIP_INSTALL is set"
    log "installing gateway dependencies ($(basename "$reqs"))"
    pip_install "$GW_VENV" -r "$reqs" || die "gateway dependency installation failed"
    echo "$stamp" >"$GW_VENV/.llm-stamp"
  fi
}

load_runtime_config() {
  local out
  out="$("$GW_PY" -m app.cli runtime)" || die "invalid configuration (see messages above)"
  eval "$out"
}

# ---------------------------------------------------------------------- engine venv
setup_engine() {
  local out stamp_file="$ENGINE_VENV/.llm-stamp"
  out="$("$GW_PY" -m app.cli install-plan)" || die "cannot determine which vLLM build to install"
  eval "$out"
  ENGINE_DEVICE_RESOLVED="$PLAN_DEVICE"
  have gcc || have cc || die "a C/C++ compiler is required by vLLM's JIT kernels: apt-get install build-essential / dnf groupinstall 'Development Tools'"
  [[ -x "$ENGINE_VENV/bin/python" ]] && make_venv "$ENGINE_VENV" headers
  if [[ "$(cat "$stamp_file" 2>/dev/null)" == "$PLAN_STAMP" ]] && "$ENGINE_VENV/bin/python" -c 'import vllm' 2>/dev/null; then
    log "vLLM $PLAN_STAMP present ($PLAN_DEVICE)"
  else
    truthy "${SKIP_INSTALL:-0}" && die "vLLM ($PLAN_STAMP) is not installed and SKIP_INSTALL is set"
    make_venv "$ENGINE_VENV" headers
    log "installing vLLM $PLAN_STAMP for $PLAN_DEVICE - first install downloads several GB"
    if [[ -n "$UV" ]]; then
      pip_install "$ENGINE_VENV" "${PLAN_PACKAGES[@]}" --torch-backend "$PLAN_TORCH_BACKEND" \
        || die "vLLM installation failed"
    else
      local extra=()
      [[ -n "$PLAN_PIP_EXTRA_INDEX" ]] && extra=(--extra-index-url "$PLAN_PIP_EXTRA_INDEX")
      pip_install "$ENGINE_VENV" "${PLAN_PACKAGES[@]}" "${extra[@]}" || die "vLLM installation failed"
    fi
    echo "$PLAN_STAMP" >"$stamp_file"
  fi
  verify_engine
}

verify_engine() {
  local info
  info="$("$ENGINE_VENV/bin/python" -c '
import torch, vllm
cuda = torch.cuda.is_available()
print(vllm.__version__, torch.__version__, "cuda" if cuda else "cpu", torch.cuda.device_count() if cuda else 0)
' 2>/dev/null)" || die "vLLM is installed but does not import; remove $ENGINE_VENV and re-run"
  read -r VLLM_VER TORCH_VER TORCH_DEV TORCH_GPUS <<<"$info"
  log "engine runtime: vllm $VLLM_VER, torch $TORCH_VER ($TORCH_DEV, ${TORCH_GPUS} GPU)"
  if [[ "$ENGINE_DEVICE_RESOLVED" == cuda && "$TORCH_DEV" != cuda ]]; then
    die "an NVIDIA GPU was detected but torch cannot use CUDA: check the driver (nvidia-smi) and ENGINE_CUDA_VARIANT"
  fi
}

prefetch_one() { # model revision -> prints weight bytes
  "$ENGINE_VENV/bin/python" "$ROOT/scripts/prefetch_model.py" \
    --model "$1" --revision "$2" --download-dir "$CFG_ENGINE_DOWNLOAD_DIR" \
    $(truthy "$CFG_ENGINE_TRUST_REMOTE_CODE" && echo --code) | tail -n 1
}

prefetch_model() {
  truthy "${CFG_ENGINE_MODEL_PREFETCH:-true}" || return 0
  [[ -e "$CFG_ENGINE_MODEL" ]] && return 0 # local directory
  local cache="${HF_HOME:-$HOME/.cache/huggingface}" free_gb bytes draft_bytes=0 out
  [[ -n "$CFG_ENGINE_DOWNLOAD_DIR" ]] && cache="$CFG_ENGINE_DOWNLOAD_DIR"
  mkdir -p "$cache"
  free_gb="$(df -Pk "$cache" | awk 'NR==2 {print int($4/1024/1024)}')"
  [[ "$free_gb" -lt 40 ]] && warn "only ${free_gb} GB free at $cache"
  log "ensuring model $CFG_ENGINE_MODEL is in the local cache"
  bytes="$(prefetch_one "$CFG_ENGINE_MODEL" "$CFG_ENGINE_REVISION")" && [[ "$bytes" =~ ^[0-9]+$ ]] \
    || die "model download failed (gated model? set HF_TOKEN in .env)"
  if [[ -n "$CFG_ENGINE_DRAFT_MODEL" && "$ENGINE_DEVICE_RESOLVED" == cuda && ! -e "$CFG_ENGINE_DRAFT_MODEL" ]]; then
    log "ensuring speculative-decoding drafter $CFG_ENGINE_DRAFT_MODEL is in the local cache"
    draft_bytes="$(prefetch_one "$CFG_ENGINE_DRAFT_MODEL" "$CFG_ENGINE_DRAFT_REVISION")" \
      && [[ "$draft_bytes" =~ ^[0-9]+$ ]] || die "drafter download failed"
  fi
  # Fail now with a clear message instead of an OOM deep inside vLLM after minutes of loading.
  if out="$("$GW_PY" -m app.cli fit-check $((bytes + draft_bytes)))"; then
    log "memory fit: $out"
  elif truthy "${SKIP_FIT_CHECK:-0}"; then
    warn "memory fit: $out (SKIP_FIT_CHECK set, continuing)"
  else
    die "the model does not fit: $out. Use a GPU with more memory, ENGINE_TENSOR_PARALLEL_SIZE=auto on a multi-GPU instance, or a smaller checkpoint (SKIP_FIT_CHECK=1 to override)"
  fi
}

ensure_engine_key() {
  if [[ -z "${ENGINE_API_KEY:-}" ]]; then
    ENGINE_API_KEY="$("$GW_PY" -c 'import secrets; print(secrets.token_urlsafe(32))')"
    export ENGINE_API_KEY # shared by gateway and engine for this run only
  fi
}

cpu_preload() {
  # vLLM's CPU backend wants Intel OpenMP (and tcmalloc when present) preloaded.
  local libs=() lib
  lib="$(find "$ENGINE_VENV" -name 'libiomp5.so' -print -quit 2>/dev/null || true)"
  [[ -n "$lib" ]] && libs+=("$lib")
  for lib in /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 /usr/lib64/libtcmalloc_minimal.so.4 \
    /usr/lib/aarch64-linux-gnu/libtcmalloc_minimal.so.4; do
    [[ -f "$lib" ]] && libs+=("$lib") && break
  done
  local IFS=:
  echo "${libs[*]}"
}

# ----------------------------------------------------------------------- processes
start_engine() {
  port_in_use "$CFG_ENGINE_HOST" "$CFG_ENGINE_PORT" && die "engine port $CFG_ENGINE_PORT is already in use"
  case "$CFG_ENGINE_MODE" in
    managed)
      local argv_file="$RUN_DIR/engine.argv" env_out preload=""
      "$GW_PY" -m app.cli engine-args >"$argv_file" || die "cannot build the vLLM command line"
      mapfile -d '' -t ENGINE_ARGV <"$argv_file"
      env_out="$("$GW_PY" -m app.cli engine-env)" || die "cannot build the engine environment"
      [[ "$ENGINE_DEVICE_RESOLVED" == cpu ]] && preload="$(cpu_preload)"
      log "starting vLLM: vllm ${ENGINE_ARGV[*]}  (log: $ENGINE_LOG)"
      : >"$ENGINE_LOG"
      (
        eval "$env_out"
        export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}" DO_NOT_TRACK="${DO_NOT_TRACK:-1}"
        [[ -n "$preload" ]] && export LD_PRELOAD="$preload${LD_PRELOAD:+:$LD_PRELOAD}"
        exec setsid "$ENGINE_VENV/bin/vllm" "${ENGINE_ARGV[@]}" > >(tee -a "$ENGINE_LOG") 2>&1
      ) &
      ENGINE_PID=$!
      ;;
    mock)
      log "starting mock engine (no model; deterministic replies)"
      (
        export VLLM_API_KEY="$ENGINE_API_KEY" MOCK_MODEL="${ENGINE_SERVED_MODEL_NAME:-$CFG_ENGINE_MODEL}"
        exec setsid "$GW_PY" -m app.mock_engine --host "$CFG_ENGINE_HOST" --port "$CFG_ENGINE_PORT"
      ) &
      ENGINE_PID=$!
      ;;
  esac
  echo "$$ ${ENGINE_PID:-0} 0" >"$PIDFILE"
}

wait_engine() {
  local timeout="$CFG_ENGINE_STARTUP_TIMEOUT_S" start=$SECONDS next_note=$((SECONDS + 30))
  [[ "$CFG_ENGINE_MODE" == mock ]] && timeout=30
  log "waiting for the engine at $CFG_ENGINE_URL (timeout ${timeout}s)"
  while ((SECONDS - start < timeout)); do
    if [[ -n "$ENGINE_PID" ]] && ! kill -0 "$ENGINE_PID" 2>/dev/null; then
      local rc=0
      wait "$ENGINE_PID" || rc=$?
      ENGINE_PID=""
      die "engine exited during startup (exit code $rc) - see its log output above"
    fi
    if curl -fsS -m 3 "$CFG_ENGINE_URL/health" >/dev/null 2>&1; then
      log "engine healthy after $((SECONDS - start))s"
      return 0
    fi
    if [[ "$CFG_ENGINE_MODE" == managed && -f "$ENGINE_LOG" ]] && grep -qE "$ENGINE_FATAL" "$ENGINE_LOG"; then
      log "engine failed during startup; root cause from $ENGINE_LOG:"
      grep -E "Error|error:|$ENGINE_FATAL" "$ENGINE_LOG" | grep -vE "^\s*$" | tail -n 6 >&2 || true
      ENGINE_STOP_TIMEOUT_S=5 # it is already broken; do not wait for a graceful exit
      die "engine failed to start (full log: $ENGINE_LOG)"
    fi
    if ((SECONDS >= next_note)); then
      log "still loading the model... ($((SECONDS - start))s)"
      next_note=$((SECONDS + 30))
    fi
    sleep 1 &
    wait $! || true
  done
  [[ "$CFG_ENGINE_MODE" == external ]] && { warn "external engine not healthy yet; gateway will report not-ready"; return 0; }
  die "engine not healthy after ${timeout}s (raise ENGINE_STARTUP_TIMEOUT_S for large models/slow disks)"
}

start_gateway() {
  port_in_use 127.0.0.1 "$CFG_PORT" && die "gateway port $CFG_PORT is already in use"
  log "starting gateway on $CFG_HOST:$CFG_PORT"
  (exec "$GW_PY" -m app) &
  GATEWAY_PID=$!
  echo "$$ ${ENGINE_PID:-0} $GATEWAY_PID" >"$PIDFILE"
  local start=$SECONDS limit=$((60 + ${CFG_ENGINE_WARMUP_TIMEOUT_S:-0})) next_note=$((SECONDS + 30))
  ((${CFG_ENGINE_WARMUP_TIMEOUT_S:-0} > 0)) && log "gateway warming up the engine before reporting ready"
  while ((SECONDS - start < limit)); do
    kill -0 "$GATEWAY_PID" 2>/dev/null || die "gateway exited during startup"
    if curl -fsS -m 2 "http://127.0.0.1:$CFG_PORT/health/ready" >/dev/null 2>&1; then
      log "READY after $((SECONDS - start))s - http://$CFG_HOST:$CFG_PORT  (OpenAI: /v1/chat/completions, sessions: /v1/sessions, ws: /v1/realtime)"
      return 0
    fi
    if ((SECONDS >= next_note)); then
      log "still warming up... ($((SECONDS - start))s)"
      next_note=$((SECONDS + 30))
    fi
    sleep 0.5
  done
  warn "gateway is up but not ready yet (engine not serving?) - check /health"
}

stop_pid() {
  local pid="$1" name="$2" timeout="$3" group="${4:-no}" i
  [[ -z "$pid" || "$pid" == 0 ]] && return 0
  kill -0 "$pid" 2>/dev/null || return 0
  log "stopping $name (pid $pid, up to ${timeout}s)"
  kill -TERM "$pid" 2>/dev/null || true
  for ((i = 0; i < timeout * 10; i++)); do
    kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return 0; }
    sleep 0.1
  done
  warn "$name did not exit in ${timeout}s; killing"
  if [[ "$group" == yes ]]; then kill -KILL -- "-$pid" 2>/dev/null || true; fi
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

SHUTTING_DOWN=0
shutdown() {
  local code="${1:-0}"
  ((SHUTTING_DOWN)) && return
  SHUTTING_DOWN=1
  trap - TERM INT HUP
  # gateway first: it drains in-flight responses while the engine is still serving them
  stop_pid "$GATEWAY_PID" gateway $((${CFG_SHUTDOWN_GRACE_S:-20} + 15))
  stop_pid "$ENGINE_PID" engine "${ENGINE_STOP_TIMEOUT_S:-45}" yes
  rm -f "$PIDFILE"
  log "stopped"
  exit "$code"
}

supervise() {
  trap 'log "received SIGTERM"; shutdown 0' TERM
  trap 'log "received SIGINT"; shutdown 0' INT
  trap 'log "received SIGHUP"; shutdown 0' HUP
  while true; do
    sleep 2 &
    wait $! || true
    if [[ -n "$ENGINE_PID" ]] && ! kill -0 "$ENGINE_PID" 2>/dev/null; then
      log "engine process exited unexpectedly"
      ENGINE_PID=""
      shutdown 1
    fi
    if ! kill -0 "$GATEWAY_PID" 2>/dev/null; then
      log "gateway process exited unexpectedly"
      GATEWAY_PID=""
      shutdown 1
    fi
  done
}

already_running() {
  [[ -f "$PIDFILE" ]] || return 1
  local sup
  read -r sup _ _ <"$PIDFILE" || return 1
  [[ -n "$sup" ]] && kill -0 "$sup" 2>/dev/null
}

# ------------------------------------------------------------------------ commands
cmd_install() {
  preflight
  ensure_uv
  setup_gateway
  load_runtime_config
  if [[ "$CFG_ENGINE_MODE" == managed ]]; then
    setup_engine
    prefetch_model
  fi
  log "install complete"
}

cmd_start() {
  already_running && die "already running (pid $(cut -d' ' -f1 "$PIDFILE")); use ./run.sh stop"
  mkdir -p "$RUN_DIR"
  preflight
  ensure_uv
  setup_gateway
  load_runtime_config
  "$GW_PY" -m app.cli doctor || warn "doctor reported problems (continuing)"
  [[ "$CFG_APP_ENV" == production && "$CFG_AUTH_ENABLED" != true ]] && warn "production without API_KEYS (ALLOW_UNAUTHENTICATED=true)"
  # From here on, Ctrl-C / SIGTERM must stop whatever has been started.
  trap 'log "interrupted during startup"; shutdown 130' INT TERM HUP
  if [[ "$CFG_ENGINE_MODE" == managed ]]; then
    setup_engine
    prefetch_model
  fi
  if [[ "$CFG_ENGINE_MODE" != external ]]; then
    ensure_engine_key
    start_engine
  fi
  wait_engine
  start_gateway
  supervise
}

cmd_check() {
  preflight
  ensure_uv
  setup_gateway
  load_runtime_config
  "$GW_PY" -m app.cli doctor "$@"
}

cmd_test() {
  ensure_uv
  setup_gateway yes
  exec "$GW_PY" -m pytest "$@"
}

cmd_stop() {
  if ! already_running; then
    log "not running"
    rm -f "$PIDFILE"
    return 0
  fi
  local sup i
  read -r sup _ _ <"$PIDFILE"
  log "sending SIGTERM to supervisor $sup"
  kill -TERM "$sup"
  for ((i = 0; i < 1200; i++)); do
    kill -0 "$sup" 2>/dev/null || { log "stopped"; return 0; }
    sleep 0.1
  done
  die "supervisor $sup still running after 120s"
}

cmd_status() {
  if ! already_running; then
    echo "stopped"
    return 3
  fi
  local sup eng gw
  read -r sup eng gw <"$PIDFILE"
  echo "running: supervisor=$sup engine=$eng gateway=$gw"
  setup_gateway >/dev/null 2>&1 && load_runtime_config
  curl -fsS -m 3 "http://127.0.0.1:${CFG_PORT:-8000}/health/ready" && echo || { echo "not ready"; return 1; }
}

cmd_engine_cmd() {
  ensure_uv
  setup_gateway >/dev/null
  local args=()
  mapfile -d '' -t args < <("$GW_PY" -m app.cli engine-args)
  printf 'vllm'
  printf ' %q' "${args[@]}"
  printf '\n'
}

main() {
  local command="${1:-start}"
  [[ $# -gt 0 ]] && shift
  case "$command" in
    -h | --help | help) sed -n '3,21p' "$0" | sed 's/^# \{0,1\}//'; return 0 ;;
  esac
  load_env
  case "$command" in
    start) cmd_start ;;
    install) cmd_install ;;
    check | doctor) cmd_check "$@" ;;
    test) cmd_test "$@" ;;
    stop) cmd_stop ;;
    status) cmd_status ;;
    engine-cmd) cmd_engine_cmd ;;
    *) die "unknown command '$command' (try ./run.sh help)" ;;
  esac
}

main "$@"
