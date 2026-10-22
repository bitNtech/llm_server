#!/usr/bin/env bash
# Firestore emulator for local development and the test suite (no Google Cloud project needed).
#
#   scripts/firestore-emulator.sh install          download a JRE + the emulator into ~/.cache/llm-server (checksummed)
#   scripts/firestore-emulator.sh start [port]     run it in the foreground (default 8686)
#
# Then: export FIRESTORE_EMULATOR_HOST=127.0.0.1:8686  (the test suite starts its own copy automatically)
set -euo pipefail

CACHE="${LLM_SERVER_CACHE:-$HOME/.cache/llm-server}"
EMULATOR_VERSION="${FIRESTORE_EMULATOR_VERSION:-1.22.0}"
JAR="$CACHE/cloud-firestore-emulator-v${EMULATOR_VERSION}.jar"
JRE_URL="https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jre/hotspot/normal/eclipse"
JAR_URL="https://storage.googleapis.com/firebase-preview-drop/emulator/cloud-firestore-emulator-v${EMULATOR_VERSION}.jar"

java_bin() {
  if [[ -x "$CACHE/jre/bin/java" ]]; then echo "$CACHE/jre/bin/java"; return; fi
  command -v java || true
}

install() {
  mkdir -p "$CACHE"
  if [[ -z "$(java_bin)" ]]; then
    echo "downloading Temurin JRE 21 ..." >&2
    local meta sha
    meta=$(curl -fsSL "https://api.adoptium.net/v3/assets/latest/21/hotspot?architecture=x64&image_type=jre&os=linux&vendor=eclipse")
    sha=$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())[0]["binary"]["package"]["checksum"])' <<<"$meta")
    curl -fsSL -o "$CACHE/jre.tar.gz" "$JRE_URL"
    echo "$sha  $CACHE/jre.tar.gz" | sha256sum -c - >&2
    mkdir -p "$CACHE/jre" && tar -xzf "$CACHE/jre.tar.gz" -C "$CACHE/jre" --strip-components=1 && rm "$CACHE/jre.tar.gz"
  fi
  if [[ ! -f "$JAR" ]]; then
    echo "downloading Firestore emulator v${EMULATOR_VERSION} ..." >&2
    local md5
    md5=$(curl -fsSL "https://storage.googleapis.com/storage/v1/b/firebase-preview-drop/o/emulator%2Fcloud-firestore-emulator-v${EMULATOR_VERSION}.jar?fields=md5Hash" \
      | python3 -c 'import json,sys,base64; print(base64.b64decode(json.load(sys.stdin)["md5Hash"]).hex())')
    curl -fsSL -o "$JAR.part" "$JAR_URL"
    echo "$md5  $JAR.part" | md5sum -c - >&2
    mv "$JAR.part" "$JAR"
  fi
  echo "installed: $(java_bin) $JAR" >&2
}

start() {
  local port="${1:-8686}"
  [[ -f "$JAR" && -n "$(java_bin)" ]] || install
  echo "Firestore emulator on 127.0.0.1:${port}  (export FIRESTORE_EMULATOR_HOST=127.0.0.1:${port})" >&2
  exec "$(java_bin)" -jar "$JAR" --host=127.0.0.1 --port="$port"
}

case "${1:-}" in
  install) install ;;
  start) shift; start "$@" ;;
  *) echo "usage: $0 install | start [port]" >&2; exit 2 ;;
esac
