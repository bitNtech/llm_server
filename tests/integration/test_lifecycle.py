"""End-to-end lifecycle of run.sh in mock mode: startup, graceful drain on SIGTERM, supervision."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest

from tests.conftest import free_port

ROOT = Path(__file__).resolve().parents[2]
PIDFILE = ROOT / "run" / "llm-server.pid"

pytestmark = pytest.mark.skipif(PIDFILE.exists(), reason="an instance of run.sh is already running here")


def start_run_sh(tmp_path: Path) -> tuple[subprocess.Popen, str, int]:
    env_file = tmp_path / "lifecycle.env"
    env_file.write_text("# empty on purpose\n")
    port, engine_port = free_port(), free_port()
    env = {
        "PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "ENV_FILE": str(env_file), "SKIP_INSTALL": "1",
        "ENGINE_MODE": "mock", "PORT": str(port), "ENGINE_PORT": str(engine_port), "HOST": "127.0.0.1",
        "SHUTDOWN_GRACE_S": "8", "LOG_LEVEL": "WARNING", "MOCK_TOKEN_DELAY_S": "0.25",
    }
    proc = subprocess.Popen([str(ROOT / "run.sh"), "start"], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 60
    while True:
        assert proc.poll() is None, proc.stdout.read() if proc.stdout else "run.sh exited"
        try:
            if httpx.get(f"{url}/health/ready", timeout=1).status_code == 200:
                return proc, url, engine_port
        except httpx.HTTPError:
            pass
        assert time.monotonic() < deadline, "run.sh did not become ready"
        time.sleep(0.2)


def port_free(port: int) -> bool:
    try:
        httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
    except httpx.ConnectError:
        return True
    except httpx.HTTPError:
        return False
    return False


def test_sigterm_drains_inflight_response_then_stops_everything(tmp_path: Path):
    proc, url, engine_port = start_run_sh(tmp_path)
    result: dict = {}

    def stream():
        with httpx.stream("POST", f"{url}/v1/chat/completions", timeout=30, json={
                "stream": True, "messages": [{"role": "user", "content": "drain me please"}]}) as r:
            result["lines"] = [line for line in r.iter_lines() if line.startswith("data: ")]

    t = threading.Thread(target=stream)
    t.start()
    time.sleep(0.8)  # mid-stream
    proc.send_signal(signal.SIGTERM)
    t.join(30)
    assert result["lines"][-1] == "data: [DONE]"  # the caller heard the whole reply
    assert not any('"error"' in line for line in result["lines"])
    assert proc.wait(timeout=60) == 0
    assert not PIDFILE.exists()
    assert port_free(engine_port)


def test_engine_crash_stops_gateway_and_exits_nonzero(tmp_path: Path):
    proc, url, engine_port = start_run_sh(tmp_path)
    _, engine_pid, gateway_pid = (int(x) for x in PIDFILE.read_text().split())
    os.kill(engine_pid, signal.SIGKILL)
    assert proc.wait(timeout=60) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(gateway_pid, 0)
    assert not PIDFILE.exists()
