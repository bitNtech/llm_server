"""Shared fixtures. Integration tests run real sockets end to end:

    test client -> gateway (uvicorn thread) -> fake Vertex / MCP server / HTTP tool target (uvicorn threads)
                                            -> Firestore emulator (Java subprocess)

The Firestore emulator is started once per test session (scripts/firestore-emulator.sh installs it
into ~/.cache/llm-server). Every gateway gets its own collection prefix, so tests never share data.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

from app.config import Settings

for _name in list(Settings.model_fields):  # keep the developer's shell / .env out of the tests
    os.environ.pop(_name.upper(), None)
os.environ.pop("LLM_SERVER_ENV_FILE", None)

ADMIN_KEY = "admin-test-key-0123456789abcdef"
CACHE = Path(os.environ.get("LLM_SERVER_CACHE", Path.home() / ".cache" / "llm-server"))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LiveServer:
    def __init__(self, app: Any, port: int | None = None) -> None:
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.ws_url = f"ws://127.0.0.1:{self.port}"
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning", log_config=None,
                                ws_ping_interval=None, lifespan="on", timeout_graceful_shutdown=5)
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> LiveServer:
        self.thread.start()
        deadline = time.monotonic() + 20
        while not self.server.started:
            if time.monotonic() > deadline or not self.thread.is_alive():
                raise RuntimeError("server failed to start")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=30)


# ------------------------------------------------------------------ emulator
def _emulator_command(port: int) -> list[str] | None:
    java = os.environ.get("JAVA_BIN") or (str(CACHE / "jre" / "bin" / "java") if (CACHE / "jre").exists()
                                          else shutil.which("java"))
    jars = sorted(CACHE.glob("cloud-firestore-emulator-*.jar"))
    jar = os.environ.get("FIRESTORE_EMULATOR_JAR") or (str(jars[-1]) if jars else None)
    if not java or not jar:
        return None
    return [java, "-jar", jar, "--host=127.0.0.1", f"--port={port}"]


@pytest.fixture(scope="session")
def firestore_emulator() -> Iterator[str]:
    existing = os.environ.get("FIRESTORE_EMULATOR_HOST")
    if existing:
        yield existing
        return
    port = free_port()
    command = _emulator_command(port)
    if command is None:
        pytest.skip("Firestore emulator not installed: run scripts/firestore-emulator.sh install")
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    host = f"127.0.0.1:{port}"
    deadline = time.monotonic() + 60
    while True:
        try:
            if httpx.get(f"http://{host}/", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline or proc.poll() is not None:
            proc.kill()
            raise RuntimeError("Firestore emulator did not start")
        time.sleep(0.2)
    os.environ["FIRESTORE_EMULATOR_HOST"] = host
    yield host
    os.environ.pop("FIRESTORE_EMULATOR_HOST", None)
    proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()


# -------------------------------------------------------------- fake servers
@pytest.fixture(scope="session")
def fake_vertex() -> Iterator[LiveServer]:
    from tests.support import fake_vertex as fv

    server = LiveServer(fv.app).start()
    yield server
    server.stop()


@pytest.fixture
def vertex(fake_vertex: LiveServer) -> Iterator[Any]:
    """Control client for the fake Vertex endpoint; state is reset for every test."""
    client = httpx.Client(base_url=fake_vertex.url, timeout=10)
    client.post("/control", json={"reset": True})
    yield client
    client.close()


@pytest.fixture(scope="session")
def mcp_server() -> Iterator[LiveServer]:
    from tests.support.mcp_server import build_app

    server = LiveServer(build_app()).start()
    yield server
    server.stop()


@pytest.fixture(scope="session")
def tool_target() -> Iterator[LiveServer]:
    from tests.support import tool_target as tt

    server = LiveServer(tt.app).start()
    yield server
    server.stop()


@pytest.fixture
def target(tool_target: LiveServer) -> Iterator[httpx.Client]:
    client = httpx.Client(base_url=tool_target.url, timeout=10)
    client.post("/reset")
    yield client
    client.close()


# ------------------------------------------------------------------- gateway
def make_settings(vertex_url: str, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "gcp_project_id": "test-project",
        "vertex_endpoint_template": vertex_url + "/v1/projects/{project}/locations/{location}/endpoints/openapi",
        "vertex_locations": "asia-south1,global",
        "vertex_http2": False,
        "vertex_keepalive_interval_s": 0,
        "vertex_retry_backoff_s": 0.01,
        "vertex_first_token_timeout_s": 5,
        "vertex_stream_idle_timeout_s": 3,
        "firestore_collection_prefix": "t" + uuid.uuid4().hex[:10] + "_",
        "admin_api_keys": ADMIN_KEY,
        "allow_private_network_targets": True,
        "log_level": "WARNING",
        "usage_flush_interval_s": 0.3,
        "session_lease_s": 10,
        "execution_lease_s": 10,
        "shutdown_grace_s": 2,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def gateway_factory(firestore_emulator: str, fake_vertex: LiveServer) -> Iterator[Callable[..., LiveServer]]:
    from app.gcp.credentials import StaticTokenSource
    from app.main import create_app

    started: list[LiveServer] = []

    def start(wait_ready: bool = True, tokens: Any = None, **overrides: Any) -> LiveServer:
        settings = make_settings(fake_vertex.url, **overrides)
        server = LiveServer(create_app(settings, tokens=tokens or StaticTokenSource("test-token"))).start()
        server.settings = settings  # type: ignore[attr-defined]
        started.append(server)
        if wait_ready:
            deadline = time.monotonic() + 20
            while httpx.get(f"{server.url}/health/ready").status_code != 200:
                if time.monotonic() > deadline:
                    raise RuntimeError("gateway never became ready: " + httpx.get(f"{server.url}/health/ready").text)
                time.sleep(0.05)
        return server

    yield start
    for server in started:
        server.stop()


@pytest.fixture
def gateway(gateway_factory: Callable[..., LiveServer], vertex: Any) -> LiveServer:
    return gateway_factory()


class Api(httpx.Client):
    """httpx client bound to one credential (and optionally a tenant for admin keys)."""

    def __init__(self, base_url: str, key: str, tenant: str | None = None) -> None:
        headers = {"Authorization": f"Bearer {key}"}
        if tenant:
            headers["X-Tenant-ID"] = tenant
        super().__init__(base_url=base_url, headers=headers, timeout=30)
        self.key = key
        self.tenant_id = tenant


def new_tenant(server: LiveServer, name: str = "Acme Clinic") -> tuple[Api, dict[str, Any]]:
    with Api(server.url, ADMIN_KEY) as admin:
        r = admin.post("/v1/tenants", json={"name": name})
        assert r.status_code == 201, r.text
        body = r.json()
    api = Api(server.url, body["api_key"]["key"])
    api.tenant_id = body["id"]
    return api, body


@pytest.fixture
def admin(gateway: LiveServer) -> Iterator[Api]:
    with Api(gateway.url, ADMIN_KEY) as client:
        yield client


@pytest.fixture
def api(gateway: LiveServer) -> Iterator[Api]:
    client, _ = new_tenant(gateway)
    yield client
    client.close()


def sse_events(response: httpx.Response) -> list[dict[str, Any]]:
    import orjson

    events = []
    for line in response.iter_lines():
        if line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                events.append(orjson.loads(data))
    return events
