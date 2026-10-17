"""Shared fixtures: a live mock vLLM server and live gateways on ephemeral ports.

Integration tests talk to real sockets (uvicorn in background threads) so streaming,
cancellation-by-disconnect and WebSockets behave exactly as in production.
"""
from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
import uvicorn

os.environ.setdefault("MOCK_TOKEN_DELAY_S", "0.005")
os.environ.setdefault("MOCK_TTFT_S", "0.01")

from app.config import Settings  # noqa: E402

# Keep the developer's shell / .env from leaking into tests.
for _name in list(Settings.model_fields):
    os.environ.pop(_name.upper(), None)
os.environ.pop("LLM_SERVER_ENV_FILE", None)
os.environ.pop("VLLM_API_KEY", None)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LiveServer:
    def __init__(self, app: Any, port: int) -> None:
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.ws_url = f"ws://127.0.0.1:{port}"
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", log_config=None,
                                ws_ping_interval=None, lifespan="on")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> LiveServer:
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.server.started:
            if time.monotonic() > deadline or not self.thread.is_alive():
                raise RuntimeError("server failed to start")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=15)


@pytest.fixture(scope="session")
def mock_engine() -> Iterator[LiveServer]:
    from app import mock_engine as mock

    server = LiveServer(mock.app, free_port()).start()
    yield server
    server.stop()


@pytest.fixture
def engine_stats(mock_engine: LiveServer) -> Callable[[], dict[str, Any]]:
    def read() -> dict[str, Any]:
        return httpx.get(f"{mock_engine.url}/mock/stats").json()
    return read


def make_settings(engine_url: str, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "engine_mode": "external",
        "engine_base_url": engine_url,
        "engine_served_model_name": "mock-model",
        "engine_health_interval_s": 0.5,
        "log_level": "WARNING",
        "default_chat_template_kwargs": "",
        "tools_modules": "app.tools.builtin",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def gateway_factory(mock_engine: LiveServer) -> Iterator[Callable[..., LiveServer]]:
    from app.main import create_app

    started: list[LiveServer] = []

    def start(**overrides: Any) -> LiveServer:
        engine_url = overrides.pop("engine_url", mock_engine.url)
        settings = make_settings(engine_url, **overrides)
        server = LiveServer(create_app(settings), free_port()).start()
        started.append(server)
        if engine_url == mock_engine.url:
            deadline = time.monotonic() + 10
            while httpx.get(f"{server.url}/health/ready").status_code != 200:
                if time.monotonic() > deadline:
                    raise RuntimeError("gateway never became ready")
                time.sleep(0.05)
        return server

    yield start
    for server in started:
        server.stop()


@pytest.fixture
def gateway(gateway_factory: Callable[..., LiveServer]) -> LiveServer:
    return gateway_factory()


@pytest.fixture
def http(gateway: LiveServer) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=gateway.url, timeout=30) as client:
        yield client
