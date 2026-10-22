"""Graceful shutdown: readiness flips, in-flight answers finish, WebSockets are told to reconnect,
and write-behind data is persisted before the process exits."""
from __future__ import annotations

import threading
import time

import httpx
import orjson
from websockets.sync.client import connect

from tests.conftest import Api, new_tenant, sse_events


def test_graceful_shutdown(gateway_factory, vertex):
    gw = gateway_factory(shutdown_grace_s=5)
    api, _ = new_tenant(gw)
    agent = api.post("/v1/agents", json={"name": "A", "instructions": "#agent:a x"}).json()
    sid = api.post("/v1/sessions", json={"agent_id": agent["id"], "session_id": "drain-1"}).json()["id"]
    ws = connect(f"{gw.ws_url}/v1/realtime?agent_id={agent['id']}",
                 additional_headers={"Authorization": f"Bearer {api.key}"})
    assert orjson.loads(ws.recv(timeout=5))["type"] == "session.created"

    events: list = []

    def run() -> None:
        with api.stream("POST", f"/v1/sessions/{sid}/responses",
                        json={"input": "finish this [slow] sentence one two three"}) as resp:
            events.extend(sse_events(resp))

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.3)
    services = gw.server.config.app.state.services
    services.begin_drain()  # what SIGTERM does first (thread-safe)
    time.sleep(0.1)
    assert httpx.get(f"{gw.url}/health/ready").json()["status"] == "draining"
    closing = orjson.loads(ws.recv(timeout=5))
    assert closing["type"] == "session.closing" and closing["reason"] == "server_shutdown"
    t.join(15)
    assert events[-1]["status"] == "completed"  # the caller heard the whole sentence
    r = api.post(f"/v1/sessions/{sid}/responses", json={"input": "new work", "stream": False})
    assert r.status_code == 503 and r.json()["error"]["code"] == "shutting_down"  # no new work while draining
    prefix = gw.settings.firestore_collection_prefix
    gw.stop()
    ws.close()

    # a fresh instance sees everything that was written behind
    gw2 = gateway_factory(firestore_collection_prefix=prefix)
    with Api(gw2.url, api.key) as api2:
        history = api2.get(f"/v1/sessions/{sid}?include_messages=true").json()
        assert history["messages"][1]["content"].startswith("a: Echo: finish this")
        responses = api2.get(f"/v1/sessions/{sid}/responses").json()["data"]
        assert any(r["status"] == "completed" for r in responses)
