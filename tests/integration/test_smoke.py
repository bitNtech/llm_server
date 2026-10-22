"""Fast end-to-end smoke test of the main path (auth -> agent -> session -> streamed turn)."""
from __future__ import annotations

from tests.conftest import sse_events


def test_smoke(api, vertex):
    r = api.post("/v1/agents", json={"name": "Reception", "instructions": "#agent:priya Be brief.",
                                     "tools": [{"type": "builtin", "name": "get_current_time"}]})
    assert r.status_code == 201, r.text
    agent = r.json()
    r = api.post("/v1/sessions", json={"agent_id": agent["id"], "session_id": "call-1"})
    assert r.status_code == 201, r.text
    with api.stream("POST", "/v1/sessions/call-1/responses", json={"input": "hello there"}) as resp:
        assert resp.status_code == 200
        events = sse_events(resp)
    types = [e["type"] for e in events]
    assert types[0] == "response.created" and types[-1] == "response.done", types
    done = events[-1]
    assert done["status"] == "completed", done
    assert done["text"].startswith("priya: Echo: hello there")


def test_console_page(api):
    r = api.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert "/v1/realtime" in r.text and "/v1/chat/completions" in r.text
