"""Realtime WebSocket protocol (aica.realtime.v1)."""
from __future__ import annotations

import time

import orjson
import pytest
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect

from tests.conftest import new_tenant


def recv_until(ws, kind: str, timeout: float = 10) -> list[dict]:
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        event = orjson.loads(ws.recv(timeout=max(0.1, deadline - time.monotonic())))
        events.append(event)
        if event["type"] == kind:
            return events
    raise AssertionError(f"no {kind} in {[e['type'] for e in events]}")


@pytest.fixture
def tenant(gateway):
    api, _ = new_tenant(gateway)
    agent = api.post("/v1/agents", json={
        "name": "Reception", "instructions": "#agent:priya receptionist",
        "tools": [{"type": "builtin", "name": "get_current_time"}, {"type": "builtin", "name": "end_conversation"},
                  {"type": "client", "name": "play_hold_music"}]}).json()
    billing = api.post("/v1/agents", json={"name": "Billing", "instructions": "#agent:billing payments"}).json()
    yield api, agent, billing
    api.close()


def test_full_voice_turn_over_websocket(gateway, tenant, vertex):
    api, agent, _ = tenant
    with connect(f"{gateway.ws_url}/v1/realtime?agent_id={agent['id']}", subprotocols=["aica.realtime.v1"],
                 additional_headers={"Authorization": f"Bearer {api.key}"}) as ws:
        assert ws.subprotocol == "aica.realtime.v1"
        created = orjson.loads(ws.recv(timeout=5))
        assert created["type"] == "session.created" and created["protocol"] == "aica.realtime.v1"
        assert created["agent"]["name"] == "Reception" and created["seq"] == 1 and created["event_id"]
        session_id = created["session"]["id"]
        ws.send(orjson.dumps({"type": "response.create", "input": "Book me in. I am free tomorrow."}).decode())
        events = recv_until(ws, "response.done")
        assert [e["type"] for e in events][0] == "response.created"
        assert any(e["type"] == "response.text.segment" for e in events)
        seqs = [e["seq"] for e in events]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
        assert events[-1]["status"] == "completed"

        ws.send(orjson.dumps({"type": "response.create", "input": "call play_hold_music"}).decode())
        done = recv_until(ws, "response.done")[-1]
        assert done["status"] == "requires_action"
        call_id = done["tool_calls"][0]["call_id"]
        ws.send(orjson.dumps({"type": "tool.result", "call_id": call_id, "output": "music playing"}).decode())
        events = recv_until(ws, "response.done")
        assert events[0]["type"] == "tool.result.accepted"
        assert events[-1]["status"] == "completed" and "music playing" in events[-1]["text"]

        ws.send(orjson.dumps({"type": "ping", "id": 7}).decode())
        assert recv_until(ws, "pong")[-1]["id"] == 7
        ws.send(orjson.dumps({"type": "nope", "event_id": "c1"}).decode())
        err = recv_until(ws, "error")[-1]
        assert err["error"]["code"] == "unknown_event" and err["client_event_id"] == "c1"
        ws.send("not json")
        assert recv_until(ws, "error")[-1]["error"]["code"] == "invalid_json"
    # the session survives the disconnect and can be resumed
    with connect(f"{gateway.ws_url}/v1/realtime?session_id={session_id}",
                 additional_headers={"Authorization": f"Bearer {api.key}"}) as ws:
        resumed = orjson.loads(ws.recv(timeout=5))
        assert resumed["type"] == "session.resumed" and resumed["session"]["message_count"] >= 4


def test_barge_in_and_handoff_over_websocket(gateway, tenant, vertex):
    api, agent, billing = tenant
    with connect(f"{gateway.ws_url}/v1/realtime?agent_id={agent['id']}",
                 additional_headers={"Authorization": f"Bearer {api.key}"}) as ws:
        recv_until(ws, "session.created")
        ws.send(orjson.dumps({"type": "response.create", "input": "long answer [slow] one two three four five",
                              "response_id": "r1"}).decode())
        recv_until(ws, "response.text.delta")
        ws.send(orjson.dumps({"type": "response.cancel", "response_id": "r1", "spoken_text": "priya:"}).decode())
        done = recv_until(ws, "response.done")[-1]
        assert done["status"] == "cancelled" and done["response_id"] == "r1"
        ws.send(orjson.dumps({"type": "session.get"}).decode())
        state = recv_until(ws, "session.state")[-1]["session"]
        assert state["messages"][-1]["content"] == "priya:" and state["messages"][-1]["interrupted"]

        ws.send(orjson.dumps({"type": "agent.handoff", "agent_id": billing["id"], "reason": "payment"}).decode())
        handoff = recv_until(ws, "agent.handoff")[-1]
        assert handoff["to_agent"]["name"] == "Billing" and handoff["source"] == "client"
        ws.send(orjson.dumps({"type": "response.create", "input": "how much?"}).decode())
        assert recv_until(ws, "response.done")[-1]["text"].startswith("billing:")


def test_end_conversation_tool_and_session_end(gateway, tenant, vertex):
    api, agent, _ = tenant
    with connect(f"{gateway.ws_url}/v1/realtime?agent_id={agent['id']}",
                 additional_headers={"Authorization": f"Bearer {api.key}"}) as ws:
        recv_until(ws, "session.created")
        ws.send(orjson.dumps({"type": "response.create",
                              "input": 'call end_conversation {"reason": "caller said bye"}'}).decode())
        events = recv_until(ws, "response.done")
        req = next(e for e in events if e["type"] == "session.end_requested")
        assert req["reason"] == "caller said bye" and events[-1]["end_conversation"]["reason"] == "caller said bye"
        ws.send(orjson.dumps({"type": "session.end", "reason": "hangup"}).decode())
        ended = recv_until(ws, "session.ended")[-1]
        assert ended["reason"] == "hangup"


def test_realtime_tokens(gateway, tenant):
    api, agent, _ = tenant
    token = api.post("/v1/realtime/tokens", json={"ttl_s": 120, "agent_id": agent["id"],
                                                  "session_id": "browser-call-1"}).json()["token"]
    # browsers authenticate with a subprotocol (they cannot set headers)
    with connect(f"{gateway.ws_url}/v1/realtime", subprotocols=["aica.realtime.v1", f"bearer.{token}"]) as ws:
        created = orjson.loads(ws.recv(timeout=5))
        assert created["type"] == "session.created" and created["session"]["id"] == "browser-call-1"
        assert created["session"]["agent_id"] == agent["id"]
    # the token is pinned to that session: resuming works, another session is refused
    with connect(f"{gateway.ws_url}/v1/realtime?token={token}") as ws:
        assert orjson.loads(ws.recv(timeout=5))["type"] == "session.resumed"
    with connect(f"{gateway.ws_url}/v1/realtime?token={token}&session_id=other") as ws:
        assert orjson.loads(ws.recv(timeout=5))["error"]["code"] == "session_mismatch"
        with pytest.raises(ConnectionClosed):
            ws.recv(timeout=5)
        assert ws.close_code == 4403
    # tokens are useless on the REST API
    r = api.get("/v1/agents", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def test_websocket_auth_rejected(gateway):
    for url, headers in ((f"{gateway.ws_url}/v1/realtime", {}),
                         (f"{gateway.ws_url}/v1/realtime", {"Authorization": "Bearer aica_nope"}),
                         (f"{gateway.ws_url}/v1/realtime?token=aica_some_api_key", {})):
        with pytest.raises(InvalidStatus) as exc:
            with connect(url, additional_headers=headers):
                pass
        assert exc.value.response.status_code == 403


def test_unknown_session_closes_4404(gateway, tenant):
    api, _, _ = tenant
    with connect(f"{gateway.ws_url}/v1/realtime?session_id=missing",
                 additional_headers={"Authorization": f"Bearer {api.key}"}) as ws:
        assert orjson.loads(ws.recv(timeout=5))["error"]["code"] == "session_not_found"
        with pytest.raises(ConnectionClosed):
            ws.recv(timeout=5)
        assert ws.close_code == 4404
