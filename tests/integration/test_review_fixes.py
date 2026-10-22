"""Regression tests for the code-review findings: client tool calls never left unanswered,
crashes still end the response, pinned tokens, and session id validation."""
from __future__ import annotations

import orjson
import pytest
from websockets.sync.client import connect

from tests.integration.test_realtime import recv_until, tenant  # noqa: F401 (fixture)


def _open(gateway, api, agent):
    ws = connect(f"{gateway.ws_url}/v1/realtime?agent_id={agent['id']}",
                 additional_headers={"Authorization": f"Bearer {api.key}"})
    assert orjson.loads(ws.recv(timeout=5))["type"] == "session.created"
    return ws


def _send(ws, event: dict) -> None:
    ws.send(orjson.dumps(event).decode())


def _requires_action(ws) -> str:
    _send(ws, {"type": "response.create", "input": "call play_hold_music"})
    done = recv_until(ws, "response.done")[-1]
    assert done["status"] == "requires_action"
    return done["tool_calls"][0]["call_id"]


def test_oversized_tool_result_can_be_retried(gateway, tenant, vertex):  # noqa: F811
    api, agent, _ = tenant
    with _open(gateway, api, agent) as ws:
        call_id = _requires_action(ws)
        _send(ws, {"type": "tool.result", "call_id": call_id, "output": "x" * 70_000})
        assert recv_until(ws, "error")[-1]["error"]["code"] == "invalid_request"
        _send(ws, {"type": "tool.result", "call_id": call_id, "output": "short"})
        assert recv_until(ws, "response.done")[-1]["status"] == "completed"


def test_discarded_speculative_turn_reopens_tool_call(gateway, tenant, vertex):  # noqa: F811
    api, agent, _ = tenant
    with _open(gateway, api, agent) as ws:
        call_id = _requires_action(ws)
        _send(ws, {"type": "response.create", "input": "[slow] maybe", "response_id": "resp_spec1"})
        recv_until(ws, "response.created")
        _send(ws, {"type": "response.cancel", "response_id": "resp_spec1", "discard": True})
        recv_until(ws, "response.done")
        _send(ws, {"type": "session.get"})
        state = recv_until(ws, "session.state")[-1]["session"]
        assert state["pending_tool_calls"] == [call_id]
        # the reopened call is answered (abandoned) by the next user turn: no Vertex 400
        _send(ws, {"type": "response.create", "input": "hello again"})
        assert recv_until(ws, "response.done")[-1]["status"] == "completed"


def test_discard_after_speculative_reply_finished(gateway, tenant, vertex):  # noqa: F811
    api, agent, _ = tenant
    with _open(gateway, api, agent) as ws:
        _send(ws, {"type": "response.create", "input": "first"})
        recv_until(ws, "response.done")
        _send(ws, {"type": "response.create", "input": "interim words", "response_id": "resp_spec2"})
        recv_until(ws, "response.done")
        _send(ws, {"type": "response.cancel", "response_id": "resp_spec2", "discard": True})
        _send(ws, {"type": "session.get"})
        messages = recv_until(ws, "session.state")[-1]["session"]["messages"]
        assert not any("interim words" in (m.get("content") or "") for m in messages)


def test_internal_error_still_ends_response(gateway, tenant, vertex):  # noqa: F811
    api, agent, _ = tenant
    with _open(gateway, api, agent) as ws:
        _send(ws, {"type": "response.create", "instructions": 5, "input": "hi"})
        events = recv_until(ws, "response.done")
        assert events[-1]["status"] == "failed"
        _send(ws, {"type": "response.create", "input": "hi"})
        assert recv_until(ws, "response.done")[-1]["status"] == "completed"


def test_pinned_token_cannot_hand_off(gateway, tenant):  # noqa: F811
    api, agent, billing = tenant
    token = api.post("/v1/realtime/tokens", json={"ttl_s": 60, "agent_id": agent["id"]}).json()["token"]
    with connect(f"{gateway.ws_url}/v1/realtime?token={token}") as ws:
        assert orjson.loads(ws.recv(timeout=5))["type"] == "session.created"
        _send(ws, {"type": "agent.handoff", "agent_id": billing["id"]})
        assert recv_until(ws, "error")[-1]["error"]["code"] == "agent_mismatch"


@pytest.mark.parametrize("session_id", ["a%2Fb", "x%2Fmessages%2F00000005"])
def test_resume_rejects_path_like_session_ids(gateway, tenant, session_id):  # noqa: F811
    api, _, _ = tenant
    with connect(f"{gateway.ws_url}/v1/realtime?session_id={session_id}",
                 additional_headers={"Authorization": f"Bearer {api.key}"}) as ws:
        assert orjson.loads(ws.recv(timeout=5))["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("event", [
    {"type": "response.create", "say": "One moment please."},
    {"type": "conversation.item.create", "item": {"role": "user", "content": "actually, never mind"}},
])
def test_messages_while_tool_call_pending_keep_history_valid(gateway, tenant, vertex, event):  # noqa: F811
    api, agent, _ = tenant
    with _open(gateway, api, agent) as ws:
        _requires_action(ws)
        _send(ws, event)
        recv_until(ws, "response.done" if event["type"] == "response.create" else "conversation.item.created")
        _send(ws, {"type": "response.create", "input": "hello"})
        assert recv_until(ws, "response.done")[-1]["status"] == "completed"
