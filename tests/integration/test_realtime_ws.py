from __future__ import annotations

import asyncio
import time

import httpx
import orjson
import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

LOOKUP = {"type": "function", "function": {"name": "lookup_booking", "parameters": {"type": "object"}}}


async def recv(ws, timeout: float = 10.0) -> dict:
    return orjson.loads(await asyncio.wait_for(ws.recv(), timeout))


async def until(ws, type_: str, timeout: float = 10.0) -> tuple[dict, list[dict]]:
    seen: list[dict] = []
    while True:
        event = await recv(ws, timeout)
        seen.append(event)
        if event["type"] == type_:
            return event, seen


async def send(ws, **event) -> None:
    await ws.send(orjson.dumps(event).decode())


async def test_conversation_over_websocket(gateway):
    async with connect(f"{gateway.ws_url}/v1/realtime") as ws:
        created = await recv(ws)
        assert created["type"] == "session.created"
        await send(ws, type="session.update", session={"system_prompt": "Be brief.", "segmentation": True})
        assert (await recv(ws))["session"]["system_prompt"] == "Be brief."
        await send(ws, type="response.create", input="Hello there. How are you")
        done, seen = await until(ws, "response.done")
        assert seen[0]["type"] == "response.created"
        assert done["status"] == "completed" and done["timings"]["ttft_ms"] is not None
        deltas = "".join(e["delta"] for e in seen if e["type"] == "response.text.delta")
        assert deltas == done["text"]
        assert [e["text"] for e in seen if e["type"] == "response.text.segment"][0] == "Echo: Hello there."
        await send(ws, type="ping", id=1)
        assert await recv(ws) == {"type": "pong", "id": 1}


async def test_barge_in_cancel_then_new_turn(gateway):
    async with connect(f"{gateway.ws_url}/v1/realtime") as ws:
        await recv(ws)
        await send(ws, type="response.create", input="[slow] long reply please")
        await until(ws, "response.text.delta")
        await send(ws, type="response.cancel", spoken_text="Echo:")
        done, _ = await until(ws, "response.done")
        assert done["status"] == "cancelled"
        await send(ws, type="response.create", input="new question")
        done, _ = await until(ws, "response.done")
        assert done["status"] == "completed"
        await send(ws, type="session.get")
        state, _ = await until(ws, "session.state")
        conv = state["session"]["conversation"]
        assert conv[1]["content"] == "Echo:" and conv[1]["interrupted"] is True
        assert [m["role"] for m in conv] == ["user", "assistant", "user", "assistant"]


async def test_response_create_while_active_cancels_previous(gateway):
    async with connect(f"{gateway.ws_url}/v1/realtime") as ws:
        await recv(ws)
        await send(ws, type="response.create", input="[slow] first", response_id="r1")
        await until(ws, "response.text.delta")
        await send(ws, type="response.create", input="second", response_id="r2")
        first, _ = await until(ws, "response.done")
        second, _ = await until(ws, "response.done")
        assert (first["response_id"], first["status"], first["cancel_reason"]) == ("r1", "cancelled", "barge_in")
        assert (second["response_id"], second["status"]) == ("r2", "completed")


async def test_client_tool_flow(gateway):
    async with connect(f"{gateway.ws_url}/v1/realtime") as ws:
        await recv(ws)
        await send(ws, type="session.update", session={"tools": [LOOKUP]})
        await recv(ws)
        await send(ws, type="response.create", input="call lookup_booking {\"id\": 3}")
        done, seen = await until(ws, "response.done")
        assert any(e["type"] == "response.tool_call.started" for e in seen)
        assert done["status"] == "requires_action"
        call = done["tool_calls"][0]
        await send(ws, type="tool.result", call_id=call["call_id"], output={"found": True})
        done, _ = await until(ws, "response.done")
        assert done["status"] == "completed" and "found" in done["text"]


async def test_protocol_errors_keep_connection_open(gateway):
    async with connect(f"{gateway.ws_url}/v1/realtime") as ws:
        await recv(ws)
        await ws.send("{not json")
        assert (await recv(ws))["error"]["code"] == "invalid_json"
        await send(ws, type="does.not.exist", event_id="e1")
        err = await recv(ws)
        assert err["error"]["code"] == "unknown_event" and err["event_id"] == "e1"
        await send(ws, type="conversation.item.create", item={"role": "system", "content": "x"})
        assert (await recv(ws))["type"] == "error"
        await send(ws, type="ping", id="still-alive")
        assert (await recv(ws))["id"] == "still-alive"


async def test_resume_session_and_unknown_session(gateway):
    async with connect(f"{gateway.ws_url}/v1/realtime") as ws:
        sid = (await recv(ws))["session"]["id"]
        await send(ws, type="response.create", input="remember me")
        await until(ws, "response.done")
    async with connect(f"{gateway.ws_url}/v1/realtime?session_id={sid}") as ws:
        resumed = await recv(ws)
        assert resumed["type"] == "session.resumed" and resumed["session"]["messages"] == 2
    async with connect(f"{gateway.ws_url}/v1/realtime?session_id=missing") as ws:
        with pytest.raises(ConnectionClosed) as exc:
            await recv(ws)
        assert exc.value.rcvd.code == 4404


async def test_websocket_auth(gateway_factory):
    gw = gateway_factory(api_keys="ws-key")
    with pytest.raises(InvalidStatus) as exc:
        async with connect(f"{gw.ws_url}/v1/realtime"):
            pass
    assert exc.value.response.status_code == 403
    async with connect(f"{gw.ws_url}/v1/realtime?api_key=ws-key") as ws:
        assert (await recv(ws))["type"] == "session.created"
    async with connect(f"{gw.ws_url}/v1/realtime", additional_headers={"Authorization": "Bearer ws-key"}) as ws:
        assert (await recv(ws))["type"] == "session.created"


async def test_disconnect_cancels_generation(gateway, engine_stats):
    aborted = engine_stats()["aborted"]
    async with connect(f"{gateway.ws_url}/v1/realtime") as ws:
        await recv(ws)
        await send(ws, type="response.create", input="[slow] a very long reply")
        await until(ws, "response.text.delta")
    deadline = time.monotonic() + 5
    while engine_stats()["aborted"] <= aborted:
        assert time.monotonic() < deadline
        await asyncio.sleep(0.05)
    assert httpx.get(f"{gateway.url}/health").json()["inflight"] == 0


async def test_rapid_successive_requests_cancel_every_superseded_response(gateway):
    async with connect(f"{gateway.ws_url}/v1/realtime") as ws:
        await recv(ws)
        for i in range(3):
            await send(ws, type="response.create", input=f"[slow] request {i}", response_id=f"r{i}")
        await send(ws, type="response.create", input="final", response_id="final")
        done = {}
        while len(done) < 4:
            event = await recv(ws)
            if event["type"] == "response.done":
                done[event["response_id"]] = event["status"]
        assert done == {"r0": "cancelled", "r1": "cancelled", "r2": "cancelled", "final": "completed"}


async def test_cancel_during_server_tool_keeps_history_valid(gateway_factory):
    gw = gateway_factory(tools_modules="app.tools.builtin,tests.tool_fixtures")
    async with connect(f"{gw.ws_url}/v1/realtime") as ws:
        await recv(ws)
        await send(ws, type="session.update", session={"server_tools": ["slow_lookup"]})
        await recv(ws)
        await send(ws, type="response.create", input="call slow_lookup")
        await until(ws, "response.tool_call")
        await send(ws, type="response.cancel")
        done, _ = await until(ws, "response.done")
        assert done["status"] == "cancelled"
        await send(ws, type="session.get")
        state, _ = await until(ws, "session.state")
        conv = state["session"]["conversation"]
        assert [m["role"] for m in conv] == ["user", "assistant", "tool"]
        assert conv[2]["tool_call_id"] == conv[1]["tool_calls"][0]["id"]
