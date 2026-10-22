"""The conversation engine over REST/SSE: streaming, tool calling of every kind, client tools,
barge-in, speculative discard, handoffs, failures, and persistence/resume through Firestore."""
from __future__ import annotations

import threading
import time

import httpx

from tests.conftest import Api, new_tenant, sse_events

RECEPTION = {"name": "Reception", "instructions": "#agent:priya You are the hospital receptionist."}


def stream(api: Api, session_id: str, **body) -> list[dict]:
    with api.stream("POST", f"/v1/sessions/{session_id}/responses", json=body) as resp:
        assert resp.status_code == 200, resp.read()
        return sse_events(resp)


def agent(api: Api, **extra) -> dict:
    r = api.post("/v1/agents", json={**RECEPTION, **extra})
    assert r.status_code == 201, r.text
    return r.json()


def session(api: Api, agent_id: str | None = None, **extra) -> str:
    body = {"agent_id": agent_id, **extra} if agent_id else extra
    r = api.post("/v1/sessions", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_streaming_turn_with_segments_usage_and_timings(api, vertex):
    sid = session(api, agent(api)["id"])
    events = stream(api, sid, input="I need a cardiologist. Tomorrow works best.")
    types = [e["type"] for e in events]
    assert types[0] == "response.created" and types[-1] == "response.done"
    deltas = "".join(e["delta"] for e in events if e["type"] == "response.text.delta")
    segments = [e["text"] for e in events if e["type"] == "response.text.segment"]
    done = events[-1]
    assert deltas == done["text"] and done["text"].startswith("priya: Echo:")
    assert len(segments) >= 2 and " ".join(segments) == done["text"]
    assert done["usage"]["prompt_tokens"] > 0 and done["usage"]["completion_tokens"] > 0
    assert done["timings"]["ttft_ms"] is not None and done["timings"]["upstream_ttft_ms"] is not None
    assert done["model"] == "google/gemini-3.5-flash" and done["agent"]["name"] == "Reception"
    # the upstream request carried the agent instructions and the voice-latency defaults
    body = vertex.get("/stats").json()["last_body"]
    assert body["messages"][0]["role"] == "system" and "#agent:priya" in body["messages"][0]["content"]
    assert body["reasoning_effort"] == "minimal" and body["stream_options"] == {"include_usage": True}


def test_non_streaming_response_and_history(api):
    sid = session(api, agent(api)["id"])
    r = api.post(f"/v1/sessions/{sid}/responses", json={"input": "hello", "stream": False})
    assert r.status_code == 200 and r.json()["status"] == "completed"
    msgs = api.get(f"/v1/sessions/{sid}/messages").json()["data"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[1]["agent_id"]
    responses = api.get(f"/v1/sessions/{sid}/responses").json()["data"]
    assert responses[0]["status"] == "completed" and responses[0]["usage"]["total_tokens"] > 0


def test_builtin_tool_round_trip(api, vertex):
    sid = session(api, agent(api, tools=[{"type": "builtin", "name": "get_current_time"}])["id"])
    events = stream(api, sid, input='call get_current_time {"timezone": "Asia/Kolkata"}')
    kinds = [e["type"] for e in events]
    assert "response.tool_call.started" in kinds and "response.tool_call" in kinds and "response.tool_result" in kinds
    result = next(e for e in events if e["type"] == "response.tool_result")
    assert not result["is_error"] and "Asia/Kolkata" in result["output"]
    done = events[-1]
    assert done["status"] == "completed" and done["text"].startswith("priya: Tool result:")
    assert done["server_tool_calls"][0]["name"] == "get_current_time"
    # the second round replayed the tool call with its thought signature (the fake rejects it otherwise)
    replay = vertex.get("/stats").json()["last_body"]["messages"]
    call_msg = next(m for m in replay if m.get("tool_calls"))
    assert call_msg["tool_calls"][0]["extra_content"]["google"]["thought_signature"].startswith("sig-")
    calls = api.get(f"/v1/sessions/{sid}/tool_calls").json()["data"]
    assert calls[0]["status"] == "completed" and calls[0]["arguments"] == {"timezone": "Asia/Kolkata"}


def test_invalid_arguments_are_fed_back_to_the_model(api):
    a = agent(api, tools=[{"type": "client", "name": "noop", "parameters": {"type": "object"}}])
    tool = api.post("/v1/tools", json={"name": "need_day", "description": "needs day",
                                       "http": {"url": "http://127.0.0.1:9/x"},
                                       "parameters": {"type": "object", "properties": {"day": {"type": "string"}},
                                                      "required": ["day"]}}).json()
    api.patch(f"/v1/agents/{a['id']}", json={"tools": [{"type": "tool", "id": tool["id"]}]})
    sid = session(api, a["id"])
    events = stream(api, sid, input="call need_day {}")
    result = next(e for e in events if e["type"] == "response.tool_result")
    assert result["is_error"] and "required" in result["output"]
    assert events[-1]["status"] == "completed"


def test_http_tool_side_effects_are_idempotent(api, target, tool_target, vertex):
    tool = api.post("/v1/tools", json={
        "name": "book", "description": "Book a slot", "side_effects": True,
        "parameters": {"type": "object", "properties": {"day": {"type": "string"}}},
        "http": {"url": f"{tool_target.url}/book"}}).json()
    sid = session(api, agent(api, tools=[{"type": "tool", "id": tool["id"]}])["id"])
    events = stream(api, sid, input='call book {"day": "friday"}')
    result = next(e for e in events if e["type"] == "response.tool_result")
    assert '"booking_id":"B0001"' in result["output"]
    rec = [r for r in target.get("/records").json() if r["name"] == "book"]
    assert len(rec) == 1 and rec[0]["body"] == '{"day":"friday"}'
    assert rec[0]["headers"]["x-aica-session-id"] == sid and rec[0]["headers"]["idempotency-key"]
    audit = api.get(f"/v1/sessions/{sid}/tool_calls").json()["data"]
    assert audit[0]["side_effects"] and audit[0]["status"] == "completed"


def test_side_effect_replay_vs_reused_call_id(api, target, tool_target, vertex):
    tool = api.post("/v1/tools", json={
        "name": "book", "description": "Book", "side_effects": True,
        "parameters": {"type": "object", "properties": {"day": {"type": "string"}}},
        "http": {"url": f"{tool_target.url}/book"}}).json()
    sid = session(api, agent(api, tools=[{"type": "tool", "id": tool["id"]}])["id"])

    def run(text: str) -> dict:
        return next(e for e in stream(api, sid, input=text) if e["type"] == "response.tool_result")

    first = run('call book {"day": "mon"} [callid:c1]')
    replay = run('call book {"day": "mon"} [callid:c1]')  # same call replayed: answered from the record
    other = run('call book {"day": "tue"} [callid:c1]')  # same id, different call: executed
    assert not first["replayed"] and replay["replayed"] and not other["replayed"]
    assert replay["output"] == first["output"]
    bookings = [r for r in target.get("/records").json() if r["name"] == "book"]
    assert [r["body"] for r in bookings] == ['{"day":"mon"}', '{"day":"tue"}']
    assert bookings[0]["headers"]["idempotency-key"] != bookings[1]["headers"]["idempotency-key"]
    assert bookings[0]["headers"]["x-aica-call-id"] == "c1"


def test_mcp_tools(api, mcp_server, vertex):
    r = api.post("/v1/mcp/servers?verify=true", json={"name": "clinic", "url": f"{mcp_server.url}/mcp",
                                                      "tool_prefix": "clinic_", "tool_allowlist": ["lookup_order",
                                                                                                   "add", "broken"]})
    assert r.status_code == 201, r.text
    server = r.json()
    assert {t["name"] for t in server["tools"]} == {"clinic_lookup_order", "clinic_add", "clinic_broken"}
    listed = api.get(f"/v1/mcp/servers/{server['id']}/tools").json()["data"]
    assert any(t["remote_name"] == "add" and "a" in t["parameters"]["properties"] for t in listed)
    sid = session(api, agent(api, tools=[{"type": "mcp", "server_id": server["id"]}])["id"])
    events = stream(api, sid, input='call clinic_lookup_order {"order_id": "A-17"}')
    result = next(e for e in events if e["type"] == "response.tool_result")
    assert not result["is_error"] and "shipped" in result["output"]
    events = stream(api, sid, input='call clinic_add {"a": 2, "b": 40}')
    assert next(e for e in events if e["type"] == "response.tool_result")["output"] == "42"
    events = stream(api, sid, input="call clinic_broken")
    assert next(e for e in events if e["type"] == "response.tool_result")["is_error"]
    events = stream(api, sid, input='call clinic_add {"a": "x"}')  # schema violation caught before the call
    bad = next(e for e in events if e["type"] == "response.tool_result")
    assert bad["is_error"] and "invalid arguments" in bad["output"]


def test_parallel_server_and_client_tools(api, vertex):
    a = agent(api, tools=[{"type": "builtin", "name": "get_current_time"},
                          {"type": "client", "name": "transfer_call", "description": "to a human",
                           "parameters": {"type": "object", "properties": {"dept": {"type": "string"}}}}])
    sid = session(api, a["id"])
    events = stream(api, sid, input='call get_current_time,transfer_call {"dept": "billing"}')
    done = events[-1]
    assert done["status"] == "requires_action" and done["tool_calls"][0]["name"] == "transfer_call"
    sides = {e["name"]: e["side"] for e in events if e["type"] == "response.tool_call"}
    assert sides == {"get_current_time": "server", "transfer_call": "client"}
    call_id = done["tool_calls"][0]["call_id"]
    assert api.get(f"/v1/sessions/{sid}").json()["pending_tool_calls"] == [call_id]
    with api.stream("POST", f"/v1/sessions/{sid}/tool_results",
                    json={"results": [{"call_id": call_id, "output": {"transferred": True}}]}) as resp:
        events = sse_events(resp)
    assert events[-1]["status"] == "completed" and "transferred" in events[-1]["text"]
    r = api.post(f"/v1/sessions/{sid}/tool_results", json={"results": [{"call_id": call_id, "output": "x"}]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "unknown_tool_call"


def test_new_input_abandons_pending_client_calls(api):
    sid = session(api, agent(api, tools=[{"type": "client", "name": "lookup"}])["id"])
    assert stream(api, sid, input="call lookup")[-1]["status"] == "requires_action"
    assert stream(api, sid, input="never mind")[-1]["status"] == "completed"
    msgs = api.get(f"/v1/sessions/{sid}/messages").json()["data"]
    tool_msg = next(m for m in msgs if m["role"] == "tool")
    assert "moved on" in tool_msg["content"]


def test_barge_in_truncates_to_spoken_text(api, vertex):
    sid = session(api, agent(api)["id"])
    events: list = []

    def run() -> None:
        events.extend(stream(api, sid, input="tell me everything [slow] about the visiting hours please"))

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.6)
    r = api.post(f"/v1/sessions/{sid}/cancel", json={"spoken_text": "priya: Echo: tell"})
    t.join(10)
    assert r.json()["cancelled"] is True
    done = events[-1]
    assert done["status"] == "cancelled" and done["cancel_reason"] == "client_cancel"
    deadline = time.monotonic() + 3  # the upstream stream was closed (generation stopped at "Vertex")
    while vertex.get("/stats").json()["aborted"] < 1 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert vertex.get("/stats").json()["aborted"] >= 1
    last = api.get(f"/v1/sessions/{sid}/messages").json()["data"][-1]
    assert last["role"] == "assistant" and last["content"] == "priya: Echo: tell" and last["interrupted"]


def test_new_response_supersedes_running_one(api):
    sid = session(api, agent(api)["id"])
    first: list = []
    t = threading.Thread(target=lambda: first.extend(stream(api, sid, input="first [slow] one two three four")))
    t.start()
    time.sleep(0.4)
    second = stream(api, sid, input="second")
    t.join(10)
    assert first[-1]["status"] == "cancelled" and first[-1]["cancel_reason"] == "barge_in"
    assert second[-1]["status"] == "completed" and "second" in second[-1]["text"]


def test_speculative_turn_discard(api):
    sid = session(api, agent(api)["id"])
    stream(api, sid, input="hello")
    out: list = []
    t = threading.Thread(target=lambda: out.extend(stream(api, sid, input="interim guess [slow] words words")))
    t.start()
    time.sleep(0.4)
    api.post(f"/v1/sessions/{sid}/cancel", json={"discard": True})
    t.join(10)
    msgs = api.get(f"/v1/sessions/{sid}/messages").json()["data"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[0]["content"] == "hello"


def test_model_handoff_and_api_handoff(api, vertex):
    billing = agent(api, name="Billing", instructions="#agent:billing You handle payments.",
                    tools=[{"type": "builtin", "name": "get_current_time"}])
    front = agent(api, handoffs=[{"agent_id": billing["id"], "description": "payment questions"}])
    sid = session(api, front["id"])
    events = stream(api, sid, input="call transfer_to_billing")
    handoff = next(e for e in events if e["type"] == "agent.handoff")
    assert handoff["from_agent"]["name"] == "Reception" and handoff["to_agent"]["name"] == "Billing"
    done = events[-1]
    assert done["status"] == "completed" and done["text"].startswith("billing: Tool result: Transferred")
    assert done["agent"]["name"] == "Billing"
    state = api.get(f"/v1/sessions/{sid}").json()
    assert state["current_agent_id"] == billing["id"] and state["agent_id"] == front["id"]
    assert stream(api, sid, input="what do I owe?")[-1]["text"].startswith("billing: Echo")
    r = api.post(f"/v1/sessions/{sid}/handoff", json={"agent_id": front["id"], "reason": "done with billing"})
    assert r.status_code == 200 and r.json()["to_agent"]["name"] == "Reception"
    assert stream(api, sid, input="thanks")[-1]["text"].startswith("priya: Echo")


def test_say_greeting_and_opening_turn(api, vertex):
    a = agent(api, greeting="Vanakkam, City Hospital.")
    sid = session(api, a["id"])
    events = stream(api, sid, say="Vanakkam, City Hospital. How can I help?")
    assert events[-1]["status"] == "completed" and events[-1]["finish_reason"] == "say"
    before = vertex.get("/stats").json()["requests"]
    assert vertex.get("/stats").json()["requests"] == before  # no model call for `say`
    sid2 = session(api, a["id"])
    events = stream(api, sid2)  # model speaks first: an opening user turn is synthesized
    assert events[-1]["status"] == "completed"
    assert any(m["role"] == "user" for m in vertex.get("/stats").json()["last_body"]["messages"])


def test_upstream_failures(api, vertex):
    sid = session(api, agent(api)["id"])
    events = stream(api, sid, input="this will [error] break midway please")
    done = events[-1]
    assert done["status"] == "failed" and done["error"]["code"] == "provider_error"
    msgs = api.get(f"/v1/sessions/{sid}/messages").json()["data"]
    assert msgs[-1]["role"] == "assistant" and msgs[-1].get("interrupted")  # partial reply kept
    events = stream(api, sid, input="this will [stall] now")
    assert events[-1]["status"] == "failed" and events[-1]["error"]["code"] == "provider_timeout"


def test_session_resumes_on_another_instance(gateway_factory, vertex):
    first = gateway_factory(session_cache_ttl_s=0.5, session_lease_s=10)
    api, _ = new_tenant(first)
    a = agent(api)
    sid = session(api, a["id"], session_id="call-resume", metadata={"caller": "+911234"})
    stream(api, sid, input="first turn")
    # same Firestore namespace, different instance
    second = gateway_factory(firestore_collection_prefix=first.settings.firestore_collection_prefix)
    other = Api(second.url, api.key)
    r = other.post(f"/v1/sessions/{sid}/responses", json={"input": "too early", "stream": False})
    assert r.status_code == 409 and r.json()["error"]["code"] == "session_busy_elsewhere"
    deadline = time.monotonic() + 20  # first instance evicts the idle session and releases its lease
    while time.monotonic() < deadline:
        r = other.post(f"/v1/sessions/{sid}/responses", json={"input": "second turn", "stream": False})
        if r.status_code != 409:
            break
        time.sleep(0.5)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "completed"
    history = other.get(f"/v1/sessions/{sid}?include_messages=true").json()
    assert [m["content"].split(".")[0] for m in history["messages"] if m["role"] == "user"] == [
        "first turn", "second turn"]
    assert history["metadata"] == {"caller": "+911234"} and history["stats"]["turns"] == 2
    other.close()


def test_session_lifecycle(api):
    sid = session(api, metadata={"call": "1"})  # agent-less session: server default instructions
    assert api.get(f"/v1/sessions/{sid}").json()["agent_id"] is None
    r = api.patch(f"/v1/sessions/{sid}", json={"instructions": "Answer in Tamil.", "params": {"temperature": 0.2}})
    assert r.json()["instructions"] == "Answer in Tamil."
    assert api.post(f"/v1/sessions/{sid}/messages",
                    json={"messages": [{"role": "user", "content": "seeded"}]}).status_code == 200
    assert api.post(f"/v1/sessions/{sid}/messages",
                    json={"messages": [{"role": "system", "content": "x"}]}).status_code == 400
    assert api.delete(f"/v1/sessions/{sid}/messages").json()["message_count"] == 0
    r = api.post(f"/v1/sessions/{sid}/end", json={"reason": "caller hung up"})
    assert r.json()["status"] == "ended" and r.json()["end_reason"] == "caller hung up"
    r = api.post(f"/v1/sessions/{sid}/responses", json={"input": "hi", "stream": False})
    assert r.status_code == 409 and r.json()["error"]["code"] == "session_ended"
    assert api.post("/v1/sessions", json={"session_id": sid}).status_code == 409
    assert api.delete(f"/v1/sessions/{sid}").status_code == 204
    assert api.get(f"/v1/sessions/{sid}").status_code == 404
    assert api.post("/v1/sessions", json={"session_id": "bad id!"}).status_code == 400


def test_upstream_auth_failure_is_reported(gateway_factory, vertex):
    from app.gcp.credentials import StaticTokenSource

    gw = gateway_factory(tokens=StaticTokenSource("wrong-token"), wait_ready=False, vertex_startup_probe=False)
    api, _ = new_tenant(gw)
    sid = session(api)
    r = api.post(f"/v1/sessions/{sid}/responses", json={"input": "hi", "stream": False})
    assert r.status_code == 502 and r.json()["error"]["code"] == "provider_permission_denied"
    assert vertex.get("/stats").json()["auth_failures"] >= 2  # retried once with a refreshed token
    assert httpx.get(f"{gw.url}/health/live").status_code == 200
