"""OpenAI-compatible chat completions through the official OpenAI SDK (what LiveKit's OpenAI plugin
uses), thought-signature repair for SDK clients, agent pseudo-models, and Vertex route failover."""
from __future__ import annotations

import json
import threading
import time

import httpx
import openai
import pytest

from tests.conftest import new_tenant, sse_events

TIME_TOOL = {"type": "function", "function": {
    "name": "lookup_patient", "description": "Find a patient",
    "parameters": {"type": "object", "properties": {"phone": {"type": "string"}}, "required": ["phone"]}}}


@pytest.fixture
def sdk(gateway):
    api, _ = new_tenant(gateway)
    client = openai.OpenAI(base_url=f"{gateway.url}/v1", api_key=api.key, max_retries=0)
    yield client, api
    api.close()


def test_streaming_and_blocking_completions(sdk, vertex):
    client, _ = sdk
    stream = client.chat.completions.create(model="default", stream=True, stream_options={"include_usage": True},
                                            messages=[{"role": "user", "content": "வணக்கம்"}])
    chunks = list(stream)
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "Echo: வணக்கம். This is a mock reply."
    assert chunks[-1].usage is not None and chunks[-1].usage.prompt_tokens > 0
    result = client.chat.completions.create(model="gemini-3.5-flash", max_completion_tokens=50,
                                            messages=[{"role": "user", "content": "hi"}])
    assert result.choices[0].message.content.startswith("Echo: hi")
    body = vertex.get("/stats").json()["last_body"]
    assert body["max_tokens"] == 50 and "max_completion_tokens" not in body  # Vertex rejects both together
    with pytest.raises(openai.BadRequestError) as exc:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    assert exc.value.body["code"] == "model_not_allowed"


def test_tool_calls_and_signature_repair_for_sdk_clients(sdk, vertex):
    """An OpenAI SDK client drops `extra_content`; replaying its tool call must still be accepted."""
    client, _ = sdk
    messages = [{"role": "user", "content": 'call lookup_patient {"phone": "98400"}'}]
    stream = client.chat.completions.create(model="default", stream=True, tools=[TIME_TOOL], messages=messages)
    calls: dict = {}
    finish = None
    for chunk in stream:
        for choice in chunk.choices:
            for tc in choice.delta.tool_calls or []:
                calls.setdefault(tc.index, {"id": tc.id, "name": tc.function.name, "arguments": ""})
                calls[tc.index]["arguments"] += tc.function.arguments or ""
            finish = choice.finish_reason or finish
    assert finish == "tool_calls" and calls[0]["name"] == "lookup_patient"
    assert json.loads(calls[0]["arguments"]) == {"phone": "98400"}
    # the client sends history back exactly like LiveKit does: no extra_content
    messages += [{"role": "assistant", "content": None, "tool_calls": [
        {"id": calls[0]["id"], "type": "function",
         "function": {"name": "lookup_patient", "arguments": calls[0]["arguments"]}}]},
        {"role": "tool", "tool_call_id": calls[0]["id"], "content": "Asha, next visit Nov 2"}]
    result = client.chat.completions.create(model="default", tools=[TIME_TOOL], messages=messages)
    assert result.choices[0].message.content == "Tool result: Asha, next visit Nov 2"
    replayed = vertex.get("/stats").json()["last_body"]["messages"][1]["tool_calls"][0]
    assert replayed["extra_content"]["google"]["thought_signature"].startswith("sig-")  # the real one, re-attached


def test_agent_pseudo_model(sdk, vertex):
    client, api = sdk
    agent = api.post("/v1/agents", json={"name": "Desk", "instructions": "#agent:desk front desk",
                                         "tools": [{"type": "builtin", "name": "get_current_time"}]}).json()
    # server tools run inside the server; the caller only sees the final text
    result = client.chat.completions.create(model=f"agent:{agent['id']}",
                                            messages=[{"role": "user", "content": "call get_current_time"}])
    assert result.choices[0].message.content.startswith("desk: Tool result:")
    assert result.choices[0].finish_reason == "stop" and result.usage.total_tokens > 0
    # the caller's own tools pass through as normal tool calls
    stream = client.chat.completions.create(model=f"agent:{agent['id']}", stream=True, tools=[TIME_TOOL],
                                            messages=[{"role": "system", "content": "Caller speaks Tamil."},
                                                      {"role": "user", "content": "call lookup_patient"}])
    chunks = list(stream)
    tool_chunks = [tc for c in chunks for ch in c.choices for tc in ch.delta.tool_calls or []]
    assert tool_chunks[0].function.name == "lookup_patient"
    assert chunks[-1].choices[0].finish_reason == "tool_calls"
    system = vertex.get("/stats").json()["last_body"]["messages"][0]["content"]
    assert system.startswith("#agent:desk front desk") and "Caller speaks Tamil." in system
    with pytest.raises(openai.NotFoundError):
        client.chat.completions.create(model="agent:agent_missing", messages=[{"role": "user", "content": "x"}])


def test_cancel_inflight_by_request_id(sdk, gateway):
    _, api = sdk
    out: dict = {}

    def run() -> None:
        with httpx.stream("POST", f"{gateway.url}/v1/chat/completions", timeout=20,
                          headers={"Authorization": f"Bearer {api.key}", "X-Request-ID": "req-cancel-me"},
                          json={"model": "default", "stream": True,
                                "messages": [{"role": "user", "content": "go [slow] a b c d e f g"}]}) as r:
            out["body"] = r.read().decode()

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.4)
    assert api.post("/v1/requests/req-cancel-me/cancel").json()["cancelled"] is True
    t.join(10)
    assert '"code":"cancelled"' in out["body"] and out["body"].rstrip().endswith("data: [DONE]")
    assert api.post("/v1/requests/req-cancel-me/cancel").status_code == 404


def test_failover_to_next_location(gateway, vertex):
    api, _ = new_tenant(gateway)
    vertex.post("/control", json={"fail": {"location": "asia-south1", "status": 429, "count": 1}})
    r = api.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert vertex.get("/stats").json()["by_location"] == {"asia-south1": 2, "global": 1}  # probe + 429, then global
    vertex.post("/control", json={"unavailable": ["google/gemini-3.5-flash", "asia-south1"]})
    vertex.post("/control", json={"unavailable": ["google/gemini-3.5-flash", "global"]})
    r = api.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and r.json()["model"] == "google/gemini-3.1-flash-lite"  # fallback model
    providers = api.get("/v1/providers").json()["data"][0]
    assert {"model": "google/gemini-3.5-flash", "location": "asia-south1"} in providers["unavailable_routes"]


def test_all_routes_exhausted(gateway, vertex):
    api, _ = new_tenant(gateway)
    for loc in ("asia-south1", "global"):
        vertex.post("/control", json={"fail": {"location": loc, "status": 503, "count": 20}})
    r = api.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["code"] == "provider_unavailable" and len(err["details"]["attempts"]) == 4  # VERTEX_MAX_ATTEMPTS
    with api.stream("POST", "/v1/chat/completions",
                    json={"stream": True, "messages": [{"role": "user", "content": "hi"}]}) as resp:
        assert resp.status_code == 503  # failed before the first frame: a real status code, not an SSE error


def test_circuit_breaker_opens(gateway_factory, vertex):
    gw = gateway_factory(vertex_circuit_failures=2, vertex_circuit_reset_s=30)
    api, _ = new_tenant(gw)
    vertex.post("/control", json={"fail": {"location": "asia-south1", "status": 500, "count": 2}})
    for _ in range(2):
        assert api.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}).status_code == 200
    loc = {s["location"]: s for s in api.get("/v1/providers/vertex").json()["locations"]}
    assert loc["asia-south1"]["circuit"] == "open"
    before = vertex.get("/stats").json()["by_location"].get("asia-south1", 0)
    api.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
    assert vertex.get("/stats").json()["by_location"].get("asia-south1", 0) == before  # skipped while open


def test_empty_candidate_is_retried_on_the_same_route(gateway, vertex):
    api, _ = new_tenant(gateway)
    vertex.post("/control", json={"empty": 1})
    r = api.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"].startswith("Echo")
    vertex.post("/control", json={"empty": 1})
    with api.stream("POST", "/v1/chat/completions",
                    json={"stream": True, "messages": [{"role": "user", "content": "hi"}]}) as resp:
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in sse_events(resp) if c.get("choices"))
    assert text.startswith("Echo")
    by_location = vertex.get("/stats").json()["by_location"]
    assert "global" not in by_location  # retried in asia-south1, not failed over
    loc = {s["location"]: s for s in api.get("/v1/providers/vertex").json()["locations"]}
    assert loc["asia-south1"]["circuit"] == "closed" and loc["asia-south1"]["consecutive_failures"] == 0


def test_tool_followup_rounds_use_reliable_thinking_level(gateway, vertex):
    api, _ = new_tenant(gateway)
    agent = api.post("/v1/agents", json={"name": "T", "instructions": "x",
                                         "tools": [{"type": "builtin", "name": "get_current_time"}]}).json()
    sid = api.post("/v1/sessions", json={"agent_id": agent["id"]}).json()["id"]
    api.post(f"/v1/sessions/{sid}/responses", json={"input": "call get_current_time", "stream": False})
    bodies = [b["body"] for b in vertex.get("/bodies").json()]
    assert bodies[-2]["reasoning_effort"] == "minimal"  # user turn: fastest setting
    assert bodies[-1]["messages"][-1]["role"] == "tool" and bodies[-1]["reasoning_effort"] == "low"
