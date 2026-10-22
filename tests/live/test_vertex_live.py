"""Live tests against real Vertex AI (opt-in: LIVE_VERTEX=1, Application Default Credentials and
GCP_PROJECT_ID / LIVE_GCP_PROJECT). Firestore still runs on the emulator. They spend a few cents.

    LIVE_VERTEX=1 LIVE_GCP_PROJECT=aica-voice-pilot .venv/bin/python -m pytest tests/live -s
"""
from __future__ import annotations

import os
import statistics
import time
import uuid

import openai
import orjson
import pytest
from websockets.sync.client import connect

from tests.conftest import ADMIN_KEY, Api, LiveServer, new_tenant, sse_events

pytestmark = pytest.mark.live

PROJECT = os.environ.get("LIVE_GCP_PROJECT") or os.environ.get("GCP_PROJECT_ID") or ""
if not os.environ.get("LIVE_VERTEX") or not PROJECT:
    pytest.skip("live Vertex tests are opt-in (LIVE_VERTEX=1 LIVE_GCP_PROJECT=<project>)", allow_module_level=True)


@pytest.fixture(scope="module")
def live_gateway(firestore_emulator):
    from app.config import Settings
    from app.main import create_app

    settings = Settings(_env_file=None, gcp_project_id=PROJECT, admin_api_keys=ADMIN_KEY,
                        vertex_locations=os.environ.get("LIVE_VERTEX_LOCATIONS", "asia-south1,global"),
                        firestore_collection_prefix="live" + uuid.uuid4().hex[:8] + "_",
                        allow_private_network_targets=True, log_level="WARNING")
    server = LiveServer(create_app(settings)).start()
    deadline = time.monotonic() + 60
    import httpx
    while httpx.get(f"{server.url}/health/ready").status_code != 200:
        assert time.monotonic() < deadline, httpx.get(f"{server.url}/health/ready").text
        time.sleep(0.2)
    yield server
    server.stop()


@pytest.fixture(scope="module")
def tenant(live_gateway):
    api, _ = new_tenant(live_gateway, "Live")
    yield api
    api.close()


def turn(api: Api, sid: str, text: str) -> list[dict]:
    with api.stream("POST", f"/v1/sessions/{sid}/responses", json={"input": text}) as resp:
        assert resp.status_code == 200, resp.read()
        return sse_events(resp)


def test_streaming_turn(tenant):
    agent = tenant.post("/v1/agents", json={
        "name": "Reception", "instructions": "You are Priya, a hospital receptionist in Chennai. "
                                             "Reply in one short spoken sentence."}).json()
    sid = tenant.post("/v1/sessions", json={"agent_id": agent["id"]}).json()["id"]
    events = turn(tenant, sid, "Hi, what are your visiting hours?")
    done = events[-1]
    assert done["status"] == "completed", done
    assert done["text"] and done["usage"]["prompt_tokens"] > 0
    assert any(e["type"] == "response.text.segment" for e in events)
    print(f"\nreply={done['text']!r} model={done['model']} timings={done['timings']}")


def test_multi_turn_parallel_tool_calls_with_real_signatures(tenant):
    agent = tenant.post("/v1/agents", json={
        "name": "Clock", "tools": [{"type": "builtin", "name": "get_current_time"}],
        "instructions": "Always use get_current_time to answer questions about time; call it once per city "
                        "(in parallel). Answer in one short sentence."}).json()
    sid = tenant.post("/v1/sessions", json={"agent_id": agent["id"]}).json()["id"]
    events = turn(tenant, sid, "What time is it right now in Chennai (Asia/Kolkata) and in London (Europe/London)?")
    done = events[-1]
    assert done["status"] == "completed", done
    calls = [e for e in events if e["type"] == "response.tool_result"]
    assert len(calls) >= 2 and not any(c["is_error"] for c in calls), calls
    # the next turn replays the history (tool calls + Gemini thought signatures) to Vertex
    done2 = turn(tenant, sid, "And what about Tokyo (Asia/Tokyo)?")[-1]
    assert done2["status"] == "completed", done2
    print(f"\nturn1={done['text']!r}\nturn2={done2['text']!r}")


def test_model_driven_handoff(tenant):
    billing = tenant.post("/v1/agents", json={
        "name": "Billing", "instructions": "You are the billing desk. Start your reply with 'Billing desk here'. "
                                          "Answer in one short sentence."}).json()
    front = tenant.post("/v1/agents", json={
        "name": "Front Desk", "instructions": "You are the front desk. If the user mentions a bill, invoice or "
                                              "payment, immediately call transfer_to_billing without replying.",
        "handoffs": [{"agent_id": billing["id"], "description": "bills, invoices and payments"}]}).json()
    sid = tenant.post("/v1/sessions", json={"agent_id": front["id"]}).json()["id"]
    events = turn(tenant, sid, "I have a question about my hospital bill from last week.")
    handoffs = [e for e in events if e["type"] == "agent.handoff"]
    done = events[-1]
    assert handoffs and handoffs[0]["to_agent"]["name"] == "Billing", [e["type"] for e in events]
    assert done["status"] == "completed" and done["agent"]["name"] == "Billing" and done["text"], done
    assert turn(tenant, sid, "How can I pay it?")[-1]["status"] == "completed"
    print(f"\nafter handoff={done['text']!r}")


def test_openai_sdk_tool_round_trip_without_extra_content(live_gateway, tenant):
    client = openai.OpenAI(base_url=f"{live_gateway.url}/v1", api_key=tenant.key, max_retries=0)
    tool = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                             "parameters": {"type": "object", "properties": {
                                                 "city": {"type": "string"}}, "required": ["city"]}}}
    messages = [{"role": "user", "content": "What's the weather in Chennai? Use the tool."}]
    first = client.chat.completions.create(model="default", tools=[tool], messages=messages)
    calls = first.choices[0].message.tool_calls
    assert calls and calls[0].function.name == "get_weather"
    # replay like LiveKit/OpenAI SDKs do (no extra_content): the gateway re-attaches the signature
    messages += [{"role": "assistant", "content": None, "tool_calls": [
        {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
        for c in calls]}]
    messages += [{"role": "tool", "tool_call_id": c.id, "content": '{"forecast": "sunny, 31C"}'} for c in calls]
    second = client.chat.completions.create(model="default", tools=[tool], messages=messages)
    assert "31" in (second.choices[0].message.content or ""), second
    stream = client.chat.completions.create(model="default", stream=True, stream_options={"include_usage": True},
                                            messages=[{"role": "user", "content": "Say hello in Tamil."}])
    chunks = list(stream)
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert chunks[-1].usage and chunks[-1].usage.total_tokens > 0


def test_structured_output_workflow(tenant):
    wf = tenant.post("/v1/workflows", json={"name": "extract", "steps": [{
        "id": "lead", "type": "llm", "prompt": "Transcript: {{input.transcript}}",
        "instructions": "Extract the caller's name and whether they want an appointment.",
        "response_format": {"type": "json_schema", "json_schema": {"name": "lead", "schema": {
            "type": "object", "properties": {"name": {"type": "string"}, "wants_appointment": {"type": "boolean"}},
            "required": ["name", "wants_appointment"]}}}}]}).json()
    run = tenant.post(f"/v1/workflows/{wf['id']}/run", json={
        "input": {"transcript": "Hi, this is Karthik. I'd like to book a dental check-up next week."}}).json()
    deadline = time.monotonic() + 60
    doc = run
    while doc["status"] not in ("completed", "failed") and time.monotonic() < deadline:
        time.sleep(0.5)
        doc = tenant.get(f"/v1/executions/{run['id']}").json()
    assert doc["status"] == "completed", doc
    assert doc["result"]["wants_appointment"] is True and "Karthik" in doc["result"]["name"]


def test_latency_over_websocket(live_gateway, tenant):
    """Client-observed TTFT over the realtime WebSocket vs. Vertex's own TTFT (gateway overhead)."""
    agent = tenant.post("/v1/agents", json={"name": "Fast", "instructions": "Reply with one short sentence.",
                                            "params": {"max_tokens": 60}}).json()
    samples = []
    with connect(f"{live_gateway.ws_url}/v1/realtime?agent_id={agent['id']}",
                 additional_headers={"Authorization": f"Bearer {tenant.key}"}) as ws:
        ws.recv(timeout=10)
        for i in range(8):
            start = time.perf_counter()
            ws.send(orjson.dumps({"type": "response.create", "input": f"Say a short greeting number {i}."}).decode())
            first = None
            while True:
                event = orjson.loads(ws.recv(timeout=30))
                if event["type"] == "response.text.delta" and first is None:
                    first = time.perf_counter() - start
                if event["type"] == "response.done":
                    assert event["status"] == "completed", event
                    samples.append((first * 1000, event["timings"]["upstream_ttft_ms"],
                                    event["timings"]["gateway_overhead_ms"]))
                    break
    client_ttft = statistics.median(s[0] for s in samples)
    upstream = statistics.median(s[1] for s in samples)
    overhead = statistics.median(s[2] for s in samples)
    print(f"\nWS TTFT p50={client_ttft:.0f}ms  Vertex TTFT p50={upstream:.0f}ms  gateway overhead p50={overhead:.1f}ms"
          f"  samples={[(round(a), round(b), c) for a, b, c in samples]}")
    assert overhead < 50  # the server's own contribution; Vertex dominates TTFT


def test_admin_cleanup_note(live_gateway):
    with Api(live_gateway.url, ADMIN_KEY) as admin:
        assert admin.get("/health").json()["provider"]["ready"]
