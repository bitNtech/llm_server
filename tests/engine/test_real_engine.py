"""Tests against a real vLLM engine (skipped unless LLM_TEST_ENGINE_URL is set).

    LLM_TEST_ENGINE_URL=http://127.0.0.1:8001 LLM_TEST_ENGINE_KEY=<ENGINE_API_KEY> ./run.sh test tests/engine

Use it on a fresh EC2 instance (or with a small CPU model locally) to verify the model's chat
template, tool parser, prefix caching and abort-on-cancel through the gateway.
"""
from __future__ import annotations

import os
import re
import time

import httpx
import orjson
import pytest

ENGINE_URL = os.environ.get("LLM_TEST_ENGINE_URL", "")
pytestmark = [pytest.mark.engine, pytest.mark.skipif(not ENGINE_URL, reason="LLM_TEST_ENGINE_URL not set")]


@pytest.fixture
def http(gateway_factory):
    served = httpx.get(f"{ENGINE_URL}/v1/models", headers=_engine_headers()).json()["data"][0]["id"]
    gw = gateway_factory(engine_url=ENGINE_URL, engine_api_key=os.environ.get("LLM_TEST_ENGINE_KEY", ""),
                         engine_served_model_name=served, default_chat_template_kwargs='{"enable_thinking": false}',
                         engine_first_token_timeout_s=300, generation_timeout_s=600)
    deadline = time.monotonic() + 30
    while httpx.get(f"{gw.url}/health/ready").status_code != 200:
        assert time.monotonic() < deadline
        time.sleep(0.2)
    with httpx.Client(base_url=gw.url, timeout=600) as client:
        yield client


def _engine_headers() -> dict[str, str]:
    key = os.environ.get("LLM_TEST_ENGINE_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _running_requests() -> float:
    text = httpx.get(f"{ENGINE_URL}/metrics").text
    match = re.search(r"^vllm:num_requests_running\{[^}]*\} ([0-9.]+)", text, re.M)
    return float(match.group(1)) if match else 0.0


def test_streaming_chat(http: httpx.Client):
    r = http.post("/v1/chat/completions", json={"stream": True, "max_tokens": 24,
                                                "messages": [{"role": "user", "content": "Say hello."}]})
    frames = [line[6:] for line in r.text.splitlines() if line.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    text = "".join((orjson.loads(f)["choices"][0]["delta"].get("content") or "") for f in frames[:-1]
                   if orjson.loads(f).get("choices"))
    assert text.strip()
    assert "<think>" not in text


def test_session_turns_hit_the_prefix_cache(http: httpx.Client):
    system = "You are a hospital receptionist. " + "Be polite and concise. " * 40
    sid = http.post("/v1/sessions", json={"system_prompt": system, "params": {"max_tokens": 16}}).json()["id"]
    http.post(f"/v1/sessions/{sid}/responses", json={"input": "Hello", "stream": False})
    second = http.post(f"/v1/sessions/{sid}/responses", json={"input": "What are your hours?", "stream": False}).json()
    assert second["status"] in ("completed", "incomplete")
    assert second["usage"]["cached_tokens"] > 0, second["usage"]


def test_tool_call_is_parsed(http: httpx.Client):
    tool = {"type": "function", "function": {
        "name": "book_appointment", "description": "Book a doctor appointment",
        "parameters": {"type": "object", "properties": {"department": {"type": "string"}},
                       "required": ["department"]}}}
    # tool_choice="auto": the gemma4 parser does not enforce "required" yet (vllm#53363)
    sid = http.post("/v1/sessions", json={
        "system_prompt": "You book appointments. Always use the book_appointment tool to book.",
        "tools": [tool], "params": {"max_tokens": 128, "temperature": 0}}).json()["id"]
    r = http.post(f"/v1/sessions/{sid}/responses", json={
        "input": "Book me a cardiology appointment.", "stream": False}).json()
    assert r["status"] == "requires_action", r
    assert r["tool_calls"][0]["name"] == "book_appointment"
    assert "department" in orjson.loads(r["tool_calls"][0]["arguments"])


def test_cancel_aborts_engine_request(http: httpx.Client):
    sid = http.post("/v1/sessions", json={"params": {"max_tokens": 2000}}).json()["id"]
    with http.stream("POST", f"/v1/sessions/{sid}/responses",
                     json={"input": "Count from 1 to 500, one number per line."}) as r:
        for line in r.iter_lines():
            if '"response.text.delta"' in line:
                break
    deadline = time.monotonic() + 10
    while _running_requests() > 0:
        assert time.monotonic() < deadline, "engine kept generating after the client left"
        time.sleep(0.2)
