from __future__ import annotations

import threading
import time

import httpx
import orjson


def sse_frames(text: str) -> list[str]:
    return [line[6:] for line in text.splitlines() if line.startswith("data: ")]


def test_health_endpoints(http: httpx.Client):
    assert http.get("/health/live").json() == {"status": "ok"}
    ready = http.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["model"] == "mock-model"
    detail = http.get("/health").json()
    assert detail["engine"]["ready"] and detail["engine"]["max_model_len"] == 4096
    assert detail["server_tools"] == ["get_current_time"]


def test_models_and_tokenize(http: httpx.Client):
    assert http.get("/v1/models").json()["data"][0]["id"] == "mock-model"
    count = http.post("/v1/tokenize", json={"messages": [{"role": "user", "content": "hello world"}]}).json()
    assert count["count"] > 0


def test_non_streaming_completion(http: httpx.Client):
    r = http.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]},
                  headers={"X-Request-ID": "my-req-1"})
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "my-req-1"
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Echo: hello. This is a mock reply."
    assert body["usage"]["completion_tokens"] > 0


def test_streaming_completion_hides_unrequested_usage(http: httpx.Client):
    r = http.post("/v1/chat/completions", json={"model": "default", "stream": True,
                                                "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    frames = sse_frames(r.text)
    assert frames[-1] == "[DONE]"
    chunks = [orjson.loads(f) for f in frames[:-1]]
    assert all(c["choices"] for c in chunks)  # no usage-only chunk the client did not ask for
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert text.strip() == "Echo: hi. This is a mock reply."

    r = http.post("/v1/chat/completions", json={"stream": True, "stream_options": {"include_usage": True},
                                                "messages": [{"role": "user", "content": "hi"}]})
    usage = [orjson.loads(f) for f in sse_frames(r.text)[:-1] if not orjson.loads(f)["choices"]]
    assert usage and usage[0]["usage"]["prompt_tokens"] > 0


def test_tool_call_deltas_are_forwarded(http: httpx.Client):
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    r = http.post("/v1/chat/completions", json={
        "stream": True, "tools": tools, "messages": [{"role": "user", "content": 'call lookup {"id": 7}'}]})
    chunks = [orjson.loads(f) for f in sse_frames(r.text)[:-1]]
    args = "".join(tc["function"].get("arguments", "") for c in chunks
                   for tc in c["choices"][0]["delta"].get("tool_calls", []))
    assert orjson.loads(args) == {"id": 7}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_errors_use_the_envelope(http: httpx.Client):
    r = http.post("/v1/chat/completions", json={"max_tokens": 10**6, "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    assert r.json()["error"]["request_id"] == r.headers["x-request-id"]
    r = http.post("/v1/chat/completions", content=b"{nope", headers={"content-type": "application/json"})
    assert r.status_code == 400
    r = http.post("/v1/chat/completions", json={"model": "other", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
    r = http.post("/v1/chat/completions", json={"stream": True, "model": "other",
                                                "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 404  # failure before the first frame keeps its status code
    assert http.get("/nope").status_code == 404


def test_in_stream_engine_error_becomes_error_frame(http: httpx.Client):
    r = http.post("/v1/chat/completions", json={"stream": True, "messages": [{"role": "user", "content": "[error]"}]})
    frames = sse_frames(r.text)
    assert frames[-1] == "[DONE]"
    assert orjson.loads(frames[-2])["error"]["code"] == "engine_error"


def test_body_size_limit(gateway_factory):
    gw = gateway_factory(max_request_body_bytes=2048)
    r = httpx.post(f"{gw.url}/v1/chat/completions", json={"messages": [{"role": "user", "content": "x" * 5000}]})
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"


def test_authentication(gateway_factory):
    gw = gateway_factory(api_keys="sekret", metrics_require_auth=True)
    body = {"messages": [{"role": "user", "content": "x"}]}
    assert httpx.post(f"{gw.url}/v1/chat/completions", json=body).status_code == 401
    assert httpx.post(f"{gw.url}/v1/chat/completions", json=body, headers={"X-API-Key": "bad"}).status_code == 401
    ok = httpx.post(f"{gw.url}/v1/chat/completions", json=body, headers={"Authorization": "Bearer sekret"})
    assert ok.status_code == 200
    assert httpx.get(f"{gw.url}/v1/chat/completions?api_key=sekret").status_code in (401, 405)
    assert httpx.get(f"{gw.url}/health/ready").status_code == 200  # probes stay open
    assert httpx.get(f"{gw.url}/health").status_code == 401
    assert httpx.get(f"{gw.url}/metrics").status_code == 401
    assert httpx.get(f"{gw.url}/metrics", headers={"X-API-Key": "sekret"}).status_code == 200


def test_rate_limit(gateway_factory):
    gw = gateway_factory(rate_limit_per_minute=2)
    body = {"messages": [{"role": "user", "content": "x"}]}
    codes = [httpx.post(f"{gw.url}/v1/chat/completions", json=body).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_cancel_by_request_id(http: httpx.Client, engine_stats):
    aborted = engine_stats()["aborted"]
    result: dict = {}

    def consume():
        result["text"] = http.post("/v1/chat/completions", headers={"X-Request-ID": "cancel-me"}, json={
            "stream": True, "messages": [{"role": "user", "content": "[slow] tell me a long story"}]}).text

    t = threading.Thread(target=consume)
    t.start()
    time.sleep(0.5)
    assert http.post("/v1/requests/cancel-me/cancel").json() == {"id": "cancel-me", "cancelled": True}
    t.join(10)
    frames = sse_frames(result["text"])
    assert orjson.loads(frames[-2])["error"]["code"] == "cancelled"
    assert http.post("/v1/requests/cancel-me/cancel").status_code == 404
    _wait_for(lambda: engine_stats()["aborted"] > aborted)


def test_client_disconnect_aborts_engine_request(gateway, engine_stats):
    aborted = engine_stats()["aborted"]
    with httpx.Client(base_url=gateway.url) as client, client.stream("POST", "/v1/chat/completions", json={
        "stream": True, "messages": [{"role": "user", "content": "[slow] a long answer please"}]}) as r:
        for line in r.iter_lines():
            if '"content":"Echo: "' in line:
                break  # leave mid-stream
    _wait_for(lambda: engine_stats()["aborted"] > aborted)


def test_engine_down_returns_503(gateway_factory):
    gw = gateway_factory(engine_url="http://127.0.0.1:9")  # nothing listens there
    assert httpx.get(f"{gw.url}/health/ready").status_code == 503
    r = httpx.post(f"{gw.url}/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 503 and r.json()["error"]["code"] == "engine_unavailable"
    assert r.headers.get("retry-after")


def test_metrics_exposed(http: httpx.Client):
    http.post("/v1/chat/completions", json={"stream": True, "messages": [{"role": "user", "content": "hi"}]})
    text = http.get("/metrics").text
    assert "llm_time_to_first_token_seconds_bucket" in text
    assert 'gateway_http_requests_total{method="POST",route="/v1/chat/completions"' in text
    assert "vllm:num_requests_running" in http.get("/metrics/engine").text


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.05)
