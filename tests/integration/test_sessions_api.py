from __future__ import annotations

import threading
import time

import httpx
import orjson

LOOKUP = {"type": "function", "function": {"name": "lookup_booking", "description": "Find a booking",
                                           "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}}}}


def events(text: str) -> list[dict]:
    return [orjson.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def new_session(http: httpx.Client, **spec) -> str:
    r = http.post("/v1/sessions", json=spec)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_session_crud(http: httpx.Client):
    sid = new_session(http, session_id="call-42", system_prompt="Be brief.", metadata={"caller": "+91"},
                      params={"temperature": 0.2})
    assert sid == "call-42"
    assert http.post("/v1/sessions", json={"session_id": "call-42"}).status_code == 409
    got = http.get(f"/v1/sessions/{sid}").json()
    assert got["system_prompt"] == "Be brief." and got["params"] == {"temperature": 0.2}
    patched = http.patch(f"/v1/sessions/{sid}", json={"system_prompt": "Be kind.", "server_tools": "all"}).json()
    assert patched["system_prompt"] == "Be kind." and patched["server_tools"] == ["get_current_time"]
    assert http.patch(f"/v1/sessions/{sid}", json={"server_tools": ["nope"]}).status_code == 400
    assert http.patch(f"/v1/sessions/{sid}", json={"params": {"model": "x"}}).status_code == 400
    assert http.patch(f"/v1/sessions/{sid}", json={"unknown_field": 1}).status_code == 400
    assert any(s["id"] == sid for s in http.get("/v1/sessions").json()["data"])
    assert http.delete(f"/v1/sessions/{sid}").status_code == 204
    assert http.get(f"/v1/sessions/{sid}").json()["error"]["code"] == "session_not_found"


def test_non_streaming_turns_build_history(http: httpx.Client, engine_stats):
    sid = new_session(http, system_prompt="You are a receptionist.")
    first = http.post(f"/v1/sessions/{sid}/responses", json={"input": "hello", "stream": False}).json()
    assert first["status"] == "completed" and first["text"].strip() == "Echo: hello. This is a mock reply."
    assert first["timings"]["ttft_ms"] is not None and first["usage"]["prompt_tokens"] > 0
    http.post(f"/v1/sessions/{sid}/responses", json={"input": "and again", "stream": False})
    sent = engine_stats()["last_body"]
    assert [m["role"] for m in sent["messages"]] == ["system", "user", "assistant", "user"]
    assert sent["max_tokens"] == 200 and sent["stream_options"] == {"include_usage": True}
    conv = http.get(f"/v1/sessions/{sid}").json()
    assert [m["role"] for m in conv["conversation"]] == ["user", "assistant", "user", "assistant"]
    assert conv["stats"]["turns"] == 2


def test_streaming_events_and_segments(http: httpx.Client):
    sid = new_session(http, segmentation=True)
    r = http.post(f"/v1/sessions/{sid}/responses", json={"input": "First part. Second part"})
    evs = events(r.text)
    types = [e["type"] for e in evs]
    assert types[0] == "response.created" and types[-1] == "response.done"
    assert "response.text.delta" in types
    segments = [e["text"] for e in evs if e["type"] == "response.text.segment"]
    assert segments == ["Echo: First part.", "Second part.", "This is a mock reply."]
    assert evs[-1]["status"] == "completed"


def test_server_tool_loop(http: httpx.Client):
    sid = new_session(http, server_tools=["get_current_time"])
    r = http.post(f"/v1/sessions/{sid}/responses",
                  json={"input": 'call get_current_time {"timezone": "Asia/Kolkata"}', "stream": False}).json()
    assert r["status"] == "completed"
    executed = r["server_tool_calls"][0]
    assert executed["name"] == "get_current_time" and not executed["is_error"]
    assert "Asia/Kolkata" in executed["output"]
    assert r["text"].startswith("Tool result:")
    roles = [m["role"] for m in http.get(f"/v1/sessions/{sid}").json()["conversation"]]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_client_tool_round_trip(http: httpx.Client):
    sid = new_session(http, tools=[LOOKUP])
    r = http.post(f"/v1/sessions/{sid}/responses", json={"input": 'call lookup_booking {"id": 7}', "stream": False})
    body = r.json()
    assert body["status"] == "requires_action"
    call = body["tool_calls"][0]
    assert call["name"] == "lookup_booking" and orjson.loads(call["arguments"]) == {"id": 7}
    bad = http.post(f"/v1/sessions/{sid}/tool_results", json={"results": [{"call_id": "nope", "output": "x"}]})
    assert bad.status_code == 400
    r = http.post(f"/v1/sessions/{sid}/tool_results", json={
        "results": [{"call_id": call["call_id"], "output": {"slot": "5pm"}}], "stream": False})
    done = r.json()
    assert done["status"] == "completed" and '"slot":"5pm"' in done["text"]


def test_new_input_abandons_pending_tool_calls(http: httpx.Client, engine_stats):
    sid = new_session(http, tools=[LOOKUP])
    http.post(f"/v1/sessions/{sid}/responses", json={"input": "call lookup_booking", "stream": False})
    http.post(f"/v1/sessions/{sid}/responses", json={"input": "never mind", "stream": False})
    roles = [m["role"] for m in engine_stats()["last_body"]["messages"]]
    assert roles == ["user", "assistant", "tool", "user"]  # every tool call keeps a result


def test_cancel_with_spoken_text_and_discard(http: httpx.Client):
    sid = new_session(http)
    out: dict = {}

    def run(key: str, text: str):
        out[key] = http.post(f"/v1/sessions/{sid}/responses", json={"input": text, "stream": False}).json()

    t = threading.Thread(target=run, args=("a", "[slow] tell me about visiting hours"))
    t.start()
    time.sleep(0.6)
    r = http.post(f"/v1/sessions/{sid}/cancel", json={"spoken_text": "Echo:"}).json()
    t.join(10)
    assert r["cancelled"] and out["a"]["status"] == "cancelled"
    conv = http.get(f"/v1/sessions/{sid}").json()["conversation"]
    assert conv[-1] == {**conv[-1], "role": "assistant", "content": "Echo:", "interrupted": True}

    t = threading.Thread(target=run, args=("b", "[slow] speculative interim transcript"))
    t.start()
    time.sleep(0.4)
    assert http.post(f"/v1/sessions/{sid}/cancel", json={"discard": True}).json()["cancelled"]
    t.join(10)
    assert len(http.get(f"/v1/sessions/{sid}").json()["conversation"]) == len(conv)


def test_barge_in_by_new_response(http: httpx.Client):
    sid = new_session(http)
    out: dict = {}
    t = threading.Thread(target=lambda: out.update(first=http.post(
        f"/v1/sessions/{sid}/responses", json={"input": "[slow] long answer", "stream": False}).json()))
    t.start()
    time.sleep(0.4)
    second = http.post(f"/v1/sessions/{sid}/responses", json={"input": "stop, new question", "stream": False}).json()
    t.join(10)
    assert out["first"]["status"] == "cancelled" and out["first"]["cancel_reason"] == "barge_in"
    assert second["status"] == "completed"
    conv = http.get(f"/v1/sessions/{sid}").json()["conversation"]
    assert [m["role"] for m in conv] == ["user", "assistant", "user", "assistant"]
    assert conv[1].get("interrupted") is True


def test_length_and_failure_statuses(http: httpx.Client):
    sid = new_session(http)
    r = http.post(f"/v1/sessions/{sid}/responses", json={"input": "hi", "params": {"max_tokens": 2}, "stream": False})
    assert r.json()["status"] == "incomplete" and r.json()["finish_reason"] == "length"
    r = http.post(f"/v1/sessions/{sid}/responses", json={"input": "[error]"})
    done = events(r.text)[-1]
    assert done["status"] == "failed" and done["error"]["code"] == "engine_error"
    r = http.post(f"/v1/sessions/{sid}/responses", json={"input": "[error]", "stream": False})
    assert r.status_code == 502
    assert http.post(f"/v1/sessions/{sid}/responses", json={"input": {"role": "system", "content": "x"}}).status_code == 400


def test_context_window_is_managed(gateway_factory, engine_stats):
    gw = gateway_factory(context_max_tokens=400, default_max_tokens=64)
    with httpx.Client(base_url=gw.url, timeout=30) as http:
        sid = new_session(http, system_prompt="Short prompt.")
        for i in range(12):
            r = http.post(f"/v1/sessions/{sid}/responses",
                          json={"input": f"turn {i} " + "padding words here " * 12, "stream": False})
            assert r.json()["status"] == "completed", r.text
        stats = http.get(f"/v1/sessions/{sid}").json()["stats"]
        assert stats["trimmed_messages"] > 0
        sent = engine_stats()["last_body"]["messages"]
        assert sent[0]["role"] == "system" and sent[1]["role"] == "user"
        assert sent[-1]["content"].startswith("turn 11")


def test_append_and_clear_messages(http: httpx.Client):
    sid = new_session(http)
    r = http.post(f"/v1/sessions/{sid}/messages", json={"messages": [
        {"role": "user", "content": "context A"}, {"role": "assistant", "content": "noted"}]})
    assert [m["role"] for m in r.json()["added"]] == ["user", "assistant"]
    assert http.post(f"/v1/sessions/{sid}/responses", json={"stream": False}).json()["status"] == "completed"
    assert http.delete(f"/v1/sessions/{sid}/messages").json()["messages"] == 0
    assert http.post(f"/v1/sessions/{sid}/responses", json={"stream": False}).status_code == 400  # no user message
