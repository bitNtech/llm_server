"""Asynchronous executions, workflows, signed webhooks, post-session hooks, recovery and usage."""
from __future__ import annotations

import time
from datetime import UTC, datetime

import orjson

from app.security.crypto import verify_signature
from tests.conftest import Api, new_tenant


def wait_status(api: Api, execution_id: str, statuses: set[str], timeout: float = 20) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        doc = api.get(f"/v1/executions/{execution_id}").json()
        if doc["status"] in statuses:
            return doc
        time.sleep(0.1)
    raise AssertionError(f"execution stuck in {doc['status']}: {doc}")


def wait_webhook(target, timeout: float = 10) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        hooks = [r for r in target.get("/records").json() if r["name"] == "webhook"]
        if hooks:
            return hooks[-1]
        time.sleep(0.1)
    raise AssertionError("no webhook delivered")


def test_agent_execution_with_server_tools_and_signed_callback(gateway, target, tool_target, vertex):
    api, tenant = new_tenant(gateway)
    agent = api.post("/v1/agents", json={"name": "Summarizer", "instructions": "#agent:sum summarize",
                                         "tools": [{"type": "builtin", "name": "get_current_time"},
                                                   {"type": "client", "name": "cannot_run_async"}]}).json()
    r = api.post("/v1/executions", json={"type": "agent", "agent_id": agent["id"], "input": "call get_current_time",
                                         "callback_url": f"{tool_target.url}/webhook", "metadata": {"job": 1}},
                 headers={"Idempotency-Key": "exec-1"})
    assert r.status_code == 202, r.text
    execution = r.json()
    assert execution["status"] in ("queued", "running")
    replay = api.post("/v1/executions", json={"type": "agent", "agent_id": agent["id"], "input": "call get_current_time",
                                              "callback_url": f"{tool_target.url}/webhook", "metadata": {"job": 1}},
                      headers={"Idempotency-Key": "exec-1"})
    assert replay.json()["id"] == execution["id"]
    doc = wait_status(api, execution["id"], {"completed", "failed"})
    assert doc["status"] == "completed", doc
    assert doc["result"]["text"].startswith("sum: Tool result:")
    assert doc["result"]["server_tool_calls"][0]["name"] == "get_current_time"
    hook = wait_webhook(target)
    assert hook["headers"]["x-aica-event"] == "execution.completed"
    assert verify_signature(tenant["webhook_secret"], hook["body"].encode(), hook["headers"]["x-aica-signature"])
    payload = orjson.loads(hook["body"])
    assert payload["data"]["id"] == execution["id"] and payload["data"]["metadata"] == {"job": 1}
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not api.get(f"/v1/executions/{execution['id']}").json().get("callback"):
        time.sleep(0.1)
    assert api.get(f"/v1/executions/{execution['id']}").json()["callback"]["status"] == "delivered"


def test_workflow_with_llm_condition_tool_and_set(gateway, target, tool_target, vertex):
    api, _ = new_tenant(gateway)
    tool = api.post("/v1/tools", json={"name": "crm_push", "description": "push lead", "side_effects": True,
                                       "http": {"url": f"{tool_target.url}/book"},
                                       "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}}
                    ).json()
    workflow = api.post("/v1/workflows", json={
        "name": "post_call",
        "steps": [
            {"id": "extract", "type": "llm", "prompt": "[json] {{input.transcript}}",
             "response_format": {"type": "json_object"}},
            {"id": "is_lead", "type": "condition",
             "if": {"left": "{{steps.extract.output.interested}}", "op": "truthy"}, "then": "push", "else": "skip"},
            {"id": "push", "type": "tool", "tool": {"type": "tool", "id": tool["id"]},
             "arguments": {"name": "{{steps.extract.output.name}}"}, "next": "done"},
            {"id": "skip", "type": "set", "value": "not a lead", "next": "done"},
            {"id": "done", "type": "set", "value": {"lead": "{{steps.extract.output.name}}",
                                                   "booking": "{{steps.push.output.booking_id}}"}},
        ],
        "output": "{{steps.done.output}}"}).json()
    assert workflow["version"] == 1
    run = api.post(f"/v1/workflows/{workflow['id']}/run", json={"input": {"transcript": "yes I am interested"}})
    assert run.status_code == 202, run.text
    doc = wait_status(api, run.json()["id"], {"completed", "failed"})
    assert doc["status"] == "completed", doc
    assert doc["result"] == {"lead": "Asha", "booking": "B0001"}
    assert set(doc["steps"]) == {"extract", "is_lead", "push", "done"}
    assert doc["steps"]["extract"]["output"] == {"ok": True, "interested": True, "name": "Asha"}
    # branch not taken
    run = api.post(f"/v1/workflows/{workflow['id']}/run", json={"input": {"transcript": "no thanks"}})
    doc = wait_status(api, run.json()["id"], {"completed", "failed"})
    assert "skip" in doc["steps"] and "push" not in doc["steps"]
    # SSE events of a finished execution replay from the stored document
    with api.stream("GET", f"/v1/executions/{run.json()['id']}/events") as resp:
        lines = [line for line in resp.iter_lines() if line.startswith("event:")]
    assert "event: execution.completed" in lines


def test_workflow_failure_and_validation(gateway, vertex):
    api, _ = new_tenant(gateway)
    r = api.post("/v1/workflows", json={"name": "bad", "steps": [{"id": "a", "type": "llm"}]})
    assert r.status_code == 400
    wf = api.post("/v1/workflows", json={"name": "fails", "steps": [
        {"id": "call", "type": "tool", "tool": {"type": "builtin", "name": "get_current_time"},
         "arguments": {"timezone": 5}},  # schema violation -> step fails
        {"id": "never", "type": "set", "value": 1}]}).json()
    doc = wait_status(api, api.post(f"/v1/workflows/{wf['id']}/run", json={}).json()["id"], {"completed", "failed"})
    assert doc["status"] == "failed" and doc["error"]["code"] == "tool_failed"
    assert doc["steps"]["call"]["status"] == "failed" and "never" not in doc["steps"]


def test_cancel_running_execution(gateway, vertex):
    api, _ = new_tenant(gateway)
    agent = api.post("/v1/agents", json={"name": "Slow", "instructions": "x"}).json()
    run = api.post("/v1/executions", json={"type": "agent", "agent_id": agent["id"],
                                           "input": "slow [slow] a b c d e f g h i j k l m n o p"}).json()
    wait_status(api, run["id"], {"running"})
    assert api.post(f"/v1/executions/{run['id']}/cancel").status_code == 200
    doc = wait_status(api, run["id"], {"cancelled", "completed", "failed"})
    assert doc["status"] == "cancelled"
    r = api.post(f"/v1/executions/{run['id']}/cancel")
    assert r.status_code == 409 and r.json()["error"]["code"] == "execution_finished"


def test_execution_recovered_by_another_instance(gateway_factory, vertex):
    """An instance that never runs executions accepts one; a worker instance claims it from the queue."""
    intake = gateway_factory(execution_workers=0)
    api, _ = new_tenant(intake)
    agent = api.post("/v1/agents", json={"name": "A", "instructions": "x"}).json()
    run = api.post("/v1/executions", json={"type": "agent", "agent_id": agent["id"], "input": "hello"}).json()
    time.sleep(0.5)
    assert api.get(f"/v1/executions/{run['id']}").json()["status"] == "queued"
    gateway_factory(firestore_collection_prefix=intake.settings.firestore_collection_prefix)
    doc = wait_status(api, run["id"], {"completed", "failed"}, timeout=30)
    assert doc["status"] == "completed" and doc["attempts"] == 1


def test_session_end_triggers_workflow(gateway, vertex):
    api, _ = new_tenant(gateway)
    wf = api.post("/v1/workflows", json={"name": "after_call", "steps": [
        {"id": "count", "type": "set", "value": {"session": "{{input.session_id}}",
                                                 "first_user_line": "{{input.transcript.0.content}}"}}]}).json()
    agent = api.post("/v1/agents", json={"name": "Desk", "instructions": "x",
                                         "on_session_end": {"workflow_id": wf["id"]}}).json()
    sid = api.post("/v1/sessions", json={"agent_id": agent["id"], "session_id": "call-77"}).json()["id"]
    api.post(f"/v1/sessions/{sid}/responses", json={"input": "book a scan", "stream": False})
    api.post(f"/v1/sessions/{sid}/end", json={"reason": "hangup"})
    deadline = time.monotonic() + 15
    runs = []
    while time.monotonic() < deadline and not runs:
        runs = [e for e in api.get("/v1/executions").json()["data"] if e.get("workflow_id") == wf["id"]]
        time.sleep(0.1)
    doc = wait_status(api, runs[0]["id"], {"completed", "failed"})
    assert doc["result"] == {"session": "call-77", "first_user_line": "book a scan"}
    assert doc["metadata"] == {"trigger": "session_end", "session_id": "call-77"}


def test_usage_is_tracked(gateway, vertex):
    api, _ = new_tenant(gateway)
    agent = api.post("/v1/agents", json={"name": "U", "instructions": "x"}).json()
    sid = api.post("/v1/sessions", json={"agent_id": agent["id"]}).json()["id"]
    for text in ("one", "two"):
        api.post(f"/v1/sessions/{sid}/responses", json={"input": text, "stream": False})
    api.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "three"}]})
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    usage = api.get(f"/v1/usage?start={today}&end={today}").json()
    assert usage["totals"]["requests"] == 3 and usage["totals"]["prompt_tokens"] > 0
    by_api = api.get(f"/v1/usage?start={today}&end={today}&group_by=api").json()["groups"]
    assert by_api["session"]["requests"] == 2 and by_api["chat"]["requests"] == 1
    by_agent = api.get(f"/v1/usage?start={today}&end={today}&group_by=agent").json()["groups"]
    assert by_agent[agent["id"]]["requests"] == 2
    by_model = api.get(f"/v1/usage?start={today}&end={today}&group_by=model").json()["groups"]
    assert by_model["google_gemini-3_5-flash"]["requests"] == 3
    # another tenant sees nothing
    other, _ = new_tenant(gateway, "Other")
    assert other.get(f"/v1/usage?start={today}&end={today}").json()["totals"]["requests"] == 0
