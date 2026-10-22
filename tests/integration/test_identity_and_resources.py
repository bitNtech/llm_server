"""Authentication, authorization, tenant isolation, resource CRUD, versioning and idempotency."""
from __future__ import annotations

import httpx

from tests.conftest import ADMIN_KEY, Api, new_tenant

AGENT = {"name": "Reception", "instructions": "#agent:priya Answer briefly.",
         "tools": [{"type": "builtin", "name": "get_current_time"}]}


def test_health_endpoints(gateway, admin):
    assert httpx.get(f"{gateway.url}/health/live").json() == {"status": "ok"}
    ready = httpx.get(f"{gateway.url}/health/ready").json()
    assert ready["status"] == "ready" and ready["model"] == "google/gemini-3.5-flash"
    detail = admin.get("/health").json()
    assert detail["provider"]["ready"] and detail["firestore"]["healthy"] and detail["firestore"]["emulator"]
    assert {loc["location"] for loc in detail["provider"]["locations"]} == {"asia-south1", "global"}
    metrics = httpx.get(f"{gateway.url}/metrics").text
    assert "llm_generations_total" in metrics and "vertex_ready 1.0" in metrics


def test_authentication_failures(gateway):
    assert httpx.get(f"{gateway.url}/v1/agents").status_code == 401
    r = httpx.get(f"{gateway.url}/v1/agents", headers={"Authorization": "Bearer aica_wrong"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    assert "request_id" in r.json()["error"] and r.headers["x-request-id"]
    with Api(gateway.url, ADMIN_KEY) as admin:
        r = admin.get("/v1/agents")  # admin keys must pick a tenant
        assert r.status_code == 400 and r.json()["error"]["code"] == "tenant_required"


def test_roles_and_scopes(gateway):
    owner, tenant = new_tenant(gateway)
    viewer_key = owner.post("/v1/api-keys", json={"name": "dash", "role": "viewer"}).json()["key"]
    runtime_key = owner.post("/v1/api-keys", json={"name": "voice", "role": "runtime"}).json()["key"]
    with Api(gateway.url, viewer_key) as viewer, Api(gateway.url, runtime_key) as runtime:
        assert viewer.get("/v1/agents").status_code == 200
        r = viewer.post("/v1/agents", json=AGENT)
        assert r.status_code == 403 and r.json()["error"]["code"] == "insufficient_scope"
        assert runtime.post("/v1/agents", json=AGENT).status_code == 403
        assert runtime.post("/v1/sessions", json={}).status_code == 201
        assert runtime.post("/v1/api-keys", json={"name": "x"}).status_code == 403
        me = runtime.get("/v1/me").json()
        assert me["role"] == "runtime" and me["tenant_id"] == tenant["id"] and "inference" in me["scopes"]
    # revocation is immediate on this instance
    key_id = next(k["id"] for k in owner.get("/v1/api-keys").json()["data"] if k["name"] == "voice")
    assert owner.delete(f"/v1/api-keys/{key_id}").json()["revoked_at"]
    assert httpx.get(f"{gateway.url}/v1/me", headers={"Authorization": f"Bearer {runtime_key}"}).status_code == 401


def test_tenant_isolation(gateway):
    a, _ = new_tenant(gateway, "A")
    b, _ = new_tenant(gateway, "B")
    agent = a.post("/v1/agents", json=AGENT).json()
    a.post("/v1/sessions", json={"agent_id": agent["id"], "session_id": "same-id"})
    assert b.get(f"/v1/agents/{agent['id']}").status_code == 404
    assert b.get("/v1/agents").json()["data"] == []
    assert b.get("/v1/sessions/same-id").status_code == 404
    # the same client-chosen session id can exist independently in another tenant
    assert b.post("/v1/sessions", json={"session_id": "same-id"}).status_code == 201
    r = b.post("/v1/sessions", json={"agent_id": agent["id"]})
    assert r.status_code == 404 and r.json()["error"]["code"] == "agent_not_found"
    # a tenant key cannot impersonate another tenant
    r = a.get("/v1/agents", headers={"X-Tenant-ID": b.tenant_id})
    assert r.status_code == 403


def test_suspended_tenant_is_locked_out(gateway, admin):
    client, tenant = new_tenant(gateway)
    assert admin.patch(f"/v1/tenants/{tenant['id']}", json={"status": "suspended"}).status_code == 200
    r = client.get("/v1/agents")
    assert r.status_code == 403 and r.json()["error"]["code"] == "tenant_inactive"


def test_agent_versions_and_etags(api):
    r = api.post("/v1/agents", json=AGENT)
    assert r.status_code == 201 and r.headers["etag"] == '"v1"'
    agent = r.json()
    r = api.patch(f"/v1/agents/{agent['id']}", json={"instructions": "v2 rules"}, headers={"If-Match": '"v1"'})
    assert r.status_code == 200 and r.json()["version"] == 2
    r = api.patch(f"/v1/agents/{agent['id']}", json={"instructions": "stale"}, headers={"If-Match": '"v1"'})
    assert r.status_code == 412 and r.json()["error"]["details"]["current_version"] == 2
    versions = api.get(f"/v1/agents/{agent['id']}/versions").json()["data"]
    assert [v["version"] for v in versions] == [1, 2]
    assert api.get(f"/v1/agents/{agent['id']}/versions/1").json()["instructions"] == AGENT["instructions"]
    assert api.delete(f"/v1/agents/{agent['id']}").status_code == 204
    assert api.get(f"/v1/agents/{agent['id']}").status_code == 404
    assert api.get(f"/v1/agents/{agent['id']}/versions/2").status_code == 200  # pinned sessions keep working


def test_agent_validation(api):
    r = api.post("/v1/agents", json={**AGENT, "tools": [{"type": "builtin", "name": "nope"}]})
    assert r.status_code == 400 and "available" in r.json()["error"]["details"]
    r = api.post("/v1/agents", json={**AGENT, "model": "gpt-4o"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "model_not_allowed"
    r = api.post("/v1/agents", json={**AGENT, "handoffs": [{"agent_id": "agent_missing"}]})
    assert r.status_code == 400
    r = api.post("/v1/agents", json={**AGENT, "params": {"max_tokens": 10**9}})
    assert r.status_code == 400
    r = api.post("/v1/agents", json={**AGENT, "unknown": 1})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error"
    client_tool = {"type": "client", "name": "transfer_call", "parameters": {"type": "object", "properties": {}}}
    r = api.post("/v1/agents", json={**AGENT, "tools": [client_tool, client_tool]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "duplicate_tool_name"


def test_tools_secrets_are_write_only(api, target, tool_target):
    r = api.post("/v1/tools", json={
        "name": "book_appointment", "description": "Book", "side_effects": True,
        "parameters": {"type": "object", "properties": {"day": {"type": "string"}}, "required": ["day"]},
        "http": {"url": f"{tool_target.url}/book", "headers": {"X-Static": "1"}},
        "secret_headers": {"Authorization": "Bearer crm-secret"}, "signing_secret": "s" * 32})
    assert r.status_code == 201, r.text
    tool = r.json()
    assert tool["secret_header_names"] == ["Authorization"] and tool["signing_secret_set"]
    assert "crm-secret" not in r.text and "secret_headers_enc" not in tool
    r = api.post(f"/v1/tools/{tool['id']}/invoke", json={"arguments": {"day": "monday"}})
    assert r.status_code == 200 and r.json()["data"]["booking_id"].startswith("B")
    rec = target.get("/records").json()[-1]
    assert rec["headers"]["authorization"] == "Bearer crm-secret" and rec["headers"]["x-static"] == "1"
    assert rec["headers"]["x-aica-signature"].startswith("t=") and rec["headers"]["idempotency-key"]
    r = api.post(f"/v1/tools/{tool['id']}/invoke", json={"arguments": {}})
    assert r.json()["is_error"] and "required" in r.json()["output"]
    assert api.post("/v1/tools", json={"name": "get_current_time", "description": "x",
                                       "http": {"url": f"{tool_target.url}/book"}}).status_code == 400


def test_idempotency_keys(api):
    headers = {"Idempotency-Key": "create-agent-1"}
    first = api.post("/v1/agents", json=AGENT, headers=headers)
    second = api.post("/v1/agents", json=AGENT, headers=headers)
    assert first.status_code == second.status_code == 201
    assert first.json()["id"] == second.json()["id"] and second.headers["idempotent-replayed"] == "true"
    assert len(api.get("/v1/agents").json()["data"]) == 1
    r = api.post("/v1/agents", json={**AGENT, "name": "Other"}, headers=headers)
    assert r.status_code == 409 and r.json()["error"]["code"] == "idempotency_key_reused"


def test_providers_and_models(api):
    providers = api.get("/v1/providers").json()["data"]
    assert providers[0]["id"] == "vertex" and providers[0]["context_window"] == 1_048_576
    agent = api.post("/v1/agents", json=AGENT).json()
    ids = [m["id"] for m in api.get("/v1/models").json()["data"]]
    assert "google/gemini-3.5-flash" in ids and f"agent:{agent['id']}" in ids
