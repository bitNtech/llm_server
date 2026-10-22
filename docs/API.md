# HTTP API reference

Base URL: `https://<host>` (local: `http://127.0.0.1:8000`). The live OpenAPI document is at `/docs` and
`/openapi.json` in development (`DOCS_ENABLED=true` to expose it in production). The realtime WebSocket is
documented in [REALTIME.md](REALTIME.md), and LiveKit integration in [LIVEKIT.md](LIVEKIT.md).

## Conventions

**Authentication.** Send `Authorization: Bearer <key>` (or `X-API-Key: <key>`).

| Credential | Created by | Can do |
|---|---|---|
| Platform admin key (`ADMIN_API_KEYS`) | the operator (environment) | manage tenants; act inside any tenant with `X-Tenant-ID: <tenant>` |
| Tenant API key `aica_…` | `POST /v1/tenants` (first owner key), `POST /v1/api-keys` | whatever its role allows, only inside its own tenant |
| Realtime token (JWT) | `POST /v1/realtime/tokens` | the realtime WebSocket only, optionally pinned to one session and/or agent |

**Roles:**

| Role | Permissions |
|---|---|
| `owner` | everything in the tenant, including API keys and the webhook secret |
| `developer` | everything except creating or revoking keys |
| `runtime` | read configuration; run sessions, realtime, chat completions and executions (for voice workers) |
| `viewer` | read-only, plus usage |

A missing permission returns `403 insufficient_scope`.

**Tenant isolation.** Every resource lives under its tenant. Ids from another tenant behave exactly like ids that do not exist (`404`).

**Errors.** Every error, whether in an HTTP body, an SSE frame or a WebSocket `error` event, uses one envelope:

```json
{"error": {"type": "invalid_request_error", "code": "agent_not_found", "message": "...", "request_id": "req_...", "details": {}}}
```

| HTTP | Typical `code`s |
|---|---|
| 400 | `invalid_request`, `validation_error`, `model_not_allowed`, `provider_rejected_request`, `context_length_exceeded`, `unknown_tool_call`, `invalid_cursor`, `forbidden_target` |
| 401 | `invalid_api_key`, `api_key_expired`, `invalid_token`, `token_expired` |
| 403 | `insufficient_scope`, `tenant_inactive`, `tenant_mismatch`, `admin_required` |
| 404 | `not_found`, `agent_not_found`, `session_not_found`, `model_not_found`, `execution_not_found`, … |
| 409 | `already_exists`, `session_exists`, `session_ended`, `session_busy_elsewhere`, `response_in_progress`, `idempotency_key_reused`, `idempotency_in_progress`, `execution_finished` |
| 412 | `version_mismatch` (stale `If-Match`) |
| 413 | `payload_too_large` |
| 429 | `rate_limited`, `overloaded`, `provider_overloaded` (with a `Retry-After` header) |
| 502 / 503 / 504 | `provider_error`, `provider_permission_denied`, `provider_empty_response`, `provider_unavailable`, `store_unavailable`, `shutting_down`, `provider_timeout` |

**Request ids.** Every response carries `X-Request-ID`. You may send your own (`[A-Za-z0-9._:-]{1,128}`), and it is also the id used to cancel an in-flight chat completion.

**Optimistic concurrency.** Agents, tools, MCP servers and workflows return `ETag: "v<version>"`. Send `If-Match: "v<version>"` on `PATCH`/`DELETE` to refuse lost updates (`412`).

**Idempotency.** Send `Idempotency-Key: <unique>` on any create/run `POST`. A retry with the same key and body returns the stored response with `Idempotent-Replayed: true`. The same key with a different body returns `409 idempotency_key_reused`. Keys are kept for 24 h per tenant; the stored response is encrypted with `SECRETS_ENCRYPTION_KEY` (it can contain a new API key). Streaming responses are not stored; use a client-chosen `response_id` for those.

**Pagination.** List endpoints take `limit` (1-200) and `cursor`, and return `{"object": "list", "data": [...], "next_cursor": "..."}`. Session history is paginated by message `seq` (`after_seq`).

## Health and observability

| Route | Auth | Purpose |
|---|---|---|
| `GET /health/live` | none | the process is serving |
| `GET /health/ready` | none | 200 when Vertex is usable (credentials, startup probe, at least one route), Firestore is reachable and the server is not draining; otherwise 503 with `reasons` |
| `GET /health` | any credential | detail: provider routes and circuit breakers, Firestore, MCP connections, admission queue, cached sessions, WebSockets, executions, write backlog |
| `GET /metrics` | optional (`METRICS_REQUIRE_AUTH`) | Prometheus metrics: TTFT (client and Vertex), inter-token latency, generations, tokens, failovers, circuit state, tool calls, MCP, Firestore latency and backlog, executions, webhooks, WebSockets |

## Identity

| Route | Permission | Body → result |
|---|---|---|
| `GET /v1/me` | any | `{kind, key_id, role, tenant_id, scopes, rate_limit_per_minute}` |
| `POST /v1/tenants` | admin | `{name, id?, settings?: {rate_limit_per_minute}}` → tenant + `api_key.key` (first owner key) + `webhook_secret`. **Both secrets are shown only here.** |
| `GET /v1/tenants`, `GET /v1/tenants/{id}` | admin (a tenant may read itself) | tenant (`webhook_secret_set`, never the secret) |
| `PATCH /v1/tenants/{id}` | admin | `{name?, status?: active\|suspended, settings?}`. Suspended tenants get `403 tenant_inactive`. |
| `DELETE /v1/tenants/{id}` | admin | marks the tenant deleted and revokes all its keys |
| `POST /v1/tenants/{id}/webhook-secret` | `keys:write` | rotates the signing secret, returns it once |
| `POST /v1/api-keys` | `keys:write` | `{name, role, expires_in_days?}` → metadata + `key` (shown once) |
| `GET /v1/api-keys`, `GET /v1/api-keys/{id}` | `keys:read` | metadata (`prefix`, `role`, `created_at`, `expires_at`, `revoked_at`) |
| `DELETE /v1/api-keys/{id}` | `keys:write` | revokes the key. This is immediate on the serving instance and takes up to `API_KEY_CACHE_TTL_S` elsewhere. |
| `POST /v1/realtime/tokens` | `inference` | `{ttl_s?, session_id?, agent_id?}` → `{token, expires_at}` for one browser/mobile/room client |

## Providers and models

Provider configuration comes from the environment, never from the API (it holds credentials), so these routes are read-only.

| Route | Result |
|---|---|
| `GET /v1/providers`, `GET /v1/providers/vertex` | default/fallback/allowed models, context window, thinking level, credential state, the startup probe, and per-location circuit state (`closed`/`open`/`half_open`) with recent errors, plus model routes currently known to be unavailable |
| `GET /v1/models` | OpenAI-style list: every allowed Vertex model (with `context_window`) and one `agent:<id>` pseudo-model per agent |

## Agents

An agent is versioned configuration, in the LiveKit Agents sense: instructions, model, sampling parameters, tools, handoff targets and hooks. Every `PATCH` creates a new immutable version. Sessions pin the version they started with, so editing an agent never changes a call in progress.

```json
POST /v1/agents
{
  "name": "Reception",
  "description": "Front desk of City Hospital",
  "instructions": "You are Priya, the voice receptionist of City Hospital, Chennai. One or two short sentences.",
  "model": "google/gemini-3.5-flash",
  "params": {"temperature": 0.4, "max_tokens": 200, "reasoning_effort": "minimal"},
  "tools": [
    {"type": "builtin", "name": "get_current_time"},
    {"type": "builtin", "name": "end_conversation"},
    {"type": "tool", "id": "tool_4b1e..."},
    {"type": "mcp", "server_id": "mcp_9c2a...", "tools": ["lookup_order"]},
    {"type": "client", "name": "transfer_to_human", "description": "Connect the caller to staff",
     "parameters": {"type": "object", "properties": {"department": {"type": "string"}}}}
  ],
  "handoffs": [{"agent_id": "agent_billing...", "description": "bills, invoices and payments"}],
  "greeting": "Vanakkam, City Hospital. How can I help?",
  "segmentation": true,
  "max_tool_rounds": 6,
  "on_session_end": {"workflow_id": "workflow_post_call..."},
  "metadata": {"team": "front-desk"}
}
```

| Field | Notes |
|---|---|
| `model` | any model in `VERTEX_MODEL` / `VERTEX_FALLBACK_MODELS` / `VERTEX_ALLOWED_MODELS` (`google/` prefix optional). The default is `VERTEX_MODEL`. |
| `params` | `temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `reasoning_effort` (`minimal`/`low`/`medium`/`high`), `response_format`, `tool_choice`, `parallel_tool_calls` |
| `tools[]` | `builtin` (by name, see `GET /v1/tools/builtin`); `tool` (a tenant HTTP or client tool, by id); `mcp` (all or the listed tools of an MCP server); `client` (inline function executed by the caller). References are validated on save, and model-facing names must be unique. |
| `handoffs[]` | each becomes a `transfer_to_<agent name>` tool (override with `name`). When the model calls it, the session's active agent switches and the new agent answers in the same response. |
| `greeting` | stored for the caller to speak with `response.create {"say": ...}` (no model call) |
| `on_session_end` | workflow started (as an execution) when the session ends, with `{session_id, agent_id, end_reason, metadata, transcript}` as input |

Routes: `POST /v1/agents` (201, `ETag`) · `GET /v1/agents` · `GET /v1/agents/{id}` · `PATCH /v1/agents/{id}` (`If-Match`) · `DELETE /v1/agents/{id}` (existing sessions keep their pinned version) · `GET /v1/agents/{id}/versions` · `GET /v1/agents/{id}/versions/{n}`.

## Tools

Tenant-defined tools are either `http` (the server calls your endpoint) or `client` (the caller executes them). Built-ins ship with the server. MCP tools come from MCP servers.

```json
POST /v1/tools
{
  "name": "book_appointment",
  "description": "Book a doctor's appointment for the caller",
  "parameters": {"type": "object", "properties": {"doctor": {"type": "string"}, "date": {"type": "string", "format": "date"}},
                 "required": ["doctor", "date"]},
  "kind": "http",
  "http": {"url": "https://crm.example.com/patients/{patient_id}/appointments", "method": "POST",
           "headers": {"X-Source": "voice"}, "body_mode": "arguments"},
  "secret_headers": {"Authorization": "Bearer crm-token"},
  "signing_secret": "a-long-random-string-for-hmac",
  "side_effects": true,
  "timeout_s": 5
}
```

How an HTTP tool call is executed:

1. The arguments are validated against `parameters` (JSON Schema). Invalid arguments go back to the model as an error result so it can correct itself; nothing is called.
2. `{name}` placeholders in the URL are filled from the arguments (URL-encoded). GET/DELETE send the remaining arguments as query parameters. POST/PUT/PATCH send JSON: the arguments (`body_mode: arguments`), or `{tool, call_id, session_id, agent_id, arguments}` (`envelope`).
3. Headers sent: your static headers, the decrypted `secret_headers`, `Idempotency-Key` (stable per tool call), `X-AICA-Tenant-Id`, `X-AICA-Session-Id`, `X-AICA-Agent-Id`, `X-AICA-Call-Id`, `X-AICA-Tool`, W3C `traceparent`, and `X-AICA-Signature: t=<unix>,v1=<hex HMAC-SHA256(signing_secret, "<t>.<body>")>` when a signing secret is set.
4. A 2xx response (JSON or text, at most `HTTP_TOOL_MAX_RESPONSE_BYTES`) becomes the tool result. Other statuses become an error result. Redirects are not followed.
5. Targets must be public `https` URLs. Private, loopback, link-local and metadata addresses are rejected unless `ALLOW_PRIVATE_NETWORK_TARGETS` or `ALLOWED_TARGET_HOSTS` allows them.

**Exactly-once side effects.** With `side_effects: true` the call is recorded in Firestore *before* it runs. A replay of the same tool call (a client retry, a resumed execution, a duplicate delivery) returns the stored result instead of running again. A tool with side effects also keeps running if the caller barges in, so its outcome is known.

Secrets (`secret_headers`, `signing_secret`) are encrypted at rest (`SECRETS_ENCRYPTION_KEY`) and never returned. Responses show `secret_header_names` and `signing_secret_set`.

Routes: `GET /v1/tools/builtin` · `POST /v1/tools` · `GET /v1/tools` · `GET /v1/tools/{id}` · `PATCH /v1/tools/{id}` · `DELETE /v1/tools/{id}` · `POST /v1/tools/{id}/invoke {arguments}` (run an HTTP tool once, for testing).

Built-in tools: `get_current_time {timezone?}` and `end_conversation {reason?}`. The second emits `session.end_requested` so the voice runtime can hang up after the goodbye.

## MCP servers

```json
POST /v1/mcp/servers?verify=true
{"name": "crm", "url": "https://mcp.example.com/mcp", "transport": "streamable_http",
 "secret_headers": {"Authorization": "Bearer ..."}, "tool_prefix": "crm_",
 "tool_allowlist": ["lookup_patient", "create_ticket"], "side_effect_tools": ["create_ticket"], "timeout_s": 15}
```

- `transport` is `streamable_http` (default) or `sse` (legacy).
- Tools are exposed to the model as `<tool_prefix><remote name>`. Tools listed in `side_effect_tools`, or annotated `destructiveHint` by the server, get exactly-once execution.
- The server keeps one live MCP session per tenant and server, caches the tool listing for `MCP_TOOLS_CACHE_TTL_S`, and reconnects with backoff.
- A broken MCP server never fails a call: its tools are left out of that turn and a warning is logged.
- `verify=true` connects and lists tools on creation.
- Local stdio servers cannot be registered through the API, because that would let a tenant start processes.

Routes: `POST /v1/mcp/servers` · `GET /v1/mcp/servers` · `GET/PATCH/DELETE /v1/mcp/servers/{id}` · `GET /v1/mcp/servers/{id}/tools` · `POST /v1/mcp/servers/{id}/refresh`.

## Sessions (stateful conversations)

A session holds the conversation: history, active agent, client tools, parameters and metadata. It is persisted in Firestore and lives on one server instance at a time. Voice runtimes normally drive it over the realtime WebSocket; these REST routes expose the same operations.

| Route | Body → result |
|---|---|
| `POST /v1/sessions` | `{session_id?, agent_id?, agent_version?, instructions?, model? (agent-less only), tools?, params?, metadata?, segmentation?, messages?}` → 201 session (`session_id` may be your call id) |
| `GET /v1/sessions` | summaries, newest first |
| `GET /v1/sessions/{id}?include_messages=` | session: status, agent, current agent, stats (turns, tokens, tool calls, handoffs, trims), pending client tool calls, active response |
| `PATCH /v1/sessions/{id}` | `{instructions?, tools?, params?, metadata?, segmentation?}` |
| `DELETE /v1/sessions/{id}` | deletes the session, its history and responses |
| `POST /v1/sessions/{id}/end` | `{reason?}` → ends the session (cancels a running response) and runs the agent's `on_session_end` workflow |
| `GET /v1/sessions/{id}/messages?after_seq=&limit=` | persisted history: `{seq, role, content, tool_calls?, tool_call_id?, name?, agent_id?, interrupted?, created_at}` |
| `POST /v1/sessions/{id}/messages` | `{messages: [...]}` append without generating (user/assistant/tool roles) |
| `DELETE /v1/sessions/{id}/messages` | clear the conversation, keep the configuration |
| `POST /v1/sessions/{id}/responses` | `{input?, params?, tools?, instructions?, tool_choice?, segmentation?, say?, stream (default true), response_id?}` → SSE events (see below) or one JSON result |
| `GET /v1/sessions/{id}/responses[/{rid}]` | past responses: status, finish reason, text, agent, model, location, usage, timings, error |
| `POST /v1/sessions/{id}/cancel` | `{response_id?, spoken_text?, discard?}` barge-in (see [REALTIME.md](REALTIME.md#barge-in-and-speculative-turns)) |
| `POST /v1/sessions/{id}/truncate` | `{spoken_text}` keep only the heard part of the last reply |
| `POST /v1/sessions/{id}/tool_results` | `{results: [{call_id, output, is_error?}], continue (default true), stream, params?}` submit client tool results; once nothing is pending the turn continues |
| `POST /v1/sessions/{id}/handoff` | `{agent_id, reason?, cancel_active?}` switch the active agent (history is kept) |
| `GET /v1/sessions/{id}/tool_calls` | audit trail of executed server tool calls (arguments, status, output, timing) |

`input` is a string (a user utterance), one OpenAI-format message, or a list of them. `instructions` adds per-response instructions (like LiveKit's `generate_reply(instructions=...)`). `say` speaks fixed text without calling the model. With no input and no prior user turn, the model opens the conversation.

**Response streaming (SSE).** `event: <type>` / `data: <json>` frames carry the same events as the WebSocket: `response.created`, `response.text.delta`, `response.text.segment`, `response.tool_call.started`, `response.tool_call`, `response.tool_result`, `agent.handoff`, `session.end_requested`, `response.done`. The non-streaming result is the `response.done` body plus `segments` and `handoffs`:

```json
{"object": "response", "response_id": "resp_...", "session_id": "call-123", "status": "completed",
 "finish_reason": "stop", "text": "Dr. Rao is free tomorrow at 10. Shall I book it?",
 "agent": {"id": "agent_...", "version": 3, "name": "Reception", "model": "google/gemini-3.5-flash"},
 "model": "google/gemini-3.5-flash",
 "usage": {"prompt_tokens": 812, "completion_tokens": 21, "total_tokens": 833, "cached_tokens": 512, "reasoning_tokens": 0},
 "timings": {"queue_ms": 0.0, "ttft_ms": 452.3, "upstream_ttft_ms": 452.1, "gateway_overhead_ms": 0.2, "total_ms": 690.4},
 "server_tool_calls": [{"call_id": "...", "name": "get_current_time", "arguments": "{}", "output": "...", "is_error": false}],
 "tool_calls": [{"call_id": "...", "name": "transfer_to_human", "arguments": "{\"department\":\"cardiology\"}"}]}
```

`status` is one of:

| Status | Meaning |
|---|---|
| `completed` | the turn finished normally |
| `incomplete` | `max_tokens` or `MAX_TOOL_ROUNDS` was reached |
| `requires_action` | client tools are pending: run them and submit `tool_results` |
| `cancelled` | the response was cancelled (`cancel_reason` says why) |
| `failed` | an error occurred (see `error`) |

## Workflows and executions

**Workflows** are declarative step lists run asynchronously. Step types:

| Type | Fields |
|---|---|
| `llm` | `prompt`, `instructions?`, `model?`, `response_format?`, `max_tokens?`, `temperature?` |
| `agent` | `agent_id`, `input` (one non-interactive agent turn with its server tools) |
| `tool` | `tool: {type: builtin\|tool\|mcp, …}`, `arguments` |
| `condition` | `if: {left, op, right}`, `then`, `else` |
| `set` | `value` |

- Templates are `{{ dotted.path }}` lookups into `input`, `steps.<id>.output` and `execution.id`. A field that is exactly one placeholder keeps the value's type. No expressions are evaluated.
- Condition `op`s: `eq ne gt gte lt lte contains in exists not_exists truthy falsy`.
- Steps run in order unless `next` (or `then`/`else`) names a step id or `end`. `on_error: continue` tolerates a failed step.
- A workflow's `output` template forms the result.
- Each step's result is persisted as it completes. A resumed run skips finished steps, and tool steps are idempotent per execution.

**Executions** run an agent turn or a workflow in the background.

| Route | Notes |
|---|---|
| `POST /v1/executions` | `{type: agent, agent_id, agent_version?, input, session_id?, params?, response_format?, callback_url?, metadata?}` or `{type: workflow, workflow_id, input, callback_url?, metadata?}` → 202 |
| `POST /v1/workflows/{id}/run` | `{input?, callback_url?, metadata?}` → 202 execution |
| `GET /v1/executions`, `GET /v1/executions/{id}` | `status` (`queued`/`running`/`completed`/`failed`/`cancelled`), `steps`, `result`, `error`, `attempts`, `callback` (delivery outcome) |
| `GET /v1/executions/{id}/events` | SSE: `execution.started`, `step.started`, `step.completed`, `step.failed`, `agent.response`, `execution.completed\|failed\|cancelled` |
| `POST /v1/executions/{id}/cancel` | cancels a queued or running execution (`409 execution_finished` afterwards) |

An agent execution with `session_id` continues that session. Without one, it uses a throwaway conversation. Client tools are not available to executions.

**Webhooks.** With `callback_url`, a signed `POST` follows completion:

- Headers: `X-AICA-Event: execution.<status>`, `X-AICA-Delivery: <id>` (stable across retries; de-duplicate on it), and `X-AICA-Signature: t=..,v1=..`. The signature is HMAC-SHA256 with the tenant webhook secret over `"<t>.<body>"`. Reject timestamps older than 5 minutes.
- Body: `{"event", "delivery_id", "data": {id, type, status, result, error, metadata, agent_id, workflow_id}}`.
- Network errors, 429 and 5xx are retried with exponential backoff (`WEBHOOK_MAX_ATTEMPTS`).

Verify a signature in Python:

```python
import hashlib, hmac, time
def verify(secret: str, body: bytes, header: str) -> bool:
    parts = dict(p.split("=", 1) for p in header.split(","))
    if abs(time.time() - int(parts["t"])) > 300:
        return False
    mac = hmac.new(secret.encode(), f"{parts['t']}.".encode() + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(mac, parts["v1"])
```

## OpenAI-compatible chat completions

`POST /v1/chat/completions` accepts the OpenAI request body (`messages`, `tools`, `tool_choice`, `response_format`, `stream`, `stream_options`, `temperature`, `max_tokens`/`max_completion_tokens`, `stop`, `seed`, `reasoning_effort`, …) and answers in OpenAI format, streaming with SSE when `stream: true`. It is stateless: the caller sends the whole conversation.

- `model`: `"default"`, `"gemini-3.5-flash"` / `"google/gemini-3.5-flash"` (any allowed model), or `"agent:<agent_id>[@version]"`.
- With `agent:<id>` the agent's instructions come first (your system messages are appended), its server-side tools run inside the server, and only the final text streams to you. Your own `tools` pass through as normal tool calls (`finish_reason: "tool_calls"`).
- Gemini thought signatures are kept for you. OpenAI SDKs and LiveKit drop the `extra_content` field, so the server re-attaches the signatures it streamed when the tool call comes back.
- If Vertex fails before the first frame, you get a real HTTP status. After streaming starts, failures arrive as an error frame followed by `[DONE]`. Cancelling with `POST /v1/requests/{X-Request-ID}/cancel` ends the stream with a `cancelled` error frame.

## Usage

`GET /v1/usage?start=YYYY-MM-DD&end=YYYY-MM-DD&group_by=day|model|agent|api` → `{totals, group_by, groups}`. The counters are `requests`, `prompt_tokens`, `completion_tokens`, `cached_tokens`, `reasoning_tokens` and `tool_calls`, per tenant and UTC day. `api` is `session`, `realtime`, `chat`, `execution` or `workflow`.
