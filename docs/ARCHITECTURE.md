# Architecture

## What the server is

An LLM inference and orchestration server for the AICA voice agent. **Vertex AI** (Gemini, through the
OpenAI-compatible endpoint) does the inference. **Firestore** (database `llm_server`) holds state. The server
itself is a stateless-by-design FastAPI process that holds live conversations in memory while calls are active.

```
 LiveKit worker / SIP bridge ──WS /v1/realtime──┐
 OpenAI SDKs, LiveKit plugin ──/v1/chat/completions──┤   ┌────────────── server instance ──────────────┐
 Back office / CRM ──REST /v1/agents, tools, …──────┼──►│ auth · tenants · rate limits · idempotency  │──HTTPS/2──► Vertex AI (asia-south1 → global)
 Webhook receivers ◄──signed callbacks──────────────┘   │ conversation runtime · tool loop · handoffs │──────────► MCP servers / HTTP tools
                                                         │ sessions (leases) · executions · usage      │──gRPC─────► Firestore `llm_server`
 Prometheus / OTLP collector ◄──metrics / traces────────└──────────────────────────────────────────────┘
```

ASR, TTS, WebRTC, SIP and telephony stay in the voice runtime. This server owns the conversation.

## Modules

| Path | Responsibility |
|---|---|
| `app/config.py` | All settings (environment / `.env`), validated at startup; production refuses to start without its secrets. |
| `app/gcp/credentials.py` | One Google credential for Vertex and Firestore. Access tokens are refreshed in the background before expiry, single-flight. |
| `app/providers/vertex.py` | Vertex client: model × location routing, retries and failover before the first token, per-location circuit breakers, empty-response retry, request normalisation, HTTP/2 pool, keep-warm, startup probe. |
| `app/providers/signatures.py` | Gemini thought-signature cache and repair for stateless clients. |
| `app/store/` | Firestore wrapper (timeouts, error mapping, metrics), tenant-scoped repositories, ordered write-behind. |
| `app/security/` | API keys, roles and scopes, realtime JWTs, Fernet secret encryption, SSRF guard, rate limiter. |
| `app/tools/` | Tool model and JSON-Schema validation, built-ins, HTTP tools, MCP manager, executor (idempotency, timeouts, audit). |
| `app/engine/` | Conversation runtime (turns, tool loop, handoffs, interruptions), live sessions and leases, context window, admission control, generation handles, sentence segmenter. |
| `app/services/` | OpenAI-compatible chat, executions, workflows, webhooks, usage, idempotency. |
| `app/api/` | HTTP routes, realtime WebSocket, middleware (request ids, body limits, metrics, spans). |

## A turn, end to end

```
response.create ─► cancel the running response (barge-in) ─► session lock
  ─► append input ─► resolve the active agent (pinned version, cached) ─► build the toolset
     (builtin / HTTP / MCP / client / transfer_to_*)
  ─► context window (estimated tokens, trim whole turns if over CONTEXT_MAX_TOKENS)
  ─► admission slot ─► Vertex stream (route chain, first-token deadline)
       text deltas ─► response.text.delta + sentence segments for TTS
       tool calls  ─► response.tool_call.started (filler cue) … response.tool_call
  ─► no tool calls: store the reply ─► response.done
     server tools: run them in parallel (JSON-Schema validated, timeouts, side effects exactly once),
                   inject the results in call order ─► next round
     handoff tool: switch the active agent ─► agent.handoff ─► the new agent answers in the next round
     client tools: register them as pending ─► response.done {requires_action}
                   ─► tool.result ─► the turn continues as a new response
  ─► usage counters, response record and session state (write-behind)
```

Thought signatures: every assistant message stores Gemini's `extra_content` and the model that produced it. When the history is replayed to a different model (after failover), its signatures are swapped for the documented placeholder, because signatures are specific to one model.

### Interruptions

Each response is a producer task with an event queue (`engine/generation.py`). Barge-in, REST cancel, client disconnect, timeout and shutdown all call `Generation.cancel()`.

- Cancelling the task closes the Vertex stream, which stops generation upstream at once.
- Read-only tool calls are cancelled. Tools with side effects are shielded: they finish and their result is recorded.
- The history is settled. The partial reply is kept (or truncated to `spoken_text`), and every tool call receives a result, because Gemini requires call/response pairs.
- `discard` removes the whole speculative turn.

## Persistence and concurrency control

**Firestore layout.** Tenant isolation is structural: every tenant-owned document lives under `tenants/{tenant}`, and repositories build paths only from the authenticated tenant.

```
tenants/{t}                       agents/{a}(+versions/{n}) · tools · mcp_servers · workflows
                                  sessions/{s}/messages/{seq} · sessions/{s}/responses/{r}
                                  tool_calls/{scope}__{call} · executions/{e} · idempotency/{h} · usage_daily/{day}
api_keys/{sha256(key)}            execution_queue/{e}
```

No composite indexes are needed: every query uses one field.

**What happens on the hot path.**

- During a call, reads come from memory.
- Messages, response records, session state and tool-call audit records are written **behind**, in order per session, and retried on transient errors. A voice turn therefore never waits on Firestore.
- Writes that must be durable before the next step are synchronous: session creation, configuration changes, idempotency records, and claims for tools with side effects.

**Session leases.** A live session is owned by one instance through a lease on its document (`SESSION_LEASE_S`, renewed while busy or recently used).

- Another instance asking for the session gets `409 session_busy_elsewhere` (WebSocket close 4409).
- When a session is idle past `SESSION_CACHE_TTL_S`, its owner flushes the pending writes, releases the lease and drops it from memory. Any instance can then load it from Firestore.
- After a crash, the lease simply expires. The writes acknowledged before the crash are already in Firestore.
- WebSockets are sticky by nature. For REST, use session affinity (Cloud Run `sessionAffinity`, or route on the session id).

**Other concurrency controls.**

| Mechanism | What it protects |
|---|---|
| Per-session `asyncio.Lock` | serialises turns within one session |
| Optimistic concurrency (`version`, `If-Match`) | concurrent edits of agents, tools, MCP servers and workflows |
| Firestore transactions | lease and claim operations |
| `create()` | idempotency records and side-effect claims (conflicts on duplicates) |

## Executions

`execution_queue/{id}` is the work queue. Workers claim items in a transaction (owner plus lease) and heartbeat while they run; a scanner on every instance picks up queued and expired items. Workflow step results are persisted as they complete, so a resumed run skips finished steps. Tool steps use the step id as the tool call id, so side effects are not repeated. On shutdown, unfinished executions are returned to the queue.

## Vertex routing and resilience

- **Routes.** For model M: M@asia-south1, M@global, then each fallback model at each location.
  - A **404** marks a route unavailable for 10 minutes.
  - A **429, 5xx, timeout or connection error** moves on to the next route within the first-token deadline (`VERTEX_FIRST_TOKEN_TIMEOUT_S`, `VERTEX_MAX_ATTEMPTS`).
  - A **401** refreshes the token and retries the same route once.
  - Consecutive failures open a location's **circuit** for `VERTEX_CIRCUIT_RESET_S`; open circuits are tried last.
- **Retries happen only before the first token.** The stream is read ahead to the first meaningful chunk, so failures before it can still fail over. After that nothing is retried, so a voice agent never repeats itself.
- **Empty candidates.** Measured on 3.5-flash with `minimal` thinking: up to 6 of 6 follow-ups after a tool result came back with no text, no tool call and no finish reason. The server (1) uses `VERTEX_TOOL_FOLLOWUP_REASONING_EFFORT=low` for rounds whose last message is a tool result, which measured equally fast and 0 empties in 12 calls, and (2) retries an empty candidate once on the same route.
- **Request normalisation.**
  - Never sends both `max_tokens` and `max_completion_tokens` (Vertex rejects that).
  - Gives Gemini a user turn when the agent speaks first (Vertex rejects input that is only a system message).
  - Strips gateway-internal fields.

## Latency

Measured from the development machine in India against `aica-voice-pilot`:

| Path | TTFT p50 |
|---|---|
| `gemini-3.5-flash` (minimal thinking), asia-south1, direct | 571-583 ms (best 501) |
| same, global endpoint | 1259 ms |
| through the server over WebSocket, concurrency 1 / 4 | 541 / 469 ms (`scripts/bench.py`) |
| server overhead (client TTFT − Vertex TTFT) | **0.2 ms** median |

A 100 ms TTFT is not reachable with any hosted Gemini model in this project: the TLS round trip to Mumbai alone is about 25-60 ms, and Vertex's own time to first token was ≥400 ms for every model tested.

What the server does to stay off the critical path:

- token refresh in the background
- one warm HTTP/2 connection per location (startup probe plus keep-warm)
- no Firestore or tokenizer calls during a turn
- cached agent, tool and auth lookups
- sentence segments, so TTS starts on the first phrase
- `response.tool_call.started`, so the runtime can play a filler
- server-side tools that skip a client round trip

## Security

- **API keys** are 256-bit random values, stored as SHA-256 hashes, compared in constant time and never logged (logs carry the key id). Roles map to scopes on every route.
- **Admin keys** come from the environment and act inside a tenant only with an explicit `X-Tenant-ID`.
- **Realtime tokens** are short-lived HS256 JWTs, valid only on the WebSocket, and can be pinned to a session or agent.
- **Tenant secrets** (tool auth headers, signing secrets, MCP credentials, webhook secrets) are encrypted with Fernet (`SECRETS_ENCRYPTION_KEY`, rotation by key list) and never returned.
- **Outbound calls** to tenant URLs (HTTP tools, MCP, webhooks) are checked against private, loopback, link-local and metadata addresses, and redirects are not followed. Production requires https.
- **Limits:** request bodies, WebSocket frames, message sizes, tool output size, messages per session, `max_tokens`, rate limits per key (tenant overridable) and admission control. `/docs` is off in production.
- **Production refuses to start** without admin keys, an encryption key and a realtime token secret.

## Observability

- **Logs** are JSON in production, in the Cloud Logging format (`severity`, `logging.googleapis.com/trace`), and carry request, session, tenant and connection ids plus trace ids.
- **Metrics** (`/metrics`): TTFT (client and Vertex, by model and location), inter-token latency, generations by status, tokens by model, failovers, circuit state, admission queue and rejections, cancellations by reason, tool calls by kind and status, MCP errors, Firestore latency and write backlog, executions, webhooks and WebSockets.
- **Traces:** OpenTelemetry over OTLP/HTTP when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Spans cover HTTP requests, WebSocket connections, responses, Vertex streams (with location, attempts and TTFT), tool executions and executions. `traceparent` is propagated to HTTP tools, MCP servers and webhooks.

## Known limitations

- Provider configuration is environment-only (by design: it contains credentials); `/v1/providers` is read-only.
- MCP servers must be remote (Streamable HTTP or SSE). Local stdio servers are not supported through the API.
- In the stateless `/v1/chat/completions` route, the caller owns the history, so server-side tool calls are not part of it and handoffs inside the server are disabled. Use sessions or the WebSocket for server-managed conversations.
- Exact token counts come from Vertex usage after each round. Before a round the size is estimated (calibrated per session), and that is enough for trimming decisions.
