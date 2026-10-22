# AICA LLM Server

The LLM inference and orchestration server for the AICA voice agent. **Vertex AI** (Gemini 3.5 Flash by default) does the inference and **Firestore** (`llm_server`) holds the state. Voice runtimes such as LiveKit stay in charge of audio, ASR, TTS, WebRTC and SIP.

```
LiveKit worker / SIP bridge ──WS /v1/realtime (aica.realtime.v1)──┐
OpenAI SDKs / LiveKit OpenAI plugin ──POST /v1/chat/completions───┼─► server ─► Vertex AI (asia-south1 → global, failover)
Back office ──REST: agents, tools, MCP, workflows, sessions, …────┘        └─► Firestore · MCP servers · HTTP tools · webhooks
```

## What it does

- **Conversation engine.** Streamed turns, sentence segments for TTS, multi-round tool calling (parallel server tools, client tools with `requires_action`), handoffs between agents, barge-in that keeps only what the caller heard, speculative-turn discard, and context-window management. History is persisted.
- **Agents (LiveKit-style).** Versioned instructions, model, parameters, tools and handoff targets. Sessions pin the version they started with.
- **Tools.**
  - Built-ins (`get_current_time`, `end_conversation`).
  - Tenant HTTP tools: encrypted credentials, HMAC signing, SSRF guard.
  - MCP servers (Streamable HTTP or SSE).
  - Client tools executed by the voice runtime.
  - Every tool gets JSON-Schema validation; tools with side effects run exactly once.
- **Sessions.** REST/SSE and a versioned realtime WebSocket. Firestore leases give each session one owner instance, and a session resumes on another instance after a failover.
- **Async executions and workflows.** Agent runs and declarative step pipelines (llm / agent / tool / condition / set). Each step is persisted, crashed runs recover, and completion triggers a signed webhook. Post-call workflows can start automatically when a session ends.
- **OpenAI-compatible `/v1/chat/completions`.** Direct model access or `agent:<id>` with server-side tools. Gemini thought signatures are repaired for SDK and LiveKit clients.
- **Platform.**
  - Tenants with role-scoped API keys, ephemeral realtime tokens, rate limits and idempotency keys.
  - ETag optimistic concurrency, usage tracking (day / model / agent / API).
  - Prometheus metrics, OpenTelemetry tracing, Cloud Logging JSON logs.
  - Readiness and liveness checks, graceful drain.

## Model choice and latency (measured)

Every Gemini model the project can use was benchmarked from India: streaming time to first token on a warm connection, with a voice-agent prompt. **`google/gemini-3.5-flash` in `asia-south1` with minimal thinking** was the fastest model with a 1,048,576-token context window (verified with an over-limit request).

| Measurement | p50 TTFT |
|---|---|
| Vertex direct, asia-south1 | 571-583 ms |
| Through this server over WebSocket (concurrency 1 / 4) | 541 / 469 ms |
| Server overhead (client TTFT − Vertex TTFT) | **0.2 ms** |

**Target not met: TTFT under 100 ms is not achievable with any hosted Gemini model available to this project.** The network round trip to the Vertex region is already 25-60 ms, and Vertex's own time to first token was at least 400 ms for every model tested. The server keeps its own contribution to about 0.2 ms; the rest is Vertex. Details are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#latency).

## Quick start (local)

```bash
gcloud auth application-default login     # Vertex access
cp .env.example .env                       # set GCP_PROJECT_ID and ADMIN_API_KEYS
./run.sh dev                               # Firestore emulator + server on http://127.0.0.1:8000 (real Vertex)

ADMIN=<your admin key>
curl -s localhost:8000/v1/tenants -H "Authorization: Bearer $ADMIN" -H 'content-type: application/json' \
     -d '{"name": "Dev"}'                  # → owner API key (aica_...)
KEY=aica_...
curl -s localhost:8000/v1/agents -H "Authorization: Bearer $KEY" -H 'content-type: application/json' -d '{
  "name": "Reception", "instructions": "You are Priya, receptionist of City Hospital. One short sentence.",
  "tools": [{"type": "builtin", "name": "get_current_time"}]}'
curl -N localhost:8000/v1/sessions -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
     -d '{"agent_id": "agent_...", "session_id": "call-1"}'
curl -N localhost:8000/v1/sessions/call-1/responses -H "Authorization: Bearer $KEY" \
     -H 'content-type: application/json' -d '{"input": "What time is it in Chennai?"}'
```

Open `http://127.0.0.1:8000/` for the **developer console**: one page that can call every HTTP route (example bodies, captured ids, SSE streaming, idempotency and `If-Match` headers, copy as curl) and drive the realtime WebSocket (conversation view, barge-in, client tool results, every client event).

| Command | What it does |
|---|---|
| `./run.sh` | install, validate configuration, run (production entry point: `python -m app`) |
| `./run.sh dev` | same, with a local Firestore emulator |
| `./run.sh check` | print the effective configuration and credential status |
| `./run.sh test` | 84 unit + integration tests: Firestore emulator, a fake Vertex that matches the live API, a real MCP server |
| `./run.sh test-live` | 7 live tests against real Vertex: streaming, parallel tools with thought signatures, handoff, OpenAI SDK, structured output, latency |
| `scripts/bench.py` | WebSocket TTFT benchmark against a running server |
| `scripts/livekit_compat.py` | drive a running server through the real LiveKit OpenAI plugin |

## Documentation

- [docs/API.md](docs/API.md): every HTTP route, auth model, errors, idempotency, tools, MCP, workflows, executions, webhooks, usage
- [docs/REALTIME.md](docs/REALTIME.md): realtime WebSocket protocol `aica.realtime.v1` (events, barge-in, voice flow)
- [docs/LIVEKIT.md](docs/LIVEKIT.md): using the server from LiveKit Agents (verified with `livekit-agents` 1.8.8)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): design, turn flow, persistence and leases, failover, latency, security
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) and [.env.example](.env.example): settings
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md): GCP setup, secrets, Cloud Run, VM, operations and troubleshooting

## Layout

```
app/config.py          settings          app/main.py        wiring, lifecycle, graceful drain
app/api/               routes + WS       app/engine/        runtime, sessions, context, agents, admission
app/providers/         Vertex client     app/tools/         builtin, HTTP, MCP, executor
app/store/             Firestore         app/services/      chat, executions, workflows, webhooks, usage
app/security/          auth, crypto, SSRF guard, rate limits
tests/                 unit, integration (emulator + fakes), live (real Vertex)
deploy/                Cloud Run service, systemd unit       scripts/  emulator, bench, LiveKit check
```
