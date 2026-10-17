# Architecture

## Goals

A self-hosted LLM endpoint for a real-time voice agent. The priorities, in order:

1. **Low and predictable time to first token (TTFT)**, because the caller hears silence until the first sentence is ready.
2. **Instant cancellation** when the caller barges in. The GPU must stop working on a reply nobody will hear.
3. **Correct conversation state** across interruptions, tool calls and long calls.
4. **Operational simplicity:** one repository and one `run.sh`, the same on a laptop and on EC2.

## Topology

```
                         ┌──────────────────────────── one host (laptop or EC2) ───────────────────────────┐
 voice pipeline  ──WS──► │  Gateway (FastAPI, :8000)                     vLLM OpenAI server (127.0.0.1:8001) │
 (ASR → LLM → TTS)       │  auth · rate limit · sessions · context  ──►  continuous batching · paged KV     │
 OpenAI SDK clients ─────►  tool loop · barge-in · admission       HTTP  prefix caching · chunked prefill   │
 Prometheus ─────────────►  metrics · health · request ids         SSE   tool/reasoning parsers · /metrics  │
                         └────────────────────────── supervised by run.sh (systemd in production) ─────────┘
```

**Decision: the vLLM OpenAI server runs as a sidecar process, and the gateway talks to it over loopback HTTP.**
The alternative was embedding `AsyncLLM` inside FastAPI. The sidecar approach was chosen because:

- It keeps everything vLLM already does well, natively and version-proof: chat templates, the tool-call parsers (`gemma4`, `qwen3_coder`, `hermes`, …), reasoning parsers, structured outputs, `/tokenize`, Prometheus metrics, request-id propagation and abort-on-disconnect. Re-implementing these against vLLM's internal Python API would break on most vLLM upgrades.
- It isolates failures. A CUDA error or an engine OOM kills the engine, not the process holding every caller's WebSocket. `run.sh` notices and exits non-zero, and systemd restarts the pair.
- It lets the gateway be developed and tested without a GPU (`ENGINE_MODE=mock`), point at a remote engine (`ENGINE_MODE=external`), or point at vLLM's official Docker image.
- The cost is one loopback HTTP hop, well under 1 ms per request. Streaming adds no per-token round trips because chunks are forwarded as they arrive.

The engine binds to `127.0.0.1` and requires `ENGINE_API_KEY`. `run.sh` generates a fresh random key on every start and passes it through the environment, never on argv, so it cannot be read from `ps`. The gateway is the only door.

## Gateway modules

| Module | Responsibility |
|---|---|
| `app/config.py` | Single typed source of truth (pydantic-settings). Used by the gateway **and** by `run.sh` (through `app/cli.py`), so they can never disagree. |
| `app/hardware.py`, `app/plan.py` | Detect CPU, RAM, glibc, NVIDIA driver, GPUs and compute capability. Choose the vLLM build (CUDA 13.0 / CUDA 12.9 / CPU wheel) and build the exact `vllm serve` argv and environment. |
| `app/engine/client.py` | Pooled keep-alive `httpx` client. Applies a first-token deadline and an inter-chunk idle deadline to streams. Closing a stream aborts the request inside vLLM. |
| `app/engine/monitor.py` | Background health and model discovery (`/health`, `/v1/models` → served name, `max_model_len`). Drives readiness. |
| `app/core/admission.py` | Bounded concurrency plus a bounded FIFO queue with a deadline. Fails fast with 429 rather than slowing every active call. |
| `app/core/generation.py` | A generation is a producer task plus an event queue. It can be cancelled from anywhere by one `task.cancel()`, and its `finally` always closes the engine stream. |
| `app/core/conversation.py` | Stateful turns: context fitting, streaming, the server/client tool loop, barge-in, speculative-turn discard, and statuses. |
| `app/core/openai_proxy.py` | Stateless OpenAI-compatible pass-through. Engine SSE frames are forwarded byte-for-byte. |
| `app/sessions/store.py` | In-memory sessions with TTL and LRU eviction. Sessions with a live WebSocket or an active response are never evicted. |
| `app/sessions/context.py` | Context-window management and the optional background summarization. |
| `app/core/segmenter.py` | Incremental sentence segmentation, so TTS can start on the first complete phrase. |
| `app/tools/` | Server-side tool registry (`@tool` decorator, timeouts, argument checks; sync tools run in threads). |
| `app/api/*` | REST, SSE and WebSocket surfaces, auth, middleware, health and metrics. |

## Request flows

### Stateless (OpenAI-compatible)

`POST /v1/chat/completions` → auth and rate limit → defaults and limits applied (max tokens, sampling defaults, `chat_template_kwargs`) → admission slot → engine stream → frames relayed. If the engine fails *before* the first frame, the client gets a real HTTP status (404, 503, …). Once streaming has started, a failure arrives as an OpenAI-style error frame followed by `[DONE]`.

### Stateful session turn (REST/SSE or WebSocket)

```
response.create ─► (cancel the running response if barge-in policy = cancel)
                ─► session lock ─► append input ─► context.prepare()  (estimate; exact /tokenize only near the limit)
                ─► admission slot ─► vLLM stream ─► text deltas, sentence segments, tool-call deltas
                ─► tool calls?  server-side: run them (parallel, with timeouts), append results, next round
                                client-side: status=requires_action → client sends tool.result → continue
                ─► append assistant message ─► response.done {status, text, usage, timings}
                ─► (summarize strategy) background compaction if near the window
```

## Latency design

- **Prefix caching:** history is append-only, and trimming happens in large steps (down to `CONTEXT_TRIM_TARGET_RATIO` of the budget). That keeps the system prompt and earlier turns identical across turns, so vLLM reuses their KV cache and only prefills the new utterance. `usage.cached_tokens` in every `response.done` shows the hit rate.
- **No tokenizer call on the hot path:** prompt size is estimated from a per-session chars-per-token ratio, calibrated from exact `usage.prompt_tokens` after each turn. `/tokenize` is called only when the estimate is above 85% of the budget.
- **Thinking is off by default** (`DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'`), because hidden reasoning is silent time on a phone call.
- **Sentence segments** (`response.text.segment`) let TTS start on the first complete phrase rather than waiting for the whole reply.
- **Server-side tools** skip the client round trip. `response.tool_call.started` is emitted as soon as the tool name is known, so the agent can play a filler ("one moment") while the arguments stream.
- **Admission control** keeps inter-token latency stable under load. Size `MAX_CONCURRENT_GENERATIONS` with `scripts/bench.py`.
- **Cancellation:** closing the HTTP stream to vLLM aborts the request at once and frees its KV blocks. Every cancel path (barge-in, REST cancel, client disconnect, timeouts, shutdown) goes through `Generation.cancel()`.

## Conversation integrity

- **Barge-in:** `response.cancel {spoken_text}` truncates the stored assistant reply to what the caller actually heard (marked `interrupted`), so the model does not believe it said things the caller never heard.
- **Speculative turns:** start a response on an interim transcript. If the final transcript differs, `response.cancel {discard: true}` removes the whole turn (input and partial reply).
- **Tool-call hygiene:** if the caller moves on while client tool calls are pending, each one gets a synthetic "no result" tool message, so the history stays valid for the chat template.
- **Template constraints:** Qwen-style templates require exactly one leading system message and at least one user message. The summary is therefore merged into the system message, system messages cannot be injected mid-conversation, and trimming always cuts at user-turn boundaries.

## Configuration flow

`.env` → `run.sh` parses it safely (it never sources or executes the file; real environment variables win) → `python -m app.cli` validates it with the same pydantic `Settings` class the gateway uses → hardware plan → `vllm serve` argv and env. Invalid configuration stops startup with a message naming the variable.

## Scaling

- **Vertical (one GPU box):** raise `MAX_CONCURRENT_GENERATIONS` and `ENGINE_MAX_NUM_SEQS` together, measured with `scripts/bench.py`. Use `ENGINE_KV_CACHE_DTYPE=fp8` on Ada or Hopper GPUs for roughly twice the KV capacity. Use `ENGINE_TENSOR_PARALLEL_SIZE=auto` on multi-GPU instances when the model does not fit on one GPU.
- **Horizontal:** sessions are in-process, so run one gateway+engine pair per instance and route each call to one instance for its lifetime. A WebSocket is naturally sticky; for REST sessions use ALB stickiness or route on the session id. The stateless `/v1/chat/completions` route can be load-balanced freely. A shared session store (Redis) can be added behind the `SessionStore` interface if calls must move between instances.
- **Multiple uvicorn workers are intentionally not used.** The engine is the bottleneck, and one asyncio worker handles thousands of streams.

## Security

- API keys are compared in constant time against every configured key. Logs carry only a SHA-256 fingerprint.
- Production refuses to start without keys (or an explicit `ALLOW_UNAUTHENTICATED=true` behind a trusted auth proxy), and wildcard CORS is rejected.
- Request bodies, WebSocket frames, message sizes, tool-output sizes, session counts and messages per session are all bounded. Session parameters are whitelisted.
- The engine is reachable only on loopback, with a per-start secret. `/docs` is off in production. `/metrics` can require auth.
- The model is never given server tools a session did not opt into.

## Observability

- Structured logs: JSON in production, text in development. Request, session and connection ids are attached to every record.
- `GET /metrics` exposes gateway Prometheus metrics: TTFT, inter-chunk latency, generation duration by status, admission wait and rejections, cancellations by reason, tokens (prompt, completion, cached), tool calls and latency, context trims, sessions, WebSockets, engine readiness.
- `GET /metrics/engine` proxies vLLM's own metrics: queue depth, KV-cache usage, prefix-cache hit rate, per-phase latencies.
- `X-Request-ID` is accepted or generated, returned to the client, and forwarded to vLLM as its request id.
- `GET /health/live`, `GET /health/ready` (engine healthy, model loaded, not draining) and `GET /health` (authenticated detail).
