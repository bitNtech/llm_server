# API reference

Base URL `http://<host>:8000`. When `API_KEYS` is set, send `Authorization: Bearer <key>` or `X-API-Key: <key>`. WebSocket clients may use `?api_key=<key>` instead. Every response carries `X-Request-ID`, and you may supply your own.

Errors always look like this:

```json
{"error": {"type": "invalid_request_error", "code": "context_length_exceeded", "message": "...", "request_id": "req_..."}}
```

| HTTP | `code` examples |
|---|---|
| 400 | `invalid_request`, `validation_error`, `engine_rejected_request`, `context_length_exceeded`, `unknown_tool_call` |
| 401 | `invalid_api_key` |
| 404 | `not_found`, `session_not_found`, `model_not_found`, `generation_not_found` |
| 409 | `conflict`, `response_in_progress` (with `WS_BARGE_IN=reject`) |
| 413 | `payload_too_large` |
| 429 | `overloaded`, `rate_limited` (+ `Retry-After`) |
| 502/503/504 | `engine_error`, `engine_unavailable`, `shutting_down`, `engine_timeout` |

## Health and metrics

| Route | Auth | Purpose |
|---|---|---|
| `GET /health/live` | no | process is up |
| `GET /health/ready` | no | 200 when the engine is healthy with the model loaded and the gateway is not draining, else 503 |
| `GET /health` | yes | detail: engine, admission queue, sessions, WebSockets, in-flight count |
| `GET /metrics` | optional | gateway Prometheus metrics |
| `GET /metrics/engine` | optional | vLLM Prometheus metrics (proxied) |

## OpenAI-compatible (stateless)

Point any OpenAI SDK at `http://host:8000/v1`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="<key or anything>")
stream = client.chat.completions.create(model="default", stream=True,
                                        messages=[{"role": "user", "content": "வணக்கம்!"}])
```

| Route | Notes |
|---|---|
| `POST /v1/chat/completions` | Full OpenAI body: `tools`, `tool_choice`, `response_format`, `stop`, … plus vLLM extras (`top_k`, `min_p`, `repetition_penalty`, `chat_template_kwargs`, …). `model` may be omitted or `"default"`. Gateway defaults fill unset sampling fields; `max_tokens` is capped by `MAX_TOKENS_LIMIT`. Streaming frames are vLLM's (content, `tool_calls` and `reasoning_content` deltas). |
| `GET /v1/models` | Served models with `max_model_len` |
| `POST /v1/tokenize` | `{"messages": [...]}` → `{"count", "max_model_len", "tokens"}` using the real chat template |
| `POST /v1/requests/{id}/cancel` | Cancel any in-flight generation by its `X-Request-ID` (or a session `response_id`) |

## Sessions (stateful REST / SSE)

```bash
curl -s localhost:8000/v1/sessions -H 'content-type: application/json' -d '{
  "session_id": "call-123",
  "system_prompt": "You are the front desk of City Hospital. Answer in one or two short sentences.",
  "server_tools": ["get_current_time"],
  "tools": [{"type": "function", "function": {"name": "transfer_call", "description": "Transfer to a human",
             "parameters": {"type": "object", "properties": {"department": {"type": "string"}}}}}],
  "params": {"temperature": 0.3, "max_tokens": 128},
  "segmentation": true
}'

curl -N localhost:8000/v1/sessions/call-123/responses -H 'content-type: application/json' \
     -d '{"input": "I want to see a cardiologist tomorrow"}'
```

| Route | Body | Result |
|---|---|---|
| `POST /v1/sessions` | `session_id?`, `system_prompt?`, `tools?`, `server_tools?` (names or `"all"`), `params?`, `metadata?`, `segmentation?`, `include_reasoning?`, `messages?` (seed history) | 201 + session |
| `GET /v1/sessions` | `?limit&offset` | summaries |
| `GET /v1/sessions/{id}` | `?include_messages=true` | session with `conversation`, `stats` (tokens, cached tokens, trims) |
| `PATCH /v1/sessions/{id}` | any field of the create body | updated session |
| `DELETE /v1/sessions/{id}` | | 204; cancels its running response |
| `POST /v1/sessions/{id}/messages` | `{"messages": [...]}` | append without generating (user/assistant/tool) |
| `DELETE /v1/sessions/{id}/messages` | | clear history and summary, keep config |
| `POST /v1/sessions/{id}/responses` | `input?` (string, message or list), `params?`, `tools?`, `segmentation?`, `stream` (default true), `response_id?` | SSE events (below) or one JSON result |
| `POST /v1/sessions/{id}/cancel` | `response_id?`, `spoken_text?`, `discard?` | `{"cancelled": bool}` |
| `POST /v1/sessions/{id}/truncate` | `spoken_text` | trim the last assistant reply to what was heard |
| `POST /v1/sessions/{id}/tool_results` | `results: [{call_id, output, is_error?}]`, `continue` (default true), `stream`, `params?` | continues the turn once nothing is pending |

Non-streaming result:

```json
{"response_id": "resp_…", "session_id": "call-123", "status": "completed", "finish_reason": "stop",
 "text": "…", "usage": {"prompt_tokens": 812, "completion_tokens": 21, "total_tokens": 833, "cached_tokens": 768},
 "timings": {"queue_ms": 0.1, "ttft_ms": 41.2, "total_ms": 220.5},
 "segments": ["…", "…"], "server_tool_calls": [{"call_id": "…", "name": "get_current_time", "arguments": "{}", "output": "…", "is_error": false}],
 "tool_calls": [{"call_id": "…", "name": "transfer_call", "arguments": "{\"department\": \"cardiology\"}"}]}
```

`status` is one of `completed`, `incomplete` (max tokens or max tool rounds), `requires_action` (client tools pending: answer them with `tool_results`), `cancelled` or `failed` (with `error`).

## Realtime WebSocket: `WS /v1/realtime[?session_id=…]`

One connection drives one session. Without `session_id` a new session is created. With it, an existing one is resumed: sessions survive disconnects until `SESSION_TTL_S`. Frames are JSON text, and every frame has a `type`.

### Client → server

| type | fields |
|---|---|
| `session.update` | `session: {system_prompt, tools, server_tools, params, metadata, segmentation, include_reasoning}` |
| `session.get` | (none) → `session.state` with the conversation |
| `response.create` | `input?`, `params?`, `tools?`, `segmentation?`, `response_id?` |
| `response.cancel` | `response_id?`, `spoken_text?`, `discard?` |
| `tool.result` | `call_id`, `output` (string or JSON), `is_error?`, `continue?` (default true), `params?` |
| `conversation.item.create` | `item: {role, content, …}` |
| `conversation.truncate` | `spoken_text` |
| `conversation.clear` | (none) |
| `ping` | `id?` |

Any client frame may carry `event_id`. Errors caused by that frame echo it.

### Server → client

| type | fields |
|---|---|
| `session.created` / `session.resumed` / `session.updated` | `session` |
| `response.created` | `response_id`, `session_id` |
| `response.text.delta` | `response_id`, `delta` |
| `response.text.segment` | `response_id`, `text`, `index` (complete sentence for TTS, when segmentation is on) |
| `response.reasoning.delta` | `response_id`, `delta` (only with `include_reasoning`) |
| `response.tool_call.started` | `response_id`, `call_id`, `name` (play a filler now) |
| `response.tool_call` | `response_id`, `call_id`, `name`, `arguments`, `side` (`server`/`client`) |
| `response.tool_result` | `response_id`, `call_id`, `name`, `output`, `is_error` (server tools) |
| `response.done` | same body as the non-streaming result |
| `conversation.item.created` / `conversation.truncated` / `conversation.cleared` | |
| `session.closing` | server is shutting down; reconnect to another instance |
| `pong` | `id` |
| `error` | `error: {type, code, message}`, `event_id?` |

Close codes: `1000` normal, `1001` server shutting down, `1009` frame too large, `1013` client too slow to read, `4404` unknown `session_id`, `4408` idle timeout. A failed authentication is rejected during the handshake (HTTP 403).

### Recommended voice-agent flow

```
caller audio → ASR ──final transcript──► response.create {input}
                     ◄── response.text.segment ──► TTS (start speaking on the first segment)
caller starts talking (VAD) ──► response.cancel {spoken_text: <text already played>}
                                (the reply is truncated to what was heard; the GPU stops at once)
interim transcript stable ──► response.create {input: interim}         (speculative, optional)
final differs ──► response.cancel {discard: true} ──► response.create {input: final}
response.tool_call.started ──► play "one moment…"
response.done {status: "requires_action"} ──► run the tool ──► tool.result {call_id, output}
```
