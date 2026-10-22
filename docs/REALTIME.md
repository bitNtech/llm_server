# Realtime WebSocket protocol — `aica.realtime.v1`

`WS /v1/realtime` is the voice-agent interface: one connection drives one conversation session. The voice
runtime (LiveKit agent worker, SIP/telephony bridge, browser) owns audio, ASR and TTS. This server owns the
conversation: context, streaming generation, tools, handoffs and interruptions.

## Connecting

```
wss://<host>/v1/realtime?agent_id=<agent>            new session bound to an agent
wss://<host>/v1/realtime?session_id=<id>              resume an existing session (404 -> close 4404)
wss://<host>/v1/realtime?session_id=<id>&create=true  resume, or create it with that id (e.g. the call id)
wss://<host>/v1/realtime                              new agent-less session (server default instructions)
```

**Authentication.** The first match wins:

1. `Authorization: Bearer <api key | realtime token>`. Server-side runtimes such as LiveKit workers send the API key in this header.
2. Subprotocol `bearer.<realtime token>`, for browsers, which cannot set headers: `new WebSocket(url, ["aica.realtime.v1", "bearer." + token])`.
3. `?token=<realtime token>`.
4. `?api_key=` only when `WS_QUERY_API_KEY=true`. Not recommended, because query strings end up in proxy logs.

Mint realtime tokens server-side with `POST /v1/realtime/tokens {"ttl_s": 600, "agent_id": "...", "session_id": "..."}` using a key that has the `inference` permission. A token pinned to a session or agent cannot be used for another (close `4403`). A failed authentication is rejected during the handshake (HTTP 403).

**Versioning.** Offer the subprotocol `aica.realtime.v1`; the server echoes it, and `session.created` carries `"protocol": "aica.realtime.v1"`. Within v1, changes are additive only (new event types or fields). Clients must ignore what they do not know. A breaking change would ship as `aica.realtime.v2`.

**Frames.** JSON text, one event per frame, every event has a `type`. Server events also carry:

| Field | Meaning |
|---|---|
| `event_id` | unique id of this event |
| `seq` | per-connection counter that increases by one with every event, so gaps and reordering are detectable |
| `client_event_id` | on `error` events: the `event_id` of the client frame that caused it |

## Client → server

| `type` | Fields | Effect |
|---|---|---|
| `response.create` | `input?` (string, message or list), `instructions?`, `params?`, `tools?`, `tool_choice?`, `segmentation?`, `say?`, `response_id?` | Start the next assistant turn. A running response is cancelled first (barge-in), or rejected with `409 response_in_progress` when `BARGE_IN=reject`. `say` speaks fixed text with no model call. With no input before any user turn, the model opens the conversation. |
| `response.cancel` | `response_id?`, `spoken_text?`, `discard?` | Cancel the running response. `spoken_text` truncates the stored reply to what the caller heard. `discard` removes the whole turn (speculative responses), also when that response has already finished (pass its `response_id`). Client tool calls the dropped turn had answered are pending again. Handled ahead of queued events, and it also cancels `response.create` events not started yet. |
| `tool.result` | `call_id`, `output` (string or JSON), `is_error?`, `continue?` (default true), `params?` | Answer a client tool call. When nothing else is pending (and `continue`), the turn continues as a new response. |
| `session.update` | `session: {instructions?, tools?, params?, metadata?, segmentation?}` | Update configuration. |
| `session.get` | — | Returns `session.state` with the full history. |
| `session.end` | `reason?` | End the session (runs the agent's `on_session_end` workflow), send `session.ended`, close 1000. |
| `conversation.item.create` | `item: {role: user\|assistant\|tool, content, tool_call_id?}` | Append a message without generating. |
| `conversation.truncate` | `spoken_text` | Same as `response.cancel {spoken_text}` for a reply that already finished. |
| `conversation.clear` | — | Clear the history (configuration stays). |
| `agent.handoff` | `agent_id`, `reason?` | Switch the active agent (cancels a running response). Refused (`agent_mismatch`) for a token pinned to another agent. |
| `ping` | `id?` | Liveness check at the application level → `pong`. |

Any client frame may carry `event_id`. Errors caused by that frame echo it as `client_event_id`.

## Server → client

| `type` | Fields |
|---|---|
| `session.created` / `session.resumed` | `protocol`, `session` (id, agent_id, current_agent_id, status, message_count, pending_tool_calls, stats, …), `agent` (`id`, `version`, `name`, `model`, `greeting`) |
| `session.updated` / `session.state` | `session` (`session.state` includes `messages`) |
| `session.ended` | `session_id`, `reason`, `ended_at` |
| `session.closing` | `reason: server_shutdown`. The server is draining: reconnect with `?session_id=` and another instance resumes the call. |
| `session.end_requested` | `session_id`, `response_id`, `reason`. The agent called `end_conversation`; hang up after the current audio finishes. |
| `response.created` | `response_id`, `session_id`, `agent_id` |
| `response.text.delta` | `response_id`, `delta` |
| `response.text.segment` | `response_id`, `text`, `index`. A complete sentence or phrase: feed it to TTS. Text spoken before a tool call is flushed as a segment while the tool runs. |
| `response.tool_call.started` | `response_id`, `call_id`, `name`, `kind`. The model chose a tool: play a filler ("one moment") now. |
| `response.tool_call` | `response_id`, `call_id`, `name`, `arguments` (JSON string), `kind` (`builtin`/`http`/`mcp`/`client`/`handoff`), `side` (`server`/`client`/`handoff`) |
| `response.tool_result` | `response_id`, `call_id`, `name`, `output`, `is_error`, `duration_ms`, `replayed` (server tools) |
| `agent.handoff` | `session_id`, `response_id`, `from_agent`, `to_agent`, `reason`, `source` (`model`/`client`/`api`) |
| `response.done` | `response_id`, `session_id`, `status`, `finish_reason`, `text`, `agent`, `model`, `usage`, `timings`, `tool_calls?` (pending client calls), `server_tool_calls?`, `end_conversation?`, `error?`, `cancel_reason?` |
| `tool.result.accepted` | `call_id`, `pending_tool_calls` |
| `conversation.item.created` / `conversation.truncated` / `conversation.cleared` | `item` / `truncated` / — |
| `pong` | `id` |
| `error` | `error: {type, code, message, details?}`, `client_event_id?` |

`status` in `response.done` is `completed`, `incomplete`, `requires_action`, `cancelled` or `failed`. `timings` holds `queue_ms`, `ttft_ms`, `upstream_ttft_ms` (Vertex), `gateway_overhead_ms` and `total_ms`.

**Close codes.**

| Code | Meaning |
|---|---|
| 1000 | normal / session ended |
| 1001 | server shutting down |
| 1009 | frame larger than `WS_MAX_MESSAGE_BYTES` |
| 1013 | client too slow to read (send queue full) |
| 4403 | token restricted to another session/agent |
| 4404 | unknown session |
| 4408 | idle (`WS_IDLE_TIMEOUT_S`) |
| 4409 | session active on another instance (retry shortly) |

## Voice-agent flow

```
connect ?agent_id=A ──────────────────────────────► session.created {agent: {greeting}}
response.create {say: greeting} ──────────────────► response.text.segment ─► TTS
ASR final transcript ─► response.create {input} ──► response.text.segment ─► TTS (speak the first segment at once)
                                                    response.tool_call.started ─► play "one moment…"
                                                    response.tool_result (server tools ran inside the server)
                                                    response.done {status: completed}
model needs the runtime (client tool) ◄──────────── response.done {status: requires_action, tool_calls}
run it (e.g. transfer the SIP call) ─► tool.result {call_id, output} ─► the turn continues
model transfers to another agent ◄───────────────── agent.handoff {to_agent} (switch the TTS voice if you like)
caller says goodbye ◄────────────────────────────── session.end_requested ─► hang up after playback ─► session.end
```

### Barge-in and speculative turns

- **Barge-in.** When VAD detects the caller speaking over the agent, send `response.cancel {spoken_text: <text already played>}` and then `response.create` with the new transcript. Generation stops at Vertex immediately, and the stored reply is cut to what the caller actually heard (marked `interrupted`), so the model never believes it said things the caller did not hear. Sending `response.create` alone also cancels the running response; use `response.cancel` first when you know what was heard.
- **Speculative responses.** Start `response.create` on a stable interim transcript to save ASR finalisation time. If the final transcript differs, send `response.cancel {discard: true}` (input and partial reply are removed from history) and then `response.create {input: final}`.
- **Tools under interruption.** Read-only tool calls are cancelled with the response. Tools with side effects run to completion and their result is recorded, so a later identical call is answered from the record instead of booking twice. History stays valid either way, because every tool call gets a result.

## Minimal Python client

```python
import json
from websockets.sync.client import connect

with connect("wss://llm.example.com/v1/realtime?agent_id=agent_123", subprotocols=["aica.realtime.v1"],
             additional_headers={"Authorization": "Bearer aica_..."}) as ws:
    print(json.loads(ws.recv())["type"])                     # session.created
    ws.send(json.dumps({"type": "response.create", "input": "I need a cardiologist tomorrow"}))
    while True:
        event = json.loads(ws.recv())
        if event["type"] == "response.text.segment":
            speak(event["text"])                               # your TTS
        elif event["type"] == "response.done":
            break
```
