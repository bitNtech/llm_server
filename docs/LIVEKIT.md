# Using the server from LiveKit Agents

LiveKit Agents owns the media side: WebRTC/SIP rooms, VAD, turn detection, STT and TTS. This server is
the LLM. There are two ways to connect them.

## Option A: OpenAI plugin (verified, no code changes)

LiveKit's `livekit-plugins-openai` speaks the Chat Completions API, which this server implements. Point it at the server:

```python
from livekit.agents import Agent, AgentSession, function_tool
from livekit.plugins import openai, deepgram, cartesia, silero

llm = openai.LLM(
    base_url="https://llm.example.com/v1",
    api_key=os.environ["AICA_API_KEY"],        # a tenant key with the `runtime` role
    model="agent:agent_5f2c...",                # or "default" / "gemini-3.5-flash" for the raw model
)

@function_tool
async def transfer_to_human(department: str) -> str:
    """Transfer the caller to a staff member."""
    ...   # LiveKit executes its own tools (SIP transfer, room actions, ...)

session = AgentSession(stt=deepgram.STT(), llm=llm, tts=cartesia.TTS(), vad=silero.VAD.load())
await session.start(room=ctx.room, agent=Agent(instructions="Caller is on a phone line.", tools=[transfer_to_human]))
```

What you get:

- **Streaming.** Tokens are relayed as Vertex produces them. The server adds 0.2 ms (median) to time to first token.
- **`model="agent:<id>"`.** The server prepends the agent's instructions; LiveKit's `instructions` arrive as an extra system message. The agent's **server-side tools** (built-ins, HTTP tools, MCP tools) run inside the server: LiveKit never sees them and only receives the final text. **LiveKit's own function tools** are passed through as normal tool calls, which LiveKit executes and sends back.
- **Gemini thought signatures.** Gemini 3.x rejects a replayed tool call without its signature (HTTP 400). LiveKit's OpenAI plugin does not keep the signature, so the server remembers the signatures it streamed (per tenant) and re-attaches them when LiveKit sends the conversation back.
- **Interruptions.** When LiveKit interrupts a turn it closes the HTTP stream, and the server stops generation at Vertex immediately.
- **Model routing.** Location failover, retries before the first token, circuit breakers and admission control apply to every request.

Verified: `scripts/livekit_compat.py` drives a running server through the real plugin (`livekit-agents` 1.8.8). It checks streaming text, a LiveKit-executed tool round trip with the conversation continued, and the `agent:` pseudo-model. Against Vertex `gemini-3.5-flash` (asia-south1) it passed with a TTFT of 444-616 ms:

```bash
pip install "livekit-agents[openai]"
python scripts/livekit_compat.py --base-url https://llm.example.com/v1 --api-key aica_... --agent-id agent_...
```

Trade-off: Chat Completions is stateless, so LiveKit owns the history. Server-side tool calls are not part of LiveKit's chat context, and agent handoffs inside the server are disabled on this route. Use LiveKit's own `Agent` handoffs, or option B.

## Option B: realtime WebSocket (server-managed conversation)

Use [`/v1/realtime`](REALTIME.md) from a custom LiveKit `llm.LLM` adapter, or from any runtime that feeds STT results in and TTS text out. The server then keeps the conversation, including server-side tool calls, model-driven handoffs between agents, barge-in truncation to what was heard, sentence segments for TTS, `end_conversation`, post-call workflows and persisted history. The session survives a worker restart: reconnect with `?session_id=<room or call id>`.

Map LiveKit events to protocol events like this:

| LiveKit | Realtime event |
|---|---|
| final user transcript | `response.create {input}` |
| `generate_reply(instructions=...)` | `response.create {instructions}` |
| `say(text)` | `response.create {say: text}` |
| user started speaking while the agent talks | `response.cancel {spoken_text}` |
| your function tool result | `tool.result {call_id, output}` |

Then speak `response.text.segment` and hang up on `session.end_requested`.

## Latency notes for voice

- Measured from India against `aica-voice-pilot`, `gemini-3.5-flash` in asia-south1 has a time to first token of **~420-580 ms p50** at concurrency 1-4. The server's own overhead is **~0.2 ms**. Over 100 ms of the total is inherent to Vertex: network round trip plus the model's prefill and first token. No hosted Gemini model in this project reached 100 ms.
- Keep `VERTEX_LOCATIONS=asia-south1,global` (Mumbai first) and run the server in asia-south1 next to Vertex.
- Speak `response.text.segment` as soon as it arrives instead of waiting for `response.done`.
- `VERTEX_REASONING_EFFORT=minimal` (the default) gives the fastest first token on user turns.
