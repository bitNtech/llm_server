"""LiveKit Agents compatibility check: drives a running server through the *real* LiveKit OpenAI
plugin (`livekit.plugins.openai.LLM`), exactly as a LiveKit voice agent worker would.

    pip install "livekit-agents[openai]"          # in the LiveKit worker's environment
    python scripts/livekit_compat.py --base-url http://127.0.0.1:8000/v1 --api-key aica_... [--agent-id agent_...]

Checks: streamed text; a LiveKit function tool called by the model, executed by LiveKit, and the
conversation continued (Gemini thought signatures are dropped by the plugin and repaired by the
server); and, with --agent-id, the `agent:<id>` pseudo-model (server-side tools + LiveKit tools).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from livekit.agents import llm
from livekit.plugins import openai


@llm.function_tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny and 31 degrees Celsius in {city}."


async def collect(stream: llm.LLMStream) -> tuple[str, list[llm.FunctionToolCall], float | None]:
    text, calls, ttft, start = "", [], None, time.perf_counter()
    async with stream:
        async for chunk in stream:
            if chunk.delta is None:
                continue
            if chunk.delta.content:
                ttft = ttft or time.perf_counter() - start
                text += chunk.delta.content
            calls.extend(chunk.delta.tool_calls or [])
    return text, calls, ttft


async def tool_round_trip(model: llm.LLM, label: str) -> None:
    ctx = llm.ChatContext()
    ctx.add_message(role="system", content="You are a voice assistant. Use tools when helpful. One short sentence.")
    ctx.add_message(role="user", content="What's the weather in Chennai right now?")
    text, calls, _ = await collect(model.chat(chat_ctx=ctx, tools=[get_weather]))
    assert calls, f"{label}: expected a get_weather call, got text {text!r}"
    for call in calls:  # what LiveKit's AgentSession does: record the call, run the tool, record the output
        ctx.items.append(llm.FunctionCall(call_id=call.call_id, name=call.name, arguments=call.arguments,
                                          extra=call.extra))
        output = await get_weather(**json.loads(call.arguments or "{}"))
        ctx.items.append(llm.FunctionCallOutput(call_id=call.call_id, name=call.name, output=output, is_error=False))
    text, calls, ttft = await collect(model.chat(chat_ctx=ctx, tools=[get_weather]))
    assert "31" in text, f"{label}: follow-up did not use the tool result: {text!r}"
    print(f"[ok] {label}: tool call -> LiveKit executed it -> follow-up {text!r} (ttft {ttft * 1000:.0f} ms)")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="default")
    parser.add_argument("--agent-id")
    args = parser.parse_args()

    model = openai.LLM(model=args.model, base_url=args.base_url, api_key=args.api_key)
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="Say hello to a caller in Tamil, in one short sentence.")
    text, _, ttft = await collect(model.chat(chat_ctx=ctx))
    assert text, "no streamed text"
    print(f"[ok] streaming text: {text!r} (ttft {ttft * 1000:.0f} ms)")
    await tool_round_trip(model, f"model={args.model}")
    if args.agent_id:
        await tool_round_trip(openai.LLM(model=f"agent:{args.agent_id}", base_url=args.base_url,
                                         api_key=args.api_key), f"model=agent:{args.agent_id}")
    await model.aclose()
    print("LiveKit compatibility: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
