"""A tiny stand-in for vLLM's OpenAI server (ENGINE_MODE=mock, tests, frontend work without a model).

It speaks the same wire protocol the gateway uses: /health, /v1/models, /tokenize, /metrics and
streaming /v1/chat/completions (content deltas, tool-call deltas, usage, [DONE]), and it counts
client disconnects so tests can prove that cancellation reaches the engine.

Deterministic behaviour, driven by the last message:
  * last message is a tool result         -> "Tool result: <output>"
  * user text contains `call <tool_name>` -> tool call to <tool_name> (if offered), JSON args
                                             taken from a `{...}` in the text, else {}
  * user text contains `[slow]`           -> 250 ms per token
  * user text contains `[error]`          -> error frame mid-stream
  * otherwise                             -> "Echo: <text>. This is a mock reply."
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import time
import uuid
from typing import Any

import orjson
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

MODEL = os.environ.get("MOCK_MODEL", "mock-model")
MAX_MODEL_LEN = int(os.environ.get("MOCK_MAX_MODEL_LEN", "4096"))
TOKEN_DELAY_S = float(os.environ.get("MOCK_TOKEN_DELAY_S", "0.01"))
TTFT_S = float(os.environ.get("MOCK_TTFT_S", "0.02"))

app = FastAPI(title="mock-vllm")
app.state.stats = {"requests": 0, "completed": 0, "aborted": 0, "active": 0, "last_body": None}


def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _prompt_tokens(body: dict[str, Any]) -> int:
    total = 0
    for m in body.get("messages", []):
        content = m.get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        total += count_tokens(str(content)) + 4
        for call in m.get("tool_calls") or []:
            total += count_tokens(orjson.dumps(call).decode())
    if body.get("tools"):
        total += count_tokens(orjson.dumps(body["tools"]).decode())
    return total


def _authorized(request: Request) -> bool:
    key = os.environ.get("VLLM_API_KEY")
    return not key or request.headers.get("authorization") == f"Bearer {key}"


def _plan(body: dict[str, Any]) -> tuple[list[str], dict[str, Any] | None, float, bool]:
    """Return (reply tokens, tool call, per-token delay, inject_error)."""
    messages = body.get("messages") or []
    last = messages[-1] if messages else {}
    text = str(last.get("content") or "")
    delay = 0.25 if "[slow]" in text else TOKEN_DELAY_S
    if last.get("role") == "tool":
        return [w + " " for w in f"Tool result: {text}".split()], None, delay, False
    offered = {t["function"]["name"] for t in body.get("tools") or [] if t.get("type") == "function"}
    match = re.search(r"call ([A-Za-z0-9_\-]+)", text)
    if match and match.group(1) in offered and body.get("tool_choice") != "none":
        args_match = re.search(r"\{.*\}", text)
        args = args_match.group(0) if args_match else "{}"
        return [], {"name": match.group(1), "arguments": args}, delay, False
    reply = f"Echo: {text.strip()}. This is a mock reply."
    return [w + " " for w in reply.split()], None, delay, "[error]" in text


@app.get("/health")
async def health() -> PlainTextResponse:
    return PlainTextResponse("")


@app.get("/v1/models")
async def models(request: Request) -> JSONResponse:
    if not _authorized(request):
        return JSONResponse({"error": {"message": "Unauthorized"}}, status_code=401)
    return JSONResponse({"object": "list", "data": [
        {"id": MODEL, "object": "model", "root": MODEL, "owned_by": "mock", "max_model_len": MAX_MODEL_LEN}]})


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    stats = app.state.stats
    return PlainTextResponse(
        f"vllm:num_requests_running {stats['active']}\nmock_aborted_total {stats['aborted']}\n")


@app.get("/mock/stats")
async def mock_stats() -> dict:
    return app.state.stats


@app.post("/tokenize")
async def tokenize(request: Request) -> JSONResponse:
    body = orjson.loads(await request.body())
    count = _prompt_tokens(body) if "messages" in body else count_tokens(body.get("prompt", ""))
    return JSONResponse({"count": count, "max_model_len": MAX_MODEL_LEN, "tokens": []})


@app.post("/v1/chat/completions")
async def chat(request: Request) -> Any:
    if not _authorized(request):
        return JSONResponse({"error": {"message": "Unauthorized"}}, status_code=401)
    body = orjson.loads(await request.body())
    stats = app.state.stats
    stats["requests"] += 1
    stats["last_body"] = body
    if body.get("model") != MODEL:
        return JSONResponse({"error": {"message": f"The model `{body.get('model')}` does not exist.",
                                       "type": "NotFoundError", "code": 404}}, status_code=404)
    if body.get("priority") and os.environ.get("MOCK_SCHEDULING", "fcfs") == "fcfs":
        return JSONResponse({"error": {"message": "priority requires --scheduling-policy priority"}},
                            status_code=400)
    tokens, tool_call, delay, inject_error = _plan(body)
    max_tokens = int(body.get("max_tokens") or 10_000)
    finish = "tool_calls" if tool_call else "stop"
    if len(tokens) > max_tokens:
        tokens, finish = tokens[:max_tokens], "length"
    prompt_tokens = _prompt_tokens(body)
    completion_tokens = len(tokens) + (count_tokens(tool_call["arguments"]) + 2 if tool_call else 0)
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
             "total_tokens": prompt_tokens + completion_tokens,
             "prompt_tokens_details": {"cached_tokens": 0}}
    rid = "chatcmpl-" + (request.headers.get("x-request-id") or uuid.uuid4().hex)
    created = int(time.time())

    if not body.get("stream"):
        await asyncio.sleep(TTFT_S + delay * len(tokens))
        message: dict[str, Any] = {"role": "assistant", "content": "".join(tokens).strip() or None}
        if tool_call:
            message["tool_calls"] = [{"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                                      "function": tool_call}]
        stats["completed"] += 1
        return JSONResponse({"id": rid, "object": "chat.completion", "created": created, "model": MODEL,
                             "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                             "usage": usage})

    def frame(delta: dict[str, Any], finish_reason: str | None = None) -> bytes:
        chunk = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
        return b"data: " + orjson.dumps(chunk) + b"\n\n"

    async def stream() -> Any:
        stats["active"] += 1
        finished = False
        try:
            await asyncio.sleep(TTFT_S)
            yield frame({"role": "assistant", "content": ""})
            for i, tok in enumerate(tokens):
                if inject_error and i == 2:
                    yield b'data: {"error": {"message": "mock engine failure", "code": 500}}\n\n'
                    finished = True
                    return
                yield frame({"content": tok})
                await asyncio.sleep(delay)
            if tool_call:
                call_id = "call_" + uuid.uuid4().hex[:12]
                yield frame({"tool_calls": [{"index": 0, "id": call_id, "type": "function",
                                             "function": {"name": tool_call["name"], "arguments": ""}}]})
                args = tool_call["arguments"]
                for i in range(0, len(args), 8):
                    yield frame({"tool_calls": [{"index": 0, "function": {"arguments": args[i:i + 8]}}]})
            yield frame({}, finish)
            if (body.get("stream_options") or {}).get("include_usage"):
                yield b"data: " + orjson.dumps({"id": rid, "object": "chat.completion.chunk", "created": created,
                                                "model": MODEL, "choices": [], "usage": usage}) + b"\n\n"
            yield b"data: [DONE]\n\n"
            finished = True
        finally:
            stats["active"] -= 1
            if finished:
                stats["completed"] += 1
            else:
                stats["aborted"] += 1  # client (the gateway) disconnected: vLLM would abort here

    return StreamingResponse(stream(), media_type="text/event-stream")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="mock vLLM OpenAI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
