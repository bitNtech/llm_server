"""A stand-in for Vertex AI's OpenAI-compatible endpoint, faithful to what the live API does
(verified against aica-voice-pilot): `[{"error": {...}}]` error bodies, bearer auth, per-location
model availability, tool calls streamed whole with `extra_content.google.thought_signature` on the
first call, usage in the final chunk, and HTTP 400 when a replayed tool call lacks its signature.

Deterministic replies, driven by the last message:
  * last message is a tool result         -> "Tool result: <output>"
  * user text contains `call a,b`         -> tool calls to a and b (if offered), JSON args from `{...}`
  * user text contains `[slow]`           -> 150 ms per token        `[stall]` -> stops after one token
  * user text contains `[error]`          -> error frame mid-stream  `[json]`  -> JSON object reply
  * user text contains `[callid:X]`       -> tool calls use the id X (Vertex ids are not guaranteed unique)
  * otherwise                             -> "Echo: <text>. This is a mock reply."
A system message containing `#agent:<name>` prefixes replies with "<name>: " (to observe handoffs).
`POST /control` injects failures per location; `GET /stats` exposes counters and the last request.
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid
from typing import Any

import orjson
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

TOKEN = "test-token"
MODELS = {"google/gemini-3.5-flash", "google/gemini-3.1-flash-lite", "google/gemini-3.5-flash-lite"}
SIG_PLACEHOLDER = "skip_thought_signature_validator"

app = FastAPI(title="fake-vertex")
STATE: dict[str, Any] = {}


def reset() -> None:
    STATE.clear()
    STATE.update({"requests": 0, "by_location": {}, "aborted": 0, "completed": 0, "last_body": None,
                  "failures": {}, "unavailable": set(), "bodies": [], "token": TOKEN, "auth_failures": 0,
                  "empty": 0})


reset()


def error(status: int, message: str, gstatus: str = "INVALID_ARGUMENT") -> JSONResponse:
    return JSONResponse([{"error": {"code": status, "message": message, "status": gstatus}}], status_code=status)


@app.head("/")
@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"error": "not found"}, status_code=404)


@app.post("/control")
async def control(request: Request) -> dict:
    body = await request.json()
    if body.get("reset"):
        reset()
    if "fail" in body:  # {"fail": {"location": "asia-south1", "status": 429, "count": 2}}
        f = body["fail"]
        STATE["failures"][f["location"]] = [f["status"], f.get("count", 1)]
    if "unavailable" in body:  # {"unavailable": ["google/gemini-3.5-flash", "asia-south1"]}
        STATE["unavailable"].add(tuple(body["unavailable"]))
    if "token" in body:
        STATE["token"] = body["token"]
    if "empty" in body:  # next N answers are empty candidates (a real Vertex behaviour)
        STATE["empty"] = int(body["empty"])
    return {"ok": True}


@app.get("/stats")
async def stats() -> dict:
    return {k: (sorted(map(list, v)) if isinstance(v, set) else v) for k, v in STATE.items() if k != "bodies"}


@app.get("/bodies")
async def bodies() -> list:
    return STATE["bodies"][-20:]


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _prompt_tokens(body: dict[str, Any]) -> int:
    return max(1, len(orjson.dumps(body.get("messages", []))) // 4)


def _plan(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages") or []
    system = " ".join(_text(m.get("content")) for m in messages if m.get("role") == "system")
    agent = re.search(r"#agent:(\w+)", system)
    prefix = f"{agent.group(1)}: " if agent else ""
    last = messages[-1] if messages else {}
    offered = {t["function"]["name"]: t for t in body.get("tools") or []}
    if last.get("role") == "tool":
        return {"text": f"{prefix}Tool result: {_text(last.get('content'))}"}
    user = next((_text(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "")
    if "[json]" in user:
        return {"text": orjson.dumps({"ok": True, "interested": "yes" in user, "name": "Asha"}).decode()}
    match = re.search(r"call ([\w.,:-]+)", user)
    choice = body.get("tool_choice")
    if match and choice != "none":
        names = [n for n in match.group(1).split(",") if n in offered]
        if names:
            args_match = re.search(r"(\{.*\})", user)
            args = args_match.group(1) if args_match else "{}"
            return {"calls": [(n, args) for n in names], "text": prefix + "Let me check that. " if "[say]" in user
                    else None}
    return {"text": f"{prefix}Echo: {user.strip()}. This is a mock reply.", "slow": "[slow]" in user,
            "error": "[error]" in user, "stall": "[stall]" in user}


def _validate(body: dict[str, Any]) -> JSONResponse | None:
    messages = body.get("messages") or []
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            answered = set()
            for follow in messages[i + 1:]:
                if follow.get("role") != "tool":
                    break
                answered.add(follow.get("tool_call_id"))
            if any(c.get("id") not in answered for c in m["tool_calls"]):
                return error(400, "Please ensure that the number of function response parts is equal to the "
                                  "number of function call parts of the function call turn.")
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            first = m["tool_calls"][0]
            sig = ((first.get("extra_content") or {}).get("google") or {}).get("thought_signature")
            if not sig:
                return error(400, "Function call is missing a thought_signature in functionCall parts. This is "
                                  "required for tools to work correctly.")
        if m.get("role") not in ("system", "developer", "user", "assistant", "tool"):
            return error(400, f"invalid role {m.get('role')}")
    if not any(m.get("role") in ("user", "assistant", "tool") for m in body.get("messages") or []):
        return error(400, "Unable to submit request because Model input cannot be empty.")
    if "max_tokens" in body and "max_completion_tokens" in body:
        return error(400, "Cannot specify both 'max_tokens' and 'max_completion_tokens' in a request.")
    return None


def _chunk(cid: str, model: str, delta: dict[str, Any], finish: str | None = None,
           usage: dict[str, Any] | None = None) -> bytes:
    payload: dict[str, Any] = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                               "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        payload["usage"] = usage
    return b"data: " + orjson.dumps(payload) + b"\n\n"


@app.post("/v1/projects/{project}/locations/{location}/endpoints/openapi/chat/completions")
async def chat(project: str, location: str, request: Request):  # noqa: C901
    STATE["requests"] += 1
    STATE["by_location"][location] = STATE["by_location"].get(location, 0) + 1
    if request.headers.get("authorization") != f"Bearer {STATE['token']}":
        STATE["auth_failures"] += 1
        return error(401, "Request had invalid authentication credentials.", "UNAUTHENTICATED")
    body = await request.json()
    STATE["last_body"] = body
    STATE["bodies"].append({"location": location, "body": body})
    model = body.get("model")
    if model not in MODELS or (model, location) in STATE["unavailable"]:
        return error(404, f"Publisher model `projects/{project}/locations/{location}/publishers/google/models/"
                          f"{str(model).split('/')[-1]}` was not found", "NOT_FOUND")
    fail = STATE["failures"].get(location)
    if fail and fail[1] > 0:
        fail[1] -= 1
        return error(fail[0], "Resource exhausted. Please try again later.", "RESOURCE_EXHAUSTED")
    invalid = _validate(body)
    if invalid is not None:
        return invalid
    cid = uuid.uuid4().hex[:20]
    if STATE["empty"] > 0:
        STATE["empty"] -= 1
        empty = {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
                 "choices": [{"index": 0, "logprobs": None}], "usage": {"prompt_tokens": 5, "total_tokens": 5}}
        if not body.get("stream"):
            return JSONResponse(empty)
        empty["object"] = "chat.completion.chunk"
        return StreamingResponse(iter([b"data: " + orjson.dumps(empty) + b"\n\n", b"data: [DONE]\n\n"]),
                                 media_type="text/event-stream")
    plan = _plan(body)
    max_tokens = int(body.get("max_tokens") or 1000)
    prompt_tokens = _prompt_tokens(body)
    fixed_id = re.search(r"\[callid:(\w+)\]", _text((body.get("messages") or [{}])[-1].get("content")))
    calls = [{"id": fixed_id.group(1) if fixed_id else f"call_{uuid.uuid4().int % 10_000_000}", "type": "function",
              "index": i, "function": {"name": name, "arguments": args}}
             for i, (name, args) in enumerate(plan.get("calls") or [])]
    if calls:
        calls[0]["extra_content"] = {"google": {"thought_signature": f"sig-{cid}"}}
    words = (plan.get("text") or "").split(" ") if plan.get("text") else []
    finish = "tool_calls" if calls else "stop"
    if len(words) > max_tokens:
        words, finish = words[:max_tokens], "length"
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": max(1, len(words) + 4 * len(calls)),
             "total_tokens": prompt_tokens + len(words), "prompt_tokens_details": {"cached_tokens": 0},
             "extra_properties": {"google": {"traffic_type": "ON_DEMAND"}}}

    if not body.get("stream"):
        message: dict[str, Any] = {"role": "assistant", "content": " ".join(words) or None}
        if calls:
            message["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in calls]
        STATE["completed"] += 1
        return JSONResponse({"id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
                             "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                             "usage": usage})

    async def stream():
        finished = False
        try:
            await asyncio.sleep(0.01)
            for i, word in enumerate(words):
                if plan.get("error") and i == 2:
                    yield b"data: " + orjson.dumps({"error": {"message": "mock engine failure", "code": 500}}) + b"\n\n"
                    return
                if plan.get("stall") and i == 1:
                    await asyncio.sleep(60)
                yield _chunk(cid, model, {"role": "assistant", "content": (" " if i else "") + word})
                await asyncio.sleep(0.15 if plan.get("slow") else 0.002)
            for call in calls:
                yield _chunk(cid, model, {"role": "assistant", "tool_calls": [call]})
            yield _chunk(cid, model, {"role": "assistant", "content": ""}, finish, usage)
            yield b"data: [DONE]\n\n"
            finished = True
            STATE["completed"] += 1
        finally:
            if not finished:
                STATE["aborted"] += 1

    return StreamingResponse(stream(), media_type="text/event-stream")
