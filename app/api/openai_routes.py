"""OpenAI-compatible, stateless routes (point any OpenAI SDK at http://host:port/v1)."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import orjson
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.responses import FastJSONResponse
from app.api.security import require_auth, require_auth_rate_limited
from app.core.errors import BadRequest, GatewayError, NotFound
from app.core.generation import Generation

router = APIRouter(prefix="/v1", tags=["openai-compatible"])

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


async def json_body(request: Request) -> Any:
    raw = await request.body()
    if not raw:
        raise BadRequest("request body is required")
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError as exc:
        raise BadRequest(f"invalid JSON: {exc}") from None


async def relay(gen: Generation, first: Any) -> AsyncIterator[bytes]:
    try:
        yield first
        async for item in gen.events():
            if isinstance(item, GatewayError):
                yield b"data: " + orjson.dumps(item.to_dict(gen.id)) + b"\n\ndata: [DONE]\n\n"
                return
            yield item
    finally:
        if not gen.done:  # client went away mid-stream: abort the engine request
            gen.cancel("client_disconnect")


@router.get("/models", summary="Served models", dependencies=[Depends(require_auth)])
async def models(request: Request) -> FastJSONResponse:
    data = await request.app.state.services.engine.models()
    return FastJSONResponse({"object": "list", "data": data})


@router.post("/chat/completions", summary="Chat completions (OpenAI-compatible, SSE when stream=true)",
             dependencies=[Depends(require_auth_rate_limited)])
async def chat_completions(request: Request) -> Any:
    services = request.app.state.services
    request_id = request.state.request_id
    body = services.openai.prepare(await json_body(request))
    if not body.get("stream"):
        return FastJSONResponse(await services.openai.complete(body, request_id))

    gen = await services.openai.stream(body, request_id)
    events = gen.events()
    first = await anext(events, None)
    if isinstance(first, GatewayError):
        raise first  # failed before the first frame: answer with a proper status code
    if first is None:
        raise GatewayError("engine closed the stream without data")
    return StreamingResponse(relay(gen, first), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/tokenize", summary="Count/tokenize a prompt with the served model's template",
             dependencies=[Depends(require_auth)])
async def tokenize(request: Request) -> FastJSONResponse:
    services = request.app.state.services
    body = await json_body(request)
    if not isinstance(body, dict):
        raise BadRequest("request body must be a JSON object")
    body.setdefault("model", services.monitor.state.model or services.settings.served_model_name)
    return FastJSONResponse(await services.engine.tokenize(body, request.state.request_id))


@router.post("/requests/{request_id}/cancel", summary="Cancel any in-flight generation by id",
             dependencies=[Depends(require_auth)])
async def cancel_request(request_id: str, request: Request) -> dict:
    if not request.app.state.services.registry.cancel(request_id, "client_cancel"):
        raise NotFound(f"no in-flight generation {request_id}", code="generation_not_found")
    return {"id": request_id, "cancelled": True}
