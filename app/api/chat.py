"""OpenAI-compatible, stateless routes: point any OpenAI SDK (or LiveKit's OpenAI plugin) at /v1."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import orjson
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.deps import json_body, require, services
from app.api.executions import SSE_HEADERS
from app.api.responses import FastJSONResponse
from app.core.errors import GatewayError, NotFound
from app.engine.generation import Generation
from app.security.auth import Principal

router = APIRouter(prefix="/v1", tags=["openai-compatible"])


async def relay(gen: Generation, first: Any) -> AsyncIterator[bytes]:
    try:
        yield first
        async for item in gen.events():
            if isinstance(item, GatewayError):
                yield b"data: " + orjson.dumps(item.to_dict(gen.id)) + b"\n\ndata: [DONE]\n\n"
                return
            yield item
    finally:
        if not gen.done:  # client went away mid-stream: stop generation upstream
            gen.cancel("client_disconnect")


@router.post("/chat/completions", summary="Chat completions (OpenAI-compatible; SSE when stream=true)")
async def chat_completions(request: Request,
                           principal: Principal = Depends(require("inference", rate_limited=True))) -> Any:
    svc = services(request)
    request_id = request.state.request_id
    body, _ = await json_body(request)
    req = svc.chat.prepare(body)
    if not req.stream:
        return FastJSONResponse(await svc.chat.complete(req, principal.tenant, request_id))
    gen = svc.chat.stream(req, principal.tenant, request_id)
    events = gen.events()
    first = await anext(events, None)
    if isinstance(first, GatewayError):
        raise first  # failed before the first frame: answer with a real status code
    if first is None:
        raise GatewayError("the model closed the stream without data", code="provider_error", status=502)
    return StreamingResponse(relay(gen, first), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/requests/{request_id}/cancel", summary="Cancel an in-flight generation by request/response id")
async def cancel_request(request_id: str, request: Request,
                         principal: Principal = Depends(require("inference"))) -> dict:
    if not services(request).registry.cancel(principal.tenant, request_id, "client_cancel"):
        raise NotFound(f"no in-flight generation {request_id}", code="generation_not_found")
    return {"id": request_id, "cancelled": True}
