"""Stateful session API: conversation history, context management, responses, tool results."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import orjson
from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import StreamingResponse

from app.api.openai_routes import SSE_HEADERS
from app.api.responses import FastJSONResponse
from app.api.schemas import (
    CancelRequest,
    MessagesAppend,
    ResponseCreate,
    SessionCreate,
    SessionSpec,
    ToolResultsSubmit,
    TruncateRequest,
)
from app.api.security import require_auth, require_auth_rate_limited
from app.core.errors import GatewayError
from app.core.generation import Generation

router = APIRouter(prefix="/v1/sessions", tags=["sessions"], dependencies=[Depends(require_auth)])


def _services(request: Request) -> Any:
    return request.app.state.services


async def sse_events(gen: Generation) -> AsyncIterator[bytes]:
    try:
        async for event in gen.events():
            payload = orjson.dumps({"type": event.type, **event.data})
            yield b"event: " + event.type.encode() + b"\ndata: " + payload + b"\n\n"
    finally:
        if not gen.done:
            gen.cancel("client_disconnect")


async def collect(gen: Generation) -> dict[str, Any]:
    """Run a response to completion and fold its events into one JSON result."""
    done: dict[str, Any] = {}
    executed: dict[str, dict[str, Any]] = {}
    segments: list[str] = []
    try:
        async for event in gen.events():
            if event.type == "response.done":
                done = event.data
            elif event.type == "response.tool_call" and event.data["side"] == "server":
                executed[event.data["call_id"]] = {"name": event.data["name"],
                                                   "arguments": event.data["arguments"]}
            elif event.type == "response.tool_result":
                executed.setdefault(event.data["call_id"], {}).update(
                    output=event.data["output"], is_error=event.data["is_error"])
            elif event.type == "response.text.segment":
                segments.append(event.data["text"])
    finally:
        if not gen.done:
            gen.cancel("client_disconnect")
    if done.get("status") == "failed" and isinstance(gen.error, GatewayError):
        raise gen.error
    if executed:
        done["server_tool_calls"] = [{"call_id": k, **v} for k, v in executed.items()]
    if segments:
        done["segments"] = segments
    return done


async def respond(gen: Generation, stream: bool) -> Any:
    if stream:
        return StreamingResponse(sse_events(gen), media_type="text/event-stream", headers=SSE_HEADERS)
    return FastJSONResponse(await collect(gen))


@router.post("", status_code=201, summary="Create a session")
async def create_session(body: SessionCreate, request: Request) -> FastJSONResponse:
    services = _services(request)
    session = services.conversation.create_session(body.model_dump(exclude_none=True))
    for raw in body.messages or []:
        session.add_openai(raw, services.settings.max_input_chars)
    return FastJSONResponse(session.full_view(), status_code=201)


@router.get("", summary="List sessions (most recently active first)")
async def list_sessions(request: Request, limit: int = Query(50, ge=1, le=500),
                        offset: int = Query(0, ge=0)) -> dict:
    store = _services(request).store
    return {"data": [s.summary_view() for s in store.list(limit, offset)], "total": len(store)}


@router.get("/{session_id}", summary="Get a session")
async def get_session(session_id: str, request: Request, include_messages: bool = True) -> FastJSONResponse:
    session = _services(request).store.get(session_id)
    return FastJSONResponse(session.full_view(include_messages))


@router.patch("/{session_id}", summary="Update system prompt, tools, params or metadata")
async def update_session(session_id: str, body: SessionSpec, request: Request) -> FastJSONResponse:
    services = _services(request)
    session = services.store.get(session_id)
    services.conversation.update_session(session, body.model_dump(exclude_none=True))
    return FastJSONResponse(session.full_view(include_messages=False))


@router.delete("/{session_id}", status_code=204, summary="Delete a session (cancels its response)")
async def delete_session(session_id: str, request: Request) -> Response:
    await _services(request).store.delete(session_id)
    return Response(status_code=204)


@router.post("/{session_id}/messages", summary="Append messages without generating")
async def append_messages(session_id: str, body: MessagesAppend, request: Request) -> dict:
    services = _services(request)
    session = services.store.get(session_id)
    services.store.check_capacity(session, len(body.messages))
    async with session.lock:
        added = [session.add_openai(m, services.settings.max_input_chars).to_public() for m in body.messages]
    return {"added": added, "messages": len(session.messages)}


@router.delete("/{session_id}/messages", summary="Clear the conversation (keeps configuration)")
async def clear_messages(session_id: str, request: Request) -> dict:
    session = _services(request).store.get(session_id)
    if session.active:
        session.active.cancel("conversation_cleared")
    async with session.lock:
        session.messages.clear()
        session.summary = ""
        session.pending_tool_calls.clear()
    return {"session_id": session_id, "messages": 0}


@router.post("/{session_id}/responses", summary="Generate the next assistant turn (SSE when stream=true)",
             dependencies=[Depends(require_auth_rate_limited)])
async def create_response(session_id: str, body: ResponseCreate, request: Request) -> Any:
    services = _services(request)
    session = services.store.get(session_id)
    overrides = {"params": body.params, "tools": body.tools, "segmentation": body.segmentation}
    gen = services.conversation.start_response(session, body.input, overrides, body.response_id)
    return await respond(gen, body.stream)


@router.post("/{session_id}/cancel", summary="Cancel the running response (barge-in)")
async def cancel_response(session_id: str, body: CancelRequest, request: Request) -> dict:
    services = _services(request)
    session = services.store.get(session_id)
    cancelled = await services.conversation.cancel_response(
        session, body.response_id, body.spoken_text, discard=body.discard)
    return {"session_id": session_id, "cancelled": cancelled}


@router.post("/{session_id}/truncate", summary="Keep only the heard part of the last assistant reply")
async def truncate(session_id: str, body: TruncateRequest, request: Request) -> dict:
    services = _services(request)
    session = services.store.get(session_id)
    return {"session_id": session_id, "truncated": await services.conversation.truncate(session, body.spoken_text)}


@router.post("/{session_id}/tool_results", summary="Submit client tool results (and continue)",
             dependencies=[Depends(require_auth_rate_limited)])
async def tool_results(session_id: str, body: ToolResultsSubmit, request: Request) -> Any:
    services = _services(request)
    session = services.store.get(session_id)
    remaining = 0
    for result in body.results:
        remaining = await services.conversation.submit_tool_result(
            session, result.call_id, result.output, result.is_error)
    if not body.continue_response or remaining:
        return {"session_id": session_id, "pending_tool_calls": list(session.pending_tool_calls)}
    gen = services.conversation.start_response(session, None, {"params": body.params})
    return await respond(gen, body.stream)
