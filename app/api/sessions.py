"""Stateful sessions over REST/SSE: lifecycle, history, responses, barge-in, client tool results,
handoffs and the tool-call audit trail. The realtime WebSocket drives the same operations."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import orjson
from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import StreamingResponse

from app.api.deps import json_body, require, services
from app.api.executions import SSE_HEADERS
from app.api.responses import FastJSONResponse
from app.api.schemas import (
    CancelRequest,
    EndRequest,
    HandoffRequest,
    MessagesAppend,
    ResponseCreate,
    SessionCreate,
    SessionUpdate,
    ToolResultsSubmit,
    TruncateRequest,
)
from app.core.errors import BadRequest, Conflict, GatewayError, NotFound
from app.engine.agents import client_tool_spec, validate_params
from app.engine.generation import Generation
from app.engine.runtime import TurnOptions
from app.engine.sessions import LiveSession, Message, doc_view
from app.security.auth import Principal
from app.store.repos import strip_internal

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


# ----------------------------------------------------------------- helpers
async def session_fields(svc: Any, tenant_id: str, spec: dict[str, Any], creating: bool) -> dict[str, Any]:
    """Validate session configuration (shared by REST and the WebSocket `session.update`)."""
    out: dict[str, Any] = {}
    if creating:
        agent_id = spec.get("agent_id")
        if agent_id:
            agent = await svc.resolver.resolve(tenant_id, agent_id, spec.get("agent_version"))
            out.update(agent_id=agent.id, agent_version=agent.version, current_agent_id=agent.id,
                       current_agent_version=agent.version)
            if spec.get("model"):
                raise BadRequest("model is set by the agent; omit it or use an agent-less session")
        elif spec.get("model"):
            out["model"] = svc.provider.resolve_model(spec["model"])
    if spec.get("instructions") is not None:
        if not isinstance(spec["instructions"], str) or len(spec["instructions"]) > 200_000:
            raise BadRequest("instructions must be a string under 200k characters")
        out["instructions"] = spec["instructions"]
    if spec.get("tools") is not None:
        if not isinstance(spec["tools"], list):
            raise BadRequest("tools must be a list of OpenAI function tools")
        for tool in spec["tools"]:
            client_tool_spec(tool)
        out["tools"] = spec["tools"]
    if spec.get("params") is not None:
        out["params"] = validate_params(spec["params"], svc.settings.max_tokens_limit)
    if spec.get("metadata") is not None:
        if not isinstance(spec["metadata"], dict) or len(orjson.dumps(spec["metadata"])) > 16_384:
            raise BadRequest("metadata must be an object under 16 KiB")
        out["metadata"] = spec["metadata"]
    if spec.get("segmentation") is not None:
        out["segmentation"] = bool(spec["segmentation"])
    return out


def turn_options(spec: dict[str, Any], api: str) -> TurnOptions:
    return TurnOptions(input=spec.get("input"), params=spec.get("params"), tools=spec.get("tools"),
                       instructions=spec.get("instructions"), tool_choice=spec.get("tool_choice"),
                       segmentation=spec.get("segmentation"), say=spec.get("say"),
                       response_id=spec.get("response_id"), api=api)


async def sse_events(gen: Generation) -> AsyncIterator[bytes]:
    try:
        async for event in gen.events():
            payload = orjson.dumps({"type": event.type, **event.data}, default=str)
            yield b"event: " + event.type.encode() + b"\ndata: " + payload + b"\n\n"
    finally:
        if not gen.done:
            gen.cancel("client_disconnect")


async def collect(gen: Generation) -> dict[str, Any]:
    """Run a response to completion and fold its events into one JSON result."""
    done: dict[str, Any] = {}
    segments: list[str] = []
    handoffs: list[dict[str, Any]] = []
    try:
        async for event in gen.events():
            if event.type == "response.done":
                done = dict(event.data)
            elif event.type == "response.text.segment":
                segments.append(event.data["text"])
            elif event.type == "agent.handoff":
                handoffs.append(event.data)
    finally:
        if not gen.done:
            gen.cancel("client_disconnect")
    if done.get("status") == "failed" and isinstance(gen.error, GatewayError) and not done.get("text"):
        raise gen.error
    if segments:
        done["segments"] = segments
    if handoffs:
        done["handoffs"] = handoffs
    return {"object": "response", **done}


async def respond(gen: Generation, stream: bool) -> Response:
    if stream:
        return StreamingResponse(sse_events(gen), media_type="text/event-stream", headers=SSE_HEADERS)
    return FastJSONResponse(await collect(gen))


async def live(request: Request, principal: Principal, session_id: str) -> LiveSession:
    return await services(request).sessions.get(principal.tenant, session_id)


# ------------------------------------------------------------------ routes
@router.post("", status_code=201, summary="Create a session (optionally bound to an agent)")
async def create_session(request: Request, principal: Principal = Depends(require("sessions:write"))) -> Response:
    svc = services(request)
    data, raw = await json_body(request)
    spec = SessionCreate.model_validate(data).model_dump(exclude_none=True)
    fields = await session_fields(svc, principal.tenant, spec, creating=True)

    async def handler() -> Response:
        session = await svc.sessions.create(principal.tenant, {**fields, "channel": "api"}, spec.get("session_id"),
                                            spec.get("messages"), svc.settings.max_input_chars)
        return FastJSONResponse(session.view(include_messages=True), status_code=201)

    return await svc.idempotency.run(request, principal.tenant, raw, handler)


@router.get("", summary="List sessions (newest first)")
async def list_sessions(request: Request, limit: int = Query(50, ge=1, le=200), cursor: str | None = None,
                        principal: Principal = Depends(require("sessions:read"))) -> dict:
    page = await services(request).repos.sessions.list(principal.tenant, limit, cursor)
    return {"object": "list", "data": [doc_view(d) for d in page.items], "next_cursor": page.next_cursor}


@router.get("/{session_id}", summary="Get a session")
async def get_session(session_id: str, request: Request, include_messages: bool = False,
                      principal: Principal = Depends(require("sessions:read"))) -> dict:
    return await services(request).sessions.describe(principal.tenant, session_id, include_messages)


@router.patch("/{session_id}", summary="Update instructions, client tools, params, metadata or segmentation")
async def update_session(session_id: str, body: SessionUpdate, request: Request,
                         principal: Principal = Depends(require("sessions:write"))) -> dict:
    svc = services(request)
    session = await live(request, principal, session_id)
    fields = await session_fields(svc, principal.tenant, body.model_dump(exclude_unset=True), creating=False)
    for key, value in fields.items():
        setattr(session, key, value)
    svc.sessions.save_state(session)
    return session.view()


@router.delete("/{session_id}", status_code=204, summary="Delete a session and its history")
async def delete_session(session_id: str, request: Request,
                         principal: Principal = Depends(require("sessions:write"))) -> Response:
    await services(request).sessions.delete(principal.tenant, session_id)
    return Response(status_code=204)


@router.post("/{session_id}/end", summary="End a session (runs the agent's on_session_end workflow)")
async def end_session(session_id: str, body: EndRequest, request: Request,
                      principal: Principal = Depends(require("sessions:write"))) -> dict:
    svc = services(request)
    session = await live(request, principal, session_id)
    await svc.sessions.end(session, body.reason)
    return session.view()


@router.get("/{session_id}/messages", summary="Conversation history (paginated by seq)")
async def get_messages(session_id: str, request: Request, after_seq: int = Query(0, ge=0),
                       limit: int = Query(200, ge=1, le=1000),
                       principal: Principal = Depends(require("sessions:read"))) -> dict:
    svc = services(request)
    session = svc.sessions.cached(principal.tenant, session_id)
    if session is not None:
        msgs = [m for m in session.messages if m.seq > after_seq][:limit]
        data = [m.to_public() for m in msgs]
    else:
        if await svc.repos.sessions.find(principal.tenant, session_id) is None:
            raise NotFound(f"session {session_id} not found", code="session_not_found")
        rows = await svc.repos.sessions.load_messages(principal.tenant, session_id, after_seq, limit)
        data = [Message.from_doc(r).to_public() for r in rows]
    next_seq = data[-1]["seq"] if len(data) == limit else None
    return {"object": "list", "data": data, "next_after_seq": next_seq}


@router.post("/{session_id}/messages", summary="Append messages without generating")
async def append_messages(session_id: str, body: MessagesAppend, request: Request,
                          principal: Principal = Depends(require("sessions:write"))) -> dict:
    svc = services(request)
    session = await live(request, principal, session_id)
    if len(session.messages) + len(body.messages) > svc.settings.max_messages_per_session:
        raise BadRequest("session would exceed MAX_MESSAGES_PER_SESSION", code="session_full")
    async with session.lock:
        added = [session.add_openai(m, svc.settings.max_input_chars).to_public() for m in body.messages]
    svc.sessions.save_state(session)
    return {"object": "list", "data": added, "message_count": len(session.messages)}


@router.delete("/{session_id}/messages", summary="Clear the conversation (keeps configuration)")
async def clear_messages(session_id: str, request: Request,
                         principal: Principal = Depends(require("sessions:write"))) -> dict:
    svc = services(request)
    session = await live(request, principal, session_id)
    if session.active is not None:
        session.active.cancel("conversation_cleared")
    async with session.lock:
        session.clear()
    svc.sessions.save_state(session)
    return {"session_id": session_id, "message_count": 0}


@router.post("/{session_id}/responses", summary="Generate the next assistant turn (SSE when stream=true)")
async def create_response(session_id: str, request: Request,
                          principal: Principal = Depends(require("sessions:write", rate_limited=True))) -> Response:
    svc = services(request)
    data, raw = await json_body(request)
    body = ResponseCreate.model_validate(data)
    session = await live(request, principal, session_id)

    async def handler() -> Response:
        gen = svc.runtime.start_response(session, turn_options(body.model_dump(exclude_none=True), "session"))
        return await respond(gen, body.stream)

    if body.stream:
        return await handler()
    return await svc.idempotency.run(request, principal.tenant, raw, handler)


@router.get("/{session_id}/responses", summary="Past responses (status, usage, timings)")
async def list_responses(session_id: str, request: Request, limit: int = Query(50, ge=1, le=200),
                         cursor: str | None = None, principal: Principal = Depends(require("sessions:read"))) -> dict:
    svc = services(request)
    await svc.writer.flush(f"{principal.tenant}/{session_id}", timeout=5)  # read-your-writes on this instance
    page = await svc.repos.sessions.list_responses(principal.tenant, session_id, limit, cursor)
    return {"object": "list", "data": [{"object": "response", **strip_internal(d)} for d in page.items],
            "next_cursor": page.next_cursor}


@router.get("/{session_id}/responses/{response_id}", summary="One past response")
async def get_response(session_id: str, response_id: str, request: Request,
                       principal: Principal = Depends(require("sessions:read"))) -> dict:
    svc = services(request)
    await svc.writer.flush(f"{principal.tenant}/{session_id}", timeout=5)
    doc = await svc.repos.sessions.get_response(principal.tenant, session_id, response_id)
    return {"object": "response", **strip_internal(doc)}


@router.post("/{session_id}/cancel", summary="Cancel the running response (barge-in)")
async def cancel_response(session_id: str, body: CancelRequest, request: Request,
                          principal: Principal = Depends(require("sessions:write"))) -> dict:
    svc = services(request)
    session = await live(request, principal, session_id)
    cancelled = await svc.runtime.cancel_response(session, body.response_id, body.spoken_text, body.discard)
    return {"session_id": session_id, "cancelled": cancelled}


@router.post("/{session_id}/truncate", summary="Keep only the heard part of the last assistant reply")
async def truncate(session_id: str, body: TruncateRequest, request: Request,
                   principal: Principal = Depends(require("sessions:write"))) -> dict:
    svc = services(request)
    session = await live(request, principal, session_id)
    return {"session_id": session_id, "truncated": await svc.runtime.truncate(session, body.spoken_text)}


@router.post("/{session_id}/tool_results", summary="Submit client tool results (and continue the turn)")
async def tool_results(session_id: str, request: Request,
                       principal: Principal = Depends(require("sessions:write", rate_limited=True))) -> Response:
    svc = services(request)
    data, _ = await json_body(request)
    body = ToolResultsSubmit.model_validate(data)
    session = await live(request, principal, session_id)
    remaining = 0
    for result in body.results:
        remaining = await svc.runtime.submit_tool_result(session, result.call_id, result.output, result.is_error)
    if not body.continue_response or remaining:
        return FastJSONResponse({"session_id": session_id, "pending_tool_calls": list(session.pending_tool_calls)})
    gen = svc.runtime.start_response(session, TurnOptions(params=body.params, api="session"))
    return await respond(gen, body.stream)


@router.post("/{session_id}/handoff", summary="Switch the session's active agent")
async def handoff(session_id: str, body: HandoffRequest, request: Request,
                  principal: Principal = Depends(require("sessions:write"))) -> dict:
    svc = services(request)
    session = await live(request, principal, session_id)
    if session.active is not None and not session.active.done:
        if not body.cancel_active:
            raise Conflict("a response is running; retry with cancel_active=true", code="response_in_progress")
        await svc.runtime.cancel_response(session)
    async with session.lock:
        previous, target = await svc.runtime.handoff(session, body.agent_id, body.reason, "api")
    return {"session_id": session_id, "from_agent": previous.public(), "to_agent": target.public(),
            "reason": body.reason}


@router.get("/{session_id}/tool_calls", summary="Executed server tool calls (audit trail)")
async def list_tool_calls(session_id: str, request: Request, limit: int = Query(200, ge=1, le=1000),
                          principal: Principal = Depends(require("sessions:read"))) -> dict:
    svc = services(request)
    await svc.writer.flush(f"{principal.tenant}/{session_id}", timeout=5)
    docs = await svc.repos.tool_calls.list_scope(principal.tenant, session_id, limit)
    return {"object": "list", "data": docs}
