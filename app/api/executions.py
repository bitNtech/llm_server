"""Asynchronous executions (agent runs and workflow runs)."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import orjson
from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import StreamingResponse

from app.api.deps import json_body, require, services
from app.api.responses import FastJSONResponse
from app.api.schemas import ExecutionCreate
from app.security.auth import Principal
from app.services.usage import summarize
from app.store.repos import strip_internal

router = APIRouter(prefix="/v1/executions", tags=["executions"])
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def execution_view(doc: dict[str, Any]) -> dict[str, Any]:
    view = {"object": "execution", **strip_internal(doc)}
    view.pop("workflow", None)
    view.pop("owner", None)
    return view


@router.post("", status_code=202, summary="Start an agent or workflow execution")
async def create_execution(request: Request,
                           principal: Principal = Depends(require("executions:write", rate_limited=True))) -> Response:
    svc = services(request)
    data, raw = await json_body(request)
    spec = ExecutionCreate.model_validate(data).model_dump(exclude_none=True)

    async def handler() -> Response:
        doc = await svc.executions.submit(principal.tenant, spec, principal.key_id)
        return FastJSONResponse(execution_view(doc), status_code=202)

    return await svc.idempotency.run(request, principal.tenant, raw, handler)


@router.get("", summary="List executions (newest first)")
async def list_executions(request: Request, limit: int = Query(50, ge=1, le=200), cursor: str | None = None,
                          principal: Principal = Depends(require("executions:read"))) -> dict:
    page = await services(request).repos.executions.list(principal.tenant, limit, cursor)
    return {"object": "list", "data": [execution_view(d) for d in page.items], "next_cursor": page.next_cursor}


@router.get("/{execution_id}", summary="Get an execution (status, step results, result, error)")
async def get_execution(execution_id: str, request: Request,
                        principal: Principal = Depends(require("executions:read"))) -> dict:
    return execution_view(await services(request).repos.executions.get(principal.tenant, execution_id))


@router.post("/{execution_id}/cancel", summary="Cancel a queued or running execution")
async def cancel_execution(execution_id: str, request: Request,
                           principal: Principal = Depends(require("executions:write"))) -> dict:
    return execution_view(await services(request).executions.cancel(principal.tenant, execution_id))


@router.get("/{execution_id}/events", summary="Progress as Server-Sent Events until the execution ends")
async def execution_events(execution_id: str, request: Request,
                           principal: Principal = Depends(require("executions:read"))) -> StreamingResponse:
    svc = services(request)
    await svc.repos.executions.get(principal.tenant, execution_id)  # 404 before streaming

    async def stream() -> AsyncIterator[bytes]:
        async for event in svc.executions.events(principal.tenant, execution_id):
            if event.get("type") == "keepalive":
                yield b": keepalive\n\n"
                continue
            yield b"event: " + event["type"].encode() + b"\ndata: " + orjson.dumps(event, default=str) + b"\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)


usage_router = APIRouter(prefix="/v1", tags=["usage"])


@usage_router.get("/usage", summary="Token and request usage by day, model, agent or API")
async def usage(request: Request, start: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
                end: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
                group_by: str = Query("day", pattern=r"^(day|model|agent|api)$"),
                principal: Principal = Depends(require("usage:read"))) -> dict:
    svc = services(request)
    await svc.usage.flush()  # include this instance's unflushed counters
    days = await svc.repos.usage.range(principal.tenant, start, end)
    return {"object": "usage", "tenant_id": principal.tenant, "start": start, "end": end, **summarize(days, group_by)}
