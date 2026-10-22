"""Agents: create, list, read, update (each update is a new immutable version), delete, versions."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request, Response

from app.api.deps import etag, if_match, json_body, require, services
from app.api.responses import FastJSONResponse
from app.api.schemas import AgentCreate, AgentUpdate
from app.core.errors import BadRequest
from app.security.auth import Principal
from app.store.repos import strip_internal

router = APIRouter(prefix="/v1/agents", tags=["agents"])


def agent_view(doc: dict[str, Any]) -> dict[str, Any]:
    return {"object": "agent", **strip_internal(doc)}


async def _check_hooks(request: Request, tenant_id: str, spec: dict[str, Any]) -> None:
    hook = spec.get("on_session_end")
    if hook and await services(request).repos.workflows.find(tenant_id, hook["workflow_id"]) is None:
        raise BadRequest(f"workflow {hook['workflow_id']} not found", code="workflow_not_found")


@router.post("", status_code=201, summary="Create an agent")
async def create_agent(request: Request, principal: Principal = Depends(require("agents:write"))) -> Response:
    svc = services(request)
    data, raw = await json_body(request)
    spec = AgentCreate.model_validate(data).model_dump(exclude_none=True)
    spec.setdefault("segmentation", True)
    await svc.resolver.validate_spec(principal.tenant, spec)
    await _check_hooks(request, principal.tenant, spec)

    async def handler() -> Response:
        doc = await svc.repos.agents.create(principal.tenant, spec)
        return FastJSONResponse(agent_view(doc), status_code=201, headers=etag(doc))

    return await svc.idempotency.run(request, principal.tenant, raw, handler)


@router.get("", summary="List agents (newest first)")
async def list_agents(request: Request, limit: int = Query(50, ge=1, le=200), cursor: str | None = None,
                      principal: Principal = Depends(require("agents:read"))) -> dict:
    page = await services(request).repos.agents.list(principal.tenant, limit, cursor)
    return {"object": "list", "data": [agent_view(d) for d in page.items], "next_cursor": page.next_cursor}


@router.get("/{agent_id}", summary="Get an agent (latest version)")
async def get_agent(agent_id: str, request: Request,
                    principal: Principal = Depends(require("agents:read"))) -> FastJSONResponse:
    doc = await services(request).repos.agents.get(principal.tenant, agent_id)
    return FastJSONResponse(agent_view(doc), headers=etag(doc))


@router.patch("/{agent_id}", summary="Update an agent (creates a new version; supports If-Match)")
async def update_agent(agent_id: str, body: AgentUpdate, request: Request,
                       principal: Principal = Depends(require("agents:write"))) -> FastJSONResponse:
    svc = services(request)
    changes = body.model_dump(exclude_unset=True)
    for required in ("name", "instructions"):
        if required in changes and not changes[required]:
            raise BadRequest(f"{required} cannot be empty")
    current = await svc.repos.agents.get(principal.tenant, agent_id)
    merged = {**current, **changes}
    await svc.resolver.validate_spec(principal.tenant, merged, self_id=agent_id)
    await _check_hooks(request, principal.tenant, merged)
    doc = await svc.repos.agents.update(principal.tenant, agent_id, changes, if_match(request))
    svc.resolver.invalidate(principal.tenant, "agent", agent_id)
    return FastJSONResponse(agent_view(doc), headers=etag(doc))


@router.delete("/{agent_id}", status_code=204, summary="Delete an agent (running sessions keep their version)")
async def delete_agent(agent_id: str, request: Request,
                       principal: Principal = Depends(require("agents:write"))) -> Response:
    svc = services(request)
    await svc.repos.agents.delete(principal.tenant, agent_id, if_match(request))
    svc.resolver.invalidate(principal.tenant, "agent", agent_id)
    return Response(status_code=204)


@router.get("/{agent_id}/versions", summary="List an agent's versions (oldest first)")
async def list_versions(agent_id: str, request: Request, limit: int = Query(50, ge=1, le=200),
                        cursor: str | None = None, principal: Principal = Depends(require("agents:read"))) -> dict:
    page = await services(request).repos.agents.list_versions(principal.tenant, agent_id, limit, cursor)
    if not page.items and cursor is None:
        await services(request).repos.agents.get(principal.tenant, agent_id)
    return {"object": "list", "data": [agent_view(d) for d in page.items], "next_cursor": page.next_cursor}


@router.get("/{agent_id}/versions/{version}", summary="Get one agent version")
async def get_version(agent_id: str, version: int, request: Request,
                      principal: Principal = Depends(require("agents:read"))) -> dict:
    return agent_view(await services(request).repos.agents.get_version(principal.tenant, agent_id, version))
