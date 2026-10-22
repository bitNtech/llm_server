"""MCP server registrations and their discovered tools."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request, Response

from app.api.deps import etag, if_match, json_body, require, services
from app.api.responses import FastJSONResponse
from app.api.schemas import McpServerCreate, McpServerUpdate
from app.security.auth import Principal
from app.store.repos import strip_internal
from app.tools.base import ToolSpec

router = APIRouter(prefix="/v1/mcp/servers", tags=["mcp"])


def server_view(doc: dict[str, Any]) -> dict[str, Any]:
    view = {"object": "mcp_server", **strip_internal(doc)}
    view["secret_header_names"] = doc.get("secret_header_names", [])
    return view


def spec_view(spec: ToolSpec) -> dict[str, Any]:
    return {"name": spec.name, "remote_name": spec.target["remote_name"], "description": spec.description,
            "parameters": spec.parameters, "side_effects": spec.side_effects}


async def _normalize(request: Request, spec: dict[str, Any]) -> dict[str, Any]:
    svc = services(request)
    out = dict(spec)
    if "url" in out:
        await svc.guard.check(out["url"], "MCP server URL")
    secret_headers = out.pop("secret_headers", None)
    if secret_headers is not None:
        out["secret_headers_enc"] = svc.box.encrypt(secret_headers) if secret_headers else None
        out["secret_header_names"] = sorted(secret_headers)
    return out


@router.post("", status_code=201, summary="Register an MCP server (Streamable HTTP or SSE)")
async def create_server(request: Request, verify: bool = Query(False, description="connect and list tools now"),
                        principal: Principal = Depends(require("mcp:write"))) -> Response:
    svc = services(request)
    data, raw = await json_body(request)
    spec = await _normalize(request, McpServerCreate.model_validate(data).model_dump(exclude_none=True))
    spec.setdefault("enabled", True)
    spec.setdefault("tool_prefix", "")

    async def handler() -> Response:
        doc = await svc.repos.mcp_servers.create(principal.tenant, spec)
        body = server_view(doc)
        if verify:
            body["tools"] = [spec_view(s) for s in await svc.mcp.tool_specs(principal.tenant, doc, force=True)]
        return FastJSONResponse(body, status_code=201, headers=etag(doc))

    return await svc.idempotency.run(request, principal.tenant, raw, handler)


@router.get("", summary="List MCP servers")
async def list_servers(request: Request, limit: int = Query(50, ge=1, le=200), cursor: str | None = None,
                       principal: Principal = Depends(require("mcp:read"))) -> dict:
    page = await services(request).repos.mcp_servers.list(principal.tenant, limit, cursor)
    return {"object": "list", "data": [server_view(d) for d in page.items], "next_cursor": page.next_cursor}


@router.get("/{server_id}", summary="Get an MCP server")
async def get_server(server_id: str, request: Request,
                     principal: Principal = Depends(require("mcp:read"))) -> FastJSONResponse:
    doc = await services(request).repos.mcp_servers.get(principal.tenant, server_id)
    return FastJSONResponse(server_view(doc), headers=etag(doc))


@router.patch("/{server_id}", summary="Update an MCP server (reconnects; supports If-Match)")
async def update_server(server_id: str, body: McpServerUpdate, request: Request,
                        principal: Principal = Depends(require("mcp:write"))) -> FastJSONResponse:
    svc = services(request)
    changes = await _normalize(request, body.model_dump(exclude_unset=True))
    doc = await svc.repos.mcp_servers.update(principal.tenant, server_id, changes, if_match(request))
    await svc.mcp.drop(principal.tenant, server_id)
    svc.resolver.invalidate(principal.tenant, "mcp", server_id)
    return FastJSONResponse(server_view(doc), headers=etag(doc))


@router.delete("/{server_id}", status_code=204, summary="Remove an MCP server")
async def delete_server(server_id: str, request: Request,
                        principal: Principal = Depends(require("mcp:write"))) -> Response:
    svc = services(request)
    await svc.repos.mcp_servers.delete(principal.tenant, server_id, if_match(request))
    await svc.mcp.drop(principal.tenant, server_id)
    svc.resolver.invalidate(principal.tenant, "mcp", server_id)
    return Response(status_code=204)


@router.get("/{server_id}/tools", summary="Tools the server exposes (cached listing)")
async def server_tools(server_id: str, request: Request, principal: Principal = Depends(require("mcp:read"))) -> dict:
    svc = services(request)
    doc = await svc.repos.mcp_servers.get(principal.tenant, server_id)
    specs = await svc.mcp.tool_specs(principal.tenant, doc)
    return {"object": "list", "data": [spec_view(s) for s in specs]}


@router.post("/{server_id}/refresh", summary="Reconnect and re-list the server's tools")
async def refresh_server(server_id: str, request: Request,
                         principal: Principal = Depends(require("mcp:write"))) -> dict:
    svc = services(request)
    doc = await svc.repos.mcp_servers.get(principal.tenant, server_id)
    await svc.mcp.drop(principal.tenant, server_id)
    specs = await svc.mcp.tool_specs(principal.tenant, doc, force=True)
    return {"object": "list", "data": [spec_view(s) for s in specs]}
