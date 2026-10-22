"""Tools: tenant-defined HTTP and client tools, built-in catalogue, test invocation, call audit."""
from __future__ import annotations

from typing import Any

import orjson
from fastapi import APIRouter, Depends, Query, Request, Response

from app.api.deps import etag, if_match, json_body, require, services
from app.api.responses import FastJSONResponse
from app.api.schemas import ToolCreate, ToolInvoke, ToolUpdate
from app.core.errors import BadRequest
from app.core.ids import new_id
from app.security.auth import Principal
from app.store.repos import strip_internal
from app.tools import builtin
from app.tools.base import EMPTY_PARAMS, ToolContext, ToolSpec, check_name, check_schema

router = APIRouter(prefix="/v1/tools", tags=["tools"])


def tool_view(doc: dict[str, Any]) -> dict[str, Any]:
    view = {"object": "tool", **strip_internal(doc)}
    view["secret_header_names"] = doc.get("secret_header_names", [])
    view["signing_secret_set"] = bool(doc.get("signing_secret_enc"))
    return view


async def _normalize(request: Request, spec: dict[str, Any], kind: str) -> dict[str, Any]:
    svc = services(request)
    out = dict(spec)
    if "name" in out:
        check_name(out["name"])
        if out["name"] in builtin.REGISTRY or out["name"].startswith("transfer_to_"):
            raise BadRequest(f"tool name {out['name']!r} is reserved")
    if "parameters" in out:
        out["parameters"] = check_schema(out["parameters"])
    if kind == "http":
        if "http" in out:
            await svc.guard.check(out["http"]["url"].replace("{", "").replace("}", ""), "tool URL")
    elif any(k in out for k in ("http", "secret_headers", "signing_secret")):
        raise BadRequest("client tools run in the caller; they take no http configuration")
    secret_headers = out.pop("secret_headers", None)
    if secret_headers is not None:
        out["secret_headers_enc"] = svc.box.encrypt(secret_headers) if secret_headers else None
        out["secret_header_names"] = sorted(secret_headers)
    signing = out.pop("signing_secret", None)
    if signing is not None:
        out["signing_secret_enc"] = svc.box.encrypt(signing)
    return out


@router.get("/builtin", summary="Built-in server tools agents can enable")
async def list_builtin(_: Principal = Depends(require("tools:read"))) -> dict:
    return {"object": "list", "data": builtin.describe()}


@router.post("", status_code=201, summary="Create an HTTP or client tool")
async def create_tool(request: Request, principal: Principal = Depends(require("tools:write"))) -> Response:
    svc = services(request)
    data, raw = await json_body(request)
    spec = ToolCreate.model_validate(data)
    if spec.kind == "http" and spec.http is None:
        raise BadRequest("http tools need an http configuration")
    doc_spec = await _normalize(request, spec.model_dump(exclude_none=True), spec.kind)
    doc_spec.setdefault("parameters", dict(EMPTY_PARAMS))
    doc_spec.setdefault("side_effects", False)

    async def handler() -> Response:
        doc = await svc.repos.tools.create(principal.tenant, doc_spec)
        return FastJSONResponse(tool_view(doc), status_code=201, headers=etag(doc))

    return await svc.idempotency.run(request, principal.tenant, raw, handler)


@router.get("", summary="List tools")
async def list_tools(request: Request, limit: int = Query(50, ge=1, le=200), cursor: str | None = None,
                     principal: Principal = Depends(require("tools:read"))) -> dict:
    page = await services(request).repos.tools.list(principal.tenant, limit, cursor)
    return {"object": "list", "data": [tool_view(d) for d in page.items], "next_cursor": page.next_cursor}


@router.get("/{tool_id}", summary="Get a tool")
async def get_tool(tool_id: str, request: Request,
                   principal: Principal = Depends(require("tools:read"))) -> FastJSONResponse:
    doc = await services(request).repos.tools.get(principal.tenant, tool_id)
    return FastJSONResponse(tool_view(doc), headers=etag(doc))


@router.patch("/{tool_id}", summary="Update a tool (supports If-Match)")
async def update_tool(tool_id: str, body: ToolUpdate, request: Request,
                      principal: Principal = Depends(require("tools:write"))) -> FastJSONResponse:
    svc = services(request)
    current = await svc.repos.tools.get(principal.tenant, tool_id)
    changes = await _normalize(request, body.model_dump(exclude_unset=True), current["kind"])
    doc = await svc.repos.tools.update(principal.tenant, tool_id, changes, if_match(request))
    svc.resolver.invalidate(principal.tenant, "tool", tool_id)
    return FastJSONResponse(tool_view(doc), headers=etag(doc))


@router.delete("/{tool_id}", status_code=204, summary="Delete a tool")
async def delete_tool(tool_id: str, request: Request,
                      principal: Principal = Depends(require("tools:write"))) -> Response:
    svc = services(request)
    await svc.repos.tools.delete(principal.tenant, tool_id, if_match(request))
    svc.resolver.invalidate(principal.tenant, "tool", tool_id)
    return Response(status_code=204)


@router.post("/{tool_id}/invoke", summary="Run an HTTP tool once with given arguments (testing)")
async def invoke_tool(tool_id: str, body: ToolInvoke, request: Request,
                      principal: Principal = Depends(require("tools:write", rate_limited=True))) -> dict:
    svc = services(request)
    doc = await svc.repos.tools.get(principal.tenant, tool_id)
    if doc["kind"] != "http":
        raise BadRequest("only http tools run on the server")
    spec = ToolSpec(name=doc["name"], description=doc.get("description", ""), parameters=doc.get("parameters") or {},
                    kind="http", side_effects=bool(doc.get("side_effects")), timeout_s=doc.get("timeout_s"),
                    target={"tool_id": doc["id"], "http": doc["http"],
                            "secret_headers_enc": doc.get("secret_headers_enc"),
                            "signing_secret_enc": doc.get("signing_secret_enc")})
    call_id = new_id("call")
    ctx = ToolContext(tenant_id=principal.tenant, scope_id=f"invoke_{request.state.request_id}")
    outcome = await svc.executor.execute(ctx, spec, call_id, orjson.dumps(body.arguments).decode())
    return {"tool_id": tool_id, "call_id": call_id, "output": outcome.output, "data": outcome.data,
            "is_error": outcome.is_error, "duration_ms": outcome.duration_ms}
