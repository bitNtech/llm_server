"""Workflow definitions; running one creates an asynchronous execution."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request, Response

from app.api.deps import etag, if_match, json_body, require, services
from app.api.executions import execution_view
from app.api.responses import FastJSONResponse
from app.api.schemas import WorkflowCreate, WorkflowRun, WorkflowUpdate
from app.security.auth import Principal
from app.services.workflows import validate_workflow
from app.store.repos import strip_internal

router = APIRouter(prefix="/v1/workflows", tags=["workflows"])


def workflow_view(doc: dict[str, Any]) -> dict[str, Any]:
    return {"object": "workflow", **strip_internal(doc)}


@router.post("", status_code=201, summary="Create a workflow")
async def create_workflow(request: Request, principal: Principal = Depends(require("workflows:write"))) -> Response:
    svc = services(request)
    data, raw = await json_body(request)
    spec = WorkflowCreate.model_validate(data).model_dump(exclude_none=True)
    validate_workflow(spec)

    async def handler() -> Response:
        doc = await svc.repos.workflows.create(principal.tenant, spec)
        return FastJSONResponse(workflow_view(doc), status_code=201, headers=etag(doc))

    return await svc.idempotency.run(request, principal.tenant, raw, handler)


@router.get("", summary="List workflows")
async def list_workflows(request: Request, limit: int = Query(50, ge=1, le=200), cursor: str | None = None,
                         principal: Principal = Depends(require("workflows:read"))) -> dict:
    page = await services(request).repos.workflows.list(principal.tenant, limit, cursor)
    return {"object": "list", "data": [workflow_view(d) for d in page.items], "next_cursor": page.next_cursor}


@router.get("/{workflow_id}", summary="Get a workflow")
async def get_workflow(workflow_id: str, request: Request,
                       principal: Principal = Depends(require("workflows:read"))) -> FastJSONResponse:
    doc = await services(request).repos.workflows.get(principal.tenant, workflow_id)
    return FastJSONResponse(workflow_view(doc), headers=etag(doc))


@router.patch("/{workflow_id}", summary="Update a workflow (running executions keep their snapshot)")
async def update_workflow(workflow_id: str, body: WorkflowUpdate, request: Request,
                          principal: Principal = Depends(require("workflows:write"))) -> FastJSONResponse:
    svc = services(request)
    changes = body.model_dump(exclude_unset=True)
    current = await svc.repos.workflows.get(principal.tenant, workflow_id)
    validate_workflow({**current, **changes})
    doc = await svc.repos.workflows.update(principal.tenant, workflow_id, changes, if_match(request))
    return FastJSONResponse(workflow_view(doc), headers=etag(doc))


@router.delete("/{workflow_id}", status_code=204, summary="Delete a workflow")
async def delete_workflow(workflow_id: str, request: Request,
                          principal: Principal = Depends(require("workflows:write"))) -> Response:
    await services(request).repos.workflows.delete(principal.tenant, workflow_id, if_match(request))
    return Response(status_code=204)


@router.post("/{workflow_id}/run", status_code=202, summary="Start an asynchronous run")
async def run_workflow(workflow_id: str, request: Request,
                       principal: Principal = Depends(require("executions:write", rate_limited=True))) -> Response:
    svc = services(request)
    data, raw = await json_body(request)
    spec = WorkflowRun.model_validate(data)

    async def handler() -> Response:
        doc = await svc.executions.submit(principal.tenant, {"type": "workflow", "workflow_id": workflow_id,
                                                             **spec.model_dump(exclude_none=True)}, principal.key_id)
        return FastJSONResponse(execution_view(doc), status_code=202)

    return await svc.idempotency.run(request, principal.tenant, raw, handler)
