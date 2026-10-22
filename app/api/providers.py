"""Provider management (read-only: provider configuration comes from the environment, never the API)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import require, services
from app.core.errors import NotFound
from app.security.auth import Principal

router = APIRouter(prefix="/v1", tags=["providers"])


@router.get("/providers", summary="Configured LLM providers with live route health")
async def list_providers(request: Request, _: Principal = Depends(require("agents:read"))) -> dict:
    return {"object": "list", "data": [services(request).provider.snapshot()]}


@router.get("/providers/{provider_id}", summary="One provider")
async def get_provider(provider_id: str, request: Request, _: Principal = Depends(require("agents:read"))) -> dict:
    if provider_id != "vertex":
        raise NotFound(f"provider {provider_id} not found", code="provider_not_found")
    return services(request).provider.snapshot()


@router.get("/models", summary="Models (OpenAI-compatible list): Vertex models and agent:<id> pseudo-models")
async def list_models(request: Request, principal: Principal = Depends(require("inference"))) -> dict:
    svc = services(request)
    data = [{"id": model, "object": "model", "owned_by": "google-vertex",
             "context_window": svc.settings.vertex_context_window, "default": model == svc.provider.default_model}
            for model in svc.provider.allowed_models]
    page = await svc.repos.agents.list(principal.tenant, limit=200)
    data += [{"id": f"agent:{a['id']}", "object": "model", "owned_by": "agent", "name": a["name"],
              "version": a["version"]} for a in page.items]
    return {"object": "list", "data": data}
