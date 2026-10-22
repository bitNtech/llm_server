"""Prometheus scrape endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.deps import authenticate, extract_credential, services
from app.core.errors import NotFound
from app.core.metrics import REGISTRY

router = APIRouter(tags=["observability"])


@router.get("/metrics", summary="Prometheus metrics")
async def metrics(request: Request) -> Response:
    svc = services(request)
    if not svc.settings.metrics_enabled:
        raise NotFound("metrics are disabled")
    if svc.settings.metrics_require_auth:
        principal = await authenticate(request, extract_credential(request.headers))
        principal.require("usage:read")
    svc.provider.ready()  # refresh the readiness gauge
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
