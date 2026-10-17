"""Prometheus endpoints: gateway metrics and a pass-through of vLLM's engine metrics."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.security import require_auth
from app.core.errors import NotFound
from app.observability.metrics import REGISTRY

router = APIRouter(tags=["observability"])


def _guard(request: Request) -> None:
    services = request.app.state.services
    if not services.settings.metrics_enabled:
        raise NotFound("metrics are disabled")
    if services.settings.metrics_require_auth:
        require_auth(request)


@router.get("/metrics", summary="Gateway metrics (Prometheus)")
async def metrics(request: Request) -> Response:
    _guard(request)
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@router.get("/metrics/engine", summary="vLLM engine metrics (Prometheus, proxied)")
async def engine_metrics(request: Request) -> PlainTextResponse:
    _guard(request)
    text = await request.app.state.services.engine.metrics_text()
    return PlainTextResponse(text, media_type=CONTENT_TYPE_LATEST)
