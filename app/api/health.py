"""Liveness, readiness and detailed status."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from app import __version__
from app.api.responses import FastJSONResponse
from app.api.security import require_auth

router = APIRouter(tags=["health"])


@router.get("/health/live", summary="Liveness: the gateway process is serving")
async def live() -> dict:
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness: engine healthy, model loaded, not draining")
async def ready(request: Request) -> FastJSONResponse:
    services = request.app.state.services
    engine = services.monitor.state
    if services.admission.draining:
        return FastJSONResponse({"status": "draining"}, status_code=503)
    if not engine.ready:
        return FastJSONResponse({"status": "not_ready", "reason": engine.last_error}, status_code=503)
    return FastJSONResponse({"status": "ready", "model": engine.model})


@router.get("/health", summary="Detailed status", dependencies=[Depends(require_auth)])
async def status(request: Request) -> FastJSONResponse:
    services = request.app.state.services
    settings = services.settings
    engine = services.monitor.state
    body = {
        "status": "ready" if engine.ready and not services.admission.draining else "not_ready",
        "version": __version__,
        "env": settings.app_env,
        "uptime_s": round(time.monotonic() - services.started_at, 1),
        "engine": {**engine.snapshot(), "mode": settings.engine_mode, "url": settings.resolved_engine_base_url},
        "admission": services.admission.snapshot(),
        "sessions": len(services.store),
        "websockets": services.ws_connections,
        "inflight": len(services.registry),
        "server_tools": services.tools.names(),
        "context_strategy": settings.context_strategy,
    }
    return FastJSONResponse(body, status_code=200 if body["status"] == "ready" else 503)
