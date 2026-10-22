"""Liveness, readiness and detailed status."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from app import __version__
from app.api.deps import require, services
from app.api.responses import FastJSONResponse
from app.security.auth import Principal

router = APIRouter(tags=["health"])


@router.get("/health/live", summary="Liveness: the process is serving")
async def live() -> dict:
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness: Vertex usable, Firestore reachable, not draining")
async def ready(request: Request) -> FastJSONResponse:
    svc = services(request)
    if svc.admission.draining:
        return FastJSONResponse({"status": "draining"}, status_code=503)
    reasons = []
    if not svc.provider.ready():
        reasons.append("vertex: " + (svc.provider.probe_error or "not ready"))
    if not svc.db.healthy:
        reasons.append("firestore: " + (svc.db.last_error or "not reachable"))
    if reasons:
        return FastJSONResponse({"status": "not_ready", "reasons": reasons}, status_code=503)
    return FastJSONResponse({"status": "ready", "model": svc.provider.default_model})


@router.get("/health", summary="Detailed status (any valid credential)")
async def status(request: Request, principal: Principal = Depends(require(None))) -> FastJSONResponse:
    svc = services(request)
    ok = svc.provider.ready() and svc.db.healthy and not svc.admission.draining
    body = {
        "status": "ready" if ok else "not_ready",
        "version": __version__,
        "env": svc.settings.app_env,
        "instance_id": svc.settings.instance_id,
        "uptime_s": round(time.monotonic() - svc.started_at, 1),
        "provider": svc.provider.snapshot(),
        "firestore": svc.db.snapshot(),
        "mcp": svc.mcp.snapshot(None if principal.is_admin else principal.tenant_id or "-"),
        "admission": svc.admission.snapshot(),
        "sessions_cached": len(svc.sessions),
        "websockets": len(svc.connections),
        "inflight_generations": len(svc.registry),
        "executions_running": len(svc.executions.running),
        "write_backlog": svc.writer.pending,
    }
    return FastJSONResponse(body, status_code=200 if ok else 503)
