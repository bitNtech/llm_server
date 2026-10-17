"""FastAPI application factory and lifecycle."""
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api import health, metrics, openai_routes, realtime, sessions
from app.api.middleware import RequestContextMiddleware
from app.api.responses import FastJSONResponse
from app.api.security import Authenticator
from app.config import Settings, get_settings
from app.core.admission import AdmissionController
from app.core.conversation import ConversationService
from app.core.errors import GatewayError
from app.core.generation import GenerationRegistry
from app.core.openai_proxy import OpenAIProxy
from app.engine.client import EngineClient
from app.engine.monitor import EngineMonitor
from app.observability.logging import get_logger, request_id_var, setup_logging
from app.sessions.context import ContextManager
from app.sessions.store import SessionStore
from app.tools.registry import ToolRegistry

log = get_logger("app")


@dataclass
class Services:
    settings: Settings
    engine: EngineClient
    monitor: EngineMonitor
    admission: AdmissionController
    registry: GenerationRegistry
    tools: ToolRegistry
    store: SessionStore
    context: ContextManager
    conversation: ConversationService
    openai: OpenAIProxy
    auth: Authenticator
    started_at: float = field(default_factory=time.monotonic)
    ws_connections: int = 0
    connections: set = field(default_factory=set)  # live RealtimeConnection objects

    def begin_drain(self) -> None:
        """First reaction to SIGTERM: fail readiness, refuse new work, wind down WebSockets."""
        if self.admission.draining:
            return
        self.admission.draining = True
        log.info("shutdown signal: draining", extra={"inflight": len(self.registry),
                                                     "websockets": len(self.connections)})
        for conn in list(self.connections):
            conn.start_drain(self.settings.shutdown_grace_s)


def build_services(settings: Settings, engine: EngineClient | None = None) -> Services:
    engine = engine or EngineClient(settings)
    monitor = EngineMonitor(engine, settings.engine_health_interval_s, settings.served_model_name,
                            warmup=settings.engine_warmup, warmup_timeout_s=settings.engine_warmup_timeout_s)
    admission = AdmissionController(
        settings.max_concurrent_generations, settings.max_queued_generations, settings.queue_timeout_s)
    registry = GenerationRegistry()
    tools = ToolRegistry(settings.tool_timeout_s)
    tools.load_modules(settings.tools_module_list)
    store = SessionStore(settings.max_sessions, settings.session_ttl_s,
                         settings.max_messages_per_session, settings.context_chars_per_token)
    context = ContextManager(settings, engine, monitor)
    conversation = ConversationService(settings, engine, monitor, admission, registry, tools, context, store)
    return Services(
        settings=settings, engine=engine, monitor=monitor, admission=admission, registry=registry,
        tools=tools, store=store, context=context, conversation=conversation,
        openai=OpenAIProxy(settings, engine, monitor, admission, registry), auth=Authenticator(settings),
    )


def _error_response(exc: GatewayError, request: Request) -> FastJSONResponse:
    headers = {"Retry-After": str(max(1, round(exc.retry_after)))} if exc.retry_after else None
    rid = getattr(request.state, "request_id", None) or request_id_var.get()
    return FastJSONResponse(exc.to_dict(rid), status_code=exc.status, headers=headers)


def create_app(settings: Settings | None = None, engine: EngineClient | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.resolved_log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        services = build_services(settings, engine)
        app.state.services = services
        services.monitor.start()
        services.store.start()
        log.info("gateway starting", extra={
            "version": __version__, "env": settings.app_env, "engine_url": settings.resolved_engine_base_url,
            "auth": services.auth.enabled, "port": settings.port})
        try:
            yield
        finally:
            # graceful drain: refuse new work, let in-flight responses finish, then cancel stragglers
            services.begin_drain()
            inflight = len(services.registry)
            log.info("draining", extra={"inflight": inflight, "grace_s": settings.shutdown_grace_s})
            cancelled = await services.registry.drain(settings.shutdown_grace_s)
            await services.context.close()
            await services.monitor.stop()
            await services.store.stop()
            await services.engine.aclose()
            log.info("gateway stopped", extra={"cancelled_on_shutdown": cancelled})

    docs = settings.docs_are_enabled
    app = FastAPI(
        title="LLM Inference Server",
        version=__version__,
        description="vLLM-backed inference gateway for real-time voice agents.",
        lifespan=lifespan,
        default_response_class=FastJSONResponse,
        docs_url="/docs" if docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs else None,
    )

    @app.exception_handler(GatewayError)
    async def gateway_error(request: Request, exc: GatewayError) -> FastJSONResponse:
        return _error_response(exc, request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> FastJSONResponse:
        details = [{"loc": list(e.get("loc", ())), "msg": e.get("msg")} for e in exc.errors()]
        err = GatewayError("request validation failed", code="validation_error", status=400,
                           details={"errors": details})
        err.type = "invalid_request_error"
        return _error_response(err, request)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> FastJSONResponse:
        err = GatewayError(str(exc.detail), code="not_found" if exc.status_code == 404 else "http_error",
                           status=exc.status_code)
        err.type = "invalid_request_error"
        return _error_response(err, request)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> FastJSONResponse:
        log.exception("unhandled error", extra={"path": request.url.path})
        return _error_response(GatewayError("internal server error"), request)

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
            expose_headers=["X-Request-ID", "Retry-After"],
            max_age=600,
        )
    app.add_middleware(RequestContextMiddleware, max_body_bytes=settings.max_request_body_bytes,
                       access_log=settings.access_log)

    for module in (health, metrics, openai_routes, sessions, realtime):
        app.include_router(module.router)
    return app
