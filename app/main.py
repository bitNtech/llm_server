"""FastAPI application factory, service wiring and lifecycle (startup, graceful drain, shutdown)."""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from google.auth.exceptions import GoogleAuthError
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api import (
    admin,
    agents,
    chat,
    console,
    executions,
    health,
    mcp,
    metrics,
    providers,
    realtime,
    sessions,
    tools,
    workflows,
)
from app.api.middleware import RequestContextMiddleware
from app.api.responses import FastJSONResponse
from app.config import Settings, get_settings
from app.core.errors import BadRequest, GatewayError
from app.core.logging import get_logger, request_id_var, setup_logging
from app.core.tracing import setup_tracing, shutdown_tracing
from app.engine.admission import AdmissionController
from app.engine.agents import AgentResolver
from app.engine.context import ContextBuilder
from app.engine.generation import GenerationRegistry
from app.engine.runtime import ConversationRuntime
from app.engine.sessions import LiveSession, SessionManager
from app.gcp.credentials import GoogleTokenSource, TokenSource, load_credentials
from app.providers.vertex import VertexProvider
from app.security.auth import Authenticator
from app.security.crypto import SecretBox
from app.security.netguard import NetGuard
from app.services.chat import ChatService
from app.services.executions import ExecutionService
from app.services.idempotency import Idempotency
from app.services.usage import UsageRecorder
from app.services.webhooks import WebhookSender
from app.services.workflows import WorkflowRunner
from app.store.db import Database
from app.store.repos import (
    AgentRepo,
    ApiKeyRepo,
    ExecutionRepo,
    IdempotencyRepo,
    McpServerRepo,
    SessionRepo,
    TenantRepo,
    ToolCallRepo,
    ToolRepo,
    UsageRepo,
    WorkflowRepo,
)
from app.store.writer import Writer
from app.tools.executor import ToolExecutor
from app.tools.http import HttpToolRunner
from app.tools.mcp import MCPManager

log = get_logger("app")


@dataclass
class Repos:
    tenants: TenantRepo
    api_keys: ApiKeyRepo
    agents: AgentRepo
    tools: ToolRepo
    mcp_servers: McpServerRepo
    workflows: WorkflowRepo
    sessions: SessionRepo
    tool_calls: ToolCallRepo
    executions: ExecutionRepo
    idempotency: IdempotencyRepo
    usage: UsageRepo


@dataclass
class Services:
    settings: Settings
    db: Database
    repos: Repos
    writer: Writer
    tokens: TokenSource
    provider: VertexProvider
    box: SecretBox
    guard: NetGuard
    auth: Authenticator
    admission: AdmissionController
    registry: GenerationRegistry
    http_tools: HttpToolRunner
    mcp: MCPManager
    executor: ToolExecutor
    resolver: AgentResolver
    sessions: SessionManager
    context: ContextBuilder
    usage: UsageRecorder
    runtime: ConversationRuntime
    chat: ChatService
    runner: WorkflowRunner
    webhooks: WebhookSender
    executions: ExecutionService
    idempotency: Idempotency
    started_at: float = field(default_factory=time.monotonic)
    connections: set = field(default_factory=set)  # live RealtimeConnection objects
    loop: asyncio.AbstractEventLoop | None = None

    def begin_drain(self) -> None:
        """First reaction to SIGTERM: fail readiness, refuse new work, wind down WebSockets.
        Safe to call from any thread (signal handlers, supervisors)."""
        if self.loop is not None:
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is not self.loop:
                self.loop.call_soon_threadsafe(self.begin_drain)
                return
        if self.admission.draining:
            return
        self.admission.draining = True
        log.info("shutdown signal: draining", extra={"inflight": len(self.registry),
                                                     "websockets": len(self.connections)})
        for conn in list(self.connections):
            conn.start_drain(self.settings.shutdown_grace_s)


def build_services(settings: Settings, tokens: TokenSource | None = None, vertex_transport: Any = None) -> Services:
    credentials = None
    if tokens is None:
        try:
            credentials, discovered_project = load_credentials(settings.gcp_credentials_json.get_secret_value())
        except (GoogleAuthError, OSError, ValueError) as exc:
            raise RuntimeError(f"Google Cloud credentials unavailable ({exc.__class__.__name__}: {exc}). Set "
                               "GOOGLE_APPLICATION_CREDENTIALS or GCP_CREDENTIALS_JSON, use workload identity, "
                               "or run `gcloud auth application-default login`.") from exc
        if not settings.gcp_project_id and discovered_project:
            settings.gcp_project_id = discovered_project
        tokens = GoogleTokenSource(credentials)
    if not settings.gcp_project_id:
        raise ValueError("GCP_PROJECT_ID is required (it could not be discovered from the credentials)")
    db_credentials = credentials
    if settings.firestore_credentials_json.get_secret_value():
        # Firestore in another project/account: its own service account instead of the Vertex one
        try:
            db_credentials, firestore_project = load_credentials(settings.firestore_credentials_json.get_secret_value())
        except (GoogleAuthError, OSError, ValueError) as exc:
            raise RuntimeError(f"FIRESTORE_CREDENTIALS_JSON is invalid ({exc.__class__.__name__}: {exc})") from exc
        if not settings.firestore_project_id and firestore_project:
            settings.firestore_project_id = firestore_project
    db = Database(settings, db_credentials)
    repos = Repos(TenantRepo(db), ApiKeyRepo(db), AgentRepo(db), ToolRepo(db), McpServerRepo(db), WorkflowRepo(db),
                  SessionRepo(db), ToolCallRepo(db), ExecutionRepo(db), IdempotencyRepo(db), UsageRepo(db))
    writer = Writer()
    provider = VertexProvider(settings, tokens, transport=vertex_transport)
    box = SecretBox(settings.encryption_keys, settings.is_production)
    guard = NetGuard(settings.allow_private_network_targets, settings.allowed_target_host_list, settings.is_production)
    auth = Authenticator(settings, repos.api_keys, repos.tenants)
    admission = AdmissionController(settings.max_concurrent_generations, settings.max_queued_generations,
                                    settings.queue_timeout_s)
    registry = GenerationRegistry()
    http_tools = HttpToolRunner(settings, guard, box)
    mcp_manager = MCPManager(settings, box, guard)
    executor = ToolExecutor(settings, repos.tool_calls, writer, http_tools, mcp_manager, repos.mcp_servers)
    resolver = AgentResolver(settings, provider, repos.agents, repos.tools, repos.mcp_servers, mcp_manager)
    sessions_mgr = SessionManager(settings, repos.sessions, writer)
    context = ContextBuilder(settings, provider, sessions_mgr)
    usage = UsageRecorder(repos.usage, settings.usage_flush_interval_s)
    runtime = ConversationRuntime(settings, provider, admission, registry, executor, resolver, context,
                                  sessions_mgr, usage)
    chat_service = ChatService(settings, provider, admission, registry, resolver, runtime, usage)
    runner = WorkflowRunner(settings, provider, admission, runtime, resolver, executor, usage)
    webhooks = WebhookSender(settings, guard, box)
    execution_service = ExecutionService(settings, repos.executions, repos.workflows, repos.tenants, runtime,
                                         runner, resolver, sessions_mgr, webhooks, guard, writer)
    services = Services(
        settings=settings, db=db, repos=repos, writer=writer, tokens=tokens, provider=provider, box=box,
        guard=guard, auth=auth, admission=admission, registry=registry, http_tools=http_tools, mcp=mcp_manager,
        executor=executor, resolver=resolver, sessions=sessions_mgr, context=context, usage=usage,
        runtime=runtime, chat=chat_service, runner=runner, webhooks=webhooks, executions=execution_service,
        idempotency=Idempotency(repos.idempotency, box))

    async def on_session_end(session: LiveSession) -> None:
        """Run the agent's post-session workflow with the transcript."""
        if not session.agent_id:
            return
        agent = await resolver.resolve(session.tenant_id, session.agent_id, session.agent_version)
        hook = agent.on_session_end or {}
        if not hook.get("workflow_id"):
            return
        transcript = [m.to_public() for m in session.messages if m.role in ("user", "assistant") and m.content]
        await execution_service.submit(session.tenant_id, {
            "type": "workflow", "workflow_id": hook["workflow_id"],
            "input": {"session_id": session.id, "agent_id": session.agent_id, "end_reason": session.end_reason,
                      "metadata": session.metadata, "transcript": transcript},
            "metadata": {"trigger": "session_end", "session_id": session.id}}, created_by="system")

    sessions_mgr.on_end = on_session_end
    return services


def _error_response(exc: GatewayError, request: Request) -> FastJSONResponse:
    headers = {"Retry-After": str(max(1, round(exc.retry_after)))} if exc.retry_after else None
    rid = getattr(request.state, "request_id", None) or request_id_var.get()
    return FastJSONResponse(exc.to_dict(rid), status_code=exc.status, headers=headers)


def create_app(settings: Settings | None = None, tokens: TokenSource | None = None,
               vertex_transport: Any = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.resolved_log_format, settings.gcp_project_id or None)
    setup_tracing(settings.otel_service_name, settings.otel_exporter_otlp_endpoint, __version__,
                  settings.otel_sample_ratio)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        services = build_services(settings, tokens, vertex_transport)
        services.loop = asyncio.get_running_loop()
        app.state.services = services
        services.tokens.start()
        services.db.start()
        services.provider.start()
        services.sessions.start()
        services.usage.start()
        services.executions.start()
        log.info("server starting", extra={
            "version": __version__, "env": settings.app_env, "instance_id": settings.instance_id,
            "project": settings.gcp_project_id, "model": settings.vertex_model, "locations": settings.location_list,
            "firestore_database": settings.firestore_database, "port": settings.port})
        try:
            yield
        finally:
            # drain order: stop intake -> finish in-flight turns -> executions -> persist -> close clients
            services.begin_drain()
            grace = settings.shutdown_grace_s
            cancelled = await services.registry.drain(grace)
            await services.executions.drain(grace / 2)
            await services.runtime.close()
            await services.context.close()
            await services.sessions.close()
            await services.usage.close()
            await services.writer.flush(timeout=15)
            await services.webhooks.aclose()
            await services.mcp.close()
            await services.http_tools.aclose()
            await services.provider.aclose()
            await services.tokens.stop()
            await services.db.close()
            shutdown_tracing()
            log.info("server stopped", extra={"cancelled_on_shutdown": cancelled})

    docs = settings.docs_are_enabled
    app = FastAPI(
        title="AICA LLM Server",
        version=__version__,
        description="Vertex AI inference and orchestration server for real-time voice agents: agents, tools, "
                    "MCP, workflows, sessions, realtime WebSocket and an OpenAI-compatible API.",
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
        return _error_response(BadRequest("request validation failed", code="validation_error",
                                          details={"errors": details}), request)

    @app.exception_handler(ValidationError)
    async def body_validation_error(request: Request, exc: ValidationError) -> FastJSONResponse:
        details = [{"loc": list(e.get("loc", ())), "msg": e.get("msg")} for e in exc.errors()]
        return _error_response(BadRequest("request validation failed", code="validation_error",
                                          details={"errors": details}), request)

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
            CORSMiddleware, allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID", "X-Tenant-ID",
                           "Idempotency-Key", "If-Match"],
            expose_headers=["X-Request-ID", "Retry-After", "ETag", "Idempotent-Replayed"], max_age=600)
    app.add_middleware(RequestContextMiddleware, max_body_bytes=settings.max_request_body_bytes,
                       access_log=settings.access_log)

    for module in (console, health, metrics, admin, providers, agents, tools, mcp, workflows, executions, sessions,
                   chat, realtime):
        app.include_router(module.router)
    app.include_router(executions.usage_router)
    return app
