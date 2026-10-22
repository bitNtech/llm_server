"""Pure-ASGI middleware (safe for streaming and WebSockets, unlike BaseHTTPMiddleware):
request ids, W3C trace context, request spans, body-size limit, metrics and structured access logs."""
from __future__ import annotations

import re
import secrets
import time

import orjson
from opentelemetry import context as otel_context
from opentelemetry.trace import SpanKind
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.errors import PayloadTooLarge
from app.core.logging import connection_id_var, get_logger, request_id_var, session_id_var, tenant_id_var
from app.core.metrics import HTTP_LATENCY, HTTP_REQUESTS
from app.core.tracing import extract_context, tracer

log = get_logger("app.http")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_QUIET = ("/health/live", "/health/ready", "/metrics")


def _route_label(scope: Scope) -> str:
    route = scope.get("route")
    return getattr(route, "path", None) or "unmatched"


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, max_body_bytes: int, access_log: bool) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.access_log = access_log

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope.get("headers") or []}
        request_id_var.set(None)
        session_id_var.set(None)
        tenant_id_var.set(None)
        token = otel_context.attach(extract_context(headers))
        try:
            if scope["type"] == "websocket":
                connection_id_var.set("ws_" + secrets.token_hex(8))
                with tracer.start_as_current_span("WS " + scope.get("path", ""), kind=SpanKind.SERVER):
                    await self.app(scope, receive, send)
                return
            await self._http(scope, receive, send, headers)
        finally:
            otel_context.detach(token)

    async def _http(self, scope: Scope, receive: Receive, send: Send, headers: dict[str, str]) -> None:
        incoming = headers.get("x-request-id", "")
        request_id = incoming if _SAFE_ID.match(incoming) else "req_" + secrets.token_hex(12)
        request_id_var.set(request_id)
        scope.setdefault("state", {})["request_id"] = request_id
        method = scope["method"]
        start = time.perf_counter()
        status = {"code": 500, "started": False}

        length = headers.get("content-length")
        if length and length.isdigit() and int(length) > self.max_body_bytes:
            await self._reject(send, request_id)
            self._observe(scope, method, 413, start)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise PayloadTooLarge(f"request body exceeds {self.max_body_bytes} bytes")
            return message

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                status["started"] = True
                message["headers"] = list(message.get("headers", [])) + [(b"x-request-id", request_id.encode())]
                HTTP_LATENCY.labels(method, _route_label(scope)).observe(time.perf_counter() - start)
            await send(message)

        with tracer.start_as_current_span(f"{method} {scope.get('path', '')}", kind=SpanKind.SERVER) as sp:
            sp.set_attribute("http.request.method", method)
            sp.set_attribute("request.id", request_id)
            try:
                await self.app(scope, limited_receive, send_wrapper)
            except PayloadTooLarge:
                if not status["started"]:
                    await self._reject(send, request_id)
                status["code"] = 413
            finally:
                sp.set_attribute("http.response.status_code", status["code"])
                sp.set_attribute("http.route", _route_label(scope))
                self._observe(scope, method, status["code"], start)

    async def _reject(self, send: Send, request_id: str) -> None:
        body = orjson.dumps(PayloadTooLarge(f"request body exceeds {self.max_body_bytes} bytes").to_dict(request_id))
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"x-request-id", request_id.encode())]})
        await send({"type": "http.response.body", "body": body})

    def _observe(self, scope: Scope, method: str, status: int, start: float) -> None:
        route = _route_label(scope)
        HTTP_REQUESTS.labels(method, route, str(status)).inc()
        if route in _QUIET:  # probes: a 503 during startup/drain is expected, not an incident
            return
        if status >= 500 or self.access_log:
            level = log.warning if status >= 500 else log.info
            level("request", extra={"method": method, "path": scope.get("path"), "status": status,
                                    "duration_ms": round((time.perf_counter() - start) * 1000, 1)})
