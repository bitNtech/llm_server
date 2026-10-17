"""Pure-ASGI middleware (safe for streaming and WebSockets, unlike BaseHTTPMiddleware)."""
from __future__ import annotations

import re
import secrets
import time

import orjson
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.errors import PayloadTooLarge
from app.observability.logging import connection_id_var, get_logger, request_id_var
from app.observability.metrics import HTTP_LATENCY, HTTP_REQUESTS

log = get_logger("app.http")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _route_label(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path or "unmatched"


class RequestContextMiddleware:
    """Request ids, body-size limit, metrics and structured access logs."""

    def __init__(self, app: ASGIApp, max_body_bytes: int, access_log: bool) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.access_log = access_log

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            connection_id_var.set("ws_" + secrets.token_hex(8))
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(b"x-request-id", b"").decode(errors="ignore")
        request_id = incoming if _SAFE_ID.match(incoming) else "req_" + secrets.token_hex(12)
        request_id_var.set(request_id)
        scope.setdefault("state", {})["request_id"] = request_id
        method = scope["method"]
        start = time.perf_counter()
        status_holder = {"status": 500, "started": False}

        length = headers.get(b"content-length")
        if length and length.isdigit() and int(length) > self.max_body_bytes:
            await self._reject(send, request_id)
            self._observe(scope, method, 413, start, request_id)
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
                status_holder["status"] = message["status"]
                status_holder["started"] = True
                message.setdefault("headers", [])
                message["headers"] = list(message["headers"]) + [(b"x-request-id", request_id.encode())]
                HTTP_LATENCY.labels(method, _route_label(scope)).observe(time.perf_counter() - start)
            await send(message)

        try:
            await self.app(scope, limited_receive, send_wrapper)
        except PayloadTooLarge:
            if not status_holder["started"]:
                await self._reject(send, request_id)
            status_holder["status"] = 413
        finally:
            self._observe(scope, method, status_holder["status"], start, request_id)

    async def _reject(self, send: Send, request_id: str) -> None:
        body = orjson.dumps(PayloadTooLarge(f"request body exceeds {self.max_body_bytes} bytes").to_dict(request_id))
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"x-request-id", request_id.encode())]})
        await send({"type": "http.response.body", "body": body})

    def _observe(self, scope: Scope, method: str, status: int, start: float, request_id: str) -> None:
        route = _route_label(scope)
        HTTP_REQUESTS.labels(method, route, str(status)).inc()
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        if status >= 500 or (self.access_log and route not in ("/health/live", "/health/ready", "/metrics")):
            level = log.warning if status >= 500 else log.info
            level("request", extra={"method": method, "path": scope.get("path"), "status": status,
                                    "duration_ms": duration_ms})
