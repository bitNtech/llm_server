"""Route dependencies: service access, authentication/authorization, body parsing, ETags."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import orjson
from fastapi import Request
from starlette.datastructures import Headers
from starlette.requests import HTTPConnection

from app.core.errors import BadRequest, Unauthorized
from app.core.logging import tenant_id_var
from app.security.auth import Principal, looks_like_jwt

if TYPE_CHECKING:
    from app.main import Services


def services(conn: HTTPConnection) -> Services:
    return conn.app.state.services


def extract_credential(headers: Headers) -> str | None:
    auth = headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip() or None
    if headers.get("x-api-key"):
        return headers["x-api-key"].strip()
    return None


async def authenticate(conn: HTTPConnection, credential: str | None) -> Principal:
    svc = services(conn)
    if credential and looks_like_jwt(credential):
        principal = await svc.auth.verify_token(credential)
    else:
        principal = await svc.auth.authenticate(credential)
    principal = await svc.auth.act_as_tenant(principal, conn.headers.get("x-tenant-id"))
    if principal.tenant_id:
        tenant_id_var.set(principal.tenant_id)
    conn.state.principal = principal
    return principal


def require(scope: str | None, *, rate_limited: bool = False) -> Callable[[HTTPConnection], Awaitable[Principal]]:
    """Dependency: authenticate, check one permission scope (None = any valid credential),
    optionally apply the rate limit."""

    async def dependency(conn: HTTPConnection) -> Principal:
        principal = await authenticate(conn, extract_credential(conn.headers))
        if principal.kind == "realtime_token":
            raise Unauthorized("realtime tokens are only valid on the realtime WebSocket", code="invalid_token")
        if scope is not None:
            principal.require(scope)
        if rate_limited:
            services(conn).auth.rate_limit(principal)
        return principal

    dependency.__name__ = f"require_{(scope or 'auth').replace(':', '_')}"
    return dependency


async def json_body(request: Request) -> tuple[Any, bytes]:
    raw = await request.body()
    if not raw:
        raise BadRequest("request body is required")
    try:
        return orjson.loads(raw), raw
    except orjson.JSONDecodeError as exc:
        raise BadRequest(f"invalid JSON: {exc}") from None


def if_match(request: Request) -> int | None:
    """Parse `If-Match: "v3"` / `3` for optimistic concurrency."""
    raw = request.headers.get("if-match")
    if not raw:
        return None
    value = raw.strip().strip('"').removeprefix("W/").strip('"').lstrip("v")
    if not value.isdigit():
        raise BadRequest('If-Match must be a version ETag such as "v3"')
    return int(value)


def etag(doc: dict[str, Any]) -> dict[str, str]:
    return {"ETag": f'"v{doc.get("version", 0)}"'}
