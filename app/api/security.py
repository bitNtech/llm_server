"""API-key authentication and per-principal rate limiting.

Keys come from API_KEYS (comma-separated). Clients send `Authorization: Bearer <key>` or
`X-API-Key: <key>`; WebSocket clients that cannot set headers (browsers) may use `?api_key=`.
Keys are compared in constant time and never logged - logs carry a short SHA-256 fingerprint.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from collections import OrderedDict

from starlette.datastructures import Headers, QueryParams
from starlette.requests import HTTPConnection

from app.config import Settings
from app.core.errors import RateLimited, Unauthorized


def fingerprint(key: str) -> str:
    return "key_" + hashlib.sha256(key.encode()).hexdigest()[:10]


def extract_key(headers: Headers, query: QueryParams | None = None) -> str | None:
    auth = headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip() or None
    if headers.get("x-api-key"):
        return headers["x-api-key"].strip()
    if query is not None and query.get("api_key"):
        return query["api_key"]
    return None


class TokenBucket:
    __slots__ = ("capacity", "rate", "tokens", "updated")

    def __init__(self, per_minute: int) -> None:
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self.tokens = float(per_minute)
        self.updated = time.monotonic()

    def take(self) -> float:
        """Consume one token; returns 0 on success or the seconds until one is available."""
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= 1:
            self.tokens -= 1
            return 0.0
        return (1 - self.tokens) / self.rate


class Authenticator:
    def __init__(self, settings: Settings) -> None:
        self._keys = [k.encode() for k in settings.api_key_list]
        self.enabled = bool(self._keys)
        self.per_minute = settings.rate_limit_per_minute
        self._buckets: OrderedDict[str, TokenBucket] = OrderedDict()

    def authenticate(self, key: str | None, client_ip: str | None) -> str:
        """Return the principal (key fingerprint or client IP) or raise Unauthorized."""
        if not self.enabled:
            return f"ip:{client_ip or 'unknown'}"
        if not key:
            raise Unauthorized("missing API key (Authorization: Bearer <key> or X-API-Key)")
        candidate = key.encode()
        # compare against every key so timing does not reveal which (if any) matched
        matched = False
        for valid in self._keys:
            matched |= hmac.compare_digest(candidate, valid)
        if not matched:
            raise Unauthorized("invalid API key")
        return fingerprint(key)

    def rate_limit(self, principal: str) -> None:
        if not self.per_minute:
            return
        bucket = self._buckets.get(principal)
        if bucket is None:
            bucket = self._buckets[principal] = TokenBucket(self.per_minute)
            if len(self._buckets) > 50_000:
                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(principal)
        wait = bucket.take()
        if wait > 0:
            raise RateLimited(f"rate limit of {self.per_minute}/min exceeded", retry_after=round(wait, 1))


def require_auth(conn: HTTPConnection) -> str:
    """FastAPI dependency for HTTP routes (query-string keys are accepted only on WebSockets)."""
    auth: Authenticator = conn.app.state.services.auth
    client = conn.client.host if conn.client else None
    principal = auth.authenticate(extract_key(conn.headers), client)
    conn.state.principal = principal
    return principal


def require_auth_rate_limited(conn: HTTPConnection) -> str:
    principal = require_auth(conn)
    conn.app.state.services.auth.rate_limit(principal)
    return principal
