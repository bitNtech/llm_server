"""Authentication and authorization.

Credentials:
  * platform admin keys (ADMIN_API_KEYS env)  - manage tenants; act inside a tenant with X-Tenant-ID
  * tenant API keys (stored hashed in Firestore, created through /v1/api-keys) - carry one role
  * ephemeral realtime tokens (HS256 JWT, minutes) - minted by a tenant key for one WebSocket
    client (browser, mobile, LiveKit room participant), optionally pinned to a session/agent

Keys are compared in constant time and never logged; logs carry the key id. Lookups are cached for
API_KEY_CACHE_TTL_S, so a revocation takes effect immediately on the instance that revoked it and
within that TTL everywhere else.
"""
from __future__ import annotations

import hmac
import secrets
import time
from dataclasses import dataclass, field, replace
from typing import Any

import jwt

from app.config import Settings
from app.core.errors import BadRequest, Forbidden, Unauthorized
from app.core.ids import utcnow
from app.core.logging import get_logger
from app.security.ratelimit import RateLimiter
from app.store.repos import ApiKeyRepo, TenantRepo, hash_key

log = get_logger("app.auth")

READ_SCOPES = {"agents:read", "tools:read", "mcp:read", "workflows:read", "sessions:read", "executions:read",
               "usage:read", "keys:read"}
WRITE_SCOPES = {"agents:write", "tools:write", "mcp:write", "workflows:write", "sessions:write",
                "executions:write", "keys:write", "inference"}
TENANT_SCOPES = READ_SCOPES | WRITE_SCOPES
ROLES: dict[str, frozenset[str]] = {
    "owner": frozenset(TENANT_SCOPES),
    "developer": frozenset(TENANT_SCOPES - {"keys:write"}),
    "runtime": frozenset({"agents:read", "tools:read", "mcp:read", "workflows:read", "sessions:read",
                          "sessions:write", "executions:read", "executions:write", "inference"}),
    "viewer": frozenset(READ_SCOPES - {"keys:read"}),
}
ADMIN_SCOPES = frozenset(TENANT_SCOPES | {"admin"})
KEY_PREFIX = "aica_"
TOKEN_AUDIENCE = "aica-realtime"


@dataclass(frozen=True)
class Principal:
    kind: str  # admin | api_key | realtime_token
    key_id: str
    role: str
    scopes: frozenset[str]
    tenant_id: str | None = None
    session_id: str | None = None  # realtime tokens may pin one session
    agent_id: str | None = None  # ... or one agent
    rate_limit_per_minute: int = 0
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def is_admin(self) -> bool:
        return "admin" in self.scopes

    def require(self, scope: str) -> None:
        if scope not in self.scopes:
            raise Forbidden(f"this credential lacks the '{scope}' permission", code="insufficient_scope",
                            details={"required": scope, "role": self.role})

    @property
    def tenant(self) -> str:
        if not self.tenant_id:
            raise BadRequest("platform admin keys must name a tenant with the X-Tenant-ID header",
                             code="tenant_required")
        return self.tenant_id


def new_api_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


class Authenticator:
    def __init__(self, settings: Settings, keys: ApiKeyRepo, tenants: TenantRepo) -> None:
        self.settings = settings
        self.keys = keys
        self.tenants = tenants
        self._admin = [k.encode() for k in settings.admin_key_list]
        self._cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
        self._tenant_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
        self.limiter = RateLimiter()
        secret = settings.realtime_token_secret.get_secret_value()
        if not secret:
            secret = secrets.token_urlsafe(48)
            log.warning("REALTIME_TOKEN_SECRET not set: realtime tokens are valid on this process only")
        self._token_secret = secret
        if not self._admin:
            log.warning("ADMIN_API_KEYS is empty: no one can create tenants or API keys")

    # ------------------------------------------------------------------ keys
    async def authenticate(self, key: str | None) -> Principal:
        if not key:
            raise Unauthorized("missing credentials (Authorization: Bearer <key> or X-API-Key)")
        candidate = key.encode()
        is_admin = False
        for valid in self._admin:  # constant time over every admin key
            is_admin |= hmac.compare_digest(candidate, valid)
        if is_admin:
            return Principal(kind="admin", key_id="admin_" + hash_key(key)[:10], role="admin",
                             scopes=ADMIN_SCOPES)
        if not key.startswith(KEY_PREFIX):
            raise Unauthorized("invalid API key")
        key_hash = hash_key(key)
        doc = await self._lookup(key_hash)
        if doc is None or doc.get("revoked_at") is not None:
            raise Unauthorized("invalid API key")
        if doc.get("expires_at") is not None and doc["expires_at"] < utcnow():
            raise Unauthorized("API key expired", code="api_key_expired")
        tenant = await self.tenant(doc["tenant_id"])
        if tenant is None or tenant.get("status") != "active":
            raise Forbidden("tenant is suspended or deleted", code="tenant_inactive")
        role = doc.get("role", "runtime")
        return Principal(kind="api_key", key_id=doc["id"], role=role, scopes=ROLES.get(role, frozenset()),
                         tenant_id=doc["tenant_id"], rate_limit_per_minute=self._rate_for(tenant))

    async def _lookup(self, key_hash: str) -> dict[str, Any] | None:
        now = time.monotonic()
        cached = self._cache.get(key_hash)
        if cached and cached[0] > now:
            return cached[1]
        doc = await self.keys.lookup(key_hash)
        ttl = self.settings.api_key_cache_ttl_s if doc is not None else min(5.0, self.settings.api_key_cache_ttl_s)
        if len(self._cache) > 100_000:
            self._cache.clear()
        self._cache[key_hash] = (now + ttl, doc)
        return doc

    def forget(self, key_hash: str) -> None:
        self._cache.pop(key_hash, None)

    async def tenant(self, tenant_id: str) -> dict[str, Any] | None:
        now = time.monotonic()
        cached = self._tenant_cache.get(tenant_id)
        if cached and cached[0] > now:
            return cached[1]
        doc = await self.tenants.find(tenant_id)
        self._tenant_cache[tenant_id] = (now + self.settings.api_key_cache_ttl_s, doc)
        return doc

    def forget_tenant(self, tenant_id: str) -> None:
        self._tenant_cache.pop(tenant_id, None)

    def _rate_for(self, tenant: dict[str, Any]) -> int:
        override = (tenant.get("settings") or {}).get("rate_limit_per_minute")
        return int(override) if override is not None else self.settings.rate_limit_per_minute

    async def act_as_tenant(self, principal: Principal, tenant_id: str | None) -> Principal:
        """Apply X-Tenant-ID: only platform admins may choose a tenant."""
        if not tenant_id:
            return principal
        if not principal.is_admin:
            if tenant_id != principal.tenant_id:
                raise Forbidden("X-Tenant-ID does not match the credential's tenant", code="tenant_mismatch")
            return principal
        tenant = await self.tenant(tenant_id)
        if tenant is None or tenant.get("status") == "deleted":
            raise Forbidden(f"tenant {tenant_id} does not exist", code="tenant_not_found")
        return replace(principal, tenant_id=tenant_id)

    def rate_limit(self, principal: Principal) -> None:
        limit = principal.rate_limit_per_minute
        if principal.kind == "admin":
            return
        self.limiter.check(principal.key_id, limit or self.settings.rate_limit_per_minute)

    # -------------------------------------------------------- realtime tokens
    def issue_token(self, principal: Principal, ttl_s: int, session_id: str | None,
                    agent_id: str | None) -> tuple[str, int]:
        ttl_s = max(30, min(ttl_s, self.settings.realtime_token_max_ttl_s))
        now = int(time.time())
        claims = {"iss": "aica-llm-server", "aud": TOKEN_AUDIENCE, "sub": principal.key_id,
                  "tid": principal.tenant, "iat": now, "exp": now + ttl_s, "jti": secrets.token_hex(8),
                  "scp": sorted(principal.scopes & {"inference", "sessions:read", "sessions:write"})}
        if session_id:
            claims["sid"] = session_id
        if agent_id:
            claims["aid"] = agent_id
        return jwt.encode(claims, self._token_secret, algorithm="HS256"), now + ttl_s

    async def verify_token(self, token: str) -> Principal:
        try:
            claims = jwt.decode(token, self._token_secret, algorithms=["HS256"], audience=TOKEN_AUDIENCE,
                                options={"require": ["exp", "iat", "tid", "sub"]})
        except jwt.ExpiredSignatureError:
            raise Unauthorized("realtime token expired", code="token_expired") from None
        except jwt.PyJWTError:
            raise Unauthorized("invalid realtime token", code="invalid_token") from None
        tenant = await self.tenant(claims["tid"])
        if tenant is None or tenant.get("status") != "active":
            raise Forbidden("tenant is suspended or deleted", code="tenant_inactive")
        return Principal(kind="realtime_token", key_id=claims["sub"], role="realtime",
                         scopes=frozenset(claims.get("scp", [])), tenant_id=claims["tid"],
                         session_id=claims.get("sid"), agent_id=claims.get("aid"),
                         rate_limit_per_minute=self._rate_for(tenant))


def looks_like_jwt(value: str) -> bool:
    return value.count(".") == 2 and value.startswith("ey")
