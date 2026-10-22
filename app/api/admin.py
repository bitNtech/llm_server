"""Identity: who am I, tenants (platform admins) and API keys (tenant owners)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, Response

from app.api.deps import json_body, require, services
from app.api.responses import FastJSONResponse
from app.api.schemas import ApiKeyCreate, RealtimeTokenCreate, TenantCreate, TenantUpdate
from app.core.errors import Forbidden
from app.core.ids import iso, utcnow
from app.security.auth import ROLES, Principal, new_api_key
from app.security.crypto import new_secret
from app.store.repos import strip_internal

router = APIRouter(prefix="/v1", tags=["identity"])


def tenant_view(doc: dict[str, Any]) -> dict[str, Any]:
    view = strip_internal(doc)
    view["object"] = "tenant"
    view["webhook_secret_set"] = bool(doc.get("webhook_secret_enc"))
    return view


def key_view(doc: dict[str, Any]) -> dict[str, Any]:
    return {"object": "api_key", "id": doc["id"], "name": doc["name"], "role": doc["role"], "prefix": doc["prefix"],
            "tenant_id": doc["tenant_id"], "created_at": doc["created_at"], "created_by": doc.get("created_by"),
            "expires_at": doc.get("expires_at"), "revoked_at": doc.get("revoked_at"),
            "last_used_at": doc.get("last_used_at")}


@router.get("/me", summary="The authenticated principal")
async def me(principal: Principal = Depends(require(None))) -> dict:
    return {"object": "principal", "kind": principal.kind, "key_id": principal.key_id, "role": principal.role,
            "tenant_id": principal.tenant_id, "scopes": sorted(principal.scopes),
            "rate_limit_per_minute": principal.rate_limit_per_minute or None}


# ------------------------------------------------------------------ tenants
def _admin(principal: Principal) -> None:
    if not principal.is_admin:
        raise Forbidden("platform admin key required", code="admin_required")


@router.post("/tenants", status_code=201, summary="Create a tenant (+ its first owner key and webhook secret)")
async def create_tenant(request: Request, principal: Principal = Depends(require("usage:read"))) -> Response:
    _admin(principal)
    svc = services(request)
    data, raw = await json_body(request)
    spec = TenantCreate.model_validate(data)

    async def handler() -> Response:
        webhook_secret = new_secret("whsec")
        doc = await svc.repos.tenants.create(
            {"name": spec.name, "settings": spec.settings.model_dump(exclude_none=True) if spec.settings else {},
             "webhook_secret_enc": svc.box.encrypt(webhook_secret)}, spec.id)
        key = new_api_key()
        key_doc = await svc.repos.api_keys.create(key, doc["id"], "initial owner key", "owner", principal.key_id, None)
        body = {**tenant_view(doc), "webhook_secret": webhook_secret,
                "api_key": {**key_view(key_doc), "key": key}}
        return FastJSONResponse(body, status_code=201)

    return await svc.idempotency.run(request, "_platform", raw, handler)


@router.get("/tenants", summary="List tenants")
async def list_tenants(request: Request, limit: int = Query(50, ge=1, le=200), cursor: str | None = None,
                       principal: Principal = Depends(require("usage:read"))) -> dict:
    _admin(principal)
    page = await services(request).repos.tenants.list(limit, cursor)
    return {"object": "list", "data": [tenant_view(d) for d in page.items], "next_cursor": page.next_cursor}


@router.get("/tenants/{tenant_id}", summary="Get a tenant (admins, or the tenant itself)")
async def get_tenant(tenant_id: str, request: Request, principal: Principal = Depends(require("usage:read"))) -> dict:
    if not principal.is_admin and principal.tenant_id != tenant_id:
        raise Forbidden("not your tenant", code="tenant_mismatch")
    return tenant_view(await services(request).repos.tenants.get(tenant_id))


@router.patch("/tenants/{tenant_id}", summary="Rename, suspend/reactivate or change tenant settings")
async def update_tenant(tenant_id: str, body: TenantUpdate, request: Request,
                        principal: Principal = Depends(require("usage:read"))) -> dict:
    _admin(principal)
    svc = services(request)
    changes = body.model_dump(exclude_none=True)
    if "settings" in changes:
        current = await svc.repos.tenants.get(tenant_id)
        changes["settings"] = {**(current.get("settings") or {}), **changes["settings"]}
    doc = await svc.repos.tenants.update(tenant_id, changes)
    svc.auth.forget_tenant(tenant_id)
    return tenant_view(doc)


@router.delete("/tenants/{tenant_id}", status_code=204, summary="Delete a tenant (revokes all its keys)")
async def delete_tenant(tenant_id: str, request: Request,
                        principal: Principal = Depends(require("usage:read"))) -> Response:
    _admin(principal)
    svc = services(request)
    await svc.repos.tenants.get(tenant_id)
    for key_hash in await svc.repos.api_keys.revoke_all(tenant_id):
        svc.auth.forget(key_hash)
    await svc.repos.tenants.update(tenant_id, {"status": "deleted", "deleted_at": utcnow()})
    svc.auth.forget_tenant(tenant_id)
    return Response(status_code=204)


@router.post("/tenants/{tenant_id}/webhook-secret", summary="Rotate the tenant's webhook signing secret")
async def rotate_webhook_secret(tenant_id: str, request: Request,
                                principal: Principal = Depends(require("keys:write"))) -> dict:
    if not principal.is_admin and principal.tenant_id != tenant_id:
        raise Forbidden("not your tenant", code="tenant_mismatch")
    svc = services(request)
    secret = new_secret("whsec")
    await svc.repos.tenants.update(tenant_id, {"webhook_secret_enc": svc.box.encrypt(secret)})
    return {"tenant_id": tenant_id, "webhook_secret": secret}


# ----------------------------------------------------------------- API keys
@router.post("/api-keys", status_code=201, summary="Create an API key (the key is shown once)")
async def create_key(request: Request, principal: Principal = Depends(require("keys:write"))) -> Response:
    svc = services(request)
    data, raw = await json_body(request)
    spec = ApiKeyCreate.model_validate(data)
    tenant_id = principal.tenant
    if principal.role != "admin" and spec.role == "owner" and principal.role != "owner":
        raise Forbidden("only owners can create owner keys")
    if spec.role not in ROLES:
        raise Forbidden(f"unknown role {spec.role}")

    async def handler() -> Response:
        key = new_api_key()
        expires = utcnow() + timedelta(days=spec.expires_in_days) if spec.expires_in_days else None
        doc = await svc.repos.api_keys.create(key, tenant_id, spec.name, spec.role, principal.key_id, expires)
        return FastJSONResponse({**key_view(doc), "key": key}, status_code=201)

    return await svc.idempotency.run(request, tenant_id, raw, handler)


@router.get("/api-keys", summary="List the tenant's API keys")
async def list_keys(request: Request, principal: Principal = Depends(require("keys:read"))) -> dict:
    docs = await services(request).repos.api_keys.list(principal.tenant)
    return {"object": "list", "data": [key_view(d) for d in docs]}


@router.get("/api-keys/{key_id}", summary="Get an API key's metadata")
async def get_key(key_id: str, request: Request, principal: Principal = Depends(require("keys:read"))) -> dict:
    _, doc = await services(request).repos.api_keys.get(principal.tenant, key_id)
    return key_view(doc)


@router.delete("/api-keys/{key_id}", summary="Revoke an API key")
async def revoke_key(key_id: str, request: Request, principal: Principal = Depends(require("keys:write"))) -> dict:
    svc = services(request)
    key_hash, doc = await svc.repos.api_keys.revoke(principal.tenant, key_id)
    svc.auth.forget(key_hash)
    return key_view(doc)


# ---------------------------------------------------------- realtime tokens
@router.post("/realtime/tokens", status_code=201, summary="Mint a short-lived token for one realtime client")
async def realtime_token(body: RealtimeTokenCreate, request: Request,
                         principal: Principal = Depends(require("inference"))) -> dict:
    token, expires = services(request).auth.issue_token(principal, body.ttl_s, body.session_id, body.agent_id)
    return {"object": "realtime_token", "token": token, "expires_at": iso(datetime.fromtimestamp(expires, UTC)),
            "session_id": body.session_id, "agent_id": body.agent_id}
