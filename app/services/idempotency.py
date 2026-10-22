"""`Idempotency-Key` support for create/run endpoints.

A client may send `Idempotency-Key: <unique string>` on any POST that creates something or starts
work. The first request with a key runs and its JSON response is stored for 24 hours (per tenant);
a retry with the same key and the same body gets the stored response back (header
`Idempotent-Replayed: true`) without running again. Reusing a key with a different body is a 409,
and a retry that arrives while the first request is still running is a 409 with Retry-After.
Streaming responses are not stored (their side effects are protected by `response_id` instead).
Stored bodies are encrypted with SECRETS_ENCRYPTION_KEY: some (new API keys, webhook secrets) are
credentials that must not sit in Firestore in plaintext.
"""
from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable

import orjson
from fastapi import Request
from starlette.responses import Response

from app.core.errors import BadRequest, Conflict
from app.security.crypto import SecretBox
from app.store.repos import IdempotencyRepo


class Idempotency:
    def __init__(self, repo: IdempotencyRepo, box: SecretBox) -> None:
        self.repo = repo
        self.box = box

    async def run(self, request: Request, tenant_id: str, body: bytes,
                  handler: Callable[[], Awaitable[Response]]) -> Response:
        key = request.headers.get("idempotency-key")
        if not key:
            return await handler()
        if len(key) > 255:
            raise BadRequest("Idempotency-Key must be at most 255 characters")
        key_hash = hashlib.sha256(f"{request.method} {request.url.path} {key}".encode()).hexdigest()
        request_hash = hashlib.sha256(body).hexdigest()
        existing = await self.repo.begin(tenant_id, key_hash, request_hash)
        if existing is not None:
            if existing.get("request_hash") != request_hash:
                raise Conflict("Idempotency-Key was already used with a different request body",
                               code="idempotency_key_reused")
            if existing.get("status") != "completed":
                raise Conflict("a request with this Idempotency-Key is still in progress",
                               code="idempotency_in_progress", retry_after=1)
            stored = self.box.decrypt(existing.get("response_body_enc")) or ""
            return Response(stored.encode(), status_code=existing["response_status"],
                            media_type=existing.get("media_type", "application/json"),
                            headers={"Idempotent-Replayed": "true"})
        try:
            response = await handler()
        except BaseException:
            await self.repo.abandon(tenant_id, key_hash)
            raise
        body_bytes = getattr(response, "body", None)
        if body_bytes is None or response.status_code >= 500:
            await self.repo.abandon(tenant_id, key_hash)  # streaming or failed: not replayable
            return response
        await self.repo.complete(tenant_id, key_hash, response.status_code,
                                 self.box.encrypt(bytes(body_bytes).decode(errors="replace")),
                                 response.media_type or "application/json")
        return response


def canonical(body: object) -> bytes:
    return orjson.dumps(body, option=orjson.OPT_SORT_KEYS)
