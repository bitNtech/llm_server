"""Repositories over Firestore. Every tenant-owned method takes the tenant id first and can only
address documents below `tenants/{tenant_id}`; there is no cross-tenant query anywhere."""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from google.cloud import firestore

from app.core.errors import BadRequest, Conflict, NotFound, PreconditionFailed
from app.core.ids import new_id, utcnow
from app.store.db import Database

DESC = firestore.Query.DESCENDING
ASC = firestore.Query.ASCENDING


def strip_internal(doc: dict[str, Any]) -> dict[str, Any]:
    """Drop encrypted secrets and other internal fields before a document leaves the server."""
    return {k: v for k, v in doc.items() if not k.endswith("_enc") and not k.startswith("_")}


class Page:
    def __init__(self, items: list[dict[str, Any]], next_cursor: str | None) -> None:
        self.items = items
        self.next_cursor = next_cursor


async def _page(db: Database, col: Any, limit: int, cursor: str | None, order: str = "created_at",
                direction: str = DESC, where: tuple[str, str, Any] | None = None) -> Page:
    query = col
    if where is not None:
        query = query.where(filter=firestore.FieldFilter(*where))
    else:
        query = query.order_by(order, direction=direction)
    if cursor:
        snap = await db.run("cursor", col.document(cursor).get())
        if not snap.exists:
            raise BadRequest("invalid or expired cursor", code="invalid_cursor")
        query = query.start_after(snap)
    snaps = await db.query(query.limit(limit + 1), op="list")
    items = [s.to_dict() for s in snaps[:limit]]
    return Page(items, snaps[limit - 1].id if len(snaps) > limit else None)


class VersionedRepo:
    """CRUD with optimistic concurrency: every document carries an integer `version` that
    increments on each update; callers may pass `expected_version` (HTTP If-Match)."""

    collection = ""
    prefix = ""

    def __init__(self, db: Database) -> None:
        self.db = db

    def col(self, tenant_id: str) -> Any:
        return self.db.tcol(tenant_id, self.collection)

    async def create(self, tenant_id: str, data: dict[str, Any], doc_id: str | None = None) -> dict[str, Any]:
        now = utcnow()
        doc = {**data, "id": doc_id or new_id(self.prefix), "version": 1, "created_at": now, "updated_at": now}
        await self.db.run(f"{self.collection}.create", self.col(tenant_id).document(doc["id"]).create(doc))
        await self._after_write(tenant_id, doc, None)
        return doc

    async def find(self, tenant_id: str, doc_id: str) -> dict[str, Any] | None:
        return await self.db.get(self.col(tenant_id).document(doc_id), op=f"{self.collection}.get")

    async def get(self, tenant_id: str, doc_id: str) -> dict[str, Any]:
        doc = await self.find(tenant_id, doc_id)
        if doc is None:
            raise NotFound(f"{self.prefix} {doc_id} not found", code=f"{self.prefix}_not_found")
        return doc

    async def list(self, tenant_id: str, limit: int = 50, cursor: str | None = None) -> Page:
        return await _page(self.db, self.col(tenant_id), limit, cursor)

    async def update(self, tenant_id: str, doc_id: str, changes: dict[str, Any] | Callable[[dict], dict],
                     expected_version: int | None = None) -> dict[str, Any]:
        ref = self.col(tenant_id).document(doc_id)
        result: dict[str, Any] = {}

        @firestore.async_transactional
        async def txn(transaction: Any) -> None:
            snap = await ref.get(transaction=transaction)
            if not snap.exists:
                raise NotFound(f"{self.prefix} {doc_id} not found", code=f"{self.prefix}_not_found")
            current = snap.to_dict()
            if expected_version is not None and current.get("version") != expected_version:
                raise PreconditionFailed(
                    f"{self.prefix} {doc_id} is at version {current.get('version')}, not {expected_version}",
                    details={"current_version": current.get("version")})
            patch = changes(current) if callable(changes) else changes
            updated = {**current, **patch, "version": int(current.get("version", 0)) + 1, "updated_at": utcnow()}
            transaction.set(ref, updated)
            self._in_txn(transaction, tenant_id, updated, current)
            result.update(updated)

        await self.db.run(f"{self.collection}.update", txn(self.db.transaction()))
        return result

    async def delete(self, tenant_id: str, doc_id: str, expected_version: int | None = None) -> None:
        ref = self.col(tenant_id).document(doc_id)
        current = await self.get(tenant_id, doc_id)
        if expected_version is not None and current.get("version") != expected_version:
            raise PreconditionFailed(f"{self.prefix} {doc_id} is at version {current.get('version')}",
                                     details={"current_version": current.get("version")})
        await self.db.run(f"{self.collection}.delete", ref.delete())

    # hooks for subclasses that keep version history
    async def _after_write(self, tenant_id: str, doc: dict[str, Any], previous: dict[str, Any] | None) -> None:
        return None

    def _in_txn(self, transaction: Any, tenant_id: str, doc: dict[str, Any], previous: dict[str, Any]) -> None:
        return None


# ---------------------------------------------------------------------------- platform
class TenantRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def ref(self, tenant_id: str) -> Any:
        return self.db.tenant_doc(tenant_id)

    async def create(self, data: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
        now = utcnow()
        doc = {**data, "id": tenant_id or new_id("ten"), "status": "active", "version": 1,
               "created_at": now, "updated_at": now}
        await self.db.run("tenants.create", self.ref(doc["id"]).create(doc))
        return doc

    async def find(self, tenant_id: str) -> dict[str, Any] | None:
        return await self.db.get(self.ref(tenant_id), op="tenants.get")

    async def get(self, tenant_id: str) -> dict[str, Any]:
        doc = await self.find(tenant_id)
        if doc is None or doc.get("status") == "deleted":
            raise NotFound(f"tenant {tenant_id} not found", code="tenant_not_found")
        return doc

    async def list(self, limit: int, cursor: str | None) -> Page:
        return await _page(self.db, self.db.col("tenants"), limit, cursor)

    async def update(self, tenant_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        ref = self.ref(tenant_id)
        result: dict[str, Any] = {}

        @firestore.async_transactional
        async def txn(transaction: Any) -> None:
            snap = await ref.get(transaction=transaction)
            if not snap.exists:
                raise NotFound(f"tenant {tenant_id} not found", code="tenant_not_found")
            current = snap.to_dict()
            updated = {**current, **changes, "version": current.get("version", 0) + 1, "updated_at": utcnow()}
            transaction.set(ref, updated)
            result.update(updated)

        await self.db.run("tenants.update", txn(self.db.transaction()))
        return result


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class ApiKeyRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def col(self) -> Any:
        return self.db.col("api_keys")

    async def create(self, key: str, tenant_id: str, name: str, role: str, created_by: str,
                     expires_at: datetime | None) -> dict[str, Any]:
        doc = {"id": new_id("key"), "tenant_id": tenant_id, "name": name, "role": role, "prefix": key[:12],
               "created_at": utcnow(), "created_by": created_by, "expires_at": expires_at, "revoked_at": None}
        await self.db.run("api_keys.create", self.col().document(hash_key(key)).create(doc))
        return doc

    async def lookup(self, key_hash: str) -> dict[str, Any] | None:
        return await self.db.get(self.col().document(key_hash), op="api_keys.lookup")

    async def list(self, tenant_id: str) -> list[dict[str, Any]]:
        query = self.col().where(filter=firestore.FieldFilter("tenant_id", "==", tenant_id)).limit(500)
        return sorted((s.to_dict() for s in await self.db.query(query, op="api_keys.list")),
                      key=lambda d: d["created_at"], reverse=True)

    async def _by_id(self, tenant_id: str, key_id: str) -> Any:
        query = (self.col().where(filter=firestore.FieldFilter("id", "==", key_id)).limit(1))
        snaps = await self.db.query(query, op="api_keys.by_id")
        if not snaps or snaps[0].to_dict().get("tenant_id") != tenant_id:
            raise NotFound(f"API key {key_id} not found", code="api_key_not_found")
        return snaps[0]

    async def get(self, tenant_id: str, key_id: str) -> tuple[str, dict[str, Any]]:
        snap = await self._by_id(tenant_id, key_id)
        return snap.id, snap.to_dict()

    async def revoke(self, tenant_id: str, key_id: str) -> tuple[str, dict[str, Any]]:
        snap = await self._by_id(tenant_id, key_id)
        doc = snap.to_dict()
        if doc.get("revoked_at") is None:
            doc["revoked_at"] = utcnow()
            await self.db.run("api_keys.revoke", snap.reference.update({"revoked_at": doc["revoked_at"]}))
        return snap.id, doc

    async def revoke_all(self, tenant_id: str) -> list[str]:
        hashes = []
        for doc_hash, doc in [(s.id, s.to_dict()) for s in await self.db.query(
                self.col().where(filter=firestore.FieldFilter("tenant_id", "==", tenant_id)), op="api_keys.list")]:
            if doc.get("revoked_at") is None:
                await self.db.run("api_keys.revoke", self.col().document(doc_hash).update({"revoked_at": utcnow()}))
            hashes.append(doc_hash)
        return hashes

    async def touch(self, key_hash: str) -> None:
        await self.db.run("api_keys.touch", self.col().document(key_hash).update({"last_used_at": utcnow()}))


# ------------------------------------------------------------------------- resources
class AgentRepo(VersionedRepo):
    collection, prefix = "agents", "agent"

    def versions(self, tenant_id: str, agent_id: str) -> Any:
        return self.col(tenant_id).document(agent_id).collection("versions")

    async def _after_write(self, tenant_id: str, doc: dict[str, Any], previous: dict[str, Any] | None) -> None:
        await self.db.run("agents.snapshot", self.versions(tenant_id, doc["id"]).document(
            f"{doc['version']:06d}").set(doc))

    def _in_txn(self, transaction: Any, tenant_id: str, doc: dict[str, Any], previous: dict[str, Any]) -> None:
        transaction.set(self.versions(tenant_id, doc["id"]).document(f"{doc['version']:06d}"), doc)

    async def get_version(self, tenant_id: str, agent_id: str, version: int) -> dict[str, Any]:
        doc = await self.db.get(self.versions(tenant_id, agent_id).document(f"{version:06d}"), op="agents.version")
        if doc is None:
            raise NotFound(f"agent {agent_id} version {version} not found", code="agent_version_not_found")
        return doc

    async def list_versions(self, tenant_id: str, agent_id: str, limit: int, cursor: str | None) -> Page:
        return await _page(self.db, self.versions(tenant_id, agent_id), limit, cursor, order="version", direction=ASC)

    async def delete(self, tenant_id: str, doc_id: str, expected_version: int | None = None) -> None:
        # version snapshots stay: sessions pinned to them keep working until they end
        await super().delete(tenant_id, doc_id, expected_version)


class ToolRepo(VersionedRepo):
    collection, prefix = "tools", "tool"


class McpServerRepo(VersionedRepo):
    collection, prefix = "mcp_servers", "mcp"


class WorkflowRepo(VersionedRepo):
    collection, prefix = "workflows", "workflow"


# -------------------------------------------------------------------------- sessions
class SessionRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def ref(self, tenant_id: str, session_id: str) -> Any:
        return self.db.tcol(tenant_id, "sessions").document(session_id)

    def messages(self, tenant_id: str, session_id: str) -> Any:
        return self.ref(tenant_id, session_id).collection("messages")

    def responses(self, tenant_id: str, session_id: str) -> Any:
        return self.ref(tenant_id, session_id).collection("responses")

    async def create(self, tenant_id: str, doc: dict[str, Any]) -> None:
        await self.db.run("sessions.create", self.ref(tenant_id, doc["id"]).create(doc))

    async def find(self, tenant_id: str, session_id: str) -> dict[str, Any] | None:
        return await self.db.get(self.ref(tenant_id, session_id), op="sessions.get")

    async def save(self, tenant_id: str, session_id: str, fields: dict[str, Any]) -> None:
        await self.db.run("sessions.save", self.ref(tenant_id, session_id).set(fields, merge=True))

    async def list(self, tenant_id: str, limit: int, cursor: str | None) -> Page:
        return await _page(self.db, self.db.tcol(tenant_id, "sessions"), limit, cursor)

    async def load_messages(self, tenant_id: str, session_id: str, after_seq: int = 0,
                            limit: int | None = None) -> list[dict[str, Any]]:
        query = self.messages(tenant_id, session_id).order_by("seq", direction=ASC)
        if after_seq:
            query = query.start_after({"seq": after_seq})
        if limit:
            query = query.limit(limit)
        return [s.to_dict() for s in await self.db.query(query, op="messages.load")]

    async def put_message(self, tenant_id: str, session_id: str, message: dict[str, Any]) -> None:
        await self.db.run("messages.put", self.messages(tenant_id, session_id).document(
            f"{message['seq']:08d}").set(message))

    async def delete_messages(self, tenant_id: str, session_id: str, seqs: list[int] | None = None) -> None:
        col = self.messages(tenant_id, session_id)
        if seqs is None:
            seqs = [s.to_dict()["seq"] for s in await self.db.query(col.select(["seq"]), op="messages.ids")]
        for start in range(0, len(seqs), 400):
            batch = self.db.client.batch()
            for seq in seqs[start:start + 400]:
                batch.delete(col.document(f"{seq:08d}"))
            await self.db.run("messages.delete", batch.commit())

    async def put_response(self, tenant_id: str, session_id: str, response: dict[str, Any]) -> None:
        await self.db.run("responses.put", self.responses(tenant_id, session_id).document(response["id"]).set(response))

    async def list_responses(self, tenant_id: str, session_id: str, limit: int, cursor: str | None) -> Page:
        return await _page(self.db, self.responses(tenant_id, session_id), limit, cursor)

    async def get_response(self, tenant_id: str, session_id: str, response_id: str) -> dict[str, Any]:
        doc = await self.db.get(self.responses(tenant_id, session_id).document(response_id), op="responses.get")
        if doc is None:
            raise NotFound(f"response {response_id} not found", code="response_not_found")
        return doc

    async def delete(self, tenant_id: str, session_id: str) -> None:
        await self.delete_messages(tenant_id, session_id)
        col = self.responses(tenant_id, session_id)
        ids = [s.id for s in await self.db.query(col.select([]), op="responses.ids")]
        for start in range(0, len(ids), 400):
            batch = self.db.client.batch()
            for rid in ids[start:start + 400]:
                batch.delete(col.document(rid))
            await self.db.run("responses.delete", batch.commit())
        await self.db.run("sessions.delete", self.ref(tenant_id, session_id).delete())

    async def acquire_lease(self, tenant_id: str, session_id: str, owner: str, lease_s: float) -> dict[str, Any]:
        """Take (or renew) the session's ownership lease. Raises Conflict while another live
        instance holds it, NotFound if the session does not exist. Returns the session document."""
        ref = self.ref(tenant_id, session_id)
        result: dict[str, Any] = {}

        @firestore.async_transactional
        async def txn(transaction: Any) -> None:
            snap = await ref.get(transaction=transaction)
            if not snap.exists:
                raise NotFound(f"session {session_id} not found", code="session_not_found")
            doc = snap.to_dict()
            lease = doc.get("lease") or {}
            now = utcnow()
            if lease.get("owner") not in (None, owner) and lease.get("expires_at") and lease["expires_at"] > now:
                raise Conflict(f"session {session_id} is active on another server instance",
                               code="session_busy_elsewhere", retry_after=2,
                               details={"lease_expires_at": lease["expires_at"].isoformat()})
            doc["lease"] = {"owner": owner, "expires_at": now + timedelta(seconds=lease_s)}
            transaction.update(ref, {"lease": doc["lease"]})
            result.update(doc)

        await self.db.run("sessions.lease", txn(self.db.transaction()))
        return result

    async def release_lease(self, tenant_id: str, session_id: str, owner: str) -> None:
        ref = self.ref(tenant_id, session_id)

        @firestore.async_transactional
        async def txn(transaction: Any) -> None:
            snap = await ref.get(transaction=transaction)
            if snap.exists and (snap.to_dict().get("lease") or {}).get("owner") == owner:
                transaction.update(ref, {"lease": None})

        await self.db.run("sessions.release", txn(self.db.transaction()))


# ------------------------------------------------------------------------ tool calls
class ToolCallRepo:
    """One record per executed tool call, keyed by (scope, call id, fingerprint of name + arguments):
    a replay of the same call finds its record; a different call that reuses an id does not."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def ref(self, tenant_id: str, record: dict[str, Any]) -> Any:
        doc_id = f"{record['scope_id']}__{record['call_id']}__{record['fingerprint']}"[:1400]
        return self.db.tcol(tenant_id, "tool_calls").document(doc_id)

    async def claim(self, tenant_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
        """Create the record; if one already exists return it instead (idempotent execution)."""
        ref = self.ref(tenant_id, record)
        try:
            await self.db.run("tool_calls.claim", ref.create(record))
            return None
        except Conflict:
            return await self.db.get(ref, op="tool_calls.get")

    async def put(self, tenant_id: str, record: dict[str, Any]) -> None:
        await self.db.run("tool_calls.put", self.ref(tenant_id, record).set(record, merge=True))

    async def get(self, tenant_id: str, record: dict[str, Any]) -> dict[str, Any] | None:
        return await self.db.get(self.ref(tenant_id, record), op="tool_calls.get")

    async def list_scope(self, tenant_id: str, scope_id: str, limit: int) -> list[dict[str, Any]]:
        query = self.db.tcol(tenant_id, "tool_calls").where(
            filter=firestore.FieldFilter("scope_id", "==", scope_id)).limit(limit)
        return sorted((s.to_dict() for s in await self.db.query(query, op="tool_calls.list")),
                      key=lambda d: d.get("started_at") or utcnow())


# ------------------------------------------------------------------------ executions
class ExecutionRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def ref(self, tenant_id: str, execution_id: str) -> Any:
        return self.db.tcol(tenant_id, "executions").document(execution_id)

    def queue_ref(self, execution_id: str) -> Any:
        return self.db.col("execution_queue").document(execution_id)

    async def create(self, tenant_id: str, doc: dict[str, Any]) -> None:
        batch = self.db.client.batch()
        batch.create(self.ref(tenant_id, doc["id"]), doc)
        batch.create(self.queue_ref(doc["id"]), {"tenant_id": tenant_id, "status": "queued", "owner": None,
                                                 "lease_expires_at": None, "created_at": doc["created_at"]})
        await self.db.run("executions.create", batch.commit())

    async def get(self, tenant_id: str, execution_id: str) -> dict[str, Any]:
        doc = await self.db.get(self.ref(tenant_id, execution_id), op="executions.get")
        if doc is None:
            raise NotFound(f"execution {execution_id} not found", code="execution_not_found")
        return doc

    async def list(self, tenant_id: str, limit: int, cursor: str | None) -> Page:
        return await _page(self.db, self.db.tcol(tenant_id, "executions"), limit, cursor)

    async def save(self, tenant_id: str, execution_id: str, fields: dict[str, Any]) -> None:
        await self.db.run("executions.save", self.ref(tenant_id, execution_id).set(fields, merge=True))

    async def claim(self, execution_id: str, owner: str, lease_s: float, max_attempts: int) -> str | None:
        """Claim a queued (or lease-expired) execution. Returns its tenant id, or None."""
        qref = self.queue_ref(execution_id)
        claimed: dict[str, Any] = {}

        @firestore.async_transactional
        async def txn(transaction: Any) -> None:
            snap = await qref.get(transaction=transaction)
            if not snap.exists:
                return
            item = snap.to_dict()
            now = utcnow()
            expired = item.get("lease_expires_at") is not None and item["lease_expires_at"] < now
            if item["status"] == "running" and not expired:
                return
            attempts = int(item.get("attempts", 0)) + 1
            if attempts > max_attempts:
                transaction.delete(qref)
                claimed["exhausted"] = item["tenant_id"]
                return
            transaction.update(qref, {"status": "running", "owner": owner, "attempts": attempts,
                                      "lease_expires_at": now + timedelta(seconds=lease_s)})
            claimed["tenant_id"] = item["tenant_id"]
            claimed["attempts"] = attempts

        await self.db.run("execution_queue.claim", txn(self.db.transaction()))
        if "exhausted" in claimed:
            await self.save(claimed["exhausted"], execution_id, {
                "status": "failed", "finished_at": utcnow(),
                "error": {"code": "max_attempts_exceeded", "message": "execution was interrupted too many times"}})
            return None
        return claimed.get("tenant_id")

    async def renew(self, execution_id: str, owner: str, lease_s: float) -> bool:
        """Extend our lease; False if it lapsed and another instance claimed the execution."""
        qref = self.queue_ref(execution_id)
        renewed = False

        @firestore.async_transactional
        async def txn(transaction: Any) -> None:
            nonlocal renewed
            snap = await qref.get(transaction=transaction)
            if snap.exists and snap.to_dict().get("owner") == owner:
                transaction.update(qref, {"lease_expires_at": utcnow() + timedelta(seconds=lease_s)})
                renewed = True

        await self.db.run("execution_queue.renew", txn(self.db.transaction()))
        return renewed

    async def release(self, execution_id: str, owner: str, requeue: bool) -> None:
        """Requeue or remove the queue entry, only while we still own it."""
        qref = self.queue_ref(execution_id)

        @firestore.async_transactional
        async def txn(transaction: Any) -> None:
            snap = await qref.get(transaction=transaction)
            if not snap.exists or snap.to_dict().get("owner") not in (None, owner):
                return
            if requeue:
                transaction.update(qref, {"status": "queued", "owner": None, "lease_expires_at": None})
            else:
                transaction.delete(qref)

        await self.db.run("execution_queue.release", txn(self.db.transaction()))

    async def claimable(self, limit: int = 20) -> list[str]:
        col = self.db.col("execution_queue")
        queued = await self.db.query(col.where(filter=firestore.FieldFilter("status", "==", "queued")).limit(limit),
                                     op="execution_queue.scan")
        expired = await self.db.query(col.where(filter=firestore.FieldFilter("lease_expires_at", "<", utcnow()))
                                      .limit(limit), op="execution_queue.scan")
        return list(dict.fromkeys([s.id for s in queued] + [s.id for s in expired]))


# ----------------------------------------------------------------------- idempotency
class IdempotencyRepo:
    TTL = timedelta(hours=24)

    def __init__(self, db: Database) -> None:
        self.db = db

    def ref(self, tenant_id: str, key_hash: str) -> Any:
        return self.db.tcol(tenant_id, "idempotency").document(key_hash)

    async def begin(self, tenant_id: str, key_hash: str, request_hash: str) -> dict[str, Any] | None:
        """Reserve the key. Returns None when reserved now, else the existing record."""
        now = utcnow()
        record = {"status": "in_progress", "request_hash": request_hash, "created_at": now,
                  "expires_at": now + self.TTL}
        try:
            await self.db.run("idempotency.begin", self.ref(tenant_id, key_hash).create(record))
            return None
        except Conflict:
            existing = await self.db.get(self.ref(tenant_id, key_hash), op="idempotency.get")
            if existing is None or existing.get("expires_at", now) < now:
                await self.db.run("idempotency.reset", self.ref(tenant_id, key_hash).set(record))
                return None
            return existing

    async def complete(self, tenant_id: str, key_hash: str, status: int, body_enc: str, media_type: str) -> None:
        await self.db.run("idempotency.complete", self.ref(tenant_id, key_hash).update(
            {"status": "completed", "response_status": status, "response_body_enc": body_enc,
             "media_type": media_type}))

    async def abandon(self, tenant_id: str, key_hash: str) -> None:
        await self.db.run("idempotency.abandon", self.ref(tenant_id, key_hash).delete())


# ----------------------------------------------------------------------------- usage
class UsageRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def ref(self, tenant_id: str, day: str) -> Any:
        return self.db.tcol(tenant_id, "usage_daily").document(day)

    async def increment(self, tenant_id: str, day: str, fields: dict[str, Any]) -> None:
        await self.db.run("usage.increment", self.ref(tenant_id, day).set(fields, merge=True))

    async def range(self, tenant_id: str, start: str, end: str) -> list[dict[str, Any]]:
        col = self.db.tcol(tenant_id, "usage_daily")
        query = (col.where(filter=firestore.FieldFilter("date", ">=", start))
                 .where(filter=firestore.FieldFilter("date", "<=", end)).order_by("date"))
        return [s.to_dict() for s in await self.db.query(query, op="usage.range")]


def conflict_if_exists(doc: dict[str, Any] | None, what: str) -> None:
    if doc is not None:
        raise Conflict(f"{what} already exists", code="already_exists")
