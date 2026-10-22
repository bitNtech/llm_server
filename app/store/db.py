"""Firestore access (named database, default `llm_server`).

Layout (every tenant-owned document lives under its tenant, which is what enforces isolation:
a repository can only build paths below `tenants/{tenant_id}` for the authenticated tenant):

    tenants/{tenant}                                 tenant
      agents/{agent}  + versions/{000001}            agents and immutable version snapshots
      tools/{tool}                                   HTTP / client tool definitions
      mcp_servers/{server}                           MCP server registrations
      workflows/{workflow}                           workflow definitions
      sessions/{session} + messages/{seq} + responses/{response}
      tool_calls/{scope}__{call}                     tool execution records (idempotency + audit)
      executions/{execution}                         async executions
      idempotency/{hash}                             Idempotency-Key replay records
      usage_daily/{YYYY-MM-DD}                       usage counters
    api_keys/{sha256(key)}                           API keys (looked up by hash on every auth cache miss)
    execution_queue/{execution}                      claimable work items (leases)

FIRESTORE_COLLECTION_PREFIX prefixes the root collections so several environments (or test runs)
can share one database.
"""
from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable
from typing import Any, TypeVar

from google.api_core import exceptions as gexc
from google.cloud import firestore
from google.cloud.firestore_v1.async_collection import AsyncCollectionReference
from google.cloud.firestore_v1.async_document import AsyncDocumentReference

from app.config import Settings
from app.core.errors import Conflict, GatewayError, NotFound, PreconditionFailed, StoreUnavailable
from app.core.logging import get_logger
from app.core.metrics import STORE_ERRORS, STORE_OPS

log = get_logger("app.store")
T = TypeVar("T")

TRANSIENT = (gexc.ServiceUnavailable, gexc.DeadlineExceeded, gexc.InternalServerError, gexc.Aborted,
             gexc.TooManyRequests, gexc.GatewayTimeout, gexc.BadGateway)


class Database:
    def __init__(self, settings: Settings, credentials: Any = None) -> None:
        self.settings = settings
        self.prefix = settings.firestore_collection_prefix
        self.timeout = settings.firestore_timeout_s
        kwargs: dict[str, Any] = {"project": settings.firestore_project or "local-dev",
                                  "database": settings.firestore_database}
        self.emulator = os.environ.get("FIRESTORE_EMULATOR_HOST")
        if credentials is not None and not self.emulator:
            kwargs["credentials"] = credentials
        self.client = firestore.AsyncClient(**kwargs)
        self.healthy = False
        self.last_error: str | None = "not checked yet"
        self.last_ok_at: float | None = None
        self._task: asyncio.Task | None = None

    # ---------------------------------------------------------------- paths
    def col(self, name: str) -> AsyncCollectionReference:
        return self.client.collection(self.prefix + name)

    def tenant_doc(self, tenant_id: str) -> AsyncDocumentReference:
        return self.col("tenants").document(tenant_id)

    def tcol(self, tenant_id: str, name: str) -> AsyncCollectionReference:
        return self.tenant_doc(tenant_id).collection(name)

    def transaction(self) -> Any:
        return self.client.transaction()

    # ------------------------------------------------------------ execution
    async def run(self, op: str, awaitable: Awaitable[T]) -> T:
        """Await a Firestore call with a deadline, metrics and error mapping."""
        start = time.perf_counter()
        try:
            return await asyncio.wait_for(awaitable, self.timeout)
        except gexc.AlreadyExists as exc:
            raise Conflict("resource already exists", code="already_exists") from exc
        except gexc.NotFound as exc:
            raise NotFound("resource not found") from exc
        except gexc.FailedPrecondition as exc:
            STORE_ERRORS.labels(op).inc()
            raise PreconditionFailed(f"store precondition failed: {exc.message}") from exc
        except TimeoutError as exc:
            STORE_ERRORS.labels(op).inc()
            raise StoreUnavailable(f"Firestore did not answer within {self.timeout:.0f}s ({op})") from exc
        except gexc.GoogleAPICallError as exc:
            STORE_ERRORS.labels(op).inc()
            log.warning("firestore error", extra={"op": op, "error": str(exc)[:300]})
            raise StoreUnavailable(f"Firestore error during {op}: {exc.__class__.__name__}") from exc
        finally:
            STORE_OPS.labels(op).observe(time.perf_counter() - start)

    async def get(self, ref: AsyncDocumentReference, op: str = "get") -> dict[str, Any] | None:
        snap = await self.run(op, ref.get())
        return snap.to_dict() if snap.exists else None

    async def query(self, query: Any, op: str = "query") -> list[Any]:
        async def collect() -> list[Any]:
            return [snap async for snap in query.stream()]
        return await self.run(op, collect())

    # --------------------------------------------------------------- health
    async def ping(self) -> bool:
        try:
            await self.run("ping", self.col("_health").document("ping").get())
        except GatewayError as exc:
            self.healthy, self.last_error = False, exc.message
            return False
        self.healthy, self.last_error, self.last_ok_at = True, None, time.monotonic()
        return True

    def start(self) -> None:
        self._task = asyncio.create_task(self._health_loop(), name="firestore-health")

    async def _health_loop(self) -> None:
        while True:
            ok = await self.ping()
            await asyncio.sleep(15.0 if ok else 2.0)

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.client.close()

    def snapshot(self) -> dict[str, Any]:
        return {
            "database": self.settings.firestore_database,
            "project": self.settings.firestore_project or None,
            "emulator": bool(self.emulator),
            "healthy": self.healthy,
            "last_error": self.last_error,
            "last_ok_age_s": round(time.monotonic() - self.last_ok_at, 1) if self.last_ok_at else None,
        }
