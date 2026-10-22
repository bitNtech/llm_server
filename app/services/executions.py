"""Asynchronous executions: agent runs and workflows that outlive the HTTP request.

Lifecycle: queued -> running -> completed | failed | cancelled.

Work distribution uses Firestore as the queue: `execution_queue/{id}` is claimed in a transaction
(owner + lease). The owner renews the lease while it works; if an instance dies, the lease expires
and another instance's scanner claims the execution again (up to EXECUTION_MAX_ATTEMPTS). Workflow
step results are persisted as they complete, so a resumed workflow continues where it stopped and
tool steps are idempotent per execution. On graceful shutdown unfinished executions are re-queued.

Progress is observable through `GET /v1/executions/{id}` and the SSE stream
`GET /v1/executions/{id}/events`; a `callback_url` receives a signed webhook when it finishes.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from app.config import Settings
from app.core.errors import BadRequest, Conflict, GatewayError, ProviderTimeout
from app.core.ids import iso, jsonable, new_id, utcnow
from app.core.logging import get_logger, tenant_id_var
from app.core.metrics import EXECUTIONS, EXECUTIONS_RUNNING
from app.core.tracing import span
from app.engine.agents import AgentResolver
from app.engine.runtime import ConversationRuntime
from app.engine.sessions import LiveSession, SessionManager
from app.security.netguard import NetGuard
from app.services.webhooks import WebhookSender
from app.services.workflows import WorkflowRunner, run_agent_turn
from app.store.repos import ExecutionRepo, TenantRepo, WorkflowRepo
from app.store.writer import Writer

log = get_logger("app.executions")
TERMINAL = {"completed", "failed", "cancelled"}


class LiveExecution:
    def __init__(self, execution_id: str, tenant_id: str) -> None:
        self.id = execution_id
        self.tenant_id = tenant_id
        self.task: asyncio.Task | None = None
        self.events: list[dict[str, Any]] = []
        self.changed = asyncio.Condition()
        self.done = False
        self.shutdown = False  # graceful shutdown: re-queue instead of finalizing
        self.lease_lost = False  # another instance owns it now: stop without touching it

    def emit(self, kind: str, data: dict[str, Any]) -> None:
        self.events.append({"type": kind, "seq": len(self.events) + 1, "ts": iso(utcnow()), **jsonable(data)})
        asyncio.get_running_loop().create_task(self._notify())

    async def _notify(self) -> None:
        async with self.changed:
            self.changed.notify_all()


class ExecutionService:
    def __init__(self, settings: Settings, repo: ExecutionRepo, workflows: WorkflowRepo, tenants: TenantRepo,
                 runtime: ConversationRuntime, runner: WorkflowRunner, resolver: AgentResolver,
                 sessions: SessionManager, webhooks: WebhookSender, guard: NetGuard, writer: Writer) -> None:
        self.settings = settings
        self.repo = repo
        self.workflows = workflows
        self.tenants = tenants
        self.runtime = runtime
        self.runner = runner
        self.resolver = resolver
        self.sessions = sessions
        self.webhooks = webhooks
        self.guard = guard
        self.writer = writer
        self.owner = settings.instance_id
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.running: dict[str, LiveExecution] = {}
        self._tasks: list[asyncio.Task] = []
        self.accepting = True

    # ---------------------------------------------------------------- submit
    async def submit(self, tenant_id: str, spec: dict[str, Any], created_by: str) -> dict[str, Any]:
        kind = spec["type"]
        doc: dict[str, Any] = {
            "id": new_id("exec"), "type": kind, "status": "queued", "input": spec.get("input"),
            "metadata": spec.get("metadata") or {}, "callback_url": spec.get("callback_url"),
            "created_at": utcnow(), "created_by": created_by, "attempts": 0, "steps": {},
            "result": None, "error": None, "cancel_requested": False,
        }
        if doc["callback_url"]:
            self.guard.check_syntax(doc["callback_url"], "callback_url")
        if kind == "agent":
            agent = await self.resolver.resolve(tenant_id, spec["agent_id"], spec.get("agent_version"))
            doc.update(agent_id=agent.id, agent_version=agent.version, session_id=spec.get("session_id"),
                       response_format=spec.get("response_format"), params=spec.get("params"))
            if doc["input"] is None:
                raise BadRequest("agent executions need input")
        elif kind == "workflow":
            workflow = await self.workflows.get(tenant_id, spec["workflow_id"])
            doc.update(workflow_id=workflow["id"], workflow_version=workflow["version"],
                       workflow={k: workflow.get(k) for k in ("name", "steps", "output")})
        else:
            raise BadRequest("type must be agent or workflow")
        await self.repo.create(tenant_id, doc)
        if self.settings.execution_workers > 0 and self.accepting:
            self.queue.put_nowait(doc["id"])
        return doc

    # ---------------------------------------------------------------- workers
    def start(self) -> None:
        if self.settings.execution_workers <= 0:
            return
        for i in range(self.settings.execution_workers):
            self._tasks.append(asyncio.create_task(self._worker(), name=f"exec-worker-{i}"))
        self._tasks.append(asyncio.create_task(self._scanner(), name="exec-scanner"))

    async def _scanner(self) -> None:
        while self.accepting:
            try:
                for execution_id in await self.repo.claimable():
                    if execution_id not in self.running:
                        self.queue.put_nowait(execution_id)
            except GatewayError as exc:
                log.warning("execution scan failed", extra={"error": exc.message})
            await asyncio.sleep(5.0)

    async def _worker(self) -> None:
        while True:
            execution_id = await self.queue.get()
            if not self.accepting or execution_id in self.running:
                continue
            try:
                tenant_id = await self.repo.claim(execution_id, self.owner, self.settings.execution_lease_s,
                                                  self.settings.execution_max_attempts)
                if tenant_id is not None:
                    await self._execute(tenant_id, execution_id)
            except GatewayError as exc:
                log.warning("execution claim/run failed", extra={"execution_id": execution_id, "error": exc.message})
            except Exception:
                log.exception("execution worker error", extra={"execution_id": execution_id})

    def _save(self, tenant_id: str, execution_id: str, fields: dict[str, Any]) -> None:
        self.writer.submit(f"exec/{execution_id}", lambda: self.repo.save(tenant_id, execution_id, fields), "execution")

    async def _execute(self, tenant_id: str, execution_id: str) -> None:
        tenant_id_var.set(tenant_id)
        doc = await self.repo.get(tenant_id, execution_id)
        if doc["status"] in TERMINAL:
            await self.repo.release(execution_id, self.owner, requeue=False)
            return
        live = LiveExecution(execution_id, tenant_id)
        self.running[execution_id] = live
        EXECUTIONS_RUNNING.set(len(self.running))
        if doc.get("cancel_requested"):
            await self._finish(doc, live, "cancelled", None, None)
            return
        attempts = int(doc.get("attempts") or 0) + 1
        self._save(tenant_id, execution_id, {"status": "running", "started_at": utcnow(), "attempts": attempts,
                                             "owner": self.owner})
        live.emit("execution.started", {"execution_id": execution_id, "attempt": attempts})

        async def guarded() -> Any:
            with span("execution", **{"execution.id": execution_id, "execution.type": doc["type"]}):
                async with asyncio.timeout(self.settings.execution_timeout_s):
                    return await self._run(tenant_id, doc, live)

        # own task: cancelling the execution must not cancel the worker loop (and vice versa)
        live.task = asyncio.create_task(guarded(), name=f"exec-{execution_id}")
        heartbeat = asyncio.create_task(self._heartbeat(live), name=f"exec-heartbeat-{execution_id}")
        status, result, error = "completed", None, None
        started = time.perf_counter()
        try:
            await asyncio.wait([live.task])
            result = live.task.result()
        except asyncio.CancelledError:
            if live.lease_lost:
                heartbeat.cancel()
                self.running.pop(execution_id, None)
                EXECUTIONS_RUNNING.set(len(self.running))
                live.done = True
                return
            if live.shutdown:
                heartbeat.cancel()
                self._save(tenant_id, execution_id, {"status": "queued", "owner": None})
                await self.writer.flush(f"exec/{execution_id}", timeout=5)
                await self.repo.release(execution_id, self.owner, requeue=True)
                self.running.pop(execution_id, None)
                EXECUTIONS_RUNNING.set(len(self.running))
                live.done = True
                return
            status = "cancelled"
        except TimeoutError:
            status, error = "failed", ProviderTimeout(
                f"execution exceeded EXECUTION_TIMEOUT_S={self.settings.execution_timeout_s}").to_dict()["error"]
        except GatewayError as exc:
            status, error = "failed", exc.to_dict()["error"]
        except Exception as exc:
            log.exception("execution crashed", extra={"execution_id": execution_id})
            status, error = "failed", GatewayError(f"internal error: {exc.__class__.__name__}").to_dict()["error"]
        finally:
            heartbeat.cancel()
        log.info("execution finished", extra={"execution_id": execution_id, "status": status,
                                               "seconds": round(time.perf_counter() - started, 2)})
        await self._finish(doc, live, status, result, error)

    async def _finish(self, doc: dict[str, Any], live: LiveExecution, status: str, result: Any,
                      error: dict[str, Any] | None) -> None:
        tenant_id, execution_id = live.tenant_id, live.id
        fields = {"status": status, "result": jsonable(result), "error": error, "finished_at": utcnow(), "owner": None}
        self._save(tenant_id, execution_id, fields)
        await self.writer.flush(f"exec/{execution_id}", timeout=10)
        await self.repo.release(execution_id, self.owner, requeue=False)
        EXECUTIONS.labels(doc["type"], status).inc()
        live.emit(f"execution.{status}", {"execution_id": execution_id, "result": result, "error": error})
        live.done = True
        await live._notify()
        self.running.pop(execution_id, None)
        EXECUTIONS_RUNNING.set(len(self.running))
        if doc.get("callback_url"):
            tenant = await self.tenants.find(tenant_id) or {}
            payload = {"id": execution_id, "type": doc["type"], "status": status, "result": jsonable(result),
                       "error": error, "metadata": doc.get("metadata"), "agent_id": doc.get("agent_id"),
                       "workflow_id": doc.get("workflow_id")}

            async def record(outcome: dict[str, Any]) -> None:
                await self.repo.save(tenant_id, execution_id, {"callback": outcome})

            self.webhooks.send(doc["callback_url"], f"execution.{status}", payload,
                               tenant.get("webhook_secret_enc"), record)

    async def _heartbeat(self, live: LiveExecution) -> None:
        interval = self.settings.execution_lease_s / 3
        while True:
            await asyncio.sleep(interval)
            try:
                if not await self.repo.renew(live.id, self.owner, self.settings.execution_lease_s):
                    log.warning("execution lease lost; stopping", extra={"execution_id": live.id})
                    live.lease_lost = True
                    if live.task:
                        live.task.cancel()
                    return
                doc = await self.repo.get(live.tenant_id, live.id)
                if doc.get("cancel_requested") and live.task:
                    live.task.cancel()
                    return
            except GatewayError as exc:
                log.warning("execution heartbeat failed", extra={"execution_id": live.id, "error": exc.message})

    async def _run(self, tenant_id: str, doc: dict[str, Any], live: LiveExecution) -> Any:
        if doc["type"] == "agent":
            if doc.get("session_id"):
                session = await self.sessions.get(tenant_id, doc["session_id"])
            else:
                session = LiveSession(tenant_id=tenant_id, id=doc["id"], agent_id=doc["agent_id"],
                                      agent_version=doc["agent_version"], current_agent_id=doc["agent_id"],
                                      current_agent_version=doc["agent_version"], channel="execution",
                                      chars_per_token=self.settings.context_chars_per_token)
            result = await run_agent_turn(self.runtime, session, doc["input"], api="execution",
                                          params=doc.get("params"), response_format=doc.get("response_format"))
            live.emit("agent.response", {"text": result["text"], "status": result["status"]})
            return result

        def emit(kind: str, data: dict[str, Any]) -> None:
            live.emit(kind, data)
            if kind in ("step.completed", "step.failed"):
                step = {"status": "completed" if kind == "step.completed" else "failed",
                        "output": jsonable(data.get("output")), "error": data.get("error"), "at": utcnow()}
                self._save(tenant_id, doc["id"], {"steps": {data["step_id"]: step}})

        completed = {sid: step for sid, step in (doc.get("steps") or {}).items() if step.get("status") == "completed"}
        return await self.runner.run(tenant_id, doc["id"], doc["workflow"], doc.get("input"), completed, emit)

    # ------------------------------------------------------------------ control
    async def cancel(self, tenant_id: str, execution_id: str) -> dict[str, Any]:
        doc = await self.repo.get(tenant_id, execution_id)
        if doc["status"] in TERMINAL:
            raise Conflict(f"execution {execution_id} already {doc['status']}", code="execution_finished")
        live = self.running.get(execution_id)
        if live is not None and live.task is not None:
            live.task.cancel()
        else:
            await self.repo.save(tenant_id, execution_id, {"cancel_requested": True})
            if doc["status"] == "queued":
                claimed = await self.repo.claim(execution_id, self.owner, self.settings.execution_lease_s,
                                                self.settings.execution_max_attempts + 1)
                if claimed is not None:
                    await self._finish(doc, LiveExecution(execution_id, tenant_id), "cancelled", None, None)
        return {**doc, "cancel_requested": True}

    async def events(self, tenant_id: str, execution_id: str) -> AsyncIterator[dict[str, Any]]:
        live = self.running.get(execution_id)
        if live is not None and live.tenant_id == tenant_id:
            sent = 0
            while True:
                while sent < len(live.events):
                    yield live.events[sent]
                    sent += 1
                if live.done:
                    return
                async with live.changed:
                    try:
                        await asyncio.wait_for(live.changed.wait(), 15)
                    except TimeoutError:
                        yield {"type": "keepalive"}
        # not running here: follow the stored document
        last_status = None
        seen_steps: set[str] = set()
        while True:
            doc = await self.repo.get(tenant_id, execution_id)
            for step_id, step in (doc.get("steps") or {}).items():
                if step_id not in seen_steps:
                    seen_steps.add(step_id)
                    yield {"type": f"step.{step.get('status')}", "step_id": step_id, **jsonable(step)}
            if doc["status"] != last_status:
                last_status = doc["status"]
                yield {"type": f"execution.{doc['status']}", "execution_id": execution_id,
                       "result": jsonable(doc.get("result")), "error": doc.get("error")}
            if doc["status"] in TERMINAL:
                return
            await asyncio.sleep(1.0)

    async def drain(self, grace_s: float) -> None:
        """Stop taking work, let running executions finish within the grace period, then re-queue
        the rest so another instance resumes them."""
        self.accepting = False
        if self._tasks:
            self._tasks[-1].cancel()  # scanner
        running = [live for live in self.running.values() if live.task is not None]
        if running:
            deadline = time.monotonic() + grace_s
            while any(not live.done for live in running) and time.monotonic() < deadline:
                await asyncio.sleep(0.1)
            for live in running:
                if not live.done and live.task is not None:
                    live.shutdown = True
                    live.task.cancel()
            for _ in range(50):
                if all(live.done for live in running):
                    break
                await asyncio.sleep(0.1)
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
