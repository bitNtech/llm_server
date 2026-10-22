"""Server-side tool execution: argument validation, dispatch, timeouts, idempotency, audit.

Idempotency: a tool call is identified by (tenant, scope, call_id), where scope is the session,
execution or chat request it belongs to.
  * every executed call is recorded in `tool_calls` (write-behind for read-only tools)
  * tools with side effects are claimed synchronously *before* they run; a replay of the same
    call (client retry, resumed execution, duplicate delivery) returns the stored result instead
    of running the side effect twice, and a concurrent duplicate waits for the first one
Validation failures and tool errors never raise: they become an error result the model sees, so
it can correct its arguments or tell the caller what went wrong.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import time
from typing import Any

import orjson

from app.config import Settings
from app.core.errors import GatewayError
from app.core.ids import utcnow
from app.core.logging import get_logger
from app.core.metrics import TOOL_CALLS, TOOL_LATENCY
from app.core.tracing import span
from app.store.repos import McpServerRepo, ToolCallRepo
from app.store.writer import Writer
from app.tools import builtin
from app.tools.base import ArgumentValidator, ToolContext, ToolOutcome, ToolSpec, parse_arguments, render_output
from app.tools.http import HttpToolRunner
from app.tools.mcp import MCPManager

log = get_logger("app.tools")


class ToolExecutor:
    def __init__(self, settings: Settings, calls: ToolCallRepo, writer: Writer, http: HttpToolRunner,
                 mcp: MCPManager, mcp_servers: McpServerRepo) -> None:
        self.settings = settings
        self.calls = calls
        self.writer = writer
        self.http = http
        self.mcp = mcp
        self.mcp_servers = mcp_servers
        self.validator = ArgumentValidator()

    async def execute(self, ctx: ToolContext, spec: ToolSpec, call_id: str, arguments: str) -> ToolOutcome:
        start = time.perf_counter()
        with span("tool.execute", **{"tool.name": spec.name, "tool.kind": spec.kind, "tool.call_id": call_id}) as sp:
            outcome = await self._execute(ctx, spec, call_id, arguments)
            outcome.duration_ms = round((time.perf_counter() - start) * 1000, 1)
            sp.set_attribute("tool.is_error", outcome.is_error)
            sp.set_attribute("tool.replayed", outcome.replayed)
        status = "replayed" if outcome.replayed else ("error" if outcome.is_error else "ok")
        TOOL_CALLS.labels(spec.kind, status).inc()
        TOOL_LATENCY.labels(spec.kind).observe(time.perf_counter() - start)
        return outcome

    async def _execute(self, ctx: ToolContext, spec: ToolSpec, call_id: str, arguments: str) -> ToolOutcome:
        try:
            args = parse_arguments(arguments)
        except (orjson.JSONDecodeError, TypeError) as exc:
            return ToolOutcome(f"invalid arguments for {spec.name}: {exc}. Send a JSON object.", True)
        problem = self.validator.validate(spec.parameters, args)
        if problem:
            return ToolOutcome(f"invalid arguments for {spec.name}: {problem}", True)

        fingerprint = hashlib.sha256(spec.name.encode() + b"\x1f" + orjson.dumps(
            args, option=orjson.OPT_SORT_KEYS)).hexdigest()[:16]
        record = {"scope_id": ctx.scope_id, "call_id": call_id, "fingerprint": fingerprint, "name": spec.name,
                  "kind": spec.kind, "arguments": args, "agent_id": ctx.agent_id, "session_id": ctx.session_id,
                  "side_effects": spec.side_effects, "status": "running", "started_at": utcnow(),
                  "target": {k: v for k, v in spec.target.items() if not k.endswith("_enc") and k != "http"}}
        if spec.side_effects:
            existing = await self._claim(ctx, record)
            if existing is not None:
                return existing

        timeout = spec.timeout_s or self.settings.tool_timeout_s
        try:
            idempotency_key = hashlib.sha256(
                f"{ctx.tenant_id}:{ctx.scope_id}:{call_id}:{fingerprint}".encode()).hexdigest()[:40]
            outcome = await asyncio.wait_for(self._dispatch(ctx, spec, call_id, idempotency_key, args, timeout),
                                             timeout + 1)
        except TimeoutError:
            outcome = ToolOutcome(f"tool {spec.name} timed out after {timeout:.1f}s", True)
        except asyncio.CancelledError:
            if spec.side_effects:
                self._record(ctx, {**record, "status": "cancelled", "finished_at": utcnow()})
            raise
        except GatewayError as exc:
            outcome = ToolOutcome(f"tool {spec.name} failed: {exc.message}", True)
        except Exception as exc:
            log.exception("tool crashed", extra={"tool": spec.name, "kind": spec.kind})
            outcome = ToolOutcome(f"tool {spec.name} failed: {exc.__class__.__name__}", True)
        outcome.output = render_output(outcome.output, self.settings.tool_output_max_chars)
        done = {**record, "status": "failed" if outcome.is_error else "completed", "output": outcome.output,
                "is_error": outcome.is_error, "finished_at": utcnow(), "control": outcome.control}
        self._record(ctx, done)
        if spec.side_effects:  # the stored result is what makes a replay safe: wait until it is durable
            await self.writer.flush(self._key(ctx), timeout=self.settings.firestore_timeout_s)
        return outcome

    async def _claim(self, ctx: ToolContext, record: dict[str, Any]) -> ToolOutcome | None:
        try:
            existing = await self.calls.claim(ctx.tenant_id, record)
        except GatewayError as exc:
            # cannot guarantee exactly-once: refuse rather than risk a duplicate side effect
            return ToolOutcome(f"tool {record['name']} not executed: idempotency store unavailable "
                               f"({exc.message})", True)
        deadline = time.monotonic() + (self.settings.tool_timeout_s * 2)
        while existing is not None:
            if existing.get("status") in ("completed", "failed"):
                return ToolOutcome(existing.get("output", ""), bool(existing.get("is_error")),
                                   control=existing.get("control"), replayed=True)
            if existing.get("status") == "cancelled" or time.monotonic() > deadline:
                return ToolOutcome(f"tool {record['name']} was already started for this call and its outcome "
                                   f"is unknown; check before retrying", True, replayed=True)
            await asyncio.sleep(0.2)
            existing = await self.calls.get(ctx.tenant_id, record)
        return None

    def _key(self, ctx: ToolContext) -> str:
        return f"{ctx.tenant_id}/{ctx.scope_id}"

    def _record(self, ctx: ToolContext, record: dict[str, Any]) -> None:
        self.writer.submit(self._key(ctx), lambda: self.calls.put(ctx.tenant_id, record), "tool_call")

    async def _dispatch(self, ctx: ToolContext, spec: ToolSpec, call_id: str, idempotency_key: str,
                        args: dict[str, Any], timeout: float) -> ToolOutcome:
        if spec.kind == "builtin":
            tool = builtin.REGISTRY.get(spec.target.get("builtin", spec.name))
            if tool is None:
                return ToolOutcome(f"unknown builtin tool {spec.name}", True)
            kwargs = dict(args)
            if tool.wants_ctx:
                kwargs["ctx"] = ctx
            try:
                if inspect.iscoroutinefunction(tool.handler):
                    result = await tool.handler(**kwargs)
                else:
                    result = await asyncio.to_thread(tool.handler, **kwargs)
            except TypeError as exc:
                return ToolOutcome(f"invalid arguments for {spec.name}: {exc}", True)
            if isinstance(result, ToolOutcome):
                return result
            return ToolOutcome(render_output(result, self.settings.tool_output_max_chars), data=result)
        if spec.kind == "http":
            return await self.http.run(spec, args, ctx, call_id, idempotency_key, timeout)
        if spec.kind == "mcp":
            server = await self.mcp_servers.get(ctx.tenant_id, spec.target["server_id"])
            if not server.get("enabled", True):
                return ToolOutcome(f"MCP server {server['name']} is disabled", True)
            return await self.mcp.call(ctx.tenant_id, server, spec.target["remote_name"], args, timeout)
        return ToolOutcome(f"tool {spec.name} ({spec.kind}) cannot run on the server", True)
