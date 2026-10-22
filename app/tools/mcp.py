"""MCP (Model Context Protocol) integration.

Tenants register remote MCP servers (Streamable HTTP or legacy SSE). The manager keeps one live
client per (tenant, server) - opened lazily, owned by a dedicated task because the SDK's anyio
scopes must be entered and exited in the same task - caches the tool listing, and reconnects
with backoff after failures. Tools are exposed to the model as `<tool_prefix><remote name>`.
Local stdio servers are deliberately not supported through the API: that would let a tenant
spawn processes on the server.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx2
import orjson
from mcp import Client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from app.config import Settings
from app.core.errors import GatewayError
from app.core.logging import get_logger
from app.core.metrics import MCP_CONNECTIONS, MCP_ERRORS
from app.core.tracing import inject_headers
from app.security.crypto import SecretBox
from app.security.netguard import NetGuard
from app.tools.base import NAME_RE, ToolOutcome, ToolSpec

log = get_logger("app.mcp")


def content_to_text(result: Any) -> tuple[str, Any]:
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            parts.append(block.text)
        elif kind == "resource":
            resource = getattr(block, "resource", None)
            text = getattr(resource, "text", None)
            parts.append(text if text is not None else f"[resource {getattr(resource, 'uri', '')}]")
        elif kind == "resource_link":
            parts.append(f"[resource {getattr(block, 'uri', '')}]")
        else:
            parts.append(f"[{kind or 'content'}]")
    structured = getattr(result, "structured_content", None)
    if structured is not None and not parts:
        parts.append(orjson.dumps(structured, default=str).decode())
    return "\n".join(parts), structured


class MCPConnection:
    def __init__(self, server: dict[str, Any], headers: dict[str, str], settings: Settings,
                 guard: NetGuard) -> None:
        self.guard = guard
        self.server = server
        self.version = server.get("version")
        self.headers = headers
        self.settings = settings
        self.client: Client | None = None
        self.error: str | None = None
        self.connected_at: float | None = None
        self.tools: list[Any] = []
        self.tools_at = 0.0
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._next_attempt = 0.0
        self._failures = 0

    @property
    def connected(self) -> bool:
        return self.client is not None and self._task is not None and not self._task.done()

    async def ensure(self) -> Client:
        if self.connected:
            assert self.client is not None
            return self.client
        async with self._lock:
            if self.connected:
                assert self.client is not None
                return self.client
            now = time.monotonic()
            if now < self._next_attempt:
                raise GatewayError(f"MCP server {self.server['name']} unavailable: {self.error}",
                                   code="mcp_unavailable", status=503, retry_after=self._next_attempt - now)
            self._ready.clear()
            self._stop.clear()
            self.client, self.error = None, None
            self._task = asyncio.create_task(self._run(), name=f"mcp-{self.server['id']}")
            try:
                await asyncio.wait_for(self._ready.wait(), self.settings.mcp_connect_timeout_s)
            except TimeoutError:
                self.error = "connect timeout"
                self._task.cancel()
            if self.client is None:
                self._failures += 1
                self._next_attempt = time.monotonic() + min(60.0, 2.0 ** self._failures)
                MCP_ERRORS.labels("connect").inc()
                raise GatewayError(f"cannot connect to MCP server {self.server['name']}: {self.error}",
                                   code="mcp_unavailable", status=503)
            self._failures = 0
            return self.client

    async def _run(self) -> None:
        url = self.server["url"]
        timeout = httpx2.Timeout(self.settings.mcp_connect_timeout_s, read=self.settings.mcp_call_timeout_s + 30)
        try:
            hook = self.guard.response_hook("MCP server URL")  # every connection, against DNS rebinding

            def guarded_client(**kwargs: Any) -> httpx2.AsyncClient:
                client = create_mcp_http_client(**kwargs)
                client.event_hooks["response"] = [*client.event_hooks.get("response", []), hook]
                return client

            if self.server.get("transport") == "sse":
                transport: Any = sse_client(url, headers=self.headers, timeout=self.settings.mcp_connect_timeout_s,
                                            sse_read_timeout=3600, httpx_client_factory=guarded_client)
                http = None
            else:
                http = guarded_client(headers=self.headers, timeout=timeout)
                transport = streamable_http_client(url, http_client=http)
            try:
                async with Client(transport, read_timeout_seconds=self.settings.mcp_call_timeout_s,
                                  cache=None) as client:
                    self.client = client
                    self.connected_at = time.monotonic()
                    MCP_CONNECTIONS.inc()
                    self._ready.set()
                    try:
                        await self._stop.wait()
                    finally:
                        MCP_CONNECTIONS.dec()
            finally:
                if http is not None:
                    await http.aclose()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # anyio raises ExceptionGroups
            self.error = _flatten(exc)
            log.warning("mcp connection ended", extra={"server": self.server["id"], "error": self.error})
        finally:
            self.client = None
            self._ready.set()

    async def list_tools(self, force: bool = False) -> list[Any]:
        ttl = self.settings.mcp_tools_cache_ttl_s
        if not force and self.tools_at and time.monotonic() - self.tools_at < ttl:
            return self.tools
        client = await self.ensure()
        tools: list[Any] = []
        cursor = None
        try:
            for _ in range(20):
                page = await asyncio.wait_for(client.list_tools(cursor=cursor), self.settings.mcp_call_timeout_s)
                tools.extend(page.tools)
                cursor = page.next_cursor
                if not cursor:
                    break
        except Exception as exc:
            MCP_ERRORS.labels("list_tools").inc()
            await self.close()
            raise GatewayError(f"MCP tools/list failed on {self.server['name']}: {_flatten(exc)}",
                               code="mcp_error", status=502) from None
        self.tools, self.tools_at = tools, time.monotonic()
        return tools

    async def call(self, name: str, args: dict[str, Any], timeout: float) -> ToolOutcome:
        client = await self.ensure()
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(client.call_tool(name, args, read_timeout_seconds=timeout), timeout + 1)
        except TimeoutError:
            MCP_ERRORS.labels("timeout").inc()
            return ToolOutcome(f"MCP tool {name} timed out after {timeout:.0f}s", True)
        except Exception as exc:
            MCP_ERRORS.labels("call").inc()
            message = _flatten(exc)
            if "closed" in message.lower() or "connect" in message.lower():
                await self.close()
            return ToolOutcome(f"MCP tool {name} failed: {message}", True)
        text, structured = content_to_text(result)
        return ToolOutcome(text, bool(result.is_error), data=structured,
                           duration_ms=(time.perf_counter() - start) * 1000)

    async def close(self) -> None:
        self._stop.set()
        task = self._task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), 5)
            except (TimeoutError, asyncio.CancelledError, Exception):
                task.cancel()
        self.client = None
        self.tools_at = 0.0


def _flatten(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        return _flatten(exc.exceptions[0])
    return f"{exc.__class__.__name__}: {str(exc)[:300]}"


class MCPManager:
    def __init__(self, settings: Settings, box: SecretBox, guard: NetGuard) -> None:
        self.settings = settings
        self.box = box
        self.guard = guard
        self._conns: dict[tuple[str, str], MCPConnection] = {}

    async def connection(self, tenant_id: str, server: dict[str, Any]) -> MCPConnection:
        key = (tenant_id, server["id"])
        conn = self._conns.get(key)
        if conn is not None and conn.version == server.get("version"):
            return conn
        if conn is not None:
            await conn.close()
        await self.guard.check(server["url"], "MCP server URL")
        headers = dict(server.get("headers") or {})
        headers.update(self.box.decrypt(server.get("secret_headers_enc")) or {})
        headers.update(inject_headers({}))
        conn = self._conns[key] = MCPConnection(server, headers, self.settings, self.guard)
        return conn

    async def tool_specs(self, tenant_id: str, server: dict[str, Any], force: bool = False) -> list[ToolSpec]:
        conn = await self.connection(tenant_id, server)
        remote = await conn.list_tools(force=force)
        allow = server.get("tool_allowlist")
        prefix = server.get("tool_prefix") or ""
        side_effects = set(server.get("side_effect_tools") or [])
        specs = []
        for tool in remote:
            if allow is not None and tool.name not in allow:
                continue
            name = f"{prefix}{tool.name}"[:64]
            if not NAME_RE.match(name):
                log.warning("skipping MCP tool with unusable name", extra={"tool": tool.name})
                continue
            annotations = getattr(tool, "annotations", None)
            destructive = bool(annotations and getattr(annotations, "destructive_hint", False)
                               and not getattr(annotations, "read_only_hint", False))
            specs.append(ToolSpec(
                name=name, description=(tool.description or tool.title or tool.name)[:4000],
                parameters=tool.input_schema or {"type": "object", "properties": {}}, kind="mcp",
                side_effects=tool.name in side_effects or destructive, timeout_s=server.get("timeout_s"),
                target={"server_id": server["id"], "remote_name": tool.name}))
        return specs

    async def call(self, tenant_id: str, server: dict[str, Any], remote_name: str, args: dict[str, Any],
                   timeout: float) -> ToolOutcome:
        conn = await self.connection(tenant_id, server)
        return await conn.call(remote_name, args, timeout)

    async def drop(self, tenant_id: str, server_id: str) -> None:
        conn = self._conns.pop((tenant_id, server_id), None)
        if conn is not None:
            await conn.close()

    async def close(self) -> None:
        await asyncio.gather(*(c.close() for c in self._conns.values()), return_exceptions=True)
        self._conns.clear()

    def snapshot(self, tenant_id: str | None = None) -> dict[str, Any]:
        """Connection status; with tenant_id only that tenant's connections (errors can carry URLs)."""
        conns = {k: c for k, c in self._conns.items() if tenant_id is None or k[0] == tenant_id}
        return {"connections": len(conns),
                "connected": sum(1 for c in conns.values() if c.connected),
                "errors": {f"{t}/{s}": c.error for (t, s), c in conns.items() if c.error}}
