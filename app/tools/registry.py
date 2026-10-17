"""Server-side tools: Python functions the gateway executes itself inside the tool loop.

Two kinds of tools exist:
  * client tools  - OpenAI function schemas sent by the client; calls are streamed back to the
                    client, which executes them and returns results (`tool.result` / REST).
  * server tools  - registered here; executed by the gateway with a timeout, results fed back
                    to the model without a client round trip (lowest latency for voice).

Register tools in any module listed in TOOLS_MODULES:

    from app.tools import tool

    @tool(description="Look up a booking", parameters={...JSON schema...})
    async def lookup_booking(booking_id: str) -> dict: ...

A session opts into server tools by name (`server_tools: ["lookup_booking"]`).
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import orjson

from app.observability.logging import get_logger
from app.observability.metrics import TOOL_CALLS, TOOL_LATENCY

log = get_logger("app.tools")

Handler = Callable[..., Any | Awaitable[Any]]
_DECLARED: list[ServerTool] = []


@dataclass
class ServerTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    timeout_s: float | None = None

    def schema(self) -> dict[str, Any]:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def tool(
    name: str | None = None,
    *,
    description: str = "",
    parameters: dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> Callable[[Handler], Handler]:
    def decorate(fn: Handler) -> Handler:
        _DECLARED.append(ServerTool(
            name=name or fn.__name__,
            description=description or (inspect.getdoc(fn) or "").strip(),
            parameters=parameters or {"type": "object", "properties": {}},
            handler=fn,
            timeout_s=timeout_s,
        ))
        return fn
    return decorate


class ToolRegistry:
    def __init__(self, default_timeout_s: float = 10.0) -> None:
        self.default_timeout_s = default_timeout_s
        self._tools: dict[str, ServerTool] = {}

    def register(self, server_tool: ServerTool) -> None:
        if server_tool.name in self._tools:
            raise ValueError(f"duplicate server tool {server_tool.name!r}")
        self._tools[server_tool.name] = server_tool

    def load_modules(self, modules: list[str]) -> None:
        for module in modules:
            mod = importlib.import_module(module)
            for t in _DECLARED:
                if t.handler.__module__ == mod.__name__ and t.name not in self._tools:
                    self.register(t)
        if self._tools:
            log.info("server tools loaded", extra={"tools": sorted(self._tools)})

    def names(self) -> list[str]:
        return sorted(self._tools)

    def has(self, name: str) -> bool:
        return name in self._tools

    def schemas(self, names: list[str]) -> list[dict[str, Any]]:
        return [self._tools[n].schema() for n in names if n in self._tools]

    def describe(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    async def execute(self, name: str, arguments: str) -> tuple[str, bool]:
        """Run a tool; returns (output_text, is_error). Never raises."""
        server_tool = self._tools.get(name)
        if server_tool is None:
            return f"unknown tool {name!r}", True
        try:
            args = orjson.loads(arguments or "{}")
            if not isinstance(args, dict):
                raise TypeError("arguments must be a JSON object")
        except (orjson.JSONDecodeError, TypeError) as exc:
            TOOL_CALLS.labels(name, "server", "bad_arguments").inc()
            return f"invalid arguments: {exc}", True
        missing = [k for k in server_tool.parameters.get("required", []) if k not in args]
        if missing:
            TOOL_CALLS.labels(name, "server", "bad_arguments").inc()
            return f"missing required arguments: {', '.join(missing)}", True

        timeout = server_tool.timeout_s or self.default_timeout_s
        start = time.perf_counter()
        try:
            if inspect.iscoroutinefunction(server_tool.handler):
                result = await asyncio.wait_for(server_tool.handler(**args), timeout)
            else:  # sync tools run in a worker thread so they never block the event loop
                result = await asyncio.wait_for(asyncio.to_thread(server_tool.handler, **args), timeout)
        except TimeoutError:
            TOOL_CALLS.labels(name, "server", "timeout").inc()
            return f"tool {name} timed out after {timeout:.0f}s", True
        except TypeError as exc:
            TOOL_CALLS.labels(name, "server", "bad_arguments").inc()
            return f"invalid arguments: {exc}", True
        except Exception as exc:
            TOOL_CALLS.labels(name, "server", "error").inc()
            log.exception("server tool failed", extra={"tool": name})
            return f"tool {name} failed: {exc.__class__.__name__}", True
        finally:
            TOOL_LATENCY.labels(name).observe(time.perf_counter() - start)
        TOOL_CALLS.labels(name, "server", "ok").inc()
        if isinstance(result, str):
            return result, False
        return orjson.dumps(result, default=str).decode(), False
