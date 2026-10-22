"""Built-in server tools. Agents enable them by name: {"type": "builtin", "name": "get_current_time"}.

Add more with the decorator; a handler may accept `ctx: ToolContext` to learn who is calling.

    @builtin(description="...", parameters={...JSON Schema...}, timeout_s=2, side_effects=False)
    async def my_tool(arg: str, ctx: ToolContext) -> dict: ...
"""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.tools.base import EMPTY_PARAMS, ToolContext, ToolOutcome, ToolSpec

Handler = Callable[..., Any | Awaitable[Any]]


@dataclass
class BuiltinTool:
    spec: ToolSpec
    handler: Handler
    wants_ctx: bool


REGISTRY: dict[str, BuiltinTool] = {}


def builtin(name: str | None = None, *, description: str, parameters: dict[str, Any] | None = None,
            timeout_s: float | None = None, side_effects: bool = False) -> Callable[[Handler], Handler]:
    def decorate(fn: Handler) -> Handler:
        tool_name = name or fn.__name__
        spec = ToolSpec(name=tool_name, description=description, parameters=parameters or dict(EMPTY_PARAMS),
                        kind="builtin", side_effects=side_effects, timeout_s=timeout_s,
                        target={"builtin": tool_name})
        REGISTRY[tool_name] = BuiltinTool(spec, fn, "ctx" in inspect.signature(fn).parameters)
        return fn
    return decorate


def describe() -> list[dict[str, Any]]:
    return [{"name": t.spec.name, "description": t.spec.description, "parameters": t.spec.parameters,
             "side_effects": t.spec.side_effects, "timeout_s": t.spec.timeout_s} for t in REGISTRY.values()]


@builtin(
    description="Get the current date and time. Use it whenever the user asks about today's date, "
                "the day of the week, the time, or relative dates like 'tomorrow'.",
    parameters={"type": "object", "properties": {
        "timezone": {"type": "string", "description": "IANA timezone, e.g. Asia/Kolkata. Default UTC."}}},
    timeout_s=1.0,
)
def get_current_time(timezone: str = "UTC") -> dict[str, Any]:
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        zone, timezone = ZoneInfo("UTC"), "UTC"
    now = datetime.now(zone)
    return {"timezone": timezone, "iso": now.isoformat(timespec="seconds"), "date": now.strftime("%Y-%m-%d"),
            "weekday": now.strftime("%A"), "time": now.strftime("%H:%M")}


@builtin(
    description="End the conversation (hang up) after saying goodbye. Call it only when the user is done "
                "or asks to end the call.",
    parameters={"type": "object", "properties": {
        "reason": {"type": "string", "description": "Short reason, e.g. 'caller said goodbye'"}}},
    timeout_s=1.0,
)
def end_conversation(reason: str = "completed") -> ToolOutcome:
    return ToolOutcome(output="The conversation will end after your final sentence.",
                       control={"end_conversation": {"reason": reason[:200]}})


__all__ = ["REGISTRY", "ToolContext", "builtin", "describe"]
