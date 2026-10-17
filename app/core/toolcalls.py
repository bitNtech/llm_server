"""Assemble OpenAI streaming `delta.tool_calls` fragments into complete tool calls."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    index: int
    id: str = ""
    name: str = ""
    arguments: str = ""
    announced: bool = False

    def to_message(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": self.arguments or "{}"}}


@dataclass
class ToolCallAccumulator:
    calls: dict[int, ToolCall] = field(default_factory=dict)

    def add(self, deltas: list[dict[str, Any]]) -> list[ToolCall]:
        """Merge fragments; return calls whose name became known in this delta (to announce)."""
        started: list[ToolCall] = []
        for pos, delta in enumerate(deltas or []):
            index = delta.get("index", pos)
            call = self.calls.get(index)
            if call is None:
                call = self.calls[index] = ToolCall(index=index)
            if delta.get("id"):
                call.id = delta["id"]
            fn = delta.get("function") or {}
            if fn.get("name") and not call.name:  # OpenAI streams the name whole, once
                call.name = fn["name"]
            if fn.get("arguments"):
                call.arguments += fn["arguments"]
            if call.name and not call.announced:
                call.announced = True
                if not call.id:
                    call.id = f"call_{uuid.uuid4().hex[:24]}"
                started.append(call)
        return started

    def finish(self) -> list[ToolCall]:
        done = []
        for index in sorted(self.calls):
            call = self.calls[index]
            if not call.name:
                continue
            if not call.id:
                call.id = f"call_{uuid.uuid4().hex[:24]}"
            done.append(call)
        return done

    def __bool__(self) -> bool:
        return any(c.name for c in self.calls.values())
