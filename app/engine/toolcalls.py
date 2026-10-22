"""Assemble streamed `delta.tool_calls` fragments into complete tool calls.

Vertex streams each call whole (id, name, full arguments) in one fragment, but the accumulator also
handles OpenAI-style incremental fragments. Gemini's `extra_content` (thought signature) is kept per
call, because it must be sent back unchanged when the conversation continues.
"""
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
    extra_content: dict[str, Any] | None = None
    announced: bool = False

    def to_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"id": self.id, "type": "function",
                               "function": {"name": self.name, "arguments": self.arguments or "{}"}}
        if self.extra_content:
            msg["extra_content"] = self.extra_content
        return msg


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
            if fn.get("name") and not call.name:
                call.name = fn["name"]
            if fn.get("arguments"):
                call.arguments += fn["arguments"]
            if delta.get("extra_content"):
                call.extra_content = delta["extra_content"]
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
