"""Tool model shared by every tool kind.

Kinds:
  builtin  Python functions shipped with the server (app/tools/builtin.py)
  http     tenant-defined HTTPS endpoints (webhook tools), executed by the server
  mcp      tools discovered on tenant-registered MCP servers, executed by the server
  client   executed by the caller (voice runtime / LiveKit worker); the turn pauses with
           `requires_action` until the result is submitted
  handoff  synthetic `transfer_to_<agent>` tools; the runtime switches the active agent
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import orjson
from jsonschema import Draft202012Validator, SchemaError
from jsonschema.exceptions import ValidationError

from app.core.errors import BadRequest

ToolKind = Literal["builtin", "http", "mcp", "client", "handoff"]
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,63}$")
EMPTY_PARAMS: dict[str, Any] = {"type": "object", "properties": {}}


@dataclass(frozen=True)
class ToolSpec:
    name: str  # model-facing function name, unique within one request
    description: str
    parameters: dict[str, Any] = field(compare=False, hash=False)
    kind: ToolKind = "builtin"
    side_effects: bool = False
    timeout_s: float | None = None
    target: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters or EMPTY_PARAMS}}

    @property
    def server_side(self) -> bool:
        return self.kind in ("builtin", "http", "mcp")


@dataclass
class ToolContext:
    tenant_id: str
    scope_id: str  # session id, execution id or chat request id: the idempotency scope
    agent_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None


@dataclass
class ToolOutcome:
    output: str
    is_error: bool = False
    data: Any = None
    control: dict[str, Any] | None = None  # side signals (e.g. end_conversation)
    replayed: bool = False  # result came from an earlier identical execution
    duration_ms: float = 0.0


def check_name(name: str, what: str = "tool name") -> str:
    if not NAME_RE.match(name):
        raise BadRequest(f"{what} {name!r} must match {NAME_RE.pattern} (letters, digits, _ . : -; max 64)",
                         code="invalid_tool_name")
    return name


def check_schema(schema: Any, what: str = "parameters") -> dict[str, Any]:
    if schema is None:
        return dict(EMPTY_PARAMS)
    if not isinstance(schema, dict):
        raise BadRequest(f"{what} must be a JSON Schema object")
    if schema.get("type", "object") != "object":
        raise BadRequest(f"{what} must describe an object (type: object)")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise BadRequest(f"{what} is not a valid JSON Schema: {exc.message}", code="invalid_schema") from None
    if len(orjson.dumps(schema)) > 64_000:
        raise BadRequest(f"{what} is too large (64 KB max)")
    return schema


class ArgumentValidator:
    """Compiled JSON-Schema validators, cached by schema content."""

    def __init__(self) -> None:
        self._cache: dict[str, Draft202012Validator] = {}

    def validate(self, schema: dict[str, Any], args: dict[str, Any]) -> str | None:
        if not schema or not schema.get("properties") and not schema.get("required"):
            return None
        key = hashlib.sha256(orjson.dumps(schema, option=orjson.OPT_SORT_KEYS)).hexdigest()
        validator = self._cache.get(key)
        if validator is None:
            try:
                validator = Draft202012Validator(schema)
            except SchemaError:
                return None  # schema came from a remote MCP server we cannot fix; let the tool decide
            if len(self._cache) > 5000:
                self._cache.clear()
            self._cache[key] = validator
        errors = sorted(validator.iter_errors(args), key=lambda e: list(e.path))
        if not errors:
            return None
        return "; ".join(_describe(e) for e in errors[:5])


def _describe(error: ValidationError) -> str:
    path = "/".join(str(p) for p in error.path)
    return f"{path or '(arguments)'}: {error.message}"


def parse_arguments(raw: str | None) -> dict[str, Any]:
    if raw is None or not raw.strip():
        return {}
    value = orjson.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("arguments must be a JSON object")
    return value


def render_output(value: Any, max_chars: int) -> str:
    text = value if isinstance(value, str) else orjson.dumps(value, default=str).decode()
    if len(text) > max_chars:
        return text[:max_chars] + f"... [truncated {len(text) - max_chars} chars]"
    return text
