"""Conversation sessions (in-process store with TTL + LRU eviction).

A session owns the conversation history, its system prompt, tool definitions and default
sampling parameters. History is append-only except for deliberate trims/compactions, so
consecutive turns share a growing prompt prefix and hit vLLM's prefix cache.

The store is in-memory and therefore per gateway process: run one gateway worker per engine
(the default) or put sticky routing in front of several gateways. See docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import orjson

from app.core.errors import BadRequest, Conflict, NotFound, Overloaded
from app.observability.logging import get_logger
from app.observability.metrics import SESSIONS_ACTIVE

if TYPE_CHECKING:
    from app.core.generation import Generation

log = get_logger("app.sessions")

ROLES = {"user", "assistant", "tool"}
MESSAGE_OVERHEAD_TOKENS = 5


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(12)}"


def content_text(content: Any) -> str:
    """Flatten OpenAI content (str | list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content)


@dataclass
class Message:
    seq: int
    role: str
    content: str | None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    interrupted: bool = False
    created_at: float = field(default_factory=time.time)

    @property
    def chars(self) -> int:
        n = len(self.content or "")
        for call in self.tool_calls or []:
            fn = call.get("function", {})
            n += len(fn.get("name", "")) + len(fn.get("arguments", ""))
        return n

    def to_openai(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role, "content": self.content if self.content is not None else ""}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg

    def to_public(self) -> dict[str, Any]:
        data = self.to_openai()
        data["id"] = self.seq
        data["created_at"] = self.created_at
        if self.interrupted:
            data["interrupted"] = True
        return data


@dataclass
class SessionStats:
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    last_prompt_tokens: int = 0
    trimmed_messages: int = 0
    compactions: int = 0


@dataclass
class Session:
    id: str
    system_prompt: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)  # client-side OpenAI function tools
    server_tools: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)  # default sampling params for this session
    metadata: dict[str, Any] = field(default_factory=dict)
    segmentation: bool = False
    include_reasoning: bool = False
    messages: list[Message] = field(default_factory=list)
    summary: str = ""
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.monotonic)
    chars_per_token: float = 3.0
    stats: SessionStats = field(default_factory=SessionStats)
    pending_tool_calls: dict[str, str] = field(default_factory=dict)  # call_id -> tool name
    active: Generation | None = None
    attachments: int = 0  # open WebSocket connections bound to this session
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    compacting: bool = False
    _seq: int = 0

    # ---------------------------------------------------------------- history
    @property
    def last_seq(self) -> int:
        return self._seq

    def touch(self) -> None:
        self.last_active = time.monotonic()

    def add(self, role: str, content: str | None, **kw: Any) -> Message:
        self._seq += 1
        msg = Message(seq=self._seq, role=role, content=content, **kw)
        self.messages.append(msg)
        return msg

    def add_openai(self, raw: dict[str, Any], max_chars: int) -> Message:
        """Validate and append a client-supplied OpenAI-format message."""
        if not isinstance(raw, dict):
            raise BadRequest("message must be an object")
        role = raw.get("role")
        if role == "system" or role == "developer":
            raise BadRequest("set the system prompt with system_prompt (session.update), not a message")
        if role not in ROLES:
            raise BadRequest(f"unsupported message role {role!r}")
        content = content_text(raw.get("content"))
        if len(content) > max_chars:
            raise BadRequest(f"message content exceeds {max_chars} characters (MAX_INPUT_CHARS)")
        tool_calls = raw.get("tool_calls") if role == "assistant" else None
        if role == "tool":
            call_id = raw.get("tool_call_id")
            if not call_id:
                raise BadRequest("tool messages need tool_call_id")
            self.pending_tool_calls.pop(call_id, None)
            return self.add("tool", content, tool_call_id=call_id, name=raw.get("name"))
        return self.add(role, content, tool_calls=tool_calls or None)

    def truncate_last_assistant(self, spoken_text: str) -> bool:
        """Barge-in: keep only what the caller actually heard of the last assistant reply."""
        for msg in reversed(self.messages):
            if msg.role == "assistant" and msg.content:
                # the client (which drove playback) is the authority on what was heard
                msg.content = spoken_text.strip()
                msg.interrupted = True
                return True
            if msg.role == "user":
                return False
        return False

    def history_chars(self) -> int:
        return sum(m.chars for m in self.messages)

    def estimate_tokens(self, extra_chars: int = 0) -> int:
        chars = len(self.system_prompt) + len(self.summary) + self.history_chars() + extra_chars
        return int(chars / self.chars_per_token) + MESSAGE_OVERHEAD_TOKENS * (len(self.messages) + 1)

    def calibrate(self, rendered_chars: int, prompt_tokens: int) -> None:
        if prompt_tokens > 32 and rendered_chars > 64:
            observed = rendered_chars / prompt_tokens
            self.chars_per_token = round(0.5 * self.chars_per_token + 0.5 * observed, 4)

    # ------------------------------------------------------------------- view
    def summary_view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "idle_s": round(time.monotonic() - self.last_active, 1),
            "messages": len(self.messages),
            "active_response": self.active.id if self.active else None,
            "pending_tool_calls": list(self.pending_tool_calls),
            "metadata": self.metadata,
        }

    def full_view(self, include_messages: bool = True) -> dict[str, Any]:
        data = self.summary_view()
        data.update({
            "system_prompt": self.system_prompt,
            "summary": self.summary,
            "tools": self.tools,
            "server_tools": self.server_tools,
            "params": self.params,
            "segmentation": self.segmentation,
            "include_reasoning": self.include_reasoning,
            "stats": self.stats.__dict__,
            "estimated_tokens": self.estimate_tokens(len(orjson.dumps(self.tools)) if self.tools else 0),
        })
        if include_messages:
            data["conversation"] = [m.to_public() for m in self.messages]
        return data


class SessionStore:
    def __init__(self, max_sessions: int, ttl_s: float, max_messages: int, chars_per_token: float) -> None:
        self.max_sessions = max_sessions
        self.ttl_s = ttl_s
        self.max_messages = max_messages
        self.chars_per_token = chars_per_token
        self._sessions: dict[str, Session] = {}
        self._sweeper: asyncio.Task | None = None

    def __len__(self) -> int:
        return len(self._sessions)

    def start(self) -> None:
        self._sweeper = asyncio.create_task(self._sweep_loop(), name="session-sweeper")

    async def stop(self) -> None:
        if self._sweeper:
            self._sweeper.cancel()
            try:
                await self._sweeper
            except asyncio.CancelledError:
                pass

    def create(self, session_id: str | None = None, **fields: Any) -> Session:
        if session_id is not None:
            if not (1 <= len(session_id) <= 128) or not session_id.replace("-", "").replace("_", "").isalnum():
                raise BadRequest("session_id must be 1-128 chars of [A-Za-z0-9_-]")
            if session_id in self._sessions:
                raise Conflict(f"session {session_id} already exists")
        if len(self._sessions) >= self.max_sessions and not self._evict_one():
            raise Overloaded("session limit reached (MAX_SESSIONS)", retry_after=5)
        session = Session(id=session_id or new_id("sess"), chars_per_token=self.chars_per_token, **fields)
        self._sessions[session.id] = session
        SESSIONS_ACTIVE.set(len(self._sessions))
        return session

    def get(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise NotFound(f"session {session_id} not found", code="session_not_found")
        session.touch()
        return session

    def list(self, limit: int = 50, offset: int = 0) -> list[Session]:
        items = sorted(self._sessions.values(), key=lambda s: s.last_active, reverse=True)
        return items[offset: offset + limit]

    async def delete(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        SESSIONS_ACTIVE.set(len(self._sessions))
        if session is None:
            raise NotFound(f"session {session_id} not found", code="session_not_found")
        if session.active:
            session.active.cancel("session_deleted")

    def check_capacity(self, session: Session, adding: int = 1) -> None:
        if len(session.messages) + adding > self.max_messages:
            raise BadRequest(
                f"session has {len(session.messages)} messages (MAX_MESSAGES_PER_SESSION={self.max_messages})",
                code="session_full",
            )

    def _evictable(self, session: Session) -> bool:
        return session.active is None and session.attachments == 0

    def _evict_one(self) -> bool:
        candidates = [s for s in self._sessions.values() if self._evictable(s)]
        if not candidates:
            return False
        oldest = min(candidates, key=lambda s: s.last_active)
        del self._sessions[oldest.id]
        log.warning("evicted least recently used session", extra={"evicted": oldest.id})
        return True

    def sweep(self) -> int:
        now = time.monotonic()
        expired = [s.id for s in self._sessions.values()
                   if self._evictable(s) and now - s.last_active > self.ttl_s]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            log.info("expired idle sessions", extra={"count": len(expired)})
        SESSIONS_ACTIVE.set(len(self._sessions))
        return len(expired)

    async def _sweep_loop(self) -> None:
        interval = max(1.0, min(30.0, self.ttl_s / 4))
        while True:
            await asyncio.sleep(interval)
            self.sweep()
