"""Conversation sessions: in-memory live state backed by Firestore.

A live session (history, pending client tool calls, the active generation, its lock) is held by
exactly one server instance at a time, enforced by a Firestore lease. Reads and writes during a
call hit memory; changes are persisted write-behind in order (messages, responses, session state),
so a session survives restarts and can resume on another instance once the lease is free.

Persisted history is complete and append-only (barge-in truncation rewrites one message); the
context window sent to the model is a *view* of it starting at `context_start_seq`.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from app.config import Settings
from app.core.errors import BadRequest, Conflict, GatewayError, NotFound
from app.core.ids import iso, new_id, utcnow, valid_client_id
from app.core.logging import get_logger
from app.core.metrics import SESSIONS_CACHED
from app.store.repos import SessionRepo
from app.store.writer import Writer

if TYPE_CHECKING:
    from app.engine.generation import Generation

log = get_logger("app.sessions")

ROLES = {"user", "assistant", "tool"}
MESSAGE_OVERHEAD_TOKENS = 5
ABANDONED_RESULT = "No result: the conversation moved on before this tool call completed."


def content_text(content: Any) -> str:
    """Flatten OpenAI content (str | list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        if not all(isinstance(t, str) for t in parts):
            raise BadRequest("text content parts need a string `text`")
        return "".join(parts)
    return str(content)


@dataclass
class Message:
    seq: int
    role: str
    content: str | None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    agent_id: str | None = None
    model: str | None = None  # model that produced an assistant message (thought signatures are per model)
    interrupted: bool = False
    extra_content: dict[str, Any] | None = None
    created_at: Any = field(default_factory=utcnow)

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
        if self.model and (self.tool_calls or self.extra_content):
            msg["_model"] = self.model  # provider swaps signatures if another model serves the request
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        if self.extra_content and self.role == "assistant":
            msg["extra_content"] = self.extra_content
        return msg

    def to_doc(self) -> dict[str, Any]:
        return {"seq": self.seq, "role": self.role, "content": self.content, "tool_calls": self.tool_calls,
                "tool_call_id": self.tool_call_id, "name": self.name, "agent_id": self.agent_id,
                "model": self.model, "interrupted": self.interrupted, "extra_content": self.extra_content,
                "created_at": self.created_at}

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> Message:
        return cls(**{k: doc.get(k) for k in ("seq", "role", "content", "tool_calls", "tool_call_id", "name",
                                                 "agent_id", "model", "extra_content", "created_at")},
                   interrupted=bool(doc.get("interrupted")))

    def to_public(self) -> dict[str, Any]:
        data: dict[str, Any] = {"seq": self.seq, "role": self.role, "content": self.content,
                                "created_at": iso(self.created_at)}
        if self.tool_calls:
            data["tool_calls"] = [{"id": c["id"], "type": "function", "function": c["function"]}
                                  for c in self.tool_calls]
        for key in ("tool_call_id", "name", "agent_id"):
            if getattr(self, key):
                data[key] = getattr(self, key)
        if self.interrupted:
            data["interrupted"] = True
        return data


@dataclass
class SessionStats:
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    last_prompt_tokens: int = 0
    tool_calls: int = 0
    handoffs: int = 0
    trimmed_messages: int = 0
    compactions: int = 0

    def to_doc(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class LiveSession:
    tenant_id: str
    id: str
    agent_id: str | None = None
    agent_version: int | None = None
    current_agent_id: str | None = None
    current_agent_version: int | None = None
    instructions: str = ""  # session-level instructions appended to the agent's
    model: str | None = None  # agent-less sessions only
    tools: list[dict[str, Any]] = field(default_factory=list)  # extra client tools
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    segmentation: bool | None = None
    status: str = "active"
    channel: str = "api"
    created_at: Any = field(default_factory=utcnow)
    ended_at: Any = None
    end_reason: str | None = None
    messages: list[Message] = field(default_factory=list)
    summary: str = ""
    context_start_seq: int = 0
    chars_per_token: float = 4.0
    stats: SessionStats = field(default_factory=SessionStats)
    pending_tool_calls: dict[str, str] = field(default_factory=dict)  # call_id -> tool name
    active: Generation | None = None
    last_response: Generation | None = None  # newest response, for discarding a finished speculative turn
    stale: bool = False  # another instance took the lease: this copy must not run or persist anything
    attachments: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_active: float = field(default_factory=time.monotonic)
    compacting: bool = False
    lease_renewed: float = 0.0
    persist: Callable[[str, Any], None] | None = None  # set by SessionManager; None = ephemeral
    _seq: int = 0

    # ---------------------------------------------------------------- history
    @property
    def last_seq(self) -> int:
        return self._seq

    def touch(self) -> None:
        self.last_active = time.monotonic()

    def _changed(self, op: str, payload: Any) -> None:
        if self.persist is not None:
            self.persist(op, payload)

    def add(self, role: str, content: str | None, **kw: Any) -> Message:
        self._seq += 1
        msg = Message(seq=self._seq, role=role, content=content, **kw)
        self.messages.append(msg)
        self._changed("message", msg)
        return msg

    def add_openai(self, raw: Any, max_chars: int) -> Message:
        """Validate and append a client-supplied OpenAI-format message."""
        if not isinstance(raw, dict):
            raise BadRequest("message must be an object")
        role = raw.get("role")
        if role in ("system", "developer"):
            raise BadRequest("set instructions with the agent or session `instructions`, not a system message")
        if role not in ROLES:
            raise BadRequest(f"unsupported message role {role!r}")
        content = content_text(raw.get("content"))
        if len(content) > max_chars:
            raise BadRequest(f"message content exceeds {max_chars} characters (MAX_INPUT_CHARS)")
        if role == "tool":
            call_id = raw.get("tool_call_id")
            if not call_id:
                raise BadRequest("tool messages need tool_call_id")
            self.pending_tool_calls.pop(call_id, None)
            return self.add("tool", content, tool_call_id=call_id, name=raw.get("name"))
        self.abandon_open_calls()  # a call must be followed by its results before anything else
        tool_calls = raw.get("tool_calls") if role == "assistant" else None
        if tool_calls is not None and not (isinstance(tool_calls, list) and all(
                isinstance(c, dict) and isinstance(c.get("id"), str) and isinstance(c.get("function"), dict)
                for c in tool_calls)):
            raise BadRequest("assistant tool_calls must be OpenAI tool call objects")
        return self.add(role, content, tool_calls=tool_calls or None, agent_id=self.current_agent_id)

    def truncate_last_assistant(self, spoken_text: str) -> bool:
        """Barge-in: keep only what the caller actually heard of the last assistant reply."""
        for msg in reversed(self.messages):
            if msg.role == "assistant" and msg.content:
                msg.content = spoken_text.strip()
                msg.interrupted = True
                self._changed("message", msg)
                return True
            if msg.role == "user":
                return False
        return False

    def drop_after(self, seq: int) -> list[int]:
        dropped = [m.seq for m in self.messages if m.seq > seq]
        self.messages = [m for m in self.messages if m.seq <= seq]
        if dropped:
            self._changed("delete", dropped)
        return dropped

    def clear(self) -> None:
        self.messages = []
        self.summary = ""
        self.context_start_seq = self._seq
        self.pending_tool_calls.clear()
        self._changed("clear", None)

    def unanswered_tool_calls(self) -> dict[str, str]:
        """Tool calls of the latest assistant turn that have no result message yet (call_id -> name)."""
        answered: set[str | None] = set()
        for msg in reversed(self.messages):
            if msg.role == "tool":
                answered.add(msg.tool_call_id)
            elif msg.role == "assistant" and msg.tool_calls:
                return {c["id"]: (c.get("function") or {}).get("name", "") for c in msg.tool_calls
                        if c.get("id") not in answered}
            else:
                return {}
        return {}

    def abandon_open_calls(self) -> None:
        """Answer every open tool call with ABANDONED_RESULT (Gemini rejects unanswered calls)."""
        for call_id, name in {**self.unanswered_tool_calls(), **self.pending_tool_calls}.items():
            self.add("tool", ABANDONED_RESULT, tool_call_id=call_id, name=name)
        self.pending_tool_calls.clear()

    def window(self) -> list[Message]:
        return [m for m in self.messages if m.seq > self.context_start_seq]

    def calibrate(self, rendered_chars: int, prompt_tokens: int) -> None:
        if prompt_tokens > 32 and rendered_chars > 64:
            observed = rendered_chars / prompt_tokens
            self.chars_per_token = round(0.5 * self.chars_per_token + 0.5 * observed, 4)

    # ---------------------------------------------------------------- views
    def state_doc(self) -> dict[str, Any]:
        """Mutable session fields persisted after each turn / change."""
        return {
            "agent_id": self.agent_id, "agent_version": self.agent_version,
            "current_agent_id": self.current_agent_id, "current_agent_version": self.current_agent_version,
            "instructions": self.instructions, "model": self.model, "tools": self.tools, "params": self.params,
            "metadata": self.metadata, "segmentation": self.segmentation, "status": self.status,
            "ended_at": self.ended_at, "end_reason": self.end_reason, "summary": self.summary,
            "context_start_seq": self.context_start_seq, "last_seq": self._seq,
            "message_count": len(self.messages), "chars_per_token": self.chars_per_token,
            "stats": self.stats.to_doc(), "pending_tool_calls": self.pending_tool_calls,
            "updated_at": utcnow(),
        }

    def view(self, include_messages: bool = False) -> dict[str, Any]:
        data = {
            "id": self.id, "object": "session", "status": self.status, "channel": self.channel,
            "agent_id": self.agent_id, "agent_version": self.agent_version,
            "current_agent_id": self.current_agent_id, "current_agent_version": self.current_agent_version,
            "instructions": self.instructions, "model": self.model, "tools": self.tools, "params": self.params,
            "metadata": self.metadata, "segmentation": self.segmentation, "summary": self.summary,
            "created_at": iso(self.created_at), "ended_at": iso(self.ended_at), "end_reason": self.end_reason,
            "message_count": len(self.messages), "pending_tool_calls": list(self.pending_tool_calls),
            "active_response_id": self.active.id if self.active and not self.active.done else None,
            "stats": self.stats.to_doc(),
        }
        if include_messages:
            data["messages"] = [m.to_public() for m in self.messages]
        return data


def doc_view(doc: dict[str, Any]) -> dict[str, Any]:
    """Public view of a persisted session that is not loaded on this instance."""
    keys = ("id", "status", "channel", "agent_id", "agent_version", "current_agent_id", "current_agent_version",
            "instructions", "model", "tools", "params", "metadata", "segmentation", "summary", "end_reason",
            "stats")
    data = {k: doc.get(k) for k in keys}
    data.update({"object": "session", "created_at": iso(doc.get("created_at")), "ended_at": iso(doc.get("ended_at")),
                 "message_count": doc.get("message_count", 0),
                 "pending_tool_calls": list((doc.get("pending_tool_calls") or {}).keys()),
                 "active_response_id": None})
    return data


class SessionManager:
    def __init__(self, settings: Settings, repo: SessionRepo, writer: Writer) -> None:
        self.settings = settings
        self.repo = repo
        self.writer = writer
        self.owner = settings.instance_id
        self._live: dict[tuple[str, str], LiveSession] = {}
        self._loading: dict[tuple[str, str], asyncio.Future] = {}
        self._releasing: dict[tuple[str, str], asyncio.Future] = {}
        self._task: asyncio.Task | None = None
        self.on_end: Callable[[LiveSession], Any] | None = None  # post-session hooks (workflows)

    def __len__(self) -> int:
        return len(self._live)

    # ------------------------------------------------------------ persistence
    def _persister(self, session: LiveSession) -> Callable[[str, Any], None]:
        tenant, sid = session.tenant_id, session.id
        key = f"{tenant}/{sid}"

        def persist(op: str, payload: Any) -> None:
            if op == "message":
                doc = payload.to_doc()
                self.writer.submit(key, lambda: self.repo.put_message(tenant, sid, doc), "message")
            elif op == "delete":
                seqs = list(payload)
                self.writer.submit(key, lambda: self.repo.delete_messages(tenant, sid, seqs), "messages.delete")
            elif op == "clear":
                self.writer.submit(key, lambda: self.repo.delete_messages(tenant, sid), "messages.clear")
            elif op == "state":
                state = payload
                self.writer.submit(key, lambda: self.repo.save(tenant, sid, state), "session.state")
            elif op == "response":
                record = payload
                self.writer.submit(key, lambda: self.repo.put_response(tenant, sid, record), "response")
        return persist

    def save_state(self, session: LiveSession) -> None:
        if session.persist is not None:
            session.persist("state", session.state_doc())

    def save_response(self, session: LiveSession, record: dict[str, Any]) -> None:
        if session.persist is not None:
            session.persist("response", record)

    # --------------------------------------------------------------- create
    async def create(self, tenant_id: str, fields: dict[str, Any], session_id: str | None = None,
                     seed_messages: list[dict[str, Any]] | None = None, max_input_chars: int = 64_000) -> LiveSession:
        if session_id is not None and not valid_client_id(session_id):
            raise BadRequest("session_id must be 1-128 chars of [A-Za-z0-9_-]")
        if (tenant_id, session_id) in self._live:
            raise Conflict(f"session {session_id} already exists", code="session_exists")
        session = LiveSession(tenant_id=tenant_id, id=session_id or new_id("sess"),
                              chars_per_token=self.settings.context_chars_per_token, **fields)
        lease = {"owner": self.owner, "expires_at": utcnow() + timedelta(seconds=self.settings.session_lease_s)}
        doc = {**session.state_doc(), "id": session.id, "created_at": session.created_at,
               "channel": session.channel, "lease": lease}
        try:
            await self.repo.create(tenant_id, doc)
        except Conflict:
            raise Conflict(f"session {session.id} already exists", code="session_exists") from None
        session.persist = self._persister(session)
        session.lease_renewed = time.monotonic()
        self._live[(tenant_id, session.id)] = session
        SESSIONS_CACHED.set(len(self._live))
        for raw in seed_messages or []:
            session.add_openai(raw, max_input_chars)
        return session

    # ------------------------------------------------------------------ get
    async def get(self, tenant_id: str, session_id: str) -> LiveSession:
        """Load a session for reading *and writing* on this instance (takes the lease)."""
        key = (tenant_id, session_id)
        releasing = self._releasing.get(key)
        if releasing is not None:  # an eviction is flushing and releasing it: load after that finishes
            await asyncio.shield(releasing)
            return await self.get(tenant_id, session_id)
        session = self._live.get(key)
        if session is not None:
            session.touch()
            return session
        pending = self._loading.get(key)
        if pending is not None:
            return await asyncio.shield(pending)
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._loading[key] = future
        try:
            session = await self._load(tenant_id, session_id)
            self._live[key] = session
            SESSIONS_CACHED.set(len(self._live))
            future.set_result(session)
            return session
        except BaseException as exc:
            future.set_exception(exc)
            future.exception()  # mark retrieved
            raise
        finally:
            self._loading.pop(key, None)

    async def _load(self, tenant_id: str, session_id: str) -> LiveSession:
        doc = await self.repo.acquire_lease(tenant_id, session_id, self.owner, self.settings.session_lease_s)
        rows = await self.repo.load_messages(tenant_id, session_id)
        session = LiveSession(
            tenant_id=tenant_id, id=session_id, agent_id=doc.get("agent_id"), agent_version=doc.get("agent_version"),
            current_agent_id=doc.get("current_agent_id"), current_agent_version=doc.get("current_agent_version"),
            instructions=doc.get("instructions") or "", model=doc.get("model"), tools=doc.get("tools") or [],
            params=doc.get("params") or {}, metadata=doc.get("metadata") or {}, segmentation=doc.get("segmentation"),
            status=doc.get("status", "active"), channel=doc.get("channel", "api"),
            created_at=doc.get("created_at") or utcnow(), ended_at=doc.get("ended_at"),
            end_reason=doc.get("end_reason"), summary=doc.get("summary") or "",
            context_start_seq=int(doc.get("context_start_seq") or 0),
            chars_per_token=float(doc.get("chars_per_token") or self.settings.context_chars_per_token),
            stats=SessionStats(**{k: v for k, v in (doc.get("stats") or {}).items()
                                  if k in SessionStats.__dataclass_fields__}),
            pending_tool_calls=dict(doc.get("pending_tool_calls") or {}),
            messages=[Message.from_doc(r) for r in rows])
        session._seq = max([int(doc.get("last_seq") or 0)] + [m.seq for m in session.messages])
        session.persist = self._persister(session)
        session.lease_renewed = time.monotonic()
        log.info("session loaded", extra={"session_id": session_id, "messages": len(session.messages)})
        return session

    def cached(self, tenant_id: str, session_id: str) -> LiveSession | None:
        return self._live.get((tenant_id, session_id))

    async def describe(self, tenant_id: str, session_id: str, include_messages: bool) -> dict[str, Any]:
        """Read-only view: from memory if loaded here, else from Firestore (no lease taken)."""
        session = self._live.get((tenant_id, session_id))
        if session is not None:
            return session.view(include_messages)
        doc = await self.repo.find(tenant_id, session_id)
        if doc is None:
            raise NotFound(f"session {session_id} not found", code="session_not_found")
        view = doc_view(doc)
        if include_messages:
            rows = await self.repo.load_messages(tenant_id, session_id)
            view["messages"] = [Message.from_doc(r).to_public() for r in rows]
        return view

    # ------------------------------------------------------------- lifecycle
    async def end(self, session: LiveSession, reason: str) -> None:
        if session.status == "ended":
            return
        if session.active is not None and not session.active.done:
            session.active.cancel("session_ended")
            await session.active.wait(5)
        session.status = "ended"
        session.ended_at = utcnow()
        session.end_reason = reason[:200]
        self.save_state(session)
        if self.on_end is not None:
            try:
                await self.on_end(session)
            except GatewayError as exc:
                log.warning("session end hook failed", extra={"session_id": session.id, "error": exc.message})

    async def delete(self, tenant_id: str, session_id: str) -> None:
        session = self._live.pop((tenant_id, session_id), None)
        SESSIONS_CACHED.set(len(self._live))
        if session is not None:
            if session.active is not None:
                session.active.cancel("session_deleted")
            session.persist = None
            session.stale = True
            await self.writer.flush(f"{tenant_id}/{session_id}", timeout=10)
        else:
            # refuses (409) while another instance owns it, whose write-behind would recreate it
            await self.repo.acquire_lease(tenant_id, session_id, self.owner, self.settings.session_lease_s)
        await self.repo.delete(tenant_id, session_id)

    async def _renew(self, session: LiveSession) -> None:
        await self.repo.acquire_lease(session.tenant_id, session.id, self.owner, self.settings.session_lease_s)
        session.lease_renewed = time.monotonic()

    async def _release(self, session: LiveSession) -> bool:
        """Flush the session's writes, then give up the lease. Keeps the lease (returns False) while
        writes are still queued: another instance must not load history that is missing them."""
        if not await self.writer.flush(f"{session.tenant_id}/{session.id}", timeout=10):
            log.warning("session writes still pending; keeping the lease", extra={"session_id": session.id})
            return False
        try:
            await self.repo.release_lease(session.tenant_id, session.id, self.owner)
        except GatewayError as exc:
            log.warning("lease release failed", extra={"session_id": session.id, "error": exc.message})
        return True

    async def _evict(self, key: tuple[str, str], session: LiveSession) -> None:
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._releasing[key] = future
        try:
            if await self._release(session):
                self._live.pop(key, None)
                session.stale = True
                session.persist = None
        finally:
            self._releasing.pop(key, None)
            future.set_result(None)

    def start(self) -> None:
        self._task = asyncio.create_task(self._maintain(), name="session-maintenance")

    async def _maintain(self) -> None:
        """Renew leases of busy sessions; flush, release and evict idle ones."""
        interval = max(2.0, self.settings.session_lease_s / 3)
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            # least recently used first, so capacity evictions hit the coldest sessions
            for key, session in sorted(self._live.items(), key=lambda kv: kv[1].last_active):
                if self._live.get(key) is not session:
                    continue
                busy = session.attachments > 0 or (session.active is not None and not session.active.done)
                idle = now - session.last_active
                try:
                    if not busy and (idle > self.settings.session_cache_ttl_s
                                     or len(self._live) > self.settings.max_cached_sessions):
                        await self._evict(key, session)
                    elif now - session.lease_renewed > self.settings.session_lease_s / 2:
                        await self._renew(session)
                except Conflict:
                    # another instance took over after our lease lapsed: our copy is stale
                    log.warning("session lease lost; dropping local copy", extra={"session_id": session.id})
                    self._live.pop(key, None)
                    session.stale = True  # fence: attached sockets get session_busy_elsewhere
                    session.persist = None
                    if session.active is not None:
                        session.active.cancel("lease_lost")
                except GatewayError as exc:
                    log.warning("session maintenance failed", extra={"session_id": session.id,
                                                                     "error": exc.message})
            SESSIONS_CACHED.set(len(self._live))

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        for session in list(self._live.values()):
            self.save_state(session)
        await self.writer.flush(timeout=15)
        await asyncio.gather(*(self._release(s) for s in self._live.values()), return_exceptions=True)
        self._live.clear()
