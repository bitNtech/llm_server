"""Stateful conversation turns: context fitting, streaming, tool loop, barge-in.

One `response` = one user-visible turn. It may span several engine rounds when the model calls
server-side tools (executed here, then the model continues) and ends in one of:

  completed        the model finished speaking
  incomplete       max_tokens or MAX_TOOL_ROUNDS reached
  requires_action  the model called client-side tools; the client must submit results
  cancelled        barge-in / explicit cancel / disconnect / shutdown
  failed           engine or validation error (see `error`)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import orjson

from app.config import Settings
from app.core.admission import AdmissionController
from app.core.errors import BadRequest, Conflict, EngineTimeout, EngineUnavailable, GatewayError
from app.core.generation import Event, Generation, GenerationRegistry
from app.core.params import apply_defaults, validate_session_params
from app.core.segmenter import SentenceSegmenter
from app.core.toolcalls import ToolCall, ToolCallAccumulator
from app.engine.client import EngineClient
from app.engine.monitor import EngineMonitor
from app.observability.logging import get_logger, session_id_var
from app.observability.metrics import GENERATION_DURATION, GENERATIONS, INTER_TOKEN, TOKENS, TOOL_CALLS, TTFT
from app.sessions.context import ContextManager, rendered_chars
from app.sessions.store import Session, SessionStore, new_id
from app.tools.registry import ToolRegistry

log = get_logger("app.conversation")

ABANDONED_TOOL_RESULT = "No result: the conversation moved on before this tool call completed."


def normalize_input(value: Any) -> list[dict[str, Any]]:
    """`input` may be a string (user utterance), one message, or a list of messages."""
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "content": value}] if value.strip() else []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(v, dict) for v in value):
        return value
    raise BadRequest("input must be a string, a message object or a list of messages")


@dataclass
class TurnState:
    started: float = field(default_factory=time.perf_counter)
    queue_s: float = 0.0
    ttft_s: float | None = None
    round_text: str = ""
    text: str = ""
    rounds: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    prompt_tokens_last: int = 0
    completion_tokens_last: int = 0
    finish_reason: str | None = None

    def first_token(self) -> None:
        if self.ttft_s is None:
            self.ttft_s = time.perf_counter() - self.started
            TTFT.labels("session").observe(self.ttft_s)

    def usage(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "cached_tokens": self.cached_tokens}

    def timings(self) -> dict[str, float | None]:
        ms = lambda v: round(v * 1000, 1) if v is not None else None  # noqa: E731
        return {"queue_ms": ms(self.queue_s), "ttft_ms": ms(self.ttft_s),
                "total_ms": ms(time.perf_counter() - self.started)}


class ConversationService:
    def __init__(
        self,
        settings: Settings,
        engine: EngineClient,
        monitor: EngineMonitor,
        admission: AdmissionController,
        registry: GenerationRegistry,
        tools: ToolRegistry,
        context: ContextManager,
        store: SessionStore,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.monitor = monitor
        self.admission = admission
        self.registry = registry
        self.tools = tools
        self.context = context
        self.store = store

    # ------------------------------------------------------------- sessions
    def create_session(self, spec: dict[str, Any]) -> Session:
        fields = self._session_fields(spec, creating=True)
        return self.store.create(session_id=spec.get("session_id"), **fields)

    def update_session(self, session: Session, spec: dict[str, Any]) -> Session:
        for key, value in self._session_fields(spec, creating=False).items():
            setattr(session, key, value)
        return session

    def _session_fields(self, spec: dict[str, Any], creating: bool) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if "system_prompt" in spec and spec["system_prompt"] is not None:
            if not isinstance(spec["system_prompt"], str):
                raise BadRequest("system_prompt must be a string")
            if len(spec["system_prompt"]) > self.settings.max_input_chars * 4:
                raise BadRequest("system_prompt is too long")
            out["system_prompt"] = spec["system_prompt"]
        elif creating:
            out["system_prompt"] = self.settings.system_prompt
        if "tools" in spec and spec["tools"] is not None:
            out["tools"] = self._validate_tools(spec["tools"])
        if "server_tools" in spec and spec["server_tools"] is not None:
            names = spec["server_tools"]
            if names == "all":
                names = self.tools.names()
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                raise BadRequest("server_tools must be a list of tool names or 'all'")
            unknown = [n for n in names if not self.tools.has(n)]
            if unknown:
                raise BadRequest(f"unknown server tools: {', '.join(unknown)}",
                                 details={"available": self.tools.names()})
            out["server_tools"] = names
        if "params" in spec and spec["params"] is not None:
            out["params"] = validate_session_params(spec["params"])
        if "metadata" in spec and spec["metadata"] is not None:
            if not isinstance(spec["metadata"], dict) or len(orjson.dumps(spec["metadata"])) > 16_384:
                raise BadRequest("metadata must be an object under 16 KiB")
            out["metadata"] = spec["metadata"]
        for flag in ("segmentation", "include_reasoning"):
            if flag in spec and spec[flag] is not None:
                out[flag] = bool(spec[flag])
        return out

    @staticmethod
    def _validate_tools(tools: Any) -> list[dict[str, Any]]:
        if not isinstance(tools, list):
            raise BadRequest("tools must be a list of OpenAI function tools")
        for t in tools:
            fn = t.get("function") if isinstance(t, dict) else None
            if not isinstance(t, dict) or t.get("type") != "function" or not isinstance(fn, dict) \
                    or not isinstance(fn.get("name"), str) or not fn["name"]:
                raise BadRequest('each tool must look like {"type": "function", "function": {"name": ...}}')
        return tools

    # ------------------------------------------------------------- requests
    def build_request(self, session: Session, overrides: dict[str, Any]) -> tuple[dict, list | None]:
        params = {**session.params, **validate_session_params(overrides.get("params"))}
        params = apply_defaults(self.settings, params)
        tools = list(overrides["tools"]) if overrides.get("tools") is not None else list(session.tools)
        if overrides.get("tools") is not None:
            self._validate_tools(tools)
        tools += self.tools.schemas(session.server_tools)
        if not tools:
            params.pop("tool_choice", None)
            params.pop("parallel_tool_calls", None)
        return params, (tools or None)

    async def ensure_engine(self) -> None:
        if not self.monitor.state.ready and not await self.monitor.check():
            raise EngineUnavailable(f"engine not ready: {self.monitor.state.last_error}", retry_after=2)

    # ------------------------------------------------------------- responses
    def start_response(
        self,
        session: Session,
        input_value: Any = None,
        overrides: dict[str, Any] | None = None,
        response_id: str | None = None,
    ) -> Generation:
        overrides = overrides or {}
        inputs = normalize_input(input_value)
        for raw in inputs:  # validate before anything is queued
            if raw.get("role") in ("system", "developer") or raw.get("role") not in ("user", "assistant", "tool"):
                raise BadRequest(f"unsupported input role {raw.get('role')!r}")
        self.store.check_capacity(session, len(inputs) + 2)
        if not inputs and not any(m.role == "user" for m in session.messages):
            raise BadRequest("the conversation has no user message yet; send input")
        if response_id and self.registry.get(response_id):
            raise Conflict(f"response id {response_id} is already in use")
        # session.active is always the newest response (running, or queued behind the session lock),
        # so a rapid sequence of response.create calls cancels every superseded response.
        active = session.active
        if active is not None and not active.done:
            if self.settings.ws_barge_in == "reject":
                raise Conflict(f"response {active.id} is still running", code="response_in_progress")
            active.cancel("barge_in")
        gen = Generation(response_id or new_id("resp"), api="session", session_id=session.id)
        session.active = gen

        def superseded() -> Event:
            if session.active is gen:
                session.active = None
            gen.status = "cancelled"
            GENERATIONS.labels("session", "cancelled").inc()
            return Event("response.done", {
                "response_id": gen.id, "session_id": session.id, "status": "cancelled",
                "finish_reason": "cancelled", "cancel_reason": gen.cancel_reason, "text": "",
                "usage": TurnState().usage(), "timings": {"queue_ms": 0.0, "ttft_ms": None, "total_ms": 0.0}})

        gen.start(self._run(gen, session, inputs, overrides), self.registry, on_never_started=superseded)
        return gen

    async def cancel_response(
        self,
        session: Session,
        response_id: str | None = None,
        spoken_text: str | None = None,
        discard: bool = False,
    ) -> bool:
        """Cancel the running response.

        spoken_text: keep only what the caller heard of the reply (barge-in).
        discard:     drop the whole turn (its input and partial reply), e.g. a speculative response
                     started on an interim transcript that turned out different.
        """
        gen = session.active
        if gen is None or (response_id and gen.id != response_id):
            return False
        cancelled = gen.cancel("client_cancel")
        await gen.wait(timeout=5)
        async with session.lock:
            if discard and gen.base_seq is not None:
                session.messages = [m for m in session.messages if m.seq <= gen.base_seq]
                session.pending_tool_calls.clear()
            elif spoken_text is not None:
                session.truncate_last_assistant(spoken_text)
        return cancelled

    async def truncate(self, session: Session, spoken_text: str) -> bool:
        async with session.lock:
            return session.truncate_last_assistant(spoken_text)

    async def submit_tool_result(self, session: Session, call_id: str, output: Any, is_error: bool = False) -> int:
        async with session.lock:
            name = session.pending_tool_calls.pop(call_id, None)
            if name is None:
                raise BadRequest(f"no pending tool call {call_id!r}", code="unknown_tool_call")
            text = output if isinstance(output, str) else orjson.dumps(output, default=str).decode()
            if len(text) > self.settings.max_input_chars:
                raise BadRequest("tool output exceeds MAX_INPUT_CHARS")
            session.add("tool", text, tool_call_id=call_id, name=name)
            TOOL_CALLS.labels(name, "client", "error" if is_error else "ok").inc()
            return len(session.pending_tool_calls)

    # ---------------------------------------------------------------- engine
    async def _run(self, gen: Generation, session: Session, inputs: list[dict], overrides: dict) -> None:
        session_id_var.set(session.id)
        state = TurnState()
        status, error, pending_calls = "completed", None, []
        seg_enabled = overrides.get("segmentation")
        segmenter = (
            SentenceSegmenter(self.settings.segment_min_chars, self.settings.segment_max_chars)
            if (session.segmentation if seg_enabled is None else seg_enabled) else None
        )
        gen.emit(Event("response.created", {"response_id": gen.id, "session_id": session.id}))
        max_tokens = self.settings.default_max_tokens
        try:
            async with session.lock:
                gen.base_seq = session.last_seq
                try:
                    async with asyncio.timeout(self.settings.generation_timeout_s):
                        await self.ensure_engine()
                        if session.pending_tool_calls and any(m.get("role") != "tool" for m in inputs):
                            for call_id, name in session.pending_tool_calls.items():
                                session.add("tool", ABANDONED_TOOL_RESULT, tool_call_id=call_id, name=name)
                            session.pending_tool_calls.clear()
                        for raw in inputs:
                            session.add_openai(raw, self.settings.max_input_chars)
                        params, tools = self.build_request(session, overrides)
                        max_tokens = params["max_tokens"]
                        status, pending_calls = await self._rounds(gen, session, params, tools, state, segmenter)
                except asyncio.CancelledError:
                    status = "cancelled"
                except TimeoutError:
                    status = "failed"
                    limit = self.settings.generation_timeout_s
                    error = EngineTimeout(f"response exceeded GENERATION_TIMEOUT_S={limit}")
                except GatewayError as exc:
                    status, error = "failed", exc
                finally:
                    if status in ("cancelled", "failed"):
                        self._settle_interrupted(session, state)
                    if session.active is gen:
                        session.active = None
                    session.touch()
        except asyncio.CancelledError:  # cancelled while queued behind the session lock
            status = "cancelled"
            if session.active is gen:
                session.active = None

        if segmenter is not None and status != "cancelled":
            tail = segmenter.flush()
            if tail:
                gen.emit(Event("response.text.segment", {"response_id": gen.id, "text": tail,
                                                         "index": segmenter.count - 1}))
        gen.status, gen.error = status, error
        if status == "completed" and state.finish_reason == "length":
            status = gen.status = "incomplete"
        stats = session.stats
        stats.turns += 1
        stats.prompt_tokens += state.prompt_tokens
        stats.completion_tokens += state.completion_tokens
        stats.cached_prompt_tokens += state.cached_tokens
        done: dict[str, Any] = {
            "response_id": gen.id,
            "session_id": session.id,
            "status": status,
            "finish_reason": state.finish_reason if status != "cancelled" else "cancelled",
            "text": state.text,
            "usage": state.usage(),
            "timings": state.timings(),
        }
        if pending_calls:
            done["tool_calls"] = [{"call_id": c.id, "name": c.name, "arguments": c.arguments}
                                  for c in pending_calls]
        if error is not None:
            done["error"] = error.to_dict()["error"]
            log.warning("response failed", extra={"response_id": gen.id, "error": error.message})
        if status == "cancelled" and gen.cancel_reason:
            done["cancel_reason"] = gen.cancel_reason
        GENERATIONS.labels("session", status).inc()
        GENERATION_DURATION.labels("session", status).observe(time.perf_counter() - state.started)
        gen.emit(Event("response.done", done))
        if status in ("completed", "incomplete"):
            self.context.after_turn(session, max_tokens)

    @staticmethod
    def _settle_interrupted(session: Session, state: TurnState) -> None:
        """Keep history valid after an interruption: store the partial reply and give every
        tool call of the last assistant message a result (the chat template expects pairs)."""
        if state.round_text:
            session.add("assistant", state.round_text, interrupted=True)
            state.round_text = ""
        answered = set(session.pending_tool_calls)
        for msg in reversed(session.messages):
            if msg.role == "tool" and msg.tool_call_id:
                answered.add(msg.tool_call_id)
            elif msg.role == "assistant" and msg.tool_calls:
                for call in msg.tool_calls:
                    if call["id"] not in answered:
                        session.add("tool", ABANDONED_TOOL_RESULT, tool_call_id=call["id"],
                                    name=call["function"]["name"])
                break
            elif msg.role in ("user", "assistant"):
                break

    async def _rounds(
        self,
        gen: Generation,
        session: Session,
        params: dict[str, Any],
        tools: list | None,
        state: TurnState,
        segmenter: SentenceSegmenter | None,
    ) -> tuple[str, list[ToolCall]]:
        model = self.monitor.state.model or self.settings.served_model_name
        allowed_server = set(session.server_tools)
        for round_no in range(self.settings.max_tool_rounds + 1):
            state.rounds += 1
            messages = await self.context.prepare(session, params["max_tokens"], tools, gen.id)
            body: dict[str, Any] = {"model": model, "messages": messages, **params,
                                    "stream_options": {"include_usage": True}}
            if tools:
                body["tools"] = tools
            chars = rendered_chars(session, tools)
            calls = await self._stream_round(gen, session, body, state, segmenter, f"{gen.id}-{round_no}")
            session.calibrate(chars, state.prompt_tokens_last)
            session.stats.last_prompt_tokens = state.prompt_tokens_last + state.completion_tokens_last
            if segmenter is not None:
                tail = segmenter.flush()
                if tail:
                    gen.emit(Event("response.text.segment", {"response_id": gen.id, "text": tail,
                                                             "index": segmenter.count - 1}))
            round_text, state.round_text = state.round_text, ""
            if not calls:
                session.add("assistant", round_text)
                return "completed", []

            session.add("assistant", round_text or None, tool_calls=[c.to_message() for c in calls])
            server_calls = [c for c in calls if c.name in allowed_server]
            client_calls = [c for c in calls if c.name not in allowed_server]
            for c in calls:
                gen.emit(Event("response.tool_call", {
                    "response_id": gen.id, "call_id": c.id, "name": c.name, "arguments": c.arguments,
                    "side": "server" if c in server_calls else "client"}))
            results = await asyncio.gather(*(self.tools.execute(c.name, c.arguments) for c in server_calls))
            for c, (output, is_error) in zip(server_calls, results, strict=True):
                session.add("tool", output, tool_call_id=c.id, name=c.name)
                gen.emit(Event("response.tool_result", {"response_id": gen.id, "call_id": c.id,
                                                        "name": c.name, "output": output, "is_error": is_error}))
            if client_calls:
                for c in client_calls:
                    session.pending_tool_calls[c.id] = c.name
                return "requires_action", client_calls
            if round_no == self.settings.max_tool_rounds:
                state.finish_reason = "max_tool_rounds"
                return "incomplete", []
        return "incomplete", []

    async def _stream_round(
        self,
        gen: Generation,
        session: Session,
        body: dict[str, Any],
        state: TurnState,
        segmenter: SentenceSegmenter | None,
        request_id: str,
    ) -> list[ToolCall]:
        acc = ToolCallAccumulator()
        state.prompt_tokens_last = state.completion_tokens_last = 0
        async with self.admission.slot() as waited:
            state.queue_s += waited
            async with self.engine.stream_chat(body, request_id) as stream:
                last: float | None = None
                async for chunk in stream:
                    now = time.perf_counter()
                    if last is not None:
                        INTER_TOKEN.labels("session").observe(now - last)
                    last = now
                    usage = chunk.get("usage")
                    if usage:
                        self._account(state, usage)
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        if reasoning and session.include_reasoning:
                            gen.emit(Event("response.reasoning.delta", {"response_id": gen.id, "delta": reasoning}))
                        content = delta.get("content")
                        if content:
                            state.first_token()
                            state.round_text += content
                            state.text += content
                            gen.emit(Event("response.text.delta", {"response_id": gen.id, "delta": content}))
                            if segmenter is not None:
                                for seg in segmenter.push(content):
                                    gen.emit(Event("response.text.segment", {
                                        "response_id": gen.id, "text": seg, "index": segmenter.count - 1}))
                        if delta.get("tool_calls"):
                            state.first_token()
                            for call in acc.add(delta["tool_calls"]):
                                gen.emit(Event("response.tool_call.started", {
                                    "response_id": gen.id, "call_id": call.id, "name": call.name}))
                        if choice.get("finish_reason"):
                            state.finish_reason = choice["finish_reason"]
        return acc.finish()

    @staticmethod
    def _account(state: TurnState, usage: dict[str, Any]) -> None:
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        state.prompt_tokens += prompt
        state.completion_tokens += completion
        state.cached_tokens += cached
        state.prompt_tokens_last = prompt
        state.completion_tokens_last = completion
        TOKENS.labels("prompt").inc(prompt)
        TOKENS.labels("completion").inc(completion)
        TOKENS.labels("cached_prompt").inc(cached)
