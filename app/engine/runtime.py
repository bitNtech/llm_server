"""Conversation runtime: one `response` = one user-visible assistant turn.

A response may span several model rounds:

    input -> context window -> stream from Vertex (text deltas, sentence segments, tool calls)
      no tool calls          -> done (completed / incomplete on max_tokens)
      server tools           -> run in parallel (builtin / HTTP / MCP), results injected, next round
      handoff tool           -> active agent switches, the new agent continues in the next round
      client tools           -> status requires_action; the caller runs them and submits results,
                                which continues the turn in a new response
      MAX_TOOL_ROUNDS        -> incomplete

Interruptions (barge-in, cancel, disconnect, timeout, shutdown) cancel the producer task: the
Vertex stream is closed (generation stops upstream), read-only tool calls are cancelled, tools with
side effects run to completion in the background (their result is recorded for idempotency), and
the history is settled so it stays valid: the partial reply is kept (or truncated to what the
caller heard) and every tool call gets a result.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import orjson

from app.config import Settings
from app.core.errors import BadRequest, Conflict, GatewayError, ProviderTimeout
from app.core.ids import new_id, utcnow
from app.core.logging import get_logger, session_id_var, tenant_id_var
from app.core.metrics import GENERATION_DURATION, GENERATIONS, HANDOFFS, INTER_TOKEN, TTFT
from app.core.tracing import span
from app.engine.admission import AdmissionController
from app.engine.agents import AgentResolver, ResolvedAgent, Toolset, validate_params
from app.engine.context import ContextBuilder
from app.engine.generation import Event, Generation, GenerationRegistry
from app.engine.segmenter import SentenceSegmenter
from app.engine.sessions import LiveSession, SessionManager
from app.engine.toolcalls import ToolCall, ToolCallAccumulator
from app.providers.vertex import VertexProvider
from app.services.usage import UsageRecorder
from app.tools.base import ToolContext, ToolOutcome, ToolSpec
from app.tools.executor import ToolExecutor

log = get_logger("app.runtime")

INTERRUPTED_RESULT = "Interrupted before the result arrived; the action may or may not have completed."
TRANSFER_SKIPPED = "Not executed: the conversation was transferred to another agent."


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
class TurnOptions:
    input: Any = None
    params: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None  # extra client tools for this response
    instructions: str | None = None  # extra instructions for this response only
    tool_choice: Any = None
    segmentation: bool | None = None
    say: str | None = None  # speak this exact text, no model call
    response_id: str | None = None
    allow_client_tools: bool = True
    allow_handoffs: bool = True
    api: str = "session"


@dataclass
class TurnState:
    started: float = field(default_factory=time.perf_counter)
    queue_s: float = 0.0
    ttft_s: float | None = None
    upstream_ttft_s: float | None = None
    round_text: str = ""
    round_extra: dict[str, Any] | None = None
    text: str = ""
    rounds: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    prompt_tokens_last: int = 0
    finish_reason: str | None = None
    model: str | None = None
    location: str | None = None
    tool_calls: int = 0
    handoffs: int = 0
    end_conversation: dict[str, Any] | None = None
    server_tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def first_token(self, api: str) -> None:
        if self.ttft_s is None:
            self.ttft_s = time.perf_counter() - self.started
            TTFT.labels(api).observe(self.ttft_s)

    def usage(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens, "cached_tokens": self.cached_tokens,
                "reasoning_tokens": self.reasoning_tokens}

    def timings(self) -> dict[str, float | None]:
        def ms(v: float | None) -> float | None:
            return round(v * 1000, 1) if v is not None else None
        overhead = None
        if self.ttft_s is not None and self.upstream_ttft_s is not None:
            overhead = max(0.0, self.ttft_s - self.upstream_ttft_s - self.queue_s)
        return {"queue_ms": ms(self.queue_s), "ttft_ms": ms(self.ttft_s), "upstream_ttft_ms": ms(self.upstream_ttft_s),
                "gateway_overhead_ms": ms(overhead), "total_ms": ms(time.perf_counter() - self.started)}


class ConversationRuntime:
    def __init__(self, settings: Settings, provider: VertexProvider, admission: AdmissionController,
                 registry: GenerationRegistry, executor: ToolExecutor, resolver: AgentResolver,
                 context: ContextBuilder, sessions: SessionManager, usage: UsageRecorder) -> None:
        self.settings = settings
        self.provider = provider
        self.admission = admission
        self.registry = registry
        self.executor = executor
        self.resolver = resolver
        self.context = context
        self.sessions = sessions
        self.usage = usage
        self._background: set[asyncio.Task] = set()

    # ---------------------------------------------------------------- agents
    async def current_agent(self, session: LiveSession) -> ResolvedAgent:
        if session.current_agent_id:
            return await self.resolver.resolve(session.tenant_id, session.current_agent_id,
                                               session.current_agent_version)
        return self.resolver.adhoc(None, session.model)

    async def handoff(self, session: LiveSession, agent_id: str, reason: str, source: str,
                      response_id: str | None = None) -> tuple[ResolvedAgent, ResolvedAgent]:
        """Switch the session's active agent (the conversation history is kept)."""
        previous = await self.current_agent(session)
        target = await self.resolver.resolve(session.tenant_id, agent_id)
        session.current_agent_id, session.current_agent_version = target.id, target.version
        session.stats.handoffs += 1
        self.sessions.save_state(session)
        HANDOFFS.labels(source).inc()
        log.info("agent handoff", extra={"from_agent": previous.id, "to_agent": target.id, "source": source})
        return previous, target

    @staticmethod
    def handoff_event(session: LiveSession, previous: ResolvedAgent, target: ResolvedAgent, reason: str,
                      source: str, response_id: str | None) -> Event:
        return Event("agent.handoff", {"session_id": session.id, "response_id": response_id,
                                       "from_agent": previous.public(), "to_agent": target.public(),
                                       "reason": reason, "source": source})

    # ------------------------------------------------------------- responses
    def start_response(self, session: LiveSession, opts: TurnOptions) -> Generation:
        if session.status != "active":
            raise Conflict(f"session {session.id} has ended", code="session_ended")
        self._check_current(session)
        inputs = normalize_input(opts.input)
        for raw in inputs:
            if raw.get("role") not in ("user", "assistant", "tool"):
                raise BadRequest(f"unsupported input role {raw.get('role')!r}")
        if opts.say is not None and (not isinstance(opts.say, str) or not opts.say.strip()):
            raise BadRequest("say must be non-empty text")
        if len(session.messages) + len(inputs) + 2 > self.settings.max_messages_per_session:
            raise BadRequest(f"session reached MAX_MESSAGES_PER_SESSION={self.settings.max_messages_per_session}",
                             code="session_full")
        if opts.params:
            validate_params(opts.params, self.settings.max_tokens_limit)
        response_id = opts.response_id or new_id("resp")
        if self.registry.get(session.tenant_id, response_id):
            raise Conflict(f"response id {response_id} is already in use")
        # session.active is always the newest response, so rapid response.create calls cancel every
        # superseded response (barge-in) instead of queueing them
        active = session.active
        if active is not None and not active.done:
            if self.settings.barge_in == "reject":
                raise Conflict(f"response {active.id} is still running", code="response_in_progress")
            active.cancel("barge_in")
        gen = Generation(response_id, api=opts.api, tenant_id=session.tenant_id, session_id=session.id)
        session.active = gen

        def superseded() -> Event:
            if session.active is gen:
                session.active = None
            gen.status = "cancelled"
            GENERATIONS.labels(opts.api, "cancelled").inc()
            return Event("response.done", {
                "response_id": gen.id, "session_id": session.id, "status": "cancelled",
                "finish_reason": "cancelled", "cancel_reason": gen.cancel_reason, "text": "",
                "usage": TurnState().usage(), "timings": {"total_ms": 0.0}})

        gen.start(self._run(gen, session, inputs, opts), self.registry, on_never_started=superseded)
        return gen

    async def cancel_response(self, session: LiveSession, response_id: str | None = None,
                              spoken_text: str | None = None, discard: bool = False) -> bool:
        """Cancel the running response.

        spoken_text: keep only what the caller heard of the reply (barge-in).
        discard:     drop the whole turn (input and partial reply), e.g. a speculative response
                     started on an interim transcript that turned out different.
        """
        gen = session.active
        if gen is None and discard and response_id and session.last_response is not None \
                and session.last_response.id == response_id:
            gen = session.last_response  # the speculative reply already finished: still drop the turn
        if gen is None or (response_id and gen.id != response_id):
            return False
        cancelled = gen.cancel("client_cancel") or (discard and gen.done)
        await gen.wait(timeout=5)
        async with session.lock:
            if discard and gen.base_seq is not None and session.last_response is gen:
                session.drop_after(gen.base_seq)
                # calls the dropped turn had answered (or abandoned) are open again
                session.pending_tool_calls = session.unanswered_tool_calls()
                session.last_response = None
            elif spoken_text is not None:
                session.truncate_last_assistant(spoken_text)
            self.sessions.save_state(session)
        return cancelled

    @staticmethod
    def _check_current(session: LiveSession) -> None:
        if session.stale:
            raise Conflict(f"session {session.id} moved to another instance; reconnect",
                           code="session_busy_elsewhere")

    async def truncate(self, session: LiveSession, spoken_text: str) -> bool:
        async with session.lock:
            return session.truncate_last_assistant(spoken_text)

    async def submit_tool_result(self, session: LiveSession, call_id: str, output: Any, is_error: bool) -> int:
        async with session.lock:
            self._check_current(session)
            name = session.pending_tool_calls.get(call_id)
            if name is None:
                raise BadRequest(f"no pending tool call {call_id!r}", code="unknown_tool_call",
                                 details={"pending": list(session.pending_tool_calls)})
            text = output if isinstance(output, str) else orjson.dumps(output, default=str).decode()
            if len(text) > self.settings.max_input_chars:
                raise BadRequest("tool output exceeds MAX_INPUT_CHARS")
            del session.pending_tool_calls[call_id]
            session.add("tool", text, tool_call_id=call_id, name=name)
            session.stats.tool_calls += 1
            self.sessions.save_state(session)
            return len(session.pending_tool_calls)

    # ----------------------------------------------------------------- core
    async def _run(self, gen: Generation, session: LiveSession, inputs: list[dict[str, Any]],
                   opts: TurnOptions) -> None:
        tenant_id_var.set(session.tenant_id)
        session_id_var.set(session.id)
        state = TurnState()
        status: str = "completed"
        error: GatewayError | None = None
        pending: list[ToolCall] = []
        agent: ResolvedAgent | None = None
        segmenter: SentenceSegmenter | None = None
        gen.emit(Event("response.created", {"response_id": gen.id, "session_id": session.id,
                                            "agent_id": session.current_agent_id}))
        with span("response", **{"response.id": gen.id, "session.id": session.id, "tenant.id": session.tenant_id,
                                 "api": opts.api}) as sp:
            try:
                async with session.lock:
                    self._check_current(session)
                    gen.base_seq = session.last_seq
                    session.last_response = gen
                    try:
                        async with asyncio.timeout(self.settings.generation_timeout_s):
                            if opts.say is not None or any(m.get("role") != "tool" for m in inputs):
                                session.abandon_open_calls()
                            for raw in inputs:
                                session.add_openai(raw, self.settings.max_input_chars)
                            agent = await self.current_agent(session)
                            seg_on = opts.segmentation if opts.segmentation is not None else (
                                session.segmentation if session.segmentation is not None else agent.segmentation)
                            if seg_on:
                                segmenter = SentenceSegmenter(self.settings.segment_min_chars,
                                                              self.settings.segment_max_chars)
                            if opts.say is not None:
                                self._say(gen, session, agent, opts.say, state, segmenter)
                            else:
                                status, pending, agent = await self._rounds(gen, session, agent, opts, state,
                                                                            segmenter)
                    except asyncio.CancelledError:
                        status = "cancelled"
                    except TimeoutError:
                        status = "failed"
                        error = ProviderTimeout(
                            f"response exceeded GENERATION_TIMEOUT_S={self.settings.generation_timeout_s}")
                    except GatewayError as exc:
                        status, error = "failed", exc
                    except Exception as exc:
                        log.exception("response failed", extra={"response_id": gen.id})
                        status, error = "failed", GatewayError(f"internal error: {exc.__class__.__name__}")
                    finally:
                        if status in ("cancelled", "failed"):
                            self._settle_interrupted(session, state, agent)
                        if session.active is gen:
                            session.active = None
                        session.touch()
            except asyncio.CancelledError:  # cancelled while queued behind the session lock
                status = "cancelled"
                if session.active is gen:
                    session.active = None
            sp.set_attribute("response.status", status)

        if segmenter is not None and status != "cancelled":
            self._emit_segment(gen, segmenter, segmenter.flush())
        if status == "completed" and state.finish_reason == "length":
            status = "incomplete"
        gen.status, gen.error = status, error
        stats = session.stats
        stats.turns += 1
        stats.prompt_tokens += state.prompt_tokens
        stats.completion_tokens += state.completion_tokens
        stats.cached_tokens += state.cached_tokens
        stats.tool_calls += state.tool_calls
        if state.rounds:
            self.usage.record(session.tenant_id, api=opts.api, model=state.model, agent_id=session.current_agent_id,
                              usage={"prompt_tokens": state.prompt_tokens, "completion_tokens": state.completion_tokens,
                                     "prompt_tokens_details": {"cached_tokens": state.cached_tokens},
                                     "completion_tokens_details": {"reasoning_tokens": state.reasoning_tokens}},
                              tool_calls=state.tool_calls)
        done: dict[str, Any] = {
            "response_id": gen.id,
            "session_id": session.id,
            "status": status,
            "finish_reason": "cancelled" if status == "cancelled" else state.finish_reason,
            "text": state.text,
            "agent": agent.public() if agent else None,
            "model": state.model,
            "usage": state.usage(),
            "timings": state.timings(),
        }
        if pending:
            done["tool_calls"] = [{"call_id": c.id, "name": c.name, "arguments": c.arguments} for c in pending]
        if state.server_tool_calls:
            done["server_tool_calls"] = state.server_tool_calls
        if state.end_conversation:
            done["end_conversation"] = state.end_conversation
        if error is not None:
            done["error"] = error.to_dict()["error"]
            log.warning("response failed", extra={"response_id": gen.id, "code": error.code, "error": error.message})
        if status == "cancelled" and gen.cancel_reason:
            done["cancel_reason"] = gen.cancel_reason
        gen.result = done
        GENERATIONS.labels(opts.api, status).inc()
        GENERATION_DURATION.labels(opts.api, status).observe(time.perf_counter() - state.started)
        record = {k: v for k, v in done.items() if k != "session_id"}
        record.update({"id": gen.id, "created_at": utcnow(), "location": state.location, "api": opts.api})
        record["text"] = state.text[:20_000]
        self.sessions.save_response(session, record)
        self.sessions.save_state(session)
        gen.emit(Event("response.done", done))

    def _say(self, gen: Generation, session: LiveSession, agent: ResolvedAgent, text: str, state: TurnState,
             segmenter: SentenceSegmenter | None) -> None:
        state.first_token("say")
        state.text = text
        state.finish_reason = "say"
        gen.emit(Event("response.text.delta", {"response_id": gen.id, "delta": text}))
        if segmenter is not None:
            for seg in segmenter.push(text):
                self._emit_segment(gen, segmenter, seg)
        session.add("assistant", text, agent_id=agent.id)

    def _emit_segment(self, gen: Generation, segmenter: SentenceSegmenter, text: str | None) -> None:
        if text:
            gen.emit(Event("response.text.segment", {"response_id": gen.id, "text": text,
                                                     "index": segmenter.count - 1}))

    def _settle_interrupted(self, session: LiveSession, state: TurnState, agent: ResolvedAgent | None) -> None:
        """Keep history valid after an interruption: store the partial reply and give every tool
        call of the last assistant message a result (Gemini needs call/response pairs)."""
        if state.round_text:
            session.add("assistant", state.round_text, interrupted=True, agent_id=agent.id if agent else None,
                        model=state.model, extra_content=state.round_extra)
            state.round_text = ""
        answered = set(session.pending_tool_calls)
        for msg in reversed(session.messages):
            if msg.role == "tool" and msg.tool_call_id:
                answered.add(msg.tool_call_id)
            elif msg.role == "assistant" and msg.tool_calls:
                for call in msg.tool_calls:
                    if call["id"] not in answered:
                        session.add("tool", INTERRUPTED_RESULT, tool_call_id=call["id"], name=call["function"]["name"])
                break
            elif msg.role in ("user", "assistant"):
                break

    def _params(self, session: LiveSession, agent: ResolvedAgent, opts: TurnOptions) -> dict[str, Any]:
        params = {**agent.params, **session.params, **(opts.params or {})}
        params.setdefault("max_tokens", self.settings.default_max_tokens)
        if params["max_tokens"] > self.settings.max_tokens_limit:
            raise BadRequest(f"max_tokens exceeds MAX_TOKENS_LIMIT={self.settings.max_tokens_limit}")
        if self.settings.default_temperature is not None:
            params.setdefault("temperature", self.settings.default_temperature)
        return params

    async def _rounds(self, gen: Generation, session: LiveSession, agent: ResolvedAgent, opts: TurnOptions,
                      state: TurnState, segmenter: SentenceSegmenter | None
                      ) -> tuple[str, list[ToolCall], ResolvedAgent]:
        max_rounds = agent.max_tool_rounds or self.settings.max_tool_rounds
        handoffs = 0
        ctx = ToolContext(tenant_id=session.tenant_id, scope_id=session.id, session_id=session.id)
        for round_no in range(max_rounds + 1):
            params = self._params(session, agent, opts)
            toolset = await self.resolver.toolset(session.tenant_id, agent, [*session.tools, *(opts.tools or [])],
                                                  allow_client=opts.allow_client_tools,
                                                  allow_handoffs=opts.allow_handoffs
                                                  and handoffs < self.settings.max_handoffs_per_turn)
            messages, chars = self.context.build(session, agent.instructions, toolset.chars(), params["max_tokens"],
                                                 opts.instructions)
            choice = opts.tool_choice if round_no == 0 else None
            if round_no > 0 and params.get("tool_choice") not in (None, "auto", "none"):
                params.pop("tool_choice")  # a forced choice applies to the first round only
            body: dict[str, Any] = {"model": agent.model, "messages": messages, **params}
            if toolset.specs:
                body["tools"] = toolset.schemas()
                if choice is not None:
                    body["tool_choice"] = choice
            else:
                body.pop("tool_choice", None)
            state.rounds += 1
            calls = await self._stream_round(gen, session, body, state, segmenter, toolset, opts.api,
                                             request_id=f"{gen.id}-{round_no}")
            session.calibrate(chars, state.prompt_tokens_last)
            session.stats.last_prompt_tokens = state.prompt_tokens_last
            if segmenter is not None:  # speak what was said before the tool call while it runs
                self._emit_segment(gen, segmenter, segmenter.flush())
            round_text, state.round_text = state.round_text, ""
            extra, state.round_extra = state.round_extra, None
            if not calls:
                session.add("assistant", round_text, agent_id=agent.id, model=state.model, extra_content=extra)
                return "completed", [], agent

            session.add("assistant", round_text or None, tool_calls=[c.to_message() for c in calls],
                        agent_id=agent.id, model=state.model, extra_content=extra)
            state.tool_calls += len(calls)
            specs: dict[str, ToolSpec | None] = {c.id: toolset.get(c.name) for c in calls}
            handoff_call = next((c for c in calls if (s := specs[c.id]) and s.kind == "handoff"), None)
            for c in calls:
                spec = specs[c.id]
                gen.emit(Event("response.tool_call", {
                    "response_id": gen.id, "call_id": c.id, "name": c.name, "arguments": c.arguments,
                    "kind": spec.kind if spec else "unknown",
                    "side": "client" if spec and spec.kind == "client" else ("handoff" if spec and spec.kind ==
                                                                             "handoff" else "server")}))
            server = [c for c in calls if (s := specs[c.id]) is not None and s.server_side]
            outcomes = dict(zip([c.id for c in server], await asyncio.gather(
                *(self._execute(ctx, agent, specs[c.id], c) for c in server)), strict=True))

            client_pending: list[ToolCall] = []
            switched: ResolvedAgent | None = None
            for c in calls:
                spec = specs[c.id]
                if spec is None:
                    session.add("tool", f"unknown tool {c.name!r}; use only the tools provided",
                                tool_call_id=c.id, name=c.name)
                elif spec.server_side:
                    outcome = outcomes[c.id]
                    session.add("tool", outcome.output, tool_call_id=c.id, name=c.name)
                    state.server_tool_calls.append({"call_id": c.id, "name": c.name, "arguments": c.arguments,
                                                    "output": outcome.output, "is_error": outcome.is_error})
                    gen.emit(Event("response.tool_result", {
                        "response_id": gen.id, "call_id": c.id, "name": c.name, "output": outcome.output,
                        "is_error": outcome.is_error, "duration_ms": outcome.duration_ms,
                        "replayed": outcome.replayed}))
                    if outcome.control and outcome.control.get("end_conversation"):
                        state.end_conversation = outcome.control["end_conversation"]
                        gen.emit(Event("session.end_requested", {"session_id": session.id, "response_id": gen.id,
                                                                 **state.end_conversation}))
                elif spec.kind == "handoff":
                    if c is handoff_call:
                        reason = _argument(c.arguments, "reason")
                        previous, switched = await self.handoff(session, spec.target["agent_id"], reason, "model",
                                                                gen.id)
                        handoffs += 1
                        state.handoffs += 1
                        session.add("tool", f"Transferred to the {switched.name} agent. You are now the "
                                            f"{switched.name} agent; continue the conversation.",
                                    tool_call_id=c.id, name=c.name)
                        gen.emit(self.handoff_event(session, previous, switched, reason, "model", gen.id))
                    else:
                        session.add("tool", "Ignored: only one transfer can happen at a time.",
                                    tool_call_id=c.id, name=c.name)
                elif handoff_call is not None:
                    session.add("tool", TRANSFER_SKIPPED, tool_call_id=c.id, name=c.name)
                else:
                    client_pending.append(c)
            if switched is not None:
                agent = switched
                continue
            if client_pending:
                for c in client_pending:
                    session.pending_tool_calls[c.id] = c.name
                return "requires_action", client_pending, agent
            if round_no == max_rounds:
                state.finish_reason = "max_tool_rounds"
                return "incomplete", [], agent
        return "incomplete", [], agent

    async def _execute(self, ctx: ToolContext, agent: ResolvedAgent, spec: ToolSpec, call: ToolCall) -> ToolOutcome:
        ctx = ToolContext(tenant_id=ctx.tenant_id, scope_id=ctx.scope_id, session_id=ctx.session_id,
                          agent_id=agent.id, request_id=ctx.request_id)
        if not spec.side_effects:
            return await self.executor.execute(ctx, spec, call.id, call.arguments)
        # side effects: let the call finish even if the turn is interrupted, so its outcome is
        # recorded (a replay then returns it instead of running the action twice)
        task = asyncio.ensure_future(self.executor.execute(ctx, spec, call.id, call.arguments))
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return await asyncio.shield(task)

    async def _stream_round(self, gen: Generation, session: LiveSession, body: dict[str, Any], state: TurnState,
                            segmenter: SentenceSegmenter | None, toolset: Toolset, api: str,
                            request_id: str) -> list[ToolCall]:
        acc = ToolCallAccumulator()
        state.prompt_tokens_last = 0
        async with self.admission.slot() as waited:
            state.queue_s += waited
            async with self.provider.stream(body, request_id=request_id, scope=session.tenant_id) as stream:
                if state.upstream_ttft_s is None:
                    state.upstream_ttft_s = stream.stats.upstream_ttft_s
                state.model, state.location = stream.stats.model, stream.stats.location
                last: float | None = None
                async for chunk in stream:
                    now = time.perf_counter()
                    if last is not None:
                        INTER_TOKEN.labels(api).observe(now - last)
                    last = now
                    usage = chunk.get("usage")
                    if usage:
                        self._account(state, usage)
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("extra_content"):
                            state.round_extra = delta["extra_content"]
                        content = delta.get("content")
                        if content:
                            state.first_token(api)
                            state.round_text += content
                            state.text += content
                            gen.emit(Event("response.text.delta", {"response_id": gen.id, "delta": content}))
                            if segmenter is not None:
                                for seg in segmenter.push(content):
                                    self._emit_segment(gen, segmenter, seg)
                        if delta.get("tool_calls"):
                            state.first_token(api)
                            for call in acc.add(delta["tool_calls"]):
                                spec = toolset.get(call.name)
                                gen.emit(Event("response.tool_call.started", {
                                    "response_id": gen.id, "call_id": call.id, "name": call.name,
                                    "kind": spec.kind if spec else "unknown"}))
                        if choice.get("finish_reason"):
                            state.finish_reason = choice["finish_reason"]
        return acc.finish()

    @staticmethod
    def _account(state: TurnState, usage: dict[str, Any]) -> None:
        prompt = int(usage.get("prompt_tokens") or 0)
        state.prompt_tokens += prompt
        state.completion_tokens += int(usage.get("completion_tokens") or 0)
        state.cached_tokens += int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        state.reasoning_tokens += int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        state.prompt_tokens_last = prompt

    async def close(self) -> None:
        if self._background:
            await asyncio.wait(list(self._background), timeout=self.settings.tool_timeout_s)


def _argument(arguments: str, name: str) -> str:
    try:
        value = orjson.loads(arguments or "{}")
    except orjson.JSONDecodeError:
        return ""
    return str(value.get(name, ""))[:500] if isinstance(value, dict) else ""
