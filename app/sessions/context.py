"""Context-window management.

Hot path cost is zero for typical voice turns: the prompt size is *estimated* from a per-session
chars-per-token ratio that is calibrated from exact `usage.prompt_tokens` after every turn.
Only when the estimate approaches the budget does the gateway ask vLLM's `/tokenize` (which
applies the real chat template) for an exact count.

When the conversation no longer fits, whole turns (a user message plus the assistant/tool
messages answering it) are dropped from the front - never a tool call without its result.
Trimming goes down to CONTEXT_TRIM_TARGET_RATIO of the budget rather than "just enough", so the
next several turns share an unchanged prefix and keep hitting vLLM's prefix cache.

CONTEXT_STRATEGY=summarize additionally folds dropped turns into a running summary (generated
in the background at low priority, never on the hot path) that rides in the system message.
"""
from __future__ import annotations

import asyncio
from typing import Any

import orjson

from app.config import Settings
from app.core.errors import BadRequest, GatewayError
from app.engine.client import EngineClient
from app.engine.monitor import EngineMonitor
from app.observability.logging import get_logger
from app.observability.metrics import CONTEXT_TRIMS
from app.sessions.store import Message, Session

log = get_logger("app.context")

FAST_PATH_RATIO = 0.85  # below this share of the budget the estimate is trusted
COMPACT_HIGH_WATER = 0.85  # summarize strategy: compact proactively above this share


def system_text(session: Session) -> str:
    if session.summary:
        prefix = session.system_prompt + "\n\n" if session.system_prompt else ""
        return f"{prefix}Summary of the earlier conversation:\n{session.summary}"
    return session.system_prompt


def render(session: Session, messages: list[Message] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sys_text = system_text(session)
    if sys_text:
        out.append({"role": "system", "content": sys_text})
    out.extend(m.to_openai() for m in (session.messages if messages is None else messages))
    return out


def rendered_chars(session: Session, tools: list[dict[str, Any]] | None) -> int:
    tools_chars = len(orjson.dumps(tools)) if tools else 0
    return len(system_text(session)) + session.history_chars() + tools_chars


def turn_groups(messages: list[Message]) -> list[list[Message]]:
    """Split history into groups that each start at a user message."""
    groups: list[list[Message]] = []
    for msg in messages:
        if msg.role == "user" or not groups:
            groups.append([msg])
        else:
            groups[-1].append(msg)
    return groups


class ContextManager:
    def __init__(self, settings: Settings, engine: EngineClient, monitor: EngineMonitor) -> None:
        self.settings = settings
        self.engine = engine
        self.monitor = monitor
        self._background: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ budget
    def context_limit(self) -> int:
        if self.settings.context_max_tokens:
            return self.settings.context_max_tokens
        return self.monitor.state.max_model_len or self.settings.engine_max_model_len

    def budget(self, max_tokens: int) -> int:
        return self.context_limit() - max_tokens - self.settings.context_reserve_tokens

    async def count_exact(self, session: Session, tools: list[dict] | None, request_id: str) -> int:
        body: dict[str, Any] = {"messages": render(session), "add_generation_prompt": True}
        if self.monitor.state.model:
            body["model"] = self.monitor.state.model
        if tools:
            body["tools"] = tools
        kwargs = self.settings.chat_template_kwargs
        if kwargs:
            body["chat_template_kwargs"] = kwargs
        result = await self.engine.tokenize(body, request_id)
        count = int(result.get("count", 0))
        if result.get("max_model_len"):
            self.monitor.state.max_model_len = int(result["max_model_len"])
        session.calibrate(rendered_chars(session, tools), count)
        return count

    # ---------------------------------------------------------------- prepare
    async def prepare(
        self, session: Session, max_tokens: int, tools: list[dict] | None, request_id: str
    ) -> list[dict[str, Any]]:
        """Return the messages to send, trimming the session history if it does not fit."""
        budget = self.budget(max_tokens)
        if budget <= 0:
            raise BadRequest(
                f"max_tokens={max_tokens} leaves no room for the prompt in a "
                f"{self.context_limit()}-token context window"
            )
        tools_chars = len(orjson.dumps(tools)) if tools else 0
        if session.estimate_tokens(tools_chars) <= budget * FAST_PATH_RATIO:
            return render(session)
        try:
            exact = await self.count_exact(session, tools, request_id)
        except GatewayError as exc:
            if exc.status == 400:
                raise
            log.warning("tokenize failed; trusting the estimate", extra={"error": exc.message})
            exact = session.estimate_tokens(tools_chars)
        if exact <= budget:
            return render(session)

        target = int(budget * self.settings.context_trim_target_ratio)
        for _ in range(4):
            dropped = self._trim(session, exact, target)
            if dropped:
                self._on_dropped(session, dropped)
            try:
                exact = await self.count_exact(session, tools, request_id)
            except GatewayError:
                exact = session.estimate_tokens(tools_chars)
            if exact <= budget:
                return render(session)
            if not dropped:
                break
        raise BadRequest(
            f"the latest input needs ~{exact} tokens but only {budget} fit in the context window "
            f"(context {self.context_limit()} - max_tokens {max_tokens} - reserve)",
            code="context_length_exceeded",
        )

    def _trim(self, session: Session, current_tokens: int, target_tokens: int) -> list[Message]:
        groups = turn_groups(session.messages)
        if len(groups) <= 1:
            return []
        ratio = max(session.chars_per_token, 0.5)
        excess = current_tokens - target_tokens
        drop_groups = 0
        removed = 0
        # always keep the newest group (it holds the input being answered)
        while drop_groups < len(groups) - 1 and removed < excess:
            removed += int(sum(m.chars for m in groups[drop_groups]) / ratio) + 5 * len(groups[drop_groups])
            drop_groups += 1
        dropped = [m for g in groups[:drop_groups] for m in g]
        if dropped:
            keep_from = len(dropped)
            session.messages = session.messages[keep_from:]
            session.stats.trimmed_messages += len(dropped)
            CONTEXT_TRIMS.labels(self.settings.context_strategy).inc()
            log.info("context trimmed", extra={"session_id": session.id, "dropped_messages": len(dropped)})
        return dropped

    # ---------------------------------------------------------- summarization
    def _on_dropped(self, session: Session, dropped: list[Message]) -> None:
        if self.settings.context_strategy == "summarize":
            self._spawn(self._summarize(session, dropped, remove_from_history=False))

    def after_turn(self, session: Session, max_tokens: int) -> None:
        """Summarize strategy: compact proactively (in the background) before the window is full."""
        if self.settings.context_strategy != "summarize" or session.compacting:
            return
        budget = self.budget(max_tokens)
        used = session.stats.last_prompt_tokens
        if budget <= 0 or used < budget * COMPACT_HIGH_WATER:
            return
        groups = turn_groups(session.messages)
        if len(groups) <= 2:
            return
        target_drop = used - int(budget * self.settings.context_trim_target_ratio)
        ratio = max(session.chars_per_token, 0.5)
        chosen: list[Message] = []
        removed = 0
        for group in groups[:-2]:  # keep the two most recent turns verbatim
            if removed >= target_drop:
                break
            chosen.extend(group)
            removed += int(sum(m.chars for m in group) / ratio)
        if chosen:
            self._spawn(self._summarize(session, chosen, remove_from_history=True))

    async def _summarize(self, session: Session, messages: list[Message], remove_from_history: bool) -> None:
        session.compacting = True
        try:
            transcript = "\n".join(
                f"{m.role.capitalize()}: {m.content}" for m in messages if m.content and m.role != "tool"
            )
            if not transcript.strip():
                return
            user = (f"Earlier summary:\n{session.summary}\n\n" if session.summary else "") + \
                f"Conversation:\n{transcript}"
            body: dict[str, Any] = {
                "model": self.monitor.state.model or self.settings.served_model_name,
                "messages": [{"role": "system", "content": self.settings.summary_prompt},
                             {"role": "user", "content": user}],
                "max_tokens": self.settings.summary_max_tokens,
                "temperature": 0.2,
            }
            if self.settings.chat_template_kwargs:
                body["chat_template_kwargs"] = self.settings.chat_template_kwargs
            if self.settings.engine_scheduling_policy == "priority":
                body["priority"] = 10  # live turns (priority 0) are scheduled first
            result = await self.engine.chat(body, request_id=f"summary-{session.id}",
                                            timeout=self.settings.generation_timeout_s)
            summary = (result.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            summary = summary.strip()
            if not summary:
                return
            if remove_from_history:
                head = session.messages[: len(messages)]
                if [m.seq for m in head] != [m.seq for m in messages]:
                    return  # history changed meanwhile (trimmed/cleared); drop this result
                session.messages = session.messages[len(messages):]
                session.stats.trimmed_messages += len(messages)
            session.summary = summary
            session.stats.compactions += 1
            log.info("context compacted", extra={"session_id": session.id, "messages": len(messages)})
        except GatewayError as exc:
            log.warning("summarization failed", extra={"session_id": session.id, "error": exc.message})
        finally:
            session.compacting = False

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def close(self) -> None:
        for task in list(self._background):
            task.cancel()
        await asyncio.gather(*self._background, return_exceptions=True)
