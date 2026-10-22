"""Context-window management.

The prompt budget is CONTEXT_MAX_TOKENS (default 64k of Gemini's 1M window): a smaller prompt means
lower prefill latency and cost on every turn. Size is *estimated* from a per-session chars-per-token
ratio that is calibrated from Vertex's exact `usage.prompt_tokens` after every round, so there is no
tokenizer call on the hot path.

When the conversation outgrows the budget, whole turns (a user message and everything answering
it) leave the window from the front - never a tool call without its result - and trimming goes
down to CONTEXT_TRIM_TARGET_RATIO of the budget so the next turns share an unchanged prefix (Gemini
implicit context caching bills cached prefix tokens at a discount and serves them faster).
CONTEXT_STRATEGY=summarize folds trimmed turns into a running summary generated in the background.
The persisted history is never trimmed; only the window sent to the model moves.
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.config import Settings
from app.core.errors import BadRequest, GatewayError
from app.core.logging import get_logger
from app.core.metrics import CONTEXT_TRIMS
from app.engine.sessions import LiveSession, Message, SessionManager
from app.providers.vertex import VertexProvider

log = get_logger("app.context")

MESSAGE_OVERHEAD_TOKENS = 5
OPENING_TURN = "(The conversation has just started.)"
SUMMARY_PROMPT = ("Summarize the conversation so far in a few short sentences. Keep names, numbers, dates, "
                  "decisions and open questions. Write in the language the user speaks.")


def turn_groups(messages: list[Message]) -> list[list[Message]]:
    groups: list[list[Message]] = []
    for msg in messages:
        if msg.role == "user" or not groups:
            groups.append([msg])
        else:
            groups[-1].append(msg)
    return groups


class ContextBuilder:
    def __init__(self, settings: Settings, provider: VertexProvider, sessions: SessionManager) -> None:
        self.settings = settings
        self.provider = provider
        self.sessions = sessions
        self._background: set[asyncio.Task] = set()

    def budget(self, max_tokens: int) -> int:
        limit = min(self.settings.context_max_tokens, self.settings.vertex_context_window)
        return limit - max_tokens - self.settings.context_reserve_tokens

    @staticmethod
    def system_text(session: LiveSession, instructions: str) -> str:
        parts = [instructions.strip()] if instructions.strip() else []
        if session.instructions.strip():
            parts.append(session.instructions.strip())
        if session.summary:
            parts.append(f"Summary of the earlier conversation:\n{session.summary}")
        return "\n\n".join(parts)

    def estimate(self, session: LiveSession, system: str, extra_chars: int) -> int:
        window = session.window()
        chars = len(system) + sum(m.chars for m in window) + extra_chars
        return int(chars / max(session.chars_per_token, 0.5)) + MESSAGE_OVERHEAD_TOKENS * (len(window) + 1)

    def build(self, session: LiveSession, instructions: str, tools_chars: int, max_tokens: int,
              response_instructions: str | None = None) -> tuple[list[dict[str, Any]], int]:
        """Return (messages, rendered_chars), trimming the window if it does not fit."""
        budget = self.budget(max_tokens)
        if budget <= 0:
            raise BadRequest(f"max_tokens={max_tokens} leaves no room for the prompt (CONTEXT_MAX_TOKENS)")
        system = self.system_text(session, instructions)
        extra = tools_chars + len(response_instructions or "")
        estimate = self.estimate(session, system, extra)
        if estimate > budget:
            target = int(budget * self.settings.context_trim_target_ratio)
            dropped = self._trim(session, estimate - target)
            if dropped and self.settings.context_strategy == "summarize":
                self._spawn(self._summarize(session, dropped))
            system = self.system_text(session, instructions)
            estimate = self.estimate(session, system, extra)
            if estimate > budget:
                raise BadRequest(f"the latest input needs ~{estimate} tokens but only {budget} fit "
                                 f"(CONTEXT_MAX_TOKENS={self.settings.context_max_tokens})",
                                 code="context_length_exceeded")
        window = session.window()
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
        if not any(m.role == "user" for m in window):
            # Gemini needs at least one user turn; an agent greeting starts with none
            messages.append({"role": "user", "content": OPENING_TURN})
        messages.extend(m.to_openai() for m in window)
        if response_instructions:
            messages.append({"role": "system", "content": response_instructions})
        chars = len(system) + sum(m.chars for m in window) + extra
        return messages, chars

    def _trim(self, session: LiveSession, excess_tokens: int) -> list[Message]:
        groups = turn_groups(session.window())
        ratio = max(session.chars_per_token, 0.5)
        removed, drop = 0, 0
        while drop < len(groups) - 1 and removed < excess_tokens:  # the newest turn always stays
            removed += int(sum(m.chars for m in groups[drop]) / ratio) + MESSAGE_OVERHEAD_TOKENS * len(groups[drop])
            drop += 1
        dropped = [m for g in groups[:drop] for m in g]
        if dropped:
            session.context_start_seq = dropped[-1].seq
            session.stats.trimmed_messages += len(dropped)
            CONTEXT_TRIMS.labels(self.settings.context_strategy).inc()
            log.info("context trimmed", extra={"session_id": session.id, "dropped_messages": len(dropped)})
        return dropped

    async def _summarize(self, session: LiveSession, messages: list[Message]) -> None:
        if session.compacting:
            return
        session.compacting = True
        try:
            transcript = "\n".join(f"{m.role.capitalize()}: {m.content}" for m in messages
                                   if m.content and m.role in ("user", "assistant"))
            if not transcript.strip():
                return
            user = (f"Earlier summary:\n{session.summary}\n\n" if session.summary else "") + \
                f"Conversation:\n{transcript}"
            body = {"model": "default", "max_tokens": self.settings.summary_max_tokens, "temperature": 0.2,
                    "messages": [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": user}]}
            result, _ = await self.provider.complete(body, request_id=f"summary-{session.id}",
                                                     scope=session.tenant_id)
            summary = ((result.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            if summary.strip():
                session.summary = summary.strip()
                session.stats.compactions += 1
                self.sessions.save_state(session)
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
