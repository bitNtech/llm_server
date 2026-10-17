from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.config import Settings
from app.core.errors import BadRequest
from app.engine.monitor import EngineMonitor
from app.sessions.context import ContextManager, render, turn_groups
from app.sessions.store import Session


class FakeEngine:
    """Counts one token per 4 characters of content plus 4 per message (like the mock engine)."""

    def __init__(self) -> None:
        self.tokenize_calls = 0
        self.chat_calls: list[dict[str, Any]] = []

    async def tokenize(self, body: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
        self.tokenize_calls += 1
        count = sum(len(m.get("content") or "") // 4 + 4 for m in body["messages"])
        return {"count": count, "max_model_len": 1000}

    async def chat(self, body: dict[str, Any], request_id: str, timeout: float) -> dict[str, Any]:
        self.chat_calls.append(body)
        return {"choices": [{"message": {"content": "They discussed booking a slot."}}]}


def build(strategy: str = "truncate", max_tokens: int = 1000) -> tuple[ContextManager, FakeEngine]:
    settings = Settings(_env_file=None, context_strategy=strategy, context_max_tokens=max_tokens,
                        context_reserve_tokens=0, context_trim_target_ratio=0.5, context_chars_per_token=4.0)
    engine = FakeEngine()
    monitor = EngineMonitor(engine, 5.0)  # type: ignore[arg-type]
    return ContextManager(settings, engine, monitor), engine  # type: ignore[arg-type]


def conversation(turns: int, words: int = 40) -> Session:
    s = Session(id="s", system_prompt="You are a helpful receptionist.", chars_per_token=4.0)
    for i in range(turns):
        s.add("user", f"question {i} " + "word " * words)
        s.add("assistant", f"answer {i} " + "word " * words)
    return s


def test_turn_groups_keep_tool_results_with_their_turn():
    s = Session(id="s")
    s.add("assistant", "Hello, how can I help?")
    s.add("user", "book")
    s.add("assistant", None, tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}])
    s.add("tool", "ok", tool_call_id="c1")
    s.add("assistant", "done")
    s.add("user", "thanks")
    groups = turn_groups(s.messages)
    assert [[m.role for m in g] for g in groups] == [["assistant"], ["user", "assistant", "tool", "assistant"], ["user"]]


def test_summary_rides_in_the_single_system_message():
    s = conversation(1)
    s.summary = "Caller wants an appointment."
    msgs = render(s)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert "Caller wants an appointment." in msgs[0]["content"]


async def test_fast_path_skips_the_tokenizer():
    ctx, engine = build()
    msgs = await ctx.prepare(conversation(2), max_tokens=100, tools=None, request_id="r")
    assert len(msgs) == 5 and engine.tokenize_calls == 0


async def test_overflow_trims_oldest_turns_to_target():
    ctx, engine = build(max_tokens=600)
    s = conversation(10)  # ~1100 tokens
    s.add("user", "latest question")
    msgs = await ctx.prepare(s, max_tokens=100, tools=None, request_id="r")
    assert engine.tokenize_calls >= 2
    assert msgs[-1]["content"] == "latest question"
    assert msgs[1]["role"] == "user"  # history still starts at a user turn
    assert s.stats.trimmed_messages > 0
    exact = (await engine.tokenize({"messages": msgs}))["count"]
    assert exact <= 600 - 100
    assert exact <= (600 - 100) * 0.5 + 120  # trimmed to roughly the target, not just under the limit


async def test_input_that_cannot_fit_is_rejected():
    ctx, _ = build(max_tokens=200)
    s = Session(id="s", chars_per_token=4.0)
    s.add("user", "word " * 400)
    with pytest.raises(BadRequest) as exc:
        await ctx.prepare(s, max_tokens=100, tools=None, request_id="r")
    assert exc.value.code == "context_length_exceeded"


async def test_max_tokens_larger_than_window():
    ctx, _ = build(max_tokens=200)
    with pytest.raises(BadRequest):
        await ctx.prepare(conversation(1), max_tokens=500, tools=None, request_id="r")


async def test_summarize_strategy_compacts_in_background():
    ctx, engine = build(strategy="summarize", max_tokens=600)
    s = conversation(10)
    s.stats.last_prompt_tokens = 560
    before = len(s.messages)
    ctx.after_turn(s, max_tokens=50)
    await asyncio.gather(*ctx._background)
    assert engine.chat_calls, "summary request was not sent"
    assert s.summary == "They discussed booking a slot."
    assert len(s.messages) < before and s.messages[0].role == "user"
    assert s.stats.compactions == 1
