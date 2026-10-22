"""Usage tracking per tenant and UTC day.

Every generation (session turn, chat completion, execution, workflow step) reports its token usage
here. Counters are aggregated in memory and flushed every USAGE_FLUSH_INTERVAL_S with Firestore
atomic increments into `tenants/{t}/usage_daily/{YYYY-MM-DD}`, broken down by model, agent and API:
one small write per tenant per interval instead of one per request, and safe across instances.
"""
from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from google.cloud import firestore

from app.core.errors import GatewayError
from app.core.logging import get_logger
from app.core.metrics import TOKENS
from app.store.repos import UsageRepo

log = get_logger("app.usage")
COUNTERS = ("requests", "prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens", "tool_calls")
_KEY = re.compile(r"[^A-Za-z0-9_-]")


def key_of(value: str | None) -> str:
    return _KEY.sub("_", value or "none")[:120]


class UsageRecorder:
    def __init__(self, repo: UsageRepo, interval_s: float) -> None:
        self.repo = repo
        self.interval_s = interval_s
        self._pending: dict[tuple[str, str], dict[str, Any]] = {}
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def record(self, tenant_id: str, *, api: str, model: str | None, agent_id: str | None,
               usage: dict[str, Any] | None, tool_calls: int = 0) -> dict[str, int]:
        usage = usage or {}
        counts = {
            "requests": 1,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "cached_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
            "reasoning_tokens": int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
            "tool_calls": tool_calls,
        }
        TOKENS.labels(model or "unknown", "prompt").inc(counts["prompt_tokens"])
        TOKENS.labels(model or "unknown", "completion").inc(counts["completion_tokens"])
        TOKENS.labels(model or "unknown", "cached").inc(counts["cached_tokens"])
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        bucket = self._pending.setdefault((tenant_id, day), {
            "totals": defaultdict(int), "by_model": defaultdict(lambda: defaultdict(int)),
            "by_agent": defaultdict(lambda: defaultdict(int)), "by_api": defaultdict(lambda: defaultdict(int))})
        for name, value in counts.items():
            bucket["totals"][name] += value
            bucket["by_model"][key_of(model)][name] += value
            bucket["by_agent"][key_of(agent_id)][name] += value
            bucket["by_api"][key_of(api)][name] += value
        return counts

    async def flush(self) -> None:
        pending, self._pending = self._pending, {}
        for (tenant_id, day), bucket in pending.items():
            fields: dict[str, Any] = {"date": day}
            fields.update({k: firestore.Increment(v) for k, v in bucket["totals"].items() if v})
            for group in ("by_model", "by_agent", "by_api"):
                fields[group] = {name: {k: firestore.Increment(v) for k, v in counts.items() if v}
                                 for name, counts in bucket[group].items()}
            try:
                await self.repo.increment(tenant_id, day, fields)
            except GatewayError as exc:
                log.warning("usage flush failed; will retry", extra={"tenant_id": tenant_id, "error": exc.message})
                self._merge_back(tenant_id, day, bucket)

    def _merge_back(self, tenant_id: str, day: str, bucket: dict[str, Any]) -> None:
        current = self._pending.get((tenant_id, day))
        if current is None:
            self._pending[(tenant_id, day)] = bucket
            return
        for name, value in bucket["totals"].items():
            current["totals"][name] += value
        for group in ("by_model", "by_agent", "by_api"):
            for key, counts in bucket[group].items():
                for name, value in counts.items():
                    current[group][key][name] += value

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="usage-flush")

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), self.interval_s)
            except TimeoutError:
                pass
            await self.flush()

    async def close(self) -> None:
        # stop the loop between flushes (never cancel one mid-way: its buckets would be lost)
        self._stop.set()
        if self._task:
            await asyncio.gather(self._task, return_exceptions=True)
        await self.flush()


def summarize(days: list[dict[str, Any]], group_by: str) -> dict[str, Any]:
    totals = {name: sum(int(d.get(name) or 0) for d in days) for name in COUNTERS}
    if group_by == "day":
        groups = {d["date"]: {name: int(d.get(name) or 0) for name in COUNTERS} for d in days}
    else:
        groups: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(COUNTERS, 0))
        for d in days:
            for key, counts in (d.get(f"by_{group_by}") or {}).items():
                for name in COUNTERS:
                    groups[key][name] += int(counts.get(name) or 0)
    return {"totals": totals, "group_by": group_by, "groups": dict(groups)}
