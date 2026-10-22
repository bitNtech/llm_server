"""Ordered write-behind for conversation data.

A voice turn must not wait for Firestore. Message, response and session-state writes are queued
per key (one key per session) and applied in order by a background chain. Transient Firestore
errors (5xx, unavailable) are retried with backoff for up to RETRY_WINDOW_S, so an outage of several
minutes loses nothing; the session keeps its lease meanwhile (SessionManager only releases a session
whose writes are flushed). Permanent errors (4xx) are logged and skipped. `flush(key)` waits for
one session's writes (before its lease is released to another instance); `flush()` waits for
everything (graceful shutdown).
"""
from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable

from app.core.errors import GatewayError
from app.core.logging import get_logger
from app.core.metrics import PERSIST_BACKLOG

log = get_logger("app.store.writer")

RETRY_WINDOW_S = 600.0


def _backoff(attempt: int) -> float:
    return min(5.0, 0.2 * 2 ** min(attempt, 5)) * (0.5 + random.random())


class Writer:
    def __init__(self) -> None:
        self._tails: dict[str, asyncio.Task] = {}
        self.pending = 0
        self.failed = 0

    def submit(self, key: str, fn: Callable[[], Awaitable[object]], label: str) -> None:
        previous = self._tails.get(key)
        self.pending += 1
        PERSIST_BACKLOG.set(self.pending)

        async def run() -> None:
            if previous is not None:
                await asyncio.wait([previous])
            try:
                deadline = time.monotonic() + RETRY_WINDOW_S
                attempt = 0
                while True:
                    attempt += 1
                    try:
                        await fn()
                        return
                    except GatewayError as exc:
                        if exc.status < 500 or time.monotonic() > deadline:
                            self.failed += 1
                            log.error("write-behind failed", extra={"key": key, "op": label, "error": exc.message,
                                                                    "attempts": attempt})
                            return
                        if attempt in (3, 10) or attempt % 50 == 0:
                            log.warning("write-behind retrying", extra={"key": key, "op": label,
                                                                        "error": exc.message, "attempts": attempt})
                        await asyncio.sleep(_backoff(attempt))
                    except Exception:
                        self.failed += 1
                        log.exception("write-behind crashed", extra={"key": key, "op": label})
                        return
            finally:
                self.pending -= 1
                PERSIST_BACKLOG.set(self.pending)
                if self._tails.get(key) is task:
                    del self._tails[key]

        task = asyncio.create_task(run(), name=f"persist-{label}")
        self._tails[key] = task

    async def flush(self, key: str | None = None, timeout: float = 30.0) -> bool:
        tasks = [self._tails[key]] if key is not None and key in self._tails else (
            list(self._tails.values()) if key is None else [])
        if not tasks:
            return True
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        return not pending
