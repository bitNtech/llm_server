"""Admission control: bounded concurrency + bounded FIFO queue with a wait deadline.

vLLM batches everything it is given, but past its sweet spot each extra stream slows every
other stream's inter-token latency. For a voice agent it is better to fail fast with a 429
(the caller can play a fallback line) than to make every active call stutter.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.core.errors import Overloaded, ShuttingDown
from app.observability.metrics import ACTIVE_GENERATIONS, QUEUE_WAIT, QUEUED_GENERATIONS, REJECTIONS


class AdmissionController:
    def __init__(self, max_active: int, max_queue: int, queue_timeout_s: float) -> None:
        self.max_active = max_active
        self.max_queue = max_queue
        self.queue_timeout_s = queue_timeout_s
        self._sem = asyncio.Semaphore(max_active)
        self.active = 0
        self.waiting = 0
        self.draining = False

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[float]:
        """Hold one generation slot; yields the seconds spent queued."""
        if self.draining:
            REJECTIONS.labels("shutting_down").inc()
            raise ShuttingDown("server is shutting down", retry_after=5)
        start = time.perf_counter()
        if self._sem.locked() or self.waiting:
            if self.waiting >= self.max_queue:
                REJECTIONS.labels("queue_full").inc()
                raise Overloaded("too many concurrent generations; retry shortly", retry_after=1)
            self.waiting += 1
            QUEUED_GENERATIONS.set(self.waiting)
            try:
                await asyncio.wait_for(self._sem.acquire(), self.queue_timeout_s)
            except TimeoutError:
                REJECTIONS.labels("queue_timeout").inc()
                raise Overloaded(
                    f"no generation slot within {self.queue_timeout_s:.0f}s; retry shortly", retry_after=2
                ) from None
            finally:
                self.waiting -= 1
                QUEUED_GENERATIONS.set(self.waiting)
        else:
            await self._sem.acquire()
        waited = time.perf_counter() - start
        QUEUE_WAIT.observe(waited)
        self.active += 1
        ACTIVE_GENERATIONS.set(self.active)
        try:
            yield waited
        finally:
            self.active -= 1
            ACTIVE_GENERATIONS.set(self.active)
            self._sem.release()

    def snapshot(self) -> dict:
        return {"active": self.active, "queued": self.waiting, "max_active": self.max_active,
                "max_queue": self.max_queue, "draining": self.draining}
