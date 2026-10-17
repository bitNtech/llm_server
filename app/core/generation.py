"""A running generation: producer task + event queue + cancellation handle.

The producer (talking to vLLM) runs in its own task and pushes events onto a queue; the
consumer (SSE response, WebSocket sender) drains it. Decoupling the two means a generation can be
cancelled from anywhere (barge-in, REST cancel, client disconnect, shutdown) by cancelling one
task, and the producer's `finally` always closes the engine stream so vLLM aborts the request.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from app.observability.logging import get_logger
from app.observability.metrics import CANCELLATIONS

log = get_logger("app.generation")

_END = object()


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)


class Generation:
    def __init__(self, gen_id: str, api: str, session_id: str | None = None) -> None:
        self.id = gen_id
        self.api = api
        self.session_id = session_id
        self.started = time.perf_counter()
        self.cancel_reason: str | None = None
        self.task: asyncio.Task | None = None
        self.finished = asyncio.Event()
        self.status: str | None = None
        self.error: Exception | None = None
        self.base_seq: int | None = None  # session message seq before this response added any
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    def emit(self, item: Any) -> None:
        self._queue.put_nowait(item)

    def start(
        self,
        producer: Coroutine[Any, Any, None],
        registry: GenerationRegistry,
        on_never_started: Callable[[], Any] | None = None,
    ) -> None:
        """Run `producer` in its own task. If the task is cancelled before it ever runs (possible
        when a newer response supersedes it immediately), the producer body never executes, so
        `on_never_started` supplies the terminal event instead."""
        registry.add(self)
        started = False

        async def run() -> None:
            nonlocal started
            started = True
            try:
                await producer
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("generation producer crashed", extra={"response_id": self.id})
            finally:
                self._finish(registry)

        def on_done(_: asyncio.Task) -> None:
            if not started:
                producer.close()
                if on_never_started is not None:
                    self.emit(on_never_started())
                self._finish(registry)

        self.task = asyncio.create_task(run(), name=f"gen-{self.id}")
        self.task.add_done_callback(on_done)

    def _finish(self, registry: GenerationRegistry) -> None:
        if self.finished.is_set():
            return
        registry.remove(self.id)
        self._queue.put_nowait(_END)
        self.finished.set()

    def cancel(self, reason: str) -> bool:
        if self.task is None or self.task.done():
            return False
        if self.cancel_reason is None:
            self.cancel_reason = reason
            CANCELLATIONS.labels(reason).inc()
        self.task.cancel()
        return True

    @property
    def done(self) -> bool:
        return self.finished.is_set()

    async def events(self) -> AsyncIterator[Any]:
        while True:
            item = await self._queue.get()
            if item is _END:
                return
            yield item

    async def wait(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self.finished.wait(), timeout)
        except TimeoutError:
            return False
        return True


class GenerationRegistry:
    """Index of in-flight generations so any of them can be cancelled by id."""

    def __init__(self) -> None:
        self._items: dict[str, Generation] = {}

    def add(self, gen: Generation) -> None:
        self._items[gen.id] = gen

    def remove(self, gen_id: str) -> None:
        self._items.pop(gen_id, None)

    def get(self, gen_id: str) -> Generation | None:
        return self._items.get(gen_id)

    def cancel(self, gen_id: str, reason: str = "client_cancel") -> bool:
        gen = self._items.get(gen_id)
        return gen.cancel(reason) if gen else False

    def __len__(self) -> int:
        return len(self._items)

    async def drain(self, timeout: float) -> int:
        """Wait for in-flight generations, then cancel the stragglers. Returns #cancelled."""
        pending = list(self._items.values())
        if pending and timeout > 0:
            await asyncio.wait([asyncio.create_task(g.finished.wait()) for g in pending], timeout=timeout)
        stragglers = [g for g in self._items.values() if not g.done]
        for gen in stragglers:
            gen.cancel("shutdown")
        if stragglers:
            await asyncio.wait([asyncio.create_task(g.finished.wait()) for g in stragglers], timeout=5)
        return len(stragglers)
