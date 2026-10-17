"""Background engine health monitor: tracks readiness and discovers the served model."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from app.core.errors import GatewayError
from app.engine.client import EngineClient
from app.observability.logging import get_logger
from app.observability.metrics import ENGINE_UP

log = get_logger("app.engine")


@dataclass
class EngineState:
    ready: bool = False
    model: str | None = None
    max_model_len: int | None = None
    root: str | None = None
    last_ok_at: float | None = None
    last_error: str | None = "not checked yet"
    checks: int = 0
    ready_since: float | None = None
    models: list[str] = field(default_factory=list)

    def snapshot(self) -> dict:
        return {
            "ready": self.ready,
            "model": self.model,
            "max_model_len": self.max_model_len,
            "last_ok_age_s": round(time.monotonic() - self.last_ok_at, 1) if self.last_ok_at else None,
            "last_error": self.last_error,
        }


class EngineMonitor:
    def __init__(
        self,
        client: EngineClient,
        interval_s: float,
        preferred_model: str | None = None,
        warmup: bool = False,
        warmup_timeout_s: float = 900.0,
    ) -> None:
        self.client = client
        self.interval_s = interval_s
        self.preferred_model = preferred_model
        self.warmup_enabled = warmup
        self.warmup_timeout_s = warmup_timeout_s
        self.warmed = not warmup
        self._warming = False
        self.state = EngineState()
        self._task: asyncio.Task | None = None
        self._ready_event = asyncio.Event()

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="engine-monitor")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def wait_ready(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout)
        except TimeoutError:
            return False
        return True

    async def check(self) -> bool:
        state = self.state
        state.checks += 1
        healthy = await self.client.health()
        if healthy:
            try:
                models = await self.client.models()
            except GatewayError as exc:
                healthy, error = False, exc.message
            else:
                ids = [m.get("id") for m in models if m.get("id")]
                chosen = next((m for m in models if m.get("id") == self.preferred_model), None)
                chosen = chosen or (models[0] if models else None)
                if chosen is None:
                    healthy, error = False, "engine serves no model"
                else:
                    state.models = ids
                    state.model = chosen["id"]
                    state.root = chosen.get("root")
                    state.max_model_len = chosen.get("max_model_len") or state.max_model_len
                    error = None
        else:
            error = "engine /health failed"

        if healthy and not self.warmed:
            if self._warming:  # another caller is already warming up: not ready yet
                return False
            self._warming = True
            try:
                await self._warmup()
            finally:
                self._warming = False
        was_ready = state.ready
        state.ready = healthy
        ENGINE_UP.set(1 if healthy else 0)
        if healthy:
            state.last_ok_at = time.monotonic()
            state.last_error = None
            self._ready_event.set()
            if not was_ready:
                state.ready_since = time.monotonic()
                log.info("engine ready", extra={"model": state.model, "max_model_len": state.max_model_len})
        else:
            state.last_error = error
            self._ready_event.clear()
            if was_ready:
                log.error("engine became unavailable", extra={"error": error})
        return healthy

    async def _warmup(self) -> None:
        """First requests after a (re)start pay for JIT compilation / CUDA-graph lazy paths
        (tens of seconds on the CPU backend). Pay it here, before readiness turns green."""
        state = self.state
        state.last_error = "warming up"
        started = time.monotonic()
        filler = "The caller asked about appointment availability and visiting hours. " * 40
        prompts = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "system", "content": filler}, {"role": "user", "content": "Summarize in one sentence."}],
        ]
        try:
            async with asyncio.timeout(self.warmup_timeout_s):
                for i, messages in enumerate(prompts):
                    body = {"model": state.model, "messages": messages, "max_tokens": 8, "temperature": 0}
                    await self.client.chat(body, request_id=f"warmup-{i}", timeout=self.warmup_timeout_s)
                    async with self.client.stream_chat({**body, "max_tokens": 4}, f"warmup-stream-{i}") as stream:
                        async for _ in stream:
                            pass
            log.info("engine warm-up done", extra={"seconds": round(time.monotonic() - started, 1)})
        except (TimeoutError, GatewayError) as exc:
            log.warning("engine warm-up incomplete; serving anyway", extra={"error": str(exc)})
        self.warmed = True

    async def _run(self) -> None:
        while True:
            try:
                ready = await self.check()
            except Exception:  # never let the monitor die
                log.exception("engine health check crashed")
                ready = False
            # poll fast while starting up / degraded, slower when healthy
            await asyncio.sleep(self.interval_s if ready else min(1.0, self.interval_s))
