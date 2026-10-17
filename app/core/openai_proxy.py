"""Stateless OpenAI-compatible chat completions (drop-in for any OpenAI SDK client).

The engine's SSE frames are forwarded byte-for-byte (vLLM already speaks the OpenAI protocol,
including tool-call and reasoning deltas); the gateway adds auth, defaults/limits, admission
control, metrics, request ids and cancellation.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import orjson

from app.config import Settings
from app.core.admission import AdmissionController
from app.core.errors import BadRequest, EngineTimeout, EngineUnavailable, GatewayError
from app.core.generation import Generation, GenerationRegistry
from app.core.params import apply_defaults
from app.engine.client import DONE, EngineClient
from app.engine.monitor import EngineMonitor
from app.observability.logging import get_logger, request_id_var
from app.observability.metrics import GENERATION_DURATION, GENERATIONS, INTER_TOKEN, TOKENS, TTFT

log = get_logger("app.openai")


def sse_error(error: GatewayError, request_id: str | None) -> bytes:
    return b"data: " + orjson.dumps(error.to_dict(request_id)) + b"\n\ndata: [DONE]\n\n"


def _account(usage: dict[str, Any]) -> None:
    TOKENS.labels("prompt").inc(int(usage.get("prompt_tokens") or 0))
    TOKENS.labels("completion").inc(int(usage.get("completion_tokens") or 0))
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    TOKENS.labels("cached_prompt").inc(int(cached))


class OpenAIProxy:
    def __init__(
        self,
        settings: Settings,
        engine: EngineClient,
        monitor: EngineMonitor,
        admission: AdmissionController,
        registry: GenerationRegistry,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.monitor = monitor
        self.admission = admission
        self.registry = registry

    def prepare(self, body: Any) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise BadRequest("request body must be a JSON object")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise BadRequest("messages must be a non-empty list")
        total_chars = sum(len(str(m.get("content") or "")) for m in messages if isinstance(m, dict))
        if total_chars > self.settings.max_input_chars * 8:
            raise BadRequest("messages are too large")
        model = body.get("model")
        state = self.monitor.state
        if not model or model in ("default", self.settings.served_model_name):
            body["model"] = state.model or self.settings.served_model_name
        return apply_defaults(self.settings, body)

    async def _ready(self) -> None:
        if not self.monitor.state.ready and not await self.monitor.check():
            raise EngineUnavailable(f"engine not ready: {self.monitor.state.last_error}", retry_after=2)

    # ----------------------------------------------------------- non-streaming
    async def complete(self, body: dict[str, Any], request_id: str) -> dict[str, Any]:
        await self._ready()
        started = time.perf_counter()
        status = "completed"
        gen = Generation(request_id, api="openai")
        result: dict[str, Any] = {}
        error: GatewayError | None = None

        async def produce() -> None:
            nonlocal result, error
            try:
                async with self.admission.slot():
                    result = await self.engine.chat(body, request_id, self.settings.generation_timeout_s)
            except GatewayError as exc:
                error = exc

        gen.start(produce(), self.registry)
        try:
            await asyncio.shield(gen.finished.wait())
        except asyncio.CancelledError:
            gen.cancel("client_disconnect")
            raise
        if gen.cancel_reason:
            GENERATIONS.labels("openai", "cancelled").inc()
            raise BadRequest("request was cancelled", code="cancelled", status=499)
        if error is not None or not result:
            GENERATIONS.labels("openai", "failed").inc()
            raise error or GatewayError("generation failed")
        if result.get("usage"):
            _account(result["usage"])
        GENERATIONS.labels("openai", status).inc()
        GENERATION_DURATION.labels("openai", status).observe(time.perf_counter() - started)
        return result

    # --------------------------------------------------------------- streaming
    async def stream(self, body: dict[str, Any], request_id: str) -> Generation:
        await self._ready()
        client_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        gen = Generation(request_id, api="openai")
        gen.start(self._produce(gen, body, client_usage), self.registry)
        return gen

    async def _produce(self, gen: Generation, body: dict[str, Any], client_usage: bool) -> None:
        request_id_var.set(gen.id)
        started = time.perf_counter()
        first = True
        emitted = False
        status = "completed"
        try:
            async with asyncio.timeout(self.settings.generation_timeout_s):
                async with self.admission.slot(), self.engine.stream_chat(body, gen.id) as stream:
                    last: float | None = None
                    async for line, chunk in stream.raw():
                        if chunk is DONE:
                            gen.emit(b"data: [DONE]\n\n")
                            break
                        now = time.perf_counter()
                        usage = chunk.get("usage")
                        if usage:
                            _account(usage)
                        choices = chunk.get("choices") or []
                        if first and any(
                            (c.get("delta") or {}).get(k)
                            for c in choices for k in ("content", "tool_calls", "reasoning_content", "reasoning")
                        ):
                            first = False
                            TTFT.labels("openai").observe(now - started)
                        if last is not None:
                            INTER_TOKEN.labels("openai").observe(now - last)
                        last = now
                        if not choices and usage and not client_usage:
                            continue  # usage-only chunk the client did not ask for
                        gen.emit(line.encode() + b"\n\n")
                        emitted = True
        except asyncio.CancelledError:
            status = "cancelled"
            if gen.cancel_reason != "client_disconnect":
                gen.emit(sse_error(BadRequest("request was cancelled", code="cancelled"), gen.id))
            raise
        except TimeoutError:
            status = "failed"
            timeout = EngineTimeout("generation exceeded GENERATION_TIMEOUT_S")
            gen.emit(sse_error(timeout, gen.id) if emitted else timeout)
        except GatewayError as exc:
            status = "failed"
            # before any frame: hand the error object to the route so it can answer with a status code
            gen.emit(sse_error(exc, gen.id) if emitted else exc)
        finally:
            GENERATIONS.labels("openai", status).inc()
            GENERATION_DURATION.labels("openai", status).observe(time.perf_counter() - started)
