"""Async client for vLLM's OpenAI-compatible server.

One pooled keep-alive `httpx.AsyncClient` for the life of the process. Streaming responses are
consumed with a first-token deadline and an inter-chunk idle deadline; closing the response
(on cancel, timeout or client disconnect) drops the connection, which makes vLLM abort the
request and free its KV cache immediately.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import orjson

from app.config import Settings
from app.core.errors import (
    BadRequest,
    EngineError,
    EngineTimeout,
    EngineUnavailable,
    GatewayError,
    NotFound,
)
from app.observability.metrics import ENGINE_ERRORS

DONE = object()


def _error_message(body: bytes) -> str:
    try:
        data = orjson.loads(body)
    except orjson.JSONDecodeError:
        return body.decode(errors="replace")[:500] or "engine error"
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            return str(err.get("message") or err)[:2000]
        return str(err)[:2000]
    return str(data)[:2000]


def map_status(status: int, body: bytes) -> GatewayError:
    message = _error_message(body)
    if status in (400, 422):
        return BadRequest(message, code="engine_rejected_request")
    if status == 404:
        return NotFound(message, code="model_not_found")
    if status in (401, 403):
        ENGINE_ERRORS.labels("auth").inc()
        return EngineError("gateway is not authorized by the engine (ENGINE_API_KEY mismatch)")
    if status == 503:
        ENGINE_ERRORS.labels("unavailable").inc()
        return EngineUnavailable(message)
    ENGINE_ERRORS.labels(f"http_{status}").inc()
    return EngineError(message, details={"engine_status": status})


class EngineStream:
    """Iterate decoded SSE chunks (dicts) or raw `data:` lines of one streaming response."""

    def __init__(self, response: httpx.Response, first_timeout: float, idle_timeout: float) -> None:
        self._response = response
        self._lines = response.aiter_lines()
        self._first_timeout = first_timeout
        self._idle_timeout = idle_timeout
        self._got_data = False

    async def _next_line(self) -> str:
        timeout = self._idle_timeout if self._got_data else self._first_timeout
        try:
            return await asyncio.wait_for(self._lines.__anext__(), timeout)
        except TimeoutError:
            ENGINE_ERRORS.labels("first_token_timeout" if not self._got_data else "idle_timeout").inc()
            phase = "first token" if not self._got_data else "next token"
            raise EngineTimeout(f"engine produced no {phase} within {timeout:.0f}s") from None
        except httpx.HTTPError as exc:
            ENGINE_ERRORS.labels("stream_broken").inc()
            raise EngineError(f"engine stream broke: {exc.__class__.__name__}") from exc

    async def raw(self) -> AsyncIterator[tuple[str, Any]]:
        """Yield (raw_data_line, parsed_chunk|DONE). Comments/blank lines are skipped."""
        while True:
            try:
                line = await self._next_line()
            except StopAsyncIteration:
                return
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                yield line, DONE
                return
            try:
                chunk = orjson.loads(data)
            except orjson.JSONDecodeError:
                ENGINE_ERRORS.labels("bad_chunk").inc()
                continue
            if isinstance(chunk, dict) and ("error" in chunk or chunk.get("object") == "error"):
                err = chunk.get("error", chunk)
                message = err.get("message", "engine error") if isinstance(err, dict) else str(err)
                ENGINE_ERRORS.labels("in_stream").inc()
                raise EngineError(message)
            self._got_data = True
            yield line, chunk

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        async for _, chunk in self.raw():
            if chunk is DONE:
                return
            yield chunk


class EngineClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        headers = {"Content-Type": "application/json"}
        key = settings.engine_api_key.get_secret_value()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.AsyncClient(
            base_url=settings.resolved_engine_base_url,
            headers=headers,
            transport=transport,
            timeout=httpx.Timeout(
                connect=settings.engine_connect_timeout_s,
                read=None,  # enforced per chunk by EngineStream / per call by wait_for
                write=settings.engine_connect_timeout_s,
                pool=settings.engine_connect_timeout_s,
            ),
            limits=httpx.Limits(
                max_connections=settings.engine_max_connections,
                max_keepalive_connections=min(settings.engine_max_connections, 128),
                keepalive_expiry=60,
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, *, timeout: float, **kwargs: Any) -> httpx.Response:
        try:
            response = await asyncio.wait_for(self._client.request(method, path, **kwargs), timeout)
        except TimeoutError:
            ENGINE_ERRORS.labels("timeout").inc()
            raise EngineTimeout(f"engine did not answer {path} within {timeout:.0f}s") from None
        except httpx.ConnectError as exc:
            ENGINE_ERRORS.labels("connect").inc()
            raise EngineUnavailable("engine is not reachable") from exc
        except httpx.HTTPError as exc:
            ENGINE_ERRORS.labels("transport").inc()
            raise EngineError(f"engine request failed: {exc.__class__.__name__}") from exc
        if response.status_code >= 400:
            raise map_status(response.status_code, response.content)
        return response

    # ------------------------------------------------------------------ probes
    async def health(self) -> bool:
        try:
            response = await asyncio.wait_for(self._client.get("/health"), 5.0)
        except (TimeoutError, httpx.HTTPError):
            return False
        return response.status_code == 200

    async def models(self) -> list[dict[str, Any]]:
        response = await self._request("GET", "/v1/models", timeout=10.0)
        return response.json().get("data", [])

    async def metrics_text(self) -> str:
        response = await self._request("GET", "/metrics", timeout=10.0)
        return response.text

    async def tokenize(self, body: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
        headers = {"X-Request-Id": request_id} if request_id else None
        response = await self._request("POST", "/tokenize", timeout=15.0,
                                       content=orjson.dumps(body), headers=headers)
        return response.json()

    # -------------------------------------------------------------- generation
    async def chat(self, body: dict[str, Any], request_id: str, timeout: float) -> dict[str, Any]:
        body = {**body, "stream": False}
        response = await self._request(
            "POST", "/v1/chat/completions", timeout=timeout,
            content=orjson.dumps(body), headers={"X-Request-Id": request_id},
        )
        return orjson.loads(response.content)

    @asynccontextmanager
    async def stream_chat(self, body: dict[str, Any], request_id: str) -> AsyncIterator[EngineStream]:
        body = {**body, "stream": True}
        request = self._client.build_request(
            "POST", "/v1/chat/completions", content=orjson.dumps(body),
            headers={"X-Request-Id": request_id, "Accept": "text/event-stream"},
        )
        timeout = self.settings.engine_first_token_timeout_s
        try:
            response = await asyncio.wait_for(self._client.send(request, stream=True), timeout)
        except TimeoutError:
            ENGINE_ERRORS.labels("first_token_timeout").inc()
            raise EngineTimeout(f"engine did not respond within {timeout:.0f}s") from None
        except httpx.ConnectError as exc:
            ENGINE_ERRORS.labels("connect").inc()
            raise EngineUnavailable("engine is not reachable") from exc
        except httpx.HTTPError as exc:
            ENGINE_ERRORS.labels("transport").inc()
            raise EngineError(f"engine request failed: {exc.__class__.__name__}") from exc
        try:
            if response.status_code >= 400:
                raise map_status(response.status_code, await response.aread())
            yield EngineStream(response, timeout, self.settings.engine_stream_idle_timeout_s)
        finally:
            # Closing mid-stream drops the connection -> vLLM aborts the request.
            await response.aclose()
