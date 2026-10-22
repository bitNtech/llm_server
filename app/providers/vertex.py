"""Vertex AI provider (OpenAI-compatible endpoint).

    https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi
    https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/endpoints/openapi

Routing: a request for model M is tried on M at every configured location (VERTEX_LOCATIONS, in
order), then on each fallback model. A route is skipped while its location's circuit breaker is
open or after Vertex said the model does not exist there (404, remembered for 10 minutes).

Retries happen only before the first token reaches the caller: a 429/5xx/timeout/connection error
moves on to the next route within the first-token deadline. Once speech has started nothing is
retried, so a voice agent never repeats itself.

Latency: one pooled HTTP/2 client, OAuth tokens refreshed in the background, connections kept warm
while idle, and a startup probe that pays the TLS handshake before the first caller does.
"""
from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import orjson

from app.config import Settings
from app.core.errors import (
    BadRequest,
    GatewayError,
    NotFound,
    Overloaded,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
)
from app.core.logging import get_logger
from app.core.metrics import CIRCUIT_OPEN, PROVIDER_FAILOVERS, PROVIDER_READY, PROVIDER_REQUESTS, UPSTREAM_TTFT
from app.core.tracing import span
from app.gcp.credentials import TokenSource
from app.providers.signatures import PLACEHOLDER, SignatureCache, call_key, ensure_signatures, signature_of

log = get_logger("app.vertex")

DONE = object()
UNAVAILABLE_TTL_S = 600.0
# Gateway-internal keys that must never reach Vertex.
_INTERNAL_KEYS = {"api", "agent", "session_id", "segmentation"}


def _error_message(body: bytes) -> tuple[str, str]:
    """Vertex errors arrive as `[{"error": {...}}]` or `{"error": {...}}`; return (message, status)."""
    try:
        data = orjson.loads(body)
    except orjson.JSONDecodeError:
        return body.decode(errors="replace")[:500] or "Vertex error", ""
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            return str(err.get("message") or err)[:2000], str(err.get("status") or "")
        return str(err)[:2000], ""
    return str(data)[:2000], ""


@dataclass
class LocationState:
    location: str
    failures: int = 0
    open_until: float = 0.0
    requests: int = 0
    errors: int = 0
    last_error: str | None = None
    last_ok_at: float | None = None

    def available(self, now: float) -> bool:
        return now >= self.open_until

    def snapshot(self, now: float) -> dict[str, Any]:
        return {
            "location": self.location,
            "circuit": "open" if now < self.open_until else ("half_open" if self.failures else "closed"),
            "consecutive_failures": self.failures,
            "requests": self.requests,
            "errors": self.errors,
            "last_error": self.last_error,
            "last_ok_age_s": round(now - self.last_ok_at, 1) if self.last_ok_at else None,
        }


@dataclass
class Attempt:
    model: str
    location: str
    status: int | None = None
    error: str | None = None


@dataclass
class RouteError(Exception):
    error: GatewayError
    retryable: bool
    attempt: Attempt
    unavailable: bool = False  # model not served at this location
    retry_same: bool = False  # transient model hiccup (empty candidate): retry this route, not the next


@dataclass
class StreamStats:
    model: str = ""
    location: str = ""
    attempts: list[Attempt] = field(default_factory=list)
    upstream_ttft_s: float | None = None


class VertexStream:
    """Iterate parsed chunks (dicts) or raw `data:` lines of one Vertex streaming response."""

    def __init__(self, response: httpx.Response, lines: AsyncIterator[str], buffered: list[tuple[str, Any]],
                 idle_timeout: float, stats: StreamStats, on_chunk: Any) -> None:
        self._response = response
        self._lines = lines
        self._buffered = buffered
        self._idle_timeout = idle_timeout
        self.stats = stats
        self._on_chunk = on_chunk

    async def raw(self) -> AsyncIterator[tuple[str, Any]]:
        for item in self._buffered:
            yield item
            if item[1] is DONE:
                return
        self._buffered = []
        while True:
            try:
                line = await asyncio.wait_for(self._lines.__anext__(), self._idle_timeout)
            except StopAsyncIteration:
                return
            except TimeoutError:
                raise ProviderTimeout(f"Vertex stream stalled for {self._idle_timeout:.0f}s") from None
            except httpx.HTTPError as exc:
                raise ProviderError(f"Vertex stream broke: {exc.__class__.__name__}") from exc
            item = _parse_line(line)
            if item is None:
                continue
            if item[1] is not DONE:
                self._on_chunk(item[1])
            yield item
            if item[1] is DONE:
                return

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        async for _, chunk in self.raw():
            if chunk is DONE:
                return
            yield chunk


def _parse_line(line: str) -> tuple[str, Any] | None:
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return line, DONE
    try:
        chunk = orjson.loads(data)
    except orjson.JSONDecodeError:
        return None
    if isinstance(chunk, list) and chunk:
        chunk = chunk[0]
    if not isinstance(chunk, dict):
        return None
    if "error" in chunk and "choices" not in chunk:
        message, _ = _error_message(orjson.dumps(chunk))
        raise ProviderError(f"Vertex stream error: {message}")
    return line, chunk


def _meaningful(chunk: dict[str, Any]) -> bool:
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or choice.get("message") or {}
        if delta.get("content") or delta.get("tool_calls") or choice.get("finish_reason"):
            return True
    return False


def _empty(attempt: Attempt) -> RouteError:
    """Vertex sometimes returns a candidate with no content, no tool call and no finish reason."""
    attempt.error = "empty response"
    return RouteError(ProviderError("Vertex returned an empty response", code="provider_empty_response"),
                      True, attempt, retry_same=True)


class VertexProvider:
    def __init__(self, settings: Settings, tokens: TokenSource, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.tokens = tokens
        self.project = settings.gcp_project_id
        self.locations = settings.location_list
        self.default_model = settings.vertex_model
        self.allowed_models = settings.allowed_model_list
        self.fallback_models = settings.fallback_model_list
        self.states = {loc: LocationState(loc) for loc in self.locations}
        self.signatures = SignatureCache()
        self._unavailable: dict[tuple[str, str], float] = {}
        self._last_used: dict[str, float] = {}
        self._keepalive_task: asyncio.Task | None = None
        self._probe_task: asyncio.Task | None = None
        self.probe_done = not settings.vertex_startup_probe
        self.probe_error: str | None = None
        self.client = httpx.AsyncClient(
            http2=settings.vertex_http2 and transport is None,
            transport=transport,
            timeout=httpx.Timeout(connect=settings.vertex_connect_timeout_s, read=None,
                                  write=settings.vertex_connect_timeout_s, pool=settings.vertex_connect_timeout_s),
            limits=httpx.Limits(max_connections=settings.vertex_max_connections,
                                max_keepalive_connections=min(settings.vertex_max_connections, 64),
                                keepalive_expiry=300),
            headers={"Content-Type": "application/json", "User-Agent": "aica-llm-server"},
        )

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self.settings.vertex_keepalive_interval_s > 0:
            self._keepalive_task = asyncio.create_task(self._keepalive(), name="vertex-keepalive")
        if self.settings.vertex_startup_probe:
            self._probe_task = asyncio.create_task(self._probe(), name="vertex-probe")

    async def aclose(self) -> None:
        for task in (self._keepalive_task, self._probe_task):
            if task:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        await self.client.aclose()

    # ------------------------------------------------------------- routing
    def base_url(self, location: str) -> str:
        template = self.settings.vertex_endpoint_template
        if template:
            return template.format(location=location, project=self.project).rstrip("/")
        host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
        return f"https://{host}/v1/projects/{self.project}/locations/{location}/endpoints/openapi"

    def resolve_model(self, requested: str | None) -> str:
        """Map a client/agent model name to an allowed Vertex model id ('default' -> VERTEX_MODEL)."""
        if not requested or requested == "default":
            return self.default_model
        model = requested if "/" in requested else f"google/{requested}"
        if model not in self.allowed_models:
            raise BadRequest(f"model {requested!r} is not enabled on this server",
                             code="model_not_allowed", details={"allowed": self.allowed_models})
        return model

    def routes(self, model: str) -> list[tuple[str, str]]:
        now = time.monotonic()
        chain = [model, *[m for m in self.fallback_models if m != model]]
        routes: list[tuple[str, str]] = []
        deferred: list[tuple[str, str]] = []
        for m in chain:
            for loc in self.locations:
                until = self._unavailable.get((m, loc))
                if until and until > now:
                    continue
                (routes if self.states[loc].available(now) else deferred).append((m, loc))
        # open circuits are only tried after every healthy route (half-open probing under load)
        return routes + deferred

    def ready(self) -> bool:
        now = time.monotonic()
        credentials = self.tokens.snapshot()
        token_ok = bool(credentials.get("valid")) or credentials.get("type") == "static"
        ok = token_ok and self.probe_done and any(s.available(now) for s in self.states.values())
        PROVIDER_READY.set(1 if ok else 0)
        return ok

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "id": "vertex",
            "project": self.project,
            "default_model": self.default_model,
            "fallback_models": self.fallback_models,
            "allowed_models": self.allowed_models,
            "context_window": self.settings.vertex_context_window,
            "reasoning_effort": self.settings.vertex_reasoning_effort or None,
            "ready": self.ready(),
            "startup_probe": {"done": self.probe_done, "error": self.probe_error},
            "credentials": self.tokens.snapshot(),
            "locations": [s.snapshot(now) for s in self.states.values()],
            "unavailable_routes": [{"model": m, "location": loc} for (m, loc), until in self._unavailable.items()
                                   if until > now],
        }

    # ----------------------------------------------------------- body prep
    def prepare(self, body: dict[str, Any], scope: str) -> dict[str, Any]:
        out = {k: v for k, v in body.items() if k not in _INTERNAL_KEYS and v is not None}
        if "max_completion_tokens" in out:
            mct = out.pop("max_completion_tokens")
            out.setdefault("max_tokens", mct)
        if self.settings.vertex_reasoning_effort and "reasoning_effort" not in out:
            out["reasoning_effort"] = self.settings.vertex_reasoning_effort
        if out.get("reasoning_effort") == "none":
            out["reasoning_effort"] = "minimal"  # Gemini 3 cannot fully disable thinking; 'minimal' is the floor
        followup = self.settings.vertex_tool_followup_reasoning_effort
        if followup and out.get("reasoning_effort") == "minimal":
            last = next((m for m in reversed(out.get("messages") or [])
                         if isinstance(m, dict) and m.get("role") != "system"), None)
            if last is not None and last.get("role") == "tool":
                out["reasoning_effort"] = followup
        if not out.get("tools"):
            out.pop("tools", None)
            out.pop("tool_choice", None)
            out.pop("parallel_tool_calls", None)
        messages = out.get("messages") or []
        if any(m.get("role") == "assistant" and m.get("tool_calls") for m in messages if isinstance(m, dict)):
            out["messages"] = messages = [dict(m) if isinstance(m, dict) else m for m in messages]
            for m in messages:
                if isinstance(m, dict) and m.get("tool_calls"):
                    m["tool_calls"] = [dict(c) for c in m["tool_calls"]]
            ensure_signatures(messages, scope, self.signatures)
        return out

    @staticmethod
    def for_model(payload: dict[str, Any], model: str) -> bytes:
        """Serialize the payload for one attempt. Assistant messages carry a private `_model` marker
        (the model that produced them); thought signatures are model-specific, so those produced by a
        different model than the one serving this attempt are replaced with the placeholder."""
        messages = payload.get("messages") or []
        if not any(isinstance(m, dict) and "_model" in m for m in messages):
            return orjson.dumps({**payload, "model": model})
        cleaned = []
        for m in messages:
            if not isinstance(m, dict) or "_model" not in m:
                cleaned.append(m)
                continue
            produced_by = m["_model"]
            m = {k: v for k, v in m.items() if k != "_model"}
            if produced_by != model:
                m.pop("extra_content", None)
                if m.get("tool_calls"):
                    m["tool_calls"] = [{**c, "extra_content": {"google": {"thought_signature": PLACEHOLDER}}}
                                       if signature_of(c.get("extra_content")) else c for c in m["tool_calls"]]
            cleaned.append(m)
        return orjson.dumps({**payload, "model": model, "messages": cleaned})

    def remember_signatures(self, chunk: dict[str, Any], scope: str) -> None:
        for choice in chunk.get("choices") or []:
            payload = choice.get("delta") or choice.get("message") or {}
            for call in payload.get("tool_calls") or []:
                extra = call.get("extra_content")
                if signature_of(extra) and call.get("id"):
                    fn = call.get("function") or {}
                    self.signatures.put(call_key(scope, call["id"], fn.get("name", ""), fn.get("arguments") or ""),
                                        extra)

    # ------------------------------------------------------------ attempts
    async def _headers(self, request_id: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {await self.tokens.token()}", "X-Request-Id": request_id}

    def _classify(self, attempt: Attempt, status: int, body: bytes) -> RouteError:
        message, gstatus = _error_message(body)
        attempt.status, attempt.error = status, message[:300]
        if status == 400:
            return RouteError(BadRequest(f"Vertex rejected the request: {message}",
                                         code="provider_rejected_request"), False, attempt)
        if status == 404:
            return RouteError(NotFound(f"model {attempt.model} is not available in {attempt.location}",
                                       code="model_not_found"), True, attempt, unavailable=True)
        if status in (401, 403):
            return RouteError(ProviderError(f"Vertex denied access ({status} {gstatus}): {message}",
                                            code="provider_permission_denied"), True, attempt)
        if status == 429:
            return RouteError(Overloaded(f"Vertex capacity exhausted: {message}", code="provider_overloaded",
                                         retry_after=2), True, attempt)
        if status in (408, 500, 502, 503, 504):
            return RouteError(ProviderUnavailable(f"Vertex {status}: {message}"), True, attempt)
        return RouteError(ProviderError(f"Vertex {status}: {message}", details={"status": status}), False, attempt)

    def _record(self, attempt: Attempt, err: RouteError | None) -> None:
        state = self.states[attempt.location]
        state.requests += 1
        now = time.monotonic()
        if err is None:
            state.failures = 0
            state.open_until = 0.0
            state.last_ok_at = now
            CIRCUIT_OPEN.labels(attempt.location).set(0)
            PROVIDER_REQUESTS.labels(attempt.model, attempt.location, "ok").inc()
            return
        outcome = f"http_{attempt.status}" if attempt.status else ("empty" if err.retry_same else "transport")
        PROVIDER_REQUESTS.labels(attempt.model, attempt.location, outcome).inc()
        state.errors += 1
        state.last_error = attempt.error
        if err.unavailable:
            self._unavailable[(attempt.model, attempt.location)] = now + UNAVAILABLE_TTL_S
            return
        if err.retryable and not err.retry_same and attempt.status not in (401, 403):
            state.failures += 1
            if state.failures >= self.settings.vertex_circuit_failures:
                state.open_until = now + self.settings.vertex_circuit_reset_s
                CIRCUIT_OPEN.labels(attempt.location).set(1)
                log.warning("vertex circuit opened", extra={"location": attempt.location, "error": attempt.error})

    async def _run_routes(self, model: str, deadline: float, attempt_fn: Any) -> Any:
        """Call attempt_fn(model, location, timeout) over the route chain until one succeeds."""
        routes = self.routes(model)
        if not routes:
            raise NotFound(f"model {model} is not available in any configured location", code="model_not_found")
        tried: list[Attempt] = []
        last: RouteError | None = None
        auth_retried = False
        same_retried = False
        i = 0
        while i < len(routes) and len(tried) < self.settings.vertex_max_attempts:
            m, loc = routes[i]
            remaining = deadline - time.monotonic()
            if remaining <= 0.05:
                break
            attempt = Attempt(m, loc)
            tried.append(attempt)
            try:
                result = await attempt_fn(attempt, remaining)
            except RouteError as err:
                self._record(attempt, err)
                last = err
                if err.attempt.status == 401 and not auth_retried:
                    auth_retried = True
                    await self.tokens.invalidate()
                    continue  # same route, fresh token
                if err.retry_same and not same_retried:
                    same_retried = True
                    continue  # same route once more
                if not err.retryable:
                    raise err.error from None
                i += 1
                if self.settings.vertex_retry_backoff_s and i < len(routes) and not err.unavailable:
                    await asyncio.sleep(min(self.settings.vertex_retry_backoff_s * (1 + random.random()),
                                            max(0.0, deadline - time.monotonic() - 0.05)))
                continue
            self._record(attempt, None)
            if (m, loc) != routes[0]:
                PROVIDER_FAILOVERS.labels("model" if m != model else "location").inc()
            return result, tried
        if last is None:
            raise ProviderTimeout("no Vertex route answered before the first-token deadline")
        if isinstance(last.error, (Overloaded, ProviderUnavailable, ProviderError, NotFound)):
            last.error.details = {**last.error.details,
                                  "attempts": [{"model": a.model, "location": a.location, "status": a.status,
                                                "error": a.error} for a in tried]}
        raise last.error

    # ------------------------------------------------------------ streaming
    @asynccontextmanager
    async def stream(self, body: dict[str, Any], *, request_id: str, scope: str) -> AsyncIterator[VertexStream]:
        model = self.resolve_model(body.get("model"))
        payload = self.prepare({**body, "stream": True}, scope)
        payload.setdefault("stream_options", {"include_usage": True})
        started = time.perf_counter()
        deadline = time.monotonic() + self.settings.vertex_first_token_timeout_s
        stats = StreamStats()

        async def attempt_fn(attempt: Attempt, remaining: float) -> tuple[httpx.Response, Any, list]:
            url = self.base_url(attempt.location) + "/chat/completions"
            self._last_used[url.split("/")[2]] = time.monotonic()
            try:
                request = self.client.build_request("POST", url, content=self.for_model(payload, attempt.model),
                                                    headers={**await self._headers(request_id),
                                                             "Accept": "text/event-stream"})
                response = await asyncio.wait_for(self.client.send(request, stream=True), remaining)
            except TimeoutError:
                attempt.error = "timeout waiting for response headers"
                raise RouteError(ProviderTimeout(f"Vertex did not answer within {remaining:.1f}s"),
                                 True, attempt) from None
            except httpx.HTTPError as exc:
                attempt.error = f"{exc.__class__.__name__}"
                raise RouteError(ProviderUnavailable(f"Vertex unreachable: {exc.__class__.__name__}"),
                                 True, attempt) from exc
            if response.status_code >= 400:
                content = await response.aread()
                await response.aclose()
                raise self._classify(attempt, response.status_code, content)
            lines = response.aiter_lines()
            buffered: list[tuple[str, Any]] = []
            # read ahead to the first meaningful chunk so errors before it can still fail over
            try:
                while True:
                    left = deadline - time.monotonic()
                    line = await asyncio.wait_for(lines.__anext__(), max(0.05, left))
                    item = _parse_line(line)
                    if item is None:
                        continue
                    buffered.append(item)
                    if item[1] is DONE:
                        await response.aclose()
                        raise _empty(attempt)
                    if _meaningful(item[1]):
                        break
            except StopAsyncIteration:
                await response.aclose()
                raise _empty(attempt) from None
            except TimeoutError:
                await response.aclose()
                attempt.error = "no first token before deadline"
                raise RouteError(ProviderTimeout("Vertex produced no token before the first-token deadline"),
                                 True, attempt) from None
            except (httpx.HTTPError, ProviderError) as exc:
                await response.aclose()
                attempt.error = str(exc)[:200]
                raise RouteError(ProviderUnavailable(f"Vertex stream failed before the first token: {exc}"),
                                 True, attempt) from exc
            return response, lines, buffered

        with span("vertex.stream", **{"llm.model": model, "request.id": request_id}) as sp:
            (response, lines, buffered), tried = await self._run_routes(model, deadline, attempt_fn)
            stats.model, stats.location, stats.attempts = tried[-1].model, tried[-1].location, tried
            stats.upstream_ttft_s = time.perf_counter() - started
            UPSTREAM_TTFT.labels(stats.model, stats.location).observe(stats.upstream_ttft_s)
            sp.set_attribute("vertex.location", stats.location)
            sp.set_attribute("vertex.attempts", len(tried))
            sp.set_attribute("vertex.ttft_ms", round(stats.upstream_ttft_s * 1000, 1))
            for _, chunk in buffered:
                if chunk is not DONE:
                    self.remember_signatures(chunk, scope)
            try:
                yield VertexStream(response, lines, buffered, self.settings.vertex_stream_idle_timeout_s, stats,
                                   lambda c: self.remember_signatures(c, scope))
            finally:
                await response.aclose()  # closing mid-stream cancels generation upstream

    # -------------------------------------------------------- non-streaming
    async def complete(self, body: dict[str, Any], *, request_id: str, scope: str,
                       timeout: float | None = None) -> tuple[dict[str, Any], StreamStats]:
        model = self.resolve_model(body.get("model"))
        payload = self.prepare({**body, "stream": False}, scope)
        payload.pop("stream_options", None)
        deadline = time.monotonic() + (timeout or self.settings.vertex_request_timeout_s)

        async def attempt_fn(attempt: Attempt, remaining: float) -> dict[str, Any]:
            url = self.base_url(attempt.location) + "/chat/completions"
            self._last_used[url.split("/")[2]] = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    self.client.post(url, content=self.for_model(payload, attempt.model),
                                     headers=await self._headers(request_id)),
                    remaining)
            except TimeoutError:
                attempt.error = "timeout"
                raise RouteError(ProviderTimeout(f"Vertex did not answer within {remaining:.1f}s"),
                                 True, attempt) from None
            except httpx.HTTPError as exc:
                attempt.error = exc.__class__.__name__
                raise RouteError(ProviderUnavailable(f"Vertex unreachable: {exc.__class__.__name__}"),
                                 True, attempt) from exc
            if response.status_code >= 400:
                raise self._classify(attempt, response.status_code, response.content)
            result = orjson.loads(response.content)
            if not _meaningful(result):
                raise _empty(attempt)
            return result

        with span("vertex.complete", **{"llm.model": model, "request.id": request_id}):
            result, tried = await self._run_routes(model, deadline, attempt_fn)
        self.remember_signatures(result, scope)
        stats = StreamStats(model=tried[-1].model, location=tried[-1].location, attempts=tried)
        return result, stats

    # ------------------------------------------------------------ warmers
    async def _probe(self) -> None:
        """Validate access to the default model and warm the connection to the primary route."""
        body = {"model": self.default_model, "max_tokens": 1,
                "messages": [{"role": "user", "content": "ping"}]}
        for attempt_no in range(1, 31):
            try:
                await self.complete(body, request_id=f"startup-probe-{attempt_no}", scope="probe", timeout=20)
                self.probe_error = None
                self.probe_done = True
                log.info("vertex startup probe ok", extra={"model": self.default_model})
                return
            except GatewayError as exc:
                self.probe_error = exc.message[:300]
                log.warning("vertex startup probe failed", extra={"error": self.probe_error, "attempt": attempt_no})
                if exc.status == 400 or exc.code in ("model_not_found", "provider_permission_denied"):
                    # configuration problem: report not-ready loudly but keep retrying slowly
                    await asyncio.sleep(30)
                else:
                    await asyncio.sleep(min(30, 2 * attempt_no))
        self.probe_done = True  # give up gating readiness; per-request errors will surface

    async def _keepalive(self) -> None:
        interval = self.settings.vertex_keepalive_interval_s
        hosts = {self.base_url(loc).split("/")[2] for loc in self.locations}
        scheme = "https" if not self.settings.vertex_endpoint_template.startswith("http://") else "http"
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            for host in hosts:
                if now - self._last_used.get(host, 0.0) < interval:
                    continue
                try:
                    # any response keeps the pooled TLS/HTTP2 connection alive; no tokens are spent
                    await asyncio.wait_for(self.client.head(f"{scheme}://{host}/"), 5)
                    self._last_used[host] = time.monotonic()
                except (TimeoutError, httpx.HTTPError):
                    pass
