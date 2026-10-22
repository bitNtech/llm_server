"""HTTP tools: tenant-defined endpoints the server calls when the model invokes the tool.

Request contract (documented in docs/TOOLS.md):
  * URL path placeholders `{name}` are filled from the arguments (URL-encoded) and removed from them
  * GET/DELETE send the remaining arguments as query parameters; POST/PUT/PATCH send JSON:
      body_mode=arguments -> the arguments object itself
      body_mode=envelope  -> {"tool", "call_id", "session_id", "agent_id", "arguments"}
  * headers: configured static headers, decrypted secret headers, `Idempotency-Key` (stable per tool
    call, so retried/replayed calls can be de-duplicated by the receiver), X-AICA-Tenant-Id,
    X-AICA-Session-Id, X-AICA-Call-Id, W3C traceparent, and `X-AICA-Signature: t=..,v1=..`
    (HMAC-SHA256 over "<t>.<body>") when a signing secret is configured
  * 2xx JSON/text responses become the tool result; other statuses become an error result the model
    can react to. Redirects are not followed (they could bypass the network policy).
"""
from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import quote

import httpx
import orjson

from app.config import Settings
from app.core.errors import GatewayError
from app.core.tracing import inject_headers
from app.security.crypto import SecretBox, sign_payload
from app.security.netguard import NetGuard
from app.tools.base import ToolContext, ToolOutcome, ToolSpec

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_HOP_BY_HOP = {"host", "content-length", "connection", "transfer-encoding", "upgrade"}


class HttpToolRunner:
    def __init__(self, settings: Settings, guard: NetGuard, box: SecretBox) -> None:
        self.settings = settings
        self.guard = guard
        self.box = box
        self.client = httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(settings.tool_timeout_s, connect=min(5.0, settings.tool_timeout_s)),
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=64, keepalive_expiry=60),
            headers={"User-Agent": "aica-llm-server-tools"},
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def run(self, spec: ToolSpec, args: dict[str, Any], ctx: ToolContext, call_id: str,
                  idempotency_key: str, timeout: float) -> ToolOutcome:
        cfg = spec.target["http"]
        args = dict(args)
        method = cfg.get("method", "POST").upper()

        def fill(match: re.Match[str]) -> str:
            value = args.pop(match.group(1), "")
            return quote(str(value), safe="")

        url = _PLACEHOLDER.sub(fill, cfg["url"])
        await self.guard.check(url, "tool URL")
        headers = {k: v for k, v in (cfg.get("headers") or {}).items() if k.lower() not in _HOP_BY_HOP}
        secret_headers = self.box.decrypt(spec.target.get("secret_headers_enc")) or {}
        headers.update({k: v for k, v in secret_headers.items() if k.lower() not in _HOP_BY_HOP})
        headers.update({"Idempotency-Key": idempotency_key, "X-AICA-Tenant-Id": ctx.tenant_id,
                        "X-AICA-Call-Id": call_id, "X-AICA-Tool": spec.name})
        if ctx.session_id:
            headers["X-AICA-Session-Id"] = ctx.session_id
        if ctx.agent_id:
            headers["X-AICA-Agent-Id"] = ctx.agent_id
        inject_headers(headers)

        content = b""
        params = None
        if method in ("GET", "DELETE"):
            params = {k: v if isinstance(v, str) else orjson.dumps(v).decode() for k, v in args.items()}
        else:
            payload: Any = args
            if cfg.get("body_mode") == "envelope":
                payload = {"tool": spec.name, "call_id": call_id, "session_id": ctx.session_id,
                           "agent_id": ctx.agent_id, "arguments": args}
            content = orjson.dumps(payload)
            headers["Content-Type"] = "application/json"
        signing_secret = self.box.decrypt(spec.target.get("signing_secret_enc"))
        if signing_secret:
            headers["X-AICA-Signature"] = sign_payload(signing_secret, content)

        start = time.perf_counter()
        for attempt in (1, 2):
            try:
                async with self.client.stream(method, url, params=params, content=content or None,
                                              headers=headers, timeout=timeout) as response:
                    self.guard.check_peer(url, response, "tool URL")
                    body = await self._read_limited(response)
                    status = response.status_code
                    ctype = response.headers.get("content-type", "")
                break
            except httpx.ConnectError as exc:
                if attempt == 2:  # nothing reached the server: one retry is safe even for side effects
                    return ToolOutcome(f"tool endpoint unreachable: {exc.__class__.__name__}", True)
            except httpx.TimeoutException:
                return ToolOutcome(f"tool endpoint timed out after {timeout:.1f}s", True)
            except httpx.HTTPError as exc:
                return ToolOutcome(f"tool request failed: {exc.__class__.__name__}", True)
            except GatewayError as exc:
                return ToolOutcome(exc.message, True)
        duration = (time.perf_counter() - start) * 1000
        data: Any = None
        if "json" in ctype:
            try:
                data = orjson.loads(body)
            except orjson.JSONDecodeError:
                data = None
        text = orjson.dumps(data).decode() if data is not None else body.decode(errors="replace")
        if status >= 400:
            return ToolOutcome(f"HTTP {status}: {text[:2000]}", True, data=data, duration_ms=duration)
        return ToolOutcome(text, False, data=data, duration_ms=duration)

    async def _read_limited(self, response: httpx.Response) -> bytes:
        limit = self.settings.http_tool_max_response_bytes
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > limit:
                raise GatewayError(f"tool response exceeds {limit} bytes", code="tool_response_too_large")
            chunks.append(chunk)
        return b"".join(chunks)
