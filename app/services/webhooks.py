"""Signed webhook delivery with retries.

Each delivery is a JSON POST with:
    X-AICA-Event: execution.completed | execution.failed | execution.cancelled | session.ended
    X-AICA-Delivery: <unique id>   (stable across retries: receivers de-duplicate on it)
    X-AICA-Signature: t=<unix>,v1=<hex HMAC-SHA256(tenant webhook secret, "<t>.<body>")>
Retries with exponential backoff on network errors, 429 and 5xx (WEBHOOK_MAX_ATTEMPTS).
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import orjson

from app.config import Settings
from app.core.errors import GatewayError
from app.core.ids import new_id
from app.core.logging import get_logger
from app.core.metrics import WEBHOOKS
from app.core.tracing import inject_headers
from app.security.crypto import SecretBox, sign_payload
from app.security.netguard import NetGuard

log = get_logger("app.webhooks")


class WebhookSender:
    def __init__(self, settings: Settings, guard: NetGuard, box: SecretBox) -> None:
        self.settings = settings
        self.guard = guard
        self.box = box
        self.client = httpx.AsyncClient(follow_redirects=False, timeout=settings.webhook_timeout_s,
                                        headers={"User-Agent": "aica-llm-server-webhooks"},
                                        event_hooks={"response": [guard.response_hook("webhook URL")]})
        self._tasks: set[asyncio.Task] = set()

    def send(self, url: str, event: str, payload: dict[str, Any], secret_enc: str | None,
             on_result: Callable[[dict[str, Any]], Awaitable[None]] | None = None) -> None:
        task = asyncio.create_task(self._deliver(url, event, payload, secret_enc, on_result), name="webhook")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _deliver(self, url: str, event: str, payload: dict[str, Any], secret_enc: str | None,
                       on_result: Callable[[dict[str, Any]], Awaitable[None]] | None) -> None:
        delivery = new_id("dlv")
        body = orjson.dumps({"event": event, "delivery_id": delivery, "data": payload}, default=str)
        result: dict[str, Any] = {"delivery_id": delivery, "status": "failed", "attempts": 0, "last_error": None}
        try:
            await self.guard.check(url, "callback URL")
        except GatewayError as exc:
            result["last_error"] = exc.message
            WEBHOOKS.labels("rejected").inc()
            await self._report(on_result, result)
            return
        secret = self.box.decrypt(secret_enc) if secret_enc else None
        for attempt in range(1, self.settings.webhook_max_attempts + 1):
            result["attempts"] = attempt
            headers = {"Content-Type": "application/json", "X-AICA-Event": event, "X-AICA-Delivery": delivery}
            if secret:
                headers["X-AICA-Signature"] = sign_payload(secret, body)
            inject_headers(headers)
            try:
                response = await self.client.post(url, content=body, headers=headers)
                if response.status_code < 300:
                    result.update(status="delivered", response_status=response.status_code, last_error=None)
                    WEBHOOKS.labels("delivered").inc()
                    break
                result["last_error"] = f"HTTP {response.status_code}"
                if response.status_code < 500 and response.status_code != 429:
                    break  # the receiver rejected it; retrying will not help
            except httpx.HTTPError as exc:
                result["last_error"] = exc.__class__.__name__
            except GatewayError as exc:  # the URL now resolves to a forbidden address
                result["last_error"] = exc.message
                break
            if attempt < self.settings.webhook_max_attempts:
                await asyncio.sleep(min(300.0, 2.0 ** attempt) * (0.5 + random.random()))
        if result["status"] != "delivered":
            WEBHOOKS.labels("failed").inc()
            log.warning("webhook delivery failed", extra={"event": event, "error": result["last_error"]})
        await self._report(on_result, result)

    @staticmethod
    async def _report(on_result: Callable[[dict[str, Any]], Awaitable[None]] | None, result: dict[str, Any]) -> None:
        if on_result is not None:
            try:
                await on_result(result)
            except GatewayError as exc:
                log.warning("could not record webhook result", extra={"error": exc.message})

    async def aclose(self, timeout: float = 5.0) -> None:
        if self._tasks:
            await asyncio.wait(list(self._tasks), timeout=timeout)
        for task in self._tasks:
            task.cancel()
        await self.client.aclose()
