"""Gemini thought signatures.

Gemini 3+ models attach an opaque `thought_signature` (in `extra_content.google`) to the first
function call of each step and require it back, unchanged, when the conversation is replayed;
without it Vertex rejects the request with HTTP 400. Stateful sessions keep signatures in their
history. Stateless clients (OpenAI SDKs, LiveKit's OpenAI plugin) drop unknown fields, so the
gateway remembers signatures it has streamed out and re-attaches them on the way back in. When a
signature is unknown (expired, another instance, or a different model produced the call), the
documented placeholder is used so the request is still accepted.
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any

PLACEHOLDER = "skip_thought_signature_validator"


def call_key(scope: str, call_id: str, name: str, arguments: str) -> str:
    raw = f"{scope}\x1f{call_id}\x1f{name}\x1f{arguments}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def signature_of(extra: Any) -> str | None:
    if isinstance(extra, dict):
        google = extra.get("google")
        if isinstance(google, dict) and isinstance(google.get("thought_signature"), str):
            return google["thought_signature"]
    return None


class SignatureCache:
    def __init__(self, max_items: int = 200_000, ttl_s: float = 6 * 3600) -> None:
        self.max_items = max_items
        self.ttl_s = ttl_s
        self._items: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()

    def put(self, key: str, extra: dict[str, Any]) -> None:
        self._items[key] = (time.monotonic() + self.ttl_s, extra)
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def get(self, key: str) -> dict[str, Any] | None:
        item = self._items.get(key)
        if item is None:
            return None
        if item[0] < time.monotonic():
            self._items.pop(key, None)
            return None
        return item[1]

    def __len__(self) -> int:
        return len(self._items)


def ensure_signatures(messages: list[dict[str, Any]], scope: str, cache: SignatureCache | None) -> None:
    """Re-attach known signatures to assistant tool calls and guarantee the first call of every
    assistant step carries one (in place)."""
    for msg in messages:
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        calls = msg["tool_calls"]
        for call in calls:
            if signature_of(call.get("extra_content")) or cache is None:
                continue
            fn = call.get("function") or {}
            known = cache.get(call_key(scope, str(call.get("id")), fn.get("name", ""), fn.get("arguments") or ""))
            if known:
                call["extra_content"] = known
        if not signature_of(calls[0].get("extra_content")):
            calls[0]["extra_content"] = {"google": {"thought_signature": PLACEHOLDER}}
