"""Small in-process TTL cache (configuration lookups on the hot path)."""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")
_MISSING = object()


class TTLCache(Generic[K, V]):
    def __init__(self, ttl_s: float, max_items: int = 10_000) -> None:
        self.ttl_s = ttl_s
        self.max_items = max_items
        self._items: OrderedDict[K, tuple[float, V]] = OrderedDict()

    def get(self, key: K, default: Any = None) -> V | Any:
        item = self._items.get(key, _MISSING)
        if item is _MISSING:
            return default
        expires, value = item  # type: ignore[misc]
        if self.ttl_s and expires < time.monotonic():
            self._items.pop(key, None)
            return default
        self._items.move_to_end(key)
        return value

    def put(self, key: K, value: V) -> None:
        self._items[key] = (time.monotonic() + self.ttl_s, value)
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def pop(self, key: K) -> None:
        self._items.pop(key, None)

    def clear(self) -> None:
        self._items.clear()
