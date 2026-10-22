"""Per-principal token-bucket rate limiting (per instance)."""
from __future__ import annotations

import time
from collections import OrderedDict

from app.core.errors import RateLimited


class TokenBucket:
    __slots__ = ("capacity", "rate", "tokens", "updated")

    def __init__(self, per_minute: int) -> None:
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self.tokens = float(per_minute)
        self.updated = time.monotonic()

    def take(self) -> float:
        """Consume one token; returns 0 on success or the seconds until one is available."""
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= 1:
            self.tokens -= 1
            return 0.0
        return (1 - self.tokens) / self.rate


class RateLimiter:
    def __init__(self, max_buckets: int = 100_000) -> None:
        self._buckets: OrderedDict[str, TokenBucket] = OrderedDict()
        self.max_buckets = max_buckets

    def check(self, key: str, per_minute: int) -> None:
        if per_minute <= 0:
            return
        bucket = self._buckets.get(key)
        if bucket is None or bucket.capacity != per_minute:
            bucket = self._buckets[key] = TokenBucket(per_minute)
            if len(self._buckets) > self.max_buckets:
                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(key)
        wait = bucket.take()
        if wait > 0:
            raise RateLimited(f"rate limit of {per_minute} requests/min exceeded", retry_after=round(wait, 2))
