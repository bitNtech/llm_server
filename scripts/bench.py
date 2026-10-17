"""Latency benchmark for capacity planning: TTFT / inter-token latency at a given concurrency.

    .venv/bin/python scripts/bench.py --concurrency 1,8,32 --requests 64 [--url ...] [--api-key ...]

Each simulated call streams one short voice-style reply through /v1/chat/completions.
Pick the highest concurrency whose p95 TTFT still meets your latency budget, then set
MAX_CONCURRENT_GENERATIONS a little above it.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time

import httpx
import orjson

PROMPT = ("You are a hospital receptionist on a phone call. The caller says: 'Hi, I'd like to book an "
          "appointment with a cardiologist next Tuesday morning.' Reply in one or two short sentences.")


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


async def one(client: httpx.AsyncClient, max_tokens: int) -> tuple[float, list[float], int, str | None]:
    start = time.perf_counter()
    ttft, last, gaps, tokens = None, None, [], 0
    try:
        async with client.stream("POST", "/v1/chat/completions", json={
                "stream": True, "max_tokens": max_tokens, "temperature": 0.7,
                "messages": [{"role": "user", "content": PROMPT}]}) as r:
            if r.status_code != 200:
                return 0.0, [], 0, f"HTTP {r.status_code}"
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = orjson.loads(line[6:])
                if "error" in chunk:
                    return 0.0, [], 0, chunk["error"].get("code", "error")
                if any((c["delta"].get("content")) for c in chunk.get("choices", [])):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - start
                    elif last is not None:
                        gaps.append(now - last)
                    last = now
                    tokens += 1
    except httpx.HTTPError as exc:
        return 0.0, [], 0, exc.__class__.__name__
    return ttft or 0.0, gaps, tokens, None


async def run_level(url: str, headers: dict, concurrency: int, requests: int, max_tokens: int) -> None:
    limits = httpx.Limits(max_connections=concurrency + 4)
    async with httpx.AsyncClient(base_url=url, headers=headers, timeout=300, limits=limits) as client:
        sem = asyncio.Semaphore(concurrency)

        async def guarded() -> tuple[float, list[float], int, str | None]:
            async with sem:
                return await one(client, max_tokens)

        start = time.perf_counter()
        results = await asyncio.gather(*(guarded() for _ in range(requests)))
        wall = time.perf_counter() - start
    ttfts = [r[0] for r in results if r[3] is None]
    gaps = [g for r in results if r[3] is None for g in r[1]]
    tokens = sum(r[2] for r in results)
    errors = [r[3] for r in results if r[3]]
    print(f"c={concurrency:<4} ok={len(ttfts):<4} err={len(errors):<3} "
          f"TTFT p50={pct(ttfts, 50) * 1000:7.0f}ms p95={pct(ttfts, 95) * 1000:7.0f}ms  "
          f"ITL p50={pct(gaps, 50) * 1000:5.1f}ms p95={pct(gaps, 95) * 1000:5.1f}ms  "
          f"throughput={tokens / wall:7.1f} tok/s" + (f"  errors={sorted(set(errors))}" if errors else ""))
    if ttfts:
        print(f"         mean TTFT {statistics.mean(ttfts) * 1000:.0f} ms over {wall:.1f}s wall")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("BENCH_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY", ""))
    parser.add_argument("--concurrency", default="1,4,16")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    for level in (int(c) for c in args.concurrency.split(",")):
        asyncio.run(run_level(args.url, headers, level, max(args.requests, level), args.max_tokens))


if __name__ == "__main__":
    main()
