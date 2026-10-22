"""Latency benchmark against a running server (the voice path: realtime WebSocket turns).

    .venv/bin/python scripts/bench.py --url http://127.0.0.1:8000 --api-key aica_... --agent-id agent_... \
        --turns 10 --concurrency 1,4

Reports, per concurrency level, client-observed time to first token (what the caller waits for),
Vertex's own time to first token, and the server's overhead (the difference), as p50/p90/max.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import orjson
import websockets

PROMPTS = ["Hi, what are your visiting hours?", "Can I book an appointment for tomorrow morning?",
           "Do you have a cardiologist available?", "What documents should I bring?",
           "Is there parking at the hospital?"]


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else float("nan")


async def caller(url: str, key: str, agent_id: str, turns: int, out: list[tuple[float, float, float]],
                 errors: list[str]) -> None:
    ws_url = url.replace("http", "ws", 1) + f"/v1/realtime?agent_id={agent_id}"
    async with websockets.connect(ws_url, additional_headers={"Authorization": f"Bearer {key}"},
                                  max_size=2**22) as ws:
        await ws.recv()  # session.created
        for i in range(turns):
            start = time.perf_counter()
            await ws.send(orjson.dumps({"type": "response.create", "input": PROMPTS[i % len(PROMPTS)]}).decode())
            first = None
            while True:
                event = orjson.loads(await ws.recv())
                if event["type"] == "response.text.delta" and first is None:
                    first = (time.perf_counter() - start) * 1000
                if event["type"] == "response.done":
                    if event["status"] != "completed" or first is None:
                        errors.append(event.get("error", {}).get("code", event["status"]))
                    else:
                        t = event["timings"]
                        out.append((first, t["upstream_ttft_ms"] or 0.0, t["gateway_overhead_ms"] or 0.0))
                    break


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--concurrency", default="1")
    args = parser.parse_args()
    print(f"{'conc':>4} {'n':>4} {'err':>4} | {'client TTFT p50/p90/max (ms)':>30} | "
          f"{'Vertex TTFT p50/p90':>20} | {'server overhead p50/max':>24}")
    for conc in (int(c) for c in args.concurrency.split(",")):
        samples: list[tuple[float, float, float]] = []
        errors: list[str] = []
        await asyncio.gather(*(caller(args.url, args.api_key, args.agent_id, args.turns, samples, errors)
                               for _ in range(conc)))
        client = [s[0] for s in samples]
        upstream = [s[1] for s in samples]
        overhead = [s[2] for s in samples]
        print(f"{conc:>4} {len(samples):>4} {len(errors):>4} | "
              f"{statistics.median(client):>8.0f} /{pct(client, .9):>6.0f} /{max(client):>6.0f}       | "
              f"{statistics.median(upstream):>8.0f} /{pct(upstream, .9):>6.0f}   | "
              f"{statistics.median(overhead):>10.1f} /{max(overhead):>7.1f}")


if __name__ == "__main__":
    asyncio.run(main())
