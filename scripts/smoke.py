"""End-to-end smoke test against a running server (local or EC2).

    .venv/bin/python scripts/smoke.py [--url http://127.0.0.1:8000] [--api-key KEY] [--tools]

Checks readiness, OpenAI-compatible streaming, a stateful session turn, the realtime WebSocket
(including barge-in cancellation) and, with --tools, a server-side tool call. Prints latencies.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

import httpx
import orjson
from websockets.asyncio.client import connect


def ok(name: str, detail: str = "") -> None:
    print(f"  ok   {name:<34} {detail}")


def fail(name: str, detail: str) -> None:
    print(f"  FAIL {name:<34} {detail}")
    raise SystemExit(1)


def openai_stream(client: httpx.Client, prompt: str) -> tuple[str, float, float]:
    start = time.perf_counter()
    ttft, text = None, ""
    with client.stream("POST", "/v1/chat/completions", json={
            "stream": True, "max_tokens": 64, "messages": [{"role": "user", "content": prompt}]}) as r:
        if r.status_code != 200:
            fail("openai stream", f"HTTP {r.status_code}: {r.read().decode()[:300]}")
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = orjson.loads(line[6:])
            if "error" in chunk:
                fail("openai stream", str(chunk["error"]))
            for choice in chunk.get("choices", []):
                delta = choice["delta"].get("content") or ""
                if delta and ttft is None:
                    ttft = time.perf_counter() - start
                text += delta
    return text, ttft or 0.0, time.perf_counter() - start


async def realtime(ws_url: str, headers: dict[str, str], prompt: str) -> None:
    async with connect(f"{ws_url}/v1/realtime", additional_headers=headers, open_timeout=10) as ws:
        async def recv() -> dict:
            return orjson.loads(await asyncio.wait_for(ws.recv(), 120))

        created = await recv()
        if created["type"] != "session.created":
            fail("realtime", f"unexpected first event {created}")
        await ws.send(orjson.dumps({"type": "session.update", "session": {
            "system_prompt": "You are a concise, friendly assistant.", "segmentation": True,
            "params": {"max_tokens": 64}}}).decode())
        await recv()
        start = time.perf_counter()
        await ws.send(orjson.dumps({"type": "response.create", "input": prompt}).decode())
        first_segment = None
        while True:
            event = await recv()
            if event["type"] == "response.text.segment" and first_segment is None:
                first_segment = time.perf_counter() - start
            if event["type"] == "error":
                fail("realtime", str(event))
            if event["type"] == "response.done":
                break
        if event["status"] not in ("completed", "incomplete"):
            fail("realtime turn", str(event))
        seg = f"{first_segment * 1000:.0f} ms" if first_segment else "-"
        ok("realtime turn", f"ttft {event['timings']['ttft_ms']} ms, first TTS segment {seg}, "
                            f"{event['usage']['completion_tokens']} tokens: {event['text'][:60]!r}")

        await ws.send(orjson.dumps({"type": "response.create", "input": "Count slowly from one to fifty.",
                                    "params": {"max_tokens": 200}}).decode())
        while (await recv())["type"] != "response.text.delta":
            pass
        t0 = time.perf_counter()
        await ws.send(orjson.dumps({"type": "response.cancel", "spoken_text": "One, two"}).decode())
        while (event := await recv())["type"] != "response.done":
            pass
        if event["status"] != "cancelled":
            fail("barge-in cancel", str(event))
        ok("barge-in cancel", f"acknowledged in {(time.perf_counter() - t0) * 1000:.0f} ms")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("SMOKE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.environ.get("SMOKE_API_KEY", ""))
    parser.add_argument("--prompt", default="Say hello in one short sentence.")
    parser.add_argument("--tools", action="store_true", help="also exercise a server-side tool call")
    args = parser.parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    print(f"smoke test against {args.url}")

    with httpx.Client(base_url=args.url, headers=headers, timeout=180) as client:
        r = client.get("/health/ready")
        if r.status_code != 200:
            fail("readiness", f"HTTP {r.status_code}: {r.text}")
        ok("readiness", r.json().get("model", ""))

        text, ttft, total = openai_stream(client, args.prompt)
        ok("openai streaming", f"ttft {ttft * 1000:.0f} ms, total {total * 1000:.0f} ms: {text.strip()[:60]!r}")

        sid = client.post("/v1/sessions", json={"system_prompt": "You are a concise assistant."}).json()["id"]
        turn = {"stream": False, "params": {"max_tokens": 48}}
        client.post(f"/v1/sessions/{sid}/responses", json={"input": args.prompt, **turn})
        second = client.post(f"/v1/sessions/{sid}/responses",
                             json={"input": "Thanks! And repeat that briefly.", **turn}).json()
        if second.get("status") not in ("completed", "incomplete"):
            fail("session turns", str(second))
        usage = second["usage"]
        ok("session turns", f"turn2 ttft {second['timings']['ttft_ms']} ms, "
                            f"prefix-cached {usage['cached_tokens']}/{usage['prompt_tokens']} tokens")

        if args.tools:
            sid = client.post("/v1/sessions", json={
                "system_prompt": "Use tools when they help.", "server_tools": ["get_current_time"]}).json()["id"]
            r = client.post(f"/v1/sessions/{sid}/responses", json={
                "input": "What is the date today in Asia/Kolkata? Use the tool.", "stream": False,
                "params": {"max_tokens": 128}}).json()
            calls = r.get("server_tool_calls") or []
            if not calls:
                fail("server tool call", f"model did not call the tool: {r.get('text')!r}")
            ok("server tool call", f"{calls[0]['name']}({calls[0]['arguments']}) -> {r['text'][:50]!r}")
        client.delete(f"/v1/sessions/{sid}")

        metrics = client.get("/metrics")
        if metrics.status_code == 200 and "llm_time_to_first_token_seconds" in metrics.text:
            ok("metrics")

    ws_url = args.url.replace("http://", "ws://").replace("https://", "wss://")
    asyncio.run(realtime(ws_url, headers, args.prompt))
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
