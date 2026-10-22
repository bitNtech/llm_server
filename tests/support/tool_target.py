"""HTTP tool / webhook receiver used by the integration tests. Records every request."""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="tool-target")
RECORDS: list[dict[str, Any]] = []
BOOKINGS: dict[str, dict[str, Any]] = {}


async def _record(request: Request, name: str) -> dict[str, Any]:
    raw = await request.body()
    entry = {"name": name, "method": request.method, "path": request.url.path, "query": dict(request.query_params),
             "headers": {k.lower(): v for k, v in request.headers.items()}, "body": raw.decode(errors="replace")}
    RECORDS.append(entry)
    return entry


@app.post("/book")
async def book(request: Request) -> JSONResponse:
    entry = await _record(request, "book")
    key = entry["headers"].get("idempotency-key", "")
    if key not in BOOKINGS:  # an idempotent receiver
        BOOKINGS[key] = {"booking_id": f"B{len(BOOKINGS) + 1:04d}", "status": "confirmed"}
    return JSONResponse(BOOKINGS[key])


@app.get("/patients/{patient_id}")
async def patient(patient_id: str, request: Request) -> JSONResponse:
    await _record(request, "patient")
    return JSONResponse({"patient_id": patient_id, "name": "Asha", "next_visit": "2026-11-02"})


@app.post("/fail")
async def fail(request: Request) -> JSONResponse:
    await _record(request, "fail")
    return JSONResponse({"error": "downstream unavailable"}, status_code=503)


@app.post("/slow")
async def slow(request: Request) -> JSONResponse:
    await _record(request, "slow")
    await asyncio.sleep(1.0)
    return JSONResponse({"done": True})


@app.post("/webhook")
async def webhook(request: Request) -> JSONResponse:
    await _record(request, "webhook")
    return JSONResponse({"ok": True})


@app.get("/records")
async def records() -> list:
    return RECORDS


@app.post("/reset")
async def reset() -> dict:
    RECORDS.clear()
    BOOKINGS.clear()
    return {"ok": True}
