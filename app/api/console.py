"""Developer console at `/`: one static page to try every HTTP route and the realtime WebSocket.

The page holds no secrets. The API key typed into it stays in the browser and is only sent to this server."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import Response

router = APIRouter(include_in_schema=False)

_PAGE = (Path(__file__).resolve().parent.parent / "static" / "console.html").read_bytes()


@router.get("/")
async def console(request: Request) -> Response:
    host = request.headers.get("host", "")
    # inline script and style only; fetches and the WebSocket go to this server
    csp = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; "
           f"connect-src 'self' ws://{host} wss://{host}; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
    return Response(_PAGE, media_type="text/html; charset=utf-8", headers={
        "Content-Security-Policy": csp, "Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer"})
