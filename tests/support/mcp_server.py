"""A real MCP server (official SDK, Streamable HTTP) used by the integration tests."""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

CALLS: list[dict] = []

server = MCPServer(name="clinic-mcp", instructions="Clinic test tools")


@server.tool(description="Look up an order by id")
def lookup_order(order_id: str) -> dict:
    CALLS.append({"tool": "lookup_order", "order_id": order_id})
    return {"order_id": order_id, "status": "shipped", "eta_days": 2}


@server.tool(description="Add two integers")
def add(a: int, b: int) -> int:
    CALLS.append({"tool": "add", "a": a, "b": b})
    return a + b


@server.tool(description="Always fails")
def broken() -> str:
    raise RuntimeError("backend exploded")


def build_app():
    return server.streamable_http_app(streamable_http_path="/mcp", stateless_http=False)
