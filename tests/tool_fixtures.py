"""Server tools used by the integration tests (loaded via TOOLS_MODULES)."""
from __future__ import annotations

import asyncio

from app.tools import tool


@tool(description="A deliberately slow lookup", parameters={"type": "object", "properties": {}})
async def slow_lookup() -> str:
    await asyncio.sleep(3)
    return "done"
