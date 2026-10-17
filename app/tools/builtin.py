"""Built-in server-side tools (opt-in per session via `server_tools`)."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.tools.registry import tool


@tool(
    description="Get the current date and time. Use it whenever the user asks about today's date, "
    "the day of the week or the time.",
    parameters={
        "type": "object",
        "properties": {
            "timezone": {"type": "string", "description": "IANA timezone, e.g. Asia/Kolkata", "default": "UTC"}
        },
    },
    timeout_s=1.0,
)
def get_current_time(timezone: str = "UTC") -> dict:
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        zone, timezone = ZoneInfo("UTC"), "UTC"
    now = datetime.now(zone)
    return {"timezone": timezone, "iso": now.isoformat(timespec="seconds"),
            "date": now.strftime("%Y-%m-%d"), "weekday": now.strftime("%A"), "time": now.strftime("%H:%M")}
