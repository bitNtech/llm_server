"""Identifiers and time helpers."""
from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime
from typing import Any

_CLIENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SLUG = re.compile(r"[^a-z0-9]+")


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(12)}"


def valid_client_id(value: str) -> bool:
    """Client-chosen ids (session ids such as a call id): 1-128 chars of [A-Za-z0-9_-]."""
    return bool(_CLIENT_ID.match(value))


def slug(value: str, max_len: int = 40) -> str:
    return _SLUG.sub("_", value.lower()).strip("_")[:max_len] or "x"


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(value: Any) -> Any:
    """Render datetimes (incl. Firestore DatetimeWithNanoseconds) as ISO-8601 UTC strings."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return value


def jsonable(value: Any) -> Any:
    """Recursively convert datetimes so a Firestore document can be returned as JSON."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return iso(value)
