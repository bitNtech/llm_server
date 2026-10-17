"""orjson-backed JSON response (FastAPI deprecated its FastJSONResponse)."""
from __future__ import annotations

from typing import Any

import orjson
from starlette.responses import JSONResponse


class FastJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, default=str)
