"""orjson-backed JSON response."""
from __future__ import annotations

from typing import Any

import orjson
from starlette.responses import JSONResponse

from app.core.ids import jsonable


class FastJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return orjson.dumps(jsonable(content), default=str)
