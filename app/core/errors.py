"""Error taxonomy. Every error leaves the server as one envelope (HTTP body, SSE frame or WS event):

    {"error": {"type": "...", "code": "...", "message": "...", "request_id": "...", "details": {...}}}
"""
from __future__ import annotations

from typing import Any


class GatewayError(Exception):
    status: int = 500
    type: str = "server_error"
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status: int | None = None,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status:
            self.status = status
        self.retry_after = retry_after
        self.details = details or {}

    def to_dict(self, request_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"type": self.type, "code": self.code, "message": self.message}
        if self.details:
            body["details"] = self.details
        if request_id:
            body["request_id"] = request_id
        return {"error": body}

    @property
    def retryable(self) -> bool:
        return self.status in (429, 502, 503, 504)


class BadRequest(GatewayError):
    status, type, code = 400, "invalid_request_error", "invalid_request"


class Unauthorized(GatewayError):
    status, type, code = 401, "authentication_error", "invalid_api_key"


class Forbidden(GatewayError):
    status, type, code = 403, "permission_error", "forbidden"


class NotFound(GatewayError):
    status, type, code = 404, "invalid_request_error", "not_found"


class Conflict(GatewayError):
    status, type, code = 409, "invalid_request_error", "conflict"


class PreconditionFailed(GatewayError):
    status, type, code = 412, "invalid_request_error", "version_mismatch"


class PayloadTooLarge(GatewayError):
    status, type, code = 413, "invalid_request_error", "payload_too_large"


class RateLimited(GatewayError):
    status, type, code = 429, "rate_limit_error", "rate_limited"


class Overloaded(GatewayError):
    status, type, code = 429, "rate_limit_error", "overloaded"


class ProviderError(GatewayError):
    status, type, code = 502, "server_error", "provider_error"


class ProviderUnavailable(GatewayError):
    status, type, code = 503, "server_error", "provider_unavailable"


class StoreUnavailable(GatewayError):
    status, type, code = 503, "server_error", "store_unavailable"


class ShuttingDown(GatewayError):
    status, type, code = 503, "server_error", "shutting_down"


class ProviderTimeout(GatewayError):
    status, type, code = 504, "server_error", "provider_timeout"
