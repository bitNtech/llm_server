"""Google Cloud credentials shared by Vertex AI and Firestore.

Resolution order: GCP_CREDENTIALS_JSON (inline service-account JSON or its base64) -> Application
Default Credentials (GOOGLE_APPLICATION_CREDENTIALS key file, workload identity / metadata server on
GCP, or `gcloud auth application-default login` on a workstation).

OAuth access tokens are refreshed in a background task ahead of expiry, so the request path never
waits for a token refresh. A refresh is single-flight: concurrent callers share one refresh.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time
from datetime import UTC, datetime
from typing import Any, Protocol

import google.auth
from google.auth import exceptions as auth_exceptions
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request
from google.oauth2 import service_account

from app.core.errors import ProviderError
from app.core.logging import get_logger

log = get_logger("app.gcp")

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
REFRESH_MARGIN_S = 300.0


class TokenSource(Protocol):
    async def token(self) -> str: ...
    async def invalidate(self) -> None: ...
    def start(self) -> None: ...
    async def stop(self) -> None: ...
    def snapshot(self) -> dict[str, Any]: ...


def load_credentials(credentials_json: str) -> tuple[Credentials, str | None]:
    """Return (credentials, project id discovered from them, if any)."""
    raw = credentials_json.strip()
    if raw:
        if not raw.startswith("{"):
            try:
                raw = base64.b64decode(raw).decode()
            except (binascii.Error, UnicodeDecodeError) as exc:
                raise ValueError("GCP_CREDENTIALS_JSON is neither JSON nor base64-encoded JSON") from exc
        info = json.loads(raw)
        if info.get("type") == "service_account":
            creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            return creds, info.get("project_id")
        creds, project = google.auth.load_credentials_from_dict(info, scopes=SCOPES)
        return creds, project
    creds, project = google.auth.default(scopes=SCOPES)
    return creds, project


class GoogleTokenSource:
    def __init__(self, credentials: Credentials) -> None:
        self.credentials = credentials
        self._token: str | None = None
        self._expiry: float = 0.0  # monotonic deadline
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self.last_error: str | None = None
        self.refreshes = 0

    def _remaining(self) -> float:
        return self._expiry - time.monotonic()

    async def token(self) -> str:
        if self._token and self._remaining() > 30:
            return self._token
        return await self._refresh()

    async def invalidate(self) -> None:
        """Force the next token() to refresh (e.g. after a 401)."""
        self._expiry = 0.0

    async def _refresh(self, force: bool = False) -> str:
        """Fetch a new token. The current one stays in use (and valid) until this succeeds."""
        async with self._lock:
            if not force and self._token and self._remaining() > REFRESH_MARGIN_S / 2:
                return self._token  # another caller refreshed while we waited
            try:
                await asyncio.to_thread(self.credentials.refresh, Request())
            except auth_exceptions.GoogleAuthError as exc:
                self.last_error = f"{exc.__class__.__name__}: {str(exc)[:200]}"
                log.error("google credential refresh failed", extra={"error": self.last_error})
                raise ProviderError("could not obtain Google Cloud credentials: " + self.last_error,
                                    code="credentials_error") from exc
            token = self.credentials.token
            if not token:
                raise ProviderError("Google credentials returned no access token", code="credentials_error")
            expiry = self.credentials.expiry
            if expiry is not None:
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=UTC)
                lifetime = (expiry - datetime.now(UTC)).total_seconds()
            else:
                lifetime = 3600.0
            self._token = token
            self._expiry = time.monotonic() + max(60.0, lifetime)
            self.last_error = None
            self.refreshes += 1
            return token

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="gcp-token-refresh")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while True:
            try:
                if self._remaining() < REFRESH_MARGIN_S:
                    await self._refresh(force=True)
                await asyncio.sleep(max(5.0, min(60.0, self._remaining() - REFRESH_MARGIN_S)))
            except asyncio.CancelledError:
                raise
            except Exception:  # never let the refresher die; the request path retries on demand
                await asyncio.sleep(10.0)

    def snapshot(self) -> dict[str, Any]:
        return {
            "valid": bool(self._token) and self._remaining() > 0,
            "expires_in_s": round(self._remaining()) if self._token else None,
            "refreshes": self.refreshes,
            "last_error": self.last_error,
            "type": type(self.credentials).__name__,
        }


class StaticTokenSource:
    """Fixed token (tests, or a token minted by an external sidecar)."""

    def __init__(self, token: str) -> None:
        self._token = token

    async def token(self) -> str:
        return self._token

    async def invalidate(self) -> None:
        return None

    def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return {"valid": True, "type": "static"}
