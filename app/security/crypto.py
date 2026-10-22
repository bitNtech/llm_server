"""Encryption at rest for tenant secrets (tool auth headers, signing secrets, MCP credentials,
webhook secrets). Fernet (AES-128-CBC + HMAC-SHA256) with key rotation: SECRETS_ENCRYPTION_KEY holds
one or more keys, newest first; data encrypted with an older key still decrypts."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Any

import orjson
from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from app.core.errors import GatewayError
from app.core.logging import get_logger

log = get_logger("app.crypto")


class SecretBox:
    def __init__(self, keys: list[str], production: bool) -> None:
        if not keys:
            if production:
                raise ValueError("SECRETS_ENCRYPTION_KEY is required in production")
            keys = [Fernet.generate_key().decode()]
            log.warning("SECRETS_ENCRYPTION_KEY not set: using an ephemeral key; stored secrets will not "
                        "survive a restart (development only)")
        try:
            self._fernet = MultiFernet([Fernet(k.encode()) for k in keys])
        except ValueError as exc:
            raise ValueError("SECRETS_ENCRYPTION_KEY must be url-safe base64 Fernet key(s) "
                             "(python -c 'from cryptography.fernet import Fernet; "
                             "print(Fernet.generate_key().decode())')") from exc

    def encrypt(self, value: Any) -> str:
        return self._fernet.encrypt(orjson.dumps(value)).decode()

    def decrypt(self, token: str | None) -> Any:
        if not token:
            return None
        try:
            return orjson.loads(self._fernet.decrypt(token.encode()))
        except InvalidToken as exc:
            raise GatewayError("stored secret cannot be decrypted (SECRETS_ENCRYPTION_KEY changed?)",
                               code="secret_decrypt_failed") from exc


def new_secret(prefix: str = "whsec") -> str:
    return f"{prefix}_{secrets.token_urlsafe(32)}"


def sign_payload(secret: str, body: bytes, timestamp: int | None = None) -> str:
    """`t=<unix>,v1=<hex hmac-sha256(secret, "<t>.<body>")>`, verified by receivers to authenticate
    webhook and HTTP-tool requests and to reject replays older than a few minutes."""
    ts = int(timestamp if timestamp is not None else time.time())
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


def verify_signature(secret: str, body: bytes, header: str, tolerance_s: int = 300) -> bool:
    try:
        parts = dict(p.split("=", 1) for p in header.split(","))
        ts = int(parts["t"])
    except (ValueError, KeyError):
        return False
    if abs(time.time() - ts) > tolerance_s:
        return False
    expected = sign_payload(secret, body, ts).split("v1=", 1)[1]
    return hmac.compare_digest(expected, parts.get("v1", ""))
