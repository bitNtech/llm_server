"""Structured logging: JSON lines in production, readable text in development.

Request/session/connection ids live in context variables and are attached to every record
emitted while handling that request, including records from library loggers.
"""
from __future__ import annotations

import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

import orjson

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)
connection_id_var: ContextVar[str | None] = ContextVar("connection_id", default=None)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "color_message"}


def _context() -> dict[str, Any]:
    ctx = {}
    for key, var in (("request_id", request_id_var), ("session_id", session_id_var),
                     ("connection_id", connection_id_var)):
        value = var.get()
        if value:
            ctx[key] = value
    return ctx


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(_context())
        payload.update(_extras(record))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return orjson.dumps(payload, default=str).decode()


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created)) + f".{int(record.msecs):03d}"
        fields = {**_context(), **_extras(record)}
        tail = " ".join(f"{k}={v}" for k, v in fields.items())
        line = f"{ts} {record.levelname:<7} {record.name}: {record.getMessage()}"
        if tail:
            line += f"  [{tail}]"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(level: str = "INFO", fmt: str = "text") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # uvicorn installs its own handlers; route everything through ours.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLevelName(level)))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
