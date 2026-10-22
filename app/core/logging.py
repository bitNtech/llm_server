"""Structured logging: JSON lines in production (Cloud Logging compatible), readable text in development.

Request, session, tenant and connection ids live in context variables and are attached to every
record emitted while handling that request, including records from library loggers. When tracing
is enabled the active trace/span ids are attached as well, in the fields Cloud Logging correlates.
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
tenant_id_var: ContextVar[str | None] = ContextVar("tenant_id", default=None)
connection_id_var: ContextVar[str | None] = ContextVar("connection_id", default=None)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "color_message"}
_GCP_PROJECT: str | None = None


def _trace_ids() -> tuple[str | None, str | None]:
    try:
        from opentelemetry import trace
    except ImportError:  # pragma: no cover
        return None, None
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None, None
    return f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}"


def _context() -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    for key, var in (("request_id", request_id_var), ("session_id", session_id_var),
                     ("tenant_id", tenant_id_var), ("connection_id", connection_id_var)):
        value = var.get()
        if value:
            ctx[key] = value
    return ctx


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_context())
        trace_id, span_id = _trace_ids()
        if trace_id:
            payload["trace_id"] = trace_id
            payload["span_id"] = span_id
            if _GCP_PROJECT:
                payload["logging.googleapis.com/trace"] = f"projects/{_GCP_PROJECT}/traces/{trace_id}"
                payload["logging.googleapis.com/spanId"] = span_id
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


def setup_logging(level: str = "INFO", fmt: str = "text", gcp_project: str | None = None) -> None:
    global _GCP_PROJECT
    _GCP_PROJECT = gcp_project
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
    quiet = max(logging.WARNING, logging.getLevelName(level))
    for noisy in ("httpx", "httpcore", "httpx2", "mcp", "google", "grpc", "urllib3"):
        logging.getLogger(noisy).setLevel(quiet)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
