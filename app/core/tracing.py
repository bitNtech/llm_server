"""OpenTelemetry tracing.

Spans are always created through the OpenTelemetry API; they cost almost nothing until an SDK is
installed. `setup_tracing` installs the SDK with an OTLP/HTTP exporter only when an endpoint is
configured (OTEL_EXPORTER_OTLP_ENDPOINT), so tracing is opt-in per deployment.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import propagate, trace
from opentelemetry.trace import Span, Status, StatusCode

from app.core.logging import get_logger

log = get_logger("app.tracing")
tracer = trace.get_tracer("aica.llm_server")
_provider: Any = None


def setup_tracing(service_name: str, endpoint: str, version: str, sample_ratio: float = 1.0) -> None:
    global _provider
    if not endpoint or _provider is not None:
        return
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    resource = Resource.create({"service.name": service_name, "service.version": version})
    _provider = TracerProvider(resource=resource, sampler=ParentBased(TraceIdRatioBased(sample_ratio)))
    exporter_endpoint = endpoint.rstrip("/")
    if not exporter_endpoint.endswith("/v1/traces"):
        exporter_endpoint += "/v1/traces"
    _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=exporter_endpoint)))
    trace.set_tracer_provider(_provider)
    log.info("tracing enabled", extra={"otlp_endpoint": exporter_endpoint})


def shutdown_tracing() -> None:
    if _provider is not None:
        _provider.shutdown()


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    with tracer.start_as_current_span(name, record_exception=True, set_status_on_exception=False) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        try:
            yield current
        except BaseException as exc:
            # cancellation (barge-in) is a normal outcome, not an error
            if not isinstance(exc, (GeneratorExit,)) and exc.__class__.__name__ != "CancelledError":
                current.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
            raise


def inject_headers(headers: dict[str, str]) -> dict[str, str]:
    """Add W3C traceparent/tracestate for outbound calls (HTTP tools, MCP, webhooks)."""
    propagate.inject(headers)
    return headers


def extract_context(headers: Mapping[str, str]) -> otel_context.Context:
    return propagate.extract(headers)
