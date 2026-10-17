"""Prometheus metrics for the gateway (vLLM's own engine metrics are proxied at /metrics/engine)."""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

_LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0, 10.0, 30.0)
_ITL_BUCKETS = (0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0)
_DURATION_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 120.0)

HTTP_REQUESTS = Counter(
    "gateway_http_requests_total", "HTTP requests", ["method", "route", "status"], registry=REGISTRY
)
HTTP_LATENCY = Histogram(
    "gateway_http_request_duration_seconds", "HTTP request duration (until headers for streams)",
    ["method", "route"], buckets=_LATENCY_BUCKETS, registry=REGISTRY,
)
TTFT = Histogram(
    "llm_time_to_first_token_seconds", "Gateway-observed time to first content/tool token",
    ["api"], buckets=_LATENCY_BUCKETS, registry=REGISTRY,
)
INTER_TOKEN = Histogram(
    "llm_inter_chunk_latency_seconds", "Time between streamed chunks", ["api"],
    buckets=_ITL_BUCKETS, registry=REGISTRY,
)
GENERATION_DURATION = Histogram(
    "llm_generation_duration_seconds", "End-to-end generation duration", ["api", "status"],
    buckets=_DURATION_BUCKETS, registry=REGISTRY,
)
QUEUE_WAIT = Histogram(
    "llm_admission_wait_seconds", "Time spent waiting for a generation slot",
    buckets=_LATENCY_BUCKETS, registry=REGISTRY,
)
TOKENS = Counter("llm_tokens_total", "Tokens processed", ["kind"], registry=REGISTRY)
GENERATIONS = Counter("llm_generations_total", "Finished generations", ["api", "status"], registry=REGISTRY)
ACTIVE_GENERATIONS = Gauge("llm_active_generations", "Generations streaming now", registry=REGISTRY)
QUEUED_GENERATIONS = Gauge("llm_queued_generations", "Generations waiting for a slot", registry=REGISTRY)
REJECTIONS = Counter("llm_admission_rejections_total", "Rejected generations", ["reason"], registry=REGISTRY)
CANCELLATIONS = Counter("llm_cancellations_total", "Cancelled generations", ["reason"], registry=REGISTRY)
ENGINE_ERRORS = Counter("llm_engine_errors_total", "Engine call failures", ["kind"], registry=REGISTRY)
TOOL_CALLS = Counter("llm_tool_calls_total", "Tool calls", ["tool", "side", "status"], registry=REGISTRY)
TOOL_LATENCY = Histogram(
    "llm_tool_duration_seconds", "Server-side tool execution time", ["tool"],
    buckets=_LATENCY_BUCKETS, registry=REGISTRY,
)
CONTEXT_TRIMS = Counter("llm_context_trims_total", "Context window trims", ["strategy"], registry=REGISTRY)
SESSIONS_ACTIVE = Gauge("gateway_sessions", "Live sessions", registry=REGISTRY)
WS_CONNECTIONS = Gauge("gateway_websocket_connections", "Open WebSocket connections", registry=REGISTRY)
ENGINE_UP = Gauge("gateway_engine_ready", "1 when the engine is healthy and serving the model", registry=REGISTRY)
