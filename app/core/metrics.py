"""Prometheus metrics. Labels are bounded: no tenant, session or user ids (use logs/traces for those)."""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

_LATENCY = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.5, 2.5, 5.0, 10.0, 30.0)
_ITL = (0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0)
_DURATION = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 120.0, 300.0)

HTTP_REQUESTS = Counter("gateway_http_requests_total", "HTTP requests", ["method", "route", "status"],
                        registry=REGISTRY)
HTTP_LATENCY = Histogram("gateway_http_request_duration_seconds", "HTTP time to response headers",
                         ["method", "route"], buckets=_LATENCY, registry=REGISTRY)

TTFT = Histogram("llm_time_to_first_token_seconds", "Client-observable time to first token", ["api"],
                 buckets=_LATENCY, registry=REGISTRY)
UPSTREAM_TTFT = Histogram("llm_upstream_ttft_seconds", "Vertex request start to first token", ["model", "location"],
                          buckets=_LATENCY, registry=REGISTRY)
INTER_TOKEN = Histogram("llm_inter_chunk_latency_seconds", "Time between streamed chunks", ["api"],
                        buckets=_ITL, registry=REGISTRY)
GENERATION_DURATION = Histogram("llm_generation_duration_seconds", "End-to-end generation duration",
                                ["api", "status"], buckets=_DURATION, registry=REGISTRY)
GENERATIONS = Counter("llm_generations_total", "Finished generations", ["api", "status"], registry=REGISTRY)
TOKENS = Counter("llm_tokens_total", "Tokens processed", ["model", "kind"], registry=REGISTRY)
ACTIVE_GENERATIONS = Gauge("llm_active_generations", "Generations streaming now", registry=REGISTRY)
QUEUED_GENERATIONS = Gauge("llm_queued_generations", "Generations waiting for a slot", registry=REGISTRY)
QUEUE_WAIT = Histogram("llm_admission_wait_seconds", "Time waiting for a generation slot",
                       buckets=_LATENCY, registry=REGISTRY)
REJECTIONS = Counter("llm_admission_rejections_total", "Rejected generations", ["reason"], registry=REGISTRY)
CANCELLATIONS = Counter("llm_cancellations_total", "Cancelled generations", ["reason"], registry=REGISTRY)

PROVIDER_REQUESTS = Counter("vertex_requests_total", "Vertex attempts", ["model", "location", "outcome"],
                            registry=REGISTRY)
PROVIDER_FAILOVERS = Counter("vertex_failovers_total", "Requests served by a non-primary route", ["reason"],
                             registry=REGISTRY)
CIRCUIT_OPEN = Gauge("vertex_circuit_open", "1 while a location's circuit breaker is open", ["location"],
                     registry=REGISTRY)
PROVIDER_READY = Gauge("vertex_ready", "1 when Vertex credentials and at least one route are usable",
                       registry=REGISTRY)

TOOL_CALLS = Counter("llm_tool_calls_total", "Tool calls", ["kind", "status"], registry=REGISTRY)
TOOL_LATENCY = Histogram("llm_tool_duration_seconds", "Server-side tool execution time", ["kind"],
                         buckets=_LATENCY, registry=REGISTRY)
HANDOFFS = Counter("llm_handoffs_total", "Agent handoffs", ["source"], registry=REGISTRY)
CONTEXT_TRIMS = Counter("llm_context_trims_total", "Context window trims", ["strategy"], registry=REGISTRY)

MCP_CONNECTIONS = Gauge("mcp_connections", "Connected MCP servers", registry=REGISTRY)
MCP_ERRORS = Counter("mcp_errors_total", "MCP connection/call failures", ["kind"], registry=REGISTRY)

STORE_OPS = Histogram("firestore_operation_seconds", "Firestore operation latency", ["op"],
                      buckets=_LATENCY, registry=REGISTRY)
STORE_ERRORS = Counter("firestore_errors_total", "Firestore failures", ["op"], registry=REGISTRY)
PERSIST_BACKLOG = Gauge("firestore_write_backlog", "Queued write-behind operations", registry=REGISTRY)

EXECUTIONS = Counter("executions_total", "Finished async executions", ["type", "status"], registry=REGISTRY)
EXECUTIONS_RUNNING = Gauge("executions_running", "Executions running on this instance", registry=REGISTRY)
WEBHOOKS = Counter("webhook_deliveries_total", "Webhook deliveries", ["outcome"], registry=REGISTRY)

SESSIONS_CACHED = Gauge("gateway_sessions_cached", "Sessions held in memory", registry=REGISTRY)
WS_CONNECTIONS = Gauge("gateway_websocket_connections", "Open WebSocket connections", registry=REGISTRY)
