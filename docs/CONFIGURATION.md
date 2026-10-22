# Configuration

All settings come from environment variables, falling back to `.env` (set `LLM_SERVER_ENV_FILE` to use another file). They are validated at startup, and `./run.sh check` prints the effective setup. [`.env.example`](../.env.example) lists every variable with comments. This page covers the ones that matter most.

## Required

| Variable | Notes |
|---|---|
| `GCP_PROJECT_ID` | Project for Vertex AI and, by default, Firestore. It can be discovered from service-account credentials. |
| credentials | `GOOGLE_APPLICATION_CREDENTIALS` (key file), `GCP_CREDENTIALS_JSON` (inline JSON or base64), workload identity on GCP, or `gcloud auth application-default login` locally. Roles: `aiplatform.user`, `datastore.user`. |
| `ADMIN_API_KEYS` | Platform admin keys (production: at least 24 characters each, no placeholders). |
| `SECRETS_ENCRYPTION_KEY` | Production. Fernet key(s), newest first. Without it, development uses an ephemeral key. |
| `REALTIME_TOKEN_SECRET` | Production. At least 32 characters. |

## Model and latency

| Variable | Default | Notes |
|---|---|---|
| `VERTEX_MODEL` | `google/gemini-3.5-flash` | Fastest model with a 1M-token window available to the project (measured, see ARCHITECTURE.md). |
| `VERTEX_FALLBACK_MODELS` | `google/gemini-3.1-flash-lite` | Tried when every location of the main model fails. |
| `VERTEX_ALLOWED_MODELS` | — | Extra models agents and clients may select. |
| `VERTEX_LOCATIONS` | `asia-south1,global` | Failover order. Mumbai is about 2x faster than global from India. |
| `VERTEX_REASONING_EFFORT` | `minimal` | Fastest first token on user turns. `low`/`medium`/`high` think more. |
| `VERTEX_TOOL_FOLLOWUP_REASONING_EFFORT` | `low` | Used after tool results, because `minimal` intermittently returns empty answers there. |
| `VERTEX_FIRST_TOKEN_TIMEOUT_S` | 12 | Total budget for retries and failover before the first token. |
| `VERTEX_MAX_ATTEMPTS` | 4 | Route attempts per request. |
| `VERTEX_CIRCUIT_FAILURES` / `_RESET_S` | 5 / 20 | Per-location circuit breaker. |
| `VERTEX_KEEPALIVE_INTERVAL_S` | 25 | Keeps TLS/HTTP2 connections warm while idle. |
| `CONTEXT_MAX_TOKENS` | 65536 | Prompt budget per turn. Older turns leave the window (history stays in Firestore). Lower values mean faster prefill and lower cost. |
| `DEFAULT_MAX_TOKENS` / `MAX_TOKENS_LIMIT` | 512 / 8192 | Reply length default and ceiling. |

## Persistence

| Variable | Default | Notes |
|---|---|---|
| `FIRESTORE_DATABASE` | `llm_server` | Named native-mode database. |
| `FIRESTORE_PROJECT_ID` | `GCP_PROJECT_ID` | When Firestore lives in another project. Discovered from `FIRESTORE_CREDENTIALS_JSON` when empty. |
| `FIRESTORE_CREDENTIALS_JSON` | — | Service-account JSON (or its base64) for Firestore when it belongs to another project or account. Role: `datastore.user`. Default: the Vertex credentials. |
| `FIRESTORE_COLLECTION_PREFIX` | — | Share one database between environments (`staging_`). |
| `FIRESTORE_EMULATOR_HOST` | — | Local development and tests. |
| `SESSION_LEASE_S` | 60 | How long another instance waits before it may take over a crashed owner's session. |
| `SESSION_CACHE_TTL_S` | 180 | Idle sessions are flushed, released and dropped from memory after this. |

## Limits and safety

| Variable | Default | Notes |
|---|---|---|
| `RATE_LIMIT_PER_MINUTE` | 1200 | Per API key, per instance. Tenants override it with `settings.rate_limit_per_minute`. |
| `MAX_CONCURRENT_GENERATIONS` / `MAX_QUEUED_GENERATIONS` / `QUEUE_TIMEOUT_S` | 256 / 1024 / 2 | Admission control per instance. When it is exceeded: fast `429 overloaded`. |
| `GENERATION_TIMEOUT_S` | 120 | Whole response, including tool rounds. |
| `TOOL_TIMEOUT_S`, `MAX_TOOL_ROUNDS`, `MAX_HANDOFFS_PER_TURN` | 10, 8, 3 | Tool loop bounds. |
| `ALLOW_PRIVATE_NETWORK_TARGETS` / `ALLOWED_TARGET_HOSTS` | false / — | Outbound URL policy for HTTP tools, MCP and webhooks. |
| `BARGE_IN` | `cancel` | `reject`: a second `response.create` while one runs gets a 409. |
| `WS_IDLE_TIMEOUT_S`, `WS_MAX_MESSAGE_BYTES` | 900, 1 MiB | WebSocket limits. |

## Executions

`EXECUTION_WORKERS` (8 per instance; 0 means the instance never runs executions), `EXECUTION_LEASE_S`, `EXECUTION_MAX_ATTEMPTS`, `EXECUTION_TIMEOUT_S`, `WORKFLOW_MAX_STEPS`, `WEBHOOK_MAX_ATTEMPTS`.

## Observability

`LOG_FORMAT` (`auto`: JSON in production), `LOG_LEVEL`, `ACCESS_LOG`, `METRICS_ENABLED`, `METRICS_REQUIRE_AUTH`, `OTEL_EXPORTER_OTLP_ENDPOINT` (enables tracing), `OTEL_SAMPLE_RATIO`, `USAGE_FLUSH_INTERVAL_S`.
