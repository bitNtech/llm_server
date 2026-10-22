"""Environment-driven configuration (single source of truth).

Every setting is read from the process environment first, then from `.env` in the project root
(override the file with LLM_SERVER_ENV_FILE). Invalid configuration fails at startup with a message
naming the variable.
"""
from __future__ import annotations

import os
import re
import secrets
import socket
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
_MODEL_ID = re.compile(r"^[a-z0-9-]+/[A-Za-z0-9._-]+$")


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.environ.get("LLM_SERVER_ENV_FILE") or ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    # ------------------------------------------------------------------ server
    app_env: Literal["development", "production"] = "development"
    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)
    instance_id: str = ""  # defaults to <hostname>-<random>; owner id for session/execution leases
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["auto", "json", "text"] = "auto"
    access_log: bool = False
    docs_enabled: bool | None = None  # None -> on in development only
    shutdown_grace_s: float = Field(20.0, ge=0)
    forwarded_allow_ips: str = "127.0.0.1"
    cors_allow_origins: str = ""
    max_request_body_bytes: int = Field(2_097_152, ge=1024)

    # ------------------------------------------------------------- google cloud
    gcp_project_id: str = ""  # required; used by Vertex AI and Firestore
    gcp_credentials_json: SecretStr = SecretStr("")  # optional inline service-account JSON (or base64 of it)
    # GOOGLE_APPLICATION_CREDENTIALS (a key file path) and workload identity are read by google-auth itself.

    # ---------------------------------------------------------------- vertex ai
    vertex_locations: str = "asia-south1,global"  # ordered failover list
    vertex_model: str = "google/gemini-3.5-flash"
    vertex_fallback_models: str = "google/gemini-3.1-flash-lite"  # tried when every location of a model fails
    vertex_allowed_models: str = ""  # extra models agents/clients may select (default model + fallbacks always allowed)
    vertex_reasoning_effort: Literal["", "minimal", "low", "medium", "high"] = "minimal"
    # Gemini 3.5 with 'minimal' thinking intermittently returns an empty candidate right after a tool
    # result (measured: up to 6/6 calls); 'low' is reliable there at the same latency. "" = no override.
    vertex_tool_followup_reasoning_effort: Literal["", "low", "medium", "high"] = "low"
    vertex_context_window: int = Field(1_048_576, ge=4096)  # measured for gemini-3.5-flash
    vertex_endpoint_template: str = ""  # override, e.g. a Private Service Connect host; {location} {project}
    vertex_connect_timeout_s: float = Field(3.0, gt=0)
    vertex_first_token_timeout_s: float = Field(12.0, gt=0)
    vertex_stream_idle_timeout_s: float = Field(20.0, gt=0)
    vertex_request_timeout_s: float = Field(60.0, gt=0)  # non-streaming calls
    vertex_max_attempts: int = Field(4, ge=1, le=10)  # across routes, before the first token only
    vertex_retry_backoff_s: float = Field(0.15, ge=0)
    vertex_http2: bool = True
    vertex_max_connections: int = Field(256, ge=1)
    vertex_keepalive_interval_s: float = Field(25.0, ge=0)  # 0 = off; keeps TLS connections warm when idle
    vertex_circuit_failures: int = Field(5, ge=1)
    vertex_circuit_reset_s: float = Field(20.0, gt=0)
    vertex_startup_probe: bool = True  # one 1-token request at startup: validates access and warms the connection

    # ---------------------------------------------------------------- firestore
    firestore_database: str = "llm_server"
    firestore_project_id: str = ""  # defaults to GCP_PROJECT_ID (or the project of FIRESTORE_CREDENTIALS_JSON)
    firestore_credentials_json: SecretStr = SecretStr("")  # optional separate service account; default: GCP credentials
    firestore_collection_prefix: str = ""  # namespacing (tests, shared databases)
    firestore_timeout_s: float = Field(10.0, gt=0)

    # ----------------------------------------------------------------- security
    admin_api_keys: SecretStr = SecretStr("")  # comma-separated platform admin keys
    api_key_cache_ttl_s: float = Field(30.0, ge=0)
    rate_limit_per_minute: int = Field(1200, ge=0)  # per API key; tenants may override; 0 = off
    secrets_encryption_key: SecretStr = SecretStr("")  # Fernet key(s), comma-separated, newest first
    realtime_token_secret: SecretStr = SecretStr("")  # HMAC secret for ephemeral realtime tokens
    realtime_token_max_ttl_s: int = Field(3600, ge=30)
    allow_private_network_targets: bool = False  # HTTP tools / MCP / webhooks may reach private IPs
    allowed_target_hosts: str = ""  # hosts exempt from the private-network check
    ws_query_api_key: bool = False  # accept ?api_key= on the WebSocket (leaks into proxy logs; prefer tokens)

    # -------------------------------------------------------- inference defaults
    default_instructions: str = "You are a helpful, concise voice assistant. Reply in one or two short sentences."
    default_max_tokens: int = Field(512, ge=1)
    max_tokens_limit: int = Field(8192, ge=1)
    default_temperature: float | None = Field(None, ge=0, le=2)
    max_input_chars: int = Field(64_000, ge=1)

    # -------------------------------------------------------------- concurrency
    max_concurrent_generations: int = Field(256, ge=1)
    max_queued_generations: int = Field(1024, ge=0)
    queue_timeout_s: float = Field(2.0, gt=0)
    generation_timeout_s: float = Field(120.0, gt=0)

    # ---------------------------------------------------------- sessions/context
    session_cache_ttl_s: float = Field(180.0, gt=0)  # idle sessions leave memory (and free their lease)
    max_cached_sessions: int = Field(20_000, ge=1)
    session_lease_s: float = Field(60.0, ge=10)
    max_messages_per_session: int = Field(5000, ge=4)
    context_strategy: Literal["truncate", "summarize"] = "truncate"
    context_max_tokens: int = Field(65_536, ge=1024)  # prompt budget; caps cost and prefill latency
    context_reserve_tokens: int = Field(256, ge=0)
    context_trim_target_ratio: float = Field(0.7, gt=0.1, le=1.0)
    context_chars_per_token: float = Field(4.0, gt=0.2)
    summary_max_tokens: int = Field(400, ge=16)

    # -------------------------------------------------------------------- tools
    tool_timeout_s: float = Field(10.0, gt=0)
    max_tool_rounds: int = Field(8, ge=1, le=50)
    max_handoffs_per_turn: int = Field(3, ge=0, le=10)
    tool_output_max_chars: int = Field(20_000, ge=256)
    http_tool_max_response_bytes: int = Field(1_048_576, ge=1024)
    mcp_connect_timeout_s: float = Field(10.0, gt=0)
    mcp_call_timeout_s: float = Field(30.0, gt=0)
    mcp_tools_cache_ttl_s: float = Field(300.0, ge=0)

    # --------------------------------------------------------------- executions
    execution_workers: int = Field(8, ge=0)  # 0 = this instance never runs executions
    execution_lease_s: float = Field(60.0, ge=10)
    execution_max_attempts: int = Field(3, ge=1)
    execution_timeout_s: float = Field(900.0, gt=0)
    workflow_max_steps: int = Field(200, ge=1)
    webhook_timeout_s: float = Field(10.0, gt=0)
    webhook_max_attempts: int = Field(6, ge=1)

    # ---------------------------------------------------------------- websocket
    ws_max_message_bytes: int = Field(1_048_576, ge=1024)
    ws_send_queue_size: int = Field(2048, ge=16)
    ws_send_timeout_s: float = Field(10.0, gt=0)
    ws_idle_timeout_s: float = Field(900.0, gt=0)
    ws_ping_interval_s: float = Field(20.0, gt=0)
    barge_in: Literal["cancel", "reject"] = "cancel"
    segment_min_chars: int = Field(8, ge=1)
    segment_max_chars: int = Field(220, ge=16)

    # ------------------------------------------------------------ observability
    metrics_enabled: bool = True
    metrics_require_auth: bool = False
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "aica-llm-server"
    otel_sample_ratio: float = Field(1.0, ge=0, le=1)
    usage_flush_interval_s: float = Field(10.0, gt=0)

    # ================================================================ validators
    @field_validator("default_temperature", mode="before")
    @classmethod
    def _blank_is_none(cls, v: object) -> object:
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("vertex_model")
    @classmethod
    def _model_id(cls, v: str) -> str:
        v = v.strip()
        if not _MODEL_ID.match(v):
            raise ValueError("must look like 'google/<model>'")
        return v

    @field_validator("vertex_fallback_models", "vertex_allowed_models")
    @classmethod
    def _model_ids(cls, v: str) -> str:
        bad = [m for m in _csv(v) if not _MODEL_ID.match(m)]
        if bad:
            raise ValueError(f"model ids must look like 'google/<model>': {bad}")
        return v

    @field_validator("vertex_locations")
    @classmethod
    def _locations(cls, v: str) -> str:
        locs = _csv(v)
        if not locs or any(not re.match(r"^[a-z0-9-]+$", loc) for loc in locs):
            raise ValueError("comma-separated Vertex locations, e.g. 'asia-south1,global'")
        return v

    @field_validator("firestore_collection_prefix")
    @classmethod
    def _prefix(cls, v: str) -> str:
        if v and not re.match(r"^[A-Za-z0-9_]{1,40}$", v):
            raise ValueError("letters, digits and underscores only")
        return v

    @model_validator(mode="after")
    def _cross_checks(self) -> Settings:
        if not self.instance_id:
            self.instance_id = f"{socket.gethostname()[:40]}-{secrets.token_hex(3)}"
        if self.default_max_tokens > self.max_tokens_limit:
            raise ValueError("DEFAULT_MAX_TOKENS must be <= MAX_TOKENS_LIMIT")
        if self.context_max_tokens > self.vertex_context_window:
            raise ValueError("CONTEXT_MAX_TOKENS cannot exceed VERTEX_CONTEXT_WINDOW")
        if self.is_production:
            missing = [name for name, value in (
                ("GCP_PROJECT_ID", self.gcp_project_id),
                ("ADMIN_API_KEYS", self.admin_api_keys.get_secret_value()),
                ("SECRETS_ENCRYPTION_KEY", self.secrets_encryption_key.get_secret_value()),
                ("REALTIME_TOKEN_SECRET", self.realtime_token_secret.get_secret_value()),
            ) if not value]
            if missing:
                raise ValueError(f"APP_ENV=production requires {', '.join(missing)}")
            weak = [k for k in self.admin_key_list if len(k) < 24 or "change" in k.lower()]
            if weak:
                raise ValueError("ADMIN_API_KEYS contains placeholder or short (<24 chars) keys")
            if len(self.realtime_token_secret.get_secret_value()) < 32:
                raise ValueError("REALTIME_TOKEN_SECRET must be at least 32 characters")
            if "*" in self.cors_origin_list:
                raise ValueError("CORS_ALLOW_ORIGINS='*' is not allowed in production")
        return self

    # ================================================================== derived
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def admin_key_list(self) -> list[str]:
        return _csv(self.admin_api_keys.get_secret_value())

    @property
    def cors_origin_list(self) -> list[str]:
        return _csv(self.cors_allow_origins)

    @property
    def docs_are_enabled(self) -> bool:
        return (not self.is_production) if self.docs_enabled is None else self.docs_enabled

    @property
    def resolved_log_format(self) -> str:
        if self.log_format != "auto":
            return self.log_format
        return "json" if self.is_production else "text"

    @property
    def location_list(self) -> list[str]:
        return _csv(self.vertex_locations)

    @property
    def fallback_model_list(self) -> list[str]:
        return [m for m in _csv(self.vertex_fallback_models) if m != self.vertex_model]

    @property
    def allowed_model_list(self) -> list[str]:
        models = [self.vertex_model, *self.fallback_model_list, *_csv(self.vertex_allowed_models)]
        return list(dict.fromkeys(models))

    @property
    def allowed_target_host_list(self) -> list[str]:
        return [h.lower() for h in _csv(self.allowed_target_hosts)]

    @property
    def firestore_project(self) -> str:
        return self.firestore_project_id or self.gcp_project_id

    @property
    def encryption_keys(self) -> list[str]:
        return _csv(self.secrets_encryption_key.get_secret_value())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
