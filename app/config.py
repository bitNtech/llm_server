"""Environment-driven configuration (single source of truth for gateway *and* engine launch).

Every setting is read from the process environment first, then from `.env` in the project
root. `run.sh` uses the same class (via `python -m app.cli`) to validate the configuration
and to build the `vllm serve` command line, so the gateway and the engine never disagree.

Naming: gateway settings are unprefixed; engine (vLLM) settings use the `ENGINE_` prefix.
We deliberately avoid the `VLLM_` prefix because vLLM reads (and warns about unknown)
`VLLM_*` variables itself.
"""
from __future__ import annotations

import json
import os
import shlex
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


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
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["auto", "json", "text"] = "auto"
    access_log: bool = False
    docs_enabled: bool | None = None  # None -> enabled in development only
    shutdown_grace_s: float = Field(20.0, ge=0)
    forwarded_allow_ips: str = "127.0.0.1"  # proxies trusted for X-Forwarded-For ("*" behind an ALB)

    # ---------------------------------------------------------------- security
    api_keys: SecretStr = SecretStr("")  # comma-separated
    allow_unauthenticated: bool = False  # production refuses to start without keys unless true
    metrics_require_auth: bool = False
    cors_allow_origins: str = ""  # comma-separated, "*" allowed in development only
    rate_limit_per_minute: int = Field(0, ge=0)  # per API key (or client IP); 0 = off
    max_request_body_bytes: int = Field(1_048_576, ge=1024)

    # ------------------------------------------------- engine connection (gateway side)
    engine_mode: Literal["managed", "external", "mock"] = "managed"
    engine_host: str = "127.0.0.1"
    engine_port: int = Field(8001, ge=1, le=65535)
    engine_base_url: str = ""  # overrides host/port, e.g. for ENGINE_MODE=external
    engine_api_key: SecretStr = SecretStr("")  # run.sh generates one per start if empty
    engine_connect_timeout_s: float = Field(5.0, gt=0)
    engine_first_token_timeout_s: float = Field(15.0, gt=0)
    engine_stream_idle_timeout_s: float = Field(10.0, gt=0)
    engine_max_connections: int = Field(512, ge=1)
    engine_health_interval_s: float = Field(5.0, gt=0)
    engine_startup_timeout_s: float = Field(1800.0, gt=0)
    engine_warmup: bool = True  # JIT-compile/warm short + long prompt shapes before reporting ready
    engine_warmup_timeout_s: float = Field(900.0, gt=0)
    generation_timeout_s: float = Field(120.0, gt=0)

    # ------------------------------------------------------- inference defaults
    # Unset (empty) sampling defaults are not sent, so the model's generation_config.json applies.
    default_system_prompt: str = ""
    default_system_prompt_file: str = ""
    default_max_tokens: int = Field(200, ge=1)
    max_tokens_limit: int = Field(4096, ge=1)
    default_temperature: float | None = Field(None, ge=0, le=2)
    default_top_p: float | None = Field(None, gt=0, le=1)
    default_top_k: int | None = None
    default_min_p: float | None = Field(None, ge=0, le=1)
    default_repetition_penalty: float | None = Field(None, gt=0)
    default_presence_penalty: float | None = Field(None, ge=-2, le=2)
    default_frequency_penalty: float | None = Field(None, ge=-2, le=2)
    default_stop: str = ""  # comma-separated
    default_chat_template_kwargs: str = '{"enable_thinking": false}'  # JSON object
    max_input_chars: int = Field(32_000, ge=1)

    # -------------------------------------------------------------- concurrency
    max_concurrent_generations: int = Field(32, ge=1)
    max_queued_generations: int = Field(256, ge=0)
    queue_timeout_s: float = Field(3.0, gt=0)

    # --------------------------------------------------------- sessions/context
    session_ttl_s: float = Field(1800.0, gt=0)
    max_sessions: int = Field(10_000, ge=1)
    max_messages_per_session: int = Field(1000, ge=4)
    context_strategy: Literal["truncate", "summarize"] = "truncate"
    context_max_tokens: int = Field(0, ge=0)  # 0 = engine max_model_len
    context_reserve_tokens: int = Field(64, ge=0)
    context_trim_target_ratio: float = Field(0.7, gt=0.1, le=1.0)
    context_chars_per_token: float = Field(3.0, gt=0.2)  # initial estimate, calibrated per session
    summary_max_tokens: int = Field(256, ge=16)
    summary_prompt: str = (
        "Summarize the conversation so far in a few short sentences. Keep names, numbers, "
        "dates, decisions and open questions. Write in the language the user speaks."
    )

    # -------------------------------------------------------------------- tools
    tools_modules: str = "app.tools.builtin"  # comma-separated import paths of server-side tools
    tool_timeout_s: float = Field(10.0, gt=0)
    max_tool_rounds: int = Field(4, ge=1, le=32)

    # ---------------------------------------------------------------- websocket
    ws_max_message_bytes: int = Field(262_144, ge=1024)
    ws_send_queue_size: int = Field(1024, ge=16)
    ws_send_timeout_s: float = Field(10.0, gt=0)
    ws_idle_timeout_s: float = Field(600.0, gt=0)
    ws_ping_interval_s: float = Field(20.0, gt=0)
    ws_barge_in: Literal["cancel", "reject"] = "cancel"
    segment_min_chars: int = Field(8, ge=1)
    segment_max_chars: int = Field(220, ge=16)

    # ------------------------------------------------------------ observability
    metrics_enabled: bool = True

    # ------------------------------------------------- engine launch (run.sh -> vllm serve)
    engine_device: Literal["auto", "cuda", "cpu"] = "auto"
    # Gemma 4 26B-A4B (MoE, ~4B active) as an FP8 W8A8 checkpoint: ~29 GB, 99%+ accuracy recovery,
    # and roughly half the bytes per decoded token of BF16 (decode is memory-bandwidth bound).
    engine_model: str = "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic"
    engine_served_model_name: str = "gemma-4-26b-a4b-it"
    engine_revision: str = "ed35d7abe5d940da41b4ff06eb482feb0be8cb44"  # pinned checkpoint commit
    engine_tokenizer: str = ""
    engine_trust_remote_code: bool = False
    engine_dtype: str = "auto"
    engine_quantization: str = ""
    engine_max_model_len: int = Field(8192, ge=256)
    engine_gpu_memory_utilization: float = Field(0.90, gt=0.05, le=0.99)
    engine_kv_cache_memory_gib: float = Field(0.0, ge=0)  # 0 = auto (GPU: utilization; CPU: sized from RAM)
    engine_kv_cache_dtype: str = "auto"
    engine_tensor_parallel_size: str = "1"  # integer or "auto" (= all visible GPUs)
    engine_max_num_seqs: int = Field(32, ge=0)  # small batches = low inter-token latency; 0 = vLLM default
    engine_max_num_batched_tokens: int = Field(0, ge=0)
    engine_enable_prefix_caching: bool = True
    engine_enforce_eager: bool = False
    engine_language_model_only: bool = True  # skip vision tower of multimodal checkpoints
    engine_enable_auto_tool_choice: bool = True
    engine_tool_call_parser: str = "gemma4"
    engine_reasoning_parser: str = "gemma4"
    engine_chat_template: str = ""
    engine_generation_config: str = "auto"
    engine_scheduling_policy: Literal["fcfs", "priority"] = "fcfs"
    # Gemma 4 MTP drafter (0.4B, shares the target's KV cache): fewer target forward passes per token.
    engine_speculative_config: str = (
        '{"model": "google/gemma-4-26B-A4B-it-assistant", '
        '"revision": "6e5aaaf4c42b98394530b8fda2e95cadd65c151c", "num_speculative_tokens": 4}'
    )
    engine_attention_backend: str = "auto"  # auto | FLASH_ATTN | TRITON_ATTN | FLASHINFER | ...
    engine_download_dir: str = ""
    engine_extra_args: str = ""  # appended verbatim (shell-split) to `vllm serve`
    engine_log_requests: bool = False
    engine_cpu_reserved_cores: int = Field(1, ge=0)

    # ---------------------------------------------------------- installation
    engine_vllm_version: str = "0.31.0"  # pinned for reproducible deployments
    engine_install_spec: str = ""  # full pip spec/URL override
    engine_cuda_variant: Literal["auto", "cu130", "cu129"] = "auto"
    engine_model_prefetch: bool = True

    # ================================================================= validators
    @field_validator(
        "default_temperature", "default_top_p", "default_top_k", "default_min_p",
        "default_repetition_penalty", "default_presence_penalty", "default_frequency_penalty",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, v: Any) -> Any:
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("default_chat_template_kwargs", "engine_speculative_config")
    @classmethod
    def _json_object(cls, v: str) -> str:
        if v.strip():
            parsed = json.loads(v)
            if not isinstance(parsed, dict):
                raise ValueError("must be a JSON object")
        return v

    @field_validator("engine_tensor_parallel_size")
    @classmethod
    def _tp(cls, v: str) -> str:
        v = v.strip().lower()
        if v != "auto" and (not v.isdigit() or int(v) < 1):
            raise ValueError("must be a positive integer or 'auto'")
        return v

    @field_validator("engine_extra_args")
    @classmethod
    def _extra_args(cls, v: str) -> str:
        shlex.split(v)  # raises on unbalanced quotes
        return v

    @model_validator(mode="after")
    def _cross_checks(self) -> Settings:
        if self.app_env == "production":
            if not self.api_key_list and not self.allow_unauthenticated:
                raise ValueError(
                    "APP_ENV=production requires API_KEYS (or ALLOW_UNAUTHENTICATED=true behind "
                    "a trusted auth proxy)"
                )
            weak = [k for k in self.api_key_list if len(k) < 16 or "change_me" in k.lower()]
            if weak:
                raise ValueError("API_KEYS contains placeholder or short (<16 chars) keys; generate real ones")
            if "*" in self.cors_origin_list:
                raise ValueError("CORS_ALLOW_ORIGINS='*' is not allowed in production")
        if self.default_max_tokens > self.max_tokens_limit:
            raise ValueError("DEFAULT_MAX_TOKENS must be <= MAX_TOKENS_LIMIT")
        if self.engine_mode == "external" and not self.engine_base_url:
            raise ValueError("ENGINE_MODE=external requires ENGINE_BASE_URL")
        return self

    # ================================================================= derived
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def api_key_list(self) -> list[str]:
        return _csv(self.api_keys.get_secret_value())

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key_list)

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
    def resolved_engine_base_url(self) -> str:
        if self.engine_base_url:
            return self.engine_base_url.rstrip("/")
        return f"http://{self.engine_host}:{self.engine_port}"

    @property
    def stop_list(self) -> list[str]:
        return _csv(self.default_stop)

    @property
    def chat_template_kwargs(self) -> dict[str, Any]:
        raw = self.default_chat_template_kwargs.strip()
        return json.loads(raw) if raw else {}

    @property
    def tools_module_list(self) -> list[str]:
        return _csv(self.tools_modules)

    @property
    def system_prompt(self) -> str:
        if self.default_system_prompt_file:
            path = Path(self.default_system_prompt_file)
            if not path.is_absolute():
                path = ROOT_DIR / path
            return path.read_text(encoding="utf-8").strip()
        return self.default_system_prompt

    @property
    def speculative(self) -> dict[str, Any]:
        raw = self.engine_speculative_config.strip()
        return json.loads(raw) if raw else {}

    @property
    def served_model_name(self) -> str:
        return self.engine_served_model_name or self.engine_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
