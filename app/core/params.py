"""Sampling-parameter defaults, limits and validation shared by every API surface."""
from __future__ import annotations

from typing import Any

from app.config import Settings
from app.core.errors import BadRequest

# Parameters a session / response may set. Everything else in an OpenAI body is passed through
# untouched on the stateless /v1/chat/completions route; vLLM validates values.
SESSION_PARAM_KEYS = {
    "temperature", "top_p", "top_k", "min_p", "max_tokens", "stop", "seed",
    "presence_penalty", "frequency_penalty", "repetition_penalty", "min_tokens",
    "stop_token_ids", "logit_bias", "response_format", "structured_outputs",
    "chat_template_kwargs", "tool_choice", "parallel_tool_calls", "priority",
    "include_stop_str_in_output", "skip_special_tokens", "bad_words",
}

_DEFAULTS = (
    ("temperature", "default_temperature"),
    ("top_p", "default_top_p"),
    ("top_k", "default_top_k"),
    ("min_p", "default_min_p"),
    ("repetition_penalty", "default_repetition_penalty"),
    ("presence_penalty", "default_presence_penalty"),
    ("frequency_penalty", "default_frequency_penalty"),
)


def validate_session_params(params: dict[str, Any] | None) -> dict[str, Any]:
    if not params:
        return {}
    if not isinstance(params, dict):
        raise BadRequest("params must be an object")
    unknown = sorted(set(params) - SESSION_PARAM_KEYS)
    if unknown:
        raise BadRequest(f"unsupported params: {', '.join(unknown)}",
                         details={"allowed": sorted(SESSION_PARAM_KEYS)})
    if "max_tokens" in params and (not isinstance(params["max_tokens"], int) or params["max_tokens"] < 1):
        raise BadRequest("max_tokens must be a positive integer")
    if "chat_template_kwargs" in params and not isinstance(params["chat_template_kwargs"], dict):
        raise BadRequest("chat_template_kwargs must be an object")
    return dict(params)


def apply_defaults(settings: Settings, body: dict[str, Any]) -> dict[str, Any]:
    """Fill gateway defaults into an OpenAI chat body (in place) and enforce limits."""
    max_tokens = body.pop("max_completion_tokens", None) or body.get("max_tokens") or settings.default_max_tokens
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise BadRequest("max_tokens must be a positive integer")
    if max_tokens > settings.max_tokens_limit:
        raise BadRequest(f"max_tokens={max_tokens} exceeds MAX_TOKENS_LIMIT={settings.max_tokens_limit}")
    body["max_tokens"] = max_tokens
    for key, attr in _DEFAULTS:
        value = getattr(settings, attr)
        if value is not None and body.get(key) is None:
            body[key] = value
    if settings.stop_list and not body.get("stop"):
        body["stop"] = settings.stop_list
    defaults = settings.chat_template_kwargs
    if defaults or body.get("chat_template_kwargs"):
        body["chat_template_kwargs"] = {**defaults, **(body.get("chat_template_kwargs") or {})}
    if settings.engine_scheduling_policy != "priority":
        body.pop("priority", None)  # vLLM rejects priorities under FCFS scheduling
    if body.get("n", 1) != 1:
        raise BadRequest("n > 1 is not supported")
    return body
