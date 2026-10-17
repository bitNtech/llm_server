"""Request bodies for the session API (also documents them in OpenAPI)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionSpec(_Body):
    system_prompt: str | None = Field(None, description="Instructions; defaults to DEFAULT_SYSTEM_PROMPT")
    tools: list[dict[str, Any]] | None = Field(None, description="Client-side OpenAI function tools")
    server_tools: list[str] | str | None = Field(None, description="Server tool names to enable, or 'all'")
    params: dict[str, Any] | None = Field(None, description="Default sampling params for every response")
    metadata: dict[str, Any] | None = None
    segmentation: bool | None = Field(None, description="Emit sentence segments for TTS")
    include_reasoning: bool | None = Field(None, description="Stream reasoning deltas (thinking models)")


class SessionCreate(SessionSpec):
    session_id: str | None = Field(None, description="Client-chosen id (e.g. the call id)")
    messages: list[dict[str, Any]] | None = Field(None, description="Seed conversation history")


class MessagesAppend(_Body):
    messages: list[dict[str, Any]] = Field(min_length=1)


class ResponseCreate(_Body):
    input: str | dict[str, Any] | list[dict[str, Any]] | None = Field(
        None, description="User utterance, one message, or a list of messages")
    params: dict[str, Any] | None = Field(None, description="Per-response sampling overrides")
    tools: list[dict[str, Any]] | None = Field(None, description="Replace the session's client tools")
    segmentation: bool | None = None
    stream: bool = True
    response_id: str | None = Field(None, max_length=128)


class CancelRequest(_Body):
    response_id: str | None = None
    spoken_text: str | None = Field(None, description="What the caller actually heard (barge-in)")
    discard: bool = Field(False, description="Remove the cancelled turn (input + partial reply) from history")


class TruncateRequest(_Body):
    spoken_text: str


class ToolResult(_Body):
    call_id: str
    output: Any
    is_error: bool = False


class ToolResultsSubmit(_Body):
    results: list[ToolResult] = Field(min_length=1)
    continue_response: bool = Field(True, alias="continue")
    stream: bool = True
    params: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
