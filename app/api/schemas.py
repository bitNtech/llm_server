"""Request bodies (validated, and documented in OpenAPI). Unknown fields are rejected."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["owner", "developer", "runtime", "viewer"]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------- platform
class TenantSettings(_Body):
    rate_limit_per_minute: int | None = Field(None, ge=0, le=1_000_000)


class TenantCreate(_Body):
    name: str = Field(min_length=1, max_length=120)
    id: str | None = Field(None, pattern=r"^[a-z0-9][a-z0-9_-]{2,62}$", description="Optional readable id")
    settings: TenantSettings | None = None


class TenantUpdate(_Body):
    name: str | None = Field(None, min_length=1, max_length=120)
    status: Literal["active", "suspended"] | None = None
    settings: TenantSettings | None = None


class ApiKeyCreate(_Body):
    name: str = Field(min_length=1, max_length=120)
    role: Role = "runtime"
    expires_in_days: int | None = Field(None, ge=1, le=3650)


class RealtimeTokenCreate(_Body):
    ttl_s: int = Field(600, ge=30, le=86_400)
    session_id: str | None = Field(None, max_length=128)
    agent_id: str | None = Field(None, max_length=128)


# ------------------------------------------------------------------------ agents
class ToolRef(_Body):
    type: Literal["builtin", "tool", "mcp", "client"]
    name: str | None = Field(None, max_length=64)
    id: str | None = Field(None, max_length=128)
    server_id: str | None = Field(None, max_length=128)
    tools: list[str] | None = Field(None, description="mcp: only these remote tools (default all)")
    description: str | None = Field(None, max_length=4000)
    parameters: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _shape(self) -> ToolRef:
        need = {"builtin": "name", "tool": "id", "mcp": "server_id", "client": "name"}[self.type]
        if not getattr(self, need):
            raise ValueError(f"{self.type} tool references need '{need}'")
        return self


class HandoffRef(_Body):
    agent_id: str = Field(max_length=128)
    name: str | None = Field(None, max_length=64, description="tool name (default transfer_to_<agent name>)")
    description: str | None = Field(None, max_length=2000, description="when the model should transfer")


class OnSessionEnd(_Body):
    workflow_id: str = Field(max_length=128)


class AgentFields(_Body):
    name: str | None = Field(None, min_length=1, max_length=80)
    description: str | None = Field(None, max_length=2000)
    instructions: str | None = Field(None, max_length=200_000)
    model: str | None = Field(None, max_length=120, description="Vertex model id; default VERTEX_MODEL")
    params: dict[str, Any] | None = Field(None, description="temperature, top_p, max_tokens, reasoning_effort, ...")
    tools: list[ToolRef] | None = None
    handoffs: list[HandoffRef] | None = None
    greeting: str | None = Field(None, max_length=2000)
    segmentation: bool | None = Field(None, description="emit sentence segments for TTS (default true)")
    max_tool_rounds: int | None = Field(None, ge=1, le=50)
    on_session_end: OnSessionEnd | None = None
    metadata: dict[str, Any] | None = None


class AgentCreate(AgentFields):
    name: str = Field(min_length=1, max_length=80)
    instructions: str = Field(min_length=1, max_length=200_000)


class AgentUpdate(AgentFields):
    pass


# ------------------------------------------------------------------------- tools
class HttpConfig(_Body):
    url: str = Field(min_length=8, max_length=2000, description="https URL; {arg} placeholders are filled")
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "POST"
    headers: dict[str, str] = Field(default_factory=dict, description="non-secret static headers")
    body_mode: Literal["arguments", "envelope"] = "arguments"


class ToolFields(_Body):
    name: str | None = Field(None, max_length=64)
    description: str | None = Field(None, max_length=4000)
    parameters: dict[str, Any] | None = None
    http: HttpConfig | None = None
    secret_headers: dict[str, str] | None = Field(None, description="encrypted at rest, never returned")
    signing_secret: str | None = Field(None, min_length=16, max_length=256)
    side_effects: bool | None = Field(None, description="exactly-once execution per tool call")
    timeout_s: float | None = Field(None, gt=0, le=300)


class ToolCreate(ToolFields):
    name: str = Field(max_length=64)
    description: str = Field(max_length=4000)
    kind: Literal["http", "client"] = "http"


class ToolUpdate(ToolFields):
    pass


class ToolInvoke(_Body):
    arguments: dict[str, Any] = Field(default_factory=dict)


class McpServerFields(_Body):
    name: str | None = Field(None, min_length=1, max_length=80)
    url: str | None = Field(None, min_length=8, max_length=2000)
    transport: Literal["streamable_http", "sse"] | None = None
    headers: dict[str, str] | None = None
    secret_headers: dict[str, str] | None = None
    tool_allowlist: list[str] | None = None
    tool_prefix: str | None = Field(None, max_length=30, pattern=r"^[A-Za-z0-9_.:-]*$")
    side_effect_tools: list[str] | None = None
    timeout_s: float | None = Field(None, gt=0, le=300)
    enabled: bool | None = None


class McpServerCreate(McpServerFields):
    name: str = Field(min_length=1, max_length=80)
    url: str = Field(min_length=8, max_length=2000)
    transport: Literal["streamable_http", "sse"] = "streamable_http"


class McpServerUpdate(McpServerFields):
    pass


# --------------------------------------------------------------------- workflows
class WorkflowFields(_Body):
    name: str | None = Field(None, min_length=1, max_length=80)
    description: str | None = Field(None, max_length=2000)
    input_schema: dict[str, Any] | None = None
    steps: list[dict[str, Any]] | None = None
    output: Any = None


class WorkflowCreate(WorkflowFields):
    name: str = Field(min_length=1, max_length=80)
    steps: list[dict[str, Any]]


class WorkflowUpdate(WorkflowFields):
    pass


class WorkflowRun(_Body):
    input: Any = None
    callback_url: str | None = Field(None, max_length=2000)
    metadata: dict[str, Any] | None = None


class ExecutionCreate(_Body):
    type: Literal["agent", "workflow"]
    agent_id: str | None = None
    agent_version: int | None = Field(None, ge=1)
    workflow_id: str | None = None
    input: Any = None
    session_id: str | None = Field(None, max_length=128, description="agent runs: continue this session")
    params: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    callback_url: str | None = Field(None, max_length=2000)
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _target(self) -> ExecutionCreate:
        if self.type == "agent" and not self.agent_id:
            raise ValueError("agent executions need agent_id")
        if self.type == "workflow" and not self.workflow_id:
            raise ValueError("workflow executions need workflow_id")
        return self


# ---------------------------------------------------------------------- sessions
class SessionFields(_Body):
    instructions: str | None = Field(None, max_length=200_000, description="appended to the agent's instructions")
    tools: list[dict[str, Any]] | None = Field(None, description="extra client tools (OpenAI function format)")
    params: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    segmentation: bool | None = None


class SessionCreate(SessionFields):
    session_id: str | None = Field(None, max_length=128, description="client-chosen id, e.g. the call id")
    agent_id: str | None = None
    agent_version: int | None = Field(None, ge=1)
    model: str | None = Field(None, description="agent-less sessions only")
    messages: list[dict[str, Any]] | None = Field(None, description="seed conversation history")


class SessionUpdate(SessionFields):
    pass


class MessagesAppend(_Body):
    messages: list[dict[str, Any]] = Field(min_length=1, max_length=500)


class ResponseCreate(_Body):
    input: str | dict[str, Any] | list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    instructions: str | None = Field(None, max_length=20_000)
    tool_choice: Any = None
    segmentation: bool | None = None
    say: str | None = Field(None, max_length=10_000, description="speak exactly this text (no model call)")
    stream: bool = True
    response_id: str | None = Field(None, max_length=128)


class CancelRequest(_Body):
    response_id: str | None = None
    spoken_text: str | None = Field(None, description="what the caller actually heard (barge-in)")
    discard: bool = Field(False, description="remove the cancelled turn (input + partial reply)")


class TruncateRequest(_Body):
    spoken_text: str


class ToolResult(_Body):
    call_id: str
    output: Any
    is_error: bool = False


class ToolResultsSubmit(_Body):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    results: list[ToolResult] = Field(min_length=1)
    continue_response: bool = Field(True, alias="continue")
    stream: bool = True
    params: dict[str, Any] | None = None


class HandoffRequest(_Body):
    agent_id: str
    reason: str = Field("", max_length=500)
    cancel_active: bool = Field(False, description="cancel a running response instead of failing with 409")


class EndRequest(_Body):
    reason: str = Field("ended_by_client", max_length=200)
