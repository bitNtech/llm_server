"""Agents (LiveKit-style): instructions + model + tools + handoff targets, versioned.

An agent document is immutable per version (`agents/{id}/versions/{n}`); sessions pin the version
they started with, so editing an agent never changes a call in progress. Tool references are
resolved per turn into a `Toolset` (model-facing function name -> ToolSpec):

    {"type": "builtin", "name": "get_current_time"}
    {"type": "tool", "id": "tool_..."}                       tenant HTTP or client tool
    {"type": "mcp", "server_id": "mcp_...", "tools": [..]}   all (or listed) tools of an MCP server
    {"type": "client", "name": "...", "description": "...", "parameters": {...}}   inline client tool

Handoffs become `transfer_to_<agent>` tools; calling one switches the session's active agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import orjson

from app.config import Settings
from app.core.cache import TTLCache
from app.core.errors import BadRequest, GatewayError, NotFound
from app.core.ids import slug
from app.core.logging import get_logger
from app.providers.vertex import VertexProvider
from app.store.repos import AgentRepo, McpServerRepo, ToolRepo
from app.tools import builtin
from app.tools.base import EMPTY_PARAMS, ToolSpec, check_name, check_schema
from app.tools.mcp import MCPManager

log = get_logger("app.agents")

AGENT_PARAM_KEYS = {"temperature", "top_p", "top_k", "max_tokens", "stop", "seed", "presence_penalty",
                    "frequency_penalty", "reasoning_effort", "response_format", "tool_choice", "parallel_tool_calls"}


@dataclass
class ResolvedAgent:
    id: str | None
    version: int | None
    name: str
    instructions: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)
    handoffs: list[dict[str, Any]] = field(default_factory=list)
    segmentation: bool = True
    max_tool_rounds: int | None = None
    greeting: str | None = None
    on_session_end: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "version": self.version, "name": self.name, "model": self.model,
                "greeting": self.greeting}


@dataclass
class Toolset:
    specs: dict[str, ToolSpec] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add(self, spec: ToolSpec) -> None:
        if spec.name in self.specs:
            self.warnings.append(f"duplicate tool name {spec.name!r} ignored")
            return
        self.specs[spec.name] = spec

    def schemas(self) -> list[dict[str, Any]]:
        return [s.schema() for s in self.specs.values()]

    def get(self, name: str) -> ToolSpec | None:
        return self.specs.get(name)

    def chars(self) -> int:
        return len(orjson.dumps(self.schemas())) if self.specs else 0


def validate_params(params: dict[str, Any] | None, max_tokens_limit: int) -> dict[str, Any]:
    if not params:
        return {}
    unknown = sorted(set(params) - AGENT_PARAM_KEYS)
    if unknown:
        raise BadRequest(f"unsupported params: {', '.join(unknown)}", details={"allowed": sorted(AGENT_PARAM_KEYS)})
    mt = params.get("max_tokens")
    if mt is not None and (not isinstance(mt, int) or not 1 <= mt <= max_tokens_limit):
        raise BadRequest(f"max_tokens must be an integer between 1 and {max_tokens_limit}")
    return dict(params)


def client_tool_spec(tool: dict[str, Any]) -> ToolSpec:
    """Accept {"name","description","parameters"} or OpenAI {"type":"function","function":{...}}."""
    fn = tool.get("function") if tool.get("type") == "function" else tool
    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
        raise BadRequest("client tools need a name (OpenAI function tool format)")
    return ToolSpec(name=check_name(fn["name"]), description=str(fn.get("description") or "")[:4000],
                    parameters=check_schema(fn.get("parameters"), f"parameters of {fn['name']}"), kind="client")


class AgentResolver:
    def __init__(self, settings: Settings, provider: VertexProvider, agents: AgentRepo, tools: ToolRepo,
                 mcp_servers: McpServerRepo, mcp: MCPManager) -> None:
        self.settings = settings
        self.provider = provider
        self.agents = agents
        self.tools = tools
        self.mcp_servers = mcp_servers
        self.mcp = mcp
        self._versions: TTLCache[tuple[str, str, int], ResolvedAgent] = TTLCache(0, 5000)  # immutable
        self._docs: TTLCache[tuple[str, str, str], dict | None] = TTLCache(10.0, 20_000)

    # ---------------------------------------------------------------- agents
    def from_doc(self, doc: dict[str, Any]) -> ResolvedAgent:
        return ResolvedAgent(
            id=doc["id"], version=doc["version"], name=doc["name"], instructions=doc.get("instructions", ""),
            model=self.provider.resolve_model(doc.get("model")), params=dict(doc.get("params") or {}),
            tools=list(doc.get("tools") or []), handoffs=list(doc.get("handoffs") or []),
            segmentation=doc.get("segmentation", True), max_tool_rounds=doc.get("max_tool_rounds"),
            greeting=doc.get("greeting"), on_session_end=doc.get("on_session_end"),
            metadata=dict(doc.get("metadata") or {}))

    async def resolve(self, tenant_id: str, agent_id: str, version: int | None = None) -> ResolvedAgent:
        if version is None:
            doc = await self._doc(tenant_id, "agent", agent_id)
            if doc is None:
                raise NotFound(f"agent {agent_id} not found", code="agent_not_found")
            version = doc["version"]
        key = (tenant_id, agent_id, int(version))
        agent = self._versions.get(key)
        if agent is None:
            agent = self.from_doc(await self.agents.get_version(tenant_id, agent_id, int(version)))
            self._versions.put(key, agent)
        return agent

    def adhoc(self, instructions: str | None, model: str | None) -> ResolvedAgent:
        """Sessions without an agent: server default instructions/model."""
        return ResolvedAgent(id=None, version=None, name="default",
                             instructions=instructions or self.settings.default_instructions,
                             model=self.provider.resolve_model(model))

    def invalidate(self, tenant_id: str, kind: str, doc_id: str) -> None:
        self._docs.pop((tenant_id, kind, doc_id))

    async def _doc(self, tenant_id: str, kind: str, doc_id: str) -> dict[str, Any] | None:
        key = (tenant_id, kind, doc_id)
        cached = self._docs.get(key, default=False)
        if cached is not False:
            return cached
        repo = {"agent": self.agents, "tool": self.tools, "mcp": self.mcp_servers}[kind]
        doc = await repo.find(tenant_id, doc_id)
        self._docs.put(key, doc)
        return doc

    # ----------------------------------------------------------- validation
    async def validate_spec(self, tenant_id: str, spec: dict[str, Any], self_id: str | None = None) -> None:
        """Check an agent definition before it is stored: references exist, names are unique."""
        if spec.get("model") is not None:
            self.provider.resolve_model(spec["model"])
        validate_params(spec.get("params"), self.settings.max_tokens_limit)
        names: set[str] = set()

        def claim(name: str) -> None:
            if name in names:
                raise BadRequest(f"two tools of this agent are both named {name!r}", code="duplicate_tool_name")
            names.add(name)

        for ref in spec.get("tools") or []:
            kind = ref.get("type")
            if kind == "builtin":
                if ref.get("name") not in builtin.REGISTRY:
                    raise BadRequest(f"unknown builtin tool {ref.get('name')!r}",
                                     details={"available": sorted(builtin.REGISTRY)})
                claim(ref["name"])
            elif kind == "tool":
                doc = await self.tools.find(tenant_id, str(ref.get("id")))
                if doc is None:
                    raise BadRequest(f"tool {ref.get('id')} not found", code="tool_not_found")
                claim(doc["name"])
            elif kind == "mcp":
                if await self.mcp_servers.find(tenant_id, str(ref.get("server_id"))) is None:
                    raise BadRequest(f"MCP server {ref.get('server_id')} not found", code="mcp_server_not_found")
            elif kind == "client":
                claim(client_tool_spec(ref).name)
            else:
                raise BadRequest(f"unknown tool reference type {kind!r} (builtin, tool, mcp, client)")
        for handoff in spec.get("handoffs") or []:
            target = handoff.get("agent_id")
            if target == self_id:
                raise BadRequest("an agent cannot hand off to itself")
            doc = await self.agents.find(tenant_id, str(target))
            if doc is None:
                raise BadRequest(f"handoff target agent {target} not found", code="agent_not_found")
            claim(handoff.get("name") or f"transfer_to_{slug(doc['name'])}")

    # -------------------------------------------------------------- toolsets
    async def toolset(self, tenant_id: str, agent: ResolvedAgent, client_tools: list[dict[str, Any]] | None = None,
                      *, allow_client: bool = True, allow_handoffs: bool = True) -> Toolset:
        toolset = Toolset()
        for ref in agent.tools:
            kind = ref.get("type")
            try:
                if kind == "builtin":
                    tool = builtin.REGISTRY.get(ref.get("name", ""))
                    if tool is None:
                        toolset.warnings.append(f"builtin tool {ref.get('name')} no longer exists")
                        continue
                    toolset.add(tool.spec)
                elif kind == "tool":
                    doc = await self._doc(tenant_id, "tool", ref["id"])
                    if doc is None:
                        toolset.warnings.append(f"tool {ref['id']} was deleted")
                        continue
                    if doc["kind"] == "client":
                        if allow_client:
                            toolset.add(ToolSpec(name=doc["name"], description=doc.get("description", ""),
                                                 parameters=doc.get("parameters") or dict(EMPTY_PARAMS), kind="client"))
                    else:
                        toolset.add(ToolSpec(
                            name=doc["name"], description=doc.get("description", ""),
                            parameters=doc.get("parameters") or dict(EMPTY_PARAMS), kind="http",
                            side_effects=bool(doc.get("side_effects")), timeout_s=doc.get("timeout_s"),
                            target={"tool_id": doc["id"], "http": doc["http"],
                                    "secret_headers_enc": doc.get("secret_headers_enc"),
                                    "signing_secret_enc": doc.get("signing_secret_enc")}))
                elif kind == "mcp":
                    server = await self._doc(tenant_id, "mcp", ref["server_id"])
                    if server is None or not server.get("enabled", True):
                        toolset.warnings.append(f"MCP server {ref['server_id']} is missing or disabled")
                        continue
                    allow = set(ref["tools"]) if ref.get("tools") else None
                    for spec in await self.mcp.tool_specs(tenant_id, server):
                        if allow is None or spec.target["remote_name"] in allow:
                            toolset.add(spec)
                elif kind == "client" and allow_client:
                    toolset.add(client_tool_spec(ref))
            except GatewayError as exc:  # one broken integration must not take the whole call down
                toolset.warnings.append(f"{kind} tool unavailable: {exc.message}")
                log.warning("tool unavailable for this turn", extra={"agent_id": agent.id, "error": exc.message})
        if allow_client:
            for raw in client_tools or []:
                toolset.add(client_tool_spec(raw))
        if allow_handoffs:
            for handoff in agent.handoffs:
                doc = await self._doc(tenant_id, "agent", handoff["agent_id"])
                if doc is None:
                    toolset.warnings.append(f"handoff target {handoff['agent_id']} was deleted")
                    continue
                desc = handoff.get("description") or doc.get("description") or f"Transfer to {doc['name']}."
                toolset.add(ToolSpec(
                    name=handoff.get("name") or f"transfer_to_{slug(doc['name'])}"[:64],
                    description=f"Transfer the conversation to the {doc['name']} agent. Use when: {desc}",
                    parameters={"type": "object", "properties": {"reason": {
                        "type": "string", "description": "Why the conversation is being transferred"}}},
                    kind="handoff", target={"agent_id": doc["id"], "agent_name": doc["name"]}))
        if toolset.warnings:
            log.warning("toolset warnings", extra={"agent_id": agent.id, "warnings": toolset.warnings[:5]})
        return toolset
