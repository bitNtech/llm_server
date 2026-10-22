"""Workflows: declarative multi-step pipelines run as asynchronous executions.

Typical use: post-call processing (summarize -> extract structured data -> push to a CRM tool ->
notify), batch enrichment, or multi-agent chains. A workflow is data, not code:

    {"name": "post_call",
     "steps": [
       {"id": "summary", "type": "llm", "instructions": "Summarize the call.", "prompt": "{{input.transcript}}"},
       {"id": "extract", "type": "llm", "prompt": "{{input.transcript}}",
        "response_format": {"type": "json_schema", "json_schema": {"name": "lead", "schema": {...}}}},
       {"id": "is_lead", "type": "condition",
        "if": {"left": "{{steps.extract.output.interested}}", "op": "eq", "right": true},
        "then": "push", "else": "end"},
       {"id": "push", "type": "tool", "tool": {"type": "tool", "id": "tool_crm"},
        "arguments": {"name": "{{steps.extract.output.name}}", "summary": "{{steps.summary.output}}"}},
       {"id": "reply", "type": "agent", "agent_id": "agent_x",
        "input": "Draft a follow-up SMS: {{steps.summary.output}}"}
     ],
     "output": {"summary": "{{steps.summary.output}}", "lead": "{{steps.extract.output}}"}}

Step types: llm, agent, tool, condition, set. Steps run in order unless `next` (or a condition's
then/else) names another step id or "end". Templates are `{{ dotted.path }}` lookups only - no
expressions are evaluated. Tool steps use the step id as the tool call id within the execution,
so a resumed execution never repeats a side effect that already completed.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import orjson

from app.config import Settings
from app.core.errors import BadRequest, GatewayError
from app.engine.admission import AdmissionController
from app.engine.agents import AgentResolver, validate_params
from app.engine.runtime import ConversationRuntime, TurnOptions
from app.engine.sessions import LiveSession
from app.providers.vertex import VertexProvider
from app.services.usage import UsageRecorder
from app.tools import builtin
from app.tools.base import EMPTY_PARAMS, ToolContext, ToolSpec
from app.tools.executor import ToolExecutor

STEP_ID = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
TEMPLATE = re.compile(r"\{\{\s*([A-Za-z0-9_.\-\[\]]+)\s*\}\}")
OPS = {"eq", "ne", "gt", "gte", "lt", "lte", "contains", "in", "exists", "not_exists", "truthy", "falsy"}
STEP_TYPES = {"llm", "agent", "tool", "condition", "set"}


# ------------------------------------------------------------------ templating
def lookup(ctx: dict[str, Any], path: str) -> Any:
    value: Any = ctx
    for part in path.replace("[", ".").replace("]", "").split("."):
        if part == "":
            continue
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def render(template: Any, ctx: dict[str, Any]) -> Any:
    """Render strings, lists and objects. A string that is exactly one placeholder keeps the
    referenced value's type; placeholders inside text are interpolated (JSON for non-strings)."""
    if isinstance(template, str):
        whole = TEMPLATE.fullmatch(template.strip())
        if whole:
            return lookup(ctx, whole.group(1))

        def sub(match: re.Match[str]) -> str:
            value = lookup(ctx, match.group(1))
            if value is None:
                return ""
            return value if isinstance(value, str) else orjson.dumps(value, default=str).decode()
        return TEMPLATE.sub(sub, template)
    if isinstance(template, list):
        return [render(v, ctx) for v in template]
    if isinstance(template, dict):
        return {k: render(v, ctx) for k, v in template.items()}
    return template


def compare(left: Any, op: str, right: Any) -> bool:
    try:
        if op == "eq":
            return left == right
        if op == "ne":
            return left != right
        if op in ("gt", "gte", "lt", "lte"):
            a, b = float(left), float(right)
            return {"gt": a > b, "gte": a >= b, "lt": a < b, "lte": a <= b}[op]
        if op == "contains":
            return right in left if isinstance(left, (str, list, dict)) else False
        if op == "in":
            return left in right if isinstance(right, (str, list, dict)) else False
        if op == "exists":
            return left is not None
        if op == "not_exists":
            return left is None
        if op == "truthy":
            return bool(left)
        if op == "falsy":
            return not left
    except (TypeError, ValueError):
        return False
    return False


# ------------------------------------------------------------------ validation
def validate_workflow(spec: dict[str, Any]) -> None:
    steps = spec.get("steps")
    if not isinstance(steps, list) or not steps:
        raise BadRequest("workflow needs a non-empty steps list")
    if len(steps) > 100:
        raise BadRequest("workflow may have at most 100 steps")
    ids = []
    for step in steps:
        if not isinstance(step, dict):
            raise BadRequest("each step must be an object")
        sid = step.get("id")
        if not isinstance(sid, str) or not STEP_ID.match(sid):
            raise BadRequest(f"step id {sid!r} must match {STEP_ID.pattern}")
        if sid in ids or sid == "end":
            raise BadRequest(f"duplicate or reserved step id {sid!r}")
        ids.append(sid)
        kind = step.get("type")
        if kind not in STEP_TYPES:
            raise BadRequest(f"step {sid}: type must be one of {sorted(STEP_TYPES)}")
        if kind == "llm" and not step.get("prompt"):
            raise BadRequest(f"step {sid}: llm steps need a prompt")
        if kind == "agent" and (not step.get("agent_id") or step.get("input") is None):
            raise BadRequest(f"step {sid}: agent steps need agent_id and input")
        if kind == "tool":
            ref = step.get("tool")
            if not isinstance(ref, dict) or ref.get("type") not in ("builtin", "tool", "mcp"):
                raise BadRequest(f"step {sid}: tool must be {{type: builtin|tool|mcp, ...}}")
        if kind == "condition":
            cond = step.get("if")
            if not isinstance(cond, dict) or cond.get("op") not in OPS:
                raise BadRequest(f"step {sid}: condition needs if: {{left, op, right}} with op in {sorted(OPS)}")
            if not step.get("then"):
                raise BadRequest(f"step {sid}: condition needs then")
        if kind == "set" and "value" not in step:
            raise BadRequest(f"step {sid}: set steps need value")
        if step.get("on_error", "fail") not in ("fail", "continue"):
            raise BadRequest(f"step {sid}: on_error must be fail or continue")
    targets = set(ids) | {"end"}
    for step in steps:
        for key in ("next", "then", "else"):
            if step.get(key) is not None and step[key] not in targets:
                raise BadRequest(f"step {step['id']}: {key} references unknown step {step[key]!r}")


# --------------------------------------------------------------------- runner
class WorkflowRunner:
    def __init__(self, settings: Settings, provider: VertexProvider, admission: AdmissionController,
                 runtime: ConversationRuntime, resolver: AgentResolver, executor: ToolExecutor,
                 usage: UsageRecorder) -> None:
        self.settings = settings
        self.provider = provider
        self.admission = admission
        self.runtime = runtime
        self.resolver = resolver
        self.executor = executor
        self.usage = usage

    async def run(self, tenant_id: str, execution_id: str, workflow: dict[str, Any], inputs: Any,
                  completed: dict[str, Any], emit: Callable[[str, dict[str, Any]], None]) -> Any:
        """Run steps; `completed` holds step results of an earlier interrupted attempt (resume)."""
        steps = {s["id"]: s for s in workflow["steps"]}
        order = [s["id"] for s in workflow["steps"]]
        ctx: dict[str, Any] = {"input": inputs, "steps": dict(completed), "execution": {"id": execution_id}}
        current: str | None = order[0]
        transitions = 0
        while current and current != "end":
            transitions += 1
            if transitions > self.settings.workflow_max_steps:
                raise GatewayError(f"workflow exceeded WORKFLOW_MAX_STEPS={self.settings.workflow_max_steps}",
                                   code="workflow_step_limit", status=422)
            step = steps[current]
            following = order[order.index(current) + 1] if order.index(current) + 1 < len(order) else None
            if step["type"] == "condition":
                cond = step["if"]
                ok = compare(render(cond.get("left"), ctx), cond["op"], render(cond.get("right"), ctx))
                ctx["steps"][current] = {"status": "completed", "output": ok}
                emit("step.completed", {"step_id": current, "type": "condition", "output": ok})
                current = step["then"] if ok else step.get("else", following)
                continue
            if current in completed and completed[current].get("status") == "completed":
                current = step.get("next", following)  # resumed: already done
                continue
            emit("step.started", {"step_id": current, "type": step["type"]})
            try:
                output = await self._step(tenant_id, execution_id, step, ctx)
            except GatewayError as exc:
                ctx["steps"][current] = {"status": "failed", "error": exc.message}
                emit("step.failed", {"step_id": current, "error": exc.to_dict()["error"]})
                if step.get("on_error", "fail") == "fail":
                    raise
                current = step.get("next", following)
                continue
            ctx["steps"][current] = {"status": "completed", "output": output}
            emit("step.completed", {"step_id": current, "type": step["type"], "output": output})
            current = step.get("next", following)
        if "output" in workflow and workflow["output"] is not None:
            return render(workflow["output"], ctx)
        last = [sid for sid in order if ctx["steps"].get(sid, {}).get("status") == "completed"]
        return ctx["steps"][last[-1]]["output"] if last else None

    async def _step(self, tenant_id: str, execution_id: str, step: dict[str, Any], ctx: dict[str, Any]) -> Any:
        kind = step["type"]
        if kind == "set":
            return render(step["value"], ctx)
        if kind == "llm":
            return await self._llm(tenant_id, execution_id, step, ctx)
        if kind == "agent":
            return await self._agent(tenant_id, execution_id, step, ctx)
        return await self._tool(tenant_id, execution_id, step, ctx)

    async def _llm(self, tenant_id: str, execution_id: str, step: dict[str, Any], ctx: dict[str, Any]) -> Any:
        prompt = render(step["prompt"], ctx)
        messages = []
        if step.get("instructions"):
            messages.append({"role": "system", "content": render(step["instructions"], ctx)})
        content = prompt if isinstance(prompt, str) else orjson.dumps(prompt).decode()
        messages.append({"role": "user", "content": content})
        params = validate_params({k: step[k] for k in ("max_tokens", "temperature", "response_format",
                                                       "reasoning_effort") if step.get(k) is not None},
                                 self.settings.max_tokens_limit)
        response_format = params.get("response_format")
        if response_format is not None and not isinstance(response_format, dict):
            raise BadRequest(f"step {step['id']}: response_format must be an object")
        body: dict[str, Any] = {"model": step.get("model") or "default", "messages": messages,
                                "max_tokens": self.settings.default_max_tokens, **params}
        async with self.admission.slot():
            result, stats = await self.provider.complete(body, request_id=f"{execution_id}-{step['id']}",
                                                         scope=tenant_id, timeout=step.get("timeout_s"))
        self.usage.record(tenant_id, api="workflow", model=stats.model, agent_id=None, usage=result.get("usage"))
        text = ((result.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        if (step.get("response_format") or {}).get("type") in ("json_schema", "json_object"):
            try:
                return orjson.loads(text)
            except orjson.JSONDecodeError as exc:
                raise GatewayError(f"step {step['id']}: model did not return valid JSON",
                                   code="invalid_model_json", status=502) from exc
        return text.strip()

    async def _agent(self, tenant_id: str, execution_id: str, step: dict[str, Any], ctx: dict[str, Any]) -> Any:
        agent = await self.resolver.resolve(tenant_id, step["agent_id"], step.get("agent_version"))
        session = LiveSession(tenant_id=tenant_id, id=f"{execution_id}_{step['id']}"[:128], agent_id=agent.id,
                              agent_version=agent.version, current_agent_id=agent.id,
                              current_agent_version=agent.version, channel="workflow",
                              chars_per_token=self.settings.context_chars_per_token)
        return await run_agent_turn(self.runtime, session, render(step["input"], ctx), api="workflow",
                                    params=step.get("params"))

    async def _tool(self, tenant_id: str, execution_id: str, step: dict[str, Any], ctx: dict[str, Any]) -> Any:
        spec = await resolve_tool(self.resolver, tenant_id, step["tool"])
        args = render(step.get("arguments") or {}, ctx)
        if not isinstance(args, dict):
            raise BadRequest(f"step {step['id']}: arguments must render to an object")
        outcome = await self.executor.execute(ToolContext(tenant_id=tenant_id, scope_id=execution_id),
                                              spec, step["id"], orjson.dumps(args).decode())
        if outcome.is_error:
            raise GatewayError(f"step {step['id']}: {outcome.output}", code="tool_failed", status=502)
        if outcome.data is not None:
            return outcome.data
        try:
            return orjson.loads(outcome.output)
        except orjson.JSONDecodeError:
            return outcome.output


async def resolve_tool(resolver: AgentResolver, tenant_id: str, ref: dict[str, Any]) -> ToolSpec:
    kind = ref.get("type")
    if kind == "builtin":
        tool = builtin.REGISTRY.get(ref.get("name", ""))
        if tool is None:
            raise BadRequest(f"unknown builtin tool {ref.get('name')!r}")
        return tool.spec
    if kind == "tool":
        doc = await resolver.tools.get(tenant_id, ref.get("id", ""))
        if doc["kind"] != "http":
            raise BadRequest(f"tool {doc['id']} is a client tool and cannot run on the server")
        return ToolSpec(name=doc["name"], description=doc.get("description", ""),
                        parameters=doc.get("parameters") or dict(EMPTY_PARAMS), kind="http",
                        side_effects=bool(doc.get("side_effects")), timeout_s=doc.get("timeout_s"),
                        target={"tool_id": doc["id"], "http": doc["http"],
                                "secret_headers_enc": doc.get("secret_headers_enc"),
                                "signing_secret_enc": doc.get("signing_secret_enc")})
    if kind == "mcp":
        server = await resolver.mcp_servers.get(tenant_id, ref.get("server_id", ""))
        for spec in await resolver.mcp.tool_specs(tenant_id, server):
            if spec.target["remote_name"] == ref.get("name"):
                return spec
        raise BadRequest(f"MCP server {server['id']} has no tool {ref.get('name')!r}")
    raise BadRequest("tool reference type must be builtin, tool or mcp")


async def run_agent_turn(runtime: ConversationRuntime, session: LiveSession, input_value: Any, api: str,
                         params: dict[str, Any] | None = None, response_format: dict[str, Any] | None = None) -> Any:
    """Run one non-interactive agent turn (server tools only) and return its result."""
    if response_format:
        params = {**(params or {}), "response_format": response_format}
    gen = runtime.start_response(session, TurnOptions(input=input_value, params=params, segmentation=False,
                                                      allow_client_tools=False, api=api))
    done: dict[str, Any] = {}
    try:
        async for event in gen.events():
            if event.type == "response.done":
                done = event.data
    finally:
        if not gen.done:
            gen.cancel("execution_cancelled")
    if done.get("status") == "failed":
        err = done.get("error") or {}
        raise GatewayError(err.get("message", "agent turn failed"), code=err.get("code", "agent_failed"), status=502)
    if done.get("status") == "cancelled":
        raise GatewayError("agent turn was cancelled", code="cancelled", status=499)
    text = done.get("text", "")
    structured: Any = None
    if response_format and response_format.get("type") in ("json_schema", "json_object"):
        try:
            structured = orjson.loads(text)
        except orjson.JSONDecodeError:
            structured = None
    return {"text": text, "structured": structured, "status": done.get("status"),
            "server_tool_calls": done.get("server_tool_calls", []), "usage": done.get("usage"),
            "agent": done.get("agent"), "model": done.get("model")}
