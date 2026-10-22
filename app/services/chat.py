"""Stateless OpenAI-compatible chat completions (`POST /v1/chat/completions`).

This is the integration point for OpenAI SDKs and voice runtimes such as LiveKit Agents
(`livekit.plugins.openai.LLM(base_url=<server>/v1, api_key=<key>, model=...)`), which keep the
conversation themselves and execute their own function tools.

  model = "google/gemini-..." or "default"
      direct pass-through to Vertex: frames are relayed as they arrive (tool calls, usage), with
      gateway defaults, limits, failover, admission control and thought-signature repair
  model = "agent:<agent_id>" (or "agent:<agent_id>@<version>")
      the agent's instructions are prepended, its server-side tools (builtin / HTTP / MCP) run
      inside the server - the caller only sees the final text - and the caller's own tools are
      passed through as normal OpenAI tool calls (finish_reason "tool_calls")

Gemini requires each function call's thought signature back on the next request. OpenAI SDKs
(and LiveKit) drop the unknown `extra_content` field, so signatures the server streamed out are
remembered per tenant and re-attached automatically.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import orjson

from app.config import Settings
from app.core.errors import BadRequest, GatewayError
from app.core.metrics import GENERATION_DURATION, GENERATIONS, INTER_TOKEN, TTFT
from app.engine.admission import AdmissionController
from app.engine.agents import AgentResolver
from app.engine.generation import Generation, GenerationRegistry
from app.engine.runtime import ConversationRuntime, TurnOptions
from app.engine.sessions import LiveSession, content_text
from app.providers.vertex import DONE, VertexProvider
from app.services.usage import UsageRecorder

FINISH = {"completed": "stop", "requires_action": "tool_calls", "incomplete": "length", "cancelled": "stop"}


def sse(payload: Any) -> bytes:
    return b"data: " + orjson.dumps(payload) + b"\n\n"


def sse_error(error: GatewayError, request_id: str | None) -> bytes:
    return sse(error.to_dict(request_id)) + b"data: [DONE]\n\n"


class ChatRequest:
    def __init__(self, body: dict[str, Any], agent_id: str | None, agent_version: int | None) -> None:
        self.body = body
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.stream = bool(body.get("stream"))
        self.include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        self.requested_model = str(body.get("model") or "default")


class ChatService:
    def __init__(self, settings: Settings, provider: VertexProvider, admission: AdmissionController,
                 registry: GenerationRegistry, resolver: AgentResolver, runtime: ConversationRuntime,
                 usage: UsageRecorder) -> None:
        self.settings = settings
        self.provider = provider
        self.admission = admission
        self.registry = registry
        self.resolver = resolver
        self.runtime = runtime
        self.usage = usage

    # ------------------------------------------------------------ validation
    def prepare(self, body: Any) -> ChatRequest:
        if not isinstance(body, dict):
            raise BadRequest("request body must be a JSON object")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
            raise BadRequest("messages must be a non-empty list of message objects")
        total = sum(len(content_text(m.get("content"))) for m in messages)
        if total > self.settings.max_input_chars * 8:
            raise BadRequest("messages are too large")
        if body.get("n", 1) != 1:
            raise BadRequest("n > 1 is not supported")
        max_tokens = (body.pop("max_completion_tokens", None) or body.get("max_tokens")
                      or self.settings.default_max_tokens)
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= self.settings.max_tokens_limit:
            raise BadRequest(f"max_tokens must be between 1 and MAX_TOKENS_LIMIT={self.settings.max_tokens_limit}")
        body["max_tokens"] = max_tokens
        if self.settings.default_temperature is not None:
            body.setdefault("temperature", self.settings.default_temperature)
        model = str(body.get("model") or "default")
        if model.startswith("agent:"):
            ref = model[6:]
            agent_id, _, version = ref.partition("@")
            if not agent_id or (version and not version.isdigit()):
                raise BadRequest("model must be 'agent:<agent_id>' or 'agent:<agent_id>@<version>'")
            return ChatRequest(body, agent_id, int(version) if version else None)
        body["model"] = self.provider.resolve_model(model)
        return ChatRequest(body, None, None)

    # --------------------------------------------------------------- direct
    async def complete(self, req: ChatRequest, tenant_id: str, request_id: str) -> dict[str, Any]:
        if req.agent_id:
            return await self._agent_complete(req, tenant_id, request_id)
        started = time.perf_counter()
        gen = Generation(request_id, api="chat", tenant_id=tenant_id)
        holder: dict[str, Any] = {}

        async def produce() -> None:
            try:
                async with self.admission.slot():
                    holder["result"], holder["stats"] = await self.provider.complete(
                        req.body, request_id=request_id, scope=tenant_id, timeout=self.settings.generation_timeout_s)
            except GatewayError as exc:
                holder["error"] = exc

        gen.start(produce(), self.registry)
        try:
            await asyncio.shield(gen.finished.wait())
        except asyncio.CancelledError:
            gen.cancel("client_disconnect")
            raise
        if gen.cancel_reason:
            GENERATIONS.labels("chat", "cancelled").inc()
            raise BadRequest("request was cancelled", code="cancelled", status=499)
        if "error" in holder or "result" not in holder:
            GENERATIONS.labels("chat", "failed").inc()
            raise holder.get("error") or GatewayError("generation failed")
        result = holder["result"]
        self.usage.record(tenant_id, api="chat", model=holder["stats"].model, agent_id=None,
                          usage=result.get("usage"))
        GENERATIONS.labels("chat", "completed").inc()
        GENERATION_DURATION.labels("chat", "completed").observe(time.perf_counter() - started)
        return result

    def stream(self, req: ChatRequest, tenant_id: str, request_id: str) -> Generation:
        gen = Generation(request_id, api="chat", tenant_id=tenant_id)
        producer = self._agent_stream(gen, req, tenant_id) if req.agent_id else self._direct_stream(gen, req, tenant_id)
        gen.start(producer, self.registry)
        return gen

    async def _direct_stream(self, gen: Generation, req: ChatRequest, tenant_id: str) -> None:
        started = time.perf_counter()
        first, emitted, status = True, False, "completed"
        model = None
        body = {**req.body, "stream_options": {"include_usage": True}}
        try:
            async with asyncio.timeout(self.settings.generation_timeout_s):
                async with self.admission.slot(), self.provider.stream(body, request_id=gen.id,
                                                                        scope=tenant_id) as stream:
                    model = stream.stats.model
                    last: float | None = None
                    async for line, chunk in stream.raw():
                        if chunk is DONE:
                            gen.emit(b"data: [DONE]\n\n")
                            break
                        now = time.perf_counter()
                        if chunk.get("usage"):
                            self.usage.record(tenant_id, api="chat", model=model, agent_id=None, usage=chunk["usage"])
                        choices = chunk.get("choices") or []
                        if first and any((c.get("delta") or {}).get(k) for c in choices
                                         for k in ("content", "tool_calls")):
                            first = False
                            TTFT.labels("chat").observe(now - started)
                        if last is not None:
                            INTER_TOKEN.labels("chat").observe(now - last)
                        last = now
                        if not choices and chunk.get("usage") and not req.include_usage:
                            continue
                        gen.emit(line.encode() + b"\n\n")
                        emitted = True
        except asyncio.CancelledError:
            status = "cancelled"
            if gen.cancel_reason != "client_disconnect":  # someone else cancelled it: tell the caller
                gen.emit(sse_error(BadRequest("request was cancelled", code="cancelled"), gen.id))
            raise
        except TimeoutError:
            status = "failed"
            err = GatewayError("generation exceeded GENERATION_TIMEOUT_S", code="provider_timeout", status=504)
            gen.emit(sse_error(err, gen.id) if emitted else err)
        except GatewayError as exc:
            status = "failed"
            # before the first frame the route can still answer with a real HTTP status
            gen.emit(sse_error(exc, gen.id) if emitted else exc)
        finally:
            GENERATIONS.labels("chat", status).inc()
            GENERATION_DURATION.labels("chat", status).observe(time.perf_counter() - started)

    # ---------------------------------------------------------------- agents
    async def _agent_session(self, req: ChatRequest, tenant_id: str,
                             request_id: str) -> tuple[LiveSession, TurnOptions]:
        agent = await self.resolver.resolve(tenant_id, req.agent_id or "", req.agent_version)
        system_parts: list[str] = []
        history: list[dict[str, Any]] = []
        for m in req.body["messages"]:
            if m.get("role") in ("system", "developer"):
                system_parts.append(content_text(m.get("content")))
            else:
                history.append(m)
        session = LiveSession(tenant_id=tenant_id, id=f"chat_{request_id}"[:128], agent_id=agent.id,
                              agent_version=agent.version, current_agent_id=agent.id,
                              current_agent_version=agent.version, instructions="\n\n".join(system_parts),
                              chars_per_token=self.settings.context_chars_per_token, channel="chat")
        for raw in history:
            session.add_openai(raw, self.settings.max_input_chars * 8)
        params = {k: req.body[k] for k in ("temperature", "top_p", "max_tokens", "stop", "seed", "presence_penalty",
                                           "frequency_penalty", "response_format", "reasoning_effort")
                  if req.body.get(k) is not None}
        opts = TurnOptions(params=params, tools=req.body.get("tools") or None, tool_choice=req.body.get("tool_choice"),
                           segmentation=False, response_id=f"{request_id}_turn", allow_handoffs=False, api="chat")
        return session, opts

    def _chunk(self, req: ChatRequest, request_id: str, delta: dict[str, Any], finish: str | None = None,
               usage: dict[str, Any] | None = None) -> dict[str, Any]:
        chunk: dict[str, Any] = {"id": f"chatcmpl-{request_id}", "object": "chat.completion.chunk",
                                 "created": int(time.time()), "model": req.requested_model,
                                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage is not None:
            chunk["usage"] = usage
        return chunk

    @staticmethod
    def _usage(done: dict[str, Any]) -> dict[str, Any]:
        u = done.get("usage") or {}
        return {"prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                "total_tokens": u.get("total_tokens", 0),
                "prompt_tokens_details": {"cached_tokens": u.get("cached_tokens", 0)}}

    def _client_calls(self, events: list[dict[str, Any]], session: LiveSession) -> list[dict[str, Any]]:
        calls = []
        for i, call in enumerate(events):
            msg = next((tc for m in reversed(session.messages) if m.tool_calls for tc in m.tool_calls
                        if tc["id"] == call["call_id"]), None)
            entry = {"index": i, "id": call["call_id"], "type": "function",
                     "function": {"name": call["name"], "arguments": call["arguments"] or "{}"}}
            if msg and msg.get("extra_content"):
                entry["extra_content"] = msg["extra_content"]
            calls.append(entry)
        return calls

    async def _agent_stream(self, gen: Generation, req: ChatRequest, tenant_id: str) -> None:
        emitted = False
        try:
            session, opts = await self._agent_session(req, tenant_id, gen.id)
            inner = self.runtime.start_response(session, opts)
            client_calls: list[dict[str, Any]] = []
            try:
                gen.emit(sse(self._chunk(req, gen.id, {"role": "assistant", "content": ""})))
                emitted = True
                async for event in inner.events():
                    if event.type == "response.text.delta":
                        gen.emit(sse(self._chunk(req, gen.id, {"content": event.data["delta"]})))
                    elif event.type == "response.tool_call" and event.data.get("side") == "client":
                        client_calls.append(event.data)
                    elif event.type == "response.done":
                        done = event.data
                        if done["status"] == "failed":
                            err = GatewayError(done["error"]["message"], code=done["error"]["code"])
                            gen.emit(sse_error(err, gen.id))
                            return
                        if client_calls:
                            gen.emit(sse(self._chunk(req, gen.id, {"tool_calls": self._client_calls(client_calls,
                                                                                                    session)})))
                        finish = FINISH.get(done["status"], "stop")
                        gen.emit(sse(self._chunk(req, gen.id, {}, finish)))
                        if req.include_usage:
                            chunk = self._chunk(req, gen.id, {}, None, self._usage(done))
                            chunk["choices"] = []
                            gen.emit(sse(chunk))
                        gen.emit(b"data: [DONE]\n\n")
            finally:
                if not inner.done:
                    inner.cancel(gen.cancel_reason or "client_disconnect")
        except GatewayError as exc:
            gen.emit(sse_error(exc, gen.id) if emitted else exc)

    async def _agent_complete(self, req: ChatRequest, tenant_id: str, request_id: str) -> dict[str, Any]:
        session, opts = await self._agent_session(req, tenant_id, request_id)
        inner = self.runtime.start_response(session, opts)
        done: dict[str, Any] = {}
        client_calls: list[dict[str, Any]] = []
        try:
            async for event in inner.events():
                if event.type == "response.tool_call" and event.data.get("side") == "client":
                    client_calls.append(event.data)
                elif event.type == "response.done":
                    done = event.data
        finally:
            if not inner.done:
                inner.cancel("client_disconnect")
        if done.get("status") == "failed":
            if isinstance(inner.error, GatewayError):
                raise inner.error
            raise GatewayError(done["error"]["message"], code=done["error"]["code"])
        message: dict[str, Any] = {"role": "assistant", "content": done.get("text") or None}
        if client_calls:
            message["tool_calls"] = [{k: v for k, v in c.items() if k != "index"}
                                     for c in self._client_calls(client_calls, session)]
        return {"id": f"chatcmpl-{request_id}", "object": "chat.completion", "created": int(time.time()),
                "model": req.requested_model,
                "choices": [{"index": 0, "message": message, "finish_reason": FINISH.get(done.get("status"), "stop")}],
                "usage": self._usage(done)}
