"""Realtime WebSocket for voice runtimes: `WS /v1/realtime`, protocol `aica.realtime.v1`.

One connection drives one session (new, or resumed with ?session_id=). JSON text frames; every
frame has a `type`. Server frames also carry `event_id` and a per-connection `seq`. The protocol
version is negotiated with the `Sec-WebSocket-Protocol: aica.realtime.v1` subprotocol (optional);
within v1 changes are additive only. See docs/REALTIME.md for the full event reference.

Authentication (first match wins): `Authorization: Bearer <api key | realtime token>`, subprotocol
`bearer.<realtime token>` (browsers cannot set headers), `?token=<realtime token>`, and - only when
WS_QUERY_API_KEY=true - `?api_key=`. Realtime tokens are minted with POST /v1/realtime/tokens and may
pin a session and/or agent.
"""
from __future__ import annotations

import asyncio
import secrets
from typing import Any

import orjson
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from app.api.deps import authenticate
from app.api.sessions import session_fields, turn_options
from app.core.errors import BadRequest, Conflict, Forbidden, GatewayError, NotFound
from app.core.ids import iso, utcnow, valid_client_id
from app.core.logging import get_logger, session_id_var
from app.core.metrics import WS_CONNECTIONS
from app.engine.generation import Generation
from app.engine.runtime import TurnOptions
from app.engine.sessions import LiveSession
from app.security.auth import Principal

router = APIRouter(tags=["realtime"])
log = get_logger("app.realtime")

PROTOCOL = "aica.realtime.v1"
CLOSE_NORMAL, CLOSE_GOING_AWAY, CLOSE_TOO_BIG, CLOSE_TRY_AGAIN = 1000, 1001, 1009, 1013
CLOSE_UNAUTHORIZED, CLOSE_FORBIDDEN, CLOSE_NOT_FOUND, CLOSE_IDLE, CLOSE_CONFLICT = 4401, 4403, 4404, 4408, 4409


class SlowConsumer(Exception):
    pass


class RealtimeConnection:
    def __init__(self, ws: WebSocket, services: Any, principal: Principal, subprotocol: str | None) -> None:
        self.ws = ws
        self.services = services
        self.settings = services.settings
        self.principal = principal
        self.subprotocol = subprotocol
        self.session: LiveSession | None = None
        self.out: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=self.settings.ws_send_queue_size)
        self.forwarders: set[asyncio.Task] = set()
        self.owned: set[str] = set()
        self.closing = False
        self.seq = 0
        self._drain_task: asyncio.Task | None = None
        # events are handled in order by one worker; response.cancel and ping skip the line, so a
        # barge-in is never stuck behind an event that waits for the running response's lock
        self.inbox: asyncio.Queue[tuple[int, dict[str, Any]] | None] = asyncio.Queue(maxsize=64)
        self.cancel_epoch = 0

    # ------------------------------------------------------------- draining
    def start_drain(self, grace_s: float) -> None:
        self._drain_task = asyncio.create_task(self._drain(grace_s), name="ws-drain")

    async def _drain(self, grace_s: float) -> None:
        """Let the response in progress finish (the caller hears the full sentence), then close."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + grace_s
        while self.owned and loop.time() < deadline:
            await asyncio.sleep(0.05)
        await asyncio.gather(*self.forwarders, return_exceptions=True)
        try:
            await self.send({"type": "session.closing", "reason": "server_shutdown",
                             "session_id": self.session.id if self.session else None})
        except SlowConsumer:
            pass
        await self.flush_out()
        await self.close(CLOSE_GOING_AWAY, "server shutting down")

    async def flush_out(self, timeout: float = 2.0) -> None:
        """Give the sender time to deliver queued events before the socket closes."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not self.out.empty() and loop.time() < deadline:
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.02)  # the last event may be in flight

    # ------------------------------------------------------------------ send
    async def send(self, event: dict[str, Any]) -> None:
        self.seq += 1
        event = {**event, "event_id": "evt_" + secrets.token_hex(8), "seq": self.seq}
        try:
            await asyncio.wait_for(self.out.put(event), self.settings.ws_send_timeout_s)
        except TimeoutError:
            raise SlowConsumer() from None

    async def _sender(self) -> None:
        timeout = self.settings.ws_send_timeout_s
        while True:
            event = await self.out.get()
            if event is None:
                return
            try:
                await asyncio.wait_for(self.ws.send_text(orjson.dumps(event, default=str).decode()), timeout)
            except TimeoutError:
                log.warning("websocket client not reading; closing")
                await self.close(CLOSE_TRY_AGAIN, "client too slow")
                return
            except (WebSocketDisconnect, RuntimeError):
                return

    async def error(self, exc: GatewayError, event_id: Any = None) -> None:
        payload: dict[str, Any] = {"type": "error", "error": exc.to_dict()["error"]}
        if event_id is not None:
            payload["client_event_id"] = event_id
        await self.send(payload)

    async def close(self, code: int, reason: str = "") -> None:
        if self.closing:
            return
        self.closing = True
        if self.ws.client_state == WebSocketState.CONNECTED:
            try:
                await self.ws.close(code, reason[:120])
            except RuntimeError:
                pass

    # ------------------------------------------------------------- lifecycle
    async def _open_session(self) -> tuple[LiveSession, str]:
        svc = self.services
        tenant = self.principal.tenant
        params = self.ws.query_params
        session_id = params.get("session_id") or self.principal.session_id
        if session_id and not valid_client_id(session_id):
            raise BadRequest("session_id must be 1-128 characters of [A-Za-z0-9_-]")
        if self.principal.session_id and session_id != self.principal.session_id:
            raise Forbidden("this token is restricted to another session", code="session_mismatch")
        agent_id = params.get("agent_id") or self.principal.agent_id
        if self.principal.agent_id and agent_id != self.principal.agent_id:
            raise Forbidden("this token is restricted to another agent", code="agent_mismatch")
        if session_id:
            try:
                session = await svc.sessions.get(tenant, session_id)
                if self.principal.agent_id and session.agent_id != self.principal.agent_id:
                    raise Forbidden("this token is restricted to another agent", code="agent_mismatch")
                return session, "session.resumed"
            except NotFound:
                if params.get("session_id") and params.get("create", "") not in ("1", "true"):
                    raise
        spec: dict[str, Any] = {"agent_id": agent_id} if agent_id else {}
        fields = await session_fields(svc, tenant, spec, creating=True)
        session = await svc.sessions.create(tenant, {**fields, "channel": "realtime"}, session_id)
        return session, "session.created"

    async def run(self) -> None:
        svc = self.services
        try:
            session, created = await self._open_session()
        except GatewayError as exc:
            await self.ws.accept(subprotocol=self.subprotocol)
            code = {404: CLOSE_NOT_FOUND, 403: CLOSE_FORBIDDEN, 409: CLOSE_CONFLICT}.get(exc.status, CLOSE_TRY_AGAIN)
            await self.ws.send_text(orjson.dumps({"type": "error", "error": exc.to_dict()["error"]}).decode())
            await self.ws.close(code, exc.message[:120])
            return
        self.session = session
        await self.ws.accept(subprotocol=self.subprotocol)
        session_id_var.set(session.id)
        session.attachments += 1
        svc.connections.add(self)
        WS_CONNECTIONS.inc()
        sender = asyncio.create_task(self._sender(), name="ws-sender")
        worker = asyncio.create_task(self._worker(), name="ws-dispatch")
        log.info("websocket connected", extra={"key_id": self.principal.key_id, "event": created})
        try:
            agent = await svc.runtime.current_agent(session)
            await self.send({"type": created, "protocol": PROTOCOL, "session": session.view(),
                             "agent": agent.public()})
            await self._receive_loop()
        except (WebSocketDisconnect, SlowConsumer):
            pass
        finally:
            if session.active is not None and session.active.id in self.owned:
                session.active.cancel("client_disconnect")  # nobody left to hear it
            session.attachments -= 1
            session.touch()
            worker.cancel()
            for task in self.forwarders:
                task.cancel()
            if self.out.full():
                sender.cancel()
            else:
                self.out.put_nowait(None)
            await asyncio.gather(sender, worker, *self.forwarders, return_exceptions=True)
            await self.close(CLOSE_NORMAL)
            svc.connections.discard(self)
            if self._drain_task and self._drain_task is not asyncio.current_task():
                self._drain_task.cancel()
            WS_CONNECTIONS.dec()
            log.info("websocket closed")

    async def _receive_loop(self) -> None:
        limit = self.settings.ws_max_message_bytes
        while True:
            try:
                message = await asyncio.wait_for(self.ws.receive(), self.settings.ws_idle_timeout_s)
            except TimeoutError:
                await self.close(CLOSE_IDLE, "idle timeout")
                return
            if message["type"] == "websocket.disconnect":
                return
            raw = message.get("text")
            if raw is None and message.get("bytes") is not None:
                raw = message["bytes"].decode("utf-8", errors="replace")
            if raw is None:
                continue
            if len(raw) > limit:
                await self.close(CLOSE_TOO_BIG, f"message exceeds {limit} bytes")
                return
            try:
                event = orjson.loads(raw)
                if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                    raise BadRequest('each message must be a JSON object with a string "type"')
            except orjson.JSONDecodeError:
                await self.error(BadRequest("invalid JSON", code="invalid_json"))
                continue
            except BadRequest as exc:
                await self.error(exc)
                continue
            if self.services.admission.draining:
                await self.error(GatewayError("server is shutting down", code="shutting_down", status=503))
                await self.close(CLOSE_GOING_AWAY, "server shutting down")
                return
            if event["type"] in ("response.cancel", "ping"):
                if event["type"] == "response.cancel":
                    self.cancel_epoch += 1  # also cancels response.create events still queued
                await self._handle(event)
            else:
                await self.inbox.put((self.cancel_epoch, event))

    async def _worker(self) -> None:
        while True:
            epoch, event = await self.inbox.get()
            if event["type"] == "response.create" and epoch < self.cancel_epoch:
                await self.error(Conflict("cancelled before it started", code="response_cancelled"),
                                 event.get("event_id"))
                continue
            try:
                await self._handle(event)
            except SlowConsumer:
                await self.close(CLOSE_TRY_AGAIN, "client too slow")
                return

    async def _handle(self, event: dict[str, Any]) -> None:
        try:
            await self.dispatch(event)
        except SlowConsumer:
            raise
        except GatewayError as exc:
            await self.error(exc, event.get("event_id"))
        except Exception:
            log.exception("websocket handler error", extra={"event_type": event.get("type")})
            await self.error(GatewayError("internal error"), event.get("event_id"))

    # -------------------------------------------------------------- handlers
    async def dispatch(self, event: dict[str, Any]) -> None:
        kind = event["type"]
        session = self.session
        assert session is not None
        svc = self.services
        runtime = svc.runtime
        session.touch()

        if kind == "response.create":
            svc.auth.rate_limit(self.principal)
            spec = {k: event.get(k) for k in ("input", "params", "tools", "instructions", "tool_choice",
                                              "segmentation", "say", "response_id")}
            self._forward(runtime.start_response(session, turn_options(spec, "realtime")))
        elif kind == "response.cancel":
            await runtime.cancel_response(session, event.get("response_id"), event.get("spoken_text"),
                                          discard=bool(event.get("discard")))
        elif kind == "tool.result":
            call_id = event.get("call_id")
            if not isinstance(call_id, str):
                raise BadRequest("tool.result needs call_id")
            remaining = await runtime.submit_tool_result(session, call_id, event.get("output", ""),
                                                         bool(event.get("is_error")))
            await self.send({"type": "tool.result.accepted", "call_id": call_id,
                             "pending_tool_calls": list(session.pending_tool_calls)})
            if remaining == 0 and event.get("continue", True):
                svc.auth.rate_limit(self.principal)
                self._forward(runtime.start_response(session, TurnOptions(params=event.get("params"),
                                                                          api="realtime")))
        elif kind == "session.update":
            spec = event.get("session")
            if not isinstance(spec, dict):
                raise BadRequest("session.update needs a session object")
            for key, value in (await session_fields(svc, session.tenant_id, spec, creating=False)).items():
                setattr(session, key, value)
            svc.sessions.save_state(session)
            await self.send({"type": "session.updated", "session": session.view()})
        elif kind == "session.get":
            await self.send({"type": "session.state", "session": session.view(include_messages=True)})
        elif kind == "session.end":
            await svc.sessions.end(session, str(event.get("reason") or "ended_by_client"))
            await self.send({"type": "session.ended", "session_id": session.id, "reason": session.end_reason,
                             "ended_at": iso(session.ended_at or utcnow())})
            await self.flush_out()
            await self.close(CLOSE_NORMAL, "session ended")
        elif kind == "conversation.item.create":
            if len(session.messages) + 1 > self.settings.max_messages_per_session:
                raise BadRequest("session is full", code="session_full")
            async with session.lock:
                item = session.add_openai(event.get("item"), self.settings.max_input_chars)
            svc.sessions.save_state(session)
            await self.send({"type": "conversation.item.created", "item": item.to_public()})
        elif kind == "conversation.truncate":
            spoken = event.get("spoken_text")
            if not isinstance(spoken, str):
                raise BadRequest("conversation.truncate needs spoken_text")
            await self.send({"type": "conversation.truncated", "truncated": await runtime.truncate(session, spoken)})
        elif kind == "conversation.clear":
            if session.active:
                session.active.cancel("conversation_cleared")
            async with session.lock:
                session.clear()
            svc.sessions.save_state(session)
            await self.send({"type": "conversation.cleared"})
        elif kind == "agent.handoff":
            agent_id = event.get("agent_id")
            if not isinstance(agent_id, str):
                raise BadRequest("agent.handoff needs agent_id")
            if self.principal.agent_id and agent_id != self.principal.agent_id:
                raise Forbidden("this token is restricted to another agent", code="agent_mismatch")
            if session.active is not None and not session.active.done:
                await runtime.cancel_response(session)
            reason = str(event.get("reason") or "")
            async with session.lock:
                previous, target = await runtime.handoff(session, agent_id, reason, "client")
            await self.send({"type": "agent.handoff", **runtime.handoff_event(
                session, previous, target, reason, "client", None).data})
        elif kind == "ping":
            await self.send({"type": "pong", "id": event.get("id")})
        else:
            raise BadRequest(f"unknown event type {kind!r}", code="unknown_event")

    def _forward(self, gen: Generation) -> None:
        self.owned.add(gen.id)

        async def pump() -> None:
            try:
                async for event in gen.events():
                    await self.send({"type": event.type, **event.data})
            except SlowConsumer:
                gen.cancel("client_too_slow")
                await self.close(CLOSE_TRY_AGAIN, "client too slow")
            finally:
                self.owned.discard(gen.id)

        task = asyncio.create_task(pump(), name=f"ws-forward-{gen.id}")
        self.forwarders.add(task)
        task.add_done_callback(self.forwarders.discard)


def _credential(ws: WebSocket) -> tuple[str | None, str | None]:
    """Return (credential, accepted subprotocol)."""
    offered = [p.strip() for p in ws.headers.get("sec-websocket-protocol", "").split(",") if p.strip()]
    subprotocol = PROTOCOL if PROTOCOL in offered else None
    auth = ws.headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip(), subprotocol
    if ws.headers.get("x-api-key"):
        return ws.headers["x-api-key"].strip(), subprotocol
    for proto in offered:
        if proto.startswith("bearer."):
            return proto[7:], subprotocol
    if ws.query_params.get("token"):
        return ws.query_params["token"], subprotocol
    return None, subprotocol


@router.websocket("/v1/realtime")
async def realtime(ws: WebSocket) -> None:
    services = ws.app.state.services
    credential, subprotocol = _credential(ws)
    if credential is None and services.settings.ws_query_api_key and ws.query_params.get("api_key"):
        credential = ws.query_params["api_key"]
    elif ws.query_params.get("token") and credential == ws.query_params.get("token") \
            and not credential.startswith("ey"):
        credential = None  # ?token= only carries realtime tokens, never API keys
    try:
        principal = await authenticate(ws, credential)
        principal.require("inference")
        principal.tenant  # noqa: B018 - admin keys must name a tenant
    except GatewayError:
        await ws.close(CLOSE_UNAUTHORIZED)  # before accept -> HTTP 403 on the handshake
        return
    if services.admission.draining:
        await ws.close(CLOSE_GOING_AWAY)
        return
    await RealtimeConnection(ws, services, principal, subprotocol).run()
