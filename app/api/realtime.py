"""WebSocket realtime protocol for voice agents: `WS /v1/realtime[?session_id=...]`.

One connection drives one session. JSON text frames, every frame has a `type`.

client -> server                              server -> client
  session.update        {session:{...}}         session.created / session.resumed / session.updated
  session.get                                   session.state
  conversation.item.create {item:{message}}     conversation.item.created
  conversation.truncate {spoken_text}           conversation.truncated
  conversation.clear                            conversation.cleared
  response.create  {input, params, tools,       response.created
                    segmentation, response_id}  response.text.delta / response.text.segment
                                                response.reasoning.delta
                                                response.tool_call.started / response.tool_call
                                                response.tool_result (server tools)
                                                response.done {status, text, usage, timings, ...}
  response.cancel  {response_id, spoken_text,   (response.done with status=cancelled)
                    discard}
  tool.result {call_id, output, is_error,       (continues automatically when nothing is pending)
               continue}
  ping {id}                                     pong {id}
                                                error {error:{type,code,message}, event_id}

Barge-in: on caller speech send `response.cancel` with `spoken_text` (what was played), then
`response.create` with the new transcript. Speculative turns: start `response.create` on an
interim transcript and send `response.cancel {discard:true}` if the final transcript differs.
"""
from __future__ import annotations

import asyncio
from typing import Any

import orjson
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from app.api.security import extract_key
from app.core.errors import BadRequest, GatewayError, NotFound
from app.core.generation import Generation
from app.observability.logging import get_logger, session_id_var
from app.observability.metrics import WS_CONNECTIONS
from app.sessions.store import Session

router = APIRouter(tags=["realtime"])
log = get_logger("app.realtime")

CLOSE_NORMAL, CLOSE_GOING_AWAY, CLOSE_TOO_BIG, CLOSE_TRY_AGAIN = 1000, 1001, 1009, 1013
CLOSE_UNAUTHORIZED, CLOSE_NOT_FOUND, CLOSE_IDLE = 4401, 4404, 4408


class SlowConsumer(Exception):
    pass


class RealtimeConnection:
    def __init__(self, ws: WebSocket, services: Any, principal: str) -> None:
        self.ws = ws
        self.services = services
        self.settings = services.settings
        self.principal = principal
        self.session: Session | None = None
        self.out: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=self.settings.ws_send_queue_size)
        self.forwarders: set[asyncio.Task] = set()
        self.owned: set[str] = set()  # response ids started by this connection
        self.closing = False
        self._drain_task: asyncio.Task | None = None

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
        for _ in range(40):  # give the sender up to ~2 s to flush
            if self.out.empty():
                break
            await asyncio.sleep(0.05)
        await self.close(CLOSE_GOING_AWAY, "server shutting down")

    # ------------------------------------------------------------------ send
    async def send(self, event: dict[str, Any]) -> None:
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
                await asyncio.wait_for(self.ws.send_text(orjson.dumps(event).decode()), timeout)
            except TimeoutError:
                log.warning("websocket client not reading; closing")
                await self.close(CLOSE_TRY_AGAIN, "client too slow")
                return
            except (WebSocketDisconnect, RuntimeError):
                return

    async def error(self, exc: GatewayError, event_id: Any = None) -> None:
        payload: dict[str, Any] = {"type": "error", "error": exc.to_dict()["error"]}
        if event_id is not None:
            payload["event_id"] = event_id
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
    async def run(self) -> None:
        services = self.services
        session_id = self.ws.query_params.get("session_id")
        if session_id:
            try:
                self.session = services.store.get(session_id)
            except NotFound:
                await self.ws.accept()
                await self.ws.close(CLOSE_NOT_FOUND, "session not found")
                return
            created = "session.resumed"
        else:
            try:
                self.session = services.conversation.create_session({})
            except GatewayError as exc:  # e.g. MAX_SESSIONS reached
                await self.ws.accept()
                await self.ws.close(CLOSE_TRY_AGAIN, exc.message[:120])
                return
            created = "session.created"
        await self.ws.accept()
        session_id_var.set(self.session.id)
        self.session.attachments += 1
        services.ws_connections += 1
        services.connections.add(self)
        WS_CONNECTIONS.inc()
        sender = asyncio.create_task(self._sender(), name="ws-sender")
        log.info("websocket connected", extra={"principal": self.principal, "event": created})
        try:
            await self.send({"type": created, "session": self.session.full_view(include_messages=False)})
            await self._receive_loop()
        except (WebSocketDisconnect, SlowConsumer):
            pass
        finally:
            session = self.session
            if session.active is not None and session.active.id in self.owned:
                session.active.cancel("client_disconnect")  # nobody left to hear it
            session.attachments -= 1
            session.touch()
            for task in self.forwarders:
                task.cancel()
            if self.out.full():
                sender.cancel()
            else:
                self.out.put_nowait(None)  # let the sender flush queued events, then stop
            await asyncio.gather(sender, *self.forwarders, return_exceptions=True)
            await self.close(CLOSE_NORMAL)
            services.ws_connections -= 1
            services.connections.discard(self)
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
                await self.error(GatewayError("server is shutting down", code="shutting_down"))
                await self.close(CLOSE_GOING_AWAY, "server shutting down")
                return
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
        conv = self.services.conversation
        session.touch()

        if kind == "response.create":
            self.services.auth.rate_limit(self.principal)
            overrides = {"params": event.get("params"), "tools": event.get("tools"),
                         "segmentation": event.get("segmentation")}
            self._forward(conv.start_response(session, event.get("input"), overrides, event.get("response_id")))
        elif kind == "response.cancel":
            await conv.cancel_response(session, event.get("response_id"), event.get("spoken_text"),
                                       discard=bool(event.get("discard")))
        elif kind == "tool.result":
            call_id = event.get("call_id")
            if not isinstance(call_id, str):
                raise BadRequest("tool.result needs call_id")
            remaining = await conv.submit_tool_result(session, call_id, event.get("output", ""),
                                                      bool(event.get("is_error")))
            if remaining == 0 and event.get("continue", True):
                self.services.auth.rate_limit(self.principal)
                self._forward(conv.start_response(session, None, {"params": event.get("params")}))
        elif kind == "session.update":
            spec = event.get("session")
            if not isinstance(spec, dict):
                raise BadRequest("session.update needs a session object")
            conv.update_session(session, spec)
            await self.send({"type": "session.updated", "session": session.full_view(include_messages=False)})
        elif kind == "session.get":
            await self.send({"type": "session.state", "session": session.full_view(include_messages=True)})
        elif kind == "conversation.item.create":
            self.services.store.check_capacity(session)
            async with session.lock:
                item = session.add_openai(event.get("item"), self.settings.max_input_chars)
            await self.send({"type": "conversation.item.created", "item": item.to_public()})
        elif kind == "conversation.truncate":
            spoken = event.get("spoken_text")
            if not isinstance(spoken, str):
                raise BadRequest("conversation.truncate needs spoken_text")
            truncated = await conv.truncate(session, spoken)
            await self.send({"type": "conversation.truncated", "truncated": truncated})
        elif kind == "conversation.clear":
            if session.active:
                session.active.cancel("conversation_cleared")
            async with session.lock:
                session.messages.clear()
                session.summary = ""
                session.pending_tool_calls.clear()
            await self.send({"type": "conversation.cleared"})
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


@router.websocket("/v1/realtime")
async def realtime(ws: WebSocket) -> None:
    services = ws.app.state.services
    client = ws.client.host if ws.client else None
    try:
        principal = services.auth.authenticate(extract_key(ws.headers, ws.query_params), client)
    except GatewayError:
        await ws.close(CLOSE_UNAUTHORIZED)  # before accept -> HTTP 403 on the handshake
        return
    if services.admission.draining:
        await ws.close(CLOSE_GOING_AWAY)
        return
    await RealtimeConnection(ws, services, principal).run()
