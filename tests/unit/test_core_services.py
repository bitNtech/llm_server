from __future__ import annotations

import asyncio
import time

import pytest

from app.api.security import Authenticator
from app.config import Settings
from app.core.admission import AdmissionController
from app.core.errors import BadRequest, Conflict, NotFound, Overloaded, RateLimited, ShuttingDown, Unauthorized
from app.core.generation import Generation, GenerationRegistry
from app.core.params import apply_defaults, validate_session_params
from app.sessions.store import SessionStore
from app.tools.registry import ServerTool, ToolRegistry


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


# --------------------------------------------------------------- admission
async def test_admission_limits_and_queue():
    adm = AdmissionController(max_active=1, max_queue=1, queue_timeout_s=0.2)
    async with adm.slot():
        waiter = asyncio.create_task(adm.slot().__aenter__())
        await asyncio.sleep(0.01)
        assert adm.waiting == 1
        with pytest.raises(Overloaded):  # queue full
            async with adm.slot():
                pass
        with pytest.raises(Overloaded):  # queue timeout
            await waiter
    async with adm.slot() as waited:
        assert waited < 0.05
    adm.draining = True
    with pytest.raises(ShuttingDown):
        async with adm.slot():
            pass


async def test_admission_queued_request_gets_slot():
    adm = AdmissionController(max_active=1, max_queue=4, queue_timeout_s=2)
    order = []

    async def job(i: int):
        async with adm.slot():
            order.append(i)
            await asyncio.sleep(0.02)

    await asyncio.gather(*(job(i) for i in range(3)))
    assert order == [0, 1, 2] and adm.active == 0


# ---------------------------------------------------------------- registry
async def test_generation_cancel_and_drain():
    reg = GenerationRegistry()
    gen = Generation("g1", "test")

    async def producer():
        gen.emit("a")
        await asyncio.sleep(10)

    gen.start(producer(), reg)
    assert reg.get("g1") is gen
    events = gen.events()
    assert await anext(events) == "a"
    assert reg.cancel("g1", "test")
    assert await gen.wait(1)
    assert gen.cancel_reason == "test" and reg.get("g1") is None
    assert [e async for e in events] == []

    slow = Generation("g2", "test")
    slow.start(asyncio.sleep(10), reg)
    assert await reg.drain(timeout=0.05) == 1 and slow.done


# ------------------------------------------------------------------ params
def test_apply_defaults_and_limits():
    s = settings(default_temperature=0.3, default_stop="</s>,END", max_tokens_limit=100, default_max_tokens=50)
    body = apply_defaults(s, {"messages": [], "temperature": 0.9, "priority": 5})
    assert body["max_tokens"] == 50 and body["temperature"] == 0.9 and body["stop"] == ["</s>", "END"]
    assert "priority" not in body  # FCFS engine would reject it
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert apply_defaults(s, {"max_completion_tokens": 20})["max_tokens"] == 20
    with pytest.raises(BadRequest):
        apply_defaults(s, {"max_tokens": 101})
    with pytest.raises(BadRequest):
        apply_defaults(s, {"n": 2})


def test_session_params_whitelist():
    assert validate_session_params({"temperature": 0.2, "max_tokens": 64}) == {"temperature": 0.2, "max_tokens": 64}
    with pytest.raises(BadRequest):
        validate_session_params({"model": "x"})
    with pytest.raises(BadRequest):
        validate_session_params({"max_tokens": 0})


# ------------------------------------------------------------------ store
async def test_store_lifecycle_eviction_and_ttl():
    store = SessionStore(max_sessions=2, ttl_s=0.05, max_messages=4, chars_per_token=3)
    a = store.create(session_id="call-1")
    with pytest.raises(Conflict):
        store.create(session_id="call-1")
    with pytest.raises(BadRequest):
        store.create(session_id="bad id!")
    b = store.create()
    b.attachments = 1  # a live WebSocket protects it from eviction
    time.sleep(0.01)
    store.create()  # evicts the least recently used evictable session (a)
    with pytest.raises(NotFound):
        store.get(a.id)
    time.sleep(0.06)
    assert store.sweep() == 1  # the unattached one expired; b is attached
    assert store.get(b.id) is b
    b.attachments = 0
    b.add("user", "1"), b.add("assistant", "2"), b.add("user", "3")
    with pytest.raises(BadRequest):
        store.check_capacity(b, 2)
    await store.delete(b.id)
    with pytest.raises(NotFound):
        await store.delete(b.id)


async def test_store_refuses_when_everything_is_busy():
    store = SessionStore(max_sessions=1, ttl_s=60, max_messages=10, chars_per_token=3)
    store.create().attachments = 1
    with pytest.raises(Overloaded):
        store.create()


def test_session_message_validation_and_truncation():
    store = SessionStore(max_sessions=5, ttl_s=60, max_messages=50, chars_per_token=3)
    s = store.create()
    with pytest.raises(BadRequest):
        s.add_openai({"role": "system", "content": "x"}, 100)
    with pytest.raises(BadRequest):
        s.add_openai({"role": "tool", "content": "x"}, 100)
    with pytest.raises(BadRequest):
        s.add_openai({"role": "user", "content": "x" * 101}, 100)
    s.add_openai({"role": "user", "content": [{"type": "text", "text": "hi"}]}, 100)
    s.add("assistant", "Your appointment is at five. Shall I also book")
    assert s.truncate_last_assistant("Your appointment is at five.")
    assert s.messages[-1].content == "Your appointment is at five." and s.messages[-1].interrupted
    s.add("user", "next")
    assert not s.truncate_last_assistant("x")  # never edits across a newer user turn


# ------------------------------------------------------------------- tools
async def test_tool_registry_execution():
    reg = ToolRegistry(default_timeout_s=0.1)

    async def slow():
        await asyncio.sleep(1)

    reg.register(ServerTool("add", "", {"type": "object", "required": ["a", "b"]}, lambda a, b: {"sum": a + b}))
    reg.register(ServerTool("slow", "", {"type": "object"}, slow))
    reg.register(ServerTool("boom", "", {"type": "object"}, lambda: 1 / 0))
    assert await reg.execute("add", '{"a": 1, "b": 2}') == ('{"sum":3}', False)
    assert (await reg.execute("add", '{"a": 1}'))[1] is True
    assert (await reg.execute("add", "not json"))[1] is True
    assert (await reg.execute("add", '{"a": 1, "b": 2, "c": 3}'))[1] is True
    assert "timed out" in (await reg.execute("slow", "{}"))[0]
    assert (await reg.execute("boom", "{}")) == ("tool boom failed: ZeroDivisionError", True)
    assert (await reg.execute("nope", "{}"))[1] is True
    with pytest.raises(ValueError):
        reg.register(ServerTool("add", "", {}, lambda: None))


def test_builtin_tools_load():
    reg = ToolRegistry()
    reg.load_modules(["app.tools.builtin"])
    assert reg.names() == ["get_current_time"]
    assert reg.schemas(["get_current_time"])[0]["function"]["name"] == "get_current_time"


# ---------------------------------------------------------------- security
def test_authenticator_and_rate_limit():
    open_auth = Authenticator(settings())
    assert open_auth.authenticate(None, "1.2.3.4") == "ip:1.2.3.4"

    auth = Authenticator(settings(api_keys="k1,k2", rate_limit_per_minute=2))
    with pytest.raises(Unauthorized):
        auth.authenticate(None, None)
    with pytest.raises(Unauthorized):
        auth.authenticate("wrong", None)
    principal = auth.authenticate("k2", None)
    assert principal.startswith("key_") and "k2" not in principal
    auth.rate_limit(principal)
    auth.rate_limit(principal)
    with pytest.raises(RateLimited) as exc:
        auth.rate_limit(principal)
    assert exc.value.retry_after and exc.value.retry_after > 0
