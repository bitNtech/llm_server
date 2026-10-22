"""Unit tests of pure components (no network, no Firestore)."""
from __future__ import annotations

import asyncio
import time

import orjson
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.core.errors import BadRequest, RateLimited
from app.engine.segmenter import SentenceSegmenter
from app.engine.toolcalls import ToolCallAccumulator
from app.providers.signatures import PLACEHOLDER, SignatureCache, call_key, ensure_signatures
from app.providers.vertex import VertexProvider, _error_message, _parse_line
from app.security.crypto import SecretBox, sign_payload, verify_signature
from app.security.netguard import NetGuard
from app.security.ratelimit import RateLimiter
from app.services.workflows import compare, render, validate_workflow
from app.tools.base import ArgumentValidator, check_name, check_schema


# ------------------------------------------------------------------ config
def test_production_requires_secrets() -> None:
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, app_env="production", gcp_project_id="p")
    assert "ADMIN_API_KEYS" in str(exc.value) and "SECRETS_ENCRYPTION_KEY" in str(exc.value)


def test_production_rejects_weak_keys() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="production", gcp_project_id="p", admin_api_keys="change_me",
                 secrets_encryption_key="x", realtime_token_secret="y" * 40)


def test_model_and_location_validation() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, vertex_model="gemini-3.5-flash")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, vertex_locations="asia south1")
    s = Settings(_env_file=None, vertex_fallback_models="google/gemini-3.1-flash-lite,google/gemini-3.5-flash")
    assert s.fallback_model_list == ["google/gemini-3.1-flash-lite"]  # the default model is not its own fallback


# --------------------------------------------------------------- segmenter
def test_segmenter_sentences_and_abbreviations() -> None:
    seg = SentenceSegmenter(min_chars=8, max_chars=60)
    out = []
    for delta in ["Hello Dr. Rao, your visit ", "is confirmed. See you ", "tomorrow at 10! Bye"]:
        out += seg.push(delta)
    assert out == ["Hello Dr. Rao, your visit is confirmed.", "See you tomorrow at 10!"]
    assert seg.flush() == "Bye"


def test_segmenter_hard_break_without_punctuation() -> None:
    seg = SentenceSegmenter(min_chars=8, max_chars=30)
    out = seg.push("word " * 20)
    assert out and all(len(s) <= 30 for s in out)


def test_segmenter_devanagari_danda() -> None:
    seg = SentenceSegmenter(min_chars=4)
    assert seg.push("नमस्ते, आप कैसे हैं। ठीक") == ["नमस्ते, आप कैसे हैं।"]


# --------------------------------------------------------------- tool calls
def test_accumulator_whole_and_fragmented_calls() -> None:
    acc = ToolCallAccumulator()
    started = acc.add([{"index": 0, "id": "call_1", "function": {"name": "a", "arguments": '{"x":'},
                        "extra_content": {"google": {"thought_signature": "sig"}}}])
    assert [c.name for c in started] == ["a"]
    acc.add([{"index": 0, "function": {"arguments": "1}"}}, {"index": 1, "function": {"name": "b"}}])
    calls = acc.finish()
    assert [(c.name, c.arguments) for c in calls] == [("a", '{"x":1}'), ("b", "")]
    assert calls[0].to_message()["extra_content"]["google"]["thought_signature"] == "sig"
    assert calls[1].id.startswith("call_") and calls[1].to_message()["function"]["arguments"] == "{}"


# --------------------------------------------------------------- signatures
def test_signatures_reattached_and_placeholder() -> None:
    cache = SignatureCache()
    cache.put(call_key("t1", "c1", "f", "{}"), {"google": {"thought_signature": "real"}})
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "assistant", "tool_calls": [{"id": "c9", "function": {"name": "g", "arguments": "{}"}}]},
    ]
    ensure_signatures(messages, "t1", cache)
    assert messages[0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "real"
    assert messages[1]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == PLACEHOLDER
    # another tenant never sees t1's signatures
    other = [{"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]}]
    ensure_signatures(other, "t2", cache)
    assert other[0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == PLACEHOLDER


def test_for_model_swaps_foreign_signatures() -> None:
    payload = {"messages": [
        {"role": "assistant", "_model": "google/a", "extra_content": {"google": {"thought_signature": "t"}},
         "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "{}"},
                         "extra_content": {"google": {"thought_signature": "real"}}}]}]}
    same = orjson.loads(VertexProvider.for_model(payload, "google/a"))
    assert same["messages"][0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "real"
    assert "_model" not in same["messages"][0]
    other = orjson.loads(VertexProvider.for_model(payload, "google/b"))
    assert other["model"] == "google/b"
    assert other["messages"][0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == PLACEHOLDER
    assert "extra_content" not in other["messages"][0]


def test_vertex_error_and_line_parsing() -> None:
    assert _error_message(b'[{"error": {"code": 429, "message": "busy", "status": "RESOURCE_EXHAUSTED"}}]') == (
        "busy", "RESOURCE_EXHAUSTED")
    assert _parse_line(": keepalive") is None
    line, chunk = _parse_line('data: {"choices": []}')
    assert chunk == {"choices": []}
    with pytest.raises(Exception, match="boom"):
        _parse_line('data: {"error": {"message": "boom"}}')


# --------------------------------------------------------------- security
def test_secretbox_roundtrip_and_rotation() -> None:
    from cryptography.fernet import Fernet

    old, new = Fernet.generate_key().decode(), Fernet.generate_key().decode()
    token = SecretBox([old], production=False).encrypt({"Authorization": "Bearer x"})
    assert SecretBox([new, old], production=False).decrypt(token) == {"Authorization": "Bearer x"}
    with pytest.raises(ValueError):
        SecretBox([], production=True)


def test_signature_verify() -> None:
    header = sign_payload("secret", b'{"a":1}')
    assert verify_signature("secret", b'{"a":1}', header)
    assert not verify_signature("other", b'{"a":1}', header)
    assert not verify_signature("secret", b'{"a":2}', header)
    stale = sign_payload("secret", b"x", timestamp=int(time.time()) - 3600)
    assert not verify_signature("secret", b"x", stale)


def test_netguard_blocks_private_targets() -> None:
    guard = NetGuard(allow_private=False, allowed_hosts=["internal.example"], require_https=True)

    async def check(url: str) -> None:
        await guard.check(url)

    for url in ("https://127.0.0.1/x", "https://10.1.2.3/", "https://169.254.169.254/latest", "https://[::1]/",
                "https://metadata.google.internal/", "http://example.com/", "ftp://example.com", "https://u:p@x.com"):
        with pytest.raises(BadRequest):
            asyncio.run(check(url))
    asyncio.run(check("https://8.8.8.8/ok"))
    asyncio.run(guard.check("http://internal.example/hook"))  # allowlisted


def test_rate_limiter() -> None:
    limiter = RateLimiter()
    for _ in range(3):
        limiter.check("k", 3)
    with pytest.raises(RateLimited) as exc:
        limiter.check("k", 3)
    assert exc.value.retry_after and exc.value.retry_after > 0


# ------------------------------------------------------------------ tools
def test_tool_name_and_schema_checks() -> None:
    check_name("crm.lookup-patient_v2")
    for bad in ("1abc", "has space", "x" * 65):
        with pytest.raises(BadRequest):
            check_name(bad)
    with pytest.raises(BadRequest):
        check_schema({"type": "array"})
    with pytest.raises(BadRequest):
        check_schema({"type": "object", "properties": {"x": {"type": "nope"}}})
    assert check_schema(None)["type"] == "object"


def test_argument_validator() -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}, "required": ["n"]}
    v = ArgumentValidator()
    assert v.validate(schema, {"n": 2}) is None
    assert "required" in v.validate(schema, {})
    assert "minimum" in v.validate(schema, {"n": 0}) or "less than" in v.validate(schema, {"n": 0})


# --------------------------------------------------------------- workflows
def test_template_rendering_keeps_types() -> None:
    ctx = {"input": {"n": 3, "tags": ["a", "b"], "who": {"name": "Asha"}}, "steps": {"s": {"output": {"ok": True}}}}
    assert render("{{input.n}}", ctx) == 3
    assert render("{{ input.tags[1] }}", ctx) == "b"
    assert render("Hi {{input.who.name}} {{input.tags}}", ctx) == 'Hi Asha ["a","b"]'
    assert render({"x": ["{{steps.s.output.ok}}"]}, ctx) == {"x": [True]}
    assert render("{{missing.path}}", ctx) is None


def test_compare_operators() -> None:
    assert compare(3, "gt", 2) and compare("5", "lte", 5) and compare("abc", "contains", "b")
    assert compare("a", "in", ["a"]) and compare(None, "not_exists", None) and compare(0, "falsy", None)
    assert not compare("x", "gt", 1)  # incomparable values are false, not errors


def test_workflow_validation() -> None:
    validate_workflow({"steps": [{"id": "a", "type": "set", "value": 1, "next": "end"}]})
    bad = [
        {"steps": []},
        {"steps": [{"id": "A", "type": "set", "value": 1}]},
        {"steps": [{"id": "a", "type": "nope"}]},
        {"steps": [{"id": "a", "type": "set", "value": 1, "next": "zzz"}]},
        {"steps": [{"id": "a", "type": "condition", "if": {"op": "eq"}}]},
        {"steps": [{"id": "a", "type": "set", "value": 1}, {"id": "a", "type": "set", "value": 2}]},
    ]
    for spec in bad:
        with pytest.raises(BadRequest):
            validate_workflow(spec)


# ------------------------------------------------------------- review fixes
def test_unanswered_tool_calls() -> None:
    from app.engine.sessions import LiveSession

    s = LiveSession(tenant_id="t", id="s")
    s.add("user", "hi")
    s.add("assistant", None, tool_calls=[{"id": "a", "function": {"name": "x"}}, {"id": "b", "function": {"name": "y"}}])
    s.add("tool", "ok", tool_call_id="a", name="x")
    assert s.unanswered_tool_calls() == {"b": "y"}
    s.add("tool", "ok", tool_call_id="b", name="y")
    assert s.unanswered_tool_calls() == {}
    with pytest.raises(BadRequest):
        s.add_openai({"role": "user", "content": [{"type": "text", "text": 123}]}, 1000)


def test_idempotent_responses_are_stored_encrypted() -> None:
    from starlette.requests import Request
    from starlette.responses import Response

    from app.services.idempotency import Idempotency

    class Repo:
        record: dict = {}

        async def begin(self, tenant_id, key_hash, request_hash):
            if self.record:
                return self.record
            self.record = {"request_hash": request_hash, "status": "running"}
            return None

        async def complete(self, tenant_id, key_hash, status, body_enc, media_type):
            self.record.update(status="completed", response_status=status, response_body_enc=body_enc,
                               media_type=media_type)

    repo = Repo()
    idem = Idempotency(repo, SecretBox([], production=False))
    request = Request({"type": "http", "method": "POST", "path": "/v1/api-keys",
                       "headers": [(b"idempotency-key", b"k1")]})

    async def handler() -> Response:
        return Response(b'{"key":"aica_secret"}', status_code=201, media_type="application/json")

    first = asyncio.run(idem.run(request, "t", b"{}", handler))
    assert first.status_code == 201
    assert "aica_secret" not in orjson.dumps(repo.record).decode()
    replay = asyncio.run(idem.run(request, "t", b"{}", handler))
    assert replay.body == b'{"key":"aica_secret"}' and replay.headers["Idempotent-Replayed"] == "true"


def test_writer_retries_transient_errors_until_they_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.errors import StoreUnavailable
    from app.store.writer import Writer

    calls = []

    async def flaky() -> None:
        calls.append(1)
        if len(calls) < 8:  # more failures than the old fixed attempt limit
            raise StoreUnavailable("firestore down")

    async def main() -> bool:
        writer = Writer()
        writer.submit("k", flaky, "test")
        return await writer.flush("k", timeout=5)

    monkeypatch.setattr("app.store.writer._backoff", lambda attempt: 0)

    assert asyncio.run(main()) and len(calls) == 8


def test_peer_check_blocks_rebound_connections() -> None:
    class Stream:
        def __init__(self, ip: str) -> None:
            self.ip = ip

        def get_extra_info(self, name: str):
            return (self.ip, 443) if name == "server_addr" else None

    class Response:
        def __init__(self, ip: str) -> None:
            self.extensions = {"network_stream": Stream(ip)}
            self.request = type("R", (), {"url": "https://tool.example.com/x"})()

    hook = NetGuard(False, [], True).response_hook("tool URL")
    asyncio.run(hook(Response("34.120.1.1")))
    for ip in ("169.254.169.254", "10.0.0.5", "::ffff:127.0.0.1"):
        with pytest.raises(BadRequest):
            asyncio.run(hook(Response(ip)))
    asyncio.run(NetGuard(True, [], True).response_hook("tool URL")(Response("10.0.0.5")))
