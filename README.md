# LLM Inference Server — vLLM + FastAPI for real-time voice agents

A self-hosted LLM endpoint for a voice agent, serving **Gemma 4 26B-A4B** (FP8 + MTP speculative decoding) with a latency-first profile. **vLLM** does the inference (continuous batching, paged KV cache, prefix caching, native tool and reasoning parsers). A **FastAPI gateway** in front of it adds the parts a phone call needs: WebSocket streaming, sessions and context management, barge-in cancellation, a tool loop, auth, admission control and observability. One `run.sh` manages the whole lifecycle on a laptop or on a GPU EC2 instance.

```
voice pipeline ──WS /v1/realtime──►┐
OpenAI SDKs ──/v1/chat/completions─►  Gateway :8000  ──loopback HTTP/SSE──►  vLLM OpenAI server 127.0.0.1:8001
REST clients ──/v1/sessions/…──────►┘  sessions · context · tools · auth · metrics      (CUDA or CPU build, picked by run.sh)
```

## Quick start

```bash
./run.sh                 # validate → install (first run only) → start vLLM → start gateway; Ctrl-C stops both gracefully
```

On the first run `run.sh` copies `.env.example` to `.env`, bootstraps `uv` and Python 3.12, creates `.venv` (gateway) and `.venv-vllm` (engine), installs the vLLM build that matches the machine, downloads the model, and starts everything. Later runs reuse what is already installed.

```bash
./run.sh check           # what was detected (CPU flags, RAM, NVIDIA driver/GPUs) and the exact `vllm serve` command
ENGINE_MODE=mock ./run.sh                                        # no model: deterministic mock engine (laptop default in .env)
ENGINE_MODEL=Qwen/Qwen3-0.6B ENGINE_REVISION= ENGINE_SERVED_MODEL_NAME= ENGINE_TOOL_CALL_PARSER=hermes ENGINE_REASONING_PARSER=qwen3 ./run.sh
                                                                 # small real model on a laptop CPU
./run.sh test            # unit + integration tests (no GPU needed)
.venv/bin/python scripts/smoke.py --tools                        # end-to-end check of a running server
.venv/bin/python scripts/bench.py --concurrency 1,8,32           # TTFT / ITL under load (capacity planning)
```

| `run.sh` command | What it does |
|---|---|
| `start` (default) | Full lifecycle in the foreground. SIGINT/SIGTERM drain the gateway, then stop vLLM. If either process dies, the other is stopped and the exit code is non-zero (for systemd restarts). |
| `install` | Virtualenvs, dependencies, vLLM (CUDA 13.0 / CUDA 12.9 / CPU wheel chosen from the hardware) and the model download. Idempotent. |
| `check` / `doctor` | Hardware and configuration report. Fails on invalid configuration. |
| `test [pytest args]` | Test suite. |
| `stop` / `status` | Control a running instance through `run/llm-server.pid`. |
| `engine-cmd` | Print the `vllm serve …` command line. |

## Features

- **Native vLLM:** the official OpenAI-compatible server with prefix caching, chunked prefill, tool parsers (`gemma4`, `qwen3_coder`, `hermes`, …), MTP speculative decoding, reasoning parsers, `/tokenize`, Prometheus metrics, request-id headers and abort-on-disconnect.
- **APIs:**
  - OpenAI-compatible `/v1/chat/completions`, so any OpenAI SDK works.
  - Stateful `/v1/sessions` REST and SSE API.
  - Realtime WebSocket `/v1/realtime` built for voice (sentence segments for TTS, tool-call "started" events for fillers).
- **Barge-in:** cancel from any path (WebSocket, REST, request id, client disconnect, timeout, shutdown) and vLLM aborts at once. The stored reply is truncated to what the caller actually heard. Speculative turns can be discarded.
- **Tools:** client-side tools (`requires_action` → `tool.result`) and server-side Python tools (`@tool`, run in-process with timeouts), in a multi-round tool loop.
- **Context management:** prompt size is estimated and calibrated from real usage, with no tokenizer call on the hot path. Trimming drops whole turns and keeps prefix-cache hits. Optional background summarization runs at low scheduling priority.
- **Production basics:** API keys, rate limits, body/frame/message limits, admission control (fast 429s rather than slow calls for everyone), graceful drain, structured JSON logs with request/session ids, Prometheus metrics for TTFT, inter-token latency, cancellations, tokens and cache hits, and health/readiness probes.
- **One config surface:** every setting comes from `.env` or the environment, validated by one pydantic model shared by the gateway and `run.sh`.

## Configuration

Everything is in [`.env.example`](.env.example), with comments. The most important settings:

| Variable | Default | Meaning |
|---|---|---|
| `APP_ENV` | `development` | `production` requires `API_KEYS`, logs JSON and disables `/docs` |
| `API_KEYS` | (empty) | comma-separated client keys |
| `ENGINE_MODE` | `managed` | `managed` (run.sh starts vLLM), `external` (`ENGINE_BASE_URL`), `mock` |
| `ENGINE_MODEL` | `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` (pinned revision) | Gemma 4 26B-A4B, FP8 W8A8 |
| `ENGINE_SPECULATIVE_CONFIG` | Gemma 4 MTP drafter, 4 tokens | speculative decoding (GPU only) |
| `ENGINE_TOOL_CALL_PARSER` / `ENGINE_REASONING_PARSER` | `gemma4` / `gemma4` | must match the model family |
| `ENGINE_MAX_NUM_SEQS` / `MAX_CONCURRENT_GENERATIONS` | `32` / `32` | small batches keep inter-token latency low; size with `scripts/bench.py` |
| `ENGINE_ATTENTION_BACKEND` | `auto` | Triton on H100/H200 with vLLM 0.30/0.31 (FA4 decode regression workaround) |
| `ENGINE_DEVICE` | `auto` | `cuda` if an NVIDIA GPU is visible, else `cpu` |
| `ENGINE_MAX_MODEL_LEN` | `8192` | shorter context = faster decode attention |
| `ENGINE_GPU_MEMORY_UTILIZATION` | `0.90` | GPU memory share for vLLM |
| `DEFAULT_MAX_TOKENS` | `200` | spoken replies are short |
| `DEFAULT_CHAT_TEMPLATE_KWARGS` | `{"enable_thinking": false}` | no silent reasoning before speaking |
| `CONTEXT_STRATEGY` | `truncate` | or `summarize` |
| `ENGINE_FIRST_TOKEN_TIMEOUT_S` / `QUEUE_TIMEOUT_S` / `GENERATION_TIMEOUT_S` | `15` / `3` / `120` | fail fast so the agent can play a fallback |

## Documentation

- [docs/API.md](docs/API.md): REST, SSE and WebSocket protocol reference, plus the recommended voice-agent flow.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): design decisions, request flows, latency design, scaling, security.
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md): prerequisites, laptop vs EC2, Gemma 4 instance sizing and latency profile, systemd, TLS/ALB, monitoring, upgrades, troubleshooting.
- [deploy/production.env.example](deploy/production.env.example): the production overrides file (`/etc/llm-server/llm-server.env`).

## Layout

```
run.sh                      lifecycle manager (validate, install, start, supervise, stop)
app/config.py               all settings (env-driven, validated)
app/hardware.py, plan.py    hardware detection → vLLM build + `vllm serve` argv
app/cli.py                  helper commands used by run.sh
app/main.py, __main__.py    FastAPI app, lifecycle, graceful drain; uvicorn entry point
app/api/                    routes: health, metrics, OpenAI-compatible, sessions, realtime WS; auth, middleware
app/core/                   admission, generation handles, conversation/tool loop, OpenAI proxy, segmenter, errors
app/engine/                 vLLM client + health monitor
app/sessions/               session store, context-window management
app/tools/                  server-side tool registry + built-ins
app/mock_engine.py          vLLM protocol stand-in for tests / ENGINE_MODE=mock
scripts/                    smoke test, benchmark, model prefetch
deploy/                     systemd unit, nginx config
tests/                      unit + integration (live servers, WebSockets, run.sh lifecycle)
```

## Adding server-side tools

```python
# mycompany/tools.py   (then TOOLS_MODULES=app.tools.builtin,mycompany.tools)
from app.tools import tool

@tool(description="Look up a patient's next appointment",
      parameters={"type": "object", "properties": {"phone": {"type": "string"}}, "required": ["phone"]},
      timeout_s=2.0)
async def next_appointment(phone: str) -> dict:
    ...
```

Sessions opt in with `"server_tools": ["next_appointment"]`. The model is never offered tools the session did not enable.
