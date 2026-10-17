# Deployment guide

The same repository and the same `./run.sh` run on a developer laptop and on a GPU EC2 instance. Nothing is hardcoded for one machine. `run.sh` detects the hardware and picks the vLLM build, while `.env` holds the deployment-specific settings.

## 1. System prerequisites

| | Local development | EC2 production |
|---|---|---|
| OS | Linux x86_64/arm64 (macOS: gateway and mock engine only) | Ubuntu 22.04/24.04 or Amazon Linux 2023 |
| Packages | `bash` ≥ 4, `curl`, `util-linux` (`setsid`), **a C/C++ compiler** (`build-essential` / `dnf groupinstall "Development Tools"`). vLLM JIT-compiles kernels at startup: torch inductor on CPU, Triton launchers on GPU | same (the DLAMI already has gcc) |
| Python | none required: `run.sh` bootstraps [uv](https://docs.astral.sh/uv/), which downloads a managed Python 3.12 that includes `Python.h` (needed by the JIT compilers). Without internet access to astral.sh, provide Python 3.11–3.14 with `venv` (`apt-get install python3.12-venv`) and set `UV_AUTO_INSTALL=0`, and install the matching `python3.X-dev` package | same |
| GPU | optional: no NVIDIA GPU → vLLM CPU backend (x86 AVX2/AVX-512 or arm64) | NVIDIA driver **≥ 580** (CUDA 13.0 build) or **≥ 525** (CUDA 12.9 build). No CUDA toolkit is needed: the wheels bring their own CUDA runtime |
| Disk | ~10 GB for virtualenvs (+ model size if you run one locally) | 100 GB+: ~15 GB for venvs, ~30 GB for Gemma 4 FP8 + drafter, plus room for an upgrade side by side |
| RAM | 8 GB minimum | ≥ 64 GiB host RAM for Gemma 4 26B (weights stream through the page cache while loading) |

`./run.sh check` prints exactly what was detected and what will be run.

## 2. Local development (laptop)

Gemma 4 26B-A4B needs a ≥ 40 GB NVIDIA GPU and its FP8 checkpoint does not run on laptop CPUs. Develop against the mock engine, a small real model on the CPU, or the EC2 engine:

```bash
ENGINE_MODE=mock ./run.sh                        # no model at all: deterministic replies, full API surface
ENGINE_MODEL=Qwen/Qwen3-0.6B ENGINE_REVISION= ENGINE_SERVED_MODEL_NAME= ENGINE_TOOL_CALL_PARSER=hermes \
  ENGINE_REASONING_PARSER=qwen3 ./run.sh         # small real model on the CPU (~20 tok/s on a Ryzen 7 5800H)
ENGINE_MODE=external ENGINE_BASE_URL=http://gpu-box:8001 ENGINE_API_KEY=... ./run.sh   # the real Gemma engine, e.g. over an SSH tunnel
./run.sh test                                    # full test suite, no GPU needed
```

AMD Radeon / Intel iGPUs are listed by `./run.sh check` but are not used. vLLM's ROCm build targets data-center Instinct GPUs. Consumer RDNA2 parts such as the RX 6500M are not supported.

### Local vs EC2: what changes automatically

| Aspect | CPU (laptop) | NVIDIA GPU (EC2) |
|---|---|---|
| vLLM build | `+cpu` wheel from the vLLM GitHub release (manylinux tag matched to glibc) | PyPI `vllm` (CUDA 13.0) if driver ≥ 580 and compute capability ≥ 7.5, else the `+cu129` wheel |
| dtype | `bfloat16` | checkpoint dtype; `float16` on pre-Ampere GPUs (T4, V100) |
| KV cache | `--kv-cache-memory-bytes` ≈ 15% of RAM (`ENGINE_KV_CACHE_MEMORY_GIB` to override) | `--gpu-memory-utilization 0.90` |
| Threads | `VLLM_CPU_OMP_THREADS_BIND=auto`, `ENGINE_CPU_RESERVED_CORES` kept free for the gateway | n/a |
| Preloads | Intel OpenMP (`libiomp5.so` from the venv) and tcmalloc when installed (`apt-get install libtcmalloc-minimal4`) | n/a |
| Parallelism | single process | `ENGINE_TENSOR_PARALLEL_SIZE=auto` uses every visible GPU |
| Startup | seconds to minutes | 1–5 min (weights, `torch.compile`, CUDA-graph capture) |

## 3. EC2 instance selection (Gemma 4 26B-A4B)

The production profile serves **Gemma 4 26B-A4B** (MoE: 26B total, ~4B active per token) as the FP8 checkpoint `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` (28.6 GB), plus the 0.4B MTP drafter `google/gemma-4-26B-A4B-it-assistant` for speculative decoding. That is ~29.4 GiB of weights. Decode speed is bound by how fast the GPU reads the active weights and KV cache, so GPU memory bandwidth decides inter-token latency.

| Instance | GPU | Fits? | Latency notes |
|---|---|---|---|
| **`p5.4xlarge`** | 1× H100 80 GB (3.35 TB/s) | yes, ~42 GB left for KV | **Lowest latency.** Native FP8. On vLLM 0.30/0.31, run.sh switches attention to Triton to avoid an FA4 decode regression (see section 9). On-Demand only in some regions (London, Mumbai, Jakarta, Tokyo, São Paulo); elsewhere use Capacity Blocks |
| **`g6e.2xlarge`** | 1× L40S 48 GB (0.86 TB/s) | yes, ~11 GB left for KV (≈ hundreds of 8k-token calls, since most layers are 1024-token sliding window) | **Best price/latency.** Native FP8 (Ada). Prefer `2xlarge` (8 vCPU, 64 GiB RAM) over `xlarge` (4 vCPU, 32 GiB): vLLM's API server, engine core and the gateway each want a core, and loading 29 GB through 32 GiB of RAM is tight |
| `p4de.24xlarge` | A100 80 GB | yes | No FP8 tensor cores: FP8 weights run as W8A16 (Marlin). Slower than H100 |
| `g6.xlarge` / `g5.xlarge` | L4 / A10G 24 GB | **no**: run.sh's fit check stops before download | `g6.12xlarge` (4× L4) with `ENGINE_TENSOR_PARALLEL_SIZE=2` fits, but tensor parallelism over PCIe adds latency to every token |
| `g7e` | RTX PRO 6000 Blackwell (SM120) | yes | Not recommended yet: FP8 MoE kernel gaps on SM120 (vllm#43507) |

Then confirm with `scripts/bench.py` at your real call concurrency.

## 4. Provisioning an instance

1. **AMI:** *AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)* is recommended. It ships a current NVIDIA driver (595.x as of mid-2026), so `run.sh` chooses the CUDA 13.0 build. With a plain Ubuntu AMI, install the driver yourself (`sudo ubuntu-drivers install --gpgpu`, then reboot) and check `nvidia-smi`.
2. **Storage:** root EBS gp3 ≥ 100 GB, or put the model cache on a separate volume: `HF_HOME=/opt/models/huggingface`.
3. **Security group:** allow inbound 443 (nginx) or 8000 (from the ALB only). Never open the engine port 8001, which is bound to 127.0.0.1 anyway.
4. **IAM / secrets:** keep `API_KEYS` and `HF_TOKEN` out of the AMI. Write them into `.env` at boot from SSM Parameter Store or Secrets Manager, or pass them as systemd `Environment=` overrides.

## 5. Install

```bash
sudo useradd --system --home /var/lib/llm-server --create-home --shell /usr/sbin/nologin llm
sudo mkdir -p /opt/llm-server /opt/models /etc/llm-server && sudo chown llm:llm /opt/llm-server /opt/models
sudo -u llm git clone <this repo> /opt/llm-server        # or rsync / CodeDeploy

# secrets + overrides live outside the repo
sudo install -o llm -g llm -m 600 /opt/llm-server/deploy/production.env.example /etc/llm-server/llm-server.env
sudo vi /etc/llm-server/llm-server.env                    # API_KEYS (real ones; placeholders are rejected), HF_TOKEN, FORWARDED_ALLOW_IPS

cd /opt/llm-server
sudo -u llm HOME=/var/lib/llm-server ENV_FILE=/etc/llm-server/llm-server.env ./run.sh check
#   -> GPU, driver, chosen vLLM build (cu130), and the exact `vllm serve` command
sudo -u llm HOME=/var/lib/llm-server ENV_FILE=/etc/llm-server/llm-server.env HF_HOME=/opt/models/huggingface ./run.sh install
#   -> venvs, vLLM, ~30 GB of weights (model + drafter), then the GPU memory fit check
```

Generate API keys with `python3 -c 'import secrets; print(secrets.token_urlsafe(32))'`.

### Run as a service

```bash
sudo cp deploy/systemd/llm-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llm-server
journalctl -u llm-server -f                               # JSON logs (gateway) + vLLM logs
curl -s localhost:8000/health/ready                        # 200 after load + CUDA graphs + warm-up (a few minutes)
.venv/bin/python scripts/smoke.py --api-key "$KEY" --tools
.venv/bin/python scripts/bench.py --api-key "$KEY" --concurrency 1,8,16,32
LLM_TEST_ENGINE_URL=http://127.0.0.1:8001 LLM_TEST_ENGINE_KEY="$(sudo cat /proc/$(pgrep -f 'python -m app$')/environ | tr '\0' '\n' | sed -n 's/^ENGINE_API_KEY=//p')" ./run.sh test tests/engine
```

`systemctl stop` sends SIGTERM to `run.sh`. Readiness switches to 503 at once, open WebSockets finish their current reply and then receive `session.closing`. In-flight responses get `SHUTDOWN_GRACE_S`, after which vLLM is stopped. `Restart=on-failure` restarts the pair if either process dies.

### TLS and WebSockets

Use either:
- **nginx on the instance:** `deploy/nginx/llm-server.conf` (buffering off for SSE, upgrade headers for WS, long read timeout); or
- **an ALB:** HTTPS listener → target group on port 8000, health check `GET /health/ready` (200), idle timeout ≥ 300 s (WebSockets), stickiness on if REST sessions are spread across instances. Set `FORWARDED_ALLOW_IPS` to the ALB subnet CIDR (or `*` in a private subnet) so client IPs are correct.

## 6. Latency profile (what is tuned and why)

| Setting | Value | Why |
|---|---|---|
| Checkpoint | FP8 W8A8 (`RedHatAI/...-FP8-dynamic`), pinned revision | Half the bytes per decoded token vs BF16, with 99–103% accuracy recovery (incl. BFCLv4 tool calling). No 4-bit: the small expert width (704) loses quality at 4 bits |
| Speculative decoding | MTP drafter `gemma-4-26B-A4B-it-assistant`, 4 tokens | Several tokens per target forward pass. It shares the target's KV cache, and vLLM keeps async scheduling on with MTP. Watch `vllm:spec_decode_*` acceptance in `/metrics/engine`. Set `ENGINE_SPECULATIVE_CONFIG=` (empty) if you run at high concurrency, where it costs throughput |
| Thinking | off (`DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'`) | Hidden reasoning is silent time on a call. The template default is also off |
| `ENGINE_LANGUAGE_MODEL_ONLY` | true | Vision tower skipped: no multimodal profiling, more KV cache |
| `ENGINE_MAX_MODEL_LEN` | 8192 | Decode attention time grows with context. Voice calls rarely need more; older turns are trimmed |
| `ENGINE_MAX_NUM_SEQS` = `MAX_CONCURRENT_GENERATIONS` | 32 | Small batches keep inter-token latency flat. Excess callers queue for ≤ `QUEUE_TIMEOUT_S`=3 s, then get 429 (play a fallback) |
| Prefix caching | on | Turn N only prefills the new utterance, not the system prompt + history |
| CUDA graphs | on (`ENGINE_ENFORCE_EAGER=false`) | Large decode-latency win |
| KV cache dtype | bf16 | Gemma 4 is mostly 1024-token sliding window, so KV is already small. FP8 KV would add quantization cost for no capacity need |
| Attention backend | auto | vLLM picks FA4 on Hopper/Blackwell and Triton elsewhere for Gemma 4's mixed head sizes. run.sh overrides to Triton on SM90 for vLLM 0.30/0.31 (section 9) |
| Sampling | model defaults (temperature 1.0, top_p 0.95, top_k 64) | Google's recommended settings. A lower temperature (e.g. `params.temperature=0.7` per session) raises the drafter acceptance rate and speeds decode |
| `DEFAULT_MAX_TOKENS` | 200 | Bounds the worst-case reply time |
| Engine warm-up | on | JIT, CUDA-graph and drafter paths are exercised before `/health/ready` turns 200 |

### Production checklist

- `APP_ENV=production`, real `API_KEYS` (placeholders and keys under 16 characters are rejected), secrets only in `/etc/llm-server/llm-server.env`.
- `ENGINE_REVISION` and the drafter `revision` in `ENGINE_SPECULATIVE_CONFIG` are pinned; `ENGINE_VLLM_VERSION=0.31.0` is pinned.
- `MAX_CONCURRENT_GENERATIONS` comes from `scripts/bench.py`: the highest concurrency whose p95 TTFT and ITL meet your budget.
- `METRICS_REQUIRE_AUTH=true` if `/metrics` is reachable from outside the VPC; `RATE_LIMIT_PER_MINUTE` if keys are shared.
- `ENGINE_SCHEDULING_POLICY=priority` together with `CONTEXT_STRATEGY=summarize`, if you enable summaries.

## 7. Monitoring

- Scrape `GET /metrics` (gateway) and `GET /metrics/engine` (vLLM) with Prometheus, or with the CloudWatch agent's Prometheus support.
- Key signals: the speculative-decoding acceptance rate (`vllm:spec_decode_num_accepted_tokens_total` / `vllm:spec_decode_num_draft_tokens_total`), `llm_time_to_first_token_seconds` p95, `llm_inter_chunk_latency_seconds` p95, `llm_admission_rejections_total`, `llm_engine_errors_total`, `gateway_engine_ready`, and vLLM's `vllm:kv_cache_usage_perc` and `vllm:num_requests_waiting`.
- Alert on `/health/ready` ≠ 200 for more than 1 minute, on rising rejections, and on a p95 TTFT above budget.
- GPU: `nvidia-smi dmon` or DCGM (shipped on the DLAMI) for utilisation and memory.

## 8. Upgrades and rollback

- **vLLM:** change `ENGINE_VLLM_VERSION` and restart. `run.sh` sees the stamp mismatch and reinstalls only the engine venv. Roll back by restoring the old value.
- **Gateway:** `git pull`, then restart (dependencies reinstall only when the requirements files change).
- **Model:** change `ENGINE_MODEL` / `ENGINE_REVISION` (and the tool/reasoning parser if the family changes), then restart.
- **Immutable hosts / AMI baking:** run `./run.sh install` during the image build, then start with `SKIP_INSTALL=1` so a production boot never touches the network for packages.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `GPU detected but torch cannot use CUDA` | Driver/runtime mismatch. Check `nvidia-smi`; force `ENGINE_CUDA_VARIANT=cu129` on drivers 525–579 |
| Engine exits with CUDA OOM at startup | Lower `ENGINE_GPU_MEMORY_UTILIZATION` or `ENGINE_MAX_MODEL_LEN`; another process may hold GPU memory (`nvidia-smi`) |
| `Python.h: No such file or directory` / `CppCompileError` | Install `python3.X-dev` and `build-essential`, or let run.sh use uv's managed Python (default). run.sh recreates an engine venv that lacks headers |
| run.sh says `engine failed to start` | The root cause is printed, and the full vLLM log is `run/engine.log` |
| `model download failed` | Gated or private repo: set `HF_TOKEN` and accept the licence on huggingface.co |
| Startup timeout | Large model or slow disk: raise `ENGINE_STARTUP_TIMEOUT_S`; `ENGINE_ENFORCE_EAGER=true` skips CUDA-graph capture |
| 429 `overloaded` | Concurrency limit reached: scale out, raise `MAX_CONCURRENT_GENERATIONS` (check ITL), or lower `DEFAULT_MAX_TOKENS` |
| 400 `context_length_exceeded` | One input alone exceeds the window: raise `ENGINE_MAX_MODEL_LEN` or send less text |
| Tool calls come back as plain text | Wrong `ENGINE_TOOL_CALL_PARSER` for the model family (see `.env.example`) |
| H100 decode slower than expected on vLLM 0.30/0.31 | FA4 ignores `num_splits` on SM90 (vllm#59362). run.sh already uses `TRITON_ATTN` there; once a vLLM release includes flash-attention#198, bump `ENGINE_VLLM_VERSION` and the workaround switches itself off |
| `tool_choice: "required"` returns prose with Gemma 4 | Known gemma4 parser gap (vllm#53363): use `auto` and instruct the model in the system prompt |
| `the model does not fit` | ~29.4 GiB of weights need a ≥ 40 GB GPU (L40S/A100/H100) or tensor parallelism |
| CPU engine very slow | Expected without AVX-512. Use a smaller model, `ENGINE_MODE=mock` for UI work, or a remote GPU engine |
