"""Turn (settings + detected hardware) into a concrete engine plan.

The plan answers three questions for run.sh:
  * which vLLM build to install (CUDA 13.0 / CUDA 12.9 / CPU wheel) and with which torch backend,
  * the exact `vllm serve` argv,
  * the extra environment the engine process needs (secrets stay out of argv / `ps`).
"""
from __future__ import annotations

import shlex
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from app.config import Settings
from app.hardware import Hardware

GITHUB_RELEASES = "https://github.com/vllm-project/vllm/releases/download"
# Newest first. vLLM's CPU wheel tag has moved between releases (2_34 / 2_35 / 2_39).
CPU_MANYLINUX_TAGS = ("2_39", "2_35", "2_34", "2_31", "2_28")

MIN_COMPUTE_CAPABILITY = (7, 0)
CU130_MIN_DRIVER = 580
CU129_MIN_DRIVER = 525
CU130_MIN_COMPUTE_CAPABILITY = (7, 5)
BF16_MIN_COMPUTE_CAPABILITY = (8, 0)
# vLLM 0.30.x/0.31.x pin a flash-attention build whose FA4 kernel ignores num_splits on SM90
# (H100/H200): decode attention runs one KV split per head, up to ~2x slower decode for models
# with few KV heads such as Gemma 4 (vllm-project/vllm#59362, fixed by flash-attention#198).
# Triton attention recovers most of it until a vLLM release picks up the fix.
FA4_SM90_REGRESSION = ("0.30.", "0.31.")


class PlanError(RuntimeError):
    pass


@dataclass
class InstallPlan:
    packages: list[str]
    torch_backend: str  # uv --torch-backend value
    pip_extra_index_url: str  # for the pip fallback (no uv)
    variant: str  # cu130 | cu129 | cpu | custom
    stamp: str  # changes whenever a reinstall is required


@dataclass
class EnginePlan:
    device: str
    argv: list[str]
    env: dict[str, str]
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- device
def resolve_device(settings: Settings, hw: Hardware) -> str:
    if settings.engine_device == "cuda":
        if not hw.has_cuda_gpu:
            raise PlanError(
                "ENGINE_DEVICE=cuda but no NVIDIA GPU is visible (nvidia-smi missing or failing). "
                "Install the NVIDIA driver, or use ENGINE_DEVICE=cpu / auto."
            )
        return "cuda"
    if settings.engine_device == "cpu":
        return "cpu"
    return "cuda" if hw.has_cuda_gpu else "cpu"


def _check_cuda(hw: Hardware) -> None:
    cc = hw.min_compute_capability
    if cc and cc < MIN_COMPUTE_CAPABILITY:
        raise PlanError(f"GPU compute capability {cc[0]}.{cc[1]} < 7.0 is not supported by vLLM")
    if hw.driver_major is None:
        raise PlanError("could not read the NVIDIA driver version from nvidia-smi")
    if hw.driver_major < CU129_MIN_DRIVER:
        raise PlanError(
            f"NVIDIA driver {hw.nvidia_driver} is too old; vLLM needs >= {CU129_MIN_DRIVER} "
            f"(CUDA 12.9 build) or >= {CU130_MIN_DRIVER} (CUDA 13.0 build)"
        )


def cuda_variant(settings: Settings, hw: Hardware) -> str:
    if settings.engine_cuda_variant != "auto":
        return settings.engine_cuda_variant
    cc = hw.min_compute_capability
    if (hw.driver_major or 0) >= CU130_MIN_DRIVER and (cc is None or cc >= CU130_MIN_COMPUTE_CAPABILITY):
        return "cu130"
    return "cu129"


# -------------------------------------------------------------------------- install
def _wheel_url(version: str, local: str, tag: str, arch: str) -> str:
    return f"{GITHUB_RELEASES}/v{version}/vllm-{version}+{local}-cp38-abi3-manylinux_{tag}_{arch}.whl"


def _url_exists(url: str, timeout: float = 15.0) -> bool:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # follows redirects
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _cpu_wheel(version: str, hw: Hardware, probe: bool) -> str:
    arch = "aarch64" if hw.arch in ("aarch64", "arm64") else "x86_64"
    glibc = hw.glibc or (99, 99)
    tags = [t for t in CPU_MANYLINUX_TAGS if tuple(int(x) for x in t.split("_")) <= glibc]
    if not tags:
        raise PlanError(f"glibc {hw.glibc} is too old for vLLM CPU wheels")
    if not probe:
        return _wheel_url(version, "cpu", tags[0], arch)
    for tag in tags:
        url = _wheel_url(version, "cpu", tag, arch)
        if _url_exists(url):
            return url
    raise PlanError(
        f"no vLLM {version} CPU wheel found for {arch} / glibc {glibc} (tried manylinux {tags}). "
        "Check network access to github.com or set ENGINE_INSTALL_SPEC."
    )


def install_plan(settings: Settings, hw: Hardware, device: str, probe: bool = True) -> InstallPlan:
    version = settings.engine_vllm_version
    if settings.engine_install_spec:
        backend = "cpu" if device == "cpu" else "auto"
        extra = "https://download.pytorch.org/whl/cpu" if device == "cpu" else ""
        return InstallPlan(shlex.split(settings.engine_install_spec), backend, extra, "custom",
                           f"custom:{settings.engine_install_spec}")
    if device == "cuda":
        _check_cuda(hw)
        variant = cuda_variant(settings, hw)
        if variant == "cu130":
            return InstallPlan([f"vllm=={version}"], "cu130", "", variant, f"{version}+cu130")
        arch = "aarch64" if hw.arch in ("aarch64", "arm64") else "x86_64"
        url = _wheel_url(version, "cu129", "2_28", arch)
        return InstallPlan([url], "cu129", "https://download.pytorch.org/whl/cu129", variant,
                           f"{version}+cu129")
    url = _cpu_wheel(version, hw, probe)
    return InstallPlan([url], "cpu", "https://download.pytorch.org/whl/cpu", "cpu", f"{version}+cpu")


# ----------------------------------------------------------------------------- argv
def _cpu_kv_cache_gib(settings: Settings, hw: Hardware) -> float:
    if settings.engine_kv_cache_memory_gib > 0:
        return settings.engine_kv_cache_memory_gib
    # Weights also live in RAM on CPU: give the KV cache ~15% of RAM, within [1, 32] GiB.
    return round(min(32.0, max(1.0, hw.mem_total_gib * 0.15)), 1)


def engine_plan(settings: Settings, hw: Hardware) -> EnginePlan:
    s = settings
    device = resolve_device(s, hw)
    warnings: list[str] = []
    env: dict[str, str] = {}
    argv = ["serve", s.engine_model, "--host", s.engine_host, "--port", str(s.engine_port)]

    def opt(flag: str, value: object) -> None:
        if value not in (None, "", 0):
            argv.extend([flag, str(value)])

    def flag(name: str, enabled: bool) -> None:
        if enabled:
            argv.append(name)

    opt("--served-model-name", s.served_model_name)
    opt("--revision", s.engine_revision)
    opt("--tokenizer", s.engine_tokenizer)
    flag("--trust-remote-code", s.engine_trust_remote_code)
    opt("--quantization", s.engine_quantization)
    opt("--max-model-len", s.engine_max_model_len)
    if s.engine_kv_cache_dtype != "auto":
        opt("--kv-cache-dtype", s.engine_kv_cache_dtype)
    opt("--max-num-seqs", s.engine_max_num_seqs)
    opt("--max-num-batched-tokens", s.engine_max_num_batched_tokens)
    argv.append("--enable-prefix-caching" if s.engine_enable_prefix_caching else "--no-enable-prefix-caching")
    flag("--enforce-eager", s.engine_enforce_eager)
    flag("--language-model-only", s.engine_language_model_only)
    if s.engine_enable_auto_tool_choice and s.engine_tool_call_parser:
        argv.extend(["--enable-auto-tool-choice", "--tool-call-parser", s.engine_tool_call_parser])
    opt("--reasoning-parser", s.engine_reasoning_parser)
    opt("--chat-template", s.engine_chat_template)
    if s.engine_generation_config != "auto":
        opt("--generation-config", s.engine_generation_config)
    if s.engine_scheduling_policy != "fcfs":
        opt("--scheduling-policy", s.engine_scheduling_policy)
    if s.engine_speculative_config and device == "cuda":
        opt("--speculative-config", s.engine_speculative_config)
    elif s.engine_speculative_config:
        warnings.append("speculative decoding is disabled on the CPU engine")
    opt("--download-dir", s.engine_download_dir)
    flag("--enable-log-requests", s.engine_log_requests)
    argv.extend([
        "--enable-prompt-tokens-details",  # cached (prefix-hit) prompt tokens in usage
        "--enable-request-id-headers",  # X-Request-Id from the gateway becomes the engine request id
        "--disable-uvicorn-access-log",
        "--disable-fastapi-docs",
        "--log-level", "DEBUG" if s.log_level == "DEBUG" else "INFO",
    ])

    if device == "cuda":
        cc = hw.min_compute_capability
        dtype = s.engine_dtype
        if dtype == "auto" and cc and cc < BF16_MIN_COMPUTE_CAPABILITY:
            dtype = "float16"
            warnings.append(f"compute capability {cc[0]}.{cc[1]} has no bfloat16: using float16")
        opt("--dtype", dtype if dtype != "auto" else "")
        backend = s.engine_attention_backend
        if backend == "auto":
            backend = ""
            if cc == (9, 0) and s.engine_vllm_version.startswith(FA4_SM90_REGRESSION):
                backend = "TRITON_ATTN"
                warnings.append("SM90 + vLLM 0.30/0.31: using TRITON_ATTN to avoid the FA4 decode regression "
                                "(vllm#59362); set ENGINE_ATTENTION_BACKEND once a fixed vLLM is pinned")
        opt("--attention-backend", backend)
        tp = len(hw.gpus) if s.engine_tensor_parallel_size == "auto" else int(s.engine_tensor_parallel_size)
        if tp > len(hw.gpus):
            raise PlanError(f"ENGINE_TENSOR_PARALLEL_SIZE={tp} but only {len(hw.gpus)} GPU(s) are visible")
        opt("--tensor-parallel-size", tp if tp > 1 else "")
        if s.engine_kv_cache_memory_gib > 0:
            opt("--kv-cache-memory-bytes", int(s.engine_kv_cache_memory_gib * 1024**3))
        else:
            opt("--gpu-memory-utilization", s.engine_gpu_memory_utilization)
    else:
        if s.engine_attention_backend not in ("auto", ""):
            opt("--attention-backend", s.engine_attention_backend)
        if not hw.has_flag("avx512f") and hw.arch in ("x86_64", "amd64"):
            warnings.append("CPU has no AVX-512: vLLM's CPU backend runs with reduced performance (AVX2)")
        opt("--dtype", "bfloat16" if s.engine_dtype == "auto" else s.engine_dtype)
        opt("--kv-cache-memory-bytes", int(_cpu_kv_cache_gib(s, hw) * 1024**3))
        env["VLLM_CPU_OMP_THREADS_BIND"] = "auto"
        env["VLLM_CPU_NUM_OF_RESERVED_CPU"] = str(s.engine_cpu_reserved_cores)
        if s.engine_tensor_parallel_size not in ("1", "auto"):
            warnings.append("tensor parallelism is ignored for the CPU engine")

    if s.engine_api_key.get_secret_value():
        env["VLLM_API_KEY"] = s.engine_api_key.get_secret_value()  # never on argv (visible in `ps`)
    argv.extend(shlex.split(s.engine_extra_args))
    return EnginePlan(device=device, argv=argv, env=env, warnings=warnings)


def fit_check(settings: Settings, hw: Hardware, weights_bytes: int) -> tuple[bool, str]:
    """Will the weights fit, leaving room for activations and KV cache? Returns (ok, message)."""
    gib = weights_bytes / 1024**3
    device = resolve_device(settings, hw)
    if device == "cuda":
        tp_setting = settings.engine_tensor_parallel_size
        tp = len(hw.gpus) if tp_setting == "auto" else int(tp_setting)
        gpus = sorted(hw.gpus, key=lambda g: g.memory_mib)[: max(tp, 1)]
        util = settings.engine_gpu_memory_utilization
        budget = sum(g.memory_mib for g in gpus) / 1024 * util
        where = f"{len(gpus)} GPU(s) x {gpus[0].memory_mib / 1024:.0f} GiB at {util:.2f}"
    else:
        budget = hw.mem_total_gib * 0.8
        where = f"{hw.mem_total_gib:.0f} GiB RAM"
    message = f"weights {gib:.1f} GiB vs usable {budget:.1f} GiB ({where})"
    # weights plus at least ~10% for activations, CUDA graphs and a minimal KV cache
    return gib * 1.1 <= budget, message
