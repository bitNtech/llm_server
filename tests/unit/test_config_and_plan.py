from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.hardware import GPU, Hardware
from app.plan import PlanError, engine_plan, fit_check, install_plan, resolve_device


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def hardware(gpus: list[GPU] | None = None, driver: str | None = None, flags=("avx2",), glibc=(2, 39),
             arch="x86_64", mem=64.0) -> Hardware:
    return Hardware(os="linux", arch=arch, cpu_model="test", cpu_count=16, cpu_flags=list(flags),
                    mem_total_gib=mem, mem_available_gib=mem / 2, glibc=glibc, nvidia_driver=driver,
                    gpus=gpus or [])


A10G = GPU(0, "NVIDIA A10G", 23028, (8, 6))
T4 = GPU(0, "Tesla T4", 15360, (7, 5))
V100 = GPU(0, "Tesla V100", 16384, (7, 0))


# ------------------------------------------------------------------ settings
def test_defaults_are_valid():
    s = settings()
    assert s.engine_mode == "managed"
    assert s.chat_template_kwargs == {"enable_thinking": False}
    assert not s.auth_enabled
    assert s.resolved_engine_base_url == "http://127.0.0.1:8001"


def test_production_requires_api_keys():
    with pytest.raises(ValidationError, match="API_KEYS"):
        settings(app_env="production")
    keys = "k" * 32 + "," + "q" * 20
    assert settings(app_env="production", api_keys=keys).api_key_list == ["k" * 32, "q" * 20]
    with pytest.raises(ValidationError, match="placeholder"):
        settings(app_env="production", api_keys="CHANGE_ME_1")
    with pytest.raises(ValidationError, match="placeholder"):
        settings(app_env="production", api_keys="short")
    assert settings(app_env="production", allow_unauthenticated=True).app_env == "production"


def test_production_rejects_wildcard_cors():
    with pytest.raises(ValidationError, match="CORS"):
        settings(app_env="production", api_keys="k" * 32, cors_allow_origins="*")


def test_blank_sampling_defaults_become_none():
    s = settings(default_temperature="", default_top_k=" ")
    assert s.default_temperature is None and s.default_top_k is None


@pytest.mark.parametrize("field,value", [
    ("default_chat_template_kwargs", "[1, 2]"),
    ("default_chat_template_kwargs", "{not json"),
    ("engine_tensor_parallel_size", "0"),
    ("engine_tensor_parallel_size", "two"),
    ("engine_extra_args", "--flag 'unbalanced"),
])
def test_invalid_values_rejected(field, value):
    with pytest.raises(ValidationError):
        settings(**{field: value})


def test_max_tokens_limit_cross_check():
    with pytest.raises(ValidationError):
        settings(default_max_tokens=500, max_tokens_limit=100)


def test_external_mode_requires_url():
    with pytest.raises(ValidationError):
        settings(engine_mode="external")


# ---------------------------------------------------------------------- plan
def test_device_resolution():
    assert resolve_device(settings(), hardware()) == "cpu"
    assert resolve_device(settings(), hardware([A10G], "580.65")) == "cuda"
    assert resolve_device(settings(engine_device="cpu"), hardware([A10G], "580.65")) == "cpu"
    with pytest.raises(PlanError):
        resolve_device(settings(engine_device="cuda"), hardware())


@pytest.mark.parametrize("gpu,driver,variant", [
    (A10G, "580.82", "cu130"),
    (A10G, "570.10", "cu129"),
    (V100, "580.82", "cu129"),  # CUDA 13 dropped Volta
    (T4, "590.10", "cu130"),
])
def test_cuda_variant_selection(gpu, driver, variant):
    plan = install_plan(settings(), hardware([gpu], driver), "cuda", probe=False)
    assert plan.variant == variant
    assert plan.torch_backend == variant
    if variant == "cu130":
        assert plan.packages == ["vllm==0.31.0"]
    else:
        assert "+cu129" in plan.packages[0]


def test_old_driver_is_rejected():
    with pytest.raises(PlanError, match="too old"):
        install_plan(settings(), hardware([A10G], "470.1"), "cuda", probe=False)


def test_cpu_wheel_matches_glibc_and_arch():
    url = install_plan(settings(), hardware(glibc=(2, 43)), "cpu", probe=False).packages[0]
    assert url.endswith("+cpu-cp38-abi3-manylinux_2_39_x86_64.whl")
    url = install_plan(settings(), hardware(glibc=(2, 35), arch="aarch64"), "cpu", probe=False).packages[0]
    assert url.endswith("manylinux_2_35_aarch64.whl")


def test_install_spec_override():
    plan = install_plan(settings(engine_install_spec="vllm==0.30.0 extra-pkg"), hardware(), "cpu")
    assert plan.packages == ["vllm==0.30.0", "extra-pkg"] and plan.variant == "custom"


def test_cpu_engine_argv():
    plan = engine_plan(settings(engine_api_key="secret", engine_extra_args="--seed 7"), hardware(mem=16))
    argv = plan.argv
    assert argv[:2] == ["serve", "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic"]
    assert "--speculative-config" not in argv  # MTP drafting needs a GPU
    assert any("speculative" in w for w in plan.warnings)
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert int(argv[argv.index("--kv-cache-memory-bytes") + 1]) == int(2.4 * 1024**3)
    assert "--gpu-memory-utilization" not in argv
    assert argv[-2:] == ["--seed", "7"]
    assert "secret" not in " ".join(argv)  # secrets never on the command line
    assert plan.env["VLLM_API_KEY"] == "secret"
    assert plan.env["VLLM_CPU_OMP_THREADS_BIND"] == "auto"
    assert any("AVX-512" in w for w in plan.warnings)


def test_gpu_engine_argv_parsers_and_parallelism():
    four = [GPU(i, "NVIDIA L4", 23034, (8, 9)) for i in range(4)]
    plan = engine_plan(settings(engine_tensor_parallel_size="auto"), hardware(four, "580.82"))
    argv = plan.argv
    assert argv[argv.index("--tensor-parallel-size") + 1] == "4"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.9"
    assert argv[argv.index("--tool-call-parser") + 1] == "gemma4"
    assert argv[argv.index("--reasoning-parser") + 1] == "gemma4"
    assert argv[argv.index("--max-num-seqs") + 1] == "32"
    spec = argv[argv.index("--speculative-config") + 1]
    assert '"google/gemma-4-26B-A4B-it-assistant"' in spec and '"num_speculative_tokens": 4' in spec
    assert "--attention-backend" not in argv  # Ada: vLLM picks Triton for Gemma 4 itself
    assert "--enable-auto-tool-choice" in argv and "--language-model-only" in argv
    assert "--dtype" not in argv  # auto on Ampere+ keeps the checkpoint dtype


def test_pre_ampere_gpu_uses_float16():
    plan = engine_plan(settings(), hardware([T4], "580.82"))
    assert plan.argv[plan.argv.index("--dtype") + 1] == "float16"


def test_tensor_parallel_larger_than_gpu_count():
    with pytest.raises(PlanError):
        engine_plan(settings(engine_tensor_parallel_size="2"), hardware([A10G], "580.82"))


def test_disabling_features_removes_flags():
    plan = engine_plan(settings(engine_tool_call_parser="", engine_reasoning_parser="",
                                engine_enable_prefix_caching=False, engine_language_model_only=False),
                       hardware([A10G], "580.82"))
    assert "--enable-auto-tool-choice" not in plan.argv
    assert "--reasoning-parser" not in plan.argv
    assert "--no-enable-prefix-caching" in plan.argv
    assert "--language-model-only" not in plan.argv


H100 = GPU(0, "NVIDIA H100 80GB HBM3", 81559, (9, 0))
L40S = GPU(0, "NVIDIA L40S", 46068, (8, 9))


def test_sm90_attention_workaround_only_for_affected_vllm():
    argv = engine_plan(settings(), hardware([H100], "595.71")).argv
    assert argv[argv.index("--attention-backend") + 1] == "TRITON_ATTN"
    argv = engine_plan(settings(engine_vllm_version="0.32.0"), hardware([H100], "595.71")).argv
    assert "--attention-backend" not in argv
    argv = engine_plan(settings(engine_attention_backend="FLASH_ATTN"), hardware([H100], "595.71")).argv
    assert argv[argv.index("--attention-backend") + 1] == "FLASH_ATTN"


def test_fit_check():
    weights = int(29.4 * 1024**3)  # Gemma 4 26B-A4B FP8 + MTP drafter
    assert fit_check(settings(), hardware([H100], "595.71"), weights)[0]
    assert fit_check(settings(), hardware([L40S], "595.71"), weights)[0]
    ok, message = fit_check(settings(), hardware([GPU(0, "NVIDIA L4", 23034, (8, 9))], "595.71"), weights)
    assert not ok and "29.4 GiB" in message
    two_l4 = [GPU(i, "NVIDIA L4", 23034, (8, 9)) for i in range(2)]
    assert fit_check(settings(engine_tensor_parallel_size="2"), hardware(two_l4, "595.71"), weights)[0]
    assert not fit_check(settings(), hardware(mem=16), weights)[0]


def test_runtime_exposes_drafter_for_prefetch(capsys):
    from app.cli import cmd_runtime

    cmd_runtime(settings())
    out = capsys.readouterr().out
    assert "CFG_ENGINE_DRAFT_MODEL=google/gemma-4-26B-A4B-it-assistant" in out
    assert "CFG_ENGINE_DRAFT_REVISION=6e5aaaf4c42b98394530b8fda2e95cadd65c151c" in out
