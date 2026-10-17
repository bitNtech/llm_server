"""Helper commands for run.sh (validation, hardware report, install plan, engine argv).

    python -m app.cli doctor [--json]   hardware + configuration report, non-zero on errors
    python -m app.cli runtime           shell assignments of the settings run.sh needs
    python -m app.cli install-plan      shell assignments describing the vLLM build to install
    python -m app.cli engine-args       NUL-separated `vllm serve` argv
    python -m app.cli engine-env        shell `export` lines for the engine process
    python -m app.cli fit-check BYTES   do weights of BYTES fit the GPU(s)/RAM? (exit 4 if not)
"""
from __future__ import annotations

import json
import shlex
import sys

from pydantic import ValidationError

from app import __version__
from app.config import Settings
from app.hardware import detect
from app.plan import PlanError, engine_plan, fit_check, install_plan, resolve_device


def _settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        print("configuration error(s):", file=sys.stderr)
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]).upper() or "settings"
            print(f"  - {loc}: {err['msg']}", file=sys.stderr)
        raise SystemExit(2) from None


def _assign(name: str, value: object) -> str:
    if isinstance(value, bool):
        value = "true" if value else "false"
    return f"{name}={shlex.quote(str(value))}"


def cmd_runtime(s: Settings) -> int:
    lines = {
        "APP_ENV": s.app_env,
        "HOST": s.host,
        "PORT": s.port,
        "ENGINE_MODE": s.engine_mode,
        "ENGINE_HOST": s.engine_host,
        "ENGINE_PORT": s.engine_port,
        "ENGINE_URL": s.resolved_engine_base_url,
        "ENGINE_MODEL": s.engine_model,
        "ENGINE_REVISION": s.engine_revision,
        "ENGINE_DOWNLOAD_DIR": s.engine_download_dir,
        "ENGINE_TRUST_REMOTE_CODE": s.engine_trust_remote_code,
        "ENGINE_MODEL_PREFETCH": s.engine_model_prefetch,
        "ENGINE_DRAFT_MODEL": s.speculative.get("model", ""),
        "ENGINE_DRAFT_REVISION": s.speculative.get("revision", "") or "",
        "ENGINE_STARTUP_TIMEOUT_S": int(s.engine_startup_timeout_s),
        "ENGINE_WARMUP_TIMEOUT_S": int(s.engine_warmup_timeout_s) if s.engine_warmup else 0,
        "SHUTDOWN_GRACE_S": int(s.shutdown_grace_s),
        "AUTH_ENABLED": s.auth_enabled,
    }
    print("\n".join(_assign(f"CFG_{k}", v) for k, v in lines.items()))
    return 0


def cmd_install_plan(s: Settings) -> int:
    hw = detect()
    try:
        device = resolve_device(s, hw)
        plan = install_plan(s, hw, device, probe=True)
    except PlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    print(_assign("PLAN_DEVICE", device))
    print(_assign("PLAN_VARIANT", plan.variant))
    print(_assign("PLAN_TORCH_BACKEND", plan.torch_backend))
    print(_assign("PLAN_PIP_EXTRA_INDEX", plan.pip_extra_index_url))
    print(_assign("PLAN_STAMP", plan.stamp))
    print("PLAN_PACKAGES=(" + " ".join(shlex.quote(p) for p in plan.packages) + ")")
    return 0


def cmd_engine_args(s: Settings) -> int:
    try:
        plan = engine_plan(s, detect())
    except PlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    for warning in plan.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    sys.stdout.write("\0".join(plan.argv) + "\0")
    return 0


def cmd_engine_env(s: Settings) -> int:
    try:
        plan = engine_plan(s, detect())
    except PlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    for key, value in plan.env.items():
        print(f"export {_assign(key, value)}")
    return 0


def cmd_doctor(s: Settings, as_json: bool) -> int:
    hw = detect()
    problems: list[str] = []
    report: dict = {"version": __version__, "hardware": hw.to_dict(), "engine_mode": s.engine_mode}
    if s.engine_mode == "managed":
        try:
            plan = engine_plan(s, hw)
            report["engine"] = {"device": plan.device, "argv": plan.argv, "warnings": plan.warnings}
            inst = install_plan(s, hw, plan.device, probe=False)
            report["install"] = {"variant": inst.variant, "packages": inst.packages,
                                 "torch_backend": inst.torch_backend}
        except PlanError as exc:
            problems.append(str(exc))
    if hw.mem_total_gib and hw.mem_total_gib < 8:
        problems.append(f"only {hw.mem_total_gib} GiB RAM: at least 8 GiB is recommended")
    if not s.auth_enabled:
        report.setdefault("notes", []).append("authentication disabled (API_KEYS empty)")
    report["problems"] = problems

    if as_json:
        print(json.dumps(report, indent=2, default=str))
        return 1 if problems else 0

    print(f"llm-server {__version__} doctor")
    print(f"  os/arch        : {hw.os}/{hw.arch}  glibc={hw.glibc}  container={hw.in_container}")
    print(f"  cpu            : {hw.cpu_model} ({hw.cpu_count} threads)  "
          f"avx2={hw.has_flag('avx2')} avx512f={hw.has_flag('avx512f')} amx={hw.has_flag('amx_tile')}")
    print(f"  memory         : {hw.mem_total_gib} GiB total, {hw.mem_available_gib} GiB available")
    if hw.gpus:
        print(f"  nvidia driver  : {hw.nvidia_driver}")
        for g in hw.gpus:
            cc = ".".join(map(str, g.compute_capability)) if g.compute_capability else "?"
            print(f"  gpu {g.index}          : {g.name}  {g.memory_mib} MiB  cc={cc}")
    else:
        print("  nvidia gpu     : none visible")
    for dev in hw.other_display_devices:
        print(f"  other display  : {dev} (not usable by vLLM CUDA)")
    print(f"  engine mode    : {s.engine_mode}")
    if "engine" in report:
        eng = report["engine"]
        print(f"  engine device  : {eng['device']}  (install: {report['install']['variant']})")
        print(f"  model          : {s.engine_model}  max_model_len={s.engine_max_model_len}")
        print("  vllm argv      : vllm " + " ".join(shlex.quote(a) for a in eng["argv"]))
        for w in eng["warnings"]:
            print(f"  warning        : {w}")
    for note in report.get("notes", []):
        print(f"  note           : {note}")
    for p in problems:
        print(f"  PROBLEM        : {p}")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 2
    command, rest = args[0], args[1:]
    s = _settings()
    if command == "runtime":
        return cmd_runtime(s)
    if command == "install-plan":
        return cmd_install_plan(s)
    if command == "engine-args":
        return cmd_engine_args(s)
    if command == "engine-env":
        return cmd_engine_env(s)
    if command == "doctor":
        return cmd_doctor(s, "--json" in rest)
    if command == "validate":
        return 0
    if command == "fit-check":
        try:
            ok, message = fit_check(s, detect(), int(rest[0]))
        except (IndexError, ValueError, PlanError) as exc:
            print(f"fit-check: {exc}", file=sys.stderr)
            return 2
        print(message)
        return 0 if ok else 4
    print(f"unknown command: {command}\n{__doc__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
