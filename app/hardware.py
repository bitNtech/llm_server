"""Best-effort hardware detection (stdlib only; safe to run before any heavy dependency exists)."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class GPU:
    index: int
    name: str
    memory_mib: int
    compute_capability: tuple[int, int] | None


@dataclass
class Hardware:
    os: str
    arch: str
    cpu_model: str
    cpu_count: int
    cpu_flags: list[str]
    mem_total_gib: float
    mem_available_gib: float
    glibc: tuple[int, int] | None
    nvidia_driver: str | None
    gpus: list[GPU] = field(default_factory=list)
    other_display_devices: list[str] = field(default_factory=list)
    in_container: bool = False

    @property
    def has_cuda_gpu(self) -> bool:
        return bool(self.gpus)

    @property
    def driver_major(self) -> int | None:
        if not self.nvidia_driver:
            return None
        try:
            return int(self.nvidia_driver.split(".")[0])
        except ValueError:
            return None

    @property
    def min_compute_capability(self) -> tuple[int, int] | None:
        caps = [g.compute_capability for g in self.gpus if g.compute_capability]
        return min(caps) if caps else None

    def has_flag(self, flag: str) -> bool:
        return flag in self.cpu_flags

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cpu_flags"] = [f for f in self.cpu_flags if f.startswith(("avx", "amx", "fma", "sse4"))]
        return data


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def _meminfo() -> tuple[float, float]:
    total = avail = 0.0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            kib = float(rest.split()[0]) if rest.split() else 0.0
            if key == "MemTotal":
                total = kib / 1024 / 1024
            elif key == "MemAvailable":
                avail = kib / 1024 / 1024
    except OSError:
        pass
    # cgroup v2 limit (containers): never plan for more memory than we may use
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit.isdigit():
            gib = int(limit) / 1024**3
            total, avail = min(total, gib), min(avail, gib)
    except OSError:
        pass
    return round(total, 2), round(avail, 2)


def _cpu() -> tuple[str, list[str]]:
    model, flags = platform.processor() or "unknown", []
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, _, value = line.partition(":")
            key = key.strip()
            if key == "model name" and model in ("unknown", "", "x86_64"):
                model = value.strip()
            elif key in ("flags", "Features") and not flags:
                flags = value.split()
    except OSError:
        pass
    return model, sorted(set(flags))


def _glibc() -> tuple[int, int] | None:
    try:
        version = os.confstr("CS_GNU_LIBC_VERSION") or ""  # "glibc 2.39"
        major, minor = version.split()[-1].split(".")[:2]
        return int(major), int(minor)
    except (ValueError, OSError, AttributeError, IndexError):
        return None


def _visible(index: int) -> bool:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return True
    allowed = {v.strip() for v in visible.split(",") if v.strip()}
    return str(index) in allowed


def _nvidia() -> tuple[str | None, list[GPU]]:
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return None, []
    gpus, driver = [], None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx, name, mem, cap, driver = parts[:5]
        try:
            cc = tuple(int(x) for x in cap.split(".")[:2]) if "." in cap else None
            gpu = GPU(int(idx), name, int(float(mem)), cc)  # type: ignore[arg-type]
        except ValueError:
            continue
        if _visible(gpu.index):
            gpus.append(gpu)
    return driver, gpus


def _other_display_devices() -> list[str]:
    out = _run(["lspci"]) or ""
    return [
        line.split(": ", 1)[-1]
        for line in out.splitlines()
        if any(k in line for k in ("VGA", "3D controller", "Display controller")) and "NVIDIA" not in line
    ]


def detect() -> Hardware:
    total, avail = _meminfo()
    model, flags = _cpu()
    driver, gpus = _nvidia()
    return Hardware(
        os=platform.system().lower(),
        arch=platform.machine().lower(),
        cpu_model=model,
        cpu_count=os.cpu_count() or 1,
        cpu_flags=flags,
        mem_total_gib=total,
        mem_available_gib=avail,
        glibc=_glibc(),
        nvidia_driver=driver,
        gpus=gpus,
        other_display_devices=_other_display_devices(),
        in_container=Path("/.dockerenv").exists() or Path("/run/.containerenv").exists(),
    )
