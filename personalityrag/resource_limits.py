from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any


PERFORMANCE_PROFILES = frozenset({"adaptive", "latency", "memory"})
DEFAULT_PERFORMANCE_PROFILE = "adaptive"
_MIB = 1024 * 1024
_GIB = 1024 * _MIB


def normalize_performance_profile(value: str | None) -> str:
    profile = str(value or DEFAULT_PERFORMANCE_PROFILE).strip().lower()
    if profile not in PERFORMANCE_PROFILES:
        raise ValueError(
            "performance_profile must be adaptive, latency, or memory"
        )
    return profile


def _profile_from_config_file() -> str:
    state_root = Path(
        os.environ.get(
            "PERSONALITYRAG_STATE_ROOT",
            str(Path(__file__).resolve().parents[1]),
        )
    )
    try:
        payload = json.loads(
            (state_root / "config" / "config.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError):
        return DEFAULT_PERFORMANCE_PROFILE
    try:
        return normalize_performance_profile(payload.get("performance_profile"))
    except ValueError:
        return DEFAULT_PERFORMANCE_PROFILE


def configured_performance_profile(value: str | None = None) -> str:
    if value is not None:
        return normalize_performance_profile(value)
    configured = os.environ.get("PERSONALITYRAG_PERFORMANCE_PROFILE")
    if configured is not None:
        try:
            return normalize_performance_profile(configured)
        except ValueError:
            return DEFAULT_PERFORMANCE_PROFILE
    return _profile_from_config_file()


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="ascii").strip()
    except OSError:
        return ""


def _cgroup_cpu_count() -> int | None:
    cpu_max = _read_text("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        parts = cpu_max.split()
        if len(parts) >= 2 and parts[0] != "max":
            try:
                quota, period = int(parts[0]), int(parts[1])
            except ValueError:
                pass
            else:
                if quota > 0 and period > 0:
                    return max(1, (quota + period - 1) // period)
    quota_text = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_text = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        quota, period = int(quota_text), int(period_text)
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return max(1, (quota + period - 1) // period)


def effective_cpu_count() -> int:
    candidates: list[int] = []
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            candidates.append(max(1, len(affinity(0))))
        except (OSError, TypeError):
            pass
    process_count = getattr(os, "process_cpu_count", None)
    if process_count is not None:
        try:
            candidates.append(max(1, int(process_count() or 1)))
        except (TypeError, ValueError):
            pass
    candidates.append(max(1, int(os.cpu_count() or 1)))
    cgroup_count = _cgroup_cpu_count()
    if cgroup_count is not None:
        candidates.append(cgroup_count)
    return max(1, min(candidates))


def _physical_memory_bytes() -> int | None:
    if os.name != "nt":
        try:
            sysconf = getattr(os, "sysconf", None)
            if sysconf is None:
                return None
            page_size = sysconf("SC_PAGE_SIZE")
            page_count = sysconf("SC_PHYS_PAGES")
            value = int(page_size) * int(page_count)
        except (TypeError, ValueError, OSError):
            return None
        return value if value > 0 else None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except (AttributeError, OSError):
        return None
    return int(status.ullTotalPhys) if ok and status.ullTotalPhys else None


def _cgroup_memory_limit_bytes() -> int | None:
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        raw = _read_text(path)
        if not raw or raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        # v1 commonly reports a sentinel close to LONG_MAX when unlimited.
        if 64 * _MIB <= value < (1 << 60):
            return value
    return None


def effective_memory_limit_bytes() -> int:
    override = os.environ.get("PERSONALITYRAG_MEMORY_LIMIT_BYTES")
    if override:
        try:
            return max(128 * _MIB, int(override))
        except ValueError:
            pass
    candidates = [
        value
        for value in (_physical_memory_bytes(), _cgroup_memory_limit_bytes())
        if value is not None and value > 0
    ]
    return min(candidates) if candidates else 2 * _GIB


def _positive_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(1, int(default))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return max(1, int(default))


def configured_io_workers(profile: str | None = None) -> int:
    profile = configured_performance_profile(profile)
    cpu_count = effective_cpu_count()
    if profile == "latency":
        default = min(8, max(2, cpu_count))
    elif profile == "memory":
        default = min(2, max(1, cpu_count))
    else:
        default = min(4, max(2, cpu_count))
    return _positive_env("PERSONALITYRAG_IO_WORKERS", default)


def configured_faiss_threads(profile: str | None = None) -> int:
    profile = configured_performance_profile(profile)
    if profile == "latency":
        default = min(8, effective_cpu_count())
    elif profile == "memory":
        default = 1
    else:
        default = min(4, effective_cpu_count())
    return _positive_env("PERSONALITYRAG_FAISS_THREADS", default)


def configured_blas_threads(profile: str | None = None) -> int:
    profile = configured_performance_profile(profile)
    if profile == "latency":
        default = min(8, effective_cpu_count())
    elif profile == "memory":
        default = 1
    else:
        default = min(4, effective_cpu_count())
    return _positive_env("PERSONALITYRAG_BLAS_THREADS", default)


def configured_sqlite_pool_size(profile: str | None = None) -> int:
    selected = configured_performance_profile(profile)
    default = 1 if selected == "memory" else min(2, effective_cpu_count())
    return min(2, _positive_env("PERSONALITYRAG_SQLITE_POOL_SIZE", default))


def configured_http_limits(profile: str | None = None) -> tuple[int, int]:
    selected = configured_performance_profile(profile)
    cpu_count = effective_cpu_count()
    if selected == "latency":
        default_max = min(64, max(16, cpu_count * 8))
        default_keepalive = min(32, max(8, cpu_count * 4))
    elif selected == "memory":
        default_max = min(8, max(4, cpu_count * 2))
        default_keepalive = min(4, max(2, cpu_count))
    else:
        default_max = min(32, max(8, cpu_count * 4))
        default_keepalive = min(16, max(4, cpu_count * 2))
    maximum = _positive_env("PERSONALITYRAG_HTTP_MAX_CONNECTIONS", default_max)
    keepalive = min(
        maximum,
        _positive_env(
            "PERSONALITYRAG_HTTP_MAX_KEEPALIVE_CONNECTIONS",
            default_keepalive,
        ),
    )
    return maximum, keepalive


def configured_surface_limits(profile: str | None = None) -> tuple[int, int]:
    selected = configured_performance_profile(profile)
    cpu_count = effective_cpu_count()
    if selected == "latency":
        foreground = min(128, max(32, cpu_count * 32))
        heartbeat = min(24, max(8, cpu_count * 6))
    elif selected == "memory":
        foreground = min(24, max(8, cpu_count * 8))
        heartbeat = min(8, max(4, cpu_count * 2))
    else:
        foreground = min(64, max(16, cpu_count * 16))
        heartbeat = min(16, max(6, cpu_count * 4))
    return (
        _positive_env("PERSONALITYRAG_FOREGROUND_CONCURRENCY", foreground),
        _positive_env("PERSONALITYRAG_HEARTBEAT_CONCURRENCY", heartbeat),
    )


def configured_provider_concurrency(profile: str | None = None) -> int:
    selected = configured_performance_profile(profile)
    cpu_count = effective_cpu_count()
    if selected == "latency":
        default = min(8, max(4, cpu_count * 2))
    elif selected == "memory":
        default = min(2, cpu_count)
    else:
        default = min(4, max(2, cpu_count * 2))
    return _positive_env("PERSONALITYRAG_PROVIDER_CONCURRENCY", default)


def runtime_memory_budget_bytes(profile: str | None = None) -> int:
    selected = configured_performance_profile(profile)
    limit = effective_memory_limit_bytes()
    if selected == "latency":
        ratio, ceiling = 0.55, 1536 * _MIB
    elif selected == "memory":
        ratio, ceiling = 0.20, 384 * _MIB
    else:
        ratio, ceiling = 0.35, 768 * _MIB
    return max(128 * _MIB, min(int(limit * ratio), ceiling))


def effective_runtime_capacity(
    configured_max: int,
    profile: str | None = None,
) -> int:
    selected = configured_performance_profile(profile)
    budget = runtime_memory_budget_bytes(selected)
    estimated_runtime = {
        "latency": 192 * _MIB,
        "adaptive": 256 * _MIB,
        "memory": 320 * _MIB,
    }[selected]
    budget_cap = max(1, budget // estimated_runtime)
    return max(1, min(int(configured_max), int(budget_cap)))


def effective_runtime_idle_minutes(
    configured_minutes: int,
    profile: str | None = None,
) -> int:
    selected = configured_performance_profile(profile)
    value = max(1, int(configured_minutes))
    if selected == "memory":
        return min(value, 5)
    if selected == "adaptive" and effective_memory_limit_bytes() <= 2 * _GIB:
        return min(value, 15)
    return value


def effective_performance_summary(profile: str | None = None) -> dict[str, Any]:
    selected = configured_performance_profile(profile)
    maximum, keepalive = configured_http_limits(selected)
    foreground, heartbeat = configured_surface_limits(selected)
    return {
        "profile": selected,
        "effective_cpu_count": effective_cpu_count(),
        "effective_memory_limit_bytes": effective_memory_limit_bytes(),
        "runtime_memory_budget_bytes": runtime_memory_budget_bytes(selected),
        "faiss_threads": configured_faiss_threads(selected),
        "blas_threads": configured_blas_threads(selected),
        "io_workers": configured_io_workers(selected),
        "sqlite_pool_max_size": configured_sqlite_pool_size(selected),
        "http_max_connections": maximum,
        "http_max_keepalive_connections": keepalive,
        "foreground_concurrency": foreground,
        "heartbeat_concurrency": heartbeat,
        "provider_concurrency": configured_provider_concurrency(selected),
    }


def configure_numeric_thread_environment() -> int:
    """Configure numeric runtimes before NumPy/FAISS are imported."""

    count = configured_blas_threads()
    override = os.environ.get("PERSONALITYRAG_BLAS_THREADS")
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if override is not None:
            os.environ[name] = str(count)
        else:
            os.environ.setdefault(name, str(count))
    return count
