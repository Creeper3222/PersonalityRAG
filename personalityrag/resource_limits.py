from __future__ import annotations

import os


def effective_cpu_count() -> int:
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return max(1, len(affinity(0)))
        except (OSError, TypeError):
            pass
    process_count = getattr(os, "process_cpu_count", None)
    if process_count is not None:
        try:
            return max(1, int(process_count() or 1))
        except (TypeError, ValueError):
            pass
    return max(1, int(os.cpu_count() or 1))


def _positive_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(1, int(default))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return max(1, int(default))


def configured_io_workers() -> int:
    default = min(8, max(2, effective_cpu_count()))
    return _positive_env("PERSONALITYRAG_IO_WORKERS", default)


def configured_faiss_threads() -> int:
    default = min(8, effective_cpu_count())
    return _positive_env("PERSONALITYRAG_FAISS_THREADS", default)


def configured_blas_threads() -> int:
    default = min(8, effective_cpu_count())
    return _positive_env("PERSONALITYRAG_BLAS_THREADS", default)


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
