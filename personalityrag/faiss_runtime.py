from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from importlib import metadata
from typing import Any

from .logger import logger


_GENERIC_FALLBACK_MARKERS = (
    "illegal instruction",
    "optimized",
    "avx",
    "simd",
    "dll load failed",
    "cannot open shared object file",
    "could not load library",
    "image not found",
    "symbol not found",
    "undefined symbol",
)


class FaissRuntimeError(RuntimeError):
    """A typed FAISS dependency failure, distinct from provider failures."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _details(result: subprocess.CompletedProcess[str]) -> str:
    value = (result.stderr or result.stdout or "").strip()
    if result.returncode < 0:
        value = f"process terminated by signal {-result.returncode}; {value}".strip()
    for raw in (os.environ.get("USERPROFILE"), os.environ.get("HOME")):
        if raw:
            value = value.replace(raw, "[user-profile]")
    return value[:2000]


def _binding_mismatch(value: str) -> bool:
    lowered = value.lower()
    return "superkmeans" in lowered or (
        "python binding" in lowered and "mismatch" in lowered
    )


def _can_try_generic(result: subprocess.CompletedProcess[str]) -> bool:
    if result.returncode < 0:
        return True
    lowered = _details(result).lower()
    return any(marker in lowered for marker in _GENERIC_FALLBACK_MARKERS)


def _version() -> str:
    try:
        return metadata.version("faiss-cpu")
    except metadata.PackageNotFoundError:
        return "unknown"


def _probe(*, generic: bool = False) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if generic:
        environment["FAISS_OPT_LEVEL"] = "generic"
    return subprocess.run(
        [sys.executable, "-c", "import faiss"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=environment,
    )


@lru_cache(maxsize=1)
def ensure_faiss_runtime() -> dict[str, Any]:
    """Probe FAISS safely before importing its native extension in-process."""

    try:
        result = _probe()
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FaissRuntimeError(
            "dependency_probe_failed",
            "FAISS runtime probe failed before vector indexes could initialize.",
        ) from exc
    if result.returncode == 0:
        return {"mode": "optimized", "version": _version()}

    details = _details(result)
    if _binding_mismatch(details):
        raise FaissRuntimeError(
            "binding_mismatch",
            "FAISS Python bindings do not match the native extension "
            f"(faiss-cpu {_version()}). This is a runtime dependency error, "
            "not an embedding-provider configuration error. Avoid faiss-cpu "
            f"1.14.2.{(' Details: ' + details) if details else ''}",
        )
    if not _can_try_generic(result):
        raise FaissRuntimeError(
            "dependency_load_failed",
            "FAISS could not load in the current Python environment; verify "
            "that its Python wrapper and native extension came from one install."
            f"{(' Details: ' + details) if details else ''}",
        )

    try:
        generic_result = _probe(generic=True)
    except (OSError, subprocess.TimeoutExpired):
        generic_result = None
    if generic_result is not None and generic_result.returncode == 0:
        os.environ["FAISS_OPT_LEVEL"] = "generic"
        logger.warning(
            "FAISS optimized extension failed; using generic instruction-set mode"
        )
        return {"mode": "generic", "version": _version()}

    generic_details = _details(generic_result) if generic_result is not None else ""
    if generic_details and generic_details != details:
        details = f"{details}; generic mode: {generic_details}".strip("; ")
    raise FaissRuntimeError(
        "instruction_set_incompatible",
        "FAISS is incompatible with the current CPU/runtime, including generic "
        f"instruction-set mode.{(' Details: ' + details) if details else ''}",
    )


def load_faiss():
    ensure_faiss_runtime()
    try:
        import faiss
    except Exception as exc:  # pragma: no cover - subprocess probe owns diagnosis
        raise FaissRuntimeError(
            "dependency_load_failed",
            "FAISS passed its subprocess probe but failed to import in-process.",
        ) from exc
    from .resource_limits import configured_faiss_threads

    faiss.omp_set_num_threads(configured_faiss_threads())
    return faiss


def configure_loaded_faiss_threads(profile: str | None = None) -> bool:
    """Apply a profile change without forcing the native runtime to import."""

    faiss = sys.modules.get("faiss")
    if faiss is None:
        return False
    from .resource_limits import configured_faiss_threads

    faiss.omp_set_num_threads(configured_faiss_threads(profile))
    return True
