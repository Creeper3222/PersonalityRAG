from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from personalityrag import faiss_runtime


def _result(returncode: int, error: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["python", "-c", "import faiss"],
        returncode=returncode,
        stdout="",
        stderr=error,
    )


@pytest.fixture(autouse=True)
def clear_probe_cache():
    faiss_runtime.ensure_faiss_runtime.cache_clear()
    yield
    faiss_runtime.ensure_faiss_runtime.cache_clear()


def test_binding_mismatch_is_dependency_error_without_generic_retry(monkeypatch):
    calls: list[bool] = []

    def probe(*, generic: bool = False):
        calls.append(generic)
        return _result(1, "AttributeError: module faiss has no attribute SuperKMeans")

    monkeypatch.setattr(faiss_runtime, "_probe", probe)
    monkeypatch.setattr(faiss_runtime, "_version", lambda: "1.14.2")

    with pytest.raises(faiss_runtime.FaissRuntimeError) as raised:
        faiss_runtime.ensure_faiss_runtime()

    assert raised.value.code == "binding_mismatch"
    assert "not an embedding-provider" in str(raised.value)
    assert calls == [False]


def test_illegal_instruction_can_use_generic_fallback(monkeypatch):
    calls: list[bool] = []

    def probe(*, generic: bool = False):
        calls.append(generic)
        return _result(0 if generic else -4, "Illegal instruction")

    monkeypatch.setattr(faiss_runtime, "_probe", probe)
    monkeypatch.setattr(faiss_runtime, "_version", lambda: "1.14.3")
    monkeypatch.delenv("FAISS_OPT_LEVEL", raising=False)

    assert faiss_runtime.ensure_faiss_runtime() == {
        "mode": "generic",
        "version": "1.14.3",
    }
    assert calls == [False, True]
    assert faiss_runtime.os.environ["FAISS_OPT_LEVEL"] == "generic"


def test_unrelated_import_error_does_not_try_generic(monkeypatch):
    calls: list[bool] = []

    def probe(*, generic: bool = False):
        calls.append(generic)
        return _result(1, "ModuleNotFoundError: no module named faiss")

    monkeypatch.setattr(faiss_runtime, "_probe", probe)

    with pytest.raises(faiss_runtime.FaissRuntimeError) as raised:
        faiss_runtime.ensure_faiss_runtime()

    assert raised.value.code == "dependency_load_failed"
    assert calls == [False]


def test_profile_change_only_reconfigures_an_already_loaded_faiss(monkeypatch):
    monkeypatch.delitem(faiss_runtime.sys.modules, "faiss", raising=False)
    assert faiss_runtime.configure_loaded_faiss_threads("memory") is False

    calls: list[int] = []
    fake = SimpleNamespace(omp_set_num_threads=calls.append)
    monkeypatch.setitem(faiss_runtime.sys.modules, "faiss", fake)
    monkeypatch.setattr(
        "personalityrag.resource_limits.configured_faiss_threads",
        lambda profile=None: 3 if profile == "latency" else 1,
    )
    assert faiss_runtime.configure_loaded_faiss_threads("latency") is True
    assert calls == [3]
