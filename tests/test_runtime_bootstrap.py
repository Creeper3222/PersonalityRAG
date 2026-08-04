from __future__ import annotations

import json
from pathlib import Path

from tools.runtime_bootstrap import (
    dependency_fingerprint,
    marker_is_current,
    python_supported,
    write_marker,
)


def test_python_runtime_accepts_cpython_310_or_newer_x64() -> None:
    assert python_supported((3, 10, 0), 64)
    assert python_supported((3, 11, 9), 64)
    assert python_supported((3, 12, 1), 64)
    assert python_supported((3, 13, 0), 64)
    assert not python_supported((3, 9, 18), 64)
    assert not python_supported((3, 12, 1), 32)
    assert not python_supported((3, 12, 1), 64, "pypy")


def test_dependency_fingerprint_tracks_both_manifests(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    lock = tmp_path / "requirements-runtime.lock"
    requirements.write_text("fastapi>=0.115,<1\n", encoding="utf-8")
    lock.write_text("fastapi==0.116.2\n", encoding="utf-8")

    first = dependency_fingerprint(
        lock_path=lock,
        requirements_path=requirements,
    )
    lock.write_text("fastapi==0.117.0\n", encoding="utf-8")
    second = dependency_fingerprint(
        lock_path=lock,
        requirements_path=requirements,
    )
    requirements.write_text("fastapi>=0.116,<1\n", encoding="utf-8")
    third = dependency_fingerprint(
        lock_path=lock,
        requirements_path=requirements,
    )

    assert first != second
    assert second != third


def test_dependency_marker_is_atomic_and_rejects_stale_data(
    tmp_path: Path,
) -> None:
    marker = tmp_path / ".personalityrag-dependencies.json"

    assert not marker_is_current(marker, "first")
    write_marker(marker, "first")
    assert marker_is_current(marker, "first")
    assert not marker_is_current(marker, "second")
    assert json.loads(marker.read_text(encoding="utf-8"))["format"] == 1
    assert not list(tmp_path.glob("*.tmp"))
    marker.write_text("[]", encoding="utf-8")
    assert not marker_is_current(marker, "first")


def test_runtime_lock_is_fully_pinned() -> None:
    root = Path(__file__).resolve().parents[1]
    lines = [
        line.strip()
        for line in (root / "requirements-runtime.lock").read_text(
            encoding="ascii"
        ).splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert len(lines) >= 20
    requirements = [line.split(";", 1)[0].strip() for line in lines]
    assert all("==" in requirement for requirement in requirements)
    assert not any(
        " >=" in requirement or "<" in requirement or "~=" in requirement
        for requirement in requirements
    )


def test_container_runtime_is_built_from_the_pinned_environment() -> None:
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    entrypoint = (root / "docker" / "entrypoint.sh").read_text(encoding="utf-8")

    assert "FROM python:3.12-slim AS base" in dockerfile
    assert "pip install -r requirements-runtime.lock" in dockerfile
    assert "python -m compileall -q" in dockerfile
    assert "PERSONALITYRAG_STATE_ROOT" in entrypoint
    assert "/sys/fs/cgroup/cpu.max" in entrypoint
    assert "OMP_NUM_THREADS" in entrypoint
    assert 'exec "$@"' in entrypoint
