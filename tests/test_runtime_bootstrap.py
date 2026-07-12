from __future__ import annotations

import json
from pathlib import Path

from tools.runtime_bootstrap import (
    dependency_fingerprint,
    marker_is_current,
    python_supported,
    write_marker,
)


def test_python_runtime_requires_cpython_312_x64_shape() -> None:
    assert python_supported((3, 12, 1), 64)
    assert not python_supported((3, 11, 9), 64)
    assert not python_supported((3, 12, 1), 32)
    assert not python_supported((3, 13, 0), 64)
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
    assert all("==" in line for line in lines)
    assert not any(" >=" in line or "<" in line or "~=" in line for line in lines)


def test_launcher_refuses_mismatched_existing_environment() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "launcher.bat"
    ).read_text(encoding="utf-8")

    assert "tools\\runtime_bootstrap.py validate" in source
    assert "will not delete or overwrite it" in source
    assert "requirements-runtime.lock" in source
    assert "tools\\runtime_bootstrap.py check" in source
    assert "tools\\runtime_bootstrap.py mark" in source
    assert "pip install" in source
