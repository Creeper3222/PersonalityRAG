from __future__ import annotations

import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

from personalityrag.update_manifest import (
    MANAGED_DIRECTORIES,
    MANAGED_FILES,
    MANIFEST_NAME,
    UpdatePackageError,
    compare_tags,
    inspect_and_extract_zip,
    validate_manifest,
    write_manifest,
)
from personalityrag.updates import UpdateService
from tools import update_helper


def _candidate(tmp_path: Path, *, version: str = "0.1.0") -> Path:
    root = tmp_path / "PersonalityRAG"
    for directory in MANAGED_DIRECTORIES:
        path = root / directory
        path.mkdir(parents=True)
        (path / "module.txt").write_text(directory, encoding="utf-8")
    for relative in MANAGED_FILES:
        path = root / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    (root / "personalityrag" / "version.py").write_text(f'VERSION = "{version}"\n', encoding="utf-8")
    write_manifest(root, version=version, tag_name=f"v{version}")
    return root


def _zip(root: Path, target: Path) -> Path:
    with zipfile.ZipFile(target, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root.parent).as_posix())
    return target


def test_semver_comparison_includes_prereleases() -> None:
    assert compare_tags("v0.1.1", "v0.1.0") > 0
    assert compare_tags("v0.1.1-rc.2", "v0.1.1-rc.1") > 0
    assert compare_tags("v0.1.1", "v0.1.1-rc.2") > 0
    assert compare_tags("v0.1.0", "v0.1.0") == 0
    with pytest.raises(UpdatePackageError):
        compare_tags("latest", "v0.1.0")


def test_runtime_version_is_centralized_in_python_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    from personalityrag.version import VERSION

    assert "v0.1.0" not in (root / "run.py").read_text(encoding="utf-8")
    assert "v0.1.0 launcher" not in (root / "launcher.bat").read_text(encoding="utf-8")
    assert 'id="sidebar-version"' in (root / "static" / "index.html").read_text(encoding="utf-8")
    assert f'version = "{VERSION}"' in (root / "pyproject.toml").read_text(encoding="utf-8")


def test_manifest_validates_every_managed_file(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    validate_manifest(manifest, expected_tag="v0.1.0", candidate_root=root)
    (root / "static" / "module.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(UpdatePackageError, match="(size|hash) mismatch"):
        validate_manifest(manifest, expected_tag="v0.1.0", candidate_root=root)


def test_manifest_blocks_incompatible_persistent_schema(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["persistent_compatibility"]["livingmemory_database"] = {"minimum": 9, "maximum": 9}
    with pytest.raises(UpdatePackageError, match="incompatible with livingmemory_database"):
        validate_manifest(manifest, expected_tag="v0.1.0")


def test_manifest_rejects_internal_runtime_version_mismatch(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    (root / "personalityrag" / "version.py").write_text('VERSION = "9.9.9"\n', encoding="utf-8")
    for entry in manifest["files"]:
        if entry["path"] == "personalityrag/version.py":
            from personalityrag.update_manifest import sha256_file

            entry["size"] = (root / entry["path"]).stat().st_size
            entry["sha256"] = sha256_file(root / entry["path"])
    with pytest.raises(UpdatePackageError, match="internal runtime version"):
        validate_manifest(manifest, expected_tag="v0.1.0", candidate_root=root)


def test_zip_requires_single_root_and_rejects_path_traversal(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("PersonalityRAG/../config/config.json", "secret")
    with pytest.raises(UpdatePackageError, match="unsafe path"):
        inspect_and_extract_zip(bad, tmp_path / "out", expected_tag="v0.1.0")


def test_zip_rejects_symlink_and_protected_content(tmp_path: Path) -> None:
    bad = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("PersonalityRAG/personalityrag/link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr(info, "target")
    with pytest.raises(UpdatePackageError, match="symbolic link"):
        inspect_and_extract_zip(bad, tmp_path / "out", expected_tag="v0.1.0")

    protected = tmp_path / "protected.zip"
    with zipfile.ZipFile(protected, "w") as archive:
        archive.writestr("PersonalityRAG/data/library.db", "data")
    with pytest.raises(UpdatePackageError, match="protected path"):
        inspect_and_extract_zip(protected, tmp_path / "out2", expected_tag="v0.1.0")


def test_valid_zip_round_trip(tmp_path: Path) -> None:
    archive = _zip(_candidate(tmp_path / "build"), tmp_path / "PersonalityRAG-v0.1.0.zip")
    root, manifest = inspect_and_extract_zip(archive, tmp_path / "extract", expected_tag="v0.1.0")
    assert root.name == "PersonalityRAG"
    assert manifest["version"] == "0.1.0"


def test_release_filter_requires_exact_windows_asset() -> None:
    base = {
        "tag_name": "v0.1.1",
        "draft": False,
        "prerelease": True,
        "assets": [{"name": "PersonalityRAG-v0.1.1.zip", "browser_download_url": "https://github.com/Creeper3222/PersonalityRAG/releases/download/v0.1.1/PersonalityRAG-v0.1.1.zip", "size": 12, "digest": "sha256:" + "a" * 64}],
    }
    assert UpdateService._normalize_release(base)["tag_name"] == "v0.1.1"
    assert UpdateService._normalize_release({**base, "draft": True}) is None
    assert UpdateService._normalize_release({**base, "assets": []}) is None
    assert UpdateService._normalize_release({**base, "tag_name": "next"}) is None
    poisoned = {**base, "assets": [{**base["assets"][0], "browser_download_url": "https://example.test/file.zip"}]}
    assert UpdateService._normalize_release(poisoned) is None


@pytest.mark.asyncio
async def test_release_check_keeps_last_successful_cache_on_network_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = UpdateService(tmp_path / "source", tmp_path / "state")
    service._write_cache({"checked_at": 1.0, "stale": False, "error": "", "releases": []})

    def fail(_: str):
        raise OSError("offline")

    monkeypatch.setattr(service, "_request_json", fail)
    monkeypatch.setattr(service, "_fallback_release_feed", lambda: (_ for _ in ()).throw(OSError("offline")))
    payload = await service.releases(refresh=True)
    assert payload["stale"] is True
    assert payload["releases"] == []


@pytest.mark.asyncio
async def test_release_check_uses_public_feed_when_api_quota_is_exhausted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = UpdateService(tmp_path / "source", tmp_path / "state")

    def api_fail(_: str):
        raise OSError("quota")

    fallback = [{
        "tag_name": "v0.1.1", "version": "0.1.1", "name": "test", "published_at": "2026-01-01T00:00:00Z",
        "prerelease": True, "notes": "test", "html_url": "https://example.test",
        "asset": {"name": "PersonalityRAG-v0.1.1.zip", "url": "https://example.test/file.zip", "size": 0, "digest": "sha256:" + "a" * 64},
    }]
    monkeypatch.setattr(service, "_request_json", api_fail)
    monkeypatch.setattr(service, "_fallback_release_feed", lambda: fallback)
    payload = await service.releases(refresh=True)
    assert payload["stale"] is False
    assert payload["releases"][0]["tag_name"] == "v0.1.1"


def test_transaction_public_payload_does_not_leak_paths(tmp_path: Path) -> None:
    service = UpdateService(tmp_path / "source", tmp_path / "state")
    public = service._public_transaction({
        "transaction_id": "abc",
        "status": "prepared",
        "source_root": "C:/secret",
        "candidate_root": "C:/secret/candidate",
    })
    assert public == {"transaction_id": "abc", "status": "prepared"}


def test_update_helper_restores_exact_managed_snapshot(tmp_path: Path) -> None:
    source = _candidate(tmp_path / "source-parent")
    backup = tmp_path / "backup"
    update_helper._backup(source, backup)
    (source / "personalityrag" / "module.txt").write_text("new", encoding="utf-8")
    (source / "static" / "extra.txt").write_text("extra", encoding="utf-8")
    (source / "data").mkdir()
    (source / "data" / "protected.txt").write_text("keep", encoding="utf-8")
    update_helper._restore(source, backup)
    assert (source / "personalityrag" / "module.txt").read_text(encoding="utf-8") == "personalityrag"
    assert not (source / "static" / "extra.txt").exists()
    assert (source / "data" / "protected.txt").read_text(encoding="utf-8") == "keep"


def test_startup_recovery_rolls_back_incomplete_transaction(tmp_path: Path) -> None:
    source = _candidate(tmp_path / "source-parent")
    state = tmp_path / "state"
    transaction_root = state / "data" / "update" / "transactions" / "abc"
    backup = transaction_root / "backup"
    transaction_root.mkdir(parents=True)
    update_helper._backup(source, backup)
    (source / "run.py").write_text("broken", encoding="utf-8")
    transaction = {
        "transaction_id": "abc",
        "status": "running",
        "stage": "replacing",
        "source_root": str(source),
    }
    transaction_file = transaction_root / "transaction.json"
    transaction_file.write_text(json.dumps(transaction), encoding="utf-8")
    assert update_helper.recover_transactions(state) == 1
    assert (source / "run.py").read_text(encoding="utf-8") == "run.py"
    assert json.loads(transaction_file.read_text(encoding="utf-8"))["stage"] == "startup_recovered"


def test_startup_recovery_marks_pre_replace_crash_safe_without_backup(tmp_path: Path) -> None:
    state = tmp_path / "state"
    transaction_root = state / "data" / "update" / "transactions" / "abc"
    transaction_root.mkdir(parents=True)
    transaction_file = transaction_root / "transaction.json"
    transaction_file.write_text(json.dumps({
        "transaction_id": "abc",
        "status": "running",
        "stage": "helper_started",
        "source_root": str(tmp_path / "source"),
    }), encoding="utf-8")
    assert update_helper.recover_transactions(state) == 0
    payload = json.loads(transaction_file.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["stage"] == "recovery_not_required"


def test_update_helper_pid_detection() -> None:
    assert update_helper._pid_is_running(os.getpid()) is True
    assert update_helper._pid_is_running(-1) is False
    assert update_helper._pid_is_running(2_147_483_647) is False
