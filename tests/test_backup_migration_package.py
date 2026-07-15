from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pyzipper
import pytest

import personalityrag.backup_migration as backup_module
from personalityrag.backup_migration import (
    PragPackageError,
    export_prag_package,
    import_prag_package,
)
from personalityrag.config import (
    AppConfig,
    ProviderConfig,
    RuntimeResidencyConfig,
    save_config,
)
from personalityrag.libraries import LibraryManager
from personalityrag.providers import EmbeddingProvider
from personalityrag.io_utils import (
    UploadSizeLimitError,
    run_blocking,
    save_upload_file,
)


class FakeProvider(EmbeddingProvider):
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.dimension = config.dimensions or 8

    async def get_embedding(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for index, value in enumerate(text.encode("utf-8")):
            vector[index % self.dimension] += (value % 17) / 17
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector.tolist()

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [await self.get_embedding(text) for text in texts]

    async def get_dimension(self) -> int:
        return self.dimension

    async def list_models(self):
        return [{"id": self.config.model}]

    async def detect_context_length(self):
        return {
            "max_context_tokens": self.config.max_context_tokens or 4096,
            "max_context_tokens_source": "auto:fake",
        }

    async def test_connection(self):
        return {
            "available": True,
            "resolved_model": self.config.model,
            "dimension": self.dimension,
        }

    async def close(self) -> None:
        return None


def _patch_fake_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda config: FakeProvider(config),
    )
    monkeypatch.setattr(
        "personalityrag.libraries.build_provider",
        lambda config: FakeProvider(config),
    )


def _read_prag(path: Path, password: str) -> tuple[dict, dict[str, bytes]]:
    with pyzipper.AESZipFile(path, "r") as archive:
        archive.setpassword(password.encode("utf-8"))
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        files = {name: archive.read(name) for name in manifest["files"]}
    return manifest, files


def _write_fixture_prag(
    path: Path,
    files: list[tuple[str, bytes]],
    *,
    declared: dict[str, dict] | None = None,
) -> None:
    file_manifest = declared or {
        name: {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
        for name, data in files
    }
    manifest = {
        "format": backup_module.PACKAGE_FORMAT,
        "version": backup_module.PACKAGE_VERSION,
        "scope": {
            "include_libraries": False,
            "include_providers": False,
        },
        "files": file_manifest,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pyzipper.AESZipFile(
            path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            encryption=pyzipper.WZ_AES,
        ) as archive:
            archive.setpassword(b"secret-pass")
            archive.setencryption(pyzipper.WZ_AES, nbits=256)
            for name, data in files:
                archive.writestr(name, data)
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
            )


def test_provider_snapshot_context_migration_handles_legacy_and_invalid_values():
    snapshot = {
        "provider_revisions": [
            {
                "config": {
                    "type": "vllm_embedding",
                    "max_context_tokens": 8192,
                    "max_context_tokens_source": "",
                }
            },
            {
                "config": {
                    "type": "openai_embedding",
                    "context_length_mode": "manual",
                    "max_context_tokens": 0,
                    "max_context_tokens_source": "",
                }
            },
            {
                "config": {
                    "type": "vllm_embedding",
                    "max_context_tokens": 0,
                    "max_context_tokens_source": "",
                }
            },
        ]
    }

    summary = backup_module._normalize_provider_snapshot_context(snapshot)

    first, second, third = [
        item["config"] for item in snapshot["provider_revisions"]
    ]
    assert first["context_length_mode"] == "manual"
    assert first["max_context_tokens_source"] == "manual:user"
    assert second["context_length_mode"] == "manual"
    assert second["max_context_tokens"] == 512
    assert second["max_context_tokens_source"] == "manual:fallback-undetected"
    assert third["context_length_mode"] == "auto"
    assert third["max_context_tokens"] == 0
    assert summary == {
        "auto_pending": 1,
        "manual_fallback": 1,
        "legacy_manual": 1,
    }


@pytest.mark.asyncio
async def test_prag_export_requires_password_and_encrypts_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _patch_fake_provider(monkeypatch)
    root = tmp_path / "source"
    config = AppConfig(
        provider=ProviderConfig(dimensions=8),
        runtime_residency=RuntimeResidencyConfig(
            idle_minutes=47,
            max_non_default_runtimes=7,
        ),
    )
    manager = LibraryManager(root, config)
    await manager.initialize()
    try:
        runtime = await manager.get_runtime("Default")
        await runtime.create_memory(
            {
                "content": "PersonalityRAG 配置包会保存核心记忆库快照。",
                "topics": ["备份迁移"],
            }
        )
        await manager.create_library(
            {
                "id": "empty_library",
                "name": "Empty Library",
                "provider_id": config.provider.id,
            }
        )
        await manager.rebuild_library("Default", None)
        target = tmp_path / "export.prag"
        with pytest.raises(PragPackageError, match="配置验证密码"):
            await export_prag_package(
                root=root,
                config=config,
                manager=manager,
                target=target,
                password="",
                include_libraries=True,
                include_providers=True,
            )

        result = await export_prag_package(
            root=root,
            config=config,
            manager=manager,
            target=target,
            password="secret-pass",
            include_libraries=True,
            include_providers=True,
        )
        assert result["size_bytes"] > 0
        with pyzipper.AESZipFile(target, "r") as archive:
            archive.setpassword(b"wrong-pass")
            with pytest.raises(RuntimeError):
                archive.read("manifest.json")

        manifest, files = _read_prag(target, "secret-pass")
        names = set(files)
        assert manifest["scope"] == {
            "include_libraries": True,
            "include_providers": True,
        }
        assert "config/global.json" in names
        assert "providers/providers.json" in names
        assert not any("/indexes/" in name or name.endswith("/CURRENT") for name in names)
        non_empty = [
            item for item in manifest["libraries"] if item["id"] == "Default"
        ][0]
        empty = [
            item for item in manifest["libraries"] if item["id"] == "empty_library"
        ][0]
        assert non_empty["files"]["livingmemory_db"] in names
        assert non_empty["files"]["conversations_db"] in names
        assert empty["empty"] is True
        assert "livingmemory_db" not in empty["files"]
        global_config = json.loads(files["config/global.json"].decode("utf-8"))
        assert "api_key" in global_config
        assert "provider" not in global_config
        assert global_config["runtime_residency"] == {
            "idle_minutes": 47,
            "max_non_default_runtimes": 7,
        }
        provider_snapshot = json.loads(files["providers/providers.json"].decode("utf-8"))
        assert "context_length_table" in provider_snapshot
        provider_config = provider_snapshot["provider_revisions"][0]["config"]
        assert provider_config["context_length_mode"] in {"auto", "manual"}
        assert "max_context_tokens" in provider_config
        assert "max_context_tokens_source" in provider_config
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_prag_import_restores_libraries_without_rebuilding_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _patch_fake_provider(monkeypatch)
    source_root = tmp_path / "source"
    source_config = AppConfig(
        provider=ProviderConfig(dimensions=8),
        runtime_residency=RuntimeResidencyConfig(
            idle_minutes=61,
            max_non_default_runtimes=9,
        ),
    )
    source_manager = LibraryManager(source_root, source_config)
    await source_manager.initialize()
    try:
        runtime = await source_manager.get_runtime("Default")
        await runtime.create_memory({"content": "导入后索引必须保持未构建。"})
        await source_manager.rebuild_library("Default", None)
        package = tmp_path / "full.prag"
        await export_prag_package(
            root=source_root,
            config=source_config,
            manager=source_manager,
            target=package,
            password="secret-pass",
            include_libraries=True,
            include_providers=True,
        )
    finally:
        await source_manager.close()

    target_root = tmp_path / "target"
    target_config = AppConfig(
        api_key="prag_target",
        provider=ProviderConfig(id="target_provider", dimensions=8),
    )
    target_manager = LibraryManager(target_root, target_config)
    await target_manager.initialize()
    try:
        config_path = target_root / "config" / "config.json"
        save_config(config_path, target_config)
        next_config, result = await import_prag_package(
            root=target_root,
            config_path=config_path,
            config=target_config,
            manager=target_manager,
            package_path=package,
            password="secret-pass",
        )
        assert next_config.api_key == source_config.api_key
        assert next_config.runtime_residency == source_config.runtime_residency
        persisted_config = json.loads(config_path.read_text(encoding="utf-8"))
        assert persisted_config["runtime_residency"] == {
            "idle_minutes": 61,
            "max_non_default_runtimes": 9,
        }
        assert result["indexes_pending"] is True
        assert result["libraries_imported"] == 1
        assert result["providers_imported"] >= 1
        default_library = await target_manager.control.default_library()
        assert default_library.id == "Default"
        libraries = await target_manager.list_libraries()
        imported = [item for item in libraries if item["id"] == "Default"][0]
        assert imported["stats"]["total_memories"] == 1
        assert imported["indexes"]["generation"] is None
        assert imported["indexes"]["document_vectors"] == 0
        assert imported["indexes"]["graph_vectors"] == 0
        assert not (target_root / "data" / "libraries" / "Default" / "indexes").exists()
        assert not await target_manager.control.has_any_running_jobs()
    finally:
        await target_manager.close()


@pytest.mark.asyncio
async def test_prag_import_reports_missing_provider_when_scope_excludes_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _patch_fake_provider(monkeypatch)
    source_root = tmp_path / "source"
    source_config = AppConfig(provider=ProviderConfig(dimensions=8))
    source_manager = LibraryManager(source_root, source_config)
    await source_manager.initialize()
    try:
        package = tmp_path / "libraries-only.prag"
        await export_prag_package(
            root=source_root,
            config=source_config,
            manager=source_manager,
            target=package,
            password="secret-pass",
            include_libraries=True,
            include_providers=False,
        )
    finally:
        await source_manager.close()

    target_root = tmp_path / "target"
    target_config = AppConfig(provider=ProviderConfig(id="other_provider", dimensions=8))
    target_manager = LibraryManager(target_root, target_config)
    await target_manager.initialize()
    try:
        next_config, result = await import_prag_package(
            root=target_root,
            config_path=target_root / "config" / "config.json",
            config=target_config,
            manager=target_manager,
            package_path=package,
            password="secret-pass",
        )
        assert next_config.provider.id == "other_provider"
        assert result["providers_imported"] == 0
        assert result["missing_provider_ids"] == [source_config.provider.id]
        library = await target_manager.control.get_library("Default")
        assert library is not None
        assert library.provider_id == source_config.provider.id
    finally:
        await target_manager.close()


@pytest.mark.asyncio
async def test_prag_import_rolls_back_after_restore_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _patch_fake_provider(monkeypatch)
    source_root = tmp_path / "source"
    source_config = AppConfig(provider=ProviderConfig(dimensions=8))
    source_manager = LibraryManager(source_root, source_config)
    await source_manager.initialize()
    try:
        package = tmp_path / "full.prag"
        await export_prag_package(
            root=source_root,
            config=source_config,
            manager=source_manager,
            target=package,
            password="secret-pass",
            include_libraries=True,
            include_providers=True,
        )
    finally:
        await source_manager.close()

    manifest, files = _read_prag(package, "secret-pass")
    providers = json.loads(files["providers/providers.json"].decode("utf-8"))
    providers["providers"][0]["latest_revision"] = 999
    files["providers/providers.json"] = json.dumps(
        providers, ensure_ascii=False, indent=2
    ).encode("utf-8")
    manifest["files"]["providers/providers.json"] = {
        "sha256": __import__("hashlib")
        .sha256(files["providers/providers.json"])
        .hexdigest(),
        "size": len(files["providers/providers.json"]),
    }
    broken = tmp_path / "broken.prag"
    with pyzipper.AESZipFile(
        broken,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        encryption=pyzipper.WZ_AES,
    ) as archive:
        archive.setpassword(b"secret-pass")
        archive.setencryption(pyzipper.WZ_AES, nbits=256)
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )

    target_root = tmp_path / "target"
    target_config = AppConfig(
        api_key="prag_original",
        provider=ProviderConfig(id="original_provider", dimensions=8),
    )
    target_manager = LibraryManager(target_root, target_config)
    await target_manager.initialize()
    try:
        config_path = target_root / "config" / "config.json"
        save_config(config_path, target_config)
        with pytest.raises(ValueError, match="missing latest revision"):
            await import_prag_package(
                root=target_root,
                config_path=config_path,
                config=target_config,
                manager=target_manager,
                package_path=broken,
                password="secret-pass",
            )
        restored = json.loads(config_path.read_text(encoding="utf-8"))
        assert restored["api_key"] == "prag_original"
        assert await target_manager.control.get_provider("original_provider")
        assert await target_manager.control.get_library("Default")
    finally:
        await target_manager.close()


@pytest.mark.asyncio
async def test_prag_import_rolls_back_after_disk_full_during_library_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_provider(monkeypatch)
    source_root = tmp_path / "source-disk-full"
    source_config = AppConfig(provider=ProviderConfig(dimensions=8))
    source_manager = LibraryManager(source_root, source_config)
    await source_manager.initialize()
    try:
        source_runtime = await source_manager.get_runtime("Default")
        await source_runtime.create_memory({"content": "new package memory"})
        package = tmp_path / "disk-full-import.prag"
        await export_prag_package(
            root=source_root,
            config=source_config,
            manager=source_manager,
            target=package,
            password="secret-pass",
            include_libraries=True,
            include_providers=True,
        )
    finally:
        await source_manager.close()

    target_root = tmp_path / "target-disk-full"
    target_config = AppConfig(
        api_key="prag_original_disk_full",
        provider=ProviderConfig(id="original_provider", dimensions=8),
    )
    target_manager = LibraryManager(target_root, target_config)
    await target_manager.initialize()
    try:
        target_runtime = await target_manager.get_runtime("Default")
        original_memory = await target_runtime.create_memory(
            {"content": "original memory survives rollback"}
        )
        config_path = target_root / "config" / "config.json"
        save_config(config_path, target_config)
        original_sqlite_backup = backup_module.sqlite_backup

        def disk_full_during_install(source: Path, target: Path) -> None:
            libraries_root = target_manager.data_dir / "libraries"
            if (
                target.name == "livingmemory.db"
                and libraries_root in target.parents
            ):
                raise OSError(28, "No space left on device")
            original_sqlite_backup(source, target)

        monkeypatch.setattr(
            backup_module,
            "sqlite_backup",
            disk_full_during_install,
        )
        with pytest.raises(OSError, match="No space left"):
            await import_prag_package(
                root=target_root,
                config_path=config_path,
                config=target_config,
                manager=target_manager,
                package_path=package,
                password="secret-pass",
            )

        restored_config = json.loads(config_path.read_text(encoding="utf-8"))
        assert restored_config["api_key"] == "prag_original_disk_full"
        restored_runtime = await target_manager.get_runtime("Default")
        restored_memory = await restored_runtime.storage.get_document(
            int(original_memory["id"])
        )
        assert restored_memory is not None
        assert restored_memory["text"] == "original memory survives rollback"
        assert await target_manager.control.get_provider("original_provider")
    finally:
        await target_manager.close()


@pytest.mark.parametrize(
    ("files", "declared", "message"),
    [
        (
            [("../escape.txt", b"escape")],
            None,
            "invalid package member path",
        ),
        (
            [("config/global.json", b"{}"), ("extra.txt", b"extra")],
            {
                "config/global.json": {
                    "sha256": hashlib.sha256(b"{}").hexdigest(),
                    "size": 2,
                }
            },
            "未声明文件",
        ),
        (
            [("config/global.json", b"{}"), ("config/global.json", b"{}")],
            None,
            "重复成员",
        ),
        (
            [("config/global.json", b"{}")],
            {
                "config/global.json": {
                    "sha256": hashlib.sha256(b"{}").hexdigest(),
                    "size": 3,
                }
            },
            "声明尺寸不一致",
        ),
        (
            [("config/global.json", b"{}")],
            {
                "config/global.json": {
                    "sha256": "0" * 64,
                    "size": 2,
                }
            },
            "文件校验失败",
        ),
    ],
)
def test_prag_extraction_rejects_unsafe_or_inconsistent_members(
    tmp_path: Path,
    files: list[tuple[str, bytes]],
    declared: dict[str, dict] | None,
    message: str,
) -> None:
    package = tmp_path / "unsafe.prag"
    _write_fixture_prag(package, files, declared=declared)

    with pytest.raises(PragPackageError, match=message):
        backup_module._extract_verified_package(
            package,
            "secret-pass",
            tmp_path / "extract",
        )
    assert not (tmp_path / "escape.txt").exists()


def test_prag_extraction_streams_large_members_without_archive_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (b"personalityrag-streaming-fixture-" * 200_000)[:5_000_000]
    package = tmp_path / "streaming.prag"
    _write_fixture_prag(package, [("config/global.json", payload)])

    def forbidden_read(*args, **kwargs):
        raise AssertionError("AESZipFile.read must not be used during extraction")

    monkeypatch.setattr(pyzipper.AESZipFile, "read", forbidden_read)
    target = tmp_path / "extract"
    manifest = backup_module._extract_verified_package(
        package,
        "secret-pass",
        target,
    )

    extracted = target / "config" / "global.json"
    assert extracted.stat().st_size == len(payload)
    assert backup_module.sha256_file(extracted) == manifest["files"][
        "config/global.json"
    ]["sha256"]


def test_prag_extraction_rejects_cumulative_uncompressed_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "compressed-limit.prag"
    _write_fixture_prag(
        package,
        [("config/global.json", b"x" * 4096)],
    )
    monkeypatch.setattr(backup_module, "MAX_EXTRACTED_BYTES", 1024)

    with pytest.raises(PragPackageError, match="解压后总大小"):
        backup_module._extract_verified_package(
            package,
            "secret-pass",
            tmp_path / "extract",
        )


def test_prag_extraction_propagates_disk_full_before_state_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "disk-full.prag"
    _write_fixture_prag(package, [("config/global.json", b"{}")])
    target = tmp_path / "extract"
    original_open = Path.open

    def disk_full_open(path: Path, mode="r", *args, **kwargs):
        if path == target / "config" / "global.json" and mode == "xb":
            raise OSError(28, "No space left on device")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", disk_full_open)
    with pytest.raises(OSError, match="No space left"):
        backup_module._extract_verified_package(
            package,
            "secret-pass",
            target,
        )


@pytest.mark.asyncio
async def test_interrupted_upload_removes_partial_file(tmp_path: Path) -> None:
    class InterruptedUpload:
        def __init__(self):
            self.reads = 0
            self.closed = False

        async def read(self, size: int) -> bytes:
            self.reads += 1
            if self.reads == 1:
                return b"partial"
            raise asyncio.CancelledError()

        async def close(self) -> None:
            self.closed = True

    upload = InterruptedUpload()
    target = tmp_path / "partial.prag"
    with pytest.raises(asyncio.CancelledError):
        await save_upload_file(upload, target)

    assert upload.closed is True
    assert not target.exists()


@pytest.mark.asyncio
async def test_oversized_upload_removes_partial_file(tmp_path: Path) -> None:
    class OversizedUpload:
        def __init__(self):
            self.sent = False
            self.closed = False

        async def read(self, size: int) -> bytes:
            if self.sent:
                return b""
            self.sent = True
            return b"too-large"

        async def close(self) -> None:
            self.closed = True

    upload = OversizedUpload()
    target = tmp_path / "oversized.prag"
    with pytest.raises(UploadSizeLimitError):
        await save_upload_file(upload, target, max_bytes=4)

    assert upload.closed is True
    assert not target.exists()


@pytest.mark.asyncio
async def test_cancelled_blocking_io_waits_for_worker_before_cleanup() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_write() -> None:
        started.set()
        release.wait(timeout=2)

    task = asyncio.create_task(run_blocking(blocking_write))
    await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0.02)
    assert task.done() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
