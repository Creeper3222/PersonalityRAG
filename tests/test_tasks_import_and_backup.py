from __future__ import annotations

import asyncio
import os
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

from personalityrag.config import AppConfig, MaintenanceConfig, ProviderConfig, RecallConfig
from personalityrag.graph import GraphBuilder
from personalityrag.jobs import JobManager
from personalityrag.libraries import LibraryManager
from personalityrag.providers import EmbeddingProvider
from personalityrag.service import PersonalityRAGService
from personalityrag.storage import Storage
from personalityrag.text import TextProcessor


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


async def _clear_graph_tables(storage: Storage) -> None:
    async with storage.connect() as db:
        await db.execute("DELETE FROM livingmemory_graph_entries_fts")
        await db.execute("DELETE FROM graph_entry_nodes")
        await db.execute("DELETE FROM graph_entries")
        await db.execute("DELETE FROM graph_edges")
        await db.execute("DELETE FROM graph_nodes")
        await db.commit()


@pytest.mark.asyncio
async def test_job_manager_runs_tasks_fifo_and_clears_finished_history_on_restart(
    tmp_path: Path,
):
    storage = Storage(tmp_path)
    await storage.initialize()
    jobs = JobManager(storage)
    await jobs.clear_for_startup()
    release_first = asyncio.Event()
    second_started = asyncio.Event()

    async def first(progress):
        await progress(0.25, "first running")
        await release_first.wait()
        return {"order": 1}

    async def second(progress):
        second_started.set()
        await progress(0.5, "second running")
        return {"order": 2}

    first_id = await jobs.start("index_rebuild", first, library_id="lib")
    second_id = await jobs.start("library_copy", second, library_id="lib2")
    await asyncio.sleep(0.1)
    first_job = await jobs.get(first_id)
    second_job = await jobs.get(second_id)
    assert first_job and first_job["status"] == "running"
    assert second_job and second_job["status"] == "queued"
    assert not second_started.is_set()

    release_first.set()
    first_result, second_result = await asyncio.gather(
        asyncio.wait_for(jobs.wait(first_id), timeout=5),
        asyncio.wait_for(jobs.wait(second_id), timeout=5),
    )
    assert first_result["status"] == "completed"
    assert second_result["status"] == "completed"

    active = await jobs.list(scope="active")
    finished = await jobs.list(scope="finished")
    assert active == []
    assert [item["id"] for item in finished] == [second_id, first_id]
    await jobs.clear_for_startup()
    retained = await jobs.list(scope="all")
    assert retained == []
    await jobs.close()


@pytest.mark.asyncio
async def test_backup_contains_livingmemory_and_conversations_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "PersonalityRAG"
    provider_config = ProviderConfig(dimensions=8)
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda config: FakeProvider(config),
    )
    monkeypatch.setattr(
        "personalityrag.libraries.build_provider",
        lambda config: FakeProvider(config),
    )
    manager = LibraryManager(root, AppConfig(provider=provider_config))
    await manager.initialize()
    try:
        result = await manager.backup_library("Default")
        backup = Path(result["path"])
        assert backup.exists()
        assert sorted(item.name for item in backup.iterdir()) == [
            "conversations.db",
            "livingmemory.db",
        ]
        for db_name in ("livingmemory.db", "conversations.db"):
            con = sqlite3.connect(backup / db_name)
            try:
                assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            finally:
                con.close()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_library_disallows_import_after_first_memory_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "PersonalityRAG"
    source_dir = tmp_path / "source_livingmemory"
    source_storage = Storage(source_dir)
    await source_storage.initialize()
    await source_storage.create_memory(
        {
            "content": "用于校验导入文件的基准记忆。",
            "persona_id": "Default",
            "topics": ["导入"],
        },
        TextProcessor().tokenize,
        GraphBuilder().build,
    )
    upload_db = tmp_path / "upload_after_write" / "livingmemory.db"
    upload_db.parent.mkdir(parents=True)
    shutil.copy2(source_dir / "livingmemory.db", upload_db)

    provider_config = ProviderConfig(dimensions=8)
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda config: FakeProvider(config),
    )
    monkeypatch.setattr(
        "personalityrag.libraries.build_provider",
        lambda config: FakeProvider(config),
    )
    manager = LibraryManager(root, AppConfig(provider=provider_config))
    await manager.initialize()
    try:
        created = await manager.create_library(
            {
                "id": "fresh_import_target",
                "name": "空库导入目标",
                "provider_id": provider_config.id,
            }
        )
        library_id = created["id"]
        assert await manager.library_is_empty(library_id) is True

        runtime = await manager.get_runtime(library_id)
        await runtime.create_memory(
            {
                "content": "适配器写入后的第一条正式记忆。",
                "persona_id": "Astrbot",
                "session_id": "astrbot:group:test",
                "topics": ["写入"],
            }
        )

        assert (await runtime.storage.statistics())["total_memories"] == 1
        assert await manager.library_is_empty(library_id) is False

        with pytest.raises(ValueError, match="只有全新空记忆库"):
            await manager.import_livingmemory_db(library_id, upload_db)
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_rebuild_library_recovers_missing_graph_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "PersonalityRAG"
    provider_config = ProviderConfig(dimensions=8)
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda config: FakeProvider(config),
    )
    monkeypatch.setattr(
        "personalityrag.libraries.build_provider",
        lambda config: FakeProvider(config),
    )
    manager = LibraryManager(root, AppConfig(provider=provider_config))
    await manager.initialize()
    try:
        runtime = await manager.get_runtime("Default")
        await runtime.create_memory(
            {
                "content": "贝雷特需要能从单文件恢复图记忆",
                "persona_id": "Default",
                "topics": ["安全恢复"],
                "key_facts": ["单文件恢复必须回填图记忆"],
            }
        )
        await _clear_graph_tables(runtime.storage)
        assert (await runtime.storage.graph_integrity_report())["graph_entries"] == 0

        result = await manager.rebuild_library("Default", None)
        stats = await runtime.storage.statistics()

        assert result["graph_recovery"]["rebuilt"] is True
        assert stats["graph_entries"] > 0
        assert runtime.indexes.status()["document_vectors"] == stats["total_memories"]
        assert runtime.indexes.status()["graph_vectors"] == stats["graph_entries"]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_rebuild_graph_forces_graph_entry_backfill_and_index_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "PersonalityRAG"
    provider_config = ProviderConfig(dimensions=8)
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda config: FakeProvider(config),
    )
    monkeypatch.setattr(
        "personalityrag.libraries.build_provider",
        lambda config: FakeProvider(config),
    )
    manager = LibraryManager(root, AppConfig(provider=provider_config))
    await manager.initialize()
    try:
        runtime = await manager.get_runtime("Default")
        await runtime.create_memory(
            {
                "content": "Beileite validates graph memory rebuild recovery.",
                "persona_id": "Default",
                "topics": ["restore"],
                "key_facts": ["Graph entries can be rebuilt from documents."],
            }
        )
        await _clear_graph_tables(runtime.storage)

        result = await manager.rebuild_graph("Default")
        stats = await runtime.storage.statistics()

        assert result["graph"]["graph"]["rebuilt_documents"] == 1
        assert result["graph"]["before"]["graph_entries"] == 0
        assert result["graph"]["after"]["graph_entries"] > 0
        assert stats["graph_entries"] > 0
        assert result["manifest"]["document_count"] == stats["total_memories"]
        assert result["manifest"]["graph_entry_count"] == stats["graph_entries"]
        assert runtime.indexes.status()["graph_vectors"] == stats["graph_entries"]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_daily_maintenance_applies_access_aware_decay_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    provider_config = ProviderConfig(dimensions=8)
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda config: FakeProvider(config),
    )
    service = PersonalityRAGService(
        tmp_path,
        AppConfig(
            provider=provider_config,
            recall=RecallConfig(
                decay_rate=0.1,
                access_decay_window_days=30,
                access_decay_max_count=10,
                access_count_decay_multiplier=0.5,
            ),
            maintenance=MaintenanceConfig(
                backup_enabled=False,
                auto_cleanup_enabled=False,
            ),
        ),
        data_dir=tmp_path / "library",
        library_id="test",
    )
    await service.initialize()
    try:
        now = time.time()
        memory_id = await service.storage.create_memory(
            {"content": "高访问记忆", "importance": 1.0},
            service.text.tokenize,
            service.graph_builder.build,
        )
        await service.storage.update_memory(
            memory_id,
            {
                "metadata": {
                    "importance": 1.0,
                    "access_count": 10,
                    "last_access_time": now,
                    "create_time": now - 3 * 86400,
                }
            },
            service.text.tokenize,
            service.graph_builder.build,
        )

        result = await service.run_maintenance()
        assert result["daily_maintenance_ran"] is True
        updated = await service.storage.get_document(memory_id)
        assert updated is not None
        assert updated["metadata"]["importance"] == pytest.approx(0.95)
        assert updated["metadata"]["access_count"] == 5

        second_result = await service.run_maintenance()
        assert second_result["daily_maintenance_ran"] is False
        unchanged = await service.storage.get_document(memory_id)
        assert unchanged is not None
        assert unchanged["metadata"]["importance"] == pytest.approx(0.95)
        assert unchanged["metadata"]["access_count"] == 5
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_daily_maintenance_cleanup_defaults_off_and_enabled_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    provider_config = ProviderConfig(dimensions=8)
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda config: FakeProvider(config),
    )

    default_service = PersonalityRAGService(
        tmp_path,
        AppConfig(
            provider=provider_config,
            recall=RecallConfig(decay_rate=0),
            maintenance=MaintenanceConfig(backup_enabled=False),
        ),
        data_dir=tmp_path / "default_library",
        library_id="default",
    )
    await default_service.initialize()
    try:
        old_low = await default_service.storage.create_memory(
            {"content": "默认不清理的旧低重要性记忆"},
            default_service.text.tokenize,
            default_service.graph_builder.build,
        )
        await default_service.storage.update_memory(
            old_low,
            {
                "metadata": {
                    "importance": 0.1,
                    "create_time": time.time() - 30 * 86400,
                }
            },
            default_service.text.tokenize,
            default_service.graph_builder.build,
        )
        result = await default_service.run_maintenance()
        assert result["cleaned_memories"] == 0
        assert await default_service.storage.get_document(old_low) is not None
    finally:
        await default_service.close()

    cleanup_service = PersonalityRAGService(
        tmp_path,
        AppConfig(
            provider=provider_config,
            recall=RecallConfig(decay_rate=0),
            maintenance=MaintenanceConfig(
                backup_enabled=False,
                auto_cleanup_enabled=True,
                cleanup_days_threshold=7,
                cleanup_importance_threshold=0.3,
            ),
        ),
        data_dir=tmp_path / "cleanup_library",
        library_id="cleanup",
    )
    await cleanup_service.initialize()
    try:
        old_low = await cleanup_service.storage.create_memory(
            {"content": "应该清理的旧低重要性记忆"},
            cleanup_service.text.tokenize,
            cleanup_service.graph_builder.build,
        )
        old_high = await cleanup_service.storage.create_memory(
            {"content": "应该保留的旧高重要性记忆"},
            cleanup_service.text.tokenize,
            cleanup_service.graph_builder.build,
        )
        old_time = time.time() - 30 * 86400
        for memory_id, importance in ((old_low, 0.1), (old_high, 0.8)):
            await cleanup_service.storage.update_memory(
                memory_id,
                {
                    "metadata": {
                        "importance": importance,
                        "create_time": old_time,
                    }
                },
                cleanup_service.text.tokenize,
                cleanup_service.graph_builder.build,
            )
        await cleanup_service.rebuild_indexes()

        result = await cleanup_service.run_maintenance()
        assert result["cleaned_memories"] == 1
        assert await cleanup_service.storage.get_document(old_low) is None
        assert await cleanup_service.storage.get_document(old_high) is not None
    finally:
        await cleanup_service.close()


@pytest.mark.asyncio
async def test_daily_maintenance_backup_retention_and_daily_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    provider_config = ProviderConfig(dimensions=8)
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda config: FakeProvider(config),
    )
    service = PersonalityRAGService(
        tmp_path,
        AppConfig(
            provider=provider_config,
            recall=RecallConfig(decay_rate=0),
            maintenance=MaintenanceConfig(
                backup_enabled=True,
                backup_keep_days=7,
                auto_cleanup_enabled=False,
            ),
        ),
        data_dir=tmp_path / "backup_library",
        library_id="backup",
    )
    await service.initialize()
    try:
        old_backup = service.data_dir / "backups" / "old-backup"
        old_backup.mkdir(parents=True)
        old_timestamp = time.time() - 10 * 86400
        os.utime(old_backup, (old_timestamp, old_timestamp))

        result = await service.run_maintenance()
        assert result["daily_maintenance_ran"] is True
        assert result["backup"]
        assert result["removed_backups"] == 1
        assert not old_backup.exists()

        second_result = await service.run_maintenance()
        assert second_result["daily_maintenance_ran"] is False
        assert second_result["backup"] is None
        assert second_result["removed_backups"] == 0
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_import_livingmemory_db_requires_empty_library_and_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "PersonalityRAG"
    source_dir = tmp_path / "source_livingmemory"
    source_storage = Storage(source_dir)
    await source_storage.initialize()
    memory_id = await source_storage.create_memory(
        {
            "content": "贝雷特记得澄月喜欢星空",
            "persona_id": "Default",
            "topics": ["星空"],
        },
        TextProcessor().tokenize,
        GraphBuilder().build,
    )
    assert memory_id == 1
    await _clear_graph_tables(source_storage)
    assert (await source_storage.graph_integrity_report())["graph_entries"] == 0
    await source_storage.add_conversation_message(
        {
            "session_id": "astrbot:group:new",
            "role": "user",
            "content": "这条消息还没有被总结。",
            "sender_id": "user-1",
            "sender_name": "tester",
            "group_id": "group-1",
            "platform": "astrbot",
            "timestamp": time.time(),
            "metadata": {"pending_summary": True},
        }
    )
    upload_db = tmp_path / "upload" / "livingmemory.db"
    upload_conversations_db = tmp_path / "upload" / "conversations.db"
    upload_db.parent.mkdir(parents=True)
    shutil.copy2(source_dir / "livingmemory.db", upload_db)
    shutil.copy2(source_dir / "conversations.db", upload_conversations_db)

    provider_config = ProviderConfig(dimensions=8)
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda config: FakeProvider(config),
    )
    monkeypatch.setattr(
        "personalityrag.libraries.build_provider",
        lambda config: FakeProvider(config),
    )
    manager = LibraryManager(root, AppConfig(provider=provider_config))
    await manager.initialize()
    try:
        result = await manager.import_livingmemory_db(
            "Default", upload_db, conversations_db=upload_conversations_db
        )
        assert result["stats"]["total_memories"] == 1
        assert result["conversations_source"]["counts"]["sessions"] == 1
        assert result["conversations_source"]["counts"]["messages"] == 1
        assert result["stats"]["conversation_counts"]["sessions"] == 1
        assert result["stats"]["conversation_counts"]["messages"] == 1
        assert result["stats"]["conversation_counts"]["pending_messages"] == 1
        assert result["graph_recovery"]["rebuilt"] is True
        assert result["stats"]["graph_entries"] > 0
        runtime = await manager.get_runtime("Default")
        conversation = await runtime.storage.get_conversation("astrbot:group:new")
        assert conversation is not None
        assert conversation["messages"][0]["content"] == "这条消息还没有被总结。"
        assert runtime.indexes.status()["document_vectors"] == 1
        assert runtime.indexes.status()["graph_vectors"] == result["stats"]["graph_entries"]
        assert not upload_db.exists()
        assert not upload_conversations_db.exists()

        second_upload = tmp_path / "upload2" / "livingmemory.db"
        second_upload.parent.mkdir(parents=True)
        shutil.copy2(source_dir / "livingmemory.db", second_upload)
        with pytest.raises(ValueError, match="只有全新空记忆库"):
            await manager.import_livingmemory_db("Default", second_upload)
    finally:
        await manager.close()
