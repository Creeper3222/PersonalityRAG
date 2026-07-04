from __future__ import annotations

import asyncio
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from personalityrag.config import AppConfig, ProviderConfig
from personalityrag.graph import GraphBuilder
from personalityrag.jobs import JobManager
from personalityrag.libraries import LibraryManager
from personalityrag.providers import EmbeddingProvider
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


@pytest.mark.asyncio
async def test_job_manager_runs_tasks_fifo_and_keeps_finished_history(tmp_path: Path):
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
    for _ in range(30):
        if (await jobs.get(second_id) or {}).get("status") == "completed":
            break
        await asyncio.sleep(0.05)

    active = await jobs.list(scope="active")
    finished = await jobs.list(scope="finished")
    assert active == []
    assert [item["id"] for item in finished] == [second_id, first_id]
    await jobs.clear_for_startup()
    assert await jobs.list(scope="all") == []
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
        result = await manager.backup_library("beileite")
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
            "persona_id": "beileite",
            "topics": ["星空"],
        },
        TextProcessor().tokenize,
        GraphBuilder().build,
    )
    assert memory_id == 1
    upload_db = tmp_path / "upload" / "livingmemory.db"
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
        result = await manager.import_livingmemory_db("beileite", upload_db)
        assert result["stats"]["total_memories"] == 1
        runtime = await manager.get_runtime("beileite")
        assert runtime.indexes.status()["document_vectors"] == 1
        assert runtime.indexes.status()["graph_vectors"] == result["stats"]["graph_entries"]
        assert not upload_db.exists()

        second_upload = tmp_path / "upload2" / "livingmemory.db"
        second_upload.parent.mkdir(parents=True)
        shutil.copy2(source_dir / "livingmemory.db", second_upload)
        with pytest.raises(ValueError, match="只有全新空记忆库"):
            await manager.import_livingmemory_db("beileite", second_upload)
    finally:
        await manager.close()
