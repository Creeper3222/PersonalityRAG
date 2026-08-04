from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pytest

from personalityrag.config import AppConfig, IndexRebuildSettings, ProviderConfig
from personalityrag.database_types import (
    DatabaseRef,
    LIVINGMEMORY_V8_TYPE,
    database_type_registry,
)
from personalityrag.graph import GraphBuilder
from personalityrag.indexes import DEFAULT_DOCUMENT_EMBED_CHARS
from personalityrag.libraries import LibraryManager
from personalityrag.migration import sqlite_backup
from personalityrag.providers import EmbeddingProvider
from personalityrag.storage import Storage
from personalityrag.text import TextProcessor


JOB_TIMEOUT_SECONDS = 120
EVENT_TIMEOUT_SECONDS = 30


class ControlledProvider(EmbeddingProvider):
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.dimension = int(config.dimensions or 8)
        self.block_next_batch = False
        self.fail_next_batch = False
        self.batch_started = asyncio.Event()
        self.release_batch = asyncio.Event()
        self.embedded_texts: list[str] = []

    async def get_embedding(self, text: str) -> list[float]:
        return (await self.get_embeddings([text]))[0]

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        if self.fail_next_batch:
            self.fail_next_batch = False
            self.batch_started.set()
            raise ConnectionError("fixture embedding outage")
        if self.block_next_batch:
            self.block_next_batch = False
            self.batch_started.set()
            await self.release_batch.wait()
        values: list[list[float]] = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for index, value in enumerate(text.encode("utf-8")):
                vector[index % self.dimension] += (value % 19) / 19
            norm = float(np.linalg.norm(vector))
            if norm:
                vector /= norm
            values.append(vector.tolist())
        return values

    async def get_dimension(self) -> int:
        return self.dimension

    async def list_models(self):
        return [{"id": self.config.model}]

    async def detect_context_length(self):
        return {"max_context_tokens": 4096, "max_context_tokens_source": "test"}

    async def test_connection(self):
        return {
            "available": True,
            "resolved_model": self.config.model,
            "dimension": self.dimension,
        }

    async def close(self) -> None:
        return None


async def _wait_status(manager: LibraryManager, job_id: str, status: str) -> dict:
    assert manager.jobs is not None
    for _ in range(200):
        job = await manager.jobs.get(job_id)
        if job and job["status"] == status:
            return job
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {status}: {await manager.jobs.get(job_id)}")


def _database_signature(path: Path) -> str:
    tables = (
        "documents",
        "graph_nodes",
        "graph_edges",
        "graph_entries",
        "graph_entry_nodes",
        "atoms",
    )
    digest = hashlib.sha256()
    connection = sqlite3.connect(path)
    try:
        for table in tables:
            columns = [
                row[1]
                for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
            ]
            if table.startswith("graph_"):
                columns = [
                    value
                    for value in columns
                    if value not in {"created_at", "updated_at"}
                ]
            if not columns:
                continue
            order = ",".join(f'"{value}"' for value in columns)
            for row in connection.execute(f'SELECT {order} FROM "{table}" ORDER BY rowid'):
                digest.update(repr(tuple(row)).encode("utf-8"))
                digest.update(b"\0")
    finally:
        connection.close()
    return digest.hexdigest()


def _index_signature(index: faiss.Index) -> tuple[tuple[int, ...], str]:
    ids = tuple(map(int, faiss.vector_to_array(index.id_map)))
    inner = faiss.downcast_index(index.index)
    vectors = np.asarray(
        faiss.rev_swig_ptr(inner.get_xb(), int(inner.ntotal * inner.d)),
        dtype=np.float32,
    )
    return ids, hashlib.sha256(vectors.tobytes()).hexdigest()


@pytest.mark.asyncio
async def test_import_pause_resume_matches_uninterrupted_and_stop_restores_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_storage = Storage(tmp_path / "source")
    await source_storage.initialize()
    long_source_text = "resumable-long-input-" + (
        "A" * (DEFAULT_DOCUMENT_EMBED_CHARS + 128)
    )
    for content, topic in (
        (long_source_text, "long-input"),
        ("贝雷特记得澄月喜欢星空。", "星空"),
        ("凶星和澄月一起完成了模型测试。", "测试"),
        ("雨雀提醒大家保存安全断点。", "断点"),
    ):
        await source_storage.create_memory(
            {"content": content, "persona_id": "Default", "topics": [topic]},
            TextProcessor().tokenize,
            GraphBuilder().build,
        )
    source_db = tmp_path / "source-livingmemory.db"
    await asyncio.to_thread(
        sqlite_backup, tmp_path / "source" / "livingmemory.db", source_db
    )

    rebuild = IndexRebuildSettings(
        batch_size=1,
        embedding_batch_size=1,
        tasks_limit=1,
        max_retries=1,
        retry_base_delay=0,
        batch_delay=0,
        request_delay=0,
        max_failure_ratio=0,
    )
    config = ProviderConfig(dimensions=8, index_rebuild_settings=rebuild)
    provider = ControlledProvider(config)
    monkeypatch.setattr("personalityrag.service.build_provider", lambda _config: provider)
    monkeypatch.setattr(
        "personalityrag.library_types.livingmemory_v8.manager.build_provider",
        lambda _config: provider,
    )
    manager = LibraryManager(tmp_path / "PersonalityRAG", AppConfig(provider=config))
    await manager.initialize()
    try:
        for library_id in ("normal", "paused", "stopped", "interrupted", "conflict"):
            await manager.create_library(
                {"id": library_id, "name": library_id, "provider_id": config.id}
            )
        assert manager.jobs is not None

        normal_upload = tmp_path / "normal-upload.db"
        await asyncio.to_thread(sqlite_backup, source_db, normal_upload)
        normal_id = await manager.jobs.start_resumable(
            "livingmemory_import",
            {"source_db": str(normal_upload), "conversations_db": None},
            library_id="normal",
        )
        assert (await asyncio.wait_for(manager.jobs.wait(normal_id), timeout=JOB_TIMEOUT_SECONDS))["status"] == "completed"
        assert long_source_text not in provider.embedded_texts
        assert long_source_text in "".join(provider.embedded_texts)

        paused_upload = tmp_path / "paused-upload.db"
        await asyncio.to_thread(sqlite_backup, source_db, paused_upload)
        provider.batch_started.clear()
        provider.release_batch.clear()
        provider.block_next_batch = True
        paused_id = await manager.jobs.start_resumable(
            "livingmemory_import",
            {"source_db": str(paused_upload), "conversations_db": None},
            library_id="paused",
        )
        await asyncio.wait_for(provider.batch_started.wait(), timeout=EVENT_TIMEOUT_SECONDS)
        await manager.jobs.pause(paused_id)
        provider.release_batch.set()
        paused_job = await _wait_status(manager, paused_id, "paused")
        assert paused_job["checkpoint"]["completed_documents"] == 1
        assert paused_job["capabilities"]["resume"] is True
        workspace = (
            database_type_registry.data_dir(
                manager.data_dir, DatabaseRef(LIVINGMEMORY_V8_TYPE, "paused")
            )
            / "task_checkpoints"
            / paused_id
        )
        assert len(list((workspace / "index" / "segments").glob("*.npz"))) == 1

        await manager.jobs.resume(paused_id)
        resumed = await asyncio.wait_for(manager.jobs.wait(paused_id), timeout=JOB_TIMEOUT_SECONDS)
        assert resumed["status"] == "completed"
        assert not workspace.exists()

        normal_runtime = await manager.get_runtime("normal")
        paused_runtime = await manager.get_runtime("paused")
        assert _database_signature(normal_runtime.data_dir / "livingmemory.db") == _database_signature(
            paused_runtime.data_dir / "livingmemory.db"
        )
        assert _index_signature(normal_runtime.indexes.document_index) == _index_signature(
            paused_runtime.indexes.document_index
        )
        assert _index_signature(normal_runtime.indexes.graph_index) == _index_signature(
            paused_runtime.indexes.graph_index
        )

        stopped_runtime = await manager.get_runtime("stopped")
        old_generation = stopped_runtime.indexes.status().get("generation") or ""
        stopped_upload = tmp_path / "stopped-upload.db"
        await asyncio.to_thread(sqlite_backup, source_db, stopped_upload)
        provider.batch_started.clear()
        provider.release_batch.clear()
        provider.block_next_batch = True
        stopped_id = await manager.jobs.start_resumable(
            "livingmemory_import",
            {"source_db": str(stopped_upload), "conversations_db": None},
            library_id="stopped",
        )
        await asyncio.wait_for(provider.batch_started.wait(), timeout=EVENT_TIMEOUT_SECONDS)
        await manager.jobs.stop(stopped_id)
        provider.release_batch.set()
        stopped = await asyncio.wait_for(manager.jobs.wait(stopped_id), timeout=JOB_TIMEOUT_SECONDS)
        assert stopped["status"] == "stopped"
        assert stopped["progress"] == 0
        stopped_runtime = await manager.get_runtime("stopped")
        assert await manager.library_is_empty("stopped") is True
        assert (stopped_runtime.indexes.status().get("generation") or "") == old_generation
        assert not stopped_upload.exists()

        interrupted_upload = tmp_path / "interrupted-upload.db"
        await asyncio.to_thread(sqlite_backup, source_db, interrupted_upload)
        provider.batch_started.clear()
        provider.fail_next_batch = True
        interrupted_id = await manager.jobs.start_resumable(
            "livingmemory_import",
            {"source_db": str(interrupted_upload), "conversations_db": None},
            library_id="interrupted",
        )
        await asyncio.wait_for(provider.batch_started.wait(), timeout=EVENT_TIMEOUT_SECONDS)
        interrupted = await _wait_status(manager, interrupted_id, "interrupted")
        assert interrupted["status_reason"] == "provider_unavailable"
        assert interrupted["capabilities"]["resume"] is True
        await manager.jobs.resume(interrupted_id)
        assert (await asyncio.wait_for(manager.jobs.wait(interrupted_id), timeout=JOB_TIMEOUT_SECONDS))["status"] == "completed"
        interrupted_runtime = await manager.get_runtime("interrupted")
        assert _index_signature(normal_runtime.indexes.document_index) == _index_signature(
            interrupted_runtime.indexes.document_index
        )
        assert _index_signature(normal_runtime.indexes.graph_index) == _index_signature(
            interrupted_runtime.indexes.graph_index
        )

        conflict_upload = tmp_path / "conflict-upload.db"
        await asyncio.to_thread(sqlite_backup, source_db, conflict_upload)
        provider.batch_started.clear()
        provider.release_batch.clear()
        provider.block_next_batch = True
        conflict_id = await manager.jobs.start_resumable(
            "livingmemory_import",
            {"source_db": str(conflict_upload), "conversations_db": None},
            library_id="conflict",
        )
        await asyncio.wait_for(provider.batch_started.wait(), timeout=EVENT_TIMEOUT_SECONDS)
        await manager.jobs.pause(conflict_id)
        provider.release_batch.set()
        await _wait_status(manager, conflict_id, "paused")
        conflict_runtime = await manager.get_runtime("conflict")
        async with conflict_runtime.storage.connect() as db:
            await db.execute("UPDATE documents SET text=text || ' changed' WHERE id=1")
            await db.commit()
        await manager.jobs.resume(conflict_id)
        conflict = await _wait_status(manager, conflict_id, "interrupted")
        assert conflict["status_reason"] == "source_changed"
        assert conflict["capabilities"]["stop"] is True
        await manager.jobs.stop(conflict_id)
        assert (await asyncio.wait_for(manager.jobs.wait(conflict_id), timeout=JOB_TIMEOUT_SECONDS))["status"] == "stopped"
        assert await manager.library_is_empty("conflict") is True
    finally:
        await manager.close()
