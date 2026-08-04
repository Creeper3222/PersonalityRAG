from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from personalityrag.config import AppConfig, ProviderConfig, RecallConfig
from personalityrag.graph import GraphBuilder
from personalityrag.indexes import (
    DEFAULT_DOCUMENT_EMBED_CHARS,
    IndexManager,
)
from personalityrag.io_utils import read_ab_checkpoint
from personalityrag.providers import EmbeddingProvider
from personalityrag.retrieval import RetrievalEngine
from personalityrag.service import PersonalityRAGService
from personalityrag.storage import Storage
from personalityrag.text import TextProcessor


class FakeProvider(EmbeddingProvider):
    dimension = 16

    async def get_embedding(self, text: str) -> list[float]:
        values = np.zeros(self.dimension, dtype=np.float32)
        for index, byte in enumerate(text.encode("utf-8")):
            values[index % self.dimension] += (byte % 31) / 31
        norm = np.linalg.norm(values)
        if norm:
            values /= norm
        return values.tolist()

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [await self.get_embedding(text) for text in texts]

    async def get_dimension(self) -> int:
        return self.dimension

    async def list_models(self):
        return [{"id": "fake"}]

    async def detect_context_length(self):
        return {"max_context_tokens": 4096, "max_context_tokens_source": "auto:fake"}

    async def test_connection(self):
        return {
            "available": True,
            "resolved_model": "fake",
            "dimension": self.dimension,
        }

    async def close(self) -> None:
        pass


class CountingProvider(FakeProvider):
    def __init__(self) -> None:
        self.embedded_texts: list[str] = []

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return await super().get_embeddings(texts)


class FullInputCountingProvider(CountingProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__()
        self.config = config


class StrictContextProvider(FakeProvider):
    def __init__(self, config: ProviderConfig, *, detected_tokens: int = 128) -> None:
        self.config = config
        self.detected_tokens = detected_tokens
        self.detect_calls = 0
        self.embedded_texts: list[str] = []

    async def detect_context_length(self):
        self.detect_calls += 1
        return {
            "max_context_tokens": self.detected_tokens,
            "max_context_tokens_source": "auto:strict-fixture",
        }

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        hard_char_limit = max(64, int(self.detected_tokens * 0.75))
        if any(len(text) > hard_char_limit for text in texts):
            raise ValueError("fixture rejected over-context embedding input")
        self.embedded_texts.extend(texts)
        return [await FakeProvider.get_embedding(self, text) for text in texts]


class RecordingJobContext:
    def __init__(self) -> None:
        self.checkpoints: list[dict[str, object]] = []

    async def checkpoint(
        self,
        checkpoint: dict[str, object],
        *,
        progress: float | None = None,
        message: str | None = None,
    ) -> None:
        self.checkpoints.append(
            {
                "checkpoint": checkpoint,
                "progress": progress,
                "message": message,
            }
        )

    async def control_point(self) -> None:
        return None


@pytest.mark.asyncio
async def test_write_rebuild_recall_and_delete(tmp_path: Path):
    storage = Storage(tmp_path)
    await storage.initialize()
    text = TextProcessor()
    graph = GraphBuilder()
    first = await storage.create_memory(
        {
            "content": "澄月喜欢观察星空",
            "persona_id": "贝雷特",
            "topics": ["星空"],
            "participants": ["澄月"],
            "key_facts": ["澄月喜欢星空"],
        },
        text.tokenize,
        graph.build,
    )
    second = await storage.create_memory(
        {
            "content": "哈萨维正在维护 RocketCatShell",
            "persona_id": "贝雷特",
            "topics": ["RocketCatShell"],
            "participants": ["哈萨维"],
            "key_facts": ["哈萨维维护 RocketCatShell"],
        },
        text.tokenize,
        graph.build,
    )
    provider = FakeProvider()
    indexes = IndexManager(tmp_path, storage, provider, "fake")
    await indexes.initialize()
    manifest = await indexes.rebuild(batch_size=2, concurrency=2)
    assert manifest["document_count"] == 2
    assert manifest["graph_entry_count"] > 0
    assert manifest["graph_vector_granularity"] == "memory"
    assert manifest["graph_source_memory_count"] == 2
    assert manifest["graph_vector_count"] == 2
    assert indexes.status()["graph_vectors"] == 2
    engine = RetrievalEngine(storage, indexes, text, RecallConfig())
    results = await engine.search("谁喜欢星空", 2, persona_id="贝雷特")
    assert results
    assert {item.doc_id for item in results} <= {first, second}
    assert await storage.delete_memories([first]) == 1
    assert await storage.get_document(first) is None
    await storage.close()


@pytest.mark.asyncio
async def test_resumable_rebuild_distinguishes_graph_memories_from_entries(
    tmp_path: Path,
):
    storage = Storage(tmp_path)
    await storage.initialize()
    try:
        text = TextProcessor()
        graph = GraphBuilder()
        for person, topic in (("Alice", "release"), ("Bob", "testing")):
            await storage.create_memory(
                {
                    "content": f"{person} owns the {topic} plan",
                    "topics": [topic, "project"],
                    "participants": [person],
                    "key_facts": [
                        f"{person} owns the {topic} plan",
                        f"{topic} happens on Friday",
                    ],
                },
                text.tokenize,
                graph.build,
            )

        provider = FakeProvider()
        indexes = IndexManager(tmp_path, storage, provider, "fake")
        await indexes.initialize()
        context = RecordingJobContext()
        checkpoint_dir = tmp_path / "resumable-checkpoint"
        manifest = await indexes.rebuild(
            batch_size=1,
            concurrency=1,
            job_context=context,  # type: ignore[arg-type]
            checkpoint_dir=checkpoint_dir,
        )

        stats = await storage.statistics()
        state = read_ab_checkpoint(checkpoint_dir)
        assert state is not None
        assert stats["graph_entries"] > 2
        assert state["total_graph_entries"] == 2  # legacy checkpoint key
        assert state["total_graph_source_memories"] == 2
        assert state["source_fingerprint"]["graph_entry_count"] == stats["graph_entries"]
        assert manifest["graph_entry_count"] == stats["graph_entries"]
        assert manifest["graph_source_memory_count"] == 2
        assert manifest["graph_vector_count"] == 2
        assert context.checkpoints
        latest = context.checkpoints[-1]["checkpoint"]
        assert latest["total_graph_source_memories"] == 2
        assert latest["raw_graph_entry_count"] == stats["graph_entries"]
        assert latest["graph_vector_granularity"] == "memory"
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_batch_metadata_update_and_set_delete_preserve_derived_data(tmp_path: Path):
    storage = Storage(tmp_path)
    await storage.initialize()
    text = TextProcessor()
    graph = GraphBuilder()
    memory_ids = [
        await storage.create_memory(
            {
                "content": f"batch fixture {index}",
                "session_id": "old-session",
                "persona_id": "old-persona",
                "key_facts": [f"fact {index}"],
            },
            text.tokenize,
            graph.build,
        )
        for index in range(3)
    ]

    updated = await storage.update_memories_metadata(
        [memory_ids[2], memory_ids[0], 999999],
        {
            "importance": 0.9,
            "session_id": "new-session",
            "persona_id": "new-persona",
        },
    )
    assert updated == [memory_ids[2], memory_ids[0]]
    documents = await storage.documents_for_ids(updated)
    assert all(item["metadata"]["importance"] == 0.9 for item in documents)
    assert all(item["metadata"]["session_id"] == "new-session" for item in documents)
    graph_rows = await storage.graph_entries_for_memory_ids(updated)
    assert all(item["session_id"] == "new-session" for item in graph_rows)
    assert all(item["persona_id"] == "new-persona" for item in graph_rows)

    assert await storage.delete_memories([*memory_ids, 999999]) == 3
    assert await storage.document_ids() == []
    assert await storage.graph_entry_ids() == []
    await storage.close()


@pytest.mark.asyncio
async def test_embedding_inputs_are_stably_chunked_after_context_probe(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    storage = Storage(tmp_path)
    await storage.initialize()
    long_document = "document-prefix-" + ("A" * (DEFAULT_DOCUMENT_EMBED_CHARS + 256))
    long_graph_fact = "graph-prefix-" + ("B" * (DEFAULT_DOCUMENT_EMBED_CHARS + 128))
    long_query = "query-prefix-" + ("C" * (DEFAULT_DOCUMENT_EMBED_CHARS + 64))
    await storage.create_memory(
        {
            "content": long_document,
            "persona_id": "Default",
            "topics": ["long-input"],
            "key_facts": [long_graph_fact],
        },
        TextProcessor().tokenize,
        GraphBuilder().build,
    )
    config = ProviderConfig(
        id="display_context_only",
        dimensions=FakeProvider.dimension,
        max_context_tokens=1,
        max_context_tokens_source="manual",
    )
    provider = FullInputCountingProvider(config)
    indexes = IndexManager(
        tmp_path,
        storage,
        provider,
        "fake",
        provider_id=config.id,
    )
    await indexes.initialize()

    caplog.set_level(logging.WARNING, logger="personalityrag")
    await indexes.rebuild(batch_size=1, concurrency=1)

    assert provider.config.max_context_tokens == 4096
    assert long_document not in provider.embedded_texts
    assert long_document in "".join(provider.embedded_texts)
    embedded = "".join(provider.embedded_texts)
    graph_rows = await storage.graph_memories_for_ids(
        await storage.graph_memory_ids()
    )
    assert len(graph_rows) == 1
    assert len(graph_rows[0]["content"]) == 4000
    assert graph_rows[0]["content"] in embedded
    assert "graph-prefix-" in graph_rows[0]["content"]
    assert long_graph_fact not in embedded
    assert all(len(text) <= 3072 for text in provider.embedded_texts)
    rebuild_warnings = [
        record.getMessage()
        for record in caplog.records
        if "Embedding 输入较长" in record.getMessage()
    ]
    assert rebuild_warnings
    assert any("label=index_rebuild_documents" in item for item in rebuild_warnings)
    assert any("policy=stable_char_chunk_mean_pool_v1" in item for item in rebuild_warnings)

    caplog.clear()
    provider.embedded_texts.clear()
    await indexes.search_documents(long_query, 5)
    await indexes.search_graph(long_query, 5)

    assert long_query not in provider.embedded_texts
    assert "".join(provider.embedded_texts) == long_query + long_query
    assert all(len(text) <= 3072 for text in provider.embedded_texts)
    query_warnings = [
        record.getMessage()
        for record in caplog.records
        if "Embedding 输入较长" in record.getMessage()
    ]
    assert len(query_warnings) == 2
    assert any("label=document_recall_query" in item for item in query_warnings)
    assert any("label=graph_recall_query" in item for item in query_warnings)
    await storage.close()


@pytest.mark.asyncio
async def test_strict_provider_is_probed_once_and_never_receives_oversized_input(
    tmp_path: Path,
):
    storage = Storage(tmp_path)
    await storage.initialize()
    source_text = "strict-provider-" + ("长" * 420)
    await storage.create_memory(
        {"content": source_text, "topics": ["strict"]},
        TextProcessor().tokenize,
        GraphBuilder().build,
    )
    config = ProviderConfig(
        id="strict",
        dimensions=FakeProvider.dimension,
        max_context_tokens=0,
    )
    provider = StrictContextProvider(config)
    indexes = IndexManager(tmp_path, storage, provider, "fake", provider_id=config.id)
    await indexes.initialize()

    manifest = await indexes.rebuild(batch_size=1, concurrency=1)

    assert provider.detect_calls == 1
    assert manifest["embedding_capability"]["detected_max_context_tokens"] == 128
    assert manifest["chunked_document_count"] >= 1
    assert all(len(text) <= 96 for text in provider.embedded_texts)
    assert source_text in "".join(provider.embedded_texts)

    provider.embedded_texts.clear()
    await indexes.search_documents("查询" * 180, 5)
    assert provider.detect_calls == 1
    assert all(len(text) <= 96 for text in provider.embedded_texts)
    await storage.close()


@pytest.mark.asyncio
async def test_incremental_index_trusts_persisted_context_at_or_above_128(
    tmp_path: Path,
):
    storage = Storage(tmp_path)
    await storage.initialize()
    config = ProviderConfig(
        id="trusted-short-task",
        dimensions=FakeProvider.dimension,
        max_context_tokens=128,
        max_context_tokens_source="auto:persisted",
    )
    provider = StrictContextProvider(config)
    indexes = IndexManager(tmp_path, storage, provider, "fake", provider_id=config.id)
    await indexes.initialize()
    memory_id = await storage.create_memory(
        {"content": "增量记忆" * 80, "topics": ["incremental"]},
        TextProcessor().tokenize,
        GraphBuilder().build,
    )

    result = await indexes.upsert_memories([memory_id], reason="trusted_context")

    assert result["document_vectors"] == 1
    assert provider.detect_calls == 0
    assert provider.embedded_texts
    assert all(len(text) <= 96 for text in provider.embedded_texts)
    await storage.close()


@pytest.mark.asyncio
async def test_incremental_upsert_embeds_only_new_memory(tmp_path: Path):
    storage = Storage(tmp_path)
    await storage.initialize()
    text = TextProcessor()
    graph = GraphBuilder()
    provider = CountingProvider()
    indexes = IndexManager(tmp_path, storage, provider, "fake")
    await indexes.initialize()

    first = await storage.create_memory(
        {
            "content": "first memory about sky",
            "topics": ["sky"],
            "key_facts": ["first fact"],
        },
        text.tokenize,
        graph.build,
    )
    first_update = await indexes.upsert_memories([first], reason="test_first")
    assert first_update["document_vectors"] == 1

    provider.embedded_texts.clear()
    second = await storage.create_memory(
        {
            "content": "second memory about ocean",
            "topics": ["ocean"],
            "key_facts": ["second fact"],
        },
        text.tokenize,
        graph.build,
    )
    second_update = await indexes.upsert_memories([second], reason="test_second")

    assert second_update["document_vectors"] == 2
    assert second_update["graph_vectors"] == 2
    assert len(provider.embedded_texts) == 2
    assert any("second memory" in text for text in provider.embedded_texts)
    assert not any("first memory" in text for text in provider.embedded_texts)
    await storage.close()


@pytest.mark.asyncio
async def test_full_graph_snapshot_returns_all_nodes_and_honors_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda _config: FakeProvider(),
    )
    service = PersonalityRAGService(
        tmp_path,
        AppConfig(provider=ProviderConfig(dimensions=FakeProvider.dimension)),
        data_dir=tmp_path,
    )
    try:
        await service.storage.initialize()
        builder = service._graph_builder_for_write()
        first = await service.storage.create_memory(
            {
                "content": "Alice plans a release with Bob",
                "session_id": "s1",
                "persona_id": "p1",
                "topics": ["release"],
                "participants": ["Alice", "Bob"],
                "key_facts": ["Alice works with Bob"],
            },
            service.text.tokenize,
            builder,
        )
        await service.storage.create_memory(
            {
                "content": "Carol prefers tea",
                "session_id": "s2",
                "persona_id": "p2",
                "topics": ["tea"],
                "participants": ["Carol"],
            },
            service.text.tokenize,
            builder,
        )

        full = await service.full_graph_snapshot()
        scoped = await service.full_graph_snapshot(session_id="s1", persona_id="p1")

        assert len(full["nodes"]) > len(scoped["nodes"]) >= 2
        assert {item["memory_id"] for item in scoped["memories"]} == {first}
        assert all(edge["source_memory_id"] == first for edge in scoped["edges"])
        assert all(edge["memory_id"] == edge["source_memory_id"] for edge in full["edges"])
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_service_rejects_incremental_write_when_nonempty_index_unbuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda _config: FakeProvider(),
    )
    service = PersonalityRAGService(
        tmp_path,
        AppConfig(provider=ProviderConfig(dimensions=FakeProvider.dimension)),
        data_dir=tmp_path,
    )
    try:
        await service.storage.initialize()
        await service.storage.create_memory(
            {"content": "existing unindexed memory"},
            service.text.tokenize,
            service._graph_builder_for_write(),
        )
        await service.indexes.initialize()

        with pytest.raises(RuntimeError, match="全量索引重建"):
            await service.create_memory({"content": "new memory should roll back"})

        stats = await service.storage.statistics()
        assert stats["total_memories"] == 1
        assert service.indexes.status()["document_vectors"] == 0
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_service_empty_library_first_write_creates_incremental_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda _config: FakeProvider(),
    )
    service = PersonalityRAGService(
        tmp_path,
        AppConfig(provider=ProviderConfig(dimensions=FakeProvider.dimension)),
        data_dir=tmp_path,
    )
    await service.initialize()
    try:
        created = await service.create_memory({"content": "first indexed memory"})

        index_update = created["index_update"]
        assert index_update["mode"] == "incremental"
        assert index_update["status"] == "completed"
        assert index_update["document_vectors"] == 1
        assert index_update["generation"]
        assert service.indexes.status()["document_vectors"] == 1
        assert (await service.storage.statistics())["total_memories"] == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_service_create_rolls_back_when_incremental_index_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda _config: FakeProvider(),
    )
    service = PersonalityRAGService(
        tmp_path,
        AppConfig(provider=ProviderConfig(dimensions=FakeProvider.dimension)),
        data_dir=tmp_path,
    )
    await service.initialize()

    async def fail_upsert(*args, **kwargs):
        raise RuntimeError("incremental index failed")

    service.indexes.upsert_memories = fail_upsert  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="incremental index failed"):
            await service.create_memory({"content": "should not remain"})

        assert (await service.storage.statistics())["total_memories"] == 0
        assert service.indexes.status()["document_vectors"] == 0
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_service_content_update_replaces_old_memory_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda _config: FakeProvider(),
    )
    service = PersonalityRAGService(
        tmp_path,
        AppConfig(provider=ProviderConfig(dimensions=FakeProvider.dimension)),
        data_dir=tmp_path,
    )
    await service.initialize()
    try:
        created = await service.create_memory(
            {"content": "old content about tea", "key_facts": ["old tea fact"]}
        )
        old_id = int(created["id"])

        updated = await service.update_memory(
            old_id,
            {"content": "new content about coffee"},
            rebuild=False,
        )

        assert updated is not None
        new_id = int(updated["new_memory_id"])
        assert new_id != old_id
        assert await service.storage.get_document(old_id) is None
        assert await service.storage.get_document(new_id) is not None
        assert service.indexes.status()["document_vectors"] == 1
        results = await service.retrieval.search("coffee", 5)
        assert [item.doc_id for item in results] == [new_id]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_service_persona_update_keeps_ids_and_indexes_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    provider = CountingProvider()
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda _config: provider,
    )
    service = PersonalityRAGService(
        tmp_path,
        AppConfig(provider=ProviderConfig(dimensions=FakeProvider.dimension)),
        data_dir=tmp_path,
    )
    await service.initialize()
    try:
        created = await service.create_memory(
            {
                "content": "四条旧记忆需要补回正确人格",
                "persona_id": None,
                "topics": ["人格修正"],
                "key_facts": ["旧记忆的人格字段为空"],
            }
        )
        memory_id = int(created["id"])
        generation = service.indexes.status()["generation"]
        graph_before = await service.storage.graph_entries_for_memory_ids(
            [memory_id]
        )
        graph_ids = [int(item["id"]) for item in graph_before]
        provider.embedded_texts.clear()

        updated = await service.update_memory_persona(memory_id, "贝雷特")

        assert updated is not None
        assert int(updated["id"]) == memory_id
        assert updated["metadata"]["persona_id"] == "贝雷特"
        assert updated["index_update"] == {
            "mode": "metadata_only",
            "status": "completed",
            "index_changed": False,
            "generation": generation,
            "document_vectors": service.indexes.status()["document_vectors"],
            "graph_vectors": service.indexes.status()["graph_vectors"],
        }
        assert provider.embedded_texts == []
        assert service.indexes.status()["generation"] == generation
        graph_after = await service.storage.graph_entries_for_memory_ids(
            [memory_id]
        )
        assert [int(item["id"]) for item in graph_after] == graph_ids
        assert all(item["persona_id"] == "贝雷特" for item in graph_after)
        async with service.storage.connect() as db:
            atom_personas = [
                row["persona_id"]
                for row in await (
                    await db.execute(
                        "SELECT persona_id FROM memory_atoms WHERE parent_memory_id=?",
                        (memory_id,),
                    )
                ).fetchall()
            ]
        assert atom_personas
        assert atom_personas == ["贝雷特"] * len(atom_personas)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_storage_initialization_strips_only_obsolete_memory_type(tmp_path: Path):
    storage = Storage(tmp_path)
    await storage.initialize()
    async with storage.connect() as db:
        cursor = await db.execute(
            "INSERT INTO documents(doc_id,text,metadata) VALUES(?,?,?)",
            (
                "legacy-type",
                "legacy memory",
                json.dumps(
                    {
                        "memory_type": "GROUP_CHAT",
                        "importance": 0.7,
                        "topics": ["keep"],
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        memory_id = int(cursor.lastrowid)
        await db.commit()

    await storage.initialize()
    cleaned = await storage.get_document(memory_id)
    assert cleaned is not None
    assert cleaned["metadata"] == {"importance": 0.7, "topics": ["keep"]}
    async with storage.connect() as db:
        row = await (
            await db.execute(
                "SELECT metadata FROM documents WHERE id=?", (memory_id,)
            )
        ).fetchone()
    assert json.loads(row["metadata"]) == {
        "importance": 0.7,
        "topics": ["keep"],
    }
    await storage.close()


@pytest.mark.asyncio
async def test_service_ignores_memory_type_without_rebuilding_indexes_or_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    provider = CountingProvider()
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda _config: provider,
    )
    service = PersonalityRAGService(
        tmp_path,
        AppConfig(provider=ProviderConfig(dimensions=FakeProvider.dimension)),
        data_dir=tmp_path,
    )
    await service.initialize()
    try:
        created = await service.create_memory(
            {
                "content": "群聊中的手动测试记忆",
                "persona_id": "贝雷特",
                "importance": 0.6,
                "memory_type": "GROUP_CHAT",
                "topics": ["测试"],
                "key_facts": ["这是一条测试记忆"],
                "metadata": {"memory_type": "PRIVATE_CHAT"},
            }
        )
        memory_id = int(created["id"])
        assert "memory_type" not in created["metadata"]
        generation = service.indexes.status()["generation"]
        graph_before = await service.storage.graph_entries_for_memory_ids(
            [memory_id]
        )
        graph_identity = [
            (int(item["id"]), item["content"])
            for item in graph_before
        ]
        provider.embedded_texts.clear()

        updated = await service.update_memory(
            memory_id,
            {
                "memory_type": "GROUP_CHAT",
                "status": "archived",
                "importance": 0.8,
                "metadata": {
                    "memory_type": "MANUAL",
                    "update_history": [{"description": "状态修正"}],
                },
            },
            rebuild=False,
        )

        assert updated is not None
        assert int(updated["id"]) == memory_id
        assert "new_memory_id" not in updated
        assert "memory_type" not in updated["metadata"]
        assert updated["metadata"]["status"] == "archived"
        assert updated["metadata"]["importance"] == pytest.approx(0.8)
        assert updated["metadata"]["persona_id"] == "贝雷特"
        assert updated["index_update"]["mode"] == "incremental"
        assert updated["index_update"]["removed_documents"] == [memory_id]
        assert updated["index_update"]["generation"] != generation
        assert provider.embedded_texts == []
        assert service.indexes.status()["document_vectors"] == 0

        graph_after = await service.storage.graph_entries_for_memory_ids(
            [memory_id]
        )
        assert graph_identity
        assert graph_after == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_service_content_update_rolls_back_when_old_delete_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda _config: FakeProvider(),
    )
    service = PersonalityRAGService(
        tmp_path,
        AppConfig(provider=ProviderConfig(dimensions=FakeProvider.dimension)),
        data_dir=tmp_path,
    )
    await service.initialize()
    try:
        created = await service.create_memory(
            {
                "content": "original durable memory",
                "source_messages": [
                    {"role": "user", "content": "original source message"}
                ],
            }
        )
        old_id = int(created["id"])
        original_delete = service.storage.delete_memories

        async def flaky_delete(memory_ids):
            if old_id in {int(value) for value in memory_ids}:
                raise RuntimeError("delete old failed")
            return await original_delete(memory_ids)

        service.storage.delete_memories = flaky_delete  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="delete old failed"):
            await service.update_memory(
                old_id,
                {"content": "replacement should roll back"},
                rebuild=False,
            )

        assert await service.storage.get_document(old_id) is not None
        assert [
            item["content"]
            for item in await service.storage.get_memory_source(old_id)
        ] == ["original source message"]
        assert (await service.storage.statistics())["total_memories"] == 1
        assert service.indexes.status()["document_vectors"] == 1
        results = await service.retrieval.search("original", 5)
        assert [item.doc_id for item in results] == [old_id]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_service_content_update_cancellation_rolls_back_new_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda _config: FakeProvider(),
    )
    service = PersonalityRAGService(
        tmp_path,
        AppConfig(provider=ProviderConfig(dimensions=FakeProvider.dimension)),
        data_dir=tmp_path,
    )
    await service.initialize()
    try:
        created = await service.create_memory(
            {
                "content": "original cancellation-safe memory",
                "source_messages": [
                    {"role": "user", "content": "original source survives"}
                ],
            }
        )
        old_id = int(created["id"])
        started = asyncio.Event()
        release = asyncio.Event()
        original_upsert = service.indexes.upsert_memories

        async def blocking_upsert(memory_ids, **kwargs):
            if kwargs.get("reason") == "memory_update_create_replacement":
                started.set()
                await release.wait()
            return await original_upsert(memory_ids, **kwargs)

        service.indexes.upsert_memories = blocking_upsert  # type: ignore[method-assign]
        task = asyncio.create_task(
            service.update_memory(
                old_id,
                {"content": "replacement cancelled before old deletion"},
                rebuild=False,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        documents = (await service.storage.list_documents(page_size=10))["items"]
        assert [int(item["id"]) for item in documents] == [old_id]
        assert documents[0]["text"] == "original cancellation-safe memory"
        assert [
            item["content"]
            for item in await service.storage.get_memory_source(old_id)
        ] == ["original source survives"]
        assert service.indexes.status()["document_vectors"] == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_service_content_update_cancellation_keeps_committed_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda _config: FakeProvider(),
    )
    service = PersonalityRAGService(
        tmp_path,
        AppConfig(provider=ProviderConfig(dimensions=FakeProvider.dimension)),
        data_dir=tmp_path,
    )
    await service.initialize()
    try:
        created = await service.create_memory(
            {
                "content": "old copy removed at commit boundary",
                "source_messages": [
                    {"role": "user", "content": "retained source payload"}
                ],
            }
        )
        old_id = int(created["id"])
        deleted_old = asyncio.Event()
        release = asyncio.Event()
        original_delete = service.storage.delete_memories

        async def committed_then_blocked(memory_ids):
            result = await original_delete(memory_ids)
            if old_id in {int(value) for value in memory_ids}:
                deleted_old.set()
                await release.wait()
            return result

        service.storage.delete_memories = committed_then_blocked  # type: ignore[method-assign]
        task = asyncio.create_task(
            service.update_memory(
                old_id,
                {"content": "replacement survives committed old deletion"},
                rebuild=False,
            )
        )
        await asyncio.wait_for(deleted_old.wait(), timeout=2)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        documents = (await service.storage.list_documents(page_size=10))["items"]
        assert len(documents) == 1
        replacement_id = int(documents[0]["id"])
        assert replacement_id != old_id
        assert documents[0]["text"] == "replacement survives committed old deletion"
        assert [
            item["content"]
            for item in await service.storage.get_memory_source(replacement_id)
        ] == ["retained source payload"]
        assert service.indexes.status()["document_vectors"] == 1
        results = await service.retrieval.search("replacement survives", 5)
        assert [item.doc_id for item in results] == [replacement_id]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_archive_restore_retains_source_and_rebuilds_all_derivatives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda _config: FakeProvider(),
    )
    service = PersonalityRAGService(
        tmp_path,
        AppConfig(provider=ProviderConfig(dimensions=FakeProvider.dimension)),
        data_dir=tmp_path,
    )
    await service.initialize()
    try:
        created = await service.create_memory(
            {
                "content": "Alice plans a Friday release",
                "topics": ["release"],
                "participants": ["Alice"],
                "key_facts": ["Alice plans a Friday release"],
                "source_messages": [
                    {"role": "user", "content": "I will release it on Friday"}
                ],
            }
        )
        memory_id = int(created["id"])
        before = await service.storage.statistics()
        assert before["graph_entries"] > 0
        assert before["atom_count"] > 0
        assert service.indexes.status()["document_vectors"] == 1

        archived = await service.archive_memories([memory_id], return_details=True)

        detail = await service.storage.get_document(memory_id)
        assert archived["archived"] == 1
        assert detail is not None
        assert detail["metadata"]["status"] == "archived"
        assert [
            item["content"]
            for item in await service.storage.get_memory_source(memory_id)
        ] == ["I will release it on Friday"]
        archived_stats = await service.storage.statistics()
        assert archived_stats["active_memories"] == 0
        assert archived_stats["graph_entries"] == 0
        assert archived_stats["atom_count"] == 0
        assert service.indexes.status()["document_vectors"] == 0
        assert await service.retrieval.search("Friday release", 5) == []

        # Explicit document/index and graph rebuilds must never resurrect an
        # archived memory into FTS, graph-derived data, or FAISS.
        await service.rebuild_indexes()
        await service.rebuild_graph()
        archived_after_rebuild = await service.storage.statistics()
        assert archived_after_rebuild["active_memories"] == 0
        assert archived_after_rebuild["graph_entries"] == 0
        assert archived_after_rebuild["atom_count"] == 0
        assert service.indexes.status()["document_vectors"] == 0
        assert await service.retrieval.search("Friday release", 5) == []
        assert [
            item["content"]
            for item in await service.storage.get_memory_source(memory_id)
        ] == ["I will release it on Friday"]

        restored = await service.restore_memory(memory_id)

        assert restored is not None
        assert restored["metadata"]["status"] == "active"
        assert [
            item["content"]
            for item in await service.storage.get_memory_source(memory_id)
        ] == ["I will release it on Friday"]
        restored_stats = await service.storage.statistics()
        assert restored_stats["graph_entries"] == before["graph_entries"]
        assert restored_stats["atom_count"] == before["atom_count"]
        assert service.indexes.status()["document_vectors"] == 1
        assert [item.doc_id for item in await service.retrieval.search("Friday release", 5)] == [memory_id]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_shadow_rebuild_replays_concurrent_create_and_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda _config: FakeProvider(),
    )
    service = PersonalityRAGService(
        tmp_path,
        AppConfig(provider=ProviderConfig(dimensions=FakeProvider.dimension)),
        data_dir=tmp_path,
    )
    await service.initialize()
    try:
        first = await service.create_memory(
            {
                "content": "Alice owns the original release plan",
                "topics": ["release"],
                "participants": ["Alice"],
                "key_facts": ["Alice owns the original release plan"],
            }
        )
        first_id = int(first["id"])
        assert not await service.storage.has_memory_sources_table()
        blocked = asyncio.Event()
        release = asyncio.Event()
        original_get_embeddings = service.provider.get_embeddings

        async def delay_first_shadow_batch(texts):
            if not blocked.is_set():
                blocked.set()
                await release.wait()
            return await original_get_embeddings(texts)

        service.provider.get_embeddings = delay_first_shadow_batch  # type: ignore[method-assign]
        rebuild = asyncio.create_task(service.rebuild_indexes())
        await asyncio.wait_for(blocked.wait(), timeout=2)

        second = await service.create_memory(
            {
                "content": "Bob added a concurrent verification checklist",
                "topics": ["verification"],
                "participants": ["Bob"],
                "key_facts": ["Bob added a concurrent verification checklist"],
            }
        )
        second_id = int(second["id"])
        await service.archive_memories([first_id])
        assert not rebuild.done()

        release.set()
        result = await asyncio.wait_for(rebuild, timeout=10)

        assert set(result["replayed_memory_ids"]) == {first_id, second_id}
        assert service.indexes.indexed_ids()[0] == {second_id}
        assert set(await service.storage.document_ids()) == {second_id}
        assert first_id not in {
            item.doc_id
            for item in await service.retrieval.search("original release", 5)
        }
        assert [
            item.doc_id
            for item in await service.retrieval.search(
                "concurrent verification checklist", 5
            )
        ] == [second_id]
        maintenance = service.maintenance_status()
        assert maintenance["status"] == "ready"
        assert maintenance["index_available"] is True
        assert not await service.storage.has_memory_sources_table()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_graph_vectors_are_aggregated_once_per_source_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda _config: FakeProvider(),
    )
    service = PersonalityRAGService(
        tmp_path,
        AppConfig(provider=ProviderConfig(dimensions=FakeProvider.dimension)),
        data_dir=tmp_path,
    )
    await service.initialize()
    try:
        memory_ids = []
        for person, topic in (("Alice", "release"), ("Bob", "testing")):
            created = await service.create_memory(
                {
                    "content": f"{person} owns the {topic} plan",
                    "topics": [topic, "project"],
                    "participants": [person],
                    "key_facts": [
                        f"{person} owns the {topic} plan",
                        f"{topic} happens on Friday",
                    ],
                }
            )
            memory_ids.append(int(created["id"]))

        stats = await service.storage.statistics()
        status = service.indexes.status()
        manifest = status["manifest"]
        assert stats["graph_entries"] > len(memory_ids)
        assert status["graph_vectors"] == len(memory_ids)
        assert manifest["graph_vector_granularity"] == "memory"
        assert manifest["graph_source_memory_count"] == len(memory_ids)
        assert manifest["graph_vector_count"] == len(memory_ids)
        assert manifest["graph_entry_count"] == stats["graph_entries"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_service_delete_removes_only_target_memory_from_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda _config: FakeProvider(),
    )
    service = PersonalityRAGService(
        tmp_path,
        AppConfig(provider=ProviderConfig(dimensions=FakeProvider.dimension)),
        data_dir=tmp_path,
    )
    await service.initialize()
    try:
        first = await service.create_memory({"content": "delete target alpha"})
        second = await service.create_memory({"content": "keep target beta"})
        first_id = int(first["id"])
        second_id = int(second["id"])

        result = await service.delete_memories(
            [first_id],
            rebuild=False,
            return_details=True,
        )

        assert result["deleted"] == 1
        assert result["index_update"]["removed_documents"] == [first_id]
        assert await service.storage.get_document(first_id) is None
        assert await service.storage.get_document(second_id) is not None
        assert service.indexes.status()["document_vectors"] == 1
        assert [item.doc_id for item in await service.retrieval.search("beta", 5)] == [
            second_id
        ]
        assert first_id not in [
            item.doc_id for item in await service.retrieval.search("alpha", 5)
        ]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_graph_keyword_batches_large_token_lists(tmp_path: Path):
    storage = Storage(tmp_path)
    await storage.initialize()
    text = TextProcessor()
    graph = GraphBuilder()
    await storage.create_memory(
        {
            "content": "long graph token batching smoke",
            "topics": ["RocketCatShell"],
            "participants": ["Default"],
            "key_facts": ["RocketCatShell graph token batching works"],
        },
        text.tokenize,
        graph.build,
    )
    provider = FakeProvider()
    indexes = IndexManager(tmp_path, storage, provider, "fake")
    await indexes.initialize()
    engine = RetrievalEngine(storage, indexes, text, RecallConfig())
    query = " ".join([f"token{i}" for i in range(700)] + ["RocketCatShell"])
    results = await engine._graph_keyword(query, 5, None, None)
    assert isinstance(results, list)
    await storage.close()


@pytest.mark.asyncio
async def test_graph_keyword_prioritizes_multi_node_hits(tmp_path: Path):
    storage = Storage(tmp_path)
    await storage.initialize()
    now = "2026-01-01T00:00:00Z"
    async with storage.connect() as db:
        await db.executemany(
            """INSERT INTO documents(
                id,doc_id,text,metadata,created_at,updated_at
            ) VALUES(?,?,?,?,?,?)""",
            [
                (1, "memory-1", "single tv hit", "{}", now, now),
                (2, "memory-2", "double tv hit", "{}", now, now),
            ],
        )
        await db.executemany(
            """INSERT INTO graph_nodes(
                id,node_key,node_type,node_value,canonical_value,
                metadata,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?)""",
            [
                (1, "topic:tv", "topic", "tv", "tv", "{}", now, now),
                (
                    2,
                    "fact:zhiren-tv",
                    "fact",
                    "\u667a\u4ebatv",
                    "\u667a\u4ebatv",
                    "{}",
                    now,
                    now,
                ),
            ],
        )
        await db.executemany(
            """INSERT INTO graph_entries(
                id,entry_key,source_memory_id,entry_type,content,
                metadata,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?)""",
            [
                (1, "entry:single", 1, "fact", "single tv hit", "{}", now, now),
                (2, "entry:double", 2, "fact", "double tv hit", "{}", now, now),
            ],
        )
        await db.executemany(
            "INSERT INTO graph_entry_nodes(entry_id,node_id) VALUES(?,?)",
            [(1, 1), (2, 1), (2, 2)],
        )
        await db.commit()

    provider = FakeProvider()
    indexes = IndexManager(tmp_path, storage, provider, "fake")
    await indexes.initialize()
    engine = RetrievalEngine(storage, indexes, TextProcessor(), RecallConfig())

    results = await engine._graph_keyword("\u667a\u4eba" + "tv", 2, None, None)

    assert [item.doc_id for item in results] == [2, 1]
    await storage.close()


@pytest.mark.asyncio
async def test_fts_writes_livingmemory_compatible_content(tmp_path: Path):
    storage = Storage(tmp_path)
    await storage.initialize()

    def tokenize(_text: str) -> list[str]:
        return ["tokenized", "content"]

    def graph_builder(memory_id: int, content: str, metadata: dict):
        return {
            "nodes": [],
            "edges": [],
            "entries": [
                {
                    "entry_key": f"entry:{memory_id}:{content}",
                    "source_memory_id": memory_id,
                    "session_id": metadata.get("session_id"),
                    "persona_id": metadata.get("persona_id"),
                    "entry_type": "fact",
                    "relation_type": "fact",
                    "content": f"Graph raw entry for {content}",
                    "metadata": {},
                    "node_keys": [],
                }
            ],
        }

    memory_id = await storage.create_memory(
        {"content": "Raw memory text", "persona_id": "p"},
        tokenize,
        graph_builder,
    )

    async def fts_rows():
        async with storage.connect() as db:
            doc_row = await (
                await db.execute(
                    "SELECT content FROM livingmemory_memories_fts WHERE doc_id=?",
                    (memory_id,),
                )
            ).fetchone()
            graph_row = await (
                await db.execute(
                    """SELECT content FROM livingmemory_graph_entries_fts
                    WHERE entry_id=(SELECT id FROM graph_entries WHERE source_memory_id=?)
                    """,
                    (memory_id,),
                )
            ).fetchone()
        return doc_row["content"], graph_row["content"]

    assert await fts_rows() == (
        "tokenized content",
        "Graph raw entry for Raw memory text",
    )

    await storage.update_memory(
        memory_id,
        {"content": "Updated raw memory"},
        tokenize,
        graph_builder,
    )
    assert await fts_rows() == (
        "tokenized content",
        "Graph raw entry for Updated raw memory",
    )

    await storage.rebuild_fts(tokenize)
    assert await fts_rows() == (
        "tokenized content",
        "Graph raw entry for Updated raw memory",
    )
    await storage.close()


@pytest.mark.asyncio
async def test_candidate_graph_evidence_never_returns_outside_candidate_ids(tmp_path: Path):
    storage = Storage(tmp_path)
    await storage.initialize()
    now = "2026-01-01T00:00:00Z"
    async with storage.connect() as db:
        await db.executemany(
            """INSERT INTO graph_entries(
                id,entry_key,source_memory_id,entry_type,relation_type,content,
                metadata,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?)""",
            [
                (1, "entry:1", 1, "fact", "fact", "candidate alpha", "{}", now, now),
                (2, "entry:2", 2, "fact", "fact", "candidate target", "{}", now, now),
                (99, "entry:99", 99, "fact", "fact", "outside target", "{}", now, now),
            ],
        )
        await db.executemany(
            "INSERT INTO livingmemory_graph_entries_fts(entry_id,content) VALUES(?,?)",
            [
                (1, "candidate alpha"),
                (2, "candidate target"),
                (99, "outside target"),
            ],
        )
        await db.commit()

    evidence = await storage.candidate_graph_evidence([1, 2], ["target"])

    assert set(evidence) <= {1, 2}
    assert 99 not in evidence
    assert evidence[2]["keyword_score"] > 0
    await storage.close()


@pytest.mark.asyncio
async def test_integrity_report_hashes_database_files_without_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = Storage(tmp_path)
    await storage.initialize()

    def forbidden_read_bytes(path: Path) -> bytes:
        raise AssertionError(f"read_bytes must not hash database files: {path}")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    report = await storage.integrity_report()

    assert report["livingmemory"]["integrity"] == "ok"
    assert len(report["livingmemory"]["sha256"]) == 64
    assert report["conversations"]["integrity"] == "ok"
    assert len(report["conversations"]["sha256"]) == 64
    await storage.close()
