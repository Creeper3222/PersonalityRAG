from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import httpx
import numpy as np
import pytest

from personalityrag.config import AppConfig, ProviderConfig
from personalityrag.control import ControlStore
from personalityrag.graph import GraphBuilder
from personalityrag.indexes import IndexManager
from personalityrag.libraries import LibraryManager
from personalityrag.providers import (
    EmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    VLLMEmbeddingProvider,
)
from personalityrag.storage import Storage
from personalityrag.text import TextProcessor


class FakeProvider(EmbeddingProvider):
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.dimension = config.dimensions or 8
        self.closed = False

    async def get_embedding(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for index, value in enumerate(text.encode("utf-8")):
            vector[index % self.dimension] += (value % 19) / 19
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

    async def test_connection(self):
        return {
            "available": True,
            "resolved_model": self.config.model,
            "dimension": self.dimension,
        }

    async def close(self) -> None:
        self.closed = True


class FailingProvider(FakeProvider):
    async def test_connection(self):
        return {"available": False, "error": "fixture failure"}


@pytest.mark.asyncio
async def test_vllm_resolves_served_model_and_never_sends_dimensions():
    requests: list[dict] = []

    async def handler(request: httpx.Request):
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "bge-m3",
                            "root": "BAAI/bge-m3",
                            "owned_by": "vllm",
                        }
                    ]
                },
            )
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": [1.0, 0.0, 0.0]}
                    for index, _ in enumerate(payload["input"])
                ]
            },
        )

    provider = VLLMEmbeddingProvider(
        ProviderConfig(
            id="vllm",
            type="vllm_embedding",
            model="BAAI/bge-m3",
            dimensions=3,
        )
    )
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(
        base_url="http://test/v1",
        transport=httpx.MockTransport(handler),
    )
    assert len(await provider.get_embedding("hello")) == 3
    assert requests[0]["model"] == "bge-m3"
    assert "dimensions" not in requests[0]
    await provider.close()


@pytest.mark.asyncio
async def test_openai_sends_configured_dimensions():
    requests: list[dict] = []

    async def handler(request: httpx.Request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]},
        )

    provider = OpenAIEmbeddingProvider(
        ProviderConfig(
            id="openai",
            type="openai_embedding",
            model="text-embedding-test",
            dimensions=3,
        )
    )
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(
        base_url="http://test/v1",
        transport=httpx.MockTransport(handler),
    )
    assert len(await provider.get_embedding("hello")) == 3
    assert requests[0]["dimensions"] == 3
    await provider.close()


@pytest.mark.asyncio
async def test_ollama_lists_models_and_embeds_batches():
    requests: list[dict] = []

    async def handler(request: httpx.Request):
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "nomic-embed-text:latest"}]},
            )
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "embeddings": [
                    [1.0, 0.0, 0.0] for _ in payload.get("input", [])
                ]
            },
        )

    provider = OllamaEmbeddingProvider(
        ProviderConfig(
            id="ollama",
            type="ollama_embedding",
            api_base="http://127.0.0.1:11434",
            model="nomic-embed-text",
            dimensions=3,
        )
    )
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    assert (await provider.list_models())[0]["id"] == "nomic-embed-text:latest"
    assert len(await provider.get_embeddings(["a", "b"])) == 2
    assert requests[0]["model"] == "nomic-embed-text"
    await provider.close()


@pytest.mark.asyncio
async def test_provider_revision_mask_and_usage_protection(tmp_path: Path):
    control = ControlStore(tmp_path / "system.db")
    seed = ProviderConfig(api_key="secret-value")
    await control.initialize(seed)
    listed = await control.list_providers()
    assert listed[0]["api_key"] == "********"
    assert listed[0]["has_api_key"] is True

    updated = await control.update_provider(
        seed.id, {"display_name": "新名称", "api_key": ""}
    )
    assert updated.revision == 2
    assert updated.config.api_key == "secret-value"
    cleared = await control.update_provider(
        seed.id, {"clear_api_key": True}
    )
    assert cleared.revision == 3
    assert cleared.config.api_key == ""
    await control.ensure_default_library(
        library_id="default",
        name="默认库",
        provider_id=seed.id,
        provider_revision=cleared.revision,
    )
    with pytest.raises(ValueError, match="不能停用"):
        await control.update_provider(seed.id, {"enabled": False})
    with pytest.raises(ValueError, match="不能删除"):
        await control.delete_provider(seed.id)


@pytest.mark.asyncio
async def test_provider_copy_delete_reuses_released_id_and_numbers(tmp_path: Path):
    control = ControlStore(tmp_path / "system.db")
    seed = ProviderConfig(id="vllm_embedding", display_name="Local bge-m3")
    await control.initialize(seed)

    first = await control.copy_provider(seed.id)
    assert first.provider_id == "vllm_embedding_copy"
    assert first.config.enabled is False

    await control.delete_provider(first.provider_id)
    assert await control.get_provider(first.provider_id) is None

    reused = await control.copy_provider(seed.id)
    assert reused.provider_id == "vllm_embedding_copy"

    second = await control.copy_provider(seed.id)
    assert second.provider_id == "vllm_embedding_copy2"

    third = await control.copy_provider(seed.id)
    assert third.provider_id == "vllm_embedding_copy3"

    await control.delete_provider(reused.provider_id)
    await control.delete_provider(second.provider_id)
    await control.delete_provider(third.provider_id)


@pytest.mark.asyncio
async def test_legacy_data_migrates_to_beileite_and_libraries_are_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "PersonalityRAG"
    data = root / "data"
    storage = Storage(data)
    await storage.initialize()
    original_id = await storage.create_memory(
        {
            "content": "贝雷特记得澄月喜欢星空",
            "persona_id": "贝雷特",
            "topics": ["星空"],
        },
        TextProcessor().tokenize,
        GraphBuilder().build,
    )
    provider_config = ProviderConfig(dimensions=8)
    provider = FakeProvider(provider_config)
    indexes = IndexManager(data, storage, provider, provider_config.model)
    await indexes.initialize()
    await indexes.rebuild()

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
        default = await manager.library_detail("beileite")
        assert default["name"] == "贝雷特"
        assert default["stats"]["total_memories"] == 1
        assert not (data / "livingmemory.db").exists()
        assert (data / "libraries" / "beileite" / "livingmemory.db").exists()
        marker = json.loads(
            (data / ".multilibrary_migrated_v1.json").read_text(
                encoding="utf-8"
            )
        )
        assert marker["validation"] == "passed"
        assert (
            await (await manager.get_runtime("beileite")).storage.get_document(
                original_id
            )
        )

        second = await manager.create_library(
            {
                "id": "second",
                "name": "第二记忆库",
                "provider_id": provider_config.id,
            }
        )
        assert second["indexes"]["generation"] is None
        assert second["indexes"]["document_vectors"] == 0
        assert second["indexes"]["graph_vectors"] == 0
        second_runtime = await manager.get_runtime(second["id"])
        await second_runtime.create_memory(
            {
                "content": "第二个库的独立记忆",
                "persona_id": "另一人格",
                "topics": ["隔离"],
            }
        )
        assert (await second_runtime.storage.statistics())["total_memories"] == 1
        assert (
            await (await manager.get_runtime("beileite")).storage.statistics()
        )["total_memories"] == 1
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_empty_library_with_legacy_empty_generation_is_reported_pending(
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
        created = await manager.create_library(
            {
                "id": "legacy_pending",
                "name": "旧空索引库",
                "provider_id": provider_config.id,
            }
        )
        library_dir = root / "data" / "libraries" / "legacy_pending"
        generation = "gen-legacy-empty"
        generation_dir = library_dir / "indexes" / generation
        generation_dir.mkdir(parents=True, exist_ok=True)
        (library_dir / "indexes" / "CURRENT").write_text(
            generation, encoding="utf-8"
        )
        (generation_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "generation": generation,
                    "document_count": 0,
                    "graph_entry_count": 0,
                    "configured_model": "BAAI/bge-m3",
                    "dimension": 8,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        detail = await manager.library_detail(created["id"])
        assert detail["indexes"]["generation"] is None
        assert detail["indexes"]["manifest"] is None
        assert detail["indexes"]["document_vectors"] == 0
        assert detail["indexes"]["graph_vectors"] == 0
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_failed_legacy_layout_validation_rolls_back_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "PersonalityRAG"
    data = root / "data"
    storage = Storage(data)
    await storage.initialize()
    await storage.create_memory(
        {"content": "没有旧索引时必须回滚", "persona_id": "fixture"},
        TextProcessor().tokenize,
        GraphBuilder().build,
    )
    system_before = (data / "personalityrag_system.db").read_bytes()
    provider_config = ProviderConfig(dimensions=8)
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda config: FakeProvider(config),
    )
    manager = LibraryManager(root, AppConfig(provider=provider_config))
    with pytest.raises(RuntimeError, match="文档 FAISS ID 数量"):
        await manager.initialize()
    assert (data / "livingmemory.db").exists()
    assert not (data / "libraries" / "beileite").exists()
    assert not (data / ".multilibrary_migrated_v1.json").exists()
    assert (data / "personalityrag_system.db").read_bytes() == system_before


@pytest.mark.asyncio
async def test_library_listing_stays_lazy_and_rejected_default_delete_keeps_runtime(
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

    first = LibraryManager(root, AppConfig(provider=provider_config))
    await first.initialize()
    await first.create_library(
        {"id": "second", "name": "第二记忆库", "provider_id": provider_config.id}
    )
    await first.close()

    manager = LibraryManager(root, AppConfig(provider=provider_config))
    await manager.initialize()
    try:
        assert set(manager.runtimes) == {"beileite"}
        libraries = await manager.list_libraries()
        assert {item["id"] for item in libraries} == {"beileite", "second"}
        assert set(manager.runtimes) == {"beileite"}

        runtime = manager.runtimes["beileite"]
        provider = runtime.provider
        await manager.update_library(
            "second", {"recall_settings": {"top_k": 3}}
        )
        second_runtime = await manager.get_runtime("second")
        assert second_runtime.config.recall.top_k == 3
        assert runtime.config.recall.top_k == 10
        with pytest.raises(ValueError, match="默认记忆库不能删除"):
            await manager.delete_library("beileite")
        assert manager.runtimes["beileite"] is runtime
        assert provider.closed is False
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_library_copy_uses_numbered_fallback_after_soft_delete(
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
        first = await manager.copy_library("beileite")
        second = await manager.copy_library("beileite")
        assert first["id"] == "beileite_copy"
        assert first["name"] == "贝雷特(副本)"
        assert second["id"] == "beileite_copy2"
        assert second["name"] == "贝雷特(副本2)"

        await manager.delete_library(first["id"])
        third = await manager.copy_library("beileite")
        assert third["id"] == "beileite_copy3"
        assert third["name"] == "贝雷特(副本3)"
        assert not (root / "data" / "libraries" / first["id"]).exists()
        trash_candidates = list(
            (root / "data" / "trash" / "libraries").glob("beileite_copy-*")
        )
        assert trash_candidates
        assert sorted(item.name for item in trash_candidates[0].iterdir()) == [
            "conversations.db",
            "livingmemory.db",
        ]

        libraries = {item["id"] for item in await manager.list_libraries()}
        assert libraries == {"beileite", "beileite_copy2", "beileite_copy3"}
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_deleted_library_id_can_be_recreated(
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
        created = await manager.create_library(
            {
                "id": "test_import",
                "name": "测试导入库",
                "provider_id": provider_config.id,
            }
        )
        assert created["id"] == "test_import"

        deleted = await manager.delete_library("test_import")
        assert deleted["library_id"] == "test_import"
        assert not (root / "data" / "libraries" / "test_import").exists()

        recreated = await manager.create_library(
            {
                "id": "test_import",
                "name": "重新创建的测试导入库",
                "provider_id": provider_config.id,
            }
        )
        assert recreated["id"] == "test_import"
        assert recreated["name"] == "重新创建的测试导入库"
        assert (root / "data" / "libraries" / "test_import").exists()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_recreate_library_id_after_legacy_index_generation_schema_upgrade(
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
    await manager.close()

    system_db = root / "data" / "personalityrag_system.db"
    with sqlite3.connect(system_db) as db:
        db.executescript(
            """
            ALTER TABLE index_generations RENAME TO index_generations_old;
            CREATE TABLE index_generations (
                generation TEXT NOT NULL,
                status TEXT NOT NULL,
                manifest TEXT NOT NULL,
                created_at REAL NOT NULL,
                activated_at REAL,
                PRIMARY KEY(generation)
            );
            INSERT INTO index_generations(generation,status,manifest,created_at,activated_at)
            SELECT generation,status,manifest,created_at,activated_at
            FROM index_generations_old;
            DROP TABLE index_generations_old;
            """
        )

    manager = LibraryManager(root, AppConfig(provider=provider_config))
    await manager.initialize()
    try:
        created = await manager.create_library(
            {
                "id": "test_import",
                "name": "测试导入库",
                "provider_id": provider_config.id,
            }
        )
        assert created["id"] == "test_import"
        deleted = await manager.delete_library("test_import")
        assert deleted["library_id"] == "test_import"
        recreated = await manager.create_library(
            {
                "id": "test_import",
                "name": "重建测试导入库",
                "provider_id": provider_config.id,
            }
        )
        assert recreated["id"] == "test_import"
        assert recreated["name"] == "重建测试导入库"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_provider_switch_is_atomic_and_failed_switch_preserves_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "PersonalityRAG"
    provider_config = ProviderConfig(dimensions=8)

    def factory(config: ProviderConfig):
        if config.model == "fixture-failure":
            return FailingProvider(config)
        return FakeProvider(config)

    monkeypatch.setattr("personalityrag.service.build_provider", factory)
    monkeypatch.setattr("personalityrag.libraries.build_provider", factory)
    manager = LibraryManager(root, AppConfig(provider=provider_config))
    await manager.initialize()
    try:
        runtime = await manager.get_runtime("beileite")
        old_provider = runtime.provider
        await manager.create_library(
            {
                "id": "other_library",
                "name": "另一记忆库",
                "provider_id": provider_config.id,
            }
        )
        switched = await manager.create_provider(
            {
                "id": "second_provider",
                "display_name": "第二 Provider",
                "type": "vllm_embedding",
                "enabled": True,
                "model": "fixture-ok",
                "dimensions": 8,
            }
        )
        await manager.rebuild_library("beileite", switched["id"])
        binding = await manager.control.get_library("beileite")
        assert binding is not None
        assert binding.provider_id == "second_provider"
        other_binding = await manager.control.get_library("other_library")
        assert other_binding is not None
        assert other_binding.provider_id == provider_config.id
        assert runtime.provider is not old_provider
        assert old_provider.closed is False

        failed = await manager.create_provider(
            {
                "id": "failed_provider",
                "display_name": "失败 Provider",
                "type": "vllm_embedding",
                "enabled": True,
                "model": "fixture-failure",
                "dimensions": 8,
            }
        )
        current_generation = runtime.indexes.status()["generation"]
        current_provider = runtime.provider
        with pytest.raises(RuntimeError, match="Provider 测试失败"):
            await manager.rebuild_library("beileite", failed["id"])
        binding = await manager.control.get_library("beileite")
        assert binding is not None
        assert binding.provider_id == "second_provider"
        assert runtime.provider is current_provider
        assert runtime.indexes.status()["generation"] == current_generation
    finally:
        await manager.close()
    assert old_provider.closed is True


@pytest.mark.asyncio
async def test_persona_and_session_filters_remain_library_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "PersonalityRAG"
    provider_config = ProviderConfig(dimensions=8)
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda config: FakeProvider(config),
    )
    manager = LibraryManager(root, AppConfig(provider=provider_config))
    await manager.initialize()
    try:
        runtime = await manager.get_runtime("beileite")
        await runtime.create_memory(
            {
                "content": "贝雷特喜欢夜空与星辰",
                "persona_id": "beileite",
                "session_id": "session-a",
                "topics": ["星空"],
            }
        )
        await runtime.create_memory(
            {
                "content": "另一人格喜欢清晨的海风",
                "persona_id": "other",
                "session_id": "session-b",
                "topics": ["海风"],
            }
        )
        by_persona = await runtime.retrieval.search(
            "喜欢什么", k=10, persona_id="beileite"
        )
        assert by_persona
        assert all(
            item.metadata.get("persona_id") == "beileite"
            for item in by_persona
        )

        runtime.config.recall.use_session_filtering = True
        by_session = await runtime.retrieval.search(
            "喜欢什么", k=10, session_id="session-b"
        )
        assert by_session
        assert all(
            item.metadata.get("session_id") == "session-b"
            for item in by_session
        )
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_queries_keep_using_old_snapshot_during_provider_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "PersonalityRAG"
    provider_config = ProviderConfig(dimensions=8)
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowProvider(FakeProvider):
        async def get_embeddings(
            self, texts: list[str]
        ) -> list[list[float]]:
            started.set()
            await release.wait()
            return await super().get_embeddings(texts)

    def factory(config: ProviderConfig):
        return SlowProvider(config) if config.model == "slow" else FakeProvider(config)

    monkeypatch.setattr("personalityrag.service.build_provider", factory)
    monkeypatch.setattr("personalityrag.libraries.build_provider", factory)
    manager = LibraryManager(root, AppConfig(provider=provider_config))
    await manager.initialize()
    try:
        runtime = await manager.get_runtime("beileite")
        await runtime.create_memory(
            {"content": "旧索引在重建期间仍应可查询", "persona_id": "fixture"}
        )
        old_generation = runtime.indexes.status()["generation"]
        provider = await manager.create_provider(
            {
                "id": "slow_provider",
                "display_name": "慢速 Provider",
                "type": "vllm_embedding",
                "enabled": True,
                "model": "slow",
                "dimensions": 8,
            }
        )
        rebuild = asyncio.create_task(
            manager.rebuild_library("beileite", provider["id"])
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        results = await asyncio.wait_for(
            runtime.indexes.search_documents("旧索引", 5), timeout=0.5
        )
        assert results
        assert runtime.indexes.status()["generation"] == old_generation
        release.set()
        await rebuild
        assert runtime.indexes.status()["generation"] != old_generation
    finally:
        release.set()
        await manager.close()
        # aiosqlite uses a worker thread; allow its final call_soon_threadsafe
        # notification to reach the still-open test event loop on Windows.
        await asyncio.sleep(0.05)
