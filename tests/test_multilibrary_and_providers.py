from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import httpx
import numpy as np
import pytest

from personalityrag.config import (
    AppConfig,
    ProviderConfig,
    RuntimeResidencyConfig,
)
from personalityrag.compat import LIVINGMEMORY_DATABASE_VERSION
from personalityrag.control import ControlStore
from personalityrag.graph import GraphBuilder
from personalityrag.indexes import IndexManager
from personalityrag.libraries import DEFAULT_LIBRARY_ID, LibraryManager
from personalityrag.providers import (
    EmbeddingProvider,
    GeminiEmbeddingProvider,
    NvidiaEmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    VLLMEmbeddingProvider,
    provider_config_hash,
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
        self.closed = True


class FailingProvider(FakeProvider):
    async def test_connection(self):
        return {"available": False, "error": "fixture failure"}


def _patch_fake_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "personalityrag.service.build_provider",
        lambda config: FakeProvider(config),
    )
    monkeypatch.setattr(
        "personalityrag.libraries.build_provider",
        lambda config: FakeProvider(config),
    )


@pytest.mark.asyncio
async def test_new_install_uses_Default_and_existing_default_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "PersonalityRAG"
    provider_config = ProviderConfig(dimensions=8)
    monkeypatch.setattr(
        "personalityrag.service.build_provider", lambda config: FakeProvider(config)
    )
    monkeypatch.setattr(
        "personalityrag.libraries.build_provider", lambda config: FakeProvider(config)
    )

    manager = LibraryManager(root, AppConfig(provider=provider_config))
    await manager.initialize()
    try:
        default = await manager.control.default_library()
        assert DEFAULT_LIBRARY_ID == "Default"
        assert default.id == DEFAULT_LIBRARY_ID
        await manager.update_library(default.id, {"id": "existing_default"})
    finally:
        await manager.close()

    restored = LibraryManager(root, AppConfig(provider=provider_config))
    await restored.initialize()
    try:
        default = await restored.control.default_library()
        assert default.id == "existing_default"
        assert await restored.control.get_library(DEFAULT_LIBRARY_ID) is None
    finally:
        await restored.close()


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
async def test_vllm_detects_context_length_from_models():
    async def handler(request: httpx.Request):
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "bge-m3",
                        "root": "BAAI/bge-m3",
                        "max_model_len": 8192,
                    }
                ]
            },
        )

    provider = VLLMEmbeddingProvider(
        ProviderConfig(id="vllm", type="vllm_embedding", model="BAAI/bge-m3")
    )
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(
        base_url="http://test/v1",
        transport=httpx.MockTransport(handler),
    )
    detected = await provider.detect_context_length()
    assert detected["max_context_tokens"] == 8192
    assert "max_model_len" in detected["max_context_tokens_source"]
    await provider.close()


@pytest.mark.asyncio
async def test_ollama_detects_context_length_from_show():
    async def handler(request: httpx.Request):
        assert request.url.path == "/api/show"
        return httpx.Response(
            200,
            json={"model_info": {"bge.context_length": 8192}},
        )

    provider = OllamaEmbeddingProvider(
        ProviderConfig(id="ollama", type="ollama_embedding", model="bge-m3")
    )
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    detected = await provider.detect_context_length()
    assert detected["max_context_tokens"] == 8192
    assert "/api/show" in detected["max_context_tokens_source"]
    await provider.close()


@pytest.mark.asyncio
async def test_openai_uses_known_embedding_context_length_table():
    provider = OpenAIEmbeddingProvider(
        ProviderConfig(
            id="openai",
            type="openai_embedding",
            model="text-embedding-3-small",
        )
    )
    detected = await provider.detect_context_length()
    assert detected["max_context_tokens"] == 8192
    assert "known-model-table" in detected["max_context_tokens_source"]
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
async def test_gemini_lists_models_and_embeds_batches():
    requests: list[dict] = []

    async def handler(request: httpx.Request):
        assert request.headers["x-goog-api-key"] == "gemini-secret"
        if request.url.path == "/v1beta/models":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-embedding-exp-03-07",
                            "inputTokenLimit": 8192,
                        }
                    ]
                },
            )
        assert request.url.path.endswith(
            "/models/gemini-embedding-exp-03-07:batchEmbedContents"
        )
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "embeddings": [
                    {"values": [1.0, 0.0, 0.0]}
                    for _ in payload.get("requests", [])
                ]
            },
        )

    provider = GeminiEmbeddingProvider(
        ProviderConfig(
            id="gemini",
            type="gemini_embedding",
            api_base="https://generativelanguage.googleapis.com/v1beta",
            api_key="gemini-secret",
            model="gemini-embedding-exp-03-07",
            dimensions=3,
        )
    )
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(
        base_url="https://test/v1beta",
        headers={"x-goog-api-key": "gemini-secret"},
        transport=httpx.MockTransport(handler),
    )
    assert (await provider.list_models())[0]["inputTokenLimit"] == 8192
    detected = await provider.detect_context_length()
    assert detected["max_context_tokens"] == 8192
    assert len(await provider.get_embeddings(["a", "b"])) == 2
    assert requests[0]["requests"][0]["model"].startswith("models/")
    assert requests[0]["requests"][0]["outputDimensionality"] == 3
    await provider.close()


@pytest.mark.asyncio
async def test_nvidia_sends_input_type_and_float_encoding():
    requests: list[dict] = []

    async def handler(request: httpx.Request):
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

    provider = NvidiaEmbeddingProvider(
        ProviderConfig(
            id="nvidia",
            type="nvidia_embedding",
            api_base="https://integrate.api.nvidia.com/v1",
            api_key="nvapi-secret",
            model="nvidia/llama-nemotron-embed-1b-v2",
            dimensions=3,
            input_type="passage",
        )
    )
    await provider._client.aclose()
    provider._client = httpx.AsyncClient(
        base_url="https://test/v1",
        transport=httpx.MockTransport(handler),
    )
    assert len(await provider.get_embeddings(["a", "b"])) == 2
    assert requests[0]["model"] == "nvidia/llama-nemotron-embed-1b-v2"
    assert requests[0]["input_type"] == "passage"
    assert requests[0]["encoding_format"] == "float"
    assert "dimensions" not in requests[0]
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
async def test_manager_detects_context_length_only_on_provider_create_or_endpoint_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []

    class CountingProvider(FakeProvider):
        async def detect_context_length(self):
            calls.append(self.config.api_base)
            tokens = 2222 if "new-endpoint" in self.config.api_base else 1111
            return {
                "max_context_tokens": tokens,
                "max_context_tokens_source": "auto:counting",
            }

    monkeypatch.setattr(
        "personalityrag.libraries.build_provider",
        lambda config: CountingProvider(config),
    )
    manager = LibraryManager(tmp_path / "PersonalityRAG", AppConfig())
    await manager.initialize()
    try:
        created = await manager.create_provider(
            {
                "id": "ctx_provider",
                "display_name": "Context Provider",
                "type": "vllm_embedding",
                "enabled": True,
                "api_base": "http://old-endpoint/v1",
                "model": "fixture-model",
                "dimensions": 8,
            }
        )
        assert created["max_context_tokens"] == 1111
        assert calls == ["http://old-endpoint/v1"]

        renamed = await manager.update_provider(
            "ctx_provider", {"display_name": "Renamed Context Provider"}
        )
        assert renamed["max_context_tokens"] == 1111
        assert calls == ["http://old-endpoint/v1"]

        changed = await manager.update_provider(
            "ctx_provider", {"api_base": "http://new-endpoint/v1"}
        )
        assert changed["max_context_tokens"] == 2222
        assert calls == ["http://old-endpoint/v1", "http://new-endpoint/v1"]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_manual_context_probe_for_saved_provider_keeps_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[tuple[str, str, str]] = []

    class SecretAwareProvider(FakeProvider):
        async def detect_context_length(self):
            calls.append(
                (
                    self.config.api_key,
                    self.config.api_base,
                    self.config.model,
                )
            )
            return {
                "max_context_tokens": 8192,
                "max_context_tokens_source": "auto:secret-aware",
            }

    monkeypatch.setattr(
        "personalityrag.libraries.build_provider",
        lambda config: SecretAwareProvider(config),
    )
    manager = LibraryManager(tmp_path / "PersonalityRAG", AppConfig())
    await manager.initialize()
    try:
        await manager.create_provider(
            {
                "id": "remote_ctx_provider",
                "display_name": "Remote Context Provider",
                "type": "openai_embedding",
                "enabled": True,
                "api_base": "https://old-endpoint.example/v1",
                "api_key": "top-secret",
                "model": "text-embedding-3-small",
                "dimensions": 8,
            }
        )
        calls.clear()

        detected = await manager.detect_context_length(
            {
                "api_base": "https://new-endpoint.example/v1",
                "model": "text-embedding-3-large",
            },
            provider_id="remote_ctx_provider",
        )

        assert detected["max_context_tokens"] == 8192
        assert calls == [
            (
                "top-secret",
                "https://new-endpoint.example/v1",
                "text-embedding-3-large",
            )
        ]
    finally:
        await manager.close()


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
    assert updated.revision == 1
    assert updated.config.api_key == "secret-value"
    assert updated.config.display_name != seed.display_name
    cleared = await control.update_provider(
        seed.id, {"clear_api_key": True}
    )
    assert cleared.revision == 2
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
async def test_provider_usage_ignores_display_only_revision_drift(tmp_path: Path):
    control = ControlStore(tmp_path / "system.db")
    seed = ProviderConfig(api_key="secret-value")
    await control.initialize(seed)
    await control.ensure_default_library(
        library_id="default",
        name="Default",
        provider_id=seed.id,
        provider_revision=1,
    )

    renamed = await control.update_provider(seed.id, {"display_name": "Renamed"})
    assert renamed.revision == 1
    usage = await control.provider_usage(seed.id)
    assert usage[0]["needs_rebuild"] is False

    changed = await control.update_provider(
        seed.id, {"api_base": "http://127.0.0.1:18001/v1"}
    )
    assert changed.revision == 2
    usage = await control.provider_usage(seed.id)
    assert usage[0]["needs_rebuild"] is True


@pytest.mark.asyncio
async def test_provider_usage_ignores_max_context_metadata_drift(tmp_path: Path):
    control = ControlStore(tmp_path / "system.db")
    seed = ProviderConfig(api_key="secret-value")
    await control.initialize(seed)
    await control.ensure_default_library(
        library_id="default",
        name="Default",
        provider_id=seed.id,
        provider_revision=1,
    )

    updated = await control.update_provider(
        seed.id,
        {
            "max_context_tokens": 8192,
            "max_context_tokens_source": "auto:vllm_embedding:models.max_model_len",
        },
    )
    assert updated.revision == 1
    assert updated.config.max_context_tokens == 8192
    usage = await control.provider_usage(seed.id)
    assert usage[0]["needs_rebuild"] is False


@pytest.mark.asyncio
async def test_debug_provider_revision_patch_recomputes_hash_without_rebuild(
    tmp_path: Path,
):
    control = ControlStore(tmp_path / "system.db")
    seed = ProviderConfig(api_key="secret-value")
    await control.initialize(seed)
    await control.ensure_default_library(
        library_id="default",
        name="Default",
        provider_id=seed.id,
        provider_revision=1,
    )

    summary = await control.debug_patch_provider_revision(
        seed.id,
        1,
        {
            "max_context_tokens": 8192,
            "max_context_tokens_source": "auto:vllm_embedding:models.max_model_len",
        },
    )

    assert summary["latest_revision"] == 1
    assert summary["revisions"][0]["config"]["max_context_tokens"] == 8192
    assert summary["usage"][0]["needs_rebuild"] is False


@pytest.mark.asyncio
async def test_debug_provider_revision_reset_can_restore_latest_revision(
    tmp_path: Path,
):
    control = ControlStore(tmp_path / "system.db")
    seed = ProviderConfig(api_key="secret-value")
    await control.initialize(seed)
    await control.ensure_default_library(
        library_id="default",
        name="Default",
        provider_id=seed.id,
        provider_revision=1,
    )
    changed = await control.update_provider(
        seed.id,
        {"api_base": "http://127.0.0.1:18001/v1"},
    )
    assert changed.revision == 2
    usage = await control.provider_usage(seed.id)
    assert usage[0]["needs_rebuild"] is True

    summary = await control.debug_reset_provider_revisions(
        seed.id,
        latest_revision=1,
        delete_revisions_after_latest=True,
    )

    assert summary["latest_revision"] == 1
    assert [item["revision"] for item in summary["revisions"]] == [1]
    assert summary["usage"][0]["provider_revision"] == 1
    assert summary["usage"][0]["needs_rebuild"] is False


@pytest.mark.asyncio
async def test_provider_id_change_is_blocked_while_library_uses_it(tmp_path: Path):
    control = ControlStore(tmp_path / "system.db")
    seed = ProviderConfig(id="seed_provider")
    await control.initialize(seed)
    await control.ensure_default_library(
        library_id="default",
        name="Default",
        provider_id=seed.id,
        provider_revision=1,
    )

    with pytest.raises(ValueError, match="Provider ID"):
        await control.update_provider(seed.id, {"id": "seed_locked"})


@pytest.mark.asyncio
async def test_provider_id_can_change_when_unused(tmp_path: Path):
    control = ControlStore(tmp_path / "system.db")
    seed = ProviderConfig(id="seed_provider", display_name="Seed Provider")
    await control.initialize(seed)

    renamed = await control.update_provider(seed.id, {"id": "renamed_provider"})

    assert renamed.provider_id == "renamed_provider"
    assert renamed.config.id == "renamed_provider"
    assert renamed.revision == 2
    assert await control.get_provider(seed.id) is None
    historical = await control.get_provider("renamed_provider", revision=1)
    assert historical is not None
    assert historical.config.id == "renamed_provider"
    listed = await control.list_providers()
    assert [item["id"] for item in listed] == ["renamed_provider"]


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
async def test_embedding_provider_index_rebuild_settings_are_non_semantic(
    tmp_path: Path,
):
    control = ControlStore(tmp_path / "system.db")
    seed = ProviderConfig(id="embedding_provider", dimensions=8)
    await control.initialize(seed)

    before = await control.get_provider(seed.id)
    assert before is not None
    before_hash = provider_config_hash(before.config)

    updated = await control.update_provider(
        seed.id,
        {
            "index_rebuild_settings": {
                "batch_size": 25,
                "embedding_batch_size": 4,
                "tasks_limit": 1,
                "max_retries": 7,
                "retry_base_delay": 12,
                "batch_delay": 3,
                "request_delay": 2,
                "max_failure_ratio": 0.05,
            },
        },
    )

    assert updated.revision == before.revision
    assert provider_config_hash(updated.config) == before_hash
    assert updated.config.index_rebuild_settings.batch_size == 25
    assert updated.config.index_rebuild_settings.embedding_batch_size == 4
    assert updated.config.index_rebuild_settings.max_retries == 7


@pytest.mark.asyncio
async def test_provider_kind_filtering_and_rerank_usage_protection(tmp_path: Path):
    control = ControlStore(tmp_path / "system.db")
    seed = ProviderConfig(id="embedding_provider")
    await control.initialize(seed)
    rerank = await control.create_provider(
        {
            "id": "rerank_provider",
            "display_name": "Rerank Provider",
            "type": "vllm_rerank",
            "enabled": True,
            "api_base": "http://127.0.0.1:8002",
            "api_suffix": "/v1/rerank",
            "model": "BAAI/bge-reranker-v2-m3",
            "dimensions": 0,
            "batch_size": 1,
            "concurrency": 1,
        }
    )
    embedding = await control.get_provider(seed.id)
    assert embedding is not None
    await control.create_library(
        {
            "id": "rerank_bound",
            "name": "Rerank Bound",
            "provider_id": seed.id,
            "rerank_provider_id": rerank.provider_id,
        },
        embedding,
    )

    assert [item["id"] for item in await control.list_providers("embedding")] == [
        seed.id
    ]
    assert [item["id"] for item in await control.list_providers("rerank")] == [
        rerank.provider_id
    ]
    assert all(
        item["provider_kind"] == "rerank"
        for item in control.provider_types("rerank")
    )
    usage = await control.provider_usage(rerank.provider_id)
    assert usage[0]["usage_kind"] == "rerank"
    with pytest.raises(ValueError, match="Provider ID"):
        await control.update_provider(rerank.provider_id, {"id": "renamed_rerank"})
    with pytest.raises(ValueError, match="不能删除"):
        await control.delete_provider(rerank.provider_id)


@pytest.mark.asyncio
async def test_library_rerank_binding_does_not_queue_rebuild(
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
        generation_before = runtime.indexes.status()["generation"]
        await manager.create_provider(
            {
                "id": "local_rerank",
                "display_name": "Local Rerank",
                "type": "vllm_rerank",
                "enabled": True,
                "api_base": "http://127.0.0.1:8002",
                "api_suffix": "/v1/rerank",
                "model": "BAAI/bge-reranker-v2-m3",
                "dimensions": 0,
                "batch_size": 1,
                "concurrency": 1,
            }
        )
        updated = await manager.update_library(
            "Default", {"rerank_provider_id": "local_rerank"}
        )
        assert updated["rerank_provider_id"] == "local_rerank"
        assert runtime.indexes.status()["generation"] == generation_before
        assert runtime.rerank_provider_revision is not None
        assert runtime.rerank_provider_revision.provider_id == "local_rerank"
        assert manager.jobs is not None
        assert await manager.jobs.list(scope="all") == []
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_legacy_data_migrates_to_Default_and_libraries_are_isolated(
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
        default = await manager.library_detail("Default")
        assert default["name"] == "贝雷特"
        assert default["stats"]["total_memories"] == 1
        assert not (data / "livingmemory.db").exists()
        assert (data / "libraries" / "Default" / "livingmemory.db").exists()
        marker = json.loads(
            (data / ".multilibrary_migrated_v1.json").read_text(
                encoding="utf-8"
            )
        )
        assert marker["validation"] == "passed"
        assert (
            await (await manager.get_runtime("Default")).storage.get_document(
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
        assert (
            second["metadata"]["livingmemory_database_version"]
            == LIVINGMEMORY_DATABASE_VERSION
        )
        assert (
            second["compatibility"]["livingmemory_database_version"]
            == LIVINGMEMORY_DATABASE_VERSION
        )
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
            await (await manager.get_runtime("Default")).storage.statistics()
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
    assert not (data / "libraries" / "Default").exists()
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
        assert set(manager.runtimes) == {"Default"}
        libraries = await manager.list_libraries()
        assert {item["id"] for item in libraries} == {"Default", "second"}
        assert set(manager.runtimes) == {"Default"}

        runtime = manager.runtimes["Default"]
        provider = runtime.provider
        await manager.update_library(
            "second", {"recall_settings": {"top_k": 3, "importance_weight": 2.5}}
        )
        second = await manager.control.get_library("second")
        second_runtime = await manager.get_runtime("second")
        assert second is not None
        assert "top_k" not in second.recall_settings
        assert second_runtime.config.recall.importance_weight == 2.5
        assert runtime.config.recall.importance_weight == 1.0
        assert not hasattr(second_runtime.config.recall, "top_k")
        assert not hasattr(runtime.config.recall, "top_k")
        with pytest.raises(ValueError, match="默认记忆库不能删除"):
            await manager.delete_library("Default")
        assert manager.runtimes["Default"] is runtime
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
        first = await manager.copy_library("Default")
        second = await manager.copy_library("Default")
        assert first["id"] == "Default_copy"
        assert first["name"] == "贝雷特(副本)"
        assert second["id"] == "Default_copy2"
        assert second["name"] == "贝雷特(副本2)"

        await manager.delete_library(first["id"])
        third = await manager.copy_library("Default")
        assert third["id"] == "Default_copy3"
        assert third["name"] == "贝雷特(副本3)"
        assert not (root / "data" / "libraries" / first["id"]).exists()
        trash_candidates = list(
            (root / "data" / "trash" / "libraries").glob("Default_copy-*")
        )
        assert trash_candidates
        assert sorted(item.name for item in trash_candidates[0].iterdir()) == [
            "conversations.db",
            "livingmemory.db",
        ]

        libraries = {item["id"] for item in await manager.list_libraries()}
        assert libraries == {"Default", "Default_copy2", "Default_copy3"}
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_empty_library_copy_can_be_loaded_renamed_and_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "PersonalityRAG"
    provider_config = ProviderConfig(dimensions=8)
    _patch_fake_providers(monkeypatch)

    manager = LibraryManager(root, AppConfig(provider=provider_config))
    await manager.initialize()
    try:
        source = await manager.create_library(
            {
                "id": "empty_source",
                "name": "Empty source",
                "provider_id": provider_config.id,
            }
        )
        assert source["stats"]["total_memories"] == 0

        copied = await manager.copy_library(source["id"])
        copied_id = copied["id"]
        assert copied_id == "empty_source_copy"
        assert (await manager.library_detail(copied_id))["id"] == copied_id
        assert (await manager.get_runtime(copied_id)).library_id == copied_id

        renamed = await manager.update_library(
            copied_id,
            {"id": "renamed_empty_copy", "name": "Renamed empty copy"},
        )
        assert renamed["id"] == "renamed_empty_copy"
        assert renamed["name"] == "Renamed empty copy"
        assert await manager.control.get_library(copied_id) is None
        assert not (root / "data" / "libraries" / copied_id).exists()
        assert (root / "data" / "libraries" / renamed["id"]).is_dir()

        deleted = await manager.delete_library(renamed["id"])
        assert deleted["library_id"] == renamed["id"]
        assert await manager.control.get_library(renamed["id"]) is None
        assert not (root / "data" / "libraries" / renamed["id"]).exists()

        copied_again = await manager.copy_library(source["id"])
        reclaimed = await manager.update_library(
            copied_again["id"],
            {"id": renamed["id"], "name": "Reclaimed empty copy"},
        )
        assert reclaimed["id"] == renamed["id"]
        assert reclaimed["name"] == "Reclaimed empty copy"
        assert (root / "data" / "libraries" / reclaimed["id"]).is_dir()
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
        runtime = await manager.get_runtime("Default")
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
        await manager.rebuild_library("Default", switched["id"])
        binding = await manager.control.get_library("Default")
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
            await manager.rebuild_library("Default", failed["id"])
        binding = await manager.control.get_library("Default")
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
        runtime = await manager.get_runtime("Default")
        await runtime.create_memory(
            {
                "content": "贝雷特喜欢夜空与星辰",
                "persona_id": "Default",
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
            "喜欢什么", k=10, persona_id="Default"
        )
        assert by_persona
        assert all(
            item.metadata.get("persona_id") == "Default"
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
        runtime = await manager.get_runtime("Default")
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
            manager.rebuild_library("Default", provider["id"])
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


@pytest.mark.asyncio
async def test_runtime_residency_keeps_default_and_evicts_non_default_lru(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_fake_providers(monkeypatch)
    provider = ProviderConfig(dimensions=8)
    config = AppConfig(
        provider=provider,
        runtime_residency=RuntimeResidencyConfig(
            idle_minutes=30,
            max_non_default_runtimes=2,
        ),
    )
    manager = LibraryManager(tmp_path / "PersonalityRAG", config)
    await manager.initialize()
    try:
        for library_id in ("first", "second", "third"):
            await manager.create_library(
                {
                    "id": library_id,
                    "name": library_id,
                    "provider_id": provider.id,
                }
            )

        assert set(manager.runtimes) == {"Default", "second", "third"}
        assert manager.runtimes["Default"] is not None

        await manager.get_runtime("first")
        assert set(manager.runtimes) == {"Default", "first", "third"}
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_runtime_residency_lease_allows_temporary_overflow_then_converges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_fake_providers(monkeypatch)
    provider = ProviderConfig(dimensions=8)
    config = AppConfig(
        provider=provider,
        runtime_residency=RuntimeResidencyConfig(
            idle_minutes=30,
            max_non_default_runtimes=1,
        ),
    )
    manager = LibraryManager(tmp_path / "PersonalityRAG", config)
    await manager.initialize()
    try:
        await manager.create_library(
            {"id": "held", "name": "held", "provider_id": provider.id}
        )
        await manager.acquire_runtime("held")
        await manager.create_library(
            {"id": "waiting", "name": "waiting", "provider_id": provider.id}
        )

        assert set(manager.runtimes) == {"Default", "held", "waiting"}
        assert manager.runtime_residency_status()["runtimes"]["held"][
            "lease_count"
        ] == 1

        await manager.release_runtime("held")
        non_default = set(manager.runtimes) - {"Default"}
        assert len(non_default) == 1
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_job_runtime_lease_is_acquired_only_during_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_fake_providers(monkeypatch)
    provider = ProviderConfig(dimensions=8)
    manager = LibraryManager(
        tmp_path / "PersonalityRAG",
        AppConfig(
            provider=provider,
            runtime_residency=RuntimeResidencyConfig(
                idle_minutes=30,
                max_non_default_runtimes=1,
            ),
        ),
    )
    await manager.initialize()
    try:
        await manager.create_library(
            {"id": "held", "name": "held", "provider_id": provider.id}
        )
        await manager.acquire_runtime("held")
        await manager.create_library(
            {"id": "worker", "name": "worker", "provider_id": provider.id}
        )
        assert set(manager.runtimes) == {"Default", "held", "worker"}
        observed: list[int] = []

        async def operation(progress):
            observed.append(
                manager.runtime_residency_status()["runtimes"]["worker"][
                    "lease_count"
                ]
            )
            return {"ok": True}

        assert manager.jobs is not None
        job_id = await manager.jobs.start(
            "fixture_runtime_job",
            operation,
            library_id="worker",
            dedupe_active=False,
        )
        result = await asyncio.wait_for(manager.jobs.wait(job_id), timeout=2)

        assert result["status"] == "completed"
        assert observed == [1]
        assert set(manager.runtimes) == {"Default", "held"}
        await manager.release_runtime("held")
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_runtime_hot_limit_and_idle_reload_preserve_recall_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_fake_providers(monkeypatch)
    provider = ProviderConfig(dimensions=8)
    config = AppConfig(
        provider=provider,
        runtime_residency=RuntimeResidencyConfig(
            idle_minutes=1,
            max_non_default_runtimes=2,
        ),
    )
    manager = LibraryManager(tmp_path / "PersonalityRAG", config)
    await manager.initialize()
    try:
        for library_id in ("recall", "spare"):
            await manager.create_library(
                {
                    "id": library_id,
                    "name": library_id,
                    "provider_id": provider.id,
                }
            )
        runtime = await manager.get_runtime("recall")
        await runtime.create_memory(
            {
                "content": "贝雷特喜欢在夜空下观察星辰",
                "persona_id": "fixture",
                "topics": ["星空"],
            }
        )
        before = await runtime.retrieval.search("贝雷特 星辰", 5)

        config.runtime_residency.max_non_default_runtimes = 1
        await manager.apply_runtime_residency()
        assert len(set(manager.runtimes) - {"Default"}) == 1

        if "recall" not in manager.runtimes:
            runtime = await manager.get_runtime("recall")
        manager._runtime_residency["recall"].last_used_at -= 120
        assert await manager.sweep_runtimes() == ["recall"]
        assert "recall" not in manager.runtimes
        assert "Default" in manager.runtimes

        reloaded = await manager.get_runtime("recall")
        after = await reloaded.retrieval.search("贝雷特 星辰", 5)
        assert [item.doc_id for item in after] == [item.doc_id for item in before]
        assert [item.final_score for item in after] == pytest.approx(
            [item.final_score for item in before],
            abs=1e-12,
        )
    finally:
        await manager.close()
