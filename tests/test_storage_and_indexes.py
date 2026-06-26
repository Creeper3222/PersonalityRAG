from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from personalityrag.config import RecallConfig
from personalityrag.graph import GraphBuilder
from personalityrag.indexes import IndexManager
from personalityrag.providers import EmbeddingProvider
from personalityrag.retrieval import RetrievalEngine
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

    async def test_connection(self):
        return {
            "available": True,
            "resolved_model": "fake",
            "dimension": self.dimension,
        }

    async def close(self) -> None:
        pass


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
    engine = RetrievalEngine(storage, indexes, text, RecallConfig())
    results = await engine.search("谁喜欢星空", 2, persona_id="贝雷特")
    assert results
    assert {item.doc_id for item in results} <= {first, second}
    assert await storage.delete_memories([first]) == 1
    assert await storage.get_document(first) is None

