from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from personalityrag.config import RecallConfig
from personalityrag.graph import GraphBuilder
from personalityrag.indexes import IndexManager
from personalityrag.providers import (
    EmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from personalityrag.retrieval import RetrievalEngine, SearchResult
from personalityrag.storage import Storage
from personalityrag.text import TextProcessor


class DeterministicProvider(EmbeddingProvider):
    dimension = 8

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
        return [{"id": "deterministic"}]

    async def detect_context_length(self):
        return {"max_context_tokens": 4096, "max_context_tokens_source": "auto:fake"}

    async def test_connection(self):
        return {
            "available": True,
            "resolved_model": "deterministic",
            "dimension": self.dimension,
        }

    async def close(self) -> None:
        pass


class FailingProvider(DeterministicProvider):
    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("intentional rebuild failure")


def _result(memory_id: int, score: float, content: str) -> SearchResult:
    return SearchResult(
        doc_id=memory_id,
        final_score=score,
        rrf_score=score,
        bm25_score=score,
        vector_score=score,
        content=content,
        metadata={},
        score_breakdown={},
    )


def test_dynamic_route_weights_follow_query_intent():
    engine = RetrievalEngine(
        None, None, TextProcessor(), RecallConfig()  # type: ignore[arg-type]
    )
    relation_document, relation_graph, relation_intent = engine._route_weights(
        "谁和澄月是朋友"
    )
    factual_document, factual_graph, factual_intent = engine._route_weights(
        "解释什么是人格记忆"
    )
    assert relation_graph > relation_document
    assert relation_intent == "relationship"
    assert factual_document > factual_graph
    assert factual_intent == "factual"


def test_mmr_prefers_diversity_after_highest_score():
    engine = RetrievalEngine(
        None, None, TextProcessor(), RecallConfig(mmr_lambda=0.7)  # type: ignore[arg-type]
    )
    selected = engine._mmr(
        [
            _result(1, 1.0, "澄月 喜欢 星空"),
            _result(2, 0.99, "澄月 喜欢 星空"),
            _result(3, 0.90, "哈萨维 维护 RocketCatShell"),
        ],
        2,
    )
    assert [item.doc_id for item in selected] == [1, 3]


def test_provider_bypasses_environment_proxy_for_local_and_private_urls():
    assert OpenAICompatibleEmbeddingProvider._is_local_or_private(
        "http://127.0.0.1:8001/v1"
    )
    assert OpenAICompatibleEmbeddingProvider._is_local_or_private(
        "http://192.168.1.10:8001/v1"
    )
    assert not OpenAICompatibleEmbeddingProvider._is_local_or_private(
        "https://embedding.example.com/v1"
    )


@pytest.mark.asyncio
async def test_atom_and_operation_log_are_removed_with_memory(tmp_path: Path):
    storage = Storage(tmp_path)
    await storage.initialize()
    memory_id = await storage.create_memory(
        {
            "content": "澄月记得今晚观察星空",
            "persona_id": "贝雷特",
            "topics": ["星空"],
            "key_facts": ["今晚观察星空"],
            "atoms": [
                {
                    "atom_type": "planned",
                    "content": "今晚观察星空",
                    "importance": 0.8,
                    "confidence": 0.9,
                }
            ],
        },
        TextProcessor().tokenize,
        GraphBuilder().build,
    )
    async with storage.connect() as db:
        atom = await (
            await db.execute(
                "SELECT status,ttl_days FROM memory_atoms WHERE parent_memory_id=?",
                (memory_id,),
            )
        ).fetchone()
        operation = await (
            await db.execute(
                "SELECT status,step FROM memory_write_ops WHERE memory_id=?",
                (memory_id,),
            )
        ).fetchone()
    assert atom["status"] == "active"
    assert float(atom["ttl_days"]) > 2
    assert (operation["status"], operation["step"]) == (
        "needs_index",
        "database_committed",
    )
    assert await storage.delete_memories([memory_id]) == 1
    async with storage.connect() as db:
        assert (
            await (
                await db.execute(
                    "SELECT COUNT(*) FROM memory_atoms WHERE parent_memory_id=?",
                    (memory_id,),
                )
            ).fetchone()
        )[0] == 0
        assert (
            await (
                await db.execute(
                    "SELECT COUNT(*) FROM memory_write_ops WHERE memory_id=?",
                    (memory_id,),
                )
            ).fetchone()
        )[0] == 0


@pytest.mark.asyncio
async def test_failed_rebuild_keeps_previous_generation_active(tmp_path: Path):
    storage = Storage(tmp_path)
    await storage.initialize()
    await storage.create_memory(
        {"content": "安全切换索引", "topics": ["索引"]},
        TextProcessor().tokenize,
        GraphBuilder().build,
    )
    healthy = IndexManager(
        tmp_path, storage, DeterministicProvider(), "deterministic"
    )
    await healthy.initialize()
    first = await healthy.rebuild()
    current_before = (tmp_path / "indexes" / "CURRENT").read_text().strip()
    assert current_before == first["generation"]

    failing = IndexManager(
        tmp_path, storage, FailingProvider(), "intentional-failure"
    )
    await failing.initialize()
    with pytest.raises(RuntimeError, match="intentional rebuild failure"):
        await failing.rebuild()
    assert (tmp_path / "indexes" / "CURRENT").read_text().strip() == current_before
