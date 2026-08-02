from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from personalityrag.atoms import classify_memory_atoms, compute_atom_ttl
from personalityrag.config import AppConfig, MaintenanceConfig, RecallConfig
from personalityrag.graph import GraphBuilder
from personalityrag.indexes import IndexManager
from personalityrag.providers import (
    EmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    RerankResult,
)
from personalityrag.retrieval import RetrievalEngine, SearchResult, _GraphHit
from personalityrag.service import PersonalityRAGService
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


class FakeReranker:
    def __init__(self, rows: list[RerankResult]):
        self.rows = rows
        self.calls: list[dict[str, object]] = []

    async def rerank(self, query: str, documents: list[str], top_n: int | None = None):
        self.calls.append({"query": query, "documents": documents, "top_n": top_n})
        return list(self.rows)

    async def close(self) -> None:
        pass


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


def test_service_auto_builds_atoms_from_key_facts(tmp_path: Path):
    service = PersonalityRAGService.__new__(PersonalityRAGService)
    service.config = AppConfig(maintenance=MaintenanceConfig(atom_enabled=True))

    payload = service._payload_for_write(
        {
            "content": "张三明天要开会",
            "session_id": "s1",
            "persona_id": "p1",
            "importance": 0.8,
            "topics": ["会议"],
            "participants": ["张三"],
            "key_facts": ["张三明天要开会"],
        }
    )

    assert payload["atoms"] == [
        {
            "atom_type": "planned",
            "content": "张三明天要开会",
            "entities": ["会议", "张三"],
            "importance": 0.8,
            "confidence": 0.85,
            "event_time": payload["atoms"][0]["event_time"],
            "session_id": "s1",
            "persona_id": "p1",
            "metadata": {"source": "summary_key_fact"},
        }
    ]
    assert payload["atoms"][0]["event_time"] is not None


def test_service_memory_write_prefers_request_persona_then_library_default():
    service = PersonalityRAGService.__new__(PersonalityRAGService)
    service.config = AppConfig(maintenance=MaintenanceConfig(atom_enabled=False))
    service.default_persona_id = "贝雷特"

    explicit = service._payload_for_write(
        {"content": "卡缪请求写入的记忆", "persona_id": "卡缪"}
    )
    fallback = service._payload_for_write({"content": "未携带人格的记忆"})

    assert explicit["persona_id"] == "卡缪"
    assert fallback["persona_id"] == "贝雷特"


def test_core_atom_classifier_matches_livingmemory_types():
    atoms = classify_memory_atoms(
        key_facts=[
            "明天下午3点开会讨论项目进度",
            "张三喜欢喝咖啡",
            "张三和李四是同事关系",
            "张三的生日是5月20日",
            "张三讨论了Q3项目进展",
            "嗯好",
        ],
        topics=["会议"],
        participants=["张三"],
        parent_importance=0.8,
        session_id="s1",
        persona_id="p1",
    )

    assert [atom["atom_type"] for atom in atoms] == [
        "planned",
        "preference",
        "relational",
        "factual",
        "episodic",
        "unknown",
    ]
    assert [atom["confidence"] for atom in atoms] == [0.85, 0.82, 0.80, 0.78, 0.75, 0.60]
    assert atoms[0]["event_time"] is not None
    assert all(atom["entities"] == ["会议", "张三"] for atom in atoms)
    assert all(atom["session_id"] == "s1" for atom in atoms)
    assert all(atom["persona_id"] == "p1" for atom in atoms)


def test_core_atom_ttl_matches_livingmemory_baseline():
    ttl, decay = compute_atom_ttl("relational", importance=0.5)
    assert ttl == pytest.approx(90.0)
    assert decay == "linear"

    ttl, decay = compute_atom_ttl("preference", importance=0.5)
    assert ttl == pytest.approx(60.0)
    assert decay == "exponential"

    future = 1_893_456_000.0
    ttl, decay = compute_atom_ttl(
        "planned",
        importance=0.8,
        event_time=future,
    )
    assert ttl > 2.0 * 1.3
    assert decay == "step"


def test_service_respects_atom_disabled_for_generated_atoms(tmp_path: Path):
    service = PersonalityRAGService.__new__(PersonalityRAGService)
    service.config = AppConfig(maintenance=MaintenanceConfig(atom_enabled=False))

    payload = service._payload_for_write(
        {
            "content": "张三明天要开会",
            "key_facts": ["张三明天要开会"],
            "atoms": [{"atom_type": "factual", "content": "张三明天要开会"}],
        }
    )

    assert payload["atoms"] == []


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


@pytest.mark.asyncio
async def test_graph_route_scores_graph_entry_metadata_without_loading_memory():
    class NoMemoryStorage:
        async def get_document(self, doc_id: int):
            raise AssertionError(f"memory {doc_id} should not be loaded")

    now = 1_800_000_000.0
    engine = RetrievalEngine(
        NoMemoryStorage(),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        TextProcessor(),
        RecallConfig(decay_rate=0),
    )

    async def fake_keyword(*_args, **_kwargs):
        return [
            _GraphHit(
                7,
                1.0,
                "Fact: graph entry",
                {
                    "importance": 0.2,
                    "graph_confidence": 1.0,
                    "create_time": now,
                    "last_access_time": now,
                },
            )
        ]

    async def fake_vector(*_args, **_kwargs):
        return []

    engine._graph_keyword = fake_keyword  # type: ignore[method-assign]
    engine._graph_vector = fake_vector  # type: ignore[method-assign]

    results = await engine._graph_route("graph", 1, None, None)

    assert [item.doc_id for item in results] == [7]
    assert results[0].content == "Fact: graph entry"
    assert results[0].final_score == pytest.approx(0.84)
    assert results[0].score_breakdown["graph_confidence"] == 1.0


@pytest.mark.asyncio
async def test_merge_routes_backfills_memory_for_graph_only_hits():
    class FakeStorage:
        async def get_document(self, doc_id: int):
            assert doc_id == 7
            return {
                "text": "canonical memory text",
                "metadata": {"importance": 0.9, "source": "memory"},
            }

    engine = RetrievalEngine(
        FakeStorage(),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        TextProcessor(),
        RecallConfig(),
    )
    graph_hit = SearchResult(
        doc_id=7,
        final_score=0.5,
        rrf_score=0.1,
        bm25_score=None,
        vector_score=0.8,
        content="Fact: graph entry",
        metadata={"source_memory_id": 7},
        score_breakdown={"graph_final_score": 0.5},
    )

    results = await engine._merge_routes("graph", [], [graph_hit], 1)

    assert [item.doc_id for item in results] == [7]
    assert results[0].content == "canonical memory text"
    assert results[0].metadata == {
        "importance": 0.9,
        "source": "memory",
        "has_source": False,
    }


@pytest.mark.asyncio
async def test_service_keeps_healthy_fts_during_index_rebuild():
    class HealthyStorage:
        async def fts_integrity_report(self):
            return {
                "documents": 2,
                "document_fts": 2,
                "graph_entries": 3,
                "graph_fts": 3,
                "active_atoms": 1,
                "atom_fts": 9,
            }

        async def rebuild_fts(self, _tokenize):
            raise AssertionError("healthy FTS should be preserved")

    service = PersonalityRAGService.__new__(PersonalityRAGService)
    service.storage = HealthyStorage()
    service.text = TextProcessor()
    service.library_id = "test"

    result = await service._ensure_fts_recovery_unlocked()

    assert result["rebuilt"] is False
    assert result["reason"] == "not_needed"


@pytest.mark.asyncio
async def test_service_repairs_incomplete_fts_during_index_rebuild():
    class IncompleteStorage:
        def __init__(self):
            self.calls = 0

        async def fts_integrity_report(self):
            if self.calls:
                return {
                    "documents": 2,
                    "document_fts": 2,
                    "graph_entries": 3,
                    "graph_fts": 3,
                    "active_atoms": 1,
                    "atom_fts": 1,
                }
            return {
                "documents": 2,
                "document_fts": 1,
                "graph_entries": 3,
                "graph_fts": 3,
                "active_atoms": 1,
                "atom_fts": 1,
            }

        async def rebuild_fts(self, _tokenize):
            self.calls += 1
            return {"documents": 2, "graph_entries": 3, "atoms": 1}

    storage = IncompleteStorage()
    service = PersonalityRAGService.__new__(PersonalityRAGService)
    service.storage = storage
    service.text = TextProcessor()
    service.library_id = "test"

    result = await service._ensure_fts_recovery_unlocked()

    assert result["rebuilt"] is True
    assert result["reason"] == "document_fts_count_mismatch"
    assert storage.calls == 1


@pytest.mark.asyncio
async def test_document_route_matches_livingmemory_250_weighting():
    class FakeStorage:
        async def get_document(self, doc_id: int):
            importance = 0.9 if doc_id == 2 else 0.2
            return {
                "text": f"memory {doc_id}",
                "metadata": {"importance": importance},
            }

    engine = RetrievalEngine(
        FakeStorage(),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        TextProcessor(),
        RecallConfig(
            score_alpha=0,
            score_beta=1,
            score_gamma=0,
            importance_weight=2,
        ),
    )

    async def fake_bm25(*_args, **_kwargs):
        return [(1, 1.0), (2, 0.9)]

    async def fake_vector(*_args, **_kwargs):
        return []

    engine._bm25_documents = fake_bm25  # type: ignore[method-assign]
    engine._vector_documents = fake_vector  # type: ignore[method-assign]

    results = await engine._document_route("memory", 2, None, None)

    assert [item.doc_id for item in results] == [2, 1]
    assert results[0].final_score == pytest.approx(0.9)
    assert results[0].score_breakdown == {
        "rrf_normalized": pytest.approx(0.9839),
        "importance": 0.9,
        "recency_weight": 1.0,
        "days_old": 0.0,
        "final_score": 0.9,
    }


def _rerank_service(
    *,
    graph_enabled: bool = True,
    rows: list[RerankResult] | None = None,
) -> PersonalityRAGService:
    service = PersonalityRAGService.__new__(PersonalityRAGService)
    service.library_id = "test"
    service.config = AppConfig(
        recall=RecallConfig(
            graph_memory_enabled=graph_enabled,
            document_route_weight=0.35,
            graph_route_weight=0.65,
            cross_route_bonus=0.08,
            dynamic_route_weighting=False,
        )
    )
    service.reranker = FakeReranker(rows or [])
    service.rerank_provider_revision = SimpleNamespace(
        provider_id="rerank_fixture",
        config=SimpleNamespace(type="vllm_rerank"),
    )
    service.provider = DeterministicProvider()
    service.text = TextProcessor()
    service.retrieval = SimpleNamespace(
        _route_weights=lambda _query: (0.35, 0.65, "fixed")
    )
    return service


@pytest.mark.asyncio
async def test_graph_disabled_rerank_keeps_plain_rerank_behavior():
    service = _rerank_service(
        graph_enabled=False,
        rows=[RerankResult(2, 0.9), RerankResult(1, 0.8)],
    )
    candidates = [
        _result(1, 0.3, "alpha"),
        _result(2, 0.2, "beta"),
        _result(3, 0.1, "gamma"),
    ]

    results, meta = await service.apply_rerank("query", candidates, 2)

    assert [item.doc_id for item in results] == [3, 2]
    assert service.reranker.calls[0]["top_n"] == 2
    assert meta["graph_enhanced"] is False
    assert meta["graph_candidate_count"] == 0


@pytest.mark.asyncio
async def test_graph_enhanced_rerank_promotes_candidate_with_graph_signal():
    service = _rerank_service(
        rows=[
            RerankResult(0, 1.0),
            RerankResult(1, 0.95),
            RerankResult(2, 0.1),
        ]
    )
    candidates = [
        _result(1, 0.3, "plain text winner"),
        _result(2, 0.2, "graph backed text"),
        _result(3, 0.1, "other text"),
    ]

    async def fake_graph_signals(_query, _candidates):
        return {
            2: {
                "graph_score": 1.0,
                "keyword_score": 1.0,
                "vector_score": 0.0,
                "node_score": 1.0,
                "evidence_count": 1,
                "entries": [
                    {
                        "content": "candidate graph evidence",
                        "entry_type": "fact",
                        "relation_type": "fact",
                    }
                ],
            }
        }

    service._rerank_graph_signals = fake_graph_signals  # type: ignore[method-assign]

    results, meta = await service.apply_rerank("relationship query", candidates, 2)

    assert [item.doc_id for item in results] == [2, 1]
    assert service.reranker.calls[0]["top_n"] == len(candidates)
    assert "[Graph evidence for rerank]" in service.reranker.calls[0]["documents"][1]
    assert meta["graph_enhanced"] is True
    assert meta["graph_candidate_count"] == 1
    assert results[0].score_breakdown["rerank_graph_score"] == 1.0
    assert results[0].score_breakdown["rerank_graph_evidence_count"] == 1


@pytest.mark.asyncio
async def test_rerank_graph_helper_is_scoped_to_embedding_candidates():
    class FakeStorage:
        def __init__(self):
            self.candidate_ids: list[int] | None = None

        async def candidate_graph_evidence(self, candidate_ids, *_args, **_kwargs):
            self.candidate_ids = list(candidate_ids)
            return {
                2: {
                    "keyword_score": 1.0,
                    "node_score": 0.0,
                    "graph_confidence": 0.7,
                    "entries": [{"content": "candidate evidence"}],
                },
                99: {
                    "keyword_score": 1.0,
                    "node_score": 1.0,
                    "graph_confidence": 1.0,
                    "entries": [{"content": "outside candidate evidence"}],
                },
            }

    service = _rerank_service(
        rows=[RerankResult(0, 0.9), RerankResult(1, 0.8), RerankResult(2, 0.7)]
    )
    storage = FakeStorage()
    service.storage = storage

    async def fake_vector_scores(_query, _raw):
        return {2: 0.5, 99: 1.0}

    service._rerank_graph_vector_scores = fake_vector_scores  # type: ignore[method-assign]
    candidates = [
        _result(1, 0.3, "one"),
        _result(2, 0.2, "two"),
        _result(3, 0.1, "three"),
    ]

    results, meta = await service.apply_rerank("query", candidates, 3)

    assert storage.candidate_ids == [1, 2, 3]
    assert {item.doc_id for item in results} <= {1, 2, 3}
    assert 99 not in [item.doc_id for item in results]
    assert meta["graph_candidate_count"] == 1


@pytest.mark.asyncio
async def test_rerank_graph_vector_failure_falls_back_to_plain_rerank():
    class FakeStorage:
        async def candidate_graph_evidence(self, *_args, **_kwargs):
            return {
                1: {
                    "keyword_score": 1.0,
                    "node_score": 0.0,
                    "graph_confidence": 0.7,
                    "entries": [{"content": "candidate evidence"}],
                }
            }

    service = _rerank_service(rows=[RerankResult(1, 0.9), RerankResult(0, 0.8)])
    service.storage = FakeStorage()

    async def failing_vector_scores(_query, _raw):
        raise RuntimeError("vector unavailable")

    service._rerank_graph_vector_scores = failing_vector_scores  # type: ignore[method-assign]
    candidates = [_result(1, 0.2, "one"), _result(2, 0.1, "two")]

    results, meta = await service.apply_rerank("query", candidates, 1)

    assert [item.doc_id for item in results] == [2]
    assert service.reranker.calls[0]["top_n"] == 1
    assert meta["graph_enhanced"] is False


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
