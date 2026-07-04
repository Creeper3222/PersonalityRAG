from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from personalityrag import app as app_module
from personalityrag.retrieval import SearchResult
from personalityrag.schemas import RecallRequest


def _result(memory_id: int) -> SearchResult:
    return SearchResult(
        doc_id=memory_id,
        final_score=float(memory_id),
        rrf_score=0.0,
        bm25_score=None,
        vector_score=None,
        content=f"memory {memory_id}",
        metadata={},
        score_breakdown={},
    )


class FakeRetrieval:
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def search(self, query, k, session_id=None, persona_id=None):
        self.calls.append(k)
        return [_result(index + 1) for index in range(k)]


class FakeTarget:
    def __init__(self, with_reranker: bool = True) -> None:
        self.library_id = "fixture"
        self.indexes = SimpleNamespace(status=lambda: {"generation": "gen-fixture"})
        self.retrieval = FakeRetrieval()
        self.reranker = object() if with_reranker else None
        self.rerank_provider_revision = (
            SimpleNamespace(
                provider_id="rerank_fixture",
                config=SimpleNamespace(type="vllm_rerank"),
            )
            if with_reranker
            else None
        )

    async def apply_rerank(self, query, candidates, k):
        return list(reversed(candidates))[:k], {
            "requested": True,
            "applied": True,
            "provider_id": "rerank_fixture",
            "provider_type": "vllm_rerank",
            "candidate_count": len(candidates),
            "returned": k,
        }


@pytest.mark.asyncio
async def test_recall_uses_embedding_k_as_candidate_pool_and_rerank_k_output(
    monkeypatch: pytest.MonkeyPatch,
):
    target = FakeTarget()

    async def fake_runtime(library_id=None):
        return target

    monkeypatch.setattr(app_module, "runtime", fake_runtime)
    monkeypatch.setattr(app_module.config.recall, "use_session_filtering", False)
    monkeypatch.setattr(app_module.config.recall, "use_persona_filtering", False)

    response = await app_module.recall(
        RecallRequest(
            query="fixture",
            k=20,
            rerank_k=10,
            rerank=True,
            include_baseline=True,
        ),
        "fixture",
    )

    assert target.retrieval.calls == [20]
    assert response["embedding_k"] == 20
    assert response["rerank_k"] == 10
    assert len(response["baseline_results"]) == 20
    assert len(response["results"]) == 10
    assert response["baseline_results"][0]["memory_id"] == 1
    assert response["results"][0]["memory_id"] == 20
    assert response["rerank"]["candidate_count"] == 20


@pytest.mark.asyncio
async def test_recall_rejects_rerank_k_larger_than_embedding_k(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_runtime(library_id=None):
        return FakeTarget()

    monkeypatch.setattr(app_module, "runtime", fake_runtime)
    with pytest.raises(HTTPException) as exc:
        await app_module.recall(
            RecallRequest(query="fixture", k=5, rerank_k=6, rerank=True),
            "fixture",
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_recall_disables_rerank_chain_without_bound_provider(
    monkeypatch: pytest.MonkeyPatch,
):
    target = FakeTarget(with_reranker=False)

    async def fake_runtime(library_id=None):
        return target

    monkeypatch.setattr(app_module, "runtime", fake_runtime)
    monkeypatch.setattr(app_module.config.recall, "use_session_filtering", False)
    monkeypatch.setattr(app_module.config.recall, "use_persona_filtering", False)

    response = await app_module.recall(
        RecallRequest(query="fixture", k=5, rerank_k=10, rerank=True),
        "fixture",
    )

    assert target.retrieval.calls == [5]
    assert len(response["results"]) == 5
    assert "rerank_candidates" not in response
    assert response["rerank"]["available"] is False
    assert response["rerank"]["requested"] is False
    assert response["rerank"]["applied"] is False
    assert response["rerank"].get("failed") is None
