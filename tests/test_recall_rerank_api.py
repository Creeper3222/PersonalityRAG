from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request

from personalityrag import app as app_module
from personalityrag.config import RecallConfig
from personalityrag.retrieval import SearchResult
from personalityrag.retrieval import RetrievalEngine
from personalityrag.routes import recall_graph as recall_routes
from personalityrag.schemas import RecallRequest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/memory-libraries/livingmemory_v8/fixture/recall",
            "headers": [],
            "state": {"request_id": "fixture-request-id"},
        }
    )


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
        self.filter_calls: list[tuple[str | None, str | None]] = []

    async def search(self, query, k, session_id=None, persona_id=None):
        self.calls.append(k)
        self.filter_calls.append((session_id, persona_id))
        return [_result(index + 1) for index in range(k)]


class FakeTarget:
    def __init__(self, with_reranker: bool = True) -> None:
        self.library_id = "fixture"
        self.default_persona_id = "贝雷特"
        self.config = SimpleNamespace(
            recall=RecallConfig(
                use_session_filtering=False,
                use_persona_filtering=False,
            )
        )
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


class FailingRerankTarget(FakeTarget):
    async def apply_rerank(self, query, candidates, k):
        raise RuntimeError("fixture rerank outage")


@pytest.mark.asyncio
async def test_recall_uses_embedding_k_as_candidate_pool_and_rerank_k_output(
    monkeypatch: pytest.MonkeyPatch,
):
    target = FakeTarget()

    async def fake_runtime(library_id=None):
        return target

    monkeypatch.setattr(recall_routes, "runtime", fake_runtime)
    monkeypatch.setattr(app_module.config.recall, "use_session_filtering", False)
    monkeypatch.setattr(app_module.config.recall, "use_persona_filtering", False)

    response = await app_module.recall(
        RecallRequest(
            query="fixture",
            k=75,
            rerank_k=60,
            rerank=True,
            include_baseline=True,
        ),
        _request(),
        "fixture",
        database_type="livingmemory_v8",
    )

    assert target.retrieval.calls == [75]
    assert response["embedding_k"] == 75
    assert response["rerank_k"] == 60
    assert len(response["baseline_results"]) == 75
    assert len(response["results"]) == 60
    assert response["baseline_results"][0]["memory_id"] == 1
    assert response["results"][0]["memory_id"] == 75
    assert response["rerank"]["candidate_count"] == 75


@pytest.mark.asyncio
async def test_recall_rerank_failure_falls_back_to_full_embedding_baseline(
    monkeypatch: pytest.MonkeyPatch,
):
    target = FailingRerankTarget()

    async def fake_runtime(library_id=None):
        return target

    monkeypatch.setattr(recall_routes, "runtime", fake_runtime)
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
        _request(),
        "fixture",
        database_type="livingmemory_v8",
    )

    assert target.retrieval.calls == [20]
    assert response["embedding_k"] == 20
    assert response["rerank_k"] == 10
    assert len(response["results"]) == 20
    assert len(response["baseline_results"]) == 20
    assert response["results"][0]["memory_id"] == 1
    assert response["rerank"]["failed"] is True


@pytest.mark.asyncio
async def test_recall_rejects_rerank_k_larger_than_embedding_k(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_runtime(library_id=None):
        return FakeTarget()

    monkeypatch.setattr(recall_routes, "runtime", fake_runtime)
    with pytest.raises(HTTPException) as exc:
        await app_module.recall(
            RecallRequest(query="fixture", k=5, rerank_k=6, rerank=True),
            _request(),
            "fixture",
            database_type="livingmemory_v8",
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_recall_uses_target_library_persona_and_session_filter_settings(
    monkeypatch: pytest.MonkeyPatch,
):
    target = FakeTarget(with_reranker=False)
    target.config.recall.use_session_filtering = True
    target.config.recall.use_persona_filtering = True

    async def fake_runtime(library_id=None):
        return target

    monkeypatch.setattr(recall_routes, "runtime", fake_runtime)
    monkeypatch.setattr(app_module.config.recall, "use_session_filtering", False)
    monkeypatch.setattr(app_module.config.recall, "use_persona_filtering", False)

    await app_module.recall(
        RecallRequest(
            query="persona fixture",
            k=5,
            session_id="astrbot-session",
            persona_id="astrbot-persona",
            rerank=False,
        ),
        _request(),
        "fixture",
        database_type="livingmemory_v8",
    )

    assert target.retrieval.filter_calls == [
        ("astrbot-session", "astrbot-persona")
    ]

    await app_module.recall(
        RecallRequest(
            query="fallback persona fixture",
            k=5,
            session_id="astrbot-session",
            rerank=False,
        ),
        _request(),
        "fixture",
        database_type="livingmemory_v8",
    )

    assert target.retrieval.filter_calls[-1] == (
        "astrbot-session",
        "贝雷特",
    )


@pytest.mark.asyncio
async def test_recall_disables_rerank_chain_without_bound_provider(
    monkeypatch: pytest.MonkeyPatch,
):
    target = FakeTarget(with_reranker=False)

    async def fake_runtime(library_id=None):
        return target

    monkeypatch.setattr(recall_routes, "runtime", fake_runtime)
    monkeypatch.setattr(app_module.config.recall, "use_session_filtering", False)
    monkeypatch.setattr(app_module.config.recall, "use_persona_filtering", False)

    response = await app_module.recall(
        RecallRequest(query="fixture", k=5, rerank_k=10, rerank=True),
        _request(),
        "fixture",
        database_type="livingmemory_v8",
    )

    assert target.retrieval.calls == [5]
    assert len(response["results"]) == 5
    assert "rerank_candidates" not in response
    assert response["rerank"]["available"] is False
    assert response["rerank"]["requested"] is False
    assert response["rerank"]["applied"] is False
    assert response["rerank"].get("failed") is None


@pytest.mark.asyncio
async def test_retrieval_search_does_not_cap_runtime_k_at_fifty():
    class DummyStorage:
        async def touch_documents(self, ids):
            return None

    route_calls: list[int] = []
    engine = RetrievalEngine(
        DummyStorage(),
        SimpleNamespace(),
        SimpleNamespace(),
        RecallConfig(search_cache_enabled=False, recent_memory_count=0),
    )

    async def fake_document_route(query, k, session_id, persona_id):
        route_calls.append(k)
        return []

    async def fake_graph_route(query, k, session_id, persona_id):
        route_calls.append(k)
        return []

    async def fake_merge(query, documents, graph, k):
        return [_result(index + 1) for index in range(k)]

    engine._document_route = fake_document_route  # type: ignore[method-assign]
    engine._graph_route = fake_graph_route  # type: ignore[method-assign]
    engine._merge_routes = fake_merge  # type: ignore[method-assign]

    results = await engine.search("fixture", 75)

    assert len(results) == 75
    assert route_calls == [150, 150]


def test_library_recall_ui_does_not_expose_top_k_setting():
    html = (REPO_ROOT / "static" / "index.html").read_text(encoding="utf-8")
    js = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            REPO_ROOT / "static" / "app.js",
            REPO_ROOT / "static" / "modules" / "libraries.js",
        )
    )

    assert 'id="library-recall-top-k"' not in html
    assert "DEFAULT_LIBRARY_RECALL_SETTINGS.top_k" not in js
