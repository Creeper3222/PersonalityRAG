from __future__ import annotations

import time
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from ..application_context import manager
from ..http_shared import (
    jobs,
    require_auth,
    runtime,
)
from ..logger import logger, safe_summary
from ..schemas import (
    GraphQuery,
    RecallRequest,
)


router = APIRouter()
@router.post("/api/v1/recall", dependencies=[Depends(require_auth)])
@router.post(
    "/api/v1/libraries/{library_id}/recall",
    dependencies=[Depends(require_auth)],
)
async def recall(payload: RecallRequest, library_id: str | None = None):
    target = await runtime(library_id)
    started = time.perf_counter()
    recall_config = target.config.recall
    session_filter = payload.session_id if recall_config.use_session_filtering else None
    requested_persona_id = str(payload.persona_id or "").strip()
    effective_persona_id = requested_persona_id or target.default_persona_id or None
    persona_filter = effective_persona_id if recall_config.use_persona_filtering else None
    generation = (target.indexes.status() or {}).get("generation")
    embedding_k = int(payload.k)
    rerank_k = int(payload.rerank_k or payload.k)
    use_rerank = bool(target.reranker) and (
        True if payload.rerank is None else bool(payload.rerank)
    )
    if use_rerank and rerank_k > embedding_k:
        raise HTTPException(400, "重排输出条数不能大于嵌入召回条数")
    logger.info(
        "开始召回：library_id=%s embedding_k=%s rerank_k=%s persona=%s session=%s generation=%s query=%s",
        target.library_id,
        embedding_k,
        rerank_k,
        persona_filter or "",
        session_filter or "",
        generation or "",
        safe_summary(payload.query),
    )
    candidate_k = embedding_k
    candidates = await target.retrieval.search(
        payload.query,
        candidate_k,
        session_filter,
        persona_filter,
    )
    baseline_results = candidates[:embedding_k]
    results = candidates[:embedding_k]
    rerank_meta: dict[str, Any] = {
        "requested": use_rerank,
        "applied": False,
        "available": bool(target.reranker),
        "embedding_k": embedding_k,
        "rerank_k": rerank_k,
        "provider_id": (
            target.rerank_provider_revision.provider_id
            if target.rerank_provider_revision
            else ""
        ),
        "provider_type": (
            target.rerank_provider_revision.config.type
            if target.rerank_provider_revision
            else ""
        ),
    }
    if use_rerank:
        if target.reranker:
            try:
                results, rerank_meta = await target.apply_rerank(
                    payload.query,
                    candidates,
                    rerank_k,
                )
                rerank_meta["embedding_k"] = embedding_k
                rerank_meta["rerank_k"] = rerank_k
            except Exception as exc:
                logger.warning(
                    "Rerank 失败，回退未重排结果：library_id=%s provider=%s err=%s",
                    target.library_id,
                    rerank_meta.get("provider_id") or "",
                    safe_summary(exc),
                )
                rerank_meta.update(
                    {
                        "requested": True,
                        "applied": False,
                        "failed": True,
                        "error": str(exc),
                        "candidate_count": len(candidates),
                        "embedding_k": embedding_k,
                        "rerank_k": rerank_k,
                    }
                )
                results = baseline_results
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "召回完成：library_id=%s total=%s elapsed_ms=%.2f top_ids=%s",
        target.library_id,
        len(results),
        elapsed_ms,
        [item.doc_id for item in results[:10]],
    )
    response = {
        "query": payload.query,
        "library_id": target.library_id,
        "results": [item.to_dict() for item in results],
        "total": len(results),
        "embedding_k": embedding_k,
        "rerank_k": rerank_k,
        "elapsed_time_ms": elapsed_ms,
        "rerank": rerank_meta,
    }
    if payload.include_baseline:
        response["baseline_results"] = [
            item.to_dict() for item in baseline_results
        ]
    if use_rerank:
        response["rerank_candidates"] = [item.to_dict() for item in candidates]
    return response

@router.get("/api/v1/graph/overview", dependencies=[Depends(require_auth)])
@router.get(
    "/api/v1/libraries/{library_id}/graph/overview",
    dependencies=[Depends(require_auth)],
)
async def graph_overview(
    library_id: str | None = None,
    session_id: str | None = None,
    persona_id: str | None = None,
):
    target = await runtime(library_id)
    return {
        "snapshot": await target.graph_snapshot(
            session_id=session_id,
            persona_id=persona_id,
            limit_nodes=48,
            minimum_degree=2,
        ),
        "stats": await target.storage.statistics(),
    }

@router.post("/api/v1/graph/query", dependencies=[Depends(require_auth)])
@router.post(
    "/api/v1/libraries/{library_id}/graph/query",
    dependencies=[Depends(require_auth)],
)
async def graph_query(payload: GraphQuery, library_id: str | None = None):
    target = await runtime(library_id)
    logger.debug(
        "图谱查询：library_id=%s memory_id=%s limit=%s query=%s",
        target.library_id,
        payload.memory_id or "",
        payload.limit_memories,
        safe_summary(payload.query),
    )
    memory_ids = [payload.memory_id] if payload.memory_id else None
    retrieval = []
    if payload.query and not memory_ids:
        results = await target.retrieval.search(
            payload.query,
            payload.limit_memories,
            payload.session_id,
            payload.persona_id,
        )
        memory_ids = [item.doc_id for item in results]
        retrieval = [item.to_dict() for item in results]
    snapshot = await target.graph_snapshot(
        memory_ids=memory_ids,
        session_id=payload.session_id,
        persona_id=payload.persona_id,
        limit_memories=payload.limit_memories,
        limit_nodes=48,
        minimum_degree=2,
    )
    return {"snapshot": snapshot, "retrieval": retrieval}

@router.post("/api/v1/graph/rebuild", dependencies=[Depends(require_auth)])
@router.post(
    "/api/v1/libraries/{library_id}/graph/rebuild",
    dependencies=[Depends(require_auth)],
)
async def rebuild_graph(library_id: str | None = None):
    target = await runtime(library_id)
    logger.warning(
        "提交图记忆重建任务：library_id=%s current_generation=%s",
        target.library_id,
        (target.indexes.status() or {}).get("generation") or "",
    )
    job_id = await jobs().start(
        "graph_rebuild",
        lambda progress: manager.rebuild_graph(target.library_id, progress),
        library_id=target.library_id,
    )
    logger.warning("图记忆重建任务已创建：library_id=%s job_id=%s", target.library_id, job_id)
    return {"job_id": job_id}
