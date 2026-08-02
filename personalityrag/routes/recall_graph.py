from __future__ import annotations

import time
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)

from ..http_shared import (
    LivingMemoryV8Type,
    jobs,
    livingmemory_v8_database_type,
    require_database_ref,
    require_existing_database_ref,
    require_auth,
    runtime,
)
from ..database_types import database_identity_fields
from ..library_types.livingmemory_v8.request_logging import (
    log_recall_completed,
    log_recall_failed,
    log_recall_request_received,
    log_recall_rerank_fallback,
)
from ..logger import logger
from ..schemas import (
    GraphQuery,
    RecallRequest,
)


router = APIRouter(prefix="/api/v1/memory-libraries/livingmemory_v8")


def _graph_view_payload(
    snapshot: dict[str, Any],
    stats: dict[str, Any],
    *,
    mode: str,
    session_id: str | None,
    persona_id: str | None,
    retrieval: list[dict[str, Any]] | None = None,
    query: str = "",
    memory_id: int | None = None,
) -> dict[str, Any]:
    """Build the LivingMemory 2.5.3 graph-page response without breaking old fields."""
    node_type_breakdown: dict[str, int] = {}
    relation_breakdown: dict[str, int] = {}
    for node in snapshot.get("nodes") or []:
        node_type = str(node.get("type") or "other")
        node_type_breakdown[node_type] = node_type_breakdown.get(node_type, 0) + 1
    for edge in snapshot.get("edges") or []:
        relation_type = str(edge.get("relation_type") or "related")
        relation_breakdown[relation_type] = (
            relation_breakdown.get(relation_type, 0) + 1
        )
    retrieval_items = list(retrieval or [])
    matched_memory_ids = [
        int(value)
        for item in retrieval_items
        if (value := item.get("memory_id") or item.get("doc_id")) is not None
    ]
    return {
        "enabled": True,
        "mode": mode,
        "query": query,
        "memory_id": memory_id,
        "filters": {"session_id": session_id, "persona_id": persona_id},
        "snapshot": snapshot,
        "stats": stats,
        "summary": {
            "node_type_breakdown": node_type_breakdown,
            "relation_breakdown": relation_breakdown,
        },
        # Keep the published list shape; the WebUI normalizes it to the 2.5.3 view model.
        "retrieval": retrieval_items,
        "matched_node_ids": [],
        "matched_memory_ids": matched_memory_ids,
    }


@router.post(
    "/{memory_store_id}/recall",
    dependencies=[Depends(require_auth)],
)
async def recall(
    payload: RecallRequest,
    request: Request,
    memory_store_id: str | None = None,
    database_type: str = Depends(livingmemory_v8_database_type),
):
    if memory_store_id is None:
        raise HTTPException(400, "database id is required")
    ref = require_database_ref(database_type, memory_store_id, capability="recall")
    request_id = log_recall_request_received(
        request,
        memory_store_id=memory_store_id,
        payload=payload,
    )
    started = time.perf_counter()
    try:
        target = await runtime(ref)
    except Exception as exc:
        log_recall_failed(
            request_id=request_id,
            memory_store_id=memory_store_id,
            error=exc,
        )
        raise
    maintenance_provider = getattr(target, "maintenance_status", None)
    if callable(maintenance_provider):
        maintenance = maintenance_provider()
    else:
        indexes = target.indexes.status() if hasattr(target, "indexes") else {}
        maintenance = {
            "status": "ready",
            "stage": "ready",
            "progress": 1.0,
            "active_generation": (indexes or {}).get("generation"),
            "candidate_generation": None,
            "index_available": True,
            "error_category": None,
            "error": None,
            "request_id": None,
        }
    if not maintenance.get("index_available"):
        stats = await target.storage.summary_statistics()
        if int(stats.get("active_memories") or 0) > 0:
            raise HTTPException(
                503,
                {
                    "code": "index_not_ready",
                    "message": "memory index is not ready; writes and sessions remain available",
                    "maintenance": maintenance,
                    "request_id": request_id,
                },
            )
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
    candidate_k = embedding_k
    try:
        candidates = await target.retrieval.search(
            payload.query,
            candidate_k,
            session_filter,
            persona_filter,
        )
    except Exception as exc:
        log_recall_failed(
            request_id=request_id,
            memory_store_id=memory_store_id,
            error=exc,
        )
        raise
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
                log_recall_rerank_fallback(
                    request_id=request_id,
                    memory_store_id=memory_store_id,
                    provider_id=str(rerank_meta.get("provider_id") or ""),
                    error=exc,
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
    log_recall_completed(
        request_id=request_id,
        memory_store_id=memory_store_id,
        elapsed_ms=elapsed_ms,
        embedding_candidates=len(candidates),
        returned_results=len(results),
        rerank_available=bool(target.reranker),
        rerank_applied=bool(rerank_meta.get("applied")),
        rerank_failed=bool(rerank_meta.get("failed")),
        session_filter_applied=bool(session_filter),
        persona_filter_applied=bool(persona_filter),
        generation=generation,
    )
    response = {
        "query": payload.query,
        **database_identity_fields(ref, include_deprecated=True),
        "results": [item.to_dict() for item in results],
        "total": len(results),
        "embedding_k": embedding_k,
        "rerank_k": rerank_k,
        "elapsed_time_ms": elapsed_ms,
        "rerank": rerank_meta,
        "maintenance": maintenance,
    }
    if payload.include_baseline:
        response["baseline_results"] = [
            item.to_dict() for item in baseline_results
        ]
    if use_rerank:
        response["rerank_candidates"] = [item.to_dict() for item in candidates]
    return response

@router.get(
    "/{memory_store_id}/graph/overview",
    dependencies=[Depends(require_auth)],
)
async def graph_overview(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    session_id: str | None = None,
    persona_id: str | None = None,
    full_graph: bool = False,
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="graph"
    )
    target = await runtime(ref)
    snapshot = (
        await target.full_graph_snapshot(
            session_id=session_id,
            persona_id=persona_id,
        )
        if full_graph
        else await target.graph_snapshot(
            session_id=session_id,
            persona_id=persona_id,
            limit_memories=12,
            limit_entries=36,
            limit_nodes=48,
            limit_edges=72,
            minimum_degree=0,
        )
    )
    return _graph_view_payload(
        snapshot,
        await target.storage.statistics(),
        mode="full_graph" if full_graph else "overview",
        session_id=session_id,
        persona_id=persona_id,
    )

@router.post(
    "/{memory_store_id}/graph/query",
    dependencies=[Depends(require_auth)],
)
async def graph_query(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    payload: GraphQuery,
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="graph"
    )
    target = await runtime(ref)
    logger.debug(
        "图谱查询：memory_store_id=%s memory_id=%s limit=%s query_chars=%s",
        memory_store_id,
        payload.memory_id or "",
        payload.limit_memories,
        len(payload.query or ""),
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
        limit_entries=payload.limit_entries,
        limit_nodes=payload.limit_nodes,
        limit_edges=payload.limit_edges,
        minimum_degree=0,
    )
    mode = "memory_focus" if payload.memory_id else "query" if payload.query else "overview"
    return _graph_view_payload(
        snapshot,
        await target.storage.statistics(),
        mode=mode,
        session_id=payload.session_id,
        persona_id=payload.persona_id,
        retrieval=retrieval,
        query=payload.query,
        memory_id=payload.memory_id,
    )

@router.post(
    "/{memory_store_id}/graph/rebuild",
    dependencies=[Depends(require_auth)],
)
async def rebuild_graph(database_type: LivingMemoryV8Type, memory_store_id: str):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="graph"
    )
    target = await runtime(ref)
    logger.warning(
        "提交图记忆重建任务：memory_store_id=%s current_generation=%s",
        memory_store_id,
        (target.indexes.status() or {}).get("generation") or "",
    )
    job_id = await jobs().start_resumable(
        "graph_rebuild",
        {},
        database_id=memory_store_id,
    )
    logger.warning(
        "图记忆重建任务已创建：memory_store_id=%s job_id=%s",
        memory_store_id,
        job_id,
    )
    return {"job_id": job_id}
