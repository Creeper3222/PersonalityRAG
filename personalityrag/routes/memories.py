from __future__ import annotations

import time
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)

from ..application_context import manager
from ..http_shared import (
    _resolve_memory_library_id,
    _run_memory_job,
    require_auth,
    runtime,
)
from ..logger import logger, safe_summary
from ..schemas import (
    BatchDelete,
    BatchUpdate,
    MemoryCreate,
    MemoryPersonaUpdate,
    MemoryUpdate,
)


router = APIRouter()


def _normalize_memory_update_payload(payload: MemoryUpdate) -> dict[str, Any]:
    if "persona_id" in payload.model_fields_set:
        raise HTTPException(
            400,
            "人格字段必须通过独立的人格编辑接口保存",
        )
    updates = payload.model_dump(exclude_none=True)
    scale = str(updates.pop("value_scale", "auto") or "auto").lower()
    if "importance" not in updates:
        return updates
    try:
        value = float(updates["importance"])
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "invalid importance") from exc
    if scale in {"display", "0-10", "ten"}:
        if not 0 <= value <= 10:
            raise HTTPException(400, "importance display value must be between 0 and 10")
        updates["importance"] = value / 10
    elif scale in {"stored", "normalized", "0-1"}:
        if not 0 <= value <= 1:
            raise HTTPException(400, "importance stored value must be between 0 and 1")
        updates["importance"] = value
    elif scale == "auto":
        if 0 <= value <= 1:
            updates["importance"] = value
        elif 0 <= value <= 10:
            updates["importance"] = value / 10
        else:
            raise HTTPException(400, "importance value must be between 0 and 10")
    else:
        raise HTTPException(400, "invalid importance value_scale")
    return updates

@router.get("/api/v1/memories", dependencies=[Depends(require_auth)])
@router.get(
    "/api/v1/libraries/{library_id}/memories",
    dependencies=[Depends(require_auth)],
)
async def memories(
    library_id: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=500),
    keyword: str = "",
    session_id: str | None = None,
    persona_id: str | None = None,
    status: str | None = None,
    memory_type: str | None = None,
    sort: str = "created_desc",
):
    target = await runtime(library_id)
    logger.debug(
        "查询记忆列表：library_id=%s page=%s page_size=%s keyword=%s persona=%s session=%s status=%s sort=%s",
        target.library_id,
        page,
        page_size,
        safe_summary(keyword),
        persona_id or "",
        session_id or "",
        status or "",
        sort,
    )
    return await target.storage.list_documents(
        page=page,
        page_size=page_size,
        keyword=keyword,
        session_id=session_id,
        persona_id=persona_id,
        status=status,
        memory_type=memory_type,
        sort=sort,
    )

@router.post("/api/v1/memories", dependencies=[Depends(require_auth)])
@router.post(
    "/api/v1/libraries/{library_id}/memories",
    dependencies=[Depends(require_auth)],
)
async def create_memory(payload: MemoryCreate, library_id: str | None = None):
    target_library_id = await _resolve_memory_library_id(library_id)
    payload_data = payload.model_dump()
    logger.info(
        "写入记忆请求：library_id=%s persona=%s session=%s importance=%s content_summary=%s",
        target_library_id,
        payload.persona_id or "",
        payload.session_id or "",
        payload.importance,
        safe_summary(payload.content),
    )
    started = time.perf_counter()
    async def operation(progress):
        await progress(0.05, "准备写入记忆")
        target = await manager.get_runtime(target_library_id)
        result = await target.create_memory(payload_data, rebuild=False)
        await progress(1.0, "记忆写入并完成增量索引")
        return result

    result = await _run_memory_job("memory_create", target_library_id, operation)
    logger.info(
        "写入记忆完成并完成增量索引：library_id=%s memory_id=%s generation=%s elapsed_ms=%.2f",
        target_library_id,
        result.get("id"),
        (result.get("index_update") or {}).get("generation"),
        (time.perf_counter() - started) * 1000,
    )
    return result

@router.post("/api/v1/memories/batch-delete", dependencies=[Depends(require_auth)])
@router.post(
    "/api/v1/libraries/{library_id}/memories/batch-delete",
    dependencies=[Depends(require_auth)],
)
async def batch_delete(payload: BatchDelete, library_id: str | None = None):
    target_library_id = await _resolve_memory_library_id(library_id)
    memory_ids = list(payload.memory_ids)
    logger.warning(
        "批量删除记忆请求：library_id=%s count=%s ids=%s",
        target_library_id,
        len(memory_ids),
        memory_ids[:20],
    )
    async def operation(progress):
        await progress(0.05, "准备删除记忆")
        target = await manager.get_runtime(target_library_id)
        result = await target.delete_memories(
            memory_ids, rebuild=False, return_details=True
        )
        await progress(1.0, "记忆删除并完成增量索引移除")
        return result

    delete_result = await _run_memory_job(
        "memory_delete", target_library_id, operation
    )
    deleted = int(delete_result.get("deleted", 0))
    logger.warning(
        "批量删除记忆完成并完成增量索引移除：library_id=%s deleted=%s generation=%s",
        target_library_id,
        deleted,
        ((delete_result.get("index_update") or {}).get("generation") or ""),
    )
    return delete_result

@router.post("/api/v1/memories/batch-update", dependencies=[Depends(require_auth)])
@router.post(
    "/api/v1/libraries/{library_id}/memories/batch-update",
    dependencies=[Depends(require_auth)],
)
async def batch_update(payload: BatchUpdate, library_id: str | None = None):
    target_library_id = await _resolve_memory_library_id(library_id)
    memory_ids = list(payload.memory_ids)
    updates = _normalize_memory_update_payload(payload.updates)
    logger.info(
        "批量更新记忆请求：library_id=%s count=%s fields=%s",
        target_library_id,
        len(memory_ids),
        sorted(payload.updates.model_dump(exclude_none=True).keys()),
    )
    async def operation(progress):
        await progress(0.05, "准备更新记忆")
        target = await manager.get_runtime(target_library_id)
        updated = []
        results = []
        total = max(1, len(memory_ids))
        for index, item_id in enumerate(memory_ids, start=1):
            result = await target.update_memory(item_id, updates, rebuild=False)
            if result:
                updated.append(int(result.get("id") or item_id))
                results.append(result)
            await progress(index / total, "记忆更新并完成增量索引")
        return {"updated": updated, "count": len(updated), "items": results}

    batch_result = await _run_memory_job(
        "memory_update", target_library_id, operation
    )
    logger.info(
        "批量更新记忆完成并完成增量索引：library_id=%s count=%s",
        target_library_id,
        batch_result.get("count", 0),
    )
    return batch_result

@router.get(
    "/api/v1/memories/{memory_id}", dependencies=[Depends(require_auth)]
)
@router.get(
    "/api/v1/libraries/{library_id}/memories/{memory_id}",
    dependencies=[Depends(require_auth)],
)
async def memory_detail(memory_id: int, library_id: str | None = None):
    target = await runtime(library_id)
    memory = await target.storage.get_document(memory_id)
    if not memory:
        raise HTTPException(404, "memory not found")
    memory["graph_context"] = await target.graph_snapshot(
        memory_ids=[memory_id]
    )
    return memory

@router.patch(
    "/api/v1/memories/{memory_id}", dependencies=[Depends(require_auth)]
)
@router.patch(
    "/api/v1/libraries/{library_id}/memories/{memory_id}",
    dependencies=[Depends(require_auth)],
)
async def update_memory(
    memory_id: int, payload: MemoryUpdate, library_id: str | None = None
):
    target_library_id = await _resolve_memory_library_id(library_id)
    updates = _normalize_memory_update_payload(payload)
    logger.info(
        "更新记忆请求：library_id=%s memory_id=%s fields=%s",
        target_library_id,
        memory_id,
        sorted(payload.model_dump(exclude_none=True).keys()),
    )
    async def operation(progress):
        await progress(0.05, "准备更新记忆")
        target = await manager.get_runtime(target_library_id)
        result = await target.update_memory(memory_id, updates, rebuild=False)
        await progress(1.0, "记忆更新并完成增量索引")
        return result or {"not_found": True}

    result = await _run_memory_job("memory_update", target_library_id, operation)
    if not result or result.get("not_found"):
        logger.warning("更新记忆失败：library_id=%s memory_id=%s err=not_found", target_library_id, memory_id)
        raise HTTPException(404, "memory not found")
    logger.info(
        "更新记忆完成并完成增量索引：library_id=%s old_memory_id=%s new_memory_id=%s generation=%s",
        target_library_id,
        memory_id,
        result.get("new_memory_id") or result.get("id"),
        (result.get("index_update") or {}).get("generation"),
    )
    return result


@router.patch(
    "/api/v1/memories/{memory_id}/persona",
    dependencies=[Depends(require_auth)],
)
@router.patch(
    "/api/v1/libraries/{library_id}/memories/{memory_id}/persona",
    dependencies=[Depends(require_auth)],
)
async def update_memory_persona(
    memory_id: int,
    payload: MemoryPersonaUpdate,
    library_id: str | None = None,
):
    target_library_id = await _resolve_memory_library_id(library_id)
    persona_id = str(payload.persona_id or "").strip() or None
    logger.info(
        "更新记忆人格字段请求：library_id=%s memory_id=%s persona=%s",
        target_library_id,
        memory_id,
        persona_id or "",
    )

    async def operation(progress):
        await progress(0.1, "准备更新记忆人格字段")
        target = await manager.get_runtime(target_library_id)
        result = await target.update_memory_persona(memory_id, persona_id)
        await progress(1.0, "记忆人格字段已更新，索引保持不变")
        return result or {"not_found": True}

    result = await _run_memory_job(
        "memory_update", target_library_id, operation
    )
    if not result or result.get("not_found"):
        raise HTTPException(404, "memory not found")
    logger.info(
        "更新记忆人格字段完成：library_id=%s memory_id=%s persona=%s generation=%s index_changed=false",
        target_library_id,
        memory_id,
        persona_id or "",
        (result.get("index_update") or {}).get("generation") or "",
    )
    return result

@router.delete(
    "/api/v1/memories/{memory_id}", dependencies=[Depends(require_auth)]
)
@router.delete(
    "/api/v1/libraries/{library_id}/memories/{memory_id}",
    dependencies=[Depends(require_auth)],
)
async def delete_memory(memory_id: int, library_id: str | None = None):
    target_library_id = await _resolve_memory_library_id(library_id)
    logger.warning("删除记忆请求：library_id=%s memory_id=%s", target_library_id, memory_id)
    async def operation(progress):
        await progress(0.05, "准备删除记忆")
        target = await manager.get_runtime(target_library_id)
        result = await target.delete_memories(
            [memory_id], rebuild=False, return_details=True
        )
        await progress(1.0, "记忆删除并完成增量索引移除")
        return result

    delete_result = await _run_memory_job(
        "memory_delete", target_library_id, operation
    )
    logger.warning(
        "删除记忆完成并完成增量索引移除：library_id=%s memory_id=%s deleted=%s generation=%s",
        target_library_id,
        memory_id,
        delete_result.get("deleted", 0),
        ((delete_result.get("index_update") or {}).get("generation") or ""),
    )
    return delete_result
