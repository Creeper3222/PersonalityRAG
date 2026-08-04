from __future__ import annotations

import hashlib
import hmac
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
    LivingMemoryV8Type,
    _resolve_memory_store_id,
    _run_memory_store_job,
    require_existing_database_ref,
    require_auth,
    runtime,
)
from ..logger import logger
from ..schemas import (
    BatchDelete,
    BatchUpdate,
    MemoryCreate,
    MemoryPersonaUpdate,
    MemoryResummaryCommit,
    MemorySourceUpdate,
    MemoryUpdate,
)


router = APIRouter(prefix="/api/v1/memory-libraries/livingmemory_v8")


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

@router.get(
    "/{memory_store_id}/memories",
    dependencies=[Depends(require_auth)],
)
async def memories(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=500),
    keyword: str = "",
    session_id: str | None = None,
    persona_id: str | None = None,
    status: str | None = None,
    sort: str = "created_desc",
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="memory_records"
    )
    target = await runtime(ref)
    logger.debug(
        "查询记忆列表：memory_store_id=%s page=%s page_size=%s keyword_chars=%s persona=%s session=%s status=%s sort=%s",
        memory_store_id,
        page,
        page_size,
        len(keyword),
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
        sort=sort,
    )

@router.post(
    "/{memory_store_id}/memories",
    dependencies=[Depends(require_auth)],
)
async def create_memory(database_type: LivingMemoryV8Type, memory_store_id: str, payload: MemoryCreate):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="memory_records"
    )
    target_memory_store_id = await _resolve_memory_store_id(ref)
    payload_data = payload.model_dump()
    logger.info(
        "写入记忆请求：memory_store_id=%s persona=%s session=%s importance=%s content_chars=%s",
        target_memory_store_id,
        payload.persona_id or "",
        payload.session_id or "",
        payload.importance,
        len(payload.content),
    )
    started = time.perf_counter()
    async def operation(progress):
        await progress(0.05, "准备写入记忆")
        target = await manager.get_runtime(target_memory_store_id)
        result = await target.create_memory(payload_data, rebuild=False)
        await progress(1.0, "记忆写入并完成增量索引")
        return result

    result = await _run_memory_store_job("memory_create", target_memory_store_id, operation)
    logger.info(
        "写入记忆完成并完成增量索引：memory_store_id=%s memory_id=%s generation=%s elapsed_ms=%.2f",
        target_memory_store_id,
        result.get("id"),
        (result.get("index_update") or {}).get("generation"),
        (time.perf_counter() - started) * 1000,
    )
    return result

@router.post(
    "/{memory_store_id}/memories/batch-delete",
    dependencies=[Depends(require_auth)],
)
async def batch_delete(database_type: LivingMemoryV8Type, memory_store_id: str, payload: BatchDelete):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="memory_records"
    )
    target_memory_store_id = await _resolve_memory_store_id(ref)
    memory_ids = list(payload.memory_ids)
    logger.warning(
        "批量删除记忆请求：memory_store_id=%s count=%s ids=%s",
        target_memory_store_id,
        len(memory_ids),
        memory_ids[:20],
    )
    async def operation(progress):
        await progress(0.05, "准备删除记忆")
        target = await manager.get_runtime(target_memory_store_id)
        result = await target.delete_memories(
            memory_ids, rebuild=False, return_details=True
        )
        await progress(1.0, "记忆删除并完成增量索引移除")
        return result

    delete_result = await _run_memory_store_job(
        "memory_delete", target_memory_store_id, operation
    )
    deleted = int(delete_result.get("deleted", 0))
    logger.warning(
        "批量删除记忆完成并完成增量索引移除：memory_store_id=%s deleted=%s generation=%s",
        target_memory_store_id,
        deleted,
        ((delete_result.get("index_update") or {}).get("generation") or ""),
    )
    return delete_result

@router.post(
    "/{memory_store_id}/memories/batch-update",
    dependencies=[Depends(require_auth)],
)
async def batch_update(database_type: LivingMemoryV8Type, memory_store_id: str, payload: BatchUpdate):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="memory_records"
    )
    target_memory_store_id = await _resolve_memory_store_id(ref)
    memory_ids = list(payload.memory_ids)
    updates = _normalize_memory_update_payload(payload.updates)
    logger.info(
        "批量更新记忆请求：memory_store_id=%s count=%s fields=%s",
        target_memory_store_id,
        len(memory_ids),
        sorted(payload.updates.model_dump(exclude_none=True).keys()),
    )
    async def operation(progress):
        await progress(0.05, "准备更新记忆")
        target = await manager.get_runtime(target_memory_store_id)
        if "content" not in updates:
            results = await target.update_memories_metadata(memory_ids, updates)
            await progress(1.0, "记忆元数据已在单个事务中更新")
            return {
                "updated": [int(item["id"]) for item in results],
                "count": len(results),
                "items": results,
            }
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

    batch_result = await _run_memory_store_job(
        "memory_update", target_memory_store_id, operation
    )
    logger.info(
        "批量更新记忆完成并完成增量索引：memory_store_id=%s count=%s",
        target_memory_store_id,
        batch_result.get("count", 0),
    )
    return batch_result

@router.get(
    "/{memory_store_id}/memories/{memory_id}",
    dependencies=[Depends(require_auth)],
)
async def memory_detail(database_type: LivingMemoryV8Type, memory_store_id: str, memory_id: int):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="memory_records"
    )
    target = await runtime(ref)
    memory = await target.storage.get_document(memory_id)
    if not memory:
        raise HTTPException(404, "memory not found")
    memory["graph_context"] = await target.graph_snapshot(
        memory_ids=[memory_id]
    )
    return memory


@router.get(
    "/{memory_store_id}/memories/{memory_id}/source",
    dependencies=[Depends(require_auth)],
)
async def memory_source(
    database_type: LivingMemoryV8Type, memory_store_id: str, memory_id: int
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="memory_records"
    )
    target = await runtime(ref)
    memory = await target.storage.get_document(memory_id)
    if not memory:
        raise HTTPException(404, "memory not found")
    source = await target.storage.get_memory_source(memory_id)
    return {
        "memory_id": memory_id,
        "has_source": bool(source),
        "source_message_count": len(source),
        "source_messages": source,
    }


@router.put(
    "/{memory_store_id}/memories/{memory_id}/source",
    dependencies=[Depends(require_auth)],
)
async def replace_memory_source(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    memory_id: int,
    payload: MemorySourceUpdate,
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="memory_records"
    )
    target = await runtime(ref)
    try:
        source = await target.storage.save_memory_source(
            memory_id, payload.source_messages
        )
    except KeyError as exc:
        raise HTTPException(404, "memory not found") from exc
    return {
        "memory_id": memory_id,
        "has_source": True,
        "source_message_count": len(source),
    }


@router.post(
    "/{memory_store_id}/memories/{memory_id}/archive",
    dependencies=[Depends(require_auth)],
)
async def archive_memory(
    database_type: LivingMemoryV8Type, memory_store_id: str, memory_id: int
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="memory_records"
    )
    target_memory_store_id = await _resolve_memory_store_id(ref)

    async def operation(progress):
        await progress(0.1, "正在归档记忆并移出检索索引")
        target = await manager.get_runtime(target_memory_store_id)
        result = await target.archive_memories([memory_id], return_details=True)
        await progress(1.0, "记忆已归档")
        return result

    result = await _run_memory_store_job(
        "memory_update", target_memory_store_id, operation
    )
    if int(result.get("archived") or 0) == 0:
        target = await runtime(ref)
        if not await target.storage.get_document(memory_id):
            raise HTTPException(404, "memory not found")
    return result


@router.post(
    "/{memory_store_id}/memories/{memory_id}/restore",
    dependencies=[Depends(require_auth)],
)
async def restore_memory(
    database_type: LivingMemoryV8Type, memory_store_id: str, memory_id: int
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="memory_records"
    )
    target_memory_store_id = await _resolve_memory_store_id(ref)

    async def operation(progress):
        await progress(0.1, "正在恢复记忆派生索引")
        target = await manager.get_runtime(target_memory_store_id)
        result = await target.restore_memory(memory_id)
        await progress(1.0, "记忆已恢复")
        return result or {"not_found": True}

    result = await _run_memory_store_job(
        "memory_update", target_memory_store_id, operation
    )
    if result.get("not_found"):
        raise HTTPException(404, "memory not found")
    return result


@router.post(
    "/{memory_store_id}/memories/{memory_id}/resummary",
    dependencies=[Depends(require_auth)],
)
async def commit_memory_resummary(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    memory_id: int,
    payload: MemoryResummaryCommit,
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="memory_records"
    )
    target_memory_store_id = await _resolve_memory_store_id(ref)
    target = await runtime(ref)
    current = await target.storage.get_document(memory_id)
    if not current:
        raise HTTPException(404, "memory not found")
    current_hash = hashlib.sha256(
        str(current.get("text") or "").encode("utf-8")
    ).hexdigest()
    if payload.expected_content_sha256 and not hmac.compare_digest(
        payload.expected_content_sha256.casefold(), current_hash
    ):
        raise HTTPException(409, "memory content changed after preview")
    retrieval_content = (
        payload.content.strip()
        if payload.content
        else payload.canonical_summary.strip()
    )
    updates = {
        "content": retrieval_content,
        "metadata": {
            **payload.metadata,
            "canonical_summary": payload.canonical_summary.strip(),
            "persona_summary": (
                payload.persona_summary.strip()
                if payload.persona_summary
                else retrieval_content
            ),
            "resummarized_from": memory_id,
        },
    }

    async def operation(progress):
        await progress(0.1, "正在原子替换记忆摘要")
        current_runtime = await manager.get_runtime(target_memory_store_id)
        result = await current_runtime.update_memory(memory_id, updates)
        await progress(1.0, "记忆摘要与派生索引已替换")
        return result or {"not_found": True}

    result = await _run_memory_store_job(
        "memory_update", target_memory_store_id, operation
    )
    if result.get("not_found"):
        raise HTTPException(404, "memory not found")
    return result

@router.patch(
    "/{memory_store_id}/memories/{memory_id}",
    dependencies=[Depends(require_auth)],
)
async def update_memory(
    database_type: LivingMemoryV8Type, memory_store_id: str, memory_id: int, payload: MemoryUpdate
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="memory_records"
    )
    target_memory_store_id = await _resolve_memory_store_id(ref)
    updates = _normalize_memory_update_payload(payload)
    logger.info(
        "更新记忆请求：memory_store_id=%s memory_id=%s fields=%s",
        target_memory_store_id,
        memory_id,
        sorted(payload.model_dump(exclude_none=True).keys()),
    )
    async def operation(progress):
        await progress(0.05, "准备更新记忆")
        target = await manager.get_runtime(target_memory_store_id)
        result = await target.update_memory(memory_id, updates, rebuild=False)
        await progress(1.0, "记忆更新并完成增量索引")
        return result or {"not_found": True}

    result = await _run_memory_store_job("memory_update", target_memory_store_id, operation)
    if not result or result.get("not_found"):
        logger.warning("更新记忆失败：memory_store_id=%s memory_id=%s err=not_found", target_memory_store_id, memory_id)
        raise HTTPException(404, "memory not found")
    logger.info(
        "更新记忆完成并完成增量索引：memory_store_id=%s old_memory_id=%s new_memory_id=%s generation=%s",
        target_memory_store_id,
        memory_id,
        result.get("new_memory_id") or result.get("id"),
        (result.get("index_update") or {}).get("generation"),
    )
    return result


@router.patch(
    "/{memory_store_id}/memories/{memory_id}/persona",
    dependencies=[Depends(require_auth)],
)
async def update_memory_persona(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    memory_id: int,
    payload: MemoryPersonaUpdate,
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="memory_records"
    )
    target_memory_store_id = await _resolve_memory_store_id(ref)
    persona_id = str(payload.persona_id or "").strip() or None
    logger.info(
        "更新记忆人格字段请求：memory_store_id=%s memory_id=%s persona=%s",
        target_memory_store_id,
        memory_id,
        persona_id or "",
    )

    async def operation(progress):
        await progress(0.1, "准备更新记忆人格字段")
        target = await manager.get_runtime(target_memory_store_id)
        result = await target.update_memory_persona(memory_id, persona_id)
        await progress(1.0, "记忆人格字段已更新，索引保持不变")
        return result or {"not_found": True}

    result = await _run_memory_store_job(
        "memory_update", target_memory_store_id, operation
    )
    if not result or result.get("not_found"):
        raise HTTPException(404, "memory not found")
    logger.info(
        "更新记忆人格字段完成：memory_store_id=%s memory_id=%s persona=%s generation=%s index_changed=false",
        target_memory_store_id,
        memory_id,
        persona_id or "",
        (result.get("index_update") or {}).get("generation") or "",
    )
    return result

@router.delete(
    "/{memory_store_id}/memories/{memory_id}",
    dependencies=[Depends(require_auth)],
)
async def delete_memory(database_type: LivingMemoryV8Type, memory_store_id: str, memory_id: int):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="memory_records"
    )
    target_memory_store_id = await _resolve_memory_store_id(ref)
    logger.warning("删除记忆请求：memory_store_id=%s memory_id=%s", target_memory_store_id, memory_id)
    async def operation(progress):
        await progress(0.05, "准备删除记忆")
        target = await manager.get_runtime(target_memory_store_id)
        result = await target.delete_memories(
            [memory_id], rebuild=False, return_details=True
        )
        await progress(1.0, "记忆删除并完成增量索引移除")
        return result

    delete_result = await _run_memory_store_job(
        "memory_delete", target_memory_store_id, operation
    )
    logger.warning(
        "删除记忆完成并完成增量索引移除：memory_store_id=%s memory_id=%s deleted=%s generation=%s",
        target_memory_store_id,
        memory_id,
        delete_result.get("deleted", 0),
        ((delete_result.get("index_update") or {}).get("generation") or ""),
    )
    return delete_result
