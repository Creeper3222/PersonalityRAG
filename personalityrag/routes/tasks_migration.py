from __future__ import annotations

import hashlib
import json
import secrets
import time
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from fastapi.responses import Response

from ..application_context import current_context, manager
from ..http_shared import (
    ADAPTER_ID_HEADER,
    LivingMemoryV8Type,
    jobs,
    require_existing_database_ref,
    require_auth,
    runtime,
)
from ..logger import logger
from ..jobs import JobStateConflict
from ..io_utils import run_blocking, save_upload_file
from ..io_utils import UploadSizeLimitError, atomic_write_json
from ..library_types.livingmemory_v8.transfer import (
    MAX_TRANSFER_BYTES,
    MemoryTransferError,
    apply_transfer_summaries,
    consume_transfer_preview,
    create_transfer_preview,
    existing_transfer_keys,
    export_transfer_csv,
    export_transfer_json,
    inspect_transfer_records,
    load_transfer_preview,
    parse_transfer_bytes,
    transfer_dedupe_key,
)
from ..migration import validate_conversations_db_file, validate_livingmemory_db_file
from ..schemas import (
    MemoryTransferCommit,
    MigrationRequest,
    RebuildRequest,
)


router = APIRouter()
livingmemory_router = APIRouter(prefix="/api/v1/memory-libraries/livingmemory_v8")


@livingmemory_router.get(
    "/{memory_store_id}/indexes",
    dependencies=[Depends(require_auth)],
)
async def index_status(request: Request, database_type: LivingMemoryV8Type, memory_store_id: str):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="index_rebuild"
    )
    if request.headers.get(ADAPTER_ID_HEADER):
        try:
            record = (
                await manager.control.get_library(memory_store_id)
                if memory_store_id
                else await manager.control.default_library()
            )
            if record is None:
                raise KeyError(memory_store_id or "")
            return (await manager.library_detail(ref))["indexes"]
        except KeyError as exc:
            raise HTTPException(404, "memory library not found") from exc
    target = await runtime(ref)
    stats = await target.storage.statistics()
    return manager._normalize_indexes_for_response(
        stats, target.indexes.status()
    )

@livingmemory_router.post(
    "/{memory_store_id}/indexes/rebuild",
    dependencies=[Depends(require_auth)],
)
async def rebuild_indexes(
    database_type: LivingMemoryV8Type, memory_store_id: str, payload: RebuildRequest
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="index_rebuild"
    )
    target = await runtime(ref)
    provider_id = payload.provider_id or target.provider_revision.provider_id
    logger.warning(
        "提交索引重建任务：memory_store_id=%s provider=%s current_generation=%s",
        memory_store_id,
        provider_id,
        (target.indexes.status() or {}).get("generation") or "",
    )
    job_id = await jobs().start_resumable(
        "index_rebuild",
        {"provider_id": payload.provider_id},
        database_id=memory_store_id,
        database_type=database_type,
    )
    logger.warning(
        "索引重建任务已创建：memory_store_id=%s job_id=%s",
        memory_store_id,
        job_id,
    )
    return {"job_id": job_id}

@router.get("/api/v1/jobs", dependencies=[Depends(require_auth)])
async def list_jobs(scope: str = Query("active", pattern="^(active|finished|all)$")):
    return {"items": await jobs().list(scope=scope)}

@router.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_auth)])
async def job_status(job_id: str):
    result = await jobs().get(job_id)
    if not result:
        raise HTTPException(404, "job not found")
    return result


@router.get(
    "/api/v1/jobs/{job_id}/details",
    dependencies=[Depends(require_auth)],
)
async def job_details(job_id: str):
    result = await jobs().get(job_id, detail=True)
    if not result:
        raise HTTPException(404, "job not found")
    return result


async def _control_job(job_id: str, action: str):
    job_manager = jobs()
    try:
        return await getattr(job_manager, action)(job_id)
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc
    except JobStateConflict as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/api/v1/jobs/{job_id}/pause", dependencies=[Depends(require_auth)])
async def pause_job(job_id: str):
    return await _control_job(job_id, "pause")


@router.post("/api/v1/jobs/{job_id}/resume", dependencies=[Depends(require_auth)])
async def resume_job(job_id: str):
    return await _control_job(job_id, "resume")


@router.post("/api/v1/jobs/{job_id}/stop", dependencies=[Depends(require_auth)])
async def stop_job(job_id: str):
    return await _control_job(job_id, "stop")


@router.post("/api/v1/jobs/{job_id}/cancel", dependencies=[Depends(require_auth)])
async def cancel_job(job_id: str):
    return await _control_job(job_id, "cancel")


@router.post("/api/v1/jobs/finished/clear", dependencies=[Depends(require_auth)])
async def clear_finished_jobs():
    return {"cleared": await jobs().clear_finished()}

@router.get("/api/v1/jobs/{job_id}/events", dependencies=[Depends(require_auth)])
async def job_events(job_id: str):
    job_manager = jobs()

    async def stream():
        async for payload in job_manager.subscribe(job_id):
            yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@livingmemory_router.post(
    "/{memory_store_id}/imports/livingmemory-db",
    dependencies=[Depends(require_auth)],
)
async def import_livingmemory_db_file(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    file: UploadFile = File(...),
    conversations_file: UploadFile | None = File(None),
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="livingmemory_import"
    )
    existing = await jobs().active_job_id("livingmemory_import", memory_store_id)
    if existing:
        logger.warning(
            "复用已存在的 LivingMemory 导入任务：memory_store_id=%s job_id=%s",
            memory_store_id,
            existing,
        )
        return {"job_id": existing}
    try:
        if not await manager.library_is_empty(ref):
            raise HTTPException(409, "只有全新空记忆库可以导入 livingmemory.db")
    except KeyError as exc:
        raise HTTPException(404, "memory library not found") from exc

    filename = Path(file.filename or "").name
    if filename.lower() != "livingmemory.db":
        await file.close()
        if conversations_file:
            await conversations_file.close()
        raise HTTPException(400, "请上传名为 livingmemory.db 的 LivingMemory 核心数据库文件")
    conversations_filename = (
        Path(conversations_file.filename or "").name if conversations_file else ""
    )
    if conversations_file and conversations_filename.lower() != "conversations.db":
        await file.close()
        await conversations_file.close()
        raise HTTPException(400, "可选消息记录文件必须命名为 conversations.db")
    upload_dir = current_context().state_root / "data" / "import_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_token = f"{memory_store_id}-{int(time.time())}-{secrets.token_hex(4)}"
    upload_path = upload_dir / f"{upload_token}-livingmemory.db"
    conversations_upload_path = (
        upload_dir / f"{upload_token}-conversations.db"
        if conversations_file
        else None
    )
    try:
        await save_upload_file(file, upload_path)
        await file.close()
        if conversations_file and conversations_upload_path is not None:
            await save_upload_file(conversations_file, conversations_upload_path)
            await conversations_file.close()
        try:
            source_validation = await run_blocking(
                validate_livingmemory_db_file, upload_path
            )
            source_validation["size"] = upload_path.stat().st_size
            conversations_validation = None
            if conversations_upload_path is not None:
                conversations_validation = await run_blocking(
                    validate_conversations_db_file, conversations_upload_path
                )
                conversations_validation["size"] = (
                    conversations_upload_path.stat().st_size
                )
        except Exception as exc:
            upload_path.unlink(missing_ok=True)
            if conversations_upload_path is not None:
                conversations_upload_path.unlink(missing_ok=True)
            logger.warning(
                "LivingMemory 单文件导入校验失败：memory_store_id=%s filename=%s conversations=%s err=%s",
                memory_store_id,
                filename,
                conversations_filename or "",
                exc,
            )
            raise HTTPException(400, str(exc)) from exc
        logger.warning(
            "提交 LivingMemory 单文件导入任务：memory_store_id=%s upload=%s conversations=%s",
            memory_store_id,
            upload_path,
            conversations_upload_path or "",
        )
        job_id = await jobs().start_resumable(
            "livingmemory_import",
            {
                "source_db": str(upload_path),
                "conversations_db": (
                    str(conversations_upload_path)
                    if conversations_upload_path is not None
                    else None
                ),
                "_source_validation": source_validation,
                "_conversations_validation": conversations_validation,
            },
            database_id=memory_store_id,
            database_type=database_type,
        )
        logger.warning("LivingMemory 单文件导入任务已创建：memory_store_id=%s job_id=%s", memory_store_id, job_id)
        return {"job_id": job_id}
    except HTTPException:
        raise
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        if conversations_upload_path is not None:
            conversations_upload_path.unlink(missing_ok=True)
        logger.exception("LivingMemory 单文件导入请求失败：memory_store_id=%s", memory_store_id)
        raise HTTPException(500, str(exc)) from exc

@livingmemory_router.post(
    "/{memory_store_id}/migration/livingmemory",
    dependencies=[Depends(require_auth)],
)
async def migrate_livingmemory(
    database_type: LivingMemoryV8Type, memory_store_id: str, payload: MigrationRequest
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="livingmemory_import"
    )
    target = await runtime(ref)
    source = Path(payload.source_path)
    logger.warning(
        "提交 LivingMemory 迁移任务：memory_store_id=%s source=%s mode=%s",
        memory_store_id,
        source,
        payload.mode,
    )

    job_id = await jobs().start_resumable(
        "livingmemory_migration",
        {"source_path": str(source), "mode": payload.mode},
        database_id=memory_store_id,
        database_type=database_type,
    )
    logger.warning(
        "LivingMemory 迁移任务已创建：memory_store_id=%s job_id=%s",
        memory_store_id,
        job_id,
    )
    return {"job_id": job_id}


@livingmemory_router.get(
    "/{memory_store_id}/transfers/export",
    dependencies=[Depends(require_auth)],
)
async def export_memory_transfer(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    format: str = Query("json", pattern="^(json|csv)$"),
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="memory_records"
    )
    target = await runtime(ref)
    records = await target.storage.memory_transfer_records()
    filename = f"{memory_store_id}-memories-{int(time.time())}.{format}"
    if format == "csv":
        body = await run_blocking(export_transfer_csv, records)
        media_type = "text/csv; charset=utf-8"
    else:
        body = await run_blocking(export_transfer_json, records)
        media_type = "application/json; charset=utf-8"
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@livingmemory_router.post(
    "/{memory_store_id}/transfers/imports/preview",
    dependencies=[Depends(require_auth)],
)
async def preview_memory_transfer(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    file: UploadFile = File(...),
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="memory_records"
    )
    filename = Path(file.filename or "transfer.json").name
    if Path(filename).suffix.lower() not in {".json", ".csv"}:
        await file.close()
        raise HTTPException(400, "请选择 JSON 或 CSV 记忆迁移文件")
    token = secrets.token_urlsafe(18)
    upload_path = (
        current_context().state_root
        / "data"
        / "import_uploads"
        / "livingmemory_v8"
        / f"{token}{Path(filename).suffix.lower()}"
    )
    try:
        await save_upload_file(file, upload_path, max_bytes=MAX_TRANSFER_BYTES)
        data = await run_blocking(upload_path.read_bytes)
        raw_records = await run_blocking(parse_transfer_bytes, data, filename)
        target = await runtime(ref)
        existing = await target.storage.memory_transfer_records()
        inspection = await run_blocking(
            inspect_transfer_records,
            raw_records,
            existing_transfer_keys(existing),
        )
        digest = hashlib.sha256(data).hexdigest()
        preview_id = await run_blocking(
            create_transfer_preview,
            state_root=current_context().state_root,
            database_id=memory_store_id,
            source_sha256=digest,
            inspection=inspection,
            secret=current_context().config.session_secret,
        )
    except (MemoryTransferError, UploadSizeLimitError) as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        upload_path.unlink(missing_ok=True)
    return {
        "preview_id": preview_id,
        "database_id": memory_store_id,
        "filename": filename,
        "source_sha256": digest,
        **inspection,
    }


@livingmemory_router.post(
    "/{memory_store_id}/transfers/imports/{preview_id}/commit",
    dependencies=[Depends(require_auth)],
)
async def commit_memory_transfer(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    preview_id: str,
    payload: MemoryTransferCommit,
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="memory_records"
    )
    try:
        preview = await run_blocking(
            load_transfer_preview,
            state_root=current_context().state_root,
            database_id=memory_store_id,
            preview_id=preview_id,
            secret=current_context().config.session_secret,
        )
    except MemoryTransferError as exc:
        raise HTTPException(404, str(exc)) from exc

    ready, summary_errors = apply_transfer_summaries(
        list((preview.get("inspection") or {}).get("items") or []),
        [item.model_dump(exclude_none=True) for item in payload.summaries],
    )
    target = await runtime(ref)
    existing = existing_transfer_keys(
        await target.storage.memory_transfer_records()
    )
    selected: list[dict] = []
    skipped: list[dict] = []
    seen = set(existing)
    for item in ready:
        key = str(item.get("dedupe_key") or "") or transfer_dedupe_key(
            item.get("content"), item.get("session_id"), item.get("persona_id")
        )
        item["dedupe_key"] = key
        duplicate = key in seen
        if duplicate and payload.duplicate_mode == "skip":
            skipped.append(
                {
                    "preview_item_id": item.get("preview_item_id"),
                    "row_number": item.get("row_number"),
                    "reason": "duplicate",
                }
            )
            continue
        selected.append(item)
        seen.add(key)
    if not selected and not summary_errors:
        raise HTTPException(409, "没有可导入的记忆记录")

    commit_id = secrets.token_urlsafe(18)
    commit_path = (
        current_context().state_root
        / "data"
        / "import_commits"
        / "livingmemory_v8"
        / f"{commit_id}.json"
    )
    manifest = {
        "version": 1,
        "database_id": memory_store_id,
        "source_sha256": preview.get("source_sha256"),
        "duplicate_mode": payload.duplicate_mode,
        "records": selected,
        "preflight_errors": [
            *list((preview.get("inspection") or {}).get("invalid_items") or []),
            *summary_errors,
        ],
        "skipped": skipped,
        "created_at": time.time(),
    }
    await run_blocking(atomic_write_json, commit_path, manifest)
    manifest_sha256 = hashlib.sha256(commit_path.read_bytes()).hexdigest()
    try:
        job_id = await jobs().start_resumable(
            "memory_transfer_import",
            {
                "manifest_path": str(commit_path),
                "manifest_sha256": manifest_sha256,
                "source_sha256": str(preview.get("source_sha256") or ""),
                "duplicate_mode": payload.duplicate_mode,
            },
            database_id=memory_store_id,
            database_type=database_type,
            dedupe_active=False,
        )
    except Exception:
        commit_path.unlink(missing_ok=True)
        raise
    await run_blocking(consume_transfer_preview, preview)
    return {
        "job_id": job_id,
        "planned_import": len(selected),
        "planned_skip": len(skipped),
        "preflight_errors": len(manifest["preflight_errors"]),
    }
