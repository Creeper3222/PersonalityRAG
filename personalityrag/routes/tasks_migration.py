from __future__ import annotations

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

from ..application_context import current_context, manager
from ..http_shared import (
    ADAPTER_ID_HEADER,
    jobs,
    require_auth,
    runtime,
)
from ..logger import logger
from ..io_utils import run_blocking, save_upload_file
from ..migration import validate_conversations_db_file, validate_livingmemory_db_file
from ..schemas import (
    MigrationRequest,
    RebuildRequest,
)


router = APIRouter()
@router.get("/api/v1/indexes", dependencies=[Depends(require_auth)])
@router.get(
    "/api/v1/libraries/{library_id}/indexes",
    dependencies=[Depends(require_auth)],
)
async def index_status(request: Request, library_id: str | None = None):
    if request.headers.get(ADAPTER_ID_HEADER):
        try:
            record = (
                await manager.control.get_library(library_id)
                if library_id
                else await manager.control.default_library()
            )
            if record is None:
                raise KeyError(library_id or "")
            return (await manager.library_detail(record.id))["indexes"]
        except KeyError as exc:
            raise HTTPException(404, "memory library not found") from exc
    target = await runtime(library_id)
    stats = await target.storage.statistics()
    return manager._normalize_indexes_for_response(
        stats, target.indexes.status()
    )

@router.post("/api/v1/indexes/rebuild", dependencies=[Depends(require_auth)])
@router.post(
    "/api/v1/libraries/{library_id}/indexes/rebuild",
    dependencies=[Depends(require_auth)],
)
async def rebuild_indexes(
    payload: RebuildRequest, library_id: str | None = None
):
    target = await runtime(library_id)
    provider_id = payload.provider_id or target.provider_revision.provider_id
    logger.warning(
        "提交索引重建任务：library_id=%s provider=%s current_generation=%s",
        target.library_id,
        provider_id,
        (target.indexes.status() or {}).get("generation") or "",
    )
    job_id = await jobs().start(
        "index_rebuild",
        lambda progress: manager.rebuild_library(
            target.library_id, payload.provider_id, progress
        ),
        library_id=target.library_id,
    )
    logger.warning("索引重建任务已创建：library_id=%s job_id=%s", target.library_id, job_id)
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

@router.post(
    "/api/v1/libraries/{library_id}/imports/livingmemory-db",
    dependencies=[Depends(require_auth)],
)
async def import_livingmemory_db_file(
    library_id: str,
    file: UploadFile = File(...),
    conversations_file: UploadFile | None = File(None),
):
    existing = await jobs().active_job_id("livingmemory_import", library_id)
    if existing:
        logger.warning(
            "复用已存在的 LivingMemory 导入任务：library_id=%s job_id=%s",
            library_id,
            existing,
        )
        return {"job_id": existing}
    try:
        if not await manager.library_is_empty(library_id):
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
    upload_token = f"{library_id}-{int(time.time())}-{secrets.token_hex(4)}"
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
            await run_blocking(validate_livingmemory_db_file, upload_path)
            if conversations_upload_path is not None:
                await run_blocking(
                    validate_conversations_db_file, conversations_upload_path
                )
        except Exception as exc:
            upload_path.unlink(missing_ok=True)
            if conversations_upload_path is not None:
                conversations_upload_path.unlink(missing_ok=True)
            logger.warning(
                "LivingMemory 单文件导入校验失败：library_id=%s filename=%s conversations=%s err=%s",
                library_id,
                filename,
                conversations_filename or "",
                exc,
            )
            raise HTTPException(400, str(exc)) from exc
        logger.warning(
            "提交 LivingMemory 单文件导入任务：library_id=%s upload=%s conversations=%s",
            library_id,
            upload_path,
            conversations_upload_path or "",
        )
        job_id = await jobs().start(
            "livingmemory_import",
            lambda progress: manager.import_livingmemory_db(
                library_id,
                upload_path,
                progress,
                conversations_db=conversations_upload_path,
            ),
            library_id=library_id,
            lease_runtime=False,
        )
        logger.warning("LivingMemory 单文件导入任务已创建：library_id=%s job_id=%s", library_id, job_id)
        return {"job_id": job_id}
    except HTTPException:
        raise
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        if conversations_upload_path is not None:
            conversations_upload_path.unlink(missing_ok=True)
        logger.exception("LivingMemory 单文件导入请求失败：library_id=%s", library_id)
        raise HTTPException(500, str(exc)) from exc

@router.post(
    "/api/v1/migration/livingmemory",
    dependencies=[Depends(require_auth)],
)
@router.post(
    "/api/v1/libraries/{library_id}/migration/livingmemory",
    dependencies=[Depends(require_auth)],
)
async def migrate_livingmemory(
    payload: MigrationRequest, library_id: str | None = None
):
    target = await runtime(library_id)
    source = Path(payload.source_path)
    logger.warning(
        "提交 LivingMemory 迁移任务：library_id=%s source=%s mode=%s",
        target.library_id,
        source,
        payload.mode,
    )

    async def operation(progress):
        logger.warning(
            "LivingMemory 迁移开始：library_id=%s source=%s mode=%s",
            target.library_id,
            source,
            payload.mode,
        )
        result = await target.migrator.migrate(
            source, mode=payload.mode, progress=progress
        )
        logger.warning(
            "LivingMemory 数据迁移完成，开始重建索引：library_id=%s run_id=%s",
            target.library_id,
            result.get("run_id"),
        )
        await target.storage.initialize()
        target.text = target.text.__class__(target.data_dir / "stopwords")
        target.retrieval.text = target.text
        rebuild = await manager.rebuild_library(
            target.library_id, None, progress
        )
        logger.warning(
            "LivingMemory 迁移与索引重建完成：library_id=%s generation=%s",
            target.library_id,
            (rebuild.get("manifest") or {}).get("generation"),
        )
        return {"migration": result, "rebuild": rebuild}

    job_id = await jobs().start(
        "livingmemory_migration",
        operation,
        library_id=target.library_id,
    )
    logger.warning("LivingMemory 迁移任务已创建：library_id=%s job_id=%s", target.library_id, job_id)
    return {"job_id": job_id}
