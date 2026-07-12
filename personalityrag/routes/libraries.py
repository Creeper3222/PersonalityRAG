from __future__ import annotations

import asyncio

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
)

from ..application_context import auth, manager
from ..auth import verify_password
from ..control import (
    ADAPTER_CONNECTION_TTL_SECONDS,
    AdapterConnectionChangedError,
    AdapterForcedOfflineError,
)
from ..http_shared import (
    ADAPTER_ID_HEADER,
    ADAPTER_INSTANCE_HEADER,
    ADAPTER_TYPE_HEADER,
    _adapter_header,
    _adapter_forced_offline_detail,
    _adapter_job_summary,
    _derive_library_psk,
    jobs,
    require_auth,
    require_admin_auth,
    runtime,
)
from ..identifiers import validate_identifier
from ..logger import logger
from ..schemas import (
    AdapterDisconnectRequest,
    AdapterHeartbeatRequest,
    ConversationMessageCreate,
    ConversationMetadataUpdate,
    ConversationTrimRequest,
    LibraryCreate,
    LibraryPskRequest,
    LibraryUpdate,
)


router = APIRouter()
@router.get("/api/v1/libraries", dependencies=[Depends(require_auth)])
async def libraries(
    stats_mode: str = Query("full", pattern="^(full|summary)$"),
):
    return {"items": await manager.list_libraries(stats_mode=stats_mode)}

@router.post("/api/v1/libraries", dependencies=[Depends(require_auth)])
async def create_library(payload: LibraryCreate):
    logger.info("创建记忆库：library_id=%s name=%s provider=%s", payload.id, payload.name, payload.provider_id)
    try:
        result = await manager.create_library(payload.model_dump())
        logger.info("记忆库创建完成：library_id=%s", result.get("id"))
        return result
    except ValueError as exc:
        logger.warning("记忆库创建失败：library_id=%s err=%s", payload.id, exc)
        raise HTTPException(400, str(exc)) from exc

@router.post("/api/v1/libraries/{library_id}/psk", dependencies=[Depends(require_auth)])
async def library_psk(library_id: str, payload: LibraryPskRequest):
    record = await manager.control.get_library(library_id)
    if not record:
        raise HTTPException(404, "memory library not found")
    if not auth.password_enabled:
        raise HTTPException(
            400,
            "请先在基础设置中设置 WebUI 登录密码，再查看记忆库接入密钥",
        )
    if not await asyncio.to_thread(
        verify_password, payload.password, auth.password_hash
    ):
        logger.warning("记忆库 PSK 查看鉴权失败：library_id=%s", library_id)
        raise HTTPException(401, "invalid password")
    logger.info("记忆库 PSK 已生成：library_id=%s", library_id)
    return {
        "library_id": library_id,
        "psk": _derive_library_psk(library_id),
    }

@router.get("/api/v1/libraries/{library_id}", dependencies=[Depends(require_auth)])
async def library_detail(library_id: str):
    try:
        return await manager.library_detail(library_id)
    except KeyError as exc:
        raise HTTPException(404, "memory library not found") from exc

@router.post(
    "/api/v1/libraries/{library_id}/adapters/heartbeat",
    dependencies=[Depends(require_auth)],
)
async def adapter_heartbeat(
    library_id: str,
    request: Request,
    payload: AdapterHeartbeatRequest | None = None,
):
    payload = payload or AdapterHeartbeatRequest()
    adapter_id = _adapter_header(request, ADAPTER_ID_HEADER)
    instance_id = _adapter_header(request, ADAPTER_INSTANCE_HEADER)
    adapter_type = _adapter_header(request, ADAPTER_TYPE_HEADER) or "unknown"
    if not adapter_id or not instance_id:
        raise HTTPException(
            400,
            "adapter heartbeat requires adapter id and instance id headers",
        )
    try:
        validate_identifier(library_id, field="记忆库 ID")
        validate_identifier(adapter_id, field="适配器标识ID")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    waiter = (
        manager.subscribe_adapter_disconnect(library_id, adapter_id)
        if payload.wait_seconds > 0 and not payload.manual_reconnect
        else None
    )
    try:
        connection = await manager.control.register_adapter_connection(
            library_id,
            adapter_id=adapter_id,
            instance_id=instance_id,
            adapter_type=adapter_type,
            manual_reconnect=payload.manual_reconnect,
        )
    except KeyError as exc:
        if waiter is not None:
            manager.unsubscribe_adapter_disconnect(library_id, adapter_id, waiter)
        raise HTTPException(404, "memory library not found") from exc
    except AdapterForcedOfflineError as exc:
        if waiter is not None:
            manager.unsubscribe_adapter_disconnect(library_id, adapter_id, waiter)
        raise HTTPException(
            409,
            _adapter_forced_offline_detail(exc.connection),
        ) from exc
    except ValueError as exc:
        if waiter is not None:
            manager.unsubscribe_adapter_disconnect(library_id, adapter_id, waiter)
        raise HTTPException(409, str(exc)) from exc
    if payload.manual_reconnect:
        manager.clear_adapter_forced_offline(library_id, adapter_id)
    try:
        if waiter is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(waiter),
                    timeout=float(payload.wait_seconds),
                )
            except TimeoutError:
                pass
            current = await manager.control.adapter_connection(
                library_id,
                adapter_id,
            )
            if current and current.get("state") == "forced_offline":
                raise HTTPException(
                    409,
                    _adapter_forced_offline_detail(current),
                )
            if current:
                connection = current
    finally:
        if waiter is not None:
            manager.unsubscribe_adapter_disconnect(
                library_id,
                adapter_id,
                waiter,
            )
    busy_job = (
        await manager.jobs.active_long_job(library_id)
        if manager.jobs is not None
        else await manager.control.active_long_job(library_id)
    )
    logger.debug(
        "Adapter 心跳已登记：library_id=%s adapter_id=%s instance=%s type=%s busy=%s",
        library_id,
        adapter_id,
        instance_id[:12],
        adapter_type,
        bool(busy_job),
    )
    return {
        "library_id": library_id,
        "adapter_id": adapter_id,
        "connection_state": "active",
        "active_ttl_seconds": ADAPTER_CONNECTION_TTL_SECONDS,
        "connection": connection,
        "adapter_busy": {
            "busy": bool(busy_job),
            "job": _adapter_job_summary(busy_job),
        },
    }


@router.post(
    "/api/v1/libraries/{library_id}/adapters/{adapter_id}/disconnect",
    dependencies=[Depends(require_admin_auth)],
)
async def disconnect_adapter(
    library_id: str,
    adapter_id: str,
    payload: AdapterDisconnectRequest,
):
    try:
        validate_identifier(library_id, field="记忆库 ID")
        validate_identifier(adapter_id, field="适配器标识ID")
        connection = await manager.control.force_disconnect_adapter(
            library_id,
            adapter_id,
            expected_instance_id=payload.instance_id,
        )
    except KeyError as exc:
        raise HTTPException(404, "adapter connection not found") from exc
    except AdapterConnectionChangedError as exc:
        raise HTTPException(
            409,
            {
                "code": "adapter_connection_changed",
                "message": str(exc),
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    manager.mark_adapter_forced_offline(connection)
    manager.notify_adapter_disconnect(library_id, adapter_id)
    logger.warning(
        "Adapter 已被管理员强制下线：library_id=%s adapter_id=%s instance=%s",
        library_id,
        adapter_id,
        payload.instance_id[:12],
    )
    return {
        "library_id": library_id,
        "adapter_id": adapter_id,
        "state": "forced_offline",
        "connection": connection,
    }

@router.patch("/api/v1/libraries/{library_id}", dependencies=[Depends(require_auth)])
async def update_library(library_id: str, payload: LibraryUpdate):
    logger.info("更新记忆库：library_id=%s fields=%s", library_id, sorted(payload.model_dump(exclude_none=True).keys()))
    try:
        result = await manager.update_library(
            library_id, payload.model_dump(exclude_none=True)
        )
        logger.info("记忆库更新完成：library_id=%s", library_id)
        return result
    except ValueError as exc:
        logger.warning("记忆库更新失败：library_id=%s err=%s", library_id, exc)
        status_code = 409 if "适配器连接" in str(exc) or "任务" in str(exc) else 400
        raise HTTPException(status_code, str(exc)) from exc
    except KeyError as exc:
        logger.warning("记忆库更新失败：library_id=%s err=not_found", library_id)
        raise HTTPException(404, "memory library not found") from exc

@router.delete("/api/v1/libraries/{library_id}", dependencies=[Depends(require_auth)])
async def delete_library(library_id: str):
    logger.warning("删除记忆库请求：library_id=%s", library_id)
    try:
        result = await manager.delete_library(library_id)
        logger.warning(
            "记忆库已删除并保留核心数据库：library_id=%s trash=%s",
            library_id,
            result.get("trash"),
        )
        return result
    except KeyError as exc:
        logger.warning("删除记忆库失败：library_id=%s err=not_found", library_id)
        raise HTTPException(404, "memory library not found") from exc
    except ValueError as exc:
        logger.warning("删除记忆库被拒绝：library_id=%s err=%s", library_id, exc)
        raise HTTPException(409, str(exc)) from exc

@router.post(
    "/api/v1/libraries/{library_id}/backup",
    dependencies=[Depends(require_auth)],
)
async def backup_library(library_id: str):
    logger.info("开始备份记忆库：library_id=%s", library_id)
    try:
        result = await manager.backup_library(library_id)
        logger.info("记忆库备份完成：library_id=%s path=%s", library_id, result.get("path"))
        return result
    except KeyError as exc:
        logger.warning("记忆库备份失败：library_id=%s err=not_found", library_id)
        raise HTTPException(404, "memory library not found") from exc

@router.post(
    "/api/v1/libraries/{library_id}/set-default",
    dependencies=[Depends(require_auth)],
)
async def set_default_library(library_id: str):
    logger.info("设置默认记忆库：library_id=%s", library_id)
    try:
        result = await manager.set_default(library_id)
        logger.info("默认记忆库已更新：library_id=%s", library_id)
        return result
    except KeyError as exc:
        logger.warning("设置默认记忆库失败：library_id=%s err=not_found", library_id)
        raise HTTPException(404, "memory library not found") from exc

@router.post(
    "/api/v1/libraries/{library_id}/copy",
    dependencies=[Depends(require_auth)],
)
async def copy_library(library_id: str):
    logger.warning("复制记忆库请求：library_id=%s", library_id)
    try:
        job_id = await jobs().start(
            "library_copy",
            lambda progress: manager.copy_library(library_id, progress),
            library_id=library_id,
        )
        logger.warning("复制记忆库任务已创建：library_id=%s job_id=%s", library_id, job_id)
        return {"job_id": job_id}
    except KeyError as exc:
        logger.warning("复制记忆库失败：library_id=%s err=not_found", library_id)
        raise HTTPException(404, "memory library not found") from exc
    except ValueError as exc:
        logger.warning("复制记忆库失败：library_id=%s err=%s", library_id, exc)
        raise HTTPException(409, str(exc)) from exc

@router.post(
    "/api/v1/libraries/{library_id}/conversations/messages",
    dependencies=[Depends(require_auth)],
)
async def append_conversation_message(
    library_id: str, payload: ConversationMessageCreate
):
    target = await runtime(library_id)
    result = await target.append_conversation_message(payload.model_dump())
    logger.info(
        "会话消息写入：library_id=%s session=%s role=%s duplicate=%s sender=%s",
        library_id,
        payload.session_id,
        payload.role,
        result.get("duplicate"),
        payload.sender_id or "",
    )
    return result

@router.get(
    "/api/v1/libraries/{library_id}/conversations/{session_id}",
    dependencies=[Depends(require_auth)],
)
async def get_conversation(
    library_id: str,
    session_id: str,
    limit: int | None = Query(default=None, ge=0, le=1000),
):
    target = await runtime(library_id)
    effective_limit = (
        limit
        if limit is not None
        else max(0, min(1000, int(target.config.conversation.context_window_size)))
    )
    result = await target.storage.get_conversation(
        session_id, limit=effective_limit
    )
    if result is None:
        raise HTTPException(404, "conversation not found")
    return result

@router.get(
    "/api/v1/libraries/{library_id}/conversations/{session_id}/messages",
    dependencies=[Depends(require_auth)],
)
async def get_conversation_messages(
    library_id: str,
    session_id: str,
    start_index: int | None = Query(default=None, ge=0),
    end_index: int | None = Query(default=None, ge=0),
    limit: int | None = Query(default=None, ge=0, le=1000),
):
    target = await runtime(library_id)
    if end_index is not None and start_index is not None and end_index < start_index:
        raise HTTPException(400, "end_index must be greater than or equal to start_index")
    messages = await target.storage.get_conversation_messages(
        session_id,
        start_index=start_index,
        end_index=end_index,
        limit=limit,
    )
    return {"session_id": session_id, "messages": messages, "count": len(messages)}

@router.patch(
    "/api/v1/libraries/{library_id}/conversations/{session_id}/metadata",
    dependencies=[Depends(require_auth)],
)
async def patch_conversation_metadata(
    library_id: str,
    session_id: str,
    payload: ConversationMetadataUpdate,
):
    target = await runtime(library_id)
    result = await target.storage.update_conversation_metadata(
        session_id, payload.metadata
    )
    if result is None:
        raise HTTPException(404, "conversation not found")
    logger.info(
        "会话元数据更新：library_id=%s session=%s keys=%s",
        target.library_id,
        session_id,
        sorted(payload.metadata.keys()),
    )
    return {"session": result}

@router.post(
    "/api/v1/libraries/{library_id}/conversations/{session_id}/clear",
    dependencies=[Depends(require_auth)],
)
async def clear_conversation(library_id: str, session_id: str):
    target = await runtime(library_id)
    result = await target.storage.clear_conversation(session_id)
    logger.warning(
        "会话短期上下文已清理：library_id=%s session=%s deleted=%s",
        target.library_id,
        session_id,
        result.get("deleted"),
    )
    return result

@router.post(
    "/api/v1/libraries/{library_id}/conversations/{session_id}/trim",
    dependencies=[Depends(require_auth)],
)
async def trim_conversation(
    library_id: str,
    session_id: str,
    payload: ConversationTrimRequest,
):
    target = await runtime(library_id)
    result = await target.storage.trim_conversation(
        session_id, payload.delete_count
    )
    logger.info(
        "会话已总结消息清理：library_id=%s session=%s requested=%s deleted=%s",
        target.library_id,
        session_id,
        payload.delete_count,
        result.get("deleted"),
    )
    return result
