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
    _current_adapter_connection_url,
    _adapter_forced_offline_detail,
    _adapter_job_summary,
    _derive_database_access_key,
    jobs,
    LivingMemoryV8Type,
    TextMediaV1Type,
    require_auth,
    require_admin_auth,
    require_existing_database_ref,
    runtime,
)
from ..database_types import (
    LIVINGMEMORY_V8_TYPE,
    TEXT_MEDIA_V1_TYPE,
    database_identity_fields,
    database_type_registry,
)
from ..library_types.livingmemory_v8.api_schemas import LivingMemoryV8Create
from ..identifiers import validate_identifier
from ..logger import logger
from ..schemas import (
    AdapterDisconnectRequest,
    AdapterHeartbeatRequest,
    ConversationMessageCreate,
    ConversationMetadataUpdate,
    ConversationTrimRequest,
    LibraryPskRequest,
    LibraryUpdate,
)


catalog_router = APIRouter()
livingmemory_router = APIRouter(prefix="/api/v1/memory-libraries/livingmemory_v8")
knowledge_adapter_router = APIRouter(prefix="/api/v1/knowledge-libraries/text_media_v1")


@catalog_router.get("/api/v1/database-types", dependencies=[Depends(require_auth)])
async def database_types(
    category: str | None = Query(default=None, pattern="^(memory|knowledge)$"),
):
    return {
        "items": [
            descriptor.public() for descriptor in database_type_registry.list(category)
        ]
    }


@catalog_router.get("/api/v1/databases", dependencies=[Depends(require_auth)])
async def libraries(
    stats_mode: str = Query("full", pattern="^(full|summary)$"),
):
    return {"items": await manager.list_libraries(stats_mode=stats_mode)}


@livingmemory_router.post("", dependencies=[Depends(require_auth)])
async def create_library(payload: LivingMemoryV8Create):
    logger.info(
        "创建记忆库：memory_store_id=%s name=%s provider=%s",
        payload.id,
        payload.name,
        payload.provider_id,
    )
    try:
        result = await manager.create_library(
            {**payload.model_dump(), "database_type": LIVINGMEMORY_V8_TYPE}
        )
        logger.info("记忆库创建完成：memory_store_id=%s", result.get("id"))
        return result
    except ValueError as exc:
        logger.warning("记忆库创建失败：memory_store_id=%s err=%s", payload.id, exc)
        status_code = 409 if "已存在" in str(exc) else 400
        raise HTTPException(status_code, str(exc)) from exc


@livingmemory_router.post(
    "/{memory_store_id}/access-key", dependencies=[Depends(require_auth)]
)
async def database_access_key(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    payload: LibraryPskRequest,
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="adapter_access"
    )
    if not auth.password_enabled:
        raise HTTPException(
            400,
            "请先在基础设置中设置 WebUI 登录密码，再查看记忆库接入密钥",
        )
    if not await asyncio.to_thread(
        verify_password, payload.password, auth.password_hash
    ):
        logger.warning("数据库接入密钥查看鉴权失败：database=%s", ref.key)
        raise HTTPException(401, "invalid password")
    driver = database_type_registry.require(ref.database_type)
    access_key = _derive_database_access_key(ref.database_type, ref.id)
    logger.info("数据库接入密钥已生成：database=%s", ref.key)
    return {
        **database_identity_fields(ref),
        "database_category": driver.descriptor.category,
        "key_prefix": driver.descriptor.key_prefix,
        "access_key": access_key,
        "adapter_url": _current_adapter_connection_url(),
    }


@livingmemory_router.get("/{memory_store_id}", dependencies=[Depends(require_auth)])
async def library_detail(database_type: LivingMemoryV8Type, memory_store_id: str):
    ref = await require_existing_database_ref(database_type, memory_store_id)
    try:
        return await manager.library_detail(ref)
    except KeyError as exc:
        raise HTTPException(404, "memory library not found") from exc


@livingmemory_router.post(
    "/{memory_store_id}/adapters/heartbeat",
    dependencies=[Depends(require_auth)],
)
async def adapter_heartbeat(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    request: Request,
    payload: AdapterHeartbeatRequest | None = None,
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="adapter_access"
    )
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
        validate_identifier(memory_store_id, field="记忆库 ID")
        validate_identifier(adapter_id, field="适配器标识ID")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    waiter = (
        manager.subscribe_adapter_disconnect(ref, adapter_id)
        if payload.wait_seconds > 0 and not payload.manual_reconnect
        else None
    )
    try:
        connection = await manager.control.register_adapter_connection(
            ref,
            adapter_id=adapter_id,
            instance_id=instance_id,
            adapter_type=adapter_type,
            manual_reconnect=payload.manual_reconnect,
        )
    except KeyError as exc:
        if waiter is not None:
            manager.unsubscribe_adapter_disconnect(ref, adapter_id, waiter)
        raise HTTPException(404, "memory library not found") from exc
    except AdapterForcedOfflineError as exc:
        if waiter is not None:
            manager.unsubscribe_adapter_disconnect(ref, adapter_id, waiter)
        raise HTTPException(
            409,
            _adapter_forced_offline_detail(exc.connection),
        ) from exc
    except ValueError as exc:
        if waiter is not None:
            manager.unsubscribe_adapter_disconnect(ref, adapter_id, waiter)
        raise HTTPException(409, str(exc)) from exc
    if payload.manual_reconnect:
        manager.clear_adapter_forced_offline(ref, adapter_id)
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
                ref,
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
                ref,
                adapter_id,
                waiter,
            )
    busy_job = (
        await manager.jobs.active_long_job(memory_store_id)
        if manager.jobs is not None
        else await manager.control.active_long_job(memory_store_id)
    )
    logger.debug(
        "Adapter 心跳已登记：memory_store_id=%s adapter_id=%s instance=%s type=%s busy=%s",
        memory_store_id,
        adapter_id,
        instance_id[:12],
        adapter_type,
        bool(busy_job),
    )
    target = await manager.get_runtime(ref)
    return {
        **database_identity_fields(ref, include_deprecated=True),
        "adapter_id": adapter_id,
        "connection_state": "active",
        "active_ttl_seconds": ADAPTER_CONNECTION_TTL_SECONDS,
        "connection": connection,
        "adapter_busy": {
            "busy": bool(busy_job),
            "job": _adapter_job_summary(busy_job),
        },
        "maintenance": target.maintenance_status(),
    }


@livingmemory_router.post(
    "/{memory_store_id}/adapters/{adapter_id}/disconnect",
    dependencies=[Depends(require_admin_auth)],
)
async def disconnect_adapter(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    adapter_id: str,
    payload: AdapterDisconnectRequest,
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="adapter_access"
    )
    try:
        validate_identifier(memory_store_id, field="记忆库 ID")
        validate_identifier(adapter_id, field="适配器标识ID")
        connection = await manager.control.force_disconnect_adapter(
            ref,
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
    manager.notify_adapter_disconnect(ref, adapter_id)
    logger.warning(
        "Adapter 已被管理员强制下线：memory_store_id=%s adapter_id=%s instance=%s",
        memory_store_id,
        adapter_id,
        payload.instance_id[:12],
    )
    return {
        **database_identity_fields(ref, include_deprecated=True),
        "adapter_id": adapter_id,
        "state": "forced_offline",
        "connection": connection,
    }


@knowledge_adapter_router.post(
    "/{knowledge_base_id}/adapters/heartbeat",
    dependencies=[Depends(require_auth)],
)
async def knowledge_adapter_heartbeat(
    database_type: TextMediaV1Type,
    knowledge_base_id: str,
    request: Request,
    payload: AdapterHeartbeatRequest | None = None,
):
    ref = await require_existing_database_ref(
        database_type,
        knowledge_base_id,
        capability="adapter_access",
    )
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
        validate_identifier(knowledge_base_id, field="knowledge base ID")
        validate_identifier(adapter_id, field="adapter ID")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    waiter = (
        manager.subscribe_adapter_disconnect(ref, adapter_id)
        if payload.wait_seconds > 0 and not payload.manual_reconnect
        else None
    )
    try:
        connection = await manager.control.register_adapter_connection(
            ref,
            adapter_id=adapter_id,
            instance_id=instance_id,
            adapter_type=adapter_type,
            manual_reconnect=payload.manual_reconnect,
        )
    except KeyError as exc:
        if waiter is not None:
            manager.unsubscribe_adapter_disconnect(ref, adapter_id, waiter)
        raise HTTPException(404, "knowledge database not found") from exc
    except AdapterForcedOfflineError as exc:
        if waiter is not None:
            manager.unsubscribe_adapter_disconnect(ref, adapter_id, waiter)
        raise HTTPException(
            409,
            _adapter_forced_offline_detail(exc.connection),
        ) from exc
    except ValueError as exc:
        if waiter is not None:
            manager.unsubscribe_adapter_disconnect(ref, adapter_id, waiter)
        raise HTTPException(409, str(exc)) from exc
    if payload.manual_reconnect:
        manager.clear_adapter_forced_offline(ref, adapter_id)
    try:
        if waiter is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(waiter),
                    timeout=float(payload.wait_seconds),
                )
            except TimeoutError:
                pass
            current = await manager.control.adapter_connection(ref, adapter_id)
            if current and current.get("state") == "forced_offline":
                raise HTTPException(
                    409,
                    _adapter_forced_offline_detail(current),
                )
            if current:
                connection = current
    finally:
        if waiter is not None:
            manager.unsubscribe_adapter_disconnect(ref, adapter_id, waiter)
    resource_key = database_type_registry.require(TEXT_MEDIA_V1_TYPE).resource_key(
        knowledge_base_id
    )
    busy_job = (
        await manager.jobs.active_long_job(resource_key)
        if manager.jobs is not None
        else await manager.control.active_long_job(resource_key)
    )
    logger.debug(
        "Knowledge Adapter heartbeat registered: knowledge_base_id=%s adapter_id=%s "
        "instance=%s type=%s busy=%s",
        knowledge_base_id,
        adapter_id,
        instance_id[:12],
        adapter_type,
        bool(busy_job),
    )
    return {
        **database_identity_fields(ref, include_deprecated=True),
        "adapter_id": adapter_id,
        "connection_state": "active",
        "active_ttl_seconds": ADAPTER_CONNECTION_TTL_SECONDS,
        "connection": connection,
        "adapter_busy": {
            "busy": bool(busy_job),
            "job": _adapter_job_summary(busy_job),
        },
    }


@knowledge_adapter_router.post(
    "/{knowledge_base_id}/adapters/{adapter_id}/disconnect",
    dependencies=[Depends(require_admin_auth)],
)
async def disconnect_knowledge_adapter(
    database_type: TextMediaV1Type,
    knowledge_base_id: str,
    adapter_id: str,
    payload: AdapterDisconnectRequest,
):
    ref = await require_existing_database_ref(
        database_type,
        knowledge_base_id,
        capability="adapter_access",
    )
    try:
        validate_identifier(knowledge_base_id, field="knowledge base ID")
        validate_identifier(adapter_id, field="adapter ID")
        connection = await manager.control.force_disconnect_adapter(
            ref,
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
    manager.notify_adapter_disconnect(ref, adapter_id)
    logger.warning(
        "Knowledge Adapter forced offline: knowledge_base_id=%s adapter_id=%s instance=%s",
        knowledge_base_id,
        adapter_id,
        payload.instance_id[:12],
    )
    return {
        **database_identity_fields(ref, include_deprecated=True),
        "adapter_id": adapter_id,
        "state": "forced_offline",
        "connection": connection,
    }


@livingmemory_router.patch("/{memory_store_id}", dependencies=[Depends(require_auth)])
async def update_library(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    payload: LibraryUpdate,
):
    ref = await require_existing_database_ref(database_type, memory_store_id)
    logger.info(
        "更新记忆库：memory_store_id=%s fields=%s",
        memory_store_id,
        sorted(payload.model_dump(exclude_none=True).keys()),
    )
    try:
        result = await manager.update_library(
            ref, payload.model_dump(exclude_none=True)
        )
        logger.info("记忆库更新完成：memory_store_id=%s", memory_store_id)
        return result
    except ValueError as exc:
        logger.warning("记忆库更新失败：memory_store_id=%s err=%s", memory_store_id, exc)
        status_code = 409 if "适配器连接" in str(exc) or "任务" in str(exc) else 400
        raise HTTPException(status_code, str(exc)) from exc
    except KeyError as exc:
        logger.warning(
            "记忆库更新失败：memory_store_id=%s err=not_found",
            memory_store_id,
        )
        raise HTTPException(404, "memory library not found") from exc


@livingmemory_router.delete("/{memory_store_id}", dependencies=[Depends(require_auth)])
async def delete_library(database_type: LivingMemoryV8Type, memory_store_id: str):
    ref = await require_existing_database_ref(database_type, memory_store_id)
    logger.warning("删除记忆库请求：memory_store_id=%s", memory_store_id)
    try:
        result = await manager.delete_library(ref)
        logger.warning(
            "记忆库已删除并保留核心数据库：memory_store_id=%s trash=%s",
            memory_store_id,
            result.get("trash"),
        )
        return result
    except KeyError as exc:
        logger.warning(
            "删除记忆库失败：memory_store_id=%s err=not_found",
            memory_store_id,
        )
        raise HTTPException(404, "memory library not found") from exc
    except ValueError as exc:
        logger.warning(
            "删除记忆库被拒绝：memory_store_id=%s err=%s",
            memory_store_id,
            exc,
        )
        raise HTTPException(409, str(exc)) from exc


@livingmemory_router.post(
    "/{memory_store_id}/backup",
    dependencies=[Depends(require_auth)],
    status_code=202,
)
async def backup_library(database_type: LivingMemoryV8Type, memory_store_id: str):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="backup"
    )
    logger.info("开始备份记忆库：memory_store_id=%s", memory_store_id)
    try:
        job_id = await jobs().start(
            "library_backup",
            lambda _progress: manager.backup_library(ref),
            database_id=memory_store_id,
            database_type=database_type,
            lease_runtime=False,
        )
        logger.info(
            "记忆库备份任务已创建：memory_store_id=%s job_id=%s",
            memory_store_id,
            job_id,
        )
        return {"job_id": job_id}
    except KeyError as exc:
        logger.warning(
            "记忆库备份失败：memory_store_id=%s err=not_found",
            memory_store_id,
        )
        raise HTTPException(404, "memory library not found") from exc


@livingmemory_router.post(
    "/{memory_store_id}/set-default",
    dependencies=[Depends(require_auth)],
)
async def set_default_library(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
):
    ref = await require_existing_database_ref(database_type, memory_store_id)
    logger.info("设置默认记忆库：memory_store_id=%s", memory_store_id)
    try:
        result = await manager.set_default(ref)
        logger.info("默认记忆库已更新：memory_store_id=%s", memory_store_id)
        return result
    except KeyError as exc:
        logger.warning(
            "设置默认记忆库失败：memory_store_id=%s err=not_found",
            memory_store_id,
        )
        raise HTTPException(404, "memory library not found") from exc


@livingmemory_router.post(
    "/{memory_store_id}/copy",
    dependencies=[Depends(require_auth)],
)
async def copy_library(database_type: LivingMemoryV8Type, memory_store_id: str):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="copy"
    )
    logger.warning("复制记忆库请求：memory_store_id=%s", memory_store_id)
    try:
        job_id = await jobs().start(
            "library_copy",
            lambda progress: manager.copy_library(ref, progress),
            database_id=memory_store_id,
            database_type=database_type,
        )
        logger.warning(
            "复制记忆库任务已创建：memory_store_id=%s job_id=%s",
            memory_store_id,
            job_id,
        )
        return {"job_id": job_id}
    except KeyError as exc:
        logger.warning(
            "复制记忆库失败：memory_store_id=%s err=not_found",
            memory_store_id,
        )
        raise HTTPException(404, "memory library not found") from exc
    except ValueError as exc:
        logger.warning(
            "复制记忆库失败：memory_store_id=%s err=%s",
            memory_store_id,
            exc,
        )
        raise HTTPException(409, str(exc)) from exc


@livingmemory_router.post(
    "/{memory_store_id}/conversations/messages",
    dependencies=[Depends(require_auth)],
)
async def append_conversation_message(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    payload: ConversationMessageCreate,
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="conversation_buffer"
    )
    target = await runtime(ref)
    result = await target.append_conversation_message(payload.model_dump())
    logger.info(
        "会话消息写入：memory_store_id=%s session=%s role=%s duplicate=%s sender=%s",
        memory_store_id,
        payload.session_id,
        payload.role,
        result.get("duplicate"),
        payload.sender_id or "",
    )
    return result


@livingmemory_router.get(
    "/{memory_store_id}/conversations/{session_id}",
    dependencies=[Depends(require_auth)],
)
async def get_conversation(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    session_id: str,
    limit: int | None = Query(default=None, ge=0, le=1000),
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="conversation_buffer"
    )
    target = await runtime(ref)
    effective_limit = (
        limit
        if limit is not None
        else max(0, min(1000, int(target.config.conversation.context_window_size)))
    )
    result = await target.storage.get_conversation(session_id, limit=effective_limit)
    if result is None:
        raise HTTPException(404, "conversation not found")
    return result


@livingmemory_router.get(
    "/{memory_store_id}/conversations/{session_id}/messages",
    dependencies=[Depends(require_auth)],
)
async def get_conversation_messages(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    session_id: str,
    start_index: int | None = Query(default=None, ge=0),
    end_index: int | None = Query(default=None, ge=0),
    limit: int | None = Query(default=None, ge=0, le=1000),
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="conversation_buffer"
    )
    target = await runtime(ref)
    if end_index is not None and start_index is not None and end_index < start_index:
        raise HTTPException(
            400, "end_index must be greater than or equal to start_index"
        )
    messages = await target.storage.get_conversation_messages(
        session_id,
        start_index=start_index,
        end_index=end_index,
        limit=limit,
    )
    return {"session_id": session_id, "messages": messages, "count": len(messages)}


@livingmemory_router.patch(
    "/{memory_store_id}/conversations/{session_id}/metadata",
    dependencies=[Depends(require_auth)],
)
async def patch_conversation_metadata(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    session_id: str,
    payload: ConversationMetadataUpdate,
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="conversation_buffer"
    )
    target = await runtime(ref)
    result = await target.storage.update_conversation_metadata(
        session_id, payload.metadata
    )
    if result is None:
        raise HTTPException(404, "conversation not found")
    logger.info(
        "会话元数据更新：memory_store_id=%s session=%s keys=%s",
        memory_store_id,
        session_id,
        sorted(payload.metadata.keys()),
    )
    return {"session": result}


@livingmemory_router.post(
    "/{memory_store_id}/conversations/{session_id}/clear",
    dependencies=[Depends(require_auth)],
)
async def clear_conversation(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    session_id: str,
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="conversation_buffer"
    )
    target = await runtime(ref)
    result = await target.storage.clear_conversation(session_id)
    logger.warning(
        "会话短期上下文已清理：memory_store_id=%s session=%s deleted=%s",
        memory_store_id,
        session_id,
        result.get("deleted"),
    )
    return result


@livingmemory_router.post(
    "/{memory_store_id}/conversations/{session_id}/trim",
    dependencies=[Depends(require_auth)],
)
async def trim_conversation(
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
    session_id: str,
    payload: ConversationTrimRequest,
):
    ref = await require_existing_database_ref(
        database_type, memory_store_id, capability="conversation_buffer"
    )
    target = await runtime(ref)
    result = await target.storage.trim_conversation(session_id, payload.delete_count)
    logger.info(
        "会话已总结消息清理：memory_store_id=%s session=%s requested=%s deleted=%s",
        memory_store_id,
        session_id,
        payload.delete_count,
        result.get("deleted"),
    )
    return result
