from __future__ import annotations


from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
)

from ..application_context import manager
from ..http_shared import (
    require_auth,
    require_revision_debug_access,
)
from ..logger import logger
from ..schemas import (
    DebugProviderRevisionPatch,
    DebugProviderRevisionReset,
    ProviderCopy,
    ProviderCreate,
    ProviderUpdate,
)
from .revision_debug import verify_debug_password


router = APIRouter()
@router.get("/api/v1/provider-types", dependencies=[Depends(require_auth)])
async def provider_types(kind: str | None = Query(default=None, pattern="^(embedding|rerank)$")):
    return {"items": manager.control.provider_types(kind)}

@router.get("/api/v1/providers", dependencies=[Depends(require_auth)])
async def providers(kind: str | None = Query(default=None, pattern="^(embedding|rerank)$")):
    return {"items": await manager.list_providers(kind)}

@router.post("/api/v1/providers", dependencies=[Depends(require_auth)])
async def create_provider(payload: ProviderCreate):
    logger.info("创建模型提供商：provider_id=%s type=%s model=%s", payload.id, payload.type, payload.model)
    try:
        result = await manager.create_provider(payload.model_dump())
        logger.info("模型提供商创建完成：provider_id=%s revision=%s", result.get("id"), result.get("revision"))
        return result
    except ValueError as exc:
        logger.warning("模型提供商创建失败：provider_id=%s err=%s", payload.id, exc)
        raise HTTPException(400, str(exc)) from exc

@router.patch(
    "/api/v1/providers/{provider_id}",
    dependencies=[Depends(require_auth)],
)
async def update_provider(provider_id: str, payload: ProviderUpdate):
    fields = sorted(payload.model_dump(exclude_none=True).keys())
    logger.info("更新模型提供商：provider_id=%s fields=%s", provider_id, fields)
    try:
        result = await manager.update_provider(
            provider_id, payload.model_dump(exclude_none=True)
        )
        logger.info("模型提供商更新完成：provider_id=%s revision=%s", result.get("id"), result.get("revision"))
        return result
    except KeyError as exc:
        logger.warning("模型提供商更新失败：provider_id=%s err=not_found", provider_id)
        raise HTTPException(404, "provider not found") from exc
    except ValueError as exc:
        logger.warning("模型提供商更新被拒绝：provider_id=%s err=%s", provider_id, exc)
        raise HTTPException(409, str(exc)) from exc

@router.delete(
    "/api/v1/providers/{provider_id}",
    dependencies=[Depends(require_auth)],
)
async def delete_provider(provider_id: str):
    logger.warning("删除模型提供商请求：provider_id=%s", provider_id)
    try:
        await manager.delete_provider(provider_id)
        logger.warning("模型提供商已删除：provider_id=%s", provider_id)
        return {"deleted": True}
    except KeyError as exc:
        logger.warning("删除模型提供商失败：provider_id=%s err=not_found", provider_id)
        raise HTTPException(404, "provider not found") from exc
    except ValueError as exc:
        logger.warning("删除模型提供商被拒绝：provider_id=%s err=%s", provider_id, exc)
        raise HTTPException(409, str(exc)) from exc

@router.post(
    "/api/v1/providers/{provider_id}/copy",
    dependencies=[Depends(require_auth)],
)
async def copy_provider(provider_id: str, payload: ProviderCopy):
    logger.info("复制模型提供商：source=%s new_id=%s", provider_id, payload.new_id or "")
    try:
        result = await manager.copy_provider(provider_id, payload.new_id)
        logger.info("模型提供商复制完成：source=%s target=%s", provider_id, result.get("id"))
        return result
    except KeyError as exc:
        logger.warning("复制模型提供商失败：provider_id=%s err=not_found", provider_id)
        raise HTTPException(404, "provider not found") from exc
    except ValueError as exc:
        logger.warning("复制模型提供商失败：provider_id=%s err=%s", provider_id, exc)
        raise HTTPException(400, str(exc)) from exc

@router.post(
    "/api/v1/providers/{provider_id}/test",
    dependencies=[Depends(require_auth)],
)
async def test_provider(provider_id: str):
    logger.info("测试模型提供商连接：provider_id=%s", provider_id)
    try:
        result = await manager.test_provider(provider_id)
        logger.info(
            "模型提供商连接成功：provider_id=%s model=%s dimension=%s latency_ms=%s",
            provider_id,
            result.get("resolved_model") or result.get("model"),
            result.get("dimension") or result.get("dimensions"),
            result.get("elapsed_ms"),
        )
        return result
    except KeyError as exc:
        logger.warning("测试模型提供商失败：provider_id=%s err=not_found", provider_id)
        raise HTTPException(404, "provider not found") from exc
    except Exception as exc:
        logger.warning("测试模型提供商失败：provider_id=%s err=%s", provider_id, exc)
        raise

@router.post("/api/v1/providers/test-draft", dependencies=[Depends(require_auth)])
async def test_provider_draft(payload: ProviderCreate):
    logger.info("测试草稿模型提供商：provider_id=%s type=%s model=%s", payload.id, payload.type, payload.model)
    result = await manager.test_provider_draft(payload.model_dump())
    logger.info("草稿模型提供商连接成功：provider_id=%s dimension=%s", payload.id, result.get("dimension") or result.get("dimensions"))
    return result

@router.post(
    "/api/v1/providers/detect-dimension",
    dependencies=[Depends(require_auth)],
)
async def detect_provider_dimension(payload: ProviderCreate):
    logger.info("自动检测模型维度：provider_id=%s type=%s model=%s", payload.id, payload.type, payload.model)
    result = await manager.detect_dimension(payload.model_dump())
    logger.info("模型维度检测完成：provider_id=%s dimensions=%s", payload.id, result.get("dimensions"))
    return result

@router.post(
    "/api/v1/providers/detect-context-length",
    dependencies=[Depends(require_auth)],
)
async def detect_provider_context_length(payload: ProviderCreate):
    logger.info(
        "manual provider max-context probe: provider_id=%s type=%s model=%s",
        payload.id,
        payload.type,
        payload.model,
    )
    result = await manager.detect_context_length(payload.model_dump())
    logger.info(
        "provider max-context probe finished: provider_id=%s tokens=%s source=%s",
        payload.id,
        result.get("max_context_tokens"),
        result.get("max_context_tokens_source") or "",
    )
    return result

@router.post(
    "/api/v1/providers/{provider_id}/detect-context-length",
    dependencies=[Depends(require_auth)],
)
async def detect_existing_provider_context_length(
    provider_id: str,
    payload: ProviderUpdate,
):
    logger.info("manual saved-provider max-context probe: provider_id=%s", provider_id)
    result = await manager.detect_context_length(
        payload.model_dump(exclude_unset=True),
        provider_id=provider_id,
    )
    logger.info(
        "saved-provider max-context probe finished: provider_id=%s tokens=%s source=%s",
        provider_id,
        result.get("max_context_tokens"),
        result.get("max_context_tokens_source") or "",
    )
    return result

@router.post("/api/v1/providers/test", dependencies=[Depends(require_auth)])
async def provider_test_compat():
    library = await manager.control.default_library()
    logger.info(
        "测试默认记忆库当前模型提供商：memory_store_id=%s provider=%s revision=%s",
        library.id,
        library.provider_id,
        library.provider_revision,
    )
    return await manager.test_provider(
        library.provider_id, revision=library.provider_revision
    )

@router.get(
    "/api/v1/debug/providers/{provider_id}/revisions",
    dependencies=[Depends(require_revision_debug_access)],
)
async def debug_provider_revisions(provider_id: str):
    try:
        return await manager.debug_provider_revisions(provider_id)
    except KeyError as exc:
        raise HTTPException(404, "provider not found") from exc

@router.patch(
    "/api/v1/debug/providers/{provider_id}/revisions/{revision}",
    dependencies=[Depends(require_revision_debug_access)],
)
async def debug_patch_provider_revision(
    provider_id: str,
    revision: int,
    payload: DebugProviderRevisionPatch,
    request: Request,
):
    await verify_debug_password(
        request,
        password=payload.password,
        purpose="provider_revision_patch",
        risk_confirmed=payload.risk_confirmed,
    )
    logger.warning(
        "debug provider revision patch requested: provider_id=%s revision=%s fields=%s",
        provider_id,
        revision,
        sorted(payload.patch.keys()),
    )
    try:
        return await manager.debug_patch_provider_revision(
            provider_id,
            revision,
            payload.patch,
        )
    except KeyError as exc:
        raise HTTPException(404, "provider or revision not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

@router.post(
    "/api/v1/debug/providers/{provider_id}/revisions/reset",
    dependencies=[Depends(require_revision_debug_access)],
)
async def debug_reset_provider_revisions(
    provider_id: str,
    payload: DebugProviderRevisionReset,
    request: Request,
):
    allow_dangerous = bool(payload.force_non_equivalent and payload.risk_confirmed)
    if payload.delete_revisions_after_latest or payload.force_non_equivalent:
        await verify_debug_password(
            request,
            password=payload.password,
            purpose="provider_revision_reset",
            risk_confirmed=payload.risk_confirmed,
        )
        allow_dangerous = True
    logger.warning(
        "debug provider revision reset requested: provider_id=%s latest=%s "
        "bind_libraries=%s delete_after=%s library_overrides=%s",
        provider_id,
        payload.latest_revision,
        payload.bind_libraries_to_latest,
        payload.delete_revisions_after_latest,
        sorted(payload.library_revisions.keys()),
    )
    try:
        return await manager.debug_reset_provider_revisions(
            provider_id,
            latest_revision=payload.latest_revision,
            bind_libraries_to_latest=payload.bind_libraries_to_latest,
            library_revisions=payload.library_revisions,
            delete_revisions_after_latest=payload.delete_revisions_after_latest,
            allow_non_equivalent=allow_dangerous,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
