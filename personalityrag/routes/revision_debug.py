from __future__ import annotations

import asyncio
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from ..application_context import auth, current_context, manager
from ..http_shared import (
    require_revision_debug_access,
    require_revision_debug_session_control,
)
from ..listener_surface import request_surface
from ..logger import logger
from ..revision_debug import DEBUG_COOKIE_NAME, is_loopback_client, request_id
from ..schemas import DebugDatabaseBindingPatch, DebugSessionUnlock


router = APIRouter()


def _client_limit_key(request: Request, purpose: str) -> str:
    host = str(request.client.host if request.client else "unknown").strip().lower()
    return f"{purpose}:{host}"


async def verify_debug_password(
    request: Request,
    *,
    password: str,
    purpose: str,
    risk_confirmed: bool = True,
) -> None:
    if not risk_confirmed:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "debug_risk_confirmation_required",
                "message": "risk confirmation is required",
            },
        )
    if not auth.password_enabled:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "debug_password_not_configured",
                "message": "configure a WebUI login password first",
            },
        )
    key = _client_limit_key(request, purpose)
    limiter = current_context().revision_debug
    limit = limiter.failure_status(key)
    if limit["limited"]:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "debug_password_rate_limited",
                "message": "too many failed password attempts",
                "retry_after": limit["retry_after"],
            },
            headers={"Retry-After": str(limit["retry_after"])},
        )
    verified = await asyncio.to_thread(auth.verify_login_secret, password)
    if not verified:
        limiter.record_failure(key)
        logger.warning(
            "revision debug password rejected: purpose=%s request_id=%s",
            purpose,
            request_id(request),
        )
        raise HTTPException(
            status_code=401,
            detail={
                "code": "debug_password_invalid",
                "message": "invalid WebUI password",
            },
        )
    limiter.clear_failures(key)


@router.get(
    "/api/v1/debug/session",
    dependencies=[Depends(require_revision_debug_session_control)],
)
async def debug_session_status(request: Request):
    token = request.cookies.get(DEBUG_COOKIE_NAME)
    return current_context().revision_debug.status(token, request, auth)


@router.post(
    "/api/v1/debug/session",
    dependencies=[Depends(require_revision_debug_session_control)],
)
async def unlock_debug_session(
    payload: DebugSessionUnlock,
    request: Request,
    response: Response,
):
    await verify_debug_password(
        request,
        password=payload.password,
        purpose="unlock",
    )
    token, status = current_context().revision_debug.issue(request, auth)
    response.set_cookie(
        DEBUG_COOKIE_NAME,
        token,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
        max_age=int(current_context().revision_debug.ttl_seconds),
        path="/",
    )
    logger.warning(
        "revision debug session unlocked: request_id=%s surface=%s loopback=%s",
        request_id(request),
        request_surface(request),
        str(is_loopback_client(request)).lower(),
    )
    return status


@router.delete(
    "/api/v1/debug/session",
    dependencies=[Depends(require_revision_debug_session_control)],
)
async def lock_debug_session(request: Request, response: Response):
    current_context().revision_debug.revoke(
        request.cookies.get(DEBUG_COOKIE_NAME),
        auth,
    )
    response.delete_cookie(DEBUG_COOKIE_NAME, path="/")
    logger.warning(
        "revision debug session locked: request_id=%s",
        request_id(request),
    )
    return {
        "password_configured": auth.password_enabled,
        "unlocked": False,
        "expires_at": None,
        "remaining_seconds": 0,
        "fixed_ttl_seconds": int(current_context().revision_debug.ttl_seconds),
    }


@router.get(
    "/api/v1/debug/revisions/overview",
    dependencies=[Depends(require_revision_debug_access)],
)
async def revision_debug_overview():
    return await manager.debug_revision_overview()


@router.patch(
    "/api/v1/debug/databases/{database_type}/{database_id}/bindings/{usage_kind}",
    dependencies=[Depends(require_revision_debug_access)],
)
async def patch_database_revision_binding(
    database_type: str,
    database_id: str,
    usage_kind: str,
    payload: DebugDatabaseBindingPatch,
    request: Request,
):
    allow_dangerous = bool(
        payload.assert_functional_compatibility and payload.risk_confirmed
    )
    if allow_dangerous:
        await verify_debug_password(
            request,
            password=payload.password,
            purpose="binding",
            risk_confirmed=payload.risk_confirmed,
        )
    try:
        result = await manager.debug_repair_database_binding(
            database_type=database_type,
            database_id=database_id,
            usage_kind=usage_kind,
            provider_id=payload.provider_id,
            revision=payload.revision,
            allow_non_equivalent=allow_dangerous,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    logger.warning(
        "revision debug database binding repaired: database_type=%s "
        "database_id=%s usage_kind=%s provider_id=%s revision=%s request_id=%s",
        database_type,
        database_id,
        usage_kind,
        payload.provider_id,
        payload.revision,
        request_id(request),
    )
    return result
