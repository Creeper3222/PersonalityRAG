from __future__ import annotations

import time
import uuid
import re
import asyncio
import weakref
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders

from .application_context import (
    ApplicationContext,
    activate_context,
    activate_runtime_lease_scope,
    current_context,
    release_runtime_lease_scope,
    reset_context,
)
from .http_shared import (
    ADAPTER_ID_HEADER,
    _adapter_busy_detail,
    _adapter_forced_offline_detail,
    _adapter_header,
    _adapter_database_ref_from_path,
    _adapter_request_authenticated,
    _adapter_status_request_allowed,
)
from .logger import logger
from .database_types import (
    DATABASE_CATEGORY_MEMORY,
    database_type_registry,
)
from .listener_surface import (
    ADAPTER_ACCESS_SURFACE,
    is_adapter_access_request_allowed,
    request_surface,
)
from .resource_limits import configured_surface_limits


SLOW_REQUEST_SECONDS = 1.0


class RequestContextMiddleware:
    def __init__(self, app: Any, *, context: ApplicationContext):
        self.app = app
        self.context = context
        self._surface_slots: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop,
            tuple[str, asyncio.Semaphore, asyncio.Semaphore],
        ] = weakref.WeakKeyDictionary()

    def _request_slot(self, *, heartbeat: bool) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        profile = self.context.config.performance_profile
        current = self._surface_slots.get(loop)
        if current is None or current[0] != profile:
            foreground_limit, heartbeat_limit = configured_surface_limits(profile)
            current = (
                profile,
                asyncio.Semaphore(foreground_limit),
                asyncio.Semaphore(heartbeat_limit),
            )
            self._surface_slots[loop] = current
        return current[2] if heartbeat else current[1]

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        token = activate_context(self.context)
        if scope.get("type") != "http":
            try:
                await self.app(scope, receive, send)
            finally:
                reset_context(token)
            return

        raw_request_id = dict(scope.get("headers") or []).get(b"x-request-id", b"")
        request_id = raw_request_id.decode("ascii", errors="ignore")[:128].strip()
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_id):
            request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status_code = 500
        method = str(scope.get("method") or "").upper()
        path = str(scope.get("path") or "").rstrip("/")
        expected_long_poll = (
            method == "POST"
            and re.fullmatch(
                r"/api/v1/(?:memory-libraries/livingmemory_v8|knowledge-libraries/text_media_v1)/[^/]+/adapters/heartbeat",
                path,
            )
            is not None
        )
        request_slot = self._request_slot(heartbeat=expected_long_poll)
        await request_slot.acquire()
        lease_token = activate_runtime_lease_scope(self.context.manager)

        async def send_with_request_id(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status") or 500)
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                path = str(scope.get("path") or "")
                if path == "/":
                    headers["Cache-Control"] = "no-store, max-age=0"
                    headers["Pragma"] = "no-cache"
                    headers["Expires"] = "0"
                elif path.startswith("/static/"):
                    headers["Cache-Control"] = (
                        "public, max-age=0, must-revalidate"
                    )
                    if "Pragma" in headers:
                        del headers["Pragma"]
                    if "Expires" in headers:
                        del headers["Expires"]
            await send(message)

        failed = False
        try:
            request = Request(scope, receive=receive)
            response = None
            if request_surface(request) == ADAPTER_ACCESS_SURFACE and not (
                is_adapter_access_request_allowed(
                    request.method,
                    request.url.path,
                )
            ):
                response = JSONResponse(
                    status_code=404,
                    content={"detail": "Not Found"},
                )
            if response is None:
                response = await _adapter_busy_response(request)
            if response is None:
                await self.app(scope, receive, send_with_request_id)
            else:
                await response(scope, receive, send_with_request_id)
        except BaseException:
            failed = True
            raise
        finally:
            elapsed = time.perf_counter() - started
            if path not in {"/api/v1/logs", "/api/v1/jobs"}:
                if failed or status_code >= 500:
                    log = logger.error
                elif status_code >= 400 or elapsed >= SLOW_REQUEST_SECONDS:
                    log = (
                        logger.debug
                        if expected_long_poll and status_code < 400
                        else logger.warning
                    )
                else:
                    log = logger.debug
                log(
                    "HTTP 请求完成：request_id=%s method=%s path=%s status=%s elapsed_ms=%.2f",
                    request_id,
                    method,
                    path,
                    status_code,
                    elapsed * 1000,
                )
            try:
                await release_runtime_lease_scope(lease_token)
            except Exception:
                logger.exception(
                    "HTTP 请求 runtime lease 释放失败：request_id=%s path=%s",
                    request_id,
                    scope.get("path") or "",
                )
            finally:
                request_slot.release()
                reset_context(token)


async def static_asset_cache_policy(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/":
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    elif request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "public, max-age=0, must-revalidate"
        if "Pragma" in response.headers:
            del response.headers["Pragma"]
        if "Expires" in response.headers:
            del response.headers["Expires"]
    return response


async def adapter_access_surface_guard(request: Request, call_next):
    if request_surface(request) != ADAPTER_ACCESS_SURFACE:
        return await call_next(request)
    if not is_adapter_access_request_allowed(request.method, request.url.path):
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    return await call_next(request)


async def adapter_busy_guard(request: Request, call_next):
    response = await _adapter_busy_response(request)
    if response is not None:
        return response
    return await call_next(request)


async def _adapter_busy_response(request: Request) -> JSONResponse | None:
    if not _adapter_header(request, ADAPTER_ID_HEADER):
        return None
    database_ref = _adapter_database_ref_from_path(request.url.path)
    if not database_ref:
        return None
    if not _adapter_request_authenticated(request, database_ref):
        return None
    database_id = database_ref.id
    context = current_context()
    adapter_id = _adapter_header(request, ADAPTER_ID_HEADER)
    descriptor = database_type_registry.require(database_ref.database_type).descriptor
    collection = (
        "memory-libraries" if descriptor.category == "memory" else "knowledge-libraries"
    )
    heartbeat_path = (
        f"/api/v1/{collection}/{database_ref.database_type}/"
        f"{database_id}/adapters/heartbeat"
    )
    connection = context.manager.forced_adapter_connection(
        database_ref,
        adapter_id,
    )
    if (
        connection
        and connection.get("state") == "forced_offline"
        and request.url.path.rstrip("/") != heartbeat_path
    ):
        identity_label = (
            "memory_store_id"
            if descriptor.category == DATABASE_CATEGORY_MEMORY
            else "knowledge_base_id"
        )
        logger.warning(
            "已强制下线的 Adapter 请求被拒绝：%s=%s adapter_id=%s path=%s",
            identity_label,
            database_id,
            adapter_id,
            request.url.path,
        )
        return JSONResponse(
            status_code=409,
            content={"detail": _adapter_forced_offline_detail(connection)},
        )
    if _adapter_status_request_allowed(request, database_ref):
        return None
    try:
        resource_key = database_type_registry.require(
            database_ref.database_type
        ).resource_key(database_ref.id)
        job = (
            await context.manager.jobs.active_long_job(resource_key)
            if context.manager.jobs is not None
            else await context.manager.control.active_long_job(resource_key)
        )
    except Exception:
        identity_label = (
            "memory_store_id"
            if descriptor.category == DATABASE_CATEGORY_MEMORY
            else "knowledge_base_id"
        )
        logger.exception(
            "Adapter busy guard 检查失败，放行请求：%s=%s path=%s",
            identity_label,
            database_id,
            request.url.path,
        )
        return None
    if not job:
        return None
    identity_label = (
        "memory_store_id"
        if descriptor.category == DATABASE_CATEGORY_MEMORY
        else "knowledge_base_id"
    )
    logger.warning(
        "Adapter 请求因数据库繁忙被拒绝：%s=%s adapter_id=%s path=%s job=%s",
        identity_label,
        database_id,
        _adapter_header(request, ADAPTER_ID_HEADER),
        request.url.path,
        job.get("id"),
    )
    return JSONResponse(
        status_code=409,
        content={"detail": _adapter_busy_detail(job, database_ref)},
    )
