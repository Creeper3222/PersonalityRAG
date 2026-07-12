from __future__ import annotations

import time
import uuid
import re
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
    _adapter_library_id_from_path,
    _adapter_request_authenticated,
    _adapter_status_request_allowed,
)
from .logger import logger


SLOW_REQUEST_SECONDS = 1.0


class RequestContextMiddleware:
    def __init__(self, app: Any, *, context: ApplicationContext):
        self.app = app
        self.context = context

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
        started = time.perf_counter()
        status_code = 500
        lease_token = activate_runtime_lease_scope(self.context.manager)

        async def send_with_request_id(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status") or 500)
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            elapsed = time.perf_counter() - started
            method = str(scope.get("method") or "").upper()
            path = str(scope.get("path") or "").rstrip("/")
            expected_long_poll = (
                method == "POST"
                and re.fullmatch(
                    r"/api/v1/libraries/[^/]+/adapters/heartbeat",
                    path,
                )
                is not None
            )
            if path not in {"/api/v1/logs", "/api/v1/jobs"}:
                log = (
                    logger.debug
                    if expected_long_poll or elapsed < SLOW_REQUEST_SECONDS
                    else logger.warning
                )
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


async def adapter_busy_guard(request: Request, call_next):
    if not _adapter_header(request, ADAPTER_ID_HEADER):
        return await call_next(request)
    library_id = _adapter_library_id_from_path(request.url.path)
    if not library_id:
        return await call_next(request)
    if not _adapter_request_authenticated(request, library_id):
        return await call_next(request)
    context = current_context()
    adapter_id = _adapter_header(request, ADAPTER_ID_HEADER)
    connection = context.manager.forced_adapter_connection(
        library_id,
        adapter_id,
    )
    if (
        connection
        and connection.get("state") == "forced_offline"
        and request.url.path.rstrip("/")
        != f"/api/v1/libraries/{library_id}/adapters/heartbeat"
    ):
        logger.warning(
            "已强制下线的 Adapter 请求被拒绝：library_id=%s adapter_id=%s path=%s",
            library_id,
            adapter_id,
            request.url.path,
        )
        return JSONResponse(
            status_code=409,
            content={"detail": _adapter_forced_offline_detail(connection)},
        )
    if _adapter_status_request_allowed(request, library_id):
        return await call_next(request)
    try:
        job = (
            await context.manager.jobs.active_long_job(library_id)
            if context.manager.jobs is not None
            else await context.manager.control.active_long_job(library_id)
        )
    except Exception:
        logger.exception(
            "Adapter busy guard 检查失败，放行请求：library_id=%s path=%s",
            library_id,
            request.url.path,
        )
        return await call_next(request)
    if not job:
        return await call_next(request)
    logger.warning(
        "Adapter 请求因记忆库繁忙被拒绝：library_id=%s adapter_id=%s path=%s job=%s",
        library_id,
        _adapter_header(request, ADAPTER_ID_HEADER),
        request.url.path,
        job.get("id"),
    )
    return JSONResponse(
        status_code=409,
        content={"detail": _adapter_busy_detail(job)},
    )
