from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any, Callable

from fastapi import Cookie, Depends, Header, HTTPException, Request

from .application_context import (
    auth,
    config,
    current_context,
    current_runtime_lease_scope,
    manager,
)
from .auth import COOKIE_NAME
from .config import build_access_url, build_adapter_connection_url
from .database_types import (
    LIVINGMEMORY_V8_TYPE,
    TEXT_MEDIA_V1_TYPE,
    DatabaseRef,
    database_identity_fields,
    database_type_registry,
)
from . import library_types as _registered_database_types  # noqa: F401
from .listener_surface import (
    ADAPTER_ACCESS_SURFACE,
    WEBUI_SURFACE,
    request_surface,
)
from .revision_debug import (
    DEBUG_COOKIE_NAME,
    admin_credential_subject,
    is_loopback_client,
)


ADAPTER_ID_HEADER = "x-personalityrag-adapter-id"
ADAPTER_INSTANCE_HEADER = "x-personalityrag-adapter-instance-id"
ADAPTER_TYPE_HEADER = "x-personalityrag-adapter-type"
RESTART_PROBE_SCAN_LIMIT = 80


def _adapter_header(request: Request, name: str) -> str:
    return str(request.headers.get(name) or "").strip()


def _current_adapter_connection_url() -> str:
    actual_access_port = int(
        os.environ.get("PERSONALITYRAG_ACCESS_ACTUAL_PORT") or config.access_port
    )
    return build_adapter_connection_url(
        config,
        access_port=actual_access_port,
    )


def _adapter_database_ref_from_path(path: str) -> DatabaseRef | None:
    prefixes = (
        ("/api/v1/memory-libraries/livingmemory_v8/", LIVINGMEMORY_V8_TYPE),
        ("/api/v1/knowledge-libraries/text_media_v1/", TEXT_MEDIA_V1_TYPE),
    )
    for prefix, database_type in prefixes:
        if not path.startswith(prefix):
            continue
        tail = path[len(prefix) :].strip("/")
        database_id = tail.split("/", 1)[0]
        if not database_id or database_id in {"imports"}:
            return None
        try:
            return DatabaseRef(database_type, database_id)
        except ValueError:
            return None
    return None


def _adapter_job_summary(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if not job:
        return None
    database_id = str(job.get("database_id") or job.get("library_id") or "")
    database_type = str(job.get("database_type") or LIVINGMEMORY_V8_TYPE)
    identity = (
        database_identity_fields(
            DatabaseRef(database_type, database_id),
            include_deprecated=False,
        )
        if database_id
        else {}
    )
    return {
        **identity,
        "id": job.get("id"),
        # Frozen job-table compatibility field. It may contain a resource key
        # for knowledge-base jobs rather than the public database ID.
        "library_id": job.get("library_id"),
        "kind": job.get("kind"),
        "status": job.get("status"),
        "progress": job.get("progress"),
        "message": job.get("message"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
    }


def _adapter_busy_detail(
    job: dict[str, Any] | None,
    database: DatabaseRef | None = None,
) -> dict[str, Any]:
    return {
        "code": "library_busy",
        "condition": "database_busy",
        **(
            database_identity_fields(database, include_deprecated=True)
            if database is not None
            else {}
        ),
        "message": "数据库繁忙，请求暂停",
        "job": _adapter_job_summary(job),
    }


def _adapter_forced_offline_detail(
    connection: dict[str, Any] | None,
) -> dict[str, Any]:
    connection = connection or {}
    database_id = str(
        connection.get("memory_store_id")
        or connection.get("knowledge_base_id")
        or connection.get("database_id")
        or connection.get("library_id")
        or ""
    )
    database_type = str(
        connection.get("memory_store_type")
        or connection.get("knowledge_base_type")
        or connection.get("database_type")
        or LIVINGMEMORY_V8_TYPE
    )
    identity = (
        database_identity_fields(
            DatabaseRef(database_type, database_id),
            include_deprecated=True,
        )
        if database_id
        else {}
    )
    return {
        "code": "adapter_forced_offline",
        "message": "连接被强制切断",
        **identity,
        "adapter_id": connection.get("adapter_id"),
        "disconnected_at": connection.get("disconnected_at"),
        "reason": connection.get("disconnect_reason") or "forced_by_admin",
    }


def _adapter_status_request_allowed(request: Request, ref: DatabaseRef) -> bool:
    path = request.url.path.rstrip("/")
    method = request.method.upper()
    descriptor = database_type_registry.require(ref.database_type).descriptor
    collection = (
        "memory-libraries" if descriptor.category == "memory" else "knowledge-libraries"
    )
    base = f"/api/v1/{collection}/{ref.database_type}/{ref.id}"
    status_paths = {base}
    if descriptor.category == "memory":
        status_paths.update({f"{base}/stats", f"{base}/indexes"})
    if method == "GET" and path in status_paths:
        return True
    if method == "POST" and path == f"{base}/adapters/heartbeat":
        return True
    media_path = re.fullmatch(
        rf"{re.escape(base)}/assets/[A-Za-z0-9_-]+/(content|thumbnail|signed-url)",
        path,
    )
    if media_path:
        operation = media_path.group(1)
        if method == "GET" and operation in {"content", "thumbnail"}:
            return True
        if method == "POST" and operation == "signed-url":
            return True
    return False


def _derive_database_access_key(database_type: str, database_id: str) -> str:
    driver = database_type_registry.require(database_type)
    return driver.derive_access_key(config.library_psk_secret, database_id)


def _verify_database_access_key(
    database_type: str | None,
    database_id: str | None,
    candidate: str | None,
) -> bool:
    if not database_type or not database_id:
        return False
    try:
        driver = database_type_registry.require(database_type)
    except KeyError:
        return False
    return driver.verify_access_key(
        config.library_psk_secret,
        database_id,
        candidate,
    )


def _derive_memory_store_psk(memory_store_id: str) -> str:
    return _derive_database_access_key(LIVINGMEMORY_V8_TYPE, memory_store_id)


def _verify_memory_store_psk(
    memory_store_id: str | None,
    candidate: str | None,
) -> bool:
    return _verify_database_access_key(
        LIVINGMEMORY_V8_TYPE,
        memory_store_id,
        candidate,
    )


# Deprecated compatibility aliases for v0.1.1 imports.
_derive_library_psk = _derive_memory_store_psk
_verify_library_psk = _verify_memory_store_psk


def _adapter_request_authenticated(request: Request, ref: DatabaseRef) -> bool:
    authorization = request.headers.get("authorization") or ""
    bearer = (
        authorization[7:].strip()
        if authorization.lower().startswith("bearer ")
        else None
    )
    if request_surface(request) == ADAPTER_ACCESS_SURFACE:
        return _verify_database_access_key(ref.database_type, ref.id, bearer)
    return bool(
        auth.verify_api_key(bearer)
        or auth.verify_session(request.cookies.get(COOKIE_NAME))
        or _verify_database_access_key(ref.database_type, ref.id, bearer)
    )


def _is_database_access_port_request(request: Request) -> bool:
    return bool(
        request_surface(request) == ADAPTER_ACCESS_SURFACE
        and _adapter_database_ref_from_path(request.url.path) is not None
    )


async def require_auth(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    session: Annotated[str | None, Cookie(alias=COOKIE_NAME)] = None,
) -> None:
    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()
    database_ref = _adapter_database_ref_from_path(request.url.path)
    database_id = database_ref.id if database_ref else None
    database_type = database_ref.database_type if database_ref else None
    if request_surface(request) == ADAPTER_ACCESS_SURFACE:
        if database_ref is None:
            raise HTTPException(status_code=404, detail="Not Found")
        if _verify_database_access_key(database_type, database_id, bearer):
            return
        raise HTTPException(status_code=401, detail="database access key required")
    if auth.verify_api_key(bearer) or auth.verify_session(session):
        return
    if _verify_database_access_key(database_type, database_id, bearer):
        return
    raise HTTPException(status_code=401, detail="authentication required")


async def require_admin_auth(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    session: Annotated[str | None, Cookie(alias=COOKIE_NAME)] = None,
) -> None:
    if request_surface(request) == ADAPTER_ACCESS_SURFACE:
        raise HTTPException(status_code=404, detail="Not Found")
    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()
    if auth.verify_api_key(bearer) or auth.verify_session(session):
        return
    raise HTTPException(status_code=401, detail="administrator authentication required")


def livingmemory_v8_database_type() -> str:
    return LIVINGMEMORY_V8_TYPE


LivingMemoryV8Type = Annotated[str, Depends(livingmemory_v8_database_type)]


def text_media_v1_database_type() -> str:
    return TEXT_MEDIA_V1_TYPE


TextMediaV1Type = Annotated[str, Depends(text_media_v1_database_type)]


async def require_revision_debug_access(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    session: Annotated[str | None, Cookie(alias=COOKIE_NAME)] = None,
    debug_session: Annotated[
        str | None,
        Cookie(alias=DEBUG_COOKIE_NAME),
    ] = None,
) -> None:
    if request_surface(request) != WEBUI_SURFACE or not is_loopback_client(request):
        raise HTTPException(status_code=404, detail="Not Found")
    await require_admin_auth(request, authorization, session)
    if not admin_credential_subject(request, auth):
        raise HTTPException(status_code=401, detail="administrator authentication required")
    status = current_context().revision_debug.status(debug_session, request, auth)
    if not status["unlocked"]:
        raise HTTPException(status_code=404, detail="Not Found")


async def require_revision_debug_session_control(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    session: Annotated[str | None, Cookie(alias=COOKIE_NAME)] = None,
) -> None:
    if request_surface(request) != WEBUI_SURFACE or not is_loopback_client(request):
        raise HTTPException(status_code=404, detail="Not Found")
    await require_admin_auth(request, authorization, session)


def require_database_ref(
    database_type: str,
    database_id: str,
    *,
    capability: str | None = None,
) -> DatabaseRef:
    try:
        ref = DatabaseRef(database_type, database_id)
        driver = database_type_registry.require(database_type)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "database type not found") from exc
    if capability and capability not in driver.descriptor.capabilities:
        raise HTTPException(409, f"数据库类型不支持该操作: {capability}")
    return ref


async def require_existing_database_ref(
    database_type: str,
    database_id: str,
    *,
    capability: str | None = None,
) -> DatabaseRef:
    ref = require_database_ref(
        database_type,
        database_id,
        capability=capability,
    )
    identity = await manager.control.database_identity(ref)
    if identity is None or identity.get("deleted_at") is not None:
        raise HTTPException(404, "database not found")
    if ref.database_type == LIVINGMEMORY_V8_TYPE:
        record = await manager.control.get_library(ref.id)
        if record is None or record.database_type != ref.database_type:
            raise HTTPException(404, "database not found")
    return ref


async def runtime(database: str | DatabaseRef | None):
    try:
        target_database = database
        if not target_database:
            record = await manager.control.default_library()
            target_database = DatabaseRef(record.database_type, record.id)
        scope = current_runtime_lease_scope()
        if scope is not None:
            return await scope.acquire(target_database)
        return await manager.get_runtime(target_database)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "database not found") from exc


def jobs():
    if not manager.jobs:
        raise HTTPException(503, "job manager not ready")
    return manager.jobs


async def _resolve_memory_store_id(
    database: str | DatabaseRef | None,
) -> str:
    try:
        if database:
            ref = (
                database
                if isinstance(database, DatabaseRef)
                else DatabaseRef(LIVINGMEMORY_V8_TYPE, database)
            )
            if ref.database_type != LIVINGMEMORY_V8_TYPE:
                raise KeyError(ref.key)
            record = await manager.control.get_library(ref.id)
            if not record:
                raise KeyError(ref.key)
            return record.id
        return (await manager.control.default_library()).id
    except KeyError as exc:
        raise HTTPException(404, "memory library not found") from exc


async def _run_memory_store_job(
    kind: str,
    memory_store_id: str,
    operation: Callable[[Callable[[float, str], Any]], Any],
) -> dict[str, Any]:
    job_id = await jobs().start(
        kind,
        operation,
        database_id=memory_store_id,
        dedupe_active=False,
    )
    job = await jobs().wait(job_id)
    if job.get("status") != "completed":
        raise HTTPException(
            409 if job.get("status") == "failed" else 500,
            {
                "job_id": job_id,
                "code": "memory_job_failed",
                "message": job.get("error")
                or job.get("message")
                or "memory job failed",
            },
        )
    result = job.get("result") or {}
    if isinstance(result, dict):
        result.setdefault("job_id", job_id)
        return result
    return {"job_id": job_id, "result": result}


# Deprecated helper aliases for extensions that import the old names.
_resolve_memory_library_id = _resolve_memory_store_id
_run_memory_job = _run_memory_store_job


def set_process_shutdown_callback(callback: Callable[[], None] | None) -> None:
    current_context().process_shutdown_callback = callback


def _restart_probe_urls() -> list[str]:
    return [
        build_access_url(config.access_base_url, port)
        for port in range(config.port, config.port + RESTART_PROBE_SCAN_LIMIT + 1)
    ]


def _start_detached_process(command: list[str], cwd: Path) -> None:
    kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(command, **kwargs)


def _launch_restart_helper() -> None:
    context = current_context()
    helper_command = [
        sys.executable,
        "-m",
        "personalityrag.restart_helper",
        str(os.getpid()),
        str(context.source_root),
    ]
    _start_detached_process(helper_command, context.source_root)


async def _shutdown_for_restart() -> None:
    await asyncio.sleep(0.35)
    callback = current_context().process_shutdown_callback
    if callback is not None:
        callback()
        return
    os._exit(0)
