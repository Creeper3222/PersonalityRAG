from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import os
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
from .config import build_access_url


ADAPTER_ID_HEADER = "x-personalityrag-adapter-id"
ADAPTER_INSTANCE_HEADER = "x-personalityrag-adapter-instance-id"
ADAPTER_TYPE_HEADER = "x-personalityrag-adapter-type"
RESTART_PROBE_SCAN_LIMIT = 80


def _adapter_header(request: Request, name: str) -> str:
    return str(request.headers.get(name) or "").strip()


def _adapter_library_id_from_path(path: str) -> str | None:
    prefix = "/api/v1/libraries/"
    if not path.startswith(prefix):
        return None
    tail = path[len(prefix) :].strip("/")
    if not tail:
        return None
    return tail.split("/", 1)[0]


def _adapter_job_summary(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if not job:
        return None
    return {
        "id": job.get("id"),
        "library_id": job.get("library_id"),
        "kind": job.get("kind"),
        "status": job.get("status"),
        "progress": job.get("progress"),
        "message": job.get("message"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
    }


def _adapter_busy_detail(job: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "code": "library_busy",
        "message": "记忆库繁忙，请求暂停",
        "job": _adapter_job_summary(job),
    }


def _adapter_forced_offline_detail(
    connection: dict[str, Any] | None,
) -> dict[str, Any]:
    connection = connection or {}
    return {
        "code": "adapter_forced_offline",
        "message": "连接被强制切断",
        "library_id": connection.get("library_id"),
        "adapter_id": connection.get("adapter_id"),
        "disconnected_at": connection.get("disconnected_at"),
        "reason": connection.get("disconnect_reason") or "forced_by_admin",
    }


def _adapter_status_request_allowed(request: Request, library_id: str) -> bool:
    path = request.url.path.rstrip("/")
    method = request.method.upper()
    base = f"/api/v1/libraries/{library_id}"
    if method == "GET" and path in {base, f"{base}/stats", f"{base}/indexes"}:
        return True
    if method == "POST" and path == f"{base}/adapters/heartbeat":
        return True
    return False


def _derive_library_psk(library_id: str) -> str:
    digest = hmac.new(
        config.library_psk_secret.encode("utf-8"),
        library_id.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return "psk-" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _verify_library_psk(library_id: str | None, candidate: str | None) -> bool:
    if not library_id or not candidate or not candidate.startswith("psk-"):
        return False
    expected = _derive_library_psk(library_id)
    return hmac.compare_digest(candidate, expected)


def _request_port(request: Request) -> int | None:
    if request.url.port:
        return request.url.port
    server = request.scope.get("server")
    if isinstance(server, tuple) and len(server) >= 2:
        try:
            return int(server[1])
        except (TypeError, ValueError):
            return None
    return None


def _adapter_request_authenticated(request: Request, library_id: str) -> bool:
    authorization = request.headers.get("authorization") or ""
    bearer = (
        authorization[7:].strip()
        if authorization.lower().startswith("bearer ")
        else None
    )
    actual_access_port = int(
        os.environ.get("PERSONALITYRAG_ACCESS_ACTUAL_PORT") or config.access_port
    )
    actual_webui_port = int(
        os.environ.get("PERSONALITYRAG_ACTUAL_PORT") or config.port
    )
    if (
        _request_port(request) == actual_access_port
        and actual_access_port != actual_webui_port
    ):
        return _verify_library_psk(library_id, bearer)
    return bool(
        auth.verify_api_key(bearer)
        or auth.verify_session(request.cookies.get(COOKIE_NAME))
        or _verify_library_psk(library_id, bearer)
    )


def _is_library_access_port_request(request: Request) -> bool:
    if not request.path_params.get("library_id"):
        return False
    actual_access_port = int(
        os.environ.get("PERSONALITYRAG_ACCESS_ACTUAL_PORT") or config.access_port
    )
    actual_webui_port = int(
        os.environ.get("PERSONALITYRAG_ACTUAL_PORT") or config.port
    )
    port = _request_port(request)
    return port == actual_access_port and actual_access_port != actual_webui_port


async def require_auth(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    session: Annotated[str | None, Cookie(alias=COOKIE_NAME)] = None,
) -> None:
    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()
    library_id = request.path_params.get("library_id")
    if _is_library_access_port_request(request):
        if _verify_library_psk(library_id, bearer):
            return
        raise HTTPException(status_code=401, detail="library psk required")
    if auth.verify_api_key(bearer) or auth.verify_session(session):
        return
    if _verify_library_psk(library_id, bearer):
        return
    raise HTTPException(status_code=401, detail="authentication required")


async def require_admin_auth(
    authorization: Annotated[str | None, Header()] = None,
    session: Annotated[str | None, Cookie(alias=COOKIE_NAME)] = None,
) -> None:
    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()
    if auth.verify_api_key(bearer) or auth.verify_session(session):
        return
    raise HTTPException(status_code=401, detail="administrator authentication required")


def _debug_revision_api_enabled() -> bool:
    return os.environ.get("PERSONALITYRAG_DEBUG_REVISION_API", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_loopback_client(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host == "localhost" or host == "::1" or host.startswith("127.")


async def require_revision_debug_access(
    request: Request,
    _: Annotated[None, Depends(require_auth)],
) -> None:
    if not _debug_revision_api_enabled() or not _is_loopback_client(request):
        raise HTTPException(status_code=404, detail="not found")


async def runtime(library_id: str | None):
    try:
        target_library_id = library_id
        if not target_library_id:
            target_library_id = (await manager.control.default_library()).id
        scope = current_runtime_lease_scope()
        if scope is not None:
            return await scope.acquire(target_library_id)
        return await manager.get_runtime(target_library_id)
    except KeyError as exc:
        raise HTTPException(404, "memory library not found") from exc


def jobs():
    if not manager.jobs:
        raise HTTPException(503, "job manager not ready")
    return manager.jobs


async def _resolve_memory_library_id(library_id: str | None) -> str:
    try:
        if library_id:
            record = await manager.control.get_library(library_id)
            if not record:
                raise KeyError(library_id)
            return record.id
        return (await manager.control.default_library()).id
    except KeyError as exc:
        raise HTTPException(404, "memory library not found") from exc


async def _run_memory_job(
    kind: str,
    library_id: str,
    operation: Callable[[Callable[[float, str], Any]], Any],
) -> dict[str, Any]:
    job_id = await jobs().start(
        kind,
        operation,
        library_id=library_id,
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
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
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
