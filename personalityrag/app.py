from __future__ import annotations

import json
import os
import secrets
import asyncio
import base64
import hashlib
import hmac
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Callable

from fastapi import (
    BackgroundTasks,
    Cookie,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .auth import COOKIE_NAME, AuthManager, hash_password, verify_password
from .compat import LIVINGMEMORY_DATABASE_VERSION
from .config import build_access_url, load_config, save_config
from .libraries import LibraryManager
from .logger import configure_logging, get_log_buffer, logger, safe_summary
from .migration import validate_livingmemory_db_file
from .schemas import (
    BatchDelete,
    BatchUpdate,
    ConversationMessageCreate,
    ConversationMetadataUpdate,
    ConversationTrimRequest,
    DebugProviderRevisionPatch,
    DebugProviderRevisionReset,
    GraphQuery,
    LibraryCreate,
    LibraryPskRequest,
    LibraryUpdate,
    LoginRequest,
    MemoryCreate,
    MemoryUpdate,
    MigrationRequest,
    ProviderCopy,
    ProviderCreate,
    ProviderUpdate,
    RecallRequest,
    RebuildRequest,
    SettingsUpdate,
    UiLogRequest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.json"
STATIC_DIR = ROOT / "static"
ASSETS_DIR = ROOT / "assets"
config = load_config(CONFIG_PATH)
configure_logging(
    ROOT / "data" / "logs" / "personalityrag.log",
    level_name=config.logging.level,
    file_max_bytes=config.logging.file_max_bytes,
    file_backup_count=config.logging.file_backup_count,
    web_max_entries=config.logging.web_max_entries,
    web_max_bytes=config.logging.web_max_bytes,
    web_max_entry_bytes=config.logging.web_max_entry_bytes,
)
auth = AuthManager(
    config.api_key,
    config.session_secret,
    password_hash=config.webui_password_hash,
)
manager = LibraryManager(ROOT, config)
RESTART_PROBE_SCAN_LIMIT = 80
_process_shutdown_callback: Callable[[], None] | None = None
_restart_in_progress = False


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "服务启动：host=%s webui_port=%s access_port=%s api_key_fp=%s",
        config.host,
        os.environ.get("PERSONALITYRAG_ACTUAL_PORT") or config.port,
        os.environ.get("PERSONALITYRAG_ACCESS_ACTUAL_PORT")
        or config.access_port,
        config.api_key_fingerprint,
    )
    for warning_key in (
        "PERSONALITYRAG_WEBUI_PORT_FALLBACK_WARNING",
        "PERSONALITYRAG_ACCESS_PORT_FALLBACK_WARNING",
        "PERSONALITYRAG_PORT_FALLBACK_WARNING",
    ):
        if os.environ.get(warning_key):
            logger.warning(os.environ[warning_key])
    await manager.initialize()
    logger.info("服务初始化完成：默认记忆库与模型提供商已加载")
    yield
    logger.info("服务正在关闭：释放记忆库 runtime 与模型提供商连接")
    await manager.close()
    logger.info("服务已关闭")


app = FastAPI(
    title="PersonalityRAG",
    version=__version__,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def no_cache_static_assets(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


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


def _is_library_access_port_request(request: Request) -> bool:
    if not request.path_params.get("library_id"):
        return False
    actual_access_port = int(
        os.environ.get("PERSONALITYRAG_ACCESS_ACTUAL_PORT") or config.access_port
    )
    actual_webui_port = int(os.environ.get("PERSONALITYRAG_ACTUAL_PORT") or config.port)
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
        if library_id:
            return await manager.get_runtime(library_id)
        return await manager.default_runtime()
    except KeyError as exc:
        raise HTTPException(404, "memory library not found") from exc


def jobs():
    if not manager.jobs:
        raise HTTPException(503, "job manager not ready")
    return manager.jobs


async def _enqueue_index_rebuild(library_id: str, reason: str) -> str:
    logger.warning(
        "提交写入触发的索引重建任务：library_id=%s reason=%s",
        library_id,
        reason,
    )
    job_id = await jobs().start(
        "index_rebuild",
        lambda progress: manager.rebuild_library(library_id, None, progress),
        library_id=library_id,
        dedupe_active=False,
    )
    logger.warning(
        "写入触发的索引重建任务已创建：library_id=%s job_id=%s reason=%s",
        library_id,
        job_id,
        reason,
    )
    return job_id


def set_process_shutdown_callback(callback: Callable[[], None] | None) -> None:
    global _process_shutdown_callback
    _process_shutdown_callback = callback


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
    helper_command = [
        sys.executable,
        "-m",
        "personalityrag.restart_helper",
        str(os.getpid()),
        str(ROOT),
    ]
    _start_detached_process(helper_command, ROOT)


async def _shutdown_for_restart() -> None:
    await asyncio.sleep(0.35)
    if _process_shutdown_callback is not None:
        _process_shutdown_callback()
        return
    os._exit(0)


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(ASSETS_DIR / "logo.png", media_type="image/png")


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "product": "PersonalityRAG",
        "version": __version__,
        "livingmemory_database_version": LIVINGMEMORY_DATABASE_VERSION,
        "time": time.time(),
    }


@app.get("/api/v1/auth/status")
async def auth_status(
    authorization: Annotated[str | None, Header()] = None,
    session: Annotated[str | None, Cookie(alias=COOKIE_NAME)] = None,
):
    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()
    return {
        "authenticated": bool(
            auth.verify_api_key(bearer) or auth.verify_session(session)
        ),
        "login_password_enabled": auth.password_enabled,
        "login_mode": "password" if auth.password_enabled else "api_key",
    }


@app.post("/api/v1/auth/login")
async def login(payload: LoginRequest, response: Response):
    credential = payload.credential or payload.api_key
    if not auth.verify_login_secret(credential):
        detail = "invalid password" if auth.password_enabled else "invalid API key"
        raise HTTPException(status_code=401, detail=detail)
    response.set_cookie(
        COOKIE_NAME,
        auth.issue_session(),
        httponly=True,
        samesite="strict",
        secure=False,
        max_age=auth.session_ttl_seconds,
        path="/",
    )
    return {"status": "ok", "fingerprint": config.api_key_fingerprint}


@app.post("/api/v1/auth/logout")
async def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"status": "ok"}


@app.post("/api/v1/auth/rotate", dependencies=[Depends(require_auth)])
async def rotate_key(response: Response):
    config.api_key = f"prag_{secrets.token_urlsafe(32)}"
    config.session_secret = secrets.token_urlsafe(48)
    save_config(CONFIG_PATH, config)
    auth.api_key = config.api_key
    auth.session_secret = config.session_secret
    response.delete_cookie(COOKIE_NAME, path="/")
    logger.warning("API Key 已轮换：new_fingerprint=%s", config.api_key_fingerprint)
    return {"api_key": config.api_key}


# ------------------------------ Settings ------------------------------


def _settings_payload() -> dict[str, Any]:
    actual_port = int(os.environ.get("PERSONALITYRAG_ACTUAL_PORT") or config.port)
    actual_access_port = int(
        os.environ.get("PERSONALITYRAG_ACCESS_ACTUAL_PORT")
        or config.access_port
    )
    configured_webui_url = build_access_url(config.access_base_url, config.port)
    configured_api_access_url = build_access_url(
        config.access_base_url, config.access_port
    )
    webui_url = build_access_url(config.access_base_url, actual_port)
    api_access_url = build_access_url(config.access_base_url, actual_access_port)
    return {
        "host": config.host,
        "access_base_url": config.access_base_url,
        "configured_port": config.port,
        "configured_webui_url": configured_webui_url,
        "actual_port": actual_port,
        "configured_access_port": config.access_port,
        "configured_api_access_url": configured_api_access_url,
        "actual_access_port": actual_access_port,
        "access_url": api_access_url,
        "webui_url": webui_url,
        "api_access_url": api_access_url,
        "port_fallback_active": actual_port != config.port,
        "access_port_fallback_active": actual_access_port != config.access_port,
        "login_password_enabled": auth.password_enabled,
        "login_mode": "password" if auth.password_enabled else "api_key",
        "api_key_fingerprint": config.api_key_fingerprint,
        "port_change_requires_restart": True,
        "access_port_change_requires_restart": True,
    }


@app.get("/api/v1/settings", dependencies=[Depends(require_auth)])
async def get_settings():
    return _settings_payload()


@app.patch("/api/v1/settings", dependencies=[Depends(require_auth)])
async def update_settings(payload: SettingsUpdate):
    changed: list[str] = []
    next_webui_port = payload.port if payload.port is not None else config.port
    next_access_port = (
        payload.access_port
        if payload.access_port is not None
        else config.access_port
    )
    if next_webui_port == next_access_port:
        raise HTTPException(400, "记忆库接入端口不能与 WebUI 端口相同")
    if (
        payload.access_base_url is not None
        and payload.access_base_url != config.access_base_url
    ):
        config.access_base_url = payload.access_base_url
        changed.append("access_base_url")
    if payload.port is not None and payload.port != config.port:
        config.port = payload.port
        changed.append("port")
    if (
        payload.access_port is not None
        and payload.access_port != config.access_port
    ):
        config.access_port = payload.access_port
        changed.append("access_port")
    if payload.clear_password:
        config.webui_password_hash = ""
        auth.password_hash = ""
        changed.append("password_cleared")
    elif payload.new_password is not None and payload.new_password.strip():
        password = payload.new_password.strip()
        if len(password) < 6:
            raise HTTPException(400, "登录密码至少需要 6 个字符")
        config.webui_password_hash = hash_password(password)
        auth.password_hash = config.webui_password_hash
        changed.append("password_set")
    if changed:
        save_config(CONFIG_PATH, config)
        logger.warning(
            "基础设置已更新：fields=%s restart_required=%s",
            changed,
            bool({"port", "access_port"} & set(changed)),
        )
    return {**_settings_payload(), "changed": changed}


@app.post("/api/v1/settings/restart", dependencies=[Depends(require_auth)])
async def restart_service(background_tasks: BackgroundTasks):
    global _restart_in_progress
    if _restart_in_progress:
        raise HTTPException(409, "restart already in progress")
    try:
        _launch_restart_helper()
    except Exception as exc:
        logger.exception("restart helper launch failed")
        raise HTTPException(500, "restart helper launch failed") from exc
    _restart_in_progress = True
    logger.warning("WebUI requested a PersonalityRAG restart: pid=%s", os.getpid())
    background_tasks.add_task(_shutdown_for_restart)
    return {
        **_settings_payload(),
        "restart_in_progress": True,
        "restart_probe_urls": _restart_probe_urls(),
        "restart_requested_at": time.time(),
    }


# ------------------------------ Logs ------------------------------


@app.get("/api/v1/logs", dependencies=[Depends(require_auth)])
async def logs(
    after_id: int = Query(default=0, ge=0),
    wait: float = Query(default=0.0, ge=0.0, le=30.0),
):
    buffer = get_log_buffer()
    latest_id = buffer.latest_id()
    reset_cursor = after_id > latest_id
    effective_after_id = 0 if reset_cursor else after_id
    initial_version = buffer.version
    if wait > 0 and not reset_cursor and effective_after_id >= latest_id:
        await buffer.wait_for_change(initial_version, timeout=wait)
        latest_id = buffer.latest_id()
        reset_cursor = after_id > latest_id
        effective_after_id = 0 if reset_cursor else after_id
    return {
        "items": buffer.get_entries(after_id=effective_after_id),
        "max_entries": buffer.max_entries,
        "buffer": buffer.summary(),
        "latest_id": latest_id,
        "reset": reset_cursor,
    }


@app.post("/api/v1/logs/clear", dependencies=[Depends(require_auth)])
async def clear_logs():
    buffer = get_log_buffer()
    cleared = buffer.clear()
    logger.info("WebUI 实时日志缓存已清空：cleared=%s", cleared)
    return {"ok": True, "cleared": cleared, "max_entries": buffer.max_entries}


@app.post("/api/v1/logs/ui", dependencies=[Depends(require_auth)])
async def ui_log(payload: UiLogRequest):
    level = payload.level.upper()
    message = safe_summary(payload.message, max_chars=512)
    context = {
        key: safe_summary(value, max_chars=160)
        for key, value in (payload.context or {}).items()
    }
    text = "WebUI 操作反馈：%s" % message
    if context:
        text += " context=%s" % context
    if level == "ERROR":
        logger.error(text)
    elif level == "WARN":
        logger.warning(text)
    elif level == "DEBUG":
        logger.debug(text)
    else:
        logger.info(text)
    return {"ok": True}


# ------------------------------ Libraries ------------------------------


@app.get("/api/v1/libraries", dependencies=[Depends(require_auth)])
async def libraries():
    return {"items": await manager.list_libraries()}


@app.post("/api/v1/libraries", dependencies=[Depends(require_auth)])
async def create_library(payload: LibraryCreate):
    logger.info("创建记忆库：library_id=%s name=%s provider=%s", payload.id, payload.name, payload.provider_id)
    try:
        result = await manager.create_library(payload.model_dump())
        logger.info("记忆库创建完成：library_id=%s", result.get("id"))
        return result
    except ValueError as exc:
        logger.warning("记忆库创建失败：library_id=%s err=%s", payload.id, exc)
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/v1/libraries/{library_id}/psk", dependencies=[Depends(require_auth)])
async def library_psk(library_id: str, payload: LibraryPskRequest):
    record = await manager.control.get_library(library_id)
    if not record:
        raise HTTPException(404, "memory library not found")
    if not auth.password_enabled:
        raise HTTPException(
            400,
            "请先在基础设置中设置 WebUI 登录密码，再查看记忆库接入密钥",
        )
    if not verify_password(payload.password, auth.password_hash):
        logger.warning("记忆库 PSK 查看鉴权失败：library_id=%s", library_id)
        raise HTTPException(401, "invalid password")
    logger.info("记忆库 PSK 已生成：library_id=%s", library_id)
    return {
        "library_id": library_id,
        "psk": _derive_library_psk(library_id),
    }


@app.get("/api/v1/libraries/{library_id}", dependencies=[Depends(require_auth)])
async def library_detail(library_id: str):
    try:
        return await manager.library_detail(library_id)
    except KeyError as exc:
        raise HTTPException(404, "memory library not found") from exc


@app.patch("/api/v1/libraries/{library_id}", dependencies=[Depends(require_auth)])
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
        raise HTTPException(400, str(exc)) from exc
    except KeyError as exc:
        logger.warning("记忆库更新失败：library_id=%s err=not_found", library_id)
        raise HTTPException(404, "memory library not found") from exc


@app.delete("/api/v1/libraries/{library_id}", dependencies=[Depends(require_auth)])
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


@app.post(
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


@app.post(
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


@app.post(
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


# ------------------------------ Conversations ------------------------------


@app.post(
    "/api/v1/libraries/{library_id}/conversations/messages",
    dependencies=[Depends(require_auth)],
)
async def append_conversation_message(
    library_id: str, payload: ConversationMessageCreate
):
    target = await runtime(library_id)
    result = await target.storage.add_conversation_message(payload.model_dump())
    logger.info(
        "会话消息写入：library_id=%s session=%s role=%s duplicate=%s sender=%s",
        target.library_id,
        payload.session_id,
        payload.role,
        result.get("duplicate"),
        payload.sender_id or "",
    )
    return result


@app.get(
    "/api/v1/libraries/{library_id}/conversations/{session_id}",
    dependencies=[Depends(require_auth)],
)
async def get_conversation(
    library_id: str,
    session_id: str,
    limit: int = Query(default=50, ge=0, le=500),
):
    target = await runtime(library_id)
    result = await target.storage.get_conversation(session_id, limit=limit)
    if result is None:
        raise HTTPException(404, "conversation not found")
    return result


@app.get(
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


@app.patch(
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


@app.post(
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


@app.post(
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


# ------------------------------ Providers ------------------------------


@app.get("/api/v1/provider-types", dependencies=[Depends(require_auth)])
async def provider_types(kind: str | None = Query(default=None, pattern="^(embedding|rerank)$")):
    return {"items": manager.control.provider_types(kind)}


@app.get("/api/v1/providers", dependencies=[Depends(require_auth)])
async def providers(kind: str | None = Query(default=None, pattern="^(embedding|rerank)$")):
    return {"items": await manager.control.list_providers(kind)}


@app.post("/api/v1/providers", dependencies=[Depends(require_auth)])
async def create_provider(payload: ProviderCreate):
    logger.info("创建模型提供商：provider_id=%s type=%s model=%s", payload.id, payload.type, payload.model)
    try:
        result = await manager.create_provider(payload.model_dump())
        logger.info("模型提供商创建完成：provider_id=%s revision=%s", result.get("id"), result.get("revision"))
        return result
    except ValueError as exc:
        logger.warning("模型提供商创建失败：provider_id=%s err=%s", payload.id, exc)
        raise HTTPException(400, str(exc)) from exc


@app.patch(
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


@app.delete(
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


@app.post(
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


@app.post(
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


@app.post("/api/v1/providers/test-draft", dependencies=[Depends(require_auth)])
async def test_provider_draft(payload: ProviderCreate):
    logger.info("测试草稿模型提供商：provider_id=%s type=%s model=%s", payload.id, payload.type, payload.model)
    result = await manager.test_provider_draft(payload.model_dump())
    logger.info("草稿模型提供商连接成功：provider_id=%s dimension=%s", payload.id, result.get("dimension") or result.get("dimensions"))
    return result


@app.post(
    "/api/v1/providers/detect-dimension",
    dependencies=[Depends(require_auth)],
)
async def detect_provider_dimension(payload: ProviderCreate):
    logger.info("自动检测模型维度：provider_id=%s type=%s model=%s", payload.id, payload.type, payload.model)
    result = await manager.detect_dimension(payload.model_dump())
    logger.info("模型维度检测完成：provider_id=%s dimensions=%s", payload.id, result.get("dimensions"))
    return result


@app.post(
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


@app.post(
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


# 旧单 Provider 测试入口继续测试默认记忆库当前绑定的 revision。
@app.post("/api/v1/providers/test", dependencies=[Depends(require_auth)])
async def provider_test_compat():
    library = await manager.control.default_library()
    logger.info("测试默认记忆库当前模型提供商：library_id=%s provider=%s revision=%s", library.id, library.provider_id, library.provider_revision)
    return await manager.test_provider(
        library.provider_id, revision=library.provider_revision
    )


@app.get(
    "/api/v1/debug/providers/{provider_id}/revisions",
    dependencies=[Depends(require_revision_debug_access)],
)
async def debug_provider_revisions(provider_id: str):
    try:
        return await manager.control.debug_provider_revisions(provider_id)
    except KeyError as exc:
        raise HTTPException(404, "provider not found") from exc


@app.patch(
    "/api/v1/debug/providers/{provider_id}/revisions/{revision}",
    dependencies=[Depends(require_revision_debug_access)],
)
async def debug_patch_provider_revision(
    provider_id: str,
    revision: int,
    payload: DebugProviderRevisionPatch,
):
    logger.warning(
        "debug provider revision patch requested: provider_id=%s revision=%s fields=%s",
        provider_id,
        revision,
        sorted(payload.patch.keys()),
    )
    try:
        return await manager.control.debug_patch_provider_revision(
            provider_id,
            revision,
            payload.patch,
        )
    except KeyError as exc:
        raise HTTPException(404, "provider or revision not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post(
    "/api/v1/debug/providers/{provider_id}/revisions/reset",
    dependencies=[Depends(require_revision_debug_access)],
)
async def debug_reset_provider_revisions(
    provider_id: str,
    payload: DebugProviderRevisionReset,
):
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
        return await manager.control.debug_reset_provider_revisions(
            provider_id,
            latest_revision=payload.latest_revision,
            bind_libraries_to_latest=payload.bind_libraries_to_latest,
            library_revisions=payload.library_revisions,
            delete_revisions_after_latest=payload.delete_revisions_after_latest,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# ------------------------------ Memories ------------------------------


def _normalize_memory_update_payload(payload: MemoryUpdate) -> dict[str, Any]:
    updates = payload.model_dump(exclude_none=True)
    scale = str(updates.pop("value_scale", "auto") or "auto").lower()
    if "importance" not in updates:
        return updates
    try:
        value = float(updates["importance"])
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "invalid importance") from exc
    if scale in {"display", "0-10", "ten"}:
        if not 0 <= value <= 10:
            raise HTTPException(400, "importance display value must be between 0 and 10")
        updates["importance"] = value / 10
    elif scale in {"stored", "normalized", "0-1"}:
        if not 0 <= value <= 1:
            raise HTTPException(400, "importance stored value must be between 0 and 1")
        updates["importance"] = value
    elif scale == "auto":
        if 0 <= value <= 1:
            updates["importance"] = value
        elif 0 <= value <= 10:
            updates["importance"] = value / 10
        else:
            raise HTTPException(400, "importance value must be between 0 and 10")
    else:
        raise HTTPException(400, "invalid importance value_scale")
    return updates


@app.get("/api/v1/memories", dependencies=[Depends(require_auth)])
@app.get(
    "/api/v1/libraries/{library_id}/memories",
    dependencies=[Depends(require_auth)],
)
async def memories(
    library_id: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=500),
    keyword: str = "",
    session_id: str | None = None,
    persona_id: str | None = None,
    status: str | None = None,
    memory_type: str | None = None,
    sort: str = "created_desc",
):
    target = await runtime(library_id)
    logger.debug(
        "查询记忆列表：library_id=%s page=%s page_size=%s keyword=%s persona=%s session=%s status=%s sort=%s",
        target.library_id,
        page,
        page_size,
        safe_summary(keyword),
        persona_id or "",
        session_id or "",
        status or "",
        sort,
    )
    return await target.storage.list_documents(
        page=page,
        page_size=page_size,
        keyword=keyword,
        session_id=session_id,
        persona_id=persona_id,
        status=status,
        memory_type=memory_type,
        sort=sort,
    )


@app.post("/api/v1/memories", dependencies=[Depends(require_auth)])
@app.post(
    "/api/v1/libraries/{library_id}/memories",
    dependencies=[Depends(require_auth)],
)
async def create_memory(payload: MemoryCreate, library_id: str | None = None):
    target = await runtime(library_id)
    logger.info(
        "写入记忆请求：library_id=%s persona=%s session=%s importance=%s content_summary=%s",
        target.library_id,
        payload.persona_id or "",
        payload.session_id or "",
        payload.importance,
        safe_summary(payload.content),
    )
    started = time.perf_counter()
    result = await target.create_memory(payload.model_dump(), rebuild=False)
    job_id = await _enqueue_index_rebuild(target.library_id, "memory_create")
    result["job_id"] = job_id
    logger.info(
        "写入记忆完成：library_id=%s memory_id=%s job_id=%s elapsed_ms=%.2f",
        target.library_id,
        result.get("id"),
        job_id,
        (time.perf_counter() - started) * 1000,
    )
    return result


@app.post("/api/v1/memories/batch-delete", dependencies=[Depends(require_auth)])
@app.post(
    "/api/v1/libraries/{library_id}/memories/batch-delete",
    dependencies=[Depends(require_auth)],
)
async def batch_delete(payload: BatchDelete, library_id: str | None = None):
    target = await runtime(library_id)
    logger.warning(
        "批量删除记忆请求：library_id=%s count=%s ids=%s",
        target.library_id,
        len(payload.memory_ids),
        payload.memory_ids[:20],
    )
    deleted = await target.delete_memories(payload.memory_ids, rebuild=False)
    job_id = None
    if deleted:
        job_id = await _enqueue_index_rebuild(target.library_id, "memory_batch_delete")
    logger.warning(
        "批量删除记忆完成：library_id=%s deleted=%s job_id=%s",
        target.library_id,
        deleted,
        job_id or "",
    )
    return {"deleted": deleted, "job_id": job_id}


@app.post("/api/v1/memories/batch-update", dependencies=[Depends(require_auth)])
@app.post(
    "/api/v1/libraries/{library_id}/memories/batch-update",
    dependencies=[Depends(require_auth)],
)
async def batch_update(payload: BatchUpdate, library_id: str | None = None):
    target = await runtime(library_id)
    logger.info(
        "批量更新记忆请求：library_id=%s count=%s fields=%s",
        target.library_id,
        len(payload.memory_ids),
        sorted(payload.updates.model_dump(exclude_none=True).keys()),
    )
    updates = _normalize_memory_update_payload(payload.updates)
    updated = []
    for memory_id in payload.memory_ids:
        result = await target.update_memory(memory_id, updates, rebuild=False)
        if result:
            updated.append(memory_id)
    job_id = None
    if updated:
        job_id = await _enqueue_index_rebuild(target.library_id, "memory_batch_update")
    logger.info(
        "批量更新记忆完成：library_id=%s count=%s job_id=%s",
        target.library_id,
        len(updated),
        job_id or "",
    )
    return {"updated": updated, "count": len(updated), "job_id": job_id}


@app.get(
    "/api/v1/memories/{memory_id}", dependencies=[Depends(require_auth)]
)
@app.get(
    "/api/v1/libraries/{library_id}/memories/{memory_id}",
    dependencies=[Depends(require_auth)],
)
async def memory_detail(memory_id: int, library_id: str | None = None):
    target = await runtime(library_id)
    memory = await target.storage.get_document(memory_id)
    if not memory:
        raise HTTPException(404, "memory not found")
    memory["graph_context"] = await target.graph_snapshot(
        memory_ids=[memory_id]
    )
    return memory


@app.patch(
    "/api/v1/memories/{memory_id}", dependencies=[Depends(require_auth)]
)
@app.patch(
    "/api/v1/libraries/{library_id}/memories/{memory_id}",
    dependencies=[Depends(require_auth)],
)
async def update_memory(
    memory_id: int, payload: MemoryUpdate, library_id: str | None = None
):
    target = await runtime(library_id)
    logger.info(
        "更新记忆请求：library_id=%s memory_id=%s fields=%s",
        target.library_id,
        memory_id,
        sorted(payload.model_dump(exclude_none=True).keys()),
    )
    result = await target.update_memory(
        memory_id, _normalize_memory_update_payload(payload), rebuild=False
    )
    if not result:
        logger.warning("更新记忆失败：library_id=%s memory_id=%s err=not_found", target.library_id, memory_id)
        raise HTTPException(404, "memory not found")
    job_id = await _enqueue_index_rebuild(target.library_id, "memory_update")
    result["job_id"] = job_id
    logger.info(
        "更新记忆完成：library_id=%s memory_id=%s job_id=%s",
        target.library_id,
        memory_id,
        job_id,
    )
    return result


@app.delete(
    "/api/v1/memories/{memory_id}", dependencies=[Depends(require_auth)]
)
@app.delete(
    "/api/v1/libraries/{library_id}/memories/{memory_id}",
    dependencies=[Depends(require_auth)],
)
async def delete_memory(memory_id: int, library_id: str | None = None):
    target = await runtime(library_id)
    logger.warning("删除记忆请求：library_id=%s memory_id=%s", target.library_id, memory_id)
    deleted = await target.delete_memories([memory_id], rebuild=False)
    job_id = None
    if deleted:
        job_id = await _enqueue_index_rebuild(target.library_id, "memory_delete")
    logger.warning(
        "删除记忆完成：library_id=%s memory_id=%s deleted=%s job_id=%s",
        target.library_id,
        memory_id,
        deleted,
        job_id or "",
    )
    return {"deleted": deleted, "job_id": job_id}


# ------------------------------ Recall / graph ------------------------------


@app.post("/api/v1/recall", dependencies=[Depends(require_auth)])
@app.post(
    "/api/v1/libraries/{library_id}/recall",
    dependencies=[Depends(require_auth)],
)
async def recall(payload: RecallRequest, library_id: str | None = None):
    target = await runtime(library_id)
    started = time.perf_counter()
    session_filter = payload.session_id if config.recall.use_session_filtering else None
    persona_filter = payload.persona_id if config.recall.use_persona_filtering else None
    generation = (target.indexes.status() or {}).get("generation")
    embedding_k = int(payload.k)
    rerank_k = int(payload.rerank_k or payload.k)
    use_rerank = bool(target.reranker) and (
        True if payload.rerank is None else bool(payload.rerank)
    )
    if use_rerank and rerank_k > embedding_k:
        raise HTTPException(400, "重排输出条数不能大于嵌入召回条数")
    logger.info(
        "开始召回：library_id=%s embedding_k=%s rerank_k=%s persona=%s session=%s generation=%s query=%s",
        target.library_id,
        embedding_k,
        rerank_k,
        persona_filter or "",
        session_filter or "",
        generation or "",
        safe_summary(payload.query),
    )
    candidate_k = embedding_k
    candidates = await target.retrieval.search(
        payload.query,
        candidate_k,
        session_filter,
        persona_filter,
    )
    baseline_results = candidates[:embedding_k]
    results = candidates[:embedding_k]
    rerank_meta: dict[str, Any] = {
        "requested": use_rerank,
        "applied": False,
        "available": bool(target.reranker),
        "embedding_k": embedding_k,
        "rerank_k": rerank_k,
        "provider_id": (
            target.rerank_provider_revision.provider_id
            if target.rerank_provider_revision
            else ""
        ),
        "provider_type": (
            target.rerank_provider_revision.config.type
            if target.rerank_provider_revision
            else ""
        ),
    }
    if use_rerank:
        if target.reranker:
            try:
                results, rerank_meta = await target.apply_rerank(
                    payload.query,
                    candidates,
                    rerank_k,
                )
                rerank_meta["embedding_k"] = embedding_k
                rerank_meta["rerank_k"] = rerank_k
            except Exception as exc:
                logger.warning(
                    "Rerank 失败，回退未重排结果：library_id=%s provider=%s err=%s",
                    target.library_id,
                    rerank_meta.get("provider_id") or "",
                    safe_summary(exc),
                )
                rerank_meta.update(
                    {
                        "requested": True,
                        "applied": False,
                        "failed": True,
                        "error": str(exc),
                        "candidate_count": len(candidates),
                        "embedding_k": embedding_k,
                        "rerank_k": rerank_k,
                    }
                )
                results = baseline_results[:rerank_k]
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "召回完成：library_id=%s total=%s elapsed_ms=%.2f top_ids=%s",
        target.library_id,
        len(results),
        elapsed_ms,
        [item.doc_id for item in results[:10]],
    )
    response = {
        "query": payload.query,
        "library_id": target.library_id,
        "results": [item.to_dict() for item in results],
        "total": len(results),
        "embedding_k": embedding_k,
        "rerank_k": rerank_k,
        "elapsed_time_ms": elapsed_ms,
        "rerank": rerank_meta,
    }
    if payload.include_baseline:
        response["baseline_results"] = [
            item.to_dict() for item in baseline_results
        ]
    if use_rerank:
        response["rerank_candidates"] = [item.to_dict() for item in candidates]
    return response


@app.get("/api/v1/graph/overview", dependencies=[Depends(require_auth)])
@app.get(
    "/api/v1/libraries/{library_id}/graph/overview",
    dependencies=[Depends(require_auth)],
)
async def graph_overview(
    library_id: str | None = None,
    session_id: str | None = None,
    persona_id: str | None = None,
):
    target = await runtime(library_id)
    return {
        "snapshot": await target.graph_snapshot(
            session_id=session_id, persona_id=persona_id
        ),
        "stats": await target.storage.statistics(),
    }


@app.post("/api/v1/graph/query", dependencies=[Depends(require_auth)])
@app.post(
    "/api/v1/libraries/{library_id}/graph/query",
    dependencies=[Depends(require_auth)],
)
async def graph_query(payload: GraphQuery, library_id: str | None = None):
    target = await runtime(library_id)
    logger.debug(
        "图谱查询：library_id=%s memory_id=%s limit=%s query=%s",
        target.library_id,
        payload.memory_id or "",
        payload.limit_memories,
        safe_summary(payload.query),
    )
    memory_ids = [payload.memory_id] if payload.memory_id else None
    retrieval = []
    if payload.query and not memory_ids:
        results = await target.retrieval.search(
            payload.query,
            payload.limit_memories,
            payload.session_id,
            payload.persona_id,
        )
        memory_ids = [item.doc_id for item in results]
        retrieval = [item.to_dict() for item in results]
    snapshot = await target.graph_snapshot(
        memory_ids=memory_ids,
        session_id=payload.session_id,
        persona_id=payload.persona_id,
        limit_memories=payload.limit_memories,
    )
    return {"snapshot": snapshot, "retrieval": retrieval}


# ------------------------------ Indexes / jobs / migration ------------------------------


@app.get("/api/v1/indexes", dependencies=[Depends(require_auth)])
@app.get(
    "/api/v1/libraries/{library_id}/indexes",
    dependencies=[Depends(require_auth)],
)
async def index_status(library_id: str | None = None):
    target = await runtime(library_id)
    stats = await target.storage.statistics()
    return manager._normalize_indexes_for_response(
        stats, target.indexes.status()
    )


@app.post("/api/v1/indexes/rebuild", dependencies=[Depends(require_auth)])
@app.post(
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


@app.get("/api/v1/jobs", dependencies=[Depends(require_auth)])
async def list_jobs(scope: str = Query("active", pattern="^(active|finished|all)$")):
    return {"items": await jobs().list(scope=scope)}


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_auth)])
async def job_status(job_id: str):
    result = await jobs().get(job_id)
    if not result:
        raise HTTPException(404, "job not found")
    return result


@app.get("/api/v1/jobs/{job_id}/events", dependencies=[Depends(require_auth)])
async def job_events(job_id: str):
    async def stream():
        async for payload in jobs().subscribe(job_id):
            yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post(
    "/api/v1/libraries/{library_id}/imports/livingmemory-db",
    dependencies=[Depends(require_auth)],
)
async def import_livingmemory_db_file(
    library_id: str,
    file: UploadFile = File(...),
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
        raise HTTPException(400, "请上传名为 livingmemory.db 的 LivingMemory 核心数据库文件")
    upload_dir = ROOT / "data" / "import_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / f"{library_id}-{int(time.time())}-{secrets.token_hex(4)}.db"
    try:
        with upload_path.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                handle.write(chunk)
        await file.close()
        try:
            await asyncio.to_thread(validate_livingmemory_db_file, upload_path)
        except Exception as exc:
            upload_path.unlink(missing_ok=True)
            logger.warning(
                "LivingMemory 单文件导入校验失败：library_id=%s filename=%s err=%s",
                library_id,
                filename,
                exc,
            )
            raise HTTPException(400, str(exc)) from exc
        logger.warning(
            "提交 LivingMemory 单文件导入任务：library_id=%s upload=%s",
            library_id,
            upload_path,
        )
        job_id = await jobs().start(
            "livingmemory_import",
            lambda progress: manager.import_livingmemory_db(
                library_id, upload_path, progress
            ),
            library_id=library_id,
        )
        logger.warning("LivingMemory 单文件导入任务已创建：library_id=%s job_id=%s", library_id, job_id)
        return {"job_id": job_id}
    except HTTPException:
        raise
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        logger.exception("LivingMemory 单文件导入请求失败：library_id=%s", library_id)
        raise HTTPException(500, str(exc)) from exc


@app.post(
    "/api/v1/migration/livingmemory",
    dependencies=[Depends(require_auth)],
)
@app.post(
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


@app.get("/api/v1/stats", dependencies=[Depends(require_auth)])
@app.get(
    "/api/v1/libraries/{library_id}/stats",
    dependencies=[Depends(require_auth)],
)
async def stats(library_id: str | None = None):
    target = await runtime(library_id)
    record = await manager.control.get_library(target.library_id)
    provider = (
        await manager.control.get_provider(
            record.provider_id, record.provider_revision
        )
        if record
        else None
    )
    stats_payload = await target.storage.statistics()
    return {
        **stats_payload,
        "service_version": __version__,
        "livingmemory_database_version": LIVINGMEMORY_DATABASE_VERSION,
        "library": record.public() if record else None,
        "provider": provider.public() if provider else None,
        "provider_status": await target.provider.test_connection(),
        "indexes": manager._normalize_indexes_for_response(
            stats_payload, target.indexes.status()
        ),
        "backups": await target.list_backups(),
    }


@app.get("/api/v1/integrity", dependencies=[Depends(require_auth)])
@app.get(
    "/api/v1/libraries/{library_id}/integrity",
    dependencies=[Depends(require_auth)],
)
async def integrity(library_id: str | None = None):
    return await (await runtime(library_id)).storage.integrity_report()
