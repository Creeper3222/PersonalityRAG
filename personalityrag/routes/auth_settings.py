from __future__ import annotations

import asyncio
import os
import secrets
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Cookie,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse

from .. import __version__
from ..application_context import auth, config, current_context, manager
from ..auth import COOKIE_NAME, hash_password
from ..revision_debug import DEBUG_COOKIE_NAME
from ..backup_migration import (
    MAX_PACKAGE_BYTES,
    PragPackageError,
    export_prag_package,
    import_prag_package,
)
from ..config import (
    build_access_url,
    build_adapter_connection_url,
    deployment_mode,
    is_docker_deployment,
    save_config,
)
from ..http_shared import (
    _launch_restart_helper,
    _restart_probe_urls,
    _shutdown_for_restart,
    require_auth,
)
from ..io_utils import UploadSizeLimitError, run_blocking, save_upload_file
from ..logger import logger
from ..http_pool import http_pool_status
from ..resource_limits import (
    effective_performance_summary,
    effective_runtime_capacity,
    effective_runtime_idle_minutes,
)
from ..schemas import (
    BackupMigrationExportRequest,
    LoginRequest,
    SettingsUpdate,
)


router = APIRouter()
@router.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(current_context().static_dir / "index.html")

@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(
        current_context().assets_dir / "logo.png",
        media_type="image/png",
    )

@router.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "product": "PersonalityRAG",
        "version": __version__,
        "time": time.time(),
    }

@router.get("/api/v1/auth/status")
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
        "version": __version__,
    }

@router.post("/api/v1/auth/login")
async def login(payload: LoginRequest, response: Response, request: Request):
    credential = payload.credential or payload.api_key
    if not await asyncio.to_thread(auth.verify_login_secret, credential):
        detail = "invalid password" if auth.password_enabled else "invalid API key"
        raise HTTPException(status_code=401, detail=detail)
    current_context().revision_debug.revoke(
        request.cookies.get(DEBUG_COOKIE_NAME),
        auth,
    )
    response.delete_cookie(DEBUG_COOKIE_NAME, path="/")
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

@router.post("/api/v1/auth/logout")
async def logout(response: Response, request: Request):
    current_context().revision_debug.revoke(
        request.cookies.get(DEBUG_COOKIE_NAME),
        auth,
    )
    response.delete_cookie(COOKIE_NAME, path="/")
    response.delete_cookie(DEBUG_COOKIE_NAME, path="/")
    return {"status": "ok"}

@router.post("/api/v1/auth/rotate", dependencies=[Depends(require_auth)])
async def rotate_key(response: Response):
    context = current_context()
    config.api_key = f"prag_{secrets.token_urlsafe(32)}"
    config.session_secret = secrets.token_urlsafe(48)
    save_config(context.config_path, context.config)
    auth.api_key = config.api_key
    auth.session_secret = config.session_secret
    current_context().revision_debug.revoke_all()
    response.delete_cookie(COOKIE_NAME, path="/")
    response.delete_cookie(DEBUG_COOKIE_NAME, path="/")
    logger.warning("API Key 已轮换：new_fingerprint=%s", config.api_key_fingerprint)
    return {"api_key": config.api_key}

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
    recommended_adapter_url = build_adapter_connection_url(
        config,
        access_port=actual_access_port,
    )
    residency_status = manager.runtime_residency_status()
    effective_performance = effective_performance_summary(
        config.performance_profile
    )
    effective_performance.update(
        {
            "runtime_idle_minutes": effective_runtime_idle_minutes(
                config.runtime_residency.idle_minutes,
                config.performance_profile,
            ),
            "runtime_max_non_default": effective_runtime_capacity(
                config.runtime_residency.max_non_default_runtimes,
                config.performance_profile,
            ),
            "loaded_runtime_count": int(
                residency_status.get("loaded_count") or 0
            ),
        }
    )
    effective_performance.update(http_pool_status())
    return {
        "deployment_mode": deployment_mode(),
        "managed_settings": (
            ["port", "access_port"] if is_docker_deployment() else []
        ),
        "host": config.host,
        "access_base_url": config.access_base_url,
        "public_adapter_url": config.public_adapter_url,
        "recommended_adapter_url": recommended_adapter_url,
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
        "version": __version__,
        "performance_profile": config.performance_profile,
        "effective_performance": effective_performance,
        "runtime_residency": {
            "idle_minutes": config.runtime_residency.idle_minutes,
            "max_non_default_runtimes": (
                config.runtime_residency.max_non_default_runtimes
            ),
        },
    }

@router.get("/api/v1/settings", dependencies=[Depends(require_auth)])
async def get_settings():
    return _settings_payload()

@router.patch("/api/v1/settings", dependencies=[Depends(require_auth)])
async def update_settings(payload: SettingsUpdate, response: Response):
    changed: list[str] = []
    if is_docker_deployment():
        attempted_managed_changes = [
            field_name
            for field_name, requested, current in (
                ("port", payload.port, config.port),
                ("access_port", payload.access_port, config.access_port),
            )
            if requested is not None and requested != current
        ]
        if attempted_managed_changes:
            raise HTTPException(
                409,
                {
                    "code": "docker_managed_settings",
                    "message": "Docker deployment ports are managed by Compose",
                    "fields": attempted_managed_changes,
                },
            )
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
    if (
        payload.public_adapter_url is not None
        and payload.public_adapter_url != config.public_adapter_url
    ):
        config.public_adapter_url = payload.public_adapter_url
        changed.append("public_adapter_url")
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
        config.webui_password_hash = await asyncio.to_thread(
            hash_password, password
        )
        auth.password_hash = config.webui_password_hash
        changed.append("password_set")
    residency_changed = False
    if (
        payload.performance_profile is not None
        and payload.performance_profile != config.performance_profile
    ):
        config.performance_profile = payload.performance_profile
        changed.append("performance_profile")
        residency_changed = True
    if (
        payload.runtime_idle_minutes is not None
        and payload.runtime_idle_minutes != config.runtime_residency.idle_minutes
    ):
        config.runtime_residency.idle_minutes = payload.runtime_idle_minutes
        changed.append("runtime_residency.idle_minutes")
        residency_changed = True
    if (
        payload.max_non_default_runtimes is not None
        and payload.max_non_default_runtimes
        != config.runtime_residency.max_non_default_runtimes
    ):
        config.runtime_residency.max_non_default_runtimes = (
            payload.max_non_default_runtimes
        )
        changed.append("runtime_residency.max_non_default_runtimes")
        residency_changed = True
    if residency_changed:
        await manager.apply_runtime_residency()
    if changed:
        context = current_context()
        save_config(context.config_path, context.config)
        if "performance_profile" in changed:
            from ..faiss_runtime import configure_loaded_faiss_threads
            from ..resource_quotas import reset_resource_quotas

            reset_resource_quotas()
            configure_loaded_faiss_threads(
                config.performance_profile
            )
        logger.warning(
            "基础设置已更新：fields=%s restart_required=%s",
            changed,
            bool({"port", "access_port"} & set(changed)),
        )
    if {"password_cleared", "password_set"} & set(changed):
        current_context().revision_debug.revoke_all()
        response.delete_cookie(DEBUG_COOKIE_NAME, path="/")
    return {**_settings_payload(), "changed": changed}

@router.post("/api/v1/settings/restart", dependencies=[Depends(require_auth)])
async def restart_service(background_tasks: BackgroundTasks):
    context = current_context()
    if context.restart_in_progress:
        raise HTTPException(409, "restart already in progress")
    restart_strategy = "container" if is_docker_deployment() else "process"
    if not is_docker_deployment():
        try:
            _launch_restart_helper()
        except Exception as exc:
            logger.exception("restart helper launch failed")
            raise HTTPException(500, "restart helper launch failed") from exc
    context.restart_in_progress = True
    logger.warning("WebUI requested a PersonalityRAG restart: pid=%s", os.getpid())
    background_tasks.add_task(_shutdown_for_restart)
    return {
        **_settings_payload(),
        "restart_in_progress": True,
        "restart_strategy": restart_strategy,
        "restart_probe_urls": _restart_probe_urls(),
        "restart_requested_at": time.time(),
    }

@router.post(
    "/api/v1/settings/backup-migration/export",
    dependencies=[Depends(require_auth)],
)
async def export_backup_migration(
    payload: BackupMigrationExportRequest,
    background_tasks: BackgroundTasks,
):
    if not (payload.password or "").strip():
        raise HTTPException(400, "请输入配置验证密码")
    context = current_context()
    export_dir = context.state_root / "data" / "backup_migration_exports"
    filename = f"personalityrag-{time.strftime('%Y%m%d-%H%M%S')}.prag"
    target = export_dir / filename
    try:
        await export_prag_package(
            root=context.state_root,
            config=context.config,
            manager=context.manager,
            target=target,
            password=payload.password,
            include_libraries=payload.include_libraries,
            include_providers=payload.include_providers,
        )
    except PragPackageError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("备份迁移配置包导出失败")
        raise HTTPException(500, "备份迁移配置包导出失败") from exc
    background_tasks.add_task(target.unlink, missing_ok=True)
    return FileResponse(
        target,
        media_type="application/octet-stream",
        filename=filename,
        background=background_tasks,
    )

@router.post(
    "/api/v1/settings/backup-migration/import",
    dependencies=[Depends(require_auth)],
)
async def import_backup_migration(
    response: Response,
    file: UploadFile = File(...),
    password: str = Form(...),
):
    context = current_context()
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() != ".prag":
        await file.close()
        raise HTTPException(400, "请选择 .prag 配置包")
    if not str(password or "").strip():
        await file.close()
        raise HTTPException(400, "请输入配置验证密码")
    upload_dir = context.state_root / "data" / "import_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = (
        upload_dir
        / f"backup-migration-{int(time.time())}-{secrets.token_hex(4)}.prag"
    )
    try:
        try:
            await save_upload_file(
                file,
                upload_path,
                max_bytes=MAX_PACKAGE_BYTES,
            )
        except UploadSizeLimitError as exc:
            raise PragPackageError("配置包过大") from exc
        next_config, result = await import_prag_package(
            root=context.state_root,
            config_path=context.config_path,
            config=context.config,
            manager=context.manager,
            package_path=upload_path,
            password=password,
            preserve_managed_network=is_docker_deployment(),
        )
        context.config = next_config
        auth.api_key = config.api_key
        auth.session_secret = config.session_secret
        auth.password_hash = config.webui_password_hash
        context.revision_debug.revoke_all()
        context.manager.config = context.config
        response.set_cookie(
            COOKIE_NAME,
            auth.issue_session(),
            httponly=True,
            samesite="strict",
            secure=False,
            max_age=auth.session_ttl_seconds,
            path="/",
        )
        response.delete_cookie(DEBUG_COOKIE_NAME, path="/")
        return result
    except PragPackageError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("备份迁移配置包导入失败")
        raise HTTPException(500, "备份迁移配置包导入失败") from exc
    finally:
        await file.close()
        await run_blocking(upload_path.unlink, missing_ok=True)
