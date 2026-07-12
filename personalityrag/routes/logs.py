from __future__ import annotations


from fastapi import (
    APIRouter,
    Depends,
    Query,
)

from ..http_shared import (
    require_auth,
)
from ..logger import get_log_buffer, logger, safe_summary
from ..schemas import (
    UiLogRequest,
)


router = APIRouter()
@router.get("/api/v1/logs", dependencies=[Depends(require_auth)])
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

@router.post("/api/v1/logs/clear", dependencies=[Depends(require_auth)])
async def clear_logs():
    buffer = get_log_buffer()
    cleared = buffer.clear()
    logger.info("WebUI 实时日志缓存已清空：cleared=%s", cleared)
    return {"ok": True, "cleared": cleared, "max_entries": buffer.max_entries}

@router.post("/api/v1/logs/ui", dependencies=[Depends(require_auth)])
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
