from __future__ import annotations

import os

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..application_context import current_context
from ..http_shared import _restart_probe_urls, _shutdown_for_restart, require_auth
from ..logger import logger
from ..task_types import ACTIVE_JOB_STATUSES
from ..update_manifest import UpdatePackageError


router = APIRouter()


class VersionSwitchRequest(BaseModel):
    tag_name: str = Field(min_length=2, max_length=64)


def _public_release(item: dict) -> dict:
    asset = item.get("asset") or {}
    return {
        key: item.get(key)
        for key in (
            "tag_name",
            "version",
            "name",
            "published_at",
            "prerelease",
            "notes",
            "html_url",
        )
    } | {
        "asset": {
            "name": asset.get("name"),
            "size": asset.get("size"),
            "digest": asset.get("digest"),
        }
    }


@router.get("/api/v1/updates/status", dependencies=[Depends(require_auth)])
async def update_status(refresh: bool = Query(False)):
    return await current_context().updates.status(refresh=refresh)


@router.get("/api/v1/updates/releases", dependencies=[Depends(require_auth)])
async def update_releases(refresh: bool = Query(False)):
    payload = await current_context().updates.releases(refresh=refresh)
    return {
        **{key: value for key, value in payload.items() if key != "releases"},
        "releases": [_public_release(item) for item in payload["releases"]],
    }


@router.get(
    "/api/v1/updates/transactions/{transaction_id}",
    dependencies=[Depends(require_auth)],
)
async def update_transaction(transaction_id: str):
    payload = current_context().updates.transaction(transaction_id)
    if payload is None:
        raise HTTPException(404, "update transaction not found")
    return payload


@router.post("/api/v1/updates/switch", dependencies=[Depends(require_auth)])
async def switch_version(payload: VersionSwitchRequest, background_tasks: BackgroundTasks):
    context = current_context()
    if context.restart_in_progress:
        raise HTTPException(409, "restart or version switch already in progress")
    if context.manager.jobs is None:
        raise HTTPException(503, "job manager not ready")
    active_jobs = await context.manager.jobs.list(scope="active")
    blocking = [job for job in active_jobs if job.get("status") in ACTIVE_JOB_STATUSES]
    if blocking:
        raise HTTPException(
            409,
            {
                "code": "active_jobs_block_update",
                "message": "finish, stop, or cancel all active jobs before switching versions",
                "job_ids": [job.get("id") for job in blocking],
            },
        )
    try:
        transaction = await context.updates.prepare_switch(
            tag_name=payload.tag_name,
            service_pid=os.getpid(),
            health_urls=_restart_probe_urls(),
        )
    except UpdatePackageError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        logger.exception("version switch preparation failed: tag=%s", payload.tag_name)
        raise HTTPException(503, "version switch preparation failed") from exc
    context.restart_in_progress = True
    logger.warning(
        "version switch prepared: transaction_id=%s current=%s target=%s action=%s",
        transaction.get("transaction_id"),
        transaction.get("current_version"),
        transaction.get("target_version"),
        transaction.get("action"),
    )
    background_tasks.add_task(_shutdown_for_restart)
    return {
        **transaction,
        "restart_in_progress": True,
        "restart_probe_urls": _restart_probe_urls(),
    }
