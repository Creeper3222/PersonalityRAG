from __future__ import annotations

import asyncio

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)

from .. import __version__
from ..application_context import manager
from ..http_shared import (
    ADAPTER_ID_HEADER,
    LivingMemoryV8Type,
    require_existing_database_ref,
    require_auth,
    runtime,
)


router = APIRouter(prefix="/api/v1/memory-libraries/livingmemory_v8")
@router.get(
    "/{memory_store_id}/stats",
    dependencies=[Depends(require_auth)],
)
async def stats(
    request: Request,
    database_type: LivingMemoryV8Type,
    memory_store_id: str,
):
    ref = await require_existing_database_ref(database_type, memory_store_id)
    if request.headers.get(ADAPTER_ID_HEADER):
        try:
            record = (
                await manager.control.get_library(memory_store_id)
                if memory_store_id
                else await manager.control.default_library()
            )
            if record is None:
                raise KeyError(memory_store_id or "")
        except KeyError as exc:
            raise HTTPException(404, "memory library not found") from exc
        detail = await manager.library_detail(ref)
        provider = await manager.control.get_provider(
            record.provider_id,
            record.provider_revision,
        )
        stats_payload = detail["stats"]
        memory_store = record.public()
        return {
            **stats_payload,
            "service_version": __version__,
            "memory_store": memory_store,
            "library": memory_store,
            "provider": provider.public() if provider else None,
            "provider_status": manager.cached_provider_status(provider),
            "indexes": detail["indexes"],
            "backups": await manager.list_library_backups(ref),
        }
    target = await runtime(ref)
    record, provider, stats_payload, busy_job, backups = await asyncio.gather(
        manager.control.get_library(memory_store_id),
        manager.control.get_provider(
            target.provider_revision.provider_id,
            target.provider_revision.revision,
        ),
        target.storage.statistics(),
        (
            manager.jobs.active_long_job(memory_store_id)
            if manager.jobs is not None
            else manager.control.active_long_job(memory_store_id)
        ),
        target.list_backups(),
    )
    cached_only = bool(request.headers.get(ADAPTER_ID_HEADER))
    if not cached_only:
        cached_only = bool(busy_job)
    memory_store = record.public() if record else None
    return {
        **stats_payload,
        "service_version": __version__,
        "memory_store": memory_store,
        "library": memory_store,
        "provider": provider.public() if provider else None,
        "provider_status": await manager.provider_status(
            target,
            allow_probe=not cached_only,
        ),
        "indexes": manager._normalize_indexes_for_response(
            stats_payload, target.indexes.status()
        ),
        "backups": backups,
    }

@router.get(
    "/{memory_store_id}/integrity",
    dependencies=[Depends(require_auth)],
)
async def integrity(database_type: LivingMemoryV8Type, memory_store_id: str):
    ref = await require_existing_database_ref(database_type, memory_store_id)
    return await (await runtime(ref)).storage.integrity_report()
