from __future__ import annotations


from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)

from .. import __version__
from ..application_context import manager
from ..compat import LIVINGMEMORY_DATABASE_VERSION
from ..http_shared import (
    ADAPTER_ID_HEADER,
    require_auth,
    runtime,
)


router = APIRouter()
@router.get("/api/v1/stats", dependencies=[Depends(require_auth)])
@router.get(
    "/api/v1/libraries/{library_id}/stats",
    dependencies=[Depends(require_auth)],
)
async def stats(request: Request, library_id: str | None = None):
    if request.headers.get(ADAPTER_ID_HEADER):
        try:
            record = (
                await manager.control.get_library(library_id)
                if library_id
                else await manager.control.default_library()
            )
            if record is None:
                raise KeyError(library_id or "")
        except KeyError as exc:
            raise HTTPException(404, "memory library not found") from exc
        detail = await manager.library_detail(record.id)
        provider = await manager.control.get_provider(
            record.provider_id,
            record.provider_revision,
        )
        stats_payload = detail["stats"]
        return {
            **stats_payload,
            "service_version": __version__,
            "livingmemory_database_version": LIVINGMEMORY_DATABASE_VERSION,
            "library": record.public(),
            "provider": provider.public() if provider else None,
            "provider_status": manager.cached_provider_status(provider),
            "indexes": detail["indexes"],
            "backups": await manager.list_library_backups(record.id),
        }
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
    cached_only = bool(request.headers.get(ADAPTER_ID_HEADER))
    if not cached_only:
        busy_job = (
            await manager.jobs.active_long_job(target.library_id)
            if manager.jobs is not None
            else await manager.control.active_long_job(target.library_id)
        )
        cached_only = bool(busy_job)
    return {
        **stats_payload,
        "service_version": __version__,
        "livingmemory_database_version": LIVINGMEMORY_DATABASE_VERSION,
        "library": record.public() if record else None,
        "provider": provider.public() if provider else None,
        "provider_status": await manager.provider_status(
            target,
            allow_probe=not cached_only,
        ),
        "indexes": manager._normalize_indexes_for_response(
            stats_payload, target.indexes.status()
        ),
        "backups": await target.list_backups(),
    }

@router.get("/api/v1/integrity", dependencies=[Depends(require_auth)])
@router.get(
    "/api/v1/libraries/{library_id}/integrity",
    dependencies=[Depends(require_auth)],
)
async def integrity(library_id: str | None = None):
    return await (await runtime(library_id)).storage.integrity_report()
