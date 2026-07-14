from __future__ import annotations

from collections.abc import Awaitable, Callable

import aiosqlite

from ..task_types import ADAPTER_BUSY_JOB_KINDS


Connect = Callable[[], Awaitable[aiosqlite.Connection]]


class _Repository:
    def __init__(self, connect: Connect):
        self.connect = connect


class ProviderRepository(_Repository):
    async def get_row(
        self, provider_id: str, revision: int | None = None
    ) -> aiosqlite.Row | None:
        db = await self.connect()
        try:
            if revision is None:
                return await (
                    await db.execute(
                        """SELECT pr.* FROM providers p JOIN provider_revisions pr
                        ON pr.provider_id=p.id AND pr.revision=p.latest_revision
                        WHERE p.id=? AND p.deleted_at IS NULL""",
                        (provider_id,),
                    )
                ).fetchone()
            return await (
                await db.execute(
                    """SELECT pr.* FROM providers p JOIN provider_revisions pr
                    ON pr.provider_id=p.id
                    WHERE p.id=? AND pr.revision=? AND p.deleted_at IS NULL""",
                    (provider_id, revision),
                )
            ).fetchone()
        finally:
            await db.close()

    async def rows_for_bindings(
        self, bindings: set[tuple[str, int | None]]
    ) -> list[aiosqlite.Row]:
        provider_ids = sorted({provider_id for provider_id, _ in bindings})
        if not provider_ids:
            return []
        placeholders = ",".join("?" for _ in provider_ids)
        db = await self.connect()
        try:
            return await (
                await db.execute(
                    f"""SELECT pr.*,p.latest_revision
                    FROM providers p JOIN provider_revisions pr
                    ON pr.provider_id=p.id
                    WHERE p.id IN ({placeholders}) AND p.deleted_at IS NULL""",
                    tuple(provider_ids),
                )
            ).fetchall()
        finally:
            await db.close()


class LibraryRepository(_Repository):
    async def get_row(self, library_id: str) -> aiosqlite.Row | None:
        db = await self.connect()
        try:
            return await (
                await db.execute(
                    "SELECT * FROM libraries WHERE id=? AND deleted_at IS NULL",
                    (library_id,),
                )
            ).fetchone()
        finally:
            await db.close()

    async def list_rows(self) -> list[aiosqlite.Row]:
        db = await self.connect()
        try:
            return await (
                await db.execute(
                    """SELECT * FROM libraries WHERE deleted_at IS NULL
                    ORDER BY is_default DESC,created_at"""
                )
            ).fetchall()
        finally:
            await db.close()


class AdapterRepository(_Repository):
    async def active_rows(
        self,
        library_ids: list[str],
        *,
        cutoff: float,
    ) -> list[aiosqlite.Row]:
        if not library_ids:
            return []
        placeholders = ",".join("?" for _ in library_ids)
        db = await self.connect()
        try:
            return await (
                await db.execute(
                    f"""SELECT * FROM adapter_connections
                    WHERE library_id IN ({placeholders})
                    AND state='active' AND last_seen>=?
                    ORDER BY last_seen DESC,adapter_id""",
                    (*library_ids, cutoff),
                )
            ).fetchall()
        finally:
            await db.close()


class JobRepository(_Repository):
    async def active_long_rows(
        self, library_ids: list[str]
    ) -> list[aiosqlite.Row]:
        if not library_ids:
            return []
        library_placeholders = ",".join("?" for _ in library_ids)
        kind_placeholders = ",".join("?" for _ in ADAPTER_BUSY_JOB_KINDS)
        db = await self.connect()
        try:
            return await (
                await db.execute(
                    f"""SELECT * FROM jobs
                    WHERE library_id IN ({library_placeholders})
                    AND kind IN ({kind_placeholders})
                    AND status IN ('queued','running','pausing','paused','interrupted','stopping')
                    ORDER BY created_at""",
                    (*library_ids, *ADAPTER_BUSY_JOB_KINDS),
                )
            ).fetchall()
        finally:
            await db.close()


class SnapshotRepository(_Repository):
    async def provider_rows(self) -> tuple[list[aiosqlite.Row], list[aiosqlite.Row]]:
        db = await self.connect()
        try:
            providers = await (
                await db.execute(
                    """SELECT id,latest_revision,created_at,updated_at
                    FROM providers WHERE deleted_at IS NULL ORDER BY created_at"""
                )
            ).fetchall()
            provider_ids = [str(row["id"]) for row in providers]
            revisions: list[aiosqlite.Row] = []
            if provider_ids:
                placeholders = ",".join("?" for _ in provider_ids)
                revisions = await (
                    await db.execute(
                        f"""SELECT provider_id,revision,config_json,created_at
                        FROM provider_revisions
                        WHERE provider_id IN ({placeholders})
                        ORDER BY provider_id,revision""",
                        tuple(provider_ids),
                    )
                ).fetchall()
            return providers, revisions
        finally:
            await db.close()

    async def library_rows(self) -> list[aiosqlite.Row]:
        db = await self.connect()
        try:
            return await (
                await db.execute(
                    """SELECT * FROM libraries WHERE deleted_at IS NULL
                    ORDER BY is_default DESC,created_at"""
                )
            ).fetchall()
        finally:
            await db.close()
