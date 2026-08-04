from __future__ import annotations

from collections.abc import Awaitable, Callable

import aiosqlite

from ..database_types import DatabaseRef


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


class MemoryStoreRepository(_Repository):
    """Repository for the frozen ``libraries`` compatibility table."""

    async def get_row(self, memory_store_id: str) -> aiosqlite.Row | None:
        db = await self.connect()
        try:
            return await (
                await db.execute(
                    "SELECT * FROM libraries WHERE id=? AND deleted_at IS NULL",
                    (memory_store_id,),
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


# Deprecated import alias. New runtime code uses MemoryStoreRepository while
# the SQL table name remains frozen for v0.1.1 compatibility.
LibraryRepository = MemoryStoreRepository


class AdapterRepository(_Repository):
    async def active_database_rows(
        self,
        databases: list[DatabaseRef],
        *,
        cutoff: float,
    ) -> list[aiosqlite.Row]:
        if not databases:
            return []
        clauses = " OR ".join("(database_type=? AND database_id=?)" for _ in databases)
        params = [part for ref in databases for part in (ref.database_type, ref.id)]
        db = await self.connect()
        try:
            return await (
                await db.execute(
                    f"""SELECT * FROM database_adapter_connections
                    WHERE ({clauses}) AND state='active' AND last_seen>=?
                    ORDER BY last_seen DESC,adapter_id""",
                    (*params, cutoff),
                )
            ).fetchall()
        finally:
            await db.close()

    async def active_rows(
        self,
        memory_store_ids: list[str],
        *,
        cutoff: float,
    ) -> list[aiosqlite.Row]:
        if not memory_store_ids:
            return []
        placeholders = ",".join("?" for _ in memory_store_ids)
        db = await self.connect()
        try:
            return await (
                await db.execute(
                    f"""SELECT * FROM adapter_connections
                    WHERE library_id IN ({placeholders})
                    AND state='active' AND last_seen>=?
                    ORDER BY last_seen DESC,adapter_id""",
                    (*memory_store_ids, cutoff),
                )
            ).fetchall()
        finally:
            await db.close()


class JobRepository(_Repository):
    async def active_long_rows(
        self, database_resource_keys: list[str]
    ) -> list[aiosqlite.Row]:
        if not database_resource_keys:
            return []
        resource_placeholders = ",".join("?" for _ in database_resource_keys)
        db = await self.connect()
        try:
            return await (
                await db.execute(
                    f"""SELECT * FROM jobs
                    WHERE library_id IN ({resource_placeholders})
                    AND status IN ('queued','running','pausing','paused','interrupted','stopping')
                    ORDER BY created_at""",
                    tuple(database_resource_keys),
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
