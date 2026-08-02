from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
import weakref

import aiosqlite


class SQLiteLease:
    def __init__(
        self,
        pool: SQLiteConnectionPool,
        connection: aiosqlite.Connection,
    ):
        self._pool = pool
        self._connection = connection
        self._released = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    async def close(self) -> None:
        if self._released:
            return
        self._released = True
        await self._pool.release(self._connection)

    async def __aenter__(self) -> SQLiteLease:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()


class SQLiteConnectionPool:
    """Small exclusive-lease pool for aiosqlite worker connections."""

    _instances: weakref.WeakSet[SQLiteConnectionPool] = weakref.WeakSet()

    def __init__(self, path: Path, *, size: int = 2):
        self.path = Path(path)
        self.size = max(1, int(size))
        self._available: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue()
        self._connections: set[aiosqlite.Connection] = set()
        self._leased: set[aiosqlite.Connection] = set()
        self._start_lock = asyncio.Lock()
        self._condition = asyncio.Condition()
        self._started = False
        self._closing = False
        self._instances.add(self)

    @classmethod
    async def close_open_pools(cls) -> None:
        await asyncio.gather(
            *(pool.close() for pool in tuple(cls._instances) if pool._started)
        )

    async def _new_connection(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self.path)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA busy_timeout=10000")
        await db.execute("PRAGMA foreign_keys=ON")
        return db

    async def start(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            if self._closing:
                raise RuntimeError("SQLite connection pool is closed")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            created: list[aiosqlite.Connection] = []
            try:
                for _ in range(self.size):
                    created.append(await self._new_connection())
            except BaseException:
                await asyncio.gather(
                    *(connection.close() for connection in created),
                    return_exceptions=True,
                )
                raise
            self._connections.update(created)
            for connection in created:
                self._available.put_nowait(connection)
            self._started = True

    async def acquire(self) -> SQLiteLease:
        await self.start()
        if self._closing:
            raise RuntimeError("SQLite connection pool is closed")
        connection = await self._available.get()
        async with self._condition:
            if self._closing:
                await connection.close()
                self._connections.discard(connection)
                self._condition.notify_all()
                raise RuntimeError("SQLite connection pool is closed")
            self._leased.add(connection)
        return SQLiteLease(self, connection)

    async def release(self, connection: aiosqlite.Connection) -> None:
        try:
            if connection.in_transaction:
                await connection.rollback()
        except Exception:
            await connection.close()
            async with self._condition:
                self._leased.discard(connection)
                self._connections.discard(connection)
                self._condition.notify_all()
            return
        async with self._condition:
            self._leased.discard(connection)
            if self._closing:
                self._connections.discard(connection)
                close_connection = True
            else:
                self._available.put_nowait(connection)
                close_connection = False
            self._condition.notify_all()
        if close_connection:
            await connection.close()

    async def close(self) -> None:
        async with self._condition:
            self._closing = True
        while True:
            try:
                connection = self._available.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._connections.discard(connection)
            await connection.close()
        async with self._condition:
            while self._leased:
                await self._condition.wait()
        self._started = False
        self._closing = False

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    @property
    def leased_count(self) -> int:
        return len(self._leased)
