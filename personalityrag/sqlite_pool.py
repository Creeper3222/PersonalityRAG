from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import aiosqlite


_standalone_connections: set[aiosqlite.Connection] = set()


def register_standalone_connection(connection: aiosqlite.Connection) -> None:
    _standalone_connections.add(connection)


async def close_aiosqlite_connection(connection: aiosqlite.Connection) -> None:
    """Close aiosqlite and wait until its worker thread has actually exited."""

    closing = asyncio.create_task(connection.close())
    cancelled = False
    try:
        await asyncio.shield(closing)
    except asyncio.CancelledError:
        cancelled = True
        await asyncio.gather(closing, return_exceptions=True)
    thread = getattr(connection, "_thread", None)
    if thread is not None and thread.is_alive():
        # aiosqlite 0.22 resolves close() just before its worker executes the
        # final loop break. Joining prevents that tiny race from crossing an
        # event-loop/process shutdown boundary.
        await asyncio.to_thread(thread.join, 2.0)
    _standalone_connections.discard(connection)
    if cancelled:
        raise asyncio.CancelledError


async def close_standalone_connections() -> None:
    connections = tuple(_standalone_connections)
    if connections:
        await asyncio.gather(
            *(close_aiosqlite_connection(item) for item in connections),
            return_exceptions=True,
        )


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
    """Small lazy exclusive-lease pool for aiosqlite worker connections.

    A pool starts with no worker thread, grows only when concurrent leases need
    another connection, and converges back to one warm connection after a
    burst.  This matters because every aiosqlite connection owns one worker
    thread and most PersonalityRAG databases are cold most of the time.
    """

    # Keep started pools strongly reachable until close(). A weak registry can
    # lose a forgotten pool while its aiosqlite worker thread is still alive,
    # making process/test shutdown report into an already closed event loop.
    _instances: set[SQLiteConnectionPool] = set()

    def __init__(self, path: Path, *, size: int = 2):
        self.path = Path(path)
        self.size = max(1, int(size))
        self._available: list[aiosqlite.Connection] = []
        self._available_since: dict[aiosqlite.Connection, float] = {}
        self._connections: set[aiosqlite.Connection] = set()
        self._leased: set[aiosqlite.Connection] = set()
        self._start_lock = asyncio.Lock()
        self._condition = asyncio.Condition()
        self._creating = 0
        self._waiters = 0
        self._idle_reaper_task: asyncio.Task[None] | None = None
        self._idle_seconds = 30.0
        self._started = False
        self._closing = False

    @classmethod
    async def close_open_pools(cls) -> None:
        await asyncio.gather(
            *(
                pool.close()
                for pool in tuple(cls._instances)
                if pool._started or pool._connections or pool._creating
            )
        )
        await close_standalone_connections()

    async def _new_connection(self) -> aiosqlite.Connection:
        # aiosqlite starts its worker thread while the awaitable is opening.
        # If a runtime-load task is cancelled in that window, abandoning the
        # awaitable leaks a worker which later reports into a closed event loop.
        pending_connection = aiosqlite.connect(self.path)
        worker = getattr(pending_connection, "_thread", None)
        if worker is not None:
            worker.name = f"SQLitePool-{self.path.name}"
        opening = asyncio.ensure_future(pending_connection)
        db: aiosqlite.Connection | None = None
        try:
            db = await asyncio.shield(opening)
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA busy_timeout=10000")
            await db.execute("PRAGMA foreign_keys=ON")
            return db
        except BaseException:
            if db is None:
                await asyncio.gather(opening, return_exceptions=True)
                if not opening.cancelled() and opening.exception() is None:
                    db = opening.result()
            if db is not None:
                await close_aiosqlite_connection(db)
            raise

    async def start(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            if self._closing:
                raise RuntimeError("SQLite connection pool is closed")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._started = True
            self._instances.add(self)

    async def acquire(self) -> SQLiteLease:
        await self.start()
        while True:
            async with self._condition:
                if self._closing:
                    raise RuntimeError("SQLite connection pool is closed")
                if self._available:
                    connection = self._available.pop()
                    self._available_since.pop(connection, None)
                    self._leased.add(connection)
                    return SQLiteLease(self, connection)
                if len(self._connections) + self._creating < self.size:
                    self._creating += 1
                    break
                self._waiters += 1
                try:
                    await self._condition.wait()
                finally:
                    self._waiters = max(0, self._waiters - 1)

        connection: aiosqlite.Connection | None = None
        try:
            connection = await self._new_connection()
        except BaseException:
            async with self._condition:
                self._creating = max(0, self._creating - 1)
                self._condition.notify_all()
            raise

        close_connection = False
        async with self._condition:
            self._creating = max(0, self._creating - 1)
            if self._closing:
                close_connection = True
            else:
                self._connections.add(connection)
                self._leased.add(connection)
            self._condition.notify_all()
        if close_connection:
            await close_aiosqlite_connection(connection)
            raise RuntimeError("SQLite connection pool is closed")
        return SQLiteLease(self, connection)

    async def release(self, connection: aiosqlite.Connection) -> None:
        try:
            if connection.in_transaction:
                await connection.rollback()
        except Exception:
            await close_aiosqlite_connection(connection)
            async with self._condition:
                self._leased.discard(connection)
                self._connections.discard(connection)
                self._available_since.pop(connection, None)
                self._condition.notify_all()
            return
        async with self._condition:
            self._leased.discard(connection)
            if self._closing:
                self._connections.discard(connection)
                self._available_since.pop(connection, None)
                close_connection = True
            else:
                self._available.append(connection)
                self._available_since[connection] = asyncio.get_running_loop().time()
                close_connection = False
                self._schedule_idle_reaper_locked()
            self._condition.notify_all()
        if close_connection:
            await close_aiosqlite_connection(connection)

    def _schedule_idle_reaper_locked(self) -> None:
        if len(self._connections) <= 1:
            return
        if self._idle_reaper_task is None or self._idle_reaper_task.done():
            self._idle_reaper_task = asyncio.create_task(
                self._idle_reaper_loop(),
                name=f"sqlite-pool-idle-{self.path.name}",
            )

    async def _idle_reaper_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._idle_seconds)
                to_close: list[aiosqlite.Connection] = []
                async with self._condition:
                    if self._closing or len(self._connections) <= 1:
                        return
                    now = asyncio.get_running_loop().time()
                    for connection in tuple(self._available):
                        if len(self._connections) - len(to_close) <= 1:
                            break
                        idle_since = self._available_since.get(connection, now)
                        if now - idle_since < self._idle_seconds:
                            continue
                        self._available.remove(connection)
                        self._available_since.pop(connection, None)
                        self._connections.discard(connection)
                        to_close.append(connection)
                    self._condition.notify_all()
                    should_continue = len(self._connections) > 1
                if to_close:
                    await asyncio.gather(
                        *(
                            close_aiosqlite_connection(connection)
                            for connection in to_close
                        ),
                        return_exceptions=True,
                    )
                if not should_continue:
                    return
        except asyncio.CancelledError:
            raise
        finally:
            if self._idle_reaper_task is asyncio.current_task():
                self._idle_reaper_task = None

    async def close(self) -> None:
        async with self._condition:
            self._closing = True
            self._condition.notify_all()
        reaper = self._idle_reaper_task
        if reaper is not None:
            reaper.cancel()
            await asyncio.gather(reaper, return_exceptions=True)
            self._idle_reaper_task = None
        async with self._condition:
            while self._leased or self._creating:
                await self._condition.wait()
            available = list(self._available)
            self._available.clear()
            self._available_since.clear()
            self._connections.clear()
        if available:
            await asyncio.gather(
                *(
                    close_aiosqlite_connection(connection)
                    for connection in available
                ),
                return_exceptions=True,
            )
        async with self._condition:
            self._started = False
            self._closing = False
            self._instances.discard(self)
            self._condition.notify_all()

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    @property
    def leased_count(self) -> int:
        return len(self._leased)
