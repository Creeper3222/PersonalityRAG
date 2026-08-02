from __future__ import annotations

import asyncio

import pytest

from personalityrag.sqlite_pool import SQLiteConnectionPool


@pytest.mark.asyncio
async def test_pool_reuses_connections_and_rolls_back_unfinished_transactions(tmp_path):
    pool = SQLiteConnectionPool(tmp_path / "pooled.db", size=2)
    first = await pool.acquire()
    await first.execute("CREATE TABLE values_table(value TEXT)")
    await first.commit()
    identity = id(first._connection)
    await first.execute("INSERT INTO values_table(value) VALUES('not-committed')")
    await first.close()

    seen: set[int] = set()
    for _ in range(4):
        lease = await pool.acquire()
        seen.add(id(lease._connection))
        count = int(
            (await (await lease.execute("SELECT COUNT(*) FROM values_table")).fetchone())[0]
        )
        assert count == 0
        await lease.close()

    assert identity in seen
    assert pool.connection_count == 2
    await pool.close()
    assert pool.connection_count == 0


@pytest.mark.asyncio
async def test_pool_uses_exclusive_leases_and_close_waits_for_return(tmp_path):
    pool = SQLiteConnectionPool(tmp_path / "exclusive.db", size=1)
    first = await pool.acquire()
    waiter = asyncio.create_task(pool.acquire())
    await asyncio.sleep(0)
    assert waiter.done() is False
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)

    closing = asyncio.create_task(pool.close())
    await asyncio.sleep(0)
    assert closing.done() is False
    await first.close()
    await closing
    reopened = await pool.acquire()
    assert pool.connection_count == 1
    await reopened.close()
    await pool.close()
