from __future__ import annotations

import asyncio
import time

import pytest

from personalityrag.config import AppConfig
from personalityrag.database_types import DatabaseRef, LIVINGMEMORY_V8_TYPE
from personalityrag.library_types.livingmemory_v8.manager import (
    LivingMemoryV8Manager as LibraryManager,
    RuntimeResidencyState,
)


def _ref(database_id: str) -> DatabaseRef:
    return DatabaseRef(LIVINGMEMORY_V8_TYPE, database_id)


class FakeRuntime:
    def __init__(self, library_id: str):
        self.library_id = library_id
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_cold_runtime_load_is_single_flight_without_blocking_warm_lease(
    tmp_path,
    monkeypatch,
) -> None:
    manager = LibraryManager(tmp_path, AppConfig())
    warm = FakeRuntime("Default")
    manager.runtimes[_ref("Default")] = warm
    manager._runtime_residency[_ref("Default")] = RuntimeResidencyState(
        lease_count=0,
        last_used_at=time.monotonic(),
    )
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def build(ref: DatabaseRef):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return FakeRuntime(ref.id)

    monkeypatch.setattr(manager, "_build_runtime", build)
    first = asyncio.create_task(manager.get_runtime("cold"))
    second = asyncio.create_task(manager.get_runtime("cold"))
    await started.wait()

    leased = await asyncio.wait_for(manager.acquire_runtime("Default"), timeout=0.1)
    assert leased is warm
    await manager.release_runtime("Default")
    assert first.done() is False
    assert calls == 1

    release.set()
    cold_first, cold_second = await asyncio.gather(first, second)
    assert cold_first is cold_second
    assert calls == 1
    await manager.close()


@pytest.mark.asyncio
async def test_runtime_release_does_not_sweep_without_capacity_pressure(
    tmp_path,
    monkeypatch,
) -> None:
    manager = LibraryManager(tmp_path, AppConfig())
    runtime = FakeRuntime("Default")
    manager.runtimes[_ref("Default")] = runtime
    manager._runtime_residency[_ref("Default")] = RuntimeResidencyState(
        lease_count=1,
        last_used_at=time.monotonic(),
    )

    async def unexpected_sweep(*, expire_idle: bool = True):
        raise AssertionError("request release must not sweep without pressure")

    monkeypatch.setattr(manager, "sweep_runtimes", unexpected_sweep)
    await manager.release_runtime("Default")
    assert manager._runtime_residency[_ref("Default")].lease_count == 0
    await manager.close()
