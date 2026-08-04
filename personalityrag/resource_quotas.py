from __future__ import annotations

import asyncio
import weakref
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import AsyncIterator, Iterator

from .resource_limits import configured_io_workers, configured_provider_concurrency


_lane: ContextVar[str] = ContextVar("personalityrag_resource_lane", default="foreground")


@dataclass(slots=True)
class _LoopQuotas:
    io_limit: int
    provider_limit: int
    io_total: asyncio.Semaphore
    io_background: asyncio.Semaphore
    provider_total: asyncio.Semaphore
    provider_background: asyncio.Semaphore


_quotas: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, _LoopQuotas
] = weakref.WeakKeyDictionary()


def _loop_quotas() -> _LoopQuotas:
    loop = asyncio.get_running_loop()
    current = _quotas.get(loop)
    # Settings changes explicitly reset the registry. Avoid re-reading config,
    # cgroup and CPU limits on every I/O or Provider request in the hot path.
    if current is not None:
        return current
    io_limit = configured_io_workers()
    provider_limit = configured_provider_concurrency()
    current = _LoopQuotas(
        io_limit=io_limit,
        provider_limit=provider_limit,
        io_total=asyncio.Semaphore(io_limit),
        io_background=asyncio.Semaphore(max(1, io_limit - 1)),
        provider_total=asyncio.Semaphore(provider_limit),
        provider_background=asyncio.Semaphore(max(1, provider_limit - 1)),
    )
    _quotas[loop] = current
    return current


def reset_resource_quotas() -> None:
    _quotas.clear()


@contextmanager
def resource_lane(name: str) -> Iterator[None]:
    token = _lane.set("task" if name == "task" else "foreground")
    try:
        yield
    finally:
        _lane.reset(token)


@asynccontextmanager
async def io_slot() -> AsyncIterator[None]:
    quotas = _loop_quotas()
    background = _lane.get() == "task"
    if background:
        await quotas.io_background.acquire()
    try:
        await quotas.io_total.acquire()
        try:
            yield
        finally:
            quotas.io_total.release()
    finally:
        if background:
            quotas.io_background.release()


@asynccontextmanager
async def provider_slot() -> AsyncIterator[None]:
    quotas = _loop_quotas()
    background = _lane.get() == "task"
    if background:
        await quotas.provider_background.acquire()
    try:
        await quotas.provider_total.acquire()
        try:
            yield
        finally:
            quotas.provider_total.release()
    finally:
        if background:
            quotas.provider_background.release()
