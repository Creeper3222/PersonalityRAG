from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from personalityrag.config import AppConfig, ProviderConfig
from personalityrag.control import ControlStore, ProviderRevision
from personalityrag.libraries import LibraryManager
from personalityrag.providers import provider_config_hash
from personalityrag.storage import Storage


async def _control_with_libraries(tmp_path: Path, count: int = 3) -> ControlStore:
    control = ControlStore(tmp_path / "personalityrag_system.db")
    config = ProviderConfig(id="seed_provider", dimensions=8)
    await control.initialize(config)
    provider = await control.get_provider(config.id)
    assert provider is not None
    for index in range(count):
        await control.create_library(
            {
                "id": f"library_{index}",
                "name": f"Library {index}",
                "provider_id": provider.provider_id,
            },
            provider,
        )
    return control


@pytest.mark.asyncio
async def test_control_library_and_provider_reads_are_batched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = await _control_with_libraries(tmp_path)
    connections = 0
    original = control.connect

    async def counted_connect():
        nonlocal connections
        connections += 1
        return await original()

    monkeypatch.setattr(control, "connect", counted_connect)
    libraries = await control.list_libraries()
    assert len(libraries) == 3
    assert connections == 1

    connections = 0
    providers = await control.get_providers_bulk(
        {("seed_provider", 1), ("seed_provider", None)}
    )
    assert providers[("seed_provider", 1)].provider_id == "seed_provider"
    assert providers[("seed_provider", None)].revision == 1
    assert connections == 1


@pytest.mark.asyncio
async def test_summary_library_list_uses_scalar_stats_without_full_metadata_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "PersonalityRAG"
    config = AppConfig(provider=ProviderConfig(id="seed_provider", dimensions=8))
    manager = LibraryManager(root, config)
    await manager.control.initialize(config.provider)
    provider = await manager.control.get_provider(config.provider.id)
    assert provider is not None
    await manager.control.create_library(
        {
            "id": "summary_library",
            "name": "Summary",
            "provider_id": provider.provider_id,
        },
        provider,
    )
    storage = Storage(
        root / "data" / "libraries" / "summary_library",
        system_path=manager.system_path,
    )
    await storage.initialize()
    await storage.add_conversation_message(
        {
            "session_id": "session-a",
            "role": "user",
            "content": "fixture",
        }
    )

    async def fail_full_statistics(_self):
        raise AssertionError("summary mode must not call full statistics")

    monkeypatch.setattr(Storage, "statistics", fail_full_statistics)
    payload = await manager.list_libraries(stats_mode="summary")

    assert len(payload) == 1
    stats = payload[0]["stats"]
    assert stats["total_memories"] == 0
    assert stats["session_count"] == 0
    assert stats["conversation_counts"] == {
        "sessions": 1,
        "messages": 1,
        "pending_messages": 1,
    }
    assert "sessions" not in stats
    assert "status_breakdown" not in stats


@pytest.mark.asyncio
async def test_summary_session_count_uses_active_livingmemory_sessions(
    tmp_path: Path,
) -> None:
    storage = Storage(tmp_path / "session-semantics")
    await storage.initialize()
    async with storage.connect() as db:
        await db.executemany(
            "INSERT INTO documents(id,doc_id,text,metadata) VALUES(?,?,?,?)",
            [
                (1, "doc-1", "active one", '{"session_id":"session-a"}'),
                (
                    2,
                    "doc-2",
                    "active same session",
                    '{"session_id":"session-a","status":"active"}',
                ),
                (
                    3,
                    "doc-3",
                    "archived session",
                    '{"session_id":"session-b","status":"archived"}',
                ),
                (4, "doc-4", "active two", '{"session_id":"session-c"}'),
            ],
        )
        await db.commit()
    await storage.add_conversation_message(
        {
            "session_id": "short-session",
            "role": "user",
            "content": "short-term only",
        }
    )

    summary = await storage.summary_statistics()
    full = await storage.statistics()

    assert summary["session_count"] == 2
    assert set(full["sessions"]) == {"session-a", "session-c"}
    assert summary["conversation_counts"]["sessions"] == 1


@pytest.mark.asyncio
async def test_summary_for_config_only_empty_library_does_not_create_databases(
    tmp_path: Path,
) -> None:
    storage = Storage(tmp_path / "config-only-library")

    stats = await storage.summary_statistics()

    assert stats["total_memories"] == 0
    assert stats["conversation_counts"]["sessions"] == 0
    assert not storage.db_path.exists()
    assert not storage.conversations_path.exists()


class CountingProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.release = asyncio.Event()

    async def test_connection(self):
        self.calls += 1
        await self.release.wait()
        return {"available": True, "model": "fixture"}


@pytest.mark.asyncio
async def test_provider_health_cache_is_single_flight_and_has_diagnostics(
    tmp_path: Path,
) -> None:
    config = ProviderConfig(id="provider", dimensions=8)
    manager = LibraryManager(tmp_path, AppConfig(provider=config))
    provider = CountingProvider()
    revision = ProviderRevision(
        provider_id=config.id,
        revision=1,
        config=config,
        config_sha256=provider_config_hash(config),
        created_at=time.time(),
    )
    runtime = SimpleNamespace(provider=provider, provider_revision=revision)

    requests = [
        asyncio.create_task(manager.provider_status(runtime)) for _ in range(5)
    ]
    for _ in range(10):
        await asyncio.sleep(0)
        if provider.calls:
            break
    assert provider.calls == 1
    provider.release.set()
    results = await asyncio.gather(*requests)

    assert provider.calls == 1
    assert sum(not item["cached"] for item in results) == 1
    assert all(item["available"] is True for item in results)
    assert all(item["checked_at"] is not None for item in results)
    assert all(item["age_seconds"] >= 0 for item in results)

    cached = await manager.provider_status(runtime)
    assert cached["cached"] is True
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_cached_only_provider_status_never_calls_provider(tmp_path: Path) -> None:
    config = ProviderConfig(id="provider", dimensions=8)
    manager = LibraryManager(tmp_path, AppConfig(provider=config))
    provider = CountingProvider()
    revision = ProviderRevision(
        provider_id=config.id,
        revision=1,
        config=config,
        config_sha256=provider_config_hash(config),
        created_at=time.time(),
    )
    runtime = SimpleNamespace(provider=provider, provider_revision=revision)

    status = await manager.provider_status(runtime, allow_probe=False)

    assert provider.calls == 0
    assert status == {
        "available": None,
        "status": "not_checked",
        "reason": "provider status is not cached",
        "cached": True,
        "checked_at": None,
        "age_seconds": None,
    }


def test_webui_library_list_requests_summary_mode() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "static"
        / "modules"
        / "libraries.js"
    ).read_text(encoding="utf-8")
    assert 'api("/libraries?stats_mode=summary")' in source
