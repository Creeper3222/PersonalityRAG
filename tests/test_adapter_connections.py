from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import sqlite3
import time
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from personalityrag.application import create_app
from personalityrag.application_context import ApplicationContext
from personalityrag.config import ProviderConfig
from personalityrag.config import AppConfig
from personalityrag.control import (
    AdapterConnectionChangedError,
    AdapterForcedOfflineError,
    ControlStore,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


async def _linked_library(control: ControlStore) -> str:
    provider = await control.get_provider("seed_provider")
    assert provider is not None
    record = await control.create_library(
        {"id": "linked", "name": "Linked", "provider_id": provider.provider_id},
        provider,
    )
    return record.id


@pytest.mark.asyncio
async def test_adapter_heartbeat_registers_renews_and_rejects_active_duplicate(tmp_path):
    control = ControlStore(tmp_path / "system.db")
    await control.initialize(ProviderConfig(id="seed_provider"))
    library_id = await _linked_library(control)

    first = await control.register_adapter_connection(
        library_id,
        adapter_id="Astrbot",
        instance_id="instance-a",
        adapter_type="astrbot",
    )
    assert first["adapter_id"] == "Astrbot"
    assert first["instance_id"] == "instance-a"

    renewed = await control.register_adapter_connection(
        library_id,
        adapter_id="Astrbot",
        instance_id="instance-a",
        adapter_type="astrbot",
    )
    assert renewed["connected_at"] == pytest.approx(first["connected_at"])
    assert renewed["last_seen"] >= first["last_seen"]

    with pytest.raises(ValueError, match="适配器标识ID"):
        await control.register_adapter_connection(
            library_id,
            adapter_id="Astrbot",
            instance_id="instance-b",
            adapter_type="astrbot",
        )

    takeover = await control.register_adapter_connection(
        library_id,
        adapter_id="Astrbot",
        instance_id="instance-b",
        adapter_type="astrbot",
        ttl_seconds=-1,
    )
    assert takeover["instance_id"] == "instance-b"


@pytest.mark.asyncio
async def test_forced_offline_persists_and_requires_explicit_reconnect(tmp_path):
    control = ControlStore(tmp_path / "system.db")
    await control.initialize(ProviderConfig(id="seed_provider"))
    library_id = await _linked_library(control)
    await control.register_adapter_connection(
        library_id,
        adapter_id="Astrbot",
        instance_id="instance-a",
    )

    disconnected = await control.force_disconnect_adapter(
        library_id,
        "Astrbot",
        expected_instance_id="instance-a",
    )
    assert disconnected["state"] == "forced_offline"
    assert await control.active_adapter_connections(library_id) == []

    with pytest.raises(AdapterForcedOfflineError):
        await control.register_adapter_connection(
            library_id,
            adapter_id="Astrbot",
            instance_id="instance-a",
        )

    reopened = ControlStore(tmp_path / "system.db")
    persisted = await reopened.adapter_connection(library_id, "Astrbot")
    assert persisted is not None
    assert persisted["state"] == "forced_offline"

    reconnected = await reopened.register_adapter_connection(
        library_id,
        adapter_id="Astrbot",
        instance_id="instance-a",
        manual_reconnect=True,
    )
    assert reconnected["state"] == "active"
    assert reconnected["disconnected_at"] is None


@pytest.mark.asyncio
async def test_force_disconnect_rejects_stale_instance(tmp_path):
    control = ControlStore(tmp_path / "system.db")
    await control.initialize(ProviderConfig(id="seed_provider"))
    library_id = await _linked_library(control)
    await control.register_adapter_connection(
        library_id,
        adapter_id="Astrbot",
        instance_id="instance-a",
    )

    with pytest.raises(AdapterConnectionChangedError):
        await control.force_disconnect_adapter(
            library_id,
            "Astrbot",
            expected_instance_id="instance-b",
        )
    assert len(await control.active_adapter_connections(library_id)) == 1


@pytest.mark.asyncio
async def test_old_adapter_connection_schema_is_upgraded_as_active(tmp_path):
    system_path = tmp_path / "system.db"
    with sqlite3.connect(system_path) as db:
        db.execute(
            """CREATE TABLE adapter_connections (
            library_id TEXT NOT NULL,
            adapter_id TEXT NOT NULL,
            instance_id TEXT NOT NULL,
            adapter_type TEXT NOT NULL DEFAULT 'unknown',
            connected_at REAL NOT NULL,
            last_seen REAL NOT NULL,
            PRIMARY KEY(library_id, adapter_id))"""
        )
        db.execute(
            "INSERT INTO adapter_connections VALUES(?,?,?,?,?,?)",
            ("linked", "Astrbot", "instance-a", "astrbot", time.time(), time.time()),
        )
    control = ControlStore(system_path)
    await control.initialize(ProviderConfig(id="seed_provider"))
    connection = await control.adapter_connection("linked", "Astrbot")
    assert connection is not None
    assert connection["state"] == "active"


def _library_psk(secret: str, library_id: str) -> str:
    digest = hmac.new(
        secret.encode("utf-8"),
        library_id.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return "psk-" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


@pytest.mark.asyncio
async def test_http_disconnect_wakes_heartbeat_and_manual_reconnects(tmp_path):
    config = AppConfig(api_key="admin-key", session_secret="session-key")
    context = ApplicationContext.create(
        source_root=REPO_ROOT,
        state_root=tmp_path / "state",
        config=config,
        configure_logs=False,
    )
    await context.manager.initialize()
    try:
        app = create_app(context)
        adapter_headers = {
            "Authorization": "Bearer admin-key",
            "X-PersonalityRAG-Adapter-ID": "Astrbot",
            "X-PersonalityRAG-Adapter-Instance-ID": "instance-a",
            "X-PersonalityRAG-Adapter-Type": "astrbot",
        }
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test:8765",
        ) as client:
            heartbeat = asyncio.create_task(
                client.post(
                    "/api/v1/libraries/Default/adapters/heartbeat",
                    headers=adapter_headers,
                    json={"wait_seconds": 5},
                )
            )
            for _ in range(100):
                if await context.manager.control.adapter_connection(
                    "Default", "Astrbot"
                ):
                    break
                await asyncio.sleep(0.01)

            psk_response = await client.post(
                "/api/v1/libraries/Default/adapters/Astrbot/disconnect",
                headers={
                    "Authorization": (
                        "Bearer "
                        + _library_psk(config.library_psk_secret, "Default")
                    )
                },
                json={"instance_id": "instance-a"},
            )
            assert psk_response.status_code == 401

            disconnected = await client.post(
                "/api/v1/libraries/Default/adapters/Astrbot/disconnect",
                headers={"Authorization": "Bearer admin-key"},
                json={"instance_id": "instance-a"},
            )
            assert disconnected.status_code == 200
            heartbeat_response = await asyncio.wait_for(heartbeat, timeout=1)
            assert heartbeat_response.status_code == 409
            assert (
                heartbeat_response.json()["detail"]["code"]
                == "adapter_forced_offline"
            )

            blocked = await client.get(
                "/api/v1/libraries/Default",
                headers=adapter_headers,
            )
            assert blocked.status_code == 409
            assert blocked.json()["detail"]["code"] == "adapter_forced_offline"

            reconnected = await client.post(
                "/api/v1/libraries/Default/adapters/heartbeat",
                headers=adapter_headers,
                json={"manual_reconnect": True},
            )
            assert reconnected.status_code == 200
            assert reconnected.json()["connection_state"] == "active"
            assert len(
                await context.manager.control.active_adapter_connections("Default")
            ) == 1
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_active_adapter_connection_blocks_library_delete(tmp_path):
    control = ControlStore(tmp_path / "system.db")
    await control.initialize(ProviderConfig(id="seed_provider"))
    library_id = await _linked_library(control)
    await control.register_adapter_connection(
        library_id,
        adapter_id="Astrbot",
        instance_id="instance-a",
        adapter_type="astrbot",
    )

    with pytest.raises(ValueError, match="适配器连接"):
        await control.mark_library_deleted(library_id)


@pytest.mark.asyncio
async def test_default_library_can_rename_until_an_adapter_connects_but_never_delete(tmp_path):
    control = ControlStore(tmp_path / "system.db")
    await control.initialize(ProviderConfig(id="seed_provider"))
    provider = await control.get_provider("seed_provider")
    assert provider is not None
    default = await control.ensure_default_library(
        library_id="Default",
        name="默认库",
        provider_id=provider.provider_id,
        provider_revision=provider.revision,
    )

    renamed = await control.update_library(default.id, {"id": "renamed_default"})
    assert renamed.id == "renamed_default"
    assert renamed.is_default is True

    with pytest.raises(ValueError, match="默认记忆库不能删除"):
        await control.mark_library_deleted(renamed.id)

    await control.register_adapter_connection(
        renamed.id,
        adapter_id="Astrbot",
        instance_id="instance-a",
    )
    with pytest.raises(ValueError, match="适配器连接"):
        await control.update_library(renamed.id, {"id": "blocked_default"})


@pytest.mark.asyncio
async def test_active_long_job_summary_only_tracks_full_library_jobs(tmp_path):
    control = ControlStore(tmp_path / "system.db")
    await control.initialize(ProviderConfig(id="seed_provider"))
    library_id = await _linked_library(control)
    db = await control.connect()
    now = time.time()
    try:
        await db.executemany(
            """INSERT INTO jobs
            (id,library_id,kind,status,progress,message,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?)""",
            [
                (
                    "memory-job",
                    library_id,
                    "memory_create",
                    "running",
                    0.1,
                    "short",
                    now,
                    now,
                ),
                (
                    "rebuild-job",
                    library_id,
                    "index_rebuild",
                    "queued",
                    0.0,
                    "long",
                    now + 1,
                    now + 1,
                ),
            ],
        )
        await db.commit()
    finally:
        await db.close()

    active = await control.active_long_job(library_id)
    assert active is not None
    assert active["id"] == "rebuild-job"
    assert active["kind"] == "index_rebuild"
