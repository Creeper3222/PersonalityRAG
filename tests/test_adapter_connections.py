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
from personalityrag.database_types import (
    DATABASE_CATEGORY_KNOWLEDGE,
    LIVINGMEMORY_V8_TYPE,
    TEXT_MEDIA_V1_TYPE,
    DatabaseRef,
    database_type_registry,
)
from personalityrag.http_shared import _adapter_status_request_allowed
from starlette.requests import Request


REPO_ROOT = Path(__file__).resolve().parents[1]


def _request(method: str, path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [],
            "query_string": b"",
            "server": ("test", 8765),
            "client": ("127.0.0.1", 1234),
            "scheme": "http",
        }
    )


def test_adapter_media_followups_bypass_busy_guard() -> None:
    ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, "art")
    base = "/api/v1/knowledge-libraries/text_media_v1/art/assets/image_1"
    assert _adapter_status_request_allowed(
        _request("POST", f"{base}/signed-url"),
        ref,
    )
    assert _adapter_status_request_allowed(
        _request("GET", f"{base}/content"),
        ref,
    )
    assert _adapter_status_request_allowed(
        _request("GET", f"{base}/thumbnail"),
        ref,
    )
    assert not _adapter_status_request_allowed(
        _request("POST", f"{base}/content"),
        ref,
    )


async def _linked_library(control: ControlStore) -> str:
    provider = await control.get_provider("seed_provider")
    assert provider is not None
    record = await control.create_library(
        {"id": "linked", "name": "Linked", "provider_id": provider.provider_id},
        provider,
    )
    return record.id


@pytest.mark.asyncio
async def test_adapter_heartbeat_registers_renews_and_rejects_active_duplicate(
    tmp_path,
):
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
                    "/api/v1/memory-libraries/livingmemory_v8/Default/adapters/heartbeat",
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
                "/api/v1/memory-libraries/livingmemory_v8/Default/adapters/Astrbot/disconnect",
                headers={
                    "Authorization": (
                        "Bearer " + _library_psk(config.library_psk_secret, "Default")
                    )
                },
                json={"instance_id": "instance-a"},
            )
            assert psk_response.status_code == 401

            disconnected = await client.post(
                "/api/v1/memory-libraries/livingmemory_v8/Default/adapters/Astrbot/disconnect",
                headers={"Authorization": "Bearer admin-key"},
                json={"instance_id": "instance-a"},
            )
            assert disconnected.status_code == 200
            heartbeat_response = await asyncio.wait_for(heartbeat, timeout=1)
            assert heartbeat_response.status_code == 409
            forced_detail = heartbeat_response.json()["detail"]
            assert forced_detail["code"] == "adapter_forced_offline"
            assert forced_detail["memory_store_id"] == "Default"
            assert forced_detail["memory_store_type"] == LIVINGMEMORY_V8_TYPE
            assert forced_detail["database_id"] == "Default"
            assert forced_detail["database_type"] == LIVINGMEMORY_V8_TYPE
            assert forced_detail["library_id"] == "Default"

            blocked = await client.get(
                "/api/v1/memory-libraries/livingmemory_v8/Default",
                headers=adapter_headers,
            )
            assert blocked.status_code == 409
            assert blocked.json()["detail"]["code"] == "adapter_forced_offline"

            reconnected = await client.post(
                "/api/v1/memory-libraries/livingmemory_v8/Default/adapters/heartbeat",
                headers=adapter_headers,
                json={"manual_reconnect": True},
            )
            assert reconnected.status_code == 200
            reconnect_payload = reconnected.json()
            assert reconnect_payload["connection_state"] == "active"
            assert reconnect_payload["memory_store_id"] == "Default"
            assert reconnect_payload["memory_store_type"] == LIVINGMEMORY_V8_TYPE
            assert reconnect_payload["database_id"] == "Default"
            assert reconnect_payload["database_type"] == LIVINGMEMORY_V8_TYPE
            assert reconnect_payload["library_id"] == "Default"
            assert (
                len(await context.manager.control.active_adapter_connections("Default"))
                == 1
            )
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_knowledge_adapter_multi_database_disconnect_is_isolated(tmp_path):
    config = AppConfig(
        api_key="admin-key",
        session_secret="session-key",
        library_psk_secret="knowledge-root-secret",
    )
    context = ApplicationContext.create(
        source_root=REPO_ROOT,
        state_root=tmp_path / "state",
        config=config,
        configure_logs=False,
    )
    await context.manager.initialize()
    first_ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, "knowledge_a")
    second_ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, "knowledge_b")
    await context.manager.control.register_database_identity(
        first_ref,
        category=DATABASE_CATEGORY_KNOWLEDGE,
    )
    await context.manager.control.register_database_identity(
        second_ref,
        category=DATABASE_CATEGORY_KNOWLEDGE,
    )
    try:
        app = create_app(context)
        driver = database_type_registry.require(TEXT_MEDIA_V1_TYPE)
        first_headers = {
            "Authorization": (
                "Bearer "
                + driver.derive_access_key(config.library_psk_secret, first_ref.id)
            ),
            "X-PersonalityRAG-Adapter-ID": "Astrbot",
            "X-PersonalityRAG-Adapter-Instance-ID": "instance-a",
            "X-PersonalityRAG-Adapter-Type": "astrbot-knowledge",
        }
        second_headers = {
            **first_headers,
            "Authorization": (
                "Bearer "
                + driver.derive_access_key(config.library_psk_secret, second_ref.id)
            ),
            "X-PersonalityRAG-Adapter-Instance-ID": "instance-b",
        }
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test:8765",
        ) as client:
            first = await client.post(
                "/api/v1/knowledge-libraries/text_media_v1/knowledge_a/adapters/heartbeat",
                headers=first_headers,
                json={},
            )
            second = await client.post(
                "/api/v1/knowledge-libraries/text_media_v1/knowledge_b/adapters/heartbeat",
                headers=second_headers,
                json={},
            )
            assert first.status_code == second.status_code == 200
            assert first.json()["database_type"] == TEXT_MEDIA_V1_TYPE
            assert first.json()["database_id"] == first_ref.id
            assert first.json()["knowledge_base_type"] == TEXT_MEDIA_V1_TYPE
            assert first.json()["knowledge_base_id"] == first_ref.id
            assert first.json()["library_id"] == first_ref.id

            access_key_disconnect = await client.post(
                "/api/v1/knowledge-libraries/text_media_v1/knowledge_a/adapters/Astrbot/disconnect",
                headers={"Authorization": first_headers["Authorization"]},
                json={"instance_id": "instance-a"},
            )
            assert access_key_disconnect.status_code == 401

            disconnected = await client.post(
                "/api/v1/knowledge-libraries/text_media_v1/knowledge_a/adapters/Astrbot/disconnect",
                headers={"Authorization": "Bearer admin-key"},
                json={"instance_id": "instance-a"},
            )
            assert disconnected.status_code == 200
            assert disconnected.json()["state"] == "forced_offline"

            blocked = await client.post(
                "/api/v1/knowledge-libraries/text_media_v1/knowledge_a/search",
                headers=first_headers,
                json={"query": "blocked before route execution"},
            )
            assert blocked.status_code == 409
            assert blocked.json()["detail"]["code"] == "adapter_forced_offline"
            assert blocked.json()["detail"]["database_id"] == first_ref.id
            assert blocked.json()["detail"]["knowledge_base_id"] == first_ref.id

            unaffected = await client.post(
                "/api/v1/knowledge-libraries/text_media_v1/knowledge_b/adapters/heartbeat",
                headers=second_headers,
                json={},
            )
            assert unaffected.status_code == 200

            rejected = await client.post(
                "/api/v1/knowledge-libraries/text_media_v1/knowledge_a/adapters/heartbeat",
                headers=first_headers,
                json={},
            )
            assert rejected.status_code == 409
            assert rejected.json()["detail"]["code"] == "adapter_forced_offline"

            reconnected = await client.post(
                "/api/v1/knowledge-libraries/text_media_v1/knowledge_a/adapters/heartbeat",
                headers=first_headers,
                json={"manual_reconnect": True},
            )
            assert reconnected.status_code == 200
            assert reconnected.json()["connection_state"] == "active"

        first_connections = await context.manager.control.active_adapter_connections(
            first_ref
        )
        second_connections = await context.manager.control.active_adapter_connections(
            second_ref
        )
        assert [item["instance_id"] for item in first_connections] == ["instance-a"]
        assert [item["instance_id"] for item in second_connections] == ["instance-b"]
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_knowledge_heartbeat_reports_busy_and_search_stays_guarded(tmp_path):
    config = AppConfig(
        api_key="admin-key",
        session_secret="session-key",
        library_psk_secret="knowledge-root-secret",
    )
    context = ApplicationContext.create(
        source_root=REPO_ROOT,
        state_root=tmp_path / "state",
        config=config,
        configure_logs=False,
    )
    await context.manager.initialize()
    ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, "busy_knowledge")
    await context.manager.control.register_database_identity(
        ref,
        category=DATABASE_CATEGORY_KNOWLEDGE,
    )
    resource_key = database_type_registry.require(TEXT_MEDIA_V1_TYPE).resource_key(
        ref.id
    )
    now = time.time()
    db = await context.manager.control.connect()
    try:
        await db.execute(
            """INSERT INTO jobs
            (id,library_id,database_type,database_id,kind,status,progress,
             message,created_at,updated_at)
            VALUES(?,?,?,?,?,'running',0.5,'busy',?,?)""",
            (
                "knowledge-busy-job",
                resource_key,
                TEXT_MEDIA_V1_TYPE,
                ref.id,
                "text_media_index_rebuild",
                now,
                now,
            ),
        )
        await db.commit()
    finally:
        await db.close()
    try:
        driver = database_type_registry.require(TEXT_MEDIA_V1_TYPE)
        headers = {
            "Authorization": (
                "Bearer " + driver.derive_access_key(config.library_psk_secret, ref.id)
            ),
            "X-PersonalityRAG-Adapter-ID": "Astrbot",
            "X-PersonalityRAG-Adapter-Instance-ID": "instance-busy",
            "X-PersonalityRAG-Adapter-Type": "astrbot-knowledge",
        }
        async with AsyncClient(
            transport=ASGITransport(app=create_app(context)),
            base_url="http://test:8765",
        ) as client:
            heartbeat = await client.post(
                "/api/v1/knowledge-libraries/text_media_v1/busy_knowledge/adapters/heartbeat",
                headers=headers,
                json={},
            )
            assert heartbeat.status_code == 200
            busy_state = heartbeat.json()["adapter_busy"]
            assert busy_state["busy"] is True
            assert busy_state["job"] == {
                "database_id": ref.id,
                "database_type": TEXT_MEDIA_V1_TYPE,
                "knowledge_base_id": ref.id,
                "knowledge_base_type": TEXT_MEDIA_V1_TYPE,
                "id": "knowledge-busy-job",
                "library_id": resource_key,
                "kind": "text_media_index_rebuild",
                "status": "running",
                "progress": 0.5,
                "message": "busy",
                "created_at": now,
                "updated_at": now,
            }
            blocked = await client.post(
                "/api/v1/knowledge-libraries/text_media_v1/busy_knowledge/search",
                headers=headers,
                json={"query": "blocked while rebuilding"},
            )
            assert blocked.status_code == 409
            assert blocked.json()["detail"]["code"] == "library_busy"
            assert blocked.json()["detail"]["condition"] == "database_busy"
            assert blocked.json()["detail"]["knowledge_base_id"] == ref.id
            assert blocked.json()["detail"]["database_id"] == ref.id
            assert blocked.json()["detail"]["library_id"] == ref.id
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_knowledge_forced_offline_cache_reloads_from_control_store(tmp_path):
    config = AppConfig(
        api_key="admin-key",
        session_secret="session-key",
        library_psk_secret="knowledge-root-secret",
    )
    state_root = tmp_path / "state"
    first_context = ApplicationContext.create(
        source_root=REPO_ROOT,
        state_root=state_root,
        config=config,
        configure_logs=False,
    )
    await first_context.manager.initialize()
    ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, "persisted_knowledge")
    await first_context.manager.control.register_database_identity(
        ref,
        category=DATABASE_CATEGORY_KNOWLEDGE,
    )
    await first_context.manager.control.register_adapter_connection(
        ref,
        adapter_id="Astrbot",
        instance_id="persisted-instance",
        adapter_type="astrbot-knowledge",
    )
    forced = await first_context.manager.control.force_disconnect_adapter(
        ref,
        "Astrbot",
        expected_instance_id="persisted-instance",
    )
    first_context.manager.mark_adapter_forced_offline(forced)
    await first_context.manager.close()

    second_context = ApplicationContext.create(
        source_root=REPO_ROOT,
        state_root=state_root,
        config=config,
        configure_logs=False,
    )
    await second_context.manager.initialize()
    try:
        loaded = second_context.manager.forced_adapter_connection(ref, "Astrbot")
        assert loaded is not None
        assert loaded["database_type"] == TEXT_MEDIA_V1_TYPE
        assert loaded["database_id"] == ref.id
        assert loaded["state"] == "forced_offline"
    finally:
        await second_context.manager.close()


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
async def test_default_library_can_rename_until_an_adapter_connects_but_never_delete(
    tmp_path,
):
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
async def test_active_long_job_summary_only_tracks_adapter_blocking_jobs(tmp_path):
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
                (
                    "import-job",
                    library_id,
                    "livingmemory_import",
                    "queued",
                    0.0,
                    "blocking",
                    now + 2,
                    now + 2,
                ),
            ],
        )
        await db.commit()
    finally:
        await db.close()

    active = await control.active_long_job(library_id)
    assert active is not None
    assert active["id"] == "import-job"
    assert active["kind"] == "livingmemory_import"
