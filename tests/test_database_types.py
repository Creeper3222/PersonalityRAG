from __future__ import annotations

import base64
import hashlib
import hmac
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from personalityrag.application import create_app
from personalityrag.application_context import ApplicationContext
from personalityrag.config import AppConfig, ProviderConfig
from personalityrag.control import ControlStore
from personalityrag.database_types import (
    DATABASE_CATEGORY_KNOWLEDGE,
    DATABASE_CATEGORY_MEMORY,
    LIVINGMEMORY_V8_TYPE,
    DatabaseRef,
    DatabaseTypeDescriptor,
    DatabaseTypeRegistry,
    derive_domain_separated_access_key,
)
from personalityrag.library_types.livingmemory_v8.driver import LivingMemoryV8Driver
from personalityrag.library_types.text_media_v1.driver import TextMediaV1Driver


@dataclass(slots=True)
class FixtureDriver:
    descriptor: DatabaseTypeDescriptor

    def resource_key(self, database_id: str) -> str:
        return f"{self.descriptor.id}:{database_id}"

    def create_manager(self, root: Path, config):
        raise NotImplementedError("registry-only fixture")

    def derive_access_key(self, root_secret: str, database_id: str) -> str:
        return derive_domain_separated_access_key(
            root_secret=root_secret,
            database_type=self.descriptor.id,
            database_id=database_id,
            prefix=self.descriptor.key_prefix,
        )

    def verify_access_key(
        self, root_secret: str, database_id: str, candidate: str | None
    ) -> bool:
        return bool(
            candidate
            and hmac.compare_digest(
                candidate, self.derive_access_key(root_secret, database_id)
            )
        )


def _fixture_driver(
    database_type: str,
    category: str,
    derivation_id: str,
) -> FixtureDriver:
    return FixtureDriver(
        DatabaseTypeDescriptor(
            id=database_type,
            category=category,
            display_name=database_type,
            description="fixture",
            icon="",
            capabilities=("adapter_access",),
            key_prefix="psk-" if category == DATABASE_CATEGORY_MEMORY else "pkb-",
            key_derivation_id=derivation_id,
        )
    )


def test_database_registry_separates_same_id_types_and_key_domains(tmp_path: Path):
    registry = DatabaseTypeRegistry()
    memory = _fixture_driver("fixture_memory", DATABASE_CATEGORY_MEMORY, "memory-v1")
    knowledge = _fixture_driver(
        "fixture_knowledge", DATABASE_CATEGORY_KNOWLEDGE, "knowledge-v1"
    )
    registry.register(memory)
    registry.register(knowledge)

    memory_ref = DatabaseRef(memory.descriptor.id, "shared")
    knowledge_ref = DatabaseRef(knowledge.descriptor.id, "shared")
    assert memory_ref != knowledge_ref
    assert memory.resource_key("shared") != knowledge.resource_key("shared")
    assert registry.data_dir(tmp_path, memory_ref) != registry.data_dir(
        tmp_path, knowledge_ref
    )
    memory_key = memory.derive_access_key("root", "shared")
    knowledge_key = knowledge.derive_access_key("root", "shared")
    assert memory_key.startswith("psk-")
    assert knowledge_key.startswith("pkb-")
    assert memory_key != knowledge_key
    assert not memory.verify_access_key("root", "shared", knowledge_key)
    assert not knowledge.verify_access_key("root", "shared", memory_key)

    with pytest.raises(ValueError, match="已注册"):
        registry.register(memory)
    with pytest.raises(ValueError, match="派生标识"):
        registry.register(
            _fixture_driver("another_memory", DATABASE_CATEGORY_MEMORY, "memory-v1")
        )


def test_livingmemory_v8_access_key_is_byte_for_byte_legacy_compatible():
    root_secret = "stable-root-secret"
    database_id = "same-id"
    digest = hmac.new(
        root_secret.encode("utf-8"),
        database_id.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    expected = "psk-" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    driver = LivingMemoryV8Driver()

    assert driver.derive_access_key(root_secret, database_id) == expected
    assert driver.verify_access_key(root_secret, database_id, expected)
    assert not driver.verify_access_key(root_secret, database_id, "pkb-" + expected[4:])


def test_text_media_v1_declares_read_only_adapter_access_capability():
    driver = TextMediaV1Driver()
    assert "adapter_access" in driver.descriptor.capabilities
    assert driver.descriptor.key_prefix == "pkb-"


def test_generic_database_manager_has_no_livingmemory_storage_dependencies():
    source = (
        Path(__file__).resolve().parents[1] / "personalityrag" / "libraries.py"
    ).read_text(encoding="utf-8")
    for implementation_detail in (
        "Storage",
        "IndexManager",
        "livingmemory.db",
        "livingmemory_import",
        "index_rebuild",
        "graph_rebuild",
    ):
        assert implementation_detail not in source
    assert "database_type_registry" in source
    assert "create_manager" in source


@pytest.mark.asyncio
async def test_control_catalog_backfills_old_libraries_and_allows_cross_type_ids(
    tmp_path: Path,
):
    path = tmp_path / "personalityrag_system.db"
    with sqlite3.connect(path) as db:
        db.execute(
            """CREATE TABLE libraries (
            id TEXT PRIMARY KEY,name TEXT NOT NULL,description TEXT NOT NULL DEFAULT '',
            default_persona_id TEXT NOT NULL DEFAULT '',is_default INTEGER NOT NULL DEFAULT 0,
            provider_id TEXT NOT NULL,provider_revision INTEGER NOT NULL,
            rerank_provider_id TEXT,conversation_config_json TEXT NOT NULL DEFAULT '{}',
            recall_config_json TEXT NOT NULL DEFAULT '{}',maintenance_config_json TEXT NOT NULL DEFAULT '{}',
            metadata_json TEXT NOT NULL DEFAULT '{}',deleted_at REAL,created_at REAL NOT NULL,
            updated_at REAL NOT NULL)"""
        )
        db.execute(
            """INSERT INTO libraries
            (id,name,is_default,provider_id,provider_revision,metadata_json,created_at,updated_at)
            VALUES('legacy','Legacy',1,'seed',1,'{"livingmemory_database_version":8,"keep":"value"}',1,1)"""
        )

    control = ControlStore(path)
    await control.initialize(ProviderConfig(id="seed", dimensions=8))
    try:
        legacy_ref = DatabaseRef(LIVINGMEMORY_V8_TYPE, "legacy")
        assert (await control.database_identity(legacy_ref))[
            "database_category"
        ] == "memory"
        legacy = await control.get_library("legacy")
        assert legacy is not None
        assert legacy.metadata == {"keep": "value"}
        with sqlite3.connect(path) as db:
            persisted_metadata = json.loads(
                db.execute(
                    "SELECT metadata_json FROM libraries WHERE id='legacy'"
                ).fetchone()[0]
            )
        assert persisted_metadata == {"keep": "value"}

        memory_ref = DatabaseRef("future_memory", "shared")
        knowledge_ref = DatabaseRef("future_knowledge", "shared")
        await control.register_database_identity(memory_ref, category="memory")
        await control.register_database_identity(knowledge_ref, category="knowledge")
        with pytest.raises(ValueError, match="already exists"):
            await control.register_database_identity(memory_ref, category="memory")

        db = await control.connect()
        try:
            await db.execute(
                """INSERT INTO libraries
                (id,name,is_default,provider_id,provider_revision,created_at,updated_at)
                VALUES('rollback_created','Rollback Created',0,'seed',1,2,2)"""
            )
            await db.commit()
        finally:
            await db.close()
        assert await control.database_identity(
            DatabaseRef(LIVINGMEMORY_V8_TYPE, "rollback_created")
        )
    finally:
        await control.close()


@pytest.mark.asyncio
async def test_typed_routes_reject_wrong_type_prefix_and_legacy_paths(tmp_path: Path):
    config = AppConfig(
        api_key="admin-key",
        session_secret="session-key",
        library_psk_secret="typed-root-secret",
    )
    context = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "state",
        config=config,
        configure_logs=False,
    )
    await context.manager.initialize()
    try:
        app = create_app(context)
        access_key = LivingMemoryV8Driver().derive_access_key(
            config.library_psk_secret, "Default"
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test:8765"
        ) as client:
            typed = await client.get(
                "/api/v1/memory-libraries/livingmemory_v8/Default",
                headers={"Authorization": f"Bearer {access_key}"},
            )
            wrong_type = await client.get(
                "/api/v1/memory-libraries/not_registered/Default",
                headers={"Authorization": f"Bearer {access_key}"},
            )
            wrong_prefix = await client.get(
                "/api/v1/memory-libraries/livingmemory_v8/Default",
                headers={"Authorization": f"Bearer pkb-{access_key[4:]}"},
            )
            legacy = await client.get(
                "/api/v1/libraries/Default",
                headers={"Authorization": "Bearer admin-key"},
            )
            generic_business = await client.get(
                "/api/v1/databases/livingmemory_v8/Default",
                headers={"Authorization": "Bearer admin-key"},
            )
            types = await client.get(
                "/api/v1/database-types?category=memory",
                headers={"Authorization": "Bearer admin-key"},
            )
            duplicate = await client.post(
                "/api/v1/memory-libraries/livingmemory_v8",
                headers={"Authorization": "Bearer admin-key"},
                json={
                    "id": "Default",
                    "name": "Duplicate",
                    "provider_id": config.provider.id,
                },
            )

        assert typed.status_code == 200
        assert wrong_type.status_code == 404
        assert wrong_prefix.status_code == 401
        assert legacy.status_code == 404
        assert generic_business.status_code == 404
        assert types.status_code == 200
        assert types.json()["items"][0]["id"] == "livingmemory_v8"
        typed_payload = typed.json()
        assert typed_payload["type_metadata"]["display_name"] == "LivingMemory v8"
        assert "compatibility" not in typed_payload
        assert "livingmemory_database_version" not in typed_payload["metadata"]
        assert duplicate.status_code == 409
    finally:
        await context.manager.close()
