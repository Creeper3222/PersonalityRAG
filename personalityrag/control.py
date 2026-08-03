from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiosqlite

from .config import ConversationConfig, MaintenanceConfig, ProviderConfig, RecallConfig
from .database_types import (
    DATABASE_CATEGORIES,
    DATABASE_CATEGORY_MEMORY,
    LIVINGMEMORY_V8_TYPE,
    DatabaseRef,
    database_identity_fields,
    database_type_registry,
)
from .identifiers import validate_identifier
from .providers import (
    PROVIDER_TEMPLATES,
    config_from_dict,
    masked_config,
    provider_kind,
    provider_config_hash,
)
from .version import VERSION
from .task_types import task_type_registry
from .repositories import (
    AdapterRepository,
    JobRepository,
    MemoryStoreRepository,
    ProviderRepository,
    SnapshotRepository,
)
from .sqlite_pool import SQLiteConnectionPool

ADAPTER_CONNECTION_TTL_SECONDS = 180.0
OBSOLETE_MEMORY_STORE_METADATA_KEYS = frozenset(
    {"livingmemory_database_version"}
)
# Deprecated symbol alias retained for external imports.
OBSOLETE_LIBRARY_METADATA_KEYS = OBSOLETE_MEMORY_STORE_METADATA_KEYS


class AdapterForcedOfflineError(ValueError):
    def __init__(self, connection: dict[str, Any]):
        super().__init__("连接被强制切断")
        self.connection = connection


class AdapterConnectionChangedError(ValueError):
    pass


@dataclass(slots=True)
class ProviderRevision:
    provider_id: str
    revision: int
    config: ProviderConfig
    config_sha256: str
    created_at: float

    def public(self) -> dict[str, Any]:
        return {
            **masked_config(self.config),
            "revision": self.revision,
            "config_sha256": self.config_sha256,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class MemoryStoreRecord:
    id: str
    database_type: str
    name: str
    description: str
    default_persona_id: str
    is_default: bool
    provider_id: str
    provider_revision: int
    rerank_provider_id: str
    conversation_settings: dict[str, Any]
    recall_settings: dict[str, Any]
    maintenance_settings: dict[str, Any]
    metadata: dict[str, Any]
    created_at: float
    updated_at: float

    def public(self) -> dict[str, Any]:
        return {
            **asdict(self),
            **database_identity_fields(DatabaseRef(self.database_type, self.id)),
        }


# Deprecated import alias retained for third-party code and old tests.
LibraryRecord = MemoryStoreRecord


class ControlStore:
    def __init__(self, path: Path):
        self.path = path
        self.pool = SQLiteConnectionPool(path, size=2)
        async def connect():
            return await self.connect()

        self.provider_repository = ProviderRepository(connect)
        self.memory_store_repository = MemoryStoreRepository(connect)
        # Deprecated compatibility attribute for extensions built before the
        # memory-store naming migration.
        self.library_repository = self.memory_store_repository
        self.adapter_repository = AdapterRepository(connect)
        self.job_repository = JobRepository(connect)
        self.snapshot_repository = SnapshotRepository(connect)

    @staticmethod
    def _provider_functional_payload(config: ProviderConfig) -> dict[str, Any]:
        payload = asdict(config)
        payload.pop("display_name", None)
        payload.pop("context_length_mode", None)
        payload.pop("max_context_tokens_source", None)
        payload.pop("index_rebuild_settings", None)
        payload.pop("batch_size", None)
        payload.pop("concurrency", None)
        payload.pop("max_retries", None)
        return payload

    @classmethod
    def _provider_configs_functionally_equal(
        cls, left: ProviderConfig, right: ProviderConfig
    ) -> bool:
        return cls._provider_functional_payload(left) == cls._provider_functional_payload(
            right
        )

    @staticmethod
    def _validate_memory_store_id(memory_store_id: str) -> str:
        return validate_identifier(memory_store_id, field="记忆库 ID")

    # Deprecated compatibility alias.
    _validate_library_id = _validate_memory_store_id

    @staticmethod
    def _merge_settings(defaults: dict[str, Any], *parts: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(defaults)
        for part in parts:
            if not isinstance(part, dict):
                continue
            merged.update({key: value for key, value in part.items() if key in defaults})
        return merged

    @classmethod
    def _conversation_settings(
        cls, *parts: dict[str, Any] | None
    ) -> dict[str, Any]:
        return cls._merge_settings(asdict(ConversationConfig()), *parts)

    @classmethod
    def _recall_settings(cls, *parts: dict[str, Any] | None) -> dict[str, Any]:
        return cls._merge_settings(asdict(RecallConfig()), *parts)

    @classmethod
    def _maintenance_settings(
        cls, *parts: dict[str, Any] | None
    ) -> dict[str, Any]:
        return cls._merge_settings(asdict(MaintenanceConfig()), *parts)

    async def connect(self):
        return await self.pool.acquire()

    async def close(self) -> None:
        await self.pool.close()

    async def initialize(self, seed_provider: ProviderConfig | None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = await self.connect()
        try:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS providers (
                    id TEXT PRIMARY KEY,
                    latest_revision INTEGER NOT NULL,
                    deleted_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS provider_revisions (
                    provider_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    config_json TEXT NOT NULL,
                    config_sha256 TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(provider_id, revision),
                    FOREIGN KEY(provider_id) REFERENCES providers(id)
                );
                CREATE TABLE IF NOT EXISTS libraries (
                    id TEXT PRIMARY KEY,
                    database_type TEXT NOT NULL DEFAULT 'livingmemory_v8',
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    default_persona_id TEXT NOT NULL DEFAULT '',
                    is_default INTEGER NOT NULL DEFAULT 0,
                    provider_id TEXT NOT NULL,
                    provider_revision INTEGER NOT NULL,
                    rerank_provider_id TEXT,
                    conversation_config_json TEXT NOT NULL DEFAULT '{}',
                    recall_config_json TEXT NOT NULL DEFAULT '{}',
                    maintenance_config_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    deleted_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_libraries_default
                ON libraries(is_default) WHERE is_default=1 AND deleted_at IS NULL;
                CREATE TABLE IF NOT EXISTS database_catalog (
                    database_type TEXT NOT NULL,
                    id TEXT NOT NULL,
                    database_category TEXT NOT NULL,
                    deleted_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(database_type,id)
                );
                CREATE TABLE IF NOT EXISTS library_generation_bindings (
                    library_id TEXT NOT NULL,
                    database_type TEXT NOT NULL DEFAULT 'livingmemory_v8',
                    database_id TEXT,
                    generation TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    provider_revision INTEGER NOT NULL,
                    manifest_json TEXT NOT NULL,
                    activated_at REAL NOT NULL,
                    PRIMARY KEY(library_id, generation)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    library_id TEXT,
                    database_type TEXT NOT NULL DEFAULT 'livingmemory_v8',
                    database_id TEXT,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    result TEXT,
                    error TEXT,
                    operation TEXT,
                    checkpoint TEXT,
                    status_reason TEXT NOT NULL DEFAULT '',
                    control_requested TEXT,
                    resumable INTEGER NOT NULL DEFAULT 0,
                    started_at REAL,
                    finished_at REAL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    status_history TEXT NOT NULL DEFAULT '[]',
                    database_state_before TEXT,
                    database_state_after TEXT,
                    database_state_capture_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS migration_runs (
                    id TEXT PRIMARY KEY,
                    library_id TEXT,
                    database_type TEXT NOT NULL DEFAULT 'livingmemory_v8',
                    database_id TEXT,
                    source_path TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    report_path TEXT,
                    created_at REAL NOT NULL,
                    completed_at REAL
                );
                CREATE TABLE IF NOT EXISTS index_generations (
                    library_id TEXT NOT NULL DEFAULT '',
                    database_type TEXT NOT NULL DEFAULT 'livingmemory_v8',
                    database_id TEXT,
                    generation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    manifest TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    activated_at REAL,
                    PRIMARY KEY(library_id, generation)
                );
                CREATE TABLE IF NOT EXISTS adapter_connections (
                    library_id TEXT NOT NULL,
                    database_type TEXT NOT NULL DEFAULT 'livingmemory_v8',
                    database_id TEXT,
                    adapter_id TEXT NOT NULL,
                    instance_id TEXT NOT NULL,
                    adapter_type TEXT NOT NULL DEFAULT 'unknown',
                    connected_at REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    state TEXT NOT NULL DEFAULT 'active',
                    disconnected_at REAL,
                    disconnect_reason TEXT,
                    PRIMARY KEY(library_id, adapter_id)
                );
                CREATE TABLE IF NOT EXISTS database_adapter_connections (
                    database_type TEXT NOT NULL,
                    database_id TEXT NOT NULL,
                    adapter_id TEXT NOT NULL,
                    instance_id TEXT NOT NULL,
                    adapter_type TEXT NOT NULL DEFAULT 'unknown',
                    connected_at REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    state TEXT NOT NULL DEFAULT 'active',
                    disconnected_at REAL,
                    disconnect_reason TEXT,
                    PRIMARY KEY(database_type,database_id,adapter_id)
                );
                """
            )
            await self._ensure_column(db, "jobs", "library_id", "TEXT")
            await self._ensure_column(db, "migration_runs", "library_id", "TEXT")
            await self._ensure_column(
                db,
                "index_generations",
                "library_id",
                "TEXT NOT NULL DEFAULT ''",
            )
            for table in (
                "library_generation_bindings",
                "jobs",
                "migration_runs",
                "index_generations",
                "adapter_connections",
            ):
                await self._ensure_column(
                    db,
                    table,
                    "database_type",
                    "TEXT NOT NULL DEFAULT 'livingmemory_v8'",
                )
                await self._ensure_column(db, table, "database_id", "TEXT")
                await db.execute(
                    f"UPDATE {table} SET database_id=library_id "
                    "WHERE database_id IS NULL OR database_id=''"
                )
            await self._ensure_column(db, "jobs", "operation", "TEXT")
            await self._ensure_column(db, "jobs", "checkpoint", "TEXT")
            await self._ensure_column(
                db, "jobs", "status_reason", "TEXT NOT NULL DEFAULT ''"
            )
            await self._ensure_column(db, "jobs", "control_requested", "TEXT")
            await self._ensure_column(
                db, "jobs", "resumable", "INTEGER NOT NULL DEFAULT 0"
            )
            await self._ensure_column(db, "jobs", "started_at", "REAL")
            await self._ensure_column(db, "jobs", "finished_at", "REAL")
            await self._ensure_column(
                db, "jobs", "attempt_count", "INTEGER NOT NULL DEFAULT 0"
            )
            await self._ensure_column(
                db, "jobs", "status_history", "TEXT NOT NULL DEFAULT '[]'"
            )
            await self._ensure_column(
                db, "jobs", "database_state_before", "TEXT"
            )
            await self._ensure_column(
                db, "jobs", "database_state_after", "TEXT"
            )
            await self._ensure_column(
                db, "jobs", "database_state_capture_error", "TEXT"
            )
            await self._ensure_column(db, "libraries", "rerank_provider_id", "TEXT")
            await self._ensure_column(
                db, "libraries", "conversation_config_json", "TEXT NOT NULL DEFAULT '{}'"
            )
            await self._ensure_column(
                db,
                "index_generations",
                "library_id",
                "TEXT NOT NULL DEFAULT ''",
            )
            await self._ensure_column(
                db,
                "adapter_connections",
                "state",
                "TEXT NOT NULL DEFAULT 'active'",
            )
            await self._ensure_column(
                db,
                "adapter_connections",
                "disconnected_at",
                "REAL",
            )
            await self._ensure_column(
                db,
                "adapter_connections",
                "disconnect_reason",
                "TEXT",
            )
            await db.execute(
                """INSERT INTO database_adapter_connections
                (database_type,database_id,adapter_id,instance_id,adapter_type,
                 connected_at,last_seen,state,disconnected_at,disconnect_reason)
                SELECT database_type,database_id,adapter_id,instance_id,adapter_type,
                       connected_at,last_seen,state,disconnected_at,disconnect_reason
                FROM adapter_connections WHERE true
                ON CONFLICT(database_type,database_id,adapter_id) DO UPDATE SET
                  instance_id=excluded.instance_id,
                  adapter_type=excluded.adapter_type,
                  connected_at=excluded.connected_at,
                  last_seen=excluded.last_seen,
                  state=excluded.state,
                  disconnected_at=excluded.disconnected_at,
                  disconnect_reason=excluded.disconnect_reason"""
            )
            await db.execute(
                "INSERT OR REPLACE INTO schema_info(key,value) VALUES('service_version',?)",
                (VERSION,),
            )
            columns = {
                row["name"]
                for row in await (
                    await db.execute("PRAGMA table_info(libraries)")
                ).fetchall()
            }
            if "recall_config_json" not in columns:
                await db.execute(
                    "ALTER TABLE libraries ADD COLUMN recall_config_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "conversation_config_json" not in columns:
                await db.execute(
                    "ALTER TABLE libraries ADD COLUMN conversation_config_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "maintenance_config_json" not in columns:
                await db.execute(
                    "ALTER TABLE libraries ADD COLUMN maintenance_config_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "metadata_json" not in columns:
                await db.execute(
                    "ALTER TABLE libraries ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "database_type" not in columns:
                await db.execute(
                    "ALTER TABLE libraries ADD COLUMN database_type "
                    "TEXT NOT NULL DEFAULT 'livingmemory_v8'"
                )
            await self._install_database_catalog(db)
            await self._remove_obsolete_library_metadata(db)
            await db.commit()
        finally:
            await db.close()
        if seed_provider is not None:
            await self.seed_provider(seed_provider)

    async def _provider_exists_any(self, provider_id: str) -> bool:
        db = await self.connect()
        try:
            row = await (
                await db.execute("SELECT 1 FROM providers WHERE id=?", (provider_id,))
            ).fetchone()
            return row is not None
        finally:
            await db.close()

    async def _purge_deleted_provider(self, provider_id: str) -> bool:
        if await self.provider_usage(provider_id):
            return False
        db = await self.connect()
        try:
            row = await (
                await db.execute(
                    "SELECT deleted_at FROM providers WHERE id=?",
                    (provider_id,),
                )
            ).fetchone()
            if not row or row["deleted_at"] is None:
                return False
            await db.execute(
                "DELETE FROM provider_revisions WHERE provider_id=?",
                (provider_id,),
            )
            await db.execute(
                "DELETE FROM providers WHERE id=? AND deleted_at IS NOT NULL",
                (provider_id,),
            )
            await db.commit()
            return True
        finally:
            await db.close()

    @staticmethod
    async def _ensure_column(
        db: aiosqlite.Connection, table: str, name: str, sql_type: str
    ) -> None:
        rows = await (await db.execute(f"PRAGMA table_info({table})")).fetchall()
        if name not in {row["name"] for row in rows}:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")

    async def _remove_obsolete_library_metadata(
        self, db: aiosqlite.Connection
    ) -> None:
        rows = await (
            await db.execute("SELECT id,metadata_json FROM libraries")
        ).fetchall()
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue
            if OBSOLETE_MEMORY_STORE_METADATA_KEYS.intersection(metadata):
                for key in OBSOLETE_MEMORY_STORE_METADATA_KEYS:
                    metadata.pop(key, None)
                await db.execute(
                    "UPDATE libraries SET metadata_json=? WHERE id=?",
                    (json.dumps(metadata, ensure_ascii=False), row["id"]),
                )

    async def _install_database_catalog(self, db: aiosqlite.Connection) -> None:
        await db.execute(
            """INSERT INTO database_catalog
            (database_type,id,database_category,deleted_at,created_at,updated_at)
            SELECT database_type,id,?,deleted_at,created_at,updated_at FROM libraries
            WHERE true
            ON CONFLICT(database_type,id) DO UPDATE SET
              database_category=excluded.database_category,
              deleted_at=excluded.deleted_at,
              created_at=excluded.created_at,
              updated_at=excluded.updated_at""",
            (DATABASE_CATEGORY_MEMORY,),
        )
        await db.executescript(
            f"""
            DROP TRIGGER IF EXISTS trg_libraries_catalog_insert;
            DROP TRIGGER IF EXISTS trg_libraries_catalog_update;
            DROP TRIGGER IF EXISTS trg_libraries_catalog_delete;
            CREATE TRIGGER trg_libraries_catalog_insert AFTER INSERT ON libraries
            BEGIN
              INSERT INTO database_catalog
              (database_type,id,database_category,deleted_at,created_at,updated_at)
              VALUES(NEW.database_type,NEW.id,'{DATABASE_CATEGORY_MEMORY}',
                     NEW.deleted_at,NEW.created_at,NEW.updated_at)
              ON CONFLICT(database_type,id) DO UPDATE SET
                database_category=excluded.database_category,
                deleted_at=excluded.deleted_at,
                created_at=excluded.created_at,
                updated_at=excluded.updated_at;
            END;
            CREATE TRIGGER trg_libraries_catalog_update AFTER UPDATE ON libraries
            BEGIN
              DELETE FROM database_catalog
              WHERE database_type=OLD.database_type AND id=OLD.id
                AND (OLD.database_type != NEW.database_type OR OLD.id != NEW.id);
              INSERT INTO database_catalog
              (database_type,id,database_category,deleted_at,created_at,updated_at)
              VALUES(NEW.database_type,NEW.id,'{DATABASE_CATEGORY_MEMORY}',
                     NEW.deleted_at,NEW.created_at,NEW.updated_at)
              ON CONFLICT(database_type,id) DO UPDATE SET
                database_category=excluded.database_category,
                deleted_at=excluded.deleted_at,
                created_at=excluded.created_at,
                updated_at=excluded.updated_at;
            END;
            CREATE TRIGGER trg_libraries_catalog_delete AFTER DELETE ON libraries
            BEGIN
              DELETE FROM database_catalog
              WHERE database_type=OLD.database_type AND id=OLD.id;
            END;
            """
        )

    async def register_database_identity(
        self,
        ref: DatabaseRef,
        *,
        category: str,
        created_at: float | None = None,
        updated_at: float | None = None,
    ) -> dict[str, Any]:
        if category not in DATABASE_CATEGORIES:
            raise ValueError(f"unsupported database category: {category}")
        created = float(created_at or time.time())
        updated = float(updated_at or created)
        db = await self.connect()
        try:
            await db.execute(
                """INSERT INTO database_catalog
                (database_type,id,database_category,deleted_at,created_at,updated_at)
                VALUES(?,?,?,NULL,?,?)""",
                (ref.database_type, ref.id, category, created, updated),
            )
            await db.commit()
        except aiosqlite.IntegrityError as exc:
            raise ValueError(f"database already exists: {ref.key}") from exc
        finally:
            await db.close()
        result = await self.database_identity(ref)
        if result is None:
            raise RuntimeError(f"database registration was not persisted: {ref.key}")
        return result

    async def register_database_identities(
        self,
        refs: list[DatabaseRef],
        *,
        category: str,
    ) -> list[dict[str, Any]]:
        if category not in DATABASE_CATEGORIES:
            raise ValueError(f"invalid database category: {category}")
        unique = {(ref.database_type, ref.id): ref for ref in refs}
        if len(unique) != len(refs):
            raise ValueError("duplicate database identity in batch")
        now = time.time()
        db = await self.connect()
        try:
            await db.execute("BEGIN IMMEDIATE")
            for ref in refs:
                await db.execute(
                    """INSERT INTO database_catalog
                    (database_type,id,database_category,deleted_at,created_at,updated_at)
                    VALUES(?,?,?,NULL,?,?)""",
                    (ref.database_type, ref.id, category, now, now),
                )
            await db.commit()
        except aiosqlite.IntegrityError as exc:
            await db.rollback()
            raise ValueError("database identity conflict in batch") from exc
        except Exception:
            await db.rollback()
            raise
        finally:
            await db.close()
        return [item for ref in refs if (item := await self.database_identity(ref))]

    async def database_identity(self, ref: DatabaseRef) -> dict[str, Any] | None:
        db = await self.connect()
        try:
            row = await (
                await db.execute(
                    """SELECT database_type,id,database_category,deleted_at,
                    created_at,updated_at FROM database_catalog
                    WHERE database_type=? AND id=?""",
                    (ref.database_type, ref.id),
                )
            ).fetchone()
            return dict(row) if row else None
        finally:
            await db.close()

    async def delete_database_identity(self, ref: DatabaseRef) -> None:
        db = await self.connect()
        try:
            await db.execute(
                "DELETE FROM database_catalog WHERE database_type=? AND id=?",
                (ref.database_type, ref.id),
            )
            await db.commit()
        finally:
            await db.close()

    async def rename_database_identity(self, ref: DatabaseRef, next_id: str) -> DatabaseRef:
        next_ref = DatabaseRef(ref.database_type, next_id)
        driver = database_type_registry.require(ref.database_type)
        current_key = driver.resource_key(ref.id)
        next_key = driver.resource_key(next_ref.id)
        now = time.time()
        db = await self.connect()
        try:
            await db.execute("BEGIN IMMEDIATE")
            current = await (
                await db.execute(
                    "SELECT 1 FROM database_catalog WHERE database_type=? AND id=? AND deleted_at IS NULL",
                    (ref.database_type, ref.id),
                )
            ).fetchone()
            if not current:
                raise KeyError(ref.key)
            occupied = await (
                await db.execute(
                    "SELECT 1 FROM database_catalog WHERE database_type=? AND id=?",
                    (next_ref.database_type, next_ref.id),
                )
            ).fetchone()
            if occupied:
                raise ValueError(f"database already exists: {next_ref.key}")
            await db.execute(
                "UPDATE database_catalog SET id=?,updated_at=? WHERE database_type=? AND id=?",
                (next_ref.id, now, ref.database_type, ref.id),
            )
            await db.execute(
                """UPDATE database_adapter_connections SET database_id=?
                WHERE database_type=? AND database_id=?""",
                (next_ref.id, ref.database_type, ref.id),
            )
            for table in ("jobs", "migration_runs", "index_generations", "library_generation_bindings"):
                columns = {
                    row["name"]
                    for row in await (await db.execute(f"PRAGMA table_info({table})")).fetchall()
                }
                if not {"database_type", "database_id"}.issubset(columns):
                    continue
                assignments = "database_id=?"
                params: list[Any] = [next_ref.id]
                if "library_id" in columns:
                    assignments += ",library_id=?"
                    params.append(next_key)
                params.extend((ref.database_type, ref.id))
                await db.execute(
                    f"UPDATE {table} SET {assignments} WHERE database_type=? AND database_id=?",
                    tuple(params),
                )
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        finally:
            await db.close()
        return next_ref

    async def delete_database_identities_by_type(self, database_type: str) -> None:
        db = await self.connect()
        try:
            await db.execute(
                "DELETE FROM database_catalog WHERE database_type=?",
                (database_type,),
            )
            await db.commit()
        finally:
            await db.close()

    async def list_database_identities(
        self,
        *,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        db = await self.connect()
        try:
            where = "" if include_deleted else "WHERE deleted_at IS NULL"
            rows = await (
                await db.execute(
                    f"""SELECT database_type,id,database_category,deleted_at,
                    created_at,updated_at FROM database_catalog {where}
                    ORDER BY database_type,id"""
                )
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            await db.close()
    @staticmethod
    def _clean_library_metadata(value: dict[str, Any]) -> dict[str, Any]:
        data = dict(value)
        for key in OBSOLETE_MEMORY_STORE_METADATA_KEYS:
            data.pop(key, None)
        return data

    @classmethod
    def _decode_library_metadata(cls, value: Any) -> dict[str, Any]:
        try:
            data = json.loads(value or "{}")
        except (TypeError, json.JSONDecodeError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        return cls._clean_library_metadata(data)

    @staticmethod
    def _rewrite_manifest_provider_id(
        manifest_text: str | None, old_provider_id: str, new_provider_id: str
    ) -> str:
        try:
            manifest = json.loads(manifest_text or "{}")
        except json.JSONDecodeError:
            return manifest_text or "{}"
        if not isinstance(manifest, dict):
            return manifest_text or "{}"
        if manifest.get("provider_id") != old_provider_id:
            return manifest_text or "{}"
        manifest["provider_id"] = new_provider_id
        return json.dumps(manifest, ensure_ascii=False)

    async def seed_provider(self, config: ProviderConfig) -> None:
        db = await self.connect()
        try:
            count = int(
                (await (await db.execute("SELECT COUNT(*) FROM providers")).fetchone())[0]
            )
            if count:
                return
            now = time.time()
            await db.execute(
                """INSERT INTO providers(id,latest_revision,created_at,updated_at)
                VALUES(?,1,?,?)""",
                (config.id, now, now),
            )
            await db.execute(
                """INSERT INTO provider_revisions
                (provider_id,revision,config_json,config_sha256,created_at)
                VALUES(?,1,?,?,?)""",
                (
                    config.id,
                    json.dumps(asdict(config), ensure_ascii=False),
                    provider_config_hash(config),
                    now,
                ),
            )
            await db.commit()
        finally:
            await db.close()

    async def get_provider(
        self, provider_id: str, revision: int | None = None
    ) -> ProviderRevision | None:
        row = await self.provider_repository.get_row(provider_id, revision)
        return self._provider_row(row) if row else None

    async def get_providers_bulk(
        self, bindings: set[tuple[str, int | None]]
    ) -> dict[tuple[str, int | None], ProviderRevision]:
        rows = await self.provider_repository.rows_for_bindings(bindings)
        result: dict[tuple[str, int | None], ProviderRevision] = {}
        for row in rows:
            record = self._provider_row(row)
            exact = (record.provider_id, record.revision)
            if exact in bindings:
                result[exact] = record
            latest = (record.provider_id, None)
            if (
                latest in bindings
                and record.revision == int(row["latest_revision"])
            ):
                result[latest] = record
        return result

    @staticmethod
    def _provider_row(row: aiosqlite.Row) -> ProviderRevision:
        return ProviderRevision(
            provider_id=row["provider_id"],
            revision=int(row["revision"]),
            config=config_from_dict(json.loads(row["config_json"])),
            config_sha256=row["config_sha256"],
            created_at=float(row["created_at"]),
        )

    async def provider_usage(self, provider_id: str) -> list[dict[str, Any]]:
        db = await self.connect()
        try:
            latest_row = await (
                await db.execute(
                    """SELECT pr.config_json FROM providers p JOIN provider_revisions pr
                    ON pr.provider_id=p.id AND pr.revision=p.latest_revision
                    WHERE p.id=? AND p.deleted_at IS NULL LIMIT 1""",
                    (provider_id,),
                )
            ).fetchone()
            latest_config = (
                config_from_dict(json.loads(latest_row["config_json"]))
                if latest_row
                else None
            )
            embedding_rows = await (
                await db.execute(
                    """SELECT id,database_type,name,provider_revision FROM libraries
                    WHERE provider_id=? AND deleted_at IS NULL ORDER BY name""",
                    (provider_id,),
                )
            ).fetchall()
            embedding_usage = []
            for row in embedding_rows:
                provider_revision = int(row["provider_revision"])
                bound_row = await (
                    await db.execute(
                        """SELECT config_json FROM provider_revisions
                        WHERE provider_id=? AND revision=? LIMIT 1""",
                        (provider_id, provider_revision),
                    )
                ).fetchone()
                bound_config = (
                    config_from_dict(json.loads(bound_row["config_json"]))
                    if bound_row
                    else None
                )
                needs_rebuild = (
                    True
                    if not latest_config or not bound_config
                    else not self._provider_configs_functionally_equal(
                        latest_config, bound_config
                    )
                )
                embedding_usage.append(
                    {
                        **database_identity_fields(
                            DatabaseRef(
                                str(
                                    row["database_type"]
                                    or LIVINGMEMORY_V8_TYPE
                                ),
                                str(row["id"]),
                            ),
                            include_deprecated=True,
                        ),
                        "database_name": row["name"],
                        # Deprecated compatibility alias for v0.1.1 clients.
                        "library_name": row["name"],
                        "provider_revision": provider_revision,
                        "usage_kind": "embedding",
                        "needs_rebuild": needs_rebuild,
                    }
                )
            rerank_rows = await (
                await db.execute(
                    """SELECT id,database_type,name FROM libraries
                    WHERE rerank_provider_id=? AND deleted_at IS NULL ORDER BY name""",
                    (provider_id,),
                )
            ).fetchall()
            return embedding_usage + [
                {
                    **database_identity_fields(
                        DatabaseRef(
                            str(
                                row["database_type"]
                                or LIVINGMEMORY_V8_TYPE
                            ),
                            str(row["id"]),
                        ),
                        include_deprecated=True,
                    ),
                    "database_name": row["name"],
                    # Deprecated compatibility alias for v0.1.1 clients.
                    "library_name": row["name"],
                    "provider_revision": None,
                    "usage_kind": "rerank",
                    "needs_rebuild": False,
                }
                for row in rerank_rows
            ]
        finally:
            await db.close()

    @staticmethod
    def _provider_functional_hash(config: ProviderConfig) -> str:
        payload = ControlStore._provider_functional_payload(config)
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_adapter_text(value: str, *, field: str) -> str:
        text = validate_identifier(value, field=field)
        if len(text) > 128:
            raise ValueError(f"{field} 不能超过 128 个字符")
        return text

    @staticmethod
    def _adapter_database_ref(database: str | DatabaseRef) -> DatabaseRef:
        return (
            database
            if isinstance(database, DatabaseRef)
            else DatabaseRef(LIVINGMEMORY_V8_TYPE, database)
        )

    @staticmethod
    def _adapter_connection_public(row: aiosqlite.Row) -> dict[str, Any]:
        database_id = str(row["database_id"])
        ref = DatabaseRef(
            str(row["database_type"] or LIVINGMEMORY_V8_TYPE),
            database_id,
        )
        return {
            **database_identity_fields(ref, include_deprecated=True),
            "adapter_id": row["adapter_id"],
            "instance_id": row["instance_id"],
            "adapter_type": row["adapter_type"],
            "connected_at": float(row["connected_at"]),
            "last_seen": float(row["last_seen"]),
            "state": str(row["state"] or "active"),
            "disconnected_at": (
                float(row["disconnected_at"])
                if row["disconnected_at"] is not None
                else None
            ),
            "disconnect_reason": str(row["disconnect_reason"] or ""),
        }

    async def register_adapter_connection(
        self,
        database: str | DatabaseRef,
        *,
        adapter_id: str,
        instance_id: str,
        adapter_type: str = "unknown",
        ttl_seconds: float = ADAPTER_CONNECTION_TTL_SECONDS,
        manual_reconnect: bool = False,
    ) -> dict[str, Any]:
        ref = self._adapter_database_ref(database)
        self._validate_memory_store_id(ref.id)
        record = await self.database_identity(ref)
        if not record:
            raise KeyError(ref.key)
        adapter_id = self._normalize_adapter_text(adapter_id, field="适配器标识ID")
        instance_id = self._normalize_adapter_text(instance_id, field="适配器实例ID")
        adapter_type = str(adapter_type or "unknown").strip()[:64] or "unknown"
        now = time.time()
        active_cutoff = now - float(ttl_seconds)
        db = await self.connect()
        try:
            await db.execute("BEGIN IMMEDIATE")
            row = await (
                await db.execute(
                    """SELECT * FROM database_adapter_connections
                    WHERE database_type=? AND database_id=? AND adapter_id=?""",
                    (ref.database_type, ref.id, adapter_id),
                )
            ).fetchone()
            if (
                row
                and str(row["state"] or "active") == "forced_offline"
                and not manual_reconnect
            ):
                raise AdapterForcedOfflineError(
                    self._adapter_connection_public(row)
                )
            if (
                row
                and str(row["state"] or "active") == "active"
                and str(row["instance_id"]) != instance_id
                and float(row["last_seen"] or 0) >= active_cutoff
            ):
                raise ValueError(
                    f"适配器标识ID已被其它实例占用：{adapter_id}"
                )
            same_instance = (
                row
                and str(row["state"] or "active") == "active"
                and str(row["instance_id"]) == instance_id
            )
            connected_at = float(row["connected_at"]) if same_instance else now
            await db.execute(
                """INSERT INTO database_adapter_connections
                (database_type,database_id,adapter_id,instance_id,adapter_type,connected_at,last_seen,
                 state,disconnected_at,disconnect_reason)
                VALUES(?,?,?,?,?,?,?,'active',NULL,NULL)
                ON CONFLICT(database_type,database_id,adapter_id) DO UPDATE SET
                    instance_id=excluded.instance_id,
                    adapter_type=excluded.adapter_type,
                    connected_at=excluded.connected_at,
                    last_seen=excluded.last_seen,
                    state='active',
                    disconnected_at=NULL,
                    disconnect_reason=NULL""",
                (
                    ref.database_type,
                    ref.id,
                    adapter_id,
                    instance_id,
                    adapter_type,
                    connected_at,
                    now,
                ),
            )
            if ref.database_type == LIVINGMEMORY_V8_TYPE:
                await db.execute(
                    """INSERT INTO adapter_connections
                    (library_id,database_type,database_id,adapter_id,instance_id,
                     adapter_type,connected_at,last_seen,state,disconnected_at,
                     disconnect_reason)
                    VALUES(?,?,?,?,?,?,?,?,'active',NULL,NULL)
                    ON CONFLICT(library_id,adapter_id) DO UPDATE SET
                      database_type=excluded.database_type,
                      database_id=excluded.database_id,
                      instance_id=excluded.instance_id,
                      adapter_type=excluded.adapter_type,
                      connected_at=excluded.connected_at,
                      last_seen=excluded.last_seen,
                      state='active',disconnected_at=NULL,disconnect_reason=NULL""",
                    (
                        ref.id,
                        ref.database_type,
                        ref.id,
                        adapter_id,
                        instance_id,
                        adapter_type,
                        connected_at,
                        now,
                    ),
                )
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        finally:
            await db.close()
        return (
            await self.adapter_connection(ref, adapter_id)
        ) or {
            **database_identity_fields(ref, include_deprecated=True),
            "adapter_id": adapter_id,
            "instance_id": instance_id,
            "adapter_type": adapter_type,
            "connected_at": connected_at,
            "last_seen": now,
            "state": "active",
            "disconnected_at": None,
            "disconnect_reason": "",
        }

    async def force_disconnect_adapter(
        self,
        database: str | DatabaseRef,
        adapter_id: str,
        *,
        expected_instance_id: str,
        reason: str = "forced_by_admin",
    ) -> dict[str, Any]:
        ref = self._adapter_database_ref(database)
        self._validate_memory_store_id(ref.id)
        adapter_id = self._normalize_adapter_text(adapter_id, field="适配器标识ID")
        expected_instance_id = self._normalize_adapter_text(
            expected_instance_id,
            field="适配器实例ID",
        )
        db = await self.connect()
        try:
            await db.execute("BEGIN IMMEDIATE")
            row = await (
                await db.execute(
                    """SELECT * FROM database_adapter_connections
                    WHERE database_type=? AND database_id=? AND adapter_id=?""",
                    (ref.database_type, ref.id, adapter_id),
                )
            ).fetchone()
            if not row:
                raise KeyError(adapter_id)
            if str(row["instance_id"]) != expected_instance_id:
                raise AdapterConnectionChangedError(
                    "适配器连接实例已变化，请刷新后重试"
                )
            disconnected_at = time.time()
            await db.execute(
                """UPDATE database_adapter_connections
                SET state='forced_offline',disconnected_at=?,disconnect_reason=?
                WHERE database_type=? AND database_id=? AND adapter_id=?""",
                (
                    disconnected_at,
                    str(reason or "forced_by_admin"),
                    ref.database_type,
                    ref.id,
                    adapter_id,
                ),
            )
            if ref.database_type == LIVINGMEMORY_V8_TYPE:
                await db.execute(
                    """UPDATE adapter_connections
                    SET state='forced_offline',disconnected_at=?,disconnect_reason=?
                    WHERE library_id=? AND adapter_id=?""",
                    (
                        disconnected_at,
                        str(reason or "forced_by_admin"),
                        ref.id,
                        adapter_id,
                    ),
                )
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        finally:
            await db.close()
        connection = await self.adapter_connection(ref, adapter_id)
        if not connection:
            raise KeyError(adapter_id)
        return connection

    async def adapter_connection(
        self, database: str | DatabaseRef, adapter_id: str
    ) -> dict[str, Any] | None:
        ref = self._adapter_database_ref(database)
        db = await self.connect()
        try:
            row = await (
                await db.execute(
                    """SELECT * FROM database_adapter_connections
                    WHERE database_type=? AND database_id=? AND adapter_id=?""",
                    (ref.database_type, ref.id, adapter_id),
                )
            ).fetchone()
            return self._adapter_connection_public(row) if row else None
        finally:
            await db.close()

    async def forced_adapter_connections(self) -> list[dict[str, Any]]:
        db = await self.connect()
        try:
            rows = await (
                await db.execute(
                    """SELECT * FROM database_adapter_connections
                    WHERE state='forced_offline'
                    ORDER BY database_type,database_id,adapter_id"""
                )
            ).fetchall()
            return [self._adapter_connection_public(row) for row in rows]
        finally:
            await db.close()

    async def active_adapter_connections(
        self,
        database: str | DatabaseRef,
        *,
        ttl_seconds: float = ADAPTER_CONNECTION_TTL_SECONDS,
    ) -> list[dict[str, Any]]:
        ref = self._adapter_database_ref(database)
        cutoff = time.time() - float(ttl_seconds)
        rows = await self.adapter_repository.active_database_rows([ref], cutoff=cutoff)
        return [self._adapter_connection_public(row) for row in rows]

    async def active_database_adapter_connections_map(
        self,
        databases: list[DatabaseRef],
        *,
        ttl_seconds: float = ADAPTER_CONNECTION_TTL_SECONDS,
    ) -> dict[str, list[dict[str, Any]]]:
        cutoff = time.time() - float(ttl_seconds)
        rows = await self.adapter_repository.active_database_rows(databases, cutoff=cutoff)
        result = {ref.key: [] for ref in databases}
        for row in rows:
            item = self._adapter_connection_public(row)
            result.setdefault(f"{item['database_type']}:{item['database_id']}", []).append(item)
        return result

    async def active_adapter_connections_map(
        self,
        library_ids: list[str],
        *,
        ttl_seconds: float = ADAPTER_CONNECTION_TTL_SECONDS,
    ) -> dict[str, list[dict[str, Any]]]:
        if not library_ids:
            return {}
        refs = [DatabaseRef(LIVINGMEMORY_V8_TYPE, library_id) for library_id in library_ids]
        typed = await self.active_database_adapter_connections_map(refs, ttl_seconds=ttl_seconds)
        return {library_id: typed.get(DatabaseRef(LIVINGMEMORY_V8_TYPE, library_id).key, []) for library_id in library_ids}

    @staticmethod
    def _job_public(row: aiosqlite.Row) -> dict[str, Any]:
        result = dict(row)
        result["database_resource_key"] = str(result.get("library_id") or "")
        database_id = str(result.get("database_id") or "")
        database_type = str(
            result.get("database_type") or LIVINGMEMORY_V8_TYPE
        )
        if database_id:
            result.update(
                database_identity_fields(
                    DatabaseRef(database_type, database_id),
                    include_deprecated=False,
                )
            )
        if result.get("result"):
            try:
                result["result"] = json.loads(result["result"])
            except json.JSONDecodeError:
                result["result"] = None
        return result

    @staticmethod
    def _job_resource_key(
        database: str | DatabaseRef,
        database_type: str | None = None,
    ) -> str:
        if isinstance(database, DatabaseRef):
            return database_type_registry.require(database.database_type).resource_key(
                database.id
            )
        value = str(database)
        if database_type:
            return database_type_registry.require(database_type).resource_key(value)
        prefix, separator, _identifier = value.partition(":")
        if separator:
            try:
                database_type_registry.require(prefix)
            except KeyError:
                pass
            else:
                return value
        return database_type_registry.require(LIVINGMEMORY_V8_TYPE).resource_key(value)

    async def active_long_job(
        self,
        database: str | DatabaseRef,
        *,
        database_type: str | None = None,
    ) -> dict[str, Any] | None:
        resource_key = self._job_resource_key(database, database_type)
        return (await self.active_long_jobs_map([resource_key])).get(resource_key)

    async def active_long_jobs_map(
        self, database_resource_keys: list[str]
    ) -> dict[str, dict[str, Any] | None]:
        if not database_resource_keys:
            return {}
        rows = await self.job_repository.active_long_rows(database_resource_keys)
        result: dict[str, dict[str, Any] | None] = {
            resource_key: None for resource_key in database_resource_keys
        }
        for row in rows:
            resource_key = str(row["library_id"] or "")
            if (
                resource_key
                and result.get(resource_key) is None
                and task_type_registry.get(
                    str(row["database_type"] or LIVINGMEMORY_V8_TYPE),
                    str(row["kind"] or ""),
                ).adapter_blocking
            ):
                result[resource_key] = self._job_public(row)
        return result

    async def debug_provider_revisions(self, provider_id: str) -> dict[str, Any]:
        db = await self.connect()
        try:
            provider_row = await (
                await db.execute(
                    """SELECT id,latest_revision,deleted_at,created_at,updated_at
                    FROM providers WHERE id=? LIMIT 1""",
                    (provider_id,),
                )
            ).fetchone()
            if not provider_row or provider_row["deleted_at"] is not None:
                raise KeyError(provider_id)
            revision_rows = await (
                await db.execute(
                    """SELECT provider_id,revision,config_json,config_sha256,created_at
                    FROM provider_revisions WHERE provider_id=? ORDER BY revision""",
                    (provider_id,),
                )
            ).fetchall()
            binding_rows = await (
                await db.execute(
                    """SELECT library_id,generation,provider_revision,activated_at
                    FROM library_generation_bindings
                    WHERE provider_id=? ORDER BY activated_at DESC""",
                    (provider_id,),
                )
            ).fetchall()
        finally:
            await db.close()

        latest_revision = int(provider_row["latest_revision"])
        latest_config: ProviderConfig | None = None
        parsed_revisions: list[tuple[aiosqlite.Row, ProviderConfig]] = []
        for row in revision_rows:
            config = config_from_dict(json.loads(row["config_json"]))
            parsed_revisions.append((row, config))
            if int(row["revision"]) == latest_revision:
                latest_config = config
        revisions: list[dict[str, Any]] = []
        for row, config in parsed_revisions:
            revisions.append(
                {
                    "provider_id": row["provider_id"],
                    "revision": int(row["revision"]),
                    "is_latest": int(row["revision"]) == latest_revision,
                    "config_sha256": row["config_sha256"],
                    "functional_sha256": self._provider_functional_hash(config),
                    "functionally_equal_to_latest": (
                        self._provider_configs_functionally_equal(
                            latest_config, config
                        )
                        if latest_config
                        else False
                    ),
                    "created_at": float(row["created_at"]),
                    "config": masked_config(config),
                }
            )
        return {
            "provider_id": provider_id,
            "latest_revision": latest_revision,
            "created_at": float(provider_row["created_at"]),
            "updated_at": float(provider_row["updated_at"]),
            "revisions": revisions,
            "usage": await self.provider_usage(provider_id),
            "generation_bindings": [
                {
                    "library_id": row["library_id"],
                    "generation": row["generation"],
                    "provider_revision": int(row["provider_revision"]),
                    "activated_at": float(row["activated_at"]),
                }
                for row in binding_rows
            ],
        }

    async def debug_revision_inventory(self) -> dict[str, list[dict[str, Any]]]:
        """Capture revision references from the process-global control DB.

        This is intentionally one connection-wide scan so the Debug overview
        can combine it with one scan from each typed database manager without
        repeatedly walking providers or databases.
        """

        cutoff = time.time() - ADAPTER_CONNECTION_TTL_SECONDS
        queries = {
            "providers": """SELECT id,latest_revision,deleted_at,created_at,updated_at
                FROM providers WHERE deleted_at IS NULL ORDER BY created_at,id""",
            "provider_revisions": """SELECT provider_id,revision,config_json,
                config_sha256,created_at FROM provider_revisions
                ORDER BY provider_id,revision""",
            "memory_databases": """SELECT id,database_type,name,provider_id,
                provider_revision,rerank_provider_id,created_at,updated_at
                FROM libraries WHERE deleted_at IS NULL ORDER BY created_at,id""",
            "livingmemory_generations": """SELECT library_id,database_type,
                database_id,generation,provider_id,provider_revision,
                manifest_json,activated_at FROM library_generation_bindings
                ORDER BY activated_at DESC""",
            "index_generations": """SELECT library_id,database_type,database_id,
                generation,status,manifest,created_at,activated_at
                FROM index_generations ORDER BY created_at DESC""",
            "revision_jobs": """SELECT id,library_id,database_type,database_id,
                kind,status,operation,checkpoint,created_at,updated_at FROM jobs
                WHERE checkpoint IS NOT NULL OR operation IS NOT NULL OR
                status IN ('queued','running','pausing','paused','interrupted','stopping')
                ORDER BY created_at DESC""",
            "active_adapters": """SELECT database_type,database_id,adapter_id,
                instance_id,connected_at,last_seen,state
                FROM database_adapter_connections
                WHERE state='active' AND last_seen>=?
                ORDER BY database_type,database_id,adapter_id""",
        }
        db = await self.connect()
        try:
            result: dict[str, list[dict[str, Any]]] = {}
            for name, query in queries.items():
                parameters = (cutoff,) if name == "active_adapters" else ()
                rows = await (await db.execute(query, parameters)).fetchall()
                result[name] = [dict(row) for row in rows]
            return result
        finally:
            await db.close()

    @staticmethod
    def _rewrite_revision_manifest(
        raw: str | dict[str, Any] | None,
        *,
        provider_id: str,
        revision: int,
        fingerprint: str,
    ) -> dict[str, Any]:
        if isinstance(raw, dict):
            manifest = dict(raw)
        else:
            try:
                parsed = json.loads(str(raw or "{}"))
            except json.JSONDecodeError:
                parsed = {}
            manifest = dict(parsed) if isinstance(parsed, dict) else {}
        manifest["provider_id"] = provider_id
        manifest["provider_revision"] = int(revision)
        if "provider_config_sha256" in manifest or fingerprint:
            manifest["provider_config_sha256"] = fingerprint
        if "provider_fingerprint" in manifest:
            manifest["provider_fingerprint"] = fingerprint
        capability = manifest.get("embedding_capability")
        if isinstance(capability, dict):
            capability = dict(capability)
            capability["provider_id"] = provider_id
            capability["provider_revision"] = int(revision)
            capability["provider_config_sha256"] = fingerprint
            manifest["embedding_capability"] = capability
        return manifest

    async def debug_bind_livingmemory_revision(
        self,
        database_id: str,
        *,
        provider_id: str,
        revision: int,
        fingerprint: str,
    ) -> None:
        now = time.time()
        db = await self.connect()
        try:
            await db.execute("BEGIN IMMEDIATE")
            row = await (
                await db.execute(
                    """SELECT provider_id FROM libraries WHERE id=? AND
                    database_type=? AND deleted_at IS NULL LIMIT 1""",
                    (database_id, LIVINGMEMORY_V8_TYPE),
                )
            ).fetchone()
            if row is None:
                raise KeyError(database_id)
            if str(row["provider_id"]) != provider_id:
                raise ValueError("revision repair cannot change the bound Provider ID")
            await db.execute(
                """UPDATE libraries SET provider_revision=?,updated_at=?
                WHERE id=? AND database_type=?""",
                (int(revision), now, database_id, LIVINGMEMORY_V8_TYPE),
            )
            generation_rows = await (
                await db.execute(
                    """SELECT library_id,generation,manifest_json
                    FROM library_generation_bindings WHERE database_type=?
                    AND database_id=? AND provider_id=?""",
                    (LIVINGMEMORY_V8_TYPE, database_id, provider_id),
                )
            ).fetchall()
            for generation in generation_rows:
                manifest = self._rewrite_revision_manifest(
                    generation["manifest_json"],
                    provider_id=provider_id,
                    revision=revision,
                    fingerprint=fingerprint,
                )
                await db.execute(
                    """UPDATE library_generation_bindings SET
                    provider_revision=?,manifest_json=?
                    WHERE library_id=? AND generation=?""",
                    (
                        int(revision),
                        json.dumps(manifest, ensure_ascii=False),
                        generation["library_id"],
                        generation["generation"],
                    ),
                )
            index_rows = await (
                await db.execute(
                    """SELECT library_id,generation,manifest FROM index_generations
                    WHERE database_type=? AND database_id=?""",
                    (LIVINGMEMORY_V8_TYPE, database_id),
                )
            ).fetchall()
            for generation in index_rows:
                raw_manifest = generation["manifest"]
                try:
                    parsed_manifest = json.loads(str(raw_manifest or "{}"))
                except json.JSONDecodeError:
                    parsed_manifest = {}
                manifest_provider_id = str(
                    (parsed_manifest or {}).get("provider_id") or ""
                )
                if manifest_provider_id and manifest_provider_id != provider_id:
                    continue
                manifest = self._rewrite_revision_manifest(
                    parsed_manifest,
                    provider_id=provider_id,
                    revision=revision,
                    fingerprint=fingerprint,
                )
                await db.execute(
                    """UPDATE index_generations SET manifest=?
                    WHERE library_id=? AND generation=?""",
                    (
                        json.dumps(manifest, ensure_ascii=False),
                        generation["library_id"],
                        generation["generation"],
                    ),
                )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        finally:
            await db.close()

    async def debug_patch_provider_revision(
        self,
        provider_id: str,
        revision: int,
        patch: dict[str, Any],
    ) -> dict[str, Any]:
        current = await self.get_provider(provider_id, revision)
        if not current:
            raise KeyError(provider_id)
        changes = dict(patch)
        changes.pop("provider_kind", None)
        changes.pop("has_api_key", None)
        if changes.get("api_key") == "********":
            changes.pop("api_key", None)
        if changes.pop("clear_api_key", False):
            changes["api_key"] = ""
        if "id" in changes and str(changes["id"]) != provider_id:
            raise ValueError("debug revision patch must not change provider id")
        payload = {**asdict(current.config), **changes, "id": provider_id}
        config = config_from_dict(payload)
        digest = provider_config_hash(config)
        now = time.time()
        db = await self.connect()
        try:
            await db.execute(
                """UPDATE provider_revisions
                SET config_json=?,config_sha256=?
                WHERE provider_id=? AND revision=?""",
                (
                    json.dumps(asdict(config), ensure_ascii=False),
                    digest,
                    provider_id,
                    revision,
                ),
            )
            await db.execute(
                "UPDATE providers SET updated_at=? WHERE id=?",
                (now, provider_id),
            )
            await db.commit()
        finally:
            await db.close()
        return await self.debug_provider_revisions(provider_id)

    async def debug_reset_provider_revisions(
        self,
        provider_id: str,
        *,
        latest_revision: int | None = None,
        bind_libraries_to_latest: bool = False,
        library_revisions: dict[str, int] | None = None,
        delete_revisions_after_latest: bool = False,
    ) -> dict[str, Any]:
        library_revisions = library_revisions or {}
        db = await self.connect()
        try:
            provider_row = await (
                await db.execute(
                    """SELECT latest_revision,deleted_at FROM providers
                    WHERE id=? LIMIT 1""",
                    (provider_id,),
                )
            ).fetchone()
            if not provider_row or provider_row["deleted_at"] is not None:
                raise KeyError(provider_id)
            revision_rows = await (
                await db.execute(
                    "SELECT revision FROM provider_revisions WHERE provider_id=?",
                    (provider_id,),
                )
            ).fetchall()
            known_revisions = {int(row["revision"]) for row in revision_rows}
            target_latest = int(latest_revision or provider_row["latest_revision"])
            if target_latest not in known_revisions:
                raise ValueError(f"provider revision {target_latest} does not exist")
            now = time.time()
            await db.execute(
                "UPDATE providers SET latest_revision=?,updated_at=? WHERE id=?",
                (target_latest, now, provider_id),
            )
            if bind_libraries_to_latest:
                await db.execute(
                    """UPDATE libraries SET provider_revision=?,updated_at=?
                    WHERE provider_id=? AND deleted_at IS NULL""",
                    (target_latest, now, provider_id),
                )
            for library_id, revision_value in library_revisions.items():
                revision_int = int(revision_value)
                if revision_int not in known_revisions:
                    raise ValueError(
                        f"provider revision {revision_int} does not exist"
                    )
                library_row = await (
                    await db.execute(
                        """SELECT provider_id FROM libraries
                        WHERE id=? AND deleted_at IS NULL LIMIT 1""",
                        (library_id,),
                    )
                ).fetchone()
                if not library_row:
                    raise KeyError(library_id)
                if library_row["provider_id"] != provider_id:
                    raise ValueError(
                        f"library {library_id} is not bound to provider {provider_id}"
                    )
                await db.execute(
                    """UPDATE libraries SET provider_revision=?,updated_at=?
                    WHERE id=?""",
                    (revision_int, now, library_id),
                )
            if delete_revisions_after_latest:
                active_bound_rows = await (
                    await db.execute(
                        """SELECT id,provider_revision FROM libraries
                        WHERE provider_id=? AND provider_revision>? AND deleted_at IS NULL""",
                        (provider_id, target_latest),
                    )
                ).fetchall()
                if active_bound_rows:
                    libraries = ", ".join(
                        f"{row['id']}@r{int(row['provider_revision'])}"
                        for row in active_bound_rows
                    )
                    raise ValueError(
                        "cannot delete revisions still bound by libraries: "
                        + libraries
                    )
                await db.execute(
                    "DELETE FROM provider_revisions WHERE provider_id=? AND revision>?",
                    (provider_id, target_latest),
                )
            await db.commit()
        finally:
            await db.close()
        return await self.debug_provider_revisions(provider_id)

    async def list_providers(self, kind: str | None = None) -> list[dict[str, Any]]:
        db = await self.connect()
        try:
            rows = await (
                await db.execute(
                    """SELECT pr.* FROM providers p JOIN provider_revisions pr
                    ON pr.provider_id=p.id AND pr.revision=p.latest_revision
                    WHERE p.deleted_at IS NULL ORDER BY p.created_at"""
                )
            ).fetchall()
        finally:
            await db.close()
        result = []
        for row in rows:
            record = self._provider_row(row)
            if kind and provider_kind(record.config.type) != kind:
                continue
            item = record.public()
            item["used_by"] = await self.provider_usage(record.provider_id)
            result.append(item)
        return result

    async def export_provider_snapshot(self) -> dict[str, Any]:
        provider_rows, revision_rows = (
            await self.snapshot_repository.provider_rows()
        )
        revisions = []
        for row in revision_rows:
            config = config_from_dict(json.loads(row["config_json"]))
            revisions.append(
                {
                    "provider_id": row["provider_id"],
                    "revision": int(row["revision"]),
                    "config": asdict(config),
                    "config_sha256": provider_config_hash(config),
                    "created_at": float(row["created_at"]),
                }
            )
        return {
            "providers": [
                {
                    "id": row["id"],
                    "latest_revision": int(row["latest_revision"]),
                    "created_at": float(row["created_at"]),
                    "updated_at": float(row["updated_at"]),
                }
                for row in provider_rows
            ],
            "provider_revisions": revisions,
        }

    async def restore_provider_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        providers = list(snapshot.get("providers") or [])
        revisions = list(snapshot.get("provider_revisions") or [])
        provider_ids = [
            validate_identifier(item.get("id"), field="Provider ID")
            for item in providers
        ]
        if len(set(provider_ids)) != len(provider_ids):
            raise ValueError("provider snapshot contains invalid or duplicate IDs")
        provider_id_set = set(provider_ids)
        revision_rows = []
        for item in revisions:
            provider_id = validate_identifier(
                item.get("provider_id"), field="Provider ID"
            )
            if provider_id not in provider_id_set:
                continue
            config = config_from_dict(dict(item.get("config") or {}))
            if config.id != provider_id:
                raise ValueError(f"provider config ID mismatch: {provider_id}")
            revision = int(item.get("revision") or 0)
            if revision <= 0:
                raise ValueError(f"invalid provider revision: {provider_id}")
            revision_rows.append(
                {
                    "provider_id": provider_id,
                    "revision": revision,
                    "config_json": json.dumps(asdict(config), ensure_ascii=False),
                    "config_sha256": provider_config_hash(config),
                    "created_at": float(item.get("created_at") or time.time()),
                }
            )
        revisions_by_provider: dict[str, set[int]] = {}
        for row in revision_rows:
            revisions_by_provider.setdefault(row["provider_id"], set()).add(
                int(row["revision"])
            )
        for provider in providers:
            provider_id = validate_identifier(provider.get("id"), field="Provider ID")
            latest_revision = int(provider.get("latest_revision") or 0)
            if latest_revision not in revisions_by_provider.get(provider_id, set()):
                raise ValueError(f"provider {provider_id} missing latest revision")
        db = await self.connect()
        try:
            await db.execute("PRAGMA foreign_keys=OFF")
            await db.execute("BEGIN IMMEDIATE")
            await db.execute("DELETE FROM provider_revisions")
            await db.execute("DELETE FROM providers")
            for provider in providers:
                await db.execute(
                    """INSERT INTO providers
                    (id,latest_revision,deleted_at,created_at,updated_at)
                    VALUES(?,?,NULL,?,?)""",
                    (
                        validate_identifier(provider["id"], field="Provider ID"),
                        int(provider["latest_revision"]),
                        float(provider.get("created_at") or time.time()),
                        float(provider.get("updated_at") or time.time()),
                    ),
                )
            for row in revision_rows:
                await db.execute(
                    """INSERT INTO provider_revisions
                    (provider_id,revision,config_json,config_sha256,created_at)
                    VALUES(?,?,?,?,?)""",
                    (
                        row["provider_id"],
                        row["revision"],
                        row["config_json"],
                        row["config_sha256"],
                        row["created_at"],
                    ),
                )
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        finally:
            await db.close()
        return {"providers": len(providers), "provider_revisions": len(revision_rows)}

    async def export_library_snapshot(self) -> dict[str, Any]:
        rows = await self.snapshot_repository.library_rows()
        return {
            "libraries": [self._library_row(row).public() for row in rows]
        }

    async def restore_library_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        libraries = list(snapshot.get("libraries") or [])
        if not libraries:
            raise ValueError("library snapshot is empty")
        ids = [
            self._validate_memory_store_id(str(item.get("id") or ""))
            for item in libraries
        ]
        if len(set(ids)) != len(ids):
            raise ValueError("library snapshot contains duplicate IDs")
        default_ids = [str(item.get("id")) for item in libraries if bool(item.get("is_default"))]
        default_id = default_ids[0] if default_ids else ids[0]
        now = time.time()
        db = await self.connect()
        try:
            await db.execute("PRAGMA foreign_keys=OFF")
            await db.execute("BEGIN IMMEDIATE")
            for table in (
                "library_generation_bindings",
                "index_generations",
                "adapter_connections",
                "database_adapter_connections",
                "migration_runs",
                "jobs",
                "libraries",
            ):
                await db.execute(f"DELETE FROM {table}")
            for item in libraries:
                library_id = self._validate_memory_store_id(
                    str(item.get("id") or "")
                )
                provider_id = validate_identifier(
                    item.get("provider_id"), field="Provider ID"
                )
                rerank_provider_id = str(item.get("rerank_provider_id") or "")
                if rerank_provider_id:
                    rerank_provider_id = validate_identifier(
                        rerank_provider_id, field="Provider ID"
                    )
                await db.execute(
                    """INSERT INTO libraries
                    (id,database_type,name,description,default_persona_id,is_default,provider_id,
                     provider_revision,rerank_provider_id,conversation_config_json,
                     recall_config_json,maintenance_config_json,metadata_json,
                     deleted_at,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?,?)""",
                    (
                        library_id,
                        str(item.get("database_type") or LIVINGMEMORY_V8_TYPE),
                        str(item.get("name") or library_id),
                        str(item.get("description") or ""),
                        str(item.get("default_persona_id") or ""),
                        1 if library_id == default_id else 0,
                        provider_id,
                        int(item.get("provider_revision") or 1),
                        rerank_provider_id or None,
                        json.dumps(
                            self._conversation_settings(
                                item.get("conversation_settings")
                            ),
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            self._recall_settings(item.get("recall_settings")),
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            self._maintenance_settings(
                                item.get("maintenance_settings")
                            ),
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            ControlStore._clean_library_metadata(
                                dict(item.get("metadata") or {})
                            ),
                            ensure_ascii=False,
                        ),
                        float(item.get("created_at") or now),
                        float(item.get("updated_at") or now),
                    ),
                )
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        finally:
            await db.close()
        return {"libraries": len(libraries), "default_library": default_id}

    async def create_provider(self, payload: dict[str, Any]) -> ProviderRevision:
        config = config_from_dict(payload)
        validate_identifier(config.id, field="Provider ID")
        now = time.time()
        db = await self.connect()
        try:
            existing = await (
                await db.execute("SELECT id FROM providers WHERE id=?", (config.id,))
            ).fetchone()
            if existing:
                raise ValueError(f"Provider ID {config.id} 已存在")
            await db.execute(
                """INSERT INTO providers(id,latest_revision,created_at,updated_at)
                VALUES(?,1,?,?)""",
                (config.id, now, now),
            )
            digest = provider_config_hash(config)
            await db.execute(
                """INSERT INTO provider_revisions
                (provider_id,revision,config_json,config_sha256,created_at)
                VALUES(?,1,?,?,?)""",
                (
                    config.id,
                    json.dumps(asdict(config), ensure_ascii=False),
                    digest,
                    now,
                ),
            )
            await db.commit()
            return ProviderRevision(config.id, 1, config, digest, now)
        finally:
            await db.close()

    async def update_provider(
        self, provider_id: str, payload: dict[str, Any]
    ) -> ProviderRevision:
        current = await self.get_provider(provider_id)
        if not current:
            raise KeyError(provider_id)
        changes = dict(payload)
        next_provider_id = (
            validate_identifier(changes.pop("id"), field="Provider ID")
            if "id" in changes
            else provider_id
        )
        usage = (
            await self.provider_usage(provider_id)
            if next_provider_id != provider_id
            or ("enabled" in changes and not bool(changes["enabled"]))
            else []
        )
        if next_provider_id != provider_id and usage:
            raise ValueError("被记忆库引用的 Provider ID 不允许修改")
        clear_api_key = bool(changes.pop("clear_api_key", False))
        if clear_api_key:
            changes["api_key"] = ""
        config = config_from_dict(
            {**changes, "id": next_provider_id},
            base=current.config,
            keep_secret=not clear_api_key,
        )
        if not config.enabled and usage:
            raise ValueError("该 Provider 正被记忆库使用，不能停用")
        display_only_update = (
            next_provider_id == provider_id
            and self._provider_configs_functionally_equal(current.config, config)
        )
        revision = current.revision + 1
        now = time.time()
        digest = provider_config_hash(config)
        if next_provider_id != provider_id:
            await self._purge_deleted_provider(next_provider_id)
        db = await self.connect()
        try:
            if display_only_update:
                await db.execute(
                    """UPDATE provider_revisions
                    SET config_json=?,config_sha256=?
                    WHERE provider_id=? AND revision=?""",
                    (
                        json.dumps(asdict(config), ensure_ascii=False),
                        digest,
                        provider_id,
                        current.revision,
                    ),
                )
                await db.execute(
                    "UPDATE providers SET updated_at=? WHERE id=?",
                    (now, provider_id),
                )
                await db.commit()
                return ProviderRevision(
                    provider_id, current.revision, config, digest, current.created_at
                )
            if next_provider_id != provider_id:
                occupied = await (
                    await db.execute(
                        "SELECT 1 FROM providers WHERE id=? LIMIT 1",
                        (next_provider_id,),
                    )
                ).fetchone()
                if occupied:
                    raise ValueError(f"Provider ID {next_provider_id} 已存在")
                provider_row = await (
                    await db.execute(
                        "SELECT created_at FROM providers WHERE id=? LIMIT 1",
                        (provider_id,),
                    )
                ).fetchone()
                if not provider_row:
                    raise KeyError(provider_id)
                await db.execute(
                    """INSERT INTO providers(id,latest_revision,created_at,updated_at)
                    VALUES(?,?,?,?)""",
                    (
                        next_provider_id,
                        revision,
                        float(provider_row["created_at"]),
                        now,
                    ),
                )
                historical_rows = await (
                    await db.execute(
                        """SELECT revision,config_json FROM provider_revisions
                        WHERE provider_id=? ORDER BY revision""",
                        (provider_id,),
                    )
                ).fetchall()
                for row in historical_rows:
                    config_payload = json.loads(row["config_json"] or "{}")
                    if isinstance(config_payload, dict):
                        config_payload["id"] = next_provider_id
                    historical_config = config_from_dict(config_payload)
                    await db.execute(
                        """UPDATE provider_revisions
                        SET provider_id=?,config_json=?,config_sha256=?
                        WHERE provider_id=? AND revision=?""",
                        (
                            next_provider_id,
                            json.dumps(asdict(historical_config), ensure_ascii=False),
                            provider_config_hash(historical_config),
                            provider_id,
                            int(row["revision"]),
                        ),
                    )
                await db.execute(
                    "UPDATE libraries SET provider_id=? WHERE provider_id=?",
                    (next_provider_id, provider_id),
                )
                binding_rows = await (
                    await db.execute(
                        """SELECT library_id,generation,manifest_json
                        FROM library_generation_bindings WHERE provider_id=?""",
                        (provider_id,),
                    )
                ).fetchall()
                for row in binding_rows:
                    await db.execute(
                        """UPDATE library_generation_bindings
                        SET provider_id=?,manifest_json=?
                        WHERE library_id=? AND generation=?""",
                        (
                            next_provider_id,
                            self._rewrite_manifest_provider_id(
                                row["manifest_json"],
                                provider_id,
                                next_provider_id,
                            ),
                            row["library_id"],
                            row["generation"],
                        ),
                    )
                generation_rows = await (
                    await db.execute(
                        "SELECT library_id,generation,manifest FROM index_generations"
                    )
                ).fetchall()
                for row in generation_rows:
                    updated_manifest = self._rewrite_manifest_provider_id(
                        row["manifest"], provider_id, next_provider_id
                    )
                    if updated_manifest == (row["manifest"] or "{}"):
                        continue
                    await db.execute(
                        """UPDATE index_generations SET manifest=?
                        WHERE library_id=? AND generation=?""",
                        (
                            updated_manifest,
                            row["library_id"],
                            row["generation"],
                        ),
                    )
            await db.execute(
                """INSERT INTO provider_revisions
                (provider_id,revision,config_json,config_sha256,created_at)
                VALUES(?,?,?,?,?)""",
                (
                    next_provider_id,
                    revision,
                    json.dumps(asdict(config), ensure_ascii=False),
                    digest,
                    now,
                ),
            )
            if next_provider_id == provider_id:
                await db.execute(
                    "UPDATE providers SET latest_revision=?,updated_at=? WHERE id=?",
                    (revision, now, provider_id),
                )
            else:
                await db.execute("DELETE FROM providers WHERE id=?", (provider_id,))
            await db.commit()
        finally:
            await db.close()
        return ProviderRevision(next_provider_id, revision, config, digest, now)

    async def copy_provider(
        self, provider_id: str, new_id: str | None = None
    ) -> ProviderRevision:
        current = await self.get_provider(provider_id)
        if not current:
            raise KeyError(provider_id)
        if new_id:
            new_id = validate_identifier(new_id, field="Provider ID")
            await self._purge_deleted_provider(new_id)
            candidate = new_id
            copy_index = 1
        else:
            copy_index = 1
            while True:
                candidate = (
                    f"{provider_id}_copy"
                    if copy_index == 1
                    else f"{provider_id}_copy{copy_index}"
                )
                await self._purge_deleted_provider(candidate)
                if not await self._provider_exists_any(candidate):
                    break
                copy_index += 1
        payload = asdict(current.config)
        payload["id"] = candidate
        payload["display_name"] = f"{current.config.display_name} 副本"
        payload["display_name"] = (
            f"{current.config.display_name}(副本)"
            if copy_index == 1
            else f"{current.config.display_name}(副本{copy_index})"
        )
        payload["enabled"] = False
        return await self.create_provider(payload)

    async def delete_provider(self, provider_id: str) -> None:
        if await self.provider_usage(provider_id):
            raise ValueError("该 Provider 正被记忆库使用，不能删除")
        db = await self.connect()
        try:
            existing = await (
                await db.execute("SELECT id FROM providers WHERE id=?", (provider_id,))
            ).fetchone()
            if not existing:
                raise KeyError(provider_id)
            await db.execute(
                "DELETE FROM provider_revisions WHERE provider_id=?",
                (provider_id,),
            )
            await db.execute("DELETE FROM providers WHERE id=?", (provider_id,))
            await db.commit()
        finally:
            await db.close()

    @staticmethod
    def provider_types(kind: str | None = None) -> list[dict[str, Any]]:
        items = []
        for key, value in PROVIDER_TEMPLATES.items():
            item_kind = provider_kind(key)
            if kind and item_kind != kind:
                continue
            items.append({"id": key, "provider_kind": item_kind, **value})
        return items

    async def ensure_default_library(
        self,
        *,
        library_id: str,
        name: str,
        provider_id: str,
        provider_revision: int,
        conversation_settings: dict[str, Any] | None = None,
        recall_settings: dict[str, Any] | None = None,
        maintenance_settings: dict[str, Any] | None = None,
    ) -> MemoryStoreRecord:
        library_id = self._validate_memory_store_id(library_id)
        provider_id = validate_identifier(provider_id, field="Provider ID")
        existing = await self.get_library(library_id)
        if existing:
            if (
                not existing.conversation_settings
                or not existing.recall_settings
                or not existing.maintenance_settings
            ):
                return await self.update_library(
                    library_id,
                    {
                        "conversation_settings": (
                            existing.conversation_settings
                            or conversation_settings
                            or {}
                        ),
                        "recall_settings": (
                            existing.recall_settings or recall_settings or {}
                        ),
                        "maintenance_settings": (
                            existing.maintenance_settings
                            or maintenance_settings
                            or {}
                        ),
                    },
                )
            return existing
        now = time.time()
        db = await self.connect()
        try:
            count = int(
                (
                    await (
                        await db.execute(
                            "SELECT COUNT(*) FROM libraries WHERE deleted_at IS NULL"
                        )
                    ).fetchone()
                )[0]
            )
            await db.execute(
                """INSERT INTO libraries
                (id,database_type,name,description,default_persona_id,is_default,provider_id,
                 provider_revision,rerank_provider_id,conversation_config_json,
                 recall_config_json,maintenance_config_json,metadata_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    library_id,
                    LIVINGMEMORY_V8_TYPE,
                    name,
                    "由 LivingMemory v8 正式数据迁移",
                    "",
                    1 if count == 0 else 0,
                    provider_id,
                    provider_revision,
                    None,
                    json.dumps(
                        self._conversation_settings(conversation_settings),
                        ensure_ascii=False,
                    ),
                    json.dumps(self._recall_settings(recall_settings), ensure_ascii=False),
                    json.dumps(
                        self._maintenance_settings(maintenance_settings),
                        ensure_ascii=False,
                    ),
                    "{}",
                    now,
                    now,
                ),
            )
            await db.commit()
        finally:
            await db.close()
        return (await self.get_library(library_id))  # type: ignore[return-value]

    async def create_library(
        self, payload: dict[str, Any], provider: ProviderRevision
    ) -> MemoryStoreRecord:
        library_id = self._validate_memory_store_id(
            str(payload.get("id") or "")
        )
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("记忆库名称不能为空")
        now = time.time()
        db = await self.connect()
        try:
            deleted_row = await (
                await db.execute(
                    "SELECT deleted_at FROM libraries WHERE id=? LIMIT 1",
                    (library_id,),
                )
            ).fetchone()
            if deleted_row and deleted_row["deleted_at"] is not None:
                await self._delete_soft_deleted_library_row(db, library_id)
            await db.execute(
                """INSERT INTO libraries
                (id,database_type,name,description,default_persona_id,is_default,provider_id,
                 provider_revision,rerank_provider_id,conversation_config_json,
                 recall_config_json,maintenance_config_json,metadata_json,created_at,updated_at)
                VALUES(?,?,?,?,?,0,?,?,?,?,?,?,?,?,?)""",
                (
                    library_id,
                    str(payload.get("database_type") or LIVINGMEMORY_V8_TYPE),
                    name,
                    str(payload.get("description") or ""),
                    str(payload.get("default_persona_id") or ""),
                    provider.provider_id,
                    provider.revision,
                    str(payload.get("rerank_provider_id") or "") or None,
                    json.dumps(
                        self._conversation_settings(
                            payload.get("conversation_settings")
                        ),
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        self._recall_settings(payload.get("recall_settings")),
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        self._maintenance_settings(
                            payload.get("maintenance_settings")
                        ),
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        self._clean_library_metadata(
                            dict(payload.get("metadata") or {})
                        ),
                        ensure_ascii=False,
                    ),
                    now,
                    now,
                ),
            )
            await db.commit()
        except aiosqlite.IntegrityError as exc:
            raise ValueError(f"记忆库 ID {library_id} 已存在") from exc
        finally:
            await db.close()
        return (await self.get_library(library_id))  # type: ignore[return-value]

    async def get_library(self, library_id: str) -> MemoryStoreRecord | None:
        row = await self.memory_store_repository.get_row(library_id)
        if not row:
            return None
        return self._library_row(row)

    def _library_row(self, row: aiosqlite.Row) -> MemoryStoreRecord:
        return MemoryStoreRecord(
            id=row["id"],
            database_type=str(row["database_type"] or LIVINGMEMORY_V8_TYPE),
            name=row["name"],
            description=row["description"],
            default_persona_id=row["default_persona_id"],
            is_default=bool(row["is_default"]),
            provider_id=row["provider_id"],
            provider_revision=int(row["provider_revision"]),
            rerank_provider_id=str(row["rerank_provider_id"] or ""),
            conversation_settings=self._conversation_settings(
                json.loads(row["conversation_config_json"] or "{}")
            ),
            recall_settings=self._recall_settings(
                json.loads(row["recall_config_json"] or "{}")
            ),
            maintenance_settings=self._maintenance_settings(
                json.loads(row["maintenance_config_json"] or "{}")
            ),
            metadata=self._decode_library_metadata(row["metadata_json"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    async def library_id_exists_any(self, library_id: str) -> bool:
        db = await self.connect()
        try:
            row = await (
                await db.execute("SELECT 1 FROM libraries WHERE id=? LIMIT 1", (library_id,))
            ).fetchone()
            return row is not None
        finally:
            await db.close()

    async def _delete_soft_deleted_library_row(
        self, db: aiosqlite.Connection, library_id: str
    ) -> None:
        await db.execute(
            "DELETE FROM library_generation_bindings WHERE library_id=?",
            (library_id,),
        )
        await db.execute(
            "DELETE FROM index_generations WHERE library_id=?",
            (library_id,),
        )
        await db.execute(
            "DELETE FROM migration_runs WHERE library_id=?",
            (library_id,),
        )
        await db.execute(
            "DELETE FROM jobs WHERE library_id=?",
            (library_id,),
        )
        await db.execute(
            "DELETE FROM adapter_connections WHERE library_id=?",
            (library_id,),
        )
        await db.execute(
            """DELETE FROM database_adapter_connections
            WHERE database_type=? AND database_id=?""",
            (LIVINGMEMORY_V8_TYPE, library_id),
        )
        await db.execute(
            "DELETE FROM libraries WHERE id=? AND deleted_at IS NOT NULL",
            (library_id,),
        )

    async def list_libraries(self) -> list[MemoryStoreRecord]:
        rows = await self.memory_store_repository.list_rows()
        return [self._library_row(row) for row in rows]

    async def default_library(self) -> MemoryStoreRecord:
        db = await self.connect()
        try:
            row = await (
                await db.execute(
                    """SELECT * FROM libraries WHERE is_default=1
                    AND deleted_at IS NULL LIMIT 1"""
                )
            ).fetchone()
        finally:
            await db.close()
        if not row:
            raise RuntimeError("尚未配置默认记忆库")
        record = self._library_row(row)
        if not record:
            raise RuntimeError("默认记忆库不存在")
        return record

    async def update_library(
        self, library_id: str, payload: dict[str, Any]
    ) -> MemoryStoreRecord:
        current = await self.get_library(library_id)
        if not current:
            raise KeyError(library_id)
        next_library_id = (
            self._validate_memory_store_id(payload["id"])
            if "id" in payload
            else current.id
        )
        if (
            next_library_id != current.id
            and await self.active_adapter_connections(current.id)
        ):
            raise ValueError("记忆库已被适配器连接，不能修改 ID")
        db = await self.connect()
        try:
            if next_library_id != current.id:
                occupied = await (
                    await db.execute(
                        "SELECT deleted_at FROM libraries WHERE id=? LIMIT 1",
                        (next_library_id,),
                    )
                ).fetchone()
                if occupied and occupied["deleted_at"] is None:
                    raise ValueError(f"记忆库 ID 已存在：{next_library_id}")
                if occupied:
                    await self._delete_soft_deleted_library_row(
                        db, next_library_id
                    )
                for table, column in (
                    ("library_generation_bindings", "manifest_json"),
                    ("index_generations", "manifest"),
                ):
                    rows = await (
                        await db.execute(
                            f"SELECT generation,{column} FROM {table} WHERE library_id=?",
                            (current.id,),
                        )
                    ).fetchall()
                    for row in rows:
                        manifest_text = row[column]
                        try:
                            manifest = json.loads(manifest_text or "{}")
                        except json.JSONDecodeError:
                            manifest = None
                        if isinstance(manifest, dict):
                            manifest["library_id"] = next_library_id
                            manifest_text = json.dumps(
                                manifest, ensure_ascii=False
                            )
                        await db.execute(
                            f"""UPDATE {table} SET library_id=?, {column}=?
                            WHERE library_id=? AND generation=?""",
                            (
                                next_library_id,
                                manifest_text,
                                current.id,
                                row["generation"],
                            ),
                        )
                await db.execute(
                    "UPDATE jobs SET library_id=? WHERE library_id=?",
                    (next_library_id, current.id),
                )
                await db.execute(
                    "UPDATE migration_runs SET library_id=? WHERE library_id=?",
                    (next_library_id, current.id),
                )
                await db.execute(
                    "UPDATE adapter_connections SET library_id=? WHERE library_id=?",
                    (next_library_id, current.id),
                )
                await db.execute(
                    """UPDATE database_adapter_connections SET database_id=?
                    WHERE database_type=? AND database_id=?""",
                    (next_library_id, LIVINGMEMORY_V8_TYPE, current.id),
                )
            await db.execute(
                """UPDATE libraries SET id=?,name=?,description=?,default_persona_id=?,
                rerank_provider_id=?,conversation_config_json=?,recall_config_json=?,
                maintenance_config_json=?,metadata_json=?,updated_at=?
                WHERE id=?""",
                (
                    next_library_id,
                    str(payload.get("name", current.name)).strip() or current.name,
                    str(payload.get("description", current.description)),
                    str(
                        payload.get(
                            "default_persona_id", current.default_persona_id
                        )
                    ),
                    str(
                        payload.get(
                            "rerank_provider_id", current.rerank_provider_id
                        )
                        or ""
                    )
                    or None,
                    json.dumps(
                        self._conversation_settings(
                            current.conversation_settings,
                            payload.get(
                                "conversation_settings",
                                current.conversation_settings,
                            ),
                        ),
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        self._recall_settings(
                            current.recall_settings,
                            payload.get("recall_settings", current.recall_settings),
                        ),
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        self._maintenance_settings(
                            current.maintenance_settings,
                            payload.get(
                                "maintenance_settings",
                                current.maintenance_settings,
                            ),
                        ),
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        self._clean_library_metadata(
                            {
                                **current.metadata,
                                **dict(payload.get("metadata") or {}),
                            }
                        ),
                        ensure_ascii=False,
                    ),
                    time.time(),
                    library_id,
                ),
            )
            await db.commit()
        finally:
            await db.close()
        return (await self.get_library(next_library_id))  # type: ignore[return-value]

    async def update_library_metadata(
        self, library_id: str, updates: dict[str, Any]
    ) -> MemoryStoreRecord:
        current = await self.get_library(library_id)
        if not current:
            raise KeyError(library_id)
        metadata = self._clean_library_metadata(
            {**current.metadata, **updates}
        )
        db = await self.connect()
        try:
            await db.execute(
                "UPDATE libraries SET metadata_json=?,updated_at=? WHERE id=?",
                (
                    json.dumps(metadata, ensure_ascii=False),
                    time.time(),
                    library_id,
                ),
            )
            await db.commit()
        finally:
            await db.close()
        return (await self.get_library(library_id))  # type: ignore[return-value]

    async def set_default_library(self, library_id: str) -> MemoryStoreRecord:
        if not await self.get_library(library_id):
            raise KeyError(library_id)
        db = await self.connect()
        try:
            await db.execute(
                "UPDATE libraries SET is_default=0 WHERE deleted_at IS NULL"
            )
            await db.execute(
                "UPDATE libraries SET is_default=1,updated_at=? WHERE id=?",
                (time.time(), library_id),
            )
            await db.commit()
        finally:
            await db.close()
        return (await self.get_library(library_id))  # type: ignore[return-value]

    async def bind_library(
        self,
        library_id: str,
        provider: ProviderRevision,
        manifest: dict[str, Any],
    ) -> None:
        now = time.time()
        db = await self.connect()
        try:
            await db.execute(
                """UPDATE libraries SET provider_id=?,provider_revision=?,
                updated_at=? WHERE id=?""",
                (provider.provider_id, provider.revision, now, library_id),
            )
            await db.execute(
                """INSERT OR REPLACE INTO library_generation_bindings
                (library_id,database_type,database_id,generation,provider_id,provider_revision,
                 manifest_json,activated_at) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    library_id,
                    LIVINGMEMORY_V8_TYPE,
                    library_id,
                    manifest["generation"],
                    provider.provider_id,
                    provider.revision,
                    json.dumps(manifest, ensure_ascii=False),
                    now,
                ),
            )
            await db.commit()
        finally:
            await db.close()

    async def has_running_jobs(self, library_id: str) -> bool:
        db = await self.connect()
        try:
            count = int(
                (
                    await (
                        await db.execute(
                            """SELECT COUNT(*) FROM jobs WHERE library_id=?
                            AND status IN ('queued','running','pausing','paused','interrupted','stopping')""",
                            (library_id,),
                        )
                    ).fetchone()
                )[0]
            )
            return count > 0
        finally:
            await db.close()

    async def has_any_running_jobs(self) -> bool:
        db = await self.connect()
        try:
            count = int(
                (
                    await (
                        await db.execute(
                            """SELECT COUNT(*) FROM jobs
                            WHERE status IN ('queued','running','pausing','paused','interrupted','stopping')"""
                        )
                    ).fetchone()
                )[0]
            )
            return count > 0
        finally:
            await db.close()

    async def mark_library_deleted(self, library_id: str) -> None:
        record = await self.get_library(library_id)
        if not record:
            raise KeyError(library_id)
        if record.is_default:
            raise ValueError("默认记忆库不能删除")
        if await self.active_adapter_connections(library_id):
            raise ValueError("记忆库已被适配器连接，不能删除")
        if await self.has_running_jobs(library_id):
            raise ValueError("记忆库存在运行中的任务，不能删除")
        db = await self.connect()
        try:
            await db.execute(
                "UPDATE libraries SET deleted_at=?,updated_at=? WHERE id=?",
                (time.time(), time.time(), library_id),
            )
            await db.commit()
        finally:
            await db.close()
