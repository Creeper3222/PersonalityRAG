from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiosqlite

from .compat import (
    LIVINGMEMORY_DATABASE_VERSION,
    LIVINGMEMORY_DATABASE_VERSION_LABEL,
    default_library_metadata,
)
from .config import ProviderConfig
from .providers import (
    PROVIDER_TEMPLATES,
    config_from_dict,
    masked_config,
    provider_kind,
    provider_config_hash,
)


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
class LibraryRecord:
    id: str
    name: str
    description: str
    default_persona_id: str
    is_default: bool
    provider_id: str
    provider_revision: int
    rerank_provider_id: str
    recall_settings: dict[str, Any]
    maintenance_settings: dict[str, Any]
    metadata: dict[str, Any]
    created_at: float
    updated_at: float

    def public(self) -> dict[str, Any]:
        return asdict(self)


class ControlStore:
    def __init__(self, path: Path):
        self.path = path

    @staticmethod
    def _provider_functional_payload(config: ProviderConfig) -> dict[str, Any]:
        payload = asdict(config)
        payload.pop("display_name", None)
        payload.pop("max_context_tokens", None)
        payload.pop("max_context_tokens_source", None)
        return payload

    @classmethod
    def _provider_configs_functionally_equal(
        cls, left: ProviderConfig, right: ProviderConfig
    ) -> bool:
        return cls._provider_functional_payload(left) == cls._provider_functional_payload(
            right
        )

    @staticmethod
    def _validate_library_id(library_id: str) -> str:
        value = str(library_id or "").strip()
        if not value or not value.replace("_", "").replace("-", "").isalnum():
            raise ValueError("记忆库 ID 只能包含字母、数字、下划线和连字符")
        return value

    async def connect(self):
        db = await aiosqlite.connect(self.path)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA busy_timeout=10000")
        await db.execute("PRAGMA foreign_keys=ON")
        return db

    async def initialize(self, seed_provider: ProviderConfig) -> None:
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
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    default_persona_id TEXT NOT NULL DEFAULT '',
                    is_default INTEGER NOT NULL DEFAULT 0,
                    provider_id TEXT NOT NULL,
                    provider_revision INTEGER NOT NULL,
                    rerank_provider_id TEXT,
                    recall_config_json TEXT NOT NULL DEFAULT '{}',
                    maintenance_config_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    deleted_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_libraries_default
                ON libraries(is_default) WHERE is_default=1 AND deleted_at IS NULL;
                CREATE TABLE IF NOT EXISTS library_generation_bindings (
                    library_id TEXT NOT NULL,
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
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    result TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS migration_runs (
                    id TEXT PRIMARY KEY,
                    library_id TEXT,
                    source_path TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    report_path TEXT,
                    created_at REAL NOT NULL,
                    completed_at REAL
                );
                CREATE TABLE IF NOT EXISTS index_generations (
                    library_id TEXT NOT NULL DEFAULT '',
                    generation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    manifest TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    activated_at REAL,
                    PRIMARY KEY(library_id, generation)
                );
                """
            )
            await self._ensure_column(db, "jobs", "library_id", "TEXT")
            await self._ensure_column(db, "migration_runs", "library_id", "TEXT")
            await self._ensure_column(db, "libraries", "rerank_provider_id", "TEXT")
            await self._ensure_column(
                db,
                "index_generations",
                "library_id",
                "TEXT NOT NULL DEFAULT ''",
            )
            await db.execute(
                "INSERT OR REPLACE INTO schema_info(key,value) VALUES('service_version','0.1.0')"
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
            if "maintenance_config_json" not in columns:
                await db.execute(
                    "ALTER TABLE libraries ADD COLUMN maintenance_config_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "metadata_json" not in columns:
                await db.execute(
                    "ALTER TABLE libraries ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
                )
            await self._backfill_library_metadata(db)
            await db.commit()
        finally:
            await db.close()
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

    async def _backfill_library_metadata(self, db: aiosqlite.Connection) -> None:
        rows = await (
            await db.execute("SELECT id,metadata_json FROM libraries")
        ).fetchall()
        for row in rows:
            metadata = self._decode_library_metadata(row["metadata_json"])
            if metadata.get("livingmemory_database_version") is None:
                metadata["livingmemory_database_version"] = (
                    LIVINGMEMORY_DATABASE_VERSION
                )
                await db.execute(
                    "UPDATE libraries SET metadata_json=? WHERE id=?",
                    (json.dumps(metadata, ensure_ascii=False), row["id"]),
                )

    @staticmethod
    def _decode_library_metadata(value: Any) -> dict[str, Any]:
        try:
            data = json.loads(value or "{}")
        except (TypeError, json.JSONDecodeError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        return data

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
        db = await self.connect()
        try:
            if revision is None:
                row = await (
                    await db.execute(
                        """SELECT pr.* FROM providers p JOIN provider_revisions pr
                        ON pr.provider_id=p.id AND pr.revision=p.latest_revision
                        WHERE p.id=? AND p.deleted_at IS NULL""",
                        (provider_id,),
                    )
                ).fetchone()
            else:
                row = await (
                    await db.execute(
                        """SELECT pr.* FROM providers p JOIN provider_revisions pr
                        ON pr.provider_id=p.id
                        WHERE p.id=? AND pr.revision=? AND p.deleted_at IS NULL""",
                        (provider_id, revision),
                    )
                ).fetchone()
        finally:
            await db.close()
        return self._provider_row(row) if row else None

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
                    """SELECT id,name,provider_revision FROM libraries
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
                        "library_id": row["id"],
                        "library_name": row["name"],
                        "provider_revision": provider_revision,
                        "usage_kind": "embedding",
                        "needs_rebuild": needs_rebuild,
                    }
                )
            rerank_rows = await (
                await db.execute(
                    """SELECT id,name FROM libraries
                    WHERE rerank_provider_id=? AND deleted_at IS NULL ORDER BY name""",
                    (provider_id,),
                )
            ).fetchall()
            return embedding_usage + [
                {
                    "library_id": row["id"],
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

    async def create_provider(self, payload: dict[str, Any]) -> ProviderRevision:
        config = config_from_dict(payload)
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
            str(changes.pop("id")).strip() if "id" in changes else provider_id
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
        recall_settings: dict[str, Any] | None = None,
        maintenance_settings: dict[str, Any] | None = None,
    ) -> LibraryRecord:
        existing = await self.get_library(library_id)
        if existing:
            if not existing.recall_settings or not existing.maintenance_settings:
                return await self.update_library(
                    library_id,
                    {
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
                (id,name,description,default_persona_id,is_default,provider_id,
                 provider_revision,rerank_provider_id,recall_config_json,maintenance_config_json,
                 metadata_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    library_id,
                    name,
                    f"由 {LIVINGMEMORY_DATABASE_VERSION_LABEL} 正式数据迁移",
                    "",
                    1 if count == 0 else 0,
                    provider_id,
                    provider_revision,
                    None,
                    json.dumps(recall_settings or {}, ensure_ascii=False),
                    json.dumps(maintenance_settings or {}, ensure_ascii=False),
                    json.dumps(default_library_metadata(), ensure_ascii=False),
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
    ) -> LibraryRecord:
        library_id = self._validate_library_id(str(payload.get("id") or ""))
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
                    "DELETE FROM libraries WHERE id=? AND deleted_at IS NOT NULL",
                    (library_id,),
                )
            await db.execute(
                """INSERT INTO libraries
                (id,name,description,default_persona_id,is_default,provider_id,
                 provider_revision,rerank_provider_id,recall_config_json,maintenance_config_json,
                 metadata_json,created_at,updated_at)
                VALUES(?,?,?,?,0,?,?,?,?,?,?,?,?)""",
                (
                    library_id,
                    name,
                    str(payload.get("description") or ""),
                    str(payload.get("default_persona_id") or ""),
                    provider.provider_id,
                    provider.revision,
                    str(payload.get("rerank_provider_id") or "") or None,
                    json.dumps(
                        payload.get("recall_settings") or {}, ensure_ascii=False
                    ),
                    json.dumps(
                        payload.get("maintenance_settings") or {},
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            **default_library_metadata(),
                            **dict(payload.get("metadata") or {}),
                        },
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

    async def get_library(self, library_id: str) -> LibraryRecord | None:
        db = await self.connect()
        try:
            row = await (
                await db.execute(
                    "SELECT * FROM libraries WHERE id=? AND deleted_at IS NULL",
                    (library_id,),
                )
            ).fetchone()
        finally:
            await db.close()
        if not row:
            return None
        return LibraryRecord(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            default_persona_id=row["default_persona_id"],
            is_default=bool(row["is_default"]),
            provider_id=row["provider_id"],
            provider_revision=int(row["provider_revision"]),
            rerank_provider_id=str(row["rerank_provider_id"] or ""),
            recall_settings=json.loads(row["recall_config_json"] or "{}"),
            maintenance_settings=json.loads(
                row["maintenance_config_json"] or "{}"
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

    async def list_libraries(self) -> list[LibraryRecord]:
        db = await self.connect()
        try:
            rows = await (
                await db.execute(
                    """SELECT * FROM libraries WHERE deleted_at IS NULL
                    ORDER BY is_default DESC, created_at"""
                )
            ).fetchall()
        finally:
            await db.close()
        result = []
        for row in rows:
            record = await self.get_library(row["id"])
            if record:
                result.append(record)
        return result

    async def default_library(self) -> LibraryRecord:
        db = await self.connect()
        try:
            row = await (
                await db.execute(
                    """SELECT id FROM libraries WHERE is_default=1
                    AND deleted_at IS NULL LIMIT 1"""
                )
            ).fetchone()
        finally:
            await db.close()
        if not row:
            raise RuntimeError("尚未配置默认记忆库")
        record = await self.get_library(row["id"])
        if not record:
            raise RuntimeError("默认记忆库不存在")
        return record

    async def update_library(
        self, library_id: str, payload: dict[str, Any]
    ) -> LibraryRecord:
        current = await self.get_library(library_id)
        if not current:
            raise KeyError(library_id)
        next_library_id = (
            self._validate_library_id(payload["id"])
            if "id" in payload
            else current.id
        )
        if next_library_id != current.id and current.is_default:
            raise ValueError("默认记忆库 ID 不能修改")
        db = await self.connect()
        try:
            if next_library_id != current.id:
                occupied = await (
                    await db.execute(
                        "SELECT deleted_at FROM libraries WHERE id=? LIMIT 1",
                        (next_library_id,),
                    )
                ).fetchone()
                if occupied:
                    raise ValueError(f"记忆库 ID 已存在：{next_library_id}")
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
                """UPDATE libraries SET id=?,name=?,description=?,default_persona_id=?,
                rerank_provider_id=?,recall_config_json=?,maintenance_config_json=?,updated_at=?
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
                        payload.get("recall_settings", current.recall_settings),
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        payload.get(
                            "maintenance_settings",
                            current.maintenance_settings,
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
    ) -> LibraryRecord:
        current = await self.get_library(library_id)
        if not current:
            raise KeyError(library_id)
        metadata = {**current.metadata, **updates}
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

    async def set_default_library(self, library_id: str) -> LibraryRecord:
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
                (library_id,generation,provider_id,provider_revision,
                 manifest_json,activated_at) VALUES(?,?,?,?,?,?)""",
                (
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
                            AND status IN ('queued','running')""",
                            (library_id,),
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
