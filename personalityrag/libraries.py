from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from .config import AppConfig
from .database_types import (
    DATABASE_CATEGORY_MEMORY,
    LIVINGMEMORY_V8_TYPE,
    TEXT_MEDIA_V1_TYPE,
    DatabaseDriverContext,
    DatabaseRef,
    database_identity_fields,
    database_type_registry,
)
from .control import ControlStore
from .providers import config_from_dict, masked_config, provider_kind
from .task_types import ACTIVE_JOB_STATUSES
from . import library_types as _registered_database_types  # noqa: F401
from .storage_layout import DatabaseLayout
from .logger import logger


class DatabaseManager:
    """Cross-category database dispatcher.

    Legacy LivingMemory method names remain on the surface for v0.1.1
    compatibility, but new runtime code imports this canonical class name.
    """

    def __init__(self, root: Path, config: AppConfig):
        self.root = root
        self.config = config
        data_root = root / "data"
        system_path = data_root / "personalityrag_system.db"
        self.services = DatabaseDriverContext(
            root=root,
            data_root=data_root,
            system_path=system_path,
            config=config,
            control=ControlStore(system_path),
        )
        self.storage_layout = DatabaseLayout(data_root)
        self._managers = {
            descriptor.id: database_type_registry.require(descriptor.id).create_manager(
                self.services
            )
            for descriptor in database_type_registry.list()
        }
        self._primary = self._managers[LIVINGMEMORY_V8_TYPE]
        self._adapter_disconnect_waiters: dict[
            tuple[DatabaseRef, str], set[asyncio.Future[None]]
        ] = {}
        self._forced_adapter_connections: dict[
            tuple[DatabaseRef, str], dict[str, Any]
        ] = {}
        self._online_prewarm_task: asyncio.Task[None] | None = None
        self._summary_list_task: asyncio.Task[list[dict[str, Any]]] | None = None

    def _database_ref(self, database: str | DatabaseRef) -> DatabaseRef:
        if isinstance(database, DatabaseRef):
            return database
        return DatabaseRef(LIVINGMEMORY_V8_TYPE, database)

    def _livingmemory_ref(self, database: str | DatabaseRef) -> DatabaseRef:
        return self._database_ref(database)

    def _manager_for(self, database: str | DatabaseRef):
        ref = self._database_ref(database)
        manager = self._managers.get(ref.database_type)
        if manager is None:
            raise KeyError(ref.database_type)
        return manager, ref

    @property
    def control(self):
        return self.services.control

    @property
    def jobs(self):
        return self.services.jobs

    @jobs.setter
    def jobs(self, value) -> None:
        self.services.jobs = value
        self._primary.jobs = value

    @property
    def runtimes(self) -> dict[DatabaseRef, Any]:
        combined: dict[DatabaseRef, Any] = {}
        for manager in self._managers.values():
            combined.update(manager.runtimes)
        return combined

    async def initialize(self) -> None:
        # Only registered implementations are initialized; unknown types never
        # fall through to the LivingMemory backend.
        self.storage_layout.prepare()
        for manager in self._managers.values():
            await manager.initialize()
            if manager is self._primary:
                self.services.jobs = manager.jobs
        self._forced_adapter_connections = {
            (
                DatabaseRef(
                    str(
                        item.get("memory_store_type")
                        or item.get("knowledge_base_type")
                        or item.get("database_type")
                        or LIVINGMEMORY_V8_TYPE
                    ),
                    str(
                        item.get("memory_store_id")
                        or item.get("knowledge_base_id")
                        or item.get("database_id")
                        or item["library_id"]
                    ),
                ),
                str(item["adapter_id"]),
            ): item
            for item in await self.control.forced_adapter_connections()
        }
        identities = await self.control.list_database_identities()
        refs = [
            DatabaseRef(str(item["database_type"]), str(item["id"]))
            for item in identities
            if str(item.get("database_type") or "") in self._managers
        ]
        if refs:
            active_map = await self.control.active_database_adapter_connections_map(
                refs
            )
            online_refs = [ref for ref in refs if active_map.get(ref.key)]
            if online_refs:
                self._online_prewarm_task = asyncio.create_task(
                    self._prewarm_online_runtimes(online_refs),
                    name="personalityrag-online-runtime-prewarm",
                )
        if self.services.jobs is not None:
            await self.services.jobs.recover_for_startup()

    async def _prewarm_online_runtimes(
        self,
        refs: list[DatabaseRef],
    ) -> None:
        limiter = asyncio.Semaphore(2)

        async def warm(ref: DatabaseRef) -> None:
            async with limiter:
                try:
                    await self._managers[ref.database_type].get_runtime(
                        ref,
                        touch=False,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "在线适配器数据库后台预热失败：database=%s",
                        ref.key,
                    )

        await asyncio.gather(*(warm(ref) for ref in refs))

    async def suspend_runtime_activity(self) -> None:
        """Drain cold-load activity before replacing database directories."""

        owner = asyncio.current_task()
        if self._online_prewarm_task is not None:
            self._online_prewarm_task.cancel()
            await asyncio.gather(
                self._online_prewarm_task,
                return_exceptions=True,
            )
            self._online_prewarm_task = None
        await asyncio.gather(
            *(
                manager.suspend_runtime_loading(owner=owner)
                for manager in self._managers.values()
            )
        )

    async def resume_runtime_activity(self) -> None:
        await asyncio.gather(
            *(
                manager.resume_runtime_loading()
                for manager in self._managers.values()
            )
        )

    async def close(self) -> None:
        if self._summary_list_task is not None:
            self._summary_list_task.cancel()
            await asyncio.gather(
                self._summary_list_task,
                return_exceptions=True,
            )
            self._summary_list_task = None
        if self._online_prewarm_task is not None:
            self._online_prewarm_task.cancel()
            await asyncio.gather(
                self._online_prewarm_task,
                return_exceptions=True,
            )
            self._online_prewarm_task = None
        waiters = [
            future
            for group in self._adapter_disconnect_waiters.values()
            for future in group
        ]
        self._adapter_disconnect_waiters.clear()
        self._forced_adapter_connections.clear()
        for future in waiters:
            if not future.done():
                future.cancel()
        for manager in reversed(list(self._managers.values())):
            await manager.close()
        # Offline summaries and short-lived inspection helpers own pools that
        # are intentionally not attached to a resident runtime. Close the
        # process-wide registry as the final SQLite safety net so no aiosqlite
        # worker can keep the interpreter alive after a graceful restart.
        from .sqlite_pool import SQLiteConnectionPool

        await SQLiteConnectionPool.close_open_pools()
        from .http_pool import close_http_pools

        await close_http_pools()
        self.storage_layout.close()

    def subscribe_adapter_disconnect(
        self,
        database: str | DatabaseRef,
        adapter_id: str,
    ) -> asyncio.Future[None]:
        ref = self._database_ref(database)
        future = asyncio.get_running_loop().create_future()
        self._adapter_disconnect_waiters.setdefault((ref, adapter_id), set()).add(
            future
        )
        return future

    def unsubscribe_adapter_disconnect(
        self,
        database: str | DatabaseRef,
        adapter_id: str,
        future: asyncio.Future[None],
    ) -> None:
        key = (self._database_ref(database), adapter_id)
        waiters = self._adapter_disconnect_waiters.get(key)
        if not waiters:
            return
        waiters.discard(future)
        if not waiters:
            self._adapter_disconnect_waiters.pop(key, None)

    def notify_adapter_disconnect(
        self,
        database: str | DatabaseRef,
        adapter_id: str,
    ) -> None:
        ref = self._database_ref(database)
        for future in tuple(
            self._adapter_disconnect_waiters.get((ref, adapter_id), ())
        ):
            if not future.done():
                future.set_result(None)

    def forced_adapter_connection(
        self,
        database: str | DatabaseRef,
        adapter_id: str,
    ) -> dict[str, Any] | None:
        connection = self._forced_adapter_connections.get(
            (self._database_ref(database), adapter_id)
        )
        return dict(connection) if connection else None

    def mark_adapter_forced_offline(self, connection: dict[str, Any]) -> None:
        ref = DatabaseRef(
            str(
                connection.get("memory_store_type")
                or connection.get("knowledge_base_type")
                or connection.get("database_type")
                or LIVINGMEMORY_V8_TYPE
            ),
            str(
                connection.get("memory_store_id")
                or connection.get("knowledge_base_id")
                or connection.get("database_id")
                or connection["library_id"]
            ),
        )
        self._forced_adapter_connections[(ref, str(connection["adapter_id"]))] = dict(
            connection
        )

    def clear_adapter_forced_offline(
        self,
        database: str | DatabaseRef,
        adapter_id: str,
    ) -> None:
        self._forced_adapter_connections.pop(
            (self._database_ref(database), adapter_id), None
        )

    async def _list_libraries_uncached(
        self, *args, **kwargs
    ) -> list[dict[str, Any]]:
        grouped = await asyncio.gather(
            *(
                manager.list_libraries(*args, **kwargs)
                for manager in self._managers.values()
            )
        )
        items = [item for group in grouped for item in group]
        return sorted(
            items,
            key=lambda item: (
                not bool(item.get("is_default")),
                float(item.get("created_at") or 9_999_999_999),
                str(item.get("database_type") or ""),
                str(item.get("id") or ""),
            ),
        )

    async def list_libraries(self, *args, **kwargs) -> list[dict[str, Any]]:
        if kwargs.get("stats_mode") != "summary":
            return await self._list_libraries_uncached(*args, **kwargs)

        # Coalesce only overlapping requests.  The completed result is never
        # retained, so invalidation and freshness semantics remain unchanged.
        task = self._summary_list_task
        if task is None or task.done():
            task = asyncio.create_task(
                self._list_libraries_uncached(*args, **kwargs),
                name="database-summary-snapshot",
            )
            self._summary_list_task = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done() and self._summary_list_task is task:
                self._summary_list_task = None

    async def apply_runtime_residency(self) -> dict[str, list[str]]:
        results: dict[str, list[str]] = {}
        for database_type, manager in self._managers.items():
            apply = getattr(manager, "apply_runtime_residency", None)
            if callable(apply):
                results[database_type] = list(await apply())
        return results

    def runtime_residency_status(self) -> dict[str, Any]:
        by_type: dict[str, dict[str, Any]] = {}
        loaded_databases: list[dict[str, str]] = []
        for database_type, manager in self._managers.items():
            status_factory = getattr(manager, "runtime_residency_status", None)
            if callable(status_factory):
                status = dict(status_factory())
            else:
                status = {
                    "loaded_databases": [
                        ref.public() for ref in manager.runtimes
                    ]
                }
            by_type[database_type] = status
            loaded_databases.extend(status.get("loaded_databases") or [])
        primary = dict(by_type.get(LIVINGMEMORY_V8_TYPE) or {})
        return {
            **primary,
            "loaded_databases": loaded_databases,
            "loaded_count": len(loaded_databases),
            "by_type": by_type,
        }

    def _enrich_provider_usage(self, item: dict[str, Any]) -> dict[str, Any]:
        result = dict(item)
        database_type = str(result.get("database_type") or LIVINGMEMORY_V8_TYPE)
        try:
            descriptor = database_type_registry.require(database_type).descriptor
        except KeyError:
            descriptor = None
        if descriptor is not None:
            result.setdefault("database_category", descriptor.category)
            result.setdefault("type_display_name", descriptor.display_name)
        database_id = str(
            result.get("database_id") or result.get("library_id") or ""
        )
        result.update(
            database_identity_fields(
                DatabaseRef(database_type, database_id),
                include_deprecated=True,
            )
        )
        return result

    @staticmethod
    def _provider_usage_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        category_order = {"memory": 0, "knowledge": 1}
        return (
            category_order.get(str(item.get("database_category") or ""), 9),
            str(
                item.get("database_name")
                or item.get("library_name")
                or item.get("database_id")
                or ""
            ),
            str(item.get("type_display_name") or ""),
            str(item.get("database_type") or ""),
            str(item.get("database_id") or ""),
            str(item.get("usage_kind") or ""),
        )

    async def provider_usage(
        self,
        provider_id: str,
        *,
        base_usage: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        usage = [
            self._enrich_provider_usage(item)
            for item in (
                base_usage
                if base_usage is not None
                else await self.control.provider_usage(provider_id)
            )
        ]
        seen = {
            (
                item.get("database_type"),
                item.get("database_id"),
                item.get("usage_kind") or "embedding",
            )
            for item in usage
        }
        for manager in self._managers.values():
            usage_factory = getattr(manager, "provider_usage", None)
            if not callable(usage_factory):
                continue
            for item in await usage_factory(provider_id):
                enriched = self._enrich_provider_usage(item)
                key = (
                    enriched.get("database_type"),
                    enriched.get("database_id"),
                    enriched.get("usage_kind") or "embedding",
                )
                if key in seen:
                    continue
                seen.add(key)
                usage.append(enriched)
        return sorted(usage, key=self._provider_usage_sort_key)

    async def list_providers(self, kind: str | None = None) -> list[dict[str, Any]]:
        items = await self.control.list_providers(kind)
        provider_ids = {
            str(item.get("id") or "") for item in items if item.get("id")
        }
        usage_by_provider = {
            provider_id: [
                self._enrich_provider_usage(entry)
                for entry in list(item.get("used_by") or [])
            ]
            for provider_id, item in (
                (str(item.get("id") or ""), item) for item in items
            )
            if provider_id
        }
        seen_by_provider = {
            provider_id: {
                (
                    entry.get("database_type"),
                    entry.get("database_id"),
                    entry.get("usage_kind") or "embedding",
                )
                for entry in usage
            }
            for provider_id, usage in usage_by_provider.items()
        }
        for manager in self._managers.values():
            usage_map_factory = getattr(manager, "provider_usage_map", None)
            if callable(usage_map_factory):
                manager_usage = await usage_map_factory(provider_ids)
            else:
                usage_factory = getattr(manager, "provider_usage", None)
                if not callable(usage_factory):
                    continue
                values = await asyncio.gather(
                    *(usage_factory(provider_id) for provider_id in provider_ids)
                )
                manager_usage = dict(zip(provider_ids, values, strict=True))
            for provider_id, entries in manager_usage.items():
                usage = usage_by_provider.setdefault(provider_id, [])
                seen = seen_by_provider.setdefault(provider_id, set())
                for entry in entries:
                    enriched = self._enrich_provider_usage(entry)
                    key = (
                        enriched.get("database_type"),
                        enriched.get("database_id"),
                        enriched.get("usage_kind") or "embedding",
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    usage.append(enriched)
        for item in items:
            item["used_by"] = sorted(
                usage_by_provider.get(str(item.get("id") or ""), []),
                key=self._provider_usage_sort_key,
            )
        return items

    @staticmethod
    def _debug_config_fields(config: Any) -> dict[str, Any]:
        payload = masked_config(config)
        payload.pop("api_key", None)
        return payload

    @staticmethod
    def _debug_flatten_fields(
        payload: dict[str, Any], prefix: str = ""
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in payload.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(DatabaseManager._debug_flatten_fields(value, name))
            else:
                result[name] = value
        return result

    @staticmethod
    def _debug_job_provider_references(job: dict[str, Any]) -> set[tuple[str, int]]:
        references: set[tuple[str, int]] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                provider_pairs = (
                    ("provider_id", "provider_revision"),
                    ("embedding_provider_id", "embedding_provider_revision"),
                    ("rerank_provider_id", "rerank_provider_revision"),
                )
                for provider_key, revision_key in provider_pairs:
                    provider_id = str(value.get(provider_key) or "")
                    try:
                        revision = int(value.get(revision_key) or 0)
                    except (TypeError, ValueError):
                        revision = 0
                    if provider_id and revision > 0:
                        references.add((provider_id, revision))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        for field in ("operation", "checkpoint"):
            raw = job.get(field)
            if not raw:
                continue
            try:
                value = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                continue
            visit(value)
        return references

    async def debug_revision_overview(self) -> dict[str, Any]:
        inventory = await self.control.debug_revision_inventory()
        provider_rows = {
            str(row["id"]): row for row in inventory["providers"]
        }
        revision_records: dict[tuple[str, int], Any] = {}
        revisions_by_provider: dict[str, list[dict[str, Any]]] = {}
        for row in inventory["provider_revisions"]:
            provider_id = str(row["provider_id"])
            revision = int(row["revision"])
            config = config_from_dict(json.loads(row["config_json"]))
            revision_records[(provider_id, revision)] = config
            revisions_by_provider.setdefault(provider_id, []).append(
                {
                    "provider_id": provider_id,
                    "revision": revision,
                    "config_sha256": str(row["config_sha256"]),
                    "functional_sha256": self.control._provider_functional_hash(
                        config
                    ),
                    "created_at": float(row["created_at"]),
                    "config": self._debug_config_fields(config),
                    "references": [],
                }
            )

        providers: list[dict[str, Any]] = []
        for provider_id, row in provider_rows.items():
            latest_revision = int(row["latest_revision"])
            revisions = revisions_by_provider.get(provider_id, [])
            latest = next(
                (item for item in revisions if item["revision"] == latest_revision),
                None,
            )
            latest_flat = self._debug_flatten_fields(
                dict((latest or {}).get("config") or {})
            )
            for item in revisions:
                current_flat = self._debug_flatten_fields(item["config"])
                item["is_latest"] = item["revision"] == latest_revision
                item["functionally_equal_to_latest"] = bool(
                    latest
                    and item["functional_sha256"]
                    == latest["functional_sha256"]
                )
                item["changed_fields"] = sorted(
                    key
                    for key in set(latest_flat) | set(current_flat)
                    if latest_flat.get(key) != current_flat.get(key)
                )
            latest_config = revision_records.get((provider_id, latest_revision))
            providers.append(
                {
                    "provider_id": provider_id,
                    "display_name": str(
                        getattr(latest_config, "display_name", provider_id)
                    ),
                    "provider_type": str(getattr(latest_config, "type", "")),
                    "provider_kind": (
                        provider_kind(str(getattr(latest_config, "type", "")))
                        if latest_config is not None
                        else "unknown"
                    ),
                    "latest_revision": latest_revision,
                    "created_at": float(row["created_at"]),
                    "updated_at": float(row["updated_at"]),
                    "revisions": revisions,
                }
            )

        job_by_database: dict[str, list[dict[str, Any]]] = {}
        for job in inventory["revision_jobs"]:
            if str(job.get("status") or "") not in ACTIVE_JOB_STATUSES:
                continue
            database_type = str(job.get("database_type") or LIVINGMEMORY_V8_TYPE)
            database_id = str(
                job.get("database_id") or job.get("library_id") or ""
            )
            prefix = f"{database_type}:"
            if database_id.startswith(prefix):
                database_id = database_id[len(prefix):]
            if database_id:
                job_by_database.setdefault(f"{database_type}:{database_id}", []).append(
                    {
                        "id": str(job.get("id") or ""),
                        "kind": str(job.get("kind") or ""),
                        "status": str(job.get("status") or ""),
                    }
                )
        adapter_by_database: dict[str, list[dict[str, Any]]] = {}
        for item in inventory["active_adapters"]:
            key = f"{item['database_type']}:{item['database_id']}"
            adapter_by_database.setdefault(key, []).append(
                {
                    "adapter_id": str(item.get("adapter_id") or ""),
                    "instance_id": str(item.get("instance_id") or ""),
                    "last_seen": float(item.get("last_seen") or 0),
                }
            )

        generations_by_memory: dict[str, list[dict[str, Any]]] = {}
        for row in inventory["livingmemory_generations"]:
            database_type = str(row.get("database_type") or LIVINGMEMORY_V8_TYPE)
            database_id = str(row.get("database_id") or row.get("library_id") or "")
            if database_type != LIVINGMEMORY_V8_TYPE or not database_id:
                continue
            try:
                manifest = json.loads(str(row.get("manifest_json") or "{}"))
            except json.JSONDecodeError:
                manifest = {}
            generations_by_memory.setdefault(database_id, []).append(
                {
                    "generation": str(row.get("generation") or ""),
                    "provider_id": str(row.get("provider_id") or ""),
                    "provider_revision": int(row.get("provider_revision") or 0),
                    "provider_fingerprint": str(
                        (manifest or {}).get("provider_config_sha256") or ""
                    ),
                    "activated_at": float(row.get("activated_at") or 0),
                }
            )

        databases: list[dict[str, Any]] = []
        memory_manager = self._managers[LIVINGMEMORY_V8_TYPE]
        for row in inventory["memory_databases"]:
            database_type = str(row.get("database_type") or LIVINGMEMORY_V8_TYPE)
            if database_type != LIVINGMEMORY_V8_TYPE:
                continue
            database_id = str(row["id"])
            key = f"{database_type}:{database_id}"
            file_generations = memory_manager.debug_revision_generation_manifests(
                database_id
            )
            generation_items_by_id = {
                str(item.get("generation") or ""): item
                for item in generations_by_memory.get(database_id, [])
            }
            for item in file_generations:
                generation = str(item.get("generation") or "")
                generation_items_by_id[generation] = {
                    **generation_items_by_id.get(generation, {}),
                    **item,
                }
            generation_items = list(generation_items_by_id.values())
            embedding_revision = int(row.get("provider_revision") or 0)
            embedding_record = next(
                (
                    item
                    for item in revisions_by_provider.get(
                        str(row.get("provider_id") or ""), []
                    )
                    if item["revision"] == embedding_revision
                ),
                None,
            )
            bindings = [
                {
                    "usage_kind": "embedding",
                    "binding_mode": "pinned",
                    "provider_id": str(row.get("provider_id") or ""),
                    "provider_revision": embedding_revision,
                    "provider_fingerprint": str(
                        (embedding_record or {}).get("config_sha256") or ""
                    ),
                }
            ]
            rerank_provider_id = str(row.get("rerank_provider_id") or "")
            if rerank_provider_id:
                latest = provider_rows.get(rerank_provider_id)
                latest_revision = int((latest or {}).get("latest_revision") or 0)
                latest_record = next(
                    (
                        item
                        for item in revisions_by_provider.get(rerank_provider_id, [])
                        if item["revision"] == latest_revision
                    ),
                    None,
                )
                bindings.append(
                    {
                        "usage_kind": "rerank",
                        "binding_mode": "follows_latest",
                        "provider_id": rerank_provider_id,
                        "provider_revision": latest_revision or None,
                        "provider_fingerprint": str(
                            (latest_record or {}).get("config_sha256") or ""
                        ),
                    }
                )
            databases.append(
                {
                    "database_type": database_type,
                    "database_category": DATABASE_CATEGORY_MEMORY,
                    "database_id": database_id,
                    "database_name": str(row.get("name") or database_id),
                    "type_display_name": database_type_registry.require(
                        database_type
                    ).descriptor.display_name,
                    "created_at": float(row.get("created_at") or 0),
                    "updated_at": float(row.get("updated_at") or 0),
                    "bindings": bindings,
                    "embedding_generations": generation_items,
                    "active_tasks": job_by_database.get(key, []),
                    "active_adapters": adapter_by_database.get(key, []),
                }
            )

        text_manager = self._managers.get(TEXT_MEDIA_V1_TYPE)
        if text_manager is not None:
            for database in await text_manager.debug_revision_bindings():
                key = f"{database['database_type']}:{database['database_id']}"
                database["active_tasks"] = job_by_database.get(key, [])
                database["active_adapters"] = adapter_by_database.get(key, [])
                databases.append(database)

        issues: list[dict[str, Any]] = []
        revision_lookup = {
            (provider["provider_id"], revision["revision"]): revision
            for provider in providers
            for revision in provider["revisions"]
        }
        latest_lookup = {
            provider["provider_id"]: provider["latest_revision"]
            for provider in providers
        }
        def add_reference(
            provider_id: str,
            revision: int,
            reference: dict[str, Any],
        ) -> None:
            target = revision_lookup.get((provider_id, int(revision or 0)))
            if target is not None:
                target["references"].append(reference)

        for database in databases:
            database_issues: list[dict[str, Any]] = []
            for binding in database.get("bindings", []):
                provider_id = str(binding.get("provider_id") or "")
                revision = int(binding.get("provider_revision") or 0)
                target = revision_lookup.get((provider_id, revision))
                latest_revision = latest_lookup.get(provider_id)
                binding["provider_exists"] = provider_id in provider_rows
                binding["revision_exists"] = target is not None
                binding["latest_revision"] = latest_revision
                binding["functionally_equal_to_latest"] = bool(
                    target
                    and latest_revision
                    and target["functional_sha256"]
                    == revision_lookup.get(
                        (provider_id, latest_revision), {}
                    ).get("functional_sha256")
                )
                fingerprint = str(binding.get("provider_fingerprint") or "")
                binding["fingerprint_matches_revision"] = (
                    None
                    if not fingerprint or target is None
                    else fingerprint == target["config_sha256"]
                )
                if target is not None:
                    add_reference(
                        provider_id,
                        revision,
                        {
                            "kind": "database_binding",
                            "database_type": database["database_type"],
                            "database_id": database["database_id"],
                            "usage_kind": binding["usage_kind"],
                        },
                    )
                if not binding["provider_exists"]:
                    database_issues.append(
                        {
                            "code": "missing_provider",
                            "usage_kind": binding["usage_kind"],
                            "provider_id": provider_id,
                        }
                    )
                elif not binding["revision_exists"]:
                    database_issues.append(
                        {
                            "code": "unknown_revision",
                            "usage_kind": binding["usage_kind"],
                            "provider_id": provider_id,
                            "revision": revision,
                        }
                    )
                elif binding["fingerprint_matches_revision"] is False:
                    database_issues.append(
                        {
                            "code": "fingerprint_mismatch",
                            "usage_kind": binding["usage_kind"],
                            "provider_id": provider_id,
                            "revision": revision,
                        }
                    )
                if (
                    binding["usage_kind"] == "embedding"
                    and binding["binding_mode"] == "pinned"
                ):
                    binding["needs_rebuild"] = not bool(
                        binding["functionally_equal_to_latest"]
                    )
                else:
                    binding["needs_rebuild"] = False

            for generation in database.get("embedding_generations", []):
                provider_id = str(generation.get("provider_id") or "")
                revision = int(generation.get("provider_revision") or 0)
                target = revision_lookup.get((provider_id, revision))
                fingerprint = str(generation.get("provider_fingerprint") or "")
                generation["revision_exists"] = target is not None
                generation["fingerprint_matches_revision"] = (
                    None
                    if not fingerprint or target is None
                    else fingerprint == target["config_sha256"]
                )
                add_reference(
                    provider_id,
                    revision,
                    {
                        "kind": "index_generation",
                        "database_type": database["database_type"],
                        "database_id": database["database_id"],
                        "generation": generation.get("generation")
                        or generation.get("id"),
                        "active": bool(generation.get("is_active")),
                    },
                )
                embedding_binding = next(
                    (
                        item
                        for item in database.get("bindings", [])
                        if item.get("usage_kind") == "embedding"
                    ),
                    None,
                )
                if generation.get("is_active") and embedding_binding and (
                    provider_id != embedding_binding.get("provider_id")
                    or revision != int(
                        embedding_binding.get("provider_revision") or 0
                    )
                    or generation["fingerprint_matches_revision"] is False
                ):
                    database_issues.append(
                        {
                            "code": "active_generation_binding_mismatch",
                            "generation": generation.get("generation")
                            or generation.get("id"),
                        }
                    )

            if database.get("missing_storage"):
                database_issues.append({"code": "missing_database_storage"})
            if database["database_type"] == TEXT_MEDIA_V1_TYPE:
                embedding_binding = next(
                    (
                        item
                        for item in database.get("bindings", [])
                        if item.get("usage_kind") == "embedding"
                    ),
                    None,
                )
                rerank_binding = next(
                    (
                        item
                        for item in database.get("bindings", [])
                        if item.get("usage_kind") == "rerank"
                    ),
                    None,
                )
                embedding_fp = str(
                    (embedding_binding or {}).get("provider_fingerprint") or ""
                )
                rerank_fp = str(
                    (rerank_binding or {}).get("provider_fingerprint") or ""
                )
                media_stale = any(
                    str(item.get("provider_fingerprint") or "")
                    not in {"", embedding_fp}
                    for item in database.get("media_embeddings", [])
                )
                relation_calibration_stale = any(
                    (
                        str(item.get("calibration_provider_fingerprint") or "")
                        not in {"", embedding_fp}
                        or (
                            rerank_fp
                            and str(
                                item.get(
                                    "calibration_rerank_provider_fingerprint"
                                )
                                or ""
                            )
                            not in {"", rerank_fp}
                        )
                    )
                    for item in database.get("relation_calibrations", [])
                    if str(item.get("semantic_mode") or "") == "calibrated"
                )
                strength_calibration_stale = any(
                    (
                        str(item.get("provider_fingerprint") or "")
                        not in {"", embedding_fp}
                        or (
                            rerank_fp
                            and str(item.get("rerank_provider_fingerprint") or "")
                            not in {"", rerank_fp}
                        )
                    )
                    for item in database.get("strength_calibrations", [])
                )
                calibration_stale = bool(
                    relation_calibration_stale or strength_calibration_stale
                )
                if embedding_binding:
                    embedding_binding["needs_rebuild"] = bool(
                        embedding_binding["needs_rebuild"] or media_stale
                    )
                if rerank_binding:
                    rerank_binding["needs_recalibration"] = calibration_stale
                if media_stale:
                    database_issues.append(
                        {"code": "media_embedding_fingerprint_mismatch"}
                    )
                if calibration_stale:
                    database_issues.append(
                        {"code": "media_calibration_fingerprint_mismatch"}
                    )
            database["needs_rebuild"] = any(
                bool(item.get("needs_rebuild"))
                for item in database.get("bindings", [])
            )
            database["needs_recalibration"] = any(
                bool(item.get("needs_recalibration"))
                for item in database.get("bindings", [])
            )
            database["issues"] = database_issues
            issues.extend(
                {
                    **issue,
                    "database_type": database["database_type"],
                    "database_id": database["database_id"],
                }
                for issue in database_issues
            )

        for generation in inventory["index_generations"]:
            try:
                manifest = json.loads(str(generation.get("manifest") or "{}"))
            except json.JSONDecodeError:
                manifest = {}
            for provider_id, revision in self._debug_job_provider_references(
                {"operation": manifest}
            ):
                add_reference(
                    provider_id,
                    revision,
                    {
                        "kind": "index_generation",
                        "database_type": str(
                            generation.get("database_type")
                            or LIVINGMEMORY_V8_TYPE
                        ),
                        "database_id": str(
                            generation.get("database_id")
                            or generation.get("library_id")
                            or ""
                        ),
                        "generation": str(generation.get("generation") or ""),
                        "active": str(generation.get("status") or "") == "active",
                    },
                )

        for job in inventory["revision_jobs"]:
            for provider_id, revision in self._debug_job_provider_references(job):
                add_reference(
                    provider_id,
                    revision,
                    {
                        "kind": "task_checkpoint",
                        "job_id": str(job.get("id") or ""),
                        "status": str(job.get("status") or ""),
                        "database_type": str(
                            job.get("database_type") or LIVINGMEMORY_V8_TYPE
                        ),
                        "database_id": str(
                            job.get("database_id")
                            or job.get("library_id")
                            or ""
                        ).removeprefix(
                            f"{str(job.get('database_type') or LIVINGMEMORY_V8_TYPE)}:"
                        ),
                    },
                )

        for provider in providers:
            if not any(
                revision["revision"] == provider["latest_revision"]
                for revision in provider["revisions"]
            ):
                issues.append(
                    {
                        "code": "missing_latest_revision",
                        "provider_id": provider["provider_id"],
                        "revision": provider["latest_revision"],
                    }
                )

        return {
            "schema_version": 1,
            "captured_at": time.time(),
            "providers": providers,
            "databases": sorted(
                databases,
                key=lambda item: (
                    item["database_category"],
                    item["database_type"],
                    item["database_id"],
                ),
            ),
            "issues": issues,
            "summary": {
                "provider_count": len(providers),
                "revision_count": sum(
                    len(provider["revisions"]) for provider in providers
                ),
                "database_count": len(databases),
                "issue_count": len(issues),
                "active_task_count": sum(
                    len(item.get("active_tasks", [])) for item in databases
                ),
                "active_adapter_count": sum(
                    len(item.get("active_adapters", [])) for item in databases
                ),
            },
        }

    async def debug_provider_revisions(self, provider_id: str) -> dict[str, Any]:
        overview = await self.debug_revision_overview()
        provider = next(
            (
                item
                for item in overview["providers"]
                if item["provider_id"] == provider_id
            ),
            None,
        )
        if provider is None:
            raise KeyError(provider_id)
        usage = []
        generation_bindings = []
        for database in overview["databases"]:
            for binding in database.get("bindings", []):
                if binding.get("provider_id") != provider_id:
                    continue
                usage.append(
                    {
                        "database_type": database["database_type"],
                        "database_id": database["database_id"],
                        "database_name": database["database_name"],
                        **binding,
                    }
                )
            for generation in database.get("embedding_generations", []):
                if generation.get("provider_id") != provider_id:
                    continue
                generation_bindings.append(
                    {
                        "database_type": database["database_type"],
                        "database_id": database["database_id"],
                        **generation,
                    }
                )
        return {**provider, "usage": usage, "generation_bindings": generation_bindings}

    async def _debug_assert_databases_idle(
        self, databases: list[dict[str, Any]]
    ) -> None:
        busy = [
            item
            for item in databases
            if item.get("active_tasks") or item.get("active_adapters")
        ]
        if busy:
            labels = ", ".join(
                f"{item['database_type']}:{item['database_id']}" for item in busy
            )
            raise ValueError(
                "revision mutation is blocked by active tasks or adapters: "
                + labels
            )

    async def debug_patch_provider_revision(
        self,
        provider_id: str,
        revision: int,
        patch: dict[str, Any],
    ) -> dict[str, Any]:
        before_overview = await self.debug_revision_overview()
        before = next(
            (
                item
                for item in before_overview["providers"]
                if item["provider_id"] == provider_id
            ),
            None,
        )
        if before is None:
            raise KeyError(provider_id)
        if not any(item["revision"] == revision for item in before["revisions"]):
            raise KeyError(f"{provider_id}@{revision}")
        affected = [
            database
            for database in before_overview["databases"]
            if any(
                ref.get("kind")
                in {"database_binding", "index_generation", "task_checkpoint"}
                and ref.get("database_type") == database["database_type"]
                and ref.get("database_id") == database["database_id"]
                for ref in next(
                    item
                    for item in before["revisions"]
                    if item["revision"] == revision
                )["references"]
            )
        ]
        await self._debug_assert_databases_idle(affected)
        for database in affected:
            await self.unload_runtime(
                DatabaseRef(database["database_type"], database["database_id"]),
                reason="revision_debug_provider_patch",
            )
        await self.control.debug_patch_provider_revision(
            provider_id,
            revision,
            patch,
        )
        after = await self.debug_provider_revisions(provider_id)
        return {**after, "before_snapshot": before, "after_snapshot": after}

    async def debug_reset_provider_revisions(
        self,
        provider_id: str,
        *,
        latest_revision: int | None = None,
        bind_libraries_to_latest: bool = False,
        library_revisions: dict[str, int] | None = None,
        delete_revisions_after_latest: bool = False,
        allow_non_equivalent: bool = False,
    ) -> dict[str, Any]:
        overview = await self.debug_revision_overview()
        provider = next(
            (
                item
                for item in overview["providers"]
                if item["provider_id"] == provider_id
            ),
            None,
        )
        if provider is None:
            raise KeyError(provider_id)
        known = {item["revision"]: item for item in provider["revisions"]}
        target_latest = int(latest_revision or provider["latest_revision"])
        if target_latest not in known:
            raise ValueError(f"provider revision {target_latest} does not exist")
        current_latest = known.get(provider["latest_revision"])
        latest_equivalent = bool(
            current_latest
            and known[target_latest]["functional_sha256"]
            == current_latest["functional_sha256"]
        )
        if target_latest != provider["latest_revision"] and not latest_equivalent:
            if not allow_non_equivalent:
                raise ValueError(
                    "setting a functionally different latest revision requires "
                    "an explicit compatibility assertion"
                )
        follows_latest_databases: list[dict[str, Any]] = []
        if target_latest != provider["latest_revision"]:
            follows_latest_databases = [
                database
                for database in overview["databases"]
                if any(
                    binding.get("provider_id") == provider_id
                    and binding.get("binding_mode") == "follows_latest"
                    for binding in database.get("bindings", [])
                )
            ]
            await self._debug_assert_databases_idle(follows_latest_databases)
        if delete_revisions_after_latest:
            if not allow_non_equivalent:
                raise ValueError("deleting provider revisions requires danger confirmation")
            referenced = [
                item
                for revision, item in known.items()
                if revision > target_latest and item.get("references")
            ]
            if referenced:
                labels = ", ".join(
                    f"r{item['revision']}" for item in referenced
                )
                raise ValueError(
                    "cannot delete revisions referenced by a database, index "
                    "generation, or task checkpoint: " + labels
                )
        requested = dict(library_revisions or {})
        if bind_libraries_to_latest:
            for database in overview["databases"]:
                if database["database_type"] != LIVINGMEMORY_V8_TYPE:
                    continue
                binding = next(
                    (
                        item
                        for item in database.get("bindings", [])
                        if item.get("usage_kind") == "embedding"
                        and item.get("provider_id") == provider_id
                    ),
                    None,
                )
                if binding:
                    requested.setdefault(database["database_id"], target_latest)
        if requested:
            targets = [
                item
                for item in overview["databases"]
                if item["database_type"] == LIVINGMEMORY_V8_TYPE
                and item["database_id"] in requested
            ]
            await self._debug_assert_databases_idle(targets)
            for database in targets:
                revision = int(requested[database["database_id"]])
                target = known.get(revision)
                if target is None:
                    raise ValueError(f"provider revision {revision} does not exist")
                binding = next(
                    item
                    for item in database["bindings"]
                    if item["usage_kind"] == "embedding"
                )
                current = known.get(int(binding.get("provider_revision") or 0))
                equivalent = bool(
                    current
                    and current["functional_sha256"] == target["functional_sha256"]
                )
                if not equivalent and not allow_non_equivalent:
                    raise ValueError(
                        "binding a functionally different revision requires an "
                        "explicit compatibility assertion"
                    )
        applied: list[tuple[dict[str, Any], int, str]] = []
        try:
            for database in (targets if requested else []):
                binding = next(
                    item
                    for item in database["bindings"]
                    if item["usage_kind"] == "embedding"
                )
                previous_revision = int(binding.get("provider_revision") or 0)
                previous_fingerprint = str(
                    binding.get("provider_fingerprint")
                    or (known.get(previous_revision) or {}).get("config_sha256")
                    or ""
                )
                revision = int(requested[database["database_id"]])
                await self._managers[LIVINGMEMORY_V8_TYPE].debug_rebind_revision(
                    database["database_id"],
                    provider_id=provider_id,
                    revision=revision,
                    fingerprint=known[revision]["config_sha256"],
                )
                applied.append(
                    (database, previous_revision, previous_fingerprint)
                )
            for database in follows_latest_databases:
                await self.unload_runtime(
                    DatabaseRef(
                        database["database_type"], database["database_id"]
                    ),
                    reason="revision_debug_latest_reset",
                )
            await self.control.debug_reset_provider_revisions(
                provider_id,
                latest_revision=target_latest,
                bind_libraries_to_latest=False,
                library_revisions={},
                delete_revisions_after_latest=delete_revisions_after_latest,
            )
        except BaseException:
            rollback_errors: list[str] = []
            for database, previous_revision, previous_fingerprint in reversed(
                applied
            ):
                try:
                    await self._managers[
                        LIVINGMEMORY_V8_TYPE
                    ].debug_rebind_revision(
                        database["database_id"],
                        provider_id=provider_id,
                        revision=previous_revision,
                        fingerprint=previous_fingerprint,
                    )
                except BaseException as rollback_error:
                    rollback_errors.append(
                        f"{database['database_id']}: {rollback_error}"
                    )
            if rollback_errors:
                raise RuntimeError(
                    "revision reset failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                )
            raise
        return await self.debug_provider_revisions(provider_id)

    async def debug_repair_database_binding(
        self,
        *,
        database_type: str,
        database_id: str,
        usage_kind: str,
        provider_id: str,
        revision: int,
        allow_non_equivalent: bool = False,
    ) -> dict[str, Any]:
        if usage_kind not in {"embedding", "rerank"}:
            raise ValueError("usage_kind must be embedding or rerank")
        ref = DatabaseRef(database_type, database_id)
        overview = await self.debug_revision_overview()
        database = next(
            (
                item
                for item in overview["databases"]
                if item["database_type"] == ref.database_type
                and item["database_id"] == ref.id
            ),
            None,
        )
        if database is None:
            raise KeyError(ref.key)
        binding = next(
            (
                item
                for item in database.get("bindings", [])
                if item.get("usage_kind") == usage_kind
            ),
            None,
        )
        if binding is None:
            raise ValueError(f"database has no {usage_kind} Provider binding")
        if binding.get("binding_mode") == "follows_latest":
            raise ValueError(
                "this binding follows Provider latest and cannot pin a database revision"
            )
        if str(binding.get("provider_id") or "") != provider_id:
            raise ValueError("revision repair cannot change the bound Provider ID")
        provider = next(
            (
                item
                for item in overview["providers"]
                if item["provider_id"] == provider_id
            ),
            None,
        )
        if provider is None:
            raise KeyError(provider_id)
        target = next(
            (item for item in provider["revisions"] if item["revision"] == revision),
            None,
        )
        if target is None:
            raise KeyError(f"{provider_id}@{revision}")
        expected_kind = provider_kind(str(target["config"].get("type") or ""))
        if expected_kind != usage_kind:
            raise ValueError(
                f"{provider_id}@{revision} is a {expected_kind} Provider revision"
            )
        current = next(
            (
                item
                for item in provider["revisions"]
                if item["revision"] == int(binding.get("provider_revision") or 0)
            ),
            None,
        )
        equivalent = bool(
            current
            and current["functional_sha256"] == target["functional_sha256"]
        )
        if not equivalent and not allow_non_equivalent:
            raise ValueError(
                "binding a functionally different revision requires an explicit "
                "compatibility assertion"
            )
        await self._debug_assert_databases_idle([database])
        before = database
        target_manager = self._managers.get(database_type)
        if target_manager is None:
            raise KeyError(database_type)
        if database_type == LIVINGMEMORY_V8_TYPE:
            await target_manager.debug_rebind_revision(
                database_id,
                provider_id=provider_id,
                revision=revision,
                fingerprint=target["config_sha256"],
            )
        elif database_type == TEXT_MEDIA_V1_TYPE:
            await target_manager.debug_rebind_revision(
                database_id,
                usage_kind=usage_kind,
                provider_id=provider_id,
                revision=revision,
                fingerprint=target["config_sha256"],
            )
        else:
            raise ValueError("database type does not implement revision repair")
        after_overview = await self.debug_revision_overview()
        after = next(
            item
            for item in after_overview["databases"]
            if item["database_type"] == database_type
            and item["database_id"] == database_id
        )
        return {
            "status": "updated",
            "functionally_equivalent": equivalent,
            "asserted_compatible": bool(not equivalent and allow_non_equivalent),
            "before": before,
            "after": after,
        }

    async def update_provider(
        self, provider_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if not await self.control.get_provider(provider_id):
            raise KeyError(provider_id)
        next_provider_id = str(payload.get("id") or provider_id)
        usage = (
            await self.provider_usage(provider_id)
            if next_provider_id != provider_id
            or ("enabled" in payload and not bool(payload["enabled"]))
            else []
        )
        if next_provider_id != provider_id and usage:
            raise ValueError("被数据库引用的 Provider ID 不允许修改")
        if "enabled" in payload and not bool(payload["enabled"]) and usage:
            raise ValueError("该 Provider 正在被数据库使用，不能停用")
        return await self._primary.update_provider(provider_id, payload)

    async def delete_provider(self, provider_id: str) -> None:
        if not await self.control.get_provider(provider_id):
            raise KeyError(provider_id)
        if await self.provider_usage(provider_id):
            raise ValueError("该 Provider 正在被数据库使用，不能删除")
        await self._primary.delete_provider(provider_id)

    async def create_library(self, payload: dict[str, Any]) -> dict[str, Any]:
        database_type = str(payload.get("database_type") or LIVINGMEMORY_V8_TYPE)
        manager, _ = self._manager_for(DatabaseRef(database_type, str(payload["id"])))
        return await manager.create_library(payload)

    async def library_detail(self, database: str | DatabaseRef) -> dict[str, Any]:
        manager, ref = self._manager_for(database)
        return await manager.library_detail(ref.id)

    async def update_library(
        self, database: str | DatabaseRef, payload: dict[str, Any]
    ) -> dict[str, Any]:
        manager, ref = self._manager_for(database)
        return await manager.update_library(ref.id, payload)

    async def set_default(self, database: str | DatabaseRef) -> dict[str, Any]:
        manager, ref = self._manager_for(database)
        return await manager.set_default(ref.id)

    async def copy_library(self, database: str | DatabaseRef, progress=None):
        manager, ref = self._manager_for(database)
        return await manager.copy_library(ref.id, progress=progress)

    async def backup_library(self, database: str | DatabaseRef):
        manager, ref = self._manager_for(database)
        return await manager.backup_library(ref.id)

    async def list_library_backups(self, database: str | DatabaseRef):
        manager, ref = self._manager_for(database)
        return await manager.list_library_backups(ref.id)

    async def library_is_empty(self, database: str | DatabaseRef) -> bool:
        manager, ref = self._manager_for(database)
        return await manager.library_is_empty(ref.id)

    async def import_livingmemory_db(
        self, database: str | DatabaseRef, *args, **kwargs
    ):
        manager, ref = self._manager_for(database)
        return await manager.import_livingmemory_db(ref.id, *args, **kwargs)

    async def delete_library(self, database: str | DatabaseRef):
        manager, ref = self._manager_for(database)
        return await manager.delete_library(ref.id)

    async def rebuild_library(self, database: str | DatabaseRef, *args, **kwargs):
        manager, ref = self._manager_for(database)
        return await manager.rebuild_library(ref.id, *args, **kwargs)

    async def rebuild_graph(self, database: str | DatabaseRef, *args, **kwargs):
        manager, ref = self._manager_for(database)
        return await manager.rebuild_graph(ref.id, *args, **kwargs)

    async def get_runtime(self, database: str | DatabaseRef, *args, **kwargs):
        manager, ref = self._manager_for(database)
        return await manager.get_runtime(ref, *args, **kwargs)

    async def acquire_runtime(self, database: str | DatabaseRef, *args, **kwargs):
        manager, ref = self._manager_for(database)
        return await manager.acquire_runtime(ref, *args, **kwargs)

    async def release_runtime(self, database: str | DatabaseRef, *args, **kwargs):
        manager, ref = self._manager_for(database)
        return await manager.release_runtime(ref, *args, **kwargs)

    async def unload_runtime(self, database: str | DatabaseRef, *args, **kwargs):
        manager, ref = self._manager_for(database)
        return await manager.unload_runtime(ref, *args, **kwargs)

    @asynccontextmanager
    async def runtime_lease(self, database: str | DatabaseRef, *args, **kwargs):
        manager, ref = self._manager_for(database)
        async with manager.runtime_lease(ref, *args, **kwargs) as runtime:
            yield runtime

    def __getattr__(self, name: str):
        # Provider, scheduler and diagnostics services remain process-global.
        # Keep their stable surface while type-specific data operations dispatch
        # through the methods above.
        primary = self.__dict__.get("_primary")
        if primary is None:
            raise AttributeError(name)
        return getattr(primary, name)


from .library_types.livingmemory_v8.manager import (  # noqa: E402
    DEFAULT_LIBRARY_ID,
    DEFAULT_LIBRARY_NAME,
    RuntimeResidencyState,
)

# Deprecated import alias retained for extensions and frozen tests.
LibraryManager = DatabaseManager


__all__ = [
    "DatabaseManager",
    "DEFAULT_LIBRARY_ID",
    "DEFAULT_LIBRARY_NAME",
    "LibraryManager",
    "RuntimeResidencyState",
]
