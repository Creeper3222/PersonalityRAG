from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import time
from contextlib import asynccontextmanager, closing
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .config import (
    AppConfig,
    ConversationConfig,
    MaintenanceConfig,
    ProviderConfig,
    RecallConfig,
)
from .compat import LIVINGMEMORY_DATABASE_VERSION
from .control import ControlStore, LibraryRecord, ProviderRevision
from .identifiers import validate_identifier
from .io_utils import run_blocking
from .jobs import JobManager
from .logger import logger
from .migration import (
    sqlite_backup,
    validate_conversations_db_file,
    validate_livingmemory_db_file,
)
from .providers import (
    build_provider,
    build_rerank_provider,
    config_from_dict,
    provider_kind,
)
from .service import PersonalityRAGService, scan_library_backups
from .storage import Storage


DEFAULT_LIBRARY_ID = "Default"
DEFAULT_LIBRARY_NAME = "贝雷特"
COMPATIBILITY_PAYLOAD = {
    "livingmemory_database_version": LIVINGMEMORY_DATABASE_VERSION,
}
LEGACY_ITEMS = (
    "livingmemory.db",
    "conversations.db",
    "indexes",
    "imports",
    "backups",
    "livingmemory_backups",
    "reports",
    "stopwords",
    "decay_state.json",
)
PROVIDER_HEALTH_TTL_SECONDS = 60.0
RUNTIME_SWEEP_MAX_INTERVAL_SECONDS = 60.0


@dataclass(slots=True)
class RuntimeResidencyState:
    lease_count: int
    last_used_at: float


class LibraryManager:
    def __init__(self, root: Path, config: AppConfig):
        self.root = root
        self.data_dir = root / "data"
        self.config = config
        self.system_path = self.data_dir / "personalityrag_system.db"
        self.control = ControlStore(self.system_path)
        self.runtimes: dict[str, PersonalityRAGService] = {}
        self._runtime_lock = asyncio.Lock()
        self._runtime_condition = asyncio.Condition(self._runtime_lock)
        self._runtime_residency: dict[str, RuntimeResidencyState] = {}
        self._default_library_id = DEFAULT_LIBRARY_ID
        self._runtime_sweeper_task: asyncio.Task[None] | None = None
        self._closing = False
        self.jobs: JobManager | None = None
        self._provider_health_cache: dict[
            tuple[str, int, str], tuple[float, dict[str, Any]]
        ] = {}
        self._provider_health_flights: dict[
            tuple[str, int, str], asyncio.Task[dict[str, Any]]
        ] = {}
        self._provider_health_lock = asyncio.Lock()
        self._adapter_disconnect_waiters: dict[
            tuple[str, str], set[asyncio.Future[None]]
        ] = {}
        self._forced_adapter_connections: dict[
            tuple[str, str], dict[str, Any]
        ] = {}

    async def initialize(self) -> None:
        logger.info("初始化 LibraryManager：data_dir=%s system_db=%s", self.data_dir, self.system_path)
        system_snapshot = {
            suffix: path.read_bytes()
            for suffix in ("", "-wal", "-shm")
            if (path := Path(str(self.system_path) + suffix)).exists()
        }
        migration: dict[str, Any] | None = None
        default_library_id = DEFAULT_LIBRARY_ID
        try:
            await self.control.initialize(self.config.provider)
            self._forced_adapter_connections = {
                (item["library_id"], item["adapter_id"]): item
                for item in await self.control.forced_adapter_connections()
            }
            self.jobs = JobManager(
                Storage(self.data_dir, system_path=self.system_path),
                runtime_lease_factory=self.runtime_lease,
            )
            await self.jobs.clear_for_startup()
            seed = await self.control.get_provider(self.config.provider.id)
            if not seed:
                raise RuntimeError("默认 Provider 初始化失败")
            logger.info("系统 Provider 已加载：provider=%s revision=%s", seed.provider_id, seed.revision)
            migration = await self._migrate_legacy_layout()
            try:
                default_library = await self.control.default_library()
            except RuntimeError:
                default_library = await self.control.ensure_default_library(
                    library_id=DEFAULT_LIBRARY_ID,
                    name=DEFAULT_LIBRARY_NAME,
                    provider_id=seed.provider_id,
                    provider_revision=seed.revision,
                    conversation_settings=asdict(self.config.conversation),
                    recall_settings=asdict(self.config.recall),
                    maintenance_settings=asdict(self.config.maintenance),
                )
                if not default_library.is_default:
                    default_library = await self.control.set_default_library(
                        default_library.id
                    )
            default_library_id = default_library.id
            self._default_library_id = default_library_id
            runtime = await self.get_runtime(default_library_id)
            await self._reconcile_binding(runtime)
            await self._validate_runtime(runtime)
            if migration:
                self._commit_migration_marker(migration)
            self._start_runtime_sweeper()
            logger.info("LibraryManager 初始化完成：default_library=%s", default_library_id)
        except Exception:
            logger.exception("LibraryManager 初始化失败，准备回滚可能的迁移")
            if migration:
                runtime = self.runtimes.pop(default_library_id, None)
                self._runtime_residency.pop(default_library_id, None)
                if runtime is not None:
                    await runtime.close()
                await self._rollback_legacy_layout(migration)
                for suffix in ("", "-wal", "-shm"):
                    path = Path(str(self.system_path) + suffix)
                    path.unlink(missing_ok=True)
                    if suffix in system_snapshot:
                        path.write_bytes(system_snapshot[suffix])
            raise

    def subscribe_adapter_disconnect(
        self,
        library_id: str,
        adapter_id: str,
    ) -> asyncio.Future[None]:
        future = asyncio.get_running_loop().create_future()
        self._adapter_disconnect_waiters.setdefault(
            (library_id, adapter_id), set()
        ).add(future)
        return future

    def unsubscribe_adapter_disconnect(
        self,
        library_id: str,
        adapter_id: str,
        future: asyncio.Future[None],
    ) -> None:
        key = (library_id, adapter_id)
        waiters = self._adapter_disconnect_waiters.get(key)
        if not waiters:
            return
        waiters.discard(future)
        if not waiters:
            self._adapter_disconnect_waiters.pop(key, None)

    def notify_adapter_disconnect(self, library_id: str, adapter_id: str) -> None:
        for future in tuple(
            self._adapter_disconnect_waiters.get((library_id, adapter_id), ())
        ):
            if not future.done():
                future.set_result(None)

    def forced_adapter_connection(
        self,
        library_id: str,
        adapter_id: str,
    ) -> dict[str, Any] | None:
        connection = self._forced_adapter_connections.get(
            (library_id, adapter_id)
        )
        return dict(connection) if connection else None

    def mark_adapter_forced_offline(self, connection: dict[str, Any]) -> None:
        self._forced_adapter_connections[
            (str(connection["library_id"]), str(connection["adapter_id"]))
        ] = dict(connection)

    def clear_adapter_forced_offline(
        self,
        library_id: str,
        adapter_id: str,
    ) -> None:
        self._forced_adapter_connections.pop((library_id, adapter_id), None)

    async def close(self) -> None:
        self._closing = True
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
        if self._runtime_sweeper_task is not None:
            self._runtime_sweeper_task.cancel()
            await asyncio.gather(
                self._runtime_sweeper_task, return_exceptions=True
            )
            self._runtime_sweeper_task = None
        if self.jobs is not None:
            await self.jobs.close()
        flights = list(self._provider_health_flights.values())
        for task in flights:
            task.cancel()
        if flights:
            await asyncio.gather(*flights, return_exceptions=True)
        self._provider_health_flights.clear()
        self._provider_health_cache.clear()
        for runtime in list(self.runtimes.values()):
            await runtime.close()
        self.runtimes.clear()
        self._runtime_residency.clear()

    async def provider_status(
        self,
        runtime: PersonalityRAGService,
        *,
        allow_probe: bool = True,
        ttl_seconds: float = PROVIDER_HEALTH_TTL_SECONDS,
    ) -> dict[str, Any]:
        revision = runtime.provider_revision
        key = (
            revision.provider_id,
            revision.revision,
            revision.config_sha256,
        )
        now = time.time()
        cached = self._provider_health_cache.get(key)
        if cached and now - cached[0] <= ttl_seconds:
            return self._provider_status_payload(
                cached[1],
                checked_at=cached[0],
                cached=True,
                now=now,
            )
        if not allow_probe:
            if cached:
                return self._provider_status_payload(
                    cached[1],
                    checked_at=cached[0],
                    cached=True,
                    now=now,
                )
            return {
                "available": None,
                "status": "not_checked",
                "reason": "provider status is not cached",
                "cached": True,
                "checked_at": None,
                "age_seconds": None,
            }

        async with self._provider_health_lock:
            cached = self._provider_health_cache.get(key)
            now = time.time()
            if cached and now - cached[0] <= ttl_seconds:
                return self._provider_status_payload(
                    cached[1],
                    checked_at=cached[0],
                    cached=True,
                    now=now,
                )
            task = self._provider_health_flights.get(key)
            reused = task is not None
            if task is None:
                task = asyncio.create_task(
                    self._probe_provider_status(key, runtime.provider)
                )
                self._provider_health_flights[key] = task

        result = await asyncio.shield(task)
        checked_at, cached_result = self._provider_health_cache.get(
            key, (time.time(), result)
        )
        return self._provider_status_payload(
            cached_result,
            checked_at=checked_at,
            cached=reused,
            now=time.time(),
        )

    def cached_provider_status(
        self, revision: ProviderRevision | None
    ) -> dict[str, Any]:
        if revision is None:
            return {
                "available": None,
                "status": "not_checked",
                "reason": "provider revision is unavailable",
                "cached": True,
                "checked_at": None,
                "age_seconds": None,
            }
        key = (
            revision.provider_id,
            revision.revision,
            revision.config_sha256,
        )
        cached = self._provider_health_cache.get(key)
        if cached is None:
            return {
                "available": None,
                "status": "not_checked",
                "reason": "provider status is not cached",
                "cached": True,
                "checked_at": None,
                "age_seconds": None,
            }
        return self._provider_status_payload(
            cached[1],
            checked_at=cached[0],
            cached=True,
            now=time.time(),
        )

    async def _probe_provider_status(
        self,
        key: tuple[str, int, str],
        provider: Any,
    ) -> dict[str, Any]:
        try:
            result = await provider.test_connection()
            checked_at = time.time()
            self._provider_health_cache[key] = (checked_at, dict(result))
            return dict(result)
        finally:
            self._provider_health_flights.pop(key, None)

    @staticmethod
    def _provider_status_payload(
        result: dict[str, Any],
        *,
        checked_at: float,
        cached: bool,
        now: float,
    ) -> dict[str, Any]:
        return {
            **result,
            "cached": cached,
            "checked_at": checked_at,
            "age_seconds": max(0.0, now - checked_at),
        }

    async def _migrate_legacy_layout(self) -> dict[str, Any] | None:
        return await run_blocking(self._migrate_legacy_layout_sync)

    def _migrate_legacy_layout_sync(self) -> dict[str, Any] | None:
        source_db = self.data_dir / "livingmemory.db"
        target_root = self.data_dir / "libraries" / DEFAULT_LIBRARY_ID
        marker = self.data_dir / ".multilibrary_migrated_v1.json"
        if not source_db.exists():
            return None

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_root = self.data_dir / "pre_multilibrary_backups" / timestamp
        logger.warning("检测到旧单库布局，开始迁移：source=%s target=%s backup=%s", self.data_dir, target_root, backup_root)
        backup_root.mkdir(parents=True, exist_ok=False)
        copied: dict[str, str] = {}
        for name in LEGACY_ITEMS:
            source = self.data_dir / name
            if not source.exists():
                continue
            destination = backup_root / name
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
                copied[name] = self._sha256(source)
        if "livingmemory.db" not in copied or "conversations.db" not in copied:
            raise RuntimeError("旧单库迁移备份不完整")
        for name, digest in copied.items():
            if self._sha256(backup_root / name) != digest:
                raise RuntimeError(f"旧单库备份哈希校验失败: {name}")

        target_root.mkdir(parents=True, exist_ok=True)
        moved: list[tuple[Path, Path]] = []
        try:
            for name in LEGACY_ITEMS:
                source = self.data_dir / name
                if not source.exists():
                    continue
                destination = target_root / name
                if destination.exists():
                    raise RuntimeError(f"目标记忆库已存在同名数据: {destination}")
                shutil.move(str(source), str(destination))
                moved.append((source, destination))
            if self._sha256(target_root / "livingmemory.db") != copied["livingmemory.db"]:
                raise RuntimeError("迁移后 livingmemory.db 哈希不一致")
            if self._sha256(target_root / "conversations.db") != copied["conversations.db"]:
                raise RuntimeError("迁移后 conversations.db 哈希不一致")
            return {
                "marker": str(marker),
                "backup_root": str(backup_root),
                "target_root": str(target_root),
                "hashes": copied,
            }
        except Exception:
            logger.exception("旧单库布局迁移失败，正在回滚文件移动")
            for source, destination in reversed(moved):
                if destination.exists() and not source.exists():
                    shutil.move(str(destination), str(source))
            raise

    async def _validate_runtime(
        self, runtime: PersonalityRAGService
    ) -> None:
        integrity = await runtime.storage.integrity_report()
        for name in ("livingmemory", "conversations"):
            item = integrity[name]
            if (
                not item.get("exists")
                or item.get("integrity") != "ok"
                or int(item.get("foreign_key_errors", 0)) != 0
            ):
                raise RuntimeError(f"{name} 完整性校验失败")
        stats = await runtime.storage.statistics()
        indexes = runtime.indexes.status()
        document_ids = {int(value) for value in await runtime.storage.document_ids()}
        graph_ids = {int(value) for value in await runtime.storage.graph_entry_ids()}
        indexed_document_ids, indexed_graph_ids = runtime.indexes.indexed_ids()
        if indexed_document_ids != document_ids:
            raise RuntimeError("文档 FAISS ID 数量与数据库不一致")
        if indexed_graph_ids != graph_ids:
            raise RuntimeError("图谱 FAISS ID 数量与数据库不一致")
        if int(indexes["document_vectors"]) != int(stats["total_memories"]):
            raise RuntimeError("文档 FAISS ID 数量与数据库不一致")
        if int(indexes["graph_vectors"]) != int(stats["graph_entries"]):
            raise RuntimeError("图谱 FAISS ID 数量与数据库不一致")

    def _commit_migration_marker(self, migration: dict[str, Any]) -> None:
        marker = Path(migration["marker"])
        marker.write_text(
            json.dumps(
                {
                    "version": 1,
                    "library_id": DEFAULT_LIBRARY_ID,
                    "library_name": DEFAULT_LIBRARY_NAME,
                    "migrated_at": time.time(),
                    "backup": migration["backup_root"],
                    "hashes": migration["hashes"],
                    "validation": "passed",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.warning(
            "旧单库布局迁移标记已写入：library_id=%s backup=%s",
            DEFAULT_LIBRARY_ID,
            migration["backup_root"],
        )

    async def _rollback_legacy_layout(self, migration: dict[str, Any]) -> None:
        await run_blocking(self._rollback_legacy_layout_sync, migration)

    def _rollback_legacy_layout_sync(self, migration: dict[str, Any]) -> None:
        logger.warning("正在回滚旧单库布局迁移：backup=%s", migration["backup_root"])
        target_root = Path(migration["target_root"])
        backup_root = Path(migration["backup_root"])
        shutil.rmtree(target_root, ignore_errors=True)
        for name in LEGACY_ITEMS:
            backup = backup_root / name
            destination = self.data_dir / name
            if not backup.exists() or destination.exists():
                continue
            if backup.is_dir():
                shutil.copytree(backup, destination)
            else:
                shutil.copy2(backup, destination)
        Path(migration["marker"]).unlink(missing_ok=True)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    async def _reconcile_binding(self, runtime: PersonalityRAGService) -> None:
        manifest = runtime.indexes.manifest or {}
        if not manifest.get("generation"):
            return
        record = await self.control.get_library(runtime.library_id)
        if not record:
            return
        provider = await self.control.get_provider(
            record.provider_id, record.provider_revision
        )
        if provider:
            await self.control.bind_library(
                runtime.library_id, provider, manifest
            )

    @staticmethod
    def _indexes_should_be_pending(
        stats: dict[str, Any], indexes: dict[str, Any]
    ) -> bool:
        counts = (
            stats.get("total_memories"),
            stats.get("graph_nodes"),
            stats.get("graph_edges"),
            stats.get("graph_entries"),
            stats.get("atom_count"),
            (stats.get("conversation_counts") or {}).get("sessions"),
        )
        library_is_empty = all(int(value or 0) == 0 for value in counts)
        return (
            library_is_empty
            and int(indexes.get("document_vectors") or 0) == 0
            and int(indexes.get("graph_vectors") or 0) == 0
        )

    def _normalize_indexes_for_response(
        self, stats: dict[str, Any], indexes: dict[str, Any]
    ) -> dict[str, Any]:
        if not self._indexes_should_be_pending(stats, indexes):
            return indexes
        return {
            **indexes,
            "generation": None,
            "manifest": None,
            "document_vectors": 0,
            "graph_vectors": 0,
        }

    def _start_runtime_sweeper(self) -> None:
        if self._runtime_sweeper_task is None or self._runtime_sweeper_task.done():
            self._runtime_sweeper_task = asyncio.create_task(
                self._runtime_sweeper_loop(),
                name="personalityrag-runtime-residency",
            )

    async def _runtime_sweeper_loop(self) -> None:
        while not self._closing:
            idle_seconds = max(
                60.0,
                float(self.config.runtime_residency.idle_minutes) * 60.0,
            )
            interval = min(
                RUNTIME_SWEEP_MAX_INTERVAL_SECONDS,
                max(5.0, idle_seconds / 2.0),
            )
            try:
                await asyncio.sleep(interval)
                await self.sweep_runtimes()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("记忆库 runtime 空闲回收巡检失败")

    async def _runtime_has_active_jobs(self, library_id: str) -> bool:
        return await self.control.has_running_jobs(library_id)

    async def _close_runtime_locked(
        self,
        library_id: str,
        *,
        reason: str,
        suppress_errors: bool = True,
    ) -> bool:
        runtime = self.runtimes.pop(library_id, None)
        self._runtime_residency.pop(library_id, None)
        if runtime is None:
            return False
        try:
            await runtime.close()
        except Exception:
            logger.exception(
                "记忆库 runtime 释放失败：library_id=%s reason=%s",
                library_id,
                reason,
            )
            if not suppress_errors:
                raise
        else:
            logger.info(
                "记忆库 runtime 已释放：library_id=%s reason=%s",
                library_id,
                reason,
            )
        return True

    async def _evict_lru_until_locked(
        self, max_non_default: int, *, reason: str
    ) -> list[str]:
        evicted: list[str] = []
        target = max(0, int(max_non_default))
        while True:
            non_default = [
                library_id
                for library_id in self.runtimes
                if library_id != self._default_library_id
            ]
            if len(non_default) <= target:
                break
            candidates = sorted(
                non_default,
                key=lambda item: self._runtime_residency.get(
                    item,
                    RuntimeResidencyState(0, 0.0),
                ).last_used_at,
            )
            selected: str | None = None
            for library_id in candidates:
                state = self._runtime_residency.get(library_id)
                if state is not None and state.lease_count > 0:
                    continue
                if await self._runtime_has_active_jobs(library_id):
                    continue
                selected = library_id
                break
            if selected is None:
                break
            await self._close_runtime_locked(selected, reason=reason)
            evicted.append(selected)
        return evicted

    async def _load_runtime_locked(
        self, library_id: str
    ) -> PersonalityRAGService:
        current = self.runtimes.get(library_id)
        if current is not None:
            return current
        library = await self.control.get_library(library_id)
        if not library:
            raise KeyError(library_id)
        if library_id != self._default_library_id:
            limit = max(
                1,
                int(self.config.runtime_residency.max_non_default_runtimes),
            )
            await self._evict_lru_until_locked(
                limit - 1,
                reason="capacity",
            )
        logger.info("懒加载记忆库 runtime：library_id=%s", library_id)
        provider = await self.control.get_provider(
            library.provider_id, library.provider_revision
        )
        if not provider:
            raise RuntimeError(
                f"记忆库 {library_id} 绑定的 Provider revision 不存在"
            )
        rerank_provider = None
        if library.rerank_provider_id:
            rerank_provider = await self.control.get_provider(
                library.rerank_provider_id
            )
            if not rerank_provider:
                logger.warning(
                    "记忆库绑定的 Rerank Provider 不存在，将跳过重排：library_id=%s provider=%s",
                    library_id,
                    library.rerank_provider_id,
                )
        runtime_config = replace(
            self.config,
            conversation=ConversationConfig(
                **{
                    **asdict(self.config.conversation),
                    **library.conversation_settings,
                }
            ),
            recall=RecallConfig(
                **{
                    **asdict(self.config.recall),
                    **library.recall_settings,
                }
            ),
            maintenance=MaintenanceConfig(
                **{
                    **asdict(self.config.maintenance),
                    **library.maintenance_settings,
                }
            ),
        )
        runtime = PersonalityRAGService(
            self.root,
            runtime_config,
            self.data_dir / "libraries" / library_id,
            library_id=library_id,
            default_persona_id=library.default_persona_id,
            provider_revision=provider,
            rerank_provider_revision=rerank_provider,
            system_path=self.system_path,
        )
        await runtime.initialize()
        now = time.monotonic()
        self.runtimes[library_id] = runtime
        self._runtime_residency[library_id] = RuntimeResidencyState(
            lease_count=0,
            last_used_at=now,
        )
        logger.info("记忆库 runtime 已加载：library_id=%s", library_id)
        return runtime

    async def get_runtime(self, library_id: str) -> PersonalityRAGService:
        async with self._runtime_lock:
            runtime = await self._load_runtime_locked(library_id)
            self._runtime_residency[library_id].last_used_at = time.monotonic()
            return runtime

    async def acquire_runtime(
        self, library_id: str, *, touch: bool = True
    ) -> PersonalityRAGService:
        async with self._runtime_lock:
            runtime = await self._load_runtime_locked(library_id)
            state = self._runtime_residency[library_id]
            state.lease_count += 1
            if touch:
                state.last_used_at = time.monotonic()
            return runtime

    async def release_runtime(self, library_id: str, *, touch: bool = True) -> None:
        should_converge = False
        async with self._runtime_condition:
            state = self._runtime_residency.get(library_id)
            if state is None:
                return
            state.lease_count = max(0, state.lease_count - 1)
            if touch:
                state.last_used_at = time.monotonic()
            should_converge = state.lease_count == 0
            self._runtime_condition.notify_all()
        if should_converge:
            await self.sweep_runtimes(expire_idle=False)

    @asynccontextmanager
    async def runtime_lease(self, library_id: str, *, touch: bool = True):
        runtime = await self.acquire_runtime(library_id, touch=touch)
        try:
            yield runtime
        finally:
            await self.release_runtime(library_id, touch=touch)

    async def unload_runtime(self, library_id: str, *, reason: str) -> bool:
        async with self._runtime_condition:
            while (
                state := self._runtime_residency.get(library_id)
            ) is not None and state.lease_count > 0:
                await self._runtime_condition.wait()
            return await self._close_runtime_locked(
                library_id,
                reason=reason,
                suppress_errors=False,
            )

    async def sweep_runtimes(self, *, expire_idle: bool = True) -> list[str]:
        evicted: list[str] = []
        async with self._runtime_lock:
            if expire_idle:
                now = time.monotonic()
                idle_seconds = (
                    max(1, int(self.config.runtime_residency.idle_minutes)) * 60.0
                )
                candidates = sorted(
                    (
                        library_id
                        for library_id in self.runtimes
                        if library_id != self._default_library_id
                    ),
                    key=lambda item: self._runtime_residency.get(
                        item,
                        RuntimeResidencyState(0, 0.0),
                    ).last_used_at,
                )
                for library_id in candidates:
                    state = self._runtime_residency.get(library_id)
                    if state is None or state.lease_count > 0:
                        continue
                    if now - state.last_used_at < idle_seconds:
                        continue
                    if await self._runtime_has_active_jobs(library_id):
                        continue
                    if await self._close_runtime_locked(
                        library_id,
                        reason="idle",
                    ):
                        evicted.append(library_id)
            evicted.extend(
                await self._evict_lru_until_locked(
                    max(
                        1,
                        int(
                            self.config.runtime_residency.max_non_default_runtimes
                        ),
                    ),
                    reason="capacity",
                )
            )
        return evicted

    async def apply_runtime_residency(self) -> list[str]:
        self.config.runtime_residency.idle_minutes = max(
            1, int(self.config.runtime_residency.idle_minutes)
        )
        self.config.runtime_residency.max_non_default_runtimes = max(
            1,
            int(self.config.runtime_residency.max_non_default_runtimes),
        )
        return await self.sweep_runtimes()

    def runtime_residency_status(self) -> dict[str, Any]:
        return {
            "default_library_id": self._default_library_id,
            "loaded_library_ids": list(self.runtimes),
            "runtimes": {
                library_id: {
                    "lease_count": state.lease_count,
                    "last_used_at": state.last_used_at,
                }
                for library_id, state in self._runtime_residency.items()
            },
        }

    async def default_runtime(self) -> PersonalityRAGService:
        record = await self.control.default_library()
        return await self.get_runtime(record.id)

    @staticmethod
    def _adapter_busy_summary(job: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "busy": bool(job),
            "job": job,
        }

    async def list_libraries(
        self, *, stats_mode: str = "full"
    ) -> list[dict[str, Any]]:
        if stats_mode not in {"full", "summary"}:
            raise ValueError("invalid library stats mode")
        result = []
        records = await self.control.list_libraries()
        library_ids = [record.id for record in records]
        provider_bindings = {
            (record.provider_id, record.provider_revision) for record in records
        }
        provider_bindings.update(
            (record.rerank_provider_id, None)
            for record in records
            if record.rerank_provider_id
        )
        provider_map = await self.control.get_providers_bulk(provider_bindings)
        adapter_map = await self.control.active_adapter_connections_map(library_ids)
        busy_map = (
            await self.jobs.active_long_jobs_map(library_ids)
            if self.jobs is not None
            else await self.control.active_long_jobs_map(library_ids)
        )
        for record in records:
            provider = provider_map.get(
                (record.provider_id, record.provider_revision)
            )
            rerank_provider = (
                provider_map.get((record.rerank_provider_id, None))
                if record.rerank_provider_id
                else None
            )
            runtime = self.runtimes.get(record.id)
            if runtime is not None:
                stats = (
                    await runtime.storage.summary_statistics()
                    if stats_mode == "summary"
                    else await runtime.storage.statistics()
                )
                indexes = self._normalize_indexes_for_response(
                    stats, runtime.indexes.status()
                )
            else:
                library_dir = self.data_dir / "libraries" / record.id
                storage = Storage(library_dir, system_path=self.system_path)
                stats = (
                    await storage.summary_statistics()
                    if stats_mode == "summary"
                    else await storage.statistics()
                )
                indexes = self._normalize_indexes_for_response(
                    stats, self._offline_index_status(library_dir, record)
                )
            result.append(
                {
                    **record.public(),
                    "stats": stats,
                    "indexes": indexes,
                    "provider": provider.public() if provider else None,
                    "rerank_provider": (
                        rerank_provider.public() if rerank_provider else None
                    ),
                    "adapter_connections": adapter_map.get(record.id, []),
                    "adapter_busy": self._adapter_busy_summary(
                        busy_map.get(record.id)
                    ),
                    "compatibility": dict(COMPATIBILITY_PAYLOAD),
                }
            )
        return result

    @staticmethod
    def _offline_index_status(
        library_dir: Path, record: LibraryRecord
    ) -> dict[str, Any]:
        index_root = library_dir / "indexes"
        current_file = index_root / "CURRENT"
        manifest: dict[str, Any] | None = None
        generation: str | None = None
        if current_file.exists():
            generation = current_file.read_text(encoding="utf-8").strip() or None
        if generation:
            manifest_path = index_root / generation / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    manifest = None
        return {
            "generation": generation,
            "manifest": manifest,
            "document_vectors": int(
                (manifest or {}).get("document_count", 0)
            ),
            "graph_vectors": int(
                (manifest or {}).get("graph_entry_count", 0)
            ),
            "provider_id": record.provider_id,
            "provider_revision": record.provider_revision,
        }

    async def library_detail(self, library_id: str) -> dict[str, Any]:
        record = await self.control.get_library(library_id)
        if not record:
            raise KeyError(library_id)
        provider = await self.control.get_provider(
            record.provider_id, record.provider_revision
        )
        rerank_provider = (
            await self.control.get_provider(record.rerank_provider_id)
            if record.rerank_provider_id
            else None
        )
        library_dir = self.data_dir / "libraries" / record.id
        stats = await Storage(
            library_dir,
            system_path=self.system_path,
        ).statistics()
        return {
            **record.public(),
            "stats": stats,
            "indexes": self._normalize_indexes_for_response(
                stats, self._offline_index_status(library_dir, record)
            ),
            "provider": provider.public() if provider else None,
            "rerank_provider": rerank_provider.public() if rerank_provider else None,
            "adapter_connections": await self.control.active_adapter_connections(
                record.id
            ),
            "adapter_busy": self._adapter_busy_summary(
                await self.jobs.active_long_job(record.id)
                if self.jobs is not None
                else await self.control.active_long_job(record.id)
            ),
            "compatibility": dict(COMPATIBILITY_PAYLOAD),
        }

    async def list_library_backups(self, library_id: str) -> list[dict[str, Any]]:
        library_dir = self.data_dir / "libraries" / library_id
        return await run_blocking(scan_library_backups, library_dir)

    async def create_library(self, payload: dict[str, Any]) -> dict[str, Any]:
        provider_id = validate_identifier(
            payload.get("provider_id"), field="Provider ID"
        )
        provider = await self.control.get_provider(provider_id)
        if provider and provider_kind(provider.config.type) != "embedding":
            raise ValueError("记忆库必须绑定 Embedding Provider")
        if not provider:
            raise ValueError("指定的 Provider 不存在")
        if not provider.config.enabled:
            raise ValueError("指定的 Provider 未启用")
        rerank_provider_id = str(payload.get("rerank_provider_id") or "")
        if rerank_provider_id:
            rerank_provider_id = validate_identifier(
                rerank_provider_id, field="Provider ID"
            )
            rerank_provider = await self.control.get_provider(rerank_provider_id)
            if not rerank_provider:
                raise ValueError("指定的 Rerank Provider 不存在")
            if provider_kind(rerank_provider.config.type) != "rerank":
                raise ValueError("Rerank 绑定必须选择 Rerank Provider")
            if not rerank_provider.config.enabled:
                raise ValueError("指定的 Rerank Provider 未启用")
            payload["rerank_provider_id"] = rerank_provider_id
        else:
            payload["rerank_provider_id"] = ""
        payload = {
            **payload,
            "conversation_settings": payload.get("conversation_settings")
            or asdict(self.config.conversation),
            "recall_settings": payload.get("recall_settings")
            or asdict(self.config.recall),
            "maintenance_settings": payload.get("maintenance_settings")
            or asdict(self.config.maintenance),
        }
        record = await self.control.create_library(payload, provider)
        try:
            await self.get_runtime(record.id)
            logger.info(
                "新记忆库创建完成，索引保持待构建：library_id=%s provider=%s revision=%s",
                record.id,
                provider.provider_id,
                provider.revision,
            )
        except Exception:
            logger.exception("新记忆库初始化失败，正在清理：library_id=%s", record.id)
            await self.unload_runtime(record.id, reason="create_failed")
            await self.control.mark_library_deleted(record.id)
            shutil.rmtree(
                self.data_dir / "libraries" / record.id, ignore_errors=True
            )
            raise
        return await self.library_detail(record.id)

    async def update_library(
        self, library_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        record = await self.control.get_library(library_id)
        if not record:
            raise KeyError(library_id)
        next_library_id = (
            validate_identifier(payload["id"], field="记忆库 ID")
            if "id" in payload
            else record.id
        )
        rename_requested = next_library_id != record.id
        requested_provider_id = payload.pop("provider_id", None)
        if requested_provider_id is not None:
            requested_provider_id = validate_identifier(
                requested_provider_id, field="Provider ID"
            )
        if requested_provider_id and requested_provider_id != record.provider_id:
            provider = await self.control.get_provider(requested_provider_id)
            if not provider:
                raise ValueError("指定的 Embedding Provider 不存在")
            if provider_kind(provider.config.type) != "embedding":
                raise ValueError("记忆库必须绑定 Embedding Provider")
            if not provider.config.enabled:
                raise ValueError("指定的 Embedding Provider 未启用")
        if "rerank_provider_id" in payload:
            rerank_provider_id = str(payload.get("rerank_provider_id") or "")
            if rerank_provider_id:
                rerank_provider_id = validate_identifier(
                    rerank_provider_id, field="Provider ID"
                )
                rerank_provider = await self.control.get_provider(rerank_provider_id)
                if not rerank_provider:
                    raise ValueError("指定的 Rerank Provider 不存在")
                if provider_kind(rerank_provider.config.type) != "rerank":
                    raise ValueError("Rerank 绑定必须选择 Rerank Provider")
                if not rerank_provider.config.enabled:
                    raise ValueError("指定的 Rerank Provider 未启用")
                payload["rerank_provider_id"] = rerank_provider_id
            else:
                payload["rerank_provider_id"] = ""
        rerank_changed = (
            "rerank_provider_id" in payload
            and str(payload.get("rerank_provider_id") or "")
            != str(record.rerank_provider_id or "")
        )
        settings_changed = (
            "conversation_settings" in payload
            or "recall_settings" in payload
            or "maintenance_settings" in payload
        )
        reload_runtime = (
            rename_requested
            or settings_changed
        )
        if rename_requested:
            if await self.control.active_adapter_connections(record.id):
                raise ValueError("记忆库已被适配器连接，不能修改 ID")
            if await self.control.has_running_jobs(record.id):
                raise ValueError("记忆库存在进行中的任务，暂时不能修改 ID")
            if await self._library_copy_target_reserved(next_library_id):
                raise ValueError(f"记忆库 ID 已存在：{next_library_id}")
        if reload_runtime:
            await self.unload_runtime(record.id, reason="library_settings_changed")
        if rename_requested:
            source_dir = self.data_dir / "libraries" / record.id
            target_dir = self.data_dir / "libraries" / next_library_id
            if not source_dir.exists():
                raise ValueError(f"记忆库目录不存在，无法修改 ID：{source_dir}")
            moved = False
            try:
                source_dir.replace(target_dir)
                moved = True
                self._rewrite_library_manifests(target_dir, next_library_id)
                await self.control.update_library(record.id, payload)
            except Exception:
                if moved and target_dir.exists():
                    try:
                        self._rewrite_library_manifests(target_dir, record.id)
                    except Exception:
                        logger.exception(
                            "记忆库改名回滚时重写 manifest 失败：source=%s target=%s",
                            record.id,
                            next_library_id,
                        )
                    if not source_dir.exists():
                        target_dir.replace(source_dir)
                raise
        else:
            await self.control.update_library(record.id, payload)
            if "default_persona_id" in payload and record.id in self.runtimes:
                self.runtimes[record.id].default_persona_id = str(
                    payload.get("default_persona_id") or ""
                ).strip()
            if rerank_changed and not reload_runtime:
                if record.id in self.runtimes:
                    provider = (
                        await self.control.get_provider(
                            str(payload.get("rerank_provider_id") or "")
                        )
                        if payload.get("rerank_provider_id")
                        else None
                    )
                    async with self.runtime_lease(record.id) as runtime:
                        await runtime.set_rerank_provider(provider)
        if record.is_default:
            self._default_library_id = next_library_id
            await self.get_runtime(next_library_id)
        return await self.library_detail(next_library_id)

    async def set_default(self, library_id: str) -> dict[str, Any]:
        await self.control.set_default_library(library_id)
        self._default_library_id = library_id
        await self.get_runtime(library_id)
        await self.sweep_runtimes(expire_idle=False)
        return await self.library_detail(library_id)

    async def refresh_default_library(self, *, load: bool = False) -> str:
        record = await self.control.default_library()
        self._default_library_id = record.id
        if load:
            await self.get_runtime(record.id)
        return record.id

    async def copy_library(self, library_id: str, progress=None) -> dict[str, Any]:
        source = await self.control.get_library(library_id)
        if not source:
            raise KeyError(library_id)
        source_dir = self.data_dir / "libraries" / source.id
        if not source_dir.exists():
            raise ValueError(f"源记忆库目录不存在：{source_dir}")
        if progress:
            await progress(0.03, "正在准备副本 ID 与目标目录")
        target_id, target_name = await self._next_copy_identity(source)
        target_dir = self.data_dir / "libraries" / target_id

        provider = await self.control.get_provider(
            source.provider_id, source.provider_revision
        )
        if not provider:
            raise ValueError("源记忆库绑定的 Provider revision 不存在")
        if progress:
            await progress(0.12, f"副本目标已确定：{target_id}")

        tmp_dir = self.data_dir / "libraries" / f".{target_id}.copying-{time.strftime('%Y%m%d-%H%M%S')}"
        logger.warning(
            "复制记忆库开始：source=%s target=%s source_dir=%s",
            source.id,
            target_id,
            source_dir,
        )
        record_created = False
        try:
            if progress:
                await progress(0.2, "正在复制记忆库目录和 SQLite 快照")
            await run_blocking(self._copy_library_directory, source_dir, tmp_dir)
            if progress:
                await progress(0.72, "正在重写副本索引 manifest")
            self._rewrite_library_manifests(tmp_dir, target_id)
            if progress:
                await progress(0.82, "正在提交副本目录")
            tmp_dir.replace(target_dir)
            if progress:
                await progress(0.9, "正在注册副本记忆库")
            record = await self.control.create_library(
                {
                    "id": target_id,
                    "name": target_name,
                    "description": source.description,
                    "default_persona_id": source.default_persona_id,
                    "rerank_provider_id": source.rerank_provider_id,
                    "conversation_settings": source.conversation_settings,
                    "recall_settings": source.recall_settings,
                    "maintenance_settings": source.maintenance_settings,
                    "metadata": source.metadata,
                },
                provider,
            )
            record_created = True
            current = target_dir / "indexes" / "CURRENT"
            if current.exists():
                generation = current.read_text(encoding="utf-8").strip()
                manifest_path = target_dir / "indexes" / generation / "manifest.json"
                if generation and manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    await self.control.bind_library(target_id, provider, manifest)
            logger.warning(
                "复制记忆库完成：source=%s target=%s",
                source.id,
                target_id,
            )
            if progress:
                await progress(1.0, f"复制完成：{target_id}")
            return {
                **record.public(),
                "stats": await Storage(target_dir, system_path=self.system_path).statistics(),
                "indexes": self._offline_index_status(target_dir, record),
                "provider": provider.public(),
                "compatibility": dict(COMPATIBILITY_PAYLOAD),
            }
        except Exception:
            logger.exception("复制记忆库失败，正在清理副本：source=%s target=%s", source.id, target_id)
            await run_blocking(shutil.rmtree, target_dir, ignore_errors=True)
            await run_blocking(shutil.rmtree, tmp_dir, ignore_errors=True)
            if record_created:
                try:
                    await self.control.mark_library_deleted(target_id)
                except Exception:
                    pass
            raise

    async def _next_copy_identity(self, source: LibraryRecord) -> tuple[str, str]:
        for index in range(1, 1000):
            target_id = (
                f"{source.id}_copy" if index == 1 else f"{source.id}_copy{index}"
            )
            target_name = (
                f"{source.name}(副本)" if index == 1 else f"{source.name}(副本{index})"
            )
            if not await self._library_copy_target_reserved(target_id):
                return target_id, target_name
        raise ValueError("无法生成可用的记忆库副本 ID，请先清理过多副本")

    async def _library_copy_target_reserved(self, library_id: str) -> bool:
        if await self.control.library_id_exists_any(library_id):
            return True
        library_root = self.data_dir / "libraries"
        if (library_root / library_id).exists():
            return True
        if any(library_root.glob(f".{library_id}.copying-*")):
            return True
        return False

    @staticmethod
    def _copy_library_directory(source_dir: Path, target_dir: Path) -> None:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(
            source_dir,
            target_dir,
            ignore=shutil.ignore_patterns("*.db-wal", "*.db-shm"),
        )
        for name in ("livingmemory.db", "conversations.db"):
            source_db = source_dir / name
            target_db = target_dir / name
            if not source_db.exists():
                continue
            tmp_db = target_db.with_suffix(target_db.suffix + ".tmp")
            if tmp_db.exists():
                tmp_db.unlink()
            with closing(sqlite3.connect(source_db)) as source_conn, closing(sqlite3.connect(tmp_db)) as target_conn:
                source_conn.backup(target_conn)
            tmp_db.replace(target_db)

    @staticmethod
    def _rewrite_library_manifests(library_dir: Path, library_id: str) -> None:
        index_root = library_dir / "indexes"
        if not index_root.exists():
            return
        for manifest_path in index_root.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            manifest["library_id"] = library_id
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    async def backup_library(self, library_id: str) -> dict[str, Any]:
        async with self.runtime_lease(library_id) as runtime:
            path = await runtime.backup()
        return {"library_id": library_id, "path": str(path)}

    async def library_is_empty(self, library_id: str) -> bool:
        async with self.runtime_lease(library_id) as runtime:
            stats = await runtime.storage.statistics()
        return all(
            int(value or 0) == 0
            for value in (
                stats.get("total_memories"),
                stats.get("graph_nodes"),
                stats.get("graph_edges"),
                stats.get("graph_entries"),
                stats.get("atom_count"),
                (stats.get("conversation_counts") or {}).get("sessions"),
            )
        )

    async def import_livingmemory_db(
        self,
        library_id: str,
        source_db: Path,
        progress=None,
        *,
        conversations_db: Path | None = None,
    ) -> dict[str, Any]:
        record = await self.control.get_library(library_id)
        if not record:
            raise KeyError(library_id)
        source_report = await run_blocking(
            validate_livingmemory_db_file,
            source_db,
        )
        conversations_report = (
            await run_blocking(validate_conversations_db_file, conversations_db)
            if conversations_db is not None
            else None
        )
        if progress:
            validated_message = "已验证 LivingMemory 核心数据库"
            if conversations_report is not None:
                validated_message += "与消息记录数据库"
            await progress(0.02, validated_message)
        runtime = await self.get_runtime(library_id)
        if not await self.library_is_empty(library_id):
            raise ValueError("只有全新空记忆库可以导入 livingmemory.db")

        run_id = time.strftime("%Y%m%d-%H%M%S-") + os.urandom(4).hex()
        library_dir = self.data_dir / "libraries" / library_id
        import_dir = library_dir / "imports" / run_id
        archive_dir = import_dir / "source_archive"
        report_dir = library_dir / "reports"
        archive_dir.mkdir(parents=True, exist_ok=False)
        report_dir.mkdir(parents=True, exist_ok=True)
        target_db = library_dir / "livingmemory.db"
        target_conversations_db = library_dir / "conversations.db"
        rollback_db = import_dir / "pre_import_empty_livingmemory.db"
        rollback_conversations_db = import_dir / "pre_import_empty_conversations.db"
        report_path = report_dir / f"livingmemory-db-import-{run_id}.json"
        logger.warning(
            "LivingMemory 单文件导入开始：library_id=%s source=%s conversations=%s run_id=%s",
            library_id,
            source_db,
            conversations_db or "",
            run_id,
        )
        try:
            if progress:
                await progress(0.04, "正在归档上传的 livingmemory.db")
            await run_blocking(
                sqlite_backup,
                source_db,
                archive_dir / "livingmemory.db",
            )
            if conversations_db is not None:
                await run_blocking(
                    sqlite_backup,
                    conversations_db,
                    archive_dir / "conversations.db",
                )
            if target_db.exists():
                await run_blocking(sqlite_backup, target_db, rollback_db)
            if target_conversations_db.exists():
                await run_blocking(
                    sqlite_backup,
                    target_conversations_db,
                    rollback_conversations_db,
                )
            if progress:
                await progress(0.06, "正在替换目标空库核心数据库")
            await self.unload_runtime(library_id, reason="livingmemory_import")
            for db_path in (target_db, target_conversations_db):
                for suffix in ("-wal", "-shm"):
                    Path(str(db_path) + suffix).unlink(missing_ok=True)
            tmp_db = target_db.with_suffix(".db.importing")
            tmp_db.unlink(missing_ok=True)
            await run_blocking(sqlite_backup, source_db, tmp_db)
            tmp_db.replace(target_db)
            if conversations_db is not None:
                tmp_conversations_db = target_conversations_db.with_suffix(
                    ".db.importing"
                )
                tmp_conversations_db.unlink(missing_ok=True)
                await run_blocking(
                    sqlite_backup,
                    conversations_db,
                    tmp_conversations_db,
                )
                tmp_conversations_db.replace(target_conversations_db)
            if progress:
                await progress(0.08, "正在初始化兼容表与 FTS")
            runtime = await self.get_runtime(library_id)
            await runtime.storage.initialize()
            runtime.text = runtime.text.__class__(runtime.data_dir / "stopwords")
            runtime.retrieval.text = runtime.text
            if progress:
                await progress(0.10, "正在重建导入库索引")
            async def rebuild_progress(value: float, message: str) -> None:
                if progress:
                    await progress(0.10 + max(0.0, min(1.0, value)) * 0.89, message)

            rebuild = await self.rebuild_library(library_id, None, rebuild_progress)
            await self.control.update_library_metadata(
                library_id,
                {
                    "livingmemory_database_version": source_report.get(
                        "db_version"
                    )
                },
            )
            stats = await runtime.storage.statistics()
            report = {
                "run_id": run_id,
                "library_id": library_id,
                "source": source_report,
                "conversations_source": conversations_report,
                "stats": stats,
                "graph_recovery": rebuild.get("graph_recovery"),
                "rebuild": rebuild,
                "completed_at": time.time(),
            }
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if progress:
                await progress(1.0, "导入与索引重建完成")
            logger.warning(
                "LivingMemory 单文件导入完成：library_id=%s run_id=%s generation=%s",
                library_id,
                run_id,
                (rebuild.get("manifest") or {}).get("generation"),
            )
            return {
                "run_id": run_id,
                "library_id": library_id,
                "source": source_report,
                "conversations_source": conversations_report,
                "stats": stats,
                "graph_recovery": rebuild.get("graph_recovery"),
                "rebuild": rebuild,
                "report_path": str(report_path),
            }
        except Exception:
            logger.exception(
                "LivingMemory 单文件导入失败，正在回滚：library_id=%s run_id=%s",
                library_id,
                run_id,
            )
            await self.unload_runtime(library_id, reason="livingmemory_import_rollback")
            if rollback_db.exists():
                for suffix in ("-wal", "-shm"):
                    Path(str(target_db) + suffix).unlink(missing_ok=True)
                await run_blocking(sqlite_backup, rollback_db, target_db)
            if rollback_conversations_db.exists():
                for suffix in ("-wal", "-shm"):
                    Path(str(target_conversations_db) + suffix).unlink(missing_ok=True)
                await run_blocking(
                    sqlite_backup,
                    rollback_conversations_db,
                    target_conversations_db,
                )
            await self.get_runtime(library_id)
            raise
        finally:
            source_db.unlink(missing_ok=True)
            if conversations_db is not None:
                conversations_db.unlink(missing_ok=True)

    async def delete_library(self, library_id: str) -> dict[str, Any]:
        record = await self.control.get_library(library_id)
        if not record:
            raise KeyError(library_id)
        if record.is_default:
            raise ValueError("默认记忆库不能删除")
        if await self.control.active_adapter_connections(library_id):
            raise ValueError("记忆库已被适配器连接，不能删除")
        if await self.control.has_running_jobs(library_id):
            raise ValueError("记忆库存在运行中的任务，不能删除")
        source = self.data_dir / "libraries" / library_id
        trash_dir = (
            self.data_dir
            / "trash"
            / "libraries"
            / f"{library_id}-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        trash_memory_db = trash_dir / "livingmemory.db"
        trash_conversations_db = trash_dir / "conversations.db"
        staging = (
            self.data_dir
            / "libraries"
            / f".{library_id}.deleting-{int(time.time())}-{os.urandom(3).hex()}"
        )
        await self.unload_runtime(library_id, reason="library_delete")
        if not source.exists():
            raise FileNotFoundError(f"记忆库目录不存在：{source}")

        trash_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(staging))
        try:
            if (staging / "livingmemory.db").exists():
                trash_dir.mkdir(parents=True, exist_ok=True)
                await run_blocking(
                    sqlite_backup,
                    staging / "livingmemory.db",
                    trash_memory_db,
                )
            if (staging / "conversations.db").exists():
                trash_dir.mkdir(parents=True, exist_ok=True)
                await run_blocking(
                    sqlite_backup,
                    staging / "conversations.db",
                    trash_conversations_db,
                )
            await self.control.mark_library_deleted(library_id)
            shutil.rmtree(staging, ignore_errors=True)
        except Exception:
            if source.exists():
                shutil.rmtree(source, ignore_errors=True)
            if staging.exists():
                shutil.move(str(staging), str(source))
            raise
        return {
            "library_id": library_id,
            "backup": str(trash_dir),
            "trash": str(trash_dir),
            "files": {
                "livingmemory": str(trash_memory_db) if trash_memory_db.exists() else None,
                "conversations": str(trash_conversations_db) if trash_conversations_db.exists() else None,
            },
        }

    async def rebuild_library(
        self,
        library_id: str,
        provider_id: str | None,
        progress=None,
    ) -> dict[str, Any]:
        runtime = await self.get_runtime(library_id)
        record = await self.control.get_library(library_id)
        if not record:
            raise KeyError(library_id)
        provider = await self.control.get_provider(
            provider_id or record.provider_id
        )
        if provider and provider_kind(provider.config.type) != "embedding":
            raise ValueError("索引重建必须使用 Embedding Provider")
        if not provider:
            raise ValueError("Provider 不存在")
        if not provider.config.enabled:
            raise ValueError("Provider 未启用")
        logger.warning(
            "开始重建记忆库索引：library_id=%s provider=%s revision=%s",
            library_id,
            provider.provider_id,
            provider.revision,
        )
        result = await runtime.rebuild_with_provider(provider, progress)
        await self.control.bind_library(
            library_id, provider, result["manifest"]
        )
        logger.warning(
            "记忆库索引重建并绑定完成：library_id=%s provider=%s revision=%s generation=%s",
            library_id,
            provider.provider_id,
            provider.revision,
            (result.get("manifest") or {}).get("generation"),
        )
        return result

    async def rebuild_graph(self, library_id: str, progress=None) -> dict[str, Any]:
        runtime = await self.get_runtime(library_id)
        record = await self.control.get_library(library_id)
        if not record:
            raise KeyError(library_id)
        provider = await self.control.get_provider(record.provider_id)
        if provider and provider_kind(provider.config.type) != "embedding":
            raise ValueError("图记忆重建必须使用 Embedding Provider")
        if not provider:
            raise ValueError("Provider 不存在")
        if not provider.config.enabled:
            raise ValueError("Provider 未启用")
        logger.warning(
            "开始重建记忆库图数据与索引：library_id=%s provider=%s revision=%s",
            library_id,
            provider.provider_id,
            provider.revision,
        )
        result = await runtime.rebuild_graph(progress)
        await self.control.bind_library(
            library_id,
            provider,
            result["manifest"],
        )
        logger.warning(
            "记忆库图数据与索引重建完成：library_id=%s provider=%s revision=%s generation=%s",
            library_id,
            provider.provider_id,
            provider.revision,
            (result.get("manifest") or {}).get("generation"),
        )
        return result

    async def create_provider(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = await self._with_context_length_metadata(payload, force=True)
        return (await self.control.create_provider(payload)).public()

    async def update_provider(
        self, provider_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        current = await self.control.get_provider(provider_id)
        if current:
            payload = await self._with_context_length_metadata(
                payload,
                current=current,
                force=False,
            )
        record = await self.control.update_provider(provider_id, payload)
        if provider_kind(record.config.type) == "rerank":
            for usage in await self.control.provider_usage(record.provider_id):
                if usage.get("usage_kind") != "rerank":
                    continue
                runtime = self.runtimes.get(str(usage.get("library_id") or ""))
                if runtime is not None:
                    await runtime.set_rerank_provider(record)
        return record.public()

    async def _with_context_length_metadata(
        self,
        payload: dict[str, Any],
        *,
        current: ProviderRevision | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        draft = dict(payload)
        base = current.config if current else None
        clear_api_key = bool(draft.get("clear_api_key", False))
        config = config_from_dict(
            draft,
            base=base,
            keep_secret=bool(base) and not clear_api_key,
        )
        if provider_kind(config.type) != "embedding":
            draft["max_context_tokens"] = 0
            draft["max_context_tokens_source"] = ""
            return draft
        changed = (
            force
            or not base
            or config.type != base.type
            or config.api_base != base.api_base
            or config.model != base.model
        )
        if changed:
            detected = await self._detect_context_length(config)
            if detected.get("max_context_tokens"):
                draft["max_context_tokens"] = int(detected["max_context_tokens"])
                draft["max_context_tokens_source"] = str(
                    detected.get("max_context_tokens_source") or ""
                )
                return draft
            source = str(
                draft.get("max_context_tokens_source")
                or config.max_context_tokens_source
                or ""
            )
            if source.startswith("auto:"):
                draft["max_context_tokens"] = 0
                draft["max_context_tokens_source"] = ""
            elif int(draft.get("max_context_tokens") or config.max_context_tokens or 0) > 0:
                draft["max_context_tokens"] = int(
                    draft.get("max_context_tokens") or config.max_context_tokens
                )
                draft["max_context_tokens_source"] = "manual"
            else:
                draft["max_context_tokens"] = 0
                draft["max_context_tokens_source"] = ""
            return draft
        if int(config.max_context_tokens or 0) > 0 and not str(
            config.max_context_tokens_source or ""
        ).startswith("auto:"):
            draft["max_context_tokens_source"] = "manual"
        elif int(config.max_context_tokens or 0) <= 0:
            draft["max_context_tokens_source"] = ""
        return draft

    async def _detect_context_length(self, config: ProviderConfig) -> dict[str, Any]:
        provider = build_provider(
            replace(config, max_context_tokens=0, max_context_tokens_source="")
        )
        try:
            result = await provider.detect_context_length()
            if result.get("max_context_tokens"):
                logger.info(
                    "模型上下文长度检测成功：provider=%s model=%s tokens=%s source=%s",
                    config.id,
                    config.model,
                    result.get("max_context_tokens"),
                    result.get("max_context_tokens_source") or "",
                )
            return result
        except Exception as exc:
            logger.info(
                "模型上下文长度检测跳过：provider=%s model=%s err=%s",
                config.id,
                config.model,
                exc,
            )
            return {"max_context_tokens": 0, "max_context_tokens_source": ""}
        finally:
            await provider.close()

    async def detect_context_length(
        self,
        payload: dict[str, Any],
        provider_id: str | None = None,
    ) -> dict[str, Any]:
        draft = dict(payload)
        draft["max_context_tokens"] = 0
        draft["max_context_tokens_source"] = ""
        current = await self.control.get_provider(provider_id) if provider_id else None
        clear_api_key = bool(draft.get("clear_api_key", False))
        config = config_from_dict(
            draft,
            base=current.config if current else None,
            keep_secret=bool(current) and not clear_api_key,
        )
        if provider_kind(config.type) != "embedding":
            raise ValueError("Rerank Provider does not support max context length detection")
        return await self._detect_context_length(config)

    async def copy_provider(
        self, provider_id: str, new_id: str | None
    ) -> dict[str, Any]:
        return (await self.control.copy_provider(provider_id, new_id)).public()

    async def delete_provider(self, provider_id: str) -> None:
        await self.control.delete_provider(provider_id)

    async def test_provider(
        self, provider_id: str, *, revision: int | None = None
    ) -> dict[str, Any]:
        record = await self.control.get_provider(provider_id, revision)
        if not record:
            raise KeyError(provider_id)
        logger.info("创建临时 Provider 客户端用于测试：provider=%s revision=%s", provider_id, revision or record.revision)
        provider = (
            build_rerank_provider(record.config)
            if provider_kind(record.config.type) == "rerank"
            else build_provider(record.config)
        )
        try:
            return await provider.test_connection()
        finally:
            await provider.close()

    async def test_provider_draft(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        logger.info("创建临时草稿 Provider 客户端用于测试：provider=%s type=%s", payload.get("id"), payload.get("type"))
        config = config_from_dict(payload)
        provider = (
            build_rerank_provider(config)
            if provider_kind(config.type) == "rerank"
            else build_provider(config)
        )
        try:
            return await provider.test_connection()
        finally:
            await provider.close()

    async def detect_dimension(self, payload: dict[str, Any]) -> dict[str, Any]:
        draft = dict(payload)
        draft["dimensions"] = 0
        config = config_from_dict(draft)
        if provider_kind(config.type) != "embedding":
            raise ValueError("Rerank Provider 不支持维度检测")
        provider = build_provider(config)
        try:
            vector = await provider.get_embedding("PersonalityRAG 维度检测")
            return {"dimensions": len(vector)}
        finally:
            await provider.close()
