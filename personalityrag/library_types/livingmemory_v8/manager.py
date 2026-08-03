from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ...config import (
    AppConfig,
    ConversationConfig,
    MaintenanceConfig,
    ProviderConfig,
    RecallConfig,
)
from ...database_types import (
    DatabaseDriverContext,
    LIVINGMEMORY_V8_TYPE,
    DatabaseRef,
    database_type_registry,
)
from ...context_lengths import (
    MANUAL_CONTEXT_FALLBACK_TOKENS,
    MIN_VALID_CONTEXT_TOKENS,
)
from ...control import ControlStore, MemoryStoreRecord, ProviderRevision
from ...identifiers import validate_identifier
from ...io_utils import read_ab_checkpoint, run_blocking
from ...jobs import JobManager
from ...logger import logger
from .migration import (
    sqlite_backup,
    validate_conversations_db_file,
    validate_livingmemory_db_file,
)
from ...performance import measure_phase
from ...providers import (
    build_provider,
    build_rerank_provider,
    config_from_dict,
    provider_kind,
)
from .resumable_tasks import ResumableMemoryStoreTasks
from .service import (
    PersonalityRAGService,
    scan_library_backups,
)
from .storage import Storage
from ...task_control import JobControlSignal, JobInterrupted


DEFAULT_LIBRARY_ID = "Default"
DEFAULT_LIBRARY_NAME = DEFAULT_LIBRARY_ID
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


class LivingMemoryV8Manager:
    def __init__(
        self,
        root: Path | DatabaseDriverContext,
        config: AppConfig | None = None,
    ):
        if isinstance(root, DatabaseDriverContext):
            self.services = root
            self.root = root.root
            self.data_dir = root.data_root
            self.config = root.config
            self.system_path = root.system_path
            self.control = root.control
        else:
            if config is None:
                raise TypeError("config is required without DatabaseDriverContext")
            self.root = root
            self.data_dir = root / "data"
            self.config = config
            self.system_path = self.data_dir / "personalityrag_system.db"
            self.control = ControlStore(self.system_path)
            self.services = DatabaseDriverContext(
                root=root,
                data_root=self.data_dir,
                system_path=self.system_path,
                config=config,
                control=self.control,
            )
        self.runtimes: dict[DatabaseRef, PersonalityRAGService] = {}
        self._runtime_loads: dict[DatabaseRef, asyncio.Task[PersonalityRAGService]] = {}
        self._offline_storages: dict[DatabaseRef, Storage] = {}
        self._runtime_lock = asyncio.Lock()
        self._runtime_condition = asyncio.Condition(self._runtime_lock)
        self._runtime_residency: dict[DatabaseRef, RuntimeResidencyState] = {}
        self._default_library_id = DEFAULT_LIBRARY_ID
        self._runtime_sweeper_task: asyncio.Task[None] | None = None
        self._closing = False
        self.jobs: JobManager | None = None
        self.resumable_tasks = ResumableMemoryStoreTasks(self)
        self._provider_health_cache: dict[
            tuple[str, int, str], tuple[float, dict[str, Any]]
        ] = {}
        self._provider_health_flights: dict[
            tuple[str, int, str], asyncio.Task[dict[str, Any]]
        ] = {}
        self._provider_health_lock = asyncio.Lock()
        self._initializing = False

    @staticmethod
    def _database_ref(library_id: str) -> DatabaseRef:
        return DatabaseRef(LIVINGMEMORY_V8_TYPE, library_id)

    def _type_root(self) -> Path:
        return database_type_registry.type_root(self.data_dir, LIVINGMEMORY_V8_TYPE)

    def _library_dir(self, library_id: str) -> Path:
        return database_type_registry.data_dir(
            self.data_dir, self._database_ref(library_id)
        )

    def _trash_root(self) -> Path:
        return database_type_registry.trash_type_root(
            self.data_dir, LIVINGMEMORY_V8_TYPE
        )

    async def initialize(self) -> None:
        logger.info(
            "初始化 LivingMemoryV8Manager：data_dir=%s system_db=%s",
            self.data_dir,
            self.system_path,
        )
        system_snapshot = (
            await run_blocking(self._system_snapshot_sync)
            if (self.data_dir / "livingmemory.db").exists()
            else {}
        )
        migration: dict[str, Any] | None = None
        default_library_id = DEFAULT_LIBRARY_ID
        self._initializing = True
        try:
            seed_config = (
                self.config.provider
                if self.config.bootstrap_provider_enabled
                else None
            )
            await self.control.initialize(seed_config)
            self.jobs = JobManager(
                self.control,
                runtime_lease_factory=self.runtime_lease,
            )
            self.services.jobs = self.jobs
            self.jobs.set_operation_resolver(
                self.resumable_tasks.resolve,
                database_type=LIVINGMEMORY_V8_TYPE,
            )
            self.jobs.set_database_state_provider(
                self.task_database_state,
                database_type=LIVINGMEMORY_V8_TYPE,
            )
            seed = (
                await self.control.get_provider(seed_config.id)
                if seed_config is not None
                else None
            )
            if seed_config is not None and not seed:
                raise RuntimeError("默认 Provider 初始化失败")
            if seed is not None:
                logger.info(
                    "系统 Provider 已加载：provider=%s revision=%s",
                    seed.provider_id,
                    seed.revision,
                )
            migration = await self._migrate_legacy_layout()
            try:
                default_library = await self.control.default_library()
            except RuntimeError:
                if seed is None:
                    self._default_library_id = DEFAULT_LIBRARY_ID
                    self._start_runtime_sweeper()
                    logger.info(
                        "全新安装未创建默认 Provider 或记忆库，等待用户完成首次配置"
                    )
                    self._initializing = False
                    return
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
            runtime.start_background_index_check()
            if migration:
                self._commit_migration_marker(migration)
            self._start_runtime_sweeper()
            logger.info(
                "LivingMemoryV8Manager 初始化完成：default_memory_store=%s",
                default_library_id,
            )
            self._initializing = False
        except Exception:
            self._initializing = False
            logger.exception("LivingMemoryV8Manager 初始化失败，准备回滚可能的迁移")
            if migration:
                default_ref = DatabaseRef(LIVINGMEMORY_V8_TYPE, default_library_id)
                runtime = self.runtimes.pop(default_ref, None)
                self._runtime_residency.pop(default_ref, None)
                if runtime is not None:
                    await runtime.close()
            if self.jobs is not None:
                await self.jobs.close()
                self.jobs = None
            for runtime in list(self.runtimes.values()):
                await runtime.close()
            self.runtimes.clear()
            self._runtime_residency.clear()
            await self.control.close()
            if migration:
                await self._rollback_legacy_layout(migration)
                for suffix in ("", "-wal", "-shm"):
                    path = Path(str(self.system_path) + suffix)
                    path.unlink(missing_ok=True)
                    if suffix in system_snapshot:
                        path.write_bytes(system_snapshot[suffix])
            raise

    def _system_snapshot_sync(self) -> dict[str, bytes]:
        return {
            suffix: path.read_bytes()
            for suffix in ("", "-wal", "-shm")
            if (path := Path(str(self.system_path) + suffix)).exists()
        }

    async def close(self) -> None:
        self._closing = True
        if self._runtime_sweeper_task is not None:
            self._runtime_sweeper_task.cancel()
            await asyncio.gather(self._runtime_sweeper_task, return_exceptions=True)
            self._runtime_sweeper_task = None
        if self.jobs is not None:
            await self.jobs.close()
        load_tasks = list(self._runtime_loads.values())
        for task in load_tasks:
            task.cancel()
        if load_tasks:
            await asyncio.gather(*load_tasks, return_exceptions=True)
        self._runtime_loads.clear()
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
        self._offline_storages.clear()
        await self.control.close()

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
        target_root = self._library_dir(DEFAULT_LIBRARY_ID)
        marker = self.data_dir / ".multilibrary_migrated_v1.json"
        if not source_db.exists():
            return None

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_root = self.data_dir / "pre_multilibrary_backups" / timestamp
        logger.warning(
            "检测到旧单库布局，开始迁移：source=%s target=%s backup=%s",
            self.data_dir,
            target_root,
            backup_root,
        )
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
            if (
                self._sha256(target_root / "livingmemory.db")
                != copied["livingmemory.db"]
            ):
                raise RuntimeError("迁移后 livingmemory.db 哈希不一致")
            if (
                self._sha256(target_root / "conversations.db")
                != copied["conversations.db"]
            ):
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

    async def _validate_runtime(self, runtime: PersonalityRAGService) -> None:
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
        if not (indexes or {}).get("generation"):
            # A database without a valid generation is an online, writable
            # index_not_ready state.  The startup maintenance task builds its
            # first shadow generation after this local database validation.
            return
        document_ids = {int(value) for value in await runtime.storage.document_ids()}
        granularity_resolver = getattr(
            runtime.indexes, "graph_vector_granularity", None
        )
        if callable(granularity_resolver):
            graph_granularity = granularity_resolver()
        else:
            manifest = indexes.get("manifest") or {}
            graph_granularity = (
                "memory"
                if manifest.get("graph_vector_granularity") == "memory"
                else "entry"
            )
        graph_ids = {
            int(value)
            for value in (
                await runtime.storage.graph_memory_ids()
                if graph_granularity == "memory"
                else await runtime.storage.graph_entry_ids()
            )
        }
        indexed_document_ids, indexed_graph_ids = runtime.indexes.indexed_ids()
        index_drift: list[str] = []
        if indexed_document_ids != document_ids:
            index_drift.append("document_id_set_changed")
        if indexed_graph_ids != graph_ids:
            index_drift.append("graph_id_set_changed")
        active_memories = int(
            stats.get("active_memories", stats.get("total_memories", 0)) or 0
        )
        if int(indexes["document_vectors"]) != active_memories:
            index_drift.append("document_vector_count_changed")
        expected_graph_vectors = (
            len(graph_ids)
            if graph_granularity == "memory"
            else int(stats["graph_entries"])
        )
        if int(indexes["graph_vectors"]) != expected_graph_vectors:
            index_drift.append("graph_vector_count_changed")
        if index_drift:
            # Index generations are derived data.  Keep the database runtime
            # online and let the startup shadow-maintenance pass repair drift
            # without turning an otherwise healthy library into a boot failure.
            logger.warning(
                "启动校验发现索引漂移，将在后台执行影子重建：memory_store_id=%s reasons=%s",
                getattr(runtime, "memory_store_id", ""),
                ",".join(sorted(set(index_drift))),
            )

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
            "旧单库布局迁移标记已写入：memory_store_id=%s backup=%s",
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

    async def _reuse_upload_validation(
        self,
        path: Path,
        report: dict[str, Any] | None,
        validator,
    ) -> dict[str, Any]:
        expected = dict(report or {})
        expected_sha256 = str(expected.get("sha256") or "")
        expected_size = expected.get("size")
        if expected_sha256 and expected_size is not None:
            current_size = path.stat().st_size
            with measure_phase(
                "livingmemory_import",
                "fingerprint_verify",
                bytes_count=current_size,
            ):
                current_sha256 = await run_blocking(self._sha256, path)
            if current_size != int(expected_size) or current_sha256 != expected_sha256:
                raise ValueError(f"uploaded database fingerprint changed: {path.name}")
            return expected
        return await run_blocking(validator, path)

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
            await self.control.bind_library(runtime.library_id, provider, manifest)

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

    async def _runtime_has_active_jobs(self, ref: DatabaseRef) -> bool:
        return await self.control.has_running_jobs(ref.id)

    async def _close_runtime_locked(
        self,
        ref: DatabaseRef,
        *,
        reason: str,
        suppress_errors: bool = True,
    ) -> bool:
        runtime = self.runtimes.pop(ref, None)
        self._runtime_residency.pop(ref, None)
        if runtime is None:
            return False
        try:
            await runtime.close()
        except Exception:
            logger.exception(
                "记忆库 runtime 释放失败：memory_store_id=%s reason=%s",
                ref.key,
                reason,
            )
            if not suppress_errors:
                raise
        else:
            logger.info(
                "记忆库 runtime 已释放：memory_store_id=%s reason=%s",
                ref.key,
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
                ref for ref in self.runtimes if ref != self._default_database_ref()
            ]
            if len(non_default) <= target:
                break
            candidates = sorted(
                non_default,
                key=lambda item: (
                    self._runtime_residency.get(
                        item,
                        RuntimeResidencyState(0, 0.0),
                    ).last_used_at
                ),
            )
            selected: DatabaseRef | None = None
            for ref in candidates:
                state = self._runtime_residency.get(ref)
                if state is not None and state.lease_count > 0:
                    continue
                if await self._runtime_has_active_jobs(ref):
                    continue
                selected = ref
                break
            if selected is None:
                break
            await self._close_runtime_locked(selected, reason=reason)
            evicted.append(selected.id)
        return evicted

    async def _build_runtime(self, ref: DatabaseRef) -> PersonalityRAGService:
        library_id = ref.id
        library = await self.control.get_library(library_id)
        if not library or library.database_type != ref.database_type:
            raise KeyError(ref.key)
        database_type_registry.require(library.database_type)
        if library.database_type != LIVINGMEMORY_V8_TYPE:
            raise ValueError(f"数据库类型尚未实现 runtime: {library.database_type}")
        logger.info("懒加载记忆库 runtime：memory_store_id=%s", library_id)
        provider = await self.control.get_provider(
            library.provider_id, library.provider_revision
        )
        if not provider:
            raise RuntimeError(f"记忆库 {library_id} 绑定的 Provider revision 不存在")
        rerank_provider = None
        if library.rerank_provider_id:
            rerank_provider = await self.control.get_provider(
                library.rerank_provider_id
            )
            if not rerank_provider:
                logger.warning(
                    "记忆库绑定的 Rerank Provider 不存在，将跳过重排：memory_store_id=%s provider=%s",
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
            database_type_registry.data_dir(
                self.data_dir,
                DatabaseRef(LIVINGMEMORY_V8_TYPE, library_id),
            ),
            memory_store_id=library_id,
            default_persona_id=library.default_persona_id,
            provider_revision=provider,
            rerank_provider_revision=rerank_provider,
            system_path=self.system_path,
            system_pool=self.control.pool,
        )
        try:
            await runtime.initialize(start_index_check=not self._initializing)
            return runtime
        except BaseException:
            await runtime.close()
            raise

    async def _runtime_or_load(
        self,
        ref: DatabaseRef,
        *,
        acquire: bool,
        touch: bool,
    ) -> PersonalityRAGService:
        async with self._runtime_lock:
            current = self.runtimes.get(ref)
            if current is not None:
                state = self._runtime_residency[ref]
                if acquire:
                    state.lease_count += 1
                if touch:
                    state.last_used_at = time.monotonic()
                return current
            task = self._runtime_loads.get(ref)
            if task is None:
                task = asyncio.create_task(
                    self._build_runtime(ref),
                    name=f"personalityrag-runtime-load-{ref.key}",
                )
                self._runtime_loads[ref] = task
        try:
            built = await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except BaseException:
            async with self._runtime_lock:
                if self._runtime_loads.get(ref) is task:
                    self._runtime_loads.pop(ref, None)
            raise

        retry = False
        async with self._runtime_lock:
            current = self.runtimes.get(ref)
            if current is None and self._runtime_loads.get(ref) is not task:
                retry = True
            elif current is None:
                if self._closing:
                    self._runtime_loads.pop(ref, None)
                    close_built = True
                else:
                    close_built = False
                    if ref != self._default_database_ref():
                        limit = max(
                            1,
                            int(self.config.runtime_residency.max_non_default_runtimes),
                        )
                        await self._evict_lru_until_locked(
                            limit - 1,
                            reason="capacity",
                        )
                    current = built
                    self.runtimes[ref] = current
                    self._runtime_residency[ref] = RuntimeResidencyState(
                        lease_count=0,
                        last_used_at=time.monotonic(),
                    )
                    self._runtime_loads.pop(ref, None)
                    logger.info("记忆库 runtime 已加载：database=%s", ref.key)
            else:
                close_built = current is not built
                if self._runtime_loads.get(ref) is task:
                    self._runtime_loads.pop(ref, None)
            if not retry and current is not None:
                state = self._runtime_residency[ref]
                if acquire:
                    state.lease_count += 1
                if touch:
                    state.last_used_at = time.monotonic()
        if retry:
            return await self._runtime_or_load(
                ref,
                acquire=acquire,
                touch=touch,
            )
        if close_built:
            await built.close()
            if current is None:
                raise RuntimeError("LivingMemoryV8Manager is closing")
        assert current is not None
        return current

    @staticmethod
    def _livingmemory_ref(
        database: str | DatabaseRef,
        database_type: str | None = None,
    ) -> DatabaseRef:
        ref = (
            database
            if isinstance(database, DatabaseRef)
            else DatabaseRef(database_type or LIVINGMEMORY_V8_TYPE, database)
        )
        database_type_registry.require(ref.database_type)
        if ref.database_type != LIVINGMEMORY_V8_TYPE:
            raise ValueError(f"数据库类型尚未实现 runtime: {ref.database_type}")
        return ref

    @staticmethod
    def _livingmemory_id(
        database: str | DatabaseRef,
        database_type: str | None = None,
    ) -> str:
        return LivingMemoryV8Manager._livingmemory_ref(database, database_type).id

    def _default_database_ref(self) -> DatabaseRef:
        return DatabaseRef(LIVINGMEMORY_V8_TYPE, self._default_library_id)

    async def get_runtime(
        self,
        database: str | DatabaseRef,
        *,
        database_type: str | None = None,
    ) -> PersonalityRAGService:
        ref = self._livingmemory_ref(database, database_type)
        return await self._runtime_or_load(
            ref,
            acquire=False,
            touch=True,
        )

    async def acquire_runtime(
        self,
        database: str | DatabaseRef,
        *,
        database_type: str | None = None,
        touch: bool = True,
    ) -> PersonalityRAGService:
        ref = self._livingmemory_ref(database, database_type)
        return await self._runtime_or_load(
            ref,
            acquire=True,
            touch=touch,
        )

    async def release_runtime(
        self,
        database: str | DatabaseRef,
        *,
        database_type: str | None = None,
        touch: bool = True,
    ) -> None:
        ref = self._livingmemory_ref(database, database_type)
        should_converge = False
        async with self._runtime_condition:
            state = self._runtime_residency.get(ref)
            if state is None:
                return
            state.lease_count = max(0, state.lease_count - 1)
            if touch:
                state.last_used_at = time.monotonic()
            non_default_count = sum(
                database != self._default_database_ref() for database in self.runtimes
            )
            limit = max(
                1,
                int(self.config.runtime_residency.max_non_default_runtimes),
            )
            should_converge = state.lease_count == 0 and non_default_count > limit
            self._runtime_condition.notify_all()
        if should_converge:
            await self.sweep_runtimes(expire_idle=False)

    @asynccontextmanager
    async def runtime_lease(self, database: str | DatabaseRef, *, touch: bool = True):
        runtime = await self.acquire_runtime(database, touch=touch)
        try:
            yield runtime
        finally:
            await self.release_runtime(database, touch=touch)

    async def unload_runtime(self, database: str | DatabaseRef, *, reason: str) -> bool:
        ref = self._livingmemory_ref(database)
        async with self._runtime_condition:
            while (
                state := self._runtime_residency.get(ref)
            ) is not None and state.lease_count > 0:
                await self._runtime_condition.wait()
            return await self._close_runtime_locked(
                ref,
                reason=reason,
                suppress_errors=False,
            )

    def debug_revision_generation_manifests(
        self, database_id: str
    ) -> list[dict[str, Any]]:
        index_root = self._library_dir(database_id) / "indexes"
        current = ""
        current_file = index_root / "CURRENT"
        if current_file.is_file():
            try:
                current = current_file.read_text(encoding="utf-8").strip()
            except OSError:
                current = ""
        items: list[dict[str, Any]] = []
        if not index_root.exists():
            return items
        for manifest_path in sorted(index_root.glob("*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                items.append(
                    {
                        "generation": manifest_path.parent.name,
                        "is_active": manifest_path.parent.name == current,
                        "invalid_manifest": True,
                    }
                )
                continue
            items.append(
                {
                    "generation": str(
                        manifest.get("generation") or manifest_path.parent.name
                    ),
                    "is_active": manifest_path.parent.name == current,
                    "provider_id": str(manifest.get("provider_id") or ""),
                    "provider_revision": int(
                        manifest.get("provider_revision") or 0
                    ),
                    "provider_fingerprint": str(
                        manifest.get("provider_config_sha256")
                        or manifest.get("provider_fingerprint")
                        or ""
                    ),
                    "created_at": float(manifest.get("created_at") or 0),
                    "document_count": int(manifest.get("document_count") or 0),
                    "graph_entry_count": int(
                        manifest.get("graph_entry_count") or 0
                    ),
                }
            )
        return items

    async def debug_rebind_revision(
        self,
        database_id: str,
        *,
        provider_id: str,
        revision: int,
        fingerprint: str,
    ) -> None:
        record = await self.control.get_library(database_id)
        if record is None or record.database_type != LIVINGMEMORY_V8_TYPE:
            raise KeyError(database_id)
        if record.provider_id != provider_id:
            raise ValueError("revision repair cannot change the bound Provider ID")
        await self.unload_runtime(
            DatabaseRef(LIVINGMEMORY_V8_TYPE, database_id),
            reason="revision_debug_binding_repair",
        )
        index_root = self._library_dir(database_id) / "indexes"
        originals: list[tuple[Path, str]] = []
        try:
            if index_root.exists():
                for manifest_path in sorted(index_root.glob("*/manifest.json")):
                    text = manifest_path.read_text(encoding="utf-8")
                    manifest = json.loads(text)
                    if str(manifest.get("provider_id") or "") != provider_id:
                        continue
                    originals.append((manifest_path, text))
                    rewritten = self.control._rewrite_revision_manifest(
                        manifest,
                        provider_id=provider_id,
                        revision=revision,
                        fingerprint=fingerprint,
                    )
                    temporary = manifest_path.with_suffix(
                        manifest_path.suffix + ".revision-debug.tmp"
                    )
                    temporary.write_text(
                        json.dumps(rewritten, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    os.replace(temporary, manifest_path)
            await self.control.debug_bind_livingmemory_revision(
                database_id,
                provider_id=provider_id,
                revision=revision,
                fingerprint=fingerprint,
            )
        except BaseException:
            for manifest_path, text in originals:
                try:
                    temporary = manifest_path.with_suffix(
                        manifest_path.suffix + ".revision-debug.rollback.tmp"
                    )
                    temporary.write_text(text, encoding="utf-8")
                    os.replace(temporary, manifest_path)
                except OSError:
                    logger.exception(
                        "revision debug manifest rollback failed: "
                        "memory_store_id=%s generation=%s",
                        database_id,
                        manifest_path.parent.name,
                    )
            raise

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
                        ref
                        for ref in self.runtimes
                        if ref != self._default_database_ref()
                    ),
                    key=lambda item: (
                        self._runtime_residency.get(
                            item,
                            RuntimeResidencyState(0, 0.0),
                        ).last_used_at
                    ),
                )
                for ref in candidates:
                    state = self._runtime_residency.get(ref)
                    if state is None or state.lease_count > 0:
                        continue
                    if now - state.last_used_at < idle_seconds:
                        continue
                    if await self._runtime_has_active_jobs(ref):
                        continue
                    if await self._close_runtime_locked(
                        ref,
                        reason="idle",
                    ):
                        evicted.append(ref.id)
            evicted.extend(
                await self._evict_lru_until_locked(
                    max(
                        1,
                        int(self.config.runtime_residency.max_non_default_runtimes),
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
            "default_database": self._default_database_ref().public(),
            "loaded_library_ids": [ref.id for ref in self.runtimes],
            "loaded_databases": [ref.public() for ref in self.runtimes],
            "runtimes": {
                ref.key: {
                    "database_type": ref.database_type,
                    "database_id": ref.id,
                    "lease_count": state.lease_count,
                    "last_used_at": state.last_used_at,
                }
                for ref, state in self._runtime_residency.items()
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

    async def list_libraries(self, *, stats_mode: str = "full") -> list[dict[str, Any]]:
        if stats_mode not in {"full", "summary"}:
            raise ValueError("invalid library stats mode")
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
        provider_map, adapter_map, busy_map = await asyncio.gather(
            self.control.get_providers_bulk(provider_bindings),
            self.control.active_adapter_connections_map(library_ids),
            (
                self.jobs.active_long_jobs_map(library_ids)
                if self.jobs is not None
                else self.control.active_long_jobs_map(library_ids)
            ),
        )
        stats_limit = asyncio.Semaphore(4)

        async def serialize(record: MemoryStoreRecord) -> dict[str, Any]:
            driver = database_type_registry.require(record.database_type)
            descriptor = driver.descriptor
            provider = provider_map.get((record.provider_id, record.provider_revision))
            rerank_provider = (
                provider_map.get((record.rerank_provider_id, None))
                if record.rerank_provider_id
                else None
            )
            runtime_ref = DatabaseRef(record.database_type, record.id)
            runtime = self.runtimes.get(runtime_ref)
            if runtime is not None:
                async with stats_limit:
                    stats = (
                        await runtime.storage.summary_statistics()
                        if stats_mode == "summary"
                        else await runtime.storage.statistics()
                    )
                indexes = self._normalize_indexes_for_response(
                    stats, runtime.indexes.status()
                )
            else:
                library_dir = database_type_registry.data_dir(
                    self.data_dir,
                    DatabaseRef(LIVINGMEMORY_V8_TYPE, record.id),
                )
                storage = self._offline_storages.setdefault(
                    runtime_ref,
                    Storage(library_dir, system_path=self.system_path),
                )
                async with stats_limit:
                    stats = (
                        await storage.summary_statistics()
                        if stats_mode == "summary"
                        else await storage.statistics()
                    )
                indexes = self._normalize_indexes_for_response(
                    stats, self._offline_index_status(library_dir, record)
                )
            return {
                **record.public(),
                "database_category": descriptor.category,
                "capabilities": list(descriptor.capabilities),
                "type_metadata": descriptor.public(),
                "stats": stats,
                "indexes": indexes,
                "maintenance": (
                    runtime.maintenance_status()
                    if runtime is not None
                    else {
                        "status": "idle",
                        "stage": "offline",
                        "progress": 0.0,
                        "index_available": bool(indexes.get("generation")),
                    }
                ),
                "provider": provider.public() if provider else None,
                "rerank_provider": (
                    rerank_provider.public() if rerank_provider else None
                ),
                "adapter_connections": adapter_map.get(record.id, []),
                "adapter_busy": self._adapter_busy_summary(busy_map.get(record.id)),
            }

        return list(await asyncio.gather(*(serialize(record) for record in records)))

    @staticmethod
    def _offline_index_status(
        library_dir: Path, record: MemoryStoreRecord
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
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    manifest = None
        graph_vector_count = (manifest or {}).get("graph_vector_count")
        return {
            "generation": generation,
            "manifest": manifest,
            "document_vectors": int((manifest or {}).get("document_count", 0)),
            "graph_vectors": int(
                graph_vector_count
                if graph_vector_count is not None
                else (manifest or {}).get("graph_entry_count", 0)
            ),
            "provider_id": record.provider_id,
            "provider_revision": record.provider_revision,
        }

    async def library_detail(self, library_id: str) -> dict[str, Any]:
        record = await self.control.get_library(library_id)
        if not record:
            raise KeyError(library_id)
        driver = database_type_registry.require(record.database_type)
        library_dir = database_type_registry.data_dir(
            self.data_dir,
            DatabaseRef(LIVINGMEMORY_V8_TYPE, record.id),
        )
        runtime_ref = DatabaseRef(record.database_type, record.id)
        runtime = self.runtimes.get(runtime_ref)
        storage = (
            runtime.storage
            if runtime is not None
            else self._offline_storages.setdefault(
                runtime_ref,
                Storage(library_dir, system_path=self.system_path),
            )
        )
        (
            provider,
            rerank_provider,
            stats,
            adapter_connections,
            busy_job,
        ) = await asyncio.gather(
            self.control.get_provider(
                record.provider_id,
                record.provider_revision,
            ),
            (
                self.control.get_provider(record.rerank_provider_id)
                if record.rerank_provider_id
                else asyncio.sleep(0, result=None)
            ),
            storage.statistics(),
            self.control.active_adapter_connections(record.id),
            (
                self.jobs.active_long_job(record.id)
                if self.jobs is not None
                else self.control.active_long_job(record.id)
            ),
        )
        index_status = (
            runtime.indexes.status()
            if runtime is not None
            else self._offline_index_status(library_dir, record)
        )
        return {
            **record.public(),
            "database_category": driver.descriptor.category,
            "capabilities": list(driver.descriptor.capabilities),
            "type_metadata": driver.descriptor.public(),
            "stats": stats,
            "indexes": self._normalize_indexes_for_response(stats, index_status),
            "maintenance": (
                runtime.maintenance_status()
                if runtime is not None
                else {
                    "status": "idle",
                    "stage": "offline",
                    "progress": 0.0,
                    "index_available": bool(index_status.get("generation")),
                }
            ),
            "provider": provider.public() if provider else None,
            "rerank_provider": rerank_provider.public() if rerank_provider else None,
            "adapter_connections": adapter_connections,
            "adapter_busy": self._adapter_busy_summary(busy_job),
        }

    async def task_database_state(
        self,
        memory_store_id: str,
        _task_kind: str,
    ) -> dict[str, Any]:
        """Capture the stable library/index facts used by finished-task details."""

        record = await self.control.get_library(memory_store_id)
        if record is None:
            return {
                "database": {
                    "exists": False,
                    "database_type": LIVINGMEMORY_V8_TYPE,
                    "database_id": memory_store_id,
                }
            }
        library_dir = self._library_dir(memory_store_id)
        runtime_ref = DatabaseRef(record.database_type, record.id)
        runtime = self.runtimes.get(runtime_ref)
        storage = (
            runtime.storage
            if runtime is not None
            else self._offline_storages.setdefault(
                runtime_ref,
                Storage(library_dir, system_path=self.system_path),
            )
        )
        stats = await storage.statistics()
        indexes = self._normalize_indexes_for_response(
            stats,
            runtime.indexes.status()
            if runtime is not None
            else self._offline_index_status(library_dir, record),
        )
        manifest = dict(indexes.get("manifest") or {})
        manifest_summary = {
            key: manifest.get(key)
            for key in (
                "generation",
                "provider_id",
                "provider_revision",
                "dimensions",
                "document_count",
                "graph_entry_count",
                "chunked_document_count",
                "chunked_graph_entry_count",
                "created_at",
            )
            if key in manifest
        }
        return {
            "database": {
                "exists": True,
                "database_type": record.database_type,
                "database_id": record.id,
                "name": record.name,
                "is_default": record.is_default,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
            },
            "provider_binding": {
                "embedding_provider_id": record.provider_id,
                "embedding_provider_revision": record.provider_revision,
                "rerank_provider_id": record.rerank_provider_id or None,
            },
            "statistics": stats,
            "indexes": {
                "generation": indexes.get("generation"),
                "document_vectors": int(indexes.get("document_vectors") or 0),
                "graph_vectors": int(indexes.get("graph_vectors") or 0),
                "provider_id": indexes.get("provider_id"),
                "provider_revision": indexes.get("provider_revision"),
                "manifest": manifest_summary or None,
            },
            "maintenance": (
                runtime.maintenance_status()
                if runtime is not None
                else {
                    "status": "idle",
                    "stage": "offline",
                    "progress": 0.0,
                    "index_available": bool(indexes.get("generation")),
                }
            ),
        }

    async def list_library_backups(self, library_id: str) -> list[dict[str, Any]]:
        library_dir = self._library_dir(library_id)
        return await run_blocking(scan_library_backups, library_dir)

    async def create_library(self, payload: dict[str, Any]) -> dict[str, Any]:
        database_type = str(payload.get("database_type") or LIVINGMEMORY_V8_TYPE)
        try:
            database_type_registry.require(database_type)
        except KeyError as exc:
            raise ValueError(f"不支持的数据库类型: {database_type}") from exc
        if database_type != LIVINGMEMORY_V8_TYPE:
            raise ValueError(f"数据库类型尚未实现创建流程: {database_type}")
        payload["database_type"] = database_type
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
            try:
                await self.control.default_library()
            except RuntimeError:
                record = await self.control.set_default_library(record.id)
                self._default_library_id = record.id
            await self.get_runtime(record.id)
            logger.info(
                "新记忆库创建完成，索引保持待构建：memory_store_id=%s provider=%s revision=%s",
                record.id,
                provider.provider_id,
                provider.revision,
            )
        except Exception:
            logger.exception("新记忆库初始化失败，正在清理：memory_store_id=%s", record.id)
            await self.unload_runtime(record.id, reason="create_failed")
            await self.control.mark_library_deleted(record.id)
            shutil.rmtree(
                database_type_registry.data_dir(
                    self.data_dir,
                    DatabaseRef(LIVINGMEMORY_V8_TYPE, record.id),
                ),
                ignore_errors=True,
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
        rerank_changed = "rerank_provider_id" in payload and str(
            payload.get("rerank_provider_id") or ""
        ) != str(record.rerank_provider_id or "")
        settings_changed = (
            "conversation_settings" in payload
            or "recall_settings" in payload
            or "maintenance_settings" in payload
        )
        reload_runtime = rename_requested or settings_changed
        if rename_requested:
            if await self.control.active_adapter_connections(record.id):
                raise ValueError("记忆库已被适配器连接，不能修改 ID")
            if await self.control.has_running_jobs(record.id):
                raise ValueError("记忆库存在进行中的任务，暂时不能修改 ID")
            library_root = self._type_root()
            if (
                await self.control.get_library(next_library_id)
                or (library_root / next_library_id).exists()
                or any(library_root.glob(f".{next_library_id}.copying-*"))
            ):
                raise ValueError(f"记忆库 ID 已存在：{next_library_id}")
        if reload_runtime:
            await self.unload_runtime(record.id, reason="library_settings_changed")
        if rename_requested:
            source_dir = self._library_dir(record.id)
            target_dir = self._library_dir(next_library_id)
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
            runtime_ref = DatabaseRef(record.database_type, record.id)
            if "default_persona_id" in payload and runtime_ref in self.runtimes:
                self.runtimes[runtime_ref].default_persona_id = str(
                    payload.get("default_persona_id") or ""
                ).strip()
            if rerank_changed and not reload_runtime:
                if runtime_ref in self.runtimes:
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
        source_dir = self._library_dir(source.id)
        if not source_dir.exists():
            raise ValueError(f"源记忆库目录不存在：{source_dir}")
        if progress:
            await progress(0.03, "正在准备副本 ID 与目标目录")
        target_id, target_name = await self._next_copy_identity(source)
        target_dir = self._library_dir(target_id)

        provider = await self.control.get_provider(
            source.provider_id, source.provider_revision
        )
        if not provider:
            raise ValueError("源记忆库绑定的 Provider revision 不存在")
        if progress:
            await progress(0.12, f"副本目标已确定：{target_id}")

        tmp_dir = (
            self._type_root() / f".{target_id}.copying-{time.strftime('%Y%m%d-%H%M%S')}"
        )
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
                "stats": await Storage(
                    target_dir, system_path=self.system_path
                ).statistics(),
                "indexes": self._offline_index_status(target_dir, record),
                "provider": provider.public(),
            }
        except Exception:
            logger.exception(
                "复制记忆库失败，正在清理副本：source=%s target=%s",
                source.id,
                target_id,
            )
            await run_blocking(shutil.rmtree, target_dir, ignore_errors=True)
            await run_blocking(shutil.rmtree, tmp_dir, ignore_errors=True)
            if record_created:
                try:
                    await self.control.mark_library_deleted(target_id)
                except Exception:
                    pass
            raise

    async def _next_copy_identity(
        self, source: MemoryStoreRecord
    ) -> tuple[str, str]:
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
        library_root = self._type_root()
        if (library_root / library_id).exists():
            return True
        if any(library_root.glob(f".{library_id}.copying-*")):
            return True
        return False

    @staticmethod
    def _copy_library_directory(source_dir: Path, target_dir: Path) -> None:
        database_bytes = sum(
            path.stat().st_size
            for path in (
                source_dir / "livingmemory.db",
                source_dir / "conversations.db",
            )
            if path.is_file()
        )
        with measure_phase(
            "library_copy",
            "positive_manifest",
            bytes_count=database_bytes,
        ):
            LivingMemoryV8Manager._copy_library_manifest(source_dir, target_dir)

    @staticmethod
    def _copy_library_manifest(source_dir: Path, target_dir: Path) -> None:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=False)
        for name in ("livingmemory.db", "conversations.db"):
            source_db = source_dir / name
            target_db = target_dir / name
            if not source_db.exists():
                continue
            sqlite_backup(source_db, target_db)

        for name in ("decay_state.json", ".plugin_version"):
            source = source_dir / name
            if source.is_file():
                shutil.copy2(source, target_dir / name)

        stopwords = source_dir / "stopwords"
        if stopwords.is_dir():
            shutil.copytree(stopwords, target_dir / "stopwords")

        source_indexes = source_dir / "indexes"
        target_indexes = target_dir / "indexes"
        target_indexes.mkdir(parents=True, exist_ok=True)
        current = source_indexes / "CURRENT"
        if not current.is_file():
            return
        generation = current.read_text(encoding="utf-8").strip()
        if not generation or Path(generation).name != generation:
            return
        generation_dir = source_indexes / generation
        if not generation_dir.is_dir():
            return
        shutil.copytree(generation_dir, target_indexes / generation)
        shutil.copy2(current, target_indexes / "CURRENT")

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
        task_context=None,
        checkpoint_dir: Path | None = None,
        source_validation_report: dict[str, Any] | None = None,
        conversations_validation_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        controlled = task_context is not None and checkpoint_dir is not None
        record = await self.control.get_library(library_id)
        if not record:
            raise KeyError(library_id)
        source_report = await self._reuse_upload_validation(
            source_db,
            source_validation_report,
            validate_livingmemory_db_file,
        )
        conversations_report = (
            await self._reuse_upload_validation(
                conversations_db,
                conversations_validation_report,
                validate_conversations_db_file,
            )
            if conversations_db is not None
            else None
        )
        if conversations_report is not None and not conversations_report.get(
            "normalized_hashes"
        ):
            conversations_report = await run_blocking(
                validate_conversations_db_file,
                conversations_db,
            )
        if progress:
            validated_message = "已验证 LivingMemory 核心数据库"
            if conversations_report is not None:
                validated_message += "与消息记录数据库"
            await progress(0.02, validated_message)
        runtime = await self.get_runtime(library_id)
        is_resume = bool(
            controlled
            and checkpoint_dir is not None
            and (checkpoint_dir / "import-state.json").exists()
        )
        if not is_resume and not await self.library_is_empty(library_id):
            raise ValueError("只有全新空记忆库可以导入 livingmemory.db")

        run_id = (
            str(task_context.job_id)
            if controlled
            else time.strftime("%Y%m%d-%H%M%S-") + os.urandom(4).hex()
        )
        library_dir = self._library_dir(library_id)
        import_dir = checkpoint_dir if controlled else library_dir / "imports" / run_id
        archive_dir = import_dir / "source_archive"
        report_dir = library_dir / "reports"
        archive_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        target_db = library_dir / "livingmemory.db"
        target_conversations_db = library_dir / "conversations.db"
        rollback_db = import_dir / "pre_import_empty_livingmemory.db"
        rollback_conversations_db = import_dir / "pre_import_empty_conversations.db"
        report_path = report_dir / f"livingmemory-db-import-{run_id}.json"
        logger.warning(
            "LivingMemory 单文件导入开始：memory_store_id=%s source=%s conversations=%s run_id=%s",
            library_id,
            source_db,
            conversations_db or "",
            run_id,
        )
        try:
            phase_path = import_dir / "import-state.json"
            phase = (
                json.loads(phase_path.read_text(encoding="utf-8"))
                if controlled and phase_path.exists()
                else {"phase": "created"}
            )
            if progress:
                await progress(0.04, "正在归档上传的 livingmemory.db")
            archived_db = archive_dir / "livingmemory.db"
            archived_conversations = archive_dir / "conversations.db"
            if phase.get("phase") == "created":
                await run_blocking(sqlite_backup, source_db, archived_db)
                if conversations_db is not None:
                    await run_blocking(
                        sqlite_backup,
                        conversations_db,
                        archived_conversations,
                    )
                if not controlled:
                    if target_db.exists():
                        await run_blocking(sqlite_backup, target_db, rollback_db)
                    if target_conversations_db.exists():
                        await run_blocking(
                            sqlite_backup,
                            target_conversations_db,
                            rollback_conversations_db,
                        )
                phase = {
                    "phase": "source_archived",
                    "source_sha256": self._sha256(archived_db),
                    "conversations_sha256": (
                        self._sha256(archived_conversations)
                        if archived_conversations.exists()
                        else ""
                    ),
                    "saved_at": time.time(),
                }
                phase_path.write_text(
                    json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                if controlled:
                    await task_context.checkpoint(
                        {"phase": "source_archived", "saved_at": time.time()},
                        progress=0.05,
                        message="导入源与任务前状态已安全归档",
                    )
            if progress:
                await progress(0.06, "正在替换目标空库核心数据库")
            if phase.get("phase") != "created":
                if not archived_db.exists() or self._sha256(archived_db) != phase.get(
                    "source_sha256"
                ):
                    raise JobInterrupted(
                        "checkpoint_corrupt",
                        "导入源归档缺失或 SHA-256 校验失败，需停止任务以回滚",
                    )
                expected_conversations_hash = str(
                    phase.get("conversations_sha256") or ""
                )
                if expected_conversations_hash and (
                    not archived_conversations.exists()
                    or self._sha256(archived_conversations)
                    != expected_conversations_hash
                ):
                    raise JobInterrupted(
                        "checkpoint_corrupt",
                        "消息记录归档缺失或 SHA-256 校验失败，需停止任务以回滚",
                    )
            if phase.get("phase") == "source_archived":
                await self.unload_runtime(library_id, reason="livingmemory_import")
                for db_path in (target_db, target_conversations_db):
                    for suffix in ("-wal", "-shm"):
                        Path(str(db_path) + suffix).unlink(missing_ok=True)
                tmp_db = target_db.with_suffix(".db.importing")
                tmp_db.unlink(missing_ok=True)
                await run_blocking(sqlite_backup, archived_db, tmp_db)
                tmp_db.replace(target_db)
                if archived_conversations.exists():
                    tmp_conversations_db = target_conversations_db.with_suffix(
                        ".db.importing"
                    )
                    tmp_conversations_db.unlink(missing_ok=True)
                    await run_blocking(
                        sqlite_backup,
                        archived_conversations,
                        tmp_conversations_db,
                    )
                    tmp_conversations_db.replace(target_conversations_db)
                phase = {
                    **phase,
                    "phase": "database_installed",
                    "saved_at": time.time(),
                }
                phase_path.write_text(
                    json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                if controlled:
                    await task_context.checkpoint(
                        {"phase": "database_installed", "saved_at": time.time()},
                        progress=0.08,
                        message="导入数据库已原子安装",
                    )
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

            rebuild = await self.rebuild_library(
                library_id,
                None,
                rebuild_progress,
                task_context=task_context,
                checkpoint_dir=(import_dir / "index") if controlled else None,
            )
            stats = await runtime.storage.statistics()
            target_conversations_report = (
                await run_blocking(
                    validate_conversations_db_file,
                    target_conversations_db,
                )
                if conversations_report is not None
                else None
            )
            if conversations_report is not None and target_conversations_report is not None:
                source_hashes = dict(
                    conversations_report.get("normalized_hashes") or {}
                )
                target_hashes = dict(
                    target_conversations_report.get("normalized_hashes") or {}
                )
                if source_hashes != target_hashes:
                    raise RuntimeError(
                        "conversations.db 导入后规范化哈希不一致，已拒绝完成导入"
                    )
                source_counts = dict(conversations_report.get("counts") or {})
                target_counts = dict(target_conversations_report.get("counts") or {})
                if source_counts != target_counts:
                    raise RuntimeError(
                        "conversations.db 导入后会话、消息或待总结统计不一致"
                    )
            report = {
                "run_id": run_id,
                "library_id": library_id,
                "source": source_report,
                "conversations_source": conversations_report,
                "conversations_target": target_conversations_report,
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
                "LivingMemory 单文件导入完成：memory_store_id=%s run_id=%s generation=%s",
                library_id,
                run_id,
                (rebuild.get("manifest") or {}).get("generation"),
            )
            return {
                "run_id": run_id,
                "library_id": library_id,
                "source": source_report,
                "conversations_source": conversations_report,
                "conversations_target": target_conversations_report,
                "stats": stats,
                "graph_recovery": rebuild.get("graph_recovery"),
                "rebuild": rebuild,
                "report_path": str(report_path),
            }
        except JobControlSignal:
            # The durable task runner owns pause/interruption/stop semantics.
            # Preserve the archived source, installed database and verified
            # index segments so this same job can continue safely.
            raise
        except Exception:
            if controlled:
                # ResumableMemoryStoreTasks restores the complete pre-task state
                # before the job is allowed to enter a terminal failure state.
                raise
            logger.exception(
                "LivingMemory 单文件导入失败，正在回滚：memory_store_id=%s run_id=%s",
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
            if not controlled:
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
        source = self._library_dir(library_id)
        trash_dir = (
            self._trash_root() / f"{library_id}-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        trash_memory_db = trash_dir / "livingmemory.db"
        trash_conversations_db = trash_dir / "conversations.db"
        staging = (
            self._type_root()
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
                "livingmemory": str(trash_memory_db)
                if trash_memory_db.exists()
                else None,
                "conversations": str(trash_conversations_db)
                if trash_conversations_db.exists()
                else None,
            },
        }

    async def rebuild_library(
        self,
        library_id: str,
        provider_id: str | None,
        progress=None,
        *,
        task_context=None,
        checkpoint_dir: Path | None = None,
        before_database_repair=None,
    ) -> dict[str, Any]:
        record = await self.control.get_library(library_id)
        if not record:
            raise KeyError(library_id)
        provider = await self.control.get_provider(provider_id or record.provider_id)
        if provider and provider_kind(provider.config.type) != "embedding":
            raise ValueError("索引重建必须使用 Embedding Provider")
        if not provider:
            raise ValueError("Provider 不存在")
        if not provider.config.enabled:
            raise ValueError("Provider 未启用")
        (
            provider,
            provider_metadata_changed,
        ) = await self._refresh_context_length_for_long_task(
            provider,
            checkpoint_dir=checkpoint_dir,
        )
        if provider_metadata_changed:
            await self.unload_runtime(
                library_id,
                reason="embedding_context_capability_refreshed",
            )
        runtime = await self.get_runtime(library_id)
        logger.warning(
            "开始重建记忆库索引：memory_store_id=%s provider=%s revision=%s",
            library_id,
            provider.provider_id,
            provider.revision,
        )
        result = await runtime.rebuild_with_provider(
            provider,
            progress,
            job_context=task_context,
            checkpoint_dir=checkpoint_dir,
            before_database_repair=before_database_repair,
        )
        await self.control.bind_library(library_id, provider, result["manifest"])
        logger.warning(
            "记忆库索引重建并绑定完成：memory_store_id=%s provider=%s revision=%s generation=%s",
            library_id,
            provider.provider_id,
            provider.revision,
            (result.get("manifest") or {}).get("generation"),
        )
        return result

    async def rebuild_graph(
        self,
        library_id: str,
        progress=None,
        *,
        task_context=None,
        checkpoint_dir: Path | None = None,
    ) -> dict[str, Any]:
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
        (
            provider,
            provider_metadata_changed,
        ) = await self._refresh_context_length_for_long_task(
            provider,
            checkpoint_dir=checkpoint_dir,
        )
        if provider_metadata_changed:
            await self.unload_runtime(
                library_id,
                reason="embedding_context_capability_refreshed",
            )
        runtime = await self.get_runtime(library_id)
        logger.warning(
            "开始重建记忆库图数据与索引：memory_store_id=%s provider=%s revision=%s",
            library_id,
            provider.provider_id,
            provider.revision,
        )
        result = await runtime.rebuild_graph(
            progress,
            job_context=task_context,
            checkpoint_dir=checkpoint_dir,
        )
        await self.control.bind_library(
            library_id,
            provider,
            result["manifest"],
        )
        logger.warning(
            "记忆库图数据与索引重建完成：memory_store_id=%s provider=%s revision=%s generation=%s",
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
                runtime = self.runtimes.get(
                    DatabaseRef(
                        str(usage.get("database_type") or LIVINGMEMORY_V8_TYPE),
                        str(usage.get("database_id") or usage.get("library_id") or ""),
                    )
                )
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
            draft["context_length_mode"] = "auto"
            draft["max_context_tokens"] = 0
            draft["max_context_tokens_source"] = ""
            return draft
        mode = (
            config.context_length_mode
            if config.context_length_mode in {"auto", "manual"}
            else "auto"
        )
        draft["context_length_mode"] = mode
        if mode == "manual":
            if int(config.max_context_tokens or 0) < MIN_VALID_CONTEXT_TOKENS:
                raise ValueError(
                    f"manual max_context_tokens must be >= {MIN_VALID_CONTEXT_TOKENS}"
                )
            draft["max_context_tokens"] = int(config.max_context_tokens)
            draft["max_context_tokens_source"] = (
                str(config.max_context_tokens_source or "").strip() or "manual:user"
            )
            return draft
        changed = (
            force
            or not base
            or config.type != base.type
            or config.api_base != base.api_base
            or config.model != base.model
            or getattr(base, "context_length_mode", "auto") != "auto"
            or int(config.max_context_tokens or 0) < MIN_VALID_CONTEXT_TOKENS
        )
        if changed:
            detected = await self._detect_context_length(config)
            if int(detected.get("max_context_tokens") or 0) >= MIN_VALID_CONTEXT_TOKENS:
                draft["max_context_tokens"] = int(detected["max_context_tokens"])
                draft["max_context_tokens_source"] = str(
                    detected.get("max_context_tokens_source") or ""
                )
                draft["context_length_mode"] = "auto"
                return draft
            fallback_tokens = int(
                draft.get("max_context_tokens") or config.max_context_tokens or 0
            )
            if fallback_tokens < MIN_VALID_CONTEXT_TOKENS:
                fallback_tokens = MANUAL_CONTEXT_FALLBACK_TOKENS
            draft["context_length_mode"] = "manual"
            draft["max_context_tokens"] = fallback_tokens
            draft["max_context_tokens_source"] = "manual:fallback-undetected"
            return draft
        if int(config.max_context_tokens or 0) < MIN_VALID_CONTEXT_TOKENS:
            draft["max_context_tokens"] = 0
            draft["max_context_tokens_source"] = ""
        else:
            draft["max_context_tokens"] = int(config.max_context_tokens)
            draft["max_context_tokens_source"] = str(
                config.max_context_tokens_source or ""
            )
        return draft

    async def _detect_context_length(self, config: ProviderConfig) -> dict[str, Any]:
        provider = build_provider(
            replace(
                config,
                context_length_mode="auto",
                max_context_tokens=0,
                max_context_tokens_source="",
            )
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

    async def _refresh_context_length_for_long_task(
        self,
        provider: ProviderRevision,
        *,
        checkpoint_dir: Path | None,
    ) -> tuple[ProviderRevision, bool]:
        """Probe once at long-task start and persist display capability metadata.

        A resumed job owns a verified capability snapshot in its checkpoint.  It
        must reuse that snapshot so a single logical task never changes its
        chunking boundary halfway through a rebuild.
        """
        if checkpoint_dir is not None:
            try:
                checkpoint = await run_blocking(read_ab_checkpoint, checkpoint_dir)
                capability = (checkpoint or {}).get("embedding_capability")
                if (
                    isinstance(capability, dict)
                    and int(capability.get("detected_max_context_tokens") or 0) >= 128
                ):
                    return provider, False
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                # The index task owns checkpoint-corruption handling and can
                # recover from the alternate A/B slot deterministically.
                pass

        context_length_mode = "auto"
        if provider.config.context_length_mode == "manual":
            context_length_mode = "manual"
            max_context_tokens = int(provider.config.max_context_tokens or 0)
            max_context_tokens_source = str(
                provider.config.max_context_tokens_source or "manual:user"
            )
        else:
            detected = await self._detect_context_length(provider.config)
            max_context_tokens = int(detected.get("max_context_tokens") or 0)
            max_context_tokens_source = str(
                detected.get("max_context_tokens_source") or ""
            )
        if max_context_tokens < MIN_VALID_CONTEXT_TOKENS:
            existing_tokens = int(provider.config.max_context_tokens or 0)
            max_context_tokens = (
                existing_tokens
                if existing_tokens >= MIN_VALID_CONTEXT_TOKENS
                else MANUAL_CONTEXT_FALLBACK_TOKENS
            )
            max_context_tokens_source = "manual:fallback-undetected"
            context_length_mode = "manual"
        if max_context_tokens < 128:
            raise JobInterrupted(
                "provider_context_probe_failed",
                "无法确认 Embedding Provider 的上下文长度，任务已中断并保留在安全状态",
                error=(
                    f"provider={provider.provider_id}, "
                    f"detected_tokens={max_context_tokens}, minimum=128"
                ),
            )
        if (
            int(provider.config.max_context_tokens or 0) == max_context_tokens
            and str(provider.config.max_context_tokens_source or "")
            == max_context_tokens_source
            and provider.config.context_length_mode == context_length_mode
        ):
            return provider, False
        refreshed = await self.control.update_provider(
            provider.provider_id,
            {
                "context_length_mode": context_length_mode,
                "max_context_tokens": max_context_tokens,
                "max_context_tokens_source": max_context_tokens_source,
            },
        )
        logger.info(
            "长任务已刷新 Embedding 上下文能力：provider=%s revision=%s tokens=%s source=%s",
            refreshed.provider_id,
            refreshed.revision,
            max_context_tokens,
            max_context_tokens_source,
        )
        return refreshed, True

    async def detect_context_length(
        self,
        payload: dict[str, Any],
        provider_id: str | None = None,
    ) -> dict[str, Any]:
        draft = dict(payload)
        draft["context_length_mode"] = "auto"
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
            raise ValueError(
                "Rerank Provider does not support max context length detection"
            )
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
        logger.info(
            "创建临时 Provider 客户端用于测试：provider=%s revision=%s",
            provider_id,
            revision or record.revision,
        )
        provider = (
            build_rerank_provider(record.config)
            if provider_kind(record.config.type) == "rerank"
            else build_provider(record.config)
        )
        try:
            return await provider.test_connection()
        finally:
            await provider.close()

    async def test_provider_draft(self, payload: dict[str, Any]) -> dict[str, Any]:
        logger.info(
            "创建临时草稿 Provider 客户端用于测试：provider=%s type=%s",
            payload.get("id"),
            payload.get("type"),
        )
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
