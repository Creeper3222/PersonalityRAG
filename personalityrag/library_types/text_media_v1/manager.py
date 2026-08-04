from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ...context_lengths import (
    MANUAL_CONTEXT_FALLBACK_TOKENS,
    MIN_VALID_CONTEXT_TOKENS,
)
from ...database_types import (
    DATABASE_CATEGORY_KNOWLEDGE,
    TEXT_MEDIA_V1_TYPE,
    DatabaseDriverContext,
    DatabaseRef,
    database_identity_fields,
    database_type_registry,
)
from ...identifiers import validate_identifier
from ...io_utils import run_blocking
from ...logger import logger
from ...resource_limits import (
    effective_runtime_capacity,
    effective_runtime_idle_minutes,
)
from ...providers import build_provider, build_rerank_provider, provider_kind
from .indexes import TextMediaIndex
from .package import export_tmkb, install_tmkb
from .retrieval import (
    normalize_retrieval_config,
    rerank_calibration_settings_fingerprint,
)
from .service import TextMediaService
from .storage import TextMediaStorage
from .text import (
    DEFAULT_VISUAL_INTENT_POLICY,
    PROTECTED_VISUAL_BLOCKER_TERMS,
    VISUAL_INTENT_DETECTOR_VERSION,
    normalize_visual_intent_policy,
    visual_intent_policy_fingerprint,
)
from .visual_intent_policy import migrate_legacy_visual_intent_policy
from .resumable_tasks import ResumableTextMediaTasks


@dataclass(slots=True)
class TextMediaRuntimeResidencyState:
    lease_count: int
    last_used_at: float


class TextMediaV1Manager:
    def __init__(self, services: DatabaseDriverContext):
        self.services = services
        self.root = services.root
        self.data_dir = services.data_root
        self.system_path = services.system_path
        self.config = services.config
        self.control = services.control
        self.runtimes: dict[DatabaseRef, TextMediaService] = {}
        self._loads: dict[DatabaseRef, asyncio.Task[TextMediaService]] = {}
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._residency: dict[
            DatabaseRef, TextMediaRuntimeResidencyState
        ] = {}
        self._load_finalizers: set[asyncio.Task[None]] = set()
        self._sweeper_task: asyncio.Task[None] | None = None
        self._runtime_loading_suspended = False
        self._runtime_loading_owner: asyncio.Task[Any] | None = None
        self._closing = False
        self._summary_cache: dict[
            DatabaseRef,
            tuple[
                tuple[tuple[int, int] | None, ...],
                dict[str, Any],
                dict[str, Any],
            ],
        ] = {}
        self.resumable_tasks = ResumableTextMediaTasks(self)

    def _type_root(self) -> Path:
        return database_type_registry.type_root(self.data_dir, TEXT_MEDIA_V1_TYPE)

    def _trash_root(self) -> Path:
        return database_type_registry.trash_type_root(
            self.data_dir, TEXT_MEDIA_V1_TYPE
        )

    @property
    def jobs(self):
        return self.services.jobs

    @staticmethod
    def _offline_summary(
        directory: Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            TextMediaStorage.summary_snapshot_from_disk(directory),
            TextMediaIndex.disk_status(directory),
        )

    @staticmethod
    def _summary_signature(
        directory: Path,
    ) -> tuple[tuple[int, int] | None, ...]:
        index_root = directory / "derived" / "indexes"
        paths = (
            directory / "textmediaknowledge.db",
            Path(str(directory / "textmediaknowledge.db") + "-wal"),
            index_root / "CURRENT",
            index_root / "media" / "CURRENT",
        )
        signature: list[tuple[int, int] | None] = []
        for path in paths:
            try:
                stat = path.stat()
            except FileNotFoundError:
                signature.append(None)
            else:
                # Overlay filesystems may advance ctime while SQLite opens a
                # WAL snapshot for reading. Size plus nanosecond mtime still
                # changes for every database/WAL write without turning a
                # read-only summary into a false cache invalidation.
                signature.append((int(stat.st_size), int(stat.st_mtime_ns)))
        return tuple(signature)

    @staticmethod
    def _copy_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "metadata": dict(snapshot["metadata"]),
            "stats": dict(snapshot["stats"]),
            "needs_recalibration": bool(snapshot["needs_recalibration"]),
        }

    async def _summary_from_disk(
        self,
        ref: DatabaseRef,
        directory: Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        signature = self._summary_signature(directory)
        cached = self._summary_cache.get(ref)
        if cached is not None and cached[0] == signature:
            return self._copy_summary(cached[1]), dict(cached[2])
        snapshot, indexes = await run_blocking(
            self._offline_summary,
            directory,
        )
        completed_signature = self._summary_signature(directory)
        if completed_signature == signature:
            self._summary_cache[ref] = (
                signature,
                self._copy_summary(snapshot),
                dict(indexes),
            )
        return snapshot, indexes

    async def initialize(self) -> None:
        if self.jobs is not None:
            self.jobs.set_operation_resolver(
                self.resumable_tasks.resolve,
                database_type=TEXT_MEDIA_V1_TYPE,
            )
            self.jobs.set_database_state_provider(
                self.task_database_state,
                database_type=TEXT_MEDIA_V1_TYPE,
            )
        identities = await self.control.list_database_identities()
        type_root = self._type_root()
        type_root.mkdir(parents=True, exist_ok=True)
        registered_ids = {
            str(item["id"])
            for item in identities
            if item["database_type"] == TEXT_MEDIA_V1_TYPE
        }
        for child in type_root.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith(".tmkbs-import-") or (
                child.name.startswith(".") and ".import-" in child.name
            ):
                shutil.rmtree(child, ignore_errors=True)
                continue
            if child.name in registered_ids or not (child / "textmediaknowledge.db").exists():
                continue
            quarantine = (
                self._trash_root()
                / "orphaned"
                / f"{child.name}-{int(time.time())}"
            )
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(child), str(quarantine))
        for item in identities:
            if item["database_type"] != TEXT_MEDIA_V1_TYPE:
                continue
            ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, str(item["id"]))
            directory = database_type_registry.data_dir(
                self.data_dir, ref
            )
            if not (directory / "textmediaknowledge.db").exists():
                continue
            storage = TextMediaStorage(directory)
            try:
                await storage.initialize()
                await migrate_legacy_visual_intent_policy(storage, directory)
            finally:
                await storage.close()
        self._start_runtime_sweeper()

    async def close(self) -> None:
        self._closing = True
        if self._sweeper_task is not None:
            self._sweeper_task.cancel()
            await asyncio.gather(self._sweeper_task, return_exceptions=True)
            self._sweeper_task = None
        async with self._condition:
            load_tasks = list(self._loads.values())
            self._loads.clear()
            services = list(self.runtimes.values())
            self.runtimes.clear()
            self._residency.clear()
            self._condition.notify_all()
        for task in load_tasks:
            task.cancel()
        if load_tasks:
            await asyncio.gather(*load_tasks, return_exceptions=True)
            await asyncio.sleep(0)
        if self._load_finalizers:
            await asyncio.gather(
                *list(self._load_finalizers),
                return_exceptions=True,
            )
        await asyncio.gather(
            *(service.close() for service in services),
            return_exceptions=True,
        )

    async def _build_service(self, ref: DatabaseRef) -> TextMediaService:
        directory = database_type_registry.data_dir(
            self.data_dir, ref
        )
        storage = TextMediaStorage(directory)
        await storage.initialize()
        try:
            visual_intent_policy = await migrate_legacy_visual_intent_policy(
                storage,
                directory,
            )
            meta = await storage.metadata()
        except Exception:
            await storage.close()
            raise
        provider = None
        reranker = None
        try:
            revision = await self.control.get_provider(
                str(meta["provider_id"]), int(meta["provider_revision"])
            ) if meta["provider_id"] else None
            if (
                revision is not None
                and revision.config_sha256 == str(meta["provider_fingerprint"])
                and revision.config.enabled
                and provider_kind(revision.config.type) == "embedding"
            ):
                provider = build_provider(revision.config)
            elif meta["status"] == "ready":
                await storage.update_metadata({"status": "provider_binding_required"})
            rerank_revision = (
                await self.control.get_provider(
                    str(meta["rerank_provider_id"]),
                    int(meta["rerank_provider_revision"]),
                )
                if meta.get("rerank_provider_id")
                else None
            )
            rerank_info: dict[str, Any] = {}
            if (
                rerank_revision is not None
                and rerank_revision.config_sha256
                == str(meta.get("rerank_provider_fingerprint") or "")
                and rerank_revision.config.enabled
                and provider_kind(rerank_revision.config.type) == "rerank"
            ):
                reranker = build_rerank_provider(rerank_revision.config)
                rerank_info = {
                    "id": rerank_revision.provider_id,
                    "revision": rerank_revision.revision,
                    "fingerprint": rerank_revision.config_sha256,
                }
            indexes = TextMediaIndex(directory)
            await indexes.load()
            if not indexes.status()["generation"]:
                await indexes.rebuild(storage)
            return TextMediaService(
                directory,
                storage,
                indexes,
                provider,
                reranker,
                rerank_info,
                visual_intent_policy,
            )
        except BaseException:
            if provider is not None:
                await provider.close()
            if reranker is not None:
                await reranker.close()
            await storage.close()
            raise

    async def suspend_runtime_loading(
        self,
        *,
        owner: asyncio.Task[Any] | None = None,
    ) -> None:
        """Block new cold loads and drain any in-flight single-flight load."""

        async with self._condition:
            self._runtime_loading_suspended = True
            self._runtime_loading_owner = owner or asyncio.current_task()
            self._condition.notify_all()
        while True:
            async with self._lock:
                loads = list(self._loads.values())
            if not loads:
                break
            await asyncio.gather(
                *(asyncio.shield(task) for task in loads),
                return_exceptions=True,
            )
            await asyncio.sleep(0)

    async def resume_runtime_loading(self) -> None:
        async with self._condition:
            self._runtime_loading_suspended = False
            self._runtime_loading_owner = None
            self._condition.notify_all()

    def _track_runtime_load(
        self,
        ref: DatabaseRef,
        task: asyncio.Task[TextMediaService],
    ) -> None:
        def done(completed: asyncio.Task[TextMediaService]) -> None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            finalizer = loop.create_task(
                self._finalize_runtime_load(ref, completed),
                name=f"text-media-runtime-finalize-{ref.key}",
            )
            self._load_finalizers.add(finalizer)
            finalizer.add_done_callback(self._load_finalizers.discard)

        task.add_done_callback(done)

    async def _finalize_runtime_load(
        self,
        ref: DatabaseRef,
        task: asyncio.Task[TextMediaService],
    ) -> None:
        try:
            built = task.result()
        except BaseException:
            async with self._condition:
                if self._loads.get(ref) is task:
                    self._loads.pop(ref, None)
                self._condition.notify_all()
            return
        close_built = False
        async with self._condition:
            current = self.runtimes.get(ref)
            owns_flight = self._loads.get(ref) is task
            if owns_flight:
                self._loads.pop(ref, None)
            if current is None and owns_flight and not self._closing:
                self.runtimes[ref] = built
                self._residency[ref] = TextMediaRuntimeResidencyState(
                    lease_count=0,
                    last_used_at=time.monotonic(),
                )
                logger.info("text_media_v1 runtime loaded: database=%s", ref.key)
            elif current is not built:
                close_built = True
            self._condition.notify_all()
        if close_built:
            await built.close()

    @staticmethod
    def _text_media_ref(database: str | DatabaseRef) -> DatabaseRef:
        ref = (
            database
            if isinstance(database, DatabaseRef)
            else DatabaseRef(TEXT_MEDIA_V1_TYPE, database)
        )
        if ref.database_type != TEXT_MEDIA_V1_TYPE:
            raise KeyError(ref.key)
        return ref

    async def _runtime_or_load(
        self,
        ref: DatabaseRef,
        *,
        acquire: bool,
        touch: bool,
    ) -> TextMediaService:
        async with self._lock:
            current = self.runtimes.get(ref)
            if current is not None:
                state = self._residency[ref]
                if acquire:
                    state.lease_count += 1
                if touch:
                    state.last_used_at = time.monotonic()
                return current
            task = self._loads.get(ref)
            if task is None:
                if (
                    self._runtime_loading_suspended
                    and asyncio.current_task() is not self._runtime_loading_owner
                ):
                    raise RuntimeError(
                        "runtime loading is temporarily suspended for maintenance"
                    )
                if self._closing:
                    raise RuntimeError("TextMediaV1Manager is closing")
                task = asyncio.create_task(
                    self._build_service(ref),
                    name=f"text-media-runtime-load-{ref.key}",
                )
                self._loads[ref] = task
                self._track_runtime_load(ref, task)
        try:
            built = await asyncio.shield(task)
        except BaseException:
            async with self._condition:
                if self._loads.get(ref) is task:
                    self._loads.pop(ref, None)
                self._condition.notify_all()
            raise

        close_built = False
        async with self._condition:
            current = self.runtimes.get(ref)
            if current is None:
                if self._closing:
                    close_built = True
                else:
                    current = built
                    self.runtimes[ref] = built
                    self._residency[ref] = TextMediaRuntimeResidencyState(
                        lease_count=0,
                        last_used_at=time.monotonic(),
                    )
                    logger.info(
                        "text_media_v1 runtime loaded: database=%s",
                        ref.key,
                    )
            elif current is not built:
                close_built = True
            if self._loads.get(ref) is task:
                self._loads.pop(ref, None)
            if current is not None:
                state = self._residency[ref]
                if acquire:
                    state.lease_count += 1
                if touch:
                    state.last_used_at = time.monotonic()
            self._condition.notify_all()
        if close_built:
            await built.close()
        if current is None:
            raise RuntimeError("TextMediaV1Manager is closing")
        await self.sweep_runtimes(expire_idle=False, protect={ref})
        return current

    async def get_runtime(
        self,
        database: str | DatabaseRef,
        *_,
        touch: bool = True,
        **__,
    ) -> TextMediaService:
        return await self._runtime_or_load(
            self._text_media_ref(database),
            acquire=False,
            touch=touch,
        )

    async def acquire_runtime(
        self,
        database: str | DatabaseRef,
        *_,
        touch: bool = True,
        **__,
    ) -> TextMediaService:
        return await self._runtime_or_load(
            self._text_media_ref(database),
            acquire=True,
            touch=touch,
        )

    async def release_runtime(
        self,
        database: str | DatabaseRef,
        *_,
        touch: bool = True,
        **__,
    ) -> None:
        ref = self._text_media_ref(database)
        async with self._condition:
            state = self._residency.get(ref)
            if state is None:
                return
            state.lease_count = max(0, state.lease_count - 1)
            if touch:
                state.last_used_at = time.monotonic()
            self._condition.notify_all()

    async def unload_runtime(
        self,
        database: str | DatabaseRef,
        *_,
        reason: str = "manual",
        **__,
    ) -> bool:
        ref = self._text_media_ref(database)
        async with self._condition:
            while (
                state := self._residency.get(ref)
            ) is not None and state.lease_count > 0:
                await self._condition.wait()
            service = self.runtimes.pop(ref, None)
            self._residency.pop(ref, None)
            self._condition.notify_all()
        if service is None:
            return False
        await service.close()
        logger.info(
            "text_media_v1 runtime released: database=%s reason=%s",
            ref.key,
            reason,
        )
        return True

    def _start_runtime_sweeper(self) -> None:
        if self._sweeper_task is None or self._sweeper_task.done():
            self._sweeper_task = asyncio.create_task(
                self._runtime_sweeper_loop(),
                name="text-media-runtime-residency",
            )

    async def _runtime_sweeper_loop(self) -> None:
        while not self._closing:
            idle_minutes = effective_runtime_idle_minutes(
                self.config.runtime_residency.idle_minutes,
                self.config.performance_profile,
            )
            interval = min(60.0, max(5.0, idle_minutes * 30.0))
            try:
                await asyncio.sleep(interval)
                await self.sweep_runtimes()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("text_media_v1 runtime sweep failed")

    async def _pinned_runtime_refs(
        self,
        refs: list[DatabaseRef],
    ) -> set[DatabaseRef]:
        if not refs:
            return set()
        adapter_map = await self.control.active_database_adapter_connections_map(
            refs
        )
        pinned = {ref for ref in refs if adapter_map.get(ref.key)}
        if self.jobs is not None:
            jobs = await asyncio.gather(
                *(
                    self.jobs.active_database_job(
                        ref,
                        include_read_only=True,
                    )
                    for ref in refs
                )
            )
            pinned.update(
                ref for ref, job in zip(refs, jobs, strict=True) if job
            )
        return pinned

    async def sweep_runtimes(
        self,
        *,
        expire_idle: bool = True,
        protect: set[DatabaseRef] | None = None,
    ) -> list[str]:
        protect = set(protect or ())
        now = time.monotonic()
        idle_seconds = (
            effective_runtime_idle_minutes(
                self.config.runtime_residency.idle_minutes,
                self.config.performance_profile,
            )
            * 60.0
        )
        capacity = effective_runtime_capacity(
            self.config.runtime_residency.max_non_default_runtimes,
            self.config.performance_profile,
        )
        async with self._lock:
            candidates = sorted(
                (
                    ref
                    for ref, state in self._residency.items()
                    if state.lease_count == 0 and ref not in protect
                ),
                key=lambda ref: self._residency[ref].last_used_at,
            )
            loaded_count = len(self.runtimes)
            idle_candidates = {
                ref
                for ref in candidates
                if expire_idle
                and now - self._residency[ref].last_used_at >= idle_seconds
            }
        pinned = await self._pinned_runtime_refs(candidates)
        selected: list[tuple[DatabaseRef, str]] = []
        remaining = loaded_count
        for ref in candidates:
            if ref in pinned:
                continue
            if ref in idle_candidates:
                selected.append((ref, "idle"))
                remaining -= 1
        for ref in candidates:
            if remaining <= capacity:
                break
            if ref in pinned or any(item[0] == ref for item in selected):
                continue
            selected.append((ref, "capacity"))
            remaining -= 1

        evicted: list[str] = []
        for ref, reason in selected:
            async with self._condition:
                state = self._residency.get(ref)
                if state is None or state.lease_count > 0 or ref in protect:
                    continue
                service = self.runtimes.pop(ref, None)
                self._residency.pop(ref, None)
                self._condition.notify_all()
            if service is None:
                continue
            try:
                await service.close()
            except Exception:
                logger.exception(
                    "text_media_v1 runtime release failed: database=%s reason=%s",
                    ref.key,
                    reason,
                )
            else:
                evicted.append(ref.id)
                logger.info(
                    "text_media_v1 runtime released: database=%s reason=%s",
                    ref.key,
                    reason,
                )
        return evicted

    async def apply_runtime_residency(self) -> list[str]:
        return await self.sweep_runtimes()

    def runtime_residency_status(self) -> dict[str, Any]:
        return {
            "loaded_library_ids": [ref.id for ref in self.runtimes],
            "loaded_databases": [ref.public() for ref in self.runtimes],
            "runtimes": {
                ref.key: {
                    "database_type": ref.database_type,
                    "database_id": ref.id,
                    "lease_count": state.lease_count,
                    "last_used_at": state.last_used_at,
                }
                for ref, state in self._residency.items()
            },
        }

    async def prepare_embedding_context_for_long_task(
        self,
        provider_id: str,
        *,
        expected_revision: int = 0,
        expected_fingerprint: str = "",
    ) -> tuple[Any, dict[str, Any]]:
        """Refresh and pin one Embedding Provider before a long task embeds.

        This intentionally mirrors the LivingMemory v8 long-index contract:
        automatic mode probes on every new logical task and persists the
        capability at Provider scope, while manual mode never probes.  A
        resumed task reuses the revision stored in its own durable state and
        therefore does not call this method again.
        """
        provider_id = validate_identifier(provider_id, field="Provider ID")
        provider = await self.control.get_provider(provider_id)
        if (
            provider is None
            or not provider.config.enabled
            or provider_kind(provider.config.type) != "embedding"
        ):
            raise ValueError("Embedding Provider is unavailable")
        expected_fingerprint = str(expected_fingerprint or "")
        if expected_fingerprint and (
            provider.config_sha256 != expected_fingerprint
            or (
                int(expected_revision or 0) > 0
                and provider.revision != int(expected_revision)
            )
        ):
            raise ValueError(
                "bound Embedding Provider changed; rebuild the text media "
                "knowledge library before starting this task"
            )

        previous_revision = int(provider.revision)
        previous_fingerprint = str(provider.config_sha256)
        mode = str(provider.config.context_length_mode or "auto")
        probe_attempted = False
        persisted = False
        if mode == "manual":
            max_context_tokens = int(provider.config.max_context_tokens or 0)
            if max_context_tokens < MIN_VALID_CONTEXT_TOKENS:
                raise ValueError(
                    "manual max_context_tokens must be at least "
                    f"{MIN_VALID_CONTEXT_TOKENS}"
                )
            max_context_tokens_source = str(
                provider.config.max_context_tokens_source or "manual:user"
            )
        else:
            probe_attempted = True
            probe = build_provider(
                replace(
                    provider.config,
                    context_length_mode="auto",
                    max_context_tokens=0,
                    max_context_tokens_source="",
                )
            )
            try:
                try:
                    detected = await probe.detect_context_length()
                except Exception as exc:
                    logger.info(
                        "text_media_v1 long-task context probe failed: "
                        "provider=%s error=%s",
                        provider_id,
                        exc,
                    )
                    detected = {}
            finally:
                await probe.close()
            max_context_tokens = int(
                detected.get("max_context_tokens") or 0
            )
            max_context_tokens_source = str(
                detected.get("max_context_tokens_source") or ""
            )
            mode = "auto"
            if max_context_tokens < MIN_VALID_CONTEXT_TOKENS:
                configured = int(provider.config.max_context_tokens or 0)
                max_context_tokens = (
                    configured
                    if configured >= MIN_VALID_CONTEXT_TOKENS
                    else MANUAL_CONTEXT_FALLBACK_TOKENS
                )
                max_context_tokens_source = "manual:fallback-undetected"
                mode = "manual"

        if max_context_tokens < MIN_VALID_CONTEXT_TOKENS:
            raise ValueError(
                "unable to establish a safe Embedding Provider context "
                f"length: {max_context_tokens}"
            )
        if (
            str(provider.config.context_length_mode or "auto") != mode
            or int(provider.config.max_context_tokens or 0)
            != max_context_tokens
            or str(provider.config.max_context_tokens_source or "")
            != max_context_tokens_source
        ):
            provider = await self.control.update_provider(
                provider_id,
                {
                    "context_length_mode": mode,
                    "max_context_tokens": max_context_tokens,
                    "max_context_tokens_source": max_context_tokens_source,
                },
            )
            persisted = True
        logger.info(
            "text_media_v1 long-task context capability ready: "
            "provider=%s revision=%s mode=%s tokens=%s source=%s probed=%s persisted=%s",
            provider.provider_id,
            provider.revision,
            mode,
            max_context_tokens,
            max_context_tokens_source,
            probe_attempted,
            persisted,
        )
        return provider, {
            "policy": "probe_each_long_task",
            "provider_id": provider.provider_id,
            "provider_revision": int(provider.revision),
            "provider_fingerprint": str(provider.config_sha256),
            "previous_provider_revision": previous_revision,
            "previous_provider_fingerprint": previous_fingerprint,
            "context_length_mode": mode,
            "max_context_tokens": max_context_tokens,
            "max_context_tokens_source": max_context_tokens_source,
            "probe_attempted": probe_attempted,
            "persisted": persisted,
            "binding_changed": (
                int(provider.revision) != previous_revision
                or str(provider.config_sha256) != previous_fingerprint
            ),
            "verified_at": time.time(),
        }

    @asynccontextmanager
    async def runtime_lease(self, database: str | DatabaseRef, *args, **kwargs):
        service = await self.acquire_runtime(database, *args, **kwargs)
        try:
            yield service
        finally:
            await self.release_runtime(database, *args, **kwargs)

    async def create_library(self, payload: dict[str, Any]) -> dict[str, Any]:
        database_id = validate_identifier(str(payload.get("id") or ""), field="知识库 ID")
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("知识库名称不能为空")
        provider_id = validate_identifier(str(payload.get("provider_id") or ""), field="Provider ID")
        revision = await self.control.get_provider(provider_id)
        if revision is None or provider_kind(revision.config.type) != "embedding":
            raise ValueError("知识库必须绑定 Embedding Provider")
        if not revision.config.enabled:
            raise ValueError("指定的 Provider 未启用")
        rerank_revision = None
        rerank_provider_id = str(
            payload.get("rerank_provider_id") or ""
        ).strip()
        if rerank_provider_id:
            rerank_provider_id = validate_identifier(
                rerank_provider_id, field="Rerank Provider ID"
            )
            rerank_revision = await self.control.get_provider(rerank_provider_id)
            if (
                rerank_revision is None
                or provider_kind(rerank_revision.config.type) != "rerank"
            ):
                raise ValueError("指定的 Provider 不是 Rerank Provider")
            if not rerank_revision.config.enabled:
                raise ValueError("指定的 Rerank Provider 未启用")
        ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id)
        directory = database_type_registry.data_dir(
            self.data_dir, DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id)
        )
        if directory.exists():
            raise ValueError(f"知识库 ID {database_id} 已存在")
        await self.control.register_database_identity(ref, category=DATABASE_CATEGORY_KNOWLEDGE)
        storage = TextMediaStorage(directory)
        try:
            await storage.initialize()
            await storage.create_library(
                database_id=database_id,
                name=name,
                description=str(payload.get("description") or ""),
                provider_id=revision.provider_id,
                provider_revision=revision.revision,
                provider_fingerprint=revision.config_sha256,
                rerank_provider_id=(
                    rerank_revision.provider_id if rerank_revision else ""
                ),
                rerank_provider_revision=(
                    rerank_revision.revision if rerank_revision else 0
                ),
                rerank_provider_fingerprint=(
                    rerank_revision.config_sha256 if rerank_revision else ""
                ),
            )
            await storage.close()
            await self.get_runtime(ref)
        except Exception:
            await storage.close()
            await self.control.delete_database_identity(ref)
            if directory.exists():
                shutil.rmtree(directory, ignore_errors=True)
            raise
        return await self.library_detail(database_id)

    async def library_detail(self, database_id: str) -> dict[str, Any]:
        ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id)
        service = await self.get_runtime(ref)
        meta, stats, adapter_connections, calibrations = await asyncio.gather(
            service.storage.metadata(),
            service.storage.statistics(),
            self.control.active_adapter_connections(ref),
            service.storage.list_document_media_calibrations(),
        )
        descriptor = database_type_registry.require(TEXT_MEDIA_V1_TYPE).descriptor
        provider = await self.control.get_provider(
            str(meta["provider_id"]), int(meta["provider_revision"])
        ) if meta["provider_id"] else None
        rerank_provider = (
            await self.control.get_provider(
                str(meta.get("rerank_provider_id") or ""),
                int(meta.get("rerank_provider_revision") or 0),
            )
            if meta.get("rerank_provider_id")
            else None
        )
        rerank_fingerprint = str(meta.get("rerank_provider_fingerprint") or "")
        rerank_available = bool(
            rerank_provider is not None
            and rerank_provider.config.enabled
            and provider_kind(rerank_provider.config.type) == "rerank"
            and rerank_provider.config_sha256 == rerank_fingerprint
        )
        retrieval_settings = normalize_retrieval_config(
            meta.get("retrieval_config_json")
        )
        retrieval_settings["visual_intent_policy"] = dict(
            service.visual_intent_policy
        )
        calibration_settings_fingerprint = (
            rerank_calibration_settings_fingerprint(retrieval_settings)
        )
        needs_recalibration = bool(
            rerank_fingerprint
            and any(
                str(
                    item.get("calibration_rerank_provider_fingerprint") or ""
                )
                != rerank_fingerprint
                or {
                    value
                    for value in str(
                        item.get(
                            "calibration_rerank_settings_fingerprints"
                        )
                        or ""
                    ).split(",")
                    if value
                }
                != {calibration_settings_fingerprint}
                for item in calibrations
                if str(item.get("semantic_mode") or "") == "calibrated"
            )
        )
        active_job = (
            await self.jobs.active_long_job(ref) if self.jobs is not None else None
        )
        return {
            "id": database_id,
            **database_identity_fields(ref),
            "database_category": DATABASE_CATEGORY_KNOWLEDGE,
            "name": meta["name"],
            "description": meta["description"],
            "is_default": False,
            "status": meta["status"],
            "provider_id": meta["provider_id"],
            "provider_revision": meta["provider_revision"],
            "provider": provider.public() if provider else None,
            "rerank_provider_id": str(meta.get("rerank_provider_id") or ""),
            "rerank_provider_revision": int(
                meta.get("rerank_provider_revision") or 0
            ),
            "rerank_provider": (
                rerank_provider.public() if rerank_provider else None
            ),
            "rerank_binding": {
                "bound": bool(meta.get("rerank_provider_id")),
                "available": rerank_available,
                "fingerprint": rerank_fingerprint,
                "needs_recalibration": needs_recalibration,
            },
            "capabilities": list(descriptor.capabilities),
            "type_metadata": descriptor.public(),
            "stats": stats,
            "indexes": service.indexes.status(),
            "adapter_connections": adapter_connections,
            "adapter_busy": {
                "busy": active_job is not None,
                "job": active_job,
            },
            "ingest_defaults": {
                "chunk_target": int(meta["chunk_target"]),
                "chunk_overlap": int(meta["chunk_overlap"]),
            },
            "uniform_media_strength": float(meta["uniform_media_strength"]),
            "retrieval_settings": retrieval_settings,
            "visual_intent_policy": retrieval_settings[
                "visual_intent_policy"
            ],
            "visual_intent_policy_type_defaults": (
                normalize_visual_intent_policy(
                    DEFAULT_VISUAL_INTENT_POLICY
                )
            ),
            "visual_intent_policy_is_default": (
                service.visual_intent_policy_is_default
            ),
            "visual_intent_policy_fingerprint": (
                visual_intent_policy_fingerprint(
                    retrieval_settings["visual_intent_policy"]
                )
            ),
            "visual_intent_detector_version": (
                VISUAL_INTENT_DETECTOR_VERSION
            ),
            "protected_visual_blocker_terms": list(
                PROTECTED_VISUAL_BLOCKER_TERMS
            ),
            "created_at": meta["created_at"],
            "updated_at": meta["updated_at"],
        }

    async def task_database_state(
        self,
        knowledge_base_id: str,
        _task_kind: str,
    ) -> dict[str, Any]:
        """Capture stable text-media database and active-index metadata."""

        ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, knowledge_base_id)
        directory = database_type_registry.data_dir(self.data_dir, ref)
        if not (directory / "textmediaknowledge.db").exists():
            return {
                "database": {
                    "exists": False,
                    "database_type": TEXT_MEDIA_V1_TYPE,
                    "database_id": knowledge_base_id,
                }
            }
        service = self.runtimes.get(ref)
        if service is not None:
            snapshot = await service.storage.summary_snapshot()
            indexes = service.indexes.status()
        else:
            storage = TextMediaStorage(directory)
            try:
                snapshot = await storage.summary_snapshot()
            finally:
                await storage.close()
            indexes = await run_blocking(TextMediaIndex.disk_status, directory)
        meta = snapshot["metadata"]
        stats = snapshot["stats"]
        return {
            "database": {
                "exists": True,
                "database_type": TEXT_MEDIA_V1_TYPE,
                "database_id": knowledge_base_id,
                "name": meta.get("name"),
                "status": meta.get("status"),
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
            },
            "provider_binding": {
                "embedding_provider_id": meta.get("provider_id"),
                "embedding_provider_revision": meta.get("provider_revision"),
                "embedding_provider_fingerprint": meta.get("provider_fingerprint"),
                "rerank_provider_id": meta.get("rerank_provider_id") or None,
                "rerank_provider_revision": meta.get("rerank_provider_revision") or None,
                "rerank_provider_fingerprint": (
                    meta.get("rerank_provider_fingerprint") or None
                ),
            },
            "statistics": stats,
            "indexes": {
                "generation": indexes.get("generation"),
                "loaded": bool(indexes.get("loaded")),
                "vector_count": int(indexes.get("vector_count") or 0),
                "dimensions": int(indexes.get("dimensions") or 0),
                "media_generation": indexes.get("media_generation"),
                "media_loaded": bool(indexes.get("media_loaded")),
                "media_vector_count": int(
                    indexes.get("media_vector_count") or 0
                ),
                "media_dimensions": int(indexes.get("media_dimensions") or 0),
            },
        }

    async def list_libraries(self, *, stats_mode: str = "full") -> list[dict[str, Any]]:
        if stats_mode not in {"full", "summary"}:
            raise ValueError("invalid library stats mode")
        identities = [
            item
            for item in await self.control.list_database_identities()
            if item["database_type"] == TEXT_MEDIA_V1_TYPE
        ]
        if stats_mode == "full":
            details = await asyncio.gather(
                *(
                    self.library_detail(str(item["id"]))
                    for item in identities
                ),
                return_exceptions=True,
            )
            return [
                item
                for item in details
                if isinstance(item, dict)
            ]

        descriptor = database_type_registry.require(
            TEXT_MEDIA_V1_TYPE
        ).descriptor
        refs = [
            DatabaseRef(TEXT_MEDIA_V1_TYPE, str(item["id"]))
            for item in identities
        ]
        summary_limit = asyncio.Semaphore(4)

        async def read_snapshot(
            ref: DatabaseRef,
        ) -> tuple[DatabaseRef, dict[str, Any], dict[str, Any]] | None:
            directory = database_type_registry.data_dir(self.data_dir, ref)
            if not (directory / "textmediaknowledge.db").is_file():
                return None
            async with summary_limit:
                service = self.runtimes.get(ref)
                snapshot, disk_indexes = await self._summary_from_disk(
                    ref,
                    directory,
                )
                indexes = (
                    service.indexes.status()
                    if service is not None
                    else disk_indexes
                )
            return ref, snapshot, indexes

        snapshots = [
            item
            for item in await asyncio.gather(
                *(read_snapshot(ref) for ref in refs)
            )
            if item is not None
        ]
        provider_bindings: set[tuple[str, int | None]] = set()
        resource_keys: list[str] = []
        for ref, snapshot, _indexes in snapshots:
            meta = snapshot["metadata"]
            if meta.get("provider_id"):
                provider_bindings.add(
                    (
                        str(meta["provider_id"]),
                        int(meta.get("provider_revision") or 0),
                    )
                )
            if meta.get("rerank_provider_id"):
                provider_bindings.add(
                    (
                        str(meta["rerank_provider_id"]),
                        int(meta.get("rerank_provider_revision") or 0),
                    )
                )
            resource_keys.append(
                database_type_registry.require(
                    TEXT_MEDIA_V1_TYPE
                ).resource_key(ref.id)
            )
        provider_map, adapter_map, busy_map = await asyncio.gather(
            self.control.get_providers_bulk(provider_bindings),
            self.control.active_database_adapter_connections_map(
                [item[0] for item in snapshots]
            ),
            (
                self.jobs.active_long_jobs_map(resource_keys)
                if self.jobs is not None
                else self.control.active_long_jobs_map(resource_keys)
            ),
        )

        results: list[dict[str, Any]] = []
        for ref, snapshot, indexes in snapshots:
            meta = snapshot["metadata"]
            provider = provider_map.get(
                (
                    str(meta.get("provider_id") or ""),
                    int(meta.get("provider_revision") or 0),
                )
            )
            rerank_provider = provider_map.get(
                (
                    str(meta.get("rerank_provider_id") or ""),
                    int(meta.get("rerank_provider_revision") or 0),
                )
            )
            rerank_fingerprint = str(
                meta.get("rerank_provider_fingerprint") or ""
            )
            rerank_available = bool(
                rerank_provider is not None
                and rerank_provider.config.enabled
                and provider_kind(rerank_provider.config.type) == "rerank"
                and rerank_provider.config_sha256 == rerank_fingerprint
            )
            retrieval_settings = normalize_retrieval_config(
                meta.get("retrieval_config_json")
            )
            visual_policy = normalize_visual_intent_policy(
                retrieval_settings.get("visual_intent_policy")
            )
            retrieval_settings["visual_intent_policy"] = visual_policy
            resource_key = database_type_registry.require(
                TEXT_MEDIA_V1_TYPE
            ).resource_key(ref.id)
            active_job = busy_map.get(resource_key)
            results.append(
                {
                    "id": ref.id,
                    **database_identity_fields(ref),
                    "database_category": DATABASE_CATEGORY_KNOWLEDGE,
                    "name": meta["name"],
                    "description": meta["description"],
                    "is_default": False,
                    "status": meta["status"],
                    "provider_id": meta["provider_id"],
                    "provider_revision": meta["provider_revision"],
                    "provider": provider.public() if provider else None,
                    "rerank_provider_id": str(
                        meta.get("rerank_provider_id") or ""
                    ),
                    "rerank_provider_revision": int(
                        meta.get("rerank_provider_revision") or 0
                    ),
                    "rerank_provider": (
                        rerank_provider.public() if rerank_provider else None
                    ),
                    "rerank_binding": {
                        "bound": bool(meta.get("rerank_provider_id")),
                        "available": rerank_available,
                        "fingerprint": rerank_fingerprint,
                        "needs_recalibration": bool(
                            snapshot["needs_recalibration"]
                        ),
                    },
                    "capabilities": list(descriptor.capabilities),
                    "type_metadata": descriptor.public(),
                    "stats": snapshot["stats"],
                    "indexes": indexes,
                    "adapter_connections": adapter_map.get(ref.key, []),
                    "adapter_busy": {
                        "busy": active_job is not None,
                        "job": active_job,
                    },
                    "ingest_defaults": {
                        "chunk_target": int(meta["chunk_target"]),
                        "chunk_overlap": int(meta["chunk_overlap"]),
                    },
                    "uniform_media_strength": float(
                        meta["uniform_media_strength"]
                    ),
                    "retrieval_settings": retrieval_settings,
                    "visual_intent_policy": visual_policy,
                    "visual_intent_policy_type_defaults": (
                        normalize_visual_intent_policy(
                            DEFAULT_VISUAL_INTENT_POLICY
                        )
                    ),
                    "visual_intent_policy_is_default": (
                        visual_policy
                        == normalize_visual_intent_policy(
                            DEFAULT_VISUAL_INTENT_POLICY
                        )
                    ),
                    "visual_intent_policy_fingerprint": (
                        visual_intent_policy_fingerprint(visual_policy)
                    ),
                    "visual_intent_detector_version": (
                        VISUAL_INTENT_DETECTOR_VERSION
                    ),
                    "protected_visual_blocker_terms": list(
                        PROTECTED_VISUAL_BLOCKER_TERMS
                    ),
                    "created_at": meta["created_at"],
                    "updated_at": meta["updated_at"],
                }
            )
        return results

    async def provider_usage_map(
        self,
        provider_ids: set[str],
    ) -> dict[str, list[dict[str, Any]]]:
        targets = {
            str(provider_id) for provider_id in provider_ids if str(provider_id)
        }
        results: dict[str, list[dict[str, Any]]] = {
            provider_id: [] for provider_id in targets
        }
        if not targets:
            return results
        identities = await self.control.list_database_identities()
        driver = database_type_registry.require(TEXT_MEDIA_V1_TYPE)
        descriptor = driver.descriptor
        latest_providers = await self.control.get_providers_bulk(
            {(provider_id, None) for provider_id in targets}
        )
        for item in identities:
            if item["database_type"] != TEXT_MEDIA_V1_TYPE:
                continue
            database_id = str(item["id"])
            directory = database_type_registry.data_dir(
                self.data_dir, DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id)
            )
            if not (directory / "textmediaknowledge.db").exists():
                continue
            storage = TextMediaStorage(directory)
            calibrations: list[dict[str, Any]] = []
            try:
                meta = await storage.metadata()
                calibrations = await storage.list_document_media_calibrations()
            except (KeyError, FileNotFoundError):
                continue
            finally:
                await storage.close()
            common = {
                "library_id": database_id,
                "database_type": TEXT_MEDIA_V1_TYPE,
                "database_id": database_id,
                "database_category": DATABASE_CATEGORY_KNOWLEDGE,
                "library_name": str(meta.get("name") or database_id),
                "type_display_name": descriptor.display_name,
            }
            provider_id = str(meta.get("provider_id") or "")
            if provider_id in results:
                latest_provider = latest_providers.get((provider_id, None))
                provider_revision = int(meta.get("provider_revision") or 0)
                provider_fingerprint = str(
                    meta.get("provider_fingerprint") or ""
                )
                results[provider_id].append(
                    {
                        **common,
                        "provider_revision": provider_revision,
                        "usage_kind": "embedding",
                        "needs_rebuild": (
                            True
                            if latest_provider is None
                            else provider_fingerprint
                            != latest_provider.config_sha256
                        ),
                    }
                )
            rerank_provider_id = str(meta.get("rerank_provider_id") or "")
            if rerank_provider_id in results:
                latest_provider = latest_providers.get(
                    (rerank_provider_id, None)
                )
                rerank_revision = int(
                    meta.get("rerank_provider_revision") or 0
                )
                rerank_fingerprint = str(
                    meta.get("rerank_provider_fingerprint") or ""
                )
                calibration_settings_fingerprint = (
                    rerank_calibration_settings_fingerprint(
                        meta.get("retrieval_config_json")
                    )
                )
                results[rerank_provider_id].append(
                    {
                        **common,
                        "provider_revision": rerank_revision,
                        "usage_kind": "rerank",
                        "needs_rebuild": False,
                        "needs_recalibration": (
                            True
                            if latest_provider is None
                            else (
                                rerank_fingerprint
                                != latest_provider.config_sha256
                                or any(
                                    str(
                                        calibration.get(
                                            "calibration_rerank_provider_fingerprint"
                                        )
                                        or ""
                                    )
                                    != rerank_fingerprint
                                    or {
                                        value
                                        for value in str(
                                            calibration.get(
                                                "calibration_rerank_settings_fingerprints"
                                            )
                                            or ""
                                        ).split(",")
                                        if value
                                    }
                                    != {calibration_settings_fingerprint}
                                    for calibration in calibrations
                                    if str(
                                        calibration.get("semantic_mode") or ""
                                    )
                                    == "calibrated"
                                )
                            )
                        ),
                    }
                )
        if self.jobs:
            for active in await self.jobs.list(scope="active"):
                if (
                    str(active.get("database_type") or "")
                    != TEXT_MEDIA_V1_TYPE
                    or str(active.get("kind") or "")
                    != "text_media_index_rebuild"
                ):
                    continue
                internal = await self.jobs.get(
                    str(active["id"]), internal=True
                )
                operation = dict((internal or {}).get("operation") or {})
                provider_id = str(operation.get("provider_id") or "")
                if provider_id not in results:
                    continue
                database_id = str(
                    active.get("database_id")
                    or operation.get("database_id")
                    or ""
                )
                key = (database_id, "embedding_rebuild_target")
                if any(
                    (
                        str(item.get("database_id") or ""),
                        str(item.get("usage_kind") or ""),
                    )
                    == key
                    for item in results[provider_id]
                ):
                    continue
                results[provider_id].append(
                    {
                        "library_id": database_id,
                        "database_type": TEXT_MEDIA_V1_TYPE,
                        "database_id": database_id,
                        "database_category": DATABASE_CATEGORY_KNOWLEDGE,
                        "library_name": database_id,
                        "type_display_name": descriptor.display_name,
                        "provider_revision": int(
                            operation.get("provider_revision") or 0
                        ),
                        "usage_kind": "embedding_rebuild_target",
                        "needs_rebuild": False,
                        "active_task_id": str(active["id"]),
                    }
                )
        return results

    async def provider_usage(self, provider_id: str) -> list[dict[str, Any]]:
        return (await self.provider_usage_map({provider_id})).get(provider_id, [])

    async def debug_revision_bindings(self) -> list[dict[str, Any]]:
        identities = await self.control.list_database_identities()
        descriptor = database_type_registry.require(TEXT_MEDIA_V1_TYPE).descriptor
        results: list[dict[str, Any]] = []
        for identity in identities:
            if identity["database_type"] != TEXT_MEDIA_V1_TYPE:
                continue
            database_id = str(identity["id"])
            ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id)
            directory = database_type_registry.data_dir(self.data_dir, ref)
            if not (directory / "textmediaknowledge.db").exists():
                results.append(
                    {
                        "database_type": TEXT_MEDIA_V1_TYPE,
                        "database_category": DATABASE_CATEGORY_KNOWLEDGE,
                        "database_id": database_id,
                        "database_name": database_id,
                        "type_display_name": descriptor.display_name,
                        "missing_storage": True,
                        "bindings": [],
                    }
                )
                continue
            runtime = self.runtimes.get(ref)
            storage = runtime.storage if runtime is not None else TextMediaStorage(directory)
            try:
                state = await storage.debug_revision_state()
            finally:
                if runtime is None:
                    await storage.close()
            meta = state["metadata"]
            results.append(
                {
                    "database_type": TEXT_MEDIA_V1_TYPE,
                    "database_category": DATABASE_CATEGORY_KNOWLEDGE,
                    "database_id": database_id,
                    "database_name": str(meta.get("name") or database_id),
                    "type_display_name": descriptor.display_name,
                    "status": str(meta.get("status") or ""),
                    "created_at": float(meta.get("created_at") or 0),
                    "updated_at": float(meta.get("updated_at") or 0),
                    "embedding_generations": state["embedding_generations"],
                    "media_embeddings": state["media_embeddings"],
                    "relation_calibrations": state["relation_calibrations"],
                    "strength_calibrations": state["strength_calibrations"],
                    "bindings": [
                        {
                            "usage_kind": "embedding",
                            "binding_mode": "pinned",
                            "provider_id": str(meta.get("provider_id") or ""),
                            "provider_revision": int(
                                meta.get("provider_revision") or 0
                            ),
                            "provider_fingerprint": str(
                                meta.get("provider_fingerprint") or ""
                            ),
                        },
                        *(
                            [
                                {
                                    "usage_kind": "rerank",
                                    "binding_mode": "pinned",
                                    "provider_id": str(
                                        meta.get("rerank_provider_id") or ""
                                    ),
                                    "provider_revision": int(
                                        meta.get("rerank_provider_revision") or 0
                                    ),
                                    "provider_fingerprint": str(
                                        meta.get("rerank_provider_fingerprint")
                                        or ""
                                    ),
                                }
                            ]
                            if meta.get("rerank_provider_id")
                            else []
                        ),
                    ],
                }
            )
        return results

    async def debug_rebind_revision(
        self,
        database_id: str,
        *,
        usage_kind: str,
        provider_id: str,
        revision: int,
        fingerprint: str,
    ) -> None:
        ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id)
        identity = await self.control.database_identity(ref)
        if identity is None or identity.get("deleted_at") is not None:
            raise KeyError(database_id)
        await self.unload_runtime(ref)
        directory = database_type_registry.data_dir(self.data_dir, ref)
        storage = TextMediaStorage(directory)
        try:
            await storage.debug_rebind_revision(
                usage_kind=usage_kind,
                provider_id=provider_id,
                revision=revision,
                fingerprint=fingerprint,
            )
        finally:
            await storage.close()

    async def copy_library(self, database_id: str, progress=None) -> dict[str, Any]:
        source_ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id)
        if not await self.control.database_identity(source_ref):
            raise KeyError(database_id)
        source = await self.get_runtime(source_ref)
        metadata = await source.storage.metadata()
        if progress:
            await progress(0.05, "正在准备知识库副本 ID")
        target_id, target_name = await self._next_copy_identity(
            database_id, str(metadata.get("name") or database_id)
        )
        type_root = database_type_registry.type_root(
            self.data_dir, TEXT_MEDIA_V1_TYPE
        )
        type_root.mkdir(parents=True, exist_ok=True)
        if progress:
            await progress(0.12, f"副本目标已确定：{target_id}")
        with tempfile.TemporaryDirectory(
            prefix=f".{target_id}.copying-", dir=type_root
        ) as raw:
            package = Path(raw) / f"{database_id}.tmkb"
            if progress:
                await progress(0.2, "正在创建一致性 TMKB 快照")
            await export_tmkb(service=source, target=package)
            if progress:
                await progress(0.68, "正在校验并安装知识库副本")
            result = await install_tmkb(
                manager=self,
                package_path=package,
                target_id=target_id,
                name_override=target_name,
            )
        if progress:
            await progress(1.0, f"复制完成：{target_id}")
        return result

    async def _next_copy_identity(
        self, source_id: str, source_name: str
    ) -> tuple[str, str]:
        for index in range(1, 1000):
            target_id = (
                f"{source_id}_copy" if index == 1 else f"{source_id}_copy{index}"
            )
            target_name = (
                f"{source_name}(副本)"
                if index == 1
                else f"{source_name}(副本{index})"
            )
            if not await self._copy_target_reserved(target_id):
                return target_id, target_name
        raise ValueError("无法生成可用的知识库副本 ID，请先清理过多副本")

    async def _copy_target_reserved(self, database_id: str) -> bool:
        ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id)
        if await self.control.database_identity(ref):
            return True
        target = database_type_registry.data_dir(
            self.data_dir, DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id)
        )
        if target.exists():
            return True
        type_root = target.parent
        if any(type_root.glob(f".{database_id}.copying-*")) or any(
            type_root.glob(f".{database_id}.import-*")
        ):
            return True
        trash = self._trash_root()
        return trash.exists() and any(trash.glob(f"{database_id}-*"))

    async def backup_library(self, database_id: str) -> dict[str, Any]:
        ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id)
        if not await self.control.database_identity(ref):
            raise KeyError(database_id)
        service = await self.get_runtime(ref)
        backup_root = service.root / "backups"
        backup_root.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        nanoseconds = time.time_ns() % 1_000_000_000
        target = backup_root / f"{timestamp}-{nanoseconds:09d}.tmkb"
        result = await export_tmkb(service=service, target=target)
        return {
            "database_type": TEXT_MEDIA_V1_TYPE,
            "database_id": database_id,
            "path": str(target),
            "filename": target.name,
            "size_bytes": result["size_bytes"],
            "sha256": result["sha256"],
        }

    async def update_visual_intent_policy(
        self,
        database_id: str,
        policy: object,
    ) -> dict[str, Any]:
        service = await self.get_runtime(
            DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id)
        )
        return await service.update_visual_intent_policy(policy)

    async def update_library(self, database_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        current_ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id)
        next_id = validate_identifier(str(payload.get("id") or database_id), field="知识库 ID")
        next_ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, next_id)
        rename_requested = next_ref != current_ref
        if self.jobs:
            active = await self.jobs.active_database_job(
                current_ref,
                include_read_only=rename_requested,
            )
            if active is not None:
                raise ValueError(
                    "knowledge library has an active task and cannot be edited"
                )
        values = {
            key: str(payload[key]).strip()
            for key in ("name", "description")
            if key in payload and payload[key] is not None
        }
        if (
            "uniform_media_strength" in payload
            and payload["uniform_media_strength"] is not None
        ):
            strength = float(payload["uniform_media_strength"])
            if not 0 <= strength <= 1:
                raise ValueError("uniform media strength must be between 0 and 1")
            values["uniform_media_strength"] = strength
        if "retrieval_settings" in payload and payload["retrieval_settings"] is not None:
            current_service = await self.get_runtime(current_ref)
            current_meta = await current_service.storage.metadata()
            merged_settings = normalize_retrieval_config(
                current_meta.get("retrieval_config_json")
            )
            merged_settings.update(dict(payload["retrieval_settings"]))
            normalized_settings = normalize_retrieval_config(
                merged_settings
            )
            normalized_settings.pop("visual_intent_policy", None)
            values["retrieval_config_json"] = normalized_settings
        rerank_revision = None
        rerank_binding_changed = "rerank_provider_id" in payload
        if rerank_binding_changed:
            rerank_provider_id = str(
                payload.get("rerank_provider_id") or ""
            ).strip()
            if rerank_provider_id:
                rerank_provider_id = validate_identifier(
                    rerank_provider_id, field="Rerank Provider ID"
                )
                rerank_revision = await self.control.get_provider(
                    rerank_provider_id
                )
                if (
                    rerank_revision is None
                    or provider_kind(rerank_revision.config.type) != "rerank"
                ):
                    raise ValueError("指定的 Provider 不是 Rerank Provider")
                if not rerank_revision.config.enabled:
                    raise ValueError("指定的 Rerank Provider 未启用")
                values.update(
                    {
                        "rerank_provider_id": rerank_revision.provider_id,
                        "rerank_provider_revision": rerank_revision.revision,
                        "rerank_provider_fingerprint": (
                            rerank_revision.config_sha256
                        ),
                    }
                )
            else:
                values.update(
                    {
                        "rerank_provider_id": "",
                        "rerank_provider_revision": 0,
                        "rerank_provider_fingerprint": "",
                    }
                )
        if "name" in values and not values["name"]:
            raise ValueError("知识库名称不能为空")
        if not rename_requested:
            service = await self.get_runtime(current_ref)
            await service.storage.update_metadata(values)
            if rerank_binding_changed:
                await service.set_rerank_provider(
                    (
                        build_rerank_provider(rerank_revision.config)
                        if rerank_revision is not None
                        else None
                    ),
                    (
                        {
                            "id": rerank_revision.provider_id,
                            "revision": rerank_revision.revision,
                            "fingerprint": rerank_revision.config_sha256,
                        }
                        if rerank_revision is not None
                        else {}
                    ),
                )
            return await self.library_detail(database_id)

        if await self.control.active_adapter_connections(current_ref):
            raise ValueError("知识库已被适配器连接，不能修改 ID")
        # Keep the established adapter-blocking guard as a compatibility
        # fallback for callers/tests that provide only the legacy job probe.
        if self.jobs and await self.jobs.active_long_job(current_ref):
            raise ValueError("知识库存在进行中的任务，暂时不能修改 ID")
        source = database_type_registry.data_dir(self.data_dir, current_ref)
        target = database_type_registry.data_dir(self.data_dir, next_ref)
        if await self.control.database_identity(next_ref) or target.exists():
            raise ValueError(f"知识库 ID 已存在：{next_id}")
        if not source.exists():
            raise ValueError(f"知识库目录不存在，无法修改 ID：{source}")

        await self.unload_runtime(current_ref)
        moved = False
        storage: TextMediaStorage | None = None
        try:
            source.replace(target)
            moved = True
            storage = TextMediaStorage(target)
            await storage.update_metadata({**values, "database_id": next_id})
            await storage.close()
            storage = None
            await self.control.rename_database_identity(current_ref, next_id)
        except Exception:
            if storage is not None:
                await storage.close()
            if moved and target.exists():
                rollback = TextMediaStorage(target)
                try:
                    await rollback.update_metadata({"database_id": database_id})
                finally:
                    await rollback.close()
                if not source.exists():
                    target.replace(source)
            raise
        return await self.library_detail(next_id)

    async def delete_library(self, database_id: str) -> dict[str, Any]:
        ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id)
        if await self.control.active_adapter_connections(ref):
            raise ValueError("知识库已被适配器连接，不能删除")
        if self.jobs and await self.jobs.active_database_job(ref):
            raise ValueError("知识库存在运行中的任务")
        await self.unload_runtime(ref)
        directory = database_type_registry.data_dir(self.data_dir, ref)
        if not directory.exists():
            raise KeyError(database_id)
        trash = self._trash_root() / f"{database_id}-{time.strftime('%Y%m%d-%H%M%S')}"
        trash.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(directory), str(trash))
        await self.control.delete_database_identity(ref)
        return {"id": database_id, "database_type": TEXT_MEDIA_V1_TYPE, "trash": str(trash)}

    async def library_is_empty(self, database_id: str) -> bool:
        service = await self.get_runtime(database_id)
        return (await service.storage.statistics())["documents"] == 0
