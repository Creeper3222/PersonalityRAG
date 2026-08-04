from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import json
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, AsyncIterator, Awaitable, Callable

from .logger import logger, safe_summary
from .database_types import (
    LIVINGMEMORY_V8_TYPE,
    DatabaseRef,
    database_identity_fields,
    database_type_registry,
)
from .resource_quotas import resource_lane
from . import library_types as _registered_database_types  # noqa: F401
from .task_control import (
    JobExecutionContext,
    JobInterrupted,
    JobPauseRequested,
    JobStopRequested,
    ResolvedJobOperation,
)
from .task_types import (
    ACTIVE_JOB_STATUSES,
    TERMINAL_JOB_STATUSES,
    task_type_registry,
)
from .task_details import (
    build_database_state_comparison,
    sanitize_task_value,
    task_request_summary,
)


ACTIVE_STATUSES = frozenset(ACTIVE_JOB_STATUSES)
TERMINAL_STATUSES = frozenset(TERMINAL_JOB_STATUSES)
PAUSED_STATUSES = frozenset({"paused", "interrupted"})
JOB_HISTORY_DAYS = 30
JOB_HISTORY_LIMIT = 1000
PROGRESS_PERSIST_INTERVAL_SECONDS = 1.0
SUBSCRIBER_QUEUE_SIZE = 1
JOB_SUMMARY_COLUMNS = """id,library_id,database_type,database_id,kind,status,
progress,message,error,
CASE WHEN checkpoint IS NULL THEN NULL ELSE json_object(
  'phase',json_extract(checkpoint,'$.phase'),
  'completed_documents',json_extract(checkpoint,'$.completed_documents'),
  'completed_graph_entries',json_extract(checkpoint,'$.completed_graph_entries'),
  'total_documents',json_extract(checkpoint,'$.total_documents'),
  'total_graph_entries',json_extract(checkpoint,'$.total_graph_entries'),
  'saved_at',json_extract(checkpoint,'$.saved_at')
) END AS checkpoint,
status_reason,control_requested,resumable,started_at,finished_at,attempt_count,
created_at,updated_at"""

ProgressCallback = Callable[[float, str], Awaitable[None]]
JobOperation = Callable[[ProgressCallback], Awaitable[Any]]
OperationResolver = Callable[
    [dict[str, Any]],
    ResolvedJobOperation | Awaitable[ResolvedJobOperation],
]
DatabaseStateProvider = Callable[
    [str, str],
    dict[str, Any] | Awaitable[dict[str, Any]],
]


class JobStateConflict(RuntimeError):
    pass


class JobManager:
    def __init__(
        self,
        storage: Any,
        runtime_lease_factory: Callable[[str], AsyncContextManager[Any]] | None = None,
    ):
        self.storage = storage
        self._runtime_lease_factory = runtime_lease_factory
        self._resolvers: dict[str, OperationResolver] = {}
        self._database_state_providers: dict[str, DatabaseStateProvider] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._operations: dict[str, JobOperation] = {}
        self._runtime_lease_jobs: set[str] = set()
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._completion: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._execution_done: dict[str, asyncio.Event] = {}
        self._control_events: dict[str, asyncio.Event] = {}
        self._state_events: dict[str, asyncio.Event] = {}
        self._transition_locks: dict[str, asyncio.Lock] = {}
        self._control_requests: dict[str, str] = {}
        self._last_persisted_at: dict[str, float] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._simple_tasks: dict[str, asyncio.Task[None]] = {}
        self._database_execution_locks: dict[str, asyncio.Lock] = {}
        self._current_job_id: str | None = None
        self._start_lock = asyncio.Lock()
        self._closing = False
        self._closed = False
        self.subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)

    def set_operation_resolver(
        self,
        resolver: OperationResolver,
        *,
        database_type: str = LIVINGMEMORY_V8_TYPE,
    ) -> None:
        self._resolvers[database_type] = resolver

    def set_database_state_provider(
        self,
        provider: DatabaseStateProvider,
        *,
        database_type: str = LIVINGMEMORY_V8_TYPE,
    ) -> None:
        self._database_state_providers[database_type] = provider

    @asynccontextmanager
    async def _connect(self):
        try:
            connection = self.storage.connect(system=True)
        except TypeError:
            db = await self.storage.connect()
            try:
                yield db
            finally:
                await db.close()
            return
        async with connection as db:
            yield db

    def control_request(self, job_id: str) -> str:
        return self._control_requests.get(job_id, "")

    def _signal_control(self, job_id: str) -> None:
        self._control_events.setdefault(job_id, asyncio.Event()).set()

    def _signal_state(self, job_id: str) -> None:
        event = self._state_events.setdefault(job_id, asyncio.Event())
        event.set()

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def recover_for_startup(self) -> None:
        """Recover durable task history without automatically resuming work."""
        self._closed = False
        self._closing = False
        self._operations.clear()
        self._runtime_lease_jobs.clear()
        self._snapshots.clear()
        self._completion.clear()
        self._execution_done.clear()
        self._control_events.clear()
        self._state_events.clear()
        self._transition_locks.clear()
        self._control_requests.clear()
        self._last_persisted_at.clear()
        self._simple_tasks.clear()
        self._database_execution_locks.clear()
        self.subscribers.clear()
        self._drain_queue()

        now = time.time()
        cutoff = now - JOB_HISTORY_DAYS * 24 * 60 * 60
        async with self._connect() as db:
            queued_rows = await (
                await db.execute("SELECT * FROM jobs WHERE status='queued'")
            ).fetchall()
        for row in queued_rows:
            queued_job = self._decode_row(row)
            if not queued_job.get("resumable") or not self._resolvers:
                continue
            try:
                operation = await self._resolve(queued_job)
                if operation.cancel_queued:
                    job_id = str(queued_job["id"])
                    self._snapshots[job_id] = queued_job
                    await operation.cancel_queued(JobExecutionContext(self, job_id))
                    self._snapshots.pop(job_id, None)
            except Exception:
                logger.exception(
                    "启动恢复时清理排队任务资源失败：job_id=%s", queued_job.get("id")
                )
        async with self._connect() as db:
            in_flight_rows = await (
                await db.execute(
                    "SELECT * FROM jobs WHERE status IN ('running','pausing','stopping')"
                )
            ).fetchall()
            queued = await db.execute(
                """UPDATE jobs SET status='cancelled',progress=0,
                message='服务重启，排队任务已取消',status_reason='shutdown',
                control_requested=NULL,finished_at=COALESCE(finished_at,?),updated_at=?
                WHERE status='queued'""",
                (now, now),
            )
            interrupted = await db.execute(
                """UPDATE jobs SET status=CASE WHEN resumable=1 THEN 'interrupted'
                ELSE 'cancelled' END,
                message=CASE WHEN resumable=1 THEN '检测到异常退出，已回退到最近安全断点'
                ELSE '服务异常退出，任务已取消' END,
                status_reason='process_interrupted',control_requested=NULL,
                finished_at=CASE WHEN resumable=1 THEN finished_at ELSE COALESCE(finished_at,?) END,
                updated_at=?
                WHERE status IN ('running','pausing','stopping')""",
                (now, now),
            )
            for row in queued_rows:
                payload = self._decode_row(row)
                self._append_status_history(
                    payload,
                    status="cancelled",
                    timestamp=now,
                    message="服务重启，排队任务已取消",
                    reason="shutdown",
                )
                await db.execute(
                    "UPDATE jobs SET status_history=? WHERE id=?",
                    (
                        json.dumps(
                            payload.get("status_history") or [],
                            ensure_ascii=False,
                        ),
                        payload["id"],
                    ),
                )
            for row in in_flight_rows:
                payload = self._decode_row(row)
                resumable = bool(payload.get("resumable"))
                self._append_status_history(
                    payload,
                    status="interrupted" if resumable else "cancelled",
                    timestamp=now,
                    message=(
                        "检测到异常退出，已回退到最近安全断点"
                        if resumable
                        else "服务异常退出，任务已取消"
                    ),
                    reason="process_interrupted",
                )
                await db.execute(
                    "UPDATE jobs SET status_history=? WHERE id=?",
                    (
                        json.dumps(
                            payload.get("status_history") or [],
                            ensure_ascii=False,
                        ),
                        payload["id"],
                    ),
                )
            expired = await db.execute(
                """DELETE FROM jobs WHERE status IN
                ('completed','failed','stopped','cancelled') AND updated_at<?""",
                (cutoff,),
            )
            overflow = await db.execute(
                """DELETE FROM jobs WHERE id IN (
                    SELECT id FROM jobs WHERE status IN
                    ('completed','failed','stopped','cancelled')
                    ORDER BY updated_at DESC LIMIT -1 OFFSET ?
                )""",
                (JOB_HISTORY_LIMIT,),
            )
            await db.commit()
            paused_rows = await (
                await db.execute(
                    """SELECT * FROM jobs WHERE status IN ('paused','interrupted')
                    AND resumable=1 ORDER BY created_at"""
                )
            ).fetchall()

        for row in paused_rows:
            payload = self._decode_row(row)
            job_id = str(payload["id"])
            self._snapshots[job_id] = payload
            self._completion[job_id] = asyncio.get_running_loop().create_future()
            self._execution_done[job_id] = asyncio.Event()
            self._control_events[job_id] = asyncio.Event()
            self._state_events[job_id] = asyncio.Event()
            self._transition_locks[job_id] = asyncio.Lock()
            await self._queue.put(job_id)
        if paused_rows:
            self._ensure_worker()
        logger.info(
            "任务历史已恢复：queued_cancelled=%s interrupted=%s expired=%s overflow=%s paused=%s",
            max(0, int(queued.rowcount or 0)),
            max(0, int(interrupted.rowcount or 0)),
            max(0, int(expired.rowcount or 0)),
            max(0, int(overflow.rowcount or 0)),
            len(paused_rows),
        )

    async def clear_for_startup(self) -> None:
        """Compatibility alias for the durable startup recovery workflow."""
        await self.recover_for_startup()

    async def close(self) -> None:
        if self._closed:
            return
        self._closing = True
        current_id = self._current_job_id
        if current_id:
            current = await self.get(current_id)
            if current and current.get("resumable") and current.get("status") == "stopping":
                deadline = time.monotonic() + 90.0
                while time.monotonic() < deadline:
                    latest = await self.get(current_id)
                    if not latest or latest.get("status") in TERMINAL_STATUSES | PAUSED_STATUSES:
                        break
                    event = self._state_events.setdefault(current_id, asyncio.Event())
                    event.clear()
                    try:
                        await asyncio.wait_for(event.wait(), timeout=1.0)
                    except TimeoutError:
                        pass
                current = await self.get(current_id)
            if current and current.get("resumable") and current.get("status") in {
                "running",
                "pausing",
            }:
                self._control_requests[current_id] = "shutdown"
                await self._update(
                    current_id,
                    status="pausing",
                    status_reason="shutdown",
                    control_requested="shutdown",
                    message="服务正在关闭，正在保存安全断点",
                )
                self._signal_control(current_id)
                deadline = time.monotonic() + 90.0
                while time.monotonic() < deadline:
                    latest = await self.get(current_id)
                    if not latest or latest.get("status") in PAUSED_STATUSES | TERMINAL_STATUSES:
                        break
                    event = self._state_events.setdefault(current_id, asyncio.Event())
                    event.clear()
                    try:
                        await asyncio.wait_for(event.wait(), timeout=1.0)
                    except TimeoutError:
                        pass

        active = await self.list(scope="active")
        for job in active:
            if job.get("status") == "queued":
                try:
                    await self.cancel(str(job["id"]), reason="shutdown")
                except JobStateConflict:
                    pass

        self._closed = True
        if self._worker_task:
            self._worker_task.cancel()
            await asyncio.gather(self._worker_task, return_exceptions=True)
            self._worker_task = None
        simple_tasks = list(self._simple_tasks.values())
        for task in simple_tasks:
            task.cancel()
        if simple_tasks:
            await asyncio.gather(*simple_tasks, return_exceptions=True)
        self._simple_tasks.clear()
        self._operations.clear()
        self._runtime_lease_jobs.clear()
        self._drain_queue()
        for event in self._execution_done.values():
            event.set()
        self.subscribers.clear()
        logger.info("任务管理器已安全关闭")

    async def start(
        self,
        kind: str,
        operation: JobOperation,
        *,
        database_id: str | None = None,
        library_id: str | None = None,
        database_type: str = LIVINGMEMORY_V8_TYPE,
        dedupe_active: bool = True,
        lease_runtime: bool = True,
    ) -> str:
        resolved_database_id = self._resolve_start_database_id(
            database_id,
            library_id,
        )
        return await self._start_common(
            kind,
            operation=operation,
            operation_spec=None,
            resumable=False,
            database_id=resolved_database_id,
            database_type=database_type,
            dedupe_active=dedupe_active,
            lease_runtime=lease_runtime,
        )

    async def start_resumable(
        self,
        kind: str,
        operation_spec: dict[str, Any],
        *,
        database_id: str | None = None,
        library_id: str | None = None,
        database_type: str = LIVINGMEMORY_V8_TYPE,
        dedupe_active: bool = True,
        lease_runtime: bool = False,
    ) -> str:
        if not self._resolvers:
            raise RuntimeError("resumable job resolver is not configured")
        resolved_database_id = self._resolve_start_database_id(
            database_id,
            library_id,
        )
        return await self._start_common(
            kind,
            operation=None,
            operation_spec=operation_spec,
            resumable=True,
            database_id=resolved_database_id,
            database_type=database_type,
            dedupe_active=dedupe_active,
            lease_runtime=lease_runtime,
        )

    @staticmethod
    def _resolve_start_database_id(
        database_id: str | None,
        deprecated_library_id: str | None,
    ) -> str | None:
        """Resolve the frozen ``library_id`` call alias at one boundary."""

        if (
            database_id not in (None, "")
            and deprecated_library_id not in (None, "")
            and database_id != deprecated_library_id
        ):
            raise ValueError("database_id conflicts with deprecated library_id")
        return (
            database_id
            if database_id not in (None, "")
            else deprecated_library_id
        )

    async def _start_common(
        self,
        kind: str,
        *,
        operation: JobOperation | None,
        operation_spec: dict[str, Any] | None,
        resumable: bool,
        database_id: str | None,
        database_type: str,
        dedupe_active: bool,
        lease_runtime: bool,
    ) -> str:
        resource_key = database_id
        if database_id is not None:
            driver = database_type_registry.require(database_type)
            resource_key = driver.resource_key(database_id)
        if self._closed or self._closing:
            raise RuntimeError("任务管理器已关闭")
        async with self._start_lock:
            if dedupe_active:
                existing = await self._find_active(kind, resource_key)
                if existing:
                    return existing
            job_id = uuid.uuid4().hex
            now = time.time()
            snapshot = {
                "id": job_id,
                "database_resource_key": resource_key,
                "database_type": database_type,
                "database_id": database_id,
                "kind": kind,
                "status": "queued",
                "progress": 0.0,
                "message": "等待执行",
                "result": None,
                "error": None,
                "operation": operation_spec,
                "checkpoint": None,
                "status_reason": "",
                "control_requested": "",
                "resumable": bool(resumable),
                "started_at": None,
                "finished_at": None,
                "attempt_count": 0,
                "status_history": [
                    {
                        "status": "queued",
                        "timestamp": now,
                        "message": "等待执行",
                        "reason": "",
                    }
                ],
                "database_state_before": None,
                "database_state_after": None,
                "database_state_capture_error": {},
                "created_at": now,
                "updated_at": now,
            }
            async with self._connect() as db:
                await db.execute(
                    """INSERT INTO jobs
                    (id,library_id,database_type,database_id,kind,status,progress,message,result,error,
                    operation,checkpoint,status_reason,control_requested,resumable,
                    started_at,finished_at,attempt_count,status_history,
                    database_state_before,database_state_after,database_state_capture_error,
                    created_at,updated_at)
                    VALUES(?,?,?,?,?,'queued',0,'等待执行',NULL,NULL,?,NULL,'',NULL,?,
                    NULL,NULL,0,?,NULL,NULL,?, ?,?)""",
                    (
                        job_id,
                        resource_key,
                        database_type,
                        database_id,
                        kind,
                        json.dumps(operation_spec, ensure_ascii=False)
                        if operation_spec is not None
                        else None,
                        1 if resumable else 0,
                        json.dumps(snapshot["status_history"], ensure_ascii=False),
                        json.dumps({}, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                await db.commit()
            self._snapshots[job_id] = snapshot
            self._last_persisted_at[job_id] = now
            self._completion[job_id] = asyncio.get_running_loop().create_future()
            self._execution_done[job_id] = asyncio.Event()
            self._control_events[job_id] = asyncio.Event()
            self._state_events[job_id] = asyncio.Event()
            self._transition_locks[job_id] = asyncio.Lock()
            if operation is not None:
                self._operations[job_id] = operation
            if lease_runtime and resource_key and self._runtime_lease_factory is not None:
                self._runtime_lease_jobs.add(job_id)
            definition = task_type_registry.get(database_type, kind)
            if resumable or definition.lane == "long":
                await self._queue.put(job_id)
                self._ensure_worker()
            else:
                self._ensure_simple_task(job_id)
        logger.info(
            "任务已入队：job_id=%s kind=%s database_type=%s database_id=%s",
            job_id,
            kind,
            database_type if database_id else "",
            database_id or "",
        )
        return job_id

    async def _find_active(self, kind: str, resource_key: str | None) -> str | None:
        for snapshot in sorted(
            self._snapshots.values(), key=lambda item: float(item.get("created_at") or 0)
        ):
            if (
                snapshot.get("kind") == kind
                and snapshot.get("database_resource_key") == resource_key
                and snapshot.get("status") in ACTIVE_STATUSES
            ):
                return str(snapshot["id"])
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        params: list[Any] = [kind]
        library_clause = "library_id IS NULL"
        if resource_key is not None:
            library_clause = "library_id=?"
            params.append(resource_key)
        params.extend(sorted(ACTIVE_STATUSES))
        async with self._connect() as db:
            row = await (
                await db.execute(
                    f"""SELECT id FROM jobs WHERE kind=? AND {library_clause}
                    AND status IN ({placeholders}) ORDER BY created_at LIMIT 1""",
                    tuple(params),
                )
            ).fetchone()
        return str(row["id"]) if row else None

    async def active_job_id(
        self,
        kind: str,
        database_resource_key: str | None = None,
    ) -> str | None:
        return await self._find_active(kind, database_resource_key)

    @staticmethod
    def _busy_resource_key(
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
        resource_key = self._busy_resource_key(database, database_type)
        return (await self.active_long_jobs_map([resource_key])).get(resource_key)

    async def active_database_job(
        self,
        database: str | DatabaseRef,
        *,
        database_type: str | None = None,
        include_read_only: bool = True,
    ) -> dict[str, Any] | None:
        """Return the oldest active task for one typed database resource."""
        resource_key = self._busy_resource_key(database, database_type)
        for item in await self.list(scope="active"):
            if str(item.get("database_resource_key") or "") != resource_key:
                continue
            definition = task_type_registry.get(
                str(item.get("database_type") or LIVINGMEMORY_V8_TYPE),
                str(item.get("kind") or ""),
            )
            if include_read_only or not definition.read_only:
                return item
        return None

    async def active_long_jobs_map(
        self, database_resource_keys: list[str]
    ) -> dict[str, dict[str, Any] | None]:
        result: dict[str, dict[str, Any] | None] = {
            value: None for value in database_resource_keys
        }
        if not database_resource_keys:
            return result
        all_jobs = await self.list(scope="active")
        for item in all_jobs:
            resource_key = str(item.get("database_resource_key") or "")
            if (
                resource_key in result
                and result[resource_key] is None
                and task_type_registry.get(
                    str(item.get("database_type") or LIVINGMEMORY_V8_TYPE),
                    str(item.get("kind") or ""),
                ).adapter_blocking
            ):
                result[resource_key] = item
        return result

    def _ensure_worker(self) -> None:
        if self._worker_task and not self._worker_task.done():
            return
        self._worker_task = asyncio.create_task(self._worker())

    def _ensure_simple_task(self, job_id: str) -> None:
        current = self._simple_tasks.get(job_id)
        if current is not None and not current.done():
            return
        self._simple_tasks[job_id] = asyncio.create_task(
            self._run_simple_scheduled(job_id),
            name=f"personalityrag-job-{job_id}",
        )

    def _execution_lock(self, database_resource_key: str | None) -> asyncio.Lock:
        key = str(database_resource_key or "__global__")
        return self._database_execution_locks.setdefault(key, asyncio.Lock())

    async def _run_simple_scheduled(self, job_id: str) -> None:
        try:
            job = await self.get(job_id)
            if not job:
                return
            async with self._execution_lock(job.get("database_resource_key")):
                job = await self.get(job_id)
                if not job or job.get("status") in TERMINAL_STATUSES:
                    self._operations.pop(job_id, None)
                    return
                operation = self._operations.pop(job_id, None)
                if operation is None:
                    await self._update(
                        job_id,
                        status="cancelled",
                        progress=0,
                        status_reason="operation_missing",
                        message="任务操作已丢失",
                    )
                    return
                await self._run_simple(job_id, operation)
        except asyncio.CancelledError:
            job = await self.get(job_id)
            if job and job.get("status") not in TERMINAL_STATUSES:
                await self._update(
                    job_id,
                    status="cancelled",
                    progress=0,
                    status_reason="shutdown",
                    message="服务关闭，任务已取消",
                )
            raise
        except Exception as exc:
            logger.exception("普通任务 worker 异常：job_id=%s", job_id)
            await self._best_effort_terminal_update(job_id, exc)
        finally:
            self._runtime_lease_jobs.discard(job_id)
            self._execution_done.setdefault(job_id, asyncio.Event()).set()
            self._simple_tasks.pop(job_id, None)

    async def _resolve(self, job: dict[str, Any]) -> ResolvedJobOperation:
        database_type = str(job.get("database_type") or LIVINGMEMORY_V8_TYPE)
        resolver = self._resolvers.get(database_type)
        if resolver is None:
            raise RuntimeError("resumable job resolver is not configured")
        value = resolver(job)
        if inspect.isawaitable(value):
            value = await value
        return value

    async def _worker(self) -> None:
        logger.info("任务队列 worker 已启动")
        while not self._closed:
            if self._closing:
                return
            try:
                job_id = await self._queue.get()
            except asyncio.CancelledError:
                return
            self._current_job_id = job_id
            try:
                job = await self.get(job_id)
                if not job or job.get("status") in TERMINAL_STATUSES:
                    continue
                async with self._execution_lock(job.get("database_resource_key")):
                    if job.get("resumable"):
                        await self._run_resumable(job_id)
                    else:
                        operation = self._operations.pop(job_id, None)
                        if operation is None:
                            await self._update(
                                job_id,
                                status="cancelled",
                                progress=0,
                                status_reason="operation_missing",
                                message="任务操作已丢失",
                            )
                        else:
                            await self._run_simple(job_id, operation)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("任务 worker 异常：job_id=%s", job_id)
                await self._best_effort_terminal_update(job_id, exc)
            finally:
                self._current_job_id = None
                self._runtime_lease_jobs.discard(job_id)
                self._execution_done.setdefault(job_id, asyncio.Event()).set()
                self._queue.task_done()

    async def _run_simple(self, job_id: str, operation: JobOperation) -> None:
        await self._update(job_id, status="running", message="任务已启动")

        async def progress(value: float, message: str) -> None:
            await self._update(job_id, status="running", progress=value, message=message)

        async def execute() -> None:
            try:
                with resource_lane("task"):
                    result = await operation(progress)
                await self._update(
                    job_id,
                    status="completed",
                    progress=1,
                    message="任务完成",
                    result=result,
                    error=None,
                )
            except asyncio.CancelledError:
                await self._update(
                    job_id,
                    status="cancelled",
                    progress=0,
                    status_reason="shutdown",
                    message="服务关闭，任务已取消",
                )
                raise
            except Exception as exc:
                logger.exception("任务失败：job_id=%s", job_id)
                await self._update(
                    job_id, status="failed", message="任务失败", error=str(exc)
                )

        job = await self.get(job_id) or {}
        database_resource_key = str(job.get("database_resource_key") or "")
        if (
            job_id in self._runtime_lease_jobs
            and database_resource_key
            and self._runtime_lease_factory
        ):
            database_id = str(job.get("database_id") or "")
            database_type = str(
                job.get("database_type") or LIVINGMEMORY_V8_TYPE
            )
            target = (
                DatabaseRef(database_type, database_id)
                if database_id
                else database_resource_key
            )
            async with self._runtime_lease_factory(target):
                await execute()
        else:
            await execute()

    async def _run_resumable(self, job_id: str) -> None:
        context = JobExecutionContext(self, job_id)
        while not self._closed:
            job = await self.get(job_id)
            if not job or job.get("status") in TERMINAL_STATUSES:
                return
            if job.get("status") in PAUSED_STATUSES:
                event = self._control_events.setdefault(job_id, asyncio.Event())
                await event.wait()
                event.clear()
                if self._closed:
                    return
            operation = await self._resolve(await context.job())
            request = self.control_request(job_id)
            if request == "stop":
                await self._perform_stop(job_id, operation, context)
                return
            self._control_requests.pop(job_id, None)
            await self._update(
                job_id,
                status="running",
                status_reason="",
                control_requested="",
                message="任务正在从安全断点执行",
                error=None,
            )
            try:
                with resource_lane("task"):
                    result = await operation.run(context)
                async with self._transition_locks.setdefault(job_id, asyncio.Lock()):
                    await context.control_point()
                    await self._update(
                        job_id,
                        status="completed",
                        progress=1,
                        message="任务完成",
                        result=result,
                        error=None,
                        status_reason="",
                        control_requested="",
                    )
                if operation.finalize_completed:
                    try:
                        with resource_lane("task"):
                            await operation.finalize_completed(context)
                    except Exception:
                        logger.exception(
                            "已完成任务的断点工作区清理失败：job_id=%s", job_id
                        )
                return
            except JobPauseRequested as signal:
                reason = "shutdown" if signal.reason == "shutdown" else "manual"
                stop_requested = False
                async with self._transition_locks.setdefault(job_id, asyncio.Lock()):
                    stop_requested = self.control_request(job_id) == "stop"
                    if not stop_requested:
                        await self._update(
                            job_id,
                            status="paused",
                            status_reason=reason,
                            control_requested="",
                            message=(
                                "服务关闭，任务已保存安全断点"
                                if reason == "shutdown"
                                else "任务已手动暂停"
                            ),
                        )
                        self._control_requests.pop(job_id, None)
                if stop_requested:
                    await self._perform_stop(job_id, operation, context)
                    return
                if reason == "shutdown":
                    return
            except JobInterrupted as signal:
                stop_requested = False
                async with self._transition_locks.setdefault(job_id, asyncio.Lock()):
                    stop_requested = self.control_request(job_id) == "stop"
                    if not stop_requested:
                        await self._update(
                            job_id,
                            status="interrupted",
                            status_reason=signal.reason,
                            control_requested="",
                            message=signal.message,
                            error=signal.error,
                        )
                        self._control_requests.pop(job_id, None)
                if stop_requested:
                    await self._perform_stop(job_id, operation, context)
                    return
            except JobStopRequested:
                await self._perform_stop(job_id, operation, context)
                return
            except asyncio.CancelledError:
                latest = await self.get(job_id)
                if latest and latest.get("status") not in PAUSED_STATUSES:
                    await self._update(
                        job_id,
                        status="interrupted",
                        status_reason="process_interrupted",
                        control_requested="",
                        message="任务执行被中断，保留最近安全断点",
                    )
                raise
            except Exception as exc:
                logger.exception("可恢复任务失败，开始回滚：job_id=%s", job_id)
                try:
                    with resource_lane("task"):
                        await operation.rollback(context)
                except Exception as rollback_exc:
                    logger.exception("任务失败后的回滚也失败：job_id=%s", job_id)
                    await self._update(
                        job_id,
                        status="interrupted",
                        status_reason="rollback_failed",
                        message="任务失败且回滚失败，工作区已保留",
                        error=f"{exc}; rollback: {rollback_exc}",
                    )
                    continue
                await self._update(
                    job_id,
                    status="failed",
                    status_reason="operation_failed",
                    message="任务失败，已恢复任务前状态",
                    error=str(exc),
                )
                return

    async def _perform_stop(
        self,
        job_id: str,
        operation: ResolvedJobOperation,
        context: JobExecutionContext,
    ) -> None:
        await self._update(
            job_id,
            status="stopping",
            status_reason="manual",
            control_requested="stop",
            message="正在回滚到任务执行前状态",
        )
        try:
            with resource_lane("task"):
                await operation.rollback(context)
        except Exception as exc:
            logger.exception("停止任务回滚失败：job_id=%s", job_id)
            await self._update(
                job_id,
                status="interrupted",
                status_reason="rollback_failed",
                control_requested="",
                message="停止任务时回滚失败，可重试停止",
                error=str(exc),
            )
            self._control_requests.pop(job_id, None)
            return
        self._control_requests.pop(job_id, None)
        await self._update(
            job_id,
            status="stopped",
            progress=0,
            status_reason="manual",
            control_requested="",
            message="任务已停止并恢复到执行前状态",
            error=None,
        )

    async def pause(self, job_id: str) -> dict[str, Any]:
        async with self._transition_locks.setdefault(job_id, asyncio.Lock()):
            job = await self.get(job_id)
            if not job:
                raise KeyError(job_id)
            if not job.get("resumable") or job.get("status") != "running":
                raise JobStateConflict("job cannot be paused in its current state")
            self._control_requests[job_id] = "pause"
            await self._update(
                job_id,
                status="pausing",
                status_reason="manual",
                control_requested="pause",
                message="正在保存安全断点",
            )
        return (await self.get(job_id)) or job

    async def resume(self, job_id: str) -> dict[str, Any]:
        job = await self.get(job_id)
        if not job:
            raise KeyError(job_id)
        if not job.get("resumable") or job.get("status") not in PAUSED_STATUSES:
            raise JobStateConflict("job cannot be resumed in its current state")
        self._control_requests.pop(job_id, None)
        await self._update(
            job_id,
            status="paused",
            status_reason="resume_requested",
            control_requested="resume",
            message="已请求从安全断点继续",
            error=None,
        )
        self._signal_control(job_id)
        return (await self.get(job_id)) or job

    async def stop(self, job_id: str) -> dict[str, Any]:
        async with self._transition_locks.setdefault(job_id, asyncio.Lock()):
            job = await self.get(job_id)
            if not job:
                raise KeyError(job_id)
            if not job.get("resumable") or job.get("status") not in {
                "running",
                "pausing",
                "paused",
                "interrupted",
            }:
                raise JobStateConflict("job cannot be stopped in its current state")
            self._control_requests[job_id] = "stop"
            await self._update(
                job_id,
                status="stopping",
                status_reason="manual",
                control_requested="stop",
                message="正在停止并回滚任务",
            )
        self._signal_control(job_id)
        return (await self.get(job_id)) or job

    async def cancel(self, job_id: str, *, reason: str = "manual") -> dict[str, Any]:
        job = await self.get(job_id, internal=True)
        if not job:
            raise KeyError(job_id)
        if job.get("status") != "queued":
            raise JobStateConflict("only queued jobs can be cancelled")
        if job.get("resumable"):
            operation = await self._resolve(job)
            if operation.cancel_queued:
                await operation.cancel_queued(JobExecutionContext(self, job_id))
        self._operations.pop(job_id, None)
        await self._update(
            job_id,
            status="cancelled",
            progress=0,
            status_reason=reason,
            control_requested="",
            message="排队任务已取消" if reason == "manual" else "服务关闭，排队任务已取消",
            error=None,
        )
        self._execution_done.setdefault(job_id, asyncio.Event()).set()
        return (await self.get(job_id)) or job

    async def _best_effort_terminal_update(self, job_id: str, exc: Exception) -> None:
        try:
            await self._update(
                job_id,
                status="failed",
                status_reason="manager_error",
                message="任务管理器异常",
                error=str(exc),
            )
        except Exception:
            logger.exception("任务终态补偿失败：job_id=%s", job_id)

    async def _capture_database_state(
        self,
        snapshot: dict[str, Any],
        *,
        phase: str,
        captured_at: float,
    ) -> None:
        if phase not in {"before", "after"}:
            raise ValueError("invalid task database-state phase")
        definition = task_type_registry.get(
            str(snapshot.get("database_type") or LIVINGMEMORY_V8_TYPE),
            str(snapshot.get("kind") or ""),
        )
        database_id = str(snapshot.get("database_id") or "")
        field = f"database_state_{phase}"
        if (
            not definition.database_state_comparison
            or not database_id
            or snapshot.get(field) is not None
        ):
            return
        database_type = str(
            snapshot.get("database_type") or LIVINGMEMORY_V8_TYPE
        )
        provider = self._database_state_providers.get(database_type)
        errors = dict(snapshot.get("database_state_capture_error") or {})
        if provider is None:
            errors[phase] = "database-state provider is unavailable"
            snapshot["database_state_capture_error"] = errors
            return
        try:
            value = provider(database_id, str(snapshot.get("kind") or ""))
            if inspect.isawaitable(value):
                value = await value
            safe = sanitize_task_value(value)
            if not isinstance(safe, dict):
                safe = {"value": safe}
            snapshot[field] = {
                "schema_version": 1,
                "captured_at": captured_at,
                **safe,
            }
            errors.pop(phase, None)
        except Exception as exc:
            errors[phase] = safe_summary(str(exc), max_chars=300)
            logger.warning(
                "任务库状态快照失败：job_id=%s phase=%s database_type=%s database_id=%s error=%s",
                snapshot.get("id") or "",
                phase,
                database_type,
                database_id,
                safe_summary(str(exc), max_chars=160),
            )
        snapshot["database_state_capture_error"] = errors

    @staticmethod
    def _append_status_history(
        snapshot: dict[str, Any],
        *,
        status: str,
        timestamp: float,
        message: str,
        reason: str,
    ) -> None:
        history = list(snapshot.get("status_history") or [])
        if history and str(history[-1].get("status") or "") == status:
            return
        history.append(
            {
                "status": status,
                "timestamp": timestamp,
                "message": message,
                "reason": reason,
            }
        )
        snapshot["status_history"] = history[-100:]

    async def _update(self, job_id: str, **fields: Any) -> None:
        allowed = {
            "status",
            "progress",
            "message",
            "result",
            "error",
            "checkpoint",
            "status_reason",
            "control_requested",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        snapshot = self._snapshots.get(job_id)
        if snapshot is None:
            snapshot = await self._read_job(job_id)
            if snapshot is None:
                return
            self._snapshots[job_id] = snapshot
        updates = {
            key: value
            for key, value in updates.items()
            if snapshot.get(key) != value
        }
        if not updates:
            return
        previous_status = str(snapshot.get("status") or "")
        next_status = str(updates.get("status") or previous_status)
        now = time.time()
        entering_execution = (
            next_status == "running"
            and previous_status in {"queued", "paused", "interrupted"}
        )
        entering_terminal = (
            next_status in TERMINAL_STATUSES
            and previous_status not in TERMINAL_STATUSES
        )
        if entering_execution:
            if snapshot.get("started_at") is None:
                snapshot["started_at"] = now
            snapshot["attempt_count"] = int(snapshot.get("attempt_count") or 0) + 1
            await self._capture_database_state(
                snapshot,
                phase="before",
                captured_at=now,
            )
        if entering_terminal:
            if snapshot.get("started_at") is not None:
                await self._capture_database_state(
                    snapshot,
                    phase="after",
                    captured_at=now,
                )
            snapshot["finished_at"] = now
        snapshot.update(updates)
        snapshot["updated_at"] = now
        if next_status != previous_status:
            self._append_status_history(
                snapshot,
                status=next_status,
                timestamp=now,
                message=str(snapshot.get("message") or ""),
                reason=str(snapshot.get("status_reason") or ""),
            )
        payload = dict(snapshot)
        terminal = str(payload.get("status") or "") in TERMINAL_STATUSES
        status_changed = str(payload.get("status") or "") != previous_status
        now = float(payload["updated_at"])
        should_persist = (
            terminal
            or status_changed
            or bool(
                {"checkpoint", "control_requested", "status_reason"}
                & updates.keys()
            )
            or now - self._last_persisted_at.get(job_id, 0) >= PROGRESS_PERSIST_INTERVAL_SECONDS
        )
        persisted = False
        if should_persist:
            try:
                await self._persist_snapshot(payload)
                self._last_persisted_at[job_id] = now
                persisted = True
            except Exception:
                logger.exception("任务状态持久化失败：job_id=%s", job_id)
        self._notify(job_id, self._public_payload(payload))
        self._signal_state(job_id)
        if should_persist and {"status", "progress", "message"} & updates.keys():
            logger.info(
                "任务进度：job_id=%s kind=%s status=%s progress=%.1f%% message=%s",
                job_id,
                payload.get("kind") or "",
                payload.get("status") or "",
                float(payload.get("progress") or 0) * 100,
                safe_summary(payload.get("message"), max_chars=160),
            )
        if terminal:
            future = self._completion.setdefault(
                job_id, asyncio.get_running_loop().create_future()
            )
            if not future.done():
                future.set_result(self._public_payload(payload))
            if persisted:
                self._snapshots.pop(job_id, None)
                self._last_persisted_at.pop(job_id, None)

    async def _persist_snapshot(self, snapshot: dict[str, Any]) -> None:
        def encode(value: Any) -> str | None:
            return json.dumps(value, ensure_ascii=False) if value is not None else None

        async with self._connect() as db:
            await db.execute(
                """UPDATE jobs SET status=?,progress=?,message=?,result=?,error=?,
                checkpoint=?,status_reason=?,control_requested=?,started_at=?,finished_at=?,
                attempt_count=?,status_history=?,database_state_before=?,database_state_after=?,
                database_state_capture_error=?,updated_at=? WHERE id=?""",
                (
                    snapshot.get("status"),
                    float(snapshot.get("progress") or 0),
                    str(snapshot.get("message") or ""),
                    encode(snapshot.get("result")),
                    snapshot.get("error"),
                    encode(snapshot.get("checkpoint")),
                    str(snapshot.get("status_reason") or ""),
                    str(snapshot.get("control_requested") or "") or None,
                    snapshot.get("started_at"),
                    snapshot.get("finished_at"),
                    int(snapshot.get("attempt_count") or 0),
                    encode(snapshot.get("status_history") or []),
                    encode(snapshot.get("database_state_before")),
                    encode(snapshot.get("database_state_after")),
                    encode(snapshot.get("database_state_capture_error") or {}),
                    float(snapshot.get("updated_at") or time.time()),
                    snapshot["id"],
                ),
            )
            await db.commit()

    @staticmethod
    def _capabilities(payload: dict[str, Any]) -> dict[str, bool]:
        status = str(payload.get("status") or "")
        resumable = bool(payload.get("resumable"))
        return {
            "pause": resumable and status == "running",
            "resume": resumable and status in PAUSED_STATUSES,
            "stop": resumable
            and status in {"running", "pausing", "paused", "interrupted"},
            "cancel": status == "queued",
        }

    @classmethod
    def _public_payload(
        cls,
        payload: dict[str, Any],
        *,
        detail: bool = False,
    ) -> dict[str, Any]:
        result = dict(payload)
        operation = result.pop("operation", None)
        status_history = list(result.pop("status_history", None) or [])
        state_before = result.pop("database_state_before", None)
        state_after = result.pop("database_state_after", None)
        state_errors = dict(result.pop("database_state_capture_error", None) or {})
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
        # Frozen v0.1.1 task-state compatibility alias.
        result["library_id"] = result.get("database_resource_key")
        checkpoint = result.get("checkpoint")
        if isinstance(checkpoint, dict) and not detail:
            result["checkpoint"] = {
                key: checkpoint.get(key)
                for key in (
                    "phase",
                    "completed_documents",
                    "completed_graph_entries",
                    "total_documents",
                    "total_graph_entries",
                    "saved_at",
                )
                if key in checkpoint
            }
        elif detail:
            result["checkpoint"] = sanitize_task_value(checkpoint)
        result["capabilities"] = cls._capabilities(payload)
        result["resumable"] = bool(payload.get("resumable"))
        definition = task_type_registry.get(
            str(payload.get("database_type") or LIVINGMEMORY_V8_TYPE),
            str(payload.get("kind") or ""),
        )
        result["task_type"] = definition.public()
        created_at = float(payload.get("created_at") or 0)
        terminal = str(payload.get("status") or "") in TERMINAL_STATUSES
        started_at = (
            float(payload["started_at"])
            if payload.get("started_at") is not None
            else None
        )
        finished_at = (
            float(payload["finished_at"])
            if payload.get("finished_at") is not None
            else (
                float(payload.get("updated_at") or 0) or None
                if terminal
                else None
            )
        )
        result["attempt_count"] = int(payload.get("attempt_count") or 0)
        result["timing"] = {
            "created_at": created_at or None,
            "started_at": started_at,
            "finished_at": finished_at,
            "queue_duration_seconds": (
                max(0.0, started_at - created_at)
                if started_at is not None and created_at
                else None
            ),
            "execution_duration_seconds": (
                max(0.0, finished_at - started_at)
                if finished_at is not None and started_at is not None
                else None
            ),
            "total_duration_seconds": (
                max(0.0, finished_at - created_at)
                if finished_at is not None and created_at
                else None
            ),
        }
        result["detail_available"] = terminal
        if detail:
            if not status_history:
                status_history = [
                    {
                        "status": str(payload.get("status") or ""),
                        "timestamp": float(payload.get("updated_at") or created_at),
                        "message": str(payload.get("message") or ""),
                        "reason": str(payload.get("status_reason") or ""),
                    }
                ]
            result["request_metadata"] = task_request_summary(operation)
            result["result"] = sanitize_task_value(result.get("result"))
            result["status_history"] = sanitize_task_value(status_history)
            result["database_state_comparison"] = (
                build_database_state_comparison(
                    state_before,
                    state_after,
                    state_errors,
                )
                if definition.database_state_comparison
                else None
            )
        return result

    def _notify(self, job_id: str, payload: dict[str, Any]) -> None:
        for queue in list(self.subscribers.get(job_id, set())):
            try:
                if queue.full():
                    queue.get_nowait()
                    queue.task_done()
                queue.put_nowait(dict(payload))
            except Exception:
                logger.exception("任务 SSE 通知失败：job_id=%s", job_id)

    @staticmethod
    def _decode_row(row: Any) -> dict[str, Any]:
        result = dict(row)
        result["database_resource_key"] = result.pop("library_id", None)
        for field in (
            "result",
            "operation",
            "checkpoint",
            "status_history",
            "database_state_before",
            "database_state_after",
            "database_state_capture_error",
        ):
            if result.get(field):
                try:
                    result[field] = json.loads(result[field])
                except (json.JSONDecodeError, TypeError):
                    result[field] = None
        result["resumable"] = bool(result.get("resumable"))
        result["attempt_count"] = int(result.get("attempt_count") or 0)
        if not isinstance(result.get("status_history"), list):
            result["status_history"] = []
        if not isinstance(result.get("database_state_capture_error"), dict):
            result["database_state_capture_error"] = {}
        result["control_requested"] = str(result.get("control_requested") or "")
        result["status_reason"] = str(result.get("status_reason") or "")
        return result

    async def _read_job(self, job_id: str) -> dict[str, Any] | None:
        async with self._connect() as db:
            row = await (
                await db.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
            ).fetchone()
        return self._decode_row(row) if row else None

    async def get(
        self,
        job_id: str,
        *,
        internal: bool = False,
        detail: bool = False,
    ) -> dict[str, Any] | None:
        snapshot = self._snapshots.get(job_id)
        payload = dict(snapshot) if snapshot is not None else await self._read_job(job_id)
        if payload is None or internal:
            return payload
        return self._public_payload(payload, detail=detail)

    async def wait(self, job_id: str, *, poll_interval: float = 0.05) -> dict[str, Any]:
        del poll_interval
        payload = await self.get(job_id)
        if not payload:
            raise KeyError(job_id)
        if payload.get("status") not in TERMINAL_STATUSES:
            future = self._completion.setdefault(
                job_id, asyncio.get_running_loop().create_future()
            )
            payload = dict(await asyncio.shield(future))
        execution_done = self._execution_done.get(job_id)
        if execution_done is not None:
            await execution_done.wait()
        return payload

    @staticmethod
    def _summary_source(payload: dict[str, Any]) -> dict[str, Any]:
        fields = {
            "id",
            "database_resource_key",
            "database_type",
            "database_id",
            "kind",
            "status",
            "progress",
            "message",
            "error",
            "checkpoint",
            "status_reason",
            "control_requested",
            "resumable",
            "started_at",
            "finished_at",
            "attempt_count",
            "created_at",
            "updated_at",
        }
        result = {key: payload.get(key) for key in fields}
        checkpoint = payload.get("checkpoint")
        if isinstance(checkpoint, dict):
            result["checkpoint"] = {
                key: checkpoint.get(key)
                for key in (
                    "phase",
                    "completed_documents",
                    "completed_graph_entries",
                    "total_documents",
                    "total_graph_entries",
                    "saved_at",
                )
                if key in checkpoint
            }
        return result

    @staticmethod
    def _encode_page_cursor(offset: int) -> str:
        return base64.urlsafe_b64encode(str(max(0, offset)).encode("ascii")).decode(
            "ascii"
        ).rstrip("=")

    @staticmethod
    def _decode_page_cursor(cursor: str | None) -> int:
        if not cursor:
            return 0
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("ascii")
            offset = int(raw)
        except (ValueError, UnicodeError, binascii.Error) as exc:
            raise ValueError("invalid job cursor") from exc
        if offset < 0:
            raise ValueError("invalid job cursor")
        return offset

    async def list_page(
        self,
        *,
        scope: str = "active",
        limit: int | None = None,
        cursor: str | None = None,
        include_details: bool = False,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if scope not in {"active", "finished", "all"}:
            raise ValueError("invalid job scope")
        if limit is not None and not 1 <= int(limit) <= 200:
            raise ValueError("job limit must be between 1 and 200")
        offset = self._decode_page_cursor(cursor)
        statuses = ACTIVE_STATUSES if scope == "active" else TERMINAL_STATUSES
        where = ""
        params: tuple[Any, ...] = ()
        if scope != "all":
            placeholders = ",".join("?" for _ in statuses)
            where = f"WHERE status IN ({placeholders})"
            params = tuple(sorted(statuses))
        if limit is not None and scope == "finished":
            page_size = int(limit)
            async with self._connect() as db:
                rows = await (
                    await db.execute(
                        f"SELECT {'*' if include_details else JOB_SUMMARY_COLUMNS} "
                        f"FROM jobs {where} "
                        "ORDER BY updated_at DESC, id ASC LIMIT ? OFFSET ?",
                        (*params, page_size + 1, offset),
                    )
                ).fetchall()
            has_more = len(rows) > page_size
            page = [self._decode_row(row) for row in rows[:page_size]]
            for index, item in enumerate(page):
                snapshot = self._snapshots.get(str(item.get("id") or ""))
                if snapshot is not None:
                    page[index] = (
                        dict(snapshot)
                        if include_details
                        else self._summary_source(snapshot)
                    )
            return [
                self._public_payload(item, detail=include_details)
                for item in page
            ], (
                self._encode_page_cursor(offset + page_size)
                if has_more
                else None
            )
        async with self._connect() as db:
            rows = await (
                await db.execute(
                    f"SELECT {'*' if include_details else JOB_SUMMARY_COLUMNS} "
                    f"FROM jobs {where}",
                    params,
                )
            ).fetchall()
        merged = {str(row["id"]): self._decode_row(row) for row in rows}
        for job_id, snapshot in self._snapshots.items():
            status = str(snapshot.get("status") or "")
            if scope == "active" and status not in ACTIVE_STATUSES:
                continue
            if scope == "finished" and status not in TERMINAL_STATUSES:
                continue
            merged[job_id] = (
                dict(snapshot)
                if include_details
                else self._summary_source(snapshot)
            )
        items = list(merged.values())
        if scope == "active":
            items.sort(
                key=lambda item: (
                    float(item.get("created_at") or 0),
                    str(item.get("id") or ""),
                )
            )
        elif scope == "finished":
            items.sort(
                key=lambda item: (
                    -float(item.get("updated_at") or 0),
                    str(item.get("id") or ""),
                )
            )
        else:
            items.sort(
                key=lambda item: (
                    0 if item.get("status") in ACTIVE_STATUSES else 1,
                    float(item.get("created_at") or 0)
                    if item.get("status") in ACTIVE_STATUSES
                    else -float(item.get("updated_at") or 0),
                    str(item.get("id") or ""),
                )
            )
        end = len(items) if limit is None else min(len(items), offset + int(limit))
        page = items[offset:end]
        next_cursor = (
            self._encode_page_cursor(end) if end < len(items) else None
        )
        return [
            self._public_payload(item, detail=include_details)
            for item in page
        ], next_cursor

    async def list(
        self,
        *,
        scope: str = "active",
        limit: int | None = None,
        cursor: str | None = None,
        include_details: bool = True,
    ) -> list[dict[str, Any]]:
        items, _next_cursor = await self.list_page(
            scope=scope,
            limit=limit,
            cursor=cursor,
            include_details=include_details,
        )
        return items

    async def clear_finished(self) -> int:
        async with self._connect() as db:
            cursor = await db.execute(
                "DELETE FROM jobs WHERE status IN ({})".format(
                    ",".join("?" for _ in TERMINAL_STATUSES)
                ),
                tuple(sorted(TERMINAL_STATUSES)),
            )
            await db.commit()
            deleted = max(0, int(cursor.rowcount or 0))
        for job_id in [
            job_id
            for job_id, snapshot in self._snapshots.items()
            if str(snapshot.get("status") or "") in TERMINAL_STATUSES
        ]:
            self._snapshots.pop(job_id, None)
            self._completion.pop(job_id, None)
            self._execution_done.pop(job_id, None)
            self._control_events.pop(job_id, None)
            self._state_events.pop(job_id, None)
            self._transition_locks.pop(job_id, None)
            self._control_requests.pop(job_id, None)
            self._last_persisted_at.pop(job_id, None)
            self.subscribers.pop(job_id, None)
        return deleted

    async def subscribe(self, job_id: str) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        self.subscribers[job_id].add(queue)
        try:
            current = await self.get(job_id)
            if current:
                yield current
                if current.get("status") in TERMINAL_STATUSES:
                    return
            while True:
                payload = await queue.get()
                queue.task_done()
                yield payload
                if payload.get("status") in TERMINAL_STATUSES:
                    break
        finally:
            self.subscribers[job_id].discard(queue)
            if not self.subscribers[job_id]:
                self.subscribers.pop(job_id, None)
