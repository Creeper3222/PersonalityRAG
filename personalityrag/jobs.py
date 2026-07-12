from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict
from typing import Any, AsyncContextManager, AsyncIterator, Awaitable, Callable

from .logger import logger, safe_summary
from .storage import Storage
from .task_types import ADAPTER_BUSY_JOB_KINDS


ACTIVE_STATUSES = frozenset({"queued", "running"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
JOB_HISTORY_DAYS = 30
JOB_HISTORY_LIMIT = 1000
PROGRESS_PERSIST_INTERVAL_SECONDS = 0.25
SUBSCRIBER_QUEUE_SIZE = 1

ProgressCallback = Callable[[float, str], Awaitable[None]]
JobOperation = Callable[[ProgressCallback], Awaitable[Any]]


class JobManager:
    def __init__(
        self,
        storage: Storage,
        runtime_lease_factory: Callable[[str], AsyncContextManager[Any]] | None = None,
    ):
        self.storage = storage
        self._runtime_lease_factory = runtime_lease_factory
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._operations: dict[str, JobOperation] = {}
        self._runtime_lease_jobs: set[str] = set()
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._completion: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._execution_done: dict[str, asyncio.Event] = {}
        self._last_persisted_at: dict[str, float] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._start_lock = asyncio.Lock()
        self._closed = False
        self.subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = (
            defaultdict(set)
        )

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def _clear_persistent_jobs(self) -> int:
        async with self.storage.connect(system=True) as db:
            removed = await db.execute("DELETE FROM jobs")
            await db.commit()
        return max(0, int(removed.rowcount or 0))

    async def clear_for_startup(self) -> None:
        """Reset all persisted and in-memory job state for a fresh startup."""
        self._operations.clear()
        self._runtime_lease_jobs.clear()
        self._snapshots.clear()
        self._completion.clear()
        self._execution_done.clear()
        self._last_persisted_at.clear()
        self.subscribers.clear()
        self._drain_queue()
        removed = await self._clear_persistent_jobs()
        logger.info("任务列表启动已清空：removed=%s", removed)
        return

        now = time.time()
        cutoff = now - JOB_HISTORY_DAYS * 24 * 60 * 60
        async with self.storage.connect(system=True) as db:
            stale = await db.execute(
                """UPDATE jobs SET status='cancelled',progress=0,
                message='服务重启，遗留任务已取消',updated_at=?
                WHERE status IN ('queued','running')""",
                (now,),
            )
            expired = await db.execute(
                """DELETE FROM jobs WHERE status IN ('completed','failed','cancelled')
                AND updated_at<?""",
                (cutoff,),
            )
            overflow = await db.execute(
                """DELETE FROM jobs WHERE id IN (
                    SELECT id FROM jobs
                    WHERE status IN ('completed','failed','cancelled')
                    ORDER BY updated_at DESC
                    LIMIT -1 OFFSET ?
                )""",
                (JOB_HISTORY_LIMIT,),
            )
            await db.commit()
        logger.info(
            "任务历史已恢复：cancelled_stale=%s expired_removed=%s overflow_removed=%s retention_days=%s limit=%s",
            max(0, int(stale.rowcount or 0)),
            max(0, int(expired.rowcount or 0)),
            max(0, int(overflow.rowcount or 0)),
            JOB_HISTORY_DAYS,
            JOB_HISTORY_LIMIT,
        )

    async def close(self) -> None:
        self._closed = True
        if self._worker_task:
            self._worker_task.cancel()
            await asyncio.gather(self._worker_task, return_exceptions=True)
            self._worker_task = None

        shutdown_at = time.time()
        for job_id, snapshot in list(self._snapshots.items()):
            if str(snapshot.get("status") or "") not in ACTIVE_STATUSES:
                continue
            payload = {
                **snapshot,
                "status": "cancelled",
                "progress": 0.0,
                "message": "服务正在关闭",
                "updated_at": shutdown_at,
            }
            self._snapshots[job_id] = payload
            self._notify(job_id, payload)

        for job_id, future in list(self._completion.items()):
            payload = self._snapshots.get(job_id)
            if payload is None:
                payload = await self._read_job(job_id)
            if payload is None:
                payload = {
                    "id": job_id,
                    "library_id": None,
                    "kind": "",
                    "status": "cancelled",
                    "progress": 0.0,
                    "message": "服务正在关闭",
                    "result": None,
                    "error": None,
                    "created_at": shutdown_at,
                    "updated_at": shutdown_at,
                }
            elif str(payload.get("status") or "") in ACTIVE_STATUSES:
                payload = {
                    **payload,
                    "status": "cancelled",
                    "progress": 0.0,
                    "message": "服务正在关闭",
                    "updated_at": shutdown_at,
                }
                self._snapshots[job_id] = payload
            if not future.done():
                future.set_result(dict(payload))

        for event in self._execution_done.values():
            event.set()

        removed = await self._clear_persistent_jobs()
        self._operations.clear()
        self._runtime_lease_jobs.clear()
        self._snapshots.clear()
        self._completion.clear()
        self._execution_done.clear()
        self._last_persisted_at.clear()
        self._drain_queue()
        self.subscribers.clear()
        logger.info("任务列表关闭已清空：removed=%s", removed)
        return

        queued_ids = list(self._operations)
        self._operations.clear()
        self._runtime_lease_jobs.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        for job_id in queued_ids:
            await self._update(
                job_id,
                status="cancelled",
                progress=0.0,
                message="服务正在关闭",
            )
            event = self._execution_done.get(job_id)
            if event is not None:
                event.set()
        for job_id in list(self.subscribers):
            self._notify(
                job_id,
                {
                    "id": job_id,
                    "status": "cancelled",
                    "progress": 0.0,
                    "message": "服务正在关闭",
                },
            )

    async def start(
        self,
        kind: str,
        operation: JobOperation,
        *,
        library_id: str | None = None,
        dedupe_active: bool = True,
        lease_runtime: bool = True,
    ) -> str:
        if self._closed:
            raise RuntimeError("任务管理器已关闭")
        async with self._start_lock:
            if dedupe_active:
                existing = await self._find_active(kind, library_id)
                if existing:
                    logger.warning(
                        "复用已存在的活动任务：job_id=%s kind=%s library_id=%s",
                        existing,
                        kind,
                        library_id or "",
                    )
                    return existing

            job_id = uuid.uuid4().hex
            now = time.time()
            snapshot = {
                "id": job_id,
                "library_id": library_id,
                "kind": kind,
                "status": "queued",
                "progress": 0.0,
                "message": "等待执行",
                "result": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
            async with self.storage.connect(system=True) as db:
                await db.execute(
                    """INSERT INTO jobs
                    (id,library_id,kind,status,progress,message,result,error,
                    created_at,updated_at)
                    VALUES(?,?,?,'queued',0,'等待执行',NULL,NULL,?,?)""",
                    (job_id, library_id, kind, now, now),
                )
                await db.commit()

            self._snapshots[job_id] = snapshot
            self._last_persisted_at[job_id] = now
            self._prune_completion_futures()
            self._completion[job_id] = asyncio.get_running_loop().create_future()
            self._execution_done[job_id] = asyncio.Event()
            self._operations[job_id] = operation
            if lease_runtime and library_id and self._runtime_lease_factory is not None:
                self._runtime_lease_jobs.add(job_id)
            await self._queue.put(job_id)
            self._ensure_worker()

        logger.info(
            "任务已入队：job_id=%s kind=%s library_id=%s",
            job_id,
            kind,
            library_id or "",
        )
        return job_id

    async def _find_active(self, kind: str, library_id: str | None) -> str | None:
        memory_matches = sorted(
            (
                snapshot
                for snapshot in self._snapshots.values()
                if snapshot.get("kind") == kind
                and snapshot.get("library_id") == library_id
                and snapshot.get("status") in ACTIVE_STATUSES
            ),
            key=lambda item: float(item.get("created_at") or 0.0),
        )
        if memory_matches:
            return str(memory_matches[0]["id"])

        async with self.storage.connect(system=True) as db:
            if library_id is None:
                row = await (
                    await db.execute(
                        """SELECT id FROM jobs WHERE kind=? AND library_id IS NULL
                        AND status IN ('queued','running')
                        ORDER BY created_at LIMIT 1""",
                        (kind,),
                    )
                ).fetchone()
            else:
                row = await (
                    await db.execute(
                        """SELECT id FROM jobs WHERE kind=? AND library_id=?
                        AND status IN ('queued','running')
                        ORDER BY created_at LIMIT 1""",
                        (kind, library_id),
                    )
                ).fetchone()
        return str(row["id"]) if row else None

    async def active_job_id(
        self, kind: str, library_id: str | None = None
    ) -> str | None:
        return await self._find_active(kind, library_id)

    async def active_long_job(self, library_id: str) -> dict[str, Any] | None:
        return (await self.active_long_jobs_map([library_id])).get(library_id)

    async def active_long_jobs_map(
        self, library_ids: list[str]
    ) -> dict[str, dict[str, Any] | None]:
        if not library_ids:
            return {}
        library_placeholders = ",".join("?" for _ in library_ids)
        kind_placeholders = ",".join("?" for _ in ADAPTER_BUSY_JOB_KINDS)
        async with self.storage.connect(system=True) as db:
            rows = await (
                await db.execute(
                    f"""SELECT * FROM jobs
                    WHERE library_id IN ({library_placeholders})
                    AND kind IN ({kind_placeholders})
                    AND status IN ('queued','running')
                    ORDER BY created_at""",
                    (*library_ids, *ADAPTER_BUSY_JOB_KINDS),
                )
            ).fetchall()

        merged = {str(row["id"]): self._decode_row(row) for row in rows}
        for job_id, snapshot in self._snapshots.items():
            if snapshot.get("library_id") in library_ids:
                merged[job_id] = dict(snapshot)
        result: dict[str, dict[str, Any] | None] = {
            library_id: None for library_id in library_ids
        }
        active = sorted(
            (
                item
                for item in merged.values()
                if item.get("library_id") in result
                and item.get("kind") in ADAPTER_BUSY_JOB_KINDS
                and item.get("status") in ACTIVE_STATUSES
            ),
            key=lambda item: float(item.get("created_at") or 0.0),
        )
        for item in active:
            library_id = str(item.get("library_id") or "")
            if library_id and result[library_id] is None:
                result[library_id] = item
        return result

    def _ensure_worker(self) -> None:
        if self._worker_task and not self._worker_task.done():
            return
        self._worker_task = asyncio.create_task(self._worker())

    def _prune_completion_futures(self) -> None:
        if len(self._completion) < JOB_HISTORY_LIMIT:
            return
        for job_id, future in list(self._completion.items()):
            if future.done():
                self._completion.pop(job_id, None)
                event = self._execution_done.get(job_id)
                if event is None or event.is_set():
                    self._execution_done.pop(job_id, None)
            if len(self._completion) < JOB_HISTORY_LIMIT:
                break

    async def _worker(self) -> None:
        logger.info("任务队列 worker 已启动")
        while not self._closed:
            try:
                job_id = await self._queue.get()
            except asyncio.CancelledError:
                return
            try:
                operation = self._operations.pop(job_id, None)
                if operation is None:
                    await self._update(
                        job_id,
                        status="cancelled",
                        message="任务操作已丢失",
                    )
                    continue
                await self._run(job_id, operation)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "任务队列 worker 已隔离异常并继续运行：job_id=%s err=%s",
                    job_id,
                    exc,
                )
                await self._best_effort_terminal_update(job_id, exc)
            finally:
                self._runtime_lease_jobs.discard(job_id)
                event = self._execution_done.get(job_id)
                if event is not None:
                    event.set()
                self._queue.task_done()

    async def _run(self, job_id: str, operation: JobOperation) -> None:
        job = await self.get(job_id) or {}
        logger.info(
            "任务开始：job_id=%s kind=%s library_id=%s",
            job_id,
            job.get("kind") or "",
            job.get("library_id") or "",
        )
        await self._update(job_id, status="running", message="任务已启动")

        async def progress(value: float, message: str) -> None:
            await self._update(
                job_id,
                status="running",
                progress=max(0.0, min(1.0, value)),
                message=message,
            )

        async def execute() -> None:
            try:
                result = await operation(progress)
                await self._update(
                    job_id,
                    status="completed",
                    progress=1.0,
                    message="任务完成",
                    result=result,
                    error=None,
                )
                logger.info(
                    "任务完成：job_id=%s kind=%s library_id=%s",
                    job_id,
                    job.get("kind") or "",
                    job.get("library_id") or "",
                )
            except asyncio.CancelledError:
                logger.warning("任务已取消：job_id=%s", job_id)
                await self._update(
                    job_id,
                    status="cancelled",
                    message="任务已取消",
                )
                raise
            except Exception as exc:
                logger.exception("任务失败：job_id=%s err=%s", job_id, exc)
                await self._update(
                    job_id,
                    status="failed",
                    message="任务失败",
                    error=str(exc),
                )

        library_id = str(job.get("library_id") or "")
        if (
            job_id in self._runtime_lease_jobs
            and library_id
            and self._runtime_lease_factory is not None
        ):
            async with self._runtime_lease_factory(library_id):
                await execute()
        else:
            await execute()

    async def _best_effort_terminal_update(
        self, job_id: str, exc: Exception
    ) -> None:
        try:
            await self._update(
                job_id,
                status="failed",
                message="任务管理器异常",
                error=str(exc),
            )
        except Exception:
            logger.exception("任务终态补偿失败：job_id=%s", job_id)

    async def _update(self, job_id: str, **fields: Any) -> None:
        allowed = {"status", "progress", "message", "result", "error"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        snapshot = self._snapshots.get(job_id)
        if snapshot is None:
            snapshot = await self._read_job(job_id)
            if snapshot is None:
                return
            self._snapshots[job_id] = snapshot

        previous_status = str(snapshot.get("status") or "")
        snapshot.update(updates)
        now = time.time()
        snapshot["updated_at"] = now
        payload = dict(snapshot)
        terminal = str(payload.get("status") or "") in TERMINAL_STATUSES
        status_changed = str(payload.get("status") or "") != previous_status
        last_persisted = self._last_persisted_at.get(job_id, 0.0)
        should_persist = (
            terminal
            or status_changed
            or now - last_persisted >= PROGRESS_PERSIST_INTERVAL_SECONDS
        )

        persisted = False
        if should_persist:
            attempts = 2 if terminal else 1
            for attempt in range(attempts):
                try:
                    await self._persist_snapshot(payload)
                    self._last_persisted_at[job_id] = now
                    persisted = True
                    break
                except Exception:
                    logger.exception(
                        "任务状态持久化失败：job_id=%s terminal=%s attempt=%s/%s",
                        job_id,
                        terminal,
                        attempt + 1,
                        attempts,
                    )
                    if attempt + 1 < attempts:
                        await asyncio.sleep(0)

        if should_persist and {"status", "progress", "message"} & updates.keys():
            logger.info(
                "任务进度：job_id=%s kind=%s library_id=%s status=%s progress=%.1f%% message=%s",
                job_id,
                payload.get("kind") or "",
                payload.get("library_id") or "",
                payload.get("status") or "",
                float(payload.get("progress") or 0.0) * 100,
                safe_summary(payload.get("message"), max_chars=160),
            )

        self._notify(job_id, payload)
        if terminal:
            future = self._completion.get(job_id)
            if future is None:
                future = asyncio.get_running_loop().create_future()
                self._completion[job_id] = future
            if not future.done():
                future.set_result(dict(payload))
            if persisted:
                self._snapshots.pop(job_id, None)
                self._last_persisted_at.pop(job_id, None)

    async def _persist_snapshot(self, snapshot: dict[str, Any]) -> None:
        raw_result = snapshot.get("result")
        result = (
            json.dumps(raw_result, ensure_ascii=False)
            if raw_result is not None
            else None
        )
        async with self.storage.connect(system=True) as db:
            await db.execute(
                """UPDATE jobs SET status=?,progress=?,message=?,result=?,error=?,
                updated_at=? WHERE id=?""",
                (
                    snapshot.get("status"),
                    float(snapshot.get("progress") or 0.0),
                    str(snapshot.get("message") or ""),
                    result,
                    snapshot.get("error"),
                    float(snapshot.get("updated_at") or time.time()),
                    snapshot["id"],
                ),
            )
            await db.commit()

    def _notify(self, job_id: str, payload: dict[str, Any]) -> None:
        for queue in list(self.subscribers.get(job_id, set())):
            try:
                if queue.full():
                    queue.get_nowait()
                    queue.task_done()
                queue.put_nowait(dict(payload))
            except Exception:
                logger.exception("任务 SSE 通知失败，已隔离：job_id=%s", job_id)

    @staticmethod
    def _decode_row(row: Any) -> dict[str, Any]:
        result = dict(row)
        if result.get("result"):
            try:
                result["result"] = json.loads(result["result"])
            except json.JSONDecodeError:
                result["result"] = None
        return result

    async def _read_job(self, job_id: str) -> dict[str, Any] | None:
        async with self.storage.connect(system=True) as db:
            row = await (
                await db.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
            ).fetchone()
        return self._decode_row(row) if row else None

    async def get(self, job_id: str) -> dict[str, Any] | None:
        snapshot = self._snapshots.get(job_id)
        if snapshot is not None:
            return dict(snapshot)
        return await self._read_job(job_id)

    async def wait(
        self, job_id: str, *, poll_interval: float = 0.05
    ) -> dict[str, Any]:
        del poll_interval
        payload = await self.get(job_id)
        if not payload:
            raise KeyError(job_id)
        if payload.get("status") in TERMINAL_STATUSES:
            event = self._execution_done.get(job_id)
            if event is not None:
                await event.wait()
            return payload
        future = self._completion.get(job_id)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            self._completion[job_id] = future
        result = dict(await asyncio.shield(future))
        event = self._execution_done.get(job_id)
        if event is not None:
            await event.wait()
        return result

    async def list(self, *, scope: str = "active") -> list[dict[str, Any]]:
        if scope not in {"active", "finished", "all"}:
            raise ValueError("invalid job scope")
        if scope == "active":
            where = "WHERE status IN ('queued','running')"
        elif scope == "finished":
            where = "WHERE status IN ('completed','failed','cancelled')"
        else:
            where = ""
        async with self.storage.connect(system=True) as db:
            rows = await (await db.execute(f"SELECT * FROM jobs {where}")).fetchall()

        merged = {str(row["id"]): self._decode_row(row) for row in rows}
        for job_id, snapshot in self._snapshots.items():
            status = str(snapshot.get("status") or "")
            if scope == "active" and status not in ACTIVE_STATUSES:
                continue
            if scope == "finished" and status not in TERMINAL_STATUSES:
                continue
            merged[job_id] = dict(snapshot)

        items = list(merged.values())
        if scope == "active":
            items.sort(key=lambda item: float(item.get("created_at") or 0.0))
        elif scope == "finished":
            items.sort(
                key=lambda item: float(item.get("updated_at") or 0.0),
                reverse=True,
            )
        else:
            items.sort(
                key=lambda item: (
                    0 if item.get("status") in ACTIVE_STATUSES else 1,
                    float(item.get("created_at") or 0.0)
                    if item.get("status") in ACTIVE_STATUSES
                    else -float(item.get("updated_at") or 0.0),
                )
            )
        return items

    async def subscribe(self, job_id: str) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=SUBSCRIBER_QUEUE_SIZE
        )
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
