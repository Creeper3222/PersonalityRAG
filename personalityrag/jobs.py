from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict
from typing import Any, Awaitable, Callable

from .logger import logger, safe_summary
from .storage import Storage


class JobManager:
    def __init__(self, storage: Storage):
        self.storage = storage
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._operations: dict[
            str, Callable[[Callable[[float, str], Awaitable[None]]], Awaitable[Any]]
        ] = {}
        self._worker_task: asyncio.Task | None = None
        self._closed = False
        self.subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)

    async def clear_for_startup(self) -> None:
        self._operations.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        async with self.storage.connect(system=True) as db:
            await db.execute("DELETE FROM jobs")
            await db.commit()
        logger.warning("任务列表已初始化：清空上次进程遗留任务")

    async def close(self) -> None:
        self._closed = True
        if self._worker_task:
            self._worker_task.cancel()
            await asyncio.gather(self._worker_task, return_exceptions=True)
            self._worker_task = None
        self._operations.clear()
        for queues in self.subscribers.values():
            for queue in list(queues):
                await queue.put(
                    {
                        "status": "cancelled",
                        "progress": 0,
                        "message": "服务正在关闭",
                    }
                )

    async def start(
        self,
        kind: str,
        operation: Callable[[Callable[[float, str], Awaitable[None]]], Awaitable[Any]],
        *,
        library_id: str | None = None,
        dedupe_active: bool = True,
    ) -> str:
        if self._closed:
            raise RuntimeError("任务管理器已关闭")
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
        async with self.storage.connect(system=True) as db:
            await db.execute(
                """INSERT INTO jobs
                (id,library_id,kind,status,progress,message,created_at,updated_at)
                VALUES(?,?,?, 'queued',0,'等待执行',?,?)""",
                (job_id, library_id, kind, now, now),
            )
            await db.commit()
        logger.info(
            "任务已入队：job_id=%s kind=%s library_id=%s",
            job_id,
            kind,
            library_id or "",
        )
        self._operations[job_id] = operation
        await self._queue.put(job_id)
        self._ensure_worker()
        return job_id

    async def _find_active(self, kind: str, library_id: str | None) -> str | None:
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

    def _ensure_worker(self) -> None:
        if self._worker_task and not self._worker_task.done():
            return
        self._worker_task = asyncio.create_task(self._worker())

    async def _worker(self) -> None:
        logger.info("任务队列 worker 已启动")
        while not self._closed:
            job_id = await self._queue.get()
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
            finally:
                self._queue.task_done()

    async def _run(self, job_id: str, operation):
        job = await self.get(job_id) or {}
        logger.info(
            "任务开始：job_id=%s kind=%s library_id=%s",
            job_id,
            job.get("kind") or "",
            job.get("library_id") or "",
        )
        await self._update(job_id, status="running", message="任务已启动")

        async def progress(value: float, message: str):
            await self._update(
                job_id,
                status="running",
                progress=max(0.0, min(1.0, value)),
                message=message,
            )

        try:
            result = await operation(progress)
            await self._update(
                job_id,
                status="completed",
                progress=1.0,
                message="任务完成",
                result=result,
            )
            job = await self.get(job_id) or {}
            logger.info(
                "任务完成：job_id=%s kind=%s library_id=%s",
                job_id,
                job.get("kind") or "",
                job.get("library_id") or "",
            )
        except asyncio.CancelledError:
            logger.warning("任务已取消：job_id=%s", job_id)
            await self._update(job_id, status="cancelled", message="任务已取消")
            raise
        except Exception as exc:
            logger.exception("任务失败：job_id=%s err=%s", job_id, exc)
            await self._update(
                job_id, status="failed", message="任务失败", error=str(exc)
            )

    async def _update(self, job_id: str, **fields):
        allowed = {
            "status",
            "progress",
            "message",
            "result",
            "error",
        }
        fields = {key: value for key, value in fields.items() if key in allowed}
        if "result" in fields:
            fields["result"] = json.dumps(fields["result"], ensure_ascii=False)
        sets = [f"{key}=?" for key in fields]
        values = list(fields.values())
        sets.append("updated_at=?")
        values.extend([time.time(), job_id])
        async with self.storage.connect(system=True) as db:
            await db.execute(
                f"UPDATE jobs SET {','.join(sets)} WHERE id=?", values
            )
            await db.commit()
        payload = await self.get(job_id)
        if payload and {"status", "progress", "message"} & fields.keys():
            logger.info(
                "任务进度：job_id=%s kind=%s library_id=%s status=%s progress=%.1f%% message=%s",
                job_id,
                payload.get("kind") or "",
                payload.get("library_id") or "",
                payload.get("status") or "",
                float(payload.get("progress") or 0.0) * 100,
                safe_summary(payload.get("message"), max_chars=160),
            )
        for queue in list(self.subscribers.get(job_id, set())):
            await queue.put(payload)

    async def get(self, job_id: str) -> dict[str, Any] | None:
        async with self.storage.connect(system=True) as db:
            row = await (
                await db.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        if result.get("result"):
            result["result"] = json.loads(result["result"])
        return result

    async def list(self, *, scope: str = "active") -> list[dict[str, Any]]:
        if scope not in {"active", "finished", "all"}:
            raise ValueError("invalid job scope")
        if scope == "active":
            where = "WHERE status IN ('queued','running')"
            order = "ORDER BY created_at"
        elif scope == "finished":
            where = "WHERE status IN ('completed','failed','cancelled')"
            order = "ORDER BY updated_at DESC"
        else:
            where = ""
            order = (
                "ORDER BY CASE WHEN status IN ('queued','running') THEN 0 ELSE 1 END,"
                " CASE WHEN status IN ('queued','running') THEN created_at ELSE -updated_at END"
            )
        async with self.storage.connect(system=True) as db:
            rows = await (
                await db.execute(f"SELECT * FROM jobs {where} {order}")
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            if item.get("result"):
                try:
                    item["result"] = json.loads(item["result"])
                except json.JSONDecodeError:
                    item["result"] = None
            result.append(item)
        return result

    async def subscribe(self, job_id: str):
        queue: asyncio.Queue = asyncio.Queue()
        self.subscribers[job_id].add(queue)
        try:
            current = await self.get(job_id)
            if current:
                yield current
                if current.get("status") in {
                    "completed",
                    "failed",
                    "cancelled",
                }:
                    return
            while True:
                payload = await queue.get()
                yield payload
                if payload.get("status") in {
                    "completed",
                    "failed",
                    "cancelled",
                }:
                    break
        finally:
            self.subscribers[job_id].discard(queue)
