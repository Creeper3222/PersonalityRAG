from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from personalityrag import app as app_module
from personalityrag.jobs import JobManager
from personalityrag.storage import Storage


async def _manager(tmp_path: Path) -> JobManager:
    storage = Storage(tmp_path)
    await storage.initialize()
    jobs = JobManager(storage)
    await jobs.clear_for_startup()
    return jobs


@pytest.mark.asyncio
async def test_wait_uses_completion_future_without_sqlite_polling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = await _manager(tmp_path)
    release = asyncio.Event()
    reads = 0
    original = jobs._read_job

    async def counted_read(job_id: str):
        nonlocal reads
        reads += 1
        return await original(job_id)

    monkeypatch.setattr(jobs, "_read_job", counted_read)

    async def operation(progress):
        await progress(0.5, "waiting")
        await release.wait()
        return {"ok": True}

    job_id = await jobs.start("fixture", operation, library_id="library")
    waiter = asyncio.create_task(jobs.wait(job_id))
    await asyncio.sleep(0.15)
    assert reads == 0
    assert not waiter.done()

    release.set()
    result = await asyncio.wait_for(waiter, timeout=1)
    assert result["status"] == "completed"
    assert result["result"] == {"ok": True}
    assert reads == 0
    await jobs.close()


@pytest.mark.asyncio
async def test_progress_persistence_is_throttled_but_terminal_is_immediate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = await _manager(tmp_path)
    persisted: list[dict] = []
    original = jobs._persist_snapshot

    async def counted_persist(snapshot: dict):
        persisted.append(dict(snapshot))
        await original(snapshot)

    monkeypatch.setattr(jobs, "_persist_snapshot", counted_persist)

    async def operation(progress):
        for index in range(100):
            await progress(index / 100, f"step {index}")
        return "done"

    job_id = await jobs.start("fixture", operation)
    result = await asyncio.wait_for(jobs.wait(job_id), timeout=1)

    assert result["status"] == "completed"
    assert persisted[-1]["status"] == "completed"
    assert len(persisted) <= 3
    stored = await jobs._read_job(job_id)
    assert stored and stored["status"] == "completed"
    assert stored["progress"] == 1.0
    await jobs.close()


@pytest.mark.asyncio
async def test_slow_subscriber_keeps_only_latest_progress(tmp_path: Path) -> None:
    jobs = await _manager(tmp_path)
    begin = asyncio.Event()
    emitted = asyncio.Event()
    release = asyncio.Event()

    async def operation(progress):
        await begin.wait()
        for index in range(50):
            await progress(index / 50, f"step {index}")
        emitted.set()
        await release.wait()

    job_id = await jobs.start("fixture", operation)
    stream = jobs.subscribe(job_id)
    first = await anext(stream)
    assert first["id"] == job_id
    begin.set()
    await asyncio.wait_for(emitted.wait(), timeout=1)

    queues = list(jobs.subscribers[job_id])
    assert len(queues) == 1
    assert queues[0].qsize() == 1
    latest = queues[0].get_nowait()
    assert latest["message"] == "step 49"
    queues[0].task_done()

    release.set()
    await asyncio.wait_for(jobs.wait(job_id), timeout=1)
    await stream.aclose()
    await jobs.close()


@pytest.mark.asyncio
async def test_worker_survives_progress_persistence_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = await _manager(tmp_path)
    original = jobs._persist_snapshot
    failed_once = False

    async def flaky_persist(snapshot: dict):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("fixture persistence outage")
        await original(snapshot)

    monkeypatch.setattr(jobs, "_persist_snapshot", flaky_persist)

    async def operation(progress):
        await progress(0.5, "working")
        return "ok"

    first = await jobs.start("fixture-a", operation)
    second = await jobs.start("fixture-b", operation)
    first_result = await asyncio.wait_for(jobs.wait(first), timeout=1)
    second_result = await asyncio.wait_for(jobs.wait(second), timeout=1)

    assert failed_once is True
    assert first_result["status"] == "completed"
    assert second_result["status"] == "completed"
    assert jobs._worker_task is not None and not jobs._worker_task.done()
    await jobs.close()


@pytest.mark.asyncio
async def test_startup_clears_all_persisted_jobs(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    await storage.initialize()
    now = time.time()
    old = now - 31 * 24 * 60 * 60
    rows = [
        ("stale-running", "lib", "fixture", "running", now, now),
        ("expired", "lib", "fixture", "completed", old, old),
        ("recent-finished", "lib", "fixture", "completed", now - 30, now - 30),
    ]
    async with storage.connect(system=True) as db:
        await db.executemany(
            """INSERT INTO jobs
            (id,library_id,kind,status,progress,message,created_at,updated_at)
            VALUES(?,?,?,?,0,'fixture',?,?)""",
            rows,
        )
        await db.commit()

    jobs = JobManager(storage)
    await jobs.clear_for_startup()
    history = await jobs.list(scope="all")

    assert history == []
    async with storage.connect(system=True) as db:
        remaining = await (
            await db.execute("SELECT COUNT(*) AS count FROM jobs")
        ).fetchone()
    assert remaining["count"] == 0
    await jobs.close()


@pytest.mark.asyncio
async def test_close_cancels_waiters_and_clears_persisted_jobs(
    tmp_path: Path,
) -> None:
    jobs = await _manager(tmp_path)
    release = asyncio.Event()

    async def running(progress):
        await progress(0.5, "working")
        await release.wait()

    async def queued(progress):
        await progress(0.1, "queued")
        return None

    running_id = await jobs.start("running", running)
    queued_id = await jobs.start("queued", queued)
    running_wait = asyncio.create_task(jobs.wait(running_id))
    queued_wait = asyncio.create_task(jobs.wait(queued_id))

    await asyncio.sleep(0.1)
    await jobs.close()

    running_result = await asyncio.wait_for(running_wait, timeout=1)
    queued_result = await asyncio.wait_for(queued_wait, timeout=1)
    assert running_result["status"] == "cancelled"
    assert queued_result["status"] == "cancelled"

    async with jobs.storage.connect(system=True) as db:
        remaining = await (
            await db.execute("SELECT COUNT(*) AS count FROM jobs")
        ).fetchone()
    assert remaining["count"] == 0


@pytest.mark.asyncio
async def test_active_long_job_prefers_current_memory_snapshot(tmp_path: Path) -> None:
    jobs = await _manager(tmp_path)
    release = asyncio.Event()

    async def operation(progress):
        await release.wait()

    short_id = await jobs.start(
        "memory_create",
        operation,
        library_id="linked",
        dedupe_active=False,
    )
    assert await jobs.active_long_job("linked") is None
    long_id = await jobs.start(
        "index_rebuild",
        operation,
        library_id="linked",
        dedupe_active=False,
    )
    busy = await jobs.active_long_job("linked")
    assert busy and busy["id"] == long_id
    assert busy["id"] != short_id

    release.set()
    await asyncio.wait_for(jobs.wait(short_id), timeout=1)
    await asyncio.wait_for(jobs.wait(long_id), timeout=1)
    await jobs.close()


def test_webui_tracks_submitted_jobs_with_sse_and_polling_fallback() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "static"
        / "modules"
        / "tasks-logs.js"
    ).read_text(encoding="utf-8")
    assert "new EventSource(" in source
    assert "task polling fallback failed" in source


@pytest.mark.asyncio
async def test_adapter_busy_guard_checks_auth_then_uses_in_memory_job_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeJobs:
        async def active_long_job(self, library_id: str):
            calls.append(library_id)
            return {
                "id": "busy-job",
                "library_id": library_id,
                "kind": "index_rebuild",
                "status": "running",
                "progress": 0.5,
                "message": "fixture",
            }

    monkeypatch.setattr(app_module.manager, "jobs", FakeJobs())
    headers = {
        "X-PersonalityRAG-Adapter-ID": "Astrbot",
        "X-PersonalityRAG-Adapter-Instance-ID": "fixture",
        "Authorization": f"Bearer {app_module.config.api_key}",
    }
    transport = ASGITransport(app=app_module.app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test:8765",
    ) as client:
        busy = await client.post(
            "/api/v1/libraries/Default/recall",
            headers=headers,
            json={"query": "fixture", "k": 1},
        )
        unauthorized = await client.post(
            "/api/v1/libraries/Default/recall",
            headers={**headers, "Authorization": "Bearer invalid"},
            json={"query": "fixture", "k": 1},
        )

    assert busy.status_code == 409
    assert busy.json()["detail"]["code"] == "library_busy"
    assert unauthorized.status_code == 401
    assert calls == ["Default"]
