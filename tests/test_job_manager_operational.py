from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from personalityrag import app as app_module
from personalityrag.jobs import JobManager, JobStateConflict
from personalityrag.storage import Storage
from personalityrag.task_control import ResolvedJobOperation


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
async def test_startup_recovers_recent_history_and_expires_old_jobs(tmp_path: Path) -> None:
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

    assert [item["id"] for item in history] == ["stale-running", "recent-finished"]
    assert history[0]["status"] == "cancelled"
    assert history[0]["status_reason"] == "process_interrupted"
    async with storage.connect(system=True) as db:
        remaining = await (
            await db.execute("SELECT COUNT(*) AS count FROM jobs")
        ).fetchone()
    assert remaining["count"] == 2
    await jobs.close()


@pytest.mark.asyncio
async def test_clear_finished_removes_only_terminal_history(tmp_path: Path) -> None:
    jobs = await _manager(tmp_path)
    release = asyncio.Event()

    async def running(progress):
        await progress(0.5, "working")
        await release.wait()

    async def finished(progress):
        await progress(1.0, "done")
        return {"ok": True}

    finished_id = await jobs.start("finished", finished, library_id="finished-lib")
    await asyncio.wait_for(jobs.wait(finished_id), timeout=1)
    active_id = await jobs.start("running", running, library_id="active-lib")

    cleared = await jobs.clear_finished()
    assert cleared == 1
    assert await jobs.get(finished_id) is None
    active = await jobs.get(active_id)
    assert active and active["status"] == "running"
    assert [item["id"] for item in await jobs.list(scope="finished")] == []
    assert [item["id"] for item in await jobs.list(scope="active")] == [active_id]

    release.set()
    await asyncio.wait_for(jobs.wait(active_id), timeout=1)
    await jobs.close()


@pytest.mark.asyncio
async def test_close_cancels_simple_jobs_and_retains_finished_history(
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
    assert remaining["count"] == 2


@pytest.mark.asyncio
async def test_resumable_job_pause_blocks_queue_then_resumes_same_id(
    tmp_path: Path,
) -> None:
    jobs = await _manager(tmp_path)
    reached = asyncio.Event()
    release = asyncio.Event()
    queued_started = asyncio.Event()
    attempts = 0

    async def run(context):
        nonlocal attempts
        attempts += 1
        await context.checkpoint(
            {"phase": "batch", "completed_documents": attempts},
            progress=0.25,
            message="safe batch",
        )
        reached.set()
        await release.wait()
        await context.control_point()
        return {"attempts": attempts}

    async def rollback(context):
        raise AssertionError("rollback must not run for pause/resume")

    jobs.set_operation_resolver(
        lambda job: ResolvedJobOperation(run=run, rollback=rollback)
    )
    job_id = await jobs.start_resumable("index_rebuild", {}, library_id="lib")
    await asyncio.wait_for(reached.wait(), timeout=1)
    await jobs.pause(job_id)
    release.set()

    for _ in range(50):
        paused = await jobs.get(job_id)
        if paused and paused["status"] == "paused":
            break
        await asyncio.sleep(0.02)
    assert paused and paused["status"] == "paused"
    assert paused["checkpoint"]["completed_documents"] == 1
    assert paused["capabilities"]["resume"] is True

    async def queued(progress):
        queued_started.set()
        return "after"

    queued_id = await jobs.start("after", queued, library_id="other")
    await asyncio.sleep(0.1)
    assert not queued_started.is_set()
    assert (await jobs.get(queued_id))["status"] == "queued"

    release.clear()
    await jobs.resume(job_id)
    for _ in range(50):
        if attempts == 2:
            break
        await asyncio.sleep(0.02)
    assert attempts == 2
    release.set()
    completed = await asyncio.wait_for(jobs.wait(job_id), timeout=1)
    assert completed["id"] == job_id
    assert completed["status"] == "completed"
    assert (await asyncio.wait_for(jobs.wait(queued_id), timeout=1))["status"] == "completed"
    await jobs.close()


@pytest.mark.asyncio
async def test_resumable_stop_rolls_back_and_queued_cancel_is_terminal(
    tmp_path: Path,
) -> None:
    jobs = await _manager(tmp_path)
    reached = asyncio.Event()
    release = asyncio.Event()
    rollback_calls: list[str] = []
    cancelled_specs: list[dict] = []

    async def run(context):
        await context.checkpoint({"phase": "safe"}, progress=0.5)
        reached.set()
        await release.wait()
        await context.control_point()

    async def rollback(context):
        rollback_calls.append(context.job_id)

    async def cancel_queued(context):
        cancelled_specs.append((await context.job())["operation"])

    jobs.set_operation_resolver(
        lambda job: ResolvedJobOperation(
            run=run,
            rollback=rollback,
            cancel_queued=cancel_queued,
        )
    )
    running_id = await jobs.start_resumable("index_rebuild", {}, library_id="lib")
    await asyncio.wait_for(reached.wait(), timeout=1)
    queued_id = await jobs.start_resumable(
        "livingmemory_import", {"source_db": "fixture"}, library_id="lib2"
    )
    cancelled = await jobs.cancel(queued_id)
    assert cancelled["status"] == "cancelled"
    assert cancelled["status_reason"] == "manual"
    assert cancelled_specs == [{"source_db": "fixture"}]
    with pytest.raises(JobStateConflict):
        await jobs.resume(queued_id)

    await jobs.stop(running_id)
    release.set()
    stopped = await asyncio.wait_for(jobs.wait(running_id), timeout=1)
    assert stopped["status"] == "stopped"
    assert stopped["progress"] == 0
    assert rollback_calls == [running_id]
    await jobs.close()


@pytest.mark.asyncio
async def test_shutdown_pauses_resumable_job_and_restart_requires_manual_resume(
    tmp_path: Path,
) -> None:
    storage = Storage(tmp_path)
    await storage.initialize()
    first = JobManager(storage)
    await first.recover_for_startup()
    reached = asyncio.Event()
    release = asyncio.Event()

    async def first_run(context):
        await context.checkpoint({"phase": "safe", "completed_documents": 1})
        reached.set()
        await release.wait()
        await context.control_point()

    async def rollback(context):
        return None

    first.set_operation_resolver(
        lambda job: ResolvedJobOperation(run=first_run, rollback=rollback)
    )
    job_id = await first.start_resumable("index_rebuild", {}, library_id="lib")
    queued_id = await first.start_resumable(
        "graph_rebuild", {}, library_id="other", dedupe_active=False
    )
    await asyncio.wait_for(reached.wait(), timeout=1)
    closing = asyncio.create_task(first.close())
    await asyncio.sleep(0.05)
    release.set()
    await asyncio.wait_for(closing, timeout=2)
    assert (await first.get(job_id))["status"] == "paused"
    assert (await first.get(job_id))["status_reason"] == "shutdown"
    assert (await first.get(queued_id))["status"] == "cancelled"

    resumed_runs = 0

    async def resumed_run(context):
        nonlocal resumed_runs
        resumed_runs += 1
        assert (await context.job())["checkpoint"]["phase"] == "safe"
        return {"resumed": True}

    second = JobManager(storage)
    second.set_operation_resolver(
        lambda job: ResolvedJobOperation(run=resumed_run, rollback=rollback)
    )
    await second.recover_for_startup()
    recovered = await second.get(job_id)
    assert recovered and recovered["status"] == "paused"
    await asyncio.sleep(0.05)
    assert resumed_runs == 0
    await second.resume(job_id)
    completed = await asyncio.wait_for(second.wait(job_id), timeout=1)
    assert completed["status"] == "completed"
    assert resumed_runs == 1
    await second.close()


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
