from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from personalityrag import app as app_module
from personalityrag.jobs import JobManager, JobStateConflict
from personalityrag.storage import Storage
from personalityrag.task_control import ResolvedJobOperation
from personalityrag.database_types import (
    LIVINGMEMORY_V8_TYPE,
    TEXT_MEDIA_V1_TYPE,
    DatabaseRef,
)
from personalityrag.task_types import task_type_registry


async def _manager(tmp_path: Path) -> JobManager:
    storage = Storage(tmp_path)
    await storage.initialize()
    jobs = JobManager(storage)
    await jobs.clear_for_startup()
    return jobs


@pytest.mark.asyncio
async def test_job_manager_uses_canonical_typed_database_identity(
    tmp_path: Path,
) -> None:
    jobs = await _manager(tmp_path)

    async def operation(progress):
        await progress(0.5, "running")
        return {"ok": True}

    job_id = await jobs.start(
        "fixture",
        operation,
        database_id="knowledge-a",
        database_type=TEXT_MEDIA_V1_TYPE,
    )
    await asyncio.wait_for(jobs.wait(job_id), timeout=1)
    payload = await jobs.get(job_id)

    assert payload is not None
    assert payload["database_id"] == "knowledge-a"
    assert payload["database_type"] == TEXT_MEDIA_V1_TYPE
    assert payload["knowledge_base_id"] == "knowledge-a"
    assert payload["knowledge_base_type"] == TEXT_MEDIA_V1_TYPE
    assert payload["database_resource_key"] == "text_media_v1:knowledge-a"
    assert payload["library_id"] == "text_media_v1:knowledge-a"

    with pytest.raises(ValueError, match="deprecated library_id"):
        await jobs.start(
            "fixture",
            operation,
            database_id="canonical",
            library_id="legacy-conflict",
        )
    await jobs.close()


@pytest.mark.asyncio
async def test_finished_index_job_exposes_durable_detail_and_state_comparison(
    tmp_path: Path,
) -> None:
    jobs = await _manager(tmp_path)
    database_state = {
        "database": {
            "exists": True,
            "database_type": LIVINGMEMORY_V8_TYPE,
            "database_id": "detail-library",
        },
        "statistics": {"total_memories": 2},
        "indexes": {"generation": "gen-before", "document_vectors": 2},
    }

    async def snapshot_provider(database_id: str, kind: str):
        assert database_id == "detail-library"
        assert kind == "index_rebuild"
        return {
            "database": dict(database_state["database"]),
            "statistics": dict(database_state["statistics"]),
            "indexes": dict(database_state["indexes"]),
        }

    async def run(context):
        await context.progress(0.5, "rebuilding")
        database_state["indexes"] = {
            "generation": "gen-after",
            "document_vectors": 3,
        }
        database_state["statistics"] = {"total_memories": 3}
        return {
            "generation": "gen-after",
            "path": r"D:\private\exports\result.tmkb",
            "download_token": "do-not-show",
            "content": "sensitive result body",
            "size_bytes": 1234,
        }

    async def rollback(_context):
        raise AssertionError("rollback should not run")

    jobs.set_database_state_provider(snapshot_provider)
    jobs.set_operation_resolver(
        lambda _job: ResolvedJobOperation(run=run, rollback=rollback)
    )
    job_id = await jobs.start_resumable(
        "index_rebuild",
        {
            "provider_id": "embedding-test",
            "source_path": r"D:\private\uploads\fixture.db",
            "backup_source_db": r"D:\private\backups\source.db",
            "api_key": "must-not-leak",
            "provider_api_key": "also-must-not-leak",
            "content": "sensitive body",
            "payload_bytes": 2048,
            "_internal_validation": {"secret": "hidden"},
        },
        database_id="detail-library",
    )
    completed = await asyncio.wait_for(jobs.wait(job_id), timeout=1)
    assert completed["status"] == "completed"

    detail = await jobs.get(job_id, detail=True)
    assert detail is not None
    assert detail["detail_available"] is True
    assert detail["attempt_count"] == 1
    assert detail["timing"]["started_at"] is not None
    assert detail["timing"]["finished_at"] is not None
    assert detail["timing"]["total_duration_seconds"] >= 0
    assert [item["status"] for item in detail["status_history"]] == [
        "queued",
        "running",
        "completed",
    ]
    assert detail["request_metadata"]["provider_id"] == "embedding-test"
    assert detail["request_metadata"]["source_path"] == "fixture.db"
    assert detail["request_metadata"]["backup_source_db"] == "source.db"
    assert detail["request_metadata"]["api_key"] == "[redacted]"
    assert detail["request_metadata"]["provider_api_key"] == "[redacted]"
    assert detail["request_metadata"]["payload_bytes"] == 2048
    assert detail["request_metadata"]["content"] == {
        "redacted": True,
        "characters": 14,
    }
    assert "_internal_validation" not in detail["request_metadata"]
    assert detail["result"] == {
        "generation": "gen-after",
        "path": "result.tmkb",
        "download_token": "[redacted]",
        "content": {"redacted": True, "characters": 21},
        "size_bytes": 1234,
    }
    comparison = detail["database_state_comparison"]
    assert comparison["complete"] is True
    assert comparison["changed"] is True
    assert comparison["before"]["indexes"]["generation"] == "gen-before"
    assert comparison["after"]["indexes"]["generation"] == "gen-after"
    changed_paths = {item["path"] for item in comparison["changes"]}
    assert "indexes.generation" in changed_paths
    assert "indexes.document_vectors" in changed_paths
    assert "statistics.total_memories" in changed_paths

    stored = await jobs._read_job(job_id)
    assert stored is not None
    assert stored["database_state_before"]["indexes"]["generation"] == "gen-before"
    assert stored["database_state_after"]["indexes"]["generation"] == "gen-after"
    assert stored["finished_at"] is not None
    await jobs.close()


@pytest.mark.asyncio
async def test_database_state_capture_failure_does_not_change_job_outcome(
    tmp_path: Path,
) -> None:
    jobs = await _manager(tmp_path)

    async def broken_snapshot(_database_id: str, _kind: str):
        raise RuntimeError("snapshot unavailable")

    async def operation(_progress):
        return {"ok": True}

    jobs.set_database_state_provider(broken_snapshot)
    job_id = await jobs.start(
        "index_rebuild",
        operation,
        database_id="capture-error-library",
    )
    completed = await asyncio.wait_for(jobs.wait(job_id), timeout=1)
    assert completed["status"] == "completed"
    detail = await jobs.get(job_id, detail=True)
    assert detail is not None
    comparison = detail["database_state_comparison"]
    assert comparison["available"] is False
    assert comparison["capture_errors"] == {
        "before": "snapshot unavailable",
        "after": "snapshot unavailable",
    }
    await jobs.close()


@pytest.mark.asyncio
async def test_startup_recovery_appends_durable_terminal_and_interrupted_history(
    tmp_path: Path,
) -> None:
    storage = Storage(tmp_path)
    await storage.initialize()
    first = JobManager(storage)

    async def run(_context):
        return None

    async def rollback(_context):
        return None

    first.set_operation_resolver(
        lambda _job: ResolvedJobOperation(run=run, rollback=rollback)
    )
    first._ensure_worker = lambda: None  # type: ignore[method-assign]
    queued_id = await first.start_resumable(
        "graph_rebuild", {}, database_id="queued-library"
    )
    running_id = await first.start_resumable(
        "index_rebuild", {}, database_id="running-library"
    )
    started_at = time.time() - 5
    running_history = [
        {
            "status": "queued",
            "timestamp": started_at - 1,
            "message": "等待执行",
            "reason": "",
        },
        {
            "status": "running",
            "timestamp": started_at,
            "message": "任务已启动",
            "reason": "",
        },
    ]
    async with first._connect() as db:
        await db.execute(
            """UPDATE jobs SET status='running',started_at=?,attempt_count=1,
            status_history=? WHERE id=?""",
            (
                started_at,
                json.dumps(running_history, ensure_ascii=False),
                running_id,
            ),
        )
        await db.commit()

    second = JobManager(storage)
    second.set_operation_resolver(
        lambda _job: ResolvedJobOperation(run=run, rollback=rollback)
    )
    await second.recover_for_startup()
    queued = await second.get(queued_id, detail=True)
    interrupted = await second.get(running_id, detail=True)

    assert queued is not None
    assert queued["status"] == "cancelled"
    assert [item["status"] for item in queued["status_history"]] == [
        "queued",
        "cancelled",
    ]
    assert queued["timing"]["finished_at"] is not None
    assert interrupted is not None
    assert interrupted["status"] == "interrupted"
    assert [item["status"] for item in interrupted["status_history"]] == [
        "queued",
        "running",
        "interrupted",
    ]
    assert interrupted["timing"]["finished_at"] is None

    first._closed = True
    first._drain_queue()
    await second.close()


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
    assert not jobs._simple_tasks
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
async def test_finished_job_pages_are_stable_and_summary_only(tmp_path: Path) -> None:
    jobs = await _manager(tmp_path)
    ids = []

    async def operation(progress):
        await progress(1.0, "done")
        return {"large_detail": "x" * 4096}

    for index in range(5):
        job_id = await jobs.start(
            f"fixture-{index}",
            operation,
            library_id="paged-library",
        )
        ids.append(job_id)
        await asyncio.wait_for(jobs.wait(job_id), timeout=1)

    expected = sorted(
        await asyncio.gather(*(jobs.get(job_id) for job_id in ids)),
        key=lambda item: (-float(item["updated_at"]), str(item["id"])),
    )
    first, cursor = await jobs.list_page(scope="finished", limit=2)
    second, next_cursor = await jobs.list_page(
        scope="finished",
        limit=2,
        cursor=cursor,
    )

    assert [item["id"] for item in first + second] == [
        item["id"] for item in expected[:4]
    ]
    assert cursor is not None
    assert next_cursor is not None
    assert all("result" not in item for item in first + second)
    assert all("request_metadata" not in item for item in first + second)
    detailed, _ = await jobs.list_page(
        scope="finished",
        limit=1,
        include_details=True,
    )
    assert detailed[0]["result"]["large_detail"]["truncated"] is True
    assert detailed[0]["result"]["large_detail"]["characters"] == 4096
    assert "request_metadata" in detailed[0]
    legacy = await jobs.list(scope="finished")
    assert legacy[0]["result"]["large_detail"]["truncated"] is True
    assert "request_metadata" in legacy[0]
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
async def test_resumable_pause_blocks_same_library_but_not_other_library(
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
    await asyncio.wait_for(queued_started.wait(), timeout=1)
    assert (await jobs.wait(queued_id))["status"] == "completed"

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
    await jobs.close()


@pytest.mark.asyncio
async def test_simple_jobs_are_serial_per_library_and_parallel_across_libraries(
    tmp_path: Path,
) -> None:
    jobs = await _manager(tmp_path)
    first_started = asyncio.Event()
    other_started = asyncio.Event()
    release = asyncio.Event()
    order: list[str] = []

    async def first(progress):
        order.append("first-start")
        first_started.set()
        await release.wait()
        order.append("first-end")

    async def second(progress):
        order.append("second")

    async def other(progress):
        order.append("other")
        other_started.set()

    first_id = await jobs.start(
        "memory_create", first, library_id="library-a", dedupe_active=False
    )
    second_id = await jobs.start(
        "memory_update", second, library_id="library-a", dedupe_active=False
    )
    other_id = await jobs.start(
        "memory_create", other, library_id="library-b", dedupe_active=False
    )
    await asyncio.wait_for(first_started.wait(), timeout=1)
    await asyncio.wait_for(other_started.wait(), timeout=1)
    assert "second" not in order
    assert (await jobs.wait(other_id))["status"] == "completed"

    release.set()
    assert (await jobs.wait(first_id))["status"] == "completed"
    assert (await jobs.wait(second_id))["status"] == "completed"
    assert order.index("first-end") < order.index("second")
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
async def test_active_long_job_ignores_continuous_rebuilds_and_tracks_imports(
    tmp_path: Path,
) -> None:
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
    rebuild_id = await jobs.start(
        "index_rebuild",
        operation,
        library_id="linked",
        dedupe_active=False,
    )
    assert await jobs.active_long_job("linked") is None
    long_id = await jobs.start(
        "livingmemory_import",
        operation,
        library_id="linked",
        dedupe_active=False,
    )
    busy = await jobs.active_long_job("linked")
    assert busy and busy["id"] == long_id
    assert busy["id"] != short_id

    release.set()
    await asyncio.wait_for(jobs.wait(short_id), timeout=1)
    await asyncio.wait_for(jobs.wait(rebuild_id), timeout=1)
    await asyncio.wait_for(jobs.wait(long_id), timeout=1)
    await jobs.close()


@pytest.mark.asyncio
async def test_typed_task_registry_keeps_text_media_busy_scope_isolated(
    tmp_path: Path,
) -> None:
    jobs = await _manager(tmp_path)
    release = asyncio.Event()
    started = asyncio.Event()

    async def operation(context):
        started.set()
        await release.wait()

    async def rollback(context):
        return None

    jobs.set_operation_resolver(
        lambda _job: ResolvedJobOperation(run=operation, rollback=rollback),
        database_type=TEXT_MEDIA_V1_TYPE,
    )

    job_id = await jobs.start_resumable(
        "text_media_index_rebuild",
        {},
        library_id="shared-id",
        database_type=TEXT_MEDIA_V1_TYPE,
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    typed = await jobs.active_long_job(
        DatabaseRef(TEXT_MEDIA_V1_TYPE, "shared-id")
    )
    livingmemory = await jobs.active_long_job("shared-id")
    assert typed and typed["id"] == job_id
    assert livingmemory is None
    assert typed["task_type"] == {
        "lane": "long",
        "resumable": True,
        "adapter_blocking": True,
        "runtime_pause": True,
        "read_only": False,
        "embedding_context_policy": "probe_each_long_task",
        "database_state_comparison": True,
    }

    definitions = {
        item.kind: item for item in task_type_registry.list(TEXT_MEDIA_V1_TYPE)
    }
    assert definitions["text_media_document_ingest"].resumable is True
    assert (
        definitions["text_media_document_ingest"].embedding_context_policy
        == "probe_each_long_task"
    )
    assert (
        task_type_registry.get(
            LIVINGMEMORY_V8_TYPE, "memory_create"
        ).embedding_context_policy
        == "trust_valid_config"
    )
    assert definitions["tmkb_import"].resumable is True
    assert definitions["tmkbs_import"].resumable is True
    assert definitions["text_media_image_upload"].lane == "short"
    visual_policy_task = definitions[
        "text_media_visual_intent_policy_update"
    ]
    assert visual_policy_task.lane == "short"
    assert visual_policy_task.resumable is False
    assert visual_policy_task.adapter_blocking is False
    assert visual_policy_task.embedding_context_policy == "none"
    assert definitions["library_backup"].read_only is True
    assert definitions["library_backup"].runtime_pause is False

    release.set()
    assert (await jobs.wait(job_id))["status"] == "completed"
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
        "/api/v1/memory-libraries/livingmemory_v8/Default/recall",
            headers=headers,
            json={"query": "fixture", "k": 1},
        )
        unauthorized = await client.post(
        "/api/v1/memory-libraries/livingmemory_v8/Default/recall",
            headers={**headers, "Authorization": "Bearer invalid"},
            json={"query": "fixture", "k": 1},
        )

    assert busy.status_code == 409
    assert busy.json()["detail"]["code"] == "library_busy"
    assert unauthorized.status_code == 401
    assert calls == ["Default"]
