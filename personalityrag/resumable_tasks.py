from __future__ import annotations

import json
import hashlib
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .io_utils import run_blocking
from .migration import sqlite_backup
from .task_control import JobExecutionContext, ResolvedJobOperation

if TYPE_CHECKING:
    from .libraries import LibraryManager


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sqlite_logical_state(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    counts: dict[str, int] = {}
    connection = sqlite3.connect(path)
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        tables = [
            str(row[0])
            for row in connection.execute(
                """SELECT name FROM sqlite_master WHERE type='table'
                AND name NOT LIKE 'sqlite_%' ORDER BY name"""
            ).fetchall()
        ]
        for table in tables:
            quoted = table.replace('"', '""')
            try:
                rows = connection.execute(
                    f'SELECT * FROM "{quoted}" ORDER BY rowid'
                ).fetchall()
            except sqlite3.OperationalError:
                columns = [
                    str(row[1]).replace('"', '""')
                    for row in connection.execute(
                        f'PRAGMA table_info("{quoted}")'
                    ).fetchall()
                ]
                order_by = ",".join(f'"{column}"' for column in columns)
                query = f'SELECT * FROM "{quoted}"'
                if order_by:
                    query += f" ORDER BY {order_by}"
                rows = connection.execute(query).fetchall()
            counts[table] = len(rows)
            digest.update(table.encode("utf-8"))
            digest.update(b"\0")
            for row in rows:
                digest.update(repr(tuple(row)).encode("utf-8"))
                digest.update(b"\0")
    finally:
        connection.close()
    return {"sha256": digest.hexdigest(), "counts": counts}


class ResumableLibraryTasks:
    def __init__(self, manager: "LibraryManager"):
        self.manager = manager

    def workspace(self, job: dict[str, Any]) -> Path:
        return (
            self.manager.data_dir
            / "libraries"
            / str(job["library_id"])
            / "task_checkpoints"
            / str(job["id"])
        )

    async def resolve(self, job: dict[str, Any]) -> ResolvedJobOperation:
        kind = str(job.get("kind") or "")
        if kind == "index_rebuild":
            run = self._run_index_rebuild
        elif kind == "graph_rebuild":
            run = self._run_graph_rebuild
        elif kind == "livingmemory_import":
            run = self._run_import
        elif kind == "livingmemory_migration":
            run = self._run_migration
        else:
            raise RuntimeError(f"unsupported resumable job kind: {kind}")
        return ResolvedJobOperation(
            run=run,
            rollback=self._rollback,
            cancel_queued=self._cancel_queued,
            finalize_completed=self._finalize_completed,
        )

    async def _job(self, context: JobExecutionContext) -> dict[str, Any]:
        return await context.job()

    async def _prepare_prestate(
        self, context: JobExecutionContext
    ) -> tuple[dict[str, Any], Path]:
        job = await self._job(context)
        library_id = str(job["library_id"])
        workspace = self.workspace(job)
        state_path = workspace / "prestate.json"
        if state_path.exists():
            return json.loads(state_path.read_text(encoding="utf-8")), workspace

        workspace.mkdir(parents=True, exist_ok=True)
        rollback_dir = workspace / "rollback"
        rollback_dir.mkdir(parents=True, exist_ok=True)
        library_dir = self.manager.data_dir / "libraries" / library_id
        target_db = library_dir / "livingmemory.db"
        conversations_db = library_dir / "conversations.db"
        record = await self.manager.control.get_library(library_id)
        if record is None:
            raise KeyError(library_id)
        if target_db.exists():
            await run_blocking(sqlite_backup, target_db, rollback_dir / "livingmemory.db")
        if conversations_db.exists():
            await run_blocking(
                sqlite_backup,
                conversations_db,
                rollback_dir / "conversations.db",
            )
        current_file = library_dir / "indexes" / "CURRENT"
        current_generation = (
            current_file.read_text(encoding="utf-8").strip()
            if current_file.exists()
            else ""
        )
        rollback_database = rollback_dir / "livingmemory.db"
        rollback_conversations = rollback_dir / "conversations.db"
        index_hashes: dict[str, str] = {}
        if current_generation:
            generation_dir = library_dir / "indexes" / current_generation
            for filename in ("manifest.json", "documents.index", "graph.index"):
                path = generation_dir / filename
                if path.exists():
                    index_hashes[filename] = await run_blocking(_sha256_file, path)
        db = await self.manager.control.connect()
        try:
            bindings = [
                dict(row)
                for row in await (
                    await db.execute(
                        "SELECT * FROM library_generation_bindings WHERE library_id=?",
                        (library_id,),
                    )
                ).fetchall()
            ]
            generations = [
                dict(row)
                for row in await (
                    await db.execute(
                        "SELECT * FROM index_generations WHERE library_id=?",
                        (library_id,),
                    )
                ).fetchall()
            ]
        finally:
            await db.close()
        state = {
            "version": 1,
            "library_id": library_id,
            "database_existed": target_db.exists(),
            "conversations_existed": conversations_db.exists(),
            "current_generation": current_generation,
            "current_generation_hashes": index_hashes,
            "rollback_database_sha256": (
                await run_blocking(_sha256_file, rollback_database)
                if rollback_database.exists()
                else ""
            ),
            "rollback_database_state": (
                await run_blocking(_sqlite_logical_state, rollback_database)
                if rollback_database.exists()
                else None
            ),
            "rollback_conversations_sha256": (
                await run_blocking(_sha256_file, rollback_conversations)
                if rollback_conversations.exists()
                else ""
            ),
            "rollback_conversations_state": (
                await run_blocking(_sqlite_logical_state, rollback_conversations)
                if rollback_conversations.exists()
                else None
            ),
            "library": record.public(),
            "bindings": bindings,
            "index_generations": generations,
            "created_at": time.time(),
        }
        _atomic_json(state_path, state)
        await context.checkpoint(
            {
                "phase": "prestate_saved",
                "completed_documents": 0,
                "completed_graph_entries": 0,
                "total_documents": 0,
                "total_graph_entries": 0,
                "saved_at": time.time(),
            },
            progress=0.01,
            message="任务执行前状态已安全备份",
        )
        return state, workspace

    async def _controlled_progress(
        self, context: JobExecutionContext, value: float, message: str
    ) -> None:
        await context.progress(value, message)
        await context.control_point()

    async def _run_index_rebuild(self, context: JobExecutionContext) -> dict[str, Any]:
        job = await self._job(context)
        spec = dict(job.get("operation") or {})
        _, workspace = await self._prepare_prestate(context)
        result = await self.manager.rebuild_library(
            str(job["library_id"]),
            spec.get("provider_id"),
            lambda value, message: self._controlled_progress(
                context, value, message
            ),
            task_context=context,
            checkpoint_dir=workspace / "index",
        )
        return result

    async def _run_graph_rebuild(self, context: JobExecutionContext) -> dict[str, Any]:
        job = await self._job(context)
        _, workspace = await self._prepare_prestate(context)
        result = await self.manager.rebuild_graph(
            str(job["library_id"]),
            lambda value, message: self._controlled_progress(
                context, value, message
            ),
            task_context=context,
            checkpoint_dir=workspace / "index",
        )
        return result

    async def _run_import(self, context: JobExecutionContext) -> dict[str, Any]:
        job = await self._job(context)
        spec = dict(job.get("operation") or {})
        _, workspace = await self._prepare_prestate(context)
        result = await self.manager.import_livingmemory_db(
            str(job["library_id"]),
            Path(str(spec["source_db"])),
            lambda value, message: self._controlled_progress(
                context, value, message
            ),
            conversations_db=(
                Path(str(spec["conversations_db"]))
                if spec.get("conversations_db")
                else None
            ),
            task_context=context,
            checkpoint_dir=workspace,
        )
        return result

    async def _run_migration(self, context: JobExecutionContext) -> dict[str, Any]:
        job = await self._job(context)
        spec = dict(job.get("operation") or {})
        _, workspace = await self._prepare_prestate(context)
        runtime = await self.manager.get_runtime(str(job["library_id"]))
        phase_path = workspace / "migration-state.json"
        phase = (
            json.loads(phase_path.read_text(encoding="utf-8"))
            if phase_path.exists()
            else {"phase": "created"}
        )
        if phase.get("phase") == "created":
            async def migration_progress(value: float, message: str) -> None:
                # The legacy migrator is transaction-oriented rather than
                # batch-resumable. Report progress, but only honor control at
                # the completed migration boundary below.
                await context.progress(value * 0.2, message)

            result = await runtime.migrator.migrate(
                Path(str(spec["source_path"])),
                mode=str(spec.get("mode") or "copy"),
                progress=migration_progress,
            )
            phase = {
                "phase": "migration_completed",
                "result": result,
                "saved_at": time.time(),
            }
            _atomic_json(phase_path, phase)
            await context.checkpoint(
                {
                    "phase": "migration_completed",
                    "saved_at": phase["saved_at"],
                },
                progress=0.2,
                message="迁移事务已完成并保存安全断点",
            )
        else:
            result = dict(phase.get("result") or {})
        await runtime.storage.initialize()
        rebuild = await self.manager.rebuild_library(
            str(job["library_id"]),
            None,
            lambda value, message: self._controlled_progress(
                context, 0.2 + value * 0.8, message
            ),
            task_context=context,
            checkpoint_dir=workspace / "index",
        )
        return {"migration": result, "rebuild": rebuild}

    async def _finalize_completed(self, context: JobExecutionContext) -> None:
        job = await self._job(context)
        await self._cleanup_completed(job, self.workspace(job))

    async def _cleanup_completed(self, job: dict[str, Any], workspace: Path) -> None:
        spec = dict(job.get("operation") or {})
        for key in ("source_db", "conversations_db"):
            if spec.get(key):
                Path(str(spec[key])).unlink(missing_ok=True)
        shutil.rmtree(workspace, ignore_errors=True)

    async def _cancel_queued(self, context: JobExecutionContext) -> None:
        job = await self._job(context)
        spec = dict(job.get("operation") or {})
        for key in ("source_db", "conversations_db"):
            if spec.get(key):
                Path(str(spec[key])).unlink(missing_ok=True)

    async def _rollback(self, context: JobExecutionContext) -> None:
        job = await self._job(context)
        library_id = str(job["library_id"])
        workspace = self.workspace(job)
        state_path = workspace / "prestate.json"
        if not state_path.exists():
            await self._cancel_queued(context)
            return
        state = json.loads(state_path.read_text(encoding="utf-8"))
        library_dir = self.manager.data_dir / "libraries" / library_id
        rollback_dir = workspace / "rollback"
        target_db = library_dir / "livingmemory.db"
        conversations_db = library_dir / "conversations.db"
        await self.manager.unload_runtime(library_id, reason="task_rollback")

        async def restore_db(
            target: Path,
            backup: Path,
            existed: bool,
            expected_sha256: str,
            expected_state: dict[str, Any] | None,
        ) -> None:
            for suffix in ("-wal", "-shm"):
                Path(str(target) + suffix).unlink(missing_ok=True)
            if existed:
                if not backup.exists():
                    raise RuntimeError(f"rollback database is missing: {backup.name}")
                if expected_sha256 and await run_blocking(_sha256_file, backup) != expected_sha256:
                    raise RuntimeError(f"rollback database hash mismatch: {backup.name}")
                await run_blocking(sqlite_backup, backup, target)
                if expected_state and await run_blocking(_sqlite_logical_state, target) != expected_state:
                    raise RuntimeError(f"rollback database content mismatch: {backup.name}")
            else:
                target.unlink(missing_ok=True)

        await restore_db(
            target_db,
            rollback_dir / "livingmemory.db",
            bool(state.get("database_existed")),
            str(state.get("rollback_database_sha256") or ""),
            state.get("rollback_database_state"),
        )
        await restore_db(
            conversations_db,
            rollback_dir / "conversations.db",
            bool(state.get("conversations_existed")),
            str(state.get("rollback_conversations_sha256") or ""),
            state.get("rollback_conversations_state"),
        )

        index_root = library_dir / "indexes"
        current_file = index_root / "CURRENT"
        old_generation = str(state.get("current_generation") or "")
        if old_generation:
            pointer = current_file.with_suffix(".rollback.tmp")
            pointer.write_text(old_generation, encoding="utf-8")
            os.replace(pointer, current_file)
            generation_dir = index_root / old_generation
            for filename, expected_hash in dict(
                state.get("current_generation_hashes") or {}
            ).items():
                path = generation_dir / filename
                if not path.exists() or await run_blocking(_sha256_file, path) != expected_hash:
                    raise RuntimeError(f"pre-task index generation changed: {filename}")
        else:
            current_file.unlink(missing_ok=True)
        index_state = workspace / "index" / "checkpoint.json"
        if index_state.exists():
            generation = str(
                json.loads(index_state.read_text(encoding="utf-8")).get(
                    "generation", ""
                )
            )
            if generation and generation != old_generation:
                shutil.rmtree(index_root / generation, ignore_errors=True)
                shutil.rmtree(index_root / f".{generation}.tmp", ignore_errors=True)

        library = dict(state["library"])
        db = await self.manager.control.connect()
        try:
            await db.execute(
                """UPDATE libraries SET provider_id=?,provider_revision=?,
                metadata_json=?,updated_at=? WHERE id=?""",
                (
                    library["provider_id"],
                    int(library["provider_revision"]),
                    json.dumps(library.get("metadata") or {}, ensure_ascii=False),
                    float(library["updated_at"]),
                    library_id,
                ),
            )
            await db.execute(
                "DELETE FROM library_generation_bindings WHERE library_id=?",
                (library_id,),
            )
            for row in state.get("bindings", []):
                await db.execute(
                    """INSERT INTO library_generation_bindings
                    (library_id,generation,provider_id,provider_revision,
                    manifest_json,activated_at) VALUES(?,?,?,?,?,?)""",
                    (
                        row["library_id"],
                        row["generation"],
                        row["provider_id"],
                        row["provider_revision"],
                        row["manifest_json"],
                        row["activated_at"],
                    ),
                )
            await db.execute(
                "DELETE FROM index_generations WHERE library_id=?", (library_id,)
            )
            for row in state.get("index_generations", []):
                await db.execute(
                    """INSERT INTO index_generations
                    (library_id,generation,status,manifest,created_at,activated_at)
                    VALUES(?,?,?,?,?,?)""",
                    (
                        row["library_id"],
                        row["generation"],
                        row["status"],
                        row["manifest"],
                        row["created_at"],
                        row["activated_at"],
                    ),
                )
            await db.commit()
        finally:
            await db.close()
        await self.manager.get_runtime(library_id)
        await self._cleanup_completed(job, workspace)
