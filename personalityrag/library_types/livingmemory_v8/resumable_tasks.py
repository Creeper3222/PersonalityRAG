from __future__ import annotations

import json
import hashlib
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...io_utils import atomic_write_json, read_ab_checkpoint, run_blocking
from .migration import sqlite_backup
from ...performance import measure_phase
from ...task_control import JobExecutionContext, ResolvedJobOperation
from ...database_types import DatabaseRef, LIVINGMEMORY_V8_TYPE, database_type_registry
from .transfer import existing_transfer_keys, transfer_dedupe_key

if TYPE_CHECKING:
    from .manager import LivingMemoryV8Manager


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_database(source: Path, target: Path) -> None:
    with measure_phase(
        "resumable_task",
        "rollback_snapshot",
        bytes_count=source.stat().st_size,
    ):
        sqlite_backup(source, target)


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
                cursor = connection.execute(
                    f'SELECT * FROM "{quoted}" ORDER BY rowid'
                )
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
                cursor = connection.execute(query)
            digest.update(table.encode("utf-8"))
            digest.update(b"\0")
            count = 0
            while rows := cursor.fetchmany(1000):
                count += len(rows)
                for row in rows:
                    digest.update(repr(tuple(row)).encode("utf-8"))
                    digest.update(b"\0")
            counts[table] = count
    finally:
        connection.close()
    return {"sha256": digest.hexdigest(), "counts": counts}


class ResumableMemoryStoreTasks:
    def __init__(self, manager: "LivingMemoryV8Manager"):
        self.manager = manager

    @staticmethod
    def _memory_store_id_from_job(job: dict[str, Any]) -> str:
        """Read canonical task identity with a frozen-state compatibility fallback."""

        return str(
            job.get("memory_store_id")
            or job.get("database_id")
            or job.get("library_id")
            or ""
        )

    @staticmethod
    def _memory_store_id_from_state(state: dict[str, Any]) -> str:
        return str(
            state.get("memory_store_id")
            or state.get("database_id")
            or state.get("library_id")
            or ""
        )

    def workspace(self, job: dict[str, Any]) -> Path:
        return (
            database_type_registry.data_dir(
                self.manager.data_dir,
                DatabaseRef(
                    LIVINGMEMORY_V8_TYPE,
                    self._memory_store_id_from_job(job),
                ),
            )
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
        elif kind == "memory_transfer_import":
            run = self._run_transfer_import
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

    @staticmethod
    def database_write_set(
        kind: str,
        operation: dict[str, Any] | None = None,
    ) -> tuple[str, ...]:
        if kind == "graph_rebuild":
            return ("livingmemory.db",)
        if kind == "livingmemory_import":
            return (
                ("livingmemory.db", "conversations.db")
                if (operation or {}).get("conversations_db")
                else ("livingmemory.db",)
            )
        if kind == "livingmemory_migration":
            return ("livingmemory.db", "conversations.db")
        if kind == "memory_transfer_import":
            return ("livingmemory.db",)
        if kind == "index_rebuild":
            return ()
        raise ValueError(f"unsupported resumable job kind: {kind}")

    async def _prepare_prestate(
        self,
        context: JobExecutionContext,
        *,
        databases: tuple[str, ...] = (),
    ) -> tuple[dict[str, Any], Path]:
        job = await self._job(context)
        memory_store_id = self._memory_store_id_from_job(job)
        workspace = self.workspace(job)
        state_path = workspace / "prestate.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            await self._ensure_database_snapshots(state, workspace, databases)
            return state, workspace

        workspace.mkdir(parents=True, exist_ok=True)
        rollback_dir = workspace / "rollback"
        rollback_dir.mkdir(parents=True, exist_ok=True)
        library_dir = database_type_registry.data_dir(
            self.manager.data_dir,
            DatabaseRef(LIVINGMEMORY_V8_TYPE, memory_store_id),
        )
        target_db = library_dir / "livingmemory.db"
        conversations_db = library_dir / "conversations.db"
        record = await self.manager.control.get_library(memory_store_id)
        if record is None:
            raise KeyError(memory_store_id)
        current_file = library_dir / "indexes" / "CURRENT"
        current_generation = (
            current_file.read_text(encoding="utf-8").strip()
            if current_file.exists()
            else ""
        )
        index_directories = sorted(
            item.name
            for item in (library_dir / "indexes").iterdir()
            if item.is_dir()
        ) if (library_dir / "indexes").exists() else []
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
                        (memory_store_id,),
                    )
                ).fetchall()
            ]
            generations = [
                dict(row)
                for row in await (
                    await db.execute(
                        "SELECT * FROM index_generations WHERE library_id=?",
                        (memory_store_id,),
                    )
                ).fetchall()
            ]
        finally:
            await db.close()
        state = {
            "version": 2,
            "memory_store_id": memory_store_id,
            "database_existed": target_db.exists(),
            "conversations_existed": conversations_db.exists(),
            "rollback_database_saved": False,
            "rollback_conversations_saved": False,
            "current_generation": current_generation,
            "index_directories": index_directories,
            "current_generation_hashes": index_hashes,
            "rollback_database_sha256": "",
            "rollback_database_state": None,
            "rollback_conversations_sha256": "",
            "rollback_conversations_state": None,
            "library": record.public(),
            "bindings": bindings,
            "index_generations": generations,
            "created_at": time.time(),
        }
        await run_blocking(_atomic_json, state_path, state)
        await self._ensure_database_snapshots(state, workspace, databases)
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

    async def _ensure_database_snapshots(
        self,
        state: dict[str, Any],
        workspace: Path,
        databases: tuple[str, ...],
    ) -> None:
        fields = {
            "livingmemory.db": (
                "database_existed",
                "rollback_database_saved",
                "rollback_database_sha256",
                "rollback_database_state",
            ),
            "conversations.db": (
                "conversations_existed",
                "rollback_conversations_saved",
                "rollback_conversations_sha256",
                "rollback_conversations_state",
            ),
        }
        library_dir = database_type_registry.data_dir(
            self.manager.data_dir,
            DatabaseRef(
                LIVINGMEMORY_V8_TYPE,
                self._memory_store_id_from_state(state),
            ),
        )
        rollback_dir = workspace / "rollback"
        changed = False
        for name in databases:
            if name not in fields:
                raise ValueError(f"unsupported rollback database: {name}")
            existed_key, saved_key, hash_key, logical_key = fields[name]
            backup = rollback_dir / name
            if saved_key not in state:
                state[saved_key] = backup.exists() or not bool(state.get(existed_key))
            if bool(state.get(saved_key)):
                continue
            target = library_dir / name
            existed = target.exists()
            state[existed_key] = existed
            if existed:
                await run_blocking(_snapshot_database, target, backup)
                state[hash_key] = await run_blocking(_sha256_file, backup)
                state[logical_key] = await run_blocking(
                    _sqlite_logical_state, backup
                )
            else:
                state[hash_key] = ""
                state[logical_key] = None
            state[saved_key] = True
            changed = True
        if changed:
            await run_blocking(_atomic_json, workspace / "prestate.json", state)

    async def _controlled_progress(
        self, context: JobExecutionContext, value: float, message: str
    ) -> None:
        await context.progress(value, message)
        await context.control_point()

    async def _run_index_rebuild(self, context: JobExecutionContext) -> dict[str, Any]:
        job = await self._job(context)
        spec = dict(job.get("operation") or {})
        state, workspace = await self._prepare_prestate(context)

        async def snapshot_before_database_repair() -> None:
            await self._ensure_database_snapshots(
                state,
                workspace,
                ("livingmemory.db",),
            )

        result = await self.manager.rebuild_library(
            self._memory_store_id_from_job(job),
            spec.get("provider_id"),
            lambda value, message: self._controlled_progress(
                context, value, message
            ),
            task_context=context,
            checkpoint_dir=workspace / "index",
            before_database_repair=snapshot_before_database_repair,
        )
        return result

    async def _run_graph_rebuild(self, context: JobExecutionContext) -> dict[str, Any]:
        job = await self._job(context)
        _, workspace = await self._prepare_prestate(
            context,
            databases=self.database_write_set("graph_rebuild"),
        )
        result = await self.manager.rebuild_graph(
            self._memory_store_id_from_job(job),
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
        _, workspace = await self._prepare_prestate(
            context,
            databases=self.database_write_set("livingmemory_import", spec),
        )
        result = await self.manager.import_livingmemory_db(
            self._memory_store_id_from_job(job),
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
            source_validation_report=spec.get("_source_validation"),
            conversations_validation_report=spec.get(
                "_conversations_validation"
            ),
        )
        return result

    async def _run_migration(self, context: JobExecutionContext) -> dict[str, Any]:
        job = await self._job(context)
        spec = dict(job.get("operation") or {})
        _, workspace = await self._prepare_prestate(
            context,
            databases=self.database_write_set("livingmemory_migration", spec),
        )
        memory_store_id = self._memory_store_id_from_job(job)
        runtime = await self.manager.get_runtime(memory_store_id)
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
                preserve_previous=False,
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
            memory_store_id,
            None,
            lambda value, message: self._controlled_progress(
                context, 0.2 + value * 0.8, message
            ),
            task_context=context,
            checkpoint_dir=workspace / "index",
        )
        return {"migration": result, "rebuild": rebuild}

    async def _run_transfer_import(
        self, context: JobExecutionContext
    ) -> dict[str, Any]:
        job = await self._job(context)
        spec = dict(job.get("operation") or {})
        memory_store_id = self._memory_store_id_from_job(job)
        _, workspace = await self._prepare_prestate(
            context,
            databases=self.database_write_set("memory_transfer_import", spec),
        )
        source_path = Path(str(spec.get("manifest_path") or ""))
        expected_sha256 = str(spec.get("manifest_sha256") or "")
        archived_source = workspace / "transfer-manifest.json"
        if not archived_source.exists():
            if not source_path.is_file():
                raise RuntimeError("memory transfer manifest is missing")
            if expected_sha256 and await run_blocking(
                _sha256_file, source_path
            ) != expected_sha256:
                raise RuntimeError("memory transfer manifest fingerprint changed")
            await run_blocking(shutil.copy2, source_path, archived_source)
        if expected_sha256 and await run_blocking(
            _sha256_file, archived_source
        ) != expected_sha256:
            raise RuntimeError("archived memory transfer manifest is damaged")
        manifest = json.loads(archived_source.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or str(manifest.get("database_id") or "") != memory_store_id
            or not isinstance(manifest.get("records"), list)
        ):
            raise RuntimeError("memory transfer manifest is invalid")
        records = list(manifest["records"])
        state_path = workspace / "transfer-state.json"
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.exists()
            else {
                "version": 1,
                "cursor": 0,
                "imported_ids": [],
                "skipped": list(manifest.get("skipped") or []),
                "errors": list(manifest.get("preflight_errors") or []),
                "saved_at": time.time(),
            }
        )
        cursor = max(0, int(state.get("cursor") or 0))
        runtime = await self.manager.get_runtime(memory_store_id)
        existing_keys = existing_transfer_keys(
            await runtime.storage.memory_transfer_records()
        )
        duplicate_mode = str(manifest.get("duplicate_mode") or "skip")
        batch_size = 64
        total = max(1, len(records))
        while cursor < len(records):
            await context.control_point()
            batch = records[cursor : cursor + batch_size]
            created_ids: list[int] = []
            batch_keys: list[str] = []
            for offset, raw in enumerate(batch):
                row_number = int(raw.get("row_number") or cursor + offset + 1)
                key = str(raw.get("dedupe_key") or "") or transfer_dedupe_key(
                    raw.get("content"),
                    raw.get("session_id"),
                    raw.get("persona_id"),
                )
                if duplicate_mode == "skip" and key in existing_keys:
                    state.setdefault("skipped", []).append(
                        {
                            "preview_item_id": raw.get("preview_item_id"),
                            "row_number": row_number,
                            "reason": "duplicate_at_execution",
                        }
                    )
                    continue
                payload = {
                    key_name: value
                    for key_name, value in raw.items()
                    if key_name
                    in {
                        "content",
                        "canonical_summary",
                        "persona_summary",
                        "persona_id",
                        "session_id",
                        "importance",
                        "status",
                        "topics",
                        "participants",
                        "participant_identities",
                        "key_facts",
                        "source_messages",
                        "source_time_strategy",
                        "source_time_tags",
                        "metadata",
                    }
                }
                try:
                    memory_id = await runtime.storage.create_memory(
                        runtime._payload_for_write(payload),
                        runtime.text.tokenize,
                        runtime._graph_builder_for_write(),
                    )
                except Exception as exc:
                    state.setdefault("errors", []).append(
                        {
                            "preview_item_id": raw.get("preview_item_id"),
                            "row_number": row_number,
                            "error": str(exc)[:300],
                        }
                    )
                    continue
                created_ids.append(int(memory_id))
                batch_keys.append(key)
            if created_ids:
                try:
                    index_update = await runtime.indexes.upsert_memories(
                        created_ids,
                        reason="memory_transfer_import",
                    )
                except Exception:
                    await runtime.storage.delete_memories(created_ids)
                    raise
                for memory_id in created_ids:
                    await runtime.storage.mark_memory_indexed(
                        memory_id,
                        generation=str(index_update.get("generation") or ""),
                    )
                state.setdefault("imported_ids", []).extend(created_ids)
                existing_keys.update(batch_keys)
                runtime.retrieval.invalidate()
            cursor += len(batch)
            state["cursor"] = cursor
            state["saved_at"] = time.time()
            await run_blocking(_atomic_json, state_path, state)
            await context.checkpoint(
                {
                    "phase": "importing_records",
                    "completed_documents": cursor,
                    "total_documents": len(records),
                    "imported": len(state.get("imported_ids") or []),
                    "skipped": len(state.get("skipped") or []),
                    "errors": len(state.get("errors") or []),
                    "saved_at": state["saved_at"],
                },
                progress=cursor / total,
                message=f"已处理 {cursor}/{len(records)} 条记忆",
            )
        stats = await runtime.storage.statistics()
        return {
            "database_id": memory_store_id,
            "source_sha256": str(manifest.get("source_sha256") or ""),
            "input_records": len(records),
            "imported": len(state.get("imported_ids") or []),
            "imported_ids": list(state.get("imported_ids") or []),
            "skipped": list(state.get("skipped") or []),
            "errors": list(state.get("errors") or []),
            "stats": stats,
        }

    async def _finalize_completed(self, context: JobExecutionContext) -> None:
        job = await self._job(context)
        await self._cleanup_completed(job, self.workspace(job))

    async def _cleanup_completed(self, job: dict[str, Any], workspace: Path) -> None:
        spec = dict(job.get("operation") or {})
        for key in ("source_db", "conversations_db", "manifest_path"):
            if spec.get(key):
                Path(str(spec[key])).unlink(missing_ok=True)
        shutil.rmtree(workspace, ignore_errors=True)

    async def _cancel_queued(self, context: JobExecutionContext) -> None:
        job = await self._job(context)
        spec = dict(job.get("operation") or {})
        for key in ("source_db", "conversations_db", "manifest_path"):
            if spec.get(key):
                Path(str(spec[key])).unlink(missing_ok=True)

    async def _rollback(self, context: JobExecutionContext) -> None:
        job = await self._job(context)
        memory_store_id = self._memory_store_id_from_job(job)
        workspace = self.workspace(job)
        state_path = workspace / "prestate.json"
        if not state_path.exists():
            await self._cancel_queued(context)
            return
        state = json.loads(state_path.read_text(encoding="utf-8"))
        library_dir = database_type_registry.data_dir(
            self.manager.data_dir,
            DatabaseRef(LIVINGMEMORY_V8_TYPE, memory_store_id),
        )
        rollback_dir = workspace / "rollback"
        target_db = library_dir / "livingmemory.db"
        conversations_db = library_dir / "conversations.db"
        await self.manager.unload_runtime(memory_store_id, reason="task_rollback")

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

        rollback_database = rollback_dir / "livingmemory.db"
        rollback_conversations = rollback_dir / "conversations.db"
        database_saved = bool(
            state.get(
                "rollback_database_saved",
                rollback_database.exists() or not bool(state.get("database_existed")),
            )
        )
        conversations_saved = bool(
            state.get(
                "rollback_conversations_saved",
                rollback_conversations.exists()
                or not bool(state.get("conversations_existed")),
            )
        )
        if database_saved:
            await restore_db(
                target_db,
                rollback_database,
                bool(state.get("database_existed")),
                str(state.get("rollback_database_sha256") or ""),
                state.get("rollback_database_state"),
            )
        if conversations_saved:
            await restore_db(
                conversations_db,
                rollback_conversations,
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
        original_index_directories = set(state.get("index_directories") or [])
        if index_root.exists():
            for generation_dir in index_root.iterdir():
                if (
                    generation_dir.is_dir()
                    and generation_dir.name not in original_index_directories
                ):
                    shutil.rmtree(generation_dir, ignore_errors=True)
        try:
            index_state = await run_blocking(
                read_ab_checkpoint,
                workspace / "index",
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            index_state = None
        if index_state:
            generation = str(index_state.get("generation", ""))
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
                    memory_store_id,
                ),
            )
            await db.execute(
                "DELETE FROM library_generation_bindings WHERE library_id=?",
                (memory_store_id,),
            )
            for row in state.get("bindings", []):
                await db.execute(
                    """INSERT INTO library_generation_bindings
                    (library_id,database_type,database_id,generation,provider_id,provider_revision,
                    manifest_json,activated_at) VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        row["library_id"],
                        LIVINGMEMORY_V8_TYPE,
                        memory_store_id,
                        row["generation"],
                        row["provider_id"],
                        row["provider_revision"],
                        row["manifest_json"],
                        row["activated_at"],
                    ),
                )
            await db.execute(
                "DELETE FROM index_generations WHERE library_id=?",
                (memory_store_id,),
            )
            for row in state.get("index_generations", []):
                await db.execute(
                    """INSERT INTO index_generations
                    (library_id,database_type,database_id,generation,status,manifest,created_at,activated_at)
                    VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        row["library_id"],
                        LIVINGMEMORY_V8_TYPE,
                        memory_store_id,
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
        await self.manager.get_runtime(memory_store_id)
        await self._cleanup_completed(job, workspace)


# Deprecated import alias retained for extensions that constructed the old class.
ResumableLibraryTasks = ResumableMemoryStoreTasks
