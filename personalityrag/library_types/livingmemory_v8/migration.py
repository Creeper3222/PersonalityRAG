from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import stat
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from ...io_utils import run_blocking


DB_FILES = (
    "livingmemory.db",
    "conversations.db",
    "livingmemory_graph_documents.db",
)
EXTRA_FILES = (
    "livingmemory.index",
    "livingmemory_graph.index",
    "decay_state.json",
    ".plugin_version",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sqlite_backup(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_db = sqlite3.connect(
        f"file:{source.as_posix()}?mode=ro", uri=True, timeout=30
    )
    destination = sqlite3.connect(target)
    try:
        source_db.backup(destination, pages=1024)
    finally:
        destination.close()
        source_db.close()


def table_fingerprint(path: Path) -> dict[str, Any]:
    con = sqlite3.connect(path)
    try:
        con.row_factory = sqlite3.Row
        tables = [
            row[0]
            for row in con.execute(
                """SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                AND name NOT LIKE '%_data' AND name NOT LIKE '%_idx'
                AND name NOT LIKE '%_content' AND name NOT LIKE '%_docsize'
                AND name NOT LIKE '%_config' ORDER BY name"""
            )
        ]
        result: dict[str, Any] = {}
        for table in tables:
            try:
                columns = [
                    row[1]
                    for row in con.execute(f'PRAGMA table_info("{table}")')
                ]
                cursor = con.execute(f'SELECT * FROM "{table}" ORDER BY rowid')
                digest = hashlib.sha256()
                count = 0
                while rows := cursor.fetchmany(1000):
                    count += len(rows)
                    for row in rows:
                        normalized = []
                        for value in row:
                            if isinstance(value, bytes):
                                normalized.append(
                                    {
                                        "blob_sha256": hashlib.sha256(
                                            value
                                        ).hexdigest()
                                    }
                                )
                            elif isinstance(value, str):
                                try:
                                    normalized.append(json.loads(value))
                                except (json.JSONDecodeError, TypeError):
                                    normalized.append(value)
                            else:
                                normalized.append(value)
                        digest.update(
                            json.dumps(
                                normalized,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        )
                        digest.update(b"\n")
                result[table] = {
                    "count": count,
                    "columns": columns,
                    "rows_sha256": digest.hexdigest(),
                }
            except sqlite3.DatabaseError:
                continue
        return {
            "integrity": con.execute("PRAGMA integrity_check").fetchone()[0],
            "foreign_key_errors": len(
                con.execute("PRAGMA foreign_key_check").fetchall()
            ),
            "tables": result,
        }
    finally:
        con.close()


def validate_livingmemory_db_file(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise RuntimeError("livingmemory.db 文件不存在")
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity_check 失败：{integrity}")
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','virtual table')"
            )
        }
        required = {"documents", "livingmemory_memories_fts"}
        missing = sorted(required - tables)
        if missing:
            raise RuntimeError(f"缺少 LivingMemory 核心表：{missing}")
        document_columns = {
            row[1] for row in con.execute("PRAGMA table_info(documents)")
        }
        required_columns = {"id", "doc_id", "text", "metadata"}
        missing_columns = sorted(required_columns - document_columns)
        if missing_columns:
            raise RuntimeError(f"documents 表缺少字段：{missing_columns}")

        def count(table: str) -> int:
            if table not in tables:
                return 0
            return int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])

        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "integrity": integrity,
            "tables": sorted(tables),
            "counts": {
                "total_memories": count("documents"),
                "graph_nodes": count("graph_nodes"),
                "graph_edges": count("graph_edges"),
                "graph_entries": count("graph_entries"),
                "atom_count": count("memory_atoms"),
            },
        }
    finally:
        con.close()


def validate_conversations_db_file(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise RuntimeError("conversations.db 文件不存在")
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity_check 失败：{integrity}")
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','virtual table')"
            )
        }
        required = {"sessions", "messages"}
        missing = sorted(required - tables)
        if missing:
            raise RuntimeError(f"缺少 conversations 核心表：{missing}")
        required_columns = {
            "sessions": {"id", "session_id", "created_at", "last_active_at", "message_count"},
            "messages": {"id", "session_id", "role", "content", "timestamp"},
        }
        for table, columns in required_columns.items():
            actual = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
            missing_columns = sorted(columns - actual)
            if missing_columns:
                raise RuntimeError(f"{table} 表缺少字段：{missing_columns}")

        def count(table: str) -> int:
            return int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])

        con.row_factory = sqlite3.Row

        def parsed_json(value: Any, fallback: Any) -> tuple[Any, bool]:
            if value in (None, ""):
                return fallback, True
            try:
                parsed = json.loads(str(value))
            except (json.JSONDecodeError, TypeError, ValueError):
                return fallback, False
            return parsed, True

        def normalized_value(value: Any) -> Any:
            if isinstance(value, bytes):
                return {"blob_sha256": hashlib.sha256(value).hexdigest()}
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            return value

        def rows_sha256(
            query: str,
            params: tuple[Any, ...] = (),
            *,
            json_columns: tuple[str, ...] = (),
        ) -> str:
            digest = hashlib.sha256()
            cursor = con.execute(query, params)
            while rows := cursor.fetchmany(1000):
                for row in rows:
                    payload = {
                        key: (
                            normalized_value(row[key])
                            if key in json_columns
                            else row[key]
                        )
                        for key in row.keys()
                    }
                    digest.update(
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                    digest.update(b"\n")
            return digest.hexdigest()

        sessions_state: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        pending_messages = 0
        pending_range_messages = 0
        pending_summary_count = 0
        invalid_json_count = 0
        message_count_mismatches = 0
        last_summarized_out_of_range = 0
        pending_ranges_out_of_range = 0
        session_rows = con.execute(
            """SELECT id,session_id,platform,created_at,last_active_at,
                      message_count,participants,metadata
               FROM sessions ORDER BY session_id,id"""
        ).fetchall()
        for row in session_rows:
            session_id = str(row["session_id"] or "")
            actual_count = int(
                con.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id=?",
                    (session_id,),
                ).fetchone()[0]
            )
            recorded_count = int(row["message_count"] or 0)
            participants, participants_valid = parsed_json(
                row["participants"], []
            )
            metadata, metadata_valid = parsed_json(row["metadata"], {})
            if not isinstance(participants, list):
                participants_valid = False
                participants = []
            if not isinstance(metadata, dict):
                metadata_valid = False
                metadata = {}
            invalid_json_count += int(not participants_valid) + int(
                not metadata_valid
            )
            try:
                last_summarized_index = int(
                    metadata.get("last_summarized_index") or 0
                )
            except (TypeError, ValueError):
                last_summarized_index = 0
                warnings.append(
                    {
                        "code": "invalid_last_summarized_index",
                        "session_id": session_id,
                    }
                )
            pending_messages += max(0, recorded_count - last_summarized_index)
            if recorded_count != actual_count:
                message_count_mismatches += 1
                warnings.append(
                    {
                        "code": "message_count_mismatch",
                        "session_id": session_id,
                        "message_count": recorded_count,
                        "actual_message_count": actual_count,
                    }
                )
            if last_summarized_index < 0 or last_summarized_index > actual_count:
                last_summarized_out_of_range += 1
                warnings.append(
                    {
                        "code": "last_summarized_index_out_of_range",
                        "session_id": session_id,
                        "last_summarized_index": last_summarized_index,
                        "actual_message_count": actual_count,
                    }
                )
            pending_summary = metadata.get("pending_summary")
            normalized_pending = None
            if isinstance(pending_summary, dict):
                pending_summary_count += 1
                try:
                    start_index = int(pending_summary.get("start_index") or 0)
                    end_index = int(pending_summary.get("end_index") or 0)
                    retry_count = int(pending_summary.get("retry_count") or 0)
                except (TypeError, ValueError):
                    start_index = end_index = retry_count = 0
                    warnings.append(
                        {
                            "code": "invalid_pending_summary",
                            "session_id": session_id,
                        }
                    )
                pending_range_messages += max(0, end_index - start_index)
                normalized_pending = {
                    "start_index": start_index,
                    "end_index": end_index,
                    "retry_count": retry_count,
                    "range_message_count": max(0, end_index - start_index),
                    "exceeds_message_count": end_index > recorded_count,
                    "exceeds_actual_message_count": end_index > actual_count,
                }
                if (
                    start_index < 0
                    or end_index < start_index
                    or end_index > actual_count
                ):
                    pending_ranges_out_of_range += 1
                    warnings.append(
                        {
                            "code": "pending_summary_range_out_of_range",
                            "session_id": session_id,
                            "start_index": start_index,
                            "end_index": end_index,
                            "message_count": recorded_count,
                            "actual_message_count": actual_count,
                            "preserved": True,
                        }
                    )
            elif pending_summary is not None:
                warnings.append(
                    {
                        "code": "invalid_pending_summary",
                        "session_id": session_id,
                    }
                )
            sessions_state.append(
                {
                    "session_id": session_id,
                    "message_count": recorded_count,
                    "actual_message_count": actual_count,
                    "participants": participants,
                    "last_summarized_index": last_summarized_index,
                    "pending_summary": normalized_pending,
                }
            )

        state_sha256 = hashlib.sha256(
            json.dumps(
                sessions_state,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        sessions_sha256 = rows_sha256(
            "SELECT * FROM sessions ORDER BY session_id,id",
            json_columns=("participants", "metadata"),
        )
        messages_sha256 = rows_sha256(
            "SELECT * FROM messages ORDER BY session_id,timestamp,id",
            json_columns=("metadata",),
        )
        order_sha256 = rows_sha256(
            """SELECT session_id,id,timestamp
               FROM messages ORDER BY session_id,timestamp,id"""
        )
        participants_sha256 = rows_sha256(
            "SELECT session_id,participants FROM sessions ORDER BY session_id,id",
            json_columns=("participants",),
        )
        metadata_sha256 = rows_sha256(
            """SELECT 'session' AS record_type,session_id,id,metadata
               FROM sessions
               UNION ALL
               SELECT 'message' AS record_type,session_id,id,metadata
               FROM messages
               ORDER BY record_type,session_id,id""",
            json_columns=("metadata",),
        )

        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "integrity": integrity,
            "tables": sorted(tables),
            "counts": {
                "sessions": count("sessions"),
                "messages": count("messages"),
                "pending_messages": pending_messages,
                "pending_summaries": pending_summary_count,
                "pending_range_messages": pending_range_messages,
            },
            "normalized_hashes": {
                "sessions": sessions_sha256,
                "messages": messages_sha256,
                "message_order": order_sha256,
                "participants": participants_sha256,
                "metadata": metadata_sha256,
                "summary_state": state_sha256,
            },
            "validation": {
                "message_count_mismatches": message_count_mismatches,
                "invalid_json_fields": invalid_json_count,
                "last_summarized_out_of_range": last_summarized_out_of_range,
                "pending_ranges_out_of_range": pending_ranges_out_of_range,
                "sessions": sessions_state,
                "warnings": warnings,
            },
        }
    finally:
        con.close()


class LivingMemoryMigrator:
    def __init__(self, destination_data_dir: Path):
        self.destination = destination_data_dir

    async def migrate(
        self,
        source_dir: Path,
        *,
        mode: str,
        progress: Callable[[float, str], Awaitable[None]] | None = None,
        preserve_previous: bool = True,
    ) -> dict[str, Any]:
        source_dir = source_dir.resolve()
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)
        missing = [
            name for name in DB_FILES[:2] if not (source_dir / name).exists()
        ]
        if missing:
            raise RuntimeError(f"LivingMemory source is incomplete: {missing}")

        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        archive = self.destination / "imports" / run_id / "source_archive"
        report_dir = self.destination / "reports"
        archive.mkdir(parents=True, exist_ok=False)
        report_dir.mkdir(parents=True, exist_ok=True)

        if progress:
            await progress(0.05, "正在创建 SQLite 一致快照")
        source_reports: dict[str, Any] = {}
        for index, name in enumerate(DB_FILES):
            source = source_dir / name
            if not source.exists():
                continue
            target = archive / name
            await run_blocking(sqlite_backup, source, target)
            source_reports[name] = await run_blocking(
                table_fingerprint, target
            )
            if progress:
                await progress(
                    0.08 + (index + 1) * 0.08,
                    f"已快照 {name}",
                )

        for name in EXTRA_FILES:
            source = source_dir / name
            if source.exists():
                await run_blocking(shutil.copy2, source, archive / name)
        for directory in ("stopwords", "backups"):
            source = source_dir / directory
            if source.exists():
                await run_blocking(
                    shutil.copytree,
                    source,
                    archive / directory,
                    dirs_exist_ok=True,
                )

        if progress:
            await progress(0.4, "正在安装独立工作副本")
        self.destination.mkdir(parents=True, exist_ok=True)
        previous = self.destination / "pre_migration_backups" / run_id
        if preserve_previous:
            previous.mkdir(parents=True, exist_ok=True)
        for name in DB_FILES[:2]:
            current = self.destination / name
            if preserve_previous and current.exists() and current.stat().st_size > 0:
                await run_blocking(shutil.copy2, current, previous / name)
            await run_blocking(shutil.copy2, archive / name, current)
        stopwords_source = archive / "stopwords"
        if stopwords_source.exists():
            await run_blocking(
                shutil.copytree,
                stopwords_source,
                self.destination / "stopwords",
                dirs_exist_ok=True,
            )
        imported_backups = archive / "backups"
        if imported_backups.exists():
            await run_blocking(
                shutil.copytree,
                imported_backups,
                self.destination / "livingmemory_backups",
                dirs_exist_ok=True,
            )

        if progress:
            await progress(0.6, "正在逐表校验迁移结果")
        target_reports = {
            name: await run_blocking(
                table_fingerprint, self.destination / name
            )
            for name in DB_FILES[:2]
        }
        mismatches = []
        for name in DB_FILES[:2]:
            source_report = source_reports[name]
            target_report = target_reports[name]
            for table, source_table in source_report["tables"].items():
                target_table = target_report["tables"].get(table)
                if not target_table or (
                    source_table["count"] != target_table["count"]
                    or source_table["rows_sha256"]
                    != target_table["rows_sha256"]
                ):
                    mismatches.append(f"{name}:{table}")

        archive_files = {}
        for path in archive.rglob("*"):
            if path.is_file():
                archive_files[str(path.relative_to(archive))] = {
                    "size": path.stat().st_size,
                    "sha256": await run_blocking(sha256_file, path),
                }
        report = {
            "run_id": run_id,
            "mode": mode,
            "source": str(source_dir),
            "destination": str(self.destination),
            "created_at": time.time(),
            "source_reports": source_reports,
            "target_reports": target_reports,
            "mismatches": mismatches,
            "archive_files": archive_files,
            "archive": str(archive),
            "status": "verified" if not mismatches else "failed",
        }
        report_path = report_dir / f"migration-{run_id}.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if mismatches:
            raise RuntimeError(
                "migration verification failed: " + ", ".join(mismatches)
            )
        if mode == "formal":
            for path in archive.rglob("*"):
                if path.is_file():
                    try:
                        path.chmod(stat.S_IREAD)
                    except OSError:
                        pass
        if progress:
            await progress(1.0, "LivingMemory 数据副本迁移与逐表校验完成")
        return {
            "run_id": run_id,
            "status": report["status"],
            "report_path": str(report_path),
            "archive": str(archive),
            "database_counts": {
                name: {
                    table: details["count"]
                    for table, details in report["target_reports"][name][
                        "tables"
                    ].items()
                }
                for name in DB_FILES[:2]
            },
        }
