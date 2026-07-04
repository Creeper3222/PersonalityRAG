from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable


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
                rows = con.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
                digest = hashlib.sha256()
                for row in rows:
                    normalized = []
                    for value in row:
                        if isinstance(value, bytes):
                            normalized.append(
                                {"blob_sha256": hashlib.sha256(value).hexdigest()}
                            )
                        elif isinstance(value, str):
                            try:
                                parsed = json.loads(value)
                                normalized.append(parsed)
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
                    "count": len(rows),
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

        db_version = None
        if "db_version" in tables:
            row = con.execute(
                "SELECT version FROM db_version ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is not None:
                db_version = int(row[0])

        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "integrity": integrity,
            "db_version": db_version,
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


class LivingMemoryMigrator:
    def __init__(self, destination_data_dir: Path):
        self.destination = destination_data_dir

    async def migrate(
        self,
        source_dir: Path,
        *,
        mode: str,
        progress: Callable[[float, str], Awaitable[None]] | None = None,
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
            await asyncio.to_thread(sqlite_backup, source, target)
            source_reports[name] = await asyncio.to_thread(
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
                await asyncio.to_thread(shutil.copy2, source, archive / name)
        for directory in ("stopwords", "backups"):
            source = source_dir / directory
            if source.exists():
                await asyncio.to_thread(
                    shutil.copytree,
                    source,
                    archive / directory,
                    dirs_exist_ok=True,
                )

        if progress:
            await progress(0.4, "正在安装独立工作副本")
        self.destination.mkdir(parents=True, exist_ok=True)
        previous = self.destination / "pre_migration_backups" / run_id
        previous.mkdir(parents=True, exist_ok=True)
        for name in DB_FILES[:2]:
            current = self.destination / name
            if current.exists() and current.stat().st_size > 0:
                await asyncio.to_thread(shutil.copy2, current, previous / name)
            await asyncio.to_thread(shutil.copy2, archive / name, current)
        stopwords_source = archive / "stopwords"
        if stopwords_source.exists():
            await asyncio.to_thread(
                shutil.copytree,
                stopwords_source,
                self.destination / "stopwords",
                dirs_exist_ok=True,
            )
        imported_backups = archive / "backups"
        if imported_backups.exists():
            await asyncio.to_thread(
                shutil.copytree,
                imported_backups,
                self.destination / "livingmemory_backups",
                dirs_exist_ok=True,
            )

        if progress:
            await progress(0.6, "正在逐表校验迁移结果")
        target_reports = {
            name: await asyncio.to_thread(
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
                    "sha256": await asyncio.to_thread(sha256_file, path),
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
