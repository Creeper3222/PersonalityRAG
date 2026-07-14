from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterable

import aiosqlite

from .atoms import compute_atom_ttl
from .migration import sha256_file
from .version import VERSION


def normalize_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        result = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return result if isinstance(result, dict) else {}


class Storage:
    def __init__(self, data_dir: Path, system_path: Path | None = None):
        self.data_dir = data_dir
        self.db_path = data_dir / "livingmemory.db"
        self.conversations_path = data_dir / "conversations.db"
        self.system_path = system_path or (data_dir / "personalityrag_system.db")
        self._write_lock = asyncio.Lock()

    @asynccontextmanager
    async def connect(self, *, system: bool = False):
        path = self.system_path if system else self.db_path
        db = await aiosqlite.connect(path)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA busy_timeout=10000")
        await db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
        finally:
            await db.close()

    async def initialize(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        async with self.connect() as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL UNIQUE,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS livingmemory_memories_fts
                USING fts5(content, doc_id UNINDEXED, tokenize='unicode61');
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_key TEXT NOT NULL UNIQUE,
                    node_type TEXT NOT NULL,
                    node_value TEXT NOT NULL,
                    canonical_value TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS graph_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    edge_key TEXT NOT NULL UNIQUE,
                    source_node_id INTEGER NOT NULL,
                    target_node_id INTEGER NOT NULL,
                    relation_type TEXT NOT NULL,
                    source_memory_id INTEGER NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    confidence REAL NOT NULL DEFAULT 0.8,
                    status TEXT NOT NULL DEFAULT 'active',
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(source_node_id) REFERENCES graph_nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY(target_node_id) REFERENCES graph_nodes(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS graph_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_key TEXT NOT NULL UNIQUE,
                    source_memory_id INTEGER NOT NULL,
                    session_id TEXT,
                    persona_id TEXT,
                    entry_type TEXT NOT NULL,
                    relation_type TEXT,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    edge_id INTEGER,
                    vector_doc_id INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(edge_id) REFERENCES graph_edges(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS graph_entry_nodes (
                    entry_id INTEGER NOT NULL,
                    node_id INTEGER NOT NULL,
                    PRIMARY KEY(entry_id,node_id),
                    FOREIGN KEY(entry_id) REFERENCES graph_entries(id) ON DELETE CASCADE,
                    FOREIGN KEY(node_id) REFERENCES graph_nodes(id) ON DELETE CASCADE
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS livingmemory_graph_entries_fts
                USING fts5(content, entry_id UNINDEXED, tokenize='unicode61');
                CREATE TABLE IF NOT EXISTS memory_atoms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_memory_id INTEGER NOT NULL,
                    atom_type TEXT NOT NULL DEFAULT 'unknown',
                    content TEXT NOT NULL,
                    entities TEXT DEFAULT '[]',
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    created_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL,
                    last_reinforced_at REAL,
                    event_time REAL,
                    ttl_days REAL NOT NULL DEFAULT 30.0,
                    expires_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    reinforcement_count INTEGER NOT NULL DEFAULT 0,
                    decay_type TEXT NOT NULL DEFAULT 'exponential',
                    session_id TEXT,
                    persona_id TEXT,
                    metadata TEXT DEFAULT '{}'
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_atoms_fts
                USING fts5(content, atom_id UNINDEXED, tokenize='unicode61');
                CREATE TABLE IF NOT EXISTS memory_write_ops (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    op_type TEXT NOT NULL,
                    memory_id INTEGER,
                    status TEXT NOT NULL DEFAULT 'pending',
                    step TEXT NOT NULL DEFAULT 'started',
                    payload TEXT DEFAULT '{}',
                    error TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_graph_nodes_canonical ON graph_nodes(canonical_value);
                CREATE INDEX IF NOT EXISTS idx_graph_edges_memory_id ON graph_edges(source_memory_id);
                CREATE INDEX IF NOT EXISTS idx_graph_entries_memory_id ON graph_entries(source_memory_id);
                CREATE INDEX IF NOT EXISTS idx_graph_entries_scope_latest
                ON graph_entries(session_id, persona_id, source_memory_id, id DESC);
                CREATE INDEX IF NOT EXISTS idx_atoms_parent ON memory_atoms(parent_memory_id);
                CREATE INDEX IF NOT EXISTS idx_atoms_scope_status
                ON memory_atoms(status, session_id, persona_id);
                """
            )
            await db.commit()
        async with self.connect(system=True) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS index_generations (
                    generation TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    manifest TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    activated_at REAL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    library_id TEXT,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    result TEXT,
                    error TEXT,
                    operation TEXT,
                    checkpoint TEXT,
                    status_reason TEXT NOT NULL DEFAULT '',
                    control_requested TEXT,
                    resumable INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS migration_runs (
                    id TEXT PRIMARY KEY,
                    library_id TEXT,
                    source_path TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    report_path TEXT,
                    created_at REAL NOT NULL,
                    completed_at REAL
                );
                """
            )
            job_columns = {
                row["name"]
                for row in await (await db.execute("PRAGMA table_info(jobs)")).fetchall()
            }
            if "library_id" not in job_columns:
                await db.execute("ALTER TABLE jobs ADD COLUMN library_id TEXT")
            for name, sql_type in (
                ("operation", "TEXT"),
                ("checkpoint", "TEXT"),
                ("status_reason", "TEXT NOT NULL DEFAULT ''"),
                ("control_requested", "TEXT"),
                ("resumable", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if name not in job_columns:
                    await db.execute(f"ALTER TABLE jobs ADD COLUMN {name} {sql_type}")
            migration_columns = {
                row["name"]
                for row in await (
                    await db.execute("PRAGMA table_info(migration_runs)")
                ).fetchall()
            }
            if "library_id" not in migration_columns:
                await db.execute(
                    "ALTER TABLE migration_runs ADD COLUMN library_id TEXT"
                )
            await db.execute(
                "INSERT OR REPLACE INTO schema_info(key,value) VALUES('service_version',?)",
                (VERSION,),
            )
            await db.commit()
        await self._initialize_conversations()

    async def _initialize_conversations(self) -> None:
        db = await aiosqlite.connect(self.conversations_path)
        try:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    platform TEXT,
                    created_at REAL NOT NULL,
                    last_active_at REAL NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    participants TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sender_id TEXT,
                    sender_name TEXT,
                    group_id TEXT,
                    platform TEXT,
                    timestamp REAL NOT NULL,
                    metadata TEXT DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session_time
                ON messages(session_id,timestamp);
                """
            )
            await db.commit()
        finally:
            await db.close()

    @staticmethod
    def _conversation_message_row(row: aiosqlite.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "session_id": row["session_id"],
            "role": row["role"],
            "content": row["content"],
            "sender_id": row["sender_id"],
            "sender_name": row["sender_name"],
            "group_id": row["group_id"],
            "platform": row["platform"],
            "timestamp": float(row["timestamp"]),
            "metadata": normalize_metadata(row["metadata"]),
        }

    @staticmethod
    def _conversation_session_row(row: aiosqlite.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "session_id": row["session_id"],
            "platform": row["platform"],
            "created_at": float(row["created_at"]),
            "last_active_at": float(row["last_active_at"]),
            "message_count": int(row["message_count"] or 0),
            "participants": json.loads(row["participants"] or "[]"),
            "metadata": normalize_metadata(row["metadata"]),
        }

    async def add_conversation_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_timestamp = payload.get("timestamp")
        timestamp = float(time.time() if raw_timestamp is None else raw_timestamp)
        session_id = str(payload["session_id"])
        role = str(payload["role"])
        content = str(payload.get("content") or "")
        platform = str(payload.get("platform") or "astrbot")
        sender_id = payload.get("sender_id") or session_id
        sender_id = str(sender_id) if sender_id is not None else ""
        sender_name = payload.get("sender_name")
        group_id = payload.get("group_id")
        metadata = dict(payload.get("metadata") or {})
        dedup_key = payload.get("dedup_key")
        if dedup_key:
            metadata.setdefault("dedup_key", str(dedup_key))

        async with self._write_lock:
            db = await aiosqlite.connect(self.conversations_path)
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA busy_timeout=10000")
            try:
                if dedup_key:
                    existing = await (
                        await db.execute(
                            """
                            SELECT id, session_id, role, content, sender_id, sender_name,
                                   group_id, platform, timestamp, metadata
                            FROM messages
                            WHERE json_extract(metadata,'$.dedup_key')=?
                            LIMIT 1
                            """,
                            (str(dedup_key),),
                        )
                    ).fetchone()
                    if existing:
                        session = await self._get_conversation_session(db, session_id)
                        return {
                            "message": self._conversation_message_row(existing),
                            "session": session,
                            "duplicate": True,
                        }

                await db.execute(
                    """
                    INSERT INTO sessions (
                        session_id, platform, created_at, last_active_at,
                        message_count, participants, metadata
                    )
                    VALUES (?, ?, ?, ?, 0, '[]', '{}')
                    ON CONFLICT(session_id) DO NOTHING
                    """,
                    (session_id, platform, timestamp, timestamp),
                )
                cursor = await db.execute(
                    """
                    INSERT INTO messages (
                        session_id, role, content, sender_id, sender_name,
                        group_id, platform, timestamp, metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        role,
                        content,
                        sender_id,
                        sender_name,
                        group_id,
                        platform,
                        timestamp,
                        json.dumps(metadata, ensure_ascii=False),
                    ),
                )
                message_id = int(cursor.lastrowid or 0)
                await db.execute(
                    """
                    UPDATE sessions
                    SET message_count = (
                            SELECT COUNT(*) FROM messages WHERE session_id = ?
                        ),
                        last_active_at = ?,
                        participants = CASE
                            WHEN ? = '' THEN participants
                            WHEN EXISTS (
                                SELECT 1
                                FROM json_each(COALESCE(NULLIF(participants, ''), '[]'))
                                WHERE value = ?
                            ) THEN participants
                            ELSE json_insert(
                                COALESCE(NULLIF(participants, ''), '[]'),
                                '$[#]',
                                ?
                            )
                        END
                    WHERE session_id = ?
                    """,
                    (session_id, timestamp, sender_id, sender_id, sender_id, session_id),
                )
                await db.commit()
                row = await (
                    await db.execute(
                        """
                        SELECT id, session_id, role, content, sender_id, sender_name,
                               group_id, platform, timestamp, metadata
                        FROM messages WHERE id=?
                        """,
                        (message_id,),
                    )
                ).fetchone()
                session = await self._get_conversation_session(db, session_id)
                return {
                    "message": self._conversation_message_row(row),
                    "session": session,
                    "duplicate": False,
                }
            finally:
                await db.close()

    async def _get_conversation_session(
        self, db: aiosqlite.Connection, session_id: str
    ) -> dict[str, Any] | None:
        row = await (
            await db.execute(
                """
                SELECT id, session_id, platform, created_at, last_active_at,
                       message_count, participants, metadata
                FROM sessions WHERE session_id=?
                """,
                (session_id,),
            )
        ).fetchone()
        return self._conversation_session_row(row) if row else None

    async def get_conversation(
        self, session_id: str, *, limit: int = 50
    ) -> dict[str, Any] | None:
        db = await aiosqlite.connect(self.conversations_path)
        db.row_factory = aiosqlite.Row
        try:
            session = await self._get_conversation_session(db, session_id)
            if not session:
                return None
            messages = await self.get_conversation_messages(session_id, limit=limit)
            return {"session": session, "messages": messages}
        finally:
            await db.close()

    async def get_conversation_messages(
        self,
        session_id: str,
        *,
        start_index: int | None = None,
        end_index: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        db = await aiosqlite.connect(self.conversations_path)
        db.row_factory = aiosqlite.Row
        try:
            base = """
                SELECT id, session_id, role, content, sender_id, sender_name,
                       group_id, platform, timestamp, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC, id ASC
            """
            if start_index is not None or end_index is not None:
                start = max(0, int(start_index or 0))
                if end_index is None:
                    count = max(0, int(limit or 50))
                else:
                    count = max(0, int(end_index) - start)
                rows = await (
                    await db.execute(base + " LIMIT ? OFFSET ?", (session_id, count, start))
                ).fetchall()
            elif limit is not None:
                count = max(0, int(limit))
                rows = await (
                    await db.execute(
                        """
                        SELECT * FROM (
                            SELECT id, session_id, role, content, sender_id, sender_name,
                                   group_id, platform, timestamp, metadata
                            FROM messages
                            WHERE session_id = ?
                            ORDER BY timestamp DESC, id DESC
                            LIMIT ?
                        ) ORDER BY timestamp ASC, id ASC
                        """,
                        (session_id, count),
                    )
                ).fetchall()
            else:
                rows = await (await db.execute(base, (session_id,))).fetchall()
            return [self._conversation_message_row(row) for row in rows]
        finally:
            await db.close()

    async def update_conversation_metadata(
        self, session_id: str, patch: dict[str, Any]
    ) -> dict[str, Any] | None:
        async with self._write_lock:
            db = await aiosqlite.connect(self.conversations_path)
            db.row_factory = aiosqlite.Row
            try:
                session = await self._get_conversation_session(db, session_id)
                if not session:
                    return None
                metadata = dict(session.get("metadata") or {})
                for key, value in patch.items():
                    if value is None:
                        metadata.pop(key, None)
                    else:
                        metadata[key] = value
                await db.execute(
                    "UPDATE sessions SET metadata=? WHERE session_id=?",
                    (json.dumps(metadata, ensure_ascii=False), session_id),
                )
                await db.commit()
                return await self._get_conversation_session(db, session_id)
            finally:
                await db.close()

    async def clear_conversation(self, session_id: str) -> dict[str, Any]:
        async with self._write_lock:
            db = await aiosqlite.connect(self.conversations_path)
            db.row_factory = aiosqlite.Row
            try:
                cursor = await db.execute(
                    "DELETE FROM messages WHERE session_id=?", (session_id,)
                )
                deleted = max(0, cursor.rowcount)
                await db.execute(
                    """
                    UPDATE sessions
                    SET message_count=0, participants='[]', metadata='{}'
                    WHERE session_id=?
                    """,
                    (session_id,),
                )
                await db.commit()
                return {
                    "deleted": deleted,
                    "session": await self._get_conversation_session(db, session_id),
                }
            finally:
                await db.close()

    async def trim_conversation(self, session_id: str, delete_count: int) -> dict[str, Any]:
        delete_count = max(0, int(delete_count))
        if delete_count <= 0:
            return {"deleted": 0, "session": await self.get_conversation(session_id)}
        async with self._write_lock:
            db = await aiosqlite.connect(self.conversations_path)
            db.row_factory = aiosqlite.Row
            try:
                session = await self._get_conversation_session(db, session_id)
                if not session:
                    return {"deleted": 0, "session": None}
                metadata = dict(session.get("metadata") or {})
                try:
                    last_summarized_index = int(
                        metadata.get("last_summarized_index", 0) or 0
                    )
                except (TypeError, ValueError):
                    last_summarized_index = 0
                actual_count = int(
                    (
                        await (
                            await db.execute(
                                "SELECT COUNT(*) AS value FROM messages WHERE session_id=?",
                                (session_id,),
                            )
                        ).fetchone()
                    )["value"]
                    or 0
                )
                if last_summarized_index > actual_count:
                    metadata["last_summarized_index"] = 0
                    await db.execute(
                        "UPDATE sessions SET message_count=?, metadata=? WHERE session_id=?",
                        (
                            actual_count,
                            json.dumps(metadata, ensure_ascii=False),
                            session_id,
                        ),
                    )
                    await db.commit()
                    return {"deleted": 0, "session": await self._get_conversation_session(db, session_id)}
                safe_count = min(delete_count, max(0, last_summarized_index))
                if safe_count <= 0:
                    return {"deleted": 0, "session": session}
                cursor = await db.execute(
                    """
                    DELETE FROM messages
                    WHERE id IN (
                        SELECT id FROM messages
                        WHERE session_id=?
                        ORDER BY timestamp ASC, id ASC
                        LIMIT ?
                    )
                    """,
                    (session_id, safe_count),
                )
                deleted = max(0, cursor.rowcount)
                metadata["last_summarized_index"] = max(
                    0, last_summarized_index - deleted
                )
                await db.execute(
                    "UPDATE sessions SET message_count=?, metadata=? WHERE session_id=?",
                    (
                        max(0, actual_count - deleted),
                        json.dumps(metadata, ensure_ascii=False),
                        session_id,
                    ),
                )
                await db.commit()
                return {
                    "deleted": deleted,
                    "session": await self._get_conversation_session(db, session_id),
                }
            finally:
                await db.close()

    async def rebuild_fts(self, tokenize) -> dict[str, int]:
        async with self._write_lock, self.connect() as db:
            await db.execute("DELETE FROM livingmemory_memories_fts")
            cursor = await db.execute("SELECT id,text FROM documents ORDER BY id")
            rows = await cursor.fetchall()
            memory_rows = [
                (int(row["id"]), " ".join(tokenize(row["text"] or "")))
                for row in rows
            ]
            await db.executemany(
                "INSERT INTO livingmemory_memories_fts(doc_id,content) VALUES(?,?)",
                memory_rows,
            )
            await db.execute("DELETE FROM livingmemory_graph_entries_fts")
            cursor = await db.execute("SELECT id,content FROM graph_entries ORDER BY id")
            graph_rows_raw = await cursor.fetchall()
            graph_rows = [
                (int(row["id"]), str(row["content"] or ""))
                for row in graph_rows_raw
            ]
            await db.executemany(
                "INSERT INTO livingmemory_graph_entries_fts(entry_id,content) VALUES(?,?)",
                graph_rows,
            )
            await db.execute("DELETE FROM memory_atoms_fts")
            cursor = await db.execute(
                "SELECT id,content FROM memory_atoms WHERE status='active' ORDER BY id"
            )
            atom_rows = await cursor.fetchall()
            await db.executemany(
                "INSERT INTO memory_atoms_fts(atom_id,content) VALUES(?,?)",
                [(int(row["id"]), row["content"]) for row in atom_rows],
            )
            await db.commit()
        return {
            "documents": len(memory_rows),
            "graph_entries": len(graph_rows),
            "atoms": len(atom_rows),
        }

    async def graph_integrity_report(self) -> dict[str, int]:
        async with self.connect() as db:
            async def count_sql(sql: str) -> int:
                row = await (await db.execute(sql)).fetchone()
                return int(row[0] or 0)

            documents = await count_sql("SELECT COUNT(*) FROM documents")
            graph_nodes = await count_sql("SELECT COUNT(*) FROM graph_nodes")
            graph_edges = await count_sql("SELECT COUNT(*) FROM graph_edges")
            graph_entries = await count_sql("SELECT COUNT(*) FROM graph_entries")
            graph_entry_nodes = await count_sql(
                "SELECT COUNT(*) FROM graph_entry_nodes"
            )
            graph_fts = await count_sql(
                "SELECT COUNT(*) FROM livingmemory_graph_entries_fts"
            )
            docs_with_graph = await count_sql(
                "SELECT COUNT(DISTINCT source_memory_id) FROM graph_entries"
            )
            docs_without_graph = await count_sql(
                """SELECT COUNT(*) FROM documents d
                WHERE NOT EXISTS (
                    SELECT 1 FROM graph_entries ge WHERE ge.source_memory_id=d.id
                )"""
            )
            orphan_graph_entries = await count_sql(
                """SELECT COUNT(*) FROM graph_entries ge
                WHERE NOT EXISTS (
                    SELECT 1 FROM documents d WHERE d.id=ge.source_memory_id
                )"""
            )
        return {
            "documents": documents,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "graph_entries": graph_entries,
            "graph_entry_nodes": graph_entry_nodes,
            "graph_fts": graph_fts,
            "documents_with_graph": docs_with_graph,
            "documents_without_graph": docs_without_graph,
            "orphan_graph_entries": orphan_graph_entries,
        }

    async def fts_integrity_report(self) -> dict[str, int]:
        async with self.connect() as db:
            async def count_sql(sql: str) -> int:
                try:
                    row = await (await db.execute(sql)).fetchone()
                except Exception:
                    return -1
                return int(row[0] or 0)

            documents = await count_sql("SELECT COUNT(*) FROM documents")
            document_fts = await count_sql("SELECT COUNT(*) FROM livingmemory_memories_fts")
            graph_entries = await count_sql("SELECT COUNT(*) FROM graph_entries")
            graph_fts = await count_sql("SELECT COUNT(*) FROM livingmemory_graph_entries_fts")
            active_atoms = await count_sql(
                "SELECT COUNT(*) FROM memory_atoms WHERE status='active'"
            )
            atom_fts = await count_sql("SELECT COUNT(*) FROM memory_atoms_fts")

        return {
            "documents": documents,
            "document_fts": document_fts,
            "graph_entries": graph_entries,
            "graph_fts": graph_fts,
            "active_atoms": active_atoms,
            "atom_fts": atom_fts,
        }

    async def rebuild_graph_from_documents(
        self,
        tokenize,
        graph_builder,
        progress=None,
    ) -> dict[str, int]:
        async with self._write_lock, self.connect() as db:
            total_row = await (
                await db.execute("SELECT COUNT(*) FROM documents")
            ).fetchone()
            total = int(total_row[0] or 0)
            await db.execute("BEGIN IMMEDIATE")
            await db.execute("DELETE FROM livingmemory_graph_entries_fts")
            await db.execute("DELETE FROM graph_entry_nodes")
            await db.execute("DELETE FROM graph_entries")
            await db.execute("DELETE FROM graph_edges")
            await db.execute("DELETE FROM graph_nodes")

            cursor = await db.execute(
                "SELECT id,text,metadata FROM documents ORDER BY id"
            )
            rebuilt_documents = 0
            rebuilt_entries = 0
            while True:
                rows = await cursor.fetchmany(100)
                if not rows:
                    break
                for row in rows:
                    memory_id = int(row["id"])
                    text = str(row["text"] or "")
                    metadata = normalize_metadata(row["metadata"])
                    canonical = str(metadata.get("canonical_summary") or text)
                    graph = graph_builder(memory_id, canonical, metadata)
                    await self._insert_graph(db, graph, tokenize)
                    rebuilt_documents += 1
                    rebuilt_entries += len(graph.get("entries") or [])
                if progress and total:
                    await progress(
                        rebuilt_documents / total,
                        f"已回填 {rebuilt_documents}/{total} 条记忆的图数据",
                    )
            await db.commit()
        return {
            "documents": total,
            "rebuilt_documents": rebuilt_documents,
            "graph_entries": rebuilt_entries,
        }

    async def list_documents(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        keyword: str = "",
        session_id: str | None = None,
        persona_id: str | None = None,
        status: str | None = None,
        memory_type: str | None = None,
        sort: str = "created_desc",
    ) -> dict[str, Any]:
        clauses: list[str] = []
        params: list[Any] = []
        if keyword:
            if keyword.isdigit():
                clauses.append("(CAST(id AS TEXT)=? OR text LIKE ?)")
                params.extend([keyword, f"%{keyword}%"])
            else:
                clauses.append("(text LIKE ? OR metadata LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%"])
        for key, value in (
            ("session_id", session_id),
            ("persona_id", persona_id),
            ("status", status),
            ("memory_type", memory_type),
        ):
            if value:
                default = "active" if key == "status" else None
                expression = (
                    f"COALESCE(json_extract(metadata,'$.{key}'),?)=?"
                    if default
                    else f"json_extract(metadata,'$.{key}')=?"
                )
                clauses.append(expression)
                if default:
                    params.append(default)
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_map = {
            "created_desc": "COALESCE(json_extract(metadata,'$.create_time'),0) DESC,id DESC",
            "created_asc": "COALESCE(json_extract(metadata,'$.create_time'),0) ASC,id ASC",
            "updated_desc": "COALESCE(json_extract(metadata,'$.updated_at'),json_extract(metadata,'$.create_time'),0) DESC,id DESC",
            "type_asc": "COALESCE(json_extract(metadata,'$.memory_type'),'GENERAL') ASC,id DESC",
            "importance_desc": "COALESCE(json_extract(metadata,'$.importance'),0.5) DESC,id DESC",
            "importance_asc": "COALESCE(json_extract(metadata,'$.importance'),0.5) ASC,id ASC",
            "id_desc": "id DESC",
            "id_asc": "id ASC",
        }
        order = order_map.get(sort, order_map["created_desc"])
        offset = (max(1, page) - 1) * page_size
        async with self.connect() as db:
            count_row = await (
                await db.execute(
                    f"SELECT COUNT(*) AS value FROM documents {where}", params
                )
            ).fetchone()
            rows = await (
                await db.execute(
                    f"""SELECT id,doc_id,text,metadata,created_at,updated_at
                    FROM documents {where} ORDER BY {order} LIMIT ? OFFSET ?""",
                    (*params, page_size, offset),
                )
            ).fetchall()
        return {
            "items": [self._document_row(row) for row in rows],
            "total": int(count_row["value"] if count_row else 0),
            "page": page,
            "page_size": page_size,
            "has_more": offset + page_size
            < int(count_row["value"] if count_row else 0),
        }

    async def get_document(self, memory_id: int) -> dict[str, Any] | None:
        async with self.connect() as db:
            row = await (
                await db.execute(
                    "SELECT id,doc_id,text,metadata,created_at,updated_at FROM documents WHERE id=?",
                    (memory_id,),
                )
            ).fetchone()
        return self._document_row(row) if row else None

    @staticmethod
    def _document_row(row: aiosqlite.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "doc_id": row["doc_id"],
            "text": row["text"],
            "metadata": normalize_metadata(row["metadata"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def document_ids(self) -> list[int]:
        async with self.connect() as db:
            rows = await (
                await db.execute("SELECT id FROM documents ORDER BY id")
            ).fetchall()
        return [int(row["id"]) for row in rows]

    async def graph_entry_ids(self) -> list[int]:
        async with self.connect() as db:
            rows = await (
                await db.execute("SELECT id FROM graph_entries ORDER BY id")
            ).fetchall()
        return [int(row["id"]) for row in rows]

    async def documents_for_ids(self, memory_ids: Iterable[int]) -> list[dict[str, Any]]:
        ids = sorted({int(value) for value in memory_ids if int(value) > 0})
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        async with self.connect() as db:
            rows = await (
                await db.execute(
                    f"""SELECT id,doc_id,text,metadata,created_at,updated_at
                    FROM documents WHERE id IN ({placeholders}) ORDER BY id""",
                    tuple(ids),
                )
            ).fetchall()
        return [self._document_row(row) for row in rows]

    async def iter_documents(self, batch_size: int = 500, *, after_id: int = 0):
        last_id = max(0, int(after_id))
        while True:
            async with self.connect() as db:
                rows = await (
                    await db.execute(
                        """SELECT id,doc_id,text,metadata,created_at,updated_at
                        FROM documents WHERE id>? ORDER BY id LIMIT ?""",
                        (last_id, batch_size),
                    )
                ).fetchall()
            if not rows:
                break
            yield [self._document_row(row) for row in rows]
            last_id = int(rows[-1]["id"])

    async def iter_graph_entries(
        self, batch_size: int = 500, *, after_id: int = 0
    ):
        last_id = max(0, int(after_id))
        while True:
            async with self.connect() as db:
                rows = await (
                    await db.execute(
                        """SELECT id,source_memory_id,session_id,persona_id,
                        entry_type,relation_type,content,metadata
                        FROM graph_entries WHERE id>? ORDER BY id LIMIT ?""",
                        (last_id, batch_size),
                    )
                ).fetchall()
            if not rows:
                break
            yield [
                {
                    "id": int(row["id"]),
                    "source_memory_id": int(row["source_memory_id"]),
                    "session_id": row["session_id"],
                    "persona_id": row["persona_id"],
                    "entry_type": row["entry_type"],
                    "relation_type": row["relation_type"],
                    "content": row["content"],
                    "metadata": normalize_metadata(row["metadata"]),
                }
                for row in rows
            ]
            last_id = int(rows[-1]["id"])

    async def graph_entries_for_memory_ids(
        self, memory_ids: Iterable[int]
    ) -> list[dict[str, Any]]:
        ids = sorted({int(value) for value in memory_ids if int(value) > 0})
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        async with self.connect() as db:
            rows = await (
                await db.execute(
                    f"""SELECT id,source_memory_id,session_id,persona_id,
                    entry_type,relation_type,content,metadata
                    FROM graph_entries
                    WHERE source_memory_id IN ({placeholders})
                    ORDER BY id""",
                    tuple(ids),
                )
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "source_memory_id": int(row["source_memory_id"]),
                "session_id": row["session_id"],
                "persona_id": row["persona_id"],
                "entry_type": row["entry_type"],
                "relation_type": row["relation_type"],
                "content": row["content"],
                "metadata": normalize_metadata(row["metadata"]),
            }
            for row in rows
        ]

    async def candidate_graph_evidence(
        self,
        candidate_ids: list[int],
        tokens: list[str],
        *,
        max_entries_per_candidate: int = 3,
        graph_expansion_limit: int = 24,
        graph_expansion_hops: int = 1,
        graph_second_hop_weight: float = 0.4,
    ) -> dict[int, dict[str, Any]]:
        normalized_ids = [
            int(item)
            for item in dict.fromkeys(candidate_ids)
            if int(item) > 0
        ]
        if not normalized_ids:
            return {}
        token_values = [
            str(token).strip()
            for token in dict.fromkeys(tokens)
            if str(token).strip()
        ]
        placeholders = ",".join("?" for _ in normalized_ids)
        max_entries = max(1, min(8, int(max_entries_per_candidate)))
        expansion_limit = max(1, min(200, int(graph_expansion_limit)))
        hops = max(1, min(2, int(graph_expansion_hops)))
        second_hop_weight = max(0.0, min(1.0, float(graph_second_hop_weight)))
        evidence: dict[int, dict[str, Any]] = {
            memory_id: {
                "keyword_score": 0.0,
                "node_score": 0.0,
                "graph_confidence": 0.0,
                "entries": [],
                "_entry_ids": set(),
            }
            for memory_id in normalized_ids
        }

        def confidence_from(metadata: dict[str, Any]) -> float:
            try:
                return max(0.0, min(1.0, float(metadata.get("graph_confidence", 0.7))))
            except (TypeError, ValueError):
                return 0.7

        def add_entry(
            row: dict[str, Any],
            *,
            score: float,
            source: str,
            score_field: str | None,
        ) -> None:
            memory_id = int(row["source_memory_id"])
            if memory_id not in evidence:
                return
            entry_id = int(row["id"])
            bucket = evidence[memory_id]
            if score_field:
                bucket[score_field] = max(
                    float(bucket.get(score_field) or 0.0),
                    max(0.0, min(1.0, float(score))),
                )
            if entry_id in bucket["_entry_ids"]:
                return
            metadata = normalize_metadata(row.get("metadata"))
            if score_field and score > 0:
                bucket["graph_confidence"] = max(
                    float(bucket.get("graph_confidence") or 0.0),
                    confidence_from(metadata),
                )
            bucket["_entry_ids"].add(entry_id)
            bucket["entries"].append(
                {
                    "entry_id": entry_id,
                    "content": str(row.get("content") or ""),
                    "entry_type": row.get("entry_type"),
                    "relation_type": row.get("relation_type"),
                    "metadata": metadata,
                    "source": source,
                    "score": max(0.0, min(1.0, float(score))),
                }
            )

        async def entries_for_nodes(
            db: Any,
            node_ids: list[int],
            *,
            weight: float,
            source: str,
        ) -> None:
            if not node_ids:
                return
            unique_nodes = sorted({int(node_id) for node_id in node_ids})
            node_placeholders = ",".join("?" for _ in unique_nodes)
            rows = await (
                await db.execute(
                    f"""SELECT ge.id,ge.source_memory_id,ge.content,ge.metadata,
                    ge.entry_type,ge.relation_type,
                    COUNT(DISTINCT gen.node_id) AS hit_count
                    FROM graph_entries ge
                    JOIN graph_entry_nodes gen ON gen.entry_id=ge.id
                    WHERE ge.source_memory_id IN ({placeholders})
                    AND gen.node_id IN ({node_placeholders})
                    GROUP BY ge.id
                    ORDER BY hit_count DESC, ge.id DESC
                    LIMIT ?""",
                    (*normalized_ids, *unique_nodes, expansion_limit),
                )
            ).fetchall()
            for row in rows:
                score = max(
                    0.0,
                    min(1.0, (0.35 + 0.15 * int(row["hit_count"] or 0)) * weight),
                )
                add_entry(
                    dict(row),
                    score=score,
                    source=source,
                    score_field="node_score",
                )

        async def neighbor_node_ids(
            db: Any,
            node_ids: list[int],
        ) -> list[int]:
            if not node_ids:
                return []
            unique_nodes = sorted({int(node_id) for node_id in node_ids})
            node_placeholders = ",".join("?" for _ in unique_nodes)
            rows = await (
                await db.execute(
                    f"""SELECT neighbor_id, SUM(edge_weight) AS total_weight
                    FROM (
                        SELECT target_node_id AS neighbor_id, weight AS edge_weight
                        FROM graph_edges
                        WHERE source_memory_id IN ({placeholders})
                        AND source_node_id IN ({node_placeholders})
                        AND status='active'
                        UNION ALL
                        SELECT source_node_id AS neighbor_id, weight AS edge_weight
                        FROM graph_edges
                        WHERE source_memory_id IN ({placeholders})
                        AND target_node_id IN ({node_placeholders})
                        AND status='active'
                    )
                    WHERE neighbor_id NOT IN ({node_placeholders})
                    GROUP BY neighbor_id
                    ORDER BY total_weight DESC, neighbor_id ASC
                    LIMIT ?""",
                    (
                        *normalized_ids,
                        *unique_nodes,
                        *normalized_ids,
                        *unique_nodes,
                        *unique_nodes,
                        expansion_limit,
                    ),
                )
            ).fetchall()
            return [int(row["neighbor_id"]) for row in rows]

        async with self.connect() as db:
            if token_values:
                fts = " OR ".join(
                    f'"{token.replace(chr(34), chr(34) * 2)}"'
                    for token in token_values
                )
                try:
                    rows = await (
                        await db.execute(
                            f"""SELECT ge.id,ge.source_memory_id,ge.content,ge.metadata,
                            ge.entry_type,ge.relation_type,
                            bm25(livingmemory_graph_entries_fts) AS score
                            FROM livingmemory_graph_entries_fts gf
                            JOIN graph_entries ge ON ge.id=gf.entry_id
                            WHERE livingmemory_graph_entries_fts MATCH ?
                            AND ge.source_memory_id IN ({placeholders})
                            ORDER BY score ASC LIMIT ?""",
                            (
                                fts,
                                *normalized_ids,
                                max(expansion_limit, len(normalized_ids) * max_entries),
                            ),
                        )
                    ).fetchall()
                except Exception:
                    rows = []
                if rows:
                    raw_scores = [float(row["score"]) for row in rows]
                    high, low = max(raw_scores), min(raw_scores)
                    span = high - low
                    for row in rows:
                        score = (
                            1.0
                            if span == 0
                            else (high - float(row["score"])) / span
                        )
                        add_entry(
                            dict(row),
                            score=score,
                            source="rerank_graph_keyword",
                            score_field="keyword_score",
                        )

                like_clauses = " OR ".join(
                    ["n.canonical_value LIKE ? OR n.node_value LIKE ?"]
                    * len(token_values)
                )
                like_params: list[str] = []
                for token in token_values:
                    pattern = f"%{token}%"
                    like_params.extend([pattern, pattern])
                node_rows = await (
                    await db.execute(
                        f"""SELECT DISTINCT n.id
                        FROM graph_entries ge
                        JOIN graph_entry_nodes gen ON gen.entry_id=ge.id
                        JOIN graph_nodes n ON n.id=gen.node_id
                        WHERE ge.source_memory_id IN ({placeholders})
                        AND ({like_clauses})
                        ORDER BY n.id LIMIT ?""",
                        (*normalized_ids, *like_params, expansion_limit),
                    )
                ).fetchall()
                direct_node_ids = [int(row["id"]) for row in node_rows]
                await entries_for_nodes(
                    db,
                    direct_node_ids,
                    weight=1.0,
                    source="rerank_graph_node",
                )
                first_hop_ids = await neighbor_node_ids(db, direct_node_ids)
                await entries_for_nodes(
                    db,
                    first_hop_ids,
                    weight=0.7,
                    source="rerank_graph_neighbor",
                )
                if hops >= 2 and first_hop_ids:
                    second_hop_ids = [
                        node_id
                        for node_id in await neighbor_node_ids(db, first_hop_ids)
                        if node_id not in set(direct_node_ids) | set(first_hop_ids)
                    ]
                    await entries_for_nodes(
                        db,
                        second_hop_ids,
                        weight=second_hop_weight,
                        source="rerank_graph_second_hop",
                    )

            fallback_rows = await (
                await db.execute(
                    f"""SELECT id,source_memory_id,content,metadata,entry_type,relation_type
                    FROM graph_entries
                    WHERE source_memory_id IN ({placeholders})
                    ORDER BY source_memory_id ASC, id DESC""",
                    (*normalized_ids,),
                )
            ).fetchall()
            fallback_count: dict[int, int] = {memory_id: 0 for memory_id in normalized_ids}
            for row in fallback_rows:
                memory_id = int(row["source_memory_id"])
                if fallback_count.get(memory_id, 0) >= max_entries:
                    continue
                fallback_count[memory_id] = fallback_count.get(memory_id, 0) + 1
                add_entry(
                    dict(row),
                    score=0.0,
                    source="rerank_graph_fallback",
                    score_field=None,
                )

        result: dict[int, dict[str, Any]] = {}
        for memory_id, payload in evidence.items():
            entries = sorted(
                payload["entries"],
                key=lambda item: (float(item.get("score") or 0.0), int(item.get("entry_id") or 0)),
                reverse=True,
            )[:max_entries]
            if not entries:
                continue
            result[memory_id] = {
                "keyword_score": float(payload.get("keyword_score") or 0.0),
                "node_score": float(payload.get("node_score") or 0.0),
                "graph_confidence": float(payload.get("graph_confidence") or 0.0),
                "entries": entries,
            }
        return result

    async def create_memory(
        self,
        payload: dict[str, Any],
        tokenize,
        graph_builder,
    ) -> int:
        text = str(payload.get("content") or payload.get("canonical_summary") or "").strip()
        if not text:
            raise ValueError("content is required")
        now = time.time()
        metadata = dict(payload.get("metadata") or {})
        canonical = str(payload.get("canonical_summary") or text).strip()
        metadata.update(
            {
                "session_id": payload.get("session_id"),
                "persona_id": payload.get("persona_id"),
                "importance": max(0.0, min(1.0, float(payload.get("importance", 0.5)))),
                "create_time": now,
                "last_access_time": now,
                "access_count": 0,
                "topics": list(payload.get("topics") or []),
                "participants": list(payload.get("participants") or []),
                "key_facts": list(payload.get("key_facts") or []),
                "canonical_summary": canonical,
                "persona_summary": str(payload.get("persona_summary") or canonical),
                "summary_schema_version": "v2",
                "status": str(payload.get("status") or "active"),
                "memory_type": str(payload.get("memory_type") or "GENERAL"),
                "memory_origin": str(payload.get("memory_origin") or "personalityrag_api"),
            }
        )
        op_id = uuid.uuid4().hex
        async with self._write_lock, self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            op_cursor = await db.execute(
                """INSERT INTO memory_write_ops(
                    op_type,status,step,payload,created_at,updated_at
                ) VALUES('add','pending','started',?,?,?)""",
                (json.dumps({"op_id": op_id}, ensure_ascii=False), now, now),
            )
            write_op_id = int(op_cursor.lastrowid)
            cursor = await db.execute(
                "INSERT INTO documents(doc_id,text,metadata) VALUES(?,?,?)",
                (str(uuid.uuid4()), canonical, json.dumps(metadata, ensure_ascii=False)),
            )
            memory_id = int(cursor.lastrowid)
            await db.execute(
                "INSERT INTO livingmemory_memories_fts(doc_id,content) VALUES(?,?)",
                (memory_id, " ".join(tokenize(canonical))),
            )
            graph = graph_builder(memory_id, canonical, metadata)
            await self._insert_graph(db, graph, tokenize)
            await self._insert_atoms(db, memory_id, payload.get("atoms") or [], metadata)
            await db.execute(
                """UPDATE memory_write_ops SET memory_id=?,status='needs_index',
                step='database_committed',updated_at=? WHERE id=?""",
                (memory_id, time.time(), write_op_id),
            )
            await db.commit()
        return memory_id

    async def _insert_graph(self, db, graph: dict[str, Any], tokenize) -> None:
        node_ids: dict[str, int] = {}
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for node in graph["nodes"]:
            await db.execute(
                """INSERT INTO graph_nodes(
                    node_key,node_type,node_value,canonical_value,metadata,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(node_key) DO UPDATE SET updated_at=excluded.updated_at""",
                (
                    node["node_key"],
                    node["node_type"],
                    node["value"],
                    node["canonical_value"],
                    json.dumps(node.get("metadata") or {}, ensure_ascii=False),
                    now_iso,
                    now_iso,
                ),
            )
            row = await (
                await db.execute(
                    "SELECT id FROM graph_nodes WHERE node_key=?", (node["node_key"],)
                )
            ).fetchone()
            node_ids[node["node_key"]] = int(row["id"])
        edge_ids: dict[str, int] = {}
        for edge in graph["edges"]:
            await db.execute(
                """INSERT OR REPLACE INTO graph_edges(
                    edge_key,source_node_id,target_node_id,relation_type,source_memory_id,
                    weight,confidence,status,metadata,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    edge["edge_key"],
                    node_ids[edge["source_key"]],
                    node_ids[edge["target_key"]],
                    edge["relation_type"],
                    edge["source_memory_id"],
                    edge.get("weight", 1.0),
                    edge.get("confidence", 0.8),
                    "active",
                    json.dumps(edge.get("metadata") or {}, ensure_ascii=False),
                    now_iso,
                    now_iso,
                ),
            )
            row = await (
                await db.execute(
                    "SELECT id FROM graph_edges WHERE edge_key=?", (edge["edge_key"],)
                )
            ).fetchone()
            edge_ids[edge["edge_key"]] = int(row["id"])
        for entry in graph["entries"]:
            edge_id = edge_ids.get(entry.get("edge_key", ""))
            cursor = await db.execute(
                """INSERT INTO graph_entries(
                    entry_key,source_memory_id,session_id,persona_id,entry_type,
                    relation_type,content,metadata,edge_id,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    entry["entry_key"],
                    entry["source_memory_id"],
                    entry.get("session_id"),
                    entry.get("persona_id"),
                    entry["entry_type"],
                    entry.get("relation_type"),
                    entry["content"],
                    json.dumps(entry.get("metadata") or {}, ensure_ascii=False),
                    edge_id,
                    now_iso,
                    now_iso,
                ),
            )
            entry_id = int(cursor.lastrowid)
            await db.execute(
                "UPDATE graph_entries SET vector_doc_id=? WHERE id=?",
                (entry_id, entry_id),
            )
            await db.execute(
                "INSERT INTO livingmemory_graph_entries_fts(entry_id,content) VALUES(?,?)",
                (entry_id, str(entry["content"] or "")),
            )
            await db.executemany(
                "INSERT INTO graph_entry_nodes(entry_id,node_id) VALUES(?,?)",
                [
                    (entry_id, node_ids[key])
                    for key in entry.get("node_keys", [])
                    if key in node_ids
                ],
            )

    async def _insert_atoms(
        self, db, memory_id: int, atoms: list[dict[str, Any]], metadata: dict[str, Any]
    ) -> None:
        now = time.time()
        for raw in atoms:
            content = str(raw.get("content") or "").strip()
            if not content:
                continue
            atom_type = str(raw.get("atom_type") or "unknown").lower()
            importance = max(0.0, min(1.0, float(raw.get("importance", 0.5))))
            ttl, decay = compute_atom_ttl(
                atom_type,
                importance,
                raw.get("reinforcement_count", 0),
                raw.get("event_time"),
            )
            cursor = await db.execute(
                """INSERT INTO memory_atoms(
                    parent_memory_id,atom_type,content,entities,importance,confidence,
                    created_at,last_accessed_at,event_time,ttl_days,expires_at,status,
                    reinforcement_count,decay_type,session_id,persona_id,metadata
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    memory_id,
                    atom_type,
                    content,
                    json.dumps(raw.get("entities") or [], ensure_ascii=False),
                    importance,
                    max(0.0, min(1.0, float(raw.get("confidence", 0.7)))),
                    now,
                    now,
                    raw.get("event_time"),
                    ttl,
                    now + ttl * 86400,
                    "active",
                    0,
                    decay,
                    raw.get("session_id") or metadata.get("session_id"),
                    raw.get("persona_id") or metadata.get("persona_id"),
                    json.dumps(raw.get("metadata") or {}, ensure_ascii=False),
                ),
            )
            await db.execute(
                "INSERT INTO memory_atoms_fts(atom_id,content) VALUES(?,?)",
                (int(cursor.lastrowid), content),
            )

    async def update_memory(self, memory_id: int, updates: dict[str, Any], tokenize, graph_builder) -> bool:
        current = await self.get_document(memory_id)
        if not current:
            return False
        text = str(updates.get("content", current["text"])).strip()
        if not text:
            raise ValueError("content cannot be empty")
        metadata = dict(current["metadata"])
        metadata.update(dict(updates.get("metadata") or {}))
        for key in ("importance", "status", "memory_type", "session_id", "persona_id"):
            if key in updates:
                metadata[key] = updates[key]
        metadata["updated_at"] = time.time()
        if "content" in updates:
            metadata["canonical_summary"] = text
        async with self._write_lock, self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            await db.execute(
                "UPDATE documents SET text=?,metadata=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (text, json.dumps(metadata, ensure_ascii=False), memory_id),
            )
            await db.execute(
                "DELETE FROM livingmemory_memories_fts WHERE doc_id=?", (memory_id,)
            )
            await db.execute(
                "INSERT INTO livingmemory_memories_fts(doc_id,content) VALUES(?,?)",
                (memory_id, " ".join(tokenize(text))),
            )
            await self._delete_graph(db, memory_id)
            await self._insert_graph(db, graph_builder(memory_id, text, metadata), tokenize)
            await db.commit()
        return True

    async def update_memory_metadata(
        self, memory_id: int, updates: dict[str, Any]
    ) -> bool:
        async with self._write_lock, self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            row = await (
                await db.execute(
                    "SELECT metadata FROM documents WHERE id=?", (memory_id,)
                )
            ).fetchone()
            if not row:
                await db.rollback()
                return False

            metadata = normalize_metadata(row["metadata"])
            metadata.update(dict(updates.get("metadata") or {}))
            for key in (
                "importance",
                "status",
                "memory_type",
                "session_id",
                "persona_id",
            ):
                if key in updates:
                    metadata[key] = updates[key]
            metadata["updated_at"] = time.time()
            await db.execute(
                "UPDATE documents SET metadata=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (json.dumps(metadata, ensure_ascii=False), memory_id),
            )

            graph_metadata_updates = {
                key: metadata.get(key)
                for key in (
                    "importance",
                    "session_id",
                    "persona_id",
                    "last_access_time",
                )
                if key in updates or key in dict(updates.get("metadata") or {})
            }
            if graph_metadata_updates:
                graph_rows = await (
                    await db.execute(
                        "SELECT id,metadata FROM graph_entries WHERE source_memory_id=?",
                        (memory_id,),
                    )
                ).fetchall()
                for graph_row in graph_rows:
                    graph_metadata = normalize_metadata(graph_row["metadata"])
                    graph_metadata.update(graph_metadata_updates)
                    await db.execute(
                        "UPDATE graph_entries SET metadata=?,updated_at=? WHERE id=?",
                        (
                            json.dumps(graph_metadata, ensure_ascii=False),
                            time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                            int(graph_row["id"]),
                        ),
                    )
            if "session_id" in updates or "persona_id" in updates:
                assignments = []
                params: list[Any] = []
                for key in ("session_id", "persona_id"):
                    if key in updates:
                        assignments.append(f"{key}=?")
                        params.append(metadata.get(key))
                assignments.append("updated_at=?")
                params.append(
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                )
                params.append(memory_id)
                await db.execute(
                    f"UPDATE graph_entries SET {','.join(assignments)} WHERE source_memory_id=?",
                    params,
                )
                atom_assignments = []
                atom_params: list[Any] = []
                for key in ("session_id", "persona_id"):
                    if key in updates:
                        atom_assignments.append(f"{key}=?")
                        atom_params.append(metadata.get(key))
                atom_params.append(memory_id)
                await db.execute(
                    f"UPDATE memory_atoms SET {','.join(atom_assignments)} WHERE parent_memory_id=?",
                    atom_params,
                )
            await db.commit()
        return True

    async def update_memory_persona(
        self, memory_id: int, persona_id: str | None
    ) -> bool:
        normalized_persona_id = str(persona_id or "").strip() or None
        async with self._write_lock, self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            row = await (
                await db.execute(
                    "SELECT metadata FROM documents WHERE id=?", (memory_id,)
                )
            ).fetchone()
            if not row:
                await db.rollback()
                return False
            metadata = normalize_metadata(row["metadata"])
            metadata["persona_id"] = normalized_persona_id
            metadata["updated_at"] = time.time()
            await db.execute(
                "UPDATE documents SET metadata=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (json.dumps(metadata, ensure_ascii=False), memory_id),
            )
            await db.execute(
                "UPDATE graph_entries SET persona_id=?,updated_at=? WHERE source_memory_id=?",
                (
                    normalized_persona_id,
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    memory_id,
                ),
            )
            await db.execute(
                "UPDATE memory_atoms SET persona_id=? WHERE parent_memory_id=?",
                (normalized_persona_id, memory_id),
            )
            await db.commit()
        return True

    async def delete_memories(self, memory_ids: Iterable[int]) -> int:
        ids = sorted({int(value) for value in memory_ids})
        if not ids:
            return 0
        async with self._write_lock, self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            deleted = 0
            for memory_id in ids:
                exists = await (
                    await db.execute("SELECT 1 FROM documents WHERE id=?", (memory_id,))
                ).fetchone()
                if not exists:
                    continue
                await self._delete_graph(db, memory_id)
                atom_ids = [
                    row["id"]
                    for row in await (
                        await db.execute(
                            "SELECT id FROM memory_atoms WHERE parent_memory_id=?",
                            (memory_id,),
                        )
                    ).fetchall()
                ]
                for atom_id in atom_ids:
                    await db.execute(
                        "DELETE FROM memory_atoms_fts WHERE atom_id=?", (atom_id,)
                    )
                await db.execute(
                    "DELETE FROM memory_atoms WHERE parent_memory_id=?", (memory_id,)
                )
                await db.execute(
                    "DELETE FROM livingmemory_memories_fts WHERE doc_id=?", (memory_id,)
                )
                await db.execute("DELETE FROM documents WHERE id=?", (memory_id,))
                await db.execute(
                    "DELETE FROM memory_write_ops WHERE memory_id=?", (memory_id,)
                )
                deleted += 1
            await db.commit()
        return deleted

    async def mark_memory_indexed(self, memory_id: int, *, generation: str = "") -> None:
        payload = {"index_generation": generation} if generation else {}
        async with self.connect() as db:
            await db.execute(
                """UPDATE memory_write_ops
                SET status='completed', step='index_incremental_completed',
                    payload=?, updated_at=?
                WHERE memory_id=? AND op_type='add'""",
                (json.dumps(payload, ensure_ascii=False), time.time(), int(memory_id)),
            )
            await db.commit()

    async def _delete_graph(self, db, memory_id: int) -> None:
        entry_ids = [
            int(row["id"])
            for row in await (
                await db.execute(
                    "SELECT id FROM graph_entries WHERE source_memory_id=?", (memory_id,)
                )
            ).fetchall()
        ]
        for entry_id in entry_ids:
            await db.execute(
                "DELETE FROM livingmemory_graph_entries_fts WHERE entry_id=?",
                (entry_id,),
            )
        await db.execute(
            "DELETE FROM graph_entries WHERE source_memory_id=?", (memory_id,)
        )
        await db.execute(
            "DELETE FROM graph_edges WHERE source_memory_id=?", (memory_id,)
        )
        await db.execute(
            """DELETE FROM graph_nodes WHERE id NOT IN(
                SELECT source_node_id FROM graph_edges UNION SELECT target_node_id FROM graph_edges
                UNION SELECT node_id FROM graph_entry_nodes
            )"""
        )

    async def touch_documents(self, memory_ids: Iterable[int]) -> None:
        ids = sorted({int(value) for value in memory_ids})
        if not ids:
            return
        now = time.time()
        async with self.connect() as db:
            for memory_id in ids:
                row = await (
                    await db.execute(
                        "SELECT metadata FROM documents WHERE id=?", (memory_id,)
                    )
                ).fetchone()
                if not row:
                    continue
                metadata = normalize_metadata(row["metadata"])
                metadata["last_access_time"] = now
                metadata["access_count"] = int(metadata.get("access_count", 0) or 0) + 1
                await db.execute(
                    "UPDATE documents SET metadata=? WHERE id=?",
                    (json.dumps(metadata, ensure_ascii=False), memory_id),
                )
            await db.commit()

    async def statistics(self) -> dict[str, Any]:
        async with self.connect() as db:
            counts = {}
            for key, table in (
                ("total_memories", "documents"),
                ("graph_nodes", "graph_nodes"),
                ("graph_edges", "graph_edges"),
                ("graph_entries", "graph_entries"),
                ("atom_count", "memory_atoms"),
            ):
                row = await (
                    await db.execute(f"SELECT COUNT(*) AS value FROM {table}")
                ).fetchone()
                counts[key] = int(row["value"])
            rows = await (
                await db.execute("SELECT metadata FROM documents")
            ).fetchall()
            status: dict[str, int] = {}
            sessions: dict[str, int] = {}
            importance = {f"{i}-{i+1}": 0 for i in range(10)}
            for row in rows:
                meta = normalize_metadata(row["metadata"])
                state = str(meta.get("status") or "active")
                status[state] = status.get(state, 0) + 1
                session = meta.get("session_id")
                if session and state == "active":
                    sessions[str(session)] = sessions.get(str(session), 0) + 1
                value = float(meta.get("importance", 0.5) or 0.5)
                value = value * 10 if value <= 1 else value
                bucket = min(9, max(0, int(value)))
                importance[f"{bucket}-{bucket+1}"] += 1
            atom_rows = await (
                await db.execute(
                    "SELECT atom_type,COUNT(*) AS value FROM memory_atoms GROUP BY atom_type"
                )
            ).fetchall()
        conversation = {"sessions": 0, "messages": 0, "pending_messages": 0}
        if self.conversations_path.exists():
            db = await aiosqlite.connect(self.conversations_path)
            db.row_factory = aiosqlite.Row
            try:
                for key, table in (("sessions", "sessions"), ("messages", "messages")):
                    row = await (
                        await db.execute(f"SELECT COUNT(*) AS value FROM {table}")
                    ).fetchone()
                    conversation[key] = int(row["value"])
                session_rows = await (
                    await db.execute("SELECT message_count, metadata FROM sessions")
                ).fetchall()
                pending_messages = 0
                for row in session_rows:
                    metadata = normalize_metadata(row["metadata"])
                    message_count = int(row["message_count"] or 0)
                    try:
                        last_summarized_index = int(
                            metadata.get("last_summarized_index") or 0
                        )
                    except (TypeError, ValueError):
                        last_summarized_index = 0
                    pending_messages += max(0, message_count - last_summarized_index)
                conversation["pending_messages"] = pending_messages
            finally:
                await db.close()
        return {
            **counts,
            "status_breakdown": status,
            "sessions": sessions,
            "importance_distribution": importance,
            "atom_breakdown": {
                row["atom_type"]: int(row["value"]) for row in atom_rows
            },
            "conversation_counts": conversation,
        }

    async def summary_statistics(self) -> dict[str, Any]:
        """Return only the scalar counters needed by library cards."""
        count_keys = (
            "total_memories",
            "graph_nodes",
            "graph_edges",
            "graph_entries",
            "atom_count",
        )
        counts = {key: 0 for key in count_keys}
        session_count = 0
        if self.db_path.exists():
            async with self.connect() as db:
                row = await (
                    await db.execute(
                        """SELECT
                        (SELECT COUNT(*) FROM documents) AS total_memories,
                        (SELECT COUNT(*) FROM graph_nodes) AS graph_nodes,
                        (SELECT COUNT(*) FROM graph_edges) AS graph_edges,
                        (SELECT COUNT(*) FROM graph_entries) AS graph_entries,
                        (SELECT COUNT(*) FROM memory_atoms) AS atom_count,
                        (SELECT COUNT(DISTINCT CAST(
                            json_extract(metadata,'$.session_id') AS TEXT
                        )) FROM documents
                        WHERE json_valid(metadata)
                          AND NULLIF(TRIM(CAST(
                              json_extract(metadata,'$.session_id') AS TEXT
                          )), '') IS NOT NULL
                          AND COALESCE(
                              json_extract(metadata,'$.status'), 'active'
                          ) = 'active') AS session_count"""
                    )
                ).fetchone()
            counts = {key: int(row[key] or 0) for key in count_keys}
            session_count = int(row["session_count"] or 0)
        conversation = {"sessions": 0, "messages": 0, "pending_messages": 0}
        if self.conversations_path.exists():
            db = await aiosqlite.connect(self.conversations_path)
            db.row_factory = aiosqlite.Row
            try:
                row = await (
                    await db.execute(
                        """SELECT
                        (SELECT COUNT(*) FROM sessions) AS sessions,
                        (SELECT COUNT(*) FROM messages) AS messages,
                        (SELECT COALESCE(SUM(MAX(
                            0,
                            message_count - CAST(COALESCE(CASE
                                WHEN json_valid(metadata)
                                THEN json_extract(
                                    metadata,'$.last_summarized_index'
                                )
                                ELSE 0
                            END,0) AS INTEGER)
                        )),0) FROM sessions) AS pending_messages"""
                    )
                ).fetchone()
                conversation = {
                    key: int(row[key] or 0)
                    for key in ("sessions", "messages", "pending_messages")
                }
            finally:
                await db.close()
        return {
            **counts,
            "session_count": session_count,
            "conversation_counts": conversation,
        }

    async def integrity_report(self) -> dict[str, Any]:
        def inspect(path: Path) -> dict[str, Any]:
            if not path.exists():
                return {"exists": False}
            con = sqlite3.connect(path)
            try:
                integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
                fk = con.execute("PRAGMA foreign_key_check").fetchall()
                tables = {
                    row[0]: con.execute(f'SELECT COUNT(*) FROM "{row[0]}"').fetchone()[0]
                    for row in con.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    )
                }
                return {
                    "exists": True,
                    "integrity": integrity,
                    "foreign_key_errors": len(fk),
                    "tables": tables,
                    "sha256": sha256_file(path),
                }
            finally:
                con.close()

        return {
            "livingmemory": await asyncio.to_thread(inspect, self.db_path),
            "conversations": await asyncio.to_thread(
                inspect, self.conversations_path
            ),
        }
