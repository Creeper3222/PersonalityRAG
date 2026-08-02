from __future__ import annotations

import asyncio
import copy
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
from .source import serialize_source_messages
from ...sqlite_pool import SQLiteConnectionPool
from ...version import VERSION


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


def normalize_document_metadata(value: Any) -> dict[str, Any]:
    metadata = normalize_metadata(value)
    metadata.pop("memory_type", None)
    return metadata


class Storage:
    def __init__(
        self,
        data_dir: Path,
        system_path: Path | None = None,
        *,
        pooled: bool = False,
        system_pool: SQLiteConnectionPool | None = None,
        initialize_system: bool = True,
    ):
        self.data_dir = data_dir
        self.db_path = data_dir / "livingmemory.db"
        self.conversations_path = data_dir / "conversations.db"
        self.system_path = system_path or (data_dir / "personalityrag_system.db")
        self._write_lock = asyncio.Lock()
        self._main_pool = SQLiteConnectionPool(self.db_path, size=2) if pooled else None
        self._conversations_pool = (
            SQLiteConnectionPool(self.conversations_path, size=2) if pooled else None
        )
        self._system_pool = system_pool
        self._initialize_system = bool(initialize_system and system_pool is None)
        self._mutation_revision = 0
        self._statistics_cache: dict[
            str, tuple[tuple[Any, ...], float, dict[str, Any]]
        ] = {}
        self._statistics_flights: dict[
            tuple[str, tuple[Any, ...]], asyncio.Task[dict[str, Any]]
        ] = {}
        self._statistics_lock = asyncio.Lock()

    def _invalidate_statistics(self) -> None:
        self._mutation_revision += 1
        self._statistics_cache.clear()

    def _statistics_signature(self) -> tuple[Any, ...]:
        signature: list[Any] = [self._mutation_revision]
        for base in (self.db_path, self.conversations_path):
            for suffix in ("", "-wal"):
                path = Path(str(base) + suffix)
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    signature.append(None)
                else:
                    signature.append((stat.st_size, stat.st_mtime_ns))
        return tuple(signature)

    async def _cached_statistics(
        self,
        mode: str,
        loader,
        *,
        ttl_seconds: float = 2.0,
    ) -> dict[str, Any]:
        now = time.monotonic()
        signature = self._statistics_signature()
        cached = self._statistics_cache.get(mode)
        if cached and cached[0] == signature and now - cached[1] <= ttl_seconds:
            return copy.deepcopy(cached[2])
        async with self._statistics_lock:
            cached = self._statistics_cache.get(mode)
            if cached and cached[0] == signature and now - cached[1] <= ttl_seconds:
                return copy.deepcopy(cached[2])
            flight_key = (mode, signature)
            task = self._statistics_flights.get(flight_key)
            if task is None:
                task = asyncio.create_task(loader())
                self._statistics_flights[flight_key] = task
        try:
            payload = await task
        finally:
            async with self._statistics_lock:
                if self._statistics_flights.get(flight_key) is task:
                    self._statistics_flights.pop(flight_key, None)
        completed_signature = self._statistics_signature()
        if completed_signature == signature:
            self._statistics_cache[mode] = (
                signature,
                time.monotonic(),
                copy.deepcopy(payload),
            )
        return copy.deepcopy(payload)

    @asynccontextmanager
    async def connect(self, *, system: bool = False):
        path = self.system_path if system else self.db_path
        pool = self._system_pool if system else self._main_pool
        if pool is not None:
            lease = await pool.acquire()
            total_changes = lease.total_changes
            try:
                yield lease
            finally:
                if not system and lease.total_changes != total_changes:
                    self._invalidate_statistics()
                await lease.close()
            return
        db = await aiosqlite.connect(path)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA busy_timeout=10000")
        await db.execute("PRAGMA foreign_keys=ON")
        total_changes = db.total_changes
        try:
            yield db
        finally:
            if not system and db.total_changes != total_changes:
                self._invalidate_statistics()
            await db.close()

    @asynccontextmanager
    async def conversation_connect(self):
        if self._conversations_pool is not None:
            lease = await self._conversations_pool.acquire()
            total_changes = lease.total_changes
            try:
                yield lease
            finally:
                if lease.total_changes != total_changes:
                    self._invalidate_statistics()
                await lease.close()
            return
        db = await aiosqlite.connect(self.conversations_path)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA busy_timeout=10000")
        total_changes = db.total_changes
        try:
            yield db
        finally:
            if db.total_changes != total_changes:
                self._invalidate_statistics()
            await db.close()

    async def close(self) -> None:
        pools = [
            pool
            for pool in (self._main_pool, self._conversations_pool)
            if pool is not None
        ]
        await asyncio.gather(*(pool.close() for pool in pools))

    @staticmethod
    async def _table_exists(db: Any, table_name: str) -> bool:
        row = await (
            await db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
                (table_name,),
            )
        ).fetchone()
        return row is not None

    @classmethod
    async def _ensure_memory_sources(cls, db: Any) -> None:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_sources (
                memory_id INTEGER PRIMARY KEY,
                source_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )

    @staticmethod
    def _is_missing_optional_table_error(
        error: sqlite3.OperationalError,
        table_name: str,
    ) -> bool:
        message = str(error).casefold()
        return "no such table" in message and table_name.casefold() in message

    @classmethod
    async def _memory_source_row(cls, db: Any, memory_id: int) -> Any | None:
        """Read the optional 2.5 source row without a schema probe race."""

        try:
            return await (
                await db.execute(
                    """SELECT memory_id,source_json,created_at,updated_at
                    FROM memory_sources WHERE memory_id=?""",
                    (int(memory_id),),
                )
            ).fetchone()
        except sqlite3.OperationalError as error:
            if cls._is_missing_optional_table_error(error, "memory_sources"):
                return None
            raise

    async def has_memory_sources_table(self) -> bool:
        """Inspect the optional 2.5 source table without mutating an old v8 DB."""

        async with self.connect() as db:
            return await self._table_exists(db, "memory_sources")

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
            await self._remove_obsolete_document_metadata(db)
            await db.commit()
        if self._initialize_system:
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
                    database_type TEXT NOT NULL DEFAULT 'livingmemory_v8',
                    database_id TEXT,
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
                    started_at REAL,
                    finished_at REAL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    status_history TEXT NOT NULL DEFAULT '[]',
                    database_state_before TEXT,
                    database_state_after TEXT,
                    database_state_capture_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS migration_runs (
                    id TEXT PRIMARY KEY,
                    library_id TEXT,
                    database_type TEXT NOT NULL DEFAULT 'livingmemory_v8',
                    database_id TEXT,
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
                    for row in await (
                        await db.execute("PRAGMA table_info(jobs)")
                    ).fetchall()
                }
                if "library_id" not in job_columns:
                    await db.execute("ALTER TABLE jobs ADD COLUMN library_id TEXT")
                for name, sql_type in (
                    ("database_type", "TEXT NOT NULL DEFAULT 'livingmemory_v8'"),
                    ("database_id", "TEXT"),
                    ("operation", "TEXT"),
                    ("checkpoint", "TEXT"),
                    ("status_reason", "TEXT NOT NULL DEFAULT ''"),
                    ("control_requested", "TEXT"),
                    ("resumable", "INTEGER NOT NULL DEFAULT 0"),
                    ("started_at", "REAL"),
                    ("finished_at", "REAL"),
                    ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
                    ("status_history", "TEXT NOT NULL DEFAULT '[]'"),
                    ("database_state_before", "TEXT"),
                    ("database_state_after", "TEXT"),
                    ("database_state_capture_error", "TEXT"),
                ):
                    if name not in job_columns:
                        await db.execute(
                            f"ALTER TABLE jobs ADD COLUMN {name} {sql_type}"
                        )
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
                for name, sql_type in (
                    ("database_type", "TEXT NOT NULL DEFAULT 'livingmemory_v8'"),
                    ("database_id", "TEXT"),
                ):
                    if name not in migration_columns:
                        await db.execute(
                            f"ALTER TABLE migration_runs ADD COLUMN {name} {sql_type}"
                        )
                await db.execute(
                    "INSERT OR REPLACE INTO schema_info(key,value) VALUES('service_version',?)",
                    (VERSION,),
                )
                await db.commit()
        await self._initialize_conversations()

    @staticmethod
    async def _remove_obsolete_document_metadata(db) -> int:
        cursor = await db.execute(
            "SELECT id,metadata FROM documents WHERE metadata LIKE '%\"memory_type\"%'"
        )
        updates: list[tuple[str, int]] = []
        while True:
            rows = await cursor.fetchmany(200)
            if not rows:
                break
            for row in rows:
                metadata = normalize_metadata(row["metadata"])
                if "memory_type" not in metadata:
                    continue
                metadata.pop("memory_type", None)
                updates.append(
                    (json.dumps(metadata, ensure_ascii=False), int(row["id"]))
                )
        if updates:
            await db.executemany(
                "UPDATE documents SET metadata=? WHERE id=?",
                updates,
            )
        return len(updates)

    async def _initialize_conversations(self) -> None:
        async with self.conversation_connect() as db:
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
                CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp_id
                ON messages(session_id,timestamp,id);
                CREATE INDEX IF NOT EXISTS idx_messages_dedup_key
                ON messages(json_extract(metadata,'$.dedup_key'));
                """
            )
            await db.commit()

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

        async with self._write_lock, self.conversation_connect() as db:
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
                    SET message_count = message_count + 1,
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
                (timestamp, sender_id, sender_id, sender_id, session_id),
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
        async with self.conversation_connect() as db:
            session = await self._get_conversation_session(db, session_id)
            if not session:
                return None
            messages = await self._get_conversation_messages(
                db,
                session_id,
                limit=limit,
            )
            return {"session": session, "messages": messages}

    async def get_conversation_messages(
        self,
        session_id: str,
        *,
        start_index: int | None = None,
        end_index: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        async with self.conversation_connect() as db:
            return await self._get_conversation_messages(
                db,
                session_id,
                start_index=start_index,
                end_index=end_index,
                limit=limit,
            )

    async def _get_conversation_messages(
        self,
        db: aiosqlite.Connection,
        session_id: str,
        *,
        start_index: int | None = None,
        end_index: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
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

    async def update_conversation_metadata(
        self, session_id: str, patch: dict[str, Any]
    ) -> dict[str, Any] | None:
        async with self._write_lock, self.conversation_connect() as db:
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

    async def clear_conversation(self, session_id: str) -> dict[str, Any]:
        async with self._write_lock, self.conversation_connect() as db:
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

    async def trim_conversation(
        self, session_id: str, delete_count: int
    ) -> dict[str, Any]:
        delete_count = max(0, int(delete_count))
        if delete_count <= 0:
            return {"deleted": 0, "session": await self.get_conversation(session_id)}
        async with self._write_lock, self.conversation_connect() as db:
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
                return {
                    "deleted": 0,
                    "session": await self._get_conversation_session(db, session_id),
                }
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
            metadata["last_summarized_index"] = max(0, last_summarized_index - deleted)
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

    async def rebuild_fts(self, tokenize) -> dict[str, int]:
        async with self._write_lock, self.connect() as db:
            await db.execute("DELETE FROM livingmemory_memories_fts")
            cursor = await db.execute(
                """SELECT id,text FROM documents
                WHERE COALESCE(json_extract(metadata,'$.status'),'active')='active'
                ORDER BY id"""
            )
            rows = await cursor.fetchall()
            memory_rows = [
                (int(row["id"]), " ".join(tokenize(row["text"] or ""))) for row in rows
            ]
            await db.executemany(
                "INSERT INTO livingmemory_memories_fts(doc_id,content) VALUES(?,?)",
                memory_rows,
            )
            await db.execute("DELETE FROM livingmemory_graph_entries_fts")
            cursor = await db.execute(
                """SELECT ge.id,ge.content FROM graph_entries ge
                JOIN documents d ON d.id=ge.source_memory_id
                WHERE COALESCE(json_extract(d.metadata,'$.status'),'active')='active'
                ORDER BY ge.id"""
            )
            graph_rows_raw = await cursor.fetchall()
            graph_rows = [
                (int(row["id"]), str(row["content"] or "")) for row in graph_rows_raw
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

    async def sync_memory_from(
        self,
        source: "Storage",
        memory_id: int,
        tokenize,
        graph_builder,
    ) -> str:
        """Reconcile one source memory into a shadow rebuild database."""

        memory_id = int(memory_id)
        async with source.connect() as source_db:
            document = await (
                await source_db.execute(
                    """SELECT id,doc_id,text,metadata,created_at,updated_at
                    FROM documents WHERE id=?""",
                    (memory_id,),
                )
            ).fetchone()
            source_row = await source._memory_source_row(source_db, memory_id)
            atom_rows = await (
                await source_db.execute(
                    "SELECT * FROM memory_atoms WHERE parent_memory_id=? ORDER BY id",
                    (memory_id,),
                )
            ).fetchall()

        await self.delete_memories([memory_id])
        if document is None:
            return "deleted"

        metadata = normalize_document_metadata(document["metadata"])
        async with self._write_lock, self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            await db.execute(
                """INSERT INTO documents(
                    id,doc_id,text,metadata,created_at,updated_at
                ) VALUES(?,?,?,?,?,?)""",
                (
                    memory_id,
                    document["doc_id"],
                    document["text"],
                    document["metadata"],
                    document["created_at"],
                    document["updated_at"],
                ),
            )
            if source_row is not None:
                await self._ensure_memory_sources(db)
                await db.execute(
                    """INSERT INTO memory_sources(
                        memory_id,source_json,created_at,updated_at
                    ) VALUES(?,?,?,?)""",
                    (
                        memory_id,
                        source_row["source_json"],
                        source_row["created_at"],
                        source_row["updated_at"],
                    ),
                )
            if str(metadata.get("status") or "active") == "active":
                text = str(document["text"] or "")
                await db.execute(
                    "INSERT INTO livingmemory_memories_fts(doc_id,content) VALUES(?,?)",
                    (memory_id, " ".join(tokenize(text))),
                )
                await self._insert_graph(
                    db,
                    graph_builder(memory_id, text, metadata),
                    tokenize,
                )
            if atom_rows:
                atom_columns = list(atom_rows[0].keys())
                quoted = ",".join(f'"{column}"' for column in atom_columns)
                placeholders = ",".join("?" for _ in atom_columns)
                await db.executemany(
                    f"INSERT INTO memory_atoms({quoted}) VALUES({placeholders})",
                    [tuple(row[column] for column in atom_columns) for row in atom_rows],
                )
                await db.executemany(
                    "INSERT INTO memory_atoms_fts(atom_id,content) VALUES(?,?)",
                    [
                        (int(row["id"]), str(row["content"] or ""))
                        for row in atom_rows
                        if str(row["status"] or "active") == "active"
                    ],
                )
            await db.commit()
        return (
            "active"
            if str(metadata.get("status") or "active") == "active"
            else "inactive"
        )

    async def replace_search_derivatives_from(self, source_path: Path) -> None:
        """Atomically replace graph and FTS tables from a verified shadow DB."""

        source_path = Path(source_path).resolve()
        if source_path == self.db_path.resolve() or not source_path.is_file():
            raise ValueError("shadow database path is invalid")
        async with self._write_lock, self.connect() as db:
            await db.execute("ATTACH DATABASE ? AS shadow_rebuild", (str(source_path),))
            try:
                await db.execute("BEGIN IMMEDIATE")
                await db.execute("DELETE FROM livingmemory_memories_fts")
                await db.execute("DELETE FROM livingmemory_graph_entries_fts")
                await db.execute("DELETE FROM graph_entry_nodes")
                await db.execute("DELETE FROM graph_entries")
                await db.execute("DELETE FROM graph_edges")
                await db.execute("DELETE FROM graph_nodes")

                for table in (
                    "graph_nodes",
                    "graph_edges",
                    "graph_entries",
                    "graph_entry_nodes",
                ):
                    columns = [
                        str(row["name"])
                        for row in await (
                            await db.execute(f'PRAGMA main.table_info("{table}")')
                        ).fetchall()
                    ]
                    quoted = ",".join(f'"{column}"' for column in columns)
                    await db.execute(
                        f'INSERT INTO main."{table}"({quoted}) '
                        f'SELECT {quoted} FROM shadow_rebuild."{table}"'
                    )
                await db.execute(
                    """INSERT INTO livingmemory_memories_fts(doc_id,content)
                    SELECT doc_id,content
                    FROM shadow_rebuild.livingmemory_memories_fts"""
                )
                await db.execute(
                    """INSERT INTO livingmemory_graph_entries_fts(entry_id,content)
                    SELECT entry_id,content
                    FROM shadow_rebuild.livingmemory_graph_entries_fts"""
                )
                await db.commit()
            except Exception:
                await db.rollback()
                raise
            finally:
                await db.execute("DETACH DATABASE shadow_rebuild")

    async def graph_integrity_report(self) -> dict[str, int]:
        async with self.connect() as db:

            async def count_sql(sql: str) -> int:
                row = await (await db.execute(sql)).fetchone()
                return int(row[0] or 0)

            documents = await count_sql("SELECT COUNT(*) FROM documents")
            active_documents = await count_sql(
                """SELECT COUNT(*) FROM documents
                WHERE COALESCE(json_extract(metadata,'$.status'),'active')='active'"""
            )
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
                """SELECT COUNT(DISTINCT ge.source_memory_id)
                FROM graph_entries ge
                JOIN documents d ON d.id=ge.source_memory_id
                WHERE COALESCE(json_extract(d.metadata,'$.status'),'active')='active'"""
            )
            docs_without_graph = await count_sql(
                """SELECT COUNT(*) FROM documents d
                WHERE COALESCE(json_extract(d.metadata,'$.status'),'active')='active'
                AND NOT EXISTS (
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
            "active_documents": active_documents,
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
            active_documents = await count_sql(
                """SELECT COUNT(*) FROM documents
                WHERE COALESCE(json_extract(metadata,'$.status'),'active')='active'"""
            )
            document_fts = await count_sql(
                "SELECT COUNT(*) FROM livingmemory_memories_fts"
            )
            graph_entries = await count_sql("SELECT COUNT(*) FROM graph_entries")
            active_graph_entries = await count_sql(
                """SELECT COUNT(*) FROM graph_entries ge
                JOIN documents d ON d.id=ge.source_memory_id
                WHERE COALESCE(json_extract(d.metadata,'$.status'),'active')='active'"""
            )
            graph_fts = await count_sql(
                "SELECT COUNT(*) FROM livingmemory_graph_entries_fts"
            )
            active_atoms = await count_sql(
                "SELECT COUNT(*) FROM memory_atoms WHERE status='active'"
            )
            atom_fts = await count_sql("SELECT COUNT(*) FROM memory_atoms_fts")

        return {
            "documents": documents,
            "active_documents": active_documents,
            "document_fts": document_fts,
            "graph_entries": graph_entries,
            "active_graph_entries": active_graph_entries,
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
                await db.execute(
                    """SELECT COUNT(*) FROM documents
                    WHERE COALESCE(json_extract(metadata,'$.status'),'active')='active'"""
                )
            ).fetchone()
            total = int(total_row[0] or 0)
            await db.execute("BEGIN IMMEDIATE")
            await db.execute("DELETE FROM livingmemory_graph_entries_fts")
            await db.execute("DELETE FROM graph_entry_nodes")
            await db.execute("DELETE FROM graph_entries")
            await db.execute("DELETE FROM graph_edges")
            await db.execute("DELETE FROM graph_nodes")

            cursor = await db.execute(
                """SELECT id,text,metadata FROM documents
                WHERE COALESCE(json_extract(metadata,'$.status'),'active')='active'
                ORDER BY id"""
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
            "importance_desc": "COALESCE(json_extract(metadata,'$.importance'),0.5) DESC,id DESC",
            "importance_asc": "COALESCE(json_extract(metadata,'$.importance'),0.5) ASC,id ASC",
            "id_desc": "id DESC",
            "id_asc": "id ASC",
        }
        order = order_map.get(sort, order_map["created_desc"])
        offset = (max(1, page) - 1) * page_size
        async with self.connect() as db:
            source_projection = (
                "CASE WHEN EXISTS(SELECT 1 FROM memory_sources s "
                "WHERE s.memory_id=documents.id) THEN 1 ELSE 0 END"
                if await self._table_exists(db, "memory_sources")
                else "0"
            )
            count_row = await (
                await db.execute(
                    f"SELECT COUNT(*) AS value FROM documents {where}", params
                )
            ).fetchone()
            rows = await (
                await db.execute(
                    f"""SELECT id,doc_id,text,metadata,created_at,updated_at,
                    {source_projection} AS has_source
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
            source_projection = (
                "CASE WHEN EXISTS(SELECT 1 FROM memory_sources s "
                "WHERE s.memory_id=documents.id) THEN 1 ELSE 0 END"
                if await self._table_exists(db, "memory_sources")
                else "0"
            )
            row = await (
                await db.execute(
                    f"""SELECT id,doc_id,text,metadata,created_at,updated_at,
                    {source_projection} AS has_source
                    FROM documents WHERE id=?""",
                    (memory_id,),
                )
            ).fetchone()
        return self._document_row(row) if row else None

    @staticmethod
    def _document_row(row: aiosqlite.Row) -> dict[str, Any]:
        metadata = normalize_document_metadata(row["metadata"])
        has_source = bool(row["has_source"]) if "has_source" in row.keys() else bool(
            metadata.get("has_source", False)
        )
        return {
            "id": int(row["id"]),
            "doc_id": row["doc_id"],
            "text": row["text"],
            "metadata": metadata,
            "has_source": has_source,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def get_memory_source(self, memory_id: int) -> list[dict[str, Any]]:
        async with self.connect() as db:
            row = await self._memory_source_row(db, memory_id)
        if not row:
            return []
        try:
            source = json.loads(row["source_json"])
        except (TypeError, json.JSONDecodeError):
            return []
        return source if isinstance(source, list) else []

    async def save_memory_source(
        self, memory_id: int, source_messages: Any
    ) -> list[dict[str, Any]]:
        source = serialize_source_messages(source_messages)
        if not source:
            raise ValueError("source_messages must contain at least one text message")
        now = time.time()
        async with self._write_lock, self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            exists = await (
                await db.execute("SELECT 1 FROM documents WHERE id=?", (int(memory_id),))
            ).fetchone()
            if not exists:
                await db.rollback()
                raise KeyError(memory_id)
            await self._ensure_memory_sources(db)
            await db.execute(
                """INSERT INTO memory_sources(memory_id,source_json,created_at,updated_at)
                VALUES(?,?,?,?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    source_json=excluded.source_json,updated_at=excluded.updated_at""",
                (int(memory_id), json.dumps(source, ensure_ascii=False), now, now),
            )
            row = await (
                await db.execute("SELECT metadata FROM documents WHERE id=?", (int(memory_id),))
            ).fetchone()
            metadata = normalize_document_metadata(row["metadata"])
            metadata["has_source"] = True
            metadata["source_message_count"] = len(source)
            await db.execute(
                "UPDATE documents SET metadata=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (json.dumps(metadata, ensure_ascii=False), int(memory_id)),
            )
            await db.commit()
        return source

    async def delete_memory_source(self, memory_id: int) -> bool:
        async with self._write_lock, self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            if not await self._table_exists(db, "memory_sources"):
                await db.rollback()
                return False
            cursor = await db.execute(
                "DELETE FROM memory_sources WHERE memory_id=?", (int(memory_id),)
            )
            row = await (
                await db.execute("SELECT metadata FROM documents WHERE id=?", (int(memory_id),))
            ).fetchone()
            if row:
                metadata = normalize_document_metadata(row["metadata"])
                metadata["has_source"] = False
                metadata["source_message_count"] = 0
                await db.execute(
                    "UPDATE documents SET metadata=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (json.dumps(metadata, ensure_ascii=False), int(memory_id)),
                )
            await db.commit()
        return int(cursor.rowcount or 0) > 0

    async def memory_transfer_records(
        self, memory_ids: Iterable[int] | None = None
    ) -> list[dict[str, Any]]:
        normalized_ids = (
            sorted({int(value) for value in memory_ids})
            if memory_ids is not None
            else None
        )
        if normalized_ids == []:
            return []
        async with self.connect() as db:
            source_table = await self._table_exists(db, "memory_sources")
            source_projection = "s.source_json" if source_table else "NULL"
            source_join = (
                "LEFT JOIN memory_sources s ON s.memory_id=d.id" if source_table else ""
            )
            where = ""
            params: tuple[Any, ...] = ()
            if normalized_ids is not None:
                placeholders = ",".join("?" for _ in normalized_ids)
                where = f"WHERE d.id IN ({placeholders})"
                params = tuple(normalized_ids)
            rows = await (
                await db.execute(
                    f"""SELECT d.id,d.text,d.metadata,d.created_at,d.updated_at,
                    {source_projection} AS source_json
                    FROM documents d {source_join} {where} ORDER BY d.id""",
                    params,
                )
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            metadata = normalize_document_metadata(row["metadata"])
            try:
                source = json.loads(row["source_json"] or "[]")
            except (TypeError, json.JSONDecodeError):
                source = []
            records.append(
                {
                    "original_id": int(row["id"]),
                    "content": str(row["text"] or ""),
                    "importance": float(metadata.get("importance", 0.5) or 0.5),
                    "session_id": metadata.get("session_id"),
                    "persona_id": metadata.get("persona_id"),
                    "metadata": metadata,
                    "source_messages": source if isinstance(source, list) else [],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return records

    async def document_ids(self, *, active_only: bool = True) -> list[int]:
        where = (
            "WHERE COALESCE(json_extract(metadata,'$.status'),'active')='active'"
            if active_only
            else ""
        )
        async with self.connect() as db:
            rows = await (
                await db.execute(f"SELECT id FROM documents {where} ORDER BY id")
            ).fetchall()
        return [int(row["id"]) for row in rows]

    async def graph_entry_ids(self) -> list[int]:
        async with self.connect() as db:
            rows = await (
                await db.execute("SELECT id FROM graph_entries ORDER BY id")
            ).fetchall()
        return [int(row["id"]) for row in rows]

    async def documents_for_ids(
        self, memory_ids: Iterable[int]
    ) -> list[dict[str, Any]]:
        ids = sorted({int(value) for value in memory_ids if int(value) > 0})
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        async with self.connect() as db:
            source_projection = (
                "CASE WHEN EXISTS(SELECT 1 FROM memory_sources s "
                "WHERE s.memory_id=documents.id) THEN 1 ELSE 0 END"
                if await self._table_exists(db, "memory_sources")
                else "0"
            )
            rows = await (
                await db.execute(
                    f"""SELECT id,doc_id,text,metadata,created_at,updated_at,
                    {source_projection} AS has_source
                    FROM documents WHERE id IN ({placeholders}) ORDER BY id""",
                    tuple(ids),
                )
            ).fetchall()
        return [self._document_row(row) for row in rows]

    async def iter_documents(
        self,
        batch_size: int = 500,
        *,
        after_id: int = 0,
        active_only: bool = True,
    ):
        last_id = max(0, int(after_id))
        while True:
            active_clause = (
                "AND COALESCE(json_extract(metadata,'$.status'),'active')='active'"
                if active_only
                else ""
            )
            async with self.connect() as db:
                rows = await (
                    await db.execute(
                        """SELECT id,doc_id,text,metadata,created_at,updated_at
                        FROM documents WHERE id>? """
                        + active_clause
                        + " ORDER BY id LIMIT ?",
                        (last_id, batch_size),
                    )
                ).fetchall()
            if not rows:
                break
            yield [self._document_row(row) for row in rows]
            last_id = int(rows[-1]["id"])

    @staticmethod
    def _aggregate_graph_entries(entries: list[dict[str, Any]]) -> str:
        unique = list(
            dict.fromkeys(
                str(item.get("content") or "").strip()
                for item in entries
                if str(item.get("content") or "").strip()
            )
        )
        return "\n".join(unique)[:4000]

    async def graph_memory_ids(self, *, active_only: bool = True) -> list[int]:
        active_join = (
            "AND COALESCE(json_extract(d.metadata,'$.status'),'active')='active'"
            if active_only
            else ""
        )
        async with self.connect() as db:
            rows = await (
                await db.execute(
                    """SELECT DISTINCT ge.source_memory_id AS id
                    FROM graph_entries ge
                    JOIN documents d ON d.id=ge.source_memory_id
                    WHERE TRIM(ge.content)<>'' """
                    + active_join
                    + " ORDER BY ge.source_memory_id"
                )
            ).fetchall()
        return [int(row["id"]) for row in rows]

    async def graph_entry_count(self, *, active_only: bool = True) -> int:
        active_join = (
            "AND COALESCE(json_extract(d.metadata,'$.status'),'active')='active'"
            if active_only
            else ""
        )
        async with self.connect() as db:
            row = await (
                await db.execute(
                    """SELECT COUNT(*) AS value FROM graph_entries ge
                    JOIN documents d ON d.id=ge.source_memory_id
                    WHERE TRIM(ge.content)<>'' """
                    + active_join
                )
            ).fetchone()
        return int(row["value"] if row else 0)

    async def graph_memories_for_ids(
        self, memory_ids: Iterable[int]
    ) -> list[dict[str, Any]]:
        ids = sorted({int(value) for value in memory_ids if int(value) > 0})
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        async with self.connect() as db:
            rows = await (
                await db.execute(
                    f"""SELECT ge.id,ge.source_memory_id,ge.content,ge.metadata,
                    d.text AS document_text,d.metadata AS document_metadata
                    FROM graph_entries ge
                    JOIN documents d ON d.id=ge.source_memory_id
                    WHERE ge.source_memory_id IN ({placeholders})
                    AND COALESCE(json_extract(d.metadata,'$.status'),'active')='active'
                    ORDER BY ge.source_memory_id,ge.id""",
                    tuple(ids),
                )
            ).fetchall()
        grouped: dict[int, list[dict[str, Any]]] = {}
        document_data: dict[int, tuple[str, dict[str, Any]]] = {}
        for row in rows:
            memory_id = int(row["source_memory_id"])
            grouped.setdefault(memory_id, []).append(
                {
                    "id": int(row["id"]),
                    "content": str(row["content"] or ""),
                    "metadata": normalize_document_metadata(row["metadata"]),
                }
            )
            document_data[memory_id] = (
                str(row["document_text"] or ""),
                normalize_document_metadata(row["document_metadata"]),
            )
        result: list[dict[str, Any]] = []
        for memory_id in ids:
            entries = grouped.get(memory_id, [])
            content = self._aggregate_graph_entries(entries)
            if not content:
                continue
            document_text, _document_metadata = document_data[memory_id]
            # LivingMemory 2.5.0 stores the first graph entry's metadata on
            # the aggregate memory-level vector. In particular this preserves
            # graph_confidence instead of silently falling back to 0.7.
            metadata = dict(entries[0].get("metadata") or {})
            result.append(
                {
                    "id": memory_id,
                    "source_memory_id": memory_id,
                    "content": content,
                    "document_text": document_text,
                    "metadata": {
                        **metadata,
                        "source_memory_id": memory_id,
                        "graph_vector_granularity": "memory",
                        "graph_entry_count": len(entries),
                    },
                    "graph_entry_count": len(entries),
                }
            )
        return result

    async def iter_graph_memories(
        self, batch_size: int = 500, *, after_id: int = 0
    ):
        last_id = max(0, int(after_id))
        while True:
            ids = [
                memory_id
                for memory_id in await self.graph_memory_ids(active_only=True)
                if memory_id > last_id
            ][: max(1, int(batch_size))]
            if not ids:
                break
            rows = await self.graph_memories_for_ids(ids)
            if rows:
                yield rows
            last_id = ids[-1]

    async def iter_graph_entries(self, batch_size: int = 500, *, after_id: int = 0):
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
            int(item) for item in dict.fromkeys(candidate_ids) if int(item) > 0
        ]
        if not normalized_ids:
            return {}
        token_values = [
            str(token).strip() for token in dict.fromkeys(tokens) if str(token).strip()
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
                    f'"{token.replace(chr(34), chr(34) * 2)}"' for token in token_values
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
                            1.0 if span == 0 else (high - float(row["score"])) / span
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
            fallback_count: dict[int, int] = {
                memory_id: 0 for memory_id in normalized_ids
            }
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
                key=lambda item: (
                    float(item.get("score") or 0.0),
                    int(item.get("entry_id") or 0),
                ),
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
        text = str(
            payload.get("content") or payload.get("canonical_summary") or ""
        ).strip()
        if not text:
            raise ValueError("content is required")
        now = time.time()
        metadata = normalize_document_metadata(payload.get("metadata"))
        canonical = str(payload.get("canonical_summary") or text).strip()
        source_messages = serialize_source_messages(payload.get("source_messages"))
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
                "participant_identities": list(
                    payload.get("participant_identities")
                    or metadata.get("participant_identities")
                    or []
                ),
                "key_facts": list(payload.get("key_facts") or []),
                "canonical_summary": canonical,
                "persona_summary": str(payload.get("persona_summary") or canonical),
                "summary_schema_version": "v2",
                "status": str(payload.get("status") or "active"),
                "memory_origin": str(
                    payload.get("memory_origin") or "personalityrag_api"
                ),
            }
        )
        if source_messages:
            metadata["has_source"] = True
            metadata["source_message_count"] = len(source_messages)
        source_time_strategy = str(
            payload.get("source_time_strategy") or "preserve"
        )
        metadata["source_time_strategy"] = source_time_strategy
        if source_time_strategy != "none" and isinstance(
            payload.get("source_time_tags"), dict
        ):
            metadata["source_time_tags"] = dict(payload["source_time_tags"])
        atom_types = sorted(
            {
                str(item.get("atom_type") or "unknown").casefold()
                for item in (payload.get("atoms") or [])
                if isinstance(item, dict)
            }
        )
        if atom_types:
            metadata["atom_types"] = atom_types
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
                (
                    str(uuid.uuid4()),
                    canonical,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            memory_id = int(cursor.lastrowid)
            if source_messages:
                await self._ensure_memory_sources(db)
                await db.execute(
                    """INSERT INTO memory_sources(
                        memory_id,source_json,created_at,updated_at
                    ) VALUES(?,?,?,?)""",
                    (
                        memory_id,
                        json.dumps(source_messages, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
            if metadata["status"] == "active":
                await db.execute(
                    "INSERT INTO livingmemory_memories_fts(doc_id,content) VALUES(?,?)",
                    (memory_id, " ".join(tokenize(canonical))),
                )
                graph = graph_builder(memory_id, canonical, metadata)
                await self._insert_graph(db, graph, tokenize)
                await self._insert_atoms(
                    db, memory_id, payload.get("atoms") or [], metadata
                )
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
            edge_key = str(edge["edge_key"])
            source_node_id = node_ids[edge["source_key"]]
            target_node_id = node_ids[edge["target_key"]]
            relation_type = str(edge["relation_type"])
            confidence = float(edge.get("confidence", 0.8))
            weight = float(edge.get("weight", 1.0))
            metadata_json = json.dumps(
                edge.get("metadata") or {},
                ensure_ascii=False,
            )
            row = await (
                await db.execute(
                    "SELECT id FROM graph_edges WHERE edge_key=?",
                    (edge_key,),
                )
            ).fetchone()
            if row:
                edge_id = int(row["id"])
                await db.execute(
                    """UPDATE graph_edges
                    SET weight=?,confidence=?,status='active',metadata=?,updated_at=?
                    WHERE id=?""",
                    (
                        weight,
                        confidence,
                        metadata_json,
                        now_iso,
                        edge_id,
                    ),
                )
            else:
                semantic_row = await (
                    await db.execute(
                        """SELECT id,confidence,weight FROM graph_edges
                        WHERE source_node_id=? AND target_node_id=?
                        AND relation_type=?
                        ORDER BY id ASC LIMIT 1""",
                        (
                            source_node_id,
                            target_node_id,
                            relation_type,
                        ),
                    )
                ).fetchone()
                if semantic_row:
                    edge_id = int(semantic_row["id"])
                    merged_confidence = (
                        float(semantic_row["confidence"] or 0.8) * 0.7
                        + confidence * 0.3
                    )
                    merged_weight = float(semantic_row["weight"] or 1.0) + 0.15
                    await db.execute(
                        """UPDATE graph_edges
                        SET confidence=?,weight=?,updated_at=? WHERE id=?""",
                        (
                            merged_confidence,
                            merged_weight,
                            now_iso,
                            edge_id,
                        ),
                    )
                else:
                    cursor = await db.execute(
                        """INSERT INTO graph_edges(
                            edge_key,source_node_id,target_node_id,relation_type,
                            source_memory_id,weight,confidence,status,metadata,
                            created_at,updated_at
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            edge_key,
                            source_node_id,
                            target_node_id,
                            relation_type,
                            edge["source_memory_id"],
                            weight,
                            confidence,
                            "active",
                            metadata_json,
                            now_iso,
                            now_iso,
                        ),
                    )
                    edge_id = int(cursor.lastrowid)
            edge_ids[edge_key] = edge_id
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

    async def update_memory(
        self, memory_id: int, updates: dict[str, Any], tokenize, graph_builder
    ) -> bool:
        current = await self.get_document(memory_id)
        if not current:
            return False
        text = str(updates.get("content", current["text"])).strip()
        if not text:
            raise ValueError("content cannot be empty")
        metadata = dict(current["metadata"])
        metadata.update(normalize_document_metadata(updates.get("metadata")))
        for key in ("importance", "status", "session_id", "persona_id"):
            if key in updates:
                metadata[key] = updates[key]
        metadata.pop("memory_type", None)
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
            await self._insert_graph(
                db, graph_builder(memory_id, text, metadata), tokenize
            )
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

            metadata = normalize_document_metadata(row["metadata"])
            metadata.update(normalize_document_metadata(updates.get("metadata")))
            for key in (
                "importance",
                "status",
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
                            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
                params.append(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
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

    async def update_memories_metadata(
        self,
        memory_ids: Iterable[int],
        updates: dict[str, Any],
    ) -> list[int]:
        requested = list(dict.fromkeys(int(value) for value in memory_ids))
        if not requested:
            return []
        now = time.time()
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
        async with self._write_lock, self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            rows: list[aiosqlite.Row] = []
            for chunk in self._id_chunks(requested):
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    await (
                        await db.execute(
                            f"SELECT id,metadata FROM documents WHERE id IN ({placeholders})",
                            chunk,
                        )
                    ).fetchall()
                )
            row_map = {int(row["id"]): row for row in rows}
            existing_ids = [memory_id for memory_id in requested if memory_id in row_map]
            document_updates: list[tuple[str, int]] = []
            metadata_by_id: dict[int, dict[str, Any]] = {}
            nested_updates = normalize_document_metadata(updates.get("metadata"))
            for memory_id in existing_ids:
                metadata = normalize_document_metadata(row_map[memory_id]["metadata"])
                metadata.update(nested_updates)
                for key in (
                    "importance",
                    "status",
                    "session_id",
                    "persona_id",
                ):
                    if key in updates:
                        metadata[key] = updates[key]
                metadata["updated_at"] = now
                metadata_by_id[memory_id] = metadata
                document_updates.append(
                    (json.dumps(metadata, ensure_ascii=False), memory_id)
                )
            if document_updates:
                await db.executemany(
                    "UPDATE documents SET metadata=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    document_updates,
                )

            graph_keys = {
                key
                for key in (
                    "importance",
                    "session_id",
                    "persona_id",
                    "last_access_time",
                )
                if key in updates or key in nested_updates
            }
            if graph_keys:
                graph_updates: list[tuple[str, str, int]] = []
                for chunk in self._id_chunks(existing_ids):
                    placeholders = ",".join("?" for _ in chunk)
                    graph_rows = await (
                        await db.execute(
                            f"SELECT id,source_memory_id,metadata FROM graph_entries "
                            f"WHERE source_memory_id IN ({placeholders})",
                            chunk,
                        )
                    ).fetchall()
                    for graph_row in graph_rows:
                        graph_metadata = normalize_metadata(graph_row["metadata"])
                        source_metadata = metadata_by_id[int(graph_row["source_memory_id"])]
                        graph_metadata.update(
                            {key: source_metadata.get(key) for key in graph_keys}
                        )
                        graph_updates.append(
                            (
                                json.dumps(graph_metadata, ensure_ascii=False),
                                now_iso,
                                int(graph_row["id"]),
                            )
                        )
                if graph_updates:
                    await db.executemany(
                        "UPDATE graph_entries SET metadata=?,updated_at=? WHERE id=?",
                        graph_updates,
                    )

            scope_keys = [
                key for key in ("session_id", "persona_id") if key in updates
            ]
            if scope_keys:
                for chunk in self._id_chunks(existing_ids):
                    placeholders = ",".join("?" for _ in chunk)
                    assignments = [f"{key}=?" for key in scope_keys]
                    scope_values = [updates.get(key) for key in scope_keys]
                    await db.execute(
                        f"UPDATE graph_entries SET {','.join(assignments)},updated_at=? "
                        f"WHERE source_memory_id IN ({placeholders})",
                        (*scope_values, now_iso, *chunk),
                    )
                    await db.execute(
                        f"UPDATE memory_atoms SET {','.join(assignments)} "
                        f"WHERE parent_memory_id IN ({placeholders})",
                        (*scope_values, *chunk),
                    )
            await db.commit()
        return existing_ids

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

    async def archive_memories(self, memory_ids: Iterable[int]) -> list[int]:
        ids = sorted({int(value) for value in memory_ids if int(value) > 0})
        if not ids:
            return []
        archived: list[int] = []
        archived_at = time.time()
        async with self._write_lock, self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            for memory_id in ids:
                row = await (
                    await db.execute(
                        "SELECT metadata FROM documents WHERE id=?", (memory_id,)
                    )
                ).fetchone()
                if not row:
                    continue
                metadata = normalize_document_metadata(row["metadata"])
                if str(metadata.get("status") or "active") == "archived":
                    continue
                metadata["status"] = "archived"
                metadata["archived_at"] = archived_at
                await db.execute(
                    "UPDATE documents SET metadata=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (json.dumps(metadata, ensure_ascii=False), memory_id),
                )
                await db.execute(
                    "DELETE FROM livingmemory_memories_fts WHERE doc_id=?",
                    (memory_id,),
                )
                atom_ids = [
                    int(item["id"])
                    for item in await (
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
                await self._delete_graph(db, memory_id)
                archived.append(memory_id)
            await db.commit()
        return archived

    async def restore_memory(
        self,
        memory_id: int,
        tokenize,
        graph_builder,
        atoms: list[dict[str, Any]] | None = None,
    ) -> bool:
        async with self._write_lock, self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            row = await (
                await db.execute(
                    "SELECT text,metadata FROM documents WHERE id=?", (int(memory_id),)
                )
            ).fetchone()
            if not row:
                await db.rollback()
                return False
            text = str(row["text"] or "")
            metadata = normalize_document_metadata(row["metadata"])
            if str(metadata.get("status") or "active") != "archived":
                await db.rollback()
                return True
            metadata["status"] = "active"
            metadata["restored_at"] = time.time()
            metadata.pop("archived_at", None)
            await db.execute(
                "UPDATE documents SET metadata=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (json.dumps(metadata, ensure_ascii=False), int(memory_id)),
            )
            await db.execute(
                "INSERT INTO livingmemory_memories_fts(doc_id,content) VALUES(?,?)",
                (int(memory_id), " ".join(tokenize(text))),
            )
            await self._insert_graph(
                db, graph_builder(int(memory_id), text, metadata), tokenize
            )
            await self._insert_atoms(
                db, int(memory_id), atoms or [], metadata
            )
            await db.commit()
        return True

    async def delete_memories(self, memory_ids: Iterable[int]) -> int:
        ids = sorted({int(value) for value in memory_ids})
        if not ids:
            return 0
        async with self._write_lock, self.connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            existing_ids: list[int] = []
            for chunk in self._id_chunks(ids):
                placeholders = ",".join("?" for _ in chunk)
                existing_ids.extend(
                    int(row["id"])
                    for row in await (
                        await db.execute(
                            f"SELECT id FROM documents WHERE id IN ({placeholders})",
                            chunk,
                        )
                    ).fetchall()
                )
            orphan_edge_ids = await self._prepare_graph_edge_deletion(
                db,
                existing_ids,
            )
            for chunk in self._id_chunks(existing_ids):
                placeholders = ",".join("?" for _ in chunk)
                entry_ids = [
                    int(row["id"])
                    for row in await (
                        await db.execute(
                            f"SELECT id FROM graph_entries WHERE source_memory_id IN ({placeholders})",
                            chunk,
                        )
                    ).fetchall()
                ]
                atom_ids = [
                    int(row["id"])
                    for row in await (
                        await db.execute(
                            f"SELECT id FROM memory_atoms WHERE parent_memory_id IN ({placeholders})",
                            chunk,
                        )
                    ).fetchall()
                ]
                for entry_chunk in self._id_chunks(entry_ids):
                    entry_placeholders = ",".join("?" for _ in entry_chunk)
                    await db.execute(
                        f"DELETE FROM livingmemory_graph_entries_fts "
                        f"WHERE entry_id IN ({entry_placeholders})",
                        entry_chunk,
                    )
                for atom_chunk in self._id_chunks(atom_ids):
                    atom_placeholders = ",".join("?" for _ in atom_chunk)
                    await db.execute(
                        f"DELETE FROM memory_atoms_fts WHERE atom_id IN ({atom_placeholders})",
                        atom_chunk,
                    )
                for table, column in (
                    ("graph_entries", "source_memory_id"),
                    ("memory_atoms", "parent_memory_id"),
                    ("livingmemory_memories_fts", "doc_id"),
                    ("memory_write_ops", "memory_id"),
                    ("documents", "id"),
                ):
                    await db.execute(
                        f"DELETE FROM {table} WHERE {column} IN ({placeholders})",
                        chunk,
                    )
                if await self._table_exists(db, "memory_sources"):
                    await db.execute(
                        f"DELETE FROM memory_sources WHERE memory_id IN ({placeholders})",
                        chunk,
                    )
            for edge_chunk in self._id_chunks(orphan_edge_ids):
                placeholders = ",".join("?" for _ in edge_chunk)
                await db.execute(
                    f"DELETE FROM graph_edges WHERE id IN ({placeholders})",
                    edge_chunk,
                )
            for chunk in self._id_chunks(existing_ids):
                placeholders = ",".join("?" for _ in chunk)
                await db.execute(
                    f"DELETE FROM graph_edges "
                    f"WHERE source_memory_id IN ({placeholders})",
                    chunk,
                )
            if existing_ids:
                await db.execute(
                    """DELETE FROM graph_nodes WHERE id NOT IN(
                        SELECT source_node_id FROM graph_edges
                        UNION SELECT target_node_id FROM graph_edges
                        UNION SELECT node_id FROM graph_entry_nodes
                    )"""
                )
            await db.commit()
        return len(existing_ids)

    @staticmethod
    def _id_chunks(values: Iterable[int], size: int = 400) -> list[tuple[int, ...]]:
        normalized = tuple(int(value) for value in values)
        return [
            normalized[index : index + max(1, int(size))]
            for index in range(0, len(normalized), max(1, int(size)))
        ]

    async def _prepare_graph_edge_deletion(
        self,
        db,
        memory_ids: Iterable[int],
    ) -> list[int]:
        deleting = {int(value) for value in memory_ids}
        if not deleting:
            return []

        affected_edge_ids: set[int] = set()
        for chunk in self._id_chunks(sorted(deleting)):
            placeholders = ",".join("?" for _ in chunk)
            affected_edge_ids.update(
                int(row["edge_id"])
                for row in await (
                    await db.execute(
                        f"""SELECT DISTINCT edge_id FROM graph_entries
                        WHERE source_memory_id IN ({placeholders})
                        AND edge_id IS NOT NULL""",
                        chunk,
                    )
                ).fetchall()
            )

        orphan_edge_ids: list[int] = []
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for edge_id in sorted(affected_edge_ids):
            edge = await (
                await db.execute(
                    """SELECT e.relation_type,s.node_key AS source_key,
                    t.node_key AS target_key
                    FROM graph_edges e
                    JOIN graph_nodes s ON s.id=e.source_node_id
                    JOIN graph_nodes t ON t.id=e.target_node_id
                    WHERE e.id=?""",
                    (edge_id,),
                )
            ).fetchone()
            if edge is None:
                continue
            remaining = [
                row
                for row in await (
                    await db.execute(
                        """SELECT source_memory_id,metadata FROM graph_entries
                        WHERE edge_id=? ORDER BY id""",
                        (edge_id,),
                    )
                ).fetchall()
                if int(row["source_memory_id"]) not in deleting
            ]
            if not remaining:
                orphan_edge_ids.append(edge_id)
                continue

            metadata_rows = [
                normalize_metadata(row["metadata"])
                for row in remaining
            ]
            confidence = float(
                metadata_rows[0].get("graph_confidence", 0.8) or 0.8
            )
            for metadata in metadata_rows[1:]:
                confidence = (
                    confidence * 0.7
                    + float(metadata.get("graph_confidence", 0.8) or 0.8) * 0.3
                )
            owner = int(remaining[0]["source_memory_id"])
            edge_key = (
                f"{edge['source_key']}|{edge['relation_type']}|"
                f"{edge['target_key']}|{owner}"
            )
            await db.execute(
                """UPDATE graph_edges
                SET edge_key=?,source_memory_id=?,weight=?,confidence=?,
                status='active',metadata=?,updated_at=?
                WHERE id=?""",
                (
                    edge_key,
                    owner,
                    1.0 + 0.15 * (len(remaining) - 1),
                    confidence,
                    json.dumps(
                        {
                            "summary": str(
                                metadata_rows[0].get("canonical_summary") or ""
                            )
                        },
                        ensure_ascii=False,
                    ),
                    now_iso,
                    edge_id,
                ),
            )
        return orphan_edge_ids

    async def mark_memory_indexed(
        self, memory_id: int, *, generation: str = ""
    ) -> None:
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
        orphan_edge_ids = await self._prepare_graph_edge_deletion(db, [memory_id])
        entry_ids = [
            int(row["id"])
            for row in await (
                await db.execute(
                    "SELECT id FROM graph_entries WHERE source_memory_id=?",
                    (memory_id,),
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
        for edge_id in orphan_edge_ids:
            await db.execute("DELETE FROM graph_edges WHERE id=?", (edge_id,))
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
        return await self._cached_statistics("full", self._statistics_uncached)

    async def _statistics_uncached(self) -> dict[str, Any]:
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
            row = await (
                await db.execute(
                    """SELECT COUNT(*) AS value FROM documents
                    WHERE COALESCE(json_extract(metadata,'$.status'),'active')='active'"""
                )
            ).fetchone()
            counts["active_memories"] = int(row["value"])
            rows = await (await db.execute("SELECT metadata FROM documents")).fetchall()
            status: dict[str, int] = {}
            sessions: dict[str, int] = {}
            importance = {f"{i}-{i + 1}": 0 for i in range(10)}
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
                importance[f"{bucket}-{bucket + 1}"] += 1
            atom_rows = await (
                await db.execute(
                    "SELECT atom_type,COUNT(*) AS value FROM memory_atoms GROUP BY atom_type"
                )
            ).fetchall()
        conversation = {"sessions": 0, "messages": 0, "pending_messages": 0}
        if self.conversations_path.exists():
            async with self.conversation_connect() as db:
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
        return await self._cached_statistics(
            "summary", self._summary_statistics_uncached
        )

    async def _summary_statistics_uncached(self) -> dict[str, Any]:
        """Return only the scalar counters needed by library cards."""
        count_keys = (
            "total_memories",
            "active_memories",
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
                        (SELECT COUNT(*) FROM documents
                         WHERE COALESCE(
                             json_extract(metadata,'$.status'), 'active'
                         ) = 'active') AS active_memories,
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
            async with self.conversation_connect() as db:
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
                    row[0]: con.execute(f'SELECT COUNT(*) FROM "{row[0]}"').fetchone()[
                        0
                    ]
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
            "conversations": await asyncio.to_thread(inspect, self.conversations_path),
        }
