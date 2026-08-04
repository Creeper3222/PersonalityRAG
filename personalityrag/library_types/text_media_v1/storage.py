from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ...sqlite_pool import SQLiteConnectionPool
from ...resource_limits import configured_sqlite_pool_size
from .retrieval import (
    rerank_calibration_settings_fingerprint,
    retrieval_config_json,
)
from .text import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_TARGET,
    fts_query_text,
    lexical_text,
    media_description_list,
    media_tokens,
    normalize_media_description,
)


# Internal SQLite migration revision. The public database type remains
# text_media_v1 and the TMKB package format remains v1.
SCHEMA_VERSION = 9
DATABASE_FILENAME = "textmediaknowledge.db"
DEFAULT_UNIFORM_MEDIA_STRENGTH = 0.5
RELATION_SCOPES = frozenset({"document", "entry", "chunk"})
OUTPUT_POLICIES = frozenset({"auto", "with_result", "metadata_only", "disabled"})

SUMMARY_SNAPSHOT_SQL = """SELECT lm.*,
    (SELECT COUNT(*) FROM documents
      WHERE status='ready') AS summary_documents,
    (SELECT COUNT(*) FROM entries
      WHERE status='active') AS summary_entries,
    (SELECT COUNT(*) FROM chunks
      WHERE status='active') AS summary_chunks,
    (SELECT COUNT(*) FROM assets
      WHERE kind='image' AND state='active') AS summary_images
    FROM library_meta lm WHERE singleton=1"""

SUMMARY_RECALIBRATION_SQL = """SELECT EXISTS(
  SELECT 1 FROM document_assets da
  WHERE da.semantic_mode='calibrated' AND (
    COALESCE(da.calibration_rerank_provider_fingerprint,'')<>?
    OR NOT EXISTS(
      SELECT 1 FROM chunk_media_strengths cms
      WHERE cms.document_id=da.document_id
        AND cms.asset_id=da.asset_id
        AND cms.rerank_semantic_strength IS NOT NULL
        AND json_valid(cms.calibration_details_json)
        AND COALESCE(json_extract(
          cms.calibration_details_json,
          '$.rerank_settings_fingerprint'
        ),'')=?
    )
    OR EXISTS(
      SELECT 1 FROM chunk_media_strengths cms
      WHERE cms.document_id=da.document_id
        AND cms.asset_id=da.asset_id
        AND cms.rerank_semantic_strength IS NOT NULL
        AND json_valid(cms.calibration_details_json)
        AND COALESCE(json_extract(
          cms.calibration_details_json,
          '$.rerank_settings_fingerprint'
        ),'')<>?
    )
  )
) AS value"""


def normalize_bm25_rows(
    rows: Iterable[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Convert SQLite's lower-is-better BM25 values to stable ``[0, 1]``.

    The caller supplies a fixed internal retrieval window, so this
    normalization never depends on the public Top-K requested by a client.
    """

    values = [(int(row_id), float(score)) for row_id, score in rows]
    if not values:
        return []
    high = max(score for _row_id, score in values)
    low = min(score for _row_id, score in values)
    span = high - low
    return [
        (row_id, 1.0 if span == 0 else (high - score) / span)
        for row_id, score in values
    ]


def vector_bytes(vector: Iterable[float]) -> bytes:
    values = np.asarray(list(vector), dtype="<f4")
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Embedding 向量无效")
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding 向量范数无效")
    return (values / norm).astype("<f4", copy=False).tobytes()


def media_description_set_sha256(values: Iterable[str]) -> str:
    normalized = sorted(
        normalize_media_description(value)
        for value in values
        if normalize_media_description(value)
    )
    return hashlib.sha256(
        json.dumps(
            normalized,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class TextMediaStorage:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / DATABASE_FILENAME
        self.pool = SQLiteConnectionPool(
            self.path,
            size=configured_sqlite_pool_size(),
        )

    async def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        db = await self.pool.acquire()
        try:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS library_meta (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    database_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    provider_id TEXT NOT NULL DEFAULT '',
                    provider_revision INTEGER NOT NULL DEFAULT 0,
                    provider_fingerprint TEXT NOT NULL DEFAULT '',
                    rerank_provider_id TEXT NOT NULL DEFAULT '',
                    rerank_provider_revision INTEGER NOT NULL DEFAULT 0,
                    rerank_provider_fingerprint TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'ready',
                    chunk_target INTEGER NOT NULL DEFAULT 1200,
                    chunk_overlap INTEGER NOT NULL DEFAULT 150,
                    uniform_media_strength REAL NOT NULL DEFAULT 0.5,
                    retrieval_config_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL CHECK(kind IN ('document','image')),
                    sha256 TEXT NOT NULL,
                    storage_key TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    original_name TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL DEFAULT 'active',
                    created_at REAL NOT NULL,
                    UNIQUE(kind,sha256)
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE RESTRICT,
                    parser_id TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ready',
                    error TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS entries (
                    id TEXT PRIMARY KEY,
                    document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    ordinal INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id TEXT NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    UNIQUE(entry_id,ordinal)
                );
                CREATE TABLE IF NOT EXISTS embedding_generations (
                    id TEXT PRIMARY KEY,
                    provider_id TEXT NOT NULL,
                    provider_revision INTEGER NOT NULL,
                    provider_fingerprint TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    metric TEXT NOT NULL DEFAULT 'cosine_ip',
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    activated_at REAL
                );
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    generation_id TEXT NOT NULL REFERENCES embedding_generations(id) ON DELETE CASCADE,
                    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    vector BLOB NOT NULL,
                    vector_sha256 TEXT NOT NULL,
                    PRIMARY KEY(generation_id,chunk_id)
                );
                CREATE TABLE IF NOT EXISTS active_generation (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    generation_id TEXT NOT NULL REFERENCES embedding_generations(id)
                );
                CREATE TABLE IF NOT EXISTS document_assets (
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                    role TEXT NOT NULL DEFAULT 'illustration',
                    relation_weight REAL NOT NULL DEFAULT 1,
                    caption TEXT NOT NULL DEFAULT '',
                    alt_text TEXT NOT NULL DEFAULT '',
                    output_policy TEXT NOT NULL DEFAULT 'auto',
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(document_id,asset_id)
                );
                CREATE TABLE IF NOT EXISTS entry_assets (
                    entry_id TEXT NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
                    asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                    role TEXT NOT NULL DEFAULT 'illustration',
                    relation_weight REAL NOT NULL DEFAULT 1,
                    caption TEXT NOT NULL DEFAULT '',
                    alt_text TEXT NOT NULL DEFAULT '',
                    output_policy TEXT NOT NULL DEFAULT 'auto',
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(entry_id,asset_id)
                );
                CREATE TABLE IF NOT EXISTS chunk_assets (
                    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                    role TEXT NOT NULL DEFAULT 'illustration',
                    relation_weight REAL NOT NULL DEFAULT 1,
                    caption TEXT NOT NULL DEFAULT '',
                    alt_text TEXT NOT NULL DEFAULT '',
                    output_policy TEXT NOT NULL DEFAULT 'auto',
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(chunk_id,asset_id)
                );
                CREATE TABLE IF NOT EXISTS chunk_media_strengths (
                    document_id TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                    semantic_strength REAL NOT NULL DEFAULT 1,
                    rerank_semantic_strength REAL,
                    calibration_similarity REAL,
                    calibration_rank INTEGER,
                    calibration_method TEXT NOT NULL DEFAULT 'uniform_v1',
                    provider_fingerprint TEXT NOT NULL DEFAULT '',
                    rerank_provider_fingerprint TEXT NOT NULL DEFAULT '',
                    calibration_details_json TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(chunk_id,asset_id),
                    FOREIGN KEY(document_id,asset_id)
                        REFERENCES document_assets(document_id,asset_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS ingest_batches (
                    id TEXT PRIMARY KEY,
                    parameters_json TEXT NOT NULL,
                    document_count INTEGER NOT NULL,
                    image_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'completed',
                    created_at REAL NOT NULL,
                    completed_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ingest_batch_documents (
                    batch_id TEXT NOT NULL REFERENCES ingest_batches(id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    PRIMARY KEY(batch_id,document_id),
                    UNIQUE(batch_id,ordinal)
                );
                CREATE TABLE IF NOT EXISTS ingest_batch_assets (
                    batch_id TEXT NOT NULL REFERENCES ingest_batches(id) ON DELETE CASCADE,
                    asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE RESTRICT,
                    ordinal INTEGER NOT NULL,
                    document_indexes_json TEXT NOT NULL,
                    PRIMARY KEY(batch_id,asset_id),
                    UNIQUE(batch_id,ordinal)
                );
                CREATE TABLE IF NOT EXISTS asset_media_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_id TEXT NOT NULL
                        REFERENCES assets(id) ON DELETE CASCADE,
                    media_description TEXT NOT NULL DEFAULT '',
                    normalized_description TEXT NOT NULL DEFAULT '',
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    search_text TEXT NOT NULL DEFAULT '',
                    description_source TEXT NOT NULL DEFAULT 'filename',
                    media_description_vector BLOB,
                    vector_sha256 TEXT NOT NULL DEFAULT '',
                    provider_id TEXT NOT NULL DEFAULT '',
                    provider_revision INTEGER NOT NULL DEFAULT 0,
                    provider_fingerprint TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(asset_id,normalized_description),
                    UNIQUE(asset_id,sort_order)
                );
                CREATE TABLE IF NOT EXISTS asset_media_tokens (
                    asset_id TEXT NOT NULL
                        REFERENCES assets(id) ON DELETE CASCADE,
                    token TEXT NOT NULL,
                    PRIMARY KEY(asset_id,token)
                );
                CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status,created_at);
                CREATE INDEX IF NOT EXISTS idx_entries_document ON entries(document_id,ordinal);
                CREATE INDEX IF NOT EXISTS idx_chunks_entry ON chunks(entry_id,ordinal);
                CREATE INDEX IF NOT EXISTS idx_assets_state ON assets(state,kind);
                CREATE INDEX IF NOT EXISTS idx_chunk_media_strengths_document
                    ON chunk_media_strengths(document_id,asset_id,calibration_rank);
                CREATE INDEX IF NOT EXISTS idx_asset_media_metadata_provider
                    ON asset_media_metadata(provider_fingerprint,asset_id);
                CREATE INDEX IF NOT EXISTS idx_asset_media_tokens_token
                    ON asset_media_tokens(token,asset_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    search_text,
                    content='chunks',
                    content_rowid='id',
                    tokenize='unicode61'
                );
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid,search_text) VALUES(new.id,new.search_text);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts,rowid,search_text)
                    VALUES('delete',old.id,old.search_text);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF search_text ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts,rowid,search_text)
                    VALUES('delete',old.id,old.search_text);
                    INSERT INTO chunks_fts(rowid,search_text) VALUES(new.id,new.search_text);
                END;
                CREATE VIRTUAL TABLE IF NOT EXISTS asset_media_metadata_fts USING fts5(
                    search_text,
                    content='asset_media_metadata',
                    content_rowid='id',
                    tokenize='unicode61'
                );
                CREATE TRIGGER IF NOT EXISTS asset_media_metadata_ai
                AFTER INSERT ON asset_media_metadata BEGIN
                    INSERT INTO asset_media_metadata_fts(rowid,search_text)
                    VALUES(new.id,new.search_text);
                END;
                CREATE TRIGGER IF NOT EXISTS asset_media_metadata_ad
                AFTER DELETE ON asset_media_metadata BEGIN
                    INSERT INTO asset_media_metadata_fts(
                        asset_media_metadata_fts,rowid,search_text
                    ) VALUES('delete',old.id,old.search_text);
                END;
                CREATE TRIGGER IF NOT EXISTS asset_media_metadata_au
                AFTER UPDATE OF search_text ON asset_media_metadata BEGIN
                    INSERT INTO asset_media_metadata_fts(
                        asset_media_metadata_fts,rowid,search_text
                    ) VALUES('delete',old.id,old.search_text);
                    INSERT INTO asset_media_metadata_fts(rowid,search_text)
                    VALUES(new.id,new.search_text);
                END;
                """
            )
            document_asset_columns = {
                str(row[1])
                for row in await (
                    await db.execute("PRAGMA table_info(document_assets)")
                ).fetchall()
            }
            for column, definition in (
                ("semantic_mode", "TEXT NOT NULL DEFAULT 'uniform'"),
                ("media_description", "TEXT NOT NULL DEFAULT ''"),
                ("media_description_vector", "BLOB"),
                ("calibration_method", "TEXT NOT NULL DEFAULT 'uniform_v1'"),
                ("calibration_provider_fingerprint", "TEXT NOT NULL DEFAULT ''"),
                (
                    "calibration_rerank_provider_fingerprint",
                    "TEXT NOT NULL DEFAULT ''",
                ),
                (
                    "calibration_description_set_sha256",
                    "TEXT NOT NULL DEFAULT ''",
                ),
            ):
                if column not in document_asset_columns:
                    await db.execute(
                        f"ALTER TABLE document_assets ADD COLUMN {column} {definition}"
                    )
            library_meta_columns = {
                str(row[1])
                for row in await (
                    await db.execute("PRAGMA table_info(library_meta)")
                ).fetchall()
            }
            if "uniform_media_strength" not in library_meta_columns:
                await db.execute(
                    "ALTER TABLE library_meta ADD COLUMN "
                    "uniform_media_strength REAL NOT NULL DEFAULT 0.5"
                )
            if "retrieval_config_json" not in library_meta_columns:
                await db.execute(
                    "ALTER TABLE library_meta ADD COLUMN "
                    "retrieval_config_json TEXT NOT NULL DEFAULT '{}'"
                )
            for column, definition in (
                ("rerank_provider_id", "TEXT NOT NULL DEFAULT ''"),
                ("rerank_provider_revision", "INTEGER NOT NULL DEFAULT 0"),
                ("rerank_provider_fingerprint", "TEXT NOT NULL DEFAULT ''"),
            ):
                if column not in library_meta_columns:
                    await db.execute(
                        f"ALTER TABLE library_meta ADD COLUMN {column} {definition}"
                    )
            strength_columns = {
                str(row[1])
                for row in await (
                    await db.execute("PRAGMA table_info(chunk_media_strengths)")
                ).fetchall()
            }
            for column, definition in (
                ("rerank_semantic_strength", "REAL"),
                ("rerank_provider_fingerprint", "TEXT NOT NULL DEFAULT ''"),
                ("calibration_details_json", "TEXT NOT NULL DEFAULT '{}'"),
            ):
                if column not in strength_columns:
                    await db.execute(
                        f"ALTER TABLE chunk_media_strengths ADD COLUMN {column} {definition}"
                    )
            await self._migrate_asset_media_metadata_v9(db)
            await self._backfill_asset_media_metadata(db)
            await self._backfill_asset_media_tokens(db)
            await db.execute(
                "INSERT OR REPLACE INTO schema_info(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )
            await db.commit()
        finally:
            await db.close()

    async def _migrate_asset_media_metadata_v9(self, db) -> None:
        """Expand the v8 one-row asset profile into ordered description rows."""

        columns = {
            str(row[1])
            for row in await (
                await db.execute("PRAGMA table_info(asset_media_metadata)")
            ).fetchall()
        }
        if {"normalized_description", "sort_order"}.issubset(columns):
            return
        await db.executescript(
            """
            DROP TRIGGER IF EXISTS asset_media_metadata_ai;
            DROP TRIGGER IF EXISTS asset_media_metadata_ad;
            DROP TRIGGER IF EXISTS asset_media_metadata_au;
            DROP TABLE IF EXISTS asset_media_metadata_fts;
            ALTER TABLE asset_media_metadata RENAME TO asset_media_metadata_v8;
            CREATE TABLE asset_media_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                media_description TEXT NOT NULL DEFAULT '',
                normalized_description TEXT NOT NULL DEFAULT '',
                sort_order INTEGER NOT NULL DEFAULT 0,
                search_text TEXT NOT NULL DEFAULT '',
                description_source TEXT NOT NULL DEFAULT 'filename',
                media_description_vector BLOB,
                vector_sha256 TEXT NOT NULL DEFAULT '',
                provider_id TEXT NOT NULL DEFAULT '',
                provider_revision INTEGER NOT NULL DEFAULT 0,
                provider_fingerprint TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(asset_id,normalized_description),
                UNIQUE(asset_id,sort_order)
            );
            """
        )
        old_rows = await (
            await db.execute(
                """SELECT * FROM asset_media_metadata_v8
                ORDER BY asset_id,id"""
            )
        ).fetchall()
        seen_by_asset: dict[str, set[str]] = {}
        next_order: dict[str, int] = {}
        for row in old_rows:
            asset_id = str(row["asset_id"])
            description = str(row["media_description"] or "").strip()
            if not description:
                continue
            normalized = normalize_media_description(description)
            seen_by_asset.setdefault(asset_id, set()).add(normalized)
            next_order[asset_id] = 1
            await db.execute(
                """INSERT INTO asset_media_metadata
                (id,asset_id,media_description,normalized_description,sort_order,
                 search_text,description_source,media_description_vector,
                 vector_sha256,provider_id,provider_revision,provider_fingerprint,
                 created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    int(row["id"]),
                    asset_id,
                    description,
                    normalized,
                    0,
                    lexical_text(description),
                    str(row["description_source"] or "legacy"),
                    row["media_description_vector"],
                    str(row["vector_sha256"] or ""),
                    str(row["provider_id"] or ""),
                    int(row["provider_revision"] or 0),
                    str(row["provider_fingerprint"] or ""),
                    float(row["created_at"]),
                    float(row["updated_at"]),
                ),
            )

        relation_rows = await (
            await db.execute(
                """SELECT da.asset_id,da.media_description,
                da.media_description_vector,da.calibration_provider_fingerprint,
                lm.provider_id,lm.provider_revision,lm.provider_fingerprint
                FROM document_assets da CROSS JOIN library_meta lm
                JOIN assets a ON a.id=da.asset_id
                WHERE a.kind='image' AND a.state='active'
                  AND da.output_policy<>'disabled'
                  AND trim(da.media_description)<>''
                ORDER BY da.asset_id,da.sort_order,da.document_id"""
            )
        ).fetchall()
        now = time.time()
        for row in relation_rows:
            asset_id = str(row["asset_id"])
            description = str(row["media_description"] or "").strip()
            normalized = normalize_media_description(description)
            seen = seen_by_asset.setdefault(asset_id, set())
            if normalized in seen:
                continue
            seen.add(normalized)
            order = next_order.get(asset_id, 0)
            next_order[asset_id] = order + 1
            valid_vector = (
                row["media_description_vector"] is not None
                and str(row["calibration_provider_fingerprint"] or "")
                == str(row["provider_fingerprint"] or "")
            )
            blob = (
                bytes(row["media_description_vector"]) if valid_vector else None
            )
            await db.execute(
                """INSERT INTO asset_media_metadata
                (asset_id,media_description,normalized_description,sort_order,
                 search_text,description_source,media_description_vector,
                 vector_sha256,provider_id,provider_revision,provider_fingerprint,
                 created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    asset_id,
                    description,
                    normalized,
                    order,
                    lexical_text(description),
                    "relation",
                    blob,
                    hashlib.sha256(blob).hexdigest() if blob else "",
                    str(row["provider_id"] or "") if blob else "",
                    int(row["provider_revision"] or 0) if blob else 0,
                    str(row["provider_fingerprint"] or "") if blob else "",
                    now,
                    now,
                ),
            )
        # Relationship rows are compatibility projections only in schema 9.
        # Preserve an old calibration as current only when its former
        # description is equivalent to the asset's sole canonical
        # description.  Any expanded set keeps the old conservative strengths
        # but deliberately receives an empty set hash so query diagnostics
        # report that a full recalibration is required.
        migrated_assets = await (
            await db.execute(
                """SELECT DISTINCT asset_id FROM asset_media_metadata
                ORDER BY asset_id"""
            )
        ).fetchall()
        for migrated_asset in migrated_assets:
            asset_id = str(migrated_asset["asset_id"])
            descriptions = await (
                await db.execute(
                    """SELECT media_description,media_description_vector
                    FROM asset_media_metadata
                    WHERE asset_id=? ORDER BY sort_order,id""",
                    (asset_id,),
                )
            ).fetchall()
            if not descriptions:
                continue
            primary_description = str(descriptions[0]["media_description"])
            primary_vector = descriptions[0]["media_description_vector"]
            description_hash = media_description_set_sha256(
                str(item["media_description"]) for item in descriptions
            )
            relations = await (
                await db.execute(
                    """SELECT document_id,semantic_mode,media_description
                    FROM document_assets WHERE asset_id=?""",
                    (asset_id,),
                )
            ).fetchall()
            for relation in relations:
                equivalent_single = (
                    len(descriptions) == 1
                    and normalize_media_description(
                        str(relation["media_description"] or "")
                    )
                    == normalize_media_description(primary_description)
                )
                set_hash = (
                    description_hash
                    if equivalent_single
                    or str(relation["semantic_mode"] or "uniform")
                    != "calibrated"
                    else ""
                )
                await db.execute(
                    """UPDATE document_assets SET media_description=?,
                    media_description_vector=?,
                    calibration_description_set_sha256=?
                    WHERE document_id=? AND asset_id=?""",
                    (
                        primary_description,
                        primary_vector,
                        set_hash,
                        str(relation["document_id"]),
                        asset_id,
                    ),
                )
        await db.executescript(
            """
            DROP TABLE asset_media_metadata_v8;
            CREATE INDEX IF NOT EXISTS idx_asset_media_metadata_provider
                ON asset_media_metadata(provider_fingerprint,asset_id);
            CREATE INDEX IF NOT EXISTS idx_asset_media_metadata_asset_order
                ON asset_media_metadata(asset_id,sort_order,id);
            CREATE VIRTUAL TABLE asset_media_metadata_fts USING fts5(
                search_text,
                content='asset_media_metadata',
                content_rowid='id',
                tokenize='unicode61'
            );
            CREATE TRIGGER asset_media_metadata_ai
            AFTER INSERT ON asset_media_metadata BEGIN
                INSERT INTO asset_media_metadata_fts(rowid,search_text)
                VALUES(new.id,new.search_text);
            END;
            CREATE TRIGGER asset_media_metadata_ad
            AFTER DELETE ON asset_media_metadata BEGIN
                INSERT INTO asset_media_metadata_fts(
                    asset_media_metadata_fts,rowid,search_text
                ) VALUES('delete',old.id,old.search_text);
            END;
            CREATE TRIGGER asset_media_metadata_au
            AFTER UPDATE OF search_text ON asset_media_metadata BEGIN
                INSERT INTO asset_media_metadata_fts(
                    asset_media_metadata_fts,rowid,search_text
                ) VALUES('delete',old.id,old.search_text);
                INSERT INTO asset_media_metadata_fts(rowid,search_text)
                VALUES(new.id,new.search_text);
            END;
            INSERT INTO asset_media_metadata_fts(
                asset_media_metadata_fts
            ) VALUES('rebuild');
            """
        )

    async def _backfill_asset_media_metadata(self, db) -> None:
        """Create provider-free search metadata for images from older stores."""

        now = time.time()
        rows = await (
            await db.execute(
                """SELECT a.id,a.original_name,lm.provider_id,lm.provider_revision,
                lm.provider_fingerprint
                FROM assets a CROSS JOIN library_meta lm
                LEFT JOIN asset_media_metadata amm ON amm.asset_id=a.id
                WHERE a.kind='image' AND a.state='active' AND amm.asset_id IS NULL
                ORDER BY a.created_at,a.id"""
            )
        ).fetchall()
        for row in rows:
            relation_rows = await (
                await db.execute(
                    """SELECT da.media_description,da.caption,da.alt_text,
                    da.media_description_vector,da.calibration_provider_fingerprint,
                    da.sort_order,da.document_id
                    FROM document_assets da
                    WHERE da.asset_id=? AND da.output_policy<>'disabled'
                    ORDER BY da.sort_order,da.document_id""",
                    (row["id"],),
                )
            ).fetchall()
            filename_stem = Path(str(row["original_name"] or "")).stem
            descriptions: list[tuple[str, str, bytes | None]] = []
            seen: set[str] = set()
            for relation in relation_rows:
                candidate = str(relation["media_description"] or "").strip()
                if not candidate:
                    continue
                normalized = normalize_media_description(candidate)
                if normalized in seen:
                    continue
                seen.add(normalized)
                vector_blob = None
                if (
                    relation["media_description_vector"] is not None
                    and str(relation["calibration_provider_fingerprint"] or "")
                    == str(row["provider_fingerprint"] or "")
                ):
                    vector_blob = bytes(relation["media_description_vector"])
                descriptions.append((candidate, "relation", vector_blob))
            if not descriptions:
                descriptions.append((filename_stem or "image", "filename", None))
            for order, (description, source, vector_blob) in enumerate(descriptions):
                await db.execute(
                    """INSERT INTO asset_media_metadata
                    (asset_id,media_description,normalized_description,sort_order,
                     search_text,description_source,media_description_vector,
                     vector_sha256,provider_id,provider_revision,
                     provider_fingerprint,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        row["id"],
                        description,
                        normalize_media_description(description),
                        order,
                        lexical_text(description),
                        source,
                        vector_blob,
                        hashlib.sha256(vector_blob).hexdigest()
                        if vector_blob
                        else "",
                        str(row["provider_id"] or "") if vector_blob else "",
                        int(row["provider_revision"] or 0) if vector_blob else 0,
                        str(row["provider_fingerprint"] or "")
                        if vector_blob
                        else "",
                        now,
                        now,
                    ),
                )

    @staticmethod
    def _media_profile_tokens(
        media_descriptions: Iterable[str], original_name: str
    ) -> list[str]:
        return media_tokens(
            " ".join(
                [
                    *(
                        str(value or "").strip()
                        for value in media_descriptions
                        if str(value or "").strip()
                    ),
                    Path(str(original_name or "")).stem,
                ]
            )
        )

    async def _replace_asset_media_tokens(
        self,
        db,
        *,
        asset_id: str,
        media_descriptions: Iterable[str],
        original_name: str,
    ) -> None:
        await db.execute(
            "DELETE FROM asset_media_tokens WHERE asset_id=?", (asset_id,)
        )
        tokens = self._media_profile_tokens(media_descriptions, original_name)
        if tokens:
            await db.executemany(
                "INSERT INTO asset_media_tokens(asset_id,token) VALUES(?,?)",
                ((asset_id, token) for token in tokens),
            )

    async def _backfill_asset_media_tokens(self, db) -> None:
        """Rebuild provider-free per-asset token membership deterministically."""

        await db.execute("DELETE FROM asset_media_tokens")
        rows = await (
            await db.execute(
                """SELECT amm.asset_id,amm.media_description,a.original_name
                FROM asset_media_metadata amm
                JOIN assets a ON a.id=amm.asset_id
                WHERE a.kind='image' AND a.state='active'
                ORDER BY amm.asset_id,amm.sort_order,amm.id"""
            )
        ).fetchall()
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            item = grouped.setdefault(
                str(row["asset_id"]),
                {
                    "descriptions": [],
                    "original_name": str(row["original_name"] or ""),
                },
            )
            item["descriptions"].append(str(row["media_description"] or ""))
        for asset_id, item in grouped.items():
            await self._replace_asset_media_tokens(
                db,
                asset_id=asset_id,
                media_descriptions=item["descriptions"],
                original_name=item["original_name"],
            )

    async def close(self) -> None:
        await self.pool.close()

    async def create_library(
        self,
        *,
        database_id: str,
        name: str,
        description: str,
        provider_id: str,
        provider_revision: int,
        provider_fingerprint: str,
        rerank_provider_id: str = "",
        rerank_provider_revision: int = 0,
        rerank_provider_fingerprint: str = "",
    ) -> None:
        now = time.time()
        db = await self.pool.acquire()
        try:
            await db.execute(
                """INSERT INTO library_meta
                (singleton,database_id,name,description,provider_id,provider_revision,
                 provider_fingerprint,rerank_provider_id,rerank_provider_revision,
                 rerank_provider_fingerprint,status,chunk_target,chunk_overlap,
                 retrieval_config_json,created_at,updated_at)
                VALUES(1,?,?,?,?,?,?,?,?,?,'ready',?,?,?,?,?)""",
                (
                    database_id,
                    name,
                    description,
                    provider_id,
                    int(provider_revision),
                    provider_fingerprint,
                    rerank_provider_id,
                    int(rerank_provider_revision),
                    rerank_provider_fingerprint,
                    DEFAULT_CHUNK_TARGET,
                    DEFAULT_CHUNK_OVERLAP,
                    retrieval_config_json(),
                    now,
                    now,
                ),
            )
            await db.commit()
        finally:
            await db.close()

    async def metadata(self) -> dict[str, Any]:
        db = await self.pool.acquire()
        try:
            row = await (await db.execute("SELECT * FROM library_meta WHERE singleton=1")).fetchone()
            if row is None:
                raise KeyError("library metadata")
            return dict(row)
        finally:
            await db.close()

    async def summary_snapshot(self) -> dict[str, Any]:
        """Read card/task metadata without initializing a full runtime.

        Counts and metadata come from one SQLite snapshot.  The optional
        recalibration flag is reduced in SQL and never materializes per-media
        calibration rows in Python.
        """

        db = await self.pool.acquire()
        try:
            row = await (
                await db.execute(SUMMARY_SNAPSHOT_SQL)
            ).fetchone()
            if row is None:
                raise KeyError("library metadata")
            payload = dict(row)
            stats = {
                "documents": int(payload.pop("summary_documents") or 0),
                "entries": int(payload.pop("summary_entries") or 0),
                "chunks": int(payload.pop("summary_chunks") or 0),
                "images": int(payload.pop("summary_images") or 0),
            }
            rerank_fingerprint = str(
                payload.get("rerank_provider_fingerprint") or ""
            )
            needs_recalibration = False
            if rerank_fingerprint:
                settings_fingerprint = rerank_calibration_settings_fingerprint(
                    payload.get("retrieval_config_json")
                )
                mismatch = await (
                    await db.execute(
                        SUMMARY_RECALIBRATION_SQL,
                        (
                            rerank_fingerprint,
                            settings_fingerprint,
                            settings_fingerprint,
                        ),
                    )
                ).fetchone()
                needs_recalibration = bool(mismatch["value"])
            return {
                "metadata": payload,
                "stats": stats,
                "needs_recalibration": needs_recalibration,
            }
        finally:
            await db.close()

    @classmethod
    def summary_snapshot_from_disk(cls, root: Path) -> dict[str, Any]:
        """Read the exact summary snapshot without creating an aiosqlite worker.

        This path is used only for unloaded library cards.  A short-lived,
        query-only SQLite connection sees the same WAL snapshot as the async
        implementation while avoiding one persistent thread per cold library.
        """

        path = Path(root) / DATABASE_FILENAME
        if not path.is_file():
            raise FileNotFoundError(path)
        database_uri = path.resolve().as_uri() + "?mode=ro"
        db = sqlite3.connect(
            database_uri,
            uri=True,
            timeout=5.0,
            check_same_thread=False,
        )
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA query_only=ON")
            row = db.execute(SUMMARY_SNAPSHOT_SQL).fetchone()
            if row is None:
                raise KeyError("library metadata")
            payload = dict(row)
            stats = {
                "documents": int(payload.pop("summary_documents") or 0),
                "entries": int(payload.pop("summary_entries") or 0),
                "chunks": int(payload.pop("summary_chunks") or 0),
                "images": int(payload.pop("summary_images") or 0),
            }
            rerank_fingerprint = str(
                payload.get("rerank_provider_fingerprint") or ""
            )
            needs_recalibration = False
            if rerank_fingerprint:
                settings_fingerprint = rerank_calibration_settings_fingerprint(
                    payload.get("retrieval_config_json")
                )
                mismatch = db.execute(
                    SUMMARY_RECALIBRATION_SQL,
                    (
                        rerank_fingerprint,
                        settings_fingerprint,
                        settings_fingerprint,
                    ),
                ).fetchone()
                needs_recalibration = bool(mismatch["value"])
            return {
                "metadata": payload,
                "stats": stats,
                "needs_recalibration": needs_recalibration,
            }
        finally:
            db.close()

    async def update_metadata(self, payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "database_id",
            "name",
            "description",
            "provider_id",
            "provider_revision",
            "provider_fingerprint",
            "rerank_provider_id",
            "rerank_provider_revision",
            "rerank_provider_fingerprint",
            "status",
            "uniform_media_strength",
            "retrieval_config_json",
        }
        values = {key: payload[key] for key in allowed if key in payload}
        if "uniform_media_strength" in values:
            strength = float(values["uniform_media_strength"])
            if not math.isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError("uniform media strength must be between 0 and 1")
            values["uniform_media_strength"] = strength
        if "retrieval_config_json" in values:
            values["retrieval_config_json"] = retrieval_config_json(
                values["retrieval_config_json"]
            )
        if not values:
            return await self.metadata()
        values["updated_at"] = time.time()
        assignments = ",".join(f"{key}=?" for key in values)
        db = await self.pool.acquire()
        try:
            await db.execute(
                f"UPDATE library_meta SET {assignments} WHERE singleton=1",
                tuple(values.values()),
            )
            await db.commit()
        finally:
            await db.close()
        return await self.metadata()

    async def debug_revision_state(self) -> dict[str, Any]:
        """Return revision metadata only; never return vectors or media text."""

        db = await self.pool.acquire()
        try:
            meta = await (
                await db.execute("SELECT * FROM library_meta WHERE singleton=1")
            ).fetchone()
            if meta is None:
                raise KeyError("library metadata")
            generations = await (
                await db.execute(
                    """SELECT eg.id,eg.provider_id,eg.provider_revision,
                    eg.provider_fingerprint,eg.dimensions,eg.status,
                    eg.created_at,eg.activated_at,
                    CASE WHEN ag.generation_id=eg.id THEN 1 ELSE 0 END AS is_active
                    FROM embedding_generations eg
                    LEFT JOIN active_generation ag ON ag.generation_id=eg.id
                    ORDER BY eg.created_at DESC,eg.id"""
                )
            ).fetchall()
            media_embeddings = await (
                await db.execute(
                    """SELECT provider_id,provider_revision,provider_fingerprint,
                    COUNT(*) AS item_count,
                    SUM(CASE WHEN media_description_vector IS NOT NULL THEN 1 ELSE 0 END)
                        AS vector_count
                    FROM asset_media_metadata
                    GROUP BY provider_id,provider_revision,provider_fingerprint
                    ORDER BY provider_id,provider_revision"""
                )
            ).fetchall()
            relation_calibrations = await (
                await db.execute(
                    """SELECT semantic_mode,calibration_provider_fingerprint,
                    calibration_rerank_provider_fingerprint,COUNT(*) AS item_count
                    FROM document_assets
                    GROUP BY semantic_mode,calibration_provider_fingerprint,
                    calibration_rerank_provider_fingerprint
                    ORDER BY semantic_mode"""
                )
            ).fetchall()
            strength_calibrations = await (
                await db.execute(
                    """SELECT provider_fingerprint,rerank_provider_fingerprint,
                    COUNT(*) AS item_count FROM chunk_media_strengths
                    GROUP BY provider_fingerprint,rerank_provider_fingerprint"""
                )
            ).fetchall()
            return {
                "metadata": dict(meta),
                "embedding_generations": [dict(row) for row in generations],
                "media_embeddings": [dict(row) for row in media_embeddings],
                "relation_calibrations": [
                    dict(row) for row in relation_calibrations
                ],
                "strength_calibrations": [
                    dict(row) for row in strength_calibrations
                ],
            }
        finally:
            await db.close()

    async def debug_rebind_revision(
        self,
        *,
        usage_kind: str,
        provider_id: str,
        revision: int,
        fingerprint: str,
    ) -> None:
        if usage_kind not in {"embedding", "rerank"}:
            raise ValueError("usage_kind must be embedding or rerank")
        now = time.time()
        db = await self.pool.acquire()
        try:
            await db.execute("BEGIN IMMEDIATE")
            meta = await (
                await db.execute("SELECT * FROM library_meta WHERE singleton=1")
            ).fetchone()
            if meta is None:
                raise KeyError("library metadata")
            if usage_kind == "embedding":
                if str(meta["provider_id"] or "") != provider_id:
                    raise ValueError(
                        "revision repair cannot change the bound Provider ID"
                    )
                old_fingerprint = str(meta["provider_fingerprint"] or "")
                await db.execute(
                    """UPDATE library_meta SET provider_revision=?,
                    provider_fingerprint=?,updated_at=? WHERE singleton=1""",
                    (int(revision), fingerprint, now),
                )
                await db.execute(
                    """UPDATE embedding_generations SET provider_revision=?,
                    provider_fingerprint=? WHERE provider_id=?""",
                    (int(revision), fingerprint, provider_id),
                )
                await db.execute(
                    """UPDATE asset_media_metadata SET provider_revision=?,
                    provider_fingerprint=?,updated_at=? WHERE provider_id=?""",
                    (int(revision), fingerprint, now, provider_id),
                )
                if old_fingerprint:
                    await db.execute(
                        """UPDATE document_assets SET
                        calibration_provider_fingerprint=?
                        WHERE calibration_provider_fingerprint=?""",
                        (fingerprint, old_fingerprint),
                    )
                    await db.execute(
                        """UPDATE chunk_media_strengths SET
                        provider_fingerprint=?,updated_at=?
                        WHERE provider_fingerprint=?""",
                        (fingerprint, now, old_fingerprint),
                    )
            else:
                if str(meta["rerank_provider_id"] or "") != provider_id:
                    raise ValueError(
                        "revision repair cannot change the bound Provider ID"
                    )
                old_fingerprint = str(meta["rerank_provider_fingerprint"] or "")
                await db.execute(
                    """UPDATE library_meta SET rerank_provider_revision=?,
                    rerank_provider_fingerprint=?,updated_at=? WHERE singleton=1""",
                    (int(revision), fingerprint, now),
                )
                if old_fingerprint:
                    await db.execute(
                        """UPDATE document_assets SET
                        calibration_rerank_provider_fingerprint=?
                        WHERE calibration_rerank_provider_fingerprint=?""",
                        (fingerprint, old_fingerprint),
                    )
                    await db.execute(
                        """UPDATE chunk_media_strengths SET
                        rerank_provider_fingerprint=?,updated_at=?
                        WHERE rerank_provider_fingerprint=?""",
                        (fingerprint, now, old_fingerprint),
                    )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        finally:
            await db.close()

    async def register_asset(
        self,
        *,
        kind: str,
        sha256: str,
        storage_key: str,
        mime_type: str,
        size_bytes: int,
        original_name: str,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        if kind not in {"document", "image"}:
            raise ValueError("invalid asset kind")
        db = await self.pool.acquire()
        try:
            existing = await (
                await db.execute(
                    "SELECT * FROM assets WHERE kind=? AND sha256=?",
                    (kind, sha256),
                )
            ).fetchone()
            if existing:
                return dict(existing)
            asset_id = uuid.uuid4().hex
            await db.execute(
                """INSERT INTO assets
                (id,kind,sha256,storage_key,mime_type,size_bytes,width,height,
                 original_name,state,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,'active',?)""",
                (
                    asset_id,
                    kind,
                    sha256,
                    storage_key,
                    mime_type,
                    int(size_bytes),
                    width,
                    height,
                    original_name,
                    time.time(),
                ),
            )
            await db.commit()
            row = await (await db.execute("SELECT * FROM assets WHERE id=?", (asset_id,))).fetchone()
            return dict(row)
        finally:
            await db.close()

    async def install_document(
        self,
        *,
        title: str,
        source_asset_id: str,
        content: str,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
        provider_id: str,
        provider_revision: int,
        provider_fingerprint: str,
        parser_id: str,
    ) -> dict[str, Any]:
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("文档分块与向量数量不一致")
        encoded = [vector_bytes(vector) for vector in vectors]
        dimensions = len(encoded[0]) // 4
        if any(len(item) != dimensions * 4 for item in encoded):
            raise ValueError("Embedding 向量维度不一致")
        now = time.time()
        document_id = uuid.uuid4().hex
        entry_id = uuid.uuid4().hex
        generation_id = f"gen-{int(now)}-{uuid.uuid4().hex[:8]}"
        db = await self.pool.acquire()
        try:
            await db.execute("BEGIN IMMEDIATE")
            await db.execute(
                """INSERT INTO documents
                (id,title,source_asset_id,parser_id,content_sha256,status,error,created_at,updated_at)
                VALUES(?,?,?,?,?,'ready','',?,?)""",
                (document_id, title, source_asset_id, parser_id, hashlib.sha256(content.encode("utf-8")).hexdigest(), now, now),
            )
            await db.execute(
                """INSERT INTO entries
                (id,document_id,title,body,ordinal,status,created_at,updated_at)
                VALUES(?,?,?,?,0,'active',?,?)""",
                (entry_id, document_id, title, content, now, now),
            )
            chunk_ids: list[int] = []
            for item in chunks:
                cursor = await db.execute(
                    """INSERT INTO chunks
                    (entry_id,ordinal,text,search_text,char_start,char_end,content_sha256,status)
                    VALUES(?,?,?,?,?,?,?,'active')""",
                    (
                        entry_id,
                        int(item["ordinal"]),
                        str(item["text"]),
                        str(item["search_text"]),
                        int(item["char_start"]),
                        int(item["char_end"]),
                        str(item["content_sha256"]),
                    ),
                )
                chunk_ids.append(int(cursor.lastrowid))
            await db.execute(
                """INSERT INTO embedding_generations
                (id,provider_id,provider_revision,provider_fingerprint,dimensions,metric,status,created_at,activated_at)
                VALUES(?,?,?,?,?,'cosine_ip','active',?,?)""",
                (generation_id, provider_id, int(provider_revision), provider_fingerprint, dimensions, now, now),
            )
            previous = await (await db.execute("SELECT generation_id FROM active_generation WHERE singleton=1")).fetchone()
            if previous:
                await db.execute(
                    """INSERT INTO chunk_embeddings(generation_id,chunk_id,vector,vector_sha256)
                    SELECT ?,ce.chunk_id,ce.vector,ce.vector_sha256
                    FROM chunk_embeddings ce JOIN chunks c ON c.id=ce.chunk_id
                    WHERE ce.generation_id=? AND c.status='active'""",
                    (generation_id, previous["generation_id"]),
                )
                await db.execute(
                    "UPDATE embedding_generations SET status='superseded' WHERE id=?",
                    (previous["generation_id"],),
                )
            for chunk_id, blob in zip(chunk_ids, encoded, strict=True):
                await db.execute(
                    "INSERT INTO chunk_embeddings(generation_id,chunk_id,vector,vector_sha256) VALUES(?,?,?,?)",
                    (generation_id, chunk_id, blob, hashlib.sha256(blob).hexdigest()),
                )
            await db.execute(
                "INSERT OR REPLACE INTO active_generation(singleton,generation_id) VALUES(1,?)",
                (generation_id,),
            )
            await db.execute("UPDATE library_meta SET updated_at=? WHERE singleton=1", (now,))
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        finally:
            await db.close()
        return {"document_id": document_id, "entry_id": entry_id, "chunk_ids": chunk_ids, "generation_id": generation_id}

    async def install_ingest_batch(
        self,
        *,
        batch_id: str,
        parameters: dict[str, Any],
        documents: list[dict[str, Any]],
        images: list[dict[str, Any]],
        provider_id: str,
        provider_revision: int,
        provider_fingerprint: str,
    ) -> dict[str, Any]:
        if not documents and not images:
            raise ValueError("batch must contain at least one document or image")
        encoded_documents: list[list[bytes]] = []
        encoded_images: list[list[bytes]] = []
        dimensions = 0
        for document in documents:
            chunks = list(document.get("chunks") or [])
            vectors = list(document.get("vectors") or [])
            if not chunks or len(chunks) != len(vectors):
                raise ValueError("document chunks and vectors do not match")
            encoded = [vector_bytes(vector) for vector in vectors]
            current_dimensions = len(encoded[0]) // 4
            if dimensions and current_dimensions != dimensions:
                raise ValueError("embedding dimensions do not match")
            dimensions = current_dimensions
            if any(len(item) != dimensions * 4 for item in encoded):
                raise ValueError("embedding dimensions do not match")
            encoded_documents.append(encoded)

        for image in images:
            description_items = list(image.get("media_descriptions") or [])
            if not description_items and image.get("media_description"):
                description_items = [
                    {
                        "media_description": image["media_description"],
                        "vector": image.get("media_description_vector"),
                        "description_source": image.get(
                            "description_source", "user"
                        ),
                    }
                ]
            if not description_items:
                raise ValueError("every image requires media descriptions")
            image["media_descriptions"] = description_items
            encoded_for_image: list[bytes] = []
            for item in description_items:
                description_vector = item.get("vector")
                if description_vector is None:
                    raise ValueError(
                        "every media description requires an embedding"
                    )
                encoded = vector_bytes(description_vector)
                current_dimensions = len(encoded) // 4
                if dimensions and current_dimensions != dimensions:
                    raise ValueError(
                        "media and chunk embedding dimensions do not match"
                    )
                dimensions = current_dimensions
                encoded_for_image.append(encoded)
            encoded_images.append(encoded_for_image)

        now = time.time()
        generation_id = (
            f"gen-{int(now)}-{uuid.uuid4().hex[:8]}" if documents else None
        )
        db = await self.pool.acquire()
        try:
            await db.execute("BEGIN IMMEDIATE")
            existing_batch = await (
                await db.execute("SELECT id FROM ingest_batches WHERE id=?", (batch_id,))
            ).fetchone()
            if existing_batch:
                raise ValueError("ingest batch already exists")

            async def ensure_asset(item: dict[str, Any], kind: str) -> str:
                existing = await (
                    await db.execute(
                        "SELECT id FROM assets WHERE kind=? AND sha256=?",
                        (kind, str(item["sha256"])),
                    )
                ).fetchone()
                if existing:
                    return str(existing["id"])
                asset_id = uuid.uuid4().hex
                await db.execute(
                    """INSERT INTO assets
                    (id,kind,sha256,storage_key,mime_type,size_bytes,width,height,
                     original_name,state,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?,'active',?)""",
                    (
                        asset_id,
                        kind,
                        str(item["sha256"]),
                        str(item["storage_key"]),
                        str(item["mime_type"]),
                        int(item["size_bytes"]),
                        item.get("width"),
                        item.get("height"),
                        str(item.get("original_name") or ""),
                        now,
                    ),
                )
                return asset_id

            await db.execute(
                """INSERT INTO ingest_batches
                (id,parameters_json,document_count,image_count,chunk_count,status,created_at,completed_at)
                VALUES(?,?,?,?,?,'completed',?,?)""",
                (
                    batch_id,
                    json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                    len(documents),
                    len(images),
                    sum(len(item["chunks"]) for item in documents),
                    now,
                    now,
                ),
            )

            document_rows: list[dict[str, Any]] = []
            all_chunk_ids: list[int] = []
            for ordinal, (document, encoded) in enumerate(
                zip(documents, encoded_documents, strict=True)
            ):
                source_asset_id = await ensure_asset(document, "document")
                document_id = uuid.uuid4().hex
                entry_id = uuid.uuid4().hex
                await db.execute(
                    """INSERT INTO documents
                    (id,title,source_asset_id,parser_id,content_sha256,status,error,created_at,updated_at)
                    VALUES(?,?,?,?,?,'ready','',?,?)""",
                    (
                        document_id,
                        str(document["title"]),
                        source_asset_id,
                        str(document["parser_id"]),
                        str(document["content_sha256"]),
                        now,
                        now,
                    ),
                )
                await db.execute(
                    """INSERT INTO entries
                    (id,document_id,title,body,ordinal,status,created_at,updated_at)
                    VALUES(?,?,?,?,0,'active',?,?)""",
                    (
                        entry_id,
                        document_id,
                        str(document["title"]),
                        str(document["content"]),
                        now,
                        now,
                    ),
                )
                chunk_ids: list[int] = []
                for chunk in document["chunks"]:
                    cursor = await db.execute(
                        """INSERT INTO chunks
                        (entry_id,ordinal,text,search_text,char_start,char_end,content_sha256,status)
                        VALUES(?,?,?,?,?,?,?,'active')""",
                        (
                            entry_id,
                            int(chunk["ordinal"]),
                            str(chunk["text"]),
                            str(chunk["search_text"]),
                            int(chunk["char_start"]),
                            int(chunk["char_end"]),
                            str(chunk["content_sha256"]),
                        ),
                    )
                    chunk_ids.append(int(cursor.lastrowid))
                all_chunk_ids.extend(chunk_ids)
                await db.execute(
                    "INSERT INTO ingest_batch_documents(batch_id,document_id,ordinal) VALUES(?,?,?)",
                    (batch_id, document_id, ordinal),
                )
                document_rows.append(
                    {
                        "document_id": document_id,
                        "entry_id": entry_id,
                        "chunk_ids": chunk_ids,
                    }
                )

            image_rows: list[dict[str, Any]] = []
            media_metadata_before: list[dict[str, Any]] = []
            for ordinal, (image, encoded_media_vectors) in enumerate(
                zip(images, encoded_images, strict=True)
            ):
                asset_id = await ensure_asset(image, "image")
                previous_metadata = await (
                    await db.execute(
                        """SELECT * FROM asset_media_metadata
                        WHERE asset_id=? ORDER BY sort_order,id""",
                        (asset_id,),
                    )
                ).fetchall()
                media_metadata_before.append(
                    {
                        "asset_id": asset_id,
                        "rows": [dict(row) for row in previous_metadata],
                    }
                )
                indexes = sorted({int(value) for value in image["document_indexes"]})
                if indexes and (indexes[0] < 0 or indexes[-1] >= len(document_rows)):
                    raise ValueError("image document mapping is invalid")
                description_items = list(image.get("media_descriptions") or [])
                descriptions = media_description_list(
                    [
                        str(item.get("media_description") or "")
                        for item in description_items
                    ]
                )
                await db.execute(
                    "DELETE FROM asset_media_metadata WHERE asset_id=?",
                    (asset_id,),
                )
                for order, (description, item, encoded_media_vector) in enumerate(
                    zip(
                        descriptions,
                        description_items,
                        encoded_media_vectors,
                        strict=True,
                    )
                ):
                    await db.execute(
                        """INSERT INTO asset_media_metadata
                        (asset_id,media_description,normalized_description,
                         sort_order,search_text,description_source,
                         media_description_vector,vector_sha256,provider_id,
                         provider_revision,provider_fingerprint,created_at,updated_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            asset_id,
                            description,
                            normalize_media_description(description),
                            order,
                            lexical_text(description),
                            str(item.get("description_source") or "user"),
                            encoded_media_vector,
                            hashlib.sha256(encoded_media_vector).hexdigest(),
                            provider_id,
                            int(provider_revision),
                            provider_fingerprint,
                            now,
                            now,
                        ),
                    )
                await self._replace_asset_media_tokens(
                    db,
                    asset_id=asset_id,
                    media_descriptions=descriptions,
                    original_name=str(image.get("original_name") or ""),
                )
                await db.execute(
                    """INSERT INTO ingest_batch_assets
                    (batch_id,asset_id,ordinal,document_indexes_json) VALUES(?,?,?,?)""",
                    (batch_id, asset_id, ordinal, json.dumps(indexes)),
                )
                for document_index in indexes:
                    calibration = dict(
                        (image.get("document_calibrations") or {}).get(
                            document_index, {}
                        )
                    )
                    semantic_mode = str(
                        calibration.get("semantic_mode") or "uniform"
                    )
                    media_description = descriptions[0]
                    calibration_method = str(
                        calibration.get("calibration_method") or "uniform_v1"
                    )
                    calibration_rerank_fingerprint = str(
                        calibration.get("rerank_provider_fingerprint") or ""
                    )
                    description_vector = calibration.get(
                        "media_description_vector"
                    )
                    if description_vector is None:
                        description_vector = image["media_descriptions"][0].get(
                            "vector"
                        )
                    encoded_description_vector = (
                        vector_bytes(description_vector)
                        if description_vector is not None
                        else None
                    )
                    if (
                        encoded_description_vector is not None
                        and len(encoded_description_vector) != dimensions * 4
                    ):
                        raise ValueError(
                            "media description and chunk dimensions do not match"
                        )
                    await db.execute(
                        """INSERT INTO document_assets
                        (document_id,asset_id,role,relation_weight,caption,alt_text,
                         output_policy,sort_order,semantic_mode,media_description,
                         media_description_vector,calibration_method,
                         calibration_provider_fingerprint,
                         calibration_rerank_provider_fingerprint,
                         calibration_description_set_sha256)
                        VALUES(?,?,'illustration',1,'','','auto',?,?,?,?,?,?,?,?)
                        ON CONFLICT(document_id,asset_id) DO UPDATE SET
                        role='illustration',relation_weight=1,output_policy='auto',
                        sort_order=excluded.sort_order,semantic_mode=excluded.semantic_mode,
                        media_description=excluded.media_description,
                        media_description_vector=excluded.media_description_vector,
                        calibration_method=excluded.calibration_method,
                        calibration_provider_fingerprint=excluded.calibration_provider_fingerprint,
                        calibration_rerank_provider_fingerprint=
                            excluded.calibration_rerank_provider_fingerprint,
                        calibration_description_set_sha256=
                            excluded.calibration_description_set_sha256""",
                        (
                            document_rows[document_index]["document_id"],
                            asset_id,
                            ordinal,
                            semantic_mode,
                            media_description,
                            encoded_description_vector,
                            calibration_method,
                            provider_fingerprint,
                            calibration_rerank_fingerprint,
                            media_description_set_sha256(descriptions),
                        ),
                    )
                    calibration_rows = list(calibration.get("chunks") or [])
                    chunk_ids = document_rows[document_index]["chunk_ids"]
                    if calibration_rows and len(calibration_rows) != len(chunk_ids):
                        raise ValueError("media calibration does not match document chunks")
                    if not calibration_rows:
                        calibration_rows = [
                            {
                                "semantic_strength": 1.0,
                                "calibration_similarity": None,
                                "calibration_rank": None,
                                "rerank_semantic_strength": None,
                                "calibration_details": {},
                            }
                            for _ in chunk_ids
                        ]
                    for chunk_id, calibrated in zip(
                        chunk_ids, calibration_rows, strict=True
                    ):
                        await db.execute(
                            """INSERT INTO chunk_media_strengths
                            (document_id,chunk_id,asset_id,semantic_strength,
                             rerank_semantic_strength,
                             calibration_similarity,calibration_rank,
                             calibration_method,provider_fingerprint,
                             rerank_provider_fingerprint,calibration_details_json,
                             updated_at)
                            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                            ON CONFLICT(chunk_id,asset_id) DO UPDATE SET
                            document_id=excluded.document_id,
                            semantic_strength=excluded.semantic_strength,
                            rerank_semantic_strength=excluded.rerank_semantic_strength,
                            calibration_similarity=excluded.calibration_similarity,
                            calibration_rank=excluded.calibration_rank,
                            calibration_method=excluded.calibration_method,
                            provider_fingerprint=excluded.provider_fingerprint,
                            rerank_provider_fingerprint=excluded.rerank_provider_fingerprint,
                            calibration_details_json=excluded.calibration_details_json,
                            updated_at=excluded.updated_at""",
                            (
                                document_rows[document_index]["document_id"],
                                chunk_id,
                                asset_id,
                                max(
                                    0.0,
                                    min(
                                        1.0,
                                        float(
                                            calibrated.get(
                                                "semantic_strength", 1.0
                                            )
                                        ),
                                    ),
                                ),
                                (
                                    max(
                                        0.0,
                                        min(
                                            1.0,
                                            float(calibrated["rerank_semantic_strength"]),
                                        ),
                                    )
                                    if calibrated.get("rerank_semantic_strength")
                                    is not None
                                    else None
                                ),
                                calibrated.get("calibration_similarity"),
                                calibrated.get("calibration_rank"),
                                calibration_method,
                                provider_fingerprint,
                                calibration_rerank_fingerprint,
                                json.dumps(
                                    calibrated.get("calibration_details") or {},
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                                now,
                            ),
                        )
                image_rows.append({"asset_id": asset_id, "document_indexes": indexes})

            previous_generation_id = None
            if generation_id is not None:
                await db.execute(
                    """INSERT INTO embedding_generations
                    (id,provider_id,provider_revision,provider_fingerprint,dimensions,metric,status,created_at,activated_at)
                    VALUES(?,?,?,?,?,'cosine_ip','active',?,?)""",
                    (
                        generation_id,
                        provider_id,
                        int(provider_revision),
                        provider_fingerprint,
                        dimensions,
                        now,
                        now,
                    ),
                )
                previous = await (
                    await db.execute(
                        "SELECT generation_id FROM active_generation WHERE singleton=1"
                    )
                ).fetchone()
                previous_generation_id = (
                    str(previous["generation_id"]) if previous else None
                )
                if previous:
                    await db.execute(
                        """INSERT INTO chunk_embeddings(generation_id,chunk_id,vector,vector_sha256)
                        SELECT ?,ce.chunk_id,ce.vector,ce.vector_sha256
                        FROM chunk_embeddings ce JOIN chunks c ON c.id=ce.chunk_id
                        WHERE ce.generation_id=? AND c.status='active'""",
                        (generation_id, previous["generation_id"]),
                    )
                    await db.execute(
                        "UPDATE embedding_generations SET status='superseded' WHERE id=?",
                        (previous["generation_id"],),
                    )
                for document_row, encoded in zip(
                    document_rows, encoded_documents, strict=True
                ):
                    for chunk_id, blob in zip(
                        document_row["chunk_ids"], encoded, strict=True
                    ):
                        await db.execute(
                            "INSERT INTO chunk_embeddings(generation_id,chunk_id,vector,vector_sha256) VALUES(?,?,?,?)",
                            (
                                generation_id,
                                chunk_id,
                                blob,
                                hashlib.sha256(blob).hexdigest(),
                            ),
                        )
                await db.execute(
                    "INSERT OR REPLACE INTO active_generation(singleton,generation_id) VALUES(1,?)",
                    (generation_id,),
                )
            await db.execute(
                "UPDATE library_meta SET updated_at=? WHERE singleton=1", (now,)
            )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        finally:
            await db.close()
        return {
            "batch_id": batch_id,
            "generation_id": generation_id,
            "previous_generation_id": previous_generation_id,
            "documents": document_rows,
            "images": image_rows,
            "media_metadata_before": media_metadata_before,
            "chunk_ids": all_chunk_ids,
        }

    async def rollback_ingest_batch(
        self,
        *,
        batch_id: str,
        generation_id: str | None,
        previous_generation_id: str | None,
        media_metadata_before: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        db = await self.pool.acquire()
        try:
            await db.execute("BEGIN IMMEDIATE")
            document_rows = await (
                await db.execute(
                    "SELECT document_id FROM ingest_batch_documents WHERE batch_id=?",
                    (batch_id,),
                )
            ).fetchall()
            asset_rows = await (
                await db.execute(
                    """SELECT DISTINCT a.id,a.storage_key FROM assets a
                    LEFT JOIN documents d ON d.source_asset_id=a.id
                    LEFT JOIN document_assets da ON da.asset_id=a.id
                    LEFT JOIN entry_assets ea ON ea.asset_id=a.id
                    LEFT JOIN chunk_assets ca ON ca.asset_id=a.id
                    LEFT JOIN ingest_batch_assets iba ON iba.asset_id=a.id AND iba.batch_id<>?
                    WHERE a.id IN (
                        SELECT source_asset_id FROM documents WHERE id IN (
                            SELECT document_id FROM ingest_batch_documents WHERE batch_id=?
                        )
                        UNION
                        SELECT asset_id FROM ingest_batch_assets WHERE batch_id=?
                    )""",
                    (batch_id, batch_id, batch_id),
                )
            ).fetchall()
            if generation_id:
                await db.execute("DELETE FROM active_generation WHERE singleton=1")
                if previous_generation_id:
                    await db.execute(
                        "UPDATE embedding_generations SET status='active' WHERE id=?",
                        (previous_generation_id,),
                    )
                    await db.execute(
                        "INSERT INTO active_generation(singleton,generation_id) VALUES(1,?)",
                        (previous_generation_id,),
                    )
                await db.execute(
                    "DELETE FROM embedding_generations WHERE id=?", (generation_id,)
                )
            for row in document_rows:
                await db.execute(
                    "DELETE FROM documents WHERE id=?", (row["document_id"],)
                )
            await db.execute("DELETE FROM ingest_batches WHERE id=?", (batch_id,))

            for snapshot in media_metadata_before or []:
                asset_id = str(snapshot["asset_id"])
                previous_rows = list(snapshot.get("rows") or [])
                if not previous_rows and snapshot.get("row"):
                    previous_rows = [dict(snapshot["row"])]
                if not previous_rows:
                    await db.execute(
                        "DELETE FROM asset_media_metadata WHERE asset_id=?",
                        (asset_id,),
                    )
                    await db.execute(
                        "DELETE FROM asset_media_tokens WHERE asset_id=?",
                        (asset_id,),
                    )
                    continue
                await db.execute(
                    "DELETE FROM asset_media_metadata WHERE asset_id=?",
                    (asset_id,),
                )
                columns = (
                    "id",
                    "asset_id",
                    "media_description",
                    "normalized_description",
                    "sort_order",
                    "search_text",
                    "description_source",
                    "media_description_vector",
                    "vector_sha256",
                    "provider_id",
                    "provider_revision",
                    "provider_fingerprint",
                    "created_at",
                    "updated_at",
                )
                for order, previous in enumerate(previous_rows):
                    values = {
                        **previous,
                        "normalized_description": str(
                            previous.get("normalized_description")
                            or normalize_media_description(
                                str(previous.get("media_description") or "")
                            )
                        ),
                        "sort_order": int(
                            previous.get("sort_order")
                            if previous.get("sort_order") is not None
                            else order
                        ),
                    }
                    await db.execute(
                        f"INSERT INTO asset_media_metadata ({','.join(columns)}) "
                        f"VALUES ({','.join('?' for _ in columns)})",
                        tuple(values[column] for column in columns),
                    )
                asset = await (
                    await db.execute(
                        "SELECT original_name FROM assets WHERE id=?",
                        (asset_id,),
                    )
                ).fetchone()
                await self._replace_asset_media_tokens(
                    db,
                    asset_id=asset_id,
                    media_descriptions=[
                        str(row.get("media_description") or "")
                        for row in previous_rows
                    ],
                    original_name=str(asset["original_name"] or "") if asset else "",
                )

            removable: list[str] = []
            for row in asset_rows:
                referenced = await (
                    await db.execute(
                        """SELECT
                        EXISTS(SELECT 1 FROM documents WHERE source_asset_id=?) OR
                        EXISTS(SELECT 1 FROM document_assets WHERE asset_id=?) OR
                        EXISTS(SELECT 1 FROM entry_assets WHERE asset_id=?) OR
                        EXISTS(SELECT 1 FROM chunk_assets WHERE asset_id=?) OR
                        EXISTS(SELECT 1 FROM ingest_batch_assets WHERE asset_id=?) AS value""",
                        (row["id"], row["id"], row["id"], row["id"], row["id"]),
                    )
                ).fetchone()
                if not bool(referenced["value"]):
                    await db.execute("DELETE FROM assets WHERE id=?", (row["id"],))
                    removable.append(str(row["storage_key"]))
            await db.commit()
            return removable
        except BaseException:
            await db.rollback()
            raise
        finally:
            await db.close()

    async def replace_entry(
        self,
        *,
        entry_id: str,
        title: str,
        body: str,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> dict[str, Any]:
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("条目分块与向量数量不一致")
        encoded = [vector_bytes(vector) for vector in vectors]
        dimensions = len(encoded[0]) // 4
        meta = await self.metadata()
        now = time.time()
        generation_id = f"gen-{int(now)}-{uuid.uuid4().hex[:8]}"
        db = await self.pool.acquire()
        try:
            await db.execute("BEGIN IMMEDIATE")
            existing = await (
                await db.execute("SELECT id FROM entries WHERE id=?", (entry_id,))
            ).fetchone()
            if not existing:
                raise KeyError(entry_id)
            previous = await (
                await db.execute(
                    "SELECT generation_id FROM active_generation WHERE singleton=1"
                )
            ).fetchone()
            await db.execute("DELETE FROM chunks WHERE entry_id=?", (entry_id,))
            await db.execute(
                "UPDATE entries SET title=?,body=?,updated_at=? WHERE id=?",
                (title, body, now, entry_id),
            )
            chunk_ids: list[int] = []
            for item in chunks:
                cursor = await db.execute(
                    """INSERT INTO chunks
                    (entry_id,ordinal,text,search_text,char_start,char_end,content_sha256,status)
                    VALUES(?,?,?,?,?,?,?,'active')""",
                    (
                        entry_id,
                        int(item["ordinal"]),
                        str(item["text"]),
                        str(item["search_text"]),
                        int(item["char_start"]),
                        int(item["char_end"]),
                        str(item["content_sha256"]),
                    ),
                )
                chunk_ids.append(int(cursor.lastrowid))
            await db.execute(
                """INSERT INTO embedding_generations
                (id,provider_id,provider_revision,provider_fingerprint,dimensions,metric,status,created_at,activated_at)
                VALUES(?,?,?,?,?,'cosine_ip','active',?,?)""",
                (
                    generation_id,
                    meta["provider_id"],
                    int(meta["provider_revision"]),
                    meta["provider_fingerprint"],
                    dimensions,
                    now,
                    now,
                ),
            )
            if previous:
                await db.execute(
                    """INSERT INTO chunk_embeddings(generation_id,chunk_id,vector,vector_sha256)
                    SELECT ?,ce.chunk_id,ce.vector,ce.vector_sha256 FROM chunk_embeddings ce
                    JOIN chunks c ON c.id=ce.chunk_id
                    WHERE ce.generation_id=? AND c.status='active'""",
                    (generation_id, previous["generation_id"]),
                )
                await db.execute(
                    "UPDATE embedding_generations SET status='superseded' WHERE id=?",
                    (previous["generation_id"],),
                )
            for chunk_id, blob in zip(chunk_ids, encoded, strict=True):
                await db.execute(
                    "INSERT INTO chunk_embeddings(generation_id,chunk_id,vector,vector_sha256) VALUES(?,?,?,?)",
                    (generation_id, chunk_id, blob, hashlib.sha256(blob).hexdigest()),
                )
            await db.execute(
                "INSERT OR REPLACE INTO active_generation(singleton,generation_id) VALUES(1,?)",
                (generation_id,),
            )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        finally:
            await db.close()
        return {"entry_id": entry_id, "chunk_ids": chunk_ids, "generation_id": generation_id}

    async def active_vectors(self) -> tuple[str | None, list[int], np.ndarray]:
        db = await self.pool.acquire()
        try:
            active = await (await db.execute("SELECT generation_id FROM active_generation WHERE singleton=1")).fetchone()
            if not active:
                return None, [], np.empty((0, 0), dtype=np.float32)
            rows = await (
                await db.execute(
                    """SELECT ce.chunk_id,ce.vector FROM chunk_embeddings ce
                    JOIN chunks c ON c.id=ce.chunk_id
                    WHERE ce.generation_id=? AND c.status='active' ORDER BY ce.chunk_id""",
                    (active["generation_id"],),
                )
            ).fetchall()
            if not rows:
                return str(active["generation_id"]), [], np.empty((0, 0), dtype=np.float32)
            vectors = np.stack([np.frombuffer(row["vector"], dtype="<f4") for row in rows]).astype(np.float32, copy=False)
            return str(active["generation_id"]), [int(row["chunk_id"]) for row in rows], vectors
        finally:
            await db.close()

    async def active_media_vectors(self) -> tuple[str | None, list[int], np.ndarray]:
        """Return searchable asset-description vectors for the current binding."""

        db = await self.pool.acquire()
        try:
            rows = await (
                await db.execute(
                    """SELECT amm.id,amm.vector_sha256,amm.media_description_vector,
                    lm.provider_fingerprint
                    FROM asset_media_metadata amm
                    JOIN assets a ON a.id=amm.asset_id
                    CROSS JOIN library_meta lm
                    WHERE a.kind='image' AND a.state='active'
                      AND amm.media_description_vector IS NOT NULL
                      AND amm.provider_fingerprint=lm.provider_fingerprint
                    ORDER BY amm.id"""
                )
            ).fetchall()
            if not rows:
                return None, [], np.empty((0, 0), dtype=np.float32)
            vectors = np.stack(
                [
                    np.frombuffer(row["media_description_vector"], dtype="<f4")
                    for row in rows
                ]
            ).astype(np.float32, copy=False)
            digest = hashlib.sha256()
            digest.update(str(rows[0]["provider_fingerprint"]).encode("utf-8"))
            for row in rows:
                digest.update(str(int(row["id"])).encode("ascii"))
                digest.update(str(row["vector_sha256"] or "").encode("ascii"))
            return (
                f"media-{digest.hexdigest()[:24]}",
                [int(row["id"]) for row in rows],
                vectors,
            )
        finally:
            await db.close()

    async def media_metadata_records(self) -> list[dict[str, Any]]:
        db = await self.pool.acquire()
        try:
            rows = await (
                await db.execute(
                    """SELECT amm.*,a.original_name,a.sha256,a.mime_type,a.size_bytes,
                    a.width,a.height,
                    CASE WHEN EXISTS(
                      SELECT 1 FROM document_assets da
                      WHERE da.asset_id=a.id AND da.output_policy<>'disabled'
                    ) OR EXISTS(
                      SELECT 1 FROM entry_assets ea
                      WHERE ea.asset_id=a.id AND ea.output_policy<>'disabled'
                    ) OR EXISTS(
                      SELECT 1 FROM chunk_assets ca
                      WHERE ca.asset_id=a.id AND ca.output_policy<>'disabled'
                    ) THEN 1 ELSE 0 END AS is_bound
                    FROM asset_media_metadata amm
                    JOIN assets a ON a.id=amm.asset_id
                    WHERE a.kind='image' AND a.state='active'
                    ORDER BY amm.id"""
                )
            ).fetchall()
            result = [dict(row) for row in rows]
            for item in result:
                blob = item.get("media_description_vector")
                item["media_description_vector"] = (
                    np.frombuffer(blob, dtype="<f4").copy()
                    if blob is not None
                    else None
                )
            return result
        finally:
            await db.close()

    async def media_metadata_identities(self) -> list[tuple[int, str]]:
        """Return the stable FAISS row-to-asset mapping without loading vectors."""

        db = await self.pool.acquire()
        try:
            rows = await (
                await db.execute(
                    """SELECT amm.id,amm.asset_id
                    FROM asset_media_metadata amm
                    JOIN assets a ON a.id=amm.asset_id
                    WHERE a.kind='image' AND a.state='active'
                    ORDER BY amm.id"""
                )
            ).fetchall()
            return [(int(row["id"]), str(row["asset_id"])) for row in rows]
        finally:
            await db.close()

    async def media_metadata_rows(self, row_ids: Iterable[int]) -> list[dict[str, Any]]:
        ids = sorted({int(value) for value in row_ids})
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        db = await self.pool.acquire()
        try:
            rows = await (
                await db.execute(
                    f"""SELECT amm.*,a.original_name,a.sha256,a.mime_type,
                    a.size_bytes,a.width,a.height,
                    lm.provider_fingerprint AS current_provider_fingerprint,
                    CASE WHEN EXISTS(
                      SELECT 1 FROM document_assets da
                      WHERE da.asset_id=a.id AND da.output_policy<>'disabled'
                    ) OR EXISTS(
                      SELECT 1 FROM entry_assets ea
                      WHERE ea.asset_id=a.id AND ea.output_policy<>'disabled'
                    ) OR EXISTS(
                      SELECT 1 FROM chunk_assets ca
                      WHERE ca.asset_id=a.id AND ca.output_policy<>'disabled'
                    ) THEN 1 ELSE 0 END AS is_bound
                    FROM asset_media_metadata amm
                    JOIN assets a ON a.id=amm.asset_id
                    CROSS JOIN library_meta lm
                    WHERE amm.id IN ({placeholders}) AND a.kind='image'
                      AND a.state='active' ORDER BY amm.id""",
                    tuple(ids),
                )
            ).fetchall()
            result = [dict(row) for row in rows]
            for item in result:
                blob = item.get("media_description_vector")
                if str(item.get("provider_fingerprint") or "") != str(
                    item.pop("current_provider_fingerprint", "") or ""
                ):
                    blob = None
                item["media_description_vector"] = (
                    np.frombuffer(blob, dtype="<f4").copy()
                    if blob is not None
                    else None
                )
            return result
        finally:
            await db.close()

    async def media_token_statistics(
        self,
        query_tokens: Iterable[str],
        *,
        bound_only: bool | None = None,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Return scope-isolated corpus frequency and exact token matches."""

        tokens = list(
            dict.fromkeys(str(value) for value in query_tokens if str(value))
        )
        if scope is None:
            scope = "bound" if bound_only else "all"
        if scope not in {"all", "bound", "unbound"}:
            raise ValueError("media token scope must be all, bound, or unbound")
        effective_binding = """(
                EXISTS (
                  SELECT 1 FROM document_assets da
                  WHERE da.asset_id=a.id AND da.output_policy<>'disabled'
                ) OR EXISTS (
                  SELECT 1 FROM entry_assets ea
                  WHERE ea.asset_id=a.id AND ea.output_policy<>'disabled'
                ) OR EXISTS (
                  SELECT 1 FROM chunk_assets ca
                  WHERE ca.asset_id=a.id AND ca.output_policy<>'disabled'
                )
            )"""
        bound_clause = (
            f"AND {effective_binding}"
            if scope == "bound"
            else f"AND NOT {effective_binding}"
            if scope == "unbound"
            else ""
        )
        scope_name = {
            "all": "all_active_media",
            "bound": "bound_active_media",
            "unbound": "unbound_active_media",
        }[scope]
        db = await self.pool.acquire()
        try:
            corpus_row = await (
                await db.execute(
                    f"""SELECT COUNT(DISTINCT a.id) AS value
                    FROM assets a
                    WHERE a.kind='image' AND a.state='active'
                    {bound_clause}"""
                )
            ).fetchone()
            corpus_size = int(corpus_row["value"] if corpus_row else 0)
            if not tokens:
                return {
                    "scope": scope_name,
                    "corpus_size": corpus_size,
                    "document_frequencies": {},
                    "asset_matches": {},
                    "row_ids": {},
                }
            placeholders = ",".join("?" for _ in tokens)
            rows = await (
                await db.execute(
                    f"""SELECT amm.id AS row_id,amt.asset_id,amt.token
                    FROM asset_media_tokens amt
                    JOIN assets a ON a.id=amt.asset_id
                    JOIN asset_media_metadata amm
                      ON amm.asset_id=a.id AND amm.sort_order=0
                    WHERE a.kind='image' AND a.state='active'
                      AND amt.token IN ({placeholders})
                      {bound_clause}
                    ORDER BY amm.id,amt.token""",
                    tuple(tokens),
                )
            ).fetchall()
        finally:
            await db.close()
        frequencies = {token: 0 for token in tokens}
        asset_matches: dict[str, list[str]] = {}
        row_ids: dict[str, int] = {}
        for row in rows:
            token = str(row["token"])
            asset_id = str(row["asset_id"])
            frequencies[token] = frequencies.get(token, 0) + 1
            asset_matches.setdefault(asset_id, []).append(token)
            row_ids[asset_id] = int(row["row_id"])
        return {
            "scope": scope_name,
            "corpus_size": corpus_size,
            "document_frequencies": frequencies,
            "asset_matches": asset_matches,
            "row_ids": row_ids,
        }

    async def media_metadata_lexical_search(
        self, query: str, limit: int
    ) -> list[tuple[int, float]]:
        tokens = media_tokens(query)
        context = await self.media_token_statistics(tokens, bound_only=False)
        size = int(context["corpus_size"])
        frequencies = dict(context["document_frequencies"])
        scored: list[tuple[int, float]] = []
        for asset_id, matched in dict(context["asset_matches"]).items():
            score = 0.0
            for token in matched:
                frequency = max(1, int(frequencies.get(token, 1)))
                score += float(max(1, min(4, len(token)))) * math.log(
                    1.0 + (size - frequency + 0.5) / (frequency + 0.5)
                )
            scored.append((int(context["row_ids"][asset_id]), score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[: max(1, int(limit))]

    async def upsert_asset_media_metadata(
        self,
        *,
        asset_id: str,
        media_description: str,
        description_source: str,
        vector: list[float] | np.ndarray,
        provider_id: str,
        provider_revision: int,
        provider_fingerprint: str,
    ) -> dict[str, Any]:
        result = await self.replace_asset_media_descriptions(
            asset_id=asset_id,
            descriptions=[
                {
                    "media_description": media_description,
                    "description_source": description_source,
                    "vector": vector,
                }
            ],
            provider_id=provider_id,
            provider_revision=provider_revision,
            provider_fingerprint=provider_fingerprint,
        )
        return {
            "asset_id": asset_id,
            "media_description": result["media_description"],
            "description_source": description_source,
            "provider_fingerprint": provider_fingerprint,
        }

    async def asset_media_descriptions(
        self, asset_id: str
    ) -> list[dict[str, Any]]:
        db = await self.pool.acquire()
        try:
            rows = await (
                await db.execute(
                    """SELECT amm.*,lm.provider_fingerprint AS current_fingerprint
                    FROM asset_media_metadata amm CROSS JOIN library_meta lm
                    WHERE amm.asset_id=?
                    ORDER BY amm.sort_order,amm.id""",
                    (asset_id,),
                )
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                blob = item.get("media_description_vector")
                item["media_description_vector"] = (
                    np.frombuffer(blob, dtype="<f4").copy()
                    if blob is not None
                    else None
                )
                item["vector_status"] = (
                    "lexical_only"
                    if blob is None
                    else "stale"
                    if str(item.get("provider_fingerprint") or "")
                    != str(item.pop("current_fingerprint", "") or "")
                    else "ready"
                )
                item.pop("current_fingerprint", None)
                result.append(item)
            return result
        finally:
            await db.close()

    async def replace_asset_media_descriptions(
        self,
        *,
        asset_id: str,
        descriptions: list[dict[str, Any]],
        provider_id: str,
        provider_revision: int,
        provider_fingerprint: str,
        calibrations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized = media_description_list(
            [str(item.get("media_description") or "") for item in descriptions]
        )
        if len(normalized) != len(descriptions):
            raise ValueError("media description payload does not match")
        encoded: list[bytes] = []
        for item in descriptions:
            vector = item.get("vector")
            if vector is None:
                raise ValueError("every media description requires an embedding")
            encoded.append(vector_bytes(vector))
        dimensions = len(encoded[0])
        if any(len(blob) != dimensions for blob in encoded):
            raise ValueError("media description vector dimensions do not match")
        now = time.time()
        db = await self.pool.acquire()
        try:
            await db.execute("BEGIN IMMEDIATE")
            asset = await (
                await db.execute(
                    """SELECT original_name FROM assets
                    WHERE id=? AND kind='image' AND state='active'""",
                    (asset_id,),
                )
            ).fetchone()
            if asset is None:
                raise KeyError(asset_id)
            existing = {
                str(row["normalized_description"]): dict(row)
                for row in await (
                    await db.execute(
                        """SELECT * FROM asset_media_metadata
                        WHERE asset_id=?""",
                        (asset_id,),
                    )
                ).fetchall()
            }
            await db.execute(
                "DELETE FROM asset_media_metadata WHERE asset_id=?", (asset_id,)
            )
            for order, (description, item, blob) in enumerate(
                zip(normalized, descriptions, encoded, strict=True)
            ):
                identity = normalize_media_description(description)
                prior = existing.get(identity)
                await db.execute(
                    """INSERT INTO asset_media_metadata
                    (id,asset_id,media_description,normalized_description,
                     sort_order,search_text,description_source,
                     media_description_vector,vector_sha256,provider_id,
                     provider_revision,provider_fingerprint,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        int(prior["id"]) if prior is not None else None,
                        asset_id,
                        description,
                        identity,
                        order,
                        lexical_text(description),
                        str(item.get("description_source") or "user"),
                        blob,
                        hashlib.sha256(blob).hexdigest(),
                        provider_id,
                        int(provider_revision),
                        provider_fingerprint,
                        float(prior["created_at"]) if prior is not None else now,
                        now,
                    ),
                )
            await self._replace_asset_media_tokens(
                db,
                asset_id=asset_id,
                media_descriptions=normalized,
                original_name=str(asset["original_name"] or ""),
            )
            description_hash = media_description_set_sha256(normalized)
            for calibration in calibrations or []:
                document_id = str(calibration["document_id"])
                relation = await (
                    await db.execute(
                        """SELECT 1 FROM document_assets
                        WHERE document_id=? AND asset_id=?""",
                        (document_id, asset_id),
                    )
                ).fetchone()
                if relation is None:
                    raise ValueError(
                        "media relation changed while descriptions were rebuilt"
                    )
                expected_chunk_ids = {
                    int(row[0])
                    for row in await (
                        await db.execute(
                            """SELECT c.id FROM chunks c
                            JOIN entries e ON e.id=c.entry_id
                            WHERE e.document_id=? AND c.status='active'""",
                            (document_id,),
                        )
                    ).fetchall()
                }
                rows = list(calibration.get("chunks") or [])
                if expected_chunk_ids != {
                    int(item["chunk_id"]) for item in rows
                }:
                    raise ValueError(
                        "media relation chunks changed while descriptions were rebuilt"
                    )
                primary_blob = encoded[0]
                await db.execute(
                    """UPDATE document_assets SET
                    semantic_mode=?,media_description=?,
                    media_description_vector=?,calibration_method=?,
                    calibration_provider_fingerprint=?,
                    calibration_rerank_provider_fingerprint=?,
                    calibration_description_set_sha256=?
                    WHERE document_id=? AND asset_id=?""",
                    (
                        str(calibration["semantic_mode"]),
                        normalized[0],
                        primary_blob,
                        str(calibration["calibration_method"]),
                        provider_fingerprint,
                        str(
                            calibration.get("rerank_provider_fingerprint")
                            or ""
                        ),
                        description_hash,
                        document_id,
                        asset_id,
                    ),
                )
                await db.execute(
                    """DELETE FROM chunk_media_strengths
                    WHERE document_id=? AND asset_id=?""",
                    (document_id, asset_id),
                )
                for item in rows:
                    await db.execute(
                        """INSERT INTO chunk_media_strengths
                        (document_id,chunk_id,asset_id,semantic_strength,
                         rerank_semantic_strength,calibration_similarity,
                         calibration_rank,calibration_method,
                         provider_fingerprint,rerank_provider_fingerprint,
                         calibration_details_json,updated_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            document_id,
                            int(item["chunk_id"]),
                            asset_id,
                            max(
                                0.0,
                                min(1.0, float(item["semantic_strength"])),
                            ),
                            (
                                max(
                                    0.0,
                                    min(
                                        1.0,
                                        float(
                                            item[
                                                "rerank_semantic_strength"
                                            ]
                                        ),
                                    ),
                                )
                                if item.get("rerank_semantic_strength")
                                is not None
                                else None
                            ),
                            item.get("calibration_similarity"),
                            item.get("calibration_rank"),
                            str(calibration["calibration_method"]),
                            provider_fingerprint,
                            str(
                                calibration.get(
                                    "rerank_provider_fingerprint"
                                )
                                or ""
                            ),
                            json.dumps(
                                item.get("calibration_details") or {},
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            now,
                        ),
                    )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        finally:
            await db.close()
        return {
            "asset_id": asset_id,
            "media_description": normalized[0],
            "media_descriptions": normalized,
            "description_count": len(normalized),
            "provider_fingerprint": provider_fingerprint,
        }

    async def active_chunk_records(self) -> list[dict[str, Any]]:
        """Return the canonical active corpus in stable chunk-id order."""
        db = await self.pool.acquire()
        try:
            rows = await (
                await db.execute(
                    """SELECT c.id AS chunk_id,c.text,e.document_id,e.id AS entry_id
                    FROM chunks c JOIN entries e ON e.id=c.entry_id
                    JOIN documents d ON d.id=e.document_id
                    WHERE c.status='active' AND e.status='active' AND d.status='ready'
                    ORDER BY c.id"""
                )
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            await db.close()

    @staticmethod
    async def _clone_active_generation_in_transaction(
        db: Any,
        *,
        now: float,
    ) -> str | None:
        active = await (
            await db.execute(
                """SELECT eg.* FROM active_generation ag
                JOIN embedding_generations eg ON eg.id=ag.generation_id
                WHERE ag.singleton=1"""
            )
        ).fetchone()
        if active is None:
            return None
        generation_id = f"gen-{int(now)}-{uuid.uuid4().hex[:8]}"
        await db.execute(
            """INSERT INTO embedding_generations
            (id,provider_id,provider_revision,provider_fingerprint,dimensions,
             metric,status,created_at,activated_at)
            VALUES(?,?,?,?,?,?,'active',?,?)""",
            (
                generation_id,
                active["provider_id"],
                int(active["provider_revision"]),
                active["provider_fingerprint"],
                int(active["dimensions"]),
                active["metric"],
                now,
                now,
            ),
        )
        await db.execute(
            """INSERT INTO chunk_embeddings
            (generation_id,chunk_id,vector,vector_sha256)
            SELECT ?,ce.chunk_id,ce.vector,ce.vector_sha256
            FROM chunk_embeddings ce JOIN chunks c ON c.id=ce.chunk_id
            JOIN entries e ON e.id=c.entry_id
            JOIN documents d ON d.id=e.document_id
            WHERE ce.generation_id=? AND c.status='active'
              AND e.status='active' AND d.status='ready'""",
            (generation_id, active["id"]),
        )
        await db.execute(
            "UPDATE embedding_generations SET status='superseded' WHERE id=?",
            (active["id"],),
        )
        await db.execute(
            "INSERT OR REPLACE INTO active_generation(singleton,generation_id) VALUES(1,?)",
            (generation_id,),
        )
        return generation_id

    async def replace_all_embeddings(
        self,
        *,
        generation_id: str | None,
        provider_id: str,
        provider_revision: int,
        provider_fingerprint: str,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]] | np.ndarray,
        calibrations: list[dict[str, Any]],
        media_vectors: list[dict[str, Any]],
        dimensions: int | None = None,
    ) -> dict[str, Any]:
        """Atomically activate a complete provider rebuild in this database."""
        if len(chunks) != len(vectors):
            raise ValueError("complete embedding rebuild requires every active chunk")
        chunk_ids = [int(item["chunk_id"]) for item in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("complete embedding rebuild contains duplicate chunks")
        encoded = [vector_bytes(vector) for vector in vectors]
        encoded_media = [
            (
                int(item["media_description_id"]),
                str(item["asset_id"]),
                vector_bytes(item["vector"]),
            )
            for item in media_vectors
        ]
        effective_dimensions = (
            len(encoded[0]) // 4
            if encoded
            else len(encoded_media[0][2]) // 4
            if encoded_media
            else max(0, int(dimensions or 0))
        )
        if any(len(blob) != effective_dimensions * 4 for blob in encoded):
            raise ValueError("Embedding vector dimensions do not match")
        if any(
            len(blob) != effective_dimensions * 4
            for _, _, blob in encoded_media
        ):
            raise ValueError("media vector dimensions do not match")
        now = time.time()
        db = await self.pool.acquire()
        try:
            await db.execute("BEGIN IMMEDIATE")
            expected = {
                int(row[0])
                for row in await (
                    await db.execute(
                        """SELECT c.id FROM chunks c
                        JOIN entries e ON e.id=c.entry_id
                        JOIN documents d ON d.id=e.document_id
                        WHERE c.status='active' AND e.status='active'
                          AND d.status='ready'"""
                    )
                ).fetchall()
            }
            if expected != set(chunk_ids):
                raise ValueError("active chunk set changed during provider rebuild")
            expected_media_rows = {
                (int(row[0]), str(row[1]))
                for row in await (
                    await db.execute(
                        """SELECT amm.id,amm.asset_id FROM asset_media_metadata amm
                        JOIN assets a ON a.id=amm.asset_id
                        WHERE a.kind='image' AND a.state='active'"""
                    )
                ).fetchall()
            }
            if expected_media_rows != {
                (description_id, asset_id)
                for description_id, asset_id, _ in encoded_media
            }:
                raise ValueError(
                    "active media description set changed during provider rebuild"
                )
            previous = await (
                await db.execute(
                    "SELECT generation_id FROM active_generation WHERE singleton=1"
                )
            ).fetchone()
            if generation_id is not None:
                await db.execute(
                    """INSERT INTO embedding_generations
                    (id,provider_id,provider_revision,provider_fingerprint,dimensions,
                     metric,status,created_at,activated_at)
                    VALUES(?,?,?,?,?,'cosine_ip','active',?,?)""",
                    (
                        generation_id,
                        provider_id,
                        int(provider_revision),
                        provider_fingerprint,
                        effective_dimensions,
                        now,
                        now,
                    ),
                )
                for chunk_id, blob in zip(chunk_ids, encoded, strict=True):
                    await db.execute(
                        """INSERT INTO chunk_embeddings
                        (generation_id,chunk_id,vector,vector_sha256)
                        VALUES(?,?,?,?)""",
                        (
                            generation_id,
                            chunk_id,
                            blob,
                            hashlib.sha256(blob).hexdigest(),
                        ),
                    )

            for description_id, asset_id, blob in encoded_media:
                await db.execute(
                    """UPDATE asset_media_metadata SET
                    media_description_vector=?,vector_sha256=?,provider_id=?,
                    provider_revision=?,provider_fingerprint=?,updated_at=?
                    WHERE id=? AND asset_id=?""",
                    (
                        blob,
                        hashlib.sha256(blob).hexdigest(),
                        provider_id,
                        int(provider_revision),
                        provider_fingerprint,
                        now,
                        description_id,
                        asset_id,
                    ),
                )

            for calibration in calibrations:
                document_id = str(calibration["document_id"])
                asset_id = str(calibration["asset_id"])
                rows = list(calibration["chunks"])
                relation = await (
                    await db.execute(
                        """SELECT 1 FROM document_assets
                        WHERE document_id=? AND asset_id=?""",
                        (document_id, asset_id),
                    )
                ).fetchone()
                if relation is None:
                    raise ValueError("media relation changed during provider rebuild")
                relation_chunks = {
                    int(row[0])
                    for row in await (
                        await db.execute(
                            """SELECT c.id FROM chunks c JOIN entries e ON e.id=c.entry_id
                            WHERE e.document_id=? AND c.status='active'""",
                            (document_id,),
                        )
                    ).fetchall()
                }
                if relation_chunks != {int(item["chunk_id"]) for item in rows}:
                    raise ValueError("media calibration chunk set changed during rebuild")
                description_vector = vector_bytes(
                    calibration["media_description_vector"]
                )
                if len(description_vector) != effective_dimensions * 4:
                    raise ValueError("media description vector dimensions do not match")
                await db.execute(
                    """UPDATE document_assets SET media_description=?,
                    media_description_vector=?,
                    calibration_method=?,calibration_provider_fingerprint=?,
                    calibration_rerank_provider_fingerprint=?,
                    calibration_description_set_sha256=?
                    WHERE document_id=? AND asset_id=?""",
                    (
                        str(calibration.get("media_description") or ""),
                        description_vector,
                        calibration["calibration_method"],
                        provider_fingerprint,
                        str(calibration.get("rerank_provider_fingerprint") or ""),
                        str(calibration.get("description_set_sha256") or ""),
                        document_id,
                        asset_id,
                    ),
                )
                await db.execute(
                    "DELETE FROM chunk_media_strengths WHERE document_id=? AND asset_id=?",
                    (document_id, asset_id),
                )
                for item in rows:
                    await db.execute(
                        """INSERT INTO chunk_media_strengths
                        (document_id,chunk_id,asset_id,semantic_strength,
                         rerank_semantic_strength,calibration_similarity,
                         calibration_rank,calibration_method,provider_fingerprint,
                         rerank_provider_fingerprint,calibration_details_json,updated_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            document_id,
                            int(item["chunk_id"]),
                            asset_id,
                            max(0.0, min(1.0, float(item["semantic_strength"]))),
                            (
                                max(
                                    0.0,
                                    min(1.0, float(item["rerank_semantic_strength"])),
                                )
                                if item.get("rerank_semantic_strength") is not None
                                else None
                            ),
                            item.get("calibration_similarity"),
                            item.get("calibration_rank"),
                            calibration["calibration_method"],
                            provider_fingerprint,
                            str(calibration.get("rerank_provider_fingerprint") or ""),
                            json.dumps(
                                item.get("calibration_details") or {},
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            now,
                        ),
                    )

            if previous:
                await db.execute(
                    "UPDATE embedding_generations SET status='superseded' WHERE id=?",
                    (previous["generation_id"],),
                )
            if generation_id is not None:
                await db.execute(
                    "INSERT OR REPLACE INTO active_generation(singleton,generation_id) VALUES(1,?)",
                    (generation_id,),
                )
            else:
                await db.execute("DELETE FROM active_generation WHERE singleton=1")
            await db.execute(
                """UPDATE library_meta SET provider_id=?,provider_revision=?,
                provider_fingerprint=?,status='ready',updated_at=? WHERE singleton=1""",
                (
                    provider_id,
                    int(provider_revision),
                    provider_fingerprint,
                    now,
                ),
            )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        finally:
            await db.close()
        return {
            "generation_id": generation_id,
            "vector_count": len(chunk_ids),
            "media_vector_count": len(encoded_media),
            "dimensions": effective_dimensions,
            "calibration_count": len(calibrations),
        }

    async def document_media_calibration_context(
        self, document_id: str, asset_id: str
    ) -> dict[str, Any]:
        db = await self.pool.acquire()
        try:
            relation = await (
                await db.execute(
                    """SELECT da.*,d.title AS document_title
                    FROM document_assets da JOIN documents d ON d.id=da.document_id
                    WHERE da.document_id=? AND da.asset_id=?""",
                    (document_id, asset_id),
                )
            ).fetchone()
            if relation is None:
                raise KeyError((document_id, asset_id))
            active = await (
                await db.execute(
                    "SELECT generation_id FROM active_generation WHERE singleton=1"
                )
            ).fetchone()
            if active is None:
                raise ValueError("knowledge library has no active embedding generation")
            rows = await (
                await db.execute(
                    """SELECT c.id,c.text,ce.vector FROM chunks c
                    JOIN entries e ON e.id=c.entry_id
                    JOIN chunk_embeddings ce ON ce.chunk_id=c.id
                    WHERE e.document_id=? AND c.status='active'
                      AND ce.generation_id=? ORDER BY c.id""",
                    (document_id, active["generation_id"]),
                )
            ).fetchall()
            if not rows:
                raise ValueError("document has no active embedded chunks")
            return {
                "relation": dict(relation),
                "generation_id": str(active["generation_id"]),
                "chunks": [
                    {
                        "chunk_id": int(row["id"]),
                        "text": str(row["text"]),
                        "vector": np.frombuffer(row["vector"], dtype="<f4").copy(),
                    }
                    for row in rows
                ],
            }
        finally:
            await db.close()

    async def replace_document_media_calibration(
        self,
        *,
        document_id: str,
        asset_id: str,
        semantic_mode: str,
        media_description: str,
        calibration_method: str,
        provider_fingerprint: str,
        rerank_provider_fingerprint: str = "",
        description_set_sha256: str = "",
        chunks: list[dict[str, Any]],
        media_description_vector: list[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        if semantic_mode not in {"uniform", "calibrated"}:
            raise ValueError("invalid semantic media calibration mode")
        encoded_description_vector = (
            vector_bytes(media_description_vector)
            if media_description_vector is not None
            else None
        )
        now = time.time()
        db = await self.pool.acquire()
        try:
            await db.execute("BEGIN IMMEDIATE")
            relation = await (
                await db.execute(
                    "SELECT 1 FROM document_assets WHERE document_id=? AND asset_id=?",
                    (document_id, asset_id),
                )
            ).fetchone()
            if relation is None:
                raise KeyError((document_id, asset_id))
            if not description_set_sha256:
                description_rows = await (
                    await db.execute(
                        """SELECT media_description FROM asset_media_metadata
                        WHERE asset_id=? ORDER BY sort_order,id""",
                        (asset_id,),
                    )
                ).fetchall()
                description_set_sha256 = media_description_set_sha256(
                    [str(row["media_description"]) for row in description_rows]
                )
            expected = {
                int(row[0])
                for row in await (
                    await db.execute(
                        """SELECT c.id FROM chunks c JOIN entries e ON e.id=c.entry_id
                        WHERE e.document_id=? AND c.status='active'""",
                        (document_id,),
                    )
                ).fetchall()
            }
            supplied = {int(item["chunk_id"]) for item in chunks}
            if not expected or supplied != expected or len(supplied) != len(chunks):
                raise ValueError("media calibration chunk set is stale")
            if encoded_description_vector is not None:
                active_vector = await (
                    await db.execute(
                        """SELECT LENGTH(ce.vector) AS size_bytes
                        FROM active_generation ag
                        JOIN chunk_embeddings ce
                          ON ce.generation_id=ag.generation_id
                        LIMIT 1"""
                    )
                ).fetchone()
                if (
                    active_vector is None
                    or int(active_vector["size_bytes"])
                    != len(encoded_description_vector)
                ):
                    raise ValueError(
                        "media description and chunk dimensions do not match"
                    )
            await db.execute(
                """UPDATE document_assets SET semantic_mode=?,media_description=?,
                media_description_vector=?,calibration_method=?,
                calibration_provider_fingerprint=?,
                calibration_rerank_provider_fingerprint=?,
                calibration_description_set_sha256=?
                WHERE document_id=? AND asset_id=?""",
                (
                    semantic_mode,
                    media_description,
                    encoded_description_vector,
                    calibration_method,
                    provider_fingerprint,
                    rerank_provider_fingerprint,
                    description_set_sha256,
                    document_id,
                    asset_id,
                ),
            )
            await db.execute(
                "DELETE FROM chunk_media_strengths WHERE document_id=? AND asset_id=?",
                (document_id, asset_id),
            )
            for item in chunks:
                await db.execute(
                    """INSERT INTO chunk_media_strengths
                    (document_id,chunk_id,asset_id,semantic_strength,
                     rerank_semantic_strength,calibration_similarity,
                     calibration_rank,calibration_method,provider_fingerprint,
                     rerank_provider_fingerprint,calibration_details_json,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        document_id,
                        int(item["chunk_id"]),
                        asset_id,
                        max(0.0, min(1.0, float(item["semantic_strength"]))),
                        (
                            max(
                                0.0,
                                min(
                                    1.0,
                                    float(item["rerank_semantic_strength"]),
                                ),
                            )
                            if item.get("rerank_semantic_strength") is not None
                            else None
                        ),
                        item.get("calibration_similarity"),
                        item.get("calibration_rank"),
                        calibration_method,
                        provider_fingerprint,
                        rerank_provider_fingerprint,
                        json.dumps(
                            item.get("calibration_details") or {},
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        now,
                    ),
                )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        finally:
            await db.close()
        strengths = [float(item["semantic_strength"]) for item in chunks]
        rerank_strengths = [
            float(item["rerank_semantic_strength"])
            for item in chunks
            if item.get("rerank_semantic_strength") is not None
        ]
        rerank_strength_deltas = [
            abs(
                float(item["rerank_semantic_strength"])
                - float(item["semantic_strength"])
            )
            for item in chunks
            if item.get("rerank_semantic_strength") is not None
        ]
        return {
            "document_id": document_id,
            "asset_id": asset_id,
            "semantic_mode": semantic_mode,
            "media_description": media_description,
            "calibration_method": calibration_method,
            "chunk_count": len(chunks),
            "minimum_strength": min(strengths),
            "maximum_strength": max(strengths),
            "rerank_provider_fingerprint": rerank_provider_fingerprint,
            "rerank_calibrated_chunk_count": len(rerank_strengths),
            "minimum_rerank_strength": (
                min(rerank_strengths) if rerank_strengths else None
            ),
            "maximum_rerank_strength": (
                max(rerank_strengths) if rerank_strengths else None
            ),
            "rerank_changed_chunk_count": sum(
                delta > 1e-12 for delta in rerank_strength_deltas
            ),
            "maximum_rerank_strength_delta": max(
                rerank_strength_deltas, default=0.0
            ),
        }

    async def list_document_media_calibrations(self) -> list[dict[str, Any]]:
        db = await self.pool.acquire()
        try:
            rows = await (
                await db.execute(
                    """SELECT da.document_id,da.asset_id,d.title AS document_title,
                    a.original_name,da.semantic_mode,da.media_description,
                    da.calibration_method,da.calibration_provider_fingerprint,
                    da.calibration_rerank_provider_fingerprint,
                    GROUP_CONCAT(DISTINCT CASE
                      WHEN cms.rerank_semantic_strength IS NOT NULL
                       AND json_valid(cms.calibration_details_json)
                      THEN json_extract(
                        cms.calibration_details_json,
                        '$.rerank_settings_fingerprint'
                      )
                      ELSE NULL END
                    ) AS calibration_rerank_settings_fingerprints,
                    COUNT(cms.chunk_id) AS calibrated_chunk_count,
                    COUNT(cms.rerank_semantic_strength)
                      AS rerank_calibrated_chunk_count,
                    COALESCE(MIN(cms.semantic_strength),lm.uniform_media_strength) AS minimum_strength,
                    COALESCE(MAX(cms.semantic_strength),lm.uniform_media_strength) AS maximum_strength,
                    MIN(cms.rerank_semantic_strength) AS minimum_rerank_strength,
                    MAX(cms.rerank_semantic_strength) AS maximum_rerank_strength,
                    SUM(CASE
                      WHEN cms.rerank_semantic_strength IS NOT NULL
                       AND ABS(
                         cms.rerank_semantic_strength-cms.semantic_strength
                       ) > 1e-12
                      THEN 1 ELSE 0 END
                    ) AS rerank_changed_chunk_count,
                    COALESCE(MAX(CASE
                      WHEN cms.rerank_semantic_strength IS NOT NULL
                      THEN ABS(
                        cms.rerank_semantic_strength-cms.semantic_strength
                      )
                      ELSE 0 END
                    ),0) AS maximum_rerank_strength_delta
                    FROM document_assets da JOIN documents d ON d.id=da.document_id
                    JOIN assets a ON a.id=da.asset_id
                    CROSS JOIN library_meta lm
                    LEFT JOIN chunk_media_strengths cms
                      ON cms.document_id=da.document_id AND cms.asset_id=da.asset_id
                    WHERE a.kind='image' AND a.state='active'
                    GROUP BY da.document_id,da.asset_id
                    ORDER BY d.created_at DESC,da.sort_order,da.asset_id"""
                )
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            await db.close()

    async def lexical_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        terms = fts_query_text(query)
        if not terms:
            return []
        db = await self.pool.acquire()
        try:
            rows = await (
                await db.execute(
                    """SELECT c.id,bm25(chunks_fts) AS score FROM chunks_fts
                    JOIN chunks c ON c.id=chunks_fts.rowid
                    WHERE chunks_fts MATCH ? AND c.status='active'
                    ORDER BY score,c.id LIMIT ?""",
                    (terms, int(limit)),
                )
            ).fetchall()
            return normalize_bm25_rows(
                (int(row["id"]), float(row["score"])) for row in rows
            )
        finally:
            await db.close()

    async def media_lexical_search(self, query: str) -> list[tuple[int, float]]:
        """Search only chunks that have at least one effective media binding."""
        terms = fts_query_text(query)
        if not terms:
            return []
        db = await self.pool.acquire()
        try:
            rows = await (
                await db.execute(
                    """SELECT c.id,bm25(chunks_fts) AS score FROM chunks_fts
                    JOIN chunks c ON c.id=chunks_fts.rowid
                    JOIN entries e ON e.id=c.entry_id
                    JOIN chunk_embeddings ce ON ce.chunk_id=c.id
                    JOIN active_generation ag ON ag.generation_id=ce.generation_id
                    WHERE chunks_fts MATCH ? AND c.status='active'
                      AND (
                        EXISTS (
                          SELECT 1 FROM chunk_assets ca JOIN assets a ON a.id=ca.asset_id
                          WHERE ca.chunk_id=c.id AND ca.output_policy<>'disabled'
                            AND a.kind='image' AND a.state='active'
                        ) OR EXISTS (
                          SELECT 1 FROM entry_assets ea JOIN assets a ON a.id=ea.asset_id
                          WHERE ea.entry_id=c.entry_id AND ea.output_policy<>'disabled'
                            AND a.kind='image' AND a.state='active'
                        ) OR EXISTS (
                          SELECT 1 FROM document_assets da JOIN assets a ON a.id=da.asset_id
                          WHERE da.document_id=e.document_id AND da.output_policy<>'disabled'
                            AND a.kind='image' AND a.state='active'
                        )
                      )
                    ORDER BY score,c.id""",
                    (terms,),
                )
            ).fetchall()
            return normalize_bm25_rows(
                (int(row["id"]), float(row["score"])) for row in rows
            )
        finally:
            await db.close()

    async def media_bindings(self) -> dict[int, list[dict[str, Any]]]:
        """Expand every effective media relation to its active embedded chunks."""
        db = await self.pool.acquire()
        try:
            rows = await (
                await db.execute(
                    """SELECT c.id AS result_chunk_id,'chunk' AS scope,
                    ca.chunk_id AS relation_target_id,ca.asset_id,ca.role,ca.relation_weight,
                    ca.caption,ca.alt_text,ca.output_policy,ca.sort_order,
                    COALESCE(cms.semantic_strength,lm.uniform_media_strength) AS semantic_strength,
                    cms.rerank_semantic_strength,
                    cms.calibration_similarity,cms.calibration_rank,
                    COALESCE(cms.calibration_method,'uniform_v1') AS calibration_method,
                    COALESCE(cms.provider_fingerprint,'') AS strength_provider_fingerprint,
                    COALESCE(cms.rerank_provider_fingerprint,'') AS strength_rerank_provider_fingerprint,
                    COALESCE(cms.calibration_details_json,'{}') AS calibration_details_json,
                    a.sha256,a.storage_key,a.mime_type,a.size_bytes,a.width,a.height,a.original_name,
                    '' AS media_description,NULL AS media_description_vector,
                    '' AS relation_calibration_rerank_fingerprint,
                    '' AS relation_calibration_description_set_sha256
                    FROM chunks c JOIN chunk_assets ca ON ca.chunk_id=c.id
                    JOIN assets a ON a.id=ca.asset_id
                    JOIN chunk_embeddings ce ON ce.chunk_id=c.id
                    JOIN active_generation ag ON ag.generation_id=ce.generation_id
                    CROSS JOIN library_meta lm
                    LEFT JOIN chunk_media_strengths cms
                      ON cms.chunk_id=c.id AND cms.asset_id=ca.asset_id
                    WHERE c.status='active' AND ca.output_policy<>'disabled'
                      AND a.kind='image' AND a.state='active'
                    UNION ALL
                    SELECT c.id,'entry',ea.entry_id,ea.asset_id,ea.role,ea.relation_weight,
                    ea.caption,ea.alt_text,ea.output_policy,ea.sort_order,
                    COALESCE(cms.semantic_strength,lm.uniform_media_strength),
                    cms.rerank_semantic_strength,cms.calibration_similarity,
                    cms.calibration_rank,
                    COALESCE(cms.calibration_method,'uniform_v1'),
                    COALESCE(cms.provider_fingerprint,''),
                    COALESCE(cms.rerank_provider_fingerprint,''),
                    COALESCE(cms.calibration_details_json,'{}'),
                    a.sha256,a.storage_key,a.mime_type,a.size_bytes,a.width,a.height,a.original_name,
                    '' AS media_description,NULL AS media_description_vector,
                    '' AS relation_calibration_rerank_fingerprint,
                    '' AS relation_calibration_description_set_sha256
                    FROM chunks c JOIN entry_assets ea ON ea.entry_id=c.entry_id
                    JOIN assets a ON a.id=ea.asset_id
                    JOIN chunk_embeddings ce ON ce.chunk_id=c.id
                    JOIN active_generation ag ON ag.generation_id=ce.generation_id
                    CROSS JOIN library_meta lm
                    LEFT JOIN chunk_media_strengths cms
                      ON cms.chunk_id=c.id AND cms.asset_id=ea.asset_id
                    WHERE c.status='active' AND ea.output_policy<>'disabled'
                      AND a.kind='image' AND a.state='active'
                    UNION ALL
                    SELECT c.id,'document',da.document_id,da.asset_id,da.role,da.relation_weight,
                    da.caption,da.alt_text,da.output_policy,da.sort_order,
                    COALESCE(cms.semantic_strength,lm.uniform_media_strength),
                    cms.rerank_semantic_strength,cms.calibration_similarity,
                    cms.calibration_rank,
                    COALESCE(cms.calibration_method,'uniform_v1'),
                    COALESCE(cms.provider_fingerprint,''),
                    COALESCE(cms.rerank_provider_fingerprint,''),
                    COALESCE(cms.calibration_details_json,'{}'),
                    a.sha256,a.storage_key,a.mime_type,a.size_bytes,a.width,a.height,a.original_name,
                    da.media_description,da.media_description_vector,
                    da.calibration_rerank_provider_fingerprint,
                    da.calibration_description_set_sha256
                    FROM chunks c JOIN entries e ON e.id=c.entry_id
                    JOIN document_assets da ON da.document_id=e.document_id
                    JOIN assets a ON a.id=da.asset_id
                    JOIN chunk_embeddings ce ON ce.chunk_id=c.id
                    JOIN active_generation ag ON ag.generation_id=ce.generation_id
                    CROSS JOIN library_meta lm
                    LEFT JOIN chunk_media_strengths cms
                      ON cms.chunk_id=c.id AND cms.asset_id=da.asset_id
                    WHERE c.status='active' AND da.output_policy<>'disabled'
                      AND a.kind='image' AND a.state='active'"""
                )
            ).fetchall()
            result: dict[int, list[dict[str, Any]]] = {}
            scope_rank = {"chunk": 0, "entry": 1, "document": 2}
            for row in rows:
                item = dict(row)
                chunk_id = int(item["result_chunk_id"])
                blob = item.get("media_description_vector")
                item["media_description_vector"] = (
                    np.frombuffer(blob, dtype="<f4").copy()
                    if blob is not None
                    else None
                )
                try:
                    item["calibration_details"] = json.loads(
                        str(item.pop("calibration_details_json") or "{}")
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    item["calibration_details"] = {}
                result.setdefault(chunk_id, []).append(item)
            for items in result.values():
                items.sort(
                    key=lambda value: (
                        scope_rank[str(value["scope"])],
                        int(value["sort_order"]),
                        str(value["asset_id"]),
                    )
                )
            return result
        finally:
            await db.close()

    async def chunk_rows(self, chunk_ids: Iterable[int]) -> list[dict[str, Any]]:
        ids = sorted({int(value) for value in chunk_ids})
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        db = await self.pool.acquire()
        try:
            rows = await (
                await db.execute(
                    f"""SELECT c.*,e.document_id,e.title AS entry_title,e.body AS entry_body,
                    d.title AS document_title FROM chunks c JOIN entries e ON e.id=c.entry_id
                    LEFT JOIN documents d ON d.id=e.document_id WHERE c.id IN ({placeholders})""",
                    tuple(ids),
                )
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            await db.close()

    async def link_asset(
        self,
        *,
        scope: str,
        target_id: str | int,
        asset_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if scope not in RELATION_SCOPES:
            raise ValueError("invalid relation scope")
        policy = str(payload.get("output_policy") or "auto")
        if policy not in OUTPUT_POLICIES:
            raise ValueError("invalid output policy")
        table = f"{scope}_assets"
        key = f"{scope}_id"
        db = await self.pool.acquire()
        try:
            asset = await (await db.execute("SELECT kind,state FROM assets WHERE id=?", (asset_id,))).fetchone()
            if not asset or asset["kind"] != "image" or asset["state"] != "active":
                raise KeyError(asset_id)
            await db.execute(
                f"""INSERT INTO {table}
                ({key},asset_id,role,relation_weight,caption,alt_text,output_policy,sort_order)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT({key},asset_id) DO UPDATE SET
                role=excluded.role,relation_weight=excluded.relation_weight,
                caption=excluded.caption,alt_text=excluded.alt_text,
                output_policy=excluded.output_policy,sort_order=excluded.sort_order""",
                (
                    target_id,
                    asset_id,
                    str(payload.get("role") or "illustration"),
                    max(0.0, min(1.0, float(payload.get("relation_weight", 1.0)))),
                    str(payload.get("caption") or ""),
                    str(payload.get("alt_text") or ""),
                    policy,
                    int(payload.get("sort_order") or 0),
                ),
            )
            await db.commit()
        finally:
            await db.close()
        return {"scope": scope, "target_id": target_id, "asset_id": asset_id, "output_policy": policy}

    async def attachments_for_chunks(self, chunk_ids: Iterable[int]) -> dict[int, list[dict[str, Any]]]:
        ids = sorted({int(value) for value in chunk_ids})
        result = {value: [] for value in ids}
        if not ids:
            return result
        placeholders = ",".join("?" for _ in ids)
        db = await self.pool.acquire()
        try:
            rows = await (
                await db.execute(
                    f"""SELECT c.id AS result_chunk_id,'chunk' AS scope,
                    ca.chunk_id AS relation_target_id,ca.asset_id,ca.role,ca.relation_weight,
                    ca.caption,ca.alt_text,ca.output_policy,ca.sort_order,
                    COALESCE(cms.semantic_strength,lm.uniform_media_strength) AS semantic_strength,
                    cms.calibration_similarity,cms.calibration_rank,
                    COALESCE(cms.calibration_method,'uniform_v1') AS calibration_method,
                    a.sha256,a.storage_key,a.mime_type,a.size_bytes,a.width,a.height,a.original_name,
                    '' AS media_description,NULL AS media_description_vector
                    FROM chunks c JOIN chunk_assets ca ON ca.chunk_id=c.id
                    JOIN assets a ON a.id=ca.asset_id
                    CROSS JOIN library_meta lm
                    LEFT JOIN chunk_media_strengths cms
                      ON cms.chunk_id=c.id AND cms.asset_id=ca.asset_id
                    WHERE c.id IN ({placeholders}) AND a.state='active'
                    UNION ALL
                    SELECT c.id,'entry',ea.entry_id,ea.asset_id,ea.role,ea.relation_weight,
                    ea.caption,ea.alt_text,ea.output_policy,ea.sort_order,
                    COALESCE(cms.semantic_strength,lm.uniform_media_strength),cms.calibration_similarity,
                    cms.calibration_rank,COALESCE(cms.calibration_method,'uniform_v1'),
                    a.sha256,a.storage_key,a.mime_type,a.size_bytes,a.width,a.height,a.original_name,
                    '' AS media_description,NULL AS media_description_vector
                    FROM chunks c JOIN entry_assets ea ON ea.entry_id=c.entry_id
                    JOIN assets a ON a.id=ea.asset_id
                    CROSS JOIN library_meta lm
                    LEFT JOIN chunk_media_strengths cms
                      ON cms.chunk_id=c.id AND cms.asset_id=ea.asset_id
                    WHERE c.id IN ({placeholders}) AND a.state='active'
                    UNION ALL
                    SELECT c.id,'document',da.document_id,da.asset_id,da.role,da.relation_weight,
                    da.caption,da.alt_text,da.output_policy,da.sort_order,
                    COALESCE(cms.semantic_strength,lm.uniform_media_strength),cms.calibration_similarity,
                    cms.calibration_rank,COALESCE(cms.calibration_method,'uniform_v1'),
                    a.sha256,a.storage_key,a.mime_type,a.size_bytes,a.width,a.height,a.original_name,
                    da.media_description,da.media_description_vector
                    FROM chunks c JOIN entries e ON e.id=c.entry_id
                    JOIN document_assets da ON da.document_id=e.document_id JOIN assets a ON a.id=da.asset_id
                    CROSS JOIN library_meta lm
                    LEFT JOIN chunk_media_strengths cms
                      ON cms.chunk_id=c.id AND cms.asset_id=da.asset_id
                    WHERE c.id IN ({placeholders}) AND a.state='active'""",
                    tuple(ids + ids + ids),
                )
            ).fetchall()
            rank = {"chunk": 0, "entry": 1, "document": 2}
            for chunk_id in ids:
                candidates = [dict(row) for row in rows if int(row["result_chunk_id"]) == chunk_id]
                result[chunk_id] = sorted(
                    candidates,
                    key=lambda value: (
                        rank[value["scope"]],
                        int(value["sort_order"]),
                        value["asset_id"],
                    ),
                )
                for item in result[chunk_id]:
                    blob = item.get("media_description_vector")
                    item["media_description_vector"] = (
                        np.frombuffer(blob, dtype="<f4").copy()
                        if blob is not None
                        else None
                    )
            return result
        finally:
            await db.close()

    async def list_documents(self) -> list[dict[str, Any]]:
        db = await self.pool.acquire()
        try:
            rows = await (await db.execute("SELECT * FROM documents ORDER BY created_at DESC,id")).fetchall()
            return [dict(row) for row in rows]
        finally:
            await db.close()

    async def list_document_summaries(
        self,
        *,
        query: str = "",
        offset: int = 0,
        limit: int = 20,
        sort: str = "created_desc",
    ) -> dict[str, Any]:
        normalized_query = query.strip()
        where = ""
        parameters: list[Any] = []
        if normalized_query:
            where = "WHERE d.title LIKE ? ESCAPE '\\' OR a.original_name LIKE ? ESCAPE '\\'"
            escaped = normalized_query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped}%"
            parameters.extend([pattern, pattern])
        order_by = {
            "created_asc": "d.created_at ASC,d.id ASC",
            "title_asc": "d.title COLLATE NOCASE ASC,d.id ASC",
            "title_desc": "d.title COLLATE NOCASE DESC,d.id ASC",
        }.get(sort, "d.created_at DESC,d.id DESC")
        db = await self.pool.acquire()
        try:
            total_row = await (
                await db.execute(
                    f"""SELECT COUNT(*) AS value FROM documents d
                    JOIN assets a ON a.id=d.source_asset_id {where}""",
                    tuple(parameters),
                )
            ).fetchone()
            rows = await (
                await db.execute(
                    f"""SELECT d.*,a.original_name,a.mime_type,a.size_bytes,
                    COUNT(DISTINCT c.id) AS chunk_count,
                    COUNT(DISTINCT da.asset_id) AS image_count
                    FROM documents d
                    JOIN assets a ON a.id=d.source_asset_id
                    LEFT JOIN entries e ON e.document_id=d.id
                    LEFT JOIN chunks c ON c.entry_id=e.id
                    LEFT JOIN document_assets da ON da.document_id=d.id
                    {where}
                    GROUP BY d.id
                    ORDER BY {order_by}
                    LIMIT ? OFFSET ?""",
                    tuple([*parameters, int(limit), int(offset)]),
                )
            ).fetchall()
            return {
                "items": [dict(row) for row in rows],
                "total": int(total_row["value"]),
                "offset": int(offset),
                "limit": int(limit),
            }
        finally:
            await db.close()

    async def get_document_detail(self, document_id: str) -> dict[str, Any] | None:
        db = await self.pool.acquire()
        try:
            row = await (
                await db.execute(
                    """SELECT d.*,a.original_name,a.mime_type,a.size_bytes,a.sha256 AS source_sha256,
                    COUNT(DISTINCT c.id) AS chunk_count
                    FROM documents d JOIN assets a ON a.id=d.source_asset_id
                    LEFT JOIN entries e ON e.document_id=d.id
                    LEFT JOIN chunks c ON c.entry_id=e.id
                    WHERE d.id=? GROUP BY d.id""",
                    (document_id,),
                )
            ).fetchone()
            if row is None:
                return None
            entries = await (
                await db.execute(
                    "SELECT id,title,body,ordinal,status,created_at,updated_at FROM entries WHERE document_id=? ORDER BY ordinal,id",
                    (document_id,),
                )
            ).fetchall()
            media = await (
                await db.execute(
                    """SELECT a.id AS asset_id,a.original_name,a.mime_type,a.size_bytes,a.width,a.height,
                    da.role,da.relation_weight,da.caption,da.alt_text,da.output_policy,da.sort_order,
                    da.media_description,da.semantic_mode,da.calibration_method,
                    COUNT(cms.chunk_id) AS calibrated_chunk_count,
                    MIN(cms.semantic_strength) AS minimum_strength,
                    MAX(cms.semantic_strength) AS maximum_strength
                    FROM document_assets da JOIN assets a ON a.id=da.asset_id
                    LEFT JOIN chunk_media_strengths cms
                      ON cms.document_id=da.document_id AND cms.asset_id=da.asset_id
                    WHERE da.document_id=? GROUP BY da.document_id,da.asset_id
                    ORDER BY da.sort_order,a.original_name,a.id""",
                    (document_id,),
                )
            ).fetchall()
            item = dict(row)
            item["entries"] = [dict(entry) for entry in entries]
            item["content"] = "\n\n".join(str(entry["body"]) for entry in entries)
            item["associated_media"] = [dict(asset) for asset in media]
            item["image_count"] = len(media)
            return item
        finally:
            await db.close()

    async def list_chunk_summaries(
        self,
        *,
        query: str = "",
        document_id: str = "",
        offset: int = 0,
        limit: int = 20,
        sort: str = "ordinal_asc",
    ) -> dict[str, Any]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if query.strip():
            clauses.append("(c.text LIKE ? ESCAPE '\\' OR e.title LIKE ? ESCAPE '\\' OR d.title LIKE ? ESCAPE '\\')")
            escaped = query.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped}%"
            parameters.extend([pattern, pattern, pattern])
        if document_id.strip():
            clauses.append("d.id=?")
            parameters.append(document_id.strip())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_by = {
            "ordinal_desc": "d.created_at DESC,e.ordinal DESC,c.ordinal DESC,c.id DESC",
            "id_desc": "c.id DESC",
        }.get(sort, "d.created_at DESC,e.ordinal ASC,c.ordinal ASC,c.id ASC")
        db = await self.pool.acquire()
        try:
            total_row = await (
                await db.execute(
                    f"""SELECT COUNT(*) AS value FROM chunks c
                    JOIN entries e ON e.id=c.entry_id
                    JOIN documents d ON d.id=e.document_id {where}""",
                    tuple(parameters),
                )
            ).fetchone()
            rows = await (
                await db.execute(
                    f"""SELECT c.id,c.entry_id,c.ordinal,c.text,c.char_start,c.char_end,
                    c.content_sha256,c.status,e.title AS entry_title,e.document_id,
                    d.title AS document_title,d.created_at AS document_created_at
                    FROM chunks c JOIN entries e ON e.id=c.entry_id
                    JOIN documents d ON d.id=e.document_id
                    {where} ORDER BY {order_by} LIMIT ? OFFSET ?""",
                    tuple([*parameters, int(limit), int(offset)]),
                )
            ).fetchall()
        finally:
            await db.close()
        items = [dict(row) for row in rows]
        attachments = await self.attachments_for_chunks([int(item["id"]) for item in items])
        for item in items:
            item["char_count"] = len(str(item.get("text") or ""))
            item["media_count"] = len(attachments.get(int(item["id"]), []))
        return {
            "items": items,
            "total": int(total_row["value"]),
            "offset": int(offset),
            "limit": int(limit),
        }

    async def get_chunk_detail(self, chunk_id: int) -> dict[str, Any] | None:
        db = await self.pool.acquire()
        try:
            row = await (
                await db.execute(
                    """SELECT c.*,e.title AS entry_title,e.document_id,d.title AS document_title,
                    d.parser_id,d.created_at AS document_created_at
                    FROM chunks c JOIN entries e ON e.id=c.entry_id
                    JOIN documents d ON d.id=e.document_id WHERE c.id=?""",
                    (int(chunk_id),),
                )
            ).fetchone()
        finally:
            await db.close()
        if row is None:
            return None
        item = dict(row)
        item["char_count"] = len(str(item.get("text") or ""))
        attachments = (
            await self.attachments_for_chunks([int(chunk_id)])
        ).get(int(chunk_id), [])
        public_media_fields = (
            "asset_id",
            "scope",
            "relation_target_id",
            "role",
            "relation_weight",
            "caption",
            "alt_text",
            "output_policy",
            "sort_order",
            "semantic_strength",
            "calibration_similarity",
            "calibration_rank",
            "calibration_method",
            "sha256",
            "mime_type",
            "size_bytes",
            "width",
            "height",
            "original_name",
            "media_description",
        )
        item["associated_media"] = [
            {key: attachment.get(key) for key in public_media_fields}
            for attachment in attachments
        ]
        return item

    async def list_entries(self) -> list[dict[str, Any]]:
        db = await self.pool.acquire()
        try:
            rows = await (
                await db.execute(
                    """SELECT e.*,COUNT(c.id) AS chunk_count FROM entries e
                    LEFT JOIN chunks c ON c.entry_id=e.id GROUP BY e.id ORDER BY e.created_at DESC,e.id"""
                )
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            await db.close()

    async def list_assets(self, *, kind: str | None = None) -> list[dict[str, Any]]:
        db = await self.pool.acquire()
        try:
            if kind == "image":
                rows = await (
                    await db.execute(
                        """SELECT a.*,amm.media_description,
                        amm.description_source,amm.provider_id AS media_provider_id,
                        amm.provider_revision AS media_provider_revision,
                        amm.provider_fingerprint AS media_provider_fingerprint,
                        (SELECT COUNT(*) FROM asset_media_metadata descriptions
                         WHERE descriptions.asset_id=a.id) AS media_description_count,
                        CASE
                          WHEN amm.media_description_vector IS NULL THEN 'lexical_only'
                          WHEN amm.provider_fingerprint<>lm.provider_fingerprint
                            THEN 'stale'
                          ELSE 'ready'
                        END AS media_vector_status
                        FROM assets a LEFT JOIN asset_media_metadata amm
                          ON amm.asset_id=a.id AND amm.sort_order=0
                        CROSS JOIN library_meta lm
                        WHERE a.kind='image' AND a.state='active'
                        ORDER BY a.created_at DESC,a.id"""
                    )
                ).fetchall()
            elif kind:
                rows = await (await db.execute("SELECT * FROM assets WHERE kind=? AND state='active' ORDER BY created_at DESC,id", (kind,))).fetchall()
            else:
                rows = await (await db.execute("SELECT * FROM assets WHERE state='active' ORDER BY created_at DESC,id")).fetchall()
            return [dict(row) for row in rows]
        finally:
            await db.close()

    async def get_asset(self, asset_id: str) -> dict[str, Any] | None:
        db = await self.pool.acquire()
        try:
            row = await (
                await db.execute(
                    "SELECT * FROM assets WHERE id=? AND state='active'",
                    (asset_id,),
                )
            ).fetchone()
            return dict(row) if row else None
        finally:
            await db.close()

    async def get_asset_detail(self, asset_id: str) -> dict[str, Any] | None:
        asset = await self.get_asset(asset_id)
        if asset is None:
            return None
        db = await self.pool.acquire()
        try:
            rows = await (
                await db.execute(
                    """SELECT 'document' AS scope,da.document_id AS target_id,d.title AS target_title,
                    da.role,da.relation_weight,da.caption,da.alt_text,da.output_policy,da.sort_order,
                    da.media_description,da.semantic_mode,da.calibration_method,
                    COUNT(cms.chunk_id) AS calibrated_chunk_count,
                    MIN(cms.semantic_strength) AS minimum_strength,
                    MAX(cms.semantic_strength) AS maximum_strength
                    FROM document_assets da JOIN documents d ON d.id=da.document_id
                    LEFT JOIN chunk_media_strengths cms
                      ON cms.document_id=da.document_id AND cms.asset_id=da.asset_id
                    WHERE da.asset_id=? GROUP BY da.document_id
                    UNION ALL
                    SELECT 'entry',ea.entry_id,e.title,ea.role,ea.relation_weight,ea.caption,ea.alt_text,
                    ea.output_policy,ea.sort_order,'','uniform','uniform_v1',0,NULL,NULL
                    FROM entry_assets ea JOIN entries e ON e.id=ea.entry_id WHERE ea.asset_id=?
                    UNION ALL
                    SELECT 'chunk',CAST(ca.chunk_id AS TEXT),substr(c.text,1,120),ca.role,ca.relation_weight,
                    ca.caption,ca.alt_text,ca.output_policy,ca.sort_order,'','uniform','uniform_v1',0,NULL,NULL
                    FROM chunk_assets ca JOIN chunks c ON c.id=ca.chunk_id WHERE ca.asset_id=?
                    ORDER BY scope,sort_order,target_title,target_id""",
                    (asset_id, asset_id, asset_id),
                )
            ).fetchall()
            item = dict(asset)
            metadata_rows = await (
                await db.execute(
                    """SELECT amm.id,amm.media_description,amm.normalized_description,
                    amm.sort_order,amm.search_text,
                    amm.description_source,amm.provider_id,amm.provider_revision,
                    amm.provider_fingerprint,amm.vector_sha256,amm.updated_at,
                    CASE
                      WHEN amm.media_description_vector IS NULL THEN 'lexical_only'
                      WHEN amm.provider_fingerprint<>lm.provider_fingerprint THEN 'stale'
                      ELSE 'ready'
                    END AS vector_status
                    FROM asset_media_metadata amm CROSS JOIN library_meta lm
                    WHERE amm.asset_id=? ORDER BY amm.sort_order,amm.id""",
                    (asset_id,),
                )
            ).fetchall()
            descriptions = [dict(row) for row in metadata_rows]
            item["media_descriptions"] = descriptions
            item["media_description"] = (
                str(descriptions[0]["media_description"]) if descriptions else ""
            )
            item["media_description_count"] = len(descriptions)
            item["media_metadata"] = descriptions[0] if descriptions else None
            item["relations"] = [dict(row) for row in rows]
            item["relation_count"] = len(rows)
            return item
        finally:
            await db.close()

    async def delete_documents(self, document_ids: Iterable[str]) -> dict[str, Any]:
        ids = list(dict.fromkeys(str(value) for value in document_ids if str(value)))
        if not ids:
            raise ValueError("at least one document is required")
        placeholders = ",".join("?" for _ in ids)
        db = await self.pool.acquire()
        try:
            await db.execute("BEGIN IMMEDIATE")
            rows = await (
                await db.execute(
                    f"""SELECT d.id AS document_id,a.* FROM documents d JOIN assets a
                    ON a.id=d.source_asset_id WHERE d.id IN ({placeholders})""",
                    tuple(ids),
                )
            ).fetchall()
            found = {str(row["document_id"]) for row in rows}
            missing = next((value for value in ids if value not in found), None)
            if missing is not None:
                raise KeyError(missing)
            await db.execute(
                f"DELETE FROM documents WHERE id IN ({placeholders})",
                tuple(ids),
            )
            removed_assets: list[dict[str, Any]] = []
            seen_assets: set[str] = set()
            for row in rows:
                asset_id = str(row["id"])
                if asset_id in seen_assets:
                    continue
                seen_assets.add(asset_id)
                refs = await (
                    await db.execute(
                        "SELECT COUNT(*) AS value FROM documents WHERE source_asset_id=?",
                        (asset_id,),
                    )
                ).fetchone()
                if int(refs["value"]) == 0:
                    await db.execute("DELETE FROM assets WHERE id=?", (asset_id,))
                    asset = dict(row)
                    asset.pop("document_id", None)
                    removed_assets.append(asset)
            generation_id = await self._clone_active_generation_in_transaction(
                db, now=time.time()
            )
            await db.commit()
            return {
                "document_ids": ids,
                "deleted_count": len(ids),
                "removed_source_assets": removed_assets,
                "generation_id": generation_id,
            }
        except BaseException:
            await db.rollback()
            raise
        finally:
            await db.close()

    async def delete_entry(self, entry_id: str) -> dict[str, Any]:
        db = await self.pool.acquire()
        try:
            await db.execute("BEGIN IMMEDIATE")
            row = await (
                await db.execute(
                    "SELECT document_id FROM entries WHERE id=?", (entry_id,)
                )
            ).fetchone()
            if row is None:
                raise KeyError(entry_id)
            document_id = row["document_id"]
            await db.execute("DELETE FROM entries WHERE id=?", (entry_id,))
            removed_document = False
            source_asset: dict[str, Any] | None = None
            if document_id:
                remaining = await (
                    await db.execute(
                        "SELECT COUNT(*) AS value FROM entries WHERE document_id=?",
                        (document_id,),
                    )
                ).fetchone()
                if int(remaining["value"]) == 0:
                    asset = await (
                        await db.execute(
                            """SELECT a.* FROM documents d JOIN assets a
                            ON a.id=d.source_asset_id WHERE d.id=?""",
                            (document_id,),
                        )
                    ).fetchone()
                    await db.execute("DELETE FROM documents WHERE id=?", (document_id,))
                    removed_document = True
                    if asset:
                        refs = await (
                            await db.execute(
                                "SELECT COUNT(*) AS value FROM documents WHERE source_asset_id=?",
                                (asset["id"],),
                            )
                        ).fetchone()
                        if int(refs["value"]) == 0:
                            await db.execute("DELETE FROM assets WHERE id=?", (asset["id"],))
                            source_asset = dict(asset)
            generation_id = await self._clone_active_generation_in_transaction(
                db, now=time.time()
            )
            await db.commit()
            return {
                "entry_id": entry_id,
                "document_id": document_id,
                "removed_document": removed_document,
                "removed_source_asset": source_asset,
                "generation_id": generation_id,
            }
        except BaseException:
            await db.rollback()
            raise
        finally:
            await db.close()

    async def delete_document(self, document_id: str) -> dict[str, Any]:
        db = await self.pool.acquire()
        try:
            await db.execute("BEGIN IMMEDIATE")
            asset = await (
                await db.execute(
                    """SELECT a.* FROM documents d JOIN assets a
                    ON a.id=d.source_asset_id WHERE d.id=?""",
                    (document_id,),
                )
            ).fetchone()
            if asset is None:
                raise KeyError(document_id)
            await db.execute("DELETE FROM documents WHERE id=?", (document_id,))
            refs = await (
                await db.execute(
                    "SELECT COUNT(*) AS value FROM documents WHERE source_asset_id=?",
                    (asset["id"],),
                )
            ).fetchone()
            removed_asset = None
            if int(refs["value"]) == 0:
                await db.execute("DELETE FROM assets WHERE id=?", (asset["id"],))
                removed_asset = dict(asset)
            generation_id = await self._clone_active_generation_in_transaction(
                db, now=time.time()
            )
            await db.commit()
            return {
                "document_id": document_id,
                "removed_source_asset": removed_asset,
                "generation_id": generation_id,
            }
        except BaseException:
            await db.rollback()
            raise
        finally:
            await db.close()

    async def delete_image_asset(self, asset_id: str) -> dict[str, Any]:
        db = await self.pool.acquire()
        try:
            await db.execute("BEGIN IMMEDIATE")
            row = await (
                await db.execute(
                    "SELECT * FROM assets WHERE id=? AND kind='image'", (asset_id,)
                )
            ).fetchone()
            if row is None:
                raise KeyError(asset_id)
            await db.execute(
                "DELETE FROM ingest_batch_assets WHERE asset_id=?", (asset_id,)
            )
            await db.execute("DELETE FROM assets WHERE id=?", (asset_id,))
            await db.commit()
            return dict(row)
        except BaseException:
            await db.rollback()
            raise
        finally:
            await db.close()

    async def unlink_asset(
        self, *, scope: str, target_id: str | int, asset_id: str
    ) -> bool:
        if scope not in RELATION_SCOPES:
            raise ValueError("invalid relation scope")
        table = f"{scope}_assets"
        key = f"{scope}_id"
        db = await self.pool.acquire()
        try:
            cursor = await db.execute(
                f"DELETE FROM {table} WHERE {key}=? AND asset_id=?",
                (target_id, asset_id),
            )
            await db.commit()
            return bool(cursor.rowcount)
        finally:
            await db.close()

    async def statistics(self) -> dict[str, int]:
        db = await self.pool.acquire()
        try:
            row = await (
                await db.execute(
                    """SELECT
                    (SELECT COUNT(*) FROM documents
                      WHERE status='ready') AS documents,
                    (SELECT COUNT(*) FROM entries
                      WHERE status='active') AS entries,
                    (SELECT COUNT(*) FROM chunks
                      WHERE status='active') AS chunks,
                    (SELECT COUNT(*) FROM assets
                      WHERE kind='image' AND state='active') AS images"""
                )
            ).fetchone()
            return {key: int(row[key] or 0) for key in row.keys()}
        finally:
            await db.close()

    async def validate(self) -> dict[str, Any]:
        db = await self.pool.acquire()
        try:
            integrity = [row[0] for row in await (await db.execute("PRAGMA integrity_check")).fetchall()]
            foreign_keys = [tuple(row) for row in await (await db.execute("PRAGMA foreign_key_check")).fetchall()]
            if integrity != ["ok"] or foreign_keys:
                raise ValueError("text media knowledge database integrity check failed")
            stats = await self.statistics()
            generation, ids, vectors = await self.active_vectors()
            dimensions = int(vectors.shape[1]) if vectors.size else 0
            media_rows = await (
                await db.execute(
                    """SELECT media_description_vector FROM document_assets
                    WHERE media_description_vector IS NOT NULL"""
                )
            ).fetchall()
            asset_media_rows = await (
                await db.execute(
                    """SELECT asset_id,media_description,
                    normalized_description,sort_order,
                    media_description_vector,vector_sha256
                    FROM asset_media_metadata
                    ORDER BY asset_id,sort_order,id"""
                )
            ).fetchall()
            descriptions_by_asset: dict[str, list[str]] = {}
            for row in asset_media_rows:
                asset_id = str(row["asset_id"])
                descriptions_by_asset.setdefault(asset_id, []).append(
                    str(row["media_description"])
                )
                if int(row["sort_order"]) != (
                    len(descriptions_by_asset[asset_id]) - 1
                ):
                    raise ValueError(
                        "media description order is not contiguous"
                    )
                if str(row["normalized_description"]) != (
                    normalize_media_description(
                        str(row["media_description"])
                    )
                ):
                    raise ValueError(
                        "media description normalization does not match"
                    )
            active_image_ids = {
                str(row[0])
                for row in await (
                    await db.execute(
                        """SELECT id FROM assets
                        WHERE kind='image' AND state='active'"""
                    )
                ).fetchall()
            }
            if active_image_ids != set(descriptions_by_asset):
                raise ValueError(
                    "every active image must have media descriptions"
                )
            for descriptions in descriptions_by_asset.values():
                media_description_list(descriptions)
            media_dimensions = 0
            vector_asset_media_rows = [
                row
                for row in asset_media_rows
                if row["media_description_vector"] is not None
            ]
            for row in [*media_rows, *vector_asset_media_rows]:
                blob = bytes(row["media_description_vector"])
                if not blob or len(blob) % 4:
                    raise ValueError("invalid media description vector")
                values = np.frombuffer(blob, dtype="<f4")
                if not values.size or not np.isfinite(values).all():
                    raise ValueError("invalid media description vector")
                media_dimensions = media_dimensions or int(values.size)
                if media_dimensions != int(values.size):
                    raise ValueError(
                        "media description vector dimensions do not match"
                    )
                if dimensions and dimensions != int(values.size):
                    raise ValueError("media and chunk vector dimensions do not match")
                if "vector_sha256" in row.keys() and (
                    hashlib.sha256(blob).hexdigest()
                    != str(row["vector_sha256"] or "")
                ):
                    raise ValueError("media description vector hash does not match")
            return {
                "integrity": "ok",
                "foreign_keys": 0,
                "stats": stats,
                "generation": generation,
                "vector_count": len(ids),
                "dimensions": dimensions,
                "media_vector_count": (
                    len(media_rows) + len(vector_asset_media_rows)
                ),
                "relation_media_vector_count": len(media_rows),
                "asset_media_vector_count": len(vector_asset_media_rows),
                "media_vector_dimensions": media_dimensions,
            }
        finally:
            await db.close()
