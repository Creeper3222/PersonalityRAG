from __future__ import annotations

import csv
import io
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

from personalityrag.graph import GraphBuilder, canonicalize, dedupe
from personalityrag.config import RecallConfig
from personalityrag.library_types.livingmemory_v8.indexes import (
    EmbeddingInputWarningTracker,
    IndexManager,
)
from personalityrag.retrieval import RetrievalEngine
from personalityrag.storage import Storage
from personalityrag.text import TextProcessor
from personalityrag.library_types.livingmemory_v8.transfer import (
    apply_transfer_summaries,
    create_transfer_preview,
    existing_transfer_keys,
    export_transfer_csv,
    inspect_transfer_records,
    load_transfer_preview,
    parse_transfer_bytes,
)
from personalityrag.library_types.livingmemory_v8.source import (
    serialize_source_messages,
)


class _StubIndexes:
    def __init__(self, documents: list[tuple[int, float]] | None = None) -> None:
        self.documents = documents or []

    async def search_documents(self, _query: str, _k: int):
        return list(self.documents)

    async def search_graph(self, _query: str, _k: int, fetch_k: int | None = None):
        return []


@pytest.mark.asyncio
async def test_embedding_aggregation_uses_upstream_numpy_normalization() -> None:
    raw = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    async def request_embeddings(_texts: list[str]) -> list[list[float]]:
        return raw.tolist()

    actual = await IndexManager._embed_texts_aggregated(
        object(),  # type: ignore[arg-type]
        ["normalization parity"],
        dimension=4,
        capability={"detected_max_context_tokens": 8192},
        tracker=EmbeddingInputWarningTracker("test", 4000),
        request_embeddings=request_embeddings,
        request_batch_size=1,
    )
    expected = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)

    assert np.array_equal(actual, expected)
    assert float(np.linalg.norm(actual[0])) == pytest.approx(1.0, abs=1e-7)


def _table_names(path: Path) -> set[str]:
    with sqlite3.connect(path) as db:
        return {
            str(row[0])
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }


class _SourceMessageObject:
    role = "assistant"
    content = [
        {"type": "text", "text": "文字片段"},
        {"type": "image_url", "image_url": "https://example.invalid/secret"},
    ]
    sender_id = "bot-1"
    timestamp = 456.5


class _SourceMessageWithToDict:
    def to_dict(self) -> dict[str, object]:
        return {
            "role": "user",
            "content": [{"type": "image", "image": "base64-secret"}],
            "metadata": {"is_bot_message": False, "private": "drop"},
        }


def test_graph_entity_normalization_matches_livingmemory_250() -> None:
    assert canonicalize("  （C++）  ") == "c++"
    assert canonicalize("Hello, World!") == "hello, world"
    assert canonicalize("中文，实体") == "中文，实体"
    assert dedupe(["C++", "C", "（C++）", "Hello, World!"], 10) == [
        "C++",
        "C",
        "Hello, World!",
    ]


@pytest.mark.asyncio
async def test_graph_edges_merge_semantically_across_source_memories(
    tmp_path: Path,
) -> None:
    storage = Storage(tmp_path)
    await storage.initialize()
    text = TextProcessor()
    graph = GraphBuilder()
    payload = {
        "content": "Shared graph statement",
        "topics": ["PersonalityRAG"],
        "key_facts": ["LivingMemory uses semantic graph edges"],
    }
    first_id = await storage.create_memory(payload, text.tokenize, graph.build)
    second_id = await storage.create_memory(payload, text.tokenize, graph.build)

    async with storage.connect() as db:
        edges = await (
            await db.execute(
                """SELECT id,weight,confidence FROM graph_edges
                WHERE relation_type='describes'"""
            )
        ).fetchall()
        entry_edges = await (
            await db.execute(
                """SELECT source_memory_id,edge_id FROM graph_entries
                WHERE relation_type='describes' ORDER BY source_memory_id"""
            )
        ).fetchall()

    assert len(edges) == 1
    assert float(edges[0]["weight"]) == pytest.approx(1.15)
    assert float(edges[0]["confidence"]) == pytest.approx(0.82)
    assert [int(row["source_memory_id"]) for row in entry_edges] == [
        first_id,
        second_id,
    ]

    assert {int(row["edge_id"]) for row in entry_edges} == {int(edges[0]["id"])}

    assert await storage.delete_memories([first_id]) == 1
    async with storage.connect() as db:
        surviving_edge = await (
            await db.execute(
                """SELECT id,source_memory_id,edge_key,weight,confidence
                FROM graph_edges WHERE relation_type='describes'"""
            )
        ).fetchone()
        surviving_entries = await (
            await db.execute(
                """SELECT source_memory_id,edge_id FROM graph_entries
                WHERE relation_type='describes'"""
            )
        ).fetchall()

    assert surviving_edge is not None
    assert int(surviving_edge["source_memory_id"]) == second_id
    assert str(surviving_edge["edge_key"]).endswith(f"|{second_id}")
    assert float(surviving_edge["weight"]) == pytest.approx(1.0)
    assert float(surviving_edge["confidence"]) == pytest.approx(0.82)
    assert [
        (int(row["source_memory_id"]), int(row["edge_id"]))
        for row in surviving_entries
    ] == [(second_id, int(surviving_edge["id"]))]
    await storage.close()


@pytest.mark.asyncio
async def test_graph_memory_vector_uses_first_entry_metadata(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    await storage.initialize()
    text = TextProcessor()
    graph = GraphBuilder()
    memory_id = await storage.create_memory(
        {
            "content": "Graph metadata parity",
            "importance": 0.8,
            "key_facts": ["first fact owns aggregate metadata"],
            "topics": ["fallback topic"],
        },
        text.tokenize,
        graph.build,
    )

    rows = await storage.graph_memories_for_ids([memory_id])

    assert len(rows) == 1
    assert rows[0]["metadata"]["source_memory_id"] == memory_id
    assert rows[0]["metadata"]["graph_confidence"] == pytest.approx(0.9)
    assert rows[0]["metadata"]["graph_vector_granularity"] == "memory"
    assert rows[0]["metadata"]["graph_entry_count"] > 1
    await storage.close()


def test_source_serializer_accepts_objects_and_redacts_media_payloads() -> None:
    source = serialize_source_messages(
        [_SourceMessageObject(), _SourceMessageWithToDict(), "普通文本"]
    )

    assert [item["content"] for item in source] == [
        "文字片段",
        "[图片消息]",
        "普通文本",
    ]
    assert source[0]["sender_id"] == "bot-1"
    assert source[0]["timestamp"] == pytest.approx(456.5)
    assert source[1]["metadata"] == {"is_bot_message": False}
    assert "example.invalid" not in json.dumps(source, ensure_ascii=False)
    assert "base64-secret" not in json.dumps(source, ensure_ascii=False)


@pytest.mark.asyncio
async def test_old_v8_read_does_not_create_memory_sources(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    await storage.initialize()
    assert "memory_sources" not in _table_names(storage.db_path)

    assert await storage.list_documents() == {
        "items": [],
        "total": 0,
        "page": 1,
        "page_size": 20,
        "has_more": False,
    }
    assert await storage.get_memory_source(1) == []
    assert "memory_sources" not in _table_names(storage.db_path)
    await storage.close()


@pytest.mark.asyncio
async def test_shadow_sync_tolerates_stale_optional_memory_sources_probe(
    tmp_path: Path,
) -> None:
    source = Storage(tmp_path / "source")
    shadow = Storage(tmp_path / "shadow")
    await source.initialize()
    await shadow.initialize()
    memory_id = await source.create_memory(
        {
            "content": "A legacy v8 memory must remain rebuildable from one DB file",
            "topics": ["compatibility"],
        },
        TextProcessor().tokenize,
        GraphBuilder().build,
    )
    assert not await source.has_memory_sources_table()

    original_table_exists = source._table_exists

    async def stale_optional_table_probe(db, table_name: str) -> bool:
        if table_name == "memory_sources":
            return True
        return await original_table_exists(db, table_name)

    # Reproduce the stale schema observation from the live failure: the old
    # check-then-query code trusted this probe and then queried a table that was
    # not present on the connection used by the replay.
    source._table_exists = stale_optional_table_probe  # type: ignore[method-assign]
    try:
        state = await shadow.sync_memory_from(
            source,
            memory_id,
            TextProcessor().tokenize,
            GraphBuilder().build,
        )
        assert state == "active"
        assert (await shadow.get_document(memory_id))["text"].startswith("A legacy")
        assert await shadow.get_memory_source(memory_id) == []
        source._table_exists = original_table_exists  # type: ignore[method-assign]
        assert not await source.has_memory_sources_table()
    finally:
        await shadow.close()
        await source.close()


@pytest.mark.asyncio
async def test_memory_source_is_lazy_and_cascades_on_delete(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    await storage.initialize()
    memory_id = await storage.create_memory(
        {
            "content": "用户计划周五发布新版本",
            "canonical_summary": "用户计划周五发布新版本",
            "persona_summary": "我记得你准备周五发布呀",
            "source_messages": [
                {
                    "id": 1,
                    "session_id": "session-1",
                    "role": "user",
                    "content": "我准备周五发布新版本",
                    "timestamp": 123.0,
                },
                {
                    "id": 2,
                    "session_id": "session-1",
                    "role": "assistant",
                    "content": "好的，我记住了",
                    "timestamp": 124.0,
                    "metadata": {"is_bot_message": True, "secret": "drop-me"},
                },
            ],
        },
        TextProcessor().tokenize,
        GraphBuilder().build,
    )

    assert "memory_sources" in _table_names(storage.db_path)
    detail = await storage.get_document(memory_id)
    assert detail is not None and detail["has_source"] is True
    source = await storage.get_memory_source(memory_id)
    assert [item["content"] for item in source] == [
        "我准备周五发布新版本",
        "好的，我记住了",
    ]
    assert source[1]["metadata"] == {"is_bot_message": True}

    records = await storage.memory_transfer_records([memory_id])
    assert records[0]["source_messages"] == source
    assert await storage.delete_memories([memory_id]) == 1
    assert await storage.get_memory_source(memory_id) == []
    await storage.close()


@pytest.mark.asyncio
async def test_recall_marks_only_memories_that_have_retained_source(
    tmp_path: Path,
) -> None:
    storage = Storage(tmp_path)
    await storage.initialize()
    text = TextProcessor()
    graph = GraphBuilder()
    with_source = await storage.create_memory(
        {
            "content": "source-aware release note",
            "source_messages": [
                {"role": "user", "content": "ship Friday"},
                {"role": "assistant", "content": "noted"},
            ],
        },
        text.tokenize,
        graph.build,
    )
    without_source = await storage.create_memory(
        {"content": "plain release note"}, text.tokenize, graph.build
    )
    engine = RetrievalEngine(
        storage,
        _StubIndexes([(with_source, 0.9), (without_source, 0.8)]),  # type: ignore[arg-type]
        text,
        RecallConfig(
            graph_memory_enabled=False,
            recent_memory_count=0,
            search_cache_enabled=False,
        ),
    )

    result = [item.to_dict() for item in await engine.search("release note", 2)]
    assert result[0]["has_source"] is True
    assert result[0]["metadata"]["has_source"] is True
    assert result[1]["has_source"] is False
    await storage.close()


@pytest.mark.asyncio
async def test_transfer_preview_supports_external_json_source_only_and_dedupe(
    tmp_path: Path,
) -> None:
    storage = Storage(tmp_path / "library")
    await storage.initialize()
    existing_id = await storage.create_memory(
        {"content": "Existing   memory", "session_id": "s1", "persona_id": "p1"},
        TextProcessor().tokenize,
        GraphBuilder().build,
    )
    assert existing_id > 0
    payload = {
        "records": [
            {
                "text": "Existing memory",
                "session_id": "s1",
                "persona_id": "p1",
            },
            {
                "messages": [
                    {"role": "user", "content": "I prefer tea"},
                    {"role": "assistant", "content": "I will remember"},
                ],
                "session_id": "s2",
            },
            {"messages": [{"role": "user", "content": "too short"}]},
        ]
    }
    records = parse_transfer_bytes(json.dumps(payload).encode(), "external.json")
    existing = existing_transfer_keys(await storage.memory_transfer_records())
    inspection = inspect_transfer_records(records, existing)
    assert inspection["counts"] == {
        "input": 3,
        "valid": 2,
        "invalid": 1,
        "duplicates": 1,
        "needs_summary": 1,
        "planned_skip": 1,
        "planned_import": 1,
    }

    preview_id = create_transfer_preview(
        state_root=tmp_path,
        database_id="memory-test",
        source_sha256="a" * 64,
        inspection=inspection,
        secret="test-secret",
    )
    preview = load_transfer_preview(
        state_root=tmp_path,
        database_id="memory-test",
        preview_id=preview_id,
        secret="test-secret",
    )
    ready, errors = apply_transfer_summaries(
        preview["inspection"]["items"],
        [
            {
                "preview_item_id": "2",
                "canonical_summary": "The user prefers tea.",
                "persona_summary": "I remember that you prefer tea.",
                "importance": 0.8,
            }
        ],
    )
    assert errors == []
    assert ready[1]["canonical_summary"] == "The user prefers tea."
    assert len(ready[1]["source_messages"]) == 2
    await storage.close()


def test_transfer_csv_export_escapes_formula_cells() -> None:
    exported = export_transfer_csv(
        [
            {
                "content": "=HYPERLINK(\"https://example.invalid\")",
                "metadata": {"persona_summary": "+formula", "status": "active"},
                "source_messages": [],
            }
        ]
    ).decode("utf-8-sig")
    row = next(csv.DictReader(io.StringIO(exported)))
    assert row["content"].startswith("'=")
    assert row["persona_summary"].startswith("'+")


def test_transfer_accepts_250_external_collections_and_csv_round_trip() -> None:
    payload = {
        "longTermMemories": {
            "legacy-7": {
                "memory": "  User   likes tea  ",
                "session": "session-a",
                "persona": "persona-a",
                "type": "preference",
            }
        },
        "sessions": {
            "session-b": [
                {"sender": "human", "text": "Remember the blue coat"},
                {"speaker": "bot", "message": "I will remember it"},
            ]
        },
    }
    records = parse_transfer_bytes(json.dumps(payload).encode(), "external.json")
    inspection = inspect_transfer_records(records, set())
    assert inspection["counts"]["valid"] == 2
    assert inspection["items"][0]["content"] == "User likes tea"
    assert inspection["items"][0]["metadata"]["imported_from_id"] == "legacy-7"
    assert inspection["items"][0]["metadata"]["memory_type"] == "PREFERENCE"
    assert inspection["items"][1]["needs_summary"] is True
    assert [
        item["role"] for item in inspection["items"][1]["source_messages"]
    ] == ["user", "assistant"]
    assert all(
        item["session_id"] == "session-b"
        for item in inspection["items"][1]["source_messages"]
    )

    csv_bytes = export_transfer_csv(
        [
            {
                "content": "=formula-like memory",
                "metadata": {"persona_summary": "+persona", "status": "active"},
                "source_messages": [],
            }
        ]
    )
    imported = parse_transfer_bytes(csv_bytes, "round-trip.csv")
    round_trip = inspect_transfer_records(imported, set())
    assert round_trip["items"][0]["content"] == "=formula-like memory"
    assert round_trip["items"][0]["persona_summary"] == "+persona"


@pytest.mark.asyncio
async def test_retrieval_policy_matches_livingmemory_250_rules(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    await storage.initialize()
    text = TextProcessor()
    graph = GraphBuilder()

    legacy = await storage.create_memory(
        {"content": "release plan legacy", "importance": 0.9},
        text.tokenize,
        graph.build,
    )
    preference = await storage.create_memory(
        {
            "content": "release plan preference",
            "importance": 0.9,
            "atoms": [{"atom_type": "preference", "content": "likes releases"}],
        },
        text.tokenize,
        graph.build,
    )
    low_importance = await storage.create_memory(
        {"content": "release plan unimportant", "importance": 0.2},
        text.tokenize,
        graph.build,
    )
    archived = await storage.create_memory(
        {"content": "release plan archived", "importance": 0.9, "status": "archived"},
        text.tokenize,
        graph.build,
    )

    indexes = _StubIndexes(
        [(legacy, 0.8), (preference, 0.8), (low_importance, 0.9), (archived, 0.95)]
    )
    engine = RetrievalEngine(
        storage,
        indexes,  # type: ignore[arg-type]
        text,
        RecallConfig(
            graph_memory_enabled=False,
            min_importance_for_retrieval=0.5,
            min_similarity_for_retrieval=0.7,
            recent_memory_count=0,
            memory_type_filter="event_only",
            search_cache_enabled=False,
        ),
    )
    results = await engine.search("release plan", 10)
    assert [item.doc_id for item in results] == [legacy]

    # A pure keyword match has no vector signal and must not be removed by the
    # minimum-similarity rule.
    indexes.documents = []
    keyword_results = await engine.search("release plan legacy", 10)
    assert [item.doc_id for item in keyword_results] == [legacy]
    await storage.close()


@pytest.mark.asyncio
async def test_recent_slots_reserve_and_deduplicate_results(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    await storage.initialize()
    text = TextProcessor()
    graph = GraphBuilder()
    first = await storage.create_memory(
        {"content": "alpha topic", "session_id": "s1", "persona_id": "p1"},
        text.tokenize,
        graph.build,
    )
    second = await storage.create_memory(
        {"content": "beta topic", "session_id": "s1", "persona_id": "p1"},
        text.tokenize,
        graph.build,
    )
    third = await storage.create_memory(
        {"content": "latest note", "session_id": "s1", "persona_id": "p1"},
        text.tokenize,
        graph.build,
    )
    await storage.update_memory_metadata(
        first, {"metadata": {"create_time": time.time() - 100}}
    )
    await storage.update_memory_metadata(
        second, {"metadata": {"create_time": time.time() - 50}}
    )

    engine = RetrievalEngine(
        storage,
        _StubIndexes([(first, 0.9), (second, 0.8)]),  # type: ignore[arg-type]
        text,
        RecallConfig(
            graph_memory_enabled=False,
            recent_memory_count=1,
            recent_memory_max_age_hours=72,
            search_cache_enabled=False,
        ),
    )
    results = await engine.search("topic", 2, session_id="s1", persona_id="p1")
    assert [item.doc_id for item in results] == [first, third]
    assert results[-1].score_breakdown == {"recent_memory": 1.0}
    await storage.close()
