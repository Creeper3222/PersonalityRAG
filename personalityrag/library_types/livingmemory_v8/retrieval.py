from __future__ import annotations

import asyncio
import copy
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

from ...config import RecallConfig
from .indexes import IndexManager
from .storage import Storage, normalize_metadata
from .text import TextProcessor


_GRAPH_NODE_TOKEN_QUERY_BATCH_SIZE = 200
DEFAULT_RECALL_REQUEST_K = 5


@dataclass(slots=True)
class SearchResult:
    doc_id: int
    final_score: float
    rrf_score: float
    bm25_score: float | None
    vector_score: float | None
    content: str
    metadata: dict[str, Any]
    score_breakdown: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.doc_id,
            "content": self.content,
            "has_source": bool(self.metadata.get("has_source", False)),
            "similarity_score": round(self.final_score, 6),
            "rrf_score": round(self.rrf_score, 6),
            "bm25_score": (
                round(self.bm25_score, 6)
                if self.bm25_score is not None
                else None
            ),
            "vector_score": (
                round(self.vector_score, 6)
                if self.vector_score is not None
                else None
            ),
            "metadata": self.metadata,
            "score_breakdown": self.score_breakdown,
        }


@dataclass(slots=True)
class _GraphHit:
    doc_id: int
    score: float
    content: str
    metadata: dict[str, Any]
    entry_type: str | None = None
    relation_type: str | None = None


def _clamp_score(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _graph_temporal_factor(metadata: dict[str, Any], now: float) -> float:
    ttl_days = _safe_float(metadata.get("ttl_days"), 0.0)
    if ttl_days <= 0:
        return 1.0

    last_access = _safe_float(metadata.get("last_access_time"), 0.0)
    days_since = max(0.0, (now - last_access) / 86400.0)
    effective_ttl = max(1.0, ttl_days)
    decay_type = str(metadata.get("decay_type") or "")
    if decay_type == "linear":
        return max(0.0, 1.0 - days_since / effective_ttl)
    if decay_type == "step":
        return 1.0 if days_since <= effective_ttl else 0.05

    half_life = effective_ttl / 2.0
    return math.exp(-math.log(2) * days_since / max(0.5, half_life))


def rrf_fuse(
    first: list[tuple[int, float]],
    second: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float, float | None, float | None]]:
    first_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(first)}
    second_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(second)}
    first_score = dict(first)
    second_score = dict(second)
    all_doc_ids: set[int] = set()
    for doc_id, _score in first:
        all_doc_ids.add(doc_id)
    for doc_id, _score in second:
        all_doc_ids.add(doc_id)

    result = []
    for doc_id in all_doc_ids:
        score = 0.0
        if doc_id in first_rank:
            score += 1 / (k + first_rank[doc_id] + 1)
        if doc_id in second_rank:
            score += 1 / (k + second_rank[doc_id] + 1)
        result.append(
            (doc_id, score, first_score.get(doc_id), second_score.get(doc_id))
        )
    result.sort(key=lambda item: item[1], reverse=True)
    return result


class RetrievalEngine:
    def __init__(
        self,
        storage: Storage,
        indexes: IndexManager,
        text: TextProcessor,
        config: RecallConfig,
        *,
        clock: Callable[[], float] | None = None,
    ):
        self.storage = storage
        self.indexes = indexes
        self.text = text
        self.config = config
        fixed_time = str(
            os.environ.get("PERSONALITYRAG_RECALL_FIXED_TIME") or ""
        ).strip()
        if clock is not None:
            self._clock = clock
        elif fixed_time:
            parsed_fixed_time = float(fixed_time)
            if not math.isfinite(parsed_fixed_time) or parsed_fixed_time <= 0:
                raise ValueError("PERSONALITYRAG_RECALL_FIXED_TIME must be positive")
            self._clock = lambda value=parsed_fixed_time: value
        else:
            self._clock = time.time
        self._track_access = str(
            os.environ.get("PERSONALITYRAG_RECALL_DISABLE_TOUCH") or ""
        ).strip().casefold() not in {"1", "true", "yes", "on"}
        self._cache: OrderedDict[
            tuple[Any, ...], tuple[float, list[SearchResult]]
        ] = OrderedDict()
        self._cache_generation = 0

    def invalidate(self) -> None:
        self._cache_generation += 1
        self._cache.clear()

    def _cache_key(
        self,
        query: str,
        k: int,
        session_id: str | None,
        persona_id: str | None,
    ):
        return (
            self._cache_generation,
            " ".join(query.casefold().split()),
            k,
            session_id or "",
            persona_id or "",
            bool(self.config.graph_memory_enabled),
            round(float(self.config.document_route_weight), 4),
            round(float(self.config.graph_route_weight), 4),
            int(self.config.graph_expansion_hops),
            round(float(self.config.min_importance_for_retrieval), 4),
            round(float(self.config.min_similarity_for_retrieval), 4),
            max(0, int(self.config.recent_memory_count)),
            max(0, int(self.config.recent_memory_max_age_hours)),
            str(self.config.memory_type_filter or "all"),
        )

    def _filter_by_retrieval_policy(
        self, results: list[SearchResult]
    ) -> list[SearchResult]:
        importance_threshold = _clamp_score(
            self.config.min_importance_for_retrieval, 0.0
        )
        similarity_threshold = _clamp_score(
            self.config.min_similarity_for_retrieval, 0.0
        )
        event_only = str(self.config.memory_type_filter or "all") == "event_only"
        event_atom_types = {"episodic", "planned", "factual"}
        filtered: list[SearchResult] = []
        for result in results:
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            if str(metadata.get("status") or "active") != "active":
                continue
            if (
                importance_threshold > 0
                and _clamp_score(metadata.get("importance"), 0.5)
                < importance_threshold
            ):
                continue
            signals: list[float] = []
            if result.vector_score is not None:
                signals.append(_clamp_score(result.vector_score, 0.0))
            for key in ("document_vector_score", "graph_vector_score"):
                if key in result.score_breakdown and result.score_breakdown[key] is not None:
                    signals.append(_clamp_score(result.score_breakdown[key], 0.0))
            # Keyword-only results stay eligible when they do not have a vector signal.
            if similarity_threshold > 0 and signals and max(signals) < similarity_threshold:
                continue
            atom_types = metadata.get("atom_types")
            if event_only and isinstance(atom_types, list) and atom_types:
                normalized_types = {str(value).casefold() for value in atom_types}
                if normalized_types.isdisjoint(event_atom_types):
                    continue
            filtered.append(result)
        return filtered

    async def _recent_memory_results(
        self,
        count: int,
        session_id: str | None,
        persona_id: str | None,
    ) -> list[SearchResult]:
        if count <= 0:
            return []
        conditions = [
            "COALESCE(json_extract(metadata,'$.status'),'active')='active'"
        ]
        params: list[Any] = []
        if session_id is not None:
            conditions.append("json_extract(metadata,'$.session_id')=?")
            params.append(session_id)
        if persona_id is not None:
            conditions.append("json_extract(metadata,'$.persona_id')=?")
            params.append(persona_id)
        max_age_hours = max(0, int(self.config.recent_memory_max_age_hours))
        if max_age_hours > 0:
            conditions.append("CAST(json_extract(metadata,'$.create_time') AS REAL)>=?")
            params.append(self._clock() - max_age_hours * 3600)
        params.append(max(count * 3, count))
        async with self.storage.connect() as db:
            rows = await (
                await db.execute(
                    "SELECT id,text,metadata FROM documents WHERE "
                    + " AND ".join(conditions)
                    + " ORDER BY CAST(json_extract(metadata,'$.create_time') AS REAL) DESC,id DESC LIMIT ?",
                    params,
                )
            ).fetchall()
        recent = []
        for row in rows:
            document = await self.storage.get_document(int(row["id"]))
            metadata = normalize_metadata(row["metadata"])
            if document is not None:
                metadata["has_source"] = bool(document.get("has_source", False))
            recent.append(
                SearchResult(
                    int(row["id"]),
                    1.0,
                    0.0,
                    None,
                    None,
                    str(row["text"] or ""),
                    metadata,
                    {"recent_memory": 1.0},
                )
            )
        return self._filter_by_retrieval_policy(recent)[:count]

    async def _merge_recent_memories(
        self,
        results: list[SearchResult],
        k: int,
        session_id: str | None,
        persona_id: str | None,
    ) -> list[SearchResult]:
        recent_count = min(max(0, int(self.config.recent_memory_count)), k)
        if recent_count <= 0:
            return results[:k]
        recent = await self._recent_memory_results(
            recent_count, session_id, persona_id
        )
        if not recent:
            return results[:k]
        selected = list(results[: max(0, k - recent_count)])
        selected_ids = {item.doc_id for item in selected}
        for item in recent:
            if item.doc_id not in selected_ids:
                selected.append(item)
                selected_ids.add(item.doc_id)
        for item in results:
            if len(selected) >= k:
                break
            if item.doc_id not in selected_ids:
                selected.append(item)
                selected_ids.add(item.doc_id)
        return selected[:k]

    async def search(
        self,
        query: str,
        k: int | None = None,
        session_id: str | None = None,
        persona_id: str | None = None,
    ) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []
        k = max(1, int(k or DEFAULT_RECALL_REQUEST_K))
        cache_enabled = (
            bool(self.config.search_cache_enabled)
            and int(self.config.search_cache_max_size) > 0
            and float(self.config.search_cache_ttl_seconds) > 0
        )
        key = self._cache_key(query, k, session_id, persona_id)
        if cache_enabled:
            cached = self._cache.get(key)
            if (
                cached
                and self._clock() - cached[0]
                <= self.config.search_cache_ttl_seconds
            ):
                results = copy.deepcopy(cached[1])
                if self._track_access:
                    self._touch_documents_in_background(
                        item.doc_id for item in results
                    )
                return results
        else:
            self._cache.clear()

        route_k = max(k * 2, k)
        if self.config.graph_memory_enabled:
            documents, graph = await asyncio.gather(
                self._document_route(query, route_k, session_id, persona_id),
                self._graph_route(query, route_k, session_id, persona_id),
            )
        else:
            documents = await self._document_route(
                query, route_k, session_id, persona_id
            )
            graph = []
        # LivingMemory 2.5.0 expands each route to 2K candidates, but the
        # dual-route merger itself must already contract the set back to K
        # before policy filtering and the recent-memory fill slot run.
        results = await self._merge_routes(query, documents, graph, k)
        results = self._filter_by_retrieval_policy(results)
        results = await self._merge_recent_memories(
            results, k, session_id, persona_id
        )
        if cache_enabled:
            self._cache[key] = (self._clock(), copy.deepcopy(results))
            self._cache.move_to_end(key)
            while len(self._cache) > self.config.search_cache_max_size:
                self._cache.popitem(last=False)
        if self._track_access:
            self._touch_documents_in_background(item.doc_id for item in results)
        return results

    async def _document_route(
        self,
        query: str,
        k: int,
        session_id: str | None,
        persona_id: str | None,
    ) -> list[SearchResult]:
        bm25, vector = await asyncio.gather(
            self._bm25_documents(query, k, session_id, persona_id),
            self._vector_documents(query, k, session_id, persona_id),
        )
        fused = rrf_fuse(bm25, vector, self.config.rrf_k)[:k]
        if not fused:
            return []
        max_rrf = max(item[1] for item in fused) or 1
        now = self._clock()
        results = []
        for doc_id, rrf, bm25_score, vector_score in fused:
            doc = await self.storage.get_document(doc_id)
            if not doc:
                continue
            meta = dict(doc["metadata"])
            meta["has_source"] = bool(doc.get("has_source", False))
            if str(meta.get("status") or "active") != "active":
                continue
            importance = max(0.0, min(1.0, float(meta.get("importance", 0.5))))
            reference = max(
                float(meta.get("create_time", now) or now),
                float(meta.get("last_access_time", 0) or 0),
            )
            days = max(0.0, (now - reference) / 86400)
            recency = math.exp(-self.config.decay_rate * days)
            rrf_normalized = rrf / max_rrf
            final = (
                self.config.score_alpha * rrf_normalized
                + self.config.score_beta * importance
                + self.config.score_gamma * recency
            )
            results.append(
                SearchResult(
                    doc_id,
                    final,
                    rrf,
                    bm25_score,
                    vector_score,
                    doc["text"],
                    meta,
                    {
                        "rrf_normalized": round(rrf_normalized, 4),
                        "importance": round(importance, 4),
                        "recency_weight": round(recency, 4),
                        "days_old": round(days, 2),
                        "final_score": round(final, 4),
                    },
                )
            )
        results.sort(key=lambda item: item.final_score, reverse=True)
        return self._mmr(results, k)

    async def _bm25_documents(
        self,
        query: str,
        k: int,
        session_id: str | None,
        persona_id: str | None,
    ) -> list[tuple[int, float]]:
        tokens = self.text.tokenize(query)
        if not tokens:
            return []
        fts_query = " OR ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
        )
        async with self.storage.connect() as db:
            try:
                rows = await (
                    await db.execute(
                        """SELECT f.doc_id,bm25(livingmemory_memories_fts) AS score,
                        d.metadata FROM livingmemory_memories_fts f
                        JOIN documents d ON d.id=f.doc_id
                        WHERE livingmemory_memories_fts MATCH ?
                        ORDER BY score ASC LIMIT ?""",
                        (fts_query, max(k * 10, 50)),
                    )
                ).fetchall()
            except Exception:
                rows = []
        filtered: list[tuple[int, float]] = []
        for row in rows:
            meta = normalize_metadata(row["metadata"])
            if str(meta.get("status") or "active") != "active":
                continue
            if session_id and meta.get("session_id") != session_id:
                continue
            if persona_id and meta.get("persona_id") != persona_id:
                continue
            filtered.append((int(row["doc_id"]), float(row["score"])))
            if len(filtered) >= k:
                break
        if not filtered:
            return []
        values = [item[1] for item in filtered]
        high, low = max(values), min(values)
        span = high - low
        return [
            (doc_id, 1.0 if span == 0 else (high - score) / span)
            for doc_id, score in filtered
        ]

    async def _vector_documents(
        self,
        query: str,
        k: int,
        session_id: str | None,
        persona_id: str | None,
    ) -> list[tuple[int, float]]:
        raw = await self.indexes.search_documents(query, max(k * 10, 50))
        filtered = []
        for doc_id, score in raw:
            doc = await self.storage.get_document(doc_id)
            if not doc:
                continue
            meta = doc["metadata"]
            if str(meta.get("status") or "active") != "active":
                continue
            if session_id and meta.get("session_id") != session_id:
                continue
            if persona_id and meta.get("persona_id") != persona_id:
                continue
            filtered.append((doc_id, score))
            if len(filtered) >= k:
                break
        return filtered

    async def _graph_route(
        self,
        query: str,
        k: int,
        session_id: str | None,
        persona_id: str | None,
    ) -> list[SearchResult]:
        keyword, vector = await asyncio.gather(
            self._graph_keyword(query, k, session_id, persona_id),
            self._graph_vector(query, k, session_id, persona_id),
        )
        fused = rrf_fuse(
            [(item.doc_id, item.score) for item in keyword],
            [(item.doc_id, item.score) for item in vector],
            self.config.rrf_k,
        )[:k]
        if not fused:
            return []
        keyword_by_id = {item.doc_id: item for item in keyword}
        vector_first_by_id: dict[int, _GraphHit] = {}
        for item in vector:
            vector_first_by_id.setdefault(item.doc_id, item)
        max_rrf = max(item[1] for item in fused) or 1
        results = []
        now = self._clock()
        for memory_id, rrf, keyword_score, vector_score in fused:
            hit = keyword_by_id.get(memory_id) or vector_first_by_id.get(memory_id)
            if hit is None:
                continue
            meta = dict(hit.metadata)
            if str(meta.get("status") or "active") != "active":
                continue
            importance = _clamp_score(meta.get("importance"), 0.5)
            reference = max(
                _safe_float(meta.get("create_time"), now),
                _safe_float(meta.get("last_access_time"), 0.0),
            )
            recency = math.exp(
                -self.config.decay_rate
                * max(0.0, (now - reference) / 86400)
            )
            graph_confidence = _clamp_score(meta.get("graph_confidence"), 0.7)
            temporal_factor = _graph_temporal_factor(meta, now)
            final = (
                0.55 * (rrf / max_rrf)
                + 0.2 * importance
                + 0.15 * recency
                + 0.1 * graph_confidence
            ) * temporal_factor
            results.append(
                SearchResult(
                    memory_id,
                    final,
                    rrf,
                    keyword_score,
                    vector_score,
                    hit.content,
                    meta,
                    {
                        "graph_rrf_normalized": round(rrf / max_rrf, 4),
                        "graph_importance": round(importance, 4),
                        "graph_recency_weight": round(recency, 4),
                        "graph_confidence": round(graph_confidence, 4),
                        "graph_temporal_factor": round(temporal_factor, 4),
                        "graph_final_score": round(final, 4),
                    },
                )
            )
        results.sort(key=lambda item: item.final_score, reverse=True)
        return results[:k]

    async def _graph_keyword(
        self,
        query: str,
        k: int,
        session_id: str | None,
        persona_id: str | None,
    ) -> list[_GraphHit]:
        tokens = list(
            dict.fromkeys(
                token.strip()
                for token in self.text.tokenize(query)
                if str(token).strip()
            )
        )
        if not tokens:
            return []
        fts = " OR ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
        )
        candidates: dict[int, _GraphHit] = {}

        def merge_hit(
            hit: dict[str, Any],
            weight: float = 1.0,
            match_source: str = "graph_keyword",
        ) -> None:
            memory_id = int(hit["source_memory_id"])
            score = float(hit.get("score") or 0.0)
            weighted_score = max(0.0, min(1.0, float(score) * weight))
            current = candidates.get(memory_id)
            metadata = normalize_metadata(hit.get("metadata"))
            metadata["graph_match_source"] = match_source
            metadata["graph_entry_type"] = hit.get("entry_type")
            metadata["graph_relation_type"] = hit.get("relation_type")
            if current is None or weighted_score > current.score:
                candidates[memory_id] = _GraphHit(
                    memory_id,
                    weighted_score,
                    str(hit.get("content") or ""),
                    metadata,
                    hit.get("entry_type"),
                    hit.get("relation_type"),
                )
                return
            current.score = min(1.0, current.score + weighted_score * 0.35)
            if "graph_match_source" in current.metadata:
                current.metadata["graph_match_source"] = (
                    f"{current.metadata['graph_match_source']}+{match_source}"
                )

        def scope_clause(prefix: str = "ge") -> tuple[str, list[Any]]:
            filters: list[str] = []
            params: list[Any] = []
            if session_id is not None:
                filters.append(f"{prefix}.session_id = ?")
                params.append(session_id)
            if persona_id is not None:
                filters.append(f"{prefix}.persona_id = ?")
                params.append(persona_id)
            return (f"AND {' AND '.join(filters)}" if filters else "", params)

        async def entries_for_node_ids(
            db: Any,
            node_ids: list[int],
            limit: int,
        ) -> list[dict[str, Any]]:
            if not node_ids:
                return []
            normalized_ids = sorted({int(item) for item in node_ids})
            placeholders = ",".join("?" for _ in normalized_ids)
            where_clause, scope_params = scope_clause("ge")
            rows = await (
                await db.execute(
                    f"""SELECT ge.id,ge.source_memory_id,ge.content,ge.metadata,
                    ge.entry_type,ge.relation_type,d.metadata AS document_metadata,
                    COUNT(DISTINCT gen.node_id) AS hit_count
                    FROM graph_entry_nodes gen
                    JOIN graph_entries ge ON ge.id=gen.entry_id
                    JOIN documents d ON d.id=ge.source_memory_id
                    WHERE gen.node_id IN ({placeholders})
                    AND COALESCE(json_extract(d.metadata,'$.status'),'active')='active'
                    {where_clause}
                    GROUP BY ge.id
                    ORDER BY hit_count DESC, ge.id DESC
                    LIMIT ?""",
                    (*normalized_ids, *scope_params, limit),
                )
            ).fetchall()
            return [
                {
                    "source_memory_id": int(row["source_memory_id"]),
                    "content": row["content"],
                    "metadata": {
                        **normalize_metadata(row["metadata"]),
                        **normalize_metadata(row["document_metadata"]),
                    },
                    "entry_type": row["entry_type"],
                    "relation_type": row["relation_type"],
                    "score": min(1.0, 0.35 + 0.15 * int(row["hit_count"])),
                }
                for row in rows
            ]

        async def neighbor_node_ids(
            db: Any,
            node_ids: list[int],
            limit: int,
        ) -> list[int]:
            if not node_ids:
                return []
            normalized_ids = sorted({int(item) for item in node_ids})
            placeholders = ",".join("?" for _ in normalized_ids)
            rows = await (
                await db.execute(
                    f"""SELECT neighbor_id, SUM(edge_weight) AS total_weight
                    FROM (
                        SELECT target_node_id AS neighbor_id, weight AS edge_weight
                        FROM graph_edges
                        WHERE source_node_id IN ({placeholders})
                          AND status = 'active'
                        UNION ALL
                        SELECT source_node_id AS neighbor_id, weight AS edge_weight
                        FROM graph_edges
                        WHERE target_node_id IN ({placeholders})
                          AND status = 'active'
                    )
                    WHERE neighbor_id NOT IN ({placeholders})
                    GROUP BY neighbor_id
                    ORDER BY total_weight DESC, neighbor_id ASC
                    LIMIT ?""",
                    (*normalized_ids, *normalized_ids, *normalized_ids, limit),
                )
            ).fetchall()
            return [int(row["neighbor_id"]) for row in rows]

        async with self.storage.connect() as db:
            try:
                where_clause, scope_params = scope_clause("ge")
                rows = await (
                    await db.execute(
                        f"""SELECT ge.id,ge.source_memory_id,ge.content,ge.metadata,
                        ge.entry_type,ge.relation_type,
                        d.metadata AS document_metadata,
                        bm25(livingmemory_graph_entries_fts) AS score,
                        ge.session_id,ge.persona_id
                        FROM livingmemory_graph_entries_fts gf
                        JOIN graph_entries ge ON ge.id=gf.entry_id
                        JOIN documents d ON d.id=ge.source_memory_id
                        WHERE livingmemory_graph_entries_fts MATCH ?
                        AND COALESCE(json_extract(d.metadata,'$.status'),'active')='active'
                        {where_clause}
                        ORDER BY score ASC LIMIT ?""",
                        (fts, *scope_params, max(k * 3, 12)),
                    )
                ).fetchall()
            except Exception:
                rows = []
            if rows:
                scores = [float(row["score"]) for row in rows]
                high, low = max(scores), min(scores)
                span = high - low
                for row in rows:
                    if session_id and row["session_id"] != session_id:
                        continue
                    if persona_id and row["persona_id"] != persona_id:
                        continue
                    score = (
                        1.0
                        if span == 0
                        else (high - float(row["score"])) / span
                    )
                    merge_hit(
                        {
                            "source_memory_id": int(row["source_memory_id"]),
                            "content": row["content"],
                            "metadata": {
                                **normalize_metadata(row["metadata"]),
                                **normalize_metadata(row["document_metadata"]),
                            },
                            "entry_type": row["entry_type"],
                            "relation_type": row["relation_type"],
                            "score": score,
                        },
                        1.0,
                        "graph_keyword",
                    )
            rows_by_id: dict[int, Any] = {}
            node_limit = max(k * 3, 12)
            for start in range(0, len(tokens), _GRAPH_NODE_TOKEN_QUERY_BATCH_SIZE):
                batch = tokens[start : start + _GRAPH_NODE_TOKEN_QUERY_BATCH_SIZE]
                like = [f"%{token}%" for token in batch]
                if not like:
                    continue
                clauses = " OR ".join(["canonical_value LIKE ?" for _ in like])
                node_rows = await (
                    await db.execute(
                        f"""SELECT id,canonical_value FROM graph_nodes
                        WHERE {clauses}
                        ORDER BY length(canonical_value),id LIMIT ?""",
                        (*like, node_limit),
                    )
                ).fetchall()
                for row in node_rows:
                    rows_by_id.setdefault(int(row["id"]), row)
                if len(rows_by_id) >= node_limit:
                    break
            if rows_by_id:
                node_rows = sorted(
                    rows_by_id.values(),
                    key=lambda row: (len(str(row["canonical_value"] or "")), int(row["id"])),
                )[:node_limit]
                node_ids = [int(row["id"]) for row in node_rows]
                if node_ids:
                    per_stage_limit = max(self.config.graph_expansion_limit, k * 3)
                    for hit in await entries_for_node_ids(
                        db,
                        node_ids,
                        per_stage_limit,
                    ):
                        merge_hit(hit, 0.7, "graph_neighbor")

                    first_hop_ids = await neighbor_node_ids(
                        db,
                        node_ids,
                        per_stage_limit,
                    )
                    matched_node_set = set(node_ids)
                    first_hop_ids = [
                        node_id
                        for node_id in first_hop_ids
                        if node_id not in matched_node_set
                    ]
                    for hit in await entries_for_node_ids(
                        db,
                        first_hop_ids,
                        per_stage_limit,
                    ):
                        merge_hit(hit, 0.7, "graph_edge_neighbor")

                    hops = max(1, min(2, int(self.config.graph_expansion_hops)))
                    if hops >= 2 and first_hop_ids:
                        second_hop_ids = await neighbor_node_ids(
                            db,
                            first_hop_ids,
                            per_stage_limit,
                        )
                        excluded_node_ids = matched_node_set | set(first_hop_ids)
                        second_hop_ids = [
                            node_id
                            for node_id in second_hop_ids
                            if node_id not in excluded_node_ids
                        ]
                        second_hop_weight = max(
                            0.0,
                            min(1.0, float(self.config.graph_second_hop_weight)),
                        )
                        for hit in await entries_for_node_ids(
                            db,
                            second_hop_ids,
                            per_stage_limit,
                        ):
                            merge_hit(
                                hit,
                                second_hop_weight,
                                "graph_second_hop",
                            )
        return sorted(candidates.values(), key=lambda item: item.score, reverse=True)[:k]

    async def _graph_vector(
        self,
        query: str,
        k: int,
        session_id: str | None,
        persona_id: str | None,
    ) -> list[_GraphHit]:
        has_filters = session_id is not None or persona_id is not None
        fetch_k = k * 2 if has_filters else k
        raw = await self.indexes.search_graph(query, fetch_k, fetch_k=fetch_k)
        results: list[_GraphHit] = []
        if self.indexes.graph_vector_granularity() == "memory":
            memory_rows = {
                int(item["id"]): item
                for item in await self.storage.graph_memories_for_ids(
                    [item_id for item_id, _score in raw]
                )
            }
            for memory_id, score in raw:
                item = memory_rows.get(int(memory_id))
                if not item:
                    continue
                metadata = dict(item.get("metadata") or {})
                if session_id and metadata.get("session_id") != session_id:
                    continue
                if persona_id and metadata.get("persona_id") != persona_id:
                    continue
                if str(metadata.get("status") or "active") != "active":
                    continue
                results.append(
                    _GraphHit(
                        int(memory_id),
                        float(score),
                        str(item.get("content") or ""),
                        metadata,
                    )
                )
                if len(results) >= k:
                    break
            return results
        async with self.storage.connect() as db:
            for entry_id, score in raw:
                row = await (
                    await db.execute(
                        """SELECT ge.source_memory_id,ge.session_id,ge.persona_id,
                        ge.content,ge.metadata,ge.entry_type,ge.relation_type,
                        d.metadata AS document_metadata
                        FROM graph_entries ge
                        JOIN documents d ON d.id=ge.source_memory_id
                        WHERE ge.id=?
                        AND COALESCE(json_extract(d.metadata,'$.status'),'active')='active'""",
                        (entry_id,),
                    )
                ).fetchone()
                if not row:
                    continue
                if session_id and row["session_id"] != session_id:
                    continue
                if persona_id and row["persona_id"] != persona_id:
                    continue
                results.append(
                    _GraphHit(
                        int(row["source_memory_id"]),
                        float(score),
                        str(row["content"] or ""),
                        {
                            **normalize_metadata(row["metadata"]),
                            **normalize_metadata(row["document_metadata"]),
                        },
                        row["entry_type"],
                        row["relation_type"],
                    )
                )
                if len(results) >= k:
                    break
        return results

    async def _merge_routes(
        self,
        query: str,
        documents: list[SearchResult],
        graph: list[SearchResult],
        k: int,
    ) -> list[SearchResult]:
        if not graph:
            return documents[:k]
        doc_weight, graph_weight, intent = self._route_weights(query)
        doc_max = max((item.final_score for item in documents), default=1) or 1
        graph_max = max((item.final_score for item in graph), default=1) or 1
        doc_map = {item.doc_id: item for item in documents}
        graph_map = {item.doc_id: item for item in graph}
        merged = []
        for doc_id in set(doc_map) | set(graph_map):
            doc = doc_map.get(doc_id)
            graph_item = graph_map.get(doc_id)
            doc_signal = doc.final_score / doc_max if doc else 0
            graph_signal = (
                graph_item.final_score / graph_max if graph_item else 0
            )
            bonus = self.config.cross_route_bonus if doc and graph_item else 0
            final = min(
                1.0,
                doc_weight * doc_signal + graph_weight * graph_signal + bonus,
            )
            base = doc or graph_item
            content = base.content
            metadata = base.metadata
            if doc is None:
                memory = await self.storage.get_document(doc_id)
                if not memory:
                    continue
                content = str(memory.get("text") or content)
                raw_metadata = memory.get("metadata") or metadata
                metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
                metadata = dict(metadata)
                metadata["has_source"] = bool(memory.get("has_source", False))
            breakdown = {}
            if doc:
                breakdown.update(doc.score_breakdown)
            if graph_item:
                breakdown.update(graph_item.score_breakdown)
            breakdown.update(
                {
                    "document_route_score": round(doc_signal, 4),
                    "graph_route_score": round(graph_signal, 4),
                    "document_route_weight": round(doc_weight, 4),
                    "graph_route_weight": round(graph_weight, 4),
                    "cross_route_bonus": round(bonus, 4),
                    "dual_route_final_score": round(final, 4),
                }
            )
            if intent:
                breakdown["query_intent"] = intent
            if doc:
                breakdown["document_keyword_score"] = round(
                    float(doc.bm25_score or 0.0),
                    4,
                )
                breakdown["document_vector_score"] = round(
                    float(doc.vector_score or 0.0),
                    4,
                )
            if graph_item:
                breakdown["graph_keyword_score"] = round(
                    float(graph_item.bm25_score or 0.0),
                    4,
                )
                breakdown["graph_vector_score"] = round(
                    float(graph_item.vector_score or 0.0),
                    4,
                )
            merged.append(
                SearchResult(
                    doc_id,
                    final,
                    max(doc.rrf_score if doc else 0, graph_item.rrf_score if graph_item else 0),
                    doc.bm25_score if doc else None,
                    doc.vector_score if doc else None,
                    content,
                    metadata,
                    breakdown,
                )
            )
        merged.sort(key=lambda item: item.final_score, reverse=True)
        return merged[:k]

    def _route_weights(self, query: str) -> tuple[float, float, str]:
        document = self.config.document_route_weight
        graph = self.config.graph_route_weight
        if not self.config.dynamic_route_weighting:
            return document, graph, "fixed"
        normalized = query.casefold()
        relation = any(
            term in normalized
            for term in (
                "谁",
                "和谁",
                "关系",
                "认识",
                "朋友",
                "同事",
                "家人",
                "父母",
                "妈妈",
                "爸爸",
                "老师",
                "同学",
                "partner",
                "relationship",
                "friend",
                "with whom",
            )
        )
        temporal = any(
            term in normalized
            for term in (
                "上次",
                "昨天",
                "前天",
                "刚才",
                "之前",
                "什么时候",
                "哪天",
                "最近",
                "last time",
                "yesterday",
                "recently",
                "when",
            )
        )
        factual = any(
            term in normalized
            for term in (
                "是什么",
                "什么是",
                "解释",
                "定义",
                "怎么",
                "如何",
                "why",
                "what is",
                "explain",
                "define",
                "how to",
            )
        )
        intent = "default"
        if relation:
            document -= 0.2
            graph += 0.2
            intent = "relationship"
        if temporal:
            document -= 0.1
            graph += 0.1
            intent = "temporal" if intent == "default" else intent + "+temporal"
        if factual and not relation:
            document += 0.15
            graph -= 0.15
            intent = "factual" if intent == "default" else intent + "+factual"
        document = min(0.9, max(0.15, document))
        graph = min(0.85, max(0.1, graph))
        total = document + graph
        return document / total, graph / total, intent

    def _mmr(
        self, results: list[SearchResult], k: int
    ) -> list[SearchResult]:
        if len(results) <= k:
            return results

        def token_set(text: str) -> set[str]:
            tokens = set(str(text or "").lower().split())
            return tokens or {"<empty>"}

        selected: list[SearchResult] = []
        candidates = list(results)
        token_cache = {
            item.doc_id: token_set(item.content) for item in candidates
        }
        while candidates and len(selected) < k:
            if not selected:
                selected.append(candidates.pop(0))
                continue
            best_index = 0
            best_score = float("-inf")
            for index, candidate in enumerate(candidates):
                source = token_cache[candidate.doc_id]
                max_similarity = 0.0
                for chosen in selected:
                    target = token_cache[chosen.doc_id]
                    union = source | target
                    similarity = len(source & target) / len(union) if union else 0
                    max_similarity = max(max_similarity, similarity)
                score = (
                    self.config.mmr_lambda * candidate.final_score
                    - (1 - self.config.mmr_lambda) * max_similarity
                )
                if score > best_score:
                    best_score = score
                    best_index = index
            selected.append(candidates.pop(best_index))
        return selected

    def _touch_documents_in_background(self, document_ids) -> None:
        operation = self.storage.touch_documents(document_ids)
        scheduler = getattr(self.storage, "create_background_task", None)
        if callable(scheduler):
            scheduler(operation)
            return
        # Compatibility for lightweight Storage implementations that predate
        # lifecycle-aware task tracking. The built-in Storage always takes the
        # tracked path above so shutdown can drain pending SQLite writes.
        asyncio.create_task(operation)
