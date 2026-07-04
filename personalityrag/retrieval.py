from __future__ import annotations

import asyncio
import copy
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .config import RecallConfig
from .indexes import IndexManager
from .storage import Storage, normalize_metadata
from .text import TextProcessor


_GRAPH_NODE_TOKEN_QUERY_BATCH_SIZE = 200


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


def rrf_fuse(
    first: list[tuple[int, float]],
    second: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float, float | None, float | None]]:
    first_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(first)}
    second_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(second)}
    first_score = dict(first)
    second_score = dict(second)
    result = []
    for doc_id in set(first_rank) | set(second_rank):
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
    ):
        self.storage = storage
        self.indexes = indexes
        self.text = text
        self.config = config
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
        )

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
        k = max(1, min(50, int(k or self.config.top_k)))
        key = self._cache_key(query, k, session_id, persona_id)
        cached = self._cache.get(key)
        if cached and time.time() - cached[0] <= self.config.search_cache_ttl_seconds:
            results = copy.deepcopy(cached[1])
            asyncio.create_task(
                self.storage.touch_documents(item.doc_id for item in results)
            )
            return results

        documents, graph = await asyncio.gather(
            self._document_route(query, max(k * 2, k), session_id, persona_id),
            self._graph_route(query, max(k * 2, k), session_id, persona_id),
        )
        results = await self._merge_routes(query, documents, graph, k)
        self._cache[key] = (time.time(), copy.deepcopy(results))
        self._cache.move_to_end(key)
        while len(self._cache) > self.config.search_cache_max_size:
            self._cache.popitem(last=False)
        asyncio.create_task(
            self.storage.touch_documents(item.doc_id for item in results)
        )
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
        fused = rrf_fuse(bm25, vector, self.config.rrf_k)
        if not fused:
            return []
        max_rrf = max(item[1] for item in fused) or 1
        now = time.time()
        results = []
        for doc_id, rrf, bm25_score, vector_score in fused:
            doc = await self.storage.get_document(doc_id)
            if not doc:
                continue
            meta = doc["metadata"]
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
                        "rrf_normalized": round(rrf_normalized, 6),
                        "importance": round(importance, 6),
                        "recency_weight": round(recency, 6),
                        "days_old": round(days, 3),
                        "document_final_score": round(final, 6),
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
        fused = rrf_fuse(keyword, vector, self.config.rrf_k)
        if not fused:
            return []
        max_rrf = max(item[1] for item in fused) or 1
        results = []
        now = time.time()
        for memory_id, rrf, keyword_score, vector_score in fused:
            doc = await self.storage.get_document(memory_id)
            if not doc:
                continue
            meta = doc["metadata"]
            importance = max(0.0, min(1.0, float(meta.get("importance", 0.5))))
            reference = max(
                float(meta.get("create_time", now) or now),
                float(meta.get("last_access_time", 0) or 0),
            )
            recency = math.exp(
                -self.config.decay_rate
                * max(0.0, (now - reference) / 86400)
            )
            graph_confidence = 0.7
            final = (
                0.55 * (rrf / max_rrf)
                + 0.2 * importance
                + 0.15 * recency
                + 0.1 * graph_confidence
            )
            results.append(
                SearchResult(
                    memory_id,
                    final,
                    rrf,
                    keyword_score,
                    vector_score,
                    doc["text"],
                    meta,
                    {
                        "graph_rrf_normalized": round(rrf / max_rrf, 6),
                        "graph_importance": round(importance, 6),
                        "graph_recency_weight": round(recency, 6),
                        "graph_confidence": graph_confidence,
                        "graph_final_score": round(final, 6),
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
    ) -> list[tuple[int, float]]:
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
        candidates: dict[int, float] = {}
        async with self.storage.connect() as db:
            try:
                rows = await (
                    await db.execute(
                        """SELECT ge.source_memory_id,
                        bm25(livingmemory_graph_entries_fts) AS score,
                        ge.session_id,ge.persona_id
                        FROM livingmemory_graph_entries_fts gf
                        JOIN graph_entries ge ON ge.id=gf.entry_id
                        WHERE livingmemory_graph_entries_fts MATCH ?
                        ORDER BY score ASC LIMIT ?""",
                        (fts, max(k * 5, 30)),
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
                    memory_id = int(row["source_memory_id"])
                    candidates[memory_id] = max(
                        candidates.get(memory_id, 0), score
                    )
            rows_by_id: dict[int, Any] = {}
            node_limit = max(k * 3, 20)
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
                    placeholders = ",".join("?" for _ in node_ids)
                    entry_rows = await (
                        await db.execute(
                            f"""SELECT DISTINCT ge.source_memory_id,ge.session_id,ge.persona_id
                            FROM graph_entry_nodes gen
                            JOIN graph_entries ge ON ge.id=gen.entry_id
                            WHERE gen.node_id IN ({placeholders}) LIMIT ?""",
                            (*node_ids, self.config.graph_expansion_limit),
                        )
                    ).fetchall()
                    for row in entry_rows:
                        if session_id and row["session_id"] != session_id:
                            continue
                        if persona_id and row["persona_id"] != persona_id:
                            continue
                        memory_id = int(row["source_memory_id"])
                        candidates[memory_id] = max(
                            candidates.get(memory_id, 0), 0.7
                        )
        return sorted(candidates.items(), key=lambda item: item[1], reverse=True)[
            :k
        ]

    async def _graph_vector(
        self,
        query: str,
        k: int,
        session_id: str | None,
        persona_id: str | None,
    ) -> list[tuple[int, float]]:
        raw = await self.indexes.search_graph(query, max(k * 5, 30))
        aggregated: dict[int, float] = {}
        async with self.storage.connect() as db:
            for entry_id, score in raw:
                row = await (
                    await db.execute(
                        """SELECT source_memory_id,session_id,persona_id
                        FROM graph_entries WHERE id=?""",
                        (entry_id,),
                    )
                ).fetchone()
                if not row:
                    continue
                if session_id and row["session_id"] != session_id:
                    continue
                if persona_id and row["persona_id"] != persona_id:
                    continue
                memory_id = int(row["source_memory_id"])
                aggregated[memory_id] = max(
                    aggregated.get(memory_id, -1.0), score
                )
        return sorted(aggregated.items(), key=lambda item: item[1], reverse=True)[
            :k
        ]

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
            breakdown = {}
            if doc:
                breakdown.update(doc.score_breakdown)
            if graph_item:
                breakdown.update(graph_item.score_breakdown)
            breakdown.update(
                {
                    "document_route_score": round(doc_signal, 6),
                    "graph_route_score": round(graph_signal, 6),
                    "document_route_weight": round(doc_weight, 6),
                    "graph_route_weight": round(graph_weight, 6),
                    "cross_route_bonus": round(bonus, 6),
                    "query_intent": intent,
                    "dual_route_final_score": round(final, 6),
                }
            )
            merged.append(
                SearchResult(
                    doc_id,
                    final,
                    max(doc.rrf_score if doc else 0, graph_item.rrf_score if graph_item else 0),
                    doc.bm25_score if doc else None,
                    doc.vector_score if doc else None,
                    base.content,
                    base.metadata,
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
                "关系",
                "认识",
                "朋友",
                "同事",
                "家人",
                "relationship",
                "friend",
            )
        )
        temporal = any(
            term in normalized
            for term in (
                "上次",
                "昨天",
                "之前",
                "什么时候",
                "最近",
                "last time",
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
                "如何",
                "what is",
                "explain",
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
            tokens = set(self.text.tokenize(text, remove_stopwords=False))
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
