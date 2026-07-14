from __future__ import annotations

import asyncio
import copy
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

from .config import AppConfig, IndexRebuildSettings
from .atoms import classify_memory_atoms
from .graph import GraphBuilder
from .indexes import IndexManager
from .logger import logger
from .migration import LivingMemoryMigrator, sqlite_backup
from .control import ProviderRevision
from .providers import build_provider, build_rerank_provider, provider_config_hash
from .retrieval import RetrievalEngine, SearchResult
from .storage import Storage
from .task_control import JobControlSignal
from .text import TextProcessor


RERANK_GRAPH_MAX_EVIDENCE = 3
RERANK_GRAPH_ENTRY_CHAR_LIMIT = 240
RERANK_GRAPH_DOCUMENT_CHAR_LIMIT = 1200


def _graph_k_core_node_ids(
    node_ids: set[int], edges: list[dict[str, Any]], minimum_degree: int
) -> set[int]:
    """Return the graph's k-core so every retained node has enough visible links."""
    retained = set(node_ids)
    if minimum_degree <= 0:
        return retained
    while retained:
        degrees = {node_id: 0 for node_id in retained}
        for edge in edges:
            source = int(edge["source"])
            target = int(edge["target"])
            if source in retained and target in retained and source != target:
                degrees[source] += 1
                degrees[target] += 1
        removed = {
            node_id
            for node_id, degree in degrees.items()
            if degree < minimum_degree
        }
        if not removed:
            break
        retained.difference_update(removed)
    return retained


def scan_library_backups(data_dir: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for root in (
        data_dir / "backups",
        data_dir / "livingmemory_backups",
        data_dir / "imports",
    ):
        if not root.exists():
            continue
        for path in root.iterdir():
            if not path.is_dir():
                continue
            files = [item for item in path.rglob("*") if item.is_file()]
            result.append(
                {
                    "name": path.name,
                    "directory": str(path),
                    "file_count": len(files),
                    "size_bytes": sum(item.stat().st_size for item in files),
                    "updated_at": path.stat().st_mtime,
                }
            )
    return sorted(result, key=lambda item: item["updated_at"], reverse=True)


class PersonalityRAGService:
    def __init__(
        self,
        root: Path,
        config: AppConfig,
        data_dir: Path | None = None,
        *,
        library_id: str = "",
        default_persona_id: str = "",
        provider_revision: ProviderRevision | None = None,
        rerank_provider_revision: ProviderRevision | None = None,
        system_path: Path | None = None,
    ):
        self.root = root
        self.data_dir = data_dir or (root / "data")
        self.library_id = library_id
        self.default_persona_id = str(default_persona_id or "").strip()
        self.config = config
        self.provider_revision = provider_revision or ProviderRevision(
            provider_id=config.provider.id,
            revision=1,
            config=config.provider,
            config_sha256=provider_config_hash(config.provider),
            created_at=time.time(),
        )
        self.rerank_provider_revision = rerank_provider_revision
        self.storage = Storage(self.data_dir, system_path=system_path)
        self.text = TextProcessor(self.data_dir / "stopwords")
        self.graph_builder = self._build_graph_builder()
        self.provider = build_provider(self.provider_revision.config)
        self.reranker = (
            build_rerank_provider(self.rerank_provider_revision.config)
            if self.rerank_provider_revision
            else None
        )
        self.indexes = IndexManager(
            self.data_dir,
            self.storage,
            self.provider,
            self.provider_revision.config.model,
            library_id=library_id,
            provider_id=self.provider_revision.provider_id,
            provider_revision=self.provider_revision.revision,
            provider_config_sha256=self.provider_revision.config_sha256,
        )
        self.retrieval = RetrievalEngine(
            self.storage, self.indexes, self.text, config.recall
        )
        self.migrator = LivingMemoryMigrator(self.data_dir)
        self._maintenance_task: asyncio.Task | None = None
        self._mutation_lock = asyncio.Lock()
        self._retired_providers: list[Any] = []
        self._retired_rerankers: list[Any] = []

    def _build_graph_builder(self) -> GraphBuilder:
        recall = self.config.recall
        return GraphBuilder(
            max_topics=recall.graph_max_topics,
            max_participants=recall.graph_max_participants,
            max_facts=recall.graph_max_facts,
        )

    def _graph_builder_for_write(self):
        if self.config.recall.graph_memory_enabled:
            return self.graph_builder.build

        def disabled_graph_builder(
            memory_id: int,
            content: str,
            metadata: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            return {"nodes": [], "edges": [], "entries": []}

        return disabled_graph_builder

    def _payload_for_write(self, payload: dict[str, Any]) -> dict[str, Any]:
        prepared = copy.deepcopy(payload)
        requested_persona_id = str(prepared.get("persona_id") or "").strip()
        fallback_persona_id = str(
            getattr(self, "default_persona_id", "") or ""
        ).strip()
        prepared["persona_id"] = requested_persona_id or fallback_persona_id or None
        if not self.config.maintenance.atom_enabled:
            prepared["atoms"] = []
        elif not prepared.get("atoms"):
            prepared["atoms"] = self._atoms_from_key_facts(prepared)
        return prepared

    @staticmethod
    def _atoms_from_key_facts(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return classify_memory_atoms(
            key_facts=payload.get("key_facts"),
            topics=payload.get("topics"),
            participants=payload.get("participants"),
            parent_importance=payload.get("importance", 0.5),
            session_id=payload.get("session_id"),
            persona_id=payload.get("persona_id"),
        )

    @staticmethod
    def _index_rebuild_settings(config: IndexRebuildSettings) -> dict[str, Any]:
        return {
            "batch_size": max(1, int(config.batch_size)),
            "embedding_batch_size": max(1, int(config.embedding_batch_size)),
            "tasks_limit": max(1, int(config.tasks_limit)),
            "max_retries": max(1, int(config.max_retries)),
            "retry_base_delay": max(0.0, float(config.retry_base_delay)),
            "batch_delay": max(0.0, float(config.batch_delay)),
            "request_delay": max(0.0, float(config.request_delay)),
            "max_failure_ratio": max(0.0, float(config.max_failure_ratio)),
        }

    async def initialize(self) -> None:
        logger.info(
            "初始化记忆库 runtime：library_id=%s data_dir=%s provider=%s revision=%s",
            self.library_id,
            self.data_dir,
            self.provider_revision.provider_id,
            self.provider_revision.revision,
        )
        for directory in (
            "indexes",
            "backups",
            "imports",
            "reports",
            "stopwords",
        ):
            (self.data_dir / directory).mkdir(parents=True, exist_ok=True)
        decay_state = self.data_dir / "decay_state.json"
        if not decay_state.exists():
            decay_state.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "library_id": self.library_id,
                        "last_maintenance_at": None,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        await self.storage.initialize()
        self.text = TextProcessor(self.data_dir / "stopwords")
        self.retrieval.text = self.text
        await self.indexes.initialize()
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        logger.info(
            "记忆库 runtime 初始化完成：library_id=%s generation=%s",
            self.library_id,
            (self.indexes.status() or {}).get("generation") or "",
        )

    async def close(self) -> None:
        logger.info("关闭记忆库 runtime：library_id=%s", self.library_id)
        if self._maintenance_task:
            self._maintenance_task.cancel()
            await asyncio.gather(
                self._maintenance_task, return_exceptions=True
            )
        providers = [
            self.provider,
            *self._retired_providers,
            *([self.reranker] if self.reranker else []),
            *self._retired_rerankers,
        ]
        self._retired_providers.clear()
        self._retired_rerankers.clear()
        closed: set[int] = set()
        for provider in providers:
            identity = id(provider)
            if identity in closed:
                continue
            closed.add(identity)
            await provider.close()

    async def set_rerank_provider(
        self, provider_revision: ProviderRevision | None
    ) -> None:
        candidate = (
            build_rerank_provider(provider_revision.config)
            if provider_revision
            else None
        )
        old = self.reranker
        self.reranker = candidate
        self.rerank_provider_revision = provider_revision
        if old is not None and old is not candidate:
            self._retired_rerankers.append(old)
        logger.info(
            "Rerank Provider 已切换：library_id=%s provider=%s",
            self.library_id,
            provider_revision.provider_id if provider_revision else "",
        )

    async def apply_rerank(
        self, query: str, candidates: list[SearchResult], k: int
    ) -> tuple[list[SearchResult], dict[str, Any]]:
        if not self.reranker or not self.rerank_provider_revision or not candidates:
            return candidates[:k], {
                "requested": False,
                "applied": False,
                "provider_id": "",
                "provider_type": "",
            }
        provider_id = self.rerank_provider_revision.provider_id
        provider_type = self.rerank_provider_revision.config.type
        if not self.config.recall.graph_memory_enabled:
            return await self._apply_plain_rerank(
                query, candidates, k, provider_id, provider_type
            )
        try:
            graph_signals = await self._rerank_graph_signals(query, candidates)
        except Exception as exc:
            logger.warning(
                "Rerank 图增强失败，回退纯文本重排：library_id=%s err=%s",
                self.library_id,
                exc,
            )
            return await self._apply_plain_rerank(
                query, candidates, k, provider_id, provider_type
            )
        graph_candidate_count = sum(
            1
            for signal in graph_signals.values()
            if float(signal.get("graph_score") or 0.0) > 0.0
        )
        if graph_candidate_count <= 0:
            return await self._apply_plain_rerank(
                query, candidates, k, provider_id, provider_type
            )
        return await self._apply_graph_enhanced_rerank(
            query,
            candidates,
            k,
            provider_id,
            provider_type,
            graph_signals,
            graph_candidate_count,
        )

    async def _apply_plain_rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        k: int,
        provider_id: str,
        provider_type: str,
    ) -> tuple[list[SearchResult], dict[str, Any]]:
        rerank_rows = await self.reranker.rerank(
            query,
            [item.content for item in candidates],
            k,
        )
        ordered: list[SearchResult] = []
        used_indexes: set[int] = set()
        for row in rerank_rows:
            if row.index < 0 or row.index >= len(candidates) or row.index in used_indexes:
                continue
            used_indexes.add(row.index)
            item = copy.deepcopy(candidates[row.index])
            item.score_breakdown = {
                **item.score_breakdown,
                "rerank_score": round(float(row.relevance_score), 6),
                "original_rank": row.index + 1,
                "original_score": round(float(item.final_score), 6),
                "rerank_provider_id": provider_id,
                "rerank_provider_type": provider_type,
            }
            item.final_score = float(row.relevance_score)
            ordered.append(item)
            if len(ordered) >= k:
                break
        if len(ordered) < k:
            for index, candidate in enumerate(candidates):
                if index in used_indexes:
                    continue
                item = copy.deepcopy(candidate)
                item.score_breakdown = {
                    **item.score_breakdown,
                    "original_rank": index + 1,
                    "original_score": round(float(item.final_score), 6),
                    "rerank_provider_id": provider_id,
                    "rerank_provider_type": provider_type,
                    "rerank_missing": True,
                }
                ordered.append(item)
                if len(ordered) >= k:
                    break
        return ordered[:k], {
            "requested": True,
            "applied": bool(rerank_rows),
            "provider_id": provider_id,
            "provider_type": provider_type,
            "candidate_count": len(candidates),
            "returned": len(ordered[:k]),
            "provider_returned": len(rerank_rows),
            "graph_enhanced": False,
            "graph_candidate_count": 0,
        }

    async def _apply_graph_enhanced_rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        k: int,
        provider_id: str,
        provider_type: str,
        graph_signals: dict[int, dict[str, Any]],
        graph_candidate_count: int,
    ) -> tuple[list[SearchResult], dict[str, Any]]:
        enhanced_documents = [
            self._rerank_document_with_graph_evidence(
                item.content,
                graph_signals.get(item.doc_id, {}),
            )
            for item in candidates
        ]
        rerank_rows = await self.reranker.rerank(
            query,
            enhanced_documents,
            len(candidates),
        )
        valid_rows = []
        used_indexes: set[int] = set()
        for row in rerank_rows:
            if row.index < 0 or row.index >= len(candidates) or row.index in used_indexes:
                continue
            used_indexes.add(row.index)
            valid_rows.append(row)
        raw_by_index = {
            row.index: float(row.relevance_score)
            for row in valid_rows
        }
        rank_by_index = {
            row.index: rank
            for rank, row in enumerate(valid_rows)
        }
        raw_values = list(raw_by_index.values())
        raw_low = min(raw_values) if raw_values else 0.0
        raw_high = max(raw_values) if raw_values else 0.0
        raw_span = raw_high - raw_low
        doc_weight, graph_weight, intent = self._rerank_route_weights(query)
        scored: list[tuple[float, float, float, int, SearchResult]] = []
        returned_count = max(1, len(valid_rows))
        for index, candidate in enumerate(candidates):
            item = copy.deepcopy(candidate)
            raw = raw_by_index.get(index)
            if raw is None:
                raw_score = 0.0
                normalized_raw = 0.0
                rank_score = 0.0
            else:
                raw_score = float(raw)
                normalized_raw = 1.0 if raw_span == 0 else (raw_score - raw_low) / raw_span
                rank = rank_by_index.get(index, returned_count - 1)
                rank_score = (
                    1.0
                    if returned_count <= 1
                    else 1.0 - (rank / max(1, returned_count - 1))
                )
            text_score = max(0.0, min(1.0, 0.7 * normalized_raw + 0.3 * rank_score))
            signal = graph_signals.get(candidate.doc_id, {})
            graph_score = max(0.0, min(1.0, float(signal.get("graph_score") or 0.0)))
            overlap_bonus = (
                self.config.recall.cross_route_bonus
                if text_score > 0.0 and graph_score > 0.0
                else 0.0
            )
            final = max(
                0.0,
                min(
                    1.0,
                    doc_weight * text_score
                    + graph_weight * graph_score
                    + overlap_bonus,
                ),
            )
            item.score_breakdown = {
                **item.score_breakdown,
                "rerank_raw_score": round(raw_score, 6),
                "rerank_text_score": round(text_score, 6),
                "rerank_graph_score": round(graph_score, 6),
                "rerank_graph_keyword_score": round(
                    float(signal.get("keyword_score") or 0.0), 6
                ),
                "rerank_graph_vector_score": round(
                    float(signal.get("vector_score") or 0.0), 6
                ),
                "rerank_graph_node_score": round(
                    float(signal.get("node_score") or 0.0), 6
                ),
                "rerank_graph_evidence_count": int(
                    signal.get("evidence_count") or 0
                ),
                "graph_enhanced_final_score": round(final, 6),
                "original_rank": index + 1,
                "original_score": round(float(candidate.final_score), 6),
                "rerank_provider_id": provider_id,
                "rerank_provider_type": provider_type,
                "rerank_graph_query_intent": intent,
                "rerank_graph_overlap_bonus": round(overlap_bonus, 6),
            }
            if raw is None:
                item.score_breakdown["rerank_missing"] = True
            item.final_score = final
            scored.append((final, text_score, graph_score, -index, item))
        scored.sort(key=lambda row: (row[0], row[1], row[2], row[3]), reverse=True)
        ordered = [item for *_scores, item in scored[:k]]
        return ordered, {
            "requested": True,
            "applied": bool(rerank_rows),
            "provider_id": provider_id,
            "provider_type": provider_type,
            "candidate_count": len(candidates),
            "returned": len(ordered),
            "provider_returned": len(rerank_rows),
            "graph_enhanced": True,
            "graph_candidate_count": graph_candidate_count,
            "document_route_weight": round(doc_weight, 6),
            "graph_route_weight": round(graph_weight, 6),
        }

    def _rerank_route_weights(self, query: str) -> tuple[float, float, str]:
        route_weights = getattr(self.retrieval, "_route_weights", None)
        if callable(route_weights):
            return route_weights(query)
        document = max(0.0, float(self.config.recall.document_route_weight))
        graph = max(0.0, float(self.config.recall.graph_route_weight))
        total = document + graph or 1.0
        return document / total, graph / total, "fixed"

    async def _rerank_graph_signals(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> dict[int, dict[str, Any]]:
        candidate_ids = [item.doc_id for item in candidates]
        tokens = self.text.tokenize(query)
        raw = await self.storage.candidate_graph_evidence(
            candidate_ids,
            tokens,
            max_entries_per_candidate=RERANK_GRAPH_MAX_EVIDENCE,
            graph_expansion_limit=self.config.recall.graph_expansion_limit,
            graph_expansion_hops=self.config.recall.graph_expansion_hops,
            graph_second_hop_weight=self.config.recall.graph_second_hop_weight,
        )
        if not raw:
            return {}
        vector_scores = await self._rerank_graph_vector_scores(query, raw)
        signals: dict[int, dict[str, Any]] = {}
        candidate_id_set = {item.doc_id for item in candidates}
        for memory_id, payload in raw.items():
            if int(memory_id) not in candidate_id_set:
                continue
            keyword_score = max(0.0, min(1.0, float(payload.get("keyword_score") or 0.0)))
            node_score = max(0.0, min(1.0, float(payload.get("node_score") or 0.0)))
            vector_score = max(0.0, min(1.0, float(vector_scores.get(memory_id) or 0.0)))
            confidence = max(0.0, min(1.0, float(payload.get("graph_confidence") or 0.0)))
            if confidence <= 0 and vector_score > 0:
                confidence = 0.7
            graph_score = max(
                0.0,
                min(
                    1.0,
                    0.45 * keyword_score
                    + 0.25 * vector_score
                    + 0.20 * node_score
                    + 0.10 * confidence,
                ),
            )
            entries = list(payload.get("entries") or [])[:RERANK_GRAPH_MAX_EVIDENCE]
            signals[int(memory_id)] = {
                "keyword_score": keyword_score,
                "node_score": node_score,
                "vector_score": vector_score,
                "graph_confidence": confidence,
                "graph_score": graph_score,
                "evidence_count": len(entries) if graph_score > 0 else 0,
                "entries": entries,
            }
        return signals

    async def _rerank_graph_vector_scores(
        self,
        query: str,
        evidence: dict[int, dict[str, Any]],
    ) -> dict[int, float]:
        flattened: list[tuple[int, str]] = []
        for memory_id, payload in evidence.items():
            for entry in list(payload.get("entries") or [])[:RERANK_GRAPH_MAX_EVIDENCE]:
                content = str(entry.get("content") or "").strip()
                if content:
                    flattened.append(
                        (int(memory_id), content[:RERANK_GRAPH_ENTRY_CHAR_LIMIT])
                    )
        if not flattened:
            return {}
        vectors = await self.provider.get_embeddings(
            [query[:RERANK_GRAPH_DOCUMENT_CHAR_LIMIT]]
            + [content for _memory_id, content in flattened]
        )
        if len(vectors) != len(flattened) + 1:
            raise RuntimeError("invalid rerank graph vector count")
        query_vector = vectors[0]
        scores: dict[int, float] = {}
        for (memory_id, _content), vector in zip(flattened, vectors[1:], strict=True):
            scores[memory_id] = max(
                scores.get(memory_id, 0.0),
                self._cosine_score(query_vector, vector),
            )
        return scores

    @staticmethod
    def _cosine_score(left: list[float], right: list[float]) -> float:
        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for a, b in zip(left, right, strict=False):
            fa = float(a)
            fb = float(b)
            dot += fa * fb
            left_norm += fa * fa
            right_norm += fb * fb
        if left_norm <= 0 or right_norm <= 0:
            return 0.0
        cosine = dot / math.sqrt(left_norm * right_norm)
        return max(0.0, min(1.0, cosine))

    @staticmethod
    def _rerank_document_with_graph_evidence(
        content: str,
        signal: dict[str, Any],
    ) -> str:
        entries = list(signal.get("entries") or [])[:RERANK_GRAPH_MAX_EVIDENCE]
        if not entries:
            return content
        evidence_lines = []
        for entry in entries:
            text = str(entry.get("content") or "").strip()
            if not text:
                continue
            prefix = str(entry.get("relation_type") or entry.get("entry_type") or "graph")
            evidence_lines.append(
                f"- {prefix}: {text[:RERANK_GRAPH_ENTRY_CHAR_LIMIT]}"
            )
        if not evidence_lines:
            return content
        evidence = "\n".join(evidence_lines)
        return (
            f"{content[:RERANK_GRAPH_DOCUMENT_CHAR_LIMIT]}\n\n"
            "[Graph evidence for rerank]\n"
            f"{evidence}"
        )

    async def rebuild_indexes(
        self, progress=None, *, job_context=None, checkpoint_dir=None
    ) -> dict[str, Any]:
        async with self._mutation_lock:
            return await self._rebuild_indexes_unlocked(
                progress, job_context=job_context, checkpoint_dir=checkpoint_dir
            )

    @staticmethod
    def _graph_recovery_reason(report: dict[str, Any]) -> str:
        if int(report.get("documents") or 0) <= 0:
            return ""
        if int(report.get("graph_entries") or 0) == 0:
            return "graph_entries_empty"
        if int(report.get("orphan_graph_entries") or 0) > 0:
            return "orphan_graph_entries"
        if int(report.get("graph_nodes") or 0) == 0:
            return "graph_nodes_empty"
        return ""

    @staticmethod
    def _fts_recovery_reason(report: dict[str, Any]) -> str:
        if int(report.get("document_fts") or 0) < 0:
            return "document_fts_unavailable"
        if int(report.get("graph_fts") or 0) < 0:
            return "graph_fts_unavailable"
        if int(report.get("documents") or 0) != int(report.get("document_fts") or 0):
            return "document_fts_count_mismatch"
        if int(report.get("graph_entries") or 0) != int(report.get("graph_fts") or 0):
            return "graph_fts_count_mismatch"
        return ""

    async def _ensure_graph_recovery_unlocked(self, progress=None) -> dict[str, Any]:
        before = await self.storage.graph_integrity_report()
        reason = self._graph_recovery_reason(before)
        if not reason:
            return {
                "rebuilt": False,
                "reason": "not_needed",
                "before": before,
                "after": before,
            }
        logger.warning(
            "检测到图记忆派生数据需要恢复：library_id=%s reason=%s report=%s",
            self.library_id,
            reason,
            before,
        )
        graph = await self.storage.rebuild_graph_from_documents(
            self.text.tokenize,
            self._graph_builder_for_write(),
            progress,
        )
        after = await self.storage.graph_integrity_report()
        logger.warning(
            "图记忆派生数据已恢复：library_id=%s reason=%s result=%s after=%s",
            self.library_id,
            reason,
            graph,
            after,
        )
        return {
            "rebuilt": True,
            "reason": reason,
            "graph": graph,
            "before": before,
            "after": after,
        }

    async def _ensure_fts_recovery_unlocked(
        self,
        *,
        force_reason: str = "",
    ) -> dict[str, Any]:
        before = await self.storage.fts_integrity_report()
        reason = force_reason or self._fts_recovery_reason(before)
        if not reason:
            return {
                "rebuilt": False,
                "reason": "not_needed",
                "before": before,
                "after": before,
                "rows": None,
            }
        logger.warning(
            "检测到 FTS 派生数据需要恢复：library_id=%s reason=%s report=%s",
            self.library_id,
            reason,
            before,
        )
        rows = await self.storage.rebuild_fts(self.text.tokenize)
        after = await self.storage.fts_integrity_report()
        logger.warning(
            "FTS 派生数据已恢复：library_id=%s reason=%s rows=%s after=%s",
            self.library_id,
            reason,
            rows,
            after,
        )
        return {
            "rebuilt": True,
            "reason": reason,
            "rows": rows,
            "before": before,
            "after": after,
        }

    async def _rebuild_graph_entries_unlocked(self, progress=None) -> dict[str, Any]:
        before = await self.storage.graph_integrity_report()
        graph = await self.storage.rebuild_graph_from_documents(
            self.text.tokenize,
            self._graph_builder_for_write(),
            progress,
        )
        after = await self.storage.graph_integrity_report()
        logger.warning(
            "图记忆派生数据已强制重建：library_id=%s result=%s before=%s after=%s",
            self.library_id,
            graph,
            before,
            after,
        )
        return {"graph": graph, "before": before, "after": after}

    async def _rebuild_indexes_unlocked(
        self, progress=None, *, job_context=None, checkpoint_dir=None
    ) -> dict[str, Any]:
        started = time.perf_counter()
        logger.warning(
            "索引重建开始：library_id=%s provider=%s revision=%s",
            self.library_id,
            self.provider_revision.provider_id,
            self.provider_revision.revision,
        )
        if progress:
            await progress(0.02, "正在检查图记忆派生数据")

        async def graph_progress(value: float, message: str) -> None:
            if progress:
                await progress(0.02 + max(0.0, min(1.0, value)) * 0.18, message)

        graph_recovery = await self._ensure_graph_recovery_unlocked(graph_progress)
        fts = await self._ensure_fts_recovery_unlocked(
            force_reason="graph_recovery" if graph_recovery.get("rebuilt") else ""
        )

        async def index_progress(value: float, message: str) -> None:
            if progress:
                await progress(0.20 + max(0.0, min(1.0, value)) * 0.80, message)

        manifest = await self.indexes.rebuild(
            **self._index_rebuild_settings(
                self.provider_revision.config.index_rebuild_settings
            ),
            progress=index_progress if progress else None,
            job_context=job_context,
            checkpoint_dir=checkpoint_dir,
        )
        self.retrieval.invalidate()
        logger.warning(
            "索引重建成功：library_id=%s generation=%s elapsed_ms=%.2f",
            self.library_id,
            manifest.get("generation"),
            (time.perf_counter() - started) * 1000,
        )
        return {"graph_recovery": graph_recovery, "fts": fts, "manifest": manifest}

    async def rebuild_graph(
        self, progress=None, *, job_context=None, checkpoint_dir=None
    ) -> dict[str, Any]:
        async with self._mutation_lock:
            started = time.perf_counter()
            logger.warning("图记忆重建开始：library_id=%s", self.library_id)

            async def graph_progress(value: float, message: str) -> None:
                if progress:
                    await progress(0.02 + max(0.0, min(1.0, value)) * 0.38, message)

            graph = await self._rebuild_graph_entries_unlocked(graph_progress)
            fts = await self._ensure_fts_recovery_unlocked(
                force_reason="graph_rebuild"
            )

            async def index_progress(value: float, message: str) -> None:
                if progress:
                    await progress(0.40 + max(0.0, min(1.0, value)) * 0.60, message)

            manifest = await self.indexes.rebuild(
                **self._index_rebuild_settings(
                    self.provider_revision.config.index_rebuild_settings
                ),
                progress=index_progress if progress else None,
                job_context=job_context,
                checkpoint_dir=checkpoint_dir,
            )
            self.retrieval.invalidate()
            logger.warning(
                "图记忆重建完成：library_id=%s generation=%s elapsed_ms=%.2f",
                self.library_id,
                manifest.get("generation"),
                (time.perf_counter() - started) * 1000,
            )
            return {"graph": graph, "fts": fts, "manifest": manifest}

    async def rebuild_with_provider(
        self,
        provider_revision: ProviderRevision,
        progress=None,
        *,
        job_context=None,
        checkpoint_dir=None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        logger.warning(
            "Provider 切换重建开始：library_id=%s provider=%s revision=%s",
            self.library_id,
            provider_revision.provider_id,
            provider_revision.revision,
        )
        candidate = build_provider(provider_revision.config)
        old_provider = self.provider
        async with self._mutation_lock:
            try:
                if progress:
                    await progress(0.02, "正在检查图记忆派生数据")

                async def graph_progress(value: float, message: str) -> None:
                    if progress:
                        await progress(0.02 + max(0.0, min(1.0, value)) * 0.18, message)

                graph_recovery = await self._ensure_graph_recovery_unlocked(
                    graph_progress
                )
                fts = await self._ensure_fts_recovery_unlocked(
                    force_reason=(
                        "graph_recovery" if graph_recovery.get("rebuilt") else ""
                    )
                )

                async def index_progress(value: float, message: str) -> None:
                    if progress:
                        await progress(0.20 + max(0.0, min(1.0, value)) * 0.80, message)

                manifest = await self.indexes.rebuild(
                    **self._index_rebuild_settings(
                        provider_revision.config.index_rebuild_settings
                    ),
                    progress=index_progress if progress else None,
                    provider=candidate,
                    library_id=self.library_id,
                    provider_id=provider_revision.provider_id,
                    provider_revision=provider_revision.revision,
                    provider_config_sha256=provider_revision.config_sha256,
                    provider_model=provider_revision.config.model,
                    job_context=job_context,
                    checkpoint_dir=checkpoint_dir,
                )
            except JobControlSignal:
                logger.info(
                    "Provider 切换重建已到达任务控制边界：library_id=%s provider=%s revision=%s",
                    self.library_id,
                    provider_revision.provider_id,
                    provider_revision.revision,
                )
                await candidate.close()
                raise
            except Exception:
                logger.exception(
                    "Provider 切换重建失败，保留旧索引：library_id=%s provider=%s revision=%s",
                    self.library_id,
                    provider_revision.provider_id,
                    provider_revision.revision,
                )
                await candidate.close()
                raise
            self.provider = candidate
            self.provider_revision = provider_revision
            self.retrieval.invalidate()
        if old_provider is not candidate:
            # Queries capture an immutable index/provider snapshot before awaiting
            # the embedding request. Keep replaced clients alive until runtime
            # shutdown so an in-flight query cannot lose its HTTP client midway.
            self._retired_providers.append(old_provider)
        logger.warning(
            "Provider 切换重建成功：library_id=%s provider=%s revision=%s generation=%s elapsed_ms=%.2f",
            self.library_id,
            provider_revision.provider_id,
            provider_revision.revision,
            manifest.get("generation"),
            (time.perf_counter() - started) * 1000,
        )
        return {"graph_recovery": graph_recovery, "fts": fts, "manifest": manifest}

    async def create_memory(
        self,
        payload: dict[str, Any],
        *,
        rebuild: bool = True,
        progress=None,
    ) -> dict[str, Any]:
        async with self._mutation_lock:
            memory_id = await self.storage.create_memory(
                self._payload_for_write(payload),
                self.text.tokenize,
                self._graph_builder_for_write(),
            )
            try:
                index_update = await self.indexes.upsert_memories(
                    [memory_id], reason="memory_create"
                )
                await self.storage.mark_memory_indexed(
                    memory_id,
                    generation=str(index_update.get("generation") or ""),
                )
            except Exception:
                await self.storage.delete_memories([memory_id])
                self.retrieval.invalidate()
                logger.exception(
                    "记忆增量索引失败，已回滚数据库写入：library_id=%s memory_id=%s",
                    self.library_id,
                    memory_id,
                )
                raise
            logger.info(
                "记忆已写入数据库并完成增量索引：library_id=%s memory_id=%s generation=%s",
                self.library_id,
                memory_id,
                index_update.get("generation"),
            )
            self.retrieval.invalidate()
            result = (await self.storage.get_document(memory_id)) or {"id": memory_id}
            result["index_update"] = index_update
            return result

    async def update_memory(
        self,
        memory_id: int,
        payload: dict[str, Any],
        *,
        rebuild: bool = True,
        progress=None,
    ) -> dict[str, Any] | None:
        async with self._mutation_lock:
            if "content" in payload:
                current = await self.storage.get_document(memory_id)
                if not current:
                    return None
                new_content = str(payload.get("content") or "").strip()
                if not new_content:
                    raise ValueError("content cannot be empty")
                current_metadata = dict(current.get("metadata") or {})
                merged_metadata = {
                    **current_metadata,
                    **dict(payload.get("metadata") or {}),
                }
                for key in (
                    "importance",
                    "status",
                    "memory_type",
                    "session_id",
                    "persona_id",
                ):
                    if key in payload:
                        merged_metadata[key] = payload[key]
                merged_metadata["previous_id"] = memory_id
                merged_metadata["updated_at"] = time.time()
                new_payload = {
                    "content": new_content,
                    "canonical_summary": new_content,
                    "persona_summary": merged_metadata.get(
                        "persona_summary", new_content
                    ),
                    "persona_id": merged_metadata.get("persona_id"),
                    "session_id": merged_metadata.get("session_id"),
                    "importance": merged_metadata.get("importance", 0.5),
                    "status": merged_metadata.get("status", "active"),
                    "memory_type": merged_metadata.get("memory_type", "GENERAL"),
                    "topics": merged_metadata.get("topics", []),
                    "participants": merged_metadata.get("participants", []),
                    "key_facts": merged_metadata.get("key_facts", []),
                    "metadata": merged_metadata,
                }
                old_graph_entries = await self.storage.graph_entries_for_memory_ids(
                    [memory_id]
                )
                old_graph_ids = {int(item["id"]) for item in old_graph_entries}
                new_memory_id = await self.storage.create_memory(
                    self._payload_for_write(new_payload),
                    self.text.tokenize,
                    self._graph_builder_for_write(),
                )
                new_index_update: dict[str, Any] | None = None
                try:
                    new_index_update = await self.indexes.upsert_memories(
                        [new_memory_id], reason="memory_update_create_replacement"
                    )
                    await self.storage.mark_memory_indexed(
                        new_memory_id,
                        generation=str(new_index_update.get("generation") or ""),
                    )
                    delete_index_update = await self.indexes.delete_memories_incremental(
                        [memory_id],
                        graph_entry_ids=old_graph_ids,
                        reason="memory_update_delete_old",
                    )
                    deleted = await self.storage.delete_memories([memory_id])
                    if deleted != 1:
                        raise RuntimeError(
                            f"failed to delete replaced memory {memory_id}"
                        )
                except Exception:
                    logger.exception(
                        "内容编辑替换失败，正在回滚：library_id=%s old_id=%s new_id=%s",
                        self.library_id,
                        memory_id,
                        new_memory_id,
                    )
                    if await self.storage.get_document(memory_id):
                        try:
                            await self.indexes.upsert_memories(
                                [memory_id],
                                remove_graph_entry_ids=set(),
                                reason="memory_update_restore_old",
                            )
                        except Exception:
                            logger.error(
                                "恢复旧记忆索引失败：library_id=%s memory_id=%s",
                                self.library_id,
                                memory_id,
                                exc_info=True,
                            )
                    if await self.storage.get_document(new_memory_id):
                        new_graph_entries = await self.storage.graph_entries_for_memory_ids(
                            [new_memory_id]
                        )
                        if new_index_update is not None:
                            try:
                                await self.indexes.delete_memories_incremental(
                                    [new_memory_id],
                                    graph_entry_ids={
                                        int(item["id"]) for item in new_graph_entries
                                    },
                                    reason="memory_update_rollback_new",
                                )
                            except Exception:
                                logger.error(
                                    "回滚新记忆索引失败：library_id=%s memory_id=%s",
                                    self.library_id,
                                    new_memory_id,
                                    exc_info=True,
                                )
                        await self.storage.delete_memories([new_memory_id])
                    self.retrieval.invalidate()
                    raise
                self.retrieval.invalidate()
                result = await self.storage.get_document(new_memory_id)
                if result is None:
                    result = {"id": new_memory_id}
                result["old_memory_id"] = memory_id
                result["new_memory_id"] = new_memory_id
                result["index_update"] = delete_index_update
                result["replacement_index_update"] = new_index_update
                logger.info(
                    "记忆内容编辑已按新 ID 替换完成：library_id=%s old_id=%s new_id=%s",
                    self.library_id,
                    memory_id,
                    new_memory_id,
                )
                return result

            success = await self.storage.update_memory_metadata(
                memory_id, payload
            )
            if not success:
                return None
            self.retrieval.invalidate()
            result = await self.storage.get_document(memory_id)
            if result is not None:
                index_status = self.indexes.status() or {}
                result["index_update"] = {
                    "mode": "metadata_only",
                    "status": "completed",
                    "index_changed": False,
                    "generation": index_status.get("generation"),
                    "document_vectors": index_status.get(
                        "document_vectors"
                    ),
                    "graph_vectors": index_status.get("graph_vectors"),
                }
            return result

    async def update_memory_persona(
        self, memory_id: int, persona_id: str | None
    ) -> dict[str, Any] | None:
        async with self._mutation_lock:
            success = await self.storage.update_memory_persona(
                memory_id, persona_id
            )
            if not success:
                return None
            self.retrieval.invalidate()
            result = await self.storage.get_document(memory_id)
            if result is None:
                return None
            index_status = self.indexes.status() or {}
            result["index_update"] = {
                "mode": "metadata_only",
                "status": "completed",
                "index_changed": False,
                "generation": index_status.get("generation"),
                "document_vectors": index_status.get("document_vectors"),
                "graph_vectors": index_status.get("graph_vectors"),
            }
            logger.info(
                "记忆人格字段已原位更新：library_id=%s memory_id=%s persona=%s generation=%s index_changed=false",
                self.library_id,
                memory_id,
                str(persona_id or "").strip() or "",
                index_status.get("generation") or "",
            )
            return result

    async def delete_memories(
        self,
        memory_ids: list[int],
        *,
        rebuild: bool = True,
        progress=None,
        return_details: bool = False,
    ) -> int | dict[str, Any]:
        async with self._mutation_lock:
            existing_documents = await self.storage.documents_for_ids(memory_ids)
            existing_ids = [int(item["id"]) for item in existing_documents]
            if not existing_ids:
                return {"deleted": 0, "index_update": None} if return_details else 0
            graph_entries = await self.storage.graph_entries_for_memory_ids(existing_ids)
            graph_entry_ids = {int(item["id"]) for item in graph_entries}
            index_update = await self.indexes.delete_memories_incremental(
                existing_ids,
                graph_entry_ids=graph_entry_ids,
                reason="memory_delete",
            )
            try:
                deleted = await self.storage.delete_memories(existing_ids)
            except Exception:
                logger.exception(
                    "删除记忆数据库阶段失败，正在恢复索引：library_id=%s ids=%s",
                    self.library_id,
                    existing_ids,
                )
                try:
                    await self.indexes.upsert_memories(
                        existing_ids, reason="memory_delete_restore"
                    )
                except Exception:
                    logger.error(
                        "恢复删除前索引失败：library_id=%s ids=%s",
                        self.library_id,
                        existing_ids,
                        exc_info=True,
                    )
                raise
            if deleted:
                logger.warning(
                    "记忆已删除并完成增量索引移除：library_id=%s deleted=%s generation=%s",
                    self.library_id,
                    deleted,
                    index_update.get("generation"),
                )
                self.retrieval.invalidate()
            if return_details:
                return {"deleted": deleted, "index_update": index_update}
            return deleted

    async def graph_snapshot(
        self,
        *,
        memory_ids: list[int] | None = None,
        session_id: str | None = None,
        persona_id: str | None = None,
        limit_nodes: int = 80,
        limit_edges: int = 120,
        limit_memories: int = 24,
        minimum_degree: int = 0,
    ) -> dict[str, Any]:
        limit_nodes = max(1, min(int(limit_nodes), 200))
        limit_edges = max(1, min(int(limit_edges), 400))
        minimum_degree = max(0, min(int(minimum_degree), 20))
        clauses: list[str] = []
        params: list[Any] = []
        if memory_ids:
            placeholders = ",".join("?" for _ in memory_ids)
            clauses.append(f"ge.source_memory_id IN ({placeholders})")
            params.extend(memory_ids)
        if session_id:
            clauses.append("ge.session_id=?")
            params.append(session_id)
        if persona_id:
            clauses.append("ge.persona_id=?")
            params.append(persona_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        async with self.storage.connect() as db:
            entry_rows = await (
                await db.execute(
                    f"""SELECT ge.id,ge.source_memory_id,ge.entry_type,
                    ge.relation_type,ge.content,ge.metadata
                    FROM graph_entries ge {where}
                    ORDER BY ge.id DESC LIMIT ?""",
                    (*params, max(limit_edges, limit_nodes)),
                )
            ).fetchall()
            selected_memory_ids = []
            for row in entry_rows:
                value = int(row["source_memory_id"])
                if value not in selected_memory_ids:
                    selected_memory_ids.append(value)
                if len(selected_memory_ids) >= limit_memories:
                    break
            entry_ids = [int(row["id"]) for row in entry_rows]
            if not entry_ids:
                return {
                    "nodes": [],
                    "edges": [],
                    "entries": [],
                    "memories": [],
                }
            entry_placeholders = ",".join("?" for _ in entry_ids)
            entry_node_map: dict[int, list[int]] = {}
            node_rows = await (
                await db.execute(
                    f"""SELECT DISTINCT n.id,n.node_key,n.node_type,n.node_value,
                    n.canonical_value,n.metadata
                    FROM graph_entry_nodes gen JOIN graph_nodes n ON n.id=gen.node_id
                    WHERE gen.entry_id IN ({entry_placeholders})""",
                    (*entry_ids,),
                )
            ).fetchall()
            entry_node_rows = await (
                await db.execute(
                    f"""SELECT entry_id,node_id FROM graph_entry_nodes
                    WHERE entry_id IN ({entry_placeholders})""",
                    (*entry_ids,),
                )
            ).fetchall()
            for row in entry_node_rows:
                entry_node_map.setdefault(int(row["entry_id"]), []).append(int(row["node_id"]))
            node_ids = [int(row["id"]) for row in node_rows]
            edge_rows = []
            node_stats: dict[int, dict[str, Any]] = {}
            if node_ids:
                node_placeholders = ",".join("?" for _ in node_ids)
                edge_rows = await (
                    await db.execute(
                        f"""SELECT e.id,e.source_node_id,e.target_node_id,
                        e.relation_type,e.source_memory_id,e.weight,e.confidence,e.status
                        FROM graph_edges e WHERE e.source_node_id IN ({node_placeholders})
                        AND e.target_node_id IN ({node_placeholders})
                        ORDER BY e.weight DESC,e.confidence DESC,e.id DESC""",
                        (*node_ids, *node_ids),
                    )
                ).fetchall()
                entry_stat_rows = await (
                    await db.execute(
                        f"""SELECT gen.node_id AS node_id,
                        COUNT(DISTINCT gen.entry_id) AS entry_count,
                        COUNT(DISTINCT ge.source_memory_id) AS memory_count
                        FROM graph_entry_nodes gen
                        JOIN graph_entries ge ON ge.id=gen.entry_id
                        WHERE gen.node_id IN ({node_placeholders})
                        GROUP BY gen.node_id""",
                        (*node_ids,),
                    )
                ).fetchall()
                for row in entry_stat_rows:
                    node_stats[int(row["node_id"])] = {
                        "entry_count": int(row["entry_count"] or 0),
                        "memory_count": int(row["memory_count"] or 0),
                    }
                for row in edge_rows:
                    source_id = int(row["source_node_id"])
                    target_id = int(row["target_node_id"])
                    edge_weight = float(row["weight"] or 0)
                    source_stats = node_stats.setdefault(source_id, {})
                    source_stats["degree"] = int(source_stats.get("degree", 0)) + 1
                    source_stats["weight"] = float(source_stats.get("weight", 0.0)) + edge_weight
                    target_stats = node_stats.setdefault(target_id, {})
                    target_stats["degree"] = int(target_stats.get("degree", 0)) + 1
                    target_stats["weight"] = float(target_stats.get("weight", 0.0)) + edge_weight
        edge_view = [
            {
                "id": int(row["id"]),
                "source": int(row["source_node_id"]),
                "target": int(row["target_node_id"]),
                "relation_type": row["relation_type"],
                "source_memory_id": int(row["source_memory_id"]),
                "weight": float(row["weight"]),
                "confidence": float(row["confidence"]),
                "status": row["status"],
            }
            for row in edge_rows
        ]
        allowed_node_ids = _graph_k_core_node_ids(
            set(node_ids), edge_view, minimum_degree
        )
        ranked_node_ids = sorted(
            allowed_node_ids,
            key=lambda node_id: (
                -int(node_stats.get(node_id, {}).get("degree", 0)),
                -int(node_stats.get(node_id, {}).get("memory_count", 0)),
                -float(node_stats.get(node_id, {}).get("weight", 0.0)),
                node_id,
            ),
        )
        allowed_node_ids = set(ranked_node_ids[:limit_nodes])
        edge_view = [
            edge
            for edge in edge_view
            if edge["source"] in allowed_node_ids
            and edge["target"] in allowed_node_ids
        ][:limit_edges]
        allowed_node_ids = _graph_k_core_node_ids(
            allowed_node_ids, edge_view, minimum_degree
        )
        edge_view = [
            edge
            for edge in edge_view
            if edge["source"] in allowed_node_ids
            and edge["target"] in allowed_node_ids
        ]

        visible_degrees = {node_id: 0 for node_id in allowed_node_ids}
        visible_weights = {node_id: 0.0 for node_id in allowed_node_ids}
        for edge in edge_view:
            source = edge["source"]
            target = edge["target"]
            visible_degrees[source] += 1
            visible_degrees[target] += 1
            visible_weights[source] += edge["weight"]
            visible_weights[target] += edge["weight"]

        memories = [
            item
            for memory_id in selected_memory_ids
            if (item := await self.storage.get_document(memory_id))
        ]
        return {
            "nodes": [
                {
                    "id": int(row["id"]),
                    "key": row["node_key"],
                    "type": row["node_type"],
                    "label": row["node_value"],
                    "canonical_value": row["canonical_value"],
                    "memory_count": int(node_stats.get(int(row["id"]), {}).get("memory_count", 0)),
                    "degree": visible_degrees.get(int(row["id"]), 0),
                    "entry_count": int(node_stats.get(int(row["id"]), {}).get("entry_count", 0)),
                    "weight": visible_weights.get(int(row["id"]), 0.0),
                }
                for row in node_rows
                if int(row["id"]) in allowed_node_ids
            ],
            "edges": edge_view,
            "entries": [
                {
                    "id": int(row["id"]),
                    "source_memory_id": int(row["source_memory_id"]),
                    "entry_type": row["entry_type"],
                    "relation_type": row["relation_type"],
                    "content": row["content"],
                    "node_ids": [
                        node_id
                        for node_id in entry_node_map.get(int(row["id"]), [])
                        if node_id in allowed_node_ids
                    ],
                }
                for row in entry_rows
            ],
            "memories": memories,
        }

    async def list_backups(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(scan_library_backups, self.data_dir)

    async def backup(self) -> Path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        target = self.data_dir / "backups" / timestamp
        target.mkdir(parents=True, exist_ok=False)
        await asyncio.to_thread(
            sqlite_backup,
            self.storage.db_path,
            target / "livingmemory.db",
        )
        await asyncio.to_thread(
            sqlite_backup,
            self.storage.conversations_path,
            target / "conversations.db",
        )
        logger.warning(
            "核心记忆库文件已备份：library_id=%s livingmemory=%s conversations=%s",
            self.library_id,
            target / "livingmemory.db",
            target / "conversations.db",
        )
        return target

    async def append_conversation_message(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        result = await self.storage.add_conversation_message(payload)
        if result.get("duplicate"):
            return result
        session = result.get("session") or {}
        session_id = str(payload.get("session_id") or "")
        max_messages = max(
            1,
            int(
                getattr(
                    self.config.conversation,
                    "max_messages_per_session",
                    1000,
                )
                or 1000
            ),
        )
        cleanup_batch_size = max(
            1,
            int(getattr(self.config.conversation, "cleanup_batch_size", 50) or 50),
        )
        message_count = int(session.get("message_count") or 0)
        if session_id and message_count > max_messages:
            trim = await self.storage.trim_conversation(
                session_id,
                max(cleanup_batch_size, message_count - max_messages),
            )
            result["trimmed"] = int(trim.get("deleted") or 0)
            result["session"] = trim.get("session") or session
        return result

    async def _maintenance_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(
                    max(
                        300,
                        int(
                            self.config.maintenance.atom_maintenance_interval_hours
                            * 3600
                        ),
                    )
                )
                await self.run_maintenance()
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(60)

    def _today_key(self, now: float | None = None) -> str:
        return time.strftime("%Y-%m-%d", time.localtime(now or time.time()))

    def _load_decay_state(self) -> dict[str, Any]:
        state_path = self.data_dir / "decay_state.json"
        if not state_path.exists():
            return {"version": 1, "library_id": self.library_id}
        try:
            data = json.loads(state_path.read_text(encoding="utf-8") or "{}")
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "library_id": self.library_id}
        return data if isinstance(data, dict) else {"version": 1, "library_id": self.library_id}

    def _write_decay_state(self, state: dict[str, Any]) -> None:
        state_path = self.data_dir / "decay_state.json"
        state_temp = state_path.with_suffix(".tmp")
        state_temp.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        state_temp.replace(state_path)

    async def run_maintenance(self) -> dict[str, Any]:
        now = time.time()
        today = self._today_key(now)
        state = self._load_decay_state()
        run_daily = state.get("last_daily_maintenance_date") != today
        forgot_before = (
            now
            - self.config.maintenance.atom_forget_delay_days * 86400
        )
        purge_before = (
            now - self.config.maintenance.atom_purge_delay_days * 86400
        )
        decayed = 0
        cleanup_ids: list[int] = []
        async with self.storage.connect() as db:
            if run_daily:
                decay_rate = max(0.0, min(1.0, float(self.config.recall.decay_rate)))
                access_window_days = max(
                    1.0, float(self.config.recall.access_decay_window_days)
                )
                max_access_count = max(
                    1, int(self.config.recall.access_decay_max_count)
                )
                access_count_multiplier = max(
                    0.0,
                    min(
                        1.0,
                        float(self.config.recall.access_count_decay_multiplier),
                    ),
                )
                access_window_start = now - access_window_days * 86400
                document_rows = await (
                    await db.execute("SELECT id,metadata FROM documents")
                ).fetchall()
                for row in document_rows:
                    try:
                        metadata = json.loads(row["metadata"] or "{}")
                    except (TypeError, json.JSONDecodeError):
                        metadata = {}
                    importance = max(
                        0.0, min(1.0, float(metadata.get("importance", 0.5)))
                    )
                    created_at = float(metadata.get("create_time", now) or now)
                    age_days = max(0.0, (now - created_at) / 86400)
                    effective_importance = importance
                    if decay_rate > 0:
                        access_count = max(0, int(metadata.get("access_count", 0) or 0))
                        last_access_time = float(metadata.get("last_access_time", 0) or 0)
                        recent_access_factor = (
                            1.0 if last_access_time >= access_window_start else 0.5
                        )
                        access_factor = min(1.0, access_count / max_access_count)
                        effective_decay_rate = decay_rate * (
                            1.0 - 0.5 * access_factor * recent_access_factor
                        )
                        effective_importance = max(
                            0.01,
                            round(importance * (1.0 - effective_decay_rate), 4),
                        )
                        if (
                            effective_importance != importance
                            or access_count_multiplier < 1.0
                        ):
                            metadata["importance"] = effective_importance
                            metadata["access_count"] = int(
                                access_count * access_count_multiplier
                            )
                            await db.execute(
                                "UPDATE documents SET metadata=? WHERE id=?",
                                (
                                    json.dumps(metadata, ensure_ascii=False),
                                    int(row["id"]),
                                ),
                            )
                            decayed += 1
                    if (
                        self.config.maintenance.auto_cleanup_enabled
                        and age_days
                        >= self.config.maintenance.cleanup_days_threshold
                        and effective_importance
                        < self.config.maintenance.cleanup_importance_threshold
                    ):
                        cleanup_ids.append(int(row["id"]))
            await db.execute(
                """UPDATE memory_atoms SET status='expired'
                WHERE status='active' AND expires_at<=?""",
                (now,),
            )
            await db.execute(
                """UPDATE memory_atoms SET status='forgotten'
                WHERE status='expired' AND expires_at<=?""",
                (forgot_before,),
            )
            rows = await (
                await db.execute(
                    "SELECT id FROM memory_atoms WHERE status='forgotten' AND expires_at<=?",
                    (purge_before,),
                )
            ).fetchall()
            for row in rows:
                await db.execute(
                    "DELETE FROM memory_atoms_fts WHERE atom_id=?",
                    (int(row["id"]),),
                )
            await db.execute(
                "DELETE FROM memory_atoms WHERE status='forgotten' AND expires_at<=?",
                (purge_before,),
            )
            await db.commit()
        cleaned = 0
        if run_daily and cleanup_ids:
            cleaned = await self.delete_memories(cleanup_ids)
        backup_path = None
        if run_daily and self.config.maintenance.backup_enabled:
            backup_path = str(await self.backup())
        backup_root = self.data_dir / "backups"
        removed_backups = 0
        if run_daily and backup_root.exists():
            cutoff = now - self.config.maintenance.backup_keep_days * 86400
            for path in backup_root.iterdir():
                if path.is_dir() and path.stat().st_mtime < cutoff:
                    shutil.rmtree(path)
                    removed_backups += 1
        result = {
            "decayed_memories": decayed,
            "cleaned_memories": cleaned,
            "purged_atoms": len(rows),
            "backup": backup_path,
            "removed_backups": removed_backups,
            "daily_maintenance_ran": run_daily,
        }
        state.update(
            {
                "version": 1,
                "library_id": self.library_id,
                "last_maintenance_at": time.time(),
                "result": result,
            }
        )
        if run_daily:
            state["last_daily_maintenance_date"] = today
            state["last_decay_date"] = today
        self._write_decay_state(state)
        return result
