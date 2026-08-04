from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from ...config import AppConfig, IndexRebuildSettings
from .atoms import classify_memory_atoms
from .graph import GraphBuilder
from .indexes import IndexManager, provider_functional_sha256
from ...faiss_runtime import FaissRuntimeError
from ...logger import logger
from .migration import LivingMemoryMigrator, sqlite_backup
from ...control import ProviderRevision
from ...providers import build_provider, build_rerank_provider, provider_config_hash
from .retrieval import RetrievalEngine, SearchResult
from .storage import Storage, normalize_document_metadata
from ...sqlite_pool import SQLiteConnectionPool
from ...task_control import JobControlSignal, JobInterrupted
from .text import TextProcessor


RERANK_EVIDENCE_VERSION = "livingmemory_v8_2_5_3"
RERANK_ATOM_MAX_EVIDENCE = 4
RERANK_GRAPH_MAX_EVIDENCE = 3
RERANK_ATOM_ENTRY_CHAR_LIMIT = 260
RERANK_GRAPH_ENTRY_CHAR_LIMIT = 240
RERANK_GRAPH_DOCUMENT_CHAR_LIMIT = 1200
RERANK_PERSONA_CHAR_LIMIT = 700
RERANK_DOCUMENT_TOTAL_CHAR_LIMIT = 4800


async def _finish_mutation_step(
    awaitable: Awaitable[Any],
) -> tuple[Any, asyncio.CancelledError | None]:
    """Finish one state-changing awaitable before observing caller cancellation.

    SQLite and index mutations can commit after the awaiting task is cancelled.  A
    shielded child lets the replacement state machine learn the actual result and
    perform the correct rollback instead of guessing whether the step committed.
    """

    task = asyncio.ensure_future(awaitable)
    interrupted: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            if interrupted is None:
                interrupted = exc
    return task.result(), interrupted


async def _finish_mutation_cleanup(awaitable: Awaitable[Any]) -> Any:
    """Run rollback to completion even if the caller repeats cancellation."""

    task = asyncio.ensure_future(awaitable)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise
    return task.result()


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
            node_id for node_id, degree in degrees.items() if degree < minimum_degree
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
        memory_store_id: str | None = None,
        library_id: str = "",
        default_persona_id: str = "",
        provider_revision: ProviderRevision | None = None,
        rerank_provider_revision: ProviderRevision | None = None,
        system_path: Path | None = None,
        system_pool: SQLiteConnectionPool | None = None,
    ):
        self.root = root
        self.data_dir = data_dir or (root / "data")
        self.memory_store_id = str(
            memory_store_id if memory_store_id is not None else library_id
        )
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
        self.storage = Storage(
            self.data_dir,
            system_path=system_path,
            pooled=True,
            system_pool=system_pool,
            initialize_system=False,
        )
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
            library_id=self.memory_store_id,
            provider_id=self.provider_revision.provider_id,
            provider_revision=self.provider_revision.revision,
            provider_config_sha256=self.provider_revision.config_sha256,
        )
        self.retrieval = RetrievalEngine(
            self.storage, self.indexes, self.text, config.recall
        )
        self.migrator = LivingMemoryMigrator(self.data_dir)
        self._maintenance_task: asyncio.Task | None = None
        self._index_maintenance_task: asyncio.Task | None = None
        self._mutation_lock = asyncio.Lock()
        self._rebuild_lock = asyncio.Lock()
        self._index_maintenance: dict[str, Any] = {
            "status": "idle",
            "stage": "idle",
            "progress": 0.0,
            "active_generation": None,
            "candidate_generation": None,
            "index_available": False,
            "error_category": None,
            "error": None,
            "request_id": None,
            "updated_at": time.time(),
        }
        self._activation_journal_path = self.data_dir / "maintenance-activation.json"
        self._retired_providers: list[Any] = []
        self._retired_rerankers: list[Any] = []

    @property
    def library_id(self) -> str:
        """Deprecated compatibility alias for frozen manifests and extensions."""

        return self.memory_store_id

    @library_id.setter
    def library_id(self, value: str) -> None:
        """Map legacy assignments onto the canonical memory-store identity."""

        self.memory_store_id = str(value or "")

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
        prepared.pop("memory_type", None)
        prepared["metadata"] = normalize_document_metadata(prepared.get("metadata"))
        requested_persona_id = str(prepared.get("persona_id") or "").strip()
        fallback_persona_id = str(getattr(self, "default_persona_id", "") or "").strip()
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

    def maintenance_status(self) -> dict[str, Any]:
        status = dict(self._index_maintenance)
        indexes = self.indexes.status() or {}
        status["active_generation"] = indexes.get("generation")
        status["index_available"] = bool(
            indexes.get("generation")
            or int(indexes.get("document_vectors") or 0) > 0
            or int(indexes.get("graph_vectors") or 0) > 0
        )
        return status

    def _set_index_maintenance(self, **updates: Any) -> None:
        self._index_maintenance.update(updates)
        self._index_maintenance["updated_at"] = time.time()

    @staticmethod
    def _maintenance_error_category(exc: Exception) -> str:
        if isinstance(exc, FaissRuntimeError):
            return exc.code
        name = type(exc).__name__.lower()
        message = str(exc).lower()
        if "provider" in name or "provider" in message or "embedding" in message:
            return "provider_error"
        if isinstance(exc, asyncio.CancelledError):
            return "cancelled"
        return "runtime_error"

    def _write_activation_journal(self, payload: dict[str, Any]) -> None:
        target = self._activation_journal_path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, target)

    async def _recover_pending_activation(self) -> None:
        if not self._activation_journal_path.exists():
            return
        try:
            journal = json.loads(
                self._activation_journal_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            self._set_index_maintenance(
                status="failed",
                stage="activation_recovery",
                error_category="activation_journal_corrupt",
                error=str(exc),
            )
            return
        if journal.get("phase") != "database_replaced":
            return
        generation = str(journal.get("candidate_generation") or "")
        candidate_dir = self.indexes.root / generation
        try:
            prepared = await self.indexes.prepare_external_generation(
                candidate_dir,
                self.provider,
            )
            await self.indexes.activate_prepared_generation(prepared)
            self._write_activation_journal(
                {
                    **journal,
                    "phase": "activated",
                    "recovered_at": time.time(),
                }
            )
        except Exception as exc:
            self._set_index_maintenance(
                status="partial",
                stage="activation_recovery",
                error_category=self._maintenance_error_category(exc),
                error=str(exc),
            )

    async def _storage_signatures(self, storage: Storage) -> dict[int, str]:
        signatures: dict[int, str] = {}
        async for batch in storage.iter_documents(
            batch_size=500,
            active_only=False,
        ):
            for item in batch:
                memory_id = int(item["id"])
                encoded = json.dumps(
                    {
                        "doc_id": item.get("doc_id"),
                        "text": item.get("text"),
                        "metadata": item.get("metadata") or {},
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
                import hashlib

                signatures[memory_id] = hashlib.sha256(encoded).hexdigest()
        return signatures

    async def _shadow_changed_ids(self, shadow: Storage) -> list[int]:
        live, candidate = await asyncio.gather(
            self._storage_signatures(self.storage),
            self._storage_signatures(shadow),
        )
        return sorted(
            memory_id
            for memory_id in set(live) | set(candidate)
            if live.get(memory_id) != candidate.get(memory_id)
        )

    async def _reconcile_shadow(
        self,
        shadow: Storage,
        shadow_indexes: IndexManager,
    ) -> list[int]:
        changed_ids = await self._shadow_changed_ids(shadow)
        if not changed_ids:
            return []
        active_ids: list[int] = []
        inactive_ids: list[int] = []
        for memory_id in changed_ids:
            state = await shadow.sync_memory_from(
                self.storage,
                memory_id,
                self.text.tokenize,
                self._graph_builder_for_write(),
            )
            if state == "active":
                active_ids.append(memory_id)
            else:
                inactive_ids.append(memory_id)
        del inactive_ids
        current_document_ids, current_graph_ids = shadow_indexes.indexed_ids()
        add_documents = await shadow.documents_for_ids(active_ids)
        add_graph = await shadow.graph_memories_for_ids(active_ids)
        await shadow_indexes.apply_delta(
            add_documents=add_documents,
            add_graph_entries=add_graph,
            remove_document_ids=set(changed_ids) & current_document_ids,
            remove_graph_entry_ids=set(changed_ids) & current_graph_ids,
            expected_document_ids_after=set(await shadow.document_ids()),
            expected_graph_ids_after=set(await shadow.graph_memory_ids()),
            reason="shadow_incremental_replay",
        )
        return changed_ids

    async def _index_check_reason(self) -> str:
        expected_documents = set(await self.storage.document_ids())
        expected_graph = set(await self.storage.graph_memory_ids())
        actual_documents, actual_graph = self.indexes.indexed_ids()
        manifest = self.indexes.manifest or {}
        if not manifest:
            return "missing_generation" if expected_documents or expected_graph else ""
        if actual_documents != expected_documents:
            return "document_id_set_changed"
        if self.indexes.graph_vector_granularity() != "memory":
            return "legacy_graph_vector_granularity"
        if actual_graph != expected_graph:
            return "graph_id_set_changed"
        dimension = await self.provider.get_dimension()
        if int(manifest.get("dimension") or 0) != int(dimension):
            return "provider_dimension_changed"
        current_functional = provider_functional_sha256(self.provider)
        recorded_functional = str(manifest.get("provider_functional_sha256") or "")
        if not recorded_functional:
            await self.indexes.adopt_provider_functional_sha256(current_functional)
        elif recorded_functional != current_functional:
            return "provider_functional_fingerprint_changed"
        graph_report, fts_report = await asyncio.gather(
            self.storage.graph_integrity_report(),
            self.storage.fts_integrity_report(),
        )
        return self._graph_recovery_reason(graph_report) or self._fts_recovery_reason(
            fts_report
        )

    async def _startup_index_check(self) -> None:
        await asyncio.sleep(0)
        try:
            self._set_index_maintenance(
                status="checking",
                stage="integrity_check",
                progress=0.0,
                error=None,
                error_category=None,
            )
            reason = await self._index_check_reason()
            if reason:
                await self._continuous_rebuild(reason=reason)
            else:
                self._set_index_maintenance(
                    status="ready",
                    stage="ready",
                    progress=1.0,
                )
        except asyncio.CancelledError:
            self._set_index_maintenance(
                status="cancelled",
                stage="cancelled",
            )
            raise
        except Exception as exc:
            available = bool((self.indexes.status() or {}).get("generation"))
            self._set_index_maintenance(
                status="partial" if available else "failed",
                stage="failed",
                error_category=self._maintenance_error_category(exc),
                error=str(exc),
            )

    async def initialize(self, *, start_index_check: bool = True) -> None:
        logger.info(
            "初始化记忆库 runtime：memory_store_id=%s data_dir=%s provider=%s revision=%s",
            self.memory_store_id,
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
                        "library_id": self.memory_store_id,
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
        await self._recover_pending_activation()
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        if start_index_check:
            self.start_background_index_check()
        logger.info(
            "记忆库 runtime 初始化完成：memory_store_id=%s generation=%s",
            self.memory_store_id,
            (self.indexes.status() or {}).get("generation") or "",
        )

    def start_background_index_check(self) -> None:
        task = self._index_maintenance_task
        if task is not None and not task.done():
            return
        self._index_maintenance_task = asyncio.create_task(
            self._startup_index_check(),
            name=f"livingmemory-index-check-{self.memory_store_id}",
        )

    async def close(self) -> None:
        logger.info("关闭记忆库 runtime：memory_store_id=%s", self.memory_store_id)
        if self._maintenance_task:
            self._maintenance_task.cancel()
            await asyncio.gather(self._maintenance_task, return_exceptions=True)
        if self._index_maintenance_task:
            self._index_maintenance_task.cancel()
            await asyncio.gather(
                self._index_maintenance_task,
                return_exceptions=True,
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
        await self.storage.close()

    async def cancel_startup_index_check(self) -> None:
        """Yield the rebuild lock to an explicit, user-visible maintenance job."""

        task = self._index_maintenance_task
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._index_maintenance_task = None

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
            "Rerank Provider 已切换：memory_store_id=%s provider=%s",
            self.memory_store_id,
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
        if not (
            self.config.recall.graph_memory_enabled
            or self.config.maintenance.atom_enabled
        ):
            return await self._apply_plain_rerank(
                query, candidates, k, provider_id, provider_type
            )
        try:
            evidence_signals = await self._rerank_evidence_signals(query, candidates)
        except Exception as exc:
            logger.warning(
                "Rerank 结构化证据增强失败，回退纯文本重排：memory_store_id=%s err=%s",
                self.memory_store_id,
                exc,
            )
            return await self._apply_plain_rerank(
                query, candidates, k, provider_id, provider_type
            )
        evidence_candidate_count = sum(
            1
            for signal in evidence_signals.values()
            if self._rerank_signal_has_evidence(signal)
        )
        if evidence_candidate_count <= 0:
            return await self._apply_plain_rerank(
                query, candidates, k, provider_id, provider_type
            )
        return await self._apply_evidence_enhanced_rerank(
            query,
            candidates,
            k,
            provider_id,
            provider_type,
            evidence_signals,
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
            if (
                row.index < 0
                or row.index >= len(candidates)
                or row.index in used_indexes
            ):
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
            "evidence_enhanced": False,
            "evidence_version": RERANK_EVIDENCE_VERSION,
            "evidence_candidate_count": 0,
            "atom_candidate_count": 0,
            "atom_signal_candidate_count": 0,
            "graph_enhanced": False,
            "graph_evidence_candidate_count": 0,
            "graph_candidate_count": 0,
        }

    async def _apply_evidence_enhanced_rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        k: int,
        provider_id: str,
        provider_type: str,
        evidence_signals: dict[int, dict[str, Any]],
    ) -> tuple[list[SearchResult], dict[str, Any]]:
        enhanced_documents = [
            self._rerank_document_with_evidence(
                item,
                evidence_signals.get(item.doc_id, {}),
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
            if (
                row.index < 0
                or row.index >= len(candidates)
                or row.index in used_indexes
            ):
                continue
            used_indexes.add(row.index)
            valid_rows.append(row)
        raw_by_index = {row.index: float(row.relevance_score) for row in valid_rows}
        rank_by_index = {row.index: rank for rank, row in enumerate(valid_rows)}
        raw_values = list(raw_by_index.values())
        raw_low = min(raw_values) if raw_values else 0.0
        raw_high = max(raw_values) if raw_values else 0.0
        raw_span = raw_high - raw_low
        doc_weight, graph_weight, intent = self._rerank_route_weights(query)
        if not self.config.recall.graph_memory_enabled:
            doc_weight, graph_weight = 1.0, 0.0
        graph_signal_candidate_count = sum(
            1
            for signal in evidence_signals.values()
            if float(signal.get("graph_score") or 0.0) > 0.0
        )
        graph_evidence_candidate_count = sum(
            1
            for signal in evidence_signals.values()
            if list(signal.get("graph_entries") or signal.get("entries") or [])
        )
        atom_signal_candidate_count = sum(
            1
            for signal in evidence_signals.values()
            if float(signal.get("atom_score") or 0.0) > 0.0
        )
        atom_candidate_count = sum(
            1
            for signal in evidence_signals.values()
            if list(signal.get("atom_entries") or [])
        )
        structured_metadata_candidate_count = sum(
            1
            for signal in evidence_signals.values()
            if signal.get("persona_summary")
            or signal.get("topics")
            or signal.get("participants")
        )
        evidence_candidate_count = sum(
            1
            for signal in evidence_signals.values()
            if self._rerank_signal_has_evidence(signal)
        )
        scored: list[tuple[float, float, float, float, int, SearchResult]] = []
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
                normalized_raw = (
                    1.0 if raw_span == 0 else (raw_score - raw_low) / raw_span
                )
                rank = rank_by_index.get(index, returned_count - 1)
                rank_score = (
                    1.0
                    if returned_count <= 1
                    else 1.0 - (rank / max(1, returned_count - 1))
                )
            text_score = max(0.0, min(1.0, 0.7 * normalized_raw + 0.3 * rank_score))
            signal = evidence_signals.get(candidate.doc_id, {})
            atom_score = max(0.0, min(1.0, float(signal.get("atom_score") or 0.0)))
            metadata_score = max(
                0.0, min(1.0, float(signal.get("metadata_score") or 0.0))
            )
            memory_evidence_score = max(atom_score, metadata_score)
            memory_score = (
                0.75 * text_score + 0.25 * memory_evidence_score
                if memory_evidence_score > 0.0
                else text_score
            )
            graph_score = max(0.0, min(1.0, float(signal.get("graph_score") or 0.0)))
            overlap_bonus = (
                self.config.recall.cross_route_bonus
                if memory_score > 0.0 and graph_score > 0.0
                else 0.0
            )
            if graph_score > 0.0 and graph_weight > 0.0:
                final = max(
                    0.0,
                    min(
                        1.0,
                        doc_weight * memory_score
                        + graph_weight * graph_score
                        + overlap_bonus,
                    ),
                )
            else:
                final = memory_score
            evidence_score = max(
                0.0,
                min(
                    1.0,
                    doc_weight * memory_evidence_score + graph_weight * graph_score,
                ),
            )
            graph_keyword_score = float(
                signal.get("graph_keyword_score", signal.get("keyword_score", 0.0))
                or 0.0
            )
            graph_vector_score = float(
                signal.get("graph_vector_score", signal.get("vector_score", 0.0)) or 0.0
            )
            graph_node_score = float(
                signal.get("graph_node_score", signal.get("node_score", 0.0)) or 0.0
            )
            graph_entries = list(
                signal.get("graph_entries") or signal.get("entries") or []
            )
            atom_entries = list(signal.get("atom_entries") or [])
            atom_types = sorted(
                {str(entry.get("atom_type") or "unknown") for entry in atom_entries}
            )
            item.score_breakdown = {
                **item.score_breakdown,
                "rerank_evidence_version": RERANK_EVIDENCE_VERSION,
                "rerank_raw_score": round(raw_score, 6),
                "rerank_text_score": round(text_score, 6),
                "rerank_memory_score": round(memory_score, 6),
                "rerank_memory_evidence_score": round(memory_evidence_score, 6),
                "rerank_evidence_score": round(evidence_score, 6),
                "rerank_atom_score": round(atom_score, 6),
                "rerank_atom_keyword_score": round(
                    float(signal.get("atom_keyword_score") or 0.0), 6
                ),
                "rerank_atom_entity_score": round(
                    float(signal.get("atom_entity_score") or 0.0), 6
                ),
                "rerank_atom_quality_score": round(
                    float(signal.get("atom_quality_score") or 0.0), 6
                ),
                "rerank_atom_type_score": round(
                    float(signal.get("atom_type_score") or 0.0), 6
                ),
                "rerank_atom_evidence_count": len(atom_entries),
                "rerank_atom_types": atom_types,
                "rerank_metadata_score": round(metadata_score, 6),
                "rerank_topic_score": round(float(signal.get("topic_score") or 0.0), 6),
                "rerank_participant_score": round(
                    float(signal.get("participant_score") or 0.0), 6
                ),
                "rerank_graph_score": round(graph_score, 6),
                "rerank_graph_keyword_score": round(graph_keyword_score, 6),
                "rerank_graph_vector_score": round(graph_vector_score, 6),
                "rerank_graph_vector_source": str(
                    signal.get("graph_vector_source") or "none"
                ),
                "rerank_graph_vector_granularity": str(
                    signal.get("graph_vector_granularity") or "unknown"
                ),
                "rerank_graph_node_score": round(graph_node_score, 6),
                "rerank_graph_evidence_count": len(graph_entries),
                "evidence_enhanced_final_score": round(final, 6),
                # Kept for API clients that already consume the old diagnostic.
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
            scored.append(
                (
                    final,
                    evidence_score,
                    graph_score,
                    text_score,
                    -index,
                    item,
                )
            )
        scored.sort(
            key=lambda row: (row[0], row[1], row[2], row[3], row[4]),
            reverse=True,
        )
        ordered = [item for *_scores, item in scored[:k]]
        return ordered, {
            "requested": True,
            "applied": bool(rerank_rows),
            "provider_id": provider_id,
            "provider_type": provider_type,
            "candidate_count": len(candidates),
            "returned": len(ordered),
            "provider_returned": len(rerank_rows),
            "evidence_enhanced": True,
            "evidence_version": RERANK_EVIDENCE_VERSION,
            "evidence_candidate_count": evidence_candidate_count,
            "structured_metadata_candidate_count": structured_metadata_candidate_count,
            "atom_candidate_count": atom_candidate_count,
            "atom_signal_candidate_count": atom_signal_candidate_count,
            "graph_enhanced": (
                graph_evidence_candidate_count > 0
                or graph_signal_candidate_count > 0
            ),
            "graph_evidence_candidate_count": graph_evidence_candidate_count,
            "graph_candidate_count": graph_signal_candidate_count,
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

    async def _rerank_evidence_signals(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> dict[int, dict[str, Any]]:
        atom_enabled = bool(self.config.maintenance.atom_enabled)
        graph_enabled = bool(self.config.recall.graph_memory_enabled)
        if atom_enabled and graph_enabled:
            atom_signals, graph_signals = await asyncio.gather(
                self._rerank_atom_signals(query, candidates),
                self._rerank_graph_signals(query, candidates),
            )
        elif atom_enabled:
            atom_signals = await self._rerank_atom_signals(query, candidates)
            graph_signals = {}
        elif graph_enabled:
            atom_signals = {}
            graph_signals = await self._rerank_graph_signals(query, candidates)
        else:
            return {}

        tokens = self.text.tokenize(query)
        signals: dict[int, dict[str, Any]] = {}
        for candidate in candidates:
            metadata = (
                candidate.metadata if isinstance(candidate.metadata, dict) else {}
            )
            persona_summary = str(metadata.get("persona_summary") or "").strip()
            canonical = " ".join(str(candidate.content or "").split()).casefold()
            if " ".join(persona_summary.split()).casefold() == canonical:
                persona_summary = ""
            topics = self._rerank_text_values(metadata.get("topics"))[:6]
            participant_labels, participant_values = self._rerank_participant_labels(
                metadata
            )
            topic_score = self._rerank_token_overlap(tokens, topics)
            participant_score = self._rerank_token_overlap(tokens, participant_values)
            signal = {
                **graph_signals.get(candidate.doc_id, {}),
                **atom_signals.get(candidate.doc_id, {}),
                "persona_summary": persona_summary,
                "topics": topics,
                "participants": participant_labels,
                "topic_score": topic_score,
                "participant_score": participant_score,
                "metadata_score": max(topic_score, participant_score),
            }
            if self._rerank_signal_has_evidence(signal):
                signals[candidate.doc_id] = signal
        return signals

    async def _rerank_atom_signals(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> dict[int, dict[str, Any]]:
        candidate_ids = [item.doc_id for item in candidates]
        raw = await self.storage.candidate_atom_evidence(
            candidate_ids,
            self.text.tokenize(query),
            max_atoms_per_candidate=RERANK_ATOM_MAX_EVIDENCE,
        )
        if not raw:
            return {}
        _document_weight, _graph_weight, intent = self._rerank_route_weights(query)
        candidate_id_set = set(candidate_ids)
        signals: dict[int, dict[str, Any]] = {}
        for memory_id, payload in raw.items():
            normalized_id = int(memory_id)
            if normalized_id not in candidate_id_set:
                continue
            entries = list(payload.get("entries") or [])[:RERANK_ATOM_MAX_EVIDENCE]
            keyword_score = self._bounded_rerank_score(payload.get("keyword_score"))
            entity_score = self._bounded_rerank_score(payload.get("entity_score"))
            quality_score = self._bounded_rerank_score(payload.get("quality_score"))
            matched_entries = [
                entry
                for entry in entries
                if float(entry.get("match_score") or 0.0) > 0.0
            ]
            type_score = max(
                (
                    self._rerank_atom_type_score(
                        intent, str(entry.get("atom_type") or "unknown")
                    )
                    for entry in matched_entries
                ),
                default=0.0,
            )
            atom_score = 0.0
            if max(keyword_score, entity_score) > 0.0:
                atom_score = min(
                    1.0,
                    0.45 * keyword_score
                    + 0.15 * entity_score
                    + 0.25 * quality_score
                    + 0.15 * type_score,
                )
            signals[normalized_id] = {
                "atom_score": atom_score,
                "atom_keyword_score": keyword_score,
                "atom_entity_score": entity_score,
                "atom_quality_score": quality_score,
                "atom_type_score": type_score,
                "atom_entries": entries,
            }
        return signals

    @staticmethod
    def _bounded_rerank_score(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _rerank_atom_type_score(intent: str, atom_type: str) -> float:
        normalized_intent = str(intent or "").casefold()
        normalized_type = str(atom_type or "unknown").casefold()
        if "relationship" in normalized_intent:
            return {
                "relational": 1.0,
                "preference": 0.75,
                "factual": 0.6,
                "episodic": 0.55,
            }.get(normalized_type, 0.45)
        if "temporal" in normalized_intent:
            return {
                "planned": 1.0,
                "episodic": 0.9,
                "factual": 0.65,
                "relational": 0.5,
            }.get(normalized_type, 0.45)
        if "factual" in normalized_intent:
            return {
                "factual": 1.0,
                "preference": 0.65,
                "relational": 0.6,
                "episodic": 0.55,
            }.get(normalized_type, 0.5)
        return {
            "factual": 0.8,
            "relational": 0.8,
            "preference": 0.8,
            "planned": 0.75,
            "episodic": 0.7,
        }.get(normalized_type, 0.55)

    @staticmethod
    def _rerank_text_values(value: Any) -> list[str]:
        if isinstance(value, dict):
            items = [*value.keys(), *value.values()]
        elif isinstance(value, (list, tuple, set)):
            items = list(value)
        elif value:
            items = [value]
        else:
            items = []
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item).strip()
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            result.append(text)
        return result

    @classmethod
    def _rerank_participant_labels(
        cls, metadata: dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        identities = metadata.get("participant_identities")
        if isinstance(identities, dict):
            if identities and all(
                isinstance(value, dict) for value in identities.values()
            ):
                identity_items = list(identities.values())
            else:
                identity_items = [identities]
        elif isinstance(identities, list):
            identity_items = identities
        else:
            identity_items = []
        labels: list[str] = []
        searchable: list[str] = []
        for identity in identity_items:
            if not isinstance(identity, dict):
                continue
            display_name = str(
                identity.get("display_name")
                or identity.get("latest_display_name")
                or identity.get("name")
                or ""
            ).strip()
            aliases = cls._rerank_text_values(identity.get("aliases"))
            if display_name:
                searchable.append(display_name)
            searchable.extend(aliases)
            if not display_name and aliases:
                display_name = aliases[0]
                aliases = aliases[1:]
            if not display_name:
                continue
            label = display_name
            unique_aliases = [
                alias
                for alias in aliases
                if alias.casefold() != display_name.casefold()
            ]
            if unique_aliases:
                label += f" (aliases: {', '.join(unique_aliases[:4])})"
            if bool(identity.get("is_bot")):
                label += " [bot]"
            labels.append(label)
        for participant in cls._rerank_text_values(metadata.get("participants")):
            searchable.append(participant)
            if all(participant.casefold() not in label.casefold() for label in labels):
                labels.append(participant)
        return cls._rerank_text_values(labels)[:8], cls._rerank_text_values(searchable)

    @staticmethod
    def _rerank_token_overlap(tokens: list[str], values: list[str]) -> float:
        normalized_tokens = [
            str(token).strip().casefold() for token in tokens if str(token).strip()
        ]
        if not normalized_tokens or not values:
            return 0.0
        searchable = "\n".join(str(value) for value in values).casefold()
        matched = sum(1 for token in normalized_tokens if token in searchable)
        return max(0.0, min(1.0, matched / len(normalized_tokens)))

    @staticmethod
    def _rerank_signal_has_evidence(signal: dict[str, Any]) -> bool:
        return bool(
            signal.get("persona_summary")
            or signal.get("topics")
            or signal.get("participants")
            or signal.get("atom_entries")
            or signal.get("graph_entries")
            or signal.get("entries")
        )

    async def _rerank_graph_signals(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> dict[int, dict[str, Any]]:
        candidate_ids = [item.doc_id for item in candidates]
        raw = await self.storage.candidate_graph_evidence(
            candidate_ids,
            self.text.tokenize(query),
            max_entries_per_candidate=RERANK_GRAPH_MAX_EVIDENCE,
            graph_expansion_limit=self.config.recall.graph_expansion_limit,
            graph_expansion_hops=self.config.recall.graph_expansion_hops,
            graph_second_hop_weight=self.config.recall.graph_second_hop_weight,
        )
        granularity = "memory"
        graph_granularity = getattr(
            getattr(self, "indexes", None), "graph_vector_granularity", None
        )
        if callable(graph_granularity):
            try:
                granularity = str(graph_granularity() or "unknown")
            except Exception:
                granularity = "unknown"
        signals: dict[int, dict[str, Any]] = {}
        for candidate in candidates:
            payload = raw.get(candidate.doc_id, {})
            keyword_score = self._bounded_rerank_score(payload.get("keyword_score"))
            node_score = self._bounded_rerank_score(payload.get("node_score"))
            vector_score = self._bounded_rerank_score(
                candidate.score_breakdown.get("graph_vector_score")
            )
            confidence = self._bounded_rerank_score(payload.get("graph_confidence"))
            has_match = max(keyword_score, node_score, vector_score) > 0.0
            if confidence <= 0 and has_match:
                confidence = 0.7
            graph_score = (
                min(
                    1.0,
                    0.40 * keyword_score
                    + 0.30 * vector_score
                    + 0.20 * node_score
                    + 0.10 * confidence,
                )
                if has_match
                else 0.0
            )
            entries = list(payload.get("entries") or [])[:RERANK_GRAPH_MAX_EVIDENCE]
            if not has_match:
                continue
            signals[candidate.doc_id] = {
                "graph_keyword_score": keyword_score,
                "graph_node_score": node_score,
                "graph_vector_score": vector_score,
                "graph_vector_source": (
                    (
                        "recall_memory_index"
                        if granularity == "memory"
                        else "recall_graph_index"
                    )
                    if vector_score > 0.0
                    else "none"
                ),
                "graph_vector_granularity": granularity,
                "graph_confidence": confidence,
                "graph_score": graph_score,
                "graph_entries": entries,
            }
        return signals

    @staticmethod
    def _rerank_document_with_evidence(
        candidate: SearchResult,
        signal: dict[str, Any],
    ) -> str:
        sections = [
            "[Canonical memory]\n"
            + str(candidate.content or "")[:RERANK_GRAPH_DOCUMENT_CHAR_LIMIT]
        ]
        persona_summary = str(signal.get("persona_summary") or "").strip()
        if persona_summary:
            sections.append(
                "[Persona perspective]\n" + persona_summary[:RERANK_PERSONA_CHAR_LIMIT]
            )

        atom_lines: list[str] = []
        for entry in list(signal.get("atom_entries") or [])[:RERANK_ATOM_MAX_EVIDENCE]:
            content = str(entry.get("content") or "").strip()
            if not content:
                continue
            atom_type = str(entry.get("atom_type") or "unknown")
            atom_lines.append(
                f"- {atom_type}: {content[:RERANK_ATOM_ENTRY_CHAR_LIMIT]}"
            )
        if atom_lines:
            sections.append("[Active atomic facts]\n" + "\n".join(atom_lines))

        participants = [
            str(item).strip()
            for item in list(signal.get("participants") or [])[:8]
            if str(item).strip()
        ]
        if participants:
            sections.append(
                "[Stable participants]\n"
                + "\n".join(f"- {item}" for item in participants)
            )

        topics = [
            str(item).strip()
            for item in list(signal.get("topics") or [])[:6]
            if str(item).strip()
        ]
        if topics:
            sections.append("[Topics]\n" + "\n".join(f"- {item}" for item in topics))

        graph_lines: list[str] = []
        graph_entries = list(
            signal.get("graph_entries") or signal.get("entries") or []
        )[:RERANK_GRAPH_MAX_EVIDENCE]
        for entry in graph_entries:
            content = str(entry.get("content") or "").strip()
            if not content:
                continue
            relation = str(
                entry.get("relation_type") or entry.get("entry_type") or "graph"
            )
            graph_lines.append(
                f"- {relation}: {content[:RERANK_GRAPH_ENTRY_CHAR_LIMIT]}"
            )
        if graph_lines:
            sections.append("[Graph relations]\n" + "\n".join(graph_lines))

        # Intentionally excludes memory_sources/source_messages.  Rerank receives
        # only the canonical candidate and its derived, library-owned evidence.
        return "\n\n".join(sections)[:RERANK_DOCUMENT_TOTAL_CHAR_LIMIT]

    async def rebuild_indexes(
        self,
        progress=None,
        *,
        job_context=None,
        checkpoint_dir=None,
        before_database_repair: Callable[[], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        await self.cancel_startup_index_check()
        return await self._continuous_rebuild(
            progress=progress,
            job_context=job_context,
            checkpoint_dir=checkpoint_dir,
            before_database_repair=before_database_repair,
            reason="manual_index_rebuild",
        )

    async def _continuous_rebuild(
        self,
        progress=None,
        *,
        job_context=None,
        checkpoint_dir=None,
        before_database_repair: Callable[[], Awaitable[None]] | None = None,
        reason: str,
        provider=None,
        provider_revision: ProviderRevision | None = None,
    ) -> dict[str, Any]:
        """Build graph/FTS/FAISS in shadow state and switch at a short barrier."""

        async with self._rebuild_lock:
            candidate_provider = provider or self.provider
            candidate_revision = provider_revision or self.provider_revision
            request_id = str(getattr(job_context, "job_id", "") or "") or None
            job_kind = ""
            if job_context is not None:
                job_kind = str((await job_context.job()).get("kind") or "")
            allow_concurrent_replay = job_kind not in {
                "livingmemory_import",
                "livingmemory_migration",
            }
            owned_work_dir = checkpoint_dir is None
            if checkpoint_dir is None:
                parent = self.data_dir / "indexes" / "maintenance-work"
                parent.mkdir(parents=True, exist_ok=True)
                work_dir = Path(tempfile.mkdtemp(prefix="shadow-", dir=str(parent)))
            else:
                work_dir = Path(checkpoint_dir) / "continuous-shadow"
                work_dir.mkdir(parents=True, exist_ok=True)
            shadow_data = work_dir / "data"
            shadow_data.mkdir(parents=True, exist_ok=True)
            shadow_db = shadow_data / "livingmemory.db"
            success = False
            shadow: Storage | None = None
            try:
                self._set_index_maintenance(
                    status="checking",
                    stage="snapshot",
                    progress=0.0,
                    request_id=request_id,
                    candidate_generation=None,
                    error=None,
                    error_category=None,
                )
                if before_database_repair is not None:
                    marker = work_dir / "before-database-repair.done"
                    if not marker.exists():
                        await before_database_repair()
                        marker.write_text(str(time.time()), encoding="utf-8")
                if not shadow_db.exists():
                    async with self._mutation_lock:
                        await asyncio.to_thread(
                            sqlite_backup,
                            self.storage.db_path,
                            shadow_db,
                        )
                shadow = Storage(
                    shadow_data,
                    system_path=self.storage.system_path,
                    initialize_system=False,
                )
                await shadow.initialize()
                if not allow_concurrent_replay and (
                    await self._storage_signatures(self.storage)
                    != await self._storage_signatures(shadow)
                ):
                    raise JobInterrupted(
                        "source_changed",
                        "导入或迁移任务暂停期间源数据已变化，拒绝不安全续跑",
                    )

                derivatives_marker = work_dir / "shadow-derivatives.json"
                if derivatives_marker.exists():
                    derivatives = json.loads(
                        derivatives_marker.read_text(encoding="utf-8")
                    )
                    graph_before = dict(derivatives.get("graph_before") or {})
                    graph = dict(derivatives.get("graph") or {})
                    fts = dict(derivatives.get("fts") or {})
                    graph_after = dict(derivatives.get("graph_after") or {})
                else:
                    self._set_index_maintenance(
                        status="rebuilding",
                        stage="shadow_graph",
                        progress=0.05,
                    )

                    async def graph_progress(value: float, message: str) -> None:
                        mapped = 0.05 + max(0.0, min(1.0, value)) * 0.15
                        self._set_index_maintenance(
                            status="rebuilding",
                            stage="shadow_graph",
                            progress=mapped,
                        )
                        if progress:
                            await progress(mapped, message)

                    graph_before = await shadow.graph_integrity_report()
                    graph = await shadow.rebuild_graph_from_documents(
                        self.text.tokenize,
                        self._graph_builder_for_write(),
                        graph_progress,
                    )
                    fts = await shadow.rebuild_fts(self.text.tokenize)
                    graph_after = await shadow.graph_integrity_report()
                    derivatives_marker.write_text(
                        json.dumps(
                            {
                                "graph_before": graph_before,
                                "graph": graph,
                                "fts": fts,
                                "graph_after": graph_after,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

                shadow_indexes = IndexManager(
                    shadow_data,
                    shadow,
                    candidate_provider,
                    candidate_revision.config.model,
                    library_id=self.memory_store_id,
                    provider_id=candidate_revision.provider_id,
                    provider_revision=candidate_revision.revision,
                    provider_config_sha256=candidate_revision.config_sha256,
                )
                await shadow_indexes.initialize()

                async def index_progress(value: float, message: str) -> None:
                    mapped = 0.20 + max(0.0, min(1.0, value)) * 0.65
                    self._set_index_maintenance(
                        status="rebuilding",
                        stage="shadow_vectors",
                        progress=mapped,
                    )
                    if progress:
                        await progress(mapped, message)

                await shadow_indexes.rebuild(
                    **self._index_rebuild_settings(
                        candidate_revision.config.index_rebuild_settings
                    ),
                    progress=index_progress,
                    provider=candidate_provider,
                    library_id=self.memory_store_id,
                    provider_id=candidate_revision.provider_id,
                    provider_revision=candidate_revision.revision,
                    provider_config_sha256=candidate_revision.config_sha256,
                    provider_model=candidate_revision.config.model,
                    job_context=job_context,
                    checkpoint_dir=(
                        (
                            Path(checkpoint_dir)
                            if checkpoint_dir is not None
                            else work_dir / "index-checkpoint"
                        )
                        if job_context is not None
                        else None
                    ),
                )

                self._set_index_maintenance(
                    status="rebuilding",
                    stage="incremental_replay",
                    progress=0.87,
                )
                replayed = await self._reconcile_shadow(shadow, shadow_indexes)
                generation = str(
                    (shadow_indexes.status() or {}).get("generation") or ""
                )
                candidate_dir = shadow_indexes.root / generation
                prepared = await self.indexes.prepare_external_generation(
                    candidate_dir,
                    candidate_provider,
                )
                self._set_index_maintenance(
                    status="rebuilding",
                    stage="activation_barrier",
                    progress=0.94,
                    candidate_generation=generation,
                )

                rollback_db = work_dir / "activation-rollback.db"
                async with self._mutation_lock:
                    if not allow_concurrent_replay and (
                        await self._storage_signatures(self.storage)
                        != await self._storage_signatures(shadow)
                    ):
                        raise JobInterrupted(
                            "source_changed",
                            "导入或迁移任务激活前源数据已变化，拒绝不安全激活",
                        )
                    final_replayed = await self._reconcile_shadow(
                        shadow,
                        shadow_indexes,
                    )
                    if final_replayed:
                        generation = str(
                            (shadow_indexes.status() or {}).get("generation") or ""
                        )
                        candidate_dir = shadow_indexes.root / generation
                        prepared = await self.indexes.prepare_external_generation(
                            candidate_dir,
                            candidate_provider,
                        )
                        self._set_index_maintenance(
                            candidate_generation=generation,
                        )
                    await asyncio.to_thread(
                        sqlite_backup,
                        self.storage.db_path,
                        rollback_db,
                    )
                    previous_generation = str(
                        (self.indexes.status() or {}).get("generation") or ""
                    )
                    journal = {
                        "version": 1,
                        "phase": "prepared",
                        "reason": reason,
                        "previous_generation": previous_generation,
                        "candidate_generation": generation,
                        "rollback_database": str(rollback_db),
                        "request_id": request_id,
                        "saved_at": time.time(),
                    }
                    await asyncio.to_thread(
                        self._write_activation_journal,
                        journal,
                    )
                    try:
                        await self.storage.replace_search_derivatives_from(
                            shadow.db_path
                        )
                        journal = {
                            **journal,
                            "phase": "database_replaced",
                            "saved_at": time.time(),
                        }
                        await asyncio.to_thread(
                            self._write_activation_journal,
                            journal,
                        )
                        manifest = await self.indexes.activate_prepared_generation(
                            prepared
                        )
                    except Exception:
                        await self.storage.replace_search_derivatives_from(rollback_db)
                        await asyncio.to_thread(
                            self._write_activation_journal,
                            {
                                **journal,
                                "phase": "rolled_back",
                                "saved_at": time.time(),
                            },
                        )
                        raise
                    await asyncio.to_thread(
                        self._write_activation_journal,
                        {
                            **journal,
                            "phase": "activated",
                            "saved_at": time.time(),
                        },
                    )

                self.retrieval.invalidate()
                self._set_index_maintenance(
                    status="ready",
                    stage="ready",
                    progress=1.0,
                    active_generation=manifest.get("generation"),
                    candidate_generation=None,
                    error=None,
                    error_category=None,
                )
                if progress:
                    await progress(1.0, "影子索引与派生数据已原子切换")
                success = True
                return {
                    "graph_recovery": {
                        "rebuilt": True,
                        "reason": reason,
                        "graph": graph,
                        "before": graph_before,
                        "after": graph_after,
                    },
                    "graph": {
                        "graph": graph,
                        "before": graph_before,
                        "after": graph_after,
                    },
                    "fts": {"rebuilt": True, "reason": reason, "rows": fts},
                    "manifest": manifest,
                    "replayed_memory_ids": sorted(set(replayed + final_replayed)),
                }
            except asyncio.CancelledError:
                self._set_index_maintenance(
                    status="cancelled",
                    stage="cancelled",
                    error_category="cancelled",
                )
                raise
            except Exception as exc:
                available = bool((self.indexes.status() or {}).get("generation"))
                self._set_index_maintenance(
                    status="partial" if available else "failed",
                    stage="failed",
                    error_category=self._maintenance_error_category(exc),
                    error=str(exc),
                )
                raise
            finally:
                if shadow is not None:
                    await shadow.close()
                if owned_work_dir and success:
                    await asyncio.to_thread(shutil.rmtree, work_dir, True)

    @staticmethod
    def _graph_recovery_reason(report: dict[str, Any]) -> str:
        active_documents = int(
            report.get("active_documents", report.get("documents", 0)) or 0
        )
        if active_documents <= 0:
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
        active_documents = int(
            report.get("active_documents", report.get("documents", 0)) or 0
        )
        active_graph_entries = int(
            report.get(
                "active_graph_entries",
                report.get("graph_entries", 0),
            )
            or 0
        )
        if active_documents != int(report.get("document_fts") or 0):
            return "document_fts_count_mismatch"
        if active_graph_entries != int(report.get("graph_fts") or 0):
            return "graph_fts_count_mismatch"
        return ""

    async def _ensure_graph_recovery_unlocked(
        self,
        progress=None,
        *,
        before_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        before = before_report or await self.storage.graph_integrity_report()
        reason = self._graph_recovery_reason(before)
        if not reason:
            return {
                "rebuilt": False,
                "reason": "not_needed",
                "before": before,
                "after": before,
            }
        logger.warning(
            "检测到图记忆派生数据需要恢复：memory_store_id=%s reason=%s report=%s",
            self.memory_store_id,
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
            "图记忆派生数据已恢复：memory_store_id=%s reason=%s result=%s after=%s",
            self.memory_store_id,
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
        before_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        before = before_report or await self.storage.fts_integrity_report()
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
            "检测到 FTS 派生数据需要恢复：memory_store_id=%s reason=%s report=%s",
            self.memory_store_id,
            reason,
            before,
        )
        rows = await self.storage.rebuild_fts(self.text.tokenize)
        after = await self.storage.fts_integrity_report()
        logger.warning(
            "FTS 派生数据已恢复：memory_store_id=%s reason=%s rows=%s after=%s",
            self.memory_store_id,
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
            "图记忆派生数据已强制重建：memory_store_id=%s result=%s before=%s after=%s",
            self.memory_store_id,
            graph,
            before,
            after,
        )
        return {"graph": graph, "before": before, "after": after}

    async def _rebuild_indexes_unlocked(
        self,
        progress=None,
        *,
        job_context=None,
        checkpoint_dir=None,
        before_database_repair: Callable[[], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        logger.warning(
            "索引重建开始：memory_store_id=%s provider=%s revision=%s",
            self.memory_store_id,
            self.provider_revision.provider_id,
            self.provider_revision.revision,
        )
        if progress:
            await progress(0.02, "正在检查图记忆派生数据")

        async def graph_progress(value: float, message: str) -> None:
            if progress:
                await progress(0.02 + max(0.0, min(1.0, value)) * 0.18, message)

        graph_before, fts_before = await asyncio.gather(
            self.storage.graph_integrity_report(),
            self.storage.fts_integrity_report(),
        )
        if before_database_repair is not None:
            await before_database_repair()
        graph_recovery = await self._ensure_graph_recovery_unlocked(
            graph_progress,
            before_report=graph_before,
        )
        fts = await self._ensure_fts_recovery_unlocked(
            force_reason=(
                "graph_recovery"
                if graph_recovery.get("rebuilt")
                else "index_rebuild_livingmemory_250"
            ),
            before_report=fts_before,
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
            "索引重建成功：memory_store_id=%s generation=%s elapsed_ms=%.2f",
            self.memory_store_id,
            manifest.get("generation"),
            (time.perf_counter() - started) * 1000,
        )
        return {"graph_recovery": graph_recovery, "fts": fts, "manifest": manifest}

    async def rebuild_graph(
        self, progress=None, *, job_context=None, checkpoint_dir=None
    ) -> dict[str, Any]:
        await self.cancel_startup_index_check()
        result = await self._continuous_rebuild(
            progress=progress,
            job_context=job_context,
            checkpoint_dir=checkpoint_dir,
            reason="manual_graph_rebuild",
        )
        return {
            "graph": result["graph"],
            "fts": result["fts"],
            "manifest": result["manifest"],
            "replayed_memory_ids": result.get("replayed_memory_ids", []),
        }

    async def rebuild_with_provider(
        self,
        provider_revision: ProviderRevision,
        progress=None,
        *,
        job_context=None,
        checkpoint_dir=None,
        before_database_repair: Callable[[], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        logger.warning(
            "Provider 切换重建开始：memory_store_id=%s provider=%s revision=%s",
            self.memory_store_id,
            provider_revision.provider_id,
            provider_revision.revision,
        )
        candidate = build_provider(provider_revision.config)
        old_provider = self.provider
        try:
            await self.cancel_startup_index_check()
            result = await self._continuous_rebuild(
                progress=progress,
                job_context=job_context,
                checkpoint_dir=checkpoint_dir,
                before_database_repair=before_database_repair,
                reason="provider_switch",
                provider=candidate,
                provider_revision=provider_revision,
            )
        except JobControlSignal:
            logger.info(
                "Provider 切换重建已到达任务控制边界：memory_store_id=%s provider=%s revision=%s",
                self.memory_store_id,
                provider_revision.provider_id,
                provider_revision.revision,
            )
            await candidate.close()
            raise
        except Exception:
            logger.exception(
                "Provider 切换重建失败，保留旧索引：memory_store_id=%s provider=%s revision=%s",
                self.memory_store_id,
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
            "Provider 切换重建成功：memory_store_id=%s provider=%s revision=%s generation=%s elapsed_ms=%.2f",
            self.memory_store_id,
            provider_revision.provider_id,
            provider_revision.revision,
            result["manifest"].get("generation"),
            (time.perf_counter() - started) * 1000,
        )
        return result

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
                    "记忆增量索引失败，已回滚数据库写入：memory_store_id=%s memory_id=%s",
                    self.memory_store_id,
                    memory_id,
                )
                raise
            logger.info(
                "记忆已写入数据库并完成增量索引：memory_store_id=%s memory_id=%s generation=%s",
                self.memory_store_id,
                memory_id,
                index_update.get("generation"),
            )
            self.retrieval.invalidate()
            result = (await self.storage.get_document(memory_id)) or {"id": memory_id}
            result["index_update"] = index_update
            return result

    async def _rollback_content_replacement(
        self,
        *,
        old_memory_id: int,
        new_memory_id: int | None,
        recovery_payload: dict[str, Any],
    ) -> None:
        """Reconcile a failed copy-on-write edit without losing both copies."""

        old_document = await self.storage.get_document(old_memory_id)
        new_document = (
            await self.storage.get_document(new_memory_id)
            if new_memory_id is not None
            else None
        )
        if old_document is not None:
            if new_document is not None and new_memory_id is not None:
                deleted = await self.storage.delete_memories([new_memory_id])
                if deleted != 1:
                    raise RuntimeError(
                        f"failed to remove replacement memory {new_memory_id}"
                    )
            repaired = await self._repair_replacement_indexes()
            await self.storage.mark_memory_indexed(
                old_memory_id,
                generation=str(repaired.get("generation") or ""),
            )
            return

        if new_document is not None and new_memory_id is not None:
            # The old row was already deleted.  Never delete the only surviving
            # copy; instead converge indexes on the completed replacement.
            repaired = await self._repair_replacement_indexes()
            await self.storage.mark_memory_indexed(
                new_memory_id,
                generation=str(repaired.get("generation") or ""),
            )
            return

        # This should be unreachable with transactional storage operations, but
        # recreating the replacement is safer than allowing an edit failure to
        # erase the only durable copy.
        recovered_memory_id = await self.storage.create_memory(
            self._payload_for_write(recovery_payload),
            self.text.tokenize,
            self._graph_builder_for_write(),
        )
        repaired = await self.indexes.upsert_memories(
            [recovered_memory_id],
            reason="memory_update_recover_missing_copies",
        )
        await self.storage.mark_memory_indexed(
            recovered_memory_id,
            generation=str(repaired.get("generation") or ""),
        )
        logger.critical(
            "内容编辑替换的两份记录均缺失，已重建可恢复副本：memory_store_id=%s memory_id=%s",
            self.memory_store_id,
            recovered_memory_id,
        )

    async def _repair_replacement_indexes(self) -> dict[str, Any]:
        """Rebuild only when the current vector ID sets disagree with storage."""

        expected_document_ids = set(await self.storage.document_ids())
        if self.indexes.graph_vector_granularity() == "memory":
            expected_graph_ids = set(await self.storage.graph_memory_ids())
        else:
            expected_graph_ids = set(await self.storage.graph_entry_ids())
        actual_document_ids, actual_graph_ids = self.indexes.indexed_ids()
        if (
            actual_document_ids != expected_document_ids
            or actual_graph_ids != expected_graph_ids
        ):
            return await self.indexes.rebuild()
        status = self.indexes.status()
        manifest = status.get("manifest")
        if isinstance(manifest, dict):
            return manifest
        return {"generation": status.get("generation")}

    async def update_memory(
        self,
        memory_id: int,
        payload: dict[str, Any],
        *,
        rebuild: bool = True,
        progress=None,
    ) -> dict[str, Any] | None:
        if "content" not in payload and "status" in payload:
            current = await self.storage.get_document(memory_id)
            if not current:
                return None
            current_status = str(
                (current.get("metadata") or {}).get("status") or "active"
            )
            target_status = str(payload.get("status") or "active")
            metadata_patch = {
                key: value for key, value in payload.items() if key != "status"
            }
            if target_status == "archived" and current_status != "archived":
                if metadata_patch:
                    await self.storage.update_memory_metadata(memory_id, metadata_patch)
                details = await self.archive_memories([memory_id], return_details=True)
                result = await self.storage.get_document(memory_id)
                if result is not None:
                    result["index_update"] = details.get("index_update")
                return result
            if target_status == "active" and current_status == "archived":
                if metadata_patch:
                    await self.storage.update_memory_metadata(memory_id, metadata_patch)
                return await self.restore_memory(memory_id)
        async with self._mutation_lock:
            if "content" in payload:
                current = await self.storage.get_document(memory_id)
                if not current:
                    return None
                new_content = str(payload.get("content") or "").strip()
                if not new_content:
                    raise ValueError("content cannot be empty")
                current_metadata = dict(current.get("metadata") or {})
                retained_source = await self.storage.get_memory_source(memory_id)
                merged_metadata = {
                    **current_metadata,
                    **dict(payload.get("metadata") or {}),
                }
                for key in (
                    "importance",
                    "status",
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
                    "topics": merged_metadata.get("topics", []),
                    "participants": merged_metadata.get("participants", []),
                    "key_facts": merged_metadata.get("key_facts", []),
                    "source_messages": retained_source,
                    "metadata": merged_metadata,
                }
                old_graph_entries = await self.storage.graph_entries_for_memory_ids(
                    [memory_id]
                )
                old_graph_ids = {int(item["id"]) for item in old_graph_entries}
                new_memory_id: int | None = None
                new_index_update: dict[str, Any] | None = None
                delete_index_update: dict[str, Any] | None = None
                try:
                    created, interrupted = await _finish_mutation_step(
                        self.storage.create_memory(
                            self._payload_for_write(new_payload),
                            self.text.tokenize,
                            self._graph_builder_for_write(),
                        )
                    )
                    new_memory_id = int(created)
                    if interrupted is not None:
                        raise interrupted

                    indexed, interrupted = await _finish_mutation_step(
                        self.indexes.upsert_memories(
                            [new_memory_id],
                            reason="memory_update_create_replacement",
                        )
                    )
                    new_index_update = dict(indexed)
                    if interrupted is not None:
                        raise interrupted

                    _, interrupted = await _finish_mutation_step(
                        self.storage.mark_memory_indexed(
                            new_memory_id,
                            generation=str(new_index_update.get("generation") or ""),
                        )
                    )
                    if interrupted is not None:
                        raise interrupted

                    removed, interrupted = await _finish_mutation_step(
                        self.indexes.delete_memories_incremental(
                            [memory_id],
                            graph_entry_ids=old_graph_ids,
                            reason="memory_update_delete_old",
                        )
                    )
                    delete_index_update = dict(removed)
                    if interrupted is not None:
                        raise interrupted

                    deleted, interrupted = await _finish_mutation_step(
                        self.storage.delete_memories([memory_id])
                    )
                    if deleted != 1:
                        raise RuntimeError(
                            f"failed to delete replaced memory {memory_id}"
                        )
                    if interrupted is not None:
                        raise interrupted
                except (Exception, asyncio.CancelledError):
                    logger.exception(
                        "内容编辑替换失败，正在回滚：memory_store_id=%s old_id=%s new_id=%s",
                        self.memory_store_id,
                        memory_id,
                        new_memory_id,
                    )
                    try:
                        await _finish_mutation_cleanup(
                            self._rollback_content_replacement(
                                old_memory_id=memory_id,
                                new_memory_id=new_memory_id,
                                recovery_payload=new_payload,
                            )
                        )
                    except (Exception, asyncio.CancelledError):
                        logger.critical(
                            "内容编辑替换回滚未能完整收敛：memory_store_id=%s old_id=%s new_id=%s",
                            self.memory_store_id,
                            memory_id,
                            new_memory_id,
                            exc_info=True,
                        )
                    self.retrieval.invalidate()
                    raise
                assert new_memory_id is not None
                assert delete_index_update is not None
                self.retrieval.invalidate()
                result = await self.storage.get_document(new_memory_id)
                if result is None:
                    result = {"id": new_memory_id}
                result["old_memory_id"] = memory_id
                result["new_memory_id"] = new_memory_id
                result["index_update"] = delete_index_update
                result["replacement_index_update"] = new_index_update
                logger.info(
                    "记忆内容编辑已按新 ID 替换完成：memory_store_id=%s old_id=%s new_id=%s",
                    self.memory_store_id,
                    memory_id,
                    new_memory_id,
                )
                return result

            success = await self.storage.update_memory_metadata(memory_id, payload)
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
                    "document_vectors": index_status.get("document_vectors"),
                    "graph_vectors": index_status.get("graph_vectors"),
                }
            return result

    async def archive_memories(
        self,
        memory_ids: list[int],
        *,
        return_details: bool = False,
    ) -> int | dict[str, Any]:
        async with self._mutation_lock:
            documents = await self.storage.documents_for_ids(memory_ids)
            active_ids = [
                int(item["id"])
                for item in documents
                if str((item.get("metadata") or {}).get("status") or "active")
                != "archived"
            ]
            if not active_ids:
                empty = {"archived": 0, "index_update": None}
                return empty if return_details else 0
            graph_entries = await self.storage.graph_entries_for_memory_ids(active_ids)
            index_update = await self.indexes.delete_memories_incremental(
                active_ids,
                graph_entry_ids={int(item["id"]) for item in graph_entries},
                reason="memory_archive",
            )
            try:
                archived_ids = await self.storage.archive_memories(active_ids)
                if len(archived_ids) != len(active_ids):
                    raise RuntimeError("not all requested memories were archived")
            except Exception:
                logger.exception(
                    "记忆归档数据库阶段失败，正在恢复索引：memory_store_id=%s ids=%s",
                    self.memory_store_id,
                    active_ids,
                )
                await self.indexes.upsert_memories(
                    active_ids, reason="memory_archive_restore"
                )
                raise
            self.retrieval.invalidate()
            if return_details:
                return {"archived": len(archived_ids), "index_update": index_update}
            return len(archived_ids)

    async def restore_memory(self, memory_id: int) -> dict[str, Any] | None:
        async with self._mutation_lock:
            current = await self.storage.get_document(memory_id)
            if not current:
                return None
            metadata = dict(current.get("metadata") or {})
            if str(metadata.get("status") or "active") != "archived":
                return current
            payload = {
                "key_facts": list(metadata.get("key_facts") or []),
                "topics": list(metadata.get("topics") or []),
                "participants": list(metadata.get("participants") or []),
                "importance": metadata.get("importance", 0.5),
                "session_id": metadata.get("session_id"),
                "persona_id": metadata.get("persona_id"),
            }
            atoms = self._atoms_from_key_facts(payload)
            restored = await self.storage.restore_memory(
                memory_id,
                self.text.tokenize,
                self._graph_builder_for_write(),
                atoms,
            )
            if not restored:
                return None
            try:
                index_update = await self.indexes.upsert_memories(
                    [memory_id], reason="memory_restore"
                )
            except Exception:
                await self.storage.archive_memories([memory_id])
                self.retrieval.invalidate()
                logger.exception(
                    "记忆恢复索引阶段失败，已回滚为归档状态：memory_store_id=%s memory_id=%s",
                    self.memory_store_id,
                    memory_id,
                )
                raise
            self.retrieval.invalidate()
            result = await self.storage.get_document(memory_id)
            if result is not None:
                result["index_update"] = index_update
            return result

    async def update_memory_persona(
        self, memory_id: int, persona_id: str | None
    ) -> dict[str, Any] | None:
        async with self._mutation_lock:
            success = await self.storage.update_memory_persona(memory_id, persona_id)
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
                "记忆人格字段已原位更新：memory_store_id=%s memory_id=%s persona=%s generation=%s index_changed=false",
                self.memory_store_id,
                memory_id,
                str(persona_id or "").strip() or "",
                index_status.get("generation") or "",
            )
            return result

    async def update_memories_metadata(
        self,
        memory_ids: list[int],
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        async with self._mutation_lock:
            updated_ids = await self.storage.update_memories_metadata(
                memory_ids,
                payload,
            )
            if not updated_ids:
                return []
            self.retrieval.invalidate()
            documents = await self.storage.documents_for_ids(updated_ids)
            index_status = self.indexes.status() or {}
            index_update = {
                "mode": "metadata_only",
                "status": "completed",
                "index_changed": False,
                "generation": index_status.get("generation"),
                "document_vectors": index_status.get("document_vectors"),
                "graph_vectors": index_status.get("graph_vectors"),
            }
            by_id = {int(item["id"]): item for item in documents}
            result: list[dict[str, Any]] = []
            for memory_id in updated_ids:
                item = by_id.get(memory_id)
                if item is None:
                    continue
                item["index_update"] = dict(index_update)
                result.append(item)
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
            graph_entries = await self.storage.graph_entries_for_memory_ids(
                existing_ids
            )
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
                    "删除记忆数据库阶段失败，正在恢复索引：memory_store_id=%s ids=%s",
                    self.memory_store_id,
                    existing_ids,
                )
                try:
                    await self.indexes.upsert_memories(
                        existing_ids, reason="memory_delete_restore"
                    )
                except Exception:
                    logger.error(
                        "恢复删除前索引失败：memory_store_id=%s ids=%s",
                        self.memory_store_id,
                        existing_ids,
                        exc_info=True,
                    )
                raise
            if deleted:
                logger.warning(
                    "记忆已删除并完成增量索引移除：memory_store_id=%s deleted=%s generation=%s",
                    self.memory_store_id,
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
        limit_entries: int = 40,
        minimum_degree: int = 0,
    ) -> dict[str, Any]:
        limit_nodes = max(1, min(int(limit_nodes), 200))
        limit_edges = max(1, min(int(limit_edges), 400))
        limit_entries = max(1, min(int(limit_entries), 80))
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
                    (*params, limit_entries),
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
                entry_node_map.setdefault(int(row["entry_id"]), []).append(
                    int(row["node_id"])
                )
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
                    source_stats["weight"] = (
                        float(source_stats.get("weight", 0.0)) + edge_weight
                    )
                    target_stats = node_stats.setdefault(target_id, {})
                    target_stats["degree"] = int(target_stats.get("degree", 0)) + 1
                    target_stats["weight"] = (
                        float(target_stats.get("weight", 0.0)) + edge_weight
                    )
        edge_view = [
            {
                "id": int(row["id"]),
                "source": int(row["source_node_id"]),
                "target": int(row["target_node_id"]),
                "relation_type": row["relation_type"],
                "source_memory_id": int(row["source_memory_id"]),
                "memory_id": int(row["source_memory_id"]),
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
            if edge["source"] in allowed_node_ids and edge["target"] in allowed_node_ids
        ][:limit_edges]
        allowed_node_ids = _graph_k_core_node_ids(
            allowed_node_ids, edge_view, minimum_degree
        )
        edge_view = [
            edge
            for edge in edge_view
            if edge["source"] in allowed_node_ids and edge["target"] in allowed_node_ids
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
                    "memory_count": int(
                        node_stats.get(int(row["id"]), {}).get("memory_count", 0)
                    ),
                    "degree": visible_degrees.get(int(row["id"]), 0),
                    "entry_count": int(
                        node_stats.get(int(row["id"]), {}).get("entry_count", 0)
                    ),
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
                    "memory_id": int(row["source_memory_id"]),
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

    async def full_graph_snapshot(
        self,
        *,
        session_id: str | None = None,
        persona_id: str | None = None,
    ) -> dict[str, Any]:
        """Return every graph node and edge in scope for the 2.5 dashboard."""
        filters: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            filters.append("ge.session_id=?")
            params.append(session_id)
        if persona_id is not None:
            filters.append("ge.persona_id=?")
            params.append(persona_id)
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        async with self.storage.connect() as db:
            node_rows = await (
                await db.execute(
                    f"""SELECT gn.id,gn.node_key,gn.node_type,gn.node_value,
                    gn.canonical_value,gn.metadata,
                    COUNT(DISTINCT ge.id) AS entry_count,
                    COUNT(DISTINCT ge.source_memory_id) AS memory_count
                    FROM graph_nodes gn
                    JOIN graph_entry_nodes gen ON gen.node_id=gn.id
                    JOIN graph_entries ge ON ge.id=gen.entry_id
                    {where}
                    GROUP BY gn.id ORDER BY gn.id""",
                    tuple(params),
                )
            ).fetchall()
            edge_rows = await (
                await db.execute(
                    f"""SELECT DISTINCT edge.id,edge.edge_key,
                    edge.source_node_id,edge.target_node_id,edge.relation_type,
                    edge.source_memory_id,edge.weight,edge.confidence,
                    edge.status,edge.metadata
                    FROM graph_edges edge
                    JOIN graph_entries ge
                    ON ge.source_memory_id=edge.source_memory_id
                    {where}
                    ORDER BY edge.id""",
                    tuple(params),
                )
            ).fetchall()
            memory_rows = await (
                await db.execute(
                    f"""SELECT ge.source_memory_id,ge.session_id,ge.persona_id,
                    ge.content,ge.metadata
                    FROM graph_entries ge
                    JOIN (
                        SELECT ge.source_memory_id,MAX(ge.id) AS latest_entry_id
                        FROM graph_entries ge {where}
                        GROUP BY ge.source_memory_id
                    ) latest ON latest.latest_entry_id=ge.id
                    ORDER BY ge.source_memory_id""",
                    tuple(params),
                )
            ).fetchall()

        node_map: dict[int, dict[str, Any]] = {}
        for row in node_rows:
            node_id = int(row["id"])
            node_map[node_id] = {
                "id": node_id,
                "key": row["node_key"],
                "type": row["node_type"],
                "label": row["node_value"],
                "canonical_value": row["canonical_value"],
                "entry_count": int(row["entry_count"] or 0),
                "memory_count": int(row["memory_count"] or 0),
                "degree": 0,
                "weight": 0.0,
            }
        edges: list[dict[str, Any]] = []
        for row in edge_rows:
            source = int(row["source_node_id"])
            target = int(row["target_node_id"])
            if source not in node_map or target not in node_map:
                continue
            edge = {
                "id": int(row["id"]),
                "key": row["edge_key"],
                "source": source,
                "target": target,
                "relation_type": row["relation_type"],
                "source_memory_id": int(row["source_memory_id"]),
                "memory_id": int(row["source_memory_id"]),
                "weight": float(row["weight"] or 0),
                "confidence": float(row["confidence"] or 0),
                "status": row["status"],
            }
            edges.append(edge)
            node_map[source]["degree"] += 1
            node_map[target]["degree"] += 1
        for node in node_map.values():
            node["weight"] = round(
                node["entry_count"]
                + node["memory_count"] * 0.75
                + node["degree"] * 0.35,
                4,
            )

        memories: list[dict[str, Any]] = []
        for row in memory_rows:
            metadata = normalize_document_metadata(row["metadata"])
            try:
                importance = float(metadata.get("importance") or 0)
            except (TypeError, ValueError):
                importance = 0.0
            memories.append(
                {
                    "memory_id": int(row["source_memory_id"]),
                    "summary": str(
                        metadata.get("canonical_summary") or row["content"] or ""
                    )[:500],
                    "session_id": metadata.get("session_id") or row["session_id"],
                    "persona_id": metadata.get("persona_id") or row["persona_id"],
                    "importance": importance,
                }
            )
        return {
            "nodes": sorted(
                node_map.values(),
                key=lambda item: (
                    -float(item["weight"]),
                    -int(item["degree"]),
                    str(item["label"]),
                ),
            ),
            "edges": edges,
            "entries": [],
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
            "核心记忆库文件已备份：memory_store_id=%s livingmemory=%s conversations=%s",
            self.memory_store_id,
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
            return {"version": 1, "library_id": self.memory_store_id}
        try:
            data = json.loads(state_path.read_text(encoding="utf-8") or "{}")
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "library_id": self.memory_store_id}
        return (
            data
            if isinstance(data, dict)
            else {"version": 1, "library_id": self.memory_store_id}
        )

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
        forgot_before = now - self.config.maintenance.atom_forget_delay_days * 86400
        purge_before = now - self.config.maintenance.atom_purge_delay_days * 86400
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
                protected_threshold = max(
                    0.0,
                    min(
                        1.0,
                        float(self.config.maintenance.protected_importance_threshold),
                    ),
                )
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
                    if decay_rate > 0 and importance < protected_threshold:
                        access_count = max(0, int(metadata.get("access_count", 0) or 0))
                        last_access_time = float(
                            metadata.get("last_access_time", 0) or 0
                        )
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
                        and str(metadata.get("status") or "active") == "active"
                        and age_days >= self.config.maintenance.cleanup_days_threshold
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
        archived = 0
        if run_daily and cleanup_ids:
            if self.config.maintenance.auto_archived_enabled:
                archived = int(await self.archive_memories(cleanup_ids))
            else:
                cleaned = int(await self.delete_memories(cleanup_ids))
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
            "archived_memories": archived,
            "purged_atoms": len(rows),
            "backup": backup_path,
            "removed_backups": removed_backups,
            "daily_maintenance_ran": run_daily,
        }
        state.update(
            {
                "version": 1,
                "library_id": self.memory_store_id,
                "last_maintenance_at": time.time(),
                "result": result,
            }
        )
        if run_daily:
            state["last_daily_maintenance_date"] = today
            state["last_decay_date"] = today
        self._write_decay_state(state)
        return result
