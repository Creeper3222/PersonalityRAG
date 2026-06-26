from __future__ import annotations

import asyncio
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

from .config import AppConfig
from .graph import GraphBuilder
from .indexes import IndexManager
from .jobs import JobManager
from .logger import logger
from .migration import LivingMemoryMigrator, sqlite_backup
from .control import ProviderRevision
from .providers import build_provider, provider_config_hash
from .retrieval import RetrievalEngine
from .storage import Storage
from .text import TextProcessor


class PersonalityRAGService:
    def __init__(
        self,
        root: Path,
        config: AppConfig,
        data_dir: Path | None = None,
        *,
        library_id: str = "",
        provider_revision: ProviderRevision | None = None,
        system_path: Path | None = None,
    ):
        self.root = root
        self.data_dir = data_dir or (root / "data")
        self.library_id = library_id
        self.config = config
        self.provider_revision = provider_revision or ProviderRevision(
            provider_id=config.provider.id,
            revision=1,
            config=config.provider,
            config_sha256=provider_config_hash(config.provider),
            created_at=time.time(),
        )
        self.storage = Storage(self.data_dir, system_path=system_path)
        self.text = TextProcessor(self.data_dir / "stopwords")
        self.graph_builder = GraphBuilder()
        self.provider = build_provider(self.provider_revision.config)
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
        self.jobs = JobManager(self.storage)
        self.migrator = LivingMemoryMigrator(self.data_dir)
        self._maintenance_task: asyncio.Task | None = None
        self._mutation_lock = asyncio.Lock()
        self._retired_providers: list[Any] = []

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
        providers = [self.provider, *self._retired_providers]
        self._retired_providers.clear()
        closed: set[int] = set()
        for provider in providers:
            identity = id(provider)
            if identity in closed:
                continue
            closed.add(identity)
            await provider.close()

    async def rebuild_indexes(self, progress=None) -> dict[str, Any]:
        async with self._mutation_lock:
            return await self._rebuild_indexes_unlocked(progress)

    async def _rebuild_indexes_unlocked(self, progress=None) -> dict[str, Any]:
        started = time.perf_counter()
        logger.warning(
            "索引重建开始：library_id=%s provider=%s revision=%s",
            self.library_id,
            self.provider_revision.provider_id,
            self.provider_revision.revision,
        )
        fts = await self.storage.rebuild_fts(self.text.tokenize)
        logger.info("FTS 重建完成：library_id=%s rows=%s", self.library_id, fts)
        manifest = await self.indexes.rebuild(
            batch_size=self.provider_revision.config.batch_size,
            concurrency=self.provider_revision.config.concurrency,
            progress=progress,
        )
        self.retrieval.invalidate()
        logger.warning(
            "索引重建成功：library_id=%s generation=%s elapsed_ms=%.2f",
            self.library_id,
            manifest.get("generation"),
            (time.perf_counter() - started) * 1000,
        )
        return {"fts": fts, "manifest": manifest}

    async def rebuild_with_provider(
        self,
        provider_revision: ProviderRevision,
        progress=None,
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
                fts = await self.storage.rebuild_fts(self.text.tokenize)
                logger.info("Provider 切换 FTS 重建完成：library_id=%s rows=%s", self.library_id, fts)
                manifest = await self.indexes.rebuild(
                    batch_size=provider_revision.config.batch_size,
                    concurrency=provider_revision.config.concurrency,
                    progress=progress,
                    provider=candidate,
                    library_id=self.library_id,
                    provider_id=provider_revision.provider_id,
                    provider_revision=provider_revision.revision,
                    provider_config_sha256=provider_revision.config_sha256,
                    provider_model=provider_revision.config.model,
                )
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
        return {"fts": fts, "manifest": manifest}

    async def create_memory(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._mutation_lock:
            memory_id = await self.storage.create_memory(
                payload,
                self.text.tokenize,
                self.graph_builder.build,
            )
            logger.info("记忆已写入数据库，开始重建索引：library_id=%s memory_id=%s", self.library_id, memory_id)
            await self._rebuild_indexes_unlocked()
            self.retrieval.invalidate()
            return (await self.storage.get_document(memory_id)) or {
                "id": memory_id
            }

    async def update_memory(
        self, memory_id: int, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        async with self._mutation_lock:
            success = await self.storage.update_memory(
                memory_id,
                payload,
                self.text.tokenize,
                self.graph_builder.build,
            )
            if not success:
                return None
            logger.info("记忆已更新数据库，开始重建索引：library_id=%s memory_id=%s", self.library_id, memory_id)
            await self._rebuild_indexes_unlocked()
            self.retrieval.invalidate()
            return await self.storage.get_document(memory_id)

    async def delete_memories(self, memory_ids: list[int]) -> int:
        async with self._mutation_lock:
            deleted = await self.storage.delete_memories(memory_ids)
            if deleted:
                logger.warning("记忆已删除，开始重建索引：library_id=%s deleted=%s", self.library_id, deleted)
                await self._rebuild_indexes_unlocked()
                self.retrieval.invalidate()
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
    ) -> dict[str, Any]:
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
            node_rows = await (
                await db.execute(
                    f"""SELECT DISTINCT n.id,n.node_key,n.node_type,n.node_value,
                    n.canonical_value,n.metadata
                    FROM graph_entry_nodes gen JOIN graph_nodes n ON n.id=gen.node_id
                    WHERE gen.entry_id IN ({entry_placeholders}) LIMIT ?""",
                    (*entry_ids, limit_nodes),
                )
            ).fetchall()
            node_ids = [int(row["id"]) for row in node_rows]
            edge_rows = []
            if node_ids:
                node_placeholders = ",".join("?" for _ in node_ids)
                edge_rows = await (
                    await db.execute(
                        f"""SELECT e.id,e.source_node_id,e.target_node_id,
                        e.relation_type,e.source_memory_id,e.weight,e.confidence,e.status
                        FROM graph_edges e WHERE e.source_node_id IN ({node_placeholders})
                        AND e.target_node_id IN ({node_placeholders}) LIMIT ?""",
                        (*node_ids, *node_ids, limit_edges),
                    )
                ).fetchall()
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
                }
                for row in node_rows
            ],
            "edges": [
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
            ],
            "entries": [
                {
                    "id": int(row["id"]),
                    "source_memory_id": int(row["source_memory_id"]),
                    "entry_type": row["entry_type"],
                    "relation_type": row["relation_type"],
                    "content": row["content"],
                }
                for row in entry_rows
            ],
            "memories": memories,
        }

    async def list_backups(self) -> list[dict[str, Any]]:
        roots = [
            self.data_dir / "backups",
            self.data_dir / "livingmemory_backups",
            self.data_dir / "imports",
        ]
        result = []
        for root in roots:
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

    async def run_maintenance(self) -> dict[str, Any]:
        now = time.time()
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
                if self.config.recall.decay_rate > 0:
                    next_importance = max(
                        0.0,
                        importance
                        * math.exp(-self.config.recall.decay_rate),
                    )
                    if next_importance != importance:
                        metadata["importance"] = next_importance
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
                    and importance
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
        if cleanup_ids:
            cleaned = await self.delete_memories(cleanup_ids)
        backup_path = None
        if self.config.maintenance.backup_enabled:
            backup_path = str(await self.backup())
        backup_root = self.data_dir / "backups"
        removed_backups = 0
        if backup_root.exists():
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
        }
        state_path = self.data_dir / "decay_state.json"
        state_temp = state_path.with_suffix(".tmp")
        state_temp.write_text(
            json.dumps(
                {
                    "version": 1,
                    "library_id": self.library_id,
                    "last_maintenance_at": time.time(),
                    "result": result,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        state_temp.replace(state_path)
        return result
