from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np

from .providers import EmbeddingProvider
from .storage import Storage


DEFAULT_DOCUMENT_EMBED_CHARS = 4000
DEFAULT_QUERY_EMBED_CHARS = 2000
CONTEXT_CHAR_SAFETY_RATIO = 0.9
MIN_CONTEXT_CLIP_CHARS = 64


def _configured_faiss_threads() -> int:
    raw = os.environ.get("PERSONALITYRAG_FAISS_THREADS", "8")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 8


FAISS_THREAD_COUNT = _configured_faiss_threads()
faiss.omp_set_num_threads(FAISS_THREAD_COUNT)


def _embedding_char_limit(provider: EmbeddingProvider, default_limit: int) -> int:
    config = getattr(provider, "config", None)
    tokens = int(getattr(config, "max_context_tokens", 0) or 0)
    if tokens <= 0:
        return default_limit
    safe_chars = max(MIN_CONTEXT_CLIP_CHARS, int(tokens * CONTEXT_CHAR_SAFETY_RATIO))
    return max(1, min(default_limit, safe_chars))


def _clip_for_embedding(
    text: Any,
    provider: EmbeddingProvider,
    default_limit: int,
) -> str:
    return str(text or "")[: _embedding_char_limit(provider, default_limit)]


def _path_is_ascii(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _faiss_needs_path_bridge(path: Path) -> bool:
    return os.name == "nt" and not _path_is_ascii(path)


def _safe_faiss_temp_dir() -> Path:
    candidates = [Path(tempfile.gettempdir())]
    system_root = os.environ.get("SystemRoot")
    if system_root:
        candidates.append(Path(system_root) / "Temp")
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            if _path_is_ascii(candidate):
                return candidate
        except Exception:
            continue
    return candidates[0]


def _read_faiss_index(path: Path) -> faiss.Index:
    if not _faiss_needs_path_bridge(path):
        return faiss.read_index(str(path))
    temp_dir = _safe_faiss_temp_dir()
    fd, temp_name = tempfile.mkstemp(
        prefix="personalityrag-faiss-", suffix=".index", dir=temp_dir
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        shutil.copyfile(path, temp_path)
        return faiss.read_index(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)


def _write_faiss_index(index: faiss.Index, path: Path) -> None:
    if not _faiss_needs_path_bridge(path):
        faiss.write_index(index, str(path))
        return
    temp_dir = _safe_faiss_temp_dir()
    fd, temp_name = tempfile.mkstemp(
        prefix="personalityrag-faiss-", suffix=".index", dir=temp_dir
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        faiss.write_index(index, str(temp_path))
        shutil.copyfile(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


@dataclass(slots=True)
class GenerationManifest:
    generation: str
    created_at: float
    library_id: str
    provider_id: str
    provider_revision: int
    provider_config_sha256: str
    provider_type: str
    configured_model: str
    resolved_model: str
    dimension: int
    metric: str
    document_count: int
    graph_entry_count: int
    document_ids_sha256: str
    graph_ids_sha256: str
    vector_norm_min: float
    vector_norm_max: float
    vector_norm_mean: float


@dataclass(slots=True)
class IndexSnapshot:
    document_index: faiss.Index
    graph_index: faiss.Index
    manifest: dict[str, Any] | None
    provider: EmbeddingProvider


class IndexManager:
    def __init__(
        self,
        data_dir: Path,
        storage: Storage,
        provider: EmbeddingProvider,
        provider_model: str,
        *,
        library_id: str = "",
        provider_id: str = "",
        provider_revision: int = 1,
        provider_config_sha256: str = "",
    ):
        self.root = data_dir / "indexes"
        self.root.mkdir(parents=True, exist_ok=True)
        self.current_file = self.root / "CURRENT"
        self.storage = storage
        self.provider_model = provider_model
        self.library_id = library_id
        self.provider_id = provider_id or getattr(provider, "config", None) and provider.config.id or ""
        self.provider_revision = provider_revision
        self.provider_config_sha256 = provider_config_sha256
        self._snapshot: IndexSnapshot | None = None
        self._initial_provider = provider
        self._swap_lock = asyncio.Lock()

    @staticmethod
    def _empty_index(dimension: int) -> faiss.Index:
        return faiss.IndexIDMap(faiss.IndexFlatL2(dimension))

    @property
    def provider(self) -> EmbeddingProvider:
        return self._snapshot.provider if self._snapshot else self._initial_provider

    @property
    def document_index(self) -> faiss.Index | None:
        return self._snapshot.document_index if self._snapshot else None

    @property
    def graph_index(self) -> faiss.Index | None:
        return self._snapshot.graph_index if self._snapshot else None

    @property
    def manifest(self) -> dict[str, Any] | None:
        return self._snapshot.manifest if self._snapshot else None

    async def initialize(self) -> None:
        if self.current_file.exists():
            generation = self.current_file.read_text(encoding="utf-8").strip()
            path = self.root / generation
            try:
                document_index = _read_faiss_index(path / "documents.index")
                graph_index = _read_faiss_index(path / "graph.index")
                manifest = json.loads(
                    (path / "manifest.json").read_text(encoding="utf-8")
                )
                self._snapshot = IndexSnapshot(
                    document_index,
                    graph_index,
                    manifest,
                    self._initial_provider,
                )
                return
            except Exception:
                pass
        dimension = await self._initial_provider.get_dimension()
        self._snapshot = IndexSnapshot(
            self._empty_index(dimension),
            self._empty_index(dimension),
            None,
            self._initial_provider,
        )

    @staticmethod
    def _validate_matrix(
        matrix: np.ndarray,
        *,
        expected_rows: int,
        dimension: int,
    ) -> tuple[float, float, float]:
        if matrix.shape != (expected_rows, dimension):
            raise RuntimeError(
                f"invalid embedding matrix {matrix.shape}, "
                f"expected {(expected_rows, dimension)}"
            )
        if not np.isfinite(matrix).all():
            raise RuntimeError("embedding matrix contains non-finite values")
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms <= 0):
            raise RuntimeError("embedding matrix contains zero vectors")
        return float(norms.min()), float(norms.max()), float(norms.sum())

    @staticmethod
    async def _validate_sample_recall(
        provider: EmbeddingProvider,
        index: faiss.Index,
        samples: list[tuple[int, str]],
        label: str,
    ) -> None:
        if not samples:
            return
        vectors = np.asarray(
            await provider.get_embeddings(
                [
                    _clip_for_embedding(text, provider, DEFAULT_DOCUMENT_EMBED_CHARS)
                    for _, text in samples
                ],
            ),
            dtype=np.float32,
        )
        if vectors.ndim != 2:
            raise RuntimeError(f"{label}抽样召回向量格式无效")
        faiss.normalize_L2(vectors)
        k = min(10, int(index.ntotal))
        _, result_ids = index.search(vectors, k)
        for (expected_id, _), row in zip(samples, result_ids, strict=True):
            if expected_id not in {int(item) for item in row if int(item) >= 0}:
                raise RuntimeError(
                    f"{label}抽样召回未命中自身 ID: {expected_id}"
                )

    @staticmethod
    def _index_ids(index: faiss.Index) -> set[int]:
        id_map = getattr(index, "id_map", None)
        if id_map is None:
            return set()
        return {int(value) for value in faiss.vector_to_array(id_map)}

    def indexed_ids(self) -> tuple[set[int], set[int]]:
        snapshot = self._snapshot
        if snapshot is None:
            return set(), set()
        return (
            self._index_ids(snapshot.document_index),
            self._index_ids(snapshot.graph_index),
        )

    @staticmethod
    def _remove_ids(index: faiss.Index, ids: set[int]) -> None:
        positive_ids = sorted({int(value) for value in ids if int(value) >= 0})
        if not positive_ids:
            return
        index.remove_ids(np.asarray(positive_ids, dtype=np.int64))

    @staticmethod
    def _id_hash(ids: set[int]) -> str:
        return hashlib.sha256(
            ",".join(map(str, sorted(ids))).encode()
        ).hexdigest()

    def _write_generation(
        self,
        document_index: faiss.Index,
        graph_index: faiss.Index,
        manifest: GenerationManifest,
    ) -> dict[str, Any]:
        final_manifest = asdict(manifest)
        temp_dir = self.root / f".{manifest.generation}.tmp"
        final_dir = self.root / manifest.generation
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            _write_faiss_index(document_index, temp_dir / "documents.index")
            _write_faiss_index(graph_index, temp_dir / "graph.index")
            (temp_dir / "manifest.json").write_text(
                json.dumps(final_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temp_dir, final_dir)
            pointer = self.current_file.with_suffix(".tmp")
            pointer.write_text(manifest.generation, encoding="utf-8")
            os.replace(pointer, self.current_file)
            self._snapshot = IndexSnapshot(
                document_index, graph_index, final_manifest, self.provider
            )
            self.library_id = manifest.library_id
            self.provider_id = manifest.provider_id
            self.provider_revision = manifest.provider_revision
            self.provider_config_sha256 = manifest.provider_config_sha256
            self.provider_model = manifest.configured_model
            return final_manifest
        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    async def apply_delta(
        self,
        *,
        add_documents: list[dict[str, Any]] | None = None,
        add_graph_entries: list[dict[str, Any]] | None = None,
        remove_document_ids: set[int] | None = None,
        remove_graph_entry_ids: set[int] | None = None,
        expected_document_ids_after: set[int],
        expected_graph_ids_after: set[int],
        reason: str = "incremental",
    ) -> dict[str, Any]:
        add_documents = add_documents or []
        add_graph_entries = add_graph_entries or []
        remove_document_ids = set(remove_document_ids or set())
        remove_graph_entry_ids = set(remove_graph_entry_ids or set())
        add_document_ids = {int(item["id"]) for item in add_documents}
        add_graph_ids = {int(item["id"]) for item in add_graph_entries}
        expected_document_ids_after = {
            int(value) for value in expected_document_ids_after
        }
        expected_graph_ids_after = {int(value) for value in expected_graph_ids_after}
        expected_document_ids_before = (
            expected_document_ids_after - add_document_ids
        ) | remove_document_ids
        expected_graph_ids_before = (
            expected_graph_ids_after - add_graph_ids
        ) | remove_graph_entry_ids

        async with self._swap_lock:
            snapshot = self._snapshot
            if snapshot is None:
                await self.initialize()
                snapshot = self._snapshot
            if snapshot is None:
                raise RuntimeError("索引尚未初始化")

            current_document_ids = self._index_ids(snapshot.document_index)
            current_graph_ids = self._index_ids(snapshot.graph_index)
            if (
                current_document_ids != expected_document_ids_before
                or current_graph_ids != expected_graph_ids_before
            ):
                raise RuntimeError(
                    "当前索引与数据库不一致或尚未构建，"
                    "请先手动执行一次全量索引重建后再写入记忆"
                )

            provider = snapshot.provider
            dimension = await provider.get_dimension()
            if snapshot.document_index.d != dimension or snapshot.graph_index.d != dimension:
                raise RuntimeError(
                    "当前索引维度与 Provider 不一致，请先手动执行全量索引重建"
                )
            provider_status = await provider.test_connection()
            if not provider_status.get("available"):
                raise RuntimeError(
                    f"Provider 测试失败: {provider_status.get('error') or '未知错误'}"
                )

            document_index = faiss.clone_index(snapshot.document_index)
            graph_index = faiss.clone_index(snapshot.graph_index)
            self._remove_ids(document_index, remove_document_ids | add_document_ids)
            self._remove_ids(graph_index, remove_graph_entry_ids | add_graph_ids)

            norm_values: list[float] = []

            async def add_texts(
                index: faiss.Index,
                rows: list[dict[str, Any]],
                *,
                text_key: str,
            ) -> None:
                if not rows:
                    return
                texts = [
                    _clip_for_embedding(
                        row.get(text_key) or "",
                        provider,
                        DEFAULT_DOCUMENT_EMBED_CHARS,
                    )
                    for row in rows
                ]
                ids = np.asarray([int(row["id"]) for row in rows], dtype=np.int64)
                vectors = await provider.get_embeddings(texts)
                matrix = np.asarray(vectors, dtype=np.float32)
                minimum, maximum, total_norm = self._validate_matrix(
                    matrix,
                    expected_rows=len(rows),
                    dimension=dimension,
                )
                norm_values.extend([minimum, maximum])
                if rows:
                    norm_values.append(total_norm / len(rows))
                faiss.normalize_L2(matrix)
                index.add_with_ids(matrix, ids)

            await add_texts(document_index, add_documents, text_key="text")
            await add_texts(graph_index, add_graph_entries, text_key="content")

            final_document_ids = self._index_ids(document_index)
            final_graph_ids = self._index_ids(graph_index)
            if final_document_ids != expected_document_ids_after:
                raise RuntimeError("增量文档索引 ID 集校验失败")
            if final_graph_ids != expected_graph_ids_after:
                raise RuntimeError("增量图索引 ID 集校验失败")

            previous = snapshot.manifest or {}
            generation = time.strftime("gen-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
            vector_norm_min = float(previous.get("vector_norm_min") or 0.0)
            vector_norm_max = float(previous.get("vector_norm_max") or 0.0)
            vector_norm_mean = float(previous.get("vector_norm_mean") or 0.0)
            if norm_values:
                vector_norm_min = min([value for value in norm_values if value >= 0])
                vector_norm_max = max(norm_values)
                vector_norm_mean = sum(norm_values) / len(norm_values)
            manifest = GenerationManifest(
                generation=generation,
                created_at=time.time(),
                library_id=self.library_id,
                provider_id=self.provider_id,
                provider_revision=self.provider_revision,
                provider_config_sha256=self.provider_config_sha256,
                provider_type=type(provider).__name__,
                configured_model=(
                    getattr(getattr(provider, "config", None), "model", "")
                    or self.provider_model
                ),
                resolved_model=str(
                    provider_status.get("resolved_model")
                    or previous.get("resolved_model")
                    or ""
                ),
                dimension=dimension,
                metric="IndexIDMap(IndexFlatL2)",
                document_count=len(expected_document_ids_after),
                graph_entry_count=len(expected_graph_ids_after),
                document_ids_sha256=self._id_hash(expected_document_ids_after),
                graph_ids_sha256=self._id_hash(expected_graph_ids_after),
                vector_norm_min=round(vector_norm_min, 8),
                vector_norm_max=round(vector_norm_max, 8),
                vector_norm_mean=round(vector_norm_mean, 8),
            )
            final_manifest = self._write_generation(
                document_index, graph_index, manifest
            )
            return {
                "mode": "incremental",
                "reason": reason,
                "status": "completed",
                "generation": final_manifest.get("generation"),
                "document_vectors": len(expected_document_ids_after),
                "graph_vectors": len(expected_graph_ids_after),
                "added_documents": sorted(add_document_ids),
                "removed_documents": sorted(remove_document_ids),
                "added_graph_entries": sorted(add_graph_ids),
                "removed_graph_entries": sorted(remove_graph_entry_ids),
            }

    async def upsert_memories(
        self,
        memory_ids: list[int],
        *,
        include_documents: bool = True,
        replace_documents: bool = False,
        remove_graph_entry_ids: set[int] | None = None,
        reason: str = "memory_upsert",
    ) -> dict[str, Any]:
        documents = (
            await self.storage.documents_for_ids(memory_ids)
            if include_documents
            else []
        )
        graph_entries = await self.storage.graph_entries_for_memory_ids(memory_ids)
        return await self.apply_delta(
            add_documents=documents,
            add_graph_entries=graph_entries,
            remove_document_ids={
                int(item["id"]) for item in documents
            } if replace_documents else set(),
            remove_graph_entry_ids=set(remove_graph_entry_ids or set()),
            expected_document_ids_after=set(await self.storage.document_ids()),
            expected_graph_ids_after=set(await self.storage.graph_entry_ids()),
            reason=reason,
        )

    async def delete_memories_incremental(
        self,
        memory_ids: list[int],
        *,
        graph_entry_ids: set[int],
        reason: str = "memory_delete",
    ) -> dict[str, Any]:
        document_ids = {int(value) for value in memory_ids}
        expected_document_ids_after = set(await self.storage.document_ids()) - document_ids
        expected_graph_ids_after = set(await self.storage.graph_entry_ids()) - {
            int(value) for value in graph_entry_ids
        }
        return await self.apply_delta(
            remove_document_ids=document_ids,
            remove_graph_entry_ids={int(value) for value in graph_entry_ids},
            expected_document_ids_after=expected_document_ids_after,
            expected_graph_ids_after=expected_graph_ids_after,
            reason=reason,
        )

    async def rebuild(
        self,
        *,
        batch_size: int = 64,
        concurrency: int = 2,
        embedding_batch_size: int | None = None,
        tasks_limit: int | None = None,
        max_retries: int = 1,
        retry_base_delay: float = 0.0,
        batch_delay: float = 0.0,
        request_delay: float = 0.0,
        max_failure_ratio: float = 0.0,
        progress: Callable[[float, str], Any] | None = None,
        provider: EmbeddingProvider | None = None,
        library_id: str | None = None,
        provider_id: str | None = None,
        provider_revision: int | None = None,
        provider_config_sha256: str | None = None,
        provider_model: str | None = None,
    ) -> dict[str, Any]:
        candidate = provider or self.provider
        read_batch_size = max(1, int(batch_size))
        embed_batch_size = max(1, int(embedding_batch_size or batch_size))
        worker_limit = max(1, int(tasks_limit or concurrency))
        max_retries = max(1, int(max_retries))
        retry_base_delay = max(0.0, float(retry_base_delay))
        batch_delay = max(0.0, float(batch_delay))
        request_delay = max(0.0, float(request_delay))
        max_failure_ratio = max(0.0, float(max_failure_ratio))
        dimension = await candidate.get_dimension()
        provider_status = await candidate.test_connection()
        if not provider_status.get("available"):
            raise RuntimeError(
                f"Provider 测试失败: {provider_status.get('error') or '未知错误'}"
            )
        generation = time.strftime("gen-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        temp_dir = self.root / f".{generation}.tmp"
        final_dir = self.root / generation
        temp_dir.mkdir(parents=True, exist_ok=False)
        doc_index = self._empty_index(dimension)
        graph_index = self._empty_index(dimension)

        stats = await self.storage.statistics()
        doc_total = int(stats["total_memories"])
        graph_total = int(stats["graph_entries"])
        total = max(1, doc_total + graph_total)
        done = 0
        norm_min = math.inf
        norm_max = 0.0
        norm_sum = 0.0
        norm_count = 0
        semaphore = asyncio.Semaphore(worker_limit)

        async def embed_chunks(
            texts: list[str], ids: list[int], index: faiss.Index
        ) -> None:
            nonlocal done, norm_min, norm_max, norm_sum, norm_count
            chunks = [
                (texts[i : i + embed_batch_size], ids[i : i + embed_batch_size])
                for i in range(0, len(texts), embed_batch_size)
            ]

            async def get_embeddings_with_retry(chunk_texts: list[str]):
                last_exc: Exception | None = None
                for attempt in range(max_retries):
                    try:
                        return await candidate.get_embeddings(chunk_texts)
                    except Exception as exc:
                        last_exc = exc
                        if attempt + 1 >= max_retries:
                            break
                        delay = retry_base_delay * (2**attempt)
                        if delay > 0:
                            await asyncio.sleep(delay)
                assert last_exc is not None
                raise last_exc

            async def run(chunk_texts: list[str], chunk_ids: list[int]):
                async with semaphore:
                    vectors = await get_embeddings_with_retry(chunk_texts)
                    matrix = np.asarray(vectors, dtype=np.float32)
                    minimum, maximum, total_norm = self._validate_matrix(
                        matrix,
                        expected_rows=len(chunk_ids),
                        dimension=dimension,
                    )
                    faiss.normalize_L2(matrix)
                    return (
                        matrix,
                        np.asarray(chunk_ids, dtype=np.int64),
                        minimum,
                        maximum,
                        total_norm,
                    )

            for start in range(0, len(chunks), worker_limit):
                results = await asyncio.gather(
                    *[
                        run(chunk_texts, chunk_ids)
                        for chunk_texts, chunk_ids in chunks[
                            start : start + worker_limit
                        ]
                    ]
                )
                for matrix, id_array, minimum, maximum, total_norm in results:
                    index.add_with_ids(matrix, id_array)
                    done += len(id_array)
                    norm_min = min(norm_min, minimum)
                    norm_max = max(norm_max, maximum)
                    norm_sum += total_norm
                    norm_count += len(id_array)
                if progress:
                    await progress(
                        done / total,
                        f"已生成 {done}/{doc_total + graph_total} 条向量",
                    )

        try:
            document_ids: list[int] = []
            async for batch in self.storage.iter_documents(batch_size=read_batch_size):
                ids = [int(item["id"]) for item in batch]
                await embed_chunks(
                    [
                        _clip_for_embedding(
                            item["text"],
                            candidate,
                            DEFAULT_DOCUMENT_EMBED_CHARS,
                        )
                        for item in batch
                    ],
                    ids,
                    doc_index,
                )
                document_ids.extend(ids)
                if batch_delay > 0 and len(batch) >= read_batch_size and done < total:
                    await asyncio.sleep(batch_delay)

            graph_ids: list[int] = []
            async for batch in self.storage.iter_graph_entries(batch_size=read_batch_size):
                ids = [int(item["id"]) for item in batch]
                await embed_chunks(
                    [
                        _clip_for_embedding(
                            item["content"],
                            candidate,
                            DEFAULT_DOCUMENT_EMBED_CHARS,
                        )
                        for item in batch
                    ],
                    ids,
                    graph_index,
                )
                graph_ids.extend(ids)
                if batch_delay > 0 and len(batch) >= read_batch_size and done < total:
                    await asyncio.sleep(batch_delay)

            if doc_index.ntotal != doc_total or graph_index.ntotal != graph_total:
                raise RuntimeError(
                    "zero-loss rebuild failed: "
                    f"documents={doc_index.ntotal}/{doc_total}, "
                    f"graph={graph_index.ntotal}/{graph_total}"
                )
            if set(map(int, faiss.vector_to_array(doc_index.id_map))) != set(
                document_ids
            ):
                raise RuntimeError("document vector ID set mismatch")
            if set(map(int, faiss.vector_to_array(graph_index.id_map))) != set(
                graph_ids
            ):
                raise RuntimeError("graph vector ID set mismatch")
            async with self.storage.connect() as db:
                document_samples = [
                    (int(row["id"]), str(row["text"]))
                    for row in await (
                        await db.execute(
                            "SELECT id,text FROM documents ORDER BY id LIMIT 3"
                        )
                    ).fetchall()
                ]
                graph_samples = [
                    (int(row["id"]), str(row["content"]))
                    for row in await (
                        await db.execute(
                            "SELECT id,content FROM graph_entries ORDER BY id LIMIT 3"
                        )
                    ).fetchall()
                ]
            await self._validate_sample_recall(
                candidate, doc_index, document_samples, "文档索引"
            )
            await self._validate_sample_recall(
                candidate, graph_index, graph_samples, "图谱索引"
            )

            manifest = GenerationManifest(
                generation=generation,
                created_at=time.time(),
                library_id=library_id if library_id is not None else self.library_id,
                provider_id=provider_id if provider_id is not None else self.provider_id,
                provider_revision=(
                    provider_revision
                    if provider_revision is not None
                    else self.provider_revision
                ),
                provider_config_sha256=(
                    provider_config_sha256
                    if provider_config_sha256 is not None
                    else self.provider_config_sha256
                ),
                provider_type=type(candidate).__name__,
                configured_model=(
                    provider_model
                    or getattr(getattr(candidate, "config", None), "model", "")
                    or self.provider_model
                ),
                resolved_model=str(provider_status.get("resolved_model") or ""),
                dimension=dimension,
                metric="IndexIDMap(IndexFlatL2)",
                document_count=doc_total,
                graph_entry_count=graph_total,
                document_ids_sha256=hashlib.sha256(
                    ",".join(map(str, document_ids)).encode()
                ).hexdigest(),
                graph_ids_sha256=hashlib.sha256(
                    ",".join(map(str, graph_ids)).encode()
                ).hexdigest(),
                vector_norm_min=round(0.0 if norm_count == 0 else norm_min, 8),
                vector_norm_max=round(norm_max, 8),
                vector_norm_mean=round(
                    0.0 if norm_count == 0 else norm_sum / norm_count, 8
                ),
            )
            _write_faiss_index(doc_index, temp_dir / "documents.index")
            _write_faiss_index(graph_index, temp_dir / "graph.index")
            (temp_dir / "manifest.json").write_text(
                json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temp_dir, final_dir)
            async with self._swap_lock:
                pointer = self.current_file.with_suffix(".tmp")
                pointer.write_text(generation, encoding="utf-8")
                os.replace(pointer, self.current_file)
                self._snapshot = IndexSnapshot(
                    doc_index, graph_index, asdict(manifest), candidate
                )
                self.library_id = manifest.library_id
                self.provider_id = manifest.provider_id
                self.provider_revision = manifest.provider_revision
                self.provider_config_sha256 = manifest.provider_config_sha256
                self.provider_model = manifest.configured_model
            if progress:
                await progress(1.0, "索引 generation 与 Provider 已原子切换")
            return asdict(manifest)
        except Exception:
            if temp_dir.exists():
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    async def search_documents(
        self, query: str, k: int, fetch_k: int | None = None
    ) -> list[tuple[int, float]]:
        snapshot = self._snapshot
        if snapshot is None or snapshot.document_index.ntotal == 0:
            return []
        vector = np.asarray(
            [
                await snapshot.provider.get_embedding(
                    _clip_for_embedding(
                        query,
                        snapshot.provider,
                        DEFAULT_QUERY_EMBED_CHARS,
                    ),
                )
            ],
            dtype=np.float32,
        )
        faiss.normalize_L2(vector)
        distances, ids = snapshot.document_index.search(vector, fetch_k or k)
        return [
            (int(item_id), float(1.0 - distance / 2.0))
            for item_id, distance in zip(ids[0], distances[0], strict=True)
            if int(item_id) >= 0
        ][:k]

    async def search_graph(
        self, query: str, k: int, fetch_k: int | None = None
    ) -> list[tuple[int, float]]:
        snapshot = self._snapshot
        if snapshot is None or snapshot.graph_index.ntotal == 0:
            return []
        vector = np.asarray(
            [
                await snapshot.provider.get_embedding(
                    _clip_for_embedding(
                        query,
                        snapshot.provider,
                        DEFAULT_QUERY_EMBED_CHARS,
                    ),
                )
            ],
            dtype=np.float32,
        )
        faiss.normalize_L2(vector)
        distances, ids = snapshot.graph_index.search(vector, fetch_k or k)
        return [
            (int(item_id), float(1.0 - distance / 2.0))
            for item_id, distance in zip(ids[0], distances[0], strict=True)
            if int(item_id) >= 0
        ][:k]

    def status(self) -> dict[str, Any]:
        snapshot = self._snapshot
        return {
            "generation": (snapshot.manifest or {}).get("generation")
            if snapshot
            else None,
            "manifest": snapshot.manifest if snapshot else None,
            "document_vectors": int(
                snapshot.document_index.ntotal if snapshot else 0
            ),
            "graph_vectors": int(snapshot.graph_index.ntotal if snapshot else 0),
            "provider_id": self.provider_id,
            "provider_revision": self.provider_revision,
        }
