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

from ...resource_limits import (
    configure_numeric_thread_environment,
    configured_faiss_threads,
)
from ...faiss_runtime import load_faiss

BLAS_THREAD_COUNT = configure_numeric_thread_environment()

import numpy as np

faiss = load_faiss()

from ...context_lengths import (
    MANUAL_CONTEXT_FALLBACK_TOKENS,
    MIN_VALID_CONTEXT_TOKENS,
    static_context_table_metadata,
)
from ...io_utils import (
    atomic_write_json,
    read_ab_checkpoint,
    run_blocking,
    write_ab_checkpoint,
)
from ...logger import logger
from ...providers import EmbeddingProvider
from .storage import Storage
from ...task_control import JobExecutionContext, JobInterrupted


DEFAULT_DOCUMENT_EMBED_CHARS = 4000
DEFAULT_QUERY_EMBED_CHARS = 2000
CONTEXT_CHUNK_CHAR_SAFETY_RATIO = 0.75
MIN_CONTEXT_CHUNK_CHARS = 64
EMBEDDING_CHUNKING_POLICY = "stable_char_chunk_mean_pool_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def provider_functional_sha256(provider: EmbeddingProvider) -> str:
    """Hash only settings that can change embedding output semantics."""

    config = getattr(provider, "config", None)
    try:
        payload = asdict(config)
    except (TypeError, ValueError):
        payload = dict(getattr(config, "__dict__", {}) or {})
    for key in (
        "display_name",
        "context_length_mode",
        "max_context_tokens_source",
        "index_rebuild_settings",
        "batch_size",
        "concurrency",
        "max_retries",
    ):
        payload.pop(key, None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _configured_faiss_threads() -> int:
    return configured_faiss_threads()


FAISS_THREAD_COUNT = _configured_faiss_threads()
faiss.omp_set_num_threads(FAISS_THREAD_COUNT)


@dataclass(slots=True)
class EmbeddingInputWarningTracker:
    label: str
    threshold: int
    provider_id: str = ""
    total: int = 0
    over_limit: int = 0
    max_chars: int = 0
    chunked_items: int = 0
    total_chunks: int = 0
    max_chunks_per_item: int = 0
    reported: bool = False

    def prepare(self, text: Any) -> str:
        value = str(text or "")
        length = len(value)
        self.total += 1
        self.max_chars = max(self.max_chars, length)
        if length > self.threshold:
            self.over_limit += 1
        return value

    def record_chunks(self, chunk_count: int) -> None:
        chunk_count = max(1, int(chunk_count))
        self.total_chunks += chunk_count
        self.max_chunks_per_item = max(self.max_chunks_per_item, chunk_count)
        if chunk_count > 1:
            self.chunked_items += 1

    def flush(self) -> None:
        if self.reported or (self.over_limit <= 0 and self.chunked_items <= 0):
            return
        self.reported = True
        logger.warning(
            "Embedding 输入较长：label=%s provider=%s over_soft_limit=%s/%s "
            "threshold_chars=%s max_chars=%s chunked_items=%s total_chunks=%s "
            "max_chunks_per_item=%s policy=%s",
            self.label,
            self.provider_id or "-",
            self.over_limit,
            self.total,
            self.threshold,
            self.max_chars,
            self.chunked_items,
            self.total_chunks,
            self.max_chunks_per_item,
            EMBEDDING_CHUNKING_POLICY,
        )


def _prepare_for_embedding(
    text: Any,
    provider: EmbeddingProvider,
    default_limit: int,
    tracker: EmbeddingInputWarningTracker | None = None,
) -> str:
    if tracker is not None:
        return tracker.prepare(text)
    return str(text or "")


async def _provider_capability_snapshot(
    provider: EmbeddingProvider,
    *,
    provider_id: str,
    provider_revision: int,
    provider_config_sha256: str,
    provider_status: dict[str, Any] | None = None,
    force_probe: bool = False,
    minimum_trusted_tokens: int = 128,
) -> dict[str, Any]:
    config = getattr(provider, "config", None)
    detected_tokens = 0
    detected_source = ""
    context_length_mode = str(
        getattr(config, "context_length_mode", "auto") if config is not None else "auto"
    )
    minimum_trusted_tokens = max(
        int(minimum_trusted_tokens or 0), MIN_VALID_CONTEXT_TOKENS
    )
    if config is not None and (context_length_mode == "manual" or not force_probe):
        detected_tokens = int(getattr(config, "max_context_tokens", 0) or 0)
        detected_source = str(getattr(config, "max_context_tokens_source", "") or "")
        if detected_tokens >= minimum_trusted_tokens:
            if not detected_source:
                detected_source = "config:manual"
            elif not detected_source.startswith("config:"):
                detected_source = f"config:{detected_source}"
    if detected_tokens < minimum_trusted_tokens:
        try:
            detected = await provider.detect_context_length()
            detected_tokens = int(detected.get("max_context_tokens") or 0)
            detected_source = str(detected.get("max_context_tokens_source") or "")
        except Exception as exc:
            logger.warning(
                f"Embedding Provider 上下文能力探测失败: {exc}"
            )
        if detected_tokens < minimum_trusted_tokens:
            existing_tokens = (
                int(getattr(config, "max_context_tokens", 0) or 0)
                if config is not None
                else 0
            )
            detected_tokens = (
                existing_tokens
                if existing_tokens >= minimum_trusted_tokens
                else MANUAL_CONTEXT_FALLBACK_TOKENS
            )
            detected_source = "manual:fallback-undetected"
            context_length_mode = "manual"
        if detected_tokens < minimum_trusted_tokens:
            raise RuntimeError(
                "Embedding Provider 上下文能力探测失败："
                f"tokens={detected_tokens}, minimum={minimum_trusted_tokens}"
            )
    if config is not None and detected_tokens > 0:
        try:
            setattr(config, "context_length_mode", context_length_mode)
            setattr(config, "max_context_tokens", detected_tokens)
            setattr(config, "max_context_tokens_source", detected_source)
        except Exception:
            pass
    chunk_char_limit = (
        max(MIN_CONTEXT_CHUNK_CHARS, int(detected_tokens * CONTEXT_CHUNK_CHAR_SAFETY_RATIO))
        if detected_tokens > 0
        else 0
    )
    status = provider_status or {}
    static_table = static_context_table_metadata()
    return {
        "provider_id": provider_id or getattr(config, "id", "") or "",
        "provider_revision": int(provider_revision or 0),
        "provider_config_sha256": provider_config_sha256 or "",
        "provider_type": type(provider).__name__,
        "configured_model": str(getattr(config, "model", "") or ""),
        "resolved_model": str(status.get("resolved_model") or ""),
        "detected_max_context_tokens": detected_tokens,
        "max_context_tokens_source": detected_source,
        "context_length_mode": context_length_mode,
        "context_length_table_version": static_table.get("version", ""),
        "context_length_table_sha256": static_table.get("sha256", ""),
        "detected_at": time.time(),
        "chunking_policy": EMBEDDING_CHUNKING_POLICY,
        "chunk_char_limit": chunk_char_limit,
        "chunk_char_safety_ratio": CONTEXT_CHUNK_CHAR_SAFETY_RATIO,
    }


def _split_for_embedding(
    text: str,
    *,
    capability: dict[str, Any],
    tracker: EmbeddingInputWarningTracker,
) -> list[str]:
    chunk_char_limit = int(capability.get("chunk_char_limit") or 0)
    if chunk_char_limit <= 0 or len(text) <= chunk_char_limit:
        tracker.record_chunks(1)
        return [text]
    chunks = [
        text[index : index + chunk_char_limit]
        for index in range(0, len(text), chunk_char_limit)
    ]
    tracker.record_chunks(len(chunks))
    return chunks


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


def _persist_generation_files(
    document_index: faiss.Index,
    graph_index: faiss.Index,
    manifest: dict[str, Any],
    temp_dir: Path,
    final_dir: Path,
) -> None:
    _write_faiss_index(document_index, temp_dir / "documents.index")
    _write_faiss_index(graph_index, temp_dir / "graph.index")
    (temp_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp_dir, final_dir)


def _load_checkpoint_segment(
    path: Path,
    expected_sha256: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists() or _sha256_file(path) != expected_sha256:
        raise ValueError("checkpoint segment hash mismatch")
    with np.load(path, allow_pickle=False) as payload:
        ids = np.asarray(payload["ids"], dtype=np.int64)
        vectors = np.asarray(payload["vectors"], dtype=np.float32)
    return ids, vectors


def _persist_checkpoint_segment(
    temp_path: Path,
    final_path: Path,
    ids: list[int],
    matrix: np.ndarray,
) -> str:
    with temp_path.open("wb") as handle:
        np.savez(handle, ids=np.asarray(ids, dtype=np.int64), vectors=matrix)
    os.replace(temp_path, final_path)
    return _sha256_file(final_path)


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
    embedding_capability: dict[str, Any]
    chunked_document_count: int = 0
    chunked_graph_entry_count: int = 0
    chunked_query_count: int = 0
    total_embedding_chunks: int = 0
    max_chunks_per_item: int = 1
    graph_vector_granularity: str = "memory"
    graph_source_memory_count: int = 0
    graph_vector_count: int = 0
    graph_vector_content_sha256: str = ""
    provider_functional_sha256: str = ""


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

    def graph_vector_granularity(self) -> str:
        manifest = self.manifest
        if manifest is None:
            return "memory"
        value = str(manifest.get("graph_vector_granularity") or "entry")
        return "memory" if value == "memory" else "entry"

    async def initialize(self) -> None:
        snapshot = await run_blocking(self._load_current_snapshot)
        if snapshot is not None:
            self._snapshot = snapshot
            return
        dimension = await self._initial_provider.get_dimension()
        self._snapshot = IndexSnapshot(
            self._empty_index(dimension),
            self._empty_index(dimension),
            None,
            self._initial_provider,
        )

    def _load_current_snapshot(self) -> IndexSnapshot | None:
        if self.current_file.exists():
            generation = self.current_file.read_text(encoding="utf-8").strip()
            path = self.root / generation
            try:
                document_index = _read_faiss_index(path / "documents.index")
                graph_index = _read_faiss_index(path / "graph.index")
                manifest = json.loads(
                    (path / "manifest.json").read_text(encoding="utf-8")
                )
                return IndexSnapshot(
                    document_index,
                    graph_index,
                    manifest,
                    self._initial_provider,
                )
            except Exception:
                pass
        return None

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
    async def _embed_texts_aggregated(
        provider: EmbeddingProvider,
        texts: list[str],
        *,
        dimension: int,
        capability: dict[str, Any],
        tracker: EmbeddingInputWarningTracker,
        request_embeddings: Callable[[list[str]], Any],
        request_batch_size: int,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, dimension), dtype=np.float32)
        request_batch_size = max(1, int(request_batch_size))
        fragment_texts: list[str] = []
        fragment_owner_indexes: list[int] = []
        for owner_index, text in enumerate(texts):
            prepared = _prepare_for_embedding(text, provider, tracker.threshold, tracker)
            fragments = _split_for_embedding(
                prepared,
                capability=capability,
                tracker=tracker,
            )
            for fragment in fragments:
                fragment_texts.append(fragment)
                fragment_owner_indexes.append(owner_index)
        owner_vectors: list[list[np.ndarray]] = [[] for _ in texts]
        for start in range(0, len(fragment_texts), request_batch_size):
            end = start + request_batch_size
            batch_texts = fragment_texts[start:end]
            raw_vectors = await request_embeddings(batch_texts)
            matrix = np.asarray(raw_vectors, dtype=np.float32)
            IndexManager._validate_matrix(
                matrix,
                expected_rows=len(batch_texts),
                dimension=dimension,
            )
            fragment_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / fragment_norms
            for offset, vector in enumerate(matrix):
                owner = fragment_owner_indexes[start + offset]
                owner_vectors[owner].append(np.asarray(vector, dtype=np.float32))
        rows: list[np.ndarray] = []
        for vectors in owner_vectors:
            if not vectors:
                raise RuntimeError("embedding chunk aggregation produced no vectors")
            if len(vectors) == 1:
                rows.append(vectors[0])
            else:
                stacked = np.vstack(vectors).astype(np.float32, copy=False)
                rows.append(np.mean(stacked, axis=0, dtype=np.float32))
        output = np.vstack(rows).astype(np.float32, copy=False)
        output_norms = np.linalg.norm(output, axis=1, keepdims=True)
        if np.any(output_norms <= 0):
            raise RuntimeError("embedding aggregation produced a zero vector")
        return (output / output_norms).astype(np.float32, copy=False)

    @staticmethod
    async def _validate_sample_recall(
        provider: EmbeddingProvider,
        index: faiss.Index,
        samples: list[tuple[int, str]],
        label: str,
        *,
        dimension: int | None = None,
        capability: dict[str, Any] | None = None,
    ) -> None:
        if not samples:
            return
        if dimension is None:
            dimension = await provider.get_dimension()
        if capability is None:
            capability = await _provider_capability_snapshot(
                provider,
                provider_id=str(getattr(getattr(provider, "config", None), "id", "") or ""),
                provider_revision=0,
                provider_config_sha256="",
                force_probe=False,
            )
        tracker = EmbeddingInputWarningTracker(
            f"sample_recall_{label}",
            DEFAULT_DOCUMENT_EMBED_CHARS,
            str(capability.get("provider_id") or ""),
        )
        vectors = await IndexManager._embed_texts_aggregated(
            provider,
            [text for _, text in samples],
            dimension=dimension,
            capability=capability,
            tracker=tracker,
            request_embeddings=provider.get_embeddings,
            request_batch_size=max(1, int(capability.get("sample_batch_size") or 8)),
        )
        tracker.flush()
        if vectors.ndim != 2:
            raise RuntimeError(f"{label} sample recall vector shape is invalid")
        k = min(10, int(index.ntotal))
        _, result_ids = index.search(vectors, k)
        for (expected_id, _), row in zip(samples, result_ids, strict=True):
            if expected_id not in {int(item) for item in row if int(item) >= 0}:
                raise RuntimeError(
                    f"{label} sample recall did not return itself: {expected_id}"
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

            capability = await _provider_capability_snapshot(
                provider,
                provider_id=self.provider_id,
                provider_revision=self.provider_revision,
                provider_config_sha256=self.provider_config_sha256,
                provider_status=provider_status,
                force_probe=False,
            )

            document_index = faiss.clone_index(snapshot.document_index)
            graph_index = faiss.clone_index(snapshot.graph_index)
            self._remove_ids(document_index, remove_document_ids | add_document_ids)
            self._remove_ids(graph_index, remove_graph_entry_ids | add_graph_ids)

            norm_values: list[float] = []
            document_tracker = EmbeddingInputWarningTracker(
                "incremental_index_update_documents",
                DEFAULT_DOCUMENT_EMBED_CHARS,
                self.provider_id,
            )
            graph_tracker = EmbeddingInputWarningTracker(
                "incremental_index_update_graph",
                DEFAULT_DOCUMENT_EMBED_CHARS,
                self.provider_id,
            )

            async def add_texts(
                index: faiss.Index,
                rows: list[dict[str, Any]],
                *,
                text_key: str,
            ) -> None:
                if not rows:
                    return
                texts = [str(row.get(text_key) or "") for row in rows]
                ids = np.asarray([int(row["id"]) for row in rows], dtype=np.int64)
                tracker = document_tracker if text_key == "text" else graph_tracker
                matrix = await self._embed_texts_aggregated(
                    provider,
                    texts,
                    dimension=dimension,
                    capability=capability,
                    tracker=tracker,
                    request_embeddings=provider.get_embeddings,
                    request_batch_size=max(1, len(rows)),
                )
                minimum, maximum, total_norm = self._validate_matrix(
                    matrix,
                    expected_rows=len(rows),
                    dimension=dimension,
                )
                norm_values.extend([minimum, maximum])
                if rows:
                    norm_values.append(total_norm / len(rows))
                index.add_with_ids(matrix, ids)

            try:
                await add_texts(document_index, add_documents, text_key="text")
                await add_texts(graph_index, add_graph_entries, text_key="content")
            finally:
                document_tracker.flush()
                graph_tracker.flush()

            final_document_ids = self._index_ids(document_index)
            final_graph_ids = self._index_ids(graph_index)
            if final_document_ids != expected_document_ids_after:
                raise RuntimeError("增量文档索引 ID 集校验失败")
            if final_graph_ids != expected_graph_ids_after:
                raise RuntimeError("增量图索引 ID 集校验失败")

            previous = snapshot.manifest or {}
            graph_granularity = self.graph_vector_granularity()
            graph_source_memory_count = 0
            graph_entry_count = len(expected_graph_ids_after)
            graph_content_hash = ""
            if graph_granularity == "memory":
                graph_rows_after = await self.storage.graph_memories_for_ids(
                    expected_graph_ids_after
                )
                graph_source_memory_count = len(graph_rows_after)
                graph_entry_count = sum(
                    int(item.get("graph_entry_count") or 0)
                    for item in graph_rows_after
                )
                content_hasher = hashlib.sha256()
                for item in graph_rows_after:
                    content = str(item.get("content") or "")
                    content_hasher.update(
                        f"{int(item['id'])}:{len(content)}:".encode()
                    )
                    content_hasher.update(content.encode("utf-8"))
                    content_hasher.update(b"\0")
                graph_content_hash = content_hasher.hexdigest()
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
                graph_entry_count=graph_entry_count,
                document_ids_sha256=self._id_hash(expected_document_ids_after),
                graph_ids_sha256=self._id_hash(expected_graph_ids_after),
                vector_norm_min=round(vector_norm_min, 8),
                vector_norm_max=round(vector_norm_max, 8),
                vector_norm_mean=round(vector_norm_mean, 8),
                embedding_capability=capability,
                chunked_document_count=int(previous.get("chunked_document_count") or 0)
                + document_tracker.chunked_items,
                chunked_graph_entry_count=int(
                    previous.get("chunked_graph_entry_count") or 0
                )
                + graph_tracker.chunked_items,
                total_embedding_chunks=int(previous.get("total_embedding_chunks") or 0)
                + document_tracker.total_chunks
                + graph_tracker.total_chunks,
                max_chunks_per_item=max(
                    int(previous.get("max_chunks_per_item") or 1),
                    document_tracker.max_chunks_per_item,
                    graph_tracker.max_chunks_per_item,
                    1,
                ),
                graph_vector_granularity=graph_granularity,
                graph_source_memory_count=graph_source_memory_count,
                graph_vector_count=len(expected_graph_ids_after),
                graph_vector_content_sha256=graph_content_hash,
                provider_functional_sha256=provider_functional_sha256(
                    snapshot.provider
                ),
            )
            final_manifest = await run_blocking(
                self._write_generation,
                document_index,
                graph_index,
                manifest,
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
        documents = [
            item
            for item in documents
            if str((item.get("metadata") or {}).get("status") or "active")
            == "active"
        ]
        memory_granularity = self.graph_vector_granularity() == "memory"
        graph_entries = (
            await self.storage.graph_memories_for_ids(memory_ids)
            if memory_granularity
            else await self.storage.graph_entries_for_memory_ids(memory_ids)
        )
        graph_ids_after = (
            set(await self.storage.graph_memory_ids())
            if memory_granularity
            else set(await self.storage.graph_entry_ids())
        )
        return await self.apply_delta(
            add_documents=documents,
            add_graph_entries=graph_entries,
            remove_document_ids={
                int(item["id"]) for item in documents
            } if replace_documents else set(),
            remove_graph_entry_ids=(
                {int(value) for value in memory_ids}
                if memory_granularity and replace_documents
                else set(remove_graph_entry_ids or set())
            ),
            expected_document_ids_after=set(await self.storage.document_ids()),
            expected_graph_ids_after=graph_ids_after,
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
        memory_granularity = self.graph_vector_granularity() == "memory"
        removed_graph_ids = (
            {int(value) for value in memory_ids}
            if memory_granularity
            else {int(value) for value in graph_entry_ids}
        )
        expected_graph_ids_after = (
            set(await self.storage.graph_memory_ids())
            if memory_granularity
            else set(await self.storage.graph_entry_ids())
        ) - removed_graph_ids
        return await self.apply_delta(
            remove_document_ids=document_ids,
            remove_graph_entry_ids=removed_graph_ids,
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
        job_context: JobExecutionContext | None = None,
        checkpoint_dir: Path | None = None,
    ) -> dict[str, Any]:
        if job_context is not None and checkpoint_dir is not None:
            return await self._rebuild_resumable(
                batch_size=batch_size,
                concurrency=concurrency,
                embedding_batch_size=embedding_batch_size,
                tasks_limit=tasks_limit,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
                batch_delay=batch_delay,
                request_delay=request_delay,
                max_failure_ratio=max_failure_ratio,
                progress=progress,
                provider=provider,
                library_id=library_id,
                provider_id=provider_id,
                provider_revision=provider_revision,
                provider_config_sha256=provider_config_sha256,
                provider_model=provider_model,
                job_context=job_context,
                checkpoint_dir=checkpoint_dir,
            )
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

        doc_total = len(await self.storage.document_ids())
        graph_total = len(await self.storage.graph_memory_ids())
        graph_entry_total = await self.storage.graph_entry_count()
        total = max(1, doc_total + graph_total)
        done = 0
        norm_min = math.inf
        norm_max = 0.0
        norm_sum = 0.0
        norm_count = 0
        capability = await _provider_capability_snapshot(
            candidate,
            provider_id=provider_id if provider_id is not None else self.provider_id,
            provider_revision=provider_revision
            if provider_revision is not None
            else self.provider_revision,
            provider_config_sha256=provider_config_sha256
            if provider_config_sha256 is not None
            else self.provider_config_sha256,
            provider_status=provider_status,
                force_probe=False,
        )
        semaphore = asyncio.Semaphore(worker_limit)
        document_tracker = EmbeddingInputWarningTracker(
            "index_rebuild_documents",
            DEFAULT_DOCUMENT_EMBED_CHARS,
            provider_id if provider_id is not None else self.provider_id,
        )
        graph_tracker = EmbeddingInputWarningTracker(
            "index_rebuild_graph",
            DEFAULT_DOCUMENT_EMBED_CHARS,
            provider_id if provider_id is not None else self.provider_id,
        )

        async def embed_chunks(
            texts: list[str],
            ids: list[int],
            index: faiss.Index,
            tracker: EmbeddingInputWarningTracker,
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
                    matrix = await self._embed_texts_aggregated(
                        candidate,
                        chunk_texts,
                        dimension=dimension,
                        capability=capability,
                        tracker=tracker,
                        request_embeddings=get_embeddings_with_retry,
                        request_batch_size=embed_batch_size,
                    )
                    minimum, maximum, total_norm = self._validate_matrix(
                        matrix,
                        expected_rows=len(chunk_ids),
                        dimension=dimension,
                    )
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
                    [str(item["text"] or "") for item in batch],
                    ids,
                    doc_index,
                    document_tracker,
                )
                document_ids.extend(ids)
                if batch_delay > 0 and len(batch) >= read_batch_size and done < total:
                    await asyncio.sleep(batch_delay)

            graph_ids: list[int] = []
            graph_content_hasher = hashlib.sha256()
            async for batch in self.storage.iter_graph_memories(batch_size=read_batch_size):
                ids = [int(item["id"]) for item in batch]
                for item in batch:
                    content = str(item["content"] or "")
                    graph_content_hasher.update(
                        f"{int(item['id'])}:{len(content)}:".encode()
                    )
                    graph_content_hasher.update(content.encode("utf-8"))
                    graph_content_hasher.update(b"\0")
                await embed_chunks(
                    [str(item["content"] or "") for item in batch],
                    ids,
                    graph_index,
                    graph_tracker,
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
                            """SELECT id,text FROM documents
                            WHERE COALESCE(json_extract(metadata,'$.status'),'active')='active'
                            ORDER BY id LIMIT 3"""
                        )
                    ).fetchall()
                ]
                graph_samples = [
                    (int(item["id"]), str(item["content"]))
                    for item in await self.storage.graph_memories_for_ids(
                        graph_ids[:3]
                    )
                ]
            await self._validate_sample_recall(
                candidate,
                doc_index,
                document_samples,
                "文档索引",
                dimension=dimension,
                capability=capability,
            )
            await self._validate_sample_recall(
                candidate,
                graph_index,
                graph_samples,
                "图谱索引",
                dimension=dimension,
                capability=capability,
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
                graph_entry_count=graph_entry_total,
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
                embedding_capability=capability,
                chunked_document_count=document_tracker.chunked_items,
                chunked_graph_entry_count=graph_tracker.chunked_items,
                total_embedding_chunks=(
                    document_tracker.total_chunks + graph_tracker.total_chunks
                ),
                max_chunks_per_item=max(
                    document_tracker.max_chunks_per_item,
                    graph_tracker.max_chunks_per_item,
                    1,
                ),
                graph_vector_granularity="memory",
                graph_source_memory_count=graph_total,
                graph_vector_count=graph_total,
                graph_vector_content_sha256=graph_content_hasher.hexdigest(),
                provider_functional_sha256=provider_functional_sha256(candidate),
            )
            await run_blocking(
                _persist_generation_files,
                doc_index,
                graph_index,
                asdict(manifest),
                temp_dir,
                final_dir,
            )
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
        finally:
            document_tracker.flush()
            graph_tracker.flush()

    async def prepare_external_generation(
        self,
        source_dir: Path,
        provider: EmbeddingProvider | None = None,
    ) -> IndexSnapshot:
        """Copy and validate a shadow generation without changing CURRENT."""

        candidate = provider or self.provider
        return await run_blocking(
            self._prepare_external_generation_sync,
            Path(source_dir),
            candidate,
        )

    def _prepare_external_generation_sync(
        self,
        source_dir: Path,
        provider: EmbeddingProvider,
    ) -> IndexSnapshot:
        manifest = json.loads(
            (source_dir / "manifest.json").read_text(encoding="utf-8")
        )
        generation = str(manifest.get("generation") or "").strip()
        if not generation or source_dir.name != generation:
            raise RuntimeError("shadow generation identity is invalid")
        destination = self.root / generation
        if not destination.exists():
            stage = self.root / f".{generation}.external.tmp"
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)
            shutil.copytree(source_dir, stage)
            os.replace(stage, destination)
        elif _sha256_file(destination / "manifest.json") != _sha256_file(
            source_dir / "manifest.json"
        ):
            raise RuntimeError("existing generation conflicts with shadow candidate")

        document_index = _read_faiss_index(destination / "documents.index")
        graph_index = _read_faiss_index(destination / "graph.index")
        document_ids = self._index_ids(document_index)
        graph_ids = self._index_ids(graph_index)
        if len(document_ids) != int(manifest.get("document_count") or 0):
            raise RuntimeError("shadow document vector count is invalid")
        expected_graph = int(
            manifest.get("graph_vector_count")
            or manifest.get("graph_source_memory_count")
            or 0
        )
        if len(graph_ids) != expected_graph:
            raise RuntimeError("shadow graph vector count is invalid")
        return IndexSnapshot(document_index, graph_index, manifest, provider)

    async def activate_prepared_generation(
        self, snapshot: IndexSnapshot
    ) -> dict[str, Any]:
        """Switch CURRENT to an already copied and validated generation."""

        manifest = dict(snapshot.manifest or {})
        generation = str(manifest.get("generation") or "").strip()
        if not generation or not (self.root / generation).is_dir():
            raise RuntimeError("prepared generation is unavailable")
        async with self._swap_lock:
            pointer = self.current_file.with_suffix(".tmp")
            pointer.write_text(generation, encoding="utf-8")
            os.replace(pointer, self.current_file)
            self._snapshot = snapshot
            self.library_id = str(manifest.get("library_id") or self.library_id)
            self.provider_id = str(manifest.get("provider_id") or self.provider_id)
            self.provider_revision = int(
                manifest.get("provider_revision") or self.provider_revision
            )
            self.provider_config_sha256 = str(
                manifest.get("provider_config_sha256")
                or self.provider_config_sha256
            )
            self.provider_model = str(
                manifest.get("configured_model") or self.provider_model
            )
        return manifest

    async def adopt_provider_functional_sha256(self, value: str) -> None:
        """Safely annotate a verified legacy generation without rebuilding it."""

        value = str(value or "").strip()
        snapshot = self._snapshot
        if not value or snapshot is None or not snapshot.manifest:
            return
        async with self._swap_lock:
            manifest = dict(snapshot.manifest)
            if manifest.get("provider_functional_sha256"):
                return
            generation = str(manifest.get("generation") or "")
            path = self.root / generation / "manifest.json"
            if not generation or not path.is_file():
                raise RuntimeError("legacy generation manifest is unavailable")
            manifest["provider_functional_sha256"] = value
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, path)
            self._snapshot = IndexSnapshot(
                snapshot.document_index,
                snapshot.graph_index,
                manifest,
                snapshot.provider,
            )

    async def _embedding_source_fingerprint(self) -> dict[str, Any]:
        document_hash = hashlib.sha256()
        graph_hash = hashlib.sha256()
        document_ids = hashlib.sha256()
        graph_ids = hashlib.sha256()
        document_count = 0
        graph_count = 0
        async for batch in self.storage.iter_documents(batch_size=500):
            for item in batch:
                item_id = int(item["id"])
                text = str(item["text"])
                document_ids.update(f"{item_id},".encode())
                document_hash.update(f"{item_id}:{len(text)}:".encode())
                document_hash.update(text.encode("utf-8"))
                document_hash.update(b"\0")
                document_count += 1
        async for batch in self.storage.iter_graph_memories(batch_size=500):
            for item in batch:
                item_id = int(item["id"])
                content = str(item["content"])
                graph_ids.update(f"{item_id},".encode())
                graph_hash.update(f"{item_id}:{len(content)}:".encode())
                graph_hash.update(content.encode("utf-8"))
                graph_hash.update(b"\0")
                # One resumable graph vector represents one source memory.  The
                # raw number of derived graph entries is recorded separately.
                graph_count += 1
        return {
            "document_count": document_count,
            "graph_count": graph_count,
            "document_ids_sha256": document_ids.hexdigest(),
            "graph_ids_sha256": graph_ids.hexdigest(),
            "document_text_sha256": document_hash.hexdigest(),
            "graph_content_sha256": graph_hash.hexdigest(),
            "graph_entry_count": await self.storage.graph_entry_count(),
            "graph_vector_granularity": "memory",
        }

    async def _rebuild_resumable(
        self,
        *,
        batch_size: int,
        concurrency: int,
        embedding_batch_size: int | None,
        tasks_limit: int | None,
        max_retries: int,
        retry_base_delay: float,
        batch_delay: float,
        request_delay: float,
        max_failure_ratio: float,
        progress: Callable[[float, str], Any] | None,
        provider: EmbeddingProvider | None,
        library_id: str | None,
        provider_id: str | None,
        provider_revision: int | None,
        provider_config_sha256: str | None,
        provider_model: str | None,
        job_context: JobExecutionContext,
        checkpoint_dir: Path,
    ) -> dict[str, Any]:
        del max_failure_ratio
        candidate = provider or self.provider
        read_batch_size = max(1, int(batch_size))
        embed_batch_size = max(1, int(embedding_batch_size or batch_size))
        worker_limit = max(1, int(tasks_limit or concurrency))
        max_retries = max(1, int(max_retries))
        retry_base_delay = max(0.0, float(retry_base_delay))
        batch_delay = max(0.0, float(batch_delay))
        request_delay = max(0.0, float(request_delay))
        effective_library = library_id if library_id is not None else self.library_id
        effective_provider = provider_id if provider_id is not None else self.provider_id
        effective_revision = (
            provider_revision
            if provider_revision is not None
            else self.provider_revision
        )
        effective_config_hash = (
            provider_config_sha256
            if provider_config_sha256 is not None
            else self.provider_config_sha256
        )
        effective_model = (
            provider_model
            or getattr(getattr(candidate, "config", None), "model", "")
            or self.provider_model
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        segments_dir = checkpoint_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        activation_journal_path = checkpoint_dir / "activation-journal.json"

        async def save_state(state: dict[str, Any], message: str) -> None:
            state["saved_at"] = time.time()
            await run_blocking(write_ab_checkpoint, checkpoint_dir, state)
            completed = int(state.get("completed_documents", 0)) + int(
                state.get("completed_graph_entries", 0)
            )
            total = max(
                1,
                int(state.get("total_documents", 0))
                + int(state.get("total_graph_entries", 0)),
            )
            await job_context.checkpoint(
                {
                    "phase": state.get("phase", "indexing"),
                    "completed_documents": int(
                        state.get("completed_documents", 0)
                    ),
                    "completed_graph_entries": int(
                        state.get("completed_graph_entries", 0)
                    ),
                    "completed_graph_source_memories": int(
                        state.get(
                            "completed_graph_source_memories",
                            state.get("completed_graph_entries", 0),
                        )
                    ),
                    "total_documents": int(state.get("total_documents", 0)),
                    "total_graph_entries": int(
                        state.get("total_graph_entries", 0)
                    ),
                    "total_graph_source_memories": int(
                        state.get(
                            "total_graph_source_memories",
                            state.get("total_graph_entries", 0),
                        )
                    ),
                    "raw_graph_entry_count": int(
                        (state.get("source_fingerprint") or {}).get(
                            "graph_entry_count", 0
                        )
                    ),
                    "graph_vector_granularity": "memory",
                    "saved_at": state["saved_at"],
                },
                progress=completed / total,
                message=message,
            )

        fingerprint = await self._embedding_source_fingerprint()
        try:
            state = await run_blocking(read_ab_checkpoint, checkpoint_dir)
        except Exception as exc:
            raise JobInterrupted(
                "checkpoint_corrupt",
                "最近两份索引断点均已损坏，需停止任务以回滚",
                error=str(exc),
            ) from exc
        if state is not None:
            # `*_graph_entries` are legacy checkpoint keys.  For memory-level
            # generations they count aggregate source-memory vectors; explicit
            # aliases keep old checkpoints resumable without mislabeling UI data.
            state.setdefault(
                "total_graph_source_memories",
                int(state.get("total_graph_entries", 0)),
            )
            state.setdefault(
                "completed_graph_source_memories",
                int(state.get("completed_graph_entries", 0)),
            )
            expected_identity = {
                "library_id": effective_library,
                "provider_id": effective_provider,
                "provider_revision": effective_revision,
                "provider_config_sha256": effective_config_hash,
                "provider_model": effective_model,
            }
            actual_identity = {key: state.get(key) for key in expected_identity}
            if actual_identity != expected_identity:
                raise JobInterrupted(
                    "checkpoint_conflict",
                    "Provider revision 或记忆库身份已变化，拒绝不安全续跑",
                )
            if state.get("source_fingerprint") != fingerprint:
                raise JobInterrupted(
                    "source_changed",
                    "暂停期间源数据已变化，拒绝不安全续跑",
                )
        else:
            generation = time.strftime("gen-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
            state = {
                "version": 1,
                "phase": "indexing_documents",
                "generation": generation,
                "library_id": effective_library,
                "provider_id": effective_provider,
                "provider_revision": effective_revision,
                "provider_config_sha256": effective_config_hash,
                "provider_model": effective_model,
                "source_fingerprint": fingerprint,
                "total_documents": int(fingerprint["document_count"]),
                # Retain legacy names for checkpoint compatibility.  These are
                # source-memory vector counts, not raw graph-entry counts.
                "total_graph_entries": int(fingerprint["graph_count"]),
                "total_graph_source_memories": int(fingerprint["graph_count"]),
                "completed_documents": 0,
                "completed_graph_entries": 0,
                "completed_graph_source_memories": 0,
                "last_document_id": 0,
                "last_graph_id": 0,
                "segments": [],
                "norm_min": None,
                "norm_max": 0.0,
                "norm_sum": 0.0,
                "norm_count": 0,
            }
            await save_state(state, "索引安全断点已初始化")

        generation = str(state["generation"])
        final_dir = self.root / generation
        if state.get("phase") == "index_activated":
            manifest_path = final_dir / "manifest.json"
            if not manifest_path.exists():
                raise RuntimeError("activated checkpoint is missing its manifest")
            if activation_journal_path.exists():
                journal = json.loads(
                    activation_journal_path.read_text(encoding="utf-8")
                )
                if (
                    journal.get("candidate_generation") != generation
                    or journal.get("phase") != "activated"
                ):
                    raise JobInterrupted(
                        "checkpoint_corrupt",
                        "索引激活 journal 与断点不一致，需停止任务以回滚",
                    )
            return json.loads(manifest_path.read_text(encoding="utf-8"))

        try:
            dimension = await candidate.get_dimension()
            provider_status = await candidate.test_connection()
        except Exception as exc:
            raise JobInterrupted(
                "provider_unavailable",
                "Embedding 服务不可用，任务已回退到最近安全断点",
                error=str(exc),
            ) from exc
        if not provider_status.get("available"):
            raise JobInterrupted(
                "provider_unavailable",
                "Embedding 服务不可用，任务已回退到最近安全断点",
                error=str(provider_status.get("error") or "provider unavailable"),
            )
        if state.get("dimension") not in {None, dimension}:
            raise JobInterrupted(
                "checkpoint_conflict",
                "Embedding 维度与断点不一致，拒绝不安全续跑",
            )
        state["dimension"] = dimension
        capability = state.get("embedding_capability")
        if not isinstance(capability, dict) or not capability:
            try:
                capability = await _provider_capability_snapshot(
                    candidate,
                    provider_id=effective_provider,
                    provider_revision=effective_revision,
                    provider_config_sha256=effective_config_hash,
                    provider_status=provider_status,
                    force_probe=False,
                )
            except Exception as exc:
                raise JobInterrupted(
                    "provider_context_probe_failed",
                    "Embedding Provider context probe failed; task is paused at the last safe checkpoint.",
                    error=str(exc),
                ) from exc
            state["embedding_capability"] = capability
            await save_state(
                state,
                "Embedding Provider context has been probed and checkpointed.",
            )
        elif int(capability.get("detected_max_context_tokens") or 0) < 128:
            raise JobInterrupted(
                "provider_context_probe_failed",
                "Embedding Provider context checkpoint is invalid; stop the task to roll back.",
            )

        doc_index = self._empty_index(dimension)
        graph_index = self._empty_index(dimension)
        document_ids: list[int] = []
        graph_ids: list[int] = []
        for segment in state.get("segments", []):
            path = segments_dir / str(segment["file"])
            try:
                ids, vectors = await run_blocking(
                    _load_checkpoint_segment,
                    path,
                    str(segment.get("sha256") or ""),
                )
            except (FileNotFoundError, ValueError):
                raise JobInterrupted(
                    "checkpoint_corrupt",
                    f"索引断点分段缺失或损坏：{path.name}",
                )
            target = doc_index if segment["kind"] == "documents" else graph_index
            target.add_with_ids(vectors, ids)
            if segment["kind"] == "documents":
                document_ids.extend(map(int, ids))
            else:
                graph_ids.extend(map(int, ids))

        semaphore = asyncio.Semaphore(worker_limit)
        document_tracker = EmbeddingInputWarningTracker(
            "resumable_index_rebuild_documents",
            DEFAULT_DOCUMENT_EMBED_CHARS,
            effective_provider,
        )
        graph_tracker = EmbeddingInputWarningTracker(
            "resumable_index_rebuild_graph",
            DEFAULT_DOCUMENT_EMBED_CHARS,
            effective_provider,
        )

        async def embed_batch(
            texts: list[str],
            ids: list[int],
            tracker: EmbeddingInputWarningTracker,
        ) -> np.ndarray:
            chunks = [
                (texts[i : i + embed_batch_size], ids[i : i + embed_batch_size])
                for i in range(0, len(texts), embed_batch_size)
            ]

            async def run(chunk_texts: list[str], chunk_ids: list[int]):
                async with semaphore:
                    async def request_embeddings(batch_texts: list[str]):
                        last_exc: Exception | None = None
                        for attempt in range(max_retries):
                            try:
                                result = await candidate.get_embeddings(batch_texts)
                                if request_delay > 0:
                                    await asyncio.sleep(request_delay)
                                return result
                            except Exception as exc:
                                last_exc = exc
                                if attempt + 1 < max_retries:
                                    await asyncio.sleep(retry_base_delay * (2**attempt))
                        assert last_exc is not None
                        tracker.flush()
                        raise JobInterrupted(
                            "provider_unavailable",
                            "Embedding request failed; task is paused at the last safe checkpoint.",
                            error=str(last_exc),
                        ) from last_exc

                    matrix = await self._embed_texts_aggregated(
                        candidate,
                        chunk_texts,
                        dimension=dimension,
                        capability=capability,
                        tracker=tracker,
                        request_embeddings=request_embeddings,
                        request_batch_size=embed_batch_size,
                    )
                    minimum, maximum, total_norm = self._validate_matrix(
                        matrix,
                        expected_rows=len(chunk_ids),
                        dimension=dimension,
                    )
                    return matrix, minimum, maximum, total_norm

            results: list[tuple[np.ndarray, float, float, float]] = []
            for start in range(0, len(chunks), worker_limit):
                results.extend(
                    await asyncio.gather(
                        *[
                            run(chunk_texts, chunk_ids)
                            for chunk_texts, chunk_ids in chunks[
                                start : start + worker_limit
                            ]
                        ]
                    )
                )
            matrices = [item[0] for item in results]
            state["norm_min"] = min(
                [
                    float(state["norm_min"])
                    if state.get("norm_min") is not None
                    else math.inf
                ]
                + [item[1] for item in results]
            )
            state["norm_max"] = max(
                [float(state.get("norm_max", 0.0))]
                + [item[2] for item in results]
            )
            state["norm_sum"] = float(state.get("norm_sum", 0.0)) + sum(
                item[3] for item in results
            )
            state["norm_count"] = int(state.get("norm_count", 0)) + len(ids)
            return np.concatenate(matrices, axis=0) if matrices else np.empty((0, dimension), dtype=np.float32)

        async def persist_segment(kind: str, ids: list[int], matrix: np.ndarray) -> None:
            sequence = len(state.get("segments", [])) + 1
            filename = f"{sequence:06d}-{kind}.npz"
            final_path = segments_dir / filename
            temp_path = segments_dir / f".{filename}.tmp"
            segment_sha256 = await run_blocking(
                _persist_checkpoint_segment,
                temp_path,
                final_path,
                ids,
                matrix,
            )
            state.setdefault("segments", []).append(
                {
                    "kind": kind,
                    "file": filename,
                    "sha256": segment_sha256,
                    "count": len(ids),
                    "first_id": ids[0],
                    "last_id": ids[-1],
                }
            )

        total = max(1, int(state["total_documents"]) + int(state["total_graph_entries"]))
        await job_context.control_point()
        async for batch in self.storage.iter_documents(
            batch_size=read_batch_size, after_id=int(state.get("last_document_id", 0))
        ):
            ids = [int(item["id"]) for item in batch]
            texts = [str(item["text"] or "") for item in batch]
            matrix = await embed_batch(texts, ids, document_tracker)
            await persist_segment("documents", ids, matrix)
            doc_index.add_with_ids(matrix, np.asarray(ids, dtype=np.int64))
            document_ids.extend(ids)
            state["completed_documents"] = int(state["completed_documents"]) + len(ids)
            state["last_document_id"] = ids[-1]
            state["phase"] = "indexing_documents"
            done = int(state["completed_documents"]) + int(state["completed_graph_entries"])
            await save_state(state, f"已生成 {done}/{total} 条向量")
            if progress:
                await progress(done / total, f"已生成 {done}/{total} 条向量")
            if batch_delay > 0:
                await asyncio.sleep(batch_delay)

        state["phase"] = "indexing_graph"
        async for batch in self.storage.iter_graph_memories(
            batch_size=read_batch_size, after_id=int(state.get("last_graph_id", 0))
        ):
            ids = [int(item["id"]) for item in batch]
            texts = [str(item["content"] or "") for item in batch]
            matrix = await embed_batch(texts, ids, graph_tracker)
            await persist_segment("graph", ids, matrix)
            graph_index.add_with_ids(matrix, np.asarray(ids, dtype=np.int64))
            graph_ids.extend(ids)
            state["completed_graph_entries"] = int(state["completed_graph_entries"]) + len(ids)
            state["completed_graph_source_memories"] = int(
                state.get("completed_graph_source_memories", 0)
            ) + len(ids)
            state["last_graph_id"] = ids[-1]
            done = int(state["completed_documents"]) + int(state["completed_graph_entries"])
            await save_state(state, f"已生成 {done}/{total} 条向量")
            if progress:
                await progress(done / total, f"已生成 {done}/{total} 条向量")
            if batch_delay > 0:
                await asyncio.sleep(batch_delay)

        document_tracker.flush()
        graph_tracker.flush()
        if doc_index.ntotal != int(state["total_documents"]) or graph_index.ntotal != int(state["total_graph_entries"]):
            raise RuntimeError("zero-loss resumable rebuild vector count mismatch")
        final_fingerprint = await self._embedding_source_fingerprint()
        if final_fingerprint != state["source_fingerprint"]:
            raise JobInterrupted(
                "source_changed",
                "索引激活前源数据发生变化，候选 generation 未激活",
            )
        if set(map(int, faiss.vector_to_array(doc_index.id_map))) != set(document_ids):
            raise RuntimeError("document vector ID set mismatch")
        if set(map(int, faiss.vector_to_array(graph_index.id_map))) != set(graph_ids):
            raise RuntimeError("graph vector ID set mismatch")

        async with self.storage.connect() as db:
            document_samples = [
                (int(row["id"]), str(row["text"]))
                for row in await (
                    await db.execute(
                        """SELECT id,text FROM documents
                        WHERE COALESCE(json_extract(metadata,'$.status'),'active')='active'
                        ORDER BY id LIMIT 3"""
                    )
                ).fetchall()
            ]
            graph_samples = [
                (int(item["id"]), str(item["content"]))
                for item in await self.storage.graph_memories_for_ids(
                    graph_ids[:3]
                )
            ]
        try:
            await self._validate_sample_recall(
                candidate,
                doc_index,
                document_samples,
                "文档索引",
                dimension=dimension,
                capability=capability,
            )
            await self._validate_sample_recall(
                candidate,
                graph_index,
                graph_samples,
                "图记忆索引",
                dimension=dimension,
                capability=capability,
            )
        except Exception as exc:
            raise JobInterrupted(
                "provider_unavailable",
                "Embedding 服务验证失败，任务保留在提交前安全断点",
                error=str(exc),
            ) from exc

        manifest = GenerationManifest(
            generation=generation,
            created_at=time.time(),
            library_id=effective_library,
            provider_id=effective_provider,
            provider_revision=effective_revision,
            provider_config_sha256=effective_config_hash,
            provider_type=type(candidate).__name__,
            configured_model=effective_model,
            resolved_model=str(provider_status.get("resolved_model") or ""),
            dimension=dimension,
            metric="IndexIDMap(IndexFlatL2)",
            document_count=int(state["total_documents"]),
            graph_entry_count=int(fingerprint.get("graph_entry_count") or 0),
            document_ids_sha256=hashlib.sha256(
                ",".join(map(str, document_ids)).encode()
            ).hexdigest(),
            graph_ids_sha256=hashlib.sha256(
                ",".join(map(str, graph_ids)).encode()
            ).hexdigest(),
            vector_norm_min=round(
                0.0 if not int(state["norm_count"]) else float(state["norm_min"]), 8
            ),
            vector_norm_max=round(float(state["norm_max"]), 8),
            vector_norm_mean=round(
                0.0
                if not int(state["norm_count"])
                else float(state["norm_sum"]) / int(state["norm_count"]),
                8,
            ),
            embedding_capability=capability,
            chunked_document_count=document_tracker.chunked_items,
            chunked_graph_entry_count=graph_tracker.chunked_items,
            total_embedding_chunks=(
                document_tracker.total_chunks + graph_tracker.total_chunks
            ),
            max_chunks_per_item=max(
                1,
                document_tracker.max_chunks_per_item,
                graph_tracker.max_chunks_per_item,
            ),
            graph_vector_granularity="memory",
            graph_source_memory_count=int(state["total_graph_source_memories"]),
            graph_vector_count=int(state["total_graph_source_memories"]),
            graph_vector_content_sha256=str(
                fingerprint.get("graph_content_sha256") or ""
            ),
            provider_functional_sha256=provider_functional_sha256(candidate),
        )
        state["phase"] = "ready_to_commit"
        await save_state(state, "候选索引已完成，等待原子激活")
        temp_dir = self.root / f".{generation}.tmp"
        if not final_dir.exists():
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir.mkdir(parents=True, exist_ok=False)
            await run_blocking(
                _persist_generation_files,
                doc_index,
                graph_index,
                asdict(manifest),
                temp_dir,
                final_dir,
            )
        existing_journal = None
        if activation_journal_path.exists():
            try:
                existing_journal = json.loads(
                    activation_journal_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise JobInterrupted(
                    "checkpoint_corrupt",
                    "索引激活 journal 已损坏，需停止任务以回滚",
                    error=str(exc),
                ) from exc
        previous_generation = (
            str(existing_journal.get("previous_generation") or "")
            if existing_journal
            and existing_journal.get("candidate_generation") == generation
            else (
                self.current_file.read_text(encoding="utf-8").strip()
                if self.current_file.exists()
                else ""
            )
        )
        await run_blocking(
            _atomic_json,
            activation_journal_path,
            {
                "phase": "prepared",
                "previous_generation": previous_generation,
                "candidate_generation": generation,
                "saved_at": time.time(),
            },
        )
        await job_context.control_point()
        async with self._swap_lock:
            pointer = self.current_file.with_suffix(".tmp")
            pointer.write_text(generation, encoding="utf-8")
            os.replace(pointer, self.current_file)
            self._snapshot = IndexSnapshot(doc_index, graph_index, asdict(manifest), candidate)
            self.library_id = manifest.library_id
            self.provider_id = manifest.provider_id
            self.provider_revision = manifest.provider_revision
            self.provider_config_sha256 = manifest.provider_config_sha256
            self.provider_model = manifest.configured_model
        await run_blocking(
            _atomic_json,
            activation_journal_path,
            {
                "phase": "activated",
                "previous_generation": previous_generation,
                "candidate_generation": generation,
                "saved_at": time.time(),
            },
        )
        state["phase"] = "index_activated"
        await save_state(state, "索引 generation 已激活，正在提交 Provider 绑定")
        if progress:
            await progress(1.0, "索引 generation 与 Provider 已原子切换")
        return asdict(manifest)

    async def search_documents(
        self, query: str, k: int, fetch_k: int | None = None
    ) -> list[tuple[int, float]]:
        snapshot = self._snapshot
        if snapshot is None or snapshot.document_index.ntotal == 0:
            return []
        capability = (snapshot.manifest or {}).get("embedding_capability")
        if not isinstance(capability, dict) or int(
            capability.get("detected_max_context_tokens") or 0
        ) < 128:
            capability = await _provider_capability_snapshot(
                snapshot.provider,
                provider_id=self.provider_id,
                provider_revision=self.provider_revision,
                provider_config_sha256=self.provider_config_sha256,
                force_probe=False,
            )
        tracker = EmbeddingInputWarningTracker(
            "document_recall_query",
            DEFAULT_QUERY_EMBED_CHARS,
            self.provider_id,
        )
        try:
            vector = await self._embed_texts_aggregated(
                snapshot.provider,
                [query],
                dimension=int(snapshot.document_index.d),
                capability=capability,
                tracker=tracker,
                request_embeddings=snapshot.provider.get_embeddings,
                request_batch_size=1,
            )
        finally:
            tracker.flush()
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
        capability = (snapshot.manifest or {}).get("embedding_capability")
        if not isinstance(capability, dict) or int(
            capability.get("detected_max_context_tokens") or 0
        ) < 128:
            capability = await _provider_capability_snapshot(
                snapshot.provider,
                provider_id=self.provider_id,
                provider_revision=self.provider_revision,
                provider_config_sha256=self.provider_config_sha256,
                force_probe=False,
            )
        tracker = EmbeddingInputWarningTracker(
            "graph_recall_query",
            DEFAULT_QUERY_EMBED_CHARS,
            self.provider_id,
        )
        try:
            vector = await self._embed_texts_aggregated(
                snapshot.provider,
                [query],
                dimension=int(snapshot.graph_index.d),
                capability=capability,
                tracker=tracker,
                request_embeddings=snapshot.provider.get_embeddings,
                request_batch_size=1,
            )
        finally:
            tracker.flush()
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
