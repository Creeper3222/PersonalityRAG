from __future__ import annotations

import asyncio
import hashlib
import math
import mimetypes
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from ...io_utils import run_blocking
from ...providers import EmbeddingProvider, RerankProvider
from .document_parsers import parse_document_bytes
from .embedding import embed_one_context_safe, embed_texts_context_safe
from .images import build_preview, normalize_image_bytes, write_content_addressed_image
from .indexes import TextMediaIndex
from .retrieval import (
    MEDIA_CONFIDENCE_ALGORITHM,
    MEDIA_FREQUENCY_ALGORITHM,
    UNBOUND_MEDIA_SPECIFICITY_ALGORITHM,
    MINIMUM_RERANK_REORDER_PROBABILITY,
    RERANK_FUSION_ALGORITHM,
    TEXT_RELEVANCE_ALGORITHM,
    aggregate_grounding,
    calibrated_semantic,
    clamp01,
    fuse_embedding_rerank,
    media_frequency_signals,
    unbound_media_confidence_factor,
    noisy_or,
    normalize_retrieval_config,
    rerank_calibration_settings_fingerprint,
    score_media_confidence,
    text_embedding_relevance,
)
from .storage import (
    DEFAULT_UNIFORM_MEDIA_STRENGTH,
    TextMediaStorage,
    media_description_set_sha256,
)
from .text import (
    CHUNKER_ID,
    LexiconReferenceVisualIntentDetector,
    VisualIntentDetector,
    chunk_text,
    media_description_list,
    media_collection_intent,
    media_format_groups,
    media_subject_tokens,
    media_tokens,
    normalize_media_description,
    normalize_media_text,
    normalize_text,
    normalize_visual_intent_policy,
    visual_intent,
    visual_intent_policy_fingerprint,
    weighted_token_coverage,
)
from .visual_intent_policy import (
    read_effective_visual_intent_policy,
    write_visual_intent_policy_override,
)


MEDIA_CALIBRATION_METHOD = "media_semantic_rank_v1"
RERANK_MEDIA_CALIBRATION_METHOD = "media_semantic_rank_rerank_v1"
UNIFORM_MEDIA_CALIBRATION_METHOD = "uniform_v1"
MINIMUM_SEMANTIC_STRENGTH = 0.15
MINIMUM_CALIBRATION_SCALE = 0.03
RERANK_CACHE_TTL_SECONDS = 60.0
RERANK_CACHE_MAX_ITEMS = 4096
QUERY_EMBEDDING_CACHE_TTL_SECONDS = 60.0
QUERY_EMBEDDING_CACHE_MAX_ITEMS = 1024


class RerankStageError(RuntimeError):
    """A query-stage rerank error that requires whole-request baseline fallback."""


def _rerank_pair_cache_key(fingerprint: str, query: str, document: str) -> str:
    normalized_query = normalize_text(query).casefold()
    document_hash = hashlib.sha256(
        normalize_text(document).encode("utf-8")
    ).hexdigest()
    return f"{fingerprint}:{normalized_query}:{document_hash}"


def calibrate_media_strengths(
    description_vector: list[float] | np.ndarray,
    chunk_vectors: list[list[float]] | np.ndarray,
) -> list[dict[str, Any]]:
    query = np.asarray(description_vector, dtype=np.float32)
    vectors = np.asarray(chunk_vectors, dtype=np.float32)
    if query.ndim != 1 or vectors.ndim != 2 or not len(vectors):
        raise ValueError("media calibration vectors are invalid")
    if vectors.shape[1] != query.shape[0]:
        raise ValueError("media description and chunk dimensions do not match")
    if not np.isfinite(query).all() or not np.isfinite(vectors).all():
        raise ValueError("media calibration vectors contain non-finite values")
    query_norm = float(np.linalg.norm(query))
    vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if query_norm <= 0 or np.any(vector_norms <= 0):
        raise ValueError("media calibration vectors have zero norm")
    similarities = (vectors / vector_norms) @ (query / query_norm)
    order = sorted(
        range(len(similarities)),
        key=lambda index: (-float(similarities[index]), index),
    )
    q25, q75 = np.quantile(similarities, [0.25, 0.75])
    scale = max(
        MINIMUM_CALIBRATION_SCALE,
        1.5 * max(0.0, float(q75) - float(q25)),
    )
    best = float(similarities[order[0]])
    denominator = max(1, len(order) - 1)
    results: list[dict[str, Any] | None] = [None] * len(order)
    for rank, index in enumerate(order, start=1):
        similarity = float(similarities[index])
        affinity = math.exp(-(best - similarity) / scale)
        rank_percentile = (len(order) - rank) / denominator if len(order) > 1 else 1.0
        strength = MINIMUM_SEMANTIC_STRENGTH + (
            1.0 - MINIMUM_SEMANTIC_STRENGTH
        ) * affinity * (0.8 + 0.2 * rank_percentile)
        results[index] = {
            "semantic_strength": max(
                MINIMUM_SEMANTIC_STRENGTH, min(1.0, float(strength))
            ),
            "calibration_similarity": similarity,
            "calibration_rank": rank,
        }
    return [dict(item) for item in results if item is not None]


def uniform_media_strengths(
    count: int,
    strength: float = DEFAULT_UNIFORM_MEDIA_STRENGTH,
) -> list[dict[str, Any]]:
    normalized_strength = max(0.0, min(1.0, float(strength)))
    return [
        {
            "semantic_strength": normalized_strength,
            "calibration_similarity": None,
            "calibration_rank": None,
        }
        for _ in range(max(0, int(count)))
    ]


def aggregate_media_description_calibrations(
    descriptions: list[dict[str, Any]],
    rows_by_description: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Collapse independent description calibrations without count bonuses."""

    if not descriptions or len(descriptions) != len(rows_by_description):
        raise ValueError("media descriptions and calibration rows do not match")
    row_count = len(rows_by_description[0])
    if any(len(rows) != row_count for rows in rows_by_description):
        raise ValueError("media description calibration row counts do not match")
    result: list[dict[str, Any]] = []
    for chunk_index in range(row_count):
        candidates = [
            (description, dict(rows[chunk_index]))
            for description, rows in zip(
                descriptions, rows_by_description, strict=True
            )
        ]
        embedding_description, embedding_winner = max(
            candidates,
            key=lambda item: float(item[1].get("semantic_strength") or 0.0),
        )
        rerank_description, rerank_winner = max(
            candidates,
            key=lambda item: float(
                item[1].get("rerank_semantic_strength")
                if item[1].get("rerank_semantic_strength") is not None
                else item[1].get("semantic_strength")
                or 0.0
            ),
        )
        rerank_values = [
            item[1].get("rerank_semantic_strength") for item in candidates
        ]
        details = {
            "version": 2,
            "description_count": len(descriptions),
            "rerank_applied": any(
                bool(
                    dict(row.get("calibration_details") or {}).get(
                        "rerank_applied"
                    )
                )
                for _description, row in candidates
            ),
            "rerank_settings_fingerprint": str(
                dict(
                    rerank_winner.get("calibration_details") or {}
                ).get("rerank_settings_fingerprint")
                or ""
            ),
            "embedding_winner_description_order": int(
                embedding_description.get("sort_order") or 0
            ),
            "embedding_winner_description": str(
                embedding_description["media_description"]
            ),
            "rerank_winner_description_order": int(
                rerank_description.get("sort_order") or 0
            ),
            "rerank_winner_description": str(
                rerank_description["media_description"]
            ),
            "descriptions": [
                {
                    "sort_order": int(description.get("sort_order") or 0),
                    "media_description": str(description["media_description"]),
                    "semantic_strength": round(
                        float(row.get("semantic_strength") or 0.0), 6
                    ),
                    "rerank_semantic_strength": (
                        round(float(row["rerank_semantic_strength"]), 6)
                        if row.get("rerank_semantic_strength") is not None
                        else None
                    ),
                    "calibration_similarity": (
                        round(float(row["calibration_similarity"]), 6)
                        if row.get("calibration_similarity") is not None
                        else None
                    ),
                    "calibration_rank": row.get("calibration_rank"),
                    "rerank": dict(row.get("calibration_details") or {}),
                }
                for description, row in candidates
            ],
        }
        result.append(
            {
                "semantic_strength": float(
                    embedding_winner.get("semantic_strength") or 0.0
                ),
                "rerank_semantic_strength": (
                    float(rerank_winner["rerank_semantic_strength"])
                    if any(value is not None for value in rerank_values)
                    else None
                ),
                "calibration_similarity": embedding_winner.get(
                    "calibration_similarity"
                ),
                "calibration_rank": embedding_winner.get("calibration_rank"),
                "calibration_details": details,
            }
        )
    return result


class TextMediaService:
    def __init__(
        self,
        root: Path,
        storage: TextMediaStorage,
        indexes: TextMediaIndex,
        provider: EmbeddingProvider | None,
        reranker: RerankProvider | None = None,
        rerank_provider_info: dict[str, Any] | None = None,
        visual_intent_policy: object = None,
    ):
        self.root = Path(root)
        self.storage = storage
        self.indexes = indexes
        self.provider = provider
        self.reranker = reranker
        self.rerank_provider_info = dict(rerank_provider_info or {})
        self._rerank_cache: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._query_embedding_cache: OrderedDict[
            str, tuple[float, list[float]]
        ] = OrderedDict()
        self._visual_intent_detector_cache: dict[
            str, VisualIntentDetector
        ] = {}
        self.visual_intent_policy = normalize_visual_intent_policy(
            visual_intent_policy
            if visual_intent_policy is not None
            else read_effective_visual_intent_policy(self.root),
            require_complete=True,
        )
        self.visual_intent_policy_is_default = not (
            self.root / "visual_intent_policy.csv"
        ).exists()
        self._write_lock = asyncio.Lock()

    async def close(self) -> None:
        if self.provider is not None:
            await self.provider.close()
        if self.reranker is not None:
            await self.reranker.close()
        await self.storage.close()

    @property
    def rerank_provider_fingerprint(self) -> str:
        return str(self.rerank_provider_info.get("fingerprint") or "")

    async def set_rerank_provider(
        self,
        reranker: RerankProvider | None,
        provider_info: dict[str, Any] | None = None,
    ) -> None:
        previous = self.reranker
        self.reranker = reranker
        self.rerank_provider_info = dict(provider_info or {})
        self._rerank_cache.clear()
        if previous is not None and previous is not reranker:
            await previous.close()

    async def update_visual_intent_policy(
        self,
        policy: object,
    ) -> dict[str, Any]:
        """Atomically replace query-only visual intent policy.

        Compilation happens before the metadata write.  No Provider, index,
        vector, or media-calibration state participates in this short task.
        """

        normalized_policy = normalize_visual_intent_policy(
            policy,
            require_complete=True,
        )
        detector = LexiconReferenceVisualIntentDetector(normalized_policy)
        before = dict(self.indexes.status())
        async with self._write_lock:
            meta = await self.storage.metadata()
            settings = normalize_retrieval_config(
                meta.get("retrieval_config_json")
            )
            settings.pop("visual_intent_policy", None)
            custom_written = await run_blocking(
                write_visual_intent_policy_override,
                self.root,
                normalized_policy,
            )
            await self.storage.update_metadata(
                {"retrieval_config_json": settings}
            )
            self.visual_intent_policy = normalized_policy
            self.visual_intent_policy_is_default = not custom_written
            self._visual_intent_detector_cache = {
                detector.policy_fingerprint: detector
            }
        after = dict(self.indexes.status())
        return {
            "visual_intent_policy": normalized_policy,
            "visual_intent_policy_fingerprint": detector.policy_fingerprint,
            "visual_intent_detector_version": detector.version,
            "visual_intent_policy_is_default": not custom_written,
            "provider_calls": 0,
            "index_rebuilt": False,
            "generation": after.get("generation"),
            "media_generation": after.get("media_generation"),
            "generation_unchanged": before.get("generation")
            == after.get("generation"),
            "media_generation_unchanged": before.get("media_generation")
            == after.get("media_generation"),
        }

    def _rerank_binding_available(self, meta: dict[str, Any]) -> bool:
        return bool(
            self.reranker is not None
            and str(meta.get("rerank_provider_id") or "")
            == str(self.rerank_provider_info.get("id") or "")
            and int(meta.get("rerank_provider_revision") or 0)
            == int(self.rerank_provider_info.get("revision") or 0)
            and str(meta.get("rerank_provider_fingerprint") or "")
            == self.rerank_provider_fingerprint
        )

    def _visual_intent_detector(
        self, settings: dict[str, Any]
    ) -> VisualIntentDetector:
        policy = self.visual_intent_policy
        fingerprint = visual_intent_policy_fingerprint(policy)
        detector = self._visual_intent_detector_cache.get(fingerprint)
        if detector is None:
            detector = LexiconReferenceVisualIntentDetector(policy)
            self._visual_intent_detector_cache = {fingerprint: detector}
        return detector

    async def _embed_query_inputs(
        self,
        values: list[str],
        *,
        provider_identity: str,
    ) -> tuple[list[list[float]], dict[str, Any]]:
        """Embed normalized query inputs with a short deterministic cache.

        Some remote Embedding runtimes exhibit tiny floating-point drift for
        identical consecutive requests.  Reusing the same vector within one
        interactive comparison window keeps retrieval modes, Top-K probes and
        whole-request Rerank fallback bit-for-bit comparable without changing
        stored vectors or cross-request semantics.
        """

        if not values:
            return [], {
                "input_count": 0,
                "request_count": 0,
                "cache_hits": 0,
                "provider_input_count": 0,
            }
        now = time.monotonic()
        results: list[list[float] | None] = [None] * len(values)
        misses: list[tuple[int, str, str]] = []
        cache_hits = 0
        for index, value in enumerate(values):
            identity = normalize_media_text(value)
            key = f"{provider_identity}:{identity}"
            cached = self._query_embedding_cache.get(key)
            if (
                cached is not None
                and now - cached[0] <= QUERY_EMBEDDING_CACHE_TTL_SECONDS
            ):
                results[index] = list(cached[1])
                cache_hits += 1
                self._query_embedding_cache.move_to_end(key)
            else:
                if cached is not None:
                    self._query_embedding_cache.pop(key, None)
                misses.append((index, value, key))

        provider_meta: dict[str, Any] = {}
        if misses:
            embedded, provider_meta = await embed_texts_context_safe(
                self.provider,
                [item[1] for item in misses],
            )
            for (index, _value, key), vector in zip(
                misses, embedded, strict=True
            ):
                copied = [float(component) for component in vector]
                results[index] = copied
                self._query_embedding_cache[key] = (now, copied)
                self._query_embedding_cache.move_to_end(key)
            while (
                len(self._query_embedding_cache)
                > QUERY_EMBEDDING_CACHE_MAX_ITEMS
            ):
                self._query_embedding_cache.popitem(last=False)
        if any(item is None for item in results):
            raise ValueError("Embedding query cache returned incomplete vectors")
        return [list(item) for item in results if item is not None], {
            **provider_meta,
            "input_count": len(values),
            "request_count": int(provider_meta.get("request_count") or 0),
            "cache_hits": cache_hits,
            "provider_input_count": len(misses),
        }

    async def _rerank_documents(
        self,
        query: str,
        documents: list[str],
        *,
        settings: dict[str, Any],
        scope: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if self.reranker is None or not self.rerank_provider_fingerprint:
            raise RerankStageError("rerank provider is unavailable")
        if not documents:
            return [], {
                "scope": scope,
                "candidate_count": 0,
                "cache_hits": 0,
                "provider_candidates": 0,
                "elapsed_ms": 0.0,
            }
        started = time.perf_counter()
        now = time.monotonic()
        raw_scores: list[float | None] = [None] * len(documents)
        cache_hits = 0
        misses: list[tuple[int, str, str]] = []
        for index, document in enumerate(documents):
            key = _rerank_pair_cache_key(
                self.rerank_provider_fingerprint, query, document
            )
            cached = self._rerank_cache.get(key)
            if cached is not None and now - cached[0] <= RERANK_CACHE_TTL_SECONDS:
                raw_scores[index] = float(cached[1])
                cache_hits += 1
                self._rerank_cache.move_to_end(key)
            else:
                if cached is not None:
                    self._rerank_cache.pop(key, None)
                misses.append((index, document, key))

        provider_candidates = 0
        batch_size = max(
            1,
            min(
                int(settings["rerank_candidate_limit"]),
                int(getattr(self.reranker.config, "batch_size", 32) or 32),
            ),
        )
        try:
            for start in range(0, len(misses), batch_size):
                batch = misses[start : start + batch_size]
                batch_documents = [item[1] for item in batch]
                returned = await self.reranker.rerank(
                    query, batch_documents, top_n=len(batch_documents)
                )
                provider_candidates += len(batch_documents)
                indexes = [int(item.index) for item in returned]
                if (
                    len(returned) != len(batch_documents)
                    or len(set(indexes)) != len(indexes)
                    or any(index < 0 or index >= len(batch_documents) for index in indexes)
                    or set(indexes) != set(range(len(batch_documents)))
                ):
                    raise ValueError("rerank provider returned an invalid candidate set")
                for item in returned:
                    original_index, _document, key = batch[int(item.index)]
                    raw = float(item.relevance_score)
                    if not math.isfinite(raw):
                        raise ValueError("rerank provider returned a non-finite score")
                    raw_scores[original_index] = raw
                    self._rerank_cache[key] = (now, raw)
                    self._rerank_cache.move_to_end(key)
            while len(self._rerank_cache) > RERANK_CACHE_MAX_ITEMS:
                self._rerank_cache.popitem(last=False)
        except RerankStageError:
            raise
        except Exception as exc:
            raise RerankStageError(f"{scope}: {exc}") from exc

        if any(score is None for score in raw_scores):
            raise RerankStageError(f"{scope}: rerank result is incomplete")
        ranked_indexes = sorted(
            range(len(documents)),
            key=lambda index: (-float(raw_scores[index]), index),
        )
        rank_by_index = {
            index: rank for rank, index in enumerate(ranked_indexes, start=1)
        }
        details: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_scores):
            details.append(
                {
                    "raw_score": float(raw),
                    "rank": int(rank_by_index[index]),
                    "candidate_count": len(documents),
                }
            )
        return details, {
            "scope": scope,
            "candidate_count": len(documents),
            "cache_hits": cache_hits,
            "provider_candidates": provider_candidates,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }

    async def _rerank_calibration_rows(
        self,
        *,
        description: str,
        chunks: list[dict[str, Any]],
        baseline: list[dict[str, Any]],
        settings: dict[str, Any],
        scope: str,
    ) -> list[dict[str, Any]]:
        if len(chunks) != len(baseline):
            raise ValueError("media calibration row count does not match chunks")
        rows = [dict(item) for item in baseline]
        settings_fingerprint = rerank_calibration_settings_fingerprint(settings)
        ordered_indexes = sorted(
            range(len(rows)),
            key=lambda index: (
                int(rows[index].get("calibration_rank") or 10**9),
                int(chunks[index].get("chunk_id") or index),
            ),
        )
        limit = min(int(settings["rerank_candidate_limit"]), len(ordered_indexes))
        candidate_indexes = ordered_indexes[:limit]
        details, _meta = await self._rerank_documents(
            description,
            [str(chunks[index]["text"]) for index in candidate_indexes],
            settings=settings,
            scope=scope,
        )
        for index, row in enumerate(rows):
            row["rerank_semantic_strength"] = float(row["semantic_strength"])
            row["calibration_details"] = {
                "version": 1,
                "rerank_applied": False,
                "rerank_settings_fingerprint": settings_fingerprint,
                "embedding_semantic_strength": round(
                    float(row["semantic_strength"]), 6
                ),
            }
        for local_index, chunk_index in enumerate(candidate_indexes):
            rerank = details[local_index]
            fused = fuse_embedding_rerank(
                float(rows[chunk_index]["semantic_strength"]),
                float(rerank["raw_score"]),
                rank=int(rerank["rank"]),
                candidate_count=int(rerank["candidate_count"]),
                fusion_weight=float(settings["rerank_fusion_weight"]),
                rank_bonus_weight=float(settings["rerank_rank_bonus_weight"]),
                rank_reliability_exponent=float(
                    settings["rerank_rank_reliability_exponent"]
                ),
            )
            rows[chunk_index]["rerank_semantic_strength"] = float(
                fused["fused_relevance"]
            )
            rows[chunk_index]["calibration_details"] = {
                "version": 1,
                "rerank_applied": True,
                "rerank_settings_fingerprint": settings_fingerprint,
                "embedding_semantic_strength": round(
                    float(rows[chunk_index]["semantic_strength"]), 6
                ),
                "rerank_raw_score": round(float(rerank["raw_score"]), 6),
                "rerank_rank": int(rerank["rank"]),
                **{key: round(float(value), 6) for key, value in fused.items()},
            }
        return rows

    async def ingest_document(self, *, filename: str, title: str, data: bytes) -> dict[str, Any]:
        suffix = Path(filename).suffix.lower()
        parsed = await run_blocking(parse_document_bytes, filename, data)
        content = parsed.content
        chunks = chunk_text(content, format_hint=parsed.format_hint)
        if not chunks:
            raise ValueError("文档没有可索引文本")
        if self.provider is None:
            raise ValueError("知识库尚未绑定可用的 Embedding Provider")
        vectors, _ = await embed_texts_context_safe(
            self.provider,
            [str(item["text"]) for item in chunks],
        )
        meta = await self.storage.metadata()
        digest = hashlib.sha256(data).hexdigest()
        relative = Path("assets") / "documents" / "sha256" / digest[:2] / f"{digest}{suffix}"
        target = self.root / relative
        async with self._write_lock:
            if not target.exists():
                await run_blocking(target.parent.mkdir, parents=True, exist_ok=True)
                temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
                try:
                    await run_blocking(temporary.write_bytes, data)
                    await run_blocking(os.replace, temporary, target)
                finally:
                    await run_blocking(temporary.unlink, missing_ok=True)
            asset = await self.storage.register_asset(
                kind="document",
                sha256=digest,
                storage_key=relative.as_posix(),
                mime_type=mimetypes.guess_type(filename)[0] or "text/plain",
                size_bytes=len(data),
                original_name=Path(filename).name,
            )
            result = await self.storage.install_document(
                title=title.strip() or Path(filename).stem,
                source_asset_id=asset["id"],
                content=content,
                chunks=chunks,
                vectors=vectors,
                provider_id=str(meta["provider_id"]),
                provider_revision=int(meta["provider_revision"]),
                provider_fingerprint=str(meta["provider_fingerprint"]),
                parser_id=f"{parsed.parser_id}+{CHUNKER_ID}",
            )
            result["index"] = await self.indexes.rebuild(self.storage)
            return result

    async def ingest_batch(
        self,
        *,
        batch_id: str,
        documents: list[dict[str, Any]],
        images: list[dict[str, Any]],
        chunk_target: int,
        chunk_overlap: int,
        embedding_batch_size: int,
        concurrency: int,
        max_retries: int,
        media_semantic_calibration_enabled: bool = False,
        progress: Callable[[float, str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        if self.provider is None:
            raise ValueError("知识库尚未绑定可用的 Embedding Provider")
        if not documents and not images:
            raise ValueError("批次必须至少包含一个文档或一张图片")
        if len(documents) > 10 or len(images) > 10:
            raise ValueError("每批次最多包含 10 个文档和 10 张图片")
        chunk_target = max(200, min(4000, int(chunk_target)))
        chunk_overlap = int(chunk_overlap)
        if chunk_overlap < 0 or chunk_overlap > chunk_target // 2:
            raise ValueError("分块重叠必须介于 0 和分块大小的一半之间")
        embedding_batch_size = max(1, min(128, int(embedding_batch_size)))
        concurrency = max(1, min(8, int(concurrency)))
        max_retries = max(1, min(8, int(max_retries)))

        async def report(value: float, message: str) -> None:
            if progress is not None:
                await progress(value, message)

        await report(0.04, "正在解析文档")
        prepared_documents: list[dict[str, Any]] = []
        all_texts: list[str] = []
        chunk_ranges: list[tuple[int, int]] = []
        for document in documents:
            filename = Path(str(document["filename"])).name
            suffix = Path(filename).suffix.lower()
            data = await run_blocking(Path(document["path"]).read_bytes)
            parsed = await run_blocking(parse_document_bytes, filename, data)
            content = parsed.content
            chunks = chunk_text(
                content,
                target=chunk_target,
                overlap=chunk_overlap,
                format_hint=parsed.format_hint,
            )
            if not chunks:
                raise ValueError(f"文档没有可索引文本：{filename}")
            digest = hashlib.sha256(data).hexdigest()
            relative = (
                Path("assets")
                / "documents"
                / "sha256"
                / digest[:2]
                / f"{digest}{suffix}"
            )
            start = len(all_texts)
            all_texts.extend(str(item["text"]) for item in chunks)
            chunk_ranges.append((start, len(all_texts)))
            prepared_documents.append(
                {
                    "filename": filename,
                    "title": str(document.get("title") or Path(filename).stem),
                    "data": data,
                    "content": content,
                    "content_sha256": hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                    "sha256": digest,
                    "storage_key": relative.as_posix(),
                    "mime_type": mimetypes.guess_type(filename)[0] or "text/plain",
                    "size_bytes": len(data),
                    "original_name": filename,
                    "parser_id": (
                        f"{parsed.parser_id}+{CHUNKER_ID};max={chunk_target};"
                        f"overlap={chunk_overlap}"
                    ),
                    "chunks": chunks,
                }
            )

        await report(0.12, "正在规范化图片")
        prepared_images: list[dict[str, Any]] = []
        prepared_images_by_hash: dict[str, dict[str, Any]] = {}
        for image in images:
            filename = Path(str(image["filename"])).name
            data = await run_blocking(Path(image["path"]).read_bytes)
            normalized = await run_blocking(normalize_image_bytes, data)
            indexes = sorted(
                {int(value) for value in image.get("document_indexes") or []}
            )
            if indexes and (indexes[0] < 0 or indexes[-1] >= len(documents)):
                raise ValueError(f"图片关联的文档范围无效：{filename}")
            relative = (
                Path("assets")
                / "images"
                / "sha256"
                / normalized.sha256[:2]
                / f"{normalized.sha256}.webp"
            )
            existing = prepared_images_by_hash.get(normalized.sha256)
            if existing is not None:
                raise ValueError("duplicate canonical image in one ingest batch")
            raw_descriptions = image.get("media_descriptions")
            if isinstance(raw_descriptions, list):
                raw_descriptions = [
                    (
                        str(item.get("media_description") or "")
                        if isinstance(item, dict)
                        else item
                    )
                    for item in raw_descriptions
                ]
            elif raw_descriptions is not None:
                raise ValueError("media_descriptions must be a list")
            legacy_description = str(image.get("media_description") or "")
            media_descriptions = media_description_list(
                raw_descriptions
                if raw_descriptions
                else ([legacy_description] if legacy_description.strip() else []),
                fallback=Path(filename).stem,
            )
            if (
                raw_descriptions
                and legacy_description.strip()
                and normalize_media_description(legacy_description)
                != normalize_media_description(media_descriptions[0])
            ):
                raise ValueError(
                    "media_description must match the first media_descriptions item"
                )
            description_source = (
                "user"
                if raw_descriptions or legacy_description.strip()
                else "filename"
            )
            prepared = {
                    "filename": filename,
                    "normalized": normalized,
                    "sha256": normalized.sha256,
                    "storage_key": relative.as_posix(),
                    "mime_type": normalized.mime_type,
                    "size_bytes": normalized.size_bytes,
                    "width": normalized.width,
                    "height": normalized.height,
                    "original_name": filename,
                    "document_indexes": indexes,
                    "media_description": media_descriptions[0],
                    "media_descriptions": [
                        {
                            "media_description": description,
                            "sort_order": order,
                            "description_source": description_source,
                        }
                        for order, description in enumerate(media_descriptions)
                    ],
                    "description_source": description_source,
                }
            prepared_images.append(prepared)
            prepared_images_by_hash[normalized.sha256] = prepared

        batches = [
            (start, all_texts[start : start + embedding_batch_size])
            for start in range(0, len(all_texts), embedding_batch_size)
        ]
        vectors: list[list[float] | None] = [None] * len(all_texts)
        semaphore = asyncio.Semaphore(concurrency)
        completed = 0
        completed_lock = asyncio.Lock()

        async def embed_one(start: int, texts: list[str]) -> None:
            nonlocal completed
            async with semaphore:
                last_error: Exception | None = None
                for attempt in range(max_retries):
                    try:
                        batch_vectors, _ = await embed_texts_context_safe(
                            self.provider,
                            texts,
                            request_batch_size=embedding_batch_size,
                        )
                        if len(batch_vectors) != len(texts):
                            raise ValueError("Embedding 返回数量与输入不一致")
                        vectors[start : start + len(texts)] = batch_vectors
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt + 1 >= max_retries:
                            raise RuntimeError(
                                f"Embedding 批次失败：{last_error}"
                            ) from last_error
                        await asyncio.sleep(min(8.0, float(2**attempt)))
                async with completed_lock:
                    completed += 1
                    await report(
                        0.18 + 0.52 * completed / max(1, len(batches)),
                        f"正在向量化分块 {completed}/{len(batches)}",
                    )

        await asyncio.gather(*(embed_one(start, texts) for start, texts in batches))
        if any(item is None for item in vectors):
            raise RuntimeError("Embedding 批次结果不完整")
        for document, (start, end) in zip(
            prepared_documents, chunk_ranges, strict=True
        ):
            document["vectors"] = vectors[start:end]

        description_vectors: dict[str, list[float]] = {}
        if prepared_images:
            await report(0.71, "Embedding media descriptions")
            descriptions = list(
                dict.fromkeys(
                    str(item["media_description"])
                    for image in prepared_images
                    for item in image["media_descriptions"]
                )
            )
            last_error: Exception | None = None
            for attempt in range(max_retries):
                try:
                    embedded, _ = await embed_texts_context_safe(
                        self.provider,
                        descriptions,
                        request_batch_size=embedding_batch_size,
                    )
                    if len(embedded) != len(descriptions):
                        raise ValueError(
                            "media description embedding count does not match input"
                        )
                    description_vectors = dict(
                        zip(descriptions, embedded, strict=True)
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 >= max_retries:
                        raise RuntimeError(
                            f"media description embedding failed: {last_error}"
                        ) from last_error
                    await asyncio.sleep(min(8.0, float(2**attempt)))

        meta = await self.storage.metadata()
        settings = normalize_retrieval_config(meta.get("retrieval_config_json"))
        calibration_rerank_enabled = self._rerank_binding_available(meta)
        uniform_media_strength = float(meta["uniform_media_strength"])
        for image in prepared_images:
            for description in image["media_descriptions"]:
                description["vector"] = description_vectors[
                    str(description["media_description"])
                ]
            calibrations: dict[int, dict[str, Any]] = {}
            for document_index in image["document_indexes"]:
                document_vectors = list(
                    prepared_documents[document_index]["vectors"]
                )
                if media_semantic_calibration_enabled:
                    calibrated_by_description: list[list[dict[str, Any]]] = []
                    calibration_method = MEDIA_CALIBRATION_METHOD
                    rerank_fingerprint = ""
                    for description in image["media_descriptions"]:
                        baseline = calibrate_media_strengths(
                            description_vectors[
                                str(description["media_description"])
                            ],
                            document_vectors,
                        )
                        calibrated = baseline
                        if calibration_rerank_enabled:
                            calibrated = await self._rerank_calibration_rows(
                                description=str(
                                    description["media_description"]
                                ),
                                chunks=list(
                                    prepared_documents[document_index]["chunks"]
                                ),
                                baseline=baseline,
                                settings=settings,
                                scope=(
                                    "secondary_calibration:"
                                    f"{image['sha256']}:{document_index}:"
                                    f"{description['sort_order']}"
                                ),
                            )
                        calibrated_by_description.append(calibrated)
                    calibrated = aggregate_media_description_calibrations(
                        list(image["media_descriptions"]),
                        calibrated_by_description,
                    )
                    if calibration_rerank_enabled:
                        calibration_method = RERANK_MEDIA_CALIBRATION_METHOD
                        rerank_fingerprint = self.rerank_provider_fingerprint
                    calibrations[document_index] = {
                        "semantic_mode": "calibrated",
                        "media_description": image["media_description"],
                        "media_description_vector": description_vectors[
                            image["media_description"]
                        ],
                        "calibration_method": calibration_method,
                        "rerank_provider_fingerprint": rerank_fingerprint,
                        "chunks": calibrated,
                    }
                else:
                    calibrations[document_index] = {
                        "semantic_mode": "uniform",
                        "media_description": image["media_description"],
                        "media_description_vector": description_vectors[
                            image["media_description"]
                        ],
                        "calibration_method": UNIFORM_MEDIA_CALIBRATION_METHOD,
                        "rerank_provider_fingerprint": "",
                        "chunks": uniform_media_strengths(
                            len(document_vectors), uniform_media_strength
                        ),
                    }
            image["document_calibrations"] = calibrations
            image["media_description_vector"] = description_vectors[
                image["media_description"]
            ]

        parameters = {
            "ingest_mode": (
                "mixed" if prepared_documents and prepared_images
                else "text_only" if prepared_documents
                else "media_only"
            ),
            "chunk_target": chunk_target,
            "chunk_overlap": chunk_overlap,
            "embedding_batch_size": embedding_batch_size,
            "concurrency": concurrency,
            "max_retries": max_retries,
            "media_semantic_calibration_enabled": bool(
                media_semantic_calibration_enabled
            ),
            "uniform_media_strength": uniform_media_strength,
        }
        created_files: set[Path] = set()
        installed: dict[str, Any] | None = None
        async with self._write_lock:
            try:
                await report(0.73, "正在保存批次资源")
                for document in prepared_documents:
                    target = self.root.joinpath(
                        *Path(document["storage_key"]).parts
                    )
                    if not target.exists():
                        await run_blocking(target.parent.mkdir, parents=True, exist_ok=True)
                        temporary = target.with_name(
                            f".{target.name}.{os.getpid()}.tmp"
                        )
                        try:
                            await run_blocking(temporary.write_bytes, document["data"])
                            await run_blocking(os.replace, temporary, target)
                            created_files.add(target)
                        finally:
                            await run_blocking(temporary.unlink, missing_ok=True)
                for image in prepared_images:
                    target = self.root.joinpath(*Path(image["storage_key"]).parts)
                    existed = target.exists()
                    await run_blocking(
                        write_content_addressed_image,
                        self.root,
                        image["normalized"],
                    )
                    if not existed:
                        created_files.add(target)
                    preview = (
                        self.root
                        / "derived"
                        / "previews"
                        / f"{image['sha256']}.webp"
                    )
                    if not preview.exists():
                        await run_blocking(build_preview, target, preview)
                        created_files.add(preview)

                await report(0.8, "正在原子安装文档与媒体关系")
                installed = await self.storage.install_ingest_batch(
                    batch_id=batch_id,
                    parameters=parameters,
                    documents=prepared_documents,
                    images=prepared_images,
                    provider_id=str(meta["provider_id"]),
                    provider_revision=int(meta["provider_revision"]),
                    provider_fingerprint=str(meta["provider_fingerprint"]),
                )
                await report(0.9, "正在切换检索索引")
                index = await self.indexes.rebuild(self.storage)
            except BaseException:
                try:
                    if installed is not None:
                        removable = await self.storage.rollback_ingest_batch(
                            batch_id=batch_id,
                            generation_id=installed.get("generation_id"),
                            previous_generation_id=installed.get(
                                "previous_generation_id"
                            ),
                            media_metadata_before=installed.get(
                                "media_metadata_before"
                            ),
                        )
                        for storage_key in removable:
                            created_files.add(
                                self.root.joinpath(*Path(storage_key).parts)
                            )
                        await self.indexes.rebuild(self.storage)
                finally:
                    for target in sorted(created_files, reverse=True):
                        await run_blocking(target.unlink, missing_ok=True)
                raise
        await report(1.0, "批次上传与索引构建完成")
        public_installed = {
            key: value
            for key, value in installed.items()
            if key != "media_metadata_before"
        }
        return {
            **public_installed,
            "index": index,
            "document_count": len(prepared_documents),
            "image_count": len(prepared_images),
            "chunk_count": len(all_texts),
            "parameters": parameters,
        }

    async def recalibrate_document_media(
        self,
        *,
        document_id: str,
        asset_id: str,
        enabled: bool,
        media_description: str,
        progress: Callable[[float, str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        if self.provider is None:
            raise ValueError("knowledge library has no available Embedding Provider")
        requested_description = normalize_text(media_description)
        descriptions = await self.storage.asset_media_descriptions(asset_id)
        if not descriptions:
            raise ValueError("media asset has no descriptions")
        primary_description = str(descriptions[0]["media_description"])
        if (
            requested_description
            and requested_description != primary_description
        ):
            raise ValueError(
                "media_description must match the current primary description; "
                "use asset media-description management to edit descriptions"
            )
        context = await self.storage.document_media_calibration_context(
            document_id, asset_id
        )
        chunks = list(context["chunks"])
        if progress is not None:
            await progress(0.25, "Preparing media calibration")
        meta = await self.storage.metadata()
        settings = normalize_retrieval_config(meta.get("retrieval_config_json"))
        rerank_fingerprint = ""
        if enabled:
            vectors_by_description: list[list[float] | np.ndarray] = []
            for description in descriptions:
                vector = description.get("media_description_vector")
                if (
                    vector is None
                    or str(description.get("provider_fingerprint") or "")
                    != str(meta["provider_fingerprint"])
                ):
                    vector = await embed_one_context_safe(
                        self.provider,
                        str(description["media_description"]),
                    )
                vectors_by_description.append(vector)
            calibrated_by_description: list[list[dict[str, Any]]] = []
            method = MEDIA_CALIBRATION_METHOD
            for description, vector in zip(
                descriptions, vectors_by_description, strict=True
            ):
                baseline = calibrate_media_strengths(
                    vector, [item["vector"] for item in chunks]
                )
                calibrated_rows = baseline
                if self._rerank_binding_available(meta):
                    calibrated_rows = await self._rerank_calibration_rows(
                        description=str(description["media_description"]),
                        chunks=chunks,
                        baseline=baseline,
                        settings=settings,
                        scope=(
                            f"secondary_calibration:{asset_id}:{document_id}:"
                            f"{description['sort_order']}"
                        ),
                    )
                calibrated_by_description.append(calibrated_rows)
            calibrated = aggregate_media_description_calibrations(
                descriptions, calibrated_by_description
            )
            if self._rerank_binding_available(meta):
                method = RERANK_MEDIA_CALIBRATION_METHOD
                rerank_fingerprint = self.rerank_provider_fingerprint
            mode = "calibrated"
            vector = vectors_by_description[0]
            description = primary_description
        else:
            vector = None
            calibrated = uniform_media_strengths(
                len(chunks), float(meta["uniform_media_strength"])
            )
            method = UNIFORM_MEDIA_CALIBRATION_METHOD
            mode = "uniform"
            description = primary_description
        rows = [
            {"chunk_id": chunk["chunk_id"], **item}
            for chunk, item in zip(chunks, calibrated, strict=True)
        ]
        async with self._write_lock:
            result = await self.storage.replace_document_media_calibration(
                document_id=document_id,
                asset_id=asset_id,
                semantic_mode=mode,
                media_description=description,
                media_description_vector=vector,
                calibration_method=method,
                provider_fingerprint=str(meta["provider_fingerprint"]),
                rerank_provider_fingerprint=rerank_fingerprint,
                chunks=rows,
            )
            result["index"] = await self.indexes.rebuild(self.storage)
        if progress is not None:
            await progress(1.0, "Media calibration completed")
        return result

    async def update_asset_media_descriptions(
        self,
        *,
        asset_id: str,
        media_descriptions: list[str],
        progress: Callable[[float, str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Atomically replace an asset description set and all its calibrations."""

        if self.provider is None:
            raise ValueError(
                "knowledge library has no available Embedding Provider"
            )
        descriptions = media_description_list(media_descriptions)

        async def report(value: float, message: str) -> None:
            if progress is not None:
                await progress(value, message)

        await report(0.08, "Preparing media descriptions")
        meta = await self.storage.metadata()
        settings = normalize_retrieval_config(
            meta.get("retrieval_config_json")
        )
        current = await self.storage.asset_media_descriptions(asset_id)
        if not current:
            asset = await self.storage.get_asset(asset_id)
            if not asset or str(asset.get("kind") or "") != "image":
                raise KeyError(asset_id)
        current_by_identity = {
            normalize_media_description(str(item["media_description"])): item
            for item in current
        }
        description_items: list[dict[str, Any]] = []
        missing: list[tuple[int, str]] = []
        for order, description in enumerate(descriptions):
            previous = current_by_identity.get(
                normalize_media_description(description)
            )
            vector = (
                previous.get("media_description_vector")
                if previous
                and str(previous.get("provider_fingerprint") or "")
                == str(meta["provider_fingerprint"])
                else None
            )
            item = {
                "media_description": description,
                "sort_order": order,
                "description_source": (
                    str(previous.get("description_source") or "user")
                    if previous
                    else "user"
                ),
                "vector": vector,
            }
            description_items.append(item)
            if vector is None:
                missing.append((order, description))
        if missing:
            await report(0.18, "Embedding new media descriptions")
            vectors, _ = await embed_texts_context_safe(
                self.provider, [item[1] for item in missing]
            )
            if len(vectors) != len(missing):
                raise ValueError(
                    "Embedding Provider returned incomplete media descriptions"
                )
            for (order, _description), vector in zip(
                missing, vectors, strict=True
            ):
                description_items[order]["vector"] = vector

        relations = [
            item
            for item in await self.storage.list_document_media_calibrations()
            if str(item["asset_id"]) == asset_id
        ]
        calibrated_relations = [
            item
            for item in relations
            if str(item.get("semantic_mode") or "") == "calibrated"
        ]
        rerank_required = bool(meta.get("rerank_provider_id"))
        if (
            calibrated_relations
            and rerank_required
            and not self._rerank_binding_available(meta)
        ):
            raise ValueError(
                "bound Rerank Provider is unavailable for strict recalibration"
            )

        calibrations: list[dict[str, Any]] = []
        for relation_index, relation in enumerate(relations, start=1):
            context = await self.storage.document_media_calibration_context(
                str(relation["document_id"]), asset_id
            )
            chunks = list(context["chunks"])
            semantic_mode = str(
                relation.get("semantic_mode") or "uniform"
            )
            rerank_fingerprint = ""
            if semantic_mode == "calibrated":
                rows_by_description: list[list[dict[str, Any]]] = []
                for item in description_items:
                    baseline = calibrate_media_strengths(
                        item["vector"],
                        [chunk["vector"] for chunk in chunks],
                    )
                    if rerank_required:
                        baseline = await self._rerank_calibration_rows(
                            description=str(item["media_description"]),
                            chunks=chunks,
                            baseline=baseline,
                            settings=settings,
                            scope=(
                                "description_update_calibration:"
                                f"{asset_id}:{relation['document_id']}:"
                                f"{item['sort_order']}"
                            ),
                        )
                    rows_by_description.append(baseline)
                strengths = aggregate_media_description_calibrations(
                    description_items, rows_by_description
                )
                method = (
                    RERANK_MEDIA_CALIBRATION_METHOD
                    if rerank_required
                    else MEDIA_CALIBRATION_METHOD
                )
                if rerank_required:
                    rerank_fingerprint = self.rerank_provider_fingerprint
            else:
                strengths = uniform_media_strengths(
                    len(chunks), float(meta["uniform_media_strength"])
                )
                method = UNIFORM_MEDIA_CALIBRATION_METHOD
            calibrations.append(
                {
                    "document_id": str(relation["document_id"]),
                    "semantic_mode": semantic_mode,
                    "calibration_method": method,
                    "rerank_provider_fingerprint": rerank_fingerprint,
                    "chunks": [
                        {"chunk_id": chunk["chunk_id"], **strength}
                        for chunk, strength in zip(
                            chunks, strengths, strict=True
                        )
                    ],
                }
            )
            await report(
                0.30
                + 0.45 * relation_index / max(1, len(relations)),
                "Recalibrating media relations",
            )

        async with self._write_lock:
            result = await self.storage.replace_asset_media_descriptions(
                asset_id=asset_id,
                descriptions=description_items,
                provider_id=str(meta["provider_id"]),
                provider_revision=int(meta["provider_revision"]),
                provider_fingerprint=str(meta["provider_fingerprint"]),
                calibrations=calibrations,
            )
            await report(0.82, "Rebuilding media indexes")
            result["index"] = await self.indexes.rebuild(self.storage)
        result["relation_count"] = len(relations)
        result["recalibrated_relation_count"] = len(calibrated_relations)
        await report(1.0, "Media descriptions updated")
        return result

    async def rebuild_embeddings(
        self,
        *,
        provider_id: str,
        provider_revision: int,
        provider_fingerprint: str,
        progress: Callable[[float, str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Re-embed the complete corpus and every calibrated media relation."""
        if self.provider is None:
            raise ValueError("target Embedding Provider is unavailable")

        async def report(value: float, message: str) -> None:
            if progress is not None:
                await progress(value, message)

        await report(0.05, "正在读取完整文本语料")
        chunks = await self.storage.active_chunk_records()
        texts = [str(item["text"]) for item in chunks]
        vectors: list[list[float]] = []
        if texts:
            await report(0.12, "正在使用目标 Provider 重建全部文本向量")
            vectors, _ = await embed_texts_context_safe(
                self.provider,
                texts,
            )
            if len(vectors) != len(chunks):
                raise ValueError("Embedding Provider returned an incomplete corpus")

        meta = await self.storage.metadata()
        settings = normalize_retrieval_config(meta.get("retrieval_config_json"))
        media_records = await self.storage.media_metadata_records()
        relations = [
            item
            for item in await self.storage.list_document_media_calibrations()
            if str(item.get("semantic_mode") or "") == "calibrated"
        ]
        asset_descriptions = [
            str(item.get("media_description") or "").strip()
            for item in media_records
        ]
        if any(not description for description in asset_descriptions):
            raise ValueError("every media asset requires a media description")
        descriptions = list(dict.fromkeys(asset_descriptions))
        await report(0.48, "正在重建媒体简述向量")
        description_vectors, _ = (
            await embed_texts_context_safe(self.provider, descriptions)
            if descriptions
            else ([], {})
        )
        if len(description_vectors) != len(descriptions):
            raise ValueError("Embedding Provider returned incomplete media vectors")
        description_map = dict(zip(descriptions, description_vectors, strict=True))
        vector_by_chunk = {
            int(chunk["chunk_id"]): vector
            for chunk, vector in zip(chunks, vectors, strict=True)
        }

        rerank_required = bool(meta.get("rerank_provider_id"))
        if rerank_required and not self._rerank_binding_available(meta):
            raise ValueError(
                "bound Rerank Provider is unavailable for strict recalibration"
            )
        calibrations: list[dict[str, Any]] = []
        media_records_by_asset: dict[str, list[dict[str, Any]]] = {}
        for item in media_records:
            media_records_by_asset.setdefault(
                str(item["asset_id"]), []
            ).append(item)
        for items in media_records_by_asset.values():
            items.sort(
                key=lambda item: (
                    int(item.get("sort_order") or 0),
                    int(item.get("id") or 0),
                )
            )
        for relation_index, relation in enumerate(relations, start=1):
            document_id = str(relation["document_id"])
            asset_id = str(relation["asset_id"])
            asset_description_rows = media_records_by_asset.get(asset_id) or []
            if not asset_description_rows:
                raise ValueError("calibrated media relation has no descriptions")
            context = await self.storage.document_media_calibration_context(
                document_id, asset_id
            )
            relation_chunks = [
                {
                    **item,
                    "vector": vector_by_chunk[int(item["chunk_id"])],
                }
                for item in context["chunks"]
            ]
            method = MEDIA_CALIBRATION_METHOD
            rerank_fingerprint = ""
            calibrated_by_description: list[list[dict[str, Any]]] = []
            for description_row in asset_description_rows:
                description = str(description_row["media_description"])
                baseline = calibrate_media_strengths(
                    description_map[description],
                    [item["vector"] for item in relation_chunks],
                )
                if rerank_required:
                    baseline = await self._rerank_calibration_rows(
                        description=description,
                        chunks=relation_chunks,
                        baseline=baseline,
                        settings=settings,
                        scope=(
                            "provider_rebuild_calibration:"
                            f"{asset_id}:{document_id}:"
                            f"{description_row['sort_order']}"
                        ),
                    )
                calibrated_by_description.append(baseline)
            baseline = aggregate_media_description_calibrations(
                asset_description_rows, calibrated_by_description
            )
            if rerank_required:
                method = RERANK_MEDIA_CALIBRATION_METHOD
                rerank_fingerprint = self.rerank_provider_fingerprint
            primary_description = str(
                asset_description_rows[0]["media_description"]
            )
            calibrations.append(
                {
                    "document_id": document_id,
                    "asset_id": asset_id,
                    "media_description": primary_description,
                    "media_description_vector": description_map[
                        primary_description
                    ],
                    "description_set_sha256": media_description_set_sha256(
                        [
                            str(item["media_description"])
                            for item in asset_description_rows
                        ]
                    ),
                    "calibration_method": method,
                    "rerank_provider_fingerprint": rerank_fingerprint,
                    "chunks": [
                        {"chunk_id": item["chunk_id"], **strength}
                        for item, strength in zip(
                            relation_chunks, baseline, strict=True
                        )
                    ],
                }
            )
            await report(
                0.55 + 0.2 * relation_index / max(1, len(relations)),
                "正在重新校准媒体关联强度",
            )

        media_vectors = [
            {
                "media_description_id": int(item["id"]),
                "asset_id": str(item["asset_id"]),
                "vector": description_map[str(item["media_description"]).strip()],
            }
            for item in media_records
        ]
        generation_id = (
            f"gen-{int(time.time())}-{os.urandom(4).hex()}" if chunks else None
        )
        await report(0.78, "正在原子写入新的向量代次")
        installed = await self.storage.replace_all_embeddings(
            generation_id=generation_id,
            provider_id=provider_id,
            provider_revision=provider_revision,
            provider_fingerprint=provider_fingerprint,
            chunks=chunks,
            vectors=vectors,
            calibrations=calibrations,
            media_vectors=media_vectors,
            dimensions=int(
                getattr(getattr(self.provider, "config", None), "dimensions", 0)
                or 0
            ),
        )
        await report(0.88, "正在构建新的文本媒体索引")
        index = await self.indexes.rebuild(self.storage)
        validation = await self.storage.validate()
        await report(1.0, "Provider 切换、索引重建与媒体校准已完成")
        return {
            **installed,
            "index": index,
            "validation": validation,
            "provider_id": provider_id,
            "provider_revision": int(provider_revision),
            "provider_fingerprint": provider_fingerprint,
        }

    async def upload_image(
        self,
        *,
        filename: str,
        data: bytes,
        media_descriptions: list[str] | None = None,
    ) -> dict[str, Any]:
        if self.provider is None:
            raise ValueError("知识库尚未绑定可用的 Embedding Provider")
        normalized = await run_blocking(normalize_image_bytes, data)
        requested_descriptions = media_description_list(
            media_descriptions, fallback=Path(filename).stem
        )
        meta = await self.storage.metadata()
        async with self._write_lock:
            relative = await run_blocking(write_content_addressed_image, self.root, normalized)
            asset = await self.storage.register_asset(
                kind="image",
                sha256=normalized.sha256,
                storage_key=relative.as_posix(),
                mime_type=normalized.mime_type,
                size_bytes=normalized.size_bytes,
                original_name=Path(filename).name,
                width=normalized.width,
                height=normalized.height,
            )
            preview = self.root / "derived" / "previews" / f"{normalized.sha256}.webp"
            if not preview.exists():
                await run_blocking(build_preview, self.root / relative, preview)
            existing = await self.storage.asset_media_descriptions(
                str(asset["id"])
            )
            combined: list[str] = []
            seen: set[str] = set()
            for description in [
                *(str(item["media_description"]) for item in existing),
                *requested_descriptions,
            ]:
                identity = normalize_media_description(description)
                if identity not in seen:
                    combined.append(description)
                    seen.add(identity)
            combined = media_description_list(combined)
            existing_by_identity = {
                normalize_media_description(str(item["media_description"])): item
                for item in existing
            }
            description_items: list[dict[str, Any]] = []
            for description in combined:
                previous = existing_by_identity.get(
                    normalize_media_description(description)
                )
                vector = (
                    previous.get("media_description_vector")
                    if previous
                    and str(previous.get("provider_fingerprint") or "")
                    == str(meta["provider_fingerprint"])
                    else None
                )
                if vector is None:
                    vector = await embed_one_context_safe(
                        self.provider, description
                    )
                description_items.append(
                    {
                        "media_description": description,
                        "description_source": (
                            str(previous.get("description_source") or "user")
                            if previous
                            else (
                                "filename"
                                if description == Path(filename).stem
                                else "user"
                            )
                        ),
                        "vector": vector,
                    }
                )
            await self.storage.replace_asset_media_descriptions(
                asset_id=str(asset["id"]),
                descriptions=description_items,
                provider_id=str(meta["provider_id"]),
                provider_revision=int(meta["provider_revision"]),
                provider_fingerprint=str(meta["provider_fingerprint"]),
            )
            asset["media_descriptions"] = combined
            asset["media_description"] = combined[0]
            asset["index"] = await self.indexes.rebuild(self.storage)
            return asset

    async def create_entry(self, *, title: str, body: str) -> dict[str, Any]:
        name = title.strip() or "Untitled"
        return await self.ingest_document(
            filename=f"manual-{hashlib.sha256(body.encode('utf-8')).hexdigest()[:12]}.txt",
            title=name,
            data=body.encode("utf-8"),
        )

    async def update_entry(
        self, *, entry_id: str, title: str, body: str
    ) -> dict[str, Any]:
        if self.provider is None:
            raise ValueError("知识库尚未绑定可用的 Embedding Provider")
        normalized = normalize_text(body)
        chunks = chunk_text(normalized)
        if not chunks:
            raise ValueError("条目没有可索引文本")
        vectors, _ = await embed_texts_context_safe(
            self.provider,
            [str(item["text"]) for item in chunks],
        )
        async with self._write_lock:
            result = await self.storage.replace_entry(
                entry_id=entry_id,
                title=title.strip() or "Untitled",
                body=normalized,
                chunks=chunks,
                vectors=vectors,
            )
            result["index"] = await self.indexes.rebuild(self.storage)
            return result

    async def delete_entry(self, entry_id: str) -> dict[str, Any]:
        async with self._write_lock:
            result = await self.storage.delete_entry(entry_id)
            asset = result.get("removed_source_asset")
            if asset:
                await run_blocking(
                    (self.root / str(asset["storage_key"])).unlink, missing_ok=True
                )
            result["index"] = await self.indexes.rebuild(self.storage)
            return result

    async def delete_document(self, document_id: str) -> dict[str, Any]:
        async with self._write_lock:
            result = await self.storage.delete_document(document_id)
            asset = result.get("removed_source_asset")
            if asset:
                await run_blocking(
                    (self.root / str(asset["storage_key"])).unlink, missing_ok=True
                )
            result["index"] = await self.indexes.rebuild(self.storage)
            return result

    async def delete_documents(self, document_ids: list[str]) -> dict[str, Any]:
        async with self._write_lock:
            result = await self.storage.delete_documents(document_ids)
            for asset in result.get("removed_source_assets", []):
                await run_blocking(
                    (self.root / str(asset["storage_key"])).unlink,
                    missing_ok=True,
                )
            result["index"] = await self.indexes.rebuild(self.storage)
            return result

    async def delete_image(self, asset_id: str) -> dict[str, Any]:
        async with self._write_lock:
            asset = await self.storage.delete_image_asset(asset_id)
            await run_blocking(
                (self.root / str(asset["storage_key"])).unlink, missing_ok=True
            )
            await run_blocking(
                (self.root / "derived" / "previews" / f"{asset['sha256']}.webp").unlink,
                missing_ok=True,
            )
            return {
                "asset_id": asset_id,
                "deleted": True,
                "index": await self.indexes.rebuild(self.storage),
            }

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        media_output_confidence_threshold: float = 0.6,
        media_relevance_pivot: float | None = None,
        media_score_threshold: float | None = None,
        max_media_outputs: int = 5,
        rerank: bool | None = None,
        retrieval_mode: str = "standard",
    ) -> dict[str, Any]:
        if retrieval_mode not in {"standard", "text_only", "media_only"}:
            raise ValueError("unsupported retrieval mode")
        meta = await self.storage.metadata()
        provider_available = self._rerank_binding_available(meta)
        effective = provider_available and rerank is not False
        parameters = {
            "top_k": top_k,
            "media_output_confidence_threshold": media_output_confidence_threshold,
            "media_relevance_pivot": media_relevance_pivot,
            "media_score_threshold": media_score_threshold,
            "max_media_outputs": max_media_outputs,
            "rerank_requested": rerank,
            "retrieval_mode": retrieval_mode,
        }
        if effective:
            try:
                return await self._search_once(
                    query, use_rerank=True, **parameters
                )
            except RerankStageError as exc:
                # A failed query-stage Rerank is atomic: discard all pair
                # results accumulated by the failed attempt before replaying
                # the complete Embedding baseline.
                self._rerank_cache.clear()
                baseline = await self._search_once(
                    query, use_rerank=False, **parameters
                )
                baseline["rerank"].update(
                    {
                        "effective_enabled": False,
                        "applied": False,
                        "failed": True,
                        "fallback": True,
                        "discarded_partial_results": True,
                        "fallback_reason": str(exc),
                    }
                )
                return baseline
        result = await self._search_once(query, use_rerank=False, **parameters)
        if rerank is True and not provider_available:
            result["rerank"]["unavailable_reason"] = (
                "no_matching_rerank_provider_binding"
            )
        return result

    async def _search_once(
        self,
        query: str,
        *,
        top_k: int,
        media_output_confidence_threshold: float,
        media_relevance_pivot: float | None,
        media_score_threshold: float | None,
        max_media_outputs: int,
        rerank_requested: bool | None,
        retrieval_mode: str,
        use_rerank: bool,
    ) -> dict[str, Any]:
        query = normalize_text(query)
        if not query:
            raise ValueError("检索文本不能为空")
        meta = await self.storage.metadata()
        if meta["status"] == "provider_binding_required" or self.provider is None:
            raise ValueError("知识库需要绑定匹配的 Embedding Provider 后才能检索")
        top_k = max(1, min(50, int(top_k)))
        media_output_confidence_threshold = max(
            0.0, min(1.0, float(media_output_confidence_threshold))
        )
        max_media_outputs = max(0, min(20, int(max_media_outputs)))
        settings = normalize_retrieval_config(meta.get("retrieval_config_json"))
        if (
            media_relevance_pivot is not None
            and media_score_threshold is not None
            and abs(
                float(media_relevance_pivot) - float(media_score_threshold)
            )
            > 1e-12
        ):
            raise ValueError(
                "media_relevance_pivot and deprecated "
                "media_score_threshold must match when both are provided"
            )
        requested_media_relevance_pivot = (
            clamp01(float(media_relevance_pivot))
            if media_relevance_pivot is not None
            else clamp01(float(media_score_threshold))
            if media_score_threshold is not None
            else None
        )
        effective_media_relevance_pivot = (
            requested_media_relevance_pivot
            if requested_media_relevance_pivot is not None
            else float(settings["media_relevance_pivot_fallback"])
        )
        media_relevance_pivot_source = (
            "request"
            if requested_media_relevance_pivot is not None
            else "library_fallback"
        )
        media_relevance_pivot_request_field = (
            "media_relevance_pivot"
            if media_relevance_pivot is not None
            else "media_score_threshold"
            if media_score_threshold is not None
            else None
        )
        # Local aliases keep the bound-evidence implementation readable while
        # response/API diagnostics expose the canonical universal pivot name.
        effective_media_score_threshold = effective_media_relevance_pivot
        candidate_limit = int(settings["media_candidate_limit"])
        rerank_candidate_limit = int(settings["rerank_candidate_limit"])
        rerank_scope_meta: list[dict[str, Any]] = []
        rrf_k = int(settings["rrf_k"])
        text_channel_enabled = retrieval_mode in {"standard", "text_only"}
        media_channel_enabled = retrieval_mode in {"standard", "media_only"}
        if media_channel_enabled:
            detector = self._visual_intent_detector(settings)
            intent = visual_intent(
                query,
                gate_enabled=bool(settings["visual_intent_gate_enabled"]),
                detector=detector,
            )
            intent["original_query"] = query
        else:
            # Pure text retrieval does not instantiate or execute the media
            # intent parser at all.
            intent = {
                "detected": False,
                "gate_enabled": bool(settings["visual_intent_gate_enabled"]),
                "intent_kind": "not_executed",
                "matched_terms": [],
                "matched_categories": {},
                "blocked_terms": [],
                "subject_anchor_required": False,
                "subject_anchor_terms": [],
                "protected_terms": [],
                "original_query": query,
                "media_query": "",
                "reference_span": "",
                "generation_span": "",
                "ignored_output_terms": [],
                "projection_applied": False,
                "policy_fingerprint": "",
                "detector_version": "",
                "skipped_reason": "text_only_mode",
            }
        explicit_visual_intent = bool(intent["detected"])
        media_query = (
            str(intent.get("media_query") or query)
            if media_channel_enabled
            else ""
        )
        collection_intent = (
            media_collection_intent(media_query) if media_channel_enabled else False
        )
        if (
            retrieval_mode == "media_only"
            and not bool(intent["detected"])
            and not bool(intent.get("blocked_terms"))
        ):
            intent = {
                **intent,
                # Selecting media-only mode supplies the visual intent. Explicit
                # administrative/meta negatives still remain blocked, and
                # visual subject anchors continue to protect against returning
                # a related document's image for a different named object.
                "detected": True,
                "reasons": [
                    *list(intent.get("reasons") or []),
                    "explicit_media_only_mode",
                ],
                "collection_intent": collection_intent,
            }
        query_tokens = (
            media_tokens(
                media_query,
                protected_terms=list(intent.get("protected_terms") or []),
            )
            if media_channel_enabled
            else []
        )
        query_media_format_groups = (
            media_format_groups(media_query) if media_channel_enabled else set()
        )
        query_subject_tokens = media_subject_tokens(query_tokens)

        embedding_inputs: list[str] = []
        embedding_input_index: dict[str, int] = {}

        def register_query(value: str) -> int:
            identity = normalize_media_text(value)
            if identity not in embedding_input_index:
                embedding_input_index[identity] = len(embedding_inputs)
                embedding_inputs.append(value)
            return embedding_input_index[identity]

        text_vector_index = register_query(query) if text_channel_enabled else None
        media_vector_index = (
            register_query(media_query) if media_channel_enabled else None
        )
        embedding_provider_identity = ":".join(
            (
                str(meta.get("provider_id") or ""),
                str(meta.get("provider_revision") or 0),
                str(meta.get("provider_fingerprint") or ""),
            )
        )
        embedded_queries, query_embedding_meta = await self._embed_query_inputs(
            embedding_inputs,
            provider_identity=embedding_provider_identity,
        )

        def normalized_vector(index: int | None) -> np.ndarray | None:
            if index is None:
                return None
            candidate = np.asarray(embedded_queries[index], dtype=np.float32)
            candidate_norm = float(np.linalg.norm(candidate))
            if (
                candidate.ndim != 1
                or not candidate.size
                or not np.isfinite(candidate).all()
                or candidate_norm <= 0
            ):
                raise ValueError("Embedding 查询向量无效")
            return candidate / candidate_norm

        normalized_media_query_vector = normalized_vector(media_vector_index)
        text_vector = (
            embedded_queries[text_vector_index]
            if text_vector_index is not None
            else None
        )
        media_vector = (
            embedded_queries[media_vector_index]
            if media_vector_index is not None
            else None
        )
        retrieval_limit = max(
            50,
            candidate_limit,
            rerank_candidate_limit if use_rerank else 0,
        )

        def fused_scores(
            dense_rows: list[tuple[int, float]],
            lexical_rows: list[tuple[int, float]],
        ) -> tuple[dict[int, dict[str, float]], list[int]]:
            fused: dict[int, dict[str, float]] = {}
            for rank, (chunk_id, raw) in enumerate(dense_rows, start=1):
                item = fused.setdefault(
                    chunk_id,
                    {"dense": 0.0, "lexical": 0.0, "rrf": 0.0},
                )
                item["dense"] = raw
                item["rrf"] += 1.0 / (rrf_k + rank)
            for rank, (chunk_id, raw) in enumerate(lexical_rows, start=1):
                item = fused.setdefault(
                    chunk_id,
                    {"dense": 0.0, "lexical": 0.0, "rrf": 0.0},
                )
                item["lexical"] = raw
                item["rrf"] += 1.0 / (rrf_k + rank)
            ordered = sorted(
                fused,
                key=lambda value: (-fused[value]["rrf"], value),
            )
            return fused, ordered

        def frequency_signals(
            candidate_tokens: list[str],
            context: dict[str, Any],
            *,
            allow_collection: bool,
        ) -> dict[str, Any]:
            return media_frequency_signals(
                query_tokens,
                candidate_tokens,
                dict(context.get("document_frequencies") or {}),
                int(context.get("corpus_size") or 0),
                coverage_exponent=float(
                    settings["media_lexical_coverage_exponent"]
                ),
                common_floor=float(settings["media_lexical_common_floor"]),
                oov_penalty=float(settings["media_lexical_oov_penalty"]),
                rarity_exponent=float(
                    settings["media_distinctive_rarity_exponent"]
                ),
                collection_intent=allow_collection and collection_intent,
            )

        asset_metadata_rows: list[dict[str, Any]] = []
        asset_candidate_rank: dict[str, int] = {}
        asset_scores: dict[int, dict[str, float]] = {}
        asset_frequency_signals: dict[str, dict[str, Any]] = {}
        media_frequency_context: dict[str, Any] = {
            "scope": "media_channel_disabled",
            "corpus_size": 0,
            "document_frequencies": {},
            "asset_matches": {},
            "row_ids": {},
        }
        bound_media_frequency_context = dict(media_frequency_context)
        unbound_media_frequency_context = dict(media_frequency_context)
        descriptor_rescue_asset_ids: set[str] = set()
        descriptor_rescue_rank: dict[str, int] = {}
        precomputed_media_dense: list[tuple[int, float]] | None = None
        media_catalog: list[dict[str, Any]] = []
        media_catalog_by_asset: dict[str, list[dict[str, Any]]] = {}
        primary_media_row_by_asset: dict[str, int] = {}

        def index_media_catalog(rows: list[dict[str, Any]]) -> None:
            media_catalog_by_asset.clear()
            primary_media_row_by_asset.clear()
            for row in rows:
                media_catalog_by_asset.setdefault(
                    str(row["asset_id"]), []
                ).append(row)
            for asset_id, items in media_catalog_by_asset.items():
                items.sort(
                    key=lambda item: (
                        int(item.get("sort_order") or 0),
                        int(item["id"]),
                    )
                )
                primary_media_row_by_asset[asset_id] = int(items[0]["id"])

        async def media_dense_by_asset(
            asset_limit: int,
            catalog: list[dict[str, Any]],
        ) -> tuple[list[tuple[int, float]], dict[str, int]]:
            if not catalog:
                return [], {}
            row_to_asset = {
                int(item["id"]): str(item["asset_id"]) for item in catalog
            }
            requested = min(
                len(catalog), max(asset_limit, min(len(catalog), asset_limit * 2))
            )
            dense_rows: list[tuple[int, float]] = []
            while requested:
                dense_rows = await self.indexes.search_media(
                    media_vector, requested
                )
                unique_assets = {
                    row_to_asset.get(int(row_id))
                    for row_id, _score in dense_rows
                    if row_to_asset.get(int(row_id))
                }
                if len(unique_assets) >= asset_limit or requested >= len(
                    catalog
                ):
                    break
                requested = min(len(catalog), max(requested + 1, requested * 2))
            best: dict[str, tuple[float, int]] = {}
            for row_id, score in dense_rows:
                asset_id = row_to_asset.get(int(row_id))
                if asset_id is None:
                    continue
                current = best.get(asset_id)
                if current is None or float(score) > current[0] or (
                    float(score) == current[0] and int(row_id) < current[1]
                ):
                    best[asset_id] = (float(score), int(row_id))
            ordered_assets = sorted(
                best,
                key=lambda asset_id: (
                    -best[asset_id][0],
                    asset_id,
                ),
            )[:asset_limit]
            return (
                [
                    (
                        primary_media_row_by_asset[asset_id],
                        best[asset_id][0],
                    )
                    for asset_id in ordered_assets
                ],
                {
                    asset_id: best[asset_id][1]
                    for asset_id in ordered_assets
                },
            )

        if media_channel_enabled:
            unbound_limit = int(settings["unbound_media_candidate_limit"])
            all_attachments = await self.storage.media_bindings()
            all_bound_chunk_ids = sorted(all_attachments)
            (
                media_catalog,
                bound_media_frequency_context,
                unbound_media_frequency_context,
                precomputed_media_dense,
                media_lexical,
            ) = await asyncio.gather(
                self.storage.media_metadata_records(),
                self.storage.media_token_statistics(
                    query_tokens, scope="bound"
                ),
                self.storage.media_token_statistics(
                    query_tokens, scope="unbound"
                ),
                self.indexes.score_subset(media_vector, all_bound_chunk_ids),
                self.storage.media_lexical_search(media_query),
            )
            index_media_catalog(media_catalog)
            row_by_id = {
                int(row["id"]): row
                for row in media_catalog
                if int(row.get("sort_order") or 0) == 0
            }
            bound_asset_ids = {
                str(item["asset_id"])
                for values in all_attachments.values()
                for item in values
            }
            unbound_asset_ids = {
                str(row["asset_id"])
                for row in row_by_id.values()
                if str(row["asset_id"]) not in bound_asset_ids
            }

            # Unbound media are candidates only through their asset descriptions
            # and token corpus.  The Dense search is collapsed to one score per
            # asset before the fixed candidate window is applied.
            all_asset_dense, _dense_winner = await media_dense_by_asset(
                len(row_by_id), media_catalog
            )
            unbound_dense = [
                (row_id, score)
                for row_id, score in all_asset_dense
                if str(row_by_id[row_id]["asset_id"]) in unbound_asset_ids
            ]
            unbound_lexical: list[tuple[int, float]] = []
            for asset_id, matched in dict(
                unbound_media_frequency_context.get("asset_matches") or {}
            ).items():
                signals = frequency_signals(
                    list(matched),
                    unbound_media_frequency_context,
                    allow_collection=True,
                )
                asset_frequency_signals[str(asset_id)] = signals
                row_id = int(
                    dict(
                        unbound_media_frequency_context.get("row_ids") or {}
                    )[asset_id]
                )
                unbound_lexical.append(
                    (row_id, float(signals["candidate_lexical_score"]))
                )
            unbound_lexical.sort(key=lambda item: (-item[1], item[0]))
            unbound_scores, unbound_ordered_rows = fused_scores(
                unbound_dense, unbound_lexical
            )
            unbound_ordered_rows = [
                row_id
                for row_id in unbound_ordered_rows
                if row_id in row_by_id
                and str(row_by_id[row_id]["asset_id"]) in unbound_asset_ids
            ][:unbound_limit]
            for rank, row_id in enumerate(unbound_ordered_rows, start=1):
                asset_id = str(row_by_id[row_id]["asset_id"])
                asset_scores[row_id] = unbound_scores[row_id]
                asset_candidate_rank[asset_id] = rank

            # Bound media are ranked only inside their own binding structure:
            # fixed chunk Dense+FTS+RRF candidates plus an exact distinctive
            # description rescue. Unbound assets never occupy this window.
            _, bound_ordered_chunk_ids = fused_scores(
                precomputed_media_dense,
                media_lexical,
            )
            bound_asset_order: list[str] = []
            seen_bound_assets: set[str] = set()
            for chunk_id in bound_ordered_chunk_ids:
                for attachment in all_attachments.get(chunk_id, []):
                    asset_id = str(attachment["asset_id"])
                    if (
                        asset_id in seen_bound_assets
                        or asset_id not in primary_media_row_by_asset
                    ):
                        continue
                    seen_bound_assets.add(asset_id)
                    bound_asset_order.append(asset_id)
            if bool(intent["detected"]) and not bool(
                intent.get("blocked_terms")
            ):
                rescue_candidates: list[tuple[str, float]] = []
                for asset_id, matched in dict(
                    bound_media_frequency_context.get("asset_matches") or {}
                ).items():
                    signals = frequency_signals(
                        list(matched),
                        bound_media_frequency_context,
                        allow_collection=False,
                    )
                    asset_frequency_signals[str(asset_id)] = signals
                    if float(signals["distinctive_support"]) >= float(
                        settings["media_bound_distinctive_rescue_min"]
                    ):
                        rescue_candidates.append(
                            (
                                str(asset_id),
                                float(signals["candidate_lexical_score"]),
                            )
                        )
                rescue_candidates.sort(key=lambda item: (-item[1], item[0]))
                descriptor_rescue_asset_ids = {
                    asset_id for asset_id, _score in rescue_candidates
                }
                descriptor_rescue_rank = {
                    asset_id: rank
                    for rank, (asset_id, _score) in enumerate(
                        rescue_candidates, start=1
                    )
                }
                for asset_id, _score in rescue_candidates:
                    if asset_id not in seen_bound_assets:
                        seen_bound_assets.add(asset_id)
                        bound_asset_order.append(asset_id)
            bound_asset_order = bound_asset_order[:candidate_limit]
            for rank, asset_id in enumerate(bound_asset_order, start=1):
                row_id = primary_media_row_by_asset[asset_id]
                contribution = 1.0 / (rrf_k + rank)
                asset_scores[row_id] = {
                    "dense": 0.0,
                    "lexical": float(
                        asset_frequency_signals.get(asset_id, {}).get(
                            "candidate_lexical_score", 0.0
                        )
                    ),
                    "bound_chunk_rrf": contribution,
                    "rrf": contribution,
                }
                asset_candidate_rank[asset_id] = rank

            selected_asset_ids = set(asset_candidate_rank)
            asset_metadata_rows = [
                row
                for row in media_catalog
                if str(row["asset_id"]) in selected_asset_ids
            ]
            attachments = {
                chunk_id: [
                    item
                    for item in values
                    if str(item["asset_id"]) in selected_asset_ids
                ]
                for chunk_id, values in all_attachments.items()
            }
            attachments = {
                chunk_id: values
                for chunk_id, values in attachments.items()
                if values
            }
            media_frequency_context = {
                "scope": "structure_isolated",
                "corpus_size": (
                    int(bound_media_frequency_context.get("corpus_size") or 0)
                    + int(unbound_media_frequency_context.get("corpus_size") or 0)
                ),
                "document_frequencies": {},
                "asset_matches": {},
                "row_ids": {},
            }
        else:
            attachments = {}
            media_lexical = []

        if text_channel_enabled:
            dense, lexical = await asyncio.gather(
                self.indexes.search(text_vector, retrieval_limit),
                self.storage.lexical_search(query, retrieval_limit),
            )
        else:
            dense = []
            lexical = []

        scores, rrf_ordered_ids = fused_scores(dense, lexical)
        text_rrf_rank_by_chunk = {
            chunk_id: rank
            for rank, chunk_id in enumerate(rrf_ordered_ids, start=1)
        }
        text_lexical_boost = float(settings["text_lexical_boost"])
        for chunk_id in rrf_ordered_ids:
            dense_relevance = clamp01(float(scores[chunk_id]["dense"]))
            lexical_relevance = clamp01(float(scores[chunk_id]["lexical"]))
            embedding_relevance = text_embedding_relevance(
                dense_relevance,
                lexical_relevance,
                lexical_boost=text_lexical_boost,
            )
            scores[chunk_id].update(
                {
                    "dense_relevance": dense_relevance,
                    "lexical_relevance": lexical_relevance,
                    "embedding_relevance": embedding_relevance,
                    # Compatibility alias. This is now an absolute relevance
                    # signal rather than max-normalized RRF rank.
                    "embedding_signal": embedding_relevance,
                    "rrf_rank": text_rrf_rank_by_chunk[chunk_id],
                }
            )
        initial_ordered_ids = sorted(
            rrf_ordered_ids,
            key=lambda chunk_id: (
                -float(scores[chunk_id]["embedding_relevance"]),
                -float(scores[chunk_id]["rrf"]),
                int(text_rrf_rank_by_chunk[chunk_id]),
                chunk_id,
            ),
        )
        ordered_ids = list(initial_ordered_ids)
        text_initial_rank_by_chunk = {
            chunk_id: rank
            for rank, chunk_id in enumerate(initial_ordered_ids, start=1)
        }
        if use_rerank and initial_ordered_ids:
            rerank_ids = initial_ordered_ids[:rerank_candidate_limit]
            rerank_rows = {
                int(row["id"]): row
                for row in await self.storage.chunk_rows(rerank_ids)
            }
            if set(rerank_rows) != set(rerank_ids):
                raise RerankStageError("text_chunks: candidate rows are incomplete")
            rerank_details, stage_meta = await self._rerank_documents(
                query,
                [str(rerank_rows[chunk_id]["text"]) for chunk_id in rerank_ids],
                settings=settings,
                scope="text_chunks",
            )
            rerank_scope_meta.append(stage_meta)
            for chunk_id, detail in zip(
                rerank_ids, rerank_details, strict=True
            ):
                fused = fuse_embedding_rerank(
                    float(scores[chunk_id]["embedding_signal"]),
                    float(detail["raw_score"]),
                    rank=int(detail["rank"]),
                    candidate_count=int(detail["candidate_count"]),
                    fusion_weight=float(settings["rerank_fusion_weight"]),
                    rank_bonus_weight=float(
                        settings["rerank_rank_bonus_weight"]
                    ),
                    rank_reliability_exponent=float(
                        settings["rerank_rank_reliability_exponent"]
                    ),
                )
                scores[chunk_id].update(
                    {
                        "initial_rank": text_initial_rank_by_chunk[chunk_id],
                        "rerank_raw_score": float(detail["raw_score"]),
                        "rerank_rank": int(detail["rank"]),
                        **fused,
                    }
                )
            reranked_head = sorted(
                rerank_ids,
                key=lambda chunk_id: (
                    -float(scores[chunk_id]["ordering_relevance"]),
                    int(scores[chunk_id]["rerank_rank"]),
                    text_initial_rank_by_chunk[chunk_id],
                    chunk_id,
                ),
            )
            ordered_ids = reranked_head + initial_ordered_ids[len(rerank_ids) :]
        baseline_selected = initial_ordered_ids[:top_k]
        selected = ordered_ids[:top_k]
        selected_row_ids = list(dict.fromkeys([*selected, *baseline_selected]))
        rows = {
            int(row["id"]): row
            for row in await self.storage.chunk_rows(selected_row_ids)
        }
        selected = [chunk_id for chunk_id in selected if chunk_id in rows]
        baseline_selected = [
            chunk_id for chunk_id in baseline_selected if chunk_id in rows
        ]
        standard_media_probe_count = (
            len(
                {
                    str(item["asset_id"])
                    for values in attachments.values()
                    for item in values
                }
            )
            if retrieval_mode == "standard"
            else 0
        )
        standard_media_skipped_reason: str | None = (
            "media_channel_disabled"
            if not media_channel_enabled
            else "no_bound_chunk_in_fixed_candidate_window"
            if not attachments
            else None
        )
        bound_chunk_ids = sorted(attachments)
        if not media_channel_enabled or not bound_chunk_ids:
            media_dense = []
        elif precomputed_media_dense is None:
            media_dense = await self.indexes.score_subset(
                media_vector, bound_chunk_ids
            )
        else:
            selected_bound_chunk_ids = set(bound_chunk_ids)
            media_dense = [
                item
                for item in precomputed_media_dense
                if int(item[0]) in selected_bound_chunk_ids
            ]
        media_candidates: dict[str, dict[str, Any]] = {}
        text_rank_by_chunk = {
            chunk_id: rank for rank, chunk_id in enumerate(selected, start=1)
        }
        scope_priority = {"chunk": 0, "entry": 1, "document": 2}
        asset_bindings: dict[str, dict[int, list[dict[str, Any]]]] = {}

        metadata_by_asset: dict[str, list[dict[str, Any]]] = {}
        for metadata in asset_metadata_rows:
            metadata_by_asset.setdefault(
                str(metadata["asset_id"]), []
            ).append(metadata)
        for asset_id, description_rows in metadata_by_asset.items():
            description_rows.sort(
                key=lambda item: (
                    int(item.get("sort_order") or 0),
                    int(item["id"]),
                )
            )
            primary = description_rows[0]
            description_texts = [
                str(item.get("media_description") or "")
                for item in description_rows
            ]
            filename_stem = Path(
                str(primary.get("original_name") or "")
            ).stem
            asset_direct_text = " ".join(
                value for value in [*description_texts, filename_stem] if value
            )
            asset_direct_tokens = media_tokens(asset_direct_text)
            asset_metadata_only = not bool(primary.get("is_bound"))
            asset_frequency_context = (
                unbound_media_frequency_context
                if asset_metadata_only
                else bound_media_frequency_context
            )
            frequency = frequency_signals(
                asset_direct_tokens,
                asset_frequency_context,
                allow_collection=asset_metadata_only,
            )
            asset_frequency_signals[asset_id] = frequency
            primary_row_id = int(primary["id"])
            initial = asset_scores.get(
                primary_row_id,
                {"dense": 0.0, "lexical": 0.0, "rrf": 0.0},
            )
            direct_relations: dict[str, dict[str, Any]] = {}
            for metadata in description_rows:
                description = str(metadata.get("media_description") or "")
                direct_text = " ".join(
                    value for value in (description, filename_stem) if value
                )
                direct_tokens = media_tokens(direct_text)
                description_frequency = frequency_signals(
                    direct_tokens,
                    asset_frequency_context,
                    allow_collection=asset_metadata_only,
                )
                raw_coverage, matched_tokens = weighted_token_coverage(
                    query_tokens, direct_tokens
                )
                semantic_similarity: float | None = None
                media_vector = metadata.get("media_description_vector")
                if media_vector is not None:
                    candidate_vector = np.asarray(
                        media_vector, dtype=np.float32
                    )
                    if (
                        candidate_vector.ndim == 1
                        and candidate_vector.shape
                        == normalized_media_query_vector.shape
                        and np.isfinite(candidate_vector).all()
                    ):
                        vector_norm = float(np.linalg.norm(candidate_vector))
                        if vector_norm > 0:
                            semantic_similarity = clamp01(
                                float(
                                    (candidate_vector / vector_norm)
                                    @ normalized_media_query_vector
                                )
                            )
                semantic_score = calibrated_semantic(
                    semantic_similarity,
                    float(settings["media_semantic_floor"]),
                )
                relation_key = f"description:{int(metadata['id'])}"
                direct_relations[relation_key] = {
                    "scope": "asset_media_description",
                    "relation_target_id": asset_id,
                    "media_description_id": int(metadata["id"]),
                    "media_description": description,
                    "media_description_sort_order": int(
                        metadata.get("sort_order") or 0
                    ),
                    "description_source": str(
                        metadata.get("description_source") or "user"
                    ),
                    "matched_tokens": matched_tokens,
                    "lexical_coverage": round(raw_coverage, 6),
                    "lexical_score": round(
                        float(
                            description_frequency[
                                "base_lexical_score"
                            ]
                        ),
                        6,
                    ),
                    "frequency_weighted_completeness": round(
                        float(description_frequency["completeness"]), 6
                    ),
                    "frequency_informativeness": round(
                        float(description_frequency["informativeness"]), 6
                    ),
                    "distinctive_support": round(
                        float(
                            description_frequency[
                                "distinctive_support"
                            ]
                        ),
                        6,
                    ),
                    "distinctive_membership_support": round(
                        float(
                            description_frequency[
                                "distinctive_membership_support"
                            ]
                        ),
                        6,
                    ),
                    "collection_membership_support": round(
                        float(
                            description_frequency[
                                "collection_membership_support"
                            ]
                        ),
                        6,
                    ),
                    "collection_support": round(
                        float(
                            description_frequency["collection_support"]
                        ),
                        6,
                    ),
                    "frequency_candidate_lexical_score": round(
                        float(
                            description_frequency[
                                "candidate_lexical_score"
                            ]
                        ),
                        6,
                    ),
                    "frequency_token_details": list(
                        description_frequency["token_details"]
                    ),
                    "media_vector_similarity": (
                        round(semantic_similarity, 6)
                        if semantic_similarity is not None
                        else None
                    ),
                    "calibrated_semantic_score": round(
                        semantic_score, 6
                    ),
                    "_direct_text": direct_text,
                    "_is_description": True,
                    "_asset_metadata_only": asset_metadata_only,
                }
            media_candidates[asset_id] = {
                "asset_id": asset_id,
                "sha256": primary["sha256"],
                "mime_type": primary["mime_type"],
                "size_bytes": int(primary["size_bytes"]),
                "width": primary.get("width"),
                "height": primary.get("height"),
                "original_name": primary.get("original_name") or "",
                "media_description": description_texts[0],
                "media_description_count": len(description_texts),
                "media_description_set_sha256": (
                    media_description_set_sha256(description_texts)
                ),
                "media_descriptions": [
                    {
                        "id": int(item["id"]),
                        "media_description": str(
                            item["media_description"]
                        ),
                        "sort_order": int(item.get("sort_order") or 0),
                        "description_source": str(
                            item.get("description_source") or "user"
                        ),
                    }
                    for item in description_rows
                ],
                "caption": "",
                "alt_text": "",
                "sort_order": 0,
                "media_evidence_scope": (
                    "asset_bound_only"
                    if bool(primary.get("is_bound"))
                    else "asset_metadata_only"
                ),
                "media_ranking_scope": (
                    "per_asset"
                    if bool(primary.get("is_bound"))
                    else "asset_index"
                ),
                "asset_metadata_only": not bool(primary.get("is_bound")),
                "_media_frequency_scope": str(
                    asset_frequency_context.get("scope") or ""
                ),
                "_media_frequency_corpus_size": int(
                    asset_frequency_context.get("corpus_size") or 0
                ),
                "asset_candidate_rank": asset_candidate_rank.get(asset_id),
                "asset_dense_score": round(
                    clamp01(float(initial["dense"])), 6
                ),
                "asset_lexical_score": round(
                    float(initial["lexical"]), 6
                ),
                "asset_rrf_score": round(float(initial["rrf"]), 6),
                "asset_bound_chunk_rrf_score": round(
                    float(initial.get("bound_chunk_rrf", 0.0)), 6
                ),
                "bound_chunk_count": 0,
                "candidate_chunk_count": 0,
                "descriptor_rescue": (
                    asset_id in descriptor_rescue_asset_ids
                ),
                "descriptor_rescue_rank": descriptor_rescue_rank.get(
                    asset_id
                ),
                "_policies": (
                    set() if bool(primary.get("is_bound")) else {"auto"}
                ),
                "_evidence_by_chunk": {},
                "_direct_relations": direct_relations,
                "_asset_direct_text": asset_direct_text,
                "_asset_direct_tokens": asset_direct_tokens,
            }

        for chunk_id, chunk_attachments in attachments.items():
            for attachment in chunk_attachments:
                policy = str(attachment.get("output_policy") or "auto")
                asset_id = str(attachment["asset_id"])
                asset_bindings.setdefault(asset_id, {}).setdefault(
                    chunk_id, []
                ).append(attachment)
                candidate = media_candidates.get(asset_id)
                if candidate is None:
                    # Every active media relation is expected to have at least
                    # one migrated asset-level description.
                    continue
                candidate["_policies"].add(policy)
                if attachment.get("caption"):
                    candidate["caption"] = str(attachment["caption"])
                if attachment.get("alt_text"):
                    candidate["alt_text"] = str(attachment["alt_text"])
                candidate["sort_order"] = min(
                    int(candidate["sort_order"]),
                    int(attachment.get("sort_order") or 0),
                )
                auxiliary_text = " ".join(
                    value
                    for value in (
                        str(attachment.get("caption") or ""),
                        str(attachment.get("alt_text") or ""),
                    )
                    if value
                )
                if auxiliary_text:
                    candidate["_asset_direct_text"] = " ".join(
                        value
                        for value in (
                            str(candidate.get("_asset_direct_text") or ""),
                            auxiliary_text,
                        )
                        if value
                    )
                    candidate["_asset_direct_tokens"] = media_tokens(
                        str(candidate["_asset_direct_text"])
                    )
                    for relation in candidate[
                        "_direct_relations"
                    ].values():
                        relation["_direct_text"] = " ".join(
                            value
                            for value in (
                                str(relation.get("_direct_text") or ""),
                                auxiliary_text,
                            )
                            if value
                        )

        if use_rerank and media_candidates:
            def direct_relation_signal(item: dict[str, Any]) -> float:
                return noisy_or(
                    (
                        float(settings["media_semantic_weight"])
                        * float(item["calibrated_semantic_score"]),
                        float(settings["media_lexical_boost"])
                        * float(item["lexical_score"]),
                        (
                            float(settings["unbound_media_distinctive_boost"])
                            if bool(item.get("_asset_metadata_only"))
                            else float(settings["media_bound_distinctive_boost"])
                        )
                        * float(item.get("distinctive_support", 0.0)),
                        (
                            float(settings["unbound_media_collection_boost"])
                            * float(
                                max(
                                    float(
                                        item.get(
                                            "collection_support", 0.0
                                        )
                                    ),
                                    float(
                                        item.get(
                                            "collection_membership_support",
                                            0.0,
                                        )
                                    ),
                                    float(
                                        item.get(
                                            "distinctive_membership_support",
                                            0.0,
                                        )
                                    ),
                                )
                            )
                            if bool(item.get("_asset_metadata_only"))
                            and collection_intent
                            else 0.0
                        ),
                    )
                )

            for metadata_only, scope_name in (
                (False, "bound"),
                (True, "unbound"),
            ):
                direct_asset_ids = sorted(
                    (
                        asset_id
                        for asset_id, candidate in media_candidates.items()
                        if bool(candidate.get("asset_metadata_only"))
                        is metadata_only
                    ),
                    key=lambda asset_id: (
                        -max(
                            (
                                direct_relation_signal(item)
                                for item in media_candidates[asset_id][
                                    "_direct_relations"
                                ].values()
                            ),
                            default=0.0,
                        ),
                        asset_id,
                    ),
                )
                relation_queues: dict[
                    str, list[tuple[str, dict[str, Any]]]
                ] = {}
                for asset_id in direct_asset_ids:
                    relation_queues[asset_id] = sorted(
                        (
                            (key, relation)
                            for key, relation in media_candidates[asset_id][
                                "_direct_relations"
                            ].items()
                            if relation.get("_is_description")
                            and str(relation.get("_direct_text") or "")
                        ),
                        key=lambda item: (
                            -direct_relation_signal(item[1]),
                            int(
                                item[1].get(
                                    "media_description_sort_order", 0
                                )
                            ),
                            item[0],
                        ),
                    )
                selected_descriptions: list[
                    tuple[str, str, dict[str, Any]]
                ] = []
                budget_asset_ids = direct_asset_ids[:rerank_candidate_limit]
                asset_count = len(budget_asset_ids)
                if asset_count:
                    quota, remainder = divmod(
                        rerank_candidate_limit, asset_count
                    )
                    quotas = {
                        asset_id: quota + (index < remainder)
                        for index, asset_id in enumerate(budget_asset_ids)
                    }
                    for round_index in range(
                        max(quotas.values(), default=0)
                    ):
                        for asset_id in budget_asset_ids:
                            if round_index >= quotas[asset_id]:
                                continue
                            queue = relation_queues[asset_id]
                            if round_index >= len(queue):
                                continue
                            key, relation = queue[round_index]
                            selected_descriptions.append(
                                (asset_id, key, relation)
                            )
                if not selected_descriptions:
                    continue
                direct_details, stage_meta = await self._rerank_documents(
                    media_query,
                    [
                        str(relation["_direct_text"])
                        for _asset_id, _key, relation in selected_descriptions
                    ],
                    settings=settings,
                    scope=f"media_descriptions:{scope_name}",
                )
                rerank_scope_meta.append(stage_meta)
                scored_descriptions = [
                    (asset_id, key, relation, detail)
                    for (asset_id, key, relation), detail in zip(
                        selected_descriptions, direct_details, strict=True
                    )
                ]
                best_raw_by_asset: dict[str, float] = {}
                for asset_id, _key, _relation, detail in scored_descriptions:
                    best_raw_by_asset[asset_id] = max(
                        best_raw_by_asset.get(asset_id, -math.inf),
                        float(detail["raw_score"]),
                    )
                ranked_assets = sorted(
                    best_raw_by_asset,
                    key=lambda asset_id: (
                        -best_raw_by_asset[asset_id],
                        asset_id,
                    ),
                )
                asset_rerank_rank = {
                    asset_id: rank
                    for rank, asset_id in enumerate(ranked_assets, start=1)
                }
                for asset_id, _key, relation, detail in scored_descriptions:
                    embedding_semantic = float(
                        relation["calibrated_semantic_score"]
                    )
                    fused = fuse_embedding_rerank(
                        embedding_semantic,
                        float(detail["raw_score"]),
                        rank=asset_rerank_rank[asset_id],
                        candidate_count=len(ranked_assets),
                        fusion_weight=float(
                            settings["rerank_fusion_weight"]
                        ),
                        rank_bonus_weight=float(
                            settings["rerank_rank_bonus_weight"]
                        ),
                        rank_reliability_exponent=float(
                            settings["rerank_rank_reliability_exponent"]
                        ),
                    )
                    relation.update(
                        {
                            "embedding_calibrated_semantic_score": round(
                                embedding_semantic, 6
                            ),
                            "rerank_raw_score": round(
                                float(detail["raw_score"]), 6
                            ),
                            "rerank_description_rank": int(detail["rank"]),
                            "rerank_rank": asset_rerank_rank[asset_id],
                            "rerank_probability": round(
                                float(fused["rerank_probability"]), 6
                            ),
                            "rerank_signal": round(
                                float(fused["rerank_signal"]), 6
                            ),
                            "calibrated_semantic_score": round(
                                float(fused["fused_relevance"]), 6
                            ),
                        }
                    )

        dense_by_chunk = {chunk_id: score for chunk_id, score in media_dense}
        lexical_by_chunk = {chunk_id: score for chunk_id, score in media_lexical}
        bound_chunk_rows = (
            {
                int(row["id"]): row
                for row in await self.storage.chunk_rows(bound_chunk_ids)
            }
            if use_rerank
            else {}
        )
        current_rerank_settings_fingerprint = (
            rerank_calibration_settings_fingerprint(settings)
        )
        for asset_id, chunks_for_asset in asset_bindings.items():
            candidate = media_candidates[asset_id]
            chunk_ids = sorted(chunks_for_asset)
            dense_rows = sorted(
                (
                    (chunk_id, dense_by_chunk[chunk_id])
                    for chunk_id in chunk_ids
                    if chunk_id in dense_by_chunk
                ),
                key=lambda item: (-item[1], item[0]),
            )
            lexical_rows = sorted(
                (
                    (chunk_id, lexical_by_chunk[chunk_id])
                    for chunk_id in chunk_ids
                    if chunk_id in lexical_by_chunk
                ),
                key=lambda item: (item[1], item[0]),
            )
            media_scores, ordered_media_ids = fused_scores(
                dense_rows, lexical_rows
            )
            evidence_ids = ordered_media_ids[:candidate_limit]
            initial_media_rank = {
                chunk_id: rank
                for rank, chunk_id in enumerate(evidence_ids, start=1)
            }
            media_rerank_by_chunk: dict[int, dict[str, Any]] = {}
            if use_rerank and evidence_ids:
                rerank_ids = evidence_ids[:rerank_candidate_limit]
                if any(chunk_id not in bound_chunk_rows for chunk_id in rerank_ids):
                    raise RerankStageError(
                        f"media_bound_chunks:{asset_id}: candidate rows are incomplete"
                    )
                rerank_details, stage_meta = await self._rerank_documents(
                    media_query,
                    [
                        str(bound_chunk_rows[chunk_id]["text"])
                        for chunk_id in rerank_ids
                    ],
                    settings=settings,
                    scope=f"media_bound_chunks:{asset_id}",
                )
                rerank_scope_meta.append(stage_meta)
                for chunk_id, detail in zip(
                    rerank_ids, rerank_details, strict=True
                ):
                    embedding_signal = clamp01(
                        float(media_scores[chunk_id]["dense"])
                    )
                    fused = fuse_embedding_rerank(
                        embedding_signal,
                        float(detail["raw_score"]),
                        rank=int(detail["rank"]),
                        candidate_count=int(detail["candidate_count"]),
                        fusion_weight=float(settings["rerank_fusion_weight"]),
                        rank_bonus_weight=float(
                            settings["rerank_rank_bonus_weight"]
                        ),
                        rank_reliability_exponent=float(
                            settings["rerank_rank_reliability_exponent"]
                        ),
                    )
                    media_rerank_by_chunk[chunk_id] = {
                        "initial_rank": initial_media_rank[chunk_id],
                        "rerank_raw_score": float(detail["raw_score"]),
                        "rerank_rank": int(detail["rank"]),
                        **fused,
                    }
                reranked_head = sorted(
                    rerank_ids,
                    key=lambda chunk_id: (
                        -float(
                            media_rerank_by_chunk[chunk_id]["ordering_relevance"]
                        ),
                        int(media_rerank_by_chunk[chunk_id]["rerank_rank"]),
                        initial_media_rank[chunk_id],
                        chunk_id,
                    ),
                )
                evidence_ids = reranked_head + evidence_ids[len(rerank_ids) :]
            candidate["bound_chunk_count"] = len(chunk_ids)
            candidate["candidate_chunk_count"] = len(evidence_ids)
            for rank, chunk_id in enumerate(evidence_ids, start=1):
                dense_score = max(
                    0.0,
                    min(1.0, float(media_scores[chunk_id]["dense"])),
                )
                relevance = (
                    float(
                        media_rerank_by_chunk[chunk_id]["fused_relevance"]
                    )
                    if chunk_id in media_rerank_by_chunk
                    else dense_score
                )
                for attachment in chunks_for_asset[chunk_id]:
                    weight = max(
                        0.0,
                        min(1.0, float(attachment.get("relation_weight") or 0.0)),
                    )
                    embedding_semantic_strength = max(
                        0.0,
                        min(
                            1.0,
                            float(attachment.get("semantic_strength") or 0.0),
                        ),
                    )
                    rerank_semantic_strength = attachment.get(
                        "rerank_semantic_strength"
                    )
                    strength_rerank_fingerprint = str(
                        attachment.get("strength_rerank_provider_fingerprint")
                        or ""
                    )
                    relation_rerank_fingerprint = str(
                        attachment.get(
                            "relation_calibration_rerank_fingerprint"
                        )
                        or ""
                    )
                    relation_description_set_sha256 = str(
                        attachment.get(
                            "relation_calibration_description_set_sha256"
                        )
                        or ""
                    )
                    calibration_details = dict(
                        attachment.get("calibration_details") or {}
                    )
                    strength_rerank_settings_fingerprint = str(
                        calibration_details.get("rerank_settings_fingerprint")
                        or ""
                    )
                    rerank_strength_current = bool(
                        use_rerank
                        and rerank_semantic_strength is not None
                        and strength_rerank_fingerprint
                        == self.rerank_provider_fingerprint
                        and (
                            not relation_rerank_fingerprint
                            or relation_rerank_fingerprint
                            == self.rerank_provider_fingerprint
                        )
                        and strength_rerank_settings_fingerprint
                        == current_rerank_settings_fingerprint
                        and (
                            not relation_description_set_sha256
                            or relation_description_set_sha256
                            == str(
                                candidate.get(
                                    "media_description_set_sha256"
                                )
                                or ""
                            )
                        )
                    )
                    semantic_strength = (
                        clamp01(float(rerank_semantic_strength))
                        if rerank_strength_current
                        else embedding_semantic_strength
                    )
                    evidence_score = relevance * weight * semantic_strength
                    rank_weight = 1.0 / (
                        math.log2(rank + 1)
                        ** float(settings["media_rank_decay_exponent"])
                    )
                    ranked_evidence_score = evidence_score * rank_weight
                    evidence = {
                        "chunk_id": chunk_id,
                        "rank": rank,
                        "scope": attachment["scope"],
                        "relation_target_id": attachment.get("relation_target_id"),
                        "dense_score": round(dense_score, 6),
                        "fused_relevance": round(relevance, 6),
                        "ordering_relevance": (
                            round(
                                float(
                                    media_rerank_by_chunk[chunk_id][
                                        "ordering_relevance"
                                    ]
                                ),
                                6,
                            )
                            if chunk_id in media_rerank_by_chunk
                            else round(dense_score, 6)
                        ),
                        "initial_rank": initial_media_rank.get(chunk_id),
                        "rerank_raw_score": (
                            round(
                                float(
                                    media_rerank_by_chunk[chunk_id][
                                        "rerank_raw_score"
                                    ]
                                ),
                                6,
                            )
                            if chunk_id in media_rerank_by_chunk
                            else None
                        ),
                        "rerank_rank": (
                            int(media_rerank_by_chunk[chunk_id]["rerank_rank"])
                            if chunk_id in media_rerank_by_chunk
                            else None
                        ),
                        "relation_weight": round(weight, 6),
                        "semantic_strength": round(semantic_strength, 6),
                        "embedding_semantic_strength": round(
                            embedding_semantic_strength, 6
                        ),
                        "rerank_semantic_strength": (
                            round(float(rerank_semantic_strength), 6)
                            if rerank_semantic_strength is not None
                            else None
                        ),
                        "semantic_strength_source": (
                            "rerank_calibration"
                            if rerank_strength_current
                            else "embedding_baseline"
                        ),
                        "rerank_calibration_current": rerank_strength_current,
                        "rerank_calibration_settings_fingerprint": (
                            strength_rerank_settings_fingerprint
                        ),
                        "calibration_similarity": (
                            round(float(attachment["calibration_similarity"]), 6)
                            if attachment.get("calibration_similarity") is not None
                            else None
                        ),
                        "calibration_rank": attachment.get("calibration_rank"),
                        "calibration_method": attachment.get(
                            "calibration_method", UNIFORM_MEDIA_CALIBRATION_METHOD
                        ),
                        "media_evidence_score": round(evidence_score, 6),
                        "rank_weight": round(rank_weight, 6),
                        "ranked_evidence_score": round(ranked_evidence_score, 6),
                    }
                    existing_evidence = candidate["_evidence_by_chunk"].get(chunk_id)
                    evidence_key = (
                        float(evidence["ranked_evidence_score"]),
                        -scope_priority.get(str(evidence["scope"]), 99),
                    )
                    existing_key = (
                        float(existing_evidence["ranked_evidence_score"]),
                        -scope_priority.get(str(existing_evidence["scope"]), 99),
                    ) if existing_evidence is not None else (-1.0, -99)
                    if evidence_key > existing_key:
                        candidate["_evidence_by_chunk"][chunk_id] = evidence

        def direct_relation_order_score(
            item: dict[str, Any], *, embedding_baseline: bool
        ) -> float:
            semantic_key = (
                "embedding_calibrated_semantic_score"
                if embedding_baseline
                else "calibrated_semantic_score"
            )
            semantic_value = float(
                item.get(semantic_key, item.get("calibrated_semantic_score", 0.0))
            )
            values = [
                float(settings["media_semantic_weight"]) * semantic_value,
                float(settings["media_lexical_boost"])
                * float(item.get("lexical_score", 0.0)),
            ]
            if bool(item.get("_asset_metadata_only")):
                values.extend(
                    (
                        float(settings["unbound_media_distinctive_boost"])
                        * float(item.get("distinctive_support", 0.0)),
                        float(settings["unbound_media_collection_boost"])
                        * max(
                            float(item.get("collection_support", 0.0)),
                            float(
                                item.get(
                                    "collection_membership_support", 0.0
                                )
                            ),
                            float(
                                item.get(
                                    "distinctive_membership_support", 0.0
                                )
                            ),
                        ),
                    )
                )
            else:
                values.append(
                    float(settings["media_bound_distinctive_boost"])
                    * float(item.get("distinctive_support", 0.0))
                )
            return noisy_or(values)

        decisions: list[dict[str, Any]] = []
        eligible: list[dict[str, Any]] = []
        for candidate in media_candidates.values():
            asset_metadata_only = bool(candidate.get("asset_metadata_only"))
            policies = set(candidate.pop("_policies"))
            evidence = sorted(
                candidate.pop("_evidence_by_chunk").values(),
                key=lambda item: (int(item["rank"]), int(item["chunk_id"])),
            )
            direct_relations = list(candidate.pop("_direct_relations").values())
            asset_direct_text = str(candidate.pop("_asset_direct_text", ""))
            asset_direct_tokens = list(
                candidate.pop("_asset_direct_tokens", [])
            )
            _asset_coverage, asset_matched_tokens = weighted_token_coverage(
                query_tokens, asset_direct_tokens
            )
            candidate_media_format_groups = media_format_groups(
                asset_direct_text
            )
            for relation in direct_relations:
                relation.pop("_direct_text", None)
            direct = max(
                direct_relations,
                key=lambda item: (
                    direct_relation_order_score(
                        item, embedding_baseline=False
                    ),
                    str(item["scope"]),
                    str(item["relation_target_id"]),
                ),
                default={
                    "scope": None,
                    "relation_target_id": None,
                    "matched_tokens": [],
                    "lexical_coverage": 0.0,
                    "lexical_score": 0.0,
                    "frequency_weighted_completeness": 0.0,
                    "frequency_informativeness": 0.0,
                    "distinctive_support": 0.0,
                    "distinctive_membership_support": 0.0,
                    "collection_membership_support": 0.0,
                    "collection_support": 0.0,
                    "frequency_candidate_lexical_score": 0.0,
                    "frequency_token_details": [],
                    "media_vector_similarity": None,
                    "calibrated_semantic_score": 0.0,
                },
            )
            implicit_media_lookup = (
                not bool(intent.get("blocked_terms"))
                and max(
                    float(direct["lexical_coverage"]),
                    float(
                        direct.get("frequency_weighted_completeness", 0.0)
                    ),
                )
                >= 0.8
            )
            direct_signal_enabled = bool(intent["detected"]) or implicit_media_lookup
            semantic_component = (
                clamp01(
                    float(settings["media_semantic_weight"])
                    * float(direct["calibrated_semantic_score"])
                )
                if direct_signal_enabled
                else 0.0
            )
            lexical_component = (
                clamp01(
                    float(settings["media_lexical_boost"])
                    * float(direct["lexical_score"])
                )
                if direct_signal_enabled
                else 0.0
            )
            distinctive_support = float(
                direct.get("distinctive_support", 0.0)
            )
            distinctive_membership_support = float(
                direct.get("distinctive_membership_support", 0.0)
            )
            collection_support = float(
                direct.get("collection_support", 0.0)
            )
            collection_membership_support = max(
                float(
                    direct.get("collection_membership_support", 0.0)
                ),
                collection_support,
                distinctive_membership_support,
            )
            subject_anchor_required = bool(
                intent.get("subject_anchor_required")
            )
            normalized_asset_direct_text = normalize_media_text(
                asset_direct_text
            )
            subject_anchor_terms = {
                normalize_media_text(value)
                for value in intent.get("subject_anchor_terms") or []
                if normalize_media_text(value)
            }
            subject_anchor_met = (
                not subject_anchor_required
                or any(
                    anchor in normalized_asset_direct_text
                    for anchor in subject_anchor_terms
                )
            )
            concrete_query_formats = (
                query_media_format_groups - {"generic_image"}
            )
            multi_format_query = len(concrete_query_formats) >= 2
            asset_matched_subject_tokens = (
                set(query_subject_tokens) & set(asset_matched_tokens)
            )
            asset_has_content_match = bool(
                asset_matched_subject_tokens
            )
            asset_content_anchor_support = float(
                asset_frequency_signals.get(
                    str(candidate.get("asset_id") or ""), {}
                ).get("distinctive_support", 0.0)
            )
            media_format_compatible = (
                not concrete_query_formats
                or not candidate_media_format_groups
                or bool(
                    concrete_query_formats
                    & candidate_media_format_groups
                )
                or (multi_format_query and asset_has_content_match)
            )
            content_anchor_required = (
                (
                    not asset_metadata_only
                    and bool(candidate_media_format_groups)
                    and bool(query_subject_tokens)
                    and (
                        bool(
                            query_media_format_groups
                            & {
                                "generic_image",
                                "illustration",
                                "scene",
                                "cover",
                            }
                        )
                        or (
                            retrieval_mode == "media_only"
                            and not explicit_visual_intent
                        )
                    )
                )
                or (
                    asset_metadata_only
                    and bool(concrete_query_formats)
                    and not media_format_compatible
                )
            )
            content_anchor_met = (
                not content_anchor_required
                or (
                    asset_has_content_match
                    and (
                        len(query_subject_tokens) == 1
                        or asset_content_anchor_support >= 0.25
                    )
                )
            )
            ambiguous_collection = (
                not asset_metadata_only
                and collection_intent
                and not concrete_query_formats
                and not query_subject_tokens
            )
            media_gate_open = (
                direct_signal_enabled
                and not ambiguous_collection
            )
            max_chunk_evidence = max(
                (float(item["media_evidence_score"]) for item in evidence),
                default=0.0,
            )
            grounding = aggregate_grounding(
                (float(item["ranked_evidence_score"]) for item in evidence),
                corroboration_weight=float(
                    settings["media_corroboration_weight"]
                ),
                limit=int(settings["media_corroboration_limit"]),
            )
            grounding_reliability = noisy_or(
                (
                    float(direct["calibrated_semantic_score"]),
                    clamp01(
                        max_chunk_evidence
                        / max(effective_media_score_threshold, 1e-9)
                    ),
                )
            )
            distinctive_component = (
                clamp01(
                    float(settings["unbound_media_distinctive_boost"])
                    * distinctive_support
                )
                if asset_metadata_only and direct_signal_enabled
                else clamp01(
                    float(settings["media_bound_distinctive_boost"])
                    * distinctive_support
                    * grounding_reliability
                )
                if direct_signal_enabled
                else 0.0
            )
            collection_component = (
                clamp01(
                    float(settings["unbound_media_collection_boost"])
                    * collection_membership_support
                )
                if (
                    asset_metadata_only
                    and collection_intent
                    and direct_signal_enabled
                )
                else 0.0
            )
            direct_candidate_score = noisy_or(
                (
                    semantic_component,
                    lexical_component,
                    distinctive_component,
                    collection_component,
                )
            )
            association_score = (
                direct_candidate_score
                if asset_metadata_only
                else noisy_or(
                    (
                        max_chunk_evidence,
                        semantic_component,
                        lexical_component,
                        distinctive_component,
                    )
                )
            )
            confidence_result = score_media_confidence(
                raw_direct_relevance=direct_candidate_score,
                grounding=grounding,
                evidence=(
                    (
                        float(item["media_evidence_score"]),
                        int(item["rank"]),
                    )
                    for item in evidence
                ),
                bound_media=not asset_metadata_only,
                relevance_pivot=effective_media_relevance_pivot,
                direct_signal_enabled=direct_signal_enabled,
                ambiguous_collection=ambiguous_collection,
                subject_anchor_met=subject_anchor_met,
                media_format_compatible=media_format_compatible,
                content_anchor_met=content_anchor_met,
                evidence_limit=int(settings["media_threshold_evidence_limit"]),
                rank_decay_exponent=float(
                    settings["media_threshold_rank_decay_exponent"]
                ),
                negative_reliability_exponent=float(
                    settings[
                        "media_threshold_negative_reliability_exponent"
                    ]
                ),
                positive_blend=float(settings["media_pivot_positive_blend"]),
                negative_weight=float(settings["media_pivot_negative_weight"]),
                negative_attenuation_floor=float(
                    settings["media_pivot_negative_attenuation_floor"]
                ),
                format_mismatch_factor=float(
                    settings["media_format_mismatch_factor"]
                ),
                content_mismatch_factor=float(
                    settings["media_content_mismatch_factor"]
                ),
                tail_rank=int(settings["media_corroboration_limit"]),
            )
            threshold_result = confidence_result["threshold_result"]
            adjusted_grounding = float(
                threshold_result["adjusted_grounding"]
            )
            pre_threshold_confidence = float(
                confidence_result["pre_pivot_output_confidence"]
            )
            output_confidence = float(confidence_result["output_confidence"])
            pivot_calibrated_direct_score = float(
                confidence_result["pivoted_direct_relevance"]
            )
            structural_attenuation_factor = float(
                confidence_result["structural_attenuation_factor"]
            )
            format_attenuation_factor = float(
                confidence_result["format_attenuation_factor"]
            )
            content_attenuation_factor = float(
                confidence_result["content_attenuation_factor"]
            )
            baseline_output_confidence = output_confidence
            baseline_direct_candidate_score = direct_candidate_score
            baseline_pivot_calibrated_direct_score = (
                pivot_calibrated_direct_score
            )
            baseline_gated_adjusted_grounding = adjusted_grounding
            if use_rerank:
                baseline_direct = max(
                    direct_relations,
                    key=lambda item: (
                        direct_relation_order_score(
                            item, embedding_baseline=True
                        ),
                        str(item["scope"]),
                        str(item["relation_target_id"]),
                    ),
                    default=direct,
                )
                baseline_implicit_lookup = (
                    not bool(intent.get("blocked_terms"))
                    and max(
                        float(baseline_direct["lexical_coverage"]),
                        float(
                            baseline_direct.get(
                                "frequency_weighted_completeness", 0.0
                            )
                        ),
                    )
                    >= 0.8
                )
                baseline_direct_enabled = bool(
                    intent["detected"]
                ) or baseline_implicit_lookup
                baseline_subject_met = (
                    not subject_anchor_required
                    or any(
                        anchor in normalized_asset_direct_text
                        for anchor in subject_anchor_terms
                    )
                )
                baseline_semantic_component = (
                    clamp01(
                        float(settings["media_semantic_weight"])
                        * float(
                            baseline_direct.get(
                                "embedding_calibrated_semantic_score",
                                baseline_direct["calibrated_semantic_score"],
                            )
                        )
                    )
                    if baseline_direct_enabled
                    else 0.0
                )
                baseline_lexical_component = (
                    clamp01(
                        float(settings["media_lexical_boost"])
                        * float(baseline_direct["lexical_score"])
                    )
                    if baseline_direct_enabled
                    else 0.0
                )
                baseline_evidence = sorted(
                    (
                        {
                            "rank": int(
                                item.get("initial_rank") or item["rank"]
                            ),
                            "score": clamp01(float(item["dense_score"]))
                            * clamp01(float(item["relation_weight"]))
                            * clamp01(
                                float(item["embedding_semantic_strength"])
                            ),
                        }
                        for item in evidence
                    ),
                    key=lambda item: int(item["rank"]),
                )
                baseline_grounding = aggregate_grounding(
                    (
                        float(item["score"])
                        / (
                            math.log2(int(item["rank"]) + 1)
                            ** float(settings["media_rank_decay_exponent"])
                        )
                        for item in baseline_evidence
                    ),
                    corroboration_weight=float(
                        settings["media_corroboration_weight"]
                    ),
                    limit=int(settings["media_corroboration_limit"]),
                )
                baseline_max_chunk_evidence = max(
                    (
                        float(item["score"])
                        for item in baseline_evidence
                    ),
                    default=0.0,
                )
                baseline_grounding_reliability = noisy_or(
                    (
                        float(
                            baseline_direct.get(
                                "embedding_calibrated_semantic_score",
                                baseline_direct[
                                    "calibrated_semantic_score"
                                ],
                            )
                        ),
                        clamp01(
                            baseline_max_chunk_evidence
                            / max(effective_media_score_threshold, 1e-9)
                        ),
                    )
                )
                baseline_distinctive_component = (
                    clamp01(
                        float(settings["unbound_media_distinctive_boost"])
                        * float(
                            baseline_direct.get(
                                "distinctive_support", 0.0
                            )
                        )
                    )
                    if asset_metadata_only
                    and baseline_direct_enabled
                    else clamp01(
                        float(settings["media_bound_distinctive_boost"])
                        * float(
                            baseline_direct.get(
                                "distinctive_support", 0.0
                            )
                        )
                        * baseline_grounding_reliability
                    )
                    if baseline_direct_enabled
                    else 0.0
                )
                baseline_collection_component = (
                    clamp01(
                        float(settings["unbound_media_collection_boost"])
                        * max(
                            float(
                                baseline_direct.get(
                                    "collection_support", 0.0
                                )
                            ),
                            float(
                                baseline_direct.get(
                                    "collection_membership_support",
                                    0.0,
                                )
                            ),
                            float(
                                baseline_direct.get(
                                    "distinctive_membership_support",
                                    0.0,
                                )
                            ),
                        )
                    )
                    if (
                        asset_metadata_only
                        and collection_intent
                        and baseline_direct_enabled
                    )
                    else 0.0
                )
                baseline_direct_candidate_score = noisy_or(
                    (
                        baseline_semantic_component,
                        baseline_lexical_component,
                        baseline_distinctive_component,
                        baseline_collection_component,
                    )
                )
                baseline_confidence_result = score_media_confidence(
                    raw_direct_relevance=baseline_direct_candidate_score,
                    grounding=baseline_grounding,
                    evidence=(
                        (float(item["score"]), int(item["rank"]))
                        for item in baseline_evidence
                    ),
                    bound_media=not asset_metadata_only,
                    relevance_pivot=effective_media_relevance_pivot,
                    direct_signal_enabled=baseline_direct_enabled,
                    ambiguous_collection=ambiguous_collection,
                    subject_anchor_met=baseline_subject_met,
                    media_format_compatible=media_format_compatible,
                    content_anchor_met=content_anchor_met,
                    evidence_limit=int(
                        settings["media_threshold_evidence_limit"]
                    ),
                    rank_decay_exponent=float(
                        settings["media_threshold_rank_decay_exponent"]
                    ),
                    negative_reliability_exponent=float(
                        settings[
                            "media_threshold_negative_reliability_exponent"
                        ]
                    ),
                    positive_blend=float(
                        settings["media_pivot_positive_blend"]
                    ),
                    negative_weight=float(
                        settings["media_pivot_negative_weight"]
                    ),
                    negative_attenuation_floor=float(
                        settings["media_pivot_negative_attenuation_floor"]
                    ),
                    format_mismatch_factor=float(
                        settings["media_format_mismatch_factor"]
                    ),
                    content_mismatch_factor=float(
                        settings["media_content_mismatch_factor"]
                    ),
                    tail_rank=int(settings["media_corroboration_limit"]),
                )
                baseline_threshold = baseline_confidence_result[
                    "threshold_result"
                ]
                baseline_gated_adjusted_grounding = float(
                    baseline_threshold["adjusted_grounding"]
                )
                baseline_pivot_calibrated_direct_score = float(
                    baseline_confidence_result["pivoted_direct_relevance"]
                )
                baseline_output_confidence = float(
                    baseline_confidence_result["output_confidence"]
                )
            contribution_by_rank = {
                int(item["rank"]): item
                for item in threshold_result["contributions"]
            }
            qualifying = [
                item
                for item in evidence
                if float(item["media_evidence_score"])
                >= effective_media_score_threshold
            ]
            weakening = [
                item
                for item in evidence
                if 0.0
                < float(item["media_evidence_score"])
                < effective_media_score_threshold
            ]
            for item in evidence:
                evidence_score = float(item["media_evidence_score"])
                contribution = contribution_by_rank.get(int(item["rank"]))
                if contribution is not None:
                    item.update(
                        {
                            key: contribution[key]
                            for key in (
                                "pivot_calibrated_evidence_score",
                                "threshold_rank_weight",
                                "threshold_positive_contribution",
                                "threshold_negative_contribution",
                                "threshold_effect",
                            )
                        }
                    )
                else:
                    item.update(
                        {
                            "threshold_rank_weight": None,
                            "threshold_positive_contribution": 0.0,
                            "threshold_negative_contribution": 0.0,
                            "threshold_effect": "outside_window",
                        }
                    )
                item["threshold_margin"] = round(
                    evidence_score - effective_media_score_threshold, 6
                )
            forced = "with_result" in policies and bool(evidence)
            metadata_only = policies == {"metadata_only"}
            confidence_threshold_met = (
                output_confidence >= media_output_confidence_threshold
            )
            policy = (
                "with_result"
                if forced
                else "metadata_only"
                if metadata_only
                else "auto"
            )
            threshold_eligible = (
                not metadata_only
                and confidence_threshold_met
            )
            should_output = forced or threshold_eligible
            for relation in direct_relations:
                relation.pop("_asset_metadata_only", None)
            decision = {
                **candidate,
                "confidence_algorithm": (
                    "unbound_asset_direct"
                    if asset_metadata_only
                    else "bound_chunk_grounded"
                ),
                "confidence_algorithm_version": MEDIA_CONFIDENCE_ALGORITHM,
                "output_policy": policy,
                "associated_chunk_count": len(evidence),
                "qualifying_chunk_count": len(qualifying),
                "weakening_chunk_count": len(weakening),
                "best_rank": min(
                    (int(item["rank"]) for item in evidence), default=None
                ),
                "best_qualifying_rank": min(
                    (int(item["rank"]) for item in qualifying), default=None
                ),
                "visual_intent": bool(intent["detected"]),
                "visual_intent_kind": str(
                    intent.get("intent_kind") or "none"
                ),
                "original_query": query,
                "effective_media_query": media_query,
                "visual_intent_reference_span": str(
                    intent.get("reference_span") or ""
                ),
                "visual_intent_generation_span": str(
                    intent.get("generation_span") or ""
                ),
                "visual_intent_ignored_output_terms": list(
                    intent.get("ignored_output_terms") or []
                ),
                "visual_intent_matched_categories": dict(
                    intent.get("matched_categories") or {}
                ),
                "visual_intent_projection_applied": bool(
                    intent.get("projection_applied")
                ),
                "visual_intent_policy_fingerprint": str(
                    intent.get("policy_fingerprint") or ""
                ),
                "visual_intent_detector_version": str(
                    intent.get("detector_version") or ""
                ),
                "implicit_media_lookup": implicit_media_lookup,
                "subject_anchor_required": subject_anchor_required,
                "subject_anchor_terms": list(
                    intent.get("subject_anchor_terms") or []
                ),
                "subject_anchor_met": subject_anchor_met,
                "media_gate_open": media_gate_open,
                "media_policy_gate_open": bool(
                    confidence_result["policy_gate_open"]
                ),
                "query_media_format_groups": sorted(
                    query_media_format_groups
                ),
                "candidate_media_format_groups": sorted(
                    candidate_media_format_groups
                ),
                "media_format_compatible": media_format_compatible,
                "content_anchor_required": content_anchor_required,
                "content_anchor_terms": query_subject_tokens,
                "content_anchor_matched_terms": sorted(
                    asset_matched_subject_tokens
                ),
                "content_anchor_support": round(
                    asset_content_anchor_support, 6
                ),
                "content_anchor_met": content_anchor_met,
                "matched_tokens": direct["matched_tokens"],
                "matched_media_description": direct.get(
                    "media_description"
                ),
                "matched_media_description_id": direct.get(
                    "media_description_id"
                ),
                "matched_media_description_sort_order": direct.get(
                    "media_description_sort_order"
                ),
                "media_description_count": int(
                    candidate.get("media_description_count") or 1
                ),
                "lexical_coverage": direct["lexical_coverage"],
                "lexical_score": direct["lexical_score"],
                "media_frequency_algorithm": MEDIA_FREQUENCY_ALGORITHM,
                "media_frequency_scope": str(
                    candidate.pop("_media_frequency_scope", "")
                ),
                "media_frequency_corpus_size": int(
                    candidate.pop("_media_frequency_corpus_size", 0)
                ),
                "frequency_weighted_completeness": direct.get(
                    "frequency_weighted_completeness", 0.0
                ),
                "frequency_informativeness": direct.get(
                    "frequency_informativeness", 0.0
                ),
                "frequency_candidate_lexical_score": direct.get(
                    "frequency_candidate_lexical_score", 0.0
                ),
                "frequency_token_details": list(
                    direct.get("frequency_token_details") or []
                ),
                "distinctive_support": round(distinctive_support, 6),
                "distinctive_membership_support": direct.get(
                    "distinctive_membership_support", 0.0
                ),
                "collection_support": round(collection_support, 6),
                "collection_membership_support": round(
                    collection_membership_support, 6
                ),
                "media_vector_similarity": direct["media_vector_similarity"],
                "calibrated_semantic_score": direct[
                    "calibrated_semantic_score"
                ],
                "semantic_component": round(semantic_component, 6),
                "lexical_component": round(lexical_component, 6),
                "distinctive_component": round(
                    distinctive_component, 6
                ),
                "collection_component": round(collection_component, 6),
                "grounding_reliability": round(
                    grounding_reliability, 6
                ),
                "unbound_direct_score": (
                    round(direct_candidate_score, 6)
                    if asset_metadata_only
                    else None
                ),
                "unbound_baseline_direct_score": (
                    round(baseline_direct_candidate_score, 6)
                    if asset_metadata_only
                    else None
                ),
                # Compatibility aliases retained for older dashboards.
                "media_only_direct_score": round(direct_candidate_score, 6),
                "media_only_baseline_direct_score": round(
                    baseline_direct_candidate_score, 6
                ),
                "raw_direct_relevance": round(direct_candidate_score, 6),
                "pivot_calibrated_direct_relevance": round(
                    pivot_calibrated_direct_score, 6
                ),
                "baseline_pivot_calibrated_direct_relevance": round(
                    baseline_pivot_calibrated_direct_score, 6
                ),
                "raw_relevance_score": round(association_score, 6),
                "pivot_calibrated_relevance_score": round(
                    float(confidence_result["pivoted_combined_confidence"]),
                    6,
                ),
                "structural_attenuation_factor": round(
                    structural_attenuation_factor, 6
                ),
                "format_attenuation_factor": round(
                    format_attenuation_factor, 6
                ),
                "content_attenuation_factor": round(
                    content_attenuation_factor, 6
                ),
                "media_only_grounding_direct_exponent": None,
                "media_only_grounding_gate": 1.0,
                "media_only_gated_grounding_score": round(
                    adjusted_grounding, 6
                ),
                "media_only_baseline_grounding_gate": 1.0,
                "media_only_baseline_gated_grounding_score": round(
                    baseline_gated_adjusted_grounding, 6
                ),
                "grounding_score": round(grounding, 6),
                "reinforced_grounding_score": round(
                    float(threshold_result["reinforced_grounding"]), 6
                ),
                "adjusted_grounding_score": round(adjusted_grounding, 6),
                "association_score": round(association_score, 6),
                "pre_threshold_output_confidence": round(
                    pre_threshold_confidence, 6
                ),
                "evidence_threshold_adjustment": round(
                    float(threshold_result["grounding_delta"]), 6
                ),
                "output_confidence": round(output_confidence, 6),
                "output_confidence_delta": round(
                    output_confidence - pre_threshold_confidence, 6
                ),
                "rerank_baseline_output_confidence": round(
                    baseline_output_confidence, 6
                ),
                "rerank_output_confidence_delta": round(
                    output_confidence - baseline_output_confidence, 6
                ),
                "rerank_raw_score": (
                    direct.get("rerank_raw_score")
                    if use_rerank
                    else None
                ),
                "rerank_fused_score": (
                    direct.get("calibrated_semantic_score")
                    if use_rerank and direct.get("rerank_raw_score") is not None
                    else None
                ),
                "positive_support": round(
                    float(threshold_result["positive_support"]), 6
                ),
                "negative_pressure": round(
                    float(threshold_result["negative_pressure"]), 6
                ),
                "negative_attenuation_factor": round(
                    float(
                        threshold_result["negative_attenuation_factor"]
                    ),
                    6,
                ),
                "tail_negative_pressure": round(
                    float(threshold_result["tail_negative_pressure"]), 6
                ),
                "threshold_normalization_constant": round(
                    float(threshold_result["normalization_constant"]), 6
                ),
                "threshold_evidence_limit": int(
                    threshold_result["evidence_limit"]
                ),
                "threshold_evidence_count": int(
                    threshold_result["evidence_count"]
                ),
                "media_relevance_pivot": effective_media_relevance_pivot,
                "effective_media_relevance_pivot": (
                    effective_media_relevance_pivot
                ),
                "requested_media_relevance_pivot": (
                    requested_media_relevance_pivot
                ),
                "media_relevance_pivot_fallback": float(
                    settings["media_relevance_pivot_fallback"]
                ),
                "media_relevance_pivot_source": (
                    media_relevance_pivot_source
                ),
                "media_relevance_pivot_request_field": (
                    media_relevance_pivot_request_field
                ),
                # Deprecated response projections remain populated for old
                # dashboards, including unbound assets now that the pivot is
                # universal rather than chunk-only.
                "media_score_threshold": effective_media_relevance_pivot,
                "media_score_threshold_applicable": True,
                "requested_media_score_threshold": (
                    requested_media_relevance_pivot
                ),
                "media_score_threshold_fallback": float(
                    settings["media_relevance_pivot_fallback"]
                ),
                "media_score_threshold_source": (
                    media_relevance_pivot_source
                ),
                "output_confidence_threshold": (
                    media_output_confidence_threshold
                ),
                "confidence_threshold_met": confidence_threshold_met,
                "max_evidence_score": round(max_chunk_evidence, 6),
                "calibration_strength_source": (
                    "asset_metadata_direct"
                    if asset_metadata_only
                    else "rerank_calibration"
                    if any(
                        item.get("semantic_strength_source")
                        == "rerank_calibration"
                        for item in evidence
                    )
                    else "embedding_baseline"
                ),
                "rerank_calibration_stale": bool(
                    use_rerank
                    and any(
                        item.get("rerank_semantic_strength") is not None
                        and not item.get("rerank_calibration_current")
                        for item in evidence
                    )
                ),
                "output": False,
                "eligible": should_output,
                "forced": forced,
                "reason": (
                    "metadata_only"
                    if metadata_only and not forced
                    else "media_metadata_confidence_below_threshold"
                    if (
                        asset_metadata_only
                        and not confidence_threshold_met
                        and not forced
                    )
                    else "confidence_below_threshold"
                    if not confidence_threshold_met and not forced
                    else "eligible"
                ),
                "direct_relations": direct_relations,
                "evidence": evidence,
            }
            decisions.append(decision)
            if should_output:
                eligible.append(decision)

        unbound_decisions = [
            item for item in decisions if bool(item.get("asset_metadata_only"))
        ]
        if unbound_decisions:
            # Competition is a property of unbound asset descriptions, not a
            # retrieval-mode modifier. Bound media retain their chunk-grounded
            # confidence and never enter this comparison space.
            best_direct_score = max(
                float(item["media_only_direct_score"])
                for item in unbound_decisions
            )
            best_baseline_direct_score = max(
                float(item["media_only_baseline_direct_score"])
                for item in unbound_decisions
            )
            best_collection_membership_support = max(
                float(
                    item.get("collection_membership_support") or 0.0
                )
                for item in unbound_decisions
            )
            eligible = [
                item
                for item in eligible
                if not bool(item.get("asset_metadata_only"))
            ]
            for decision in unbound_decisions:
                strongest_competitor_score = max(
                    (
                        float(item["media_only_direct_score"])
                        for item in unbound_decisions
                        if item is not decision
                    ),
                    default=0.0,
                )
                strongest_baseline_competitor_score = max(
                    (
                        float(item["media_only_baseline_direct_score"])
                        for item in unbound_decisions
                        if item is not decision
                    ),
                    default=0.0,
                )
                adjustment = unbound_media_confidence_factor(
                    float(decision["media_only_direct_score"]),
                    best_direct_score,
                    strongest_competitor_score=strongest_competitor_score,
                    collection_membership_support=(
                        float(
                            decision.get("collection_membership_support")
                            or 0.0
                        )
                        / best_collection_membership_support
                        if best_collection_membership_support > 0.0
                        else 0.0
                    ),
                    collection_intent=collection_intent,
                    competition_floor=float(
                        settings["unbound_media_competition_floor"]
                    ),
                    reliability_target=float(
                        settings["unbound_media_reliability_target"]
                    ),
                    specificity_exponent=float(
                        settings["unbound_media_specificity_exponent"]
                    ),
                    advantage_target=float(
                        settings["unbound_media_advantage_target"]
                    ),
                )
                baseline_adjustment = unbound_media_confidence_factor(
                    float(decision["media_only_baseline_direct_score"]),
                    best_baseline_direct_score,
                    strongest_competitor_score=(
                        strongest_baseline_competitor_score
                    ),
                    collection_membership_support=(
                        float(
                            decision.get("collection_membership_support")
                            or 0.0
                        )
                        / best_collection_membership_support
                        if best_collection_membership_support > 0.0
                        else 0.0
                    ),
                    collection_intent=collection_intent,
                    competition_floor=float(
                        settings["unbound_media_competition_floor"]
                    ),
                    reliability_target=float(
                        settings["unbound_media_reliability_target"]
                    ),
                    specificity_exponent=float(
                        settings["unbound_media_specificity_exponent"]
                    ),
                    advantage_target=float(
                        settings["unbound_media_advantage_target"]
                    ),
                )
                pre_competition_confidence = float(
                    decision["output_confidence"]
                )
                pre_competition_association = float(decision["association_score"])
                pre_competition_baseline = float(
                    decision["rerank_baseline_output_confidence"]
                )
                output_confidence = clamp01(
                    pre_competition_confidence
                    * float(adjustment["confidence_factor"])
                )
                baseline_output_confidence = clamp01(
                    pre_competition_baseline
                    * float(baseline_adjustment["confidence_factor"])
                )
                # Association is the direct, pre-competition relevance D.
                association_score = pre_competition_association
                confidence_threshold_met = (
                    output_confidence >= media_output_confidence_threshold
                )
                forced = bool(decision["forced"])
                metadata_only_policy = decision["output_policy"] == "metadata_only"
                should_output = forced or (
                    not metadata_only_policy and confidence_threshold_met
                )
                decision.update(
                    {
                        "media_only_specificity_algorithm": (
                            UNBOUND_MEDIA_SPECIFICITY_ALGORITHM
                        ),
                        "unbound_specificity_algorithm": (
                            UNBOUND_MEDIA_SPECIFICITY_ALGORITHM
                        ),
                        "unbound_collection_intent": collection_intent,
                        "unbound_best_direct_score": round(
                            best_direct_score, 6
                        ),
                        "unbound_pre_competition_output_confidence": round(
                            pre_competition_confidence, 6
                        ),
                        "unbound_pre_competition_association_score": round(
                            pre_competition_association, 6
                        ),
                        "unbound_direct_reliability": round(
                            float(adjustment["direct_reliability"]), 6
                        ),
                        "unbound_relative_specificity": round(
                            float(adjustment["relative_specificity"]), 6
                        ),
                        "unbound_strongest_competitor_score": round(
                            float(adjustment["strongest_competitor_score"]), 6
                        ),
                        "unbound_advantage_margin": round(
                            float(adjustment["advantage_margin"]), 6
                        ),
                        "unbound_advantage_weight": round(
                            float(adjustment["advantage_weight"]), 6
                        ),
                        "unbound_collection_membership_weight": round(
                            float(
                                adjustment["collection_membership_weight"]
                            ),
                            6,
                        ),
                        "unbound_specificity_weight": round(
                            float(adjustment["specificity_weight"]), 6
                        ),
                        "unbound_winner_support": round(
                            float(adjustment["winner_support"]), 6
                        ),
                        "unbound_competition_floor": round(
                            float(adjustment["competition_floor"]), 6
                        ),
                        "unbound_confidence_factor": round(
                            float(adjustment["confidence_factor"]), 6
                        ),
                        "unbound_reliability_target": round(
                            float(settings["unbound_media_reliability_target"]),
                            6,
                        ),
                        "unbound_specificity_exponent": round(
                            float(
                                settings[
                                    "unbound_media_specificity_exponent"
                                ]
                            ),
                            6,
                        ),
                        "unbound_advantage_target": round(
                            float(settings["unbound_media_advantage_target"]),
                            6,
                        ),
                        # Legacy names remain response aliases for clients that
                        # predate structure-aware confidence selection.
                        "media_only_collection_intent": collection_intent,
                        "media_only_best_direct_score": round(
                            best_direct_score, 6
                        ),
                        "media_only_pre_competition_output_confidence": round(
                            pre_competition_confidence, 6
                        ),
                        "media_only_pre_competition_association_score": round(
                            pre_competition_association, 6
                        ),
                        "media_only_direct_reliability": round(
                            float(adjustment["direct_reliability"]), 6
                        ),
                        "media_only_relative_specificity": round(
                            float(adjustment["relative_specificity"]), 6
                        ),
                        "media_only_strongest_competitor_score": round(
                            float(
                                adjustment["strongest_competitor_score"]
                            ),
                            6,
                        ),
                        "media_only_advantage_margin": round(
                            float(adjustment["advantage_margin"]), 6
                        ),
                        "media_only_advantage_weight": round(
                            float(adjustment["advantage_weight"]), 6
                        ),
                        "media_only_collection_membership_weight": round(
                            float(
                                adjustment[
                                    "collection_membership_weight"
                                ]
                            ),
                            6,
                        ),
                        "media_only_specificity_weight": round(
                            float(adjustment["specificity_weight"]), 6
                        ),
                        "media_only_winner_support": round(
                            float(adjustment["winner_support"]), 6
                        ),
                        "media_only_competition_floor": round(
                            float(adjustment["competition_floor"]), 6
                        ),
                        "media_only_confidence_factor": round(
                            float(adjustment["confidence_factor"]), 6
                        ),
                        "association_score": round(association_score, 6),
                        "output_confidence": round(output_confidence, 6),
                        "output_confidence_delta": round(
                            output_confidence
                            - float(decision["pre_threshold_output_confidence"]),
                            6,
                        ),
                        "rerank_baseline_output_confidence": round(
                            baseline_output_confidence, 6
                        ),
                        "rerank_output_confidence_delta": round(
                            output_confidence - baseline_output_confidence, 6
                        ),
                        "confidence_threshold_met": confidence_threshold_met,
                        "eligible": should_output,
                        "reason": (
                            "metadata_only"
                            if metadata_only_policy and not forced
                            else "media_metadata_confidence_below_threshold"
                            if (
                                bool(decision.get("asset_metadata_only"))
                                and not confidence_threshold_met
                                and not forced
                            )
                            else "confidence_below_threshold"
                            if not confidence_threshold_met and not forced
                            else "eligible"
                        ),
                    }
                )
                if should_output:
                    eligible.append(decision)

        eligible.sort(
            key=lambda item: (
                not bool(item["forced"]),
                -float(item["output_confidence"]),
                int(
                    item["best_rank"]
                    or item.get("asset_candidate_rank")
                    or 10**9
                ),
                str(item["asset_id"]),
            )
        )
        selected_asset_ids = {
            str(item["asset_id"]) for item in eligible[:max_media_outputs]
        }
        for decision in decisions:
            if str(decision["asset_id"]) in selected_asset_ids:
                decision["output"] = True
                decision["reason"] = (
                    "forced_with_result"
                    if decision["forced"]
                    else "confidence_met"
                )
            elif decision["eligible"]:
                decision["reason"] = "output_limit"

        decisions.sort(
            key=lambda item: (
                not bool(item["output"]),
                not bool(item["forced"]),
                -float(item["output_confidence"]),
                int(
                    item["best_rank"]
                    or item.get("asset_candidate_rank")
                    or 10**9
                ),
                -float(item["max_evidence_score"]),
                int(item["sort_order"]),
                str(item["asset_id"]),
            )
        )
        outputs = [
            {
                key: value
                for key, value in item.items()
                if key not in {"evidence", "direct_relations", "eligible", "forced"}
            }
            for item in decisions
            if item["output"]
        ]

        items: list[dict[str, Any]] = []
        for chunk_id in selected:
            rank = text_rank_by_chunk[chunk_id]
            row = rows[chunk_id]
            related_media = []
            for decision in decisions:
                matching = next(
                    (
                        item
                        for item in decision["evidence"]
                        if int(item["chunk_id"]) == chunk_id
                    ),
                    None,
                )
                if matching is not None:
                    related_media.append(
                        {
                            "asset_id": decision["asset_id"],
                            "original_name": decision["original_name"],
                            "media_description": decision["media_description"],
                            "output_policy": decision["output_policy"],
                            **matching,
                        }
                    )
            items.append(
                {
                    "rank": rank,
                    "entry_id": row["entry_id"],
                    "document_id": row["document_id"],
                    "chunk_id": chunk_id,
                    "title": row["entry_title"],
                    "text": row["text"],
                    "score": round(
                        float(
                            scores[chunk_id].get(
                                "ordering_relevance",
                                scores[chunk_id]["embedding_signal"],
                            )
                        ),
                        6,
                    ),
                    "initial_rank": text_initial_rank_by_chunk[chunk_id],
                    "initial_score": round(
                        float(scores[chunk_id]["embedding_relevance"]), 6
                    ),
                    "embedding_relevance": round(
                        float(scores[chunk_id]["embedding_relevance"]), 6
                    ),
                    "lexical_relevance": round(
                        float(scores[chunk_id]["lexical_relevance"]), 6
                    ),
                    "rrf_score": round(float(scores[chunk_id]["rrf"]), 6),
                    "rrf_rank": int(scores[chunk_id]["rrf_rank"]),
                    "rerank_raw_score": (
                        round(float(scores[chunk_id]["rerank_raw_score"]), 6)
                        if "rerank_raw_score" in scores[chunk_id]
                        else None
                    ),
                    "rerank_normalized_score": (
                        round(
                            float(scores[chunk_id]["rerank_probability"]), 6
                        )
                        if "rerank_probability" in scores[chunk_id]
                        else None
                    ),
                    "rerank_rank": scores[chunk_id].get("rerank_rank"),
                    "rerank_reorder_eligible": bool(
                        use_rerank
                        and "rerank_probability" in scores[chunk_id]
                        and rank != text_initial_rank_by_chunk[chunk_id]
                    ),
                    "fused_score": round(
                        float(
                            scores[chunk_id].get(
                                "fused_relevance",
                                scores[chunk_id]["embedding_signal"],
                            )
                        ),
                        6,
                    ),
                    "dense_score": round(
                        clamp01(float(scores[chunk_id]["dense"])), 6
                    ),
                    "score_breakdown": {
                        key: round(value, 6)
                        for key, value in scores[chunk_id].items()
                        if isinstance(value, (int, float))
                    },
                    "retrieval_stage": (
                        "reranked" if use_rerank else "embedding_baseline"
                    ),
                    "score_source": (
                        "rerank_fused_relevance"
                        if use_rerank
                        else "embedding_relevance"
                    ),
                    "associated_media": related_media,
                }
            )
        baseline_items = items
        if use_rerank:
            baseline_items = []
            for rank, chunk_id in enumerate(baseline_selected, start=1):
                row = rows[chunk_id]
                score_details = scores[chunk_id]
                embedding_score = float(score_details["embedding_relevance"])
                baseline_items.append(
                    {
                        "rank": rank,
                        "entry_id": row["entry_id"],
                        "document_id": row["document_id"],
                        "chunk_id": chunk_id,
                        "title": row["entry_title"],
                        "text": row["text"],
                        "score": round(embedding_score, 6),
                        "initial_rank": text_initial_rank_by_chunk[chunk_id],
                        "initial_score": round(embedding_score, 6),
                        "embedding_relevance": round(embedding_score, 6),
                        "lexical_relevance": round(
                            float(score_details["lexical_relevance"]), 6
                        ),
                        "rrf_score": round(
                            float(score_details["rrf"]), 6
                        ),
                        "rrf_rank": int(score_details["rrf_rank"]),
                        "rerank_raw_score": None,
                        "rerank_normalized_score": None,
                        "rerank_rank": None,
                        "rerank_reorder_eligible": False,
                        "fused_score": round(embedding_score, 6),
                        "dense_score": round(
                            clamp01(float(score_details["dense"])), 6
                        ),
                        "score_breakdown": {
                            key: round(float(score_details[key]), 6)
                            for key in (
                                "dense",
                                "lexical",
                                "rrf",
                                "rrf_rank",
                                "dense_relevance",
                                "lexical_relevance",
                                "embedding_relevance",
                                "embedding_signal",
                            )
                            if key in score_details
                        },
                        "retrieval_stage": "embedding_baseline",
                        "score_source": "embedding_relevance",
                        "associated_media": [],
                    }
                )
        return {
            "query": query,
            "media_query": media_query,
            "media_query_projection_applied": bool(
                intent.get("projection_applied")
            ),
            "query_embedding_count": len(embedding_inputs),
            "query_embedding": {
                "input_count": int(query_embedding_meta.get("input_count") or 0),
                "request_count": int(
                    query_embedding_meta.get("request_count") or 0
                ),
                "cache_hits": int(
                    query_embedding_meta.get("cache_hits") or 0
                ),
                "provider_input_count": int(
                    query_embedding_meta.get("provider_input_count") or 0
                ),
                "deduplicated": len(embedding_inputs) < int(
                    text_channel_enabled
                ) + int(media_channel_enabled),
            },
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "max_media_outputs": max_media_outputs,
            "generation": self.indexes.status()["generation"],
            "media_generation": self.indexes.status()["media_generation"],
            "visual_intent": intent,
            "query_tokens": query_tokens,
            "media_frequency": {
                "algorithm": MEDIA_FREQUENCY_ALGORITHM,
                "scope": "structure_isolated",
                "bound": {
                    "scope": str(
                        bound_media_frequency_context.get("scope") or ""
                    ),
                    "corpus_size": int(
                        bound_media_frequency_context.get("corpus_size") or 0
                    ),
                    "query_document_frequencies": dict(
                        bound_media_frequency_context.get(
                            "document_frequencies"
                        )
                        or {}
                    ),
                    "lexical_candidate_count": len(
                        dict(
                            bound_media_frequency_context.get("asset_matches")
                            or {}
                        )
                    ),
                },
                "unbound": {
                    "scope": str(
                        unbound_media_frequency_context.get("scope") or ""
                    ),
                    "corpus_size": int(
                        unbound_media_frequency_context.get("corpus_size") or 0
                    ),
                    "query_document_frequencies": dict(
                        unbound_media_frequency_context.get(
                            "document_frequencies"
                        )
                        or {}
                    ),
                    "lexical_candidate_count": len(
                        dict(
                            unbound_media_frequency_context.get("asset_matches")
                            or {}
                        )
                    ),
                },
                "descriptor_rescue_count": len(
                    descriptor_rescue_asset_ids
                ),
            },
            "retrieval_settings": settings,
            "text_scoring": {
                "algorithm": TEXT_RELEVANCE_ALGORITHM,
                "candidate_fusion": "faiss_fts5_rrf",
                "candidate_ordering": "embedding_relevance_desc",
                "rrf_is_public_score": False,
                "lexical_query_operator": "OR",
            },
            "rerank": {
                "requested": rerank_requested,
                "provider_available": self._rerank_binding_available(meta),
                "effective_enabled": use_rerank,
                "applied": use_rerank,
                "failed": False,
                "fallback": False,
                "discarded_partial_results": False,
                "fallback_reason": None,
                "provider_id": str(meta.get("rerank_provider_id") or ""),
                "provider_revision": int(
                    meta.get("rerank_provider_revision") or 0
                ),
                "provider_fingerprint": str(
                    meta.get("rerank_provider_fingerprint") or ""
                ),
                "candidate_limit": rerank_candidate_limit,
                "fusion_algorithm": RERANK_FUSION_ALGORITHM,
                "text_reorder_probability_floor": (
                    MINIMUM_RERANK_REORDER_PROBABILITY
                ),
                "text_ordering": (
                    "rerank_fused_relevance_desc"
                    if use_rerank
                    else "embedding_relevance_desc"
                ),
                "media_ordering": (
                    "ordering_relevance_per_asset"
                    if use_rerank
                    else "embedding_baseline_per_asset"
                ),
                "scopes": rerank_scope_meta,
                "cache_hits": sum(
                    int(item["cache_hits"]) for item in rerank_scope_meta
                ),
                "provider_candidates": sum(
                    int(item["provider_candidates"])
                    for item in rerank_scope_meta
                ),
                "elapsed_ms": round(
                    sum(float(item["elapsed_ms"]) for item in rerank_scope_meta),
                    3,
                ),
            },
            "thresholds": {
                "media_output_confidence_threshold": (
                    media_output_confidence_threshold
                ),
                "media_relevance_pivot": effective_media_relevance_pivot,
                "effective_media_relevance_pivot": (
                    effective_media_relevance_pivot
                ),
                "requested_media_relevance_pivot": (
                    requested_media_relevance_pivot
                ),
                "media_relevance_pivot_fallback": float(
                    settings["media_relevance_pivot_fallback"]
                ),
                "media_relevance_pivot_source": (
                    media_relevance_pivot_source
                ),
                "media_relevance_pivot_request_field": (
                    media_relevance_pivot_request_field
                ),
                # Deprecated aliases for older clients.
                "media_score_threshold": effective_media_relevance_pivot,
                "effective_media_score_threshold": (
                    effective_media_relevance_pivot
                ),
                "requested_media_score_threshold": (
                    requested_media_relevance_pivot
                ),
                "media_score_threshold_fallback": float(
                    settings["media_relevance_pivot_fallback"]
                ),
                "media_score_threshold_source": (
                    media_relevance_pivot_source
                ),
                "media_threshold_evidence_limit": int(
                    settings["media_threshold_evidence_limit"]
                ),
                "media_threshold_normalization_constant": round(
                    sum(
                        1.0
                        / (
                            math.log2(rank + 1)
                            ** float(
                                settings[
                                    "media_threshold_rank_decay_exponent"
                                ]
                            )
                        )
                        for rank in range(
                            1,
                            int(settings["media_threshold_evidence_limit"])
                            + 1,
                        )
                    ),
                    6,
                ),
                "media_candidates_independent_of_top_k": True,
                "media_pivot_applies_to_bound_and_unbound": True,
                "media_confidence_algorithm": MEDIA_CONFIDENCE_ALGORITHM,
                "media_pivot_positive_blend": float(
                    settings["media_pivot_positive_blend"]
                ),
                "media_pivot_negative_weight": float(
                    settings["media_pivot_negative_weight"]
                ),
                "media_pivot_negative_attenuation_floor": float(
                    settings["media_pivot_negative_attenuation_floor"]
                ),
                "media_format_mismatch_factor": float(
                    settings["media_format_mismatch_factor"]
                ),
                "media_content_mismatch_factor": float(
                    settings["media_content_mismatch_factor"]
                ),
                "media_evidence_scope": "asset_bound_only",
                "media_ranking_scope": "per_asset",
                "media_confidence_algorithms": [
                    "bound_chunk_grounded",
                    "unbound_asset_direct",
                ],
            },
            "execution": {
                "text_retrieval_executed": text_channel_enabled,
                "text_retrieval_skip_reason": (
                    None
                    if text_channel_enabled
                    else "media_only_mode"
                ),
                "media_channel_executed": media_channel_enabled,
                "media_channel_skip_reason": (
                    None if media_channel_enabled else "text_only_mode"
                ),
                "bound_media_executed": (
                    media_channel_enabled
                    and any(
                        not bool(item.get("asset_metadata_only"))
                        for item in decisions
                    )
                ),
                "bound_media_skip_reason": (
                    None
                    if media_channel_enabled
                    and any(
                        not bool(item.get("asset_metadata_only"))
                        for item in decisions
                    )
                    else standard_media_skipped_reason
                    or "no_bound_media_candidate"
                ),
                "unbound_media_executed": (
                    media_channel_enabled
                    and any(
                        bool(item.get("asset_metadata_only"))
                        for item in decisions
                    )
                ),
                "unbound_media_skip_reason": (
                    None
                    if media_channel_enabled
                    and any(
                        bool(item.get("asset_metadata_only"))
                        for item in decisions
                    )
                    else "text_only_mode"
                    if not media_channel_enabled
                    else "no_unbound_media_candidate"
                ),
                "standard_media_probe_count": standard_media_probe_count,
                "asset_index_candidate_count": len(asset_metadata_rows),
                "media_frequency_scope": "structure_isolated",
                "bound_media_frequency_scope": str(
                    bound_media_frequency_context.get("scope") or ""
                ),
                "bound_media_frequency_corpus_size": int(
                    bound_media_frequency_context.get("corpus_size") or 0
                ),
                "unbound_media_frequency_scope": str(
                    unbound_media_frequency_context.get("scope") or ""
                ),
                "unbound_media_frequency_corpus_size": int(
                    unbound_media_frequency_context.get("corpus_size") or 0
                ),
                "media_frequency_lexical_candidate_count": (
                    len(
                        dict(
                            bound_media_frequency_context.get("asset_matches")
                            or {}
                        )
                    )
                    + len(
                        dict(
                            unbound_media_frequency_context.get("asset_matches")
                            or {}
                        )
                    )
                ),
                "descriptor_rescue_candidate_count": len(
                    descriptor_rescue_asset_ids
                ),
                "bound_media_candidate_count": sum(
                    not bool(item.get("asset_metadata_only"))
                    for item in decisions
                ),
                "unbound_media_candidate_count": sum(
                    bool(item.get("asset_metadata_only"))
                    for item in decisions
                ),
                "media_only_collection_intent": (
                    collection_intent if media_channel_enabled else None
                ),
                "media_only_specificity_algorithm": (
                    UNBOUND_MEDIA_SPECIFICITY_ALGORITHM
                    if media_channel_enabled
                    else None
                ),
                "unbound_media_specificity_algorithm": (
                    UNBOUND_MEDIA_SPECIFICITY_ALGORITHM
                    if media_channel_enabled
                    else None
                ),
            },
            "baseline_items": baseline_items,
            "items": items,
            "media_outputs": outputs,
            "media_decisions": decisions,
        }
