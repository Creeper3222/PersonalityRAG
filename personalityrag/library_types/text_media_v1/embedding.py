from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from ...context_lengths import MANUAL_CONTEXT_FALLBACK_TOKENS
from ...providers import EmbeddingProvider


CONTEXT_CHUNK_CHAR_SAFETY_RATIO = 0.75
MIN_CONTEXT_TOKENS = 128
MIN_FRAGMENT_CHARS = 64
EMBEDDING_CHUNKING_POLICY = "stable_char_chunk_normalized_mean_pool_v1"


async def _embedding_context_capability(
    provider: EmbeddingProvider,
) -> dict[str, Any]:
    config = getattr(provider, "config", None)
    max_context_tokens = int(
        getattr(config, "max_context_tokens", 0) or 0
    )
    source = str(
        getattr(config, "max_context_tokens_source", "") or ""
    )
    mode = str(getattr(config, "context_length_mode", "auto") or "auto")
    if mode == "manual" and max_context_tokens < MIN_CONTEXT_TOKENS:
        raise ValueError(
            f"manual max_context_tokens must be at least {MIN_CONTEXT_TOKENS}"
        )
    if max_context_tokens < MIN_CONTEXT_TOKENS:
        detector = getattr(provider, "detect_context_length", None)
        if callable(detector):
            try:
                detected = await detector()
            except Exception:
                detected = {}
            detected_tokens = int(detected.get("max_context_tokens") or 0)
            if detected_tokens >= MIN_CONTEXT_TOKENS:
                max_context_tokens = detected_tokens
                source = str(
                    detected.get("max_context_tokens_source") or "provider_probe"
                )
                mode = "auto"
        if max_context_tokens < MIN_CONTEXT_TOKENS:
            max_context_tokens = MANUAL_CONTEXT_FALLBACK_TOKENS
            source = "manual:fallback-undetected"
            mode = "manual"
        if config is not None:
            config.context_length_mode = mode
            config.max_context_tokens = max_context_tokens
            config.max_context_tokens_source = source
    char_limit = (
        max(
            MIN_FRAGMENT_CHARS,
            int(max_context_tokens * CONTEXT_CHUNK_CHAR_SAFETY_RATIO),
        )
        if max_context_tokens >= MIN_CONTEXT_TOKENS
        else 0
    )
    return {
        "max_context_tokens": max_context_tokens,
        "max_context_tokens_source": source,
        "context_length_mode": mode,
        "chunk_char_limit": char_limit,
        "chunk_char_safety_ratio": CONTEXT_CHUNK_CHAR_SAFETY_RATIO,
        "chunking_policy": EMBEDDING_CHUNKING_POLICY,
    }


def _split_text(text: str, char_limit: int) -> list[str]:
    if char_limit <= 0 or len(text) <= char_limit:
        return [text]
    return [
        text[start : start + char_limit]
        for start in range(0, len(text), char_limit)
    ]


def _validate_vector(vector: Any, *, dimension: int | None) -> np.ndarray:
    row = np.asarray(vector, dtype=np.float32)
    if row.ndim != 1 or not row.size or not np.isfinite(row).all():
        raise ValueError("Embedding Provider returned an invalid vector")
    if dimension is not None and row.size != dimension:
        raise ValueError("Embedding Provider returned inconsistent dimensions")
    if float(np.linalg.norm(row)) <= 0:
        raise ValueError("Embedding Provider returned a zero vector")
    return row


async def embed_texts_context_safe(
    provider: EmbeddingProvider,
    texts: list[str],
    *,
    request_batch_size: int | None = None,
    concurrency: int = 1,
    max_retries: int = 1,
    retry_base_delay: float = 0.25,
    progress: Callable[[int, int], Awaitable[None]] | None = None,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Embed text without exceeding the bound Provider's context window.

    Stored source text is never rewritten. Oversized inputs are split with a
    deterministic character boundary, fragment vectors are normalized, then
    mean-pooled and normalized again into one vector for the original input.
    Inputs that fit in one request preserve the Provider vector byte-for-byte.
    """
    if not texts:
        return [], {
            "input_count": 0,
            "fragment_count": 0,
            "split_input_count": 0,
            "max_fragments_per_input": 0,
            "request_count": 0,
            **await _embedding_context_capability(provider),
        }

    capability = await _embedding_context_capability(provider)
    char_limit = int(capability["chunk_char_limit"] or 0)
    fragments: list[str] = []
    owners: list[int] = []
    owner_fragment_counts: list[int] = []
    for owner, value in enumerate(texts):
        parts = _split_text(str(value or ""), char_limit)
        owner_fragment_counts.append(len(parts))
        fragments.extend(parts)
        owners.extend([owner] * len(parts))

    config = getattr(provider, "config", None)
    batch_size = max(
        1,
        int(
            request_batch_size
            or getattr(config, "batch_size", 0)
            or len(fragments)
        ),
    )
    batches = [
        (start, fragments[start : start + batch_size])
        for start in range(0, len(fragments), batch_size)
    ]
    raw_results: list[list[float] | None] = [None] * len(fragments)
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    completed = 0
    progress_lock = asyncio.Lock()

    async def embed_batch(start: int, batch: list[str]) -> None:
        nonlocal completed
        last_error: Exception | None = None
        async with semaphore:
            for attempt in range(max(1, int(max_retries))):
                try:
                    result = await provider.get_embeddings(batch)
                    if len(result) != len(batch):
                        raise ValueError(
                            "Embedding Provider returned an incomplete batch"
                        )
                    raw_results[start : start + len(batch)] = result
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 >= max(1, int(max_retries)):
                        break
                    delay = max(0.0, float(retry_base_delay)) * (2**attempt)
                    if delay:
                        await asyncio.sleep(delay)
            if last_error is not None:
                raise last_error
            async with progress_lock:
                completed += 1
                if progress is not None:
                    await progress(completed, len(batches))

    await asyncio.gather(*(embed_batch(start, batch) for start, batch in batches))
    if any(vector is None for vector in raw_results):
        raise ValueError("Embedding Provider returned incomplete fragment results")

    dimension: int | None = None
    owner_vectors: list[list[np.ndarray]] = [[] for _ in texts]
    for owner, vector in zip(owners, raw_results, strict=True):
        row = _validate_vector(vector, dimension=dimension)
        dimension = int(row.size) if dimension is None else dimension
        owner_vectors[owner].append(row)

    aggregated: list[list[float]] = []
    for rows in owner_vectors:
        if len(rows) == 1:
            # Preserve the established single-input embedding baseline.
            aggregated.append([float(value) for value in rows[0]])
            continue
        normalized_rows = []
        for row in rows:
            normalized_rows.append(row / float(np.linalg.norm(row)))
        pooled = np.mean(
            np.vstack(normalized_rows).astype(np.float32, copy=False),
            axis=0,
            dtype=np.float32,
        )
        pooled_norm = float(np.linalg.norm(pooled))
        if not math.isfinite(pooled_norm) or pooled_norm <= 0:
            raise ValueError("Embedding fragment aggregation produced a zero vector")
        aggregated.append(
            [float(value) for value in np.asarray(pooled / pooled_norm, dtype=np.float32)]
        )

    return aggregated, {
        "input_count": len(texts),
        "fragment_count": len(fragments),
        "split_input_count": sum(count > 1 for count in owner_fragment_counts),
        "max_fragments_per_input": max(owner_fragment_counts, default=0),
        "request_count": len(batches),
        "dimensions": int(dimension or 0),
        **capability,
    }


async def embed_one_context_safe(
    provider: EmbeddingProvider,
    text: str,
) -> list[float]:
    vectors, _ = await embed_texts_context_safe(provider, [text])
    return vectors[0]
