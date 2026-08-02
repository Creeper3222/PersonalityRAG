from __future__ import annotations

import asyncio
import io
import json
import sqlite3
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pyzipper
from PIL import Image
from fastapi import Request
from httpx import ASGITransport, AsyncClient

from personalityrag.application import create_app
from personalityrag.application_context import ApplicationContext
from personalityrag.backup_migration import export_prag_package, import_prag_package
from personalityrag.config import AppConfig, ProviderConfig
from personalityrag.database_types import (
    TEXT_MEDIA_V1_TYPE,
    DatabaseRef,
    database_type_registry,
)
from personalityrag.library_types.text_media_v1.images import (
    MAX_CANONICAL_BYTES,
    ImageValidationError,
    normalize_image_bytes,
)
from personalityrag.library_types.text_media_v1.batch_package import (
    install_tmkbs_atomic,
    inspect_tmkbs,
    write_tmkbs,
)
from personalityrag.library_types.text_media_v1.embedding import (
    CONTEXT_CHUNK_CHAR_SAFETY_RATIO,
    embed_texts_context_safe,
)
from personalityrag.library_types.text_media_v1 import api as text_media_api
from personalityrag.library_types.text_media_v1.api import (
    IngestImageMapping,
    IngestBatchManifest,
    MediaDescriptionsUpdateRequest,
    SearchRequest,
    UpdateRequest,
    VisualIntentPolicyRequest,
)
from personalityrag.library_types.text_media_v1 import document_parsers
from personalityrag.library_types.text_media_v1.document_parsers import (
    parse_document_bytes,
)
from personalityrag.library_types.text_media_v1.indexes import TextMediaIndex
from personalityrag.library_types.text_media_v1.package import (
    TmkbPackageError,
    export_tmkb,
    inspect_tmkb,
    install_tmkb,
)
from personalityrag.library_types.text_media_v1.retrieval import (
    DEFAULT_RETRIEVAL_CONFIG,
    apply_evidence_threshold,
    fuse_embedding_rerank,
    media_frequency_signals,
    media_relevance_pivot,
    media_only_confidence_factor,
    noisy_or,
    normalize_retrieval_config,
    retrieval_config_json,
    rerank_calibration_settings_fingerprint,
    score_media_confidence,
    text_embedding_relevance,
)
from personalityrag.library_types.text_media_v1.service import (
    MEDIA_CALIBRATION_METHOD,
    TextMediaService,
    aggregate_media_description_calibrations,
    calibrate_media_strengths,
)
from personalityrag.library_types.text_media_v1.storage import (
    TextMediaStorage,
    normalize_bm25_rows,
)
from personalityrag.providers import RerankResult
from personalityrag.library_types.text_media_v1.text import (
    DEFAULT_VISUAL_INTENT_POLICY,
    LexiconReferenceVisualIntentDetector,
    chunk_text,
    fts_query_text,
    media_description_list,
    media_collection_intent,
    media_format_groups,
    media_subject_tokens,
    media_tokens,
    normalize_media_description,
    normalize_visual_intent_policy,
    text_query_tokens,
    visual_intent,
    visual_intent_policy_fingerprint,
    weighted_token_coverage,
)
from personalityrag.library_types.text_media_v1.visual_intent_policy import (
    parse_visual_intent_policy_csv,
)


def _search_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/knowledge-libraries/text_media_v1/review/search",
            "headers": [],
            "state": {"request_id": "fixture-search-request-id"},
        }
    )


class FixtureProvider:
    async def get_embedding(self, text: str) -> list[float]:
        return self._vector(text)

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [
            2.0 if "立绘" in text else 0.2,
            2.0 if "背景" in text else 0.2,
            1.0,
        ]

    async def close(self) -> None:
        return None


class DimensionalFixtureProvider:
    def __init__(
        self,
        config: ProviderConfig,
        *,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
        fail: bool = False,
    ):
        self.config = config
        self.started = started
        self.release = release
        self.fail = fail
        self.closed = False

    def _vector(self, text: str) -> list[float]:
        seed = sum(text.encode("utf-8")) or 1
        return [float((seed + index * 17) % 97 + 1) for index in range(self.config.dimensions)]

    async def get_embedding(self, text: str) -> list[float]:
        return (await self.get_embeddings([text]))[0]

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.fail:
            raise RuntimeError("fixture provider rebuild failure")
        max_context_tokens = int(self.config.max_context_tokens or 0)
        if max_context_tokens:
            max_chars = int(
                max_context_tokens * CONTEXT_CHUNK_CHAR_SAFETY_RATIO
            )
            if any(len(text) > max_chars for text in texts):
                raise RuntimeError("fixture context length exceeded")
        return [self._vector(text) for text in texts]

    async def close(self) -> None:
        self.closed = True


class ContextPolicyFixtureProvider(DimensionalFixtureProvider):
    def __init__(
        self,
        config: ProviderConfig,
        *,
        detect_calls: list[int],
        detected_tokens: int = 512,
    ):
        super().__init__(config)
        self.detect_calls = detect_calls
        self.detected_tokens = detected_tokens

    async def detect_context_length(self) -> dict[str, object]:
        self.detect_calls.append(int(self.config.max_context_tokens or 0))
        return {
            "max_context_tokens": self.detected_tokens,
            "max_context_tokens_source": "auto:text-media-task-fixture",
        }


@pytest.mark.asyncio
async def test_context_safe_embedding_splits_and_pools_without_rewriting_source() -> None:
    config = ProviderConfig(
        id="context-limited",
        display_name="Context limited",
        type="vllm_embedding",
        enabled=True,
        model="fixture-context",
        dimensions=5,
        context_length_mode="manual",
        max_context_tokens=128,
        max_context_tokens_source="manual:test",
    )
    provider = DimensionalFixtureProvider(config)
    short = "short text"
    long = "long context " * 100

    vectors, diagnostics = await embed_texts_context_safe(
        provider,
        [short, long],
        request_batch_size=4,
        concurrency=2,
    )

    assert vectors[0] == provider._vector(short)
    assert len(vectors) == 2
    assert len(vectors[1]) == 5
    assert np.linalg.norm(np.asarray(vectors[1], dtype=np.float32)) == pytest.approx(1.0)
    assert diagnostics["chunk_char_limit"] == 96
    assert diagnostics["split_input_count"] == 1
    assert diagnostics["fragment_count"] > diagnostics["input_count"]
    assert diagnostics["chunking_policy"] == (
        "stable_char_chunk_normalized_mean_pool_v1"
    )


@pytest.mark.asyncio
async def test_context_safe_embedding_uses_loose_policy_for_short_calls() -> None:
    valid_calls: list[int] = []
    valid = ContextPolicyFixtureProvider(
        ProviderConfig(
            id="loose-valid",
            dimensions=3,
            context_length_mode="auto",
            max_context_tokens=256,
            max_context_tokens_source="auto:cached",
        ),
        detect_calls=valid_calls,
    )
    _, valid_diagnostics = await embed_texts_context_safe(valid, ["short"])
    assert valid_calls == []
    assert valid_diagnostics["max_context_tokens"] == 256
    assert valid_diagnostics["context_length_mode"] == "auto"

    missing_calls: list[int] = []
    missing = ContextPolicyFixtureProvider(
        ProviderConfig(
            id="loose-missing",
            dimensions=3,
            context_length_mode="auto",
            max_context_tokens=0,
        ),
        detect_calls=missing_calls,
        detected_tokens=384,
    )
    _, missing_diagnostics = await embed_texts_context_safe(
        missing, ["short"]
    )
    assert missing_calls == [0]
    assert missing_diagnostics["max_context_tokens"] == 384
    assert missing.config.max_context_tokens == 384

    manual_calls: list[int] = []
    manual = ContextPolicyFixtureProvider(
        ProviderConfig(
            id="loose-manual",
            dimensions=3,
            context_length_mode="manual",
            max_context_tokens=256,
            max_context_tokens_source="manual:user",
        ),
        detect_calls=manual_calls,
    )
    await embed_texts_context_safe(manual, ["short"])
    assert manual_calls == []


def test_evidence_threshold_reinforces_and_weakens_confidence() -> None:
    reinforced = apply_evidence_threshold(
        0.6,
        [(0.8, 1), (0.4, 2)],
        threshold=0.2,
        limit=5,
        rank_decay_exponent=2,
        negative_reliability_exponent=2,
        reinforcement_weight=0.5,
        weakening_weight=0.5,
    )
    weakened = apply_evidence_threshold(
        0.6,
        [(0.4, 1), (0.3, 2)],
        threshold=0.9,
        limit=5,
        rank_decay_exponent=2,
        negative_reliability_exponent=2,
        reinforcement_weight=0.5,
        weakening_weight=0.5,
    )

    assert reinforced["positive_support"] > 0
    assert reinforced["adjusted_grounding"] > 0.6
    assert weakened["negative_pressure"] > 0
    assert weakened["adjusted_grounding"] < 0.6


def test_search_request_media_relevance_pivot_and_legacy_alias() -> None:
    assert SearchRequest(query="portrait").media_relevance_pivot is None
    assert SearchRequest(
        query="portrait", media_relevance_pivot=0.2
    ).media_relevance_pivot == 0.2
    assert SearchRequest(query="portrait").__dict__["media_score_threshold"] is None
    assert SearchRequest(
        query="portrait", media_score_threshold=None
    ).__dict__["media_score_threshold"] is None
    assert SearchRequest(
        query="portrait", media_score_threshold=0.0
    ).__dict__["media_score_threshold"] == 0.0
    with pytest.raises(
        ValueError, match="media_relevance_pivot and deprecated"
    ):
        SearchRequest(
            query="portrait",
            media_relevance_pivot=0.2,
            media_score_threshold=0.3,
        )


def test_media_relevance_pivot_endpoints_monotonicity_and_structure() -> None:
    assert media_relevance_pivot(0, 0.35) == 0
    assert media_relevance_pivot(1, 0.35) == 1
    assert media_relevance_pivot(0.35, 0.35) == pytest.approx(0.5)
    assert media_relevance_pivot(0.4, 0.01) > media_relevance_pivot(
        0.4, 0.35
    )
    assert media_relevance_pivot(0.4, 0.35) > media_relevance_pivot(
        0.4, 0.9
    )

    mismatched = score_media_confidence(
        raw_direct_relevance=0.2,
        grounding=0.5,
        evidence=[(0.5, 1)],
        bound_media=True,
        relevance_pivot=0.35,
        direct_signal_enabled=True,
        ambiguous_collection=False,
        subject_anchor_met=True,
        media_format_compatible=False,
        content_anchor_met=True,
        evidence_limit=5,
        rank_decay_exponent=1.5,
        negative_reliability_exponent=1.5,
        positive_blend=0.8,
        negative_weight=0.5,
        negative_attenuation_floor=0.05,
        format_mismatch_factor=0.1,
        content_mismatch_factor=0.15,
    )
    assert 0 < mismatched["output_confidence"] < (
        mismatched["pivoted_combined_confidence"]
    )
    blocked = score_media_confidence(
        **{
            **{
                key: value
                for key, value in {
                    "raw_direct_relevance": 0.2,
                    "grounding": 0.5,
                    "evidence": [(0.5, 1)],
                    "bound_media": True,
                    "relevance_pivot": 0.35,
                    "ambiguous_collection": False,
                    "subject_anchor_met": True,
                    "media_format_compatible": True,
                    "content_anchor_met": True,
                    "evidence_limit": 5,
                    "rank_decay_exponent": 1.5,
                    "negative_reliability_exponent": 1.5,
                    "positive_blend": 0.8,
                    "negative_weight": 0.5,
                    "negative_attenuation_floor": 0.05,
                    "format_mismatch_factor": 0.1,
                    "content_mismatch_factor": 0.15,
                }.items()
            },
            "direct_signal_enabled": False,
        }
    )
    assert blocked["output_confidence"] == 0

    common = {
        "raw_direct_relevance": 0.32,
        "grounding": 0.62,
        "evidence": [(0.55, 1), (0.31, 2), (0.02, 5)],
        "relevance_pivot": 0.35,
        "direct_signal_enabled": True,
        "ambiguous_collection": False,
        "subject_anchor_met": True,
        "media_format_compatible": True,
        "content_anchor_met": True,
        "evidence_limit": 5,
        "rank_decay_exponent": 1.5,
        "negative_reliability_exponent": 1.5,
        "positive_blend": 0.8,
        "negative_weight": 0.5,
        "negative_attenuation_floor": 0.05,
        "format_mismatch_factor": 0.1,
        "content_mismatch_factor": 0.15,
    }
    for bound_media in (False, True):
        scores = [
            score_media_confidence(
                **{
                    **common,
                    "bound_media": bound_media,
                    "relevance_pivot": pivot,
                }
            )["output_confidence"]
            for pivot in (0.0, 0.01, 0.1, 0.35, 0.7, 0.9, 1.0)
        ]
        assert scores == sorted(scores, reverse=True)


def test_search_request_rerank_has_three_state_semantics() -> None:
    assert SearchRequest(query="portrait").rerank is None
    assert SearchRequest(query="portrait", rerank=None).rerank is None
    assert SearchRequest(query="portrait", rerank=False).rerank is False
    assert SearchRequest(query="portrait", rerank=True).rerank is True


def test_search_request_retrieval_mode_and_empty_image_manifest_defaults() -> None:
    assert SearchRequest(query="portrait").retrieval_mode == "standard"
    assert SearchRequest(query="portrait").media_response_mode == "full"
    assert SearchRequest(
        query="portrait", retrieval_mode="media_only"
    ).retrieval_mode == "media_only"
    assert SearchRequest(
        query="portrait", retrieval_mode="text_only"
    ).retrieval_mode == "text_only"
    assert IngestBatchManifest().images == []


@pytest.mark.asyncio
async def test_search_description_mode_returns_only_qualified_compact_media(
    monkeypatch,
) -> None:
    def response() -> dict:
        return {
            "retrieval_mode": "standard",
            "items": [{"rank": 1, "text": "kept text"}],
            "baseline_items": [{"rank": 1, "text": "kept text"}],
            "media_outputs": [
                {
                    "asset_id": "portrait-asset",
                    "original_name": "portrait.png",
                    "media_description": "Full-body character portrait",
                    "matched_media_description": "Character portrait",
                    "output_policy": "auto",
                    "output_confidence": 0.82,
                    "association_score": 0.71,
                    "reason": "confidence_met",
                    "evidence": [{"chunk_id": 1}],
                }
            ],
            "media_decisions": [
                {
                    "asset_id": "portrait-asset",
                    "output": True,
                },
                {
                    "asset_id": "rejected-asset",
                    "output": False,
                    "reason": "confidence_below_threshold",
                },
            ],
        }

    service = SimpleNamespace(
        search=AsyncMock(side_effect=[response(), response()])
    )
    monkeypatch.setattr(text_media_api, "_ref", AsyncMock(return_value=object()))
    monkeypatch.setattr(text_media_api, "runtime", AsyncMock(return_value=service))

    full = await text_media_api.search(
        "review",
        SearchRequest(query="portrait"),
        _search_request(),
    )
    compact = await text_media_api.search(
        "review",
        SearchRequest(
            query="portrait",
            media_response_mode="descriptions_only",
        ),
        _search_request(),
    )

    assert "media_candidates" not in full
    assert full["media_outputs"][0]["content_url"].endswith(
        "/review/assets/portrait-asset/content"
    )
    assert full["media_decisions"][0]["thumbnail_url"].endswith(
        "/review/assets/portrait-asset/thumbnail"
    )
    assert compact["items"] == [{"rank": 1, "text": "kept text"}]
    assert compact["baseline_items"] == [{"rank": 1, "text": "kept text"}]
    assert compact["media_outputs"] == []
    assert compact["media_decisions"] == []
    assert compact["media_candidates"] == [
        {
            "rank": 1,
            "asset_id": "portrait-asset",
            "original_name": "portrait.png",
            "media_description": "Full-body character portrait",
            "matched_media_description": "Character portrait",
            "output_policy": "auto",
            "output_confidence": 0.82,
            "media_relevance": 0.71,
            "reason": "confidence_met",
            "fetchable": True,
        }
    ]
    assert "content_url" not in compact["media_candidates"][0]
    assert "thumbnail_url" not in compact["media_candidates"][0]
    assert "evidence" not in compact["media_candidates"][0]
    assert "rejected-asset" not in json.dumps(compact)


def test_update_request_preserves_explicit_rerank_unbind() -> None:
    assert "rerank_provider_id" not in UpdateRequest().model_dump(exclude_unset=True)
    assert UpdateRequest(rerank_provider_id=None).model_dump(exclude_unset=True) == {
        "rerank_provider_id": None
    }


def test_rerank_fusion_gates_rank_bonus_by_score_reliability() -> None:
    weak = fuse_embedding_rerank(
        0.6,
        0.01,
        rank=1,
        candidate_count=50,
        fusion_weight=0.25,
        rank_bonus_weight=0.2,
        rank_reliability_exponent=2.0,
    )
    strong = fuse_embedding_rerank(
        0.6,
        0.9,
        rank=1,
        candidate_count=50,
        fusion_weight=0.25,
        rank_bonus_weight=0.2,
        rank_reliability_exponent=2.0,
    )

    assert weak["rerank_rank_support"] < 0.001
    assert strong["rerank_rank_support"] > 0.8
    assert weak["rerank_signal"] == pytest.approx(0.0100198)
    assert weak["fused_relevance"] < 0.35
    assert weak["fused_relevance"] < 0.6
    assert strong["fused_relevance"] > weak["fused_relevance"]
    assert strong["fused_relevance"] > 0.6
    assert weak["ordering_relevance"] == pytest.approx(
        weak["fused_relevance"]
    )


def test_legacy_ineffective_rerank_weight_migrates_to_current_default() -> None:
    settings = normalize_retrieval_config({"rerank_fusion_weight": 0.00004})
    assert settings["rerank_fusion_weight"] == pytest.approx(0.30)


def test_rerank_calibration_settings_fingerprint_is_stable_and_sensitive() -> None:
    baseline = rerank_calibration_settings_fingerprint()
    assert baseline == rerank_calibration_settings_fingerprint(
        dict(DEFAULT_RETRIEVAL_CONFIG)
    )
    changed = {
        **DEFAULT_RETRIEVAL_CONFIG,
        "rerank_fusion_weight": (
            float(DEFAULT_RETRIEVAL_CONFIG["rerank_fusion_weight"]) + 0.1
        ),
    }
    assert rerank_calibration_settings_fingerprint(changed) != baseline


def test_evidence_threshold_respects_candidate_limit() -> None:
    first_only = apply_evidence_threshold(
        0.5,
        [(0.9, 1), (0.25, 2)],
        threshold=0.5,
        limit=1,
        rank_decay_exponent=2,
        negative_reliability_exponent=2,
        reinforcement_weight=0.5,
        weakening_weight=0.5,
    )
    both = apply_evidence_threshold(
        0.5,
        [(0.9, 1), (0.25, 2)],
        threshold=0.5,
        limit=2,
        rank_decay_exponent=2,
        negative_reliability_exponent=2,
        reinforcement_weight=0.5,
        weakening_weight=0.5,
    )

    assert first_only["evidence_count"] == 1
    assert both["evidence_count"] == 2
    assert both["negative_pressure"] > first_only["negative_pressure"]
    assert both["adjusted_grounding"] < first_only["adjusted_grounding"]


def test_evidence_threshold_near_zero_and_low_rank_have_tiny_influence() -> None:
    near_zero = apply_evidence_threshold(
        0.6,
        [(0.001, 1)],
        threshold=0.35,
        limit=20,
        rank_decay_exponent=2,
        negative_reliability_exponent=2,
        reinforcement_weight=0.35,
        weakening_weight=0.25,
    )
    middle = apply_evidence_threshold(
        0.6,
        [(0.20, 1)],
        threshold=0.35,
        limit=20,
        rank_decay_exponent=2,
        negative_reliability_exponent=2,
        reinforcement_weight=0.35,
        weakening_weight=0.25,
    )
    low_rank = apply_evidence_threshold(
        0.6,
        [(0.20, 20)],
        threshold=0.35,
        limit=20,
        rank_decay_exponent=2,
        negative_reliability_exponent=2,
        reinforcement_weight=0.35,
        weakening_weight=0.25,
    )

    assert near_zero["negative_pressure"] < middle["negative_pressure"] / 1000
    assert low_rank["negative_pressure"] < middle["negative_pressure"] / 10


def test_evidence_threshold_uses_fixed_normalization_constant() -> None:
    one = apply_evidence_threshold(
        0.5,
        [(0.2, 1)],
        threshold=0.35,
        limit=20,
        rank_decay_exponent=2,
        negative_reliability_exponent=2,
        reinforcement_weight=0.35,
        weakening_weight=0.25,
    )
    twenty = apply_evidence_threshold(
        0.5,
        [(0.2, 1), *[(0.001, rank) for rank in range(2, 21)]],
        threshold=0.35,
        limit=20,
        rank_decay_exponent=2,
        negative_reliability_exponent=2,
        reinforcement_weight=0.35,
        weakening_weight=0.25,
    )

    assert one["normalization_constant"] == twenty["normalization_constant"]
    assert twenty["negative_pressure"] - one["negative_pressure"] < 0.00001


def test_evidence_threshold_boundaries_are_neutral_or_non_negative() -> None:
    zero_threshold = apply_evidence_threshold(
        0.4,
        [(0.3, 1), (0.0, 2)],
        threshold=0,
        limit=5,
        rank_decay_exponent=1.5,
        negative_reliability_exponent=2.5,
        reinforcement_weight=0.5,
        weakening_weight=0.25,
    )
    exact_threshold = apply_evidence_threshold(
        0.4,
        [(0.35, 1)],
        threshold=0.35,
        limit=5,
        rank_decay_exponent=1.5,
        negative_reliability_exponent=2.5,
        reinforcement_weight=0.5,
        weakening_weight=0.25,
    )

    assert zero_threshold["negative_pressure"] == 0
    assert zero_threshold["positive_support"] > 0
    assert exact_threshold["positive_support"] == 0
    assert exact_threshold["negative_pressure"] == 0
    assert exact_threshold["contributions"][0]["threshold_effect"] == "neutral"


class FailingFixtureProvider(FixtureProvider):
    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("fixture embedding failure")


class BlockingFixtureProvider(FixtureProvider):
    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        await asyncio.sleep(60)
        return await super().get_embeddings(texts)


class CalibrationFixtureProvider(FixtureProvider):
    @staticmethod
    def _vector(text: str) -> list[float]:
        return [
            2.0 if "portrait" in text else 0.1,
            2.0 if "history" in text else 0.1,
            1.0,
        ]


class MultiImageFixtureProvider(FixtureProvider):
    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        return [
            4.0 if any(value in lowered for value in ("face", "portrait", "avatar")) else 0.05,
            4.0 if any(value in lowered for value in ("full", "armor", "outfit")) else 0.05,
            4.0 if any(value in lowered for value in ("battle", "action", "sword")) else 0.05,
            4.0 if "cat" in lowered else 0.05,
        ]


class MultiDescriptionFixtureProvider(FixtureProvider):
    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        groups = (
            ("deepseek", "深度求索"),
            ("gemini", "google", "谷歌"),
            ("chatgpt", "openai", "gpt"),
            ("claude", "anthropic", "克劳德"),
        )
        return [
            5.0 if any(value in lowered for value in group) else 0.02
            for group in groups
        ] + [1.0]


class RecordingIntentFixtureProvider(FixtureProvider):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def get_embedding(self, text: str) -> list[float]:
        self.calls.append([text])
        return self._vector(text)

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]


class FixtureReranker:
    def __init__(self, *, fail: bool = False, malformed: bool = False):
        self.config = SimpleNamespace(batch_size=64)
        self.fail = fail
        self.malformed = malformed
        self.calls: list[tuple[str, list[str]]] = []

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        self.calls.append((query, list(documents)))
        if self.fail:
            raise RuntimeError("fixture rerank failure")
        if self.malformed:
            return [RerankResult(index=0, relevance_score=0.9)]
        query_terms = set(query.casefold().split())
        rows = []
        for index, document in enumerate(documents):
            terms = set(document.casefold().split())
            overlap = len(query_terms & terms)
            rows.append(
                RerankResult(
                    index=index,
                    relevance_score=min(0.99, 0.05 + 0.3 * overlap),
                )
            )
        return sorted(rows, key=lambda item: (-item.relevance_score, item.index))

    async def close(self) -> None:
        return None


class ReverseFixtureReranker(FixtureReranker):
    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        self.calls.append((query, list(documents)))
        count = max(1, len(documents))
        return [
            RerankResult(index=index, relevance_score=(index + 1) / count)
            for index in range(len(documents) - 1, -1, -1)
        ]


class LowScoreReverseFixtureReranker(FixtureReranker):
    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        self.calls.append((query, list(documents)))
        count = max(1, len(documents))
        return [
            RerankResult(
                index=index,
                relevance_score=0.001 * (index + 1) / count,
            )
            for index in range(len(documents) - 1, -1, -1)
        ]


def _png_bytes(size: tuple[int, int] = (2200, 1100)) -> bytes:
    image = Image.new("RGB", size, (220, 40, 90))
    output = io.BytesIO()
    image.save(output, format="PNG", pnginfo=None)
    return output.getvalue()


def test_chunker_is_deterministic_and_overlaps_long_text() -> None:
    text = ("第一段介绍角色立绘。" * 90) + "\n\n" + ("第二段介绍背景。" * 90)
    first = chunk_text(text, target=300, overlap=40)
    second = chunk_text(text, target=300, overlap=40)
    assert first == second
    assert len(first) > 2
    assert [item["ordinal"] for item in first] == list(range(len(first)))
    assert all(len(str(item["content_sha256"])) == 64 for item in first)


def test_markdown_chunker_preserves_heading_paths_and_semantic_boundaries() -> None:
    alpha = "".join(f"Alpha sentence {index}. " for index in range(20))
    source = f"# Root\n\n## Alpha\n\n{alpha}\n\n## Beta\n\n- Beta one.\n- Beta two."
    first = chunk_text(source, target=220, overlap=50, format_hint="markdown")
    second = chunk_text(source, target=220, overlap=50, format_hint="markdown")

    assert first == second
    assert all(len(str(item["text"])) <= 220 for item in first)
    alpha_chunks = [item for item in first if "## Alpha" in str(item["text"])]
    beta_chunks = [item for item in first if "## Beta" in str(item["text"])]
    assert len(alpha_chunks) > 1
    assert beta_chunks
    assert all(str(item["text"]).startswith("# Root\n## Alpha\n\n") for item in alpha_chunks)
    assert all(str(item["text"]).startswith("# Root\n## Beta\n\n") for item in beta_chunks)
    assert all("## Beta" not in str(item["text"]) for item in alpha_chunks)
    assert [int(item["ordinal"]) for item in first] == list(range(len(first)))
    assert all(int(item["char_start"]) < int(item["char_end"]) for item in first)


def test_plaintext_chunker_uses_sentence_units_without_markdown_context() -> None:
    source = "".join(f"Sentence {index} is complete. " for index in range(30))
    chunks = chunk_text(source, target=200, overlap=40, format_hint="text")
    assert len(chunks) > 1
    assert all(len(str(item["text"])) <= 200 for item in chunks)
    assert all(not str(item["text"]).startswith("#") for item in chunks)
    assert all(str(item["text"]).rstrip().endswith(".") for item in chunks)


def test_plaintext_build_guide_preserves_lists_and_paragraph_boundaries() -> None:
    source = "\n\n".join(
        (
            "黑龙太刀终盘配装说明\n这是一份纯文本配装记录，不使用 Markdown 标题。",
            "武器：黑龙歼灭刀\n头部：精英·龙头盔\n胸部：精英·龙皮\n护石：挑战护石。",
            "核心技能包含看破、弱点特效、超会心和纳刀术。" * 35,
            "实战说明：纳刀术会影响特殊纳刀节奏，但不会改变居合判定窗口。" * 20,
        )
    )
    first = chunk_text(source, target=500, overlap=80, format_hint="text")
    second = chunk_text(source, target=500, overlap=80, format_hint="text")

    assert first == second
    assert len(first) > 2
    assert all(len(str(item["text"])) <= 500 for item in first)
    assert all(not str(item["text"]).startswith("#") for item in first)
    assert any("武器：黑龙歼灭刀\n头部：精英·龙头盔" in str(item["text"]) for item in first)
    assert any("纳刀术会影响特殊纳刀节奏" in str(item["text"]) for item in first)
    assert [int(item["ordinal"]) for item in first] == list(range(len(first)))


def test_text_query_tokens_recover_unknown_entity_and_build_or_fts() -> None:
    query = "\u840c\u4f9d\u662f\u8c01"
    assert text_query_tokens(query) == ["\u840c\u4f9d"]
    assert fts_query_text(query) == '"\u840c\u4f9d"'

    tokens = text_query_tokens("\u6f84\u6708\u5e73\u65f6\u7a7f\u4ec0\u4e48")
    assert "\u5e73\u65f6" in tokens
    assert "\u7a7f" in tokens
    assert "\u4ec0\u4e48" not in tokens
    assert " OR " in fts_query_text(
        "\u6f84\u6708\u5e73\u65f6\u7a7f\u4ec0\u4e48"
    )


def test_bm25_normalization_and_text_relevance_are_absolute() -> None:
    assert normalize_bm25_rows([(1, -8.0), (2, -4.0), (3, -2.0)]) == [
        (1, 1.0),
        (2, pytest.approx(1.0 / 3.0)),
        (3, 0.0),
    ]
    assert normalize_bm25_rows([(9, -1.5)]) == [(9, 1.0)]

    exact = text_embedding_relevance(0.46, 1.0, lexical_boost=0.5)
    unrelated = text_embedding_relevance(0.32, 0.0, lexical_boost=0.5)
    assert exact == pytest.approx(0.73)
    assert unrelated == pytest.approx(0.32)
    assert exact > unrelated


def test_media_tokenization_intent_gate_and_retrieval_config_validation() -> None:
    query_tokens = media_tokens("描述一下你的全身立绘！")
    coverage, matched = weighted_token_coverage(
        query_tokens, media_tokens("澄月的全身战斗服立绘设定图")
    )
    assert {"全身", "立绘"} <= set(matched)
    assert coverage > 0.8
    assert visual_intent("描述一下你的立绘")["detected"] is True
    assert visual_intent("show me your portrait")["detected"] is True
    assert visual_intent("покажи свой портрет")["detected"] is True
    assert visual_intent("你喜欢战斗吗")["detected"] is False
    assert visual_intent("立绘是谁画的")["detected"] is False
    assert visual_intent("图片上传失败")["detected"] is False
    assert visual_intent("GPT 表情包是谁画的")["detected"] is False
    weapon_intent = visual_intent("展示你手里的狱牙刀")
    assert weapon_intent["detected"] is True
    assert weapon_intent["subject_anchor_required"] is True
    assert "狱牙刀" in weapon_intent["subject_anchor_terms"]
    assert visual_intent("给我看看黑龙太刀配装图")["detected"] is True
    assert visual_intent("给我一张 GPT 表情包")["detected"] is True
    assert visual_intent("深度求索那个表情")["detected"] is True
    assert visual_intent("找一张《闪光的哈萨维》小说封面")["detected"] is True
    assert visual_intent("show me a GPT meme")["detected"] is True
    assert visual_intent("покажи мем GPT")["detected"] is True
    assert media_format_groups("给我看看黑龙太刀配装图") == {
        "generic_image"
    }
    assert media_format_groups("展示狱狼龙生态插图") == {
        "illustration",
        "generic_image",
    }
    assert media_format_groups("描述一下你的全身立绘") == {"portrait"}
    assert media_format_groups("找一张小说封面") == {"cover"}
    assert media_subject_tokens(media_tokens("给我看看黑龙太刀配装图")) == [
        "黑龙太刀",
        "配装",
    ]
    assert media_collection_intent("把四种表情包都给我看") is True
    assert media_collection_intent("展示立绘以及战斗场景") is True
    assert media_collection_intent("show me all portraits") is True
    assert media_collection_intent("看看 Claude 版表情包") is False
    assert media_tokens("给我一张 GPT 表情包") == ["gpt", "表情"]
    assert normalize_retrieval_config({}) == DEFAULT_RETRIEVAL_CONFIG
    assert DEFAULT_RETRIEVAL_CONFIG["text_lexical_boost"] == 0.6
    assert DEFAULT_RETRIEVAL_CONFIG["media_relevance_pivot_fallback"] == 0.35
    assert DEFAULT_RETRIEVAL_CONFIG["media_score_threshold_fallback"] == 0.35
    assert DEFAULT_RETRIEVAL_CONFIG["media_threshold_evidence_limit"] == 5
    assert DEFAULT_RETRIEVAL_CONFIG["media_threshold_rank_decay_exponent"] == 1.5
    assert DEFAULT_RETRIEVAL_CONFIG[
        "media_threshold_negative_reliability_exponent"
    ] == 1.5
    assert DEFAULT_RETRIEVAL_CONFIG["media_pivot_positive_blend"] == 0.7
    assert DEFAULT_RETRIEVAL_CONFIG["media_pivot_negative_weight"] == 0.35
    assert DEFAULT_RETRIEVAL_CONFIG[
        "media_pivot_negative_attenuation_floor"
    ] == 0.05
    assert DEFAULT_RETRIEVAL_CONFIG["media_format_mismatch_factor"] == 0.1
    assert DEFAULT_RETRIEVAL_CONFIG["media_content_mismatch_factor"] == 0.1
    assert DEFAULT_RETRIEVAL_CONFIG["media_threshold_reinforcement_weight"] == 0.7
    assert DEFAULT_RETRIEVAL_CONFIG["media_threshold_weakening_weight"] == 0.35
    assert DEFAULT_RETRIEVAL_CONFIG["media_lexical_common_floor"] == 0
    assert DEFAULT_RETRIEVAL_CONFIG["media_lexical_oov_penalty"] == 0.3
    assert DEFAULT_RETRIEVAL_CONFIG["media_distinctive_rarity_exponent"] == 1.5
    assert DEFAULT_RETRIEVAL_CONFIG["unbound_media_candidate_limit"] == 10
    assert DEFAULT_RETRIEVAL_CONFIG["unbound_media_distinctive_boost"] == 0.35
    assert DEFAULT_RETRIEVAL_CONFIG["unbound_media_collection_boost"] == 0.55
    assert DEFAULT_RETRIEVAL_CONFIG["unbound_media_competition_floor"] == 0.35
    assert DEFAULT_RETRIEVAL_CONFIG["unbound_media_reliability_target"] == 0.25
    assert DEFAULT_RETRIEVAL_CONFIG["unbound_media_specificity_exponent"] == 1.0
    assert DEFAULT_RETRIEVAL_CONFIG["unbound_media_advantage_target"] == 0.04
    assert DEFAULT_RETRIEVAL_CONFIG["media_bound_distinctive_boost"] == 0.1
    assert DEFAULT_RETRIEVAL_CONFIG["media_bound_distinctive_rescue_min"] == 0.8
    old_config = normalize_retrieval_config({"media_candidate_limit": 10})
    assert old_config["media_score_threshold_fallback"] == 0.35
    assert old_config["media_relevance_pivot_fallback"] == 0.35
    migrated_threshold_defaults = normalize_retrieval_config(
        {
            "media_score_threshold_fallback": 0.44,
            "media_threshold_reinforcement_weight": 0.2,
            "media_threshold_weakening_weight": 0.1,
        }
    )
    assert migrated_threshold_defaults["media_relevance_pivot_fallback"] == 0.44
    assert migrated_threshold_defaults["media_pivot_positive_blend"] == 0.7
    assert migrated_threshold_defaults["media_pivot_negative_weight"] == 0.35
    assert old_config["media_threshold_evidence_limit"] == 5
    explicit_window = normalize_retrieval_config(
        {
            "media_candidate_limit": 40,
            "media_threshold_evidence_limit": 30,
            "media_threshold_rank_decay_exponent": 2.5,
            "media_threshold_negative_reliability_exponent": 3.0,
            "media_threshold_reinforcement_weight": 0.5,
            "media_threshold_weakening_weight": 0.4,
        }
    )
    assert explicit_window["media_threshold_evidence_limit"] == 30
    assert explicit_window["media_pivot_positive_blend"] == 0.5
    assert explicit_window["media_pivot_negative_weight"] == 0.4
    assert explicit_window["media_threshold_reinforcement_weight"] == 0.5
    assert explicit_window["media_threshold_weakening_weight"] == 0.4
    with pytest.raises(ValueError, match="media_candidate_limit"):
        normalize_retrieval_config({"media_candidate_limit": 9})
    with pytest.raises(ValueError, match="media_threshold_evidence_limit"):
        normalize_retrieval_config({"media_threshold_evidence_limit": 101})
    with pytest.raises(ValueError, match="unbound_media_competition_floor"):
        normalize_retrieval_config({"media_only_competition_floor": 1.01})
    legacy = normalize_retrieval_config(
        {
            "media_only_asset_candidate_limit": 20,
            "media_only_distinctive_boost": 0.4,
            "media_only_collection_boost": 0.42,
            "media_only_competition_floor": 0.48,
            "media_only_grounding_direct_exponent": 3.0,
        }
    )
    assert legacy["unbound_media_candidate_limit"] == 20
    assert legacy["unbound_media_distinctive_boost"] == 0.4
    assert legacy["unbound_media_collection_boost"] == 0.42
    assert legacy["unbound_media_competition_floor"] == 0.48
    assert "media_only_grounding_direct_exponent" not in legacy


def test_reference_generation_intent_projects_only_the_source_media_clause() -> None:
    detector = LexiconReferenceVisualIntentDetector()
    cases = {
        "根据你的立绘画一张q版图": ("你的立绘", "看看你的立绘"),
        "先看看你的立绘，再画一张q版的": ("你的立绘", "看看你的立绘"),
        "把你的立绘画成Q版": ("你的立绘", "看看你的立绘"),
        "参考你的头像生成像素风头像": ("你的头像", "看看你的头像"),
        "把你的战斗场景改成水彩插图": (
            "你的战斗场景",
            "看看你的战斗场景",
        ),
        "用DeepSeek表情包做一张新梗图": (
            "deepseek表情包",
            "看看deepseek表情包",
        ),
        "draw a chibi based on your portrait": (
            "your portrait",
            "show your portrait",
        ),
        "нарисуй чиби на основе твоего портрета": (
            "твоего портрета",
            "покажи твоего портрета",
        ),
    }
    for query, (reference, media_query) in cases.items():
        intent = detector.analyze(query)
        assert intent["detected"] is True, query
        assert intent["intent_kind"] == "reference_generation", query
        assert intent["reference_span"] == reference, query
        assert intent["media_query"] == media_query, query
        assert intent["projection_applied"] is True, query
        assert intent["generation_span"], query

    protected = detector.analyze("根据你的立绘画一张q版图")
    assert "立绘" in protected["protected_terms"]
    assert protected["generation_span"] == "一张q版图"
    assert protected["ignored_output_terms"] == ["一张q版图"]
    projected_tokens = media_tokens(
        str(protected["media_query"]),
        protected_terms=list(protected["protected_terms"]),
    )
    assert "立绘" in projected_tokens
    assert "q" not in projected_tokens

    for query, expected_kind in (
        ("画一张Q版图", "generation_without_reference"),
        ("我喜欢画画", "generation_without_reference"),
        ("立绘是谁画的", "blocked"),
        ("图片上传失败", "blocked"),
        ("战斗策略是什么", "blocked"),
    ):
        intent = detector.analyze(query)
        assert intent["detected"] is False, query
        assert intent["intent_kind"] == expected_kind, query
        assert intent["projection_applied"] is False, query
    assert detector.analyze("立绘是谁画的", gate_enabled=False)[
        "detected"
    ] is False
    assert detector.analyze("画一张Q版图", gate_enabled=False)[
        "detected"
    ] is False


def test_visual_intent_policy_is_literal_normalized_complete_and_fingerprinted() -> None:
    defaults = normalize_visual_intent_policy(DEFAULT_VISUAL_INTENT_POLICY)
    assert defaults["generation_action_terms"]
    assert defaults["visual_object_terms"]
    assert normalize_retrieval_config({})["visual_intent_policy"] == defaults
    inherited_serialized = json.loads(retrieval_config_json())
    assert "visual_intent_policy" not in inherited_serialized
    assert normalize_retrieval_config(inherited_serialized)[
        "visual_intent_policy"
    ] == defaults

    custom = normalize_visual_intent_policy(
        {
            "visual_object_terms": ["  参考 图  ", "参考  图", "PORTRAIT"],
            "lookup_action_terms": [],
            "generation_action_terms": ["创作", "CREATE"],
            "reference_connector_terms": ["依照"],
        },
        require_complete=True,
    )
    assert custom == {
        "visual_object_terms": ["参考 图", "portrait"],
        "lookup_action_terms": [],
        "generation_action_terms": ["创作", "create"],
        "reference_connector_terms": ["依照"],
    }
    request = VisualIntentPolicyRequest(**custom)
    assert request.model_dump() == custom
    assert visual_intent_policy_fingerprint(custom) == (
        visual_intent_policy_fingerprint(dict(reversed(list(custom.items()))))
    )
    changed = {**custom, "generation_action_terms": ["绘制"]}
    assert visual_intent_policy_fingerprint(changed) != (
        visual_intent_policy_fingerprint(custom)
    )
    assert LexiconReferenceVisualIntentDetector(custom).analyze(
        "依照参考 图创作成卡通"
    )["intent_kind"] == "reference_generation"
    with pytest.raises(ValueError, match="requires all fields"):
        normalize_visual_intent_policy(
            {"visual_object_terms": ["立绘"]}, require_complete=True
        )
    with pytest.raises(ValueError, match="cannot exceed 64"):
        normalize_visual_intent_policy(
            {**custom, "visual_object_terms": ["x" * 65]},
            require_complete=True,
        )


def test_media_frequency_signals_reward_unique_terms_and_gate_collections() -> None:
    document_frequencies = {
        "gpt": 1,
        "gemini": 1,
        "claude": 1,
        "deepseek": 1,
        "表情": 4,
        "模型": 4,
    }
    settings = {
        "coverage_exponent": 1.2,
        "common_floor": 0.1,
        "oov_penalty": 0.5,
        "rarity_exponent": 1.0,
    }
    target = media_frequency_signals(
        ["gpt", "表情"],
        ["原来", "劣等", "模型", "gpt", "表情"],
        document_frequencies,
        8,
        collection_intent=False,
        **settings,
    )
    common = media_frequency_signals(
        ["表情"],
        ["原来", "劣等", "模型", "gpt", "表情"],
        document_frequencies,
        8,
        collection_intent=False,
        **settings,
    )
    wrong = media_frequency_signals(
        ["gpt", "表情"],
        ["原来", "劣等", "模型", "gemini", "表情"],
        document_frequencies,
        8,
        collection_intent=False,
        **settings,
    )
    collection = media_frequency_signals(
        ["表情"],
        ["原来", "劣等", "模型", "gpt", "表情"],
        document_frequencies,
        8,
        collection_intent=True,
        **settings,
    )
    assert target["distinctive_support"] > 0.9
    assert target["distinctive_membership_support"] == pytest.approx(1.0)
    assert target["candidate_lexical_score"] > common[
        "candidate_lexical_score"
    ]
    assert target["completeness"] > wrong["completeness"]
    assert common["collection_support"] == 0
    assert collection["collection_support"] == pytest.approx(1.0)
    selected_pair = media_frequency_signals(
        ["谷歌", "表情"],
        ["谷歌", "表情"],
        {"谷歌": 2, "表情": 4},
        4,
        collection_intent=True,
        **settings,
    )
    unrelated_pair = media_frequency_signals(
        ["谷歌", "表情"],
        ["表情"],
        {"谷歌": 2, "表情": 4},
        4,
        collection_intent=True,
        **settings,
    )
    assert selected_pair["collection_membership_support"] > 0
    assert unrelated_pair["collection_membership_support"] == 0
    assert media_frequency_signals(
        ["gpt"],
        ["gpt"],
        {"gpt": 1},
        1,
        collection_intent=False,
        **settings,
    )["distinctive_support"] == 0


def test_media_only_confidence_factor_distinguishes_single_and_collection_queries() -> None:
    target = media_only_confidence_factor(
        0.72,
        0.72,
        strongest_competitor_score=0.52,
        collection_intent=False,
    )
    similar_false_positive = media_only_confidence_factor(
        0.52,
        0.72,
        strongest_competitor_score=0.72,
        collection_intent=False,
    )
    collection_member = media_only_confidence_factor(
        0.52,
        0.72,
        strongest_competitor_score=0.72,
        collection_membership_support=1.0,
        collection_intent=True,
    )
    weak_direct = media_only_confidence_factor(
        0.14,
        0.72,
        strongest_competitor_score=0.72,
        collection_intent=False,
    )
    assert target["confidence_factor"] == pytest.approx(1.0)
    assert similar_false_positive["confidence_factor"] == pytest.approx(0.5)
    assert similar_false_positive["advantage_weight"] == 0
    assert similar_false_positive["winner_support"] == 0
    assert similar_false_positive["competition_floor"] == pytest.approx(0.5)
    assert collection_member["confidence_factor"] == pytest.approx(1.0)
    assert collection_member["competition_floor"] == 0
    assert weak_direct["direct_reliability"] == pytest.approx(0.4)
    assert weak_direct["confidence_factor"] == pytest.approx(0.5)

    tied = media_only_confidence_factor(
        0.72,
        0.72,
        strongest_competitor_score=0.72,
        collection_intent=False,
    )
    no_signal = media_only_confidence_factor(
        0.0,
        0.72,
        strongest_competitor_score=0.72,
        collection_intent=False,
    )
    assert tied["confidence_factor"] == pytest.approx(0.5)
    assert no_signal["confidence_factor"] == pytest.approx(0.5)
    assert 0.0 * no_signal["confidence_factor"] == 0.0


def test_media_only_soft_competition_preserves_family_hierarchy_under_grounding_pressure() -> None:
    direct_scores = {
        "deepseek": 0.781,
        "gpt": 0.251,
        "gemini": 0.250,
        "claude": 0.240,
        "battle": 0.092,
        "portrait": 0.059,
    }
    grounding_scores = {
        "deepseek": 0.0,
        "gpt": 0.0,
        "gemini": 0.0,
        "claude": 0.0,
        "battle": 0.430,
        "portrait": 0.432,
    }
    best = max(direct_scores.values())
    final_scores: dict[str, float] = {}
    for label, direct_score in direct_scores.items():
        gated_grounding = grounding_scores[label] * direct_score**2.5
        raw_confidence = noisy_or((direct_score, gated_grounding))
        strongest_competitor = max(
            value
            for other_label, value in direct_scores.items()
            if other_label != label
        )
        adjustment = media_only_confidence_factor(
            direct_score,
            best,
            strongest_competitor_score=strongest_competitor,
            collection_intent=False,
            competition_floor=0.5,
        )
        final_scores[label] = raw_confidence * adjustment["confidence_factor"]

    assert final_scores["deepseek"] >= 0.60
    assert max(
        final_scores[label] for label in ("gpt", "gemini", "claude")
    ) < 0.60
    assert min(
        final_scores[label] for label in ("gpt", "gemini", "claude")
    ) > max(final_scores[label] for label in ("battle", "portrait"))


def test_media_only_soft_competition_scales_to_shared_template_corpus() -> None:
    direct_scores = {"target": 0.80}
    direct_scores.update({f"sibling-{index:03d}": 0.24 for index in range(50)})
    direct_scores.update({f"distractor-{index:03d}": 0.05 for index in range(49)})
    best = max(direct_scores.values())
    final_scores: dict[str, float] = {}
    for label, direct_score in direct_scores.items():
        grounding = 0.0 if label == "target" or label.startswith("sibling") else 0.95
        raw_confidence = noisy_or((direct_score, grounding * direct_score**2.5))
        adjustment = media_only_confidence_factor(
            direct_score,
            best,
            strongest_competitor_score=max(
                value
                for other_label, value in direct_scores.items()
                if other_label != label
            ),
            collection_intent=False,
            competition_floor=0.5,
        )
        final_scores[label] = raw_confidence * adjustment["confidence_factor"]

    assert final_scores["target"] >= 0.60
    assert min(
        value for label, value in final_scores.items() if label.startswith("sibling")
    ) > max(
        value
        for label, value in final_scores.items()
        if label.startswith("distractor")
    )
    assert all(0.0 <= value <= 1.0 for value in final_scores.values())


@pytest.mark.asyncio
async def test_media_token_frequency_pressure_updates_and_deletes_df(
    tmp_path: Path,
) -> None:
    root = tmp_path / "media-token-pressure"
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="media-token-pressure",
        name="Media token pressure",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    assets: list[dict[str, object]] = []
    for index in range(100):
        shared = " sharedpair" if index < 2 else ""
        description = f"sharedtemplate emote tag{index:03d}{shared}"
        asset = await storage.register_asset(
            kind="image",
            sha256=f"{index + 1:064x}",
            storage_key=f"assets/images/{index:03d}.webp",
            mime_type="image/webp",
            size_bytes=1,
            original_name=f"{description}.jpg",
            width=1,
            height=1,
        )
        await storage.upsert_asset_media_metadata(
            asset_id=str(asset["id"]),
            media_description=description,
            description_source="user",
            vector=[1.0, 0.0],
            provider_id="fixture",
            provider_revision=1,
            provider_fingerprint="fixture-sha",
        )
        assets.append(asset)

    query_tokens = ["tag042", "sharedpair", "emote"]
    initial = await storage.media_token_statistics(
        query_tokens, bound_only=False
    )
    assert initial["corpus_size"] == 100
    assert initial["document_frequencies"] == {
        "tag042": 1,
        "sharedpair": 2,
        "emote": 100,
    }
    unique = media_frequency_signals(
        ["tag042"],
        ["tag042"],
        initial["document_frequencies"],
        initial["corpus_size"],
        coverage_exponent=1.0,
        common_floor=0.0,
        oov_penalty=0.15,
        rarity_exponent=1.5,
        collection_intent=False,
    )
    pair = media_frequency_signals(
        ["sharedpair"],
        ["sharedpair"],
        initial["document_frequencies"],
        initial["corpus_size"],
        coverage_exponent=1.0,
        common_floor=0.0,
        oov_penalty=0.15,
        rarity_exponent=1.5,
        collection_intent=False,
    )
    common = media_frequency_signals(
        ["emote"],
        ["emote"],
        initial["document_frequencies"],
        initial["corpus_size"],
        coverage_exponent=1.0,
        common_floor=0.0,
        oov_penalty=0.15,
        rarity_exponent=1.5,
        collection_intent=False,
    )
    assert unique["distinctive_support"] > pair["distinctive_support"] > 0
    assert common["distinctive_support"] == 0

    updated_description = "sharedtemplate emote tag002 sharedpair"
    await storage.upsert_asset_media_metadata(
        asset_id=str(assets[2]["id"]),
        media_description=updated_description,
        description_source="user",
        vector=[1.0, 0.0],
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    after_update = await storage.media_token_statistics(
        query_tokens, bound_only=False
    )
    assert after_update["document_frequencies"]["sharedpair"] == 3

    await storage.delete_image_asset(str(assets[1]["id"]))
    after_delete = await storage.media_token_statistics(
        query_tokens, bound_only=False
    )
    assert after_delete["corpus_size"] == 99
    assert after_delete["document_frequencies"]["sharedpair"] == 2
    assert after_delete["document_frequencies"]["tag042"] == 1
    await storage.close()


def test_image_pipeline_discards_source_format_and_enforces_profile() -> None:
    source = _png_bytes()
    normalized = normalize_image_bytes(source)
    assert normalized.data[:4] == b"RIFF"
    assert normalized.mime_type == "image/webp"
    assert max(normalized.width, normalized.height) == 1536
    assert normalized.size_bytes <= MAX_CANONICAL_BYTES
    with Image.open(io.BytesIO(normalized.data)) as image:
        assert image.format == "WEBP"
        assert "exif" not in image.info

    with pytest.raises(ImageValidationError):
        normalize_image_bytes(b"<svg xmlns='http://www.w3.org/2000/svg'></svg>")


@pytest.mark.asyncio
async def test_legacy_schema_migrates_media_strength_settings(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    path = root / "textmediaknowledge.db"
    with sqlite3.connect(path) as db:
        db.execute(
            """CREATE TABLE document_assets (
            document_id TEXT NOT NULL,asset_id TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'illustration',
            relation_weight REAL NOT NULL DEFAULT 1,
            caption TEXT NOT NULL DEFAULT '',alt_text TEXT NOT NULL DEFAULT '',
            output_policy TEXT NOT NULL DEFAULT 'auto',sort_order INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(document_id,asset_id))"""
        )
    storage = TextMediaStorage(root)
    try:
        await storage.initialize()
        await storage.create_library(
            database_id="legacy",
            name="Legacy",
            description="",
            provider_id="fixture",
            provider_revision=1,
            provider_fingerprint="fixture-sha",
        )
        db = await storage.pool.acquire()
        try:
            columns = {
                str(row[1])
                for row in await (
                    await db.execute("PRAGMA table_info(document_assets)")
                ).fetchall()
            }
            version = await (
                await db.execute(
                    "SELECT value FROM schema_info WHERE key='schema_version'"
                )
            ).fetchone()
            meta_columns = {
                str(row[1])
                for row in await (
                    await db.execute("PRAGMA table_info(library_meta)")
                ).fetchall()
            }
            default_strength = await (
                await db.execute(
                    "SELECT uniform_media_strength FROM library_meta WHERE singleton=1"
                )
            ).fetchone()
            tables = {
                str(row[0])
                for row in await (
                    await db.execute(
                        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
                    )
                ).fetchall()
            }
        finally:
            await db.close()
            assert {
                "semantic_mode",
                "media_description",
                "media_description_vector",
                "calibration_method",
                "calibration_provider_fingerprint",
                "calibration_rerank_provider_fingerprint",
            } <= columns
            assert "uniform_media_strength" in meta_columns
            assert "retrieval_config_json" in meta_columns
            assert float(default_strength[0]) == pytest.approx(0.5)
            assert {
                "rerank_provider_id",
                "rerank_provider_revision",
                "rerank_provider_fingerprint",
            } <= meta_columns
            assert version[0] == "9"
            assert "asset_media_metadata" in tables
            assert "asset_media_metadata_fts" in tables
            assert "asset_media_tokens" in tables
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_atomic_ingest_batch_tracks_members_and_document_media_relations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "library"
    staging = tmp_path / "staging"
    staging.mkdir()
    first = staging / "first.md"
    second = staging / "second.txt"
    image_path = staging / "source.png"
    first.write_text("角色立绘与战斗服。" * 120, encoding="utf-8")
    second.write_text("背景资料与角色经历。" * 80, encoding="utf-8")
    image_path.write_bytes(_png_bytes((800, 600)))

    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="batch",
        name="Batch",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    await storage.update_metadata({"uniform_media_strength": 0.35})
    service = TextMediaService(
        root, storage, TextMediaIndex(root), CalibrationFixtureProvider()
    )
    try:
        result = await service.ingest_batch(
            batch_id="batch-1",
            documents=[
                {"path": str(first), "filename": first.name, "title": "First"},
                {"path": str(second), "filename": second.name, "title": "Second"},
            ],
            images=[
                {
                    "path": str(image_path),
                    "filename": image_path.name,
                    "document_indexes": [0, 1],
                },
            ],
            chunk_target=240,
            chunk_overlap=40,
            embedding_batch_size=3,
            concurrency=2,
            max_retries=2,
        )
        assert result["document_count"] == 2
        assert result["image_count"] == 1
        assert result["images"][0]["document_indexes"] == [0, 1]
        assert result["chunk_count"] > 4
        assert (await storage.statistics())["documents"] == 2
        db = await storage.pool.acquire()
        try:
            assert int((await (await db.execute("SELECT COUNT(*) FROM ingest_batches")).fetchone())[0]) == 1
            assert int((await (await db.execute("SELECT COUNT(*) FROM ingest_batch_documents")).fetchone())[0]) == 2
            assert int((await (await db.execute("SELECT COUNT(*) FROM ingest_batch_assets")).fetchone())[0]) == 1
            strengths = await (
                await db.execute(
                    "SELECT semantic_strength FROM chunk_media_strengths ORDER BY chunk_id"
                )
            ).fetchall()
            assert len(strengths) == result["chunk_count"]
            assert {float(row[0]) for row in strengths} == {0.35}
            first_chunks = [
                int(row[0])
                for row in await (
                    await db.execute(
                        """SELECT c.id FROM chunks c JOIN entries e ON e.id=c.entry_id
                        WHERE e.document_id=? ORDER BY c.id""",
                        (result["documents"][0]["document_id"],),
                    )
                ).fetchall()
            ]
            second_chunks = [
                int(row[0])
                for row in await (
                    await db.execute(
                        """SELECT c.id FROM chunks c JOIN entries e ON e.id=c.entry_id
                        WHERE e.document_id=? ORDER BY c.id""",
                        (result["documents"][1]["document_id"],),
                    )
                ).fetchall()
            ]
        finally:
            await db.close()
        attachments = await storage.attachments_for_chunks(
            [first_chunks[0], second_chunks[0]]
        )
        assert attachments[first_chunks[0]][0]["scope"] == "document"
        assert attachments[second_chunks[0]][0]["asset_id"] == attachments[first_chunks[0]][0]["asset_id"]
        assert (await storage.statistics())["images"] == 1
        asset_id = str(attachments[first_chunks[0]][0]["asset_id"])
        assert (await service.delete_image(asset_id))["deleted"] is True
        db = await storage.pool.acquire()
        try:
            assert int(
                (
                    await (
                        await db.execute(
                            "SELECT COUNT(*) FROM ingest_batch_assets WHERE asset_id=?",
                            (asset_id,),
                        )
                    ).fetchone()
                )[0]
            ) == 0
        finally:
            await db.close()
        assert (await storage.statistics())["images"] == 0
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_content_review_queries_and_batch_document_delete(tmp_path: Path) -> None:
    root = tmp_path / "library"
    staging = tmp_path / "staging"
    staging.mkdir()
    first = staging / "alpha.md"
    second = staging / "beta.md"
    image_path = staging / "portrait.png"
    first.write_text("# Alpha\n\nPortrait and armor details. " * 40, encoding="utf-8")
    second.write_text("# Beta\n\nBackground and history details. " * 40, encoding="utf-8")
    image_path.write_bytes(_png_bytes((800, 600)))

    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="review",
        name="Review",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    service = TextMediaService(root, storage, TextMediaIndex(root), FixtureProvider())
    try:
        result = await service.ingest_batch(
            batch_id="review-batch",
            documents=[
                {"path": str(first), "filename": first.name, "title": "Alpha"},
                {"path": str(second), "filename": second.name, "title": "Beta"},
            ],
            images=[
                {
                    "path": str(image_path),
                    "filename": image_path.name,
                    "document_indexes": [0],
                    "media_description": "Portrait and armor",
                    "media_descriptions": [
                        "Portrait and armor",
                        "Full-body black armor illustration",
                    ],
                }
            ],
            chunk_target=240,
            chunk_overlap=40,
            embedding_batch_size=4,
            concurrency=2,
            max_retries=2,
            media_semantic_calibration_enabled=True,
        )
        alpha_id = str(result["documents"][0]["document_id"])
        beta_id = str(result["documents"][1]["document_id"])
        asset_id = str(result["images"][0]["asset_id"])

        page = await storage.list_document_summaries(query="alpha", limit=10)
        assert page["total"] == 1
        assert page["items"][0]["id"] == alpha_id
        assert page["items"][0]["chunk_count"] > 0
        assert page["items"][0]["image_count"] == 1

        detail = await storage.get_document_detail(alpha_id)
        assert detail is not None
        assert detail["original_name"] == "alpha.md"
        assert detail["image_count"] == 1
        assert detail["associated_media"][0]["asset_id"] == asset_id
        assert "Portrait and armor" in detail["content"]

        chunks = await storage.list_chunk_summaries(document_id=alpha_id, limit=50)
        assert chunks["total"] == detail["chunk_count"]
        chunk_id = int(chunks["items"][0]["id"])
        chunk = await storage.get_chunk_detail(chunk_id)
        assert chunk is not None
        assert chunk["document_id"] == alpha_id
        assert chunk["associated_media"][0]["asset_id"] == asset_id
        assert "media_description_vector" not in chunk["associated_media"][0]
        assert "storage_key" not in chunk["associated_media"][0]
        assert (
            "Portrait and armor"
            in chunk["associated_media"][0]["media_description"]
        )
        json.dumps(chunk)

        asset = await storage.get_asset_detail(asset_id)
        assert [
            item["media_description"] for item in asset["media_descriptions"]
        ] == [
            "Portrait and armor",
            "Full-body black armor illustration",
        ]
        assert asset["media_description_count"] == 2
        assert asset is not None
        assert asset["relation_count"] == 1
        assert asset["relations"][0]["scope"] == "document"
        assert asset["relations"][0]["target_id"] == alpha_id

        deleted = await service.delete_documents([alpha_id, beta_id])
        assert deleted["deleted_count"] == 2
        assert deleted["generation_id"] != result["generation_id"]
        generation, remaining_ids, remaining_vectors = await storage.active_vectors()
        assert generation == deleted["generation_id"]
        assert remaining_ids == []
        assert remaining_vectors.shape[0] == 0
        assert (await storage.list_document_summaries())["total"] == 0
        assert (await storage.list_chunk_summaries())["total"] == 0
        assert await storage.get_asset(asset_id) is not None
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_chunk_detail_api_adds_media_preview_urls(monkeypatch) -> None:
    item = {
        "id": 7,
        "associated_media": [
            {
                "asset_id": "portrait-asset",
                "original_name": "portrait.png",
            }
        ],
    }
    service = SimpleNamespace(
        storage=SimpleNamespace(get_chunk_detail=AsyncMock(return_value=item))
    )
    monkeypatch.setattr(text_media_api, "_ref", AsyncMock(return_value=object()))
    monkeypatch.setattr(text_media_api, "runtime", AsyncMock(return_value=service))

    result = await text_media_api.chunk_detail("review", 7)

    media = result["associated_media"][0]
    assert media["thumbnail_url"].endswith(
        "/review/assets/portrait-asset/thumbnail"
    )
    assert media["content_url"].endswith(
        "/review/assets/portrait-asset/content"
    )
    json.dumps(result)


def test_media_semantic_calibration_is_stable_and_bounded() -> None:
    chunks = [
        [1.0, 0.0, 0.0],
        [0.8, 0.2, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]
    first = calibrate_media_strengths([1.0, 0.0, 0.0], chunks)
    second = calibrate_media_strengths([2.0, 0.0, 0.0], chunks)
    assert first == second
    assert first[0]["calibration_rank"] == 1
    assert first[0]["semantic_strength"] == pytest.approx(1.0)
    assert first[-1]["semantic_strength"] < first[1]["semantic_strength"]
    assert all(0.15 <= item["semantic_strength"] <= 1.0 for item in first)


@pytest.mark.asyncio
async def test_secondary_embedding_calibrates_document_media_edges(
    tmp_path: Path,
) -> None:
    root = tmp_path / "library"
    source = tmp_path / "source.md"
    image_path = tmp_path / "source.png"
    source.write_text(("portrait combat outfit\n\n" * 80) + ("unrelated history\n\n" * 80), encoding="utf-8")
    image_path.write_bytes(_png_bytes((800, 600)))
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="calibrated",
        name="Calibrated",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    service = TextMediaService(
        root, storage, TextMediaIndex(root), CalibrationFixtureProvider()
    )
    try:
        result = await service.ingest_batch(
            batch_id="calibrated-1",
            documents=[{"path": str(source), "filename": source.name, "title": "Source"}],
            images=[{
                "path": str(image_path),
                "filename": image_path.name,
                "document_indexes": [0],
                "media_description": "portrait illustration",
            }],
            chunk_target=200,
            chunk_overlap=20,
            embedding_batch_size=4,
            concurrency=2,
            max_retries=2,
            media_semantic_calibration_enabled=True,
        )
        db = await storage.pool.acquire()
        try:
            relation = await (
                await db.execute(
                    "SELECT semantic_mode,media_description,calibration_method FROM document_assets"
                )
            ).fetchone()
            strengths = await (
                await db.execute(
                    "SELECT semantic_strength,calibration_rank FROM chunk_media_strengths ORDER BY calibration_rank"
                )
            ).fetchall()
        finally:
            await db.close()
        assert relation["semantic_mode"] == "calibrated"
        assert relation["media_description"] == "portrait illustration"
        assert relation["calibration_method"] == MEDIA_CALIBRATION_METHOD
        assert len(strengths) == result["chunk_count"]
        assert float(strengths[0]["semantic_strength"]) == pytest.approx(1.0)
        assert float(strengths[-1]["semantic_strength"]) < 1.0
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_three_images_keep_independent_descriptions_strengths_and_outputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "library"
    source = tmp_path / "source.md"
    source.write_text(
        "# Hero\n\n## Face\n\nFace portrait avatar with red eyes.\n\n"
        "## Armor\n\nFull armor outfit illustration.\n\n"
        "## Battle\n\nBattle action with a sword.\n\n"
        "## Cats\n\nThe hero feeds a stray cat.",
        encoding="utf-8",
    )
    image_specs = [
        ("avatar.png", (640, 640), "face portrait avatar"),
        ("full.png", (700, 900), "full armor outfit"),
        ("battle.png", (900, 700), "battle action sword"),
    ]
    images: list[dict[str, object]] = []
    for filename, size, description in image_specs:
        path = tmp_path / filename
        path.write_bytes(_png_bytes(size))
        images.append(
            {
                "path": str(path),
                "filename": filename,
                "document_indexes": [0],
                "media_description": description,
            }
        )

    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="multi-image",
        name="Multi image",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    service = TextMediaService(
        root, storage, TextMediaIndex(root), MultiImageFixtureProvider()
    )
    package = tmp_path / "multi-image.tmkb"
    expected_chunk_count = 0
    try:
        result = await service.ingest_batch(
            batch_id="multi-image-1",
            documents=[{"path": str(source), "filename": source.name}],
            images=images,
            chunk_target=220,
            chunk_overlap=40,
            embedding_batch_size=4,
            concurrency=2,
            max_retries=2,
            media_semantic_calibration_enabled=True,
        )
        assert result["image_count"] == 3
        expected_chunk_count = int(result["chunk_count"])

        db = await storage.pool.acquire()
        try:
            rows = await (
                await db.execute(
                    """SELECT a.original_name,da.media_description,c.text,
                    cms.semantic_strength,cms.calibration_rank
                    FROM chunk_media_strengths cms
                    JOIN assets a ON a.id=cms.asset_id
                    JOIN document_assets da
                      ON da.document_id=cms.document_id AND da.asset_id=cms.asset_id
                    JOIN chunks c ON c.id=cms.chunk_id
                    ORDER BY a.original_name,cms.calibration_rank"""
                )
            ).fetchall()
        finally:
            await db.close()

        by_image: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            by_image.setdefault(str(row["original_name"]), []).append(dict(row))
        assert set(by_image) == {"avatar.png", "full.png", "battle.png"}
        assert all(len(items) == result["chunk_count"] for items in by_image.values())
        assert "## Face" in str(by_image["avatar.png"][0]["text"])
        assert "## Armor" in str(by_image["full.png"][0]["text"])
        assert "## Battle" in str(by_image["battle.png"][0]["text"])
        assert {
            str(items[0]["media_description"]) for items in by_image.values()
        } == {item[2] for item in image_specs}
        rank_orders = {
            name: tuple(str(item["text"]) for item in items)
            for name, items in by_image.items()
        }
        assert len(set(rank_orders.values())) == 3

        for query, expected in (
            ("face portrait avatar", "avatar.png"),
            ("full armor outfit", "full.png"),
            ("battle action sword", "battle.png"),
        ):
            response = await service.search(
                query=query,
                top_k=4,
                media_output_confidence_threshold=0.5,
                media_score_threshold=0.5,
            )
            outputs = {str(item["original_name"]) for item in response["media_outputs"]}
            assert outputs == {expected}
            associated = [
                media
                for item in response["items"]
                for media in item["associated_media"]
                if media["original_name"] == expected
            ]
            assert associated
            assert all(media["media_description"] for media in associated)
        await export_tmkb(service=service, target=package)
    finally:
        await service.close()

    context = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "restored-state",
        config=AppConfig(api_key="target", session_secret="target-session"),
        configure_logs=False,
    )
    await context.manager.initialize()
    try:
        await install_tmkb(
            manager=context.manager._managers[TEXT_MEDIA_V1_TYPE],
            package_path=package,
            target_id="multi-image-restored",
            name_override="Multi image restored",
        )
        restored = await context.manager.get_runtime(
            DatabaseRef(TEXT_MEDIA_V1_TYPE, "multi-image-restored")
        )
        db = await restored.storage.pool.acquire()
        try:
            restored_rows = await (
                await db.execute(
                    """SELECT a.original_name,da.media_description,
                    COUNT(cms.chunk_id) AS strength_count
                    FROM document_assets da
                    JOIN assets a ON a.id=da.asset_id
                    JOIN chunk_media_strengths cms
                      ON cms.document_id=da.document_id AND cms.asset_id=da.asset_id
                    GROUP BY a.original_name,da.media_description
                    ORDER BY a.original_name"""
                )
            ).fetchall()
        finally:
            await db.close()
        assert {
            (str(row["original_name"]), str(row["media_description"]))
            for row in restored_rows
        } == {(item[0], item[2]) for item in image_specs}
        assert {
            int(row["strength_count"]) for row in restored_rows
        } == {expected_chunk_count}
    finally:
        await context.manager.close()


def test_multi_media_description_validation_and_legacy_contract() -> None:
    assert media_description_list(
        ["  ＤｅｅｐＳｅｅｋ\n表情包  ", "深度求索表情包"]
    ) == ["DeepSeek 表情包", "深度求索表情包"]
    assert normalize_media_description("  GPT\n表情包 ") == "gpt 表情包"

    with pytest.raises(ValueError, match="duplicate"):
        media_description_list(["Google 大模型", "ｇｏｏｇｌｅ   大模型"])
    with pytest.raises(ValueError, match="at most 20"):
        media_description_list([f"description {index}" for index in range(21)])
    with pytest.raises(ValueError, match="2000"):
        MediaDescriptionsUpdateRequest(media_descriptions=["x" * 2001])
    with pytest.raises(ValueError, match="must equal"):
        IngestImageMapping(
            media_description="DeepSeek 表情包",
            media_descriptions=["Gemini 表情包"],
        )

    compatible = IngestImageMapping(
        media_description="  ＤｅｅｐＳｅｅｋ\n表情包 ",
        media_descriptions=["DeepSeek 表情包", "深度求索表情包"],
    )
    assert compatible.media_descriptions == [
        "DeepSeek 表情包",
        "深度求索表情包",
    ]


def test_multi_media_description_calibration_takes_per_chunk_max_without_averaging() -> None:
    descriptions = [
        {
            "media_description": "full-body armor",
            "sort_order": 0,
        },
        {
            "media_description": "red-eyed face portrait",
            "sort_order": 1,
        },
    ]
    rows = aggregate_media_description_calibrations(
        descriptions,
        [
            [
                {
                    "semantic_strength": 0.90,
                    "rerank_semantic_strength": 0.30,
                    "calibration_details": {"rerank_applied": True},
                },
                {
                    "semantic_strength": 0.20,
                    "rerank_semantic_strength": 0.85,
                    "calibration_details": {"rerank_applied": True},
                },
            ],
            [
                {
                    "semantic_strength": 0.40,
                    "rerank_semantic_strength": 0.95,
                    "calibration_details": {"rerank_applied": True},
                },
                {
                    "semantic_strength": 0.80,
                    "rerank_semantic_strength": 0.25,
                    "calibration_details": {"rerank_applied": True},
                },
            ],
        ],
    )
    assert [item["semantic_strength"] for item in rows] == [0.90, 0.80]
    assert [item["rerank_semantic_strength"] for item in rows] == [
        0.95,
        0.85,
    ]
    assert rows[0]["calibration_details"][
        "embedding_winner_description"
    ] == "full-body armor"
    assert rows[0]["calibration_details"][
        "rerank_winner_description"
    ] == "red-eyed face portrait"
    assert rows[1]["calibration_details"][
        "embedding_winner_description"
    ] == "red-eyed face portrait"
    assert rows[1]["calibration_details"][
        "rerank_winner_description"
    ] == "full-body armor"


@pytest.mark.asyncio
async def test_multi_media_descriptions_collapse_by_asset_and_primary_is_score_neutral(
    tmp_path: Path,
) -> None:
    root = tmp_path / "multi-description-search"
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="multi-description-search",
        name="Multi description search",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    service = TextMediaService(
        root,
        storage,
        TextMediaIndex(root),
        MultiDescriptionFixtureProvider(),
    )
    deepseek_descriptions = [
        "DeepSeek 劣等模型表情包",
        "深度求索表情包",
        "DeepSeek 大模型梗图",
    ]
    gemini_descriptions = [
        "Gemini 劣等模型表情包",
        "Google 大模型表情包",
        "谷歌大模型表情包",
    ]
    try:
        deepseek = await service.upload_image(
            filename="deepseek.png",
            data=_png_bytes((801, 601)),
            media_descriptions=deepseek_descriptions[:2],
        )
        duplicate_upload = await service.upload_image(
            filename="deepseek-copy.png",
            data=_png_bytes((801, 601)),
            media_descriptions=[
                deepseek_descriptions[0],
                deepseek_descriptions[2],
            ],
        )
        assert duplicate_upload["id"] == deepseek["id"]
        assert duplicate_upload["media_descriptions"] == deepseek_descriptions
        gemini = await service.upload_image(
            filename="gemini.png",
            data=_png_bytes((802, 602)),
            media_descriptions=gemini_descriptions,
        )
        generation, description_ids, vectors = (
            await storage.active_media_vectors()
        )
        assert generation
        assert len(description_ids) == 6
        assert vectors.shape == (6, 5)

        before = await service.search(
            "给我深度求索表情包",
            retrieval_mode="media_only",
            rerank=False,
            media_output_confidence_threshold=0,
            max_media_outputs=5,
        )
        assert before["media_outputs"][0]["asset_id"] == deepseek["id"]
        target_before = next(
            item
            for item in before["media_decisions"]
            if item["asset_id"] == deepseek["id"]
        )
        assert target_before["matched_media_description"] == "深度求索表情包"
        assert target_before["matched_media_description_sort_order"] == 1
        assert target_before["media_description_count"] == 3
        assert len(target_before["direct_relations"]) == 3
        assert len(
            {
                item["asset_id"] for item in before["media_decisions"]
            }
        ) == 2
        baseline_confidence = target_before["output_confidence"]
        baseline_association = target_before["association_score"]

        reordered = [
            "深度求索表情包",
            "DeepSeek 劣等模型表情包",
            "DeepSeek 大模型梗图",
        ]
        await service.update_asset_media_descriptions(
            asset_id=str(deepseek["id"]),
            media_descriptions=reordered,
        )
        after = await service.search(
            "给我深度求索表情包",
            retrieval_mode="media_only",
            rerank=False,
            media_output_confidence_threshold=0,
            max_media_outputs=5,
        )
        target_after = next(
            item
            for item in after["media_decisions"]
            if item["asset_id"] == deepseek["id"]
        )
        assert after["media_outputs"][0]["asset_id"] == deepseek["id"]
        assert target_after["media_description"] == "深度求索表情包"
        assert target_after["matched_media_description"] == "深度求索表情包"
        assert target_after["matched_media_description_sort_order"] == 0
        assert target_after["output_confidence"] == pytest.approx(
            baseline_confidence, abs=1e-6
        )
        assert target_after["association_score"] == pytest.approx(
            baseline_association, abs=1e-6
        )

        await service.update_asset_media_descriptions(
            asset_id=str(deepseek["id"]),
            media_descriptions=[
                *reordered,
                "a completely unrelated viewing angle",
            ],
        )
        with_extra = await service.search(
            "给我深度求索表情包",
            retrieval_mode="media_only",
            rerank=False,
            media_output_confidence_threshold=0,
            max_media_outputs=5,
        )
        target_with_extra = next(
            item
            for item in with_extra["media_decisions"]
            if item["asset_id"] == deepseek["id"]
        )
        assert target_with_extra["output_confidence"] == pytest.approx(
            baseline_confidence, abs=1e-6
        )
        assert target_with_extra["association_score"] == pytest.approx(
            baseline_association, abs=1e-6
        )
        assert target_with_extra["media_description_count"] == 4

        detail_before_failure = await storage.get_asset_detail(
            str(deepseek["id"])
        )
        with pytest.raises(ValueError, match="duplicate"):
            await service.update_asset_media_descriptions(
                asset_id=str(deepseek["id"]),
                media_descriptions=[
                    "深度求索表情包",
                    "  深度求索表情包 ",
                ],
            )
        detail_after_failure = await storage.get_asset_detail(
            str(deepseek["id"])
        )
        assert detail_after_failure["media_descriptions"] == (
            detail_before_failure["media_descriptions"]
        )
        assert gemini["id"] != deepseek["id"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_multi_media_description_rerank_budget_is_fair_and_failure_falls_back(
    tmp_path: Path,
) -> None:
    root = tmp_path / "multi-description-rerank"
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="multi-description-rerank",
        name="Multi description rerank",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
        rerank_provider_id="fixture-rerank",
        rerank_provider_revision=1,
        rerank_provider_fingerprint="rerank-sha",
    )
    reranker = FixtureReranker()
    service = TextMediaService(
        root,
        storage,
        TextMediaIndex(root),
        MultiDescriptionFixtureProvider(),
        reranker,
        {
            "id": "fixture-rerank",
            "revision": 1,
            "fingerprint": "rerank-sha",
        },
    )
    try:
        await storage.update_metadata(
            {
                "retrieval_config_json": {
                    **DEFAULT_RETRIEVAL_CONFIG,
                    "rerank_candidate_limit": 10,
                }
            }
        )
        for index, prefix in enumerate(("alpha", "beta"), start=1):
            await service.upload_image(
                filename=f"{prefix}.png",
                data=_png_bytes((810 + index, 610 + index)),
                media_descriptions=[
                    f"{prefix} model angle {order:02d}"
                    for order in range(10)
                ],
            )
        await service.upload_image(
            filename="gamma.png",
            data=_png_bytes((813, 613)),
            media_descriptions=["gamma model only angle"],
        )

        baseline = await service.search(
            "model",
            retrieval_mode="media_only",
            rerank=False,
            media_output_confidence_threshold=0,
        )
        reranked = await service.search(
            "model",
            retrieval_mode="media_only",
            rerank=True,
            media_output_confidence_threshold=0,
        )
        assert reranked["rerank"]["applied"] is True
        assert len(reranker.calls) == 1
        reranked_documents = reranker.calls[0][1]
        alpha_count = sum(
            value.startswith("alpha ") for value in reranked_documents
        )
        beta_count = sum(
            value.startswith("beta ") for value in reranked_documents
        )
        gamma_count = sum(
            value.startswith("gamma ") for value in reranked_documents
        )
        assert alpha_count in {3, 4}
        assert beta_count in {3, 4}
        assert abs(alpha_count - beta_count) <= 1
        assert gamma_count == 1
        assert len(reranked_documents) in {7, 8}
        assert sorted(
            len(item["direct_relations"])
            for item in reranked["media_decisions"]
        ) == [1, 10, 10]

        await service.set_rerank_provider(
            FixtureReranker(malformed=True),
            {
                "id": "fixture-rerank",
                "revision": 1,
                "fingerprint": "rerank-sha",
            },
        )
        fallback = await service.search(
            "model",
            retrieval_mode="media_only",
            rerank=True,
            media_output_confidence_threshold=0,
        )
        assert fallback["rerank"]["failed"] is True
        assert fallback["rerank"]["fallback"] is True
        assert [
            (
                item["asset_id"],
                item["output_confidence"],
                item["association_score"],
                item["matched_media_description_id"],
            )
            for item in fallback["media_decisions"]
        ] == [
            (
                item["asset_id"],
                item["output_confidence"],
                item["association_score"],
                item["matched_media_description_id"],
            )
            for item in baseline["media_decisions"]
        ]
        assert all(
            item.get("rerank_raw_score") is None
            for decision in fallback["media_decisions"]
            for item in decision["direct_relations"]
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_schema_8_to_9_migrates_relation_descriptions_without_provider_calls(
    tmp_path: Path,
) -> None:
    root = tmp_path / "schema-8-to-9"
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="schema-8-to-9",
        name="Schema 8 to 9",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    service = TextMediaService(
        root,
        storage,
        TextMediaIndex(root),
        MultiDescriptionFixtureProvider(),
    )
    document = await service.ingest_document(
        filename="deepseek.md",
        title="DeepSeek",
        data=("DeepSeek model and 深度求索 model. " * 40).encode(),
    )
    image = await service.upload_image(
        filename="deepseek.png",
        data=_png_bytes((821, 621)),
        media_descriptions=["DeepSeek 劣等模型表情包"],
    )
    await storage.link_asset(
        scope="document",
        target_id=document["document_id"],
        asset_id=image["id"],
        payload={"output_policy": "auto"},
    )
    await service.recalibrate_document_media(
        document_id=document["document_id"],
        asset_id=image["id"],
        enabled=True,
        media_description="DeepSeek 劣等模型表情包",
    )
    original = (await storage.asset_media_descriptions(image["id"]))[0]
    vector_blob = np.asarray(
        original["media_description_vector"], dtype="<f4"
    ).tobytes()
    await service.close()

    path = root / "textmediaknowledge.db"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA foreign_keys=OFF")
        db.executescript(
            """
            DROP TRIGGER IF EXISTS asset_media_metadata_ai;
            DROP TRIGGER IF EXISTS asset_media_metadata_ad;
            DROP TRIGGER IF EXISTS asset_media_metadata_au;
            DROP TABLE IF EXISTS asset_media_metadata_fts;
            ALTER TABLE asset_media_metadata RENAME TO asset_media_metadata_v9;
            CREATE TABLE asset_media_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id TEXT NOT NULL UNIQUE
                    REFERENCES assets(id) ON DELETE CASCADE,
                media_description TEXT NOT NULL DEFAULT '',
                search_text TEXT NOT NULL DEFAULT '',
                description_source TEXT NOT NULL DEFAULT 'filename',
                media_description_vector BLOB,
                vector_sha256 TEXT NOT NULL DEFAULT '',
                provider_id TEXT NOT NULL DEFAULT '',
                provider_revision INTEGER NOT NULL DEFAULT 0,
                provider_fingerprint TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            INSERT INTO asset_media_metadata
            (id,asset_id,media_description,search_text,description_source,
             media_description_vector,vector_sha256,provider_id,
             provider_revision,provider_fingerprint,created_at,updated_at)
            SELECT id,asset_id,media_description,search_text,
                   description_source,media_description_vector,vector_sha256,
                   provider_id,provider_revision,provider_fingerprint,
                   created_at,updated_at
            FROM asset_media_metadata_v9 WHERE sort_order=0;
            DROP TABLE asset_media_metadata_v9;
            UPDATE schema_info SET value='8' WHERE key='schema_version';
            """
        )
        db.execute(
            """UPDATE document_assets SET media_description=?,
            media_description_vector=?,calibration_provider_fingerprint=?,
            calibration_description_set_sha256='' WHERE document_id=? AND asset_id=?""",
            (
                "深度求索表情包",
                vector_blob,
                "fixture-sha",
                document["document_id"],
                image["id"],
            ),
        )
        db.commit()

    migrated = TextMediaStorage(root)
    try:
        await migrated.initialize()
        descriptions = await migrated.asset_media_descriptions(image["id"])
        assert [
            item["media_description"] for item in descriptions
        ] == ["DeepSeek 劣等模型表情包", "深度求索表情包"]
        assert [item["sort_order"] for item in descriptions] == [0, 1]
        assert [item["vector_status"] for item in descriptions] == [
            "ready",
            "ready",
        ]
        db = await migrated.pool.acquire()
        try:
            version = await (
                await db.execute(
                    "SELECT value FROM schema_info WHERE key='schema_version'"
                )
            ).fetchone()
            relation = await (
                await db.execute(
                    """SELECT media_description,media_description_vector,
                    calibration_description_set_sha256
                    FROM document_assets WHERE document_id=? AND asset_id=?""",
                    (document["document_id"], image["id"]),
                )
            ).fetchone()
        finally:
            await db.close()
        assert version[0] == "9"
        assert relation["media_description"] == "DeepSeek 劣等模型表情包"
        assert bytes(relation["media_description_vector"]) == vector_blob
        assert relation["calibration_description_set_sha256"] == ""
    finally:
        await migrated.close()


@pytest.mark.asyncio
async def test_multi_media_description_update_rolls_back_on_rerank_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "description-update-rollback"
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="description-update-rollback",
        name="Description update rollback",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
        rerank_provider_id="fixture-rerank",
        rerank_provider_revision=1,
        rerank_provider_fingerprint="rerank-sha",
    )
    service = TextMediaService(
        root,
        storage,
        TextMediaIndex(root),
        CalibrationFixtureProvider(),
        FixtureReranker(),
        {
            "id": "fixture-rerank",
            "revision": 1,
            "fingerprint": "rerank-sha",
        },
    )
    try:
        document = await service.ingest_document(
            filename="portrait.md",
            title="Portrait",
            data=("portrait armor illustration. " * 80).encode(),
        )
        image = await service.upload_image(
            filename="portrait.png",
            data=_png_bytes((831, 631)),
            media_descriptions=["portrait armor illustration"],
        )
        await storage.link_asset(
            scope="document",
            target_id=document["document_id"],
            asset_id=image["id"],
            payload={"output_policy": "auto"},
        )
        await service.recalibrate_document_media(
            document_id=document["document_id"],
            asset_id=image["id"],
            enabled=True,
            media_description="portrait armor illustration",
        )
        updated = await service.update_asset_media_descriptions(
            asset_id=image["id"],
            media_descriptions=[
                "portrait armor illustration",
                "full-body black armor standing pose",
            ],
        )
        assert updated["description_count"] == 2
        assert updated["recalibrated_relation_count"] == 1
        db = await storage.pool.acquire()
        try:
            relation_projection = await (
                await db.execute(
                    """SELECT media_description,
                    calibration_description_set_sha256
                    FROM document_assets WHERE document_id=? AND asset_id=?""",
                    (document["document_id"], image["id"]),
                )
            ).fetchone()
            strength_diagnostic = await (
                await db.execute(
                    """SELECT calibration_details_json
                    FROM chunk_media_strengths
                    WHERE document_id=? AND asset_id=? ORDER BY chunk_id LIMIT 1""",
                    (document["document_id"], image["id"]),
                )
            ).fetchone()
        finally:
            await db.close()
        assert relation_projection["media_description"] == (
            "portrait armor illustration"
        )
        assert len(
            relation_projection["calibration_description_set_sha256"]
        ) == 64
        assert json.loads(strength_diagnostic["calibration_details_json"])[
            "description_count"
        ] == 2
        descriptions_before = [
            (
                item["id"],
                item["media_description"],
                item["sort_order"],
                np.asarray(
                    item["media_description_vector"], dtype="<f4"
                ).tobytes(),
            )
            for item in await storage.asset_media_descriptions(image["id"])
        ]
        calibrations_before = (
            await storage.list_document_media_calibrations()
        )
        await service.set_rerank_provider(
            FixtureReranker(fail=True),
            {
                "id": "fixture-rerank",
                "revision": 1,
                "fingerprint": "rerank-sha",
            },
        )
        with pytest.raises(RuntimeError, match="fixture rerank failure"):
            await service.update_asset_media_descriptions(
                asset_id=image["id"],
                media_descriptions=[
                    "portrait armor illustration",
                    "full-body black armor standing pose",
                    "red-eyed face portrait",
                ],
            )
        descriptions_after = [
            (
                item["id"],
                item["media_description"],
                item["sort_order"],
                np.asarray(
                    item["media_description_vector"], dtype="<f4"
                ).tobytes(),
            )
            for item in await storage.asset_media_descriptions(image["id"])
        ]
        assert descriptions_after == descriptions_before
        assert await storage.list_document_media_calibrations() == (
            calibrations_before
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_failed_ingest_batch_leaves_no_content_or_assets(tmp_path: Path) -> None:
    root = tmp_path / "library"
    source = tmp_path / "source.md"
    image_path = tmp_path / "source.png"
    source.write_text("failed batch content" * 80, encoding="utf-8")
    image_path.write_bytes(_png_bytes((800, 600)))
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="batch-failure",
        name="Batch failure",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    service = TextMediaService(
        root, storage, TextMediaIndex(root), FailingFixtureProvider()
    )
    try:
        with pytest.raises(RuntimeError, match="Embedding"):
            await service.ingest_batch(
                batch_id="missing-image",
                documents=[
                    {"path": str(source), "filename": source.name, "title": "Failure"}
                ],
                images=[],
                chunk_target=240,
                chunk_overlap=40,
                embedding_batch_size=4,
                concurrency=2,
                max_retries=1,
            )
        with pytest.raises(RuntimeError, match="Embedding"):
            await service.ingest_batch(
                batch_id="batch-failure-1",
                documents=[
                    {"path": str(source), "filename": source.name, "title": "Failure"}
                ],
                images=[
                    {
                        "path": str(image_path),
                        "filename": image_path.name,
                        "document_indexes": [0],
                    }
                ],
                chunk_target=240,
                chunk_overlap=40,
                embedding_batch_size=4,
                concurrency=2,
                max_retries=1,
            )
        assert await storage.statistics() == {
            "documents": 0,
            "entries": 0,
            "chunks": 0,
            "images": 0,
        }
        db = await storage.pool.acquire()
        try:
            assert int((await (await db.execute("SELECT COUNT(*) FROM ingest_batches")).fetchone())[0]) == 0
            assert int((await (await db.execute("SELECT COUNT(*) FROM assets")).fetchone())[0]) == 0
        finally:
            await db.close()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_ingest_batch_index_failure_rolls_back_database_and_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "library"
    source = tmp_path / "source.md"
    image_path = tmp_path / "source.png"
    source.write_text("illustration details " * 120, encoding="utf-8")
    image_path.write_bytes(_png_bytes((800, 600)))
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="index-failure",
        name="Index failure",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    indexes = TextMediaIndex(root)
    service = TextMediaService(root, storage, indexes, FixtureProvider())
    original_rebuild = indexes.rebuild
    rebuild_calls = 0

    async def fail_first_rebuild(current_storage):
        nonlocal rebuild_calls
        rebuild_calls += 1
        if rebuild_calls == 1:
            raise RuntimeError("fixture index failure")
        return await original_rebuild(current_storage)

    monkeypatch.setattr(indexes, "rebuild", fail_first_rebuild)
    try:
        with pytest.raises(RuntimeError, match="index failure"):
            await service.ingest_batch(
                batch_id="index-failure-1",
                documents=[{"path": str(source), "filename": source.name}],
                images=[
                    {
                        "path": str(image_path),
                        "filename": image_path.name,
                        "document_indexes": [0],
                    }
                ],
                chunk_target=240,
                chunk_overlap=40,
                embedding_batch_size=4,
                concurrency=2,
                max_retries=1,
            )
        assert await storage.statistics() == {
            "documents": 0,
            "entries": 0,
            "chunks": 0,
            "images": 0,
        }
        assert not list((root / "assets" / "documents").rglob("*.md"))
        assert not list((root / "assets" / "images").rglob("*.webp"))
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_corrupt_image_and_cancelled_ingest_leave_no_batch_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "library"
    source = tmp_path / "source.md"
    broken = tmp_path / "broken.png"
    valid = tmp_path / "valid.png"
    source.write_text("illustration details " * 120, encoding="utf-8")
    broken.write_bytes(b"not an image")
    valid.write_bytes(_png_bytes((800, 600)))
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="cancelled",
        name="Cancelled",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    service = TextMediaService(root, storage, TextMediaIndex(root), FixtureProvider())
    try:
        with pytest.raises(ImageValidationError):
            await service.ingest_batch(
                batch_id="broken-image",
                documents=[{"path": str(source), "filename": source.name}],
                images=[
                    {
                        "path": str(broken),
                        "filename": broken.name,
                        "document_indexes": [0],
                    }
                ],
                chunk_target=240,
                chunk_overlap=40,
                embedding_batch_size=4,
                concurrency=1,
                max_retries=1,
            )
        service.provider = BlockingFixtureProvider()
        task = asyncio.create_task(
            service.ingest_batch(
                batch_id="cancelled-batch",
                documents=[{"path": str(source), "filename": source.name}],
                images=[
                    {
                        "path": str(valid),
                        "filename": valid.name,
                        "document_indexes": [0],
                    }
                ],
                chunk_target=240,
                chunk_overlap=40,
                embedding_batch_size=4,
                concurrency=1,
                max_retries=1,
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await storage.statistics() == {
            "documents": 0,
            "entries": 0,
            "chunks": 0,
            "images": 0,
        }
        db = await storage.pool.acquire()
        try:
            assert int((await (await db.execute("SELECT COUNT(*) FROM ingest_batches")).fetchone())[0]) == 0
        finally:
            await db.close()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_storage_index_search_and_multilevel_asset_relation(tmp_path: Path) -> None:
    root = tmp_path / "library"
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="demo",
        name="Demo",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    provider = FixtureProvider()
    indexes = TextMediaIndex(root)
    service = TextMediaService(root, storage, indexes, provider)
    try:
        result = await service.ingest_document(
            filename="角色立绘.md",
            title="角色立绘",
            data="角色立绘使用粉色服装。\n\n这是固定知识。".encode(),
        )
        image = await service.upload_image(filename="source.png", data=_png_bytes((800, 600)))
        await storage.link_asset(
            scope="entry",
            target_id=result["entry_id"],
            asset_id=image["id"],
            payload={"caption": "角色立绘", "output_policy": "with_result"},
        )
        response = await service.search("角色立绘", top_k=5)
        assert response["items"]
        item = response["items"][0]
        assert item["title"] == "角色立绘"
        assert item["associated_media"][0]["asset_id"] == image["id"]
        assert item["associated_media"][0]["scope"] == "entry"
        assert response["media_outputs"][0]["asset_id"] == image["id"]

        await storage.link_asset(
            scope="entry",
            target_id=result["entry_id"],
            asset_id=image["id"],
            payload={"caption": "角色立绘", "output_policy": "disabled"},
        )
        hidden_response = await service.search("角色立绘", top_k=5)
        assert hidden_response["items"][0]["associated_media"] == []
        assert hidden_response["media_outputs"] == []

        generation, ids, vectors = await storage.active_vectors()
        assert generation == result["generation_id"]
        assert ids == result["chunk_ids"]
        assert vectors.dtype == np.float32
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
        assert (await storage.validate())["integrity"] == "ok"

        assert await storage.unlink_asset(
            scope="entry", target_id=result["entry_id"], asset_id=image["id"]
        )
        assert not await storage.unlink_asset(
            scope="entry", target_id=result["entry_id"], asset_id=image["id"]
        )
        image_path = root / image["storage_key"]
        await service.delete_image(image["id"])
        assert not image_path.exists()
        await service.delete_entry(result["entry_id"])
        assert await storage.statistics() == {
            "documents": 0,
            "entries": 0,
            "chunks": 0,
            "images": 0,
        }
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_full_query_rerank_and_false_baseline_switch(tmp_path: Path) -> None:
    root = tmp_path / "rerank-library"
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="rerank-demo",
        name="Rerank demo",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
        rerank_provider_id="fixture-rerank",
        rerank_provider_revision=2,
        rerank_provider_fingerprint="rerank-sha",
    )
    reranker = ReverseFixtureReranker()
    service = TextMediaService(
        root,
        storage,
        TextMediaIndex(root),
        CalibrationFixtureProvider(),
        reranker,
        {"id": "fixture-rerank", "revision": 2, "fingerprint": "rerank-sha"},
    )
    try:
        installed = await service.ingest_document(
            filename="portrait.md",
            title="Portrait",
            data=("portrait armor illustration. " * 80).encode(),
        )
        image = await service.upload_image(
            filename="portrait.png",
            data=_png_bytes((800, 600)),
            media_descriptions=["portrait armor illustration"],
        )
        await storage.link_asset(
            scope="document",
            target_id=installed["document_id"],
            asset_id=image["id"],
            payload={
                "caption": "portrait armor illustration",
                "output_policy": "auto",
            },
        )
        await storage.update_metadata(
            {
                "retrieval_config_json": {
                    **DEFAULT_RETRIEVAL_CONFIG,
                    "rerank_fusion_weight": 0.5,
                }
            }
        )

        calls_before = len(reranker.calls)
        baseline = await service.search(
            "show portrait armor",
            top_k=5,
            media_output_confidence_threshold=0,
            rerank=False,
        )
        assert len(reranker.calls) == calls_before
        assert baseline["rerank"]["applied"] is False
        assert baseline["baseline_items"] == baseline["items"]
        assert all(
            item["retrieval_stage"] == "embedding_baseline"
            for item in baseline["baseline_items"]
        )

        reranked = await service.search(
            "show portrait armor",
            top_k=5,
            media_output_confidence_threshold=0,
            rerank=None,
        )
        assert reranked["rerank"]["applied"] is True
        assert reranked["rerank"]["fusion_algorithm"] == (
            "log_odds_absolute_relevance_v8"
        )
        assert reranked["rerank"]["text_reorder_probability_floor"] == (
            pytest.approx(0.0)
        )
        assert reranked["rerank"]["text_ordering"] == (
            "rerank_fused_relevance_desc"
        )
        scopes = {item["scope"] for item in reranked["rerank"]["scopes"]}
        assert "text_chunks" in scopes
        assert "media_descriptions:bound" in scopes
        assert any(value.startswith("media_bound_chunks:") for value in scopes)
        assert [
            (item["chunk_id"], item["score"], item["dense_score"])
            for item in reranked["baseline_items"]
        ] == [
            (item["chunk_id"], item["score"], item["dense_score"])
            for item in baseline["items"]
        ]
        assert all(
            item["retrieval_stage"] == "embedding_baseline"
            and item["rerank_raw_score"] is None
            and item["rerank_rank"] is None
            for item in reranked["baseline_items"]
        )
        assert all(
            item["retrieval_stage"] == "reranked"
            for item in reranked["items"]
        )
        assert [item["chunk_id"] for item in reranked["items"]] != [
            item["chunk_id"] for item in baseline["items"]
        ]
        assert reranked["items"][0]["initial_rank"] > 1
        assert reranked["items"][0]["rerank_rank"] == 1
        assert reranked["items"][0]["rerank_reorder_eligible"] is True
        assert reranked["items"][0]["score_source"] == (
            "rerank_fused_relevance"
        )
        assert reranked["items"][0]["score"] == pytest.approx(
            reranked["items"][0]["score_breakdown"]["ordering_relevance"]
        )
        assert [item["score"] for item in reranked["items"]] != [
            item["score"] for item in baseline["items"]
        ]
        assert reranked["items"][0]["rerank_raw_score"] is not None
        evidence = reranked["media_decisions"][0]["evidence"][0]
        assert evidence["initial_rank"] >= 1
        assert evidence["rerank_raw_score"] is not None
        assert 0 <= evidence["fused_relevance"] <= 1
        assert evidence["ordering_relevance"] != pytest.approx(
            evidence["dense_score"]
        )
        direct = reranked["media_decisions"][0]["direct_relations"][0]
        assert direct["rerank_raw_score"] is not None
        assert 0 <= direct["calibrated_semantic_score"] <= 1
        assert reranked["media_decisions"][0][
            "rerank_output_confidence_delta"
        ] == pytest.approx(
            reranked["media_decisions"][0]["output_confidence"]
            - reranked["media_decisions"][0][
                "rerank_baseline_output_confidence"
            ],
            abs=2e-6,
        )
        calls_after_rerank = len(reranker.calls)
        baseline_again = await service.search(
            "show portrait armor",
            top_k=5,
            media_output_confidence_threshold=0,
            rerank=False,
        )
        assert len(reranker.calls) == calls_after_rerank
        assert baseline_again["items"] == baseline["items"]
        assert baseline_again["media_outputs"] == baseline["media_outputs"]
        assert baseline_again["media_decisions"] == baseline["media_decisions"]
        await service.set_rerank_provider(
            LowScoreReverseFixtureReranker(),
            {"id": "fixture-rerank", "revision": 2, "fingerprint": "rerank-sha"},
        )
        low_confidence = await service.search(
            "show portrait armor",
            top_k=5,
            media_output_confidence_threshold=0,
            rerank=True,
        )
        assert {item["chunk_id"] for item in low_confidence["items"]} == {
            item["chunk_id"] for item in baseline["items"]
        }
        assert max(item["score"] for item in low_confidence["items"]) < max(
            item["score"] for item in baseline["items"]
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_reference_generation_uses_original_text_query_and_projected_media_query(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reference-generation"
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="reference-generation",
        name="Reference generation",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
        rerank_provider_id="fixture-rerank",
        rerank_provider_revision=1,
        rerank_provider_fingerprint="rerank-sha",
    )
    provider = RecordingIntentFixtureProvider()
    reranker = FixtureReranker()
    service = TextMediaService(
        root,
        storage,
        TextMediaIndex(root),
        provider,
        reranker,
        {"id": "fixture-rerank", "revision": 1, "fingerprint": "rerank-sha"},
    )
    try:
        installed = await service.ingest_document(
            filename="portrait.md",
            title="澄月",
            data=("# 澄月\n\n## 立绘\n\n澄月穿着黑红铠甲的全身立绘。" * 20).encode(),
        )
        image = await service.upload_image(
            filename="澄月全身立绘.png",
            data=_png_bytes((800, 1200)),
            media_descriptions=["澄月穿着黑红铠甲的全身立绘"],
        )
        await storage.link_asset(
            scope="document",
            target_id=installed["document_id"],
            asset_id=image["id"],
            payload={"output_policy": "auto", "relation_weight": 1.0},
        )
        provider.calls.clear()
        reranker.calls.clear()

        original_query = "先看看你的立绘，再画一张q版的"
        response = await service.search(
            original_query,
            top_k=5,
            media_output_confidence_threshold=0,
            rerank=True,
        )
        assert response["query"] == original_query
        assert response["media_query"] == "看看你的立绘"
        assert response["media_query_projection_applied"] is True
        assert response["query_embedding_count"] == 2
        assert provider.calls[0] == [original_query, "看看你的立绘"]
        rerank_queries = [query for query, _documents in reranker.calls]
        assert original_query in rerank_queries
        assert "看看你的立绘" in rerank_queries
        assert response["media_outputs"][0]["asset_id"] == image["id"]
        decision = response["media_decisions"][0]
        assert decision["original_query"] == original_query
        assert decision["effective_media_query"] == "看看你的立绘"
        assert decision["visual_intent_kind"] == "reference_generation"
        assert decision["visual_intent_reference_span"] == "你的立绘"
        assert decision["visual_intent_projection_applied"] is True

        provider_call_count = len(provider.calls)
        media_only = await service.search(
            original_query,
            top_k=20,
            media_output_confidence_threshold=0,
            rerank=True,
            retrieval_mode="media_only",
        )
        assert len(provider.calls) == provider_call_count
        assert media_only["query_embedding"]["cache_hits"] == 1
        assert media_only["query_embedding"]["provider_input_count"] == 0
        assert media_only["media_outputs"] == response["media_outputs"]
        assert media_only["media_decisions"] == response["media_decisions"]

        calls_before_update = len(provider.calls)
        index_before = dict(service.indexes.status())
        custom_policy = normalize_visual_intent_policy(
            DEFAULT_VISUAL_INTENT_POLICY
        )
        custom_policy["generation_action_terms"].append("创作")
        update = await service.update_visual_intent_policy(custom_policy)
        assert len(provider.calls) == calls_before_update
        assert update["provider_calls"] == 0
        assert update["index_rebuilt"] is False
        assert update["generation_unchanged"] is True
        assert update["media_generation_unchanged"] is True
        assert service.indexes.status() == index_before

        custom_response = await service.search(
            "根据你的立绘创作一张纸片画",
            top_k=1,
            media_output_confidence_threshold=0,
            rerank=False,
        )
        assert custom_response["media_query"] == "看看你的立绘"
        assert custom_response["visual_intent"]["intent_kind"] == (
            "reference_generation"
        )

        restored = await service.update_visual_intent_policy(
            DEFAULT_VISUAL_INTENT_POLICY
        )
        assert restored["visual_intent_policy_is_default"] is True
        stored = json.loads(
            str((await storage.metadata())["retrieval_config_json"])
        )
        assert "visual_intent_policy" not in stored
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", [False, True])
async def test_query_rerank_failure_replays_whole_baseline(
    tmp_path: Path, malformed: bool
) -> None:
    root = tmp_path / "rerank-fallback"
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="rerank-fallback",
        name="Rerank fallback",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
        rerank_provider_id="fixture-rerank",
        rerank_provider_revision=1,
        rerank_provider_fingerprint="rerank-sha",
    )
    reranker = FixtureReranker(fail=not malformed, malformed=malformed)
    service = TextMediaService(
        root,
        storage,
        TextMediaIndex(root),
        FixtureProvider(),
        reranker,
        {"id": "fixture-rerank", "revision": 1, "fingerprint": "rerank-sha"},
    )
    try:
        await service.ingest_document(
            filename="portrait.md",
            title="Portrait",
            data=("角色立绘使用粉色服装。\n\n" * 300).encode(),
        )
        baseline = await service.search("角色立绘", top_k=5, rerank=False)
        fallback = await service.search("角色立绘", top_k=5, rerank=True)
        assert fallback["items"] == baseline["items"]
        assert fallback["media_outputs"] == baseline["media_outputs"]
        assert fallback["media_decisions"] == baseline["media_decisions"]
        assert fallback["rerank"]["failed"] is True
        assert fallback["rerank"]["fallback"] is True
        assert fallback["rerank"]["applied"] is False
        assert fallback["rerank"]["discarded_partial_results"] is True
        assert not service._rerank_cache
        if malformed:
            assert "invalid candidate set" in fallback["rerank"]["fallback_reason"]
        assert reranker.calls
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_media_only_rerank_failure_replays_unbound_media_baseline(
    tmp_path: Path,
) -> None:
    root = tmp_path / "media-only-rerank-fallback"
    image = tmp_path / "unbound.png"
    image.write_bytes(_png_bytes((720, 480)))
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="media-only-rerank-fallback",
        name="Media-only rerank fallback",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
        rerank_provider_id="fixture-rerank",
        rerank_provider_revision=1,
        rerank_provider_fingerprint="rerank-sha",
    )
    reranker = FixtureReranker(fail=True)
    service = TextMediaService(
        root,
        storage,
        TextMediaIndex(root),
        FixtureProvider(),
        reranker,
        {"id": "fixture-rerank", "revision": 1, "fingerprint": "rerank-sha"},
    )
    try:
        await service.ingest_batch(
            batch_id="unbound-media",
            documents=[],
            images=[
                {
                    "path": str(image),
                    "filename": "Claude版劣等模型表情包.png",
                    "document_indexes": [],
                    "media_description": "Claude版劣等模型表情包",
                }
            ],
            chunk_target=1200,
            chunk_overlap=150,
            embedding_batch_size=8,
            concurrency=1,
            max_retries=1,
            media_semantic_calibration_enabled=True,
        )
        assert reranker.calls == []
        baseline = await service.search(
            "Claude版劣等模型表情包",
            retrieval_mode="media_only",
            rerank=False,
            media_output_confidence_threshold=0,
        )
        fallback = await service.search(
            "Claude版劣等模型表情包",
            retrieval_mode="media_only",
            rerank=True,
            media_output_confidence_threshold=0,
        )
        assert fallback["items"] == fallback["baseline_items"] == []
        assert fallback["media_outputs"] == baseline["media_outputs"]
        assert fallback["media_decisions"] == baseline["media_decisions"]
        assert fallback["rerank"]["failed"] is True
        assert fallback["rerank"]["fallback"] is True
        assert fallback["rerank"]["applied"] is False
        assert reranker.calls
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_rerank_media_calibration_is_dual_path_and_atomic(tmp_path: Path) -> None:
    root = tmp_path / "rerank-calibration"
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="rerank-calibration",
        name="Rerank calibration",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
        rerank_provider_id="fixture-rerank",
        rerank_provider_revision=1,
        rerank_provider_fingerprint="rerank-sha",
    )
    service = TextMediaService(
        root,
        storage,
        TextMediaIndex(root),
        CalibrationFixtureProvider(),
        FixtureReranker(),
        {"id": "fixture-rerank", "revision": 1, "fingerprint": "rerank-sha"},
    )
    try:
        installed = await service.ingest_document(
            filename="portrait.md",
            title="Portrait",
            data=("portrait armor. " * 120).encode(),
        )
        image = await service.upload_image(
            filename="portrait.png",
            data=_png_bytes((800, 600)),
            media_descriptions=["portrait armor illustration"],
        )
        await storage.link_asset(
            scope="document",
            target_id=installed["document_id"],
            asset_id=image["id"],
            payload={"output_policy": "auto"},
        )
        calibrated = await service.recalibrate_document_media(
            document_id=installed["document_id"],
            asset_id=image["id"],
            enabled=True,
            media_description="portrait armor illustration",
        )
        assert calibrated["rerank_provider_fingerprint"] == "rerank-sha"
        assert calibrated["rerank_calibrated_chunk_count"] == calibrated[
            "chunk_count"
        ]
        assert calibrated["rerank_changed_chunk_count"] > 0
        assert calibrated["maximum_rerank_strength_delta"] > 0
        before = await storage.list_document_media_calibrations()
        assert before[0]["calibration_rerank_settings_fingerprints"] == (
            rerank_calibration_settings_fingerprint()
        )
        assert before[0]["minimum_rerank_strength"] != pytest.approx(
            before[0]["minimum_strength"]
        )
        assert before[0]["maximum_rerank_strength"] <= 1
        assert before[0]["maximum_rerank_strength_delta"] > 0
        assert before[0]["rerank_changed_chunk_count"] > 0
        assert before[0]["maximum_rerank_strength_delta"] > 0

        await storage.update_metadata(
            {
                "retrieval_config_json": {
                    **DEFAULT_RETRIEVAL_CONFIG,
                    "rerank_fusion_weight": 0.5,
                }
            }
        )
        stale = await service.search(
            "show portrait armor",
            top_k=5,
            media_output_confidence_threshold=0,
            rerank=True,
        )
        stale_decision = stale["media_decisions"][0]
        assert stale_decision["calibration_strength_source"] == "embedding_baseline"
        assert stale_decision["rerank_calibration_stale"] is True

        await service.set_rerank_provider(
            FixtureReranker(fail=True),
            {"id": "fixture-rerank", "revision": 1, "fingerprint": "rerank-sha"},
        )
        with pytest.raises(RuntimeError, match="fixture rerank failure"):
            await service.recalibrate_document_media(
                document_id=installed["document_id"],
                asset_id=image["id"],
                enabled=True,
                media_description="portrait armor illustration",
            )
        assert await storage.list_document_media_calibrations() == before
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_search_returns_chunk_top_k_and_aggregates_ranked_media_confidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "library"
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="media-search",
        name="Media search",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    service = TextMediaService(root, storage, TextMediaIndex(root), FixtureProvider())
    try:
        result = await service.ingest_document(
            filename="long.md",
            title="Long entry",
            data=(("illustration armor background " * 50 + "\n\n") * 35).encode(),
        )
        image = await service.upload_image(
            filename="source.png", data=_png_bytes((800, 600))
        )
        await storage.link_asset(
            scope="document",
            target_id=result["document_id"],
            asset_id=image["id"],
            payload={"relation_weight": 1.0, "output_policy": "auto"},
        )
        response = await service.search(
            "illustration armor",
            top_k=5,
            media_output_confidence_threshold=0.6,
            media_score_threshold=0.45,
        )
        assert len(response["items"]) == 5
        assert len({item["entry_id"] for item in response["items"]}) == 1
        assert "required_chunk_count" not in response["thresholds"]
        assert response["media_outputs"][0]["qualifying_chunk_count"] >= 5
        assert response["media_outputs"][0]["weakening_chunk_count"] >= 0
        assert response["media_outputs"][0]["best_rank"] == 1
        assert response["media_outputs"][0]["output_confidence"] >= 0.6
        assert response["media_outputs"][0]["reason"] == "confidence_met"
        assert response["media_outputs"][0]["media_score_threshold"] == 0.45
        assert response["media_outputs"][0]["output_confidence_threshold"] == 0.6
        assert response["media_outputs"][0]["confidence_threshold_met"] is True
        assert response["media_decisions"][0]["evidence"][0]["rank_weight"] == 1.0

        strict = await service.search(
            "illustration armor",
            top_k=5,
            media_output_confidence_threshold=0.6,
            media_score_threshold=1,
        )
        assert strict["media_decisions"][0]["association_score"] == pytest.approx(
            response["media_decisions"][0]["association_score"]
        )
        assert strict["media_decisions"][0]["media_score_threshold"] == 1
        assert strict["media_decisions"][0]["qualifying_chunk_count"] == 0
        assert strict["media_decisions"][0]["weakening_chunk_count"] > 0
        assert strict["media_decisions"][0]["evidence_threshold_adjustment"] < 0
        assert strict["media_decisions"][0]["output_confidence"] < (
            response["media_decisions"][0]["output_confidence"]
        )
        assert strict["media_decisions"][0]["negative_pressure"] > 0
        assert strict["media_decisions"][0]["media_score_threshold_source"] == "request"

        final_threshold_only = await service.search(
            "illustration armor",
            top_k=5,
            media_output_confidence_threshold=0.999,
            media_score_threshold=1,
        )
        assert final_threshold_only["media_outputs"] == []
        assert final_threshold_only["media_decisions"][0][
            "confidence_threshold_met"
        ] is False

        output_threshold_only = await service.search(
            "illustration armor",
            top_k=5,
            media_output_confidence_threshold=0,
            media_score_threshold=1,
        )
        assert output_threshold_only["media_outputs"]
        assert output_threshold_only["media_outputs"][0][
            "qualifying_chunk_count"
        ] == 0
        assert output_threshold_only["media_outputs"][0][
            "confidence_threshold_met"
        ] is True

        permissive = await service.search(
            "illustration armor",
            top_k=5,
            media_output_confidence_threshold=0.6,
            media_score_threshold=0,
        )
        assert permissive["media_outputs"]
        assert permissive["media_decisions"][0]["weakening_chunk_count"] == 0
        assert permissive["media_decisions"][0]["evidence_threshold_adjustment"] > 0
        assert permissive["media_decisions"][0]["output_confidence"] > (
            strict["media_decisions"][0]["output_confidence"]
        )

        total_chunks = int((await storage.statistics())["chunks"])
        for requested_top_k in (10, 20, 50):
            wider = await service.search(
                "illustration armor",
                top_k=requested_top_k,
                media_output_confidence_threshold=0.6,
                media_score_threshold=0.45,
            )
            assert len(wider["items"]) == min(requested_top_k, total_chunks)
            assert wider["media_decisions"] == response["media_decisions"]
            assert wider["media_outputs"] == response["media_outputs"]

        fallback_default = await service.search(
            "illustration armor",
            top_k=5,
            media_score_threshold=None,
        )
        assert fallback_default["thresholds"]["media_score_threshold"] == 0.35
        assert fallback_default["thresholds"][
            "requested_media_score_threshold"
        ] is None
        assert fallback_default["thresholds"][
            "media_score_threshold_source"
        ] == "library_fallback"

        await storage.update_metadata(
            {
                "retrieval_config_json": {
                    **DEFAULT_RETRIEVAL_CONFIG,
                    "media_relevance_pivot_fallback": 0.72,
                }
            }
        )
        fallback_custom = await service.search(
            "illustration armor", top_k=5, media_score_threshold=None
        )
        explicit_override = await service.search(
            "illustration armor", top_k=5, media_score_threshold=0.2
        )
        canonical_override = await service.search(
            "illustration armor", top_k=5, media_relevance_pivot=0.2
        )
        matching_aliases = await service.search(
            "illustration armor",
            top_k=5,
            media_relevance_pivot=0.2,
            media_score_threshold=0.2,
        )
        assert fallback_custom["thresholds"]["media_score_threshold"] == 0.72
        assert explicit_override["thresholds"]["media_score_threshold"] == 0.2
        assert canonical_override["thresholds"]["media_relevance_pivot"] == 0.2
        assert canonical_override["thresholds"][
            "media_relevance_pivot_request_field"
        ] == "media_relevance_pivot"
        canonical_confidences = [
            item["output_confidence"]
            for item in canonical_override["media_decisions"]
        ]
        assert [
            item["output_confidence"]
            for item in explicit_override["media_decisions"]
        ] == canonical_confidences
        assert [
            item["output_confidence"]
            for item in matching_aliases["media_decisions"]
        ] == canonical_confidences
        assert explicit_override["thresholds"][
            "media_score_threshold_source"
        ] == "request"

        context = await storage.document_media_calibration_context(
            result["document_id"], image["id"]
        )
        await storage.replace_document_media_calibration(
            document_id=result["document_id"],
            asset_id=image["id"],
            semantic_mode="calibrated",
            media_description="illustration armor",
            calibration_method=MEDIA_CALIBRATION_METHOD,
            provider_fingerprint="fixture-sha",
            chunks=[
                {
                    "chunk_id": item["chunk_id"],
                    "semantic_strength": 0.5,
                    "calibration_similarity": 0.8,
                    "calibration_rank": index,
                }
                for index, item in enumerate(context["chunks"], start=1)
            ],
        )
        semantic_weighted = await service.search(
            "illustration armor",
            top_k=1,
            media_output_confidence_threshold=0,
            media_score_threshold=0,
        )
        semantic_media = semantic_weighted["items"][0]["associated_media"][0]
        assert semantic_media["semantic_strength"] == pytest.approx(0.5)
        assert semantic_media["media_evidence_score"] == pytest.approx(
            semantic_weighted["items"][0]["dense_score"] * 0.5,
            abs=1e-6,
        )
        await service.recalibrate_document_media(
            document_id=result["document_id"],
            asset_id=image["id"],
            enabled=False,
            media_description="",
        )
        uniform = await storage.list_document_media_calibrations()
        assert uniform[0]["semantic_mode"] == "uniform"
        assert float(uniform[0]["minimum_strength"]) == pytest.approx(0.5)
        assert float(uniform[0]["maximum_strength"]) == pytest.approx(0.5)

        await storage.link_asset(
            scope="document",
            target_id=result["document_id"],
            asset_id=image["id"],
            payload={"relation_weight": 0.5, "output_policy": "auto"},
        )
        rejected = await service.search(
            "illustration armor",
            top_k=5,
            media_output_confidence_threshold=0.999,
            media_score_threshold=0.45,
        )
        assert rejected["media_outputs"] == []
        assert rejected["media_decisions"][0]["output_confidence"] < (
            response["media_decisions"][0]["output_confidence"]
        )
        assert rejected["media_decisions"][0]["reason"] == "confidence_below_threshold"
        assert rejected["media_decisions"][0]["qualifying_chunk_count"] == 0
        assert rejected["media_decisions"][0]["weakening_chunk_count"] > 0
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_media_confidence_uses_per_asset_bound_evidence_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "isolated-media-library"
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="isolated-media",
        name="Isolated media",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    service = TextMediaService(root, storage, TextMediaIndex(root), FixtureProvider())
    try:
        bound = await service.ingest_document(
            filename="bound.md",
            title="Bound portrait",
            data=(("角色立绘与盔甲背景。" * 30 + "\n\n") * 8).encode(),
        )
        target_image = await service.upload_image(
            filename="target.png", data=_png_bytes((800, 600))
        )
        await storage.link_asset(
            scope="document",
            target_id=bound["document_id"],
            asset_id=target_image["id"],
            payload={"relation_weight": 1.0, "output_policy": "auto"},
        )

        async def target_decision() -> dict[str, object]:
            response = await service.search(
                "描述角色立绘背景",
                top_k=5,
                media_output_confidence_threshold=0,
                media_score_threshold=0.35,
            )
            assert response["thresholds"]["media_evidence_scope"] == "asset_bound_only"
            assert response["thresholds"]["media_ranking_scope"] == "per_asset"
            return next(
                item
                for item in response["media_decisions"]
                if item["asset_id"] == target_image["id"]
            )

        baseline = await target_decision()
        assert baseline["bound_chunk_count"] == len(bound["chunk_ids"])
        assert baseline["candidate_chunk_count"] == len(baseline["evidence"])
        assert baseline["media_evidence_scope"] == "asset_bound_only"
        assert baseline["media_ranking_scope"] == "per_asset"

        unbound = await service.ingest_document(
            filename="same-batch-like-unbound.md",
            title="Unbound high-similarity document",
            data=(("角色立绘与盔甲背景，高相似度但没有媒体关系。" * 35 + "\n\n") * 10).encode(),
        )
        after_unbound = await target_decision()
        assert after_unbound == baseline

        other = await service.ingest_document(
            filename="other-media.md",
            title="Other media document",
            data=(("角色立绘与盔甲背景，属于另一张图片。" * 35 + "\n\n") * 10).encode(),
        )
        other_image = await service.upload_image(
            filename="other.png", data=_png_bytes((640, 640))
        )
        await storage.link_asset(
            scope="document",
            target_id=other["document_id"],
            asset_id=other_image["id"],
            payload={"relation_weight": 1.0, "output_policy": "auto"},
        )
        after_other_asset = await target_decision()
        assert after_other_asset["output_confidence"] == pytest.approx(
            baseline["output_confidence"], abs=1e-6
        )
        assert after_other_asset["association_score"] == pytest.approx(
            baseline["association_score"], abs=1e-6
        )
        assert after_other_asset["evidence"] == baseline["evidence"]
        assert after_other_asset["media_frequency_corpus_size"] == (
            baseline["media_frequency_corpus_size"] + 1
        )

        await storage.link_asset(
            scope="document",
            target_id=unbound["document_id"],
            asset_id=target_image["id"],
            payload={"relation_weight": 1.0, "output_policy": "disabled"},
        )
        after_disabled = await target_decision()
        assert after_disabled["output_confidence"] == pytest.approx(
            baseline["output_confidence"], abs=1e-6
        )
        assert after_disabled["association_score"] == pytest.approx(
            baseline["association_score"], abs=1e-6
        )
        assert after_disabled["evidence"] == baseline["evidence"]

        await storage.link_asset(
            scope="entry",
            target_id=bound["entry_id"],
            asset_id=target_image["id"],
            payload={"relation_weight": 1.0, "output_policy": "auto"},
        )
        await storage.link_asset(
            scope="chunk",
            target_id=bound["chunk_ids"][0],
            asset_id=target_image["id"],
            payload={"relation_weight": 1.0, "output_policy": "auto"},
        )
        deduplicated = await target_decision()
        evidence_ids = [item["chunk_id"] for item in deduplicated["evidence"]]
        assert len(evidence_ids) == len(set(evidence_ids))
        assert deduplicated["bound_chunk_count"] == len(bound["chunk_ids"])
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_media_fusion_uses_visual_gate_vectors_keywords_and_absolute_limit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fusion-library"
    source = tmp_path / "knowledge.md"
    source.write_text("General character biography and preferences.", encoding="utf-8")
    image_specs = [
        ("avatar.png", "character face portrait avatar"),
        ("full.png", "character full armor outfit illustration"),
        ("battle.png", "character battle action scene with sword"),
    ]
    for index, (filename, _) in enumerate(image_specs):
        (tmp_path / filename).write_bytes(_png_bytes((640 + index, 480)))

    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="fusion",
        name="Fusion",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    service = TextMediaService(
        root, storage, TextMediaIndex(root), MultiImageFixtureProvider()
    )
    try:
        installed = await service.ingest_batch(
            batch_id="fusion-batch",
            documents=[{"path": str(source), "filename": source.name}],
            images=[
                {
                    "path": str(tmp_path / filename),
                    "filename": filename,
                    "document_indexes": [0],
                    "media_description": description,
                }
                for filename, description in image_specs
            ],
            chunk_target=300,
            chunk_overlap=30,
            embedding_batch_size=4,
            concurrency=2,
            max_retries=1,
            media_semantic_calibration_enabled=True,
        )
        validation = await storage.validate()
        assert validation["relation_media_vector_count"] == 3
        assert validation["asset_media_vector_count"] == 3
        assert validation["media_vector_dimensions"] == 4

        portrait = await service.search("show me your portrait", top_k=1)
        assert portrait["visual_intent"]["detected"] is True
        assert [item["original_name"] for item in portrait["media_outputs"]] == [
            "avatar.png"
        ]
        assert portrait["media_outputs"][0]["media_vector_similarity"] > 0.99
        assert "portrait" in portrait["media_outputs"][0]["matched_tokens"]

        battle = await service.search("show the battle scene", top_k=1)
        assert battle["media_outputs"][0]["original_name"] == "battle.png"
        negative = await service.search("do you enjoy battle?", top_k=1)
        assert negative["visual_intent"]["detected"] is False
        assert negative["media_outputs"] == []

        broad = await service.search(
            "show portrait full outfit battle scene",
            top_k=1,
            media_output_confidence_threshold=0.35,
            media_score_threshold=0.35,
            max_media_outputs=1,
        )
        assert len(broad["media_outputs"]) == 1
        assert sum(
            decision["reason"] == "output_limit"
            for decision in broad["media_decisions"]
        ) >= 1

        battle_asset = next(
            item["asset_id"]
            for item in installed["images"]
            if item["asset_id"]
            == next(
                decision["asset_id"]
                for decision in battle["media_decisions"]
                if decision["original_name"] == "battle.png"
            )
        )
        await storage.link_asset(
            scope="document",
            target_id=installed["documents"][0]["document_id"],
            asset_id=battle_asset,
            payload={"output_policy": "with_result"},
        )
        forced = await service.search(
            "show me your portrait", top_k=1, max_media_outputs=1
        )
        assert forced["media_outputs"][0]["original_name"] == "battle.png"
        assert forced["media_outputs"][0]["reason"] == "forced_with_result"

        disabled = await service.search(
            "show me your portrait", top_k=1, max_media_outputs=0
        )
        assert disabled["media_outputs"] == []
        assert all(
            not decision["output"] for decision in disabled["media_decisions"]
        )
        assert any(
            decision["reason"] == "output_limit"
            for decision in disabled["media_decisions"]
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_registered_manager_creates_cross_type_same_id(tmp_path: Path) -> None:
    config = AppConfig(api_key="test", session_secret="test-session")
    context = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "state",
        config=config,
        configure_logs=False,
    )
    await context.manager.initialize()
    try:
        created = await context.manager.create_library(
            {
                "database_type": TEXT_MEDIA_V1_TYPE,
                "id": "Default",
                "name": "Knowledge Default",
                "provider_id": config.provider.id,
            }
        )
        assert created["database_type"] == TEXT_MEDIA_V1_TYPE
        assert created["database_category"] == "knowledge"
        assert DatabaseRef(TEXT_MEDIA_V1_TYPE, "Default") in context.manager.runtimes
        items = await context.manager.list_libraries(stats_mode="summary")
        assert {(item["database_type"], item["id"]) for item in items} >= {
            ("livingmemory_v8", "Default"),
            (TEXT_MEDIA_V1_TYPE, "Default"),
        }
        providers = await context.manager.list_providers("embedding")
        default_provider = next(
            item for item in providers if item["id"] == config.provider.id
        )
        assert {
            (item["database_type"], item["database_id"], item["database_category"])
            for item in default_provider["used_by"]
        } >= {
            ("livingmemory_v8", "Default", "memory"),
            (TEXT_MEDIA_V1_TYPE, "Default", "knowledge"),
        }

        knowledge_only = asdict(config.provider)
        knowledge_only["id"] = "knowledge_only"
        knowledge_only["display_name"] = "Knowledge Only"
        await context.manager.control.create_provider(knowledge_only)
        await context.manager.create_library(
            {
                "database_type": TEXT_MEDIA_V1_TYPE,
                "id": "knowledge_only_db",
                "name": "Knowledge Only DB",
                "provider_id": "knowledge_only",
            }
        )
        usage = await context.manager.provider_usage("knowledge_only")
        assert [(item["database_type"], item["database_id"]) for item in usage] == [
            (TEXT_MEDIA_V1_TYPE, "knowledge_only_db")
        ]
        with pytest.raises(ValueError):
            await context.manager.update_provider("knowledge_only", {"id": "renamed"})
        with pytest.raises(ValueError):
            await context.manager.update_provider("knowledge_only", {"enabled": False})
        with pytest.raises(ValueError):
            await context.manager.delete_provider("knowledge_only")
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_text_media_rerank_binding_usage_and_hot_unbind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AppConfig(api_key="test", session_secret="test-session")
    context = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "state",
        config=config,
        configure_logs=False,
    )
    monkeypatch.setattr(
        "personalityrag.library_types.text_media_v1.manager.build_rerank_provider",
        lambda _config: FixtureReranker(),
    )
    await context.manager.initialize()
    try:
        await context.manager.create_provider(
            {
                "id": "knowledge_rerank",
                "display_name": "Knowledge Rerank",
                "type": "vllm_rerank",
                "enabled": True,
                "api_base": "http://127.0.0.1:8002",
                "api_suffix": "/v1/rerank",
                "model": "BAAI/bge-reranker-v2-m3",
                "dimensions": 0,
                "batch_size": 8,
                "concurrency": 1,
            }
        )
        created = await context.manager.create_library(
            {
                "database_type": TEXT_MEDIA_V1_TYPE,
                "id": "rerank_usage_db",
                "name": "Rerank usage",
                "provider_id": config.provider.id,
                "rerank_provider_id": "knowledge_rerank",
            }
        )
        assert created["rerank_binding"]["bound"] is True
        assert created["rerank_binding"]["available"] is True
        assert created["rerank_binding"]["fingerprint"]
        assert created["rerank_binding"]["needs_recalibration"] is False
        usage = await context.manager.provider_usage("knowledge_rerank")
        assert len(usage) == 1
        assert usage[0]["usage_kind"] == "rerank"
        assert usage[0]["needs_rebuild"] is False
        assert usage[0]["needs_recalibration"] is False
        with pytest.raises(ValueError, match="不能删除"):
            await context.manager.delete_provider("knowledge_rerank")

        runtime = await context.manager.get_runtime(
            DatabaseRef(TEXT_MEDIA_V1_TYPE, "rerank_usage_db")
        )
        generation = runtime.indexes.status()["generation"]
        updated = await context.manager.update_library(
            DatabaseRef(TEXT_MEDIA_V1_TYPE, "rerank_usage_db"),
            {"rerank_provider_id": None},
        )
        assert updated["rerank_binding"]["bound"] is False
        assert runtime.indexes.status()["generation"] == generation
        assert await context.manager.provider_usage("knowledge_rerank") == []
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_text_media_long_embedding_tasks_probe_auto_context_each_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    detect_calls: list[int] = []

    def factory(provider_config: ProviderConfig):
        return ContextPolicyFixtureProvider(
            provider_config,
            detect_calls=detect_calls,
            detected_tokens=512,
        )

    monkeypatch.setattr(
        "personalityrag.library_types.text_media_v1.manager.build_provider",
        factory,
    )
    monkeypatch.setattr(
        "personalityrag.library_types.text_media_v1.resumable_tasks.build_provider",
        factory,
    )
    provider_config = ProviderConfig(
        id="text_context_auto",
        display_name="Text context auto",
        model="fixture-context-auto",
        dimensions=3,
        context_length_mode="auto",
        max_context_tokens=256,
        max_context_tokens_source="auto:cached",
    )
    context = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "state",
        config=AppConfig(
            api_key="test",
            session_secret="text-context-auto-session",
            provider=provider_config,
        ),
        configure_logs=False,
    )
    await context.manager.initialize()
    ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, "context_auto")
    try:
        await context.manager.create_library(
            {
                "database_type": TEXT_MEDIA_V1_TYPE,
                "id": ref.id,
                "name": "Context auto",
                "provider_id": provider_config.id,
            }
        )
        first_root = (
            context.state_root
            / "data"
            / "task_inputs"
            / TEXT_MEDIA_V1_TYPE
            / ref.id
            / "first"
        )
        first_root.mkdir(parents=True)
        document_path = first_root / "first.md"
        document_path.write_text("# Portrait\n\nfull portrait", encoding="utf-8")
        first_id = await context.manager.jobs.start_resumable(
            "text_media_document_ingest",
            {
                "input_root": str(first_root),
                "document_path": str(document_path),
                "filename": "first.md",
                "title": "First",
            },
            library_id=ref.id,
            database_type=TEXT_MEDIA_V1_TYPE,
            lease_runtime=False,
        )
        first = await asyncio.wait_for(
            context.manager.jobs.wait(first_id), timeout=5
        )
        assert first["status"] == "completed"
        assert detect_calls == [0]
        assert first["result"]["embedding_context"]["probe_attempted"] is True
        assert first["result"]["embedding_context"]["persisted"] is True
        persisted = await context.manager.control.get_provider(
            provider_config.id
        )
        assert persisted is not None
        assert persisted.config.context_length_mode == "auto"
        assert persisted.config.max_context_tokens == 512
        assert (
            persisted.config.max_context_tokens_source
            == "auto:text-media-task-fixture"
        )
        detail = await context.manager.library_detail(ref)
        assert detail["provider_revision"] == persisted.revision
        assert detail["provider"]["config_sha256"] == persisted.config_sha256

        second_root = (
            context.state_root
            / "data"
            / "task_inputs"
            / TEXT_MEDIA_V1_TYPE
            / ref.id
            / "second"
        )
        second_root.mkdir(parents=True)
        body_path = second_root / "body.txt"
        body_path.write_text("portrait face details", encoding="utf-8")
        second_id = await context.manager.jobs.start_resumable(
            "text_media_entry_create",
            {
                "input_root": str(second_root),
                "body_path": str(body_path),
                "title": "Second",
            },
            library_id=ref.id,
            database_type=TEXT_MEDIA_V1_TYPE,
            lease_runtime=False,
        )
        second = await asyncio.wait_for(
            context.manager.jobs.wait(second_id), timeout=5
        )
        assert second["status"] == "completed"
        assert detect_calls == [0, 0]
        assert second["result"]["embedding_context"]["probe_attempted"] is True
        assert second["result"]["embedding_context"]["persisted"] is False

        delete_id = await context.manager.jobs.start_resumable(
            "text_media_document_delete",
            {"document_ids": [first["result"]["document_id"]]},
            library_id=ref.id,
            database_type=TEXT_MEDIA_V1_TYPE,
            lease_runtime=False,
        )
        deleted = await asyncio.wait_for(
            context.manager.jobs.wait(delete_id), timeout=5
        )
        assert deleted["status"] == "completed"
        assert detect_calls == [0, 0]
        assert "embedding_context" not in deleted["result"]
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_text_media_long_task_manual_context_never_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    detect_calls: list[int] = []

    def factory(provider_config: ProviderConfig):
        return ContextPolicyFixtureProvider(
            provider_config,
            detect_calls=detect_calls,
            detected_tokens=999,
        )

    monkeypatch.setattr(
        "personalityrag.library_types.text_media_v1.manager.build_provider",
        factory,
    )
    monkeypatch.setattr(
        "personalityrag.library_types.text_media_v1.resumable_tasks.build_provider",
        factory,
    )
    provider_config = ProviderConfig(
        id="text_context_manual",
        display_name="Text context manual",
        model="fixture-context-manual",
        dimensions=3,
        context_length_mode="manual",
        max_context_tokens=256,
        max_context_tokens_source="manual:user",
    )
    context = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "state",
        config=AppConfig(
            api_key="test",
            session_secret="text-context-manual-session",
            provider=provider_config,
        ),
        configure_logs=False,
    )
    await context.manager.initialize()
    ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, "context_manual")
    try:
        await context.manager.create_library(
            {
                "database_type": TEXT_MEDIA_V1_TYPE,
                "id": ref.id,
                "name": "Context manual",
                "provider_id": provider_config.id,
            }
        )
        job_id = await context.manager.jobs.start_resumable(
            "text_media_index_rebuild",
            {"provider_id": provider_config.id, "reason": "manual-policy"},
            library_id=ref.id,
            database_type=TEXT_MEDIA_V1_TYPE,
            lease_runtime=False,
        )
        completed = await asyncio.wait_for(
            context.manager.jobs.wait(job_id), timeout=5
        )
        assert completed["status"] == "completed"
        assert detect_calls == []
        diagnostics = completed["result"]["embedding_context"]
        assert diagnostics["probe_attempted"] is False
        assert diagnostics["context_length_mode"] == "manual"
        assert diagnostics["max_context_tokens"] == 256
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_text_media_failed_auto_probe_persists_safe_manual_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    detect_calls: list[int] = []

    def factory(provider_config: ProviderConfig):
        return ContextPolicyFixtureProvider(
            provider_config,
            detect_calls=detect_calls,
            detected_tokens=0,
        )

    monkeypatch.setattr(
        "personalityrag.library_types.text_media_v1.manager.build_provider",
        factory,
    )
    provider_config = ProviderConfig(
        id="text_context_fallback",
        display_name="Text context fallback",
        model="fixture-context-fallback",
        dimensions=3,
        context_length_mode="auto",
        max_context_tokens=256,
        max_context_tokens_source="auto:cached",
    )
    context = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "state",
        config=AppConfig(
            api_key="test",
            session_secret="text-context-fallback-session",
            provider=provider_config,
        ),
        configure_logs=False,
    )
    await context.manager.initialize()
    try:
        text_manager = context.manager._managers[TEXT_MEDIA_V1_TYPE]
        revision, diagnostics = (
            await text_manager.prepare_embedding_context_for_long_task(
                provider_config.id
            )
        )
        assert detect_calls == [0]
        assert diagnostics["probe_attempted"] is True
        assert diagnostics["persisted"] is True
        assert revision.config.context_length_mode == "manual"
        assert revision.config.max_context_tokens == 256
        assert (
            revision.config.max_context_tokens_source
            == "manual:fallback-undetected"
        )
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_text_media_provider_switch_rebuilds_every_vector_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_started = asyncio.Event()
    target_release = asyncio.Event()
    source_config = ProviderConfig(
        id="embedding_a",
        display_name="Embedding A",
        model="fixture-a",
        dimensions=3,
    )

    def factory(provider_config: ProviderConfig):
        if provider_config.model == "fixture-b":
            return DimensionalFixtureProvider(
                provider_config,
                started=target_started,
                release=target_release,
            )
        return DimensionalFixtureProvider(
            provider_config,
            fail=provider_config.model == "fixture-failure",
        )

    monkeypatch.setattr(
        "personalityrag.library_types.text_media_v1.manager.build_provider",
        factory,
    )
    monkeypatch.setattr(
        "personalityrag.library_types.text_media_v1.resumable_tasks.build_provider",
        factory,
    )
    context = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "state",
        config=AppConfig(
            api_key="test",
            session_secret="provider-switch-session",
            provider=source_config,
        ),
        configure_logs=False,
    )
    await context.manager.initialize()
    text_manager = context.manager._managers[TEXT_MEDIA_V1_TYPE]
    ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, "switchable")
    try:
        await context.manager.create_library(
            {
                "database_type": TEXT_MEDIA_V1_TYPE,
                "id": ref.id,
                "name": "Switchable",
                "provider_id": source_config.id,
            }
        )
        service = await context.manager.get_runtime(ref)
        document = await service.ingest_document(
            filename="portrait.md",
            title="Portrait",
            data=("# Portrait\n\nfull portrait armor details " * 20).encode(),
        )
        image = await service.upload_image(
            filename="portrait.png",
            data=_png_bytes((800, 600)),
            media_descriptions=[
                "full portrait armor",
                "black and red standing pose",
                "long sword character illustration",
            ],
        )
        await service.storage.link_asset(
            scope="document",
            target_id=document["document_id"],
            asset_id=image["id"],
            payload={"role": "illustration", "relation_weight": 1.0},
        )
        await service.recalibrate_document_media(
            document_id=document["document_id"],
            asset_id=image["id"],
            enabled=True,
            media_description="full portrait armor",
        )
        original_generation = service.indexes.status()["generation"]
        original_content = (
            await service.storage.get_document_detail(document["document_id"])
        )["content"]

        target = await context.manager.create_provider(
            {
                "id": "embedding_b",
                "display_name": "Embedding B",
                "type": "vllm_embedding",
                "enabled": True,
                "model": "fixture-b",
                "dimensions": 5,
                "context_length_mode": "manual",
                "max_context_tokens": 128,
            }
        )
        job_id = await context.manager.jobs.start_resumable(
            "text_media_index_rebuild",
            {"provider_id": target["id"], "reason": "test_provider_switch"},
            library_id=ref.id,
            database_type=TEXT_MEDIA_V1_TYPE,
            lease_runtime=False,
        )
        await asyncio.wait_for(target_started.wait(), timeout=2)
        during = await text_manager.library_detail(ref.id)
        assert during["provider_id"] == source_config.id
        assert during["indexes"]["generation"] == original_generation

        target_release.set()
        completed = await asyncio.wait_for(
            context.manager.jobs.wait(job_id), timeout=5
        )
        assert completed["status"] == "completed"
        assert completed["resumable"] is True
        assert completed["task_type"]["adapter_blocking"] is True

        rebuilt = await context.manager.get_runtime(ref)
        detail = await text_manager.library_detail(ref.id)
        assert detail["provider_id"] == "embedding_b"
        assert detail["indexes"]["generation"] != original_generation
        assert detail["indexes"]["dimensions"] == 5
        assert (
            await rebuilt.storage.get_document_detail(document["document_id"])
        )["content"] == original_content
        db = await rebuilt.storage.pool.acquire()
        try:
            relation = await (
                await db.execute(
                    """SELECT length(media_description_vector) AS bytes,
                    calibration_provider_fingerprint FROM document_assets
                    WHERE document_id=? AND asset_id=?""",
                    (document["document_id"], image["id"]),
                )
            ).fetchone()
            asset_metadata = await (
                await db.execute(
                    """SELECT length(media_description_vector) AS bytes,
                    provider_fingerprint,sort_order
                    FROM asset_media_metadata
                    WHERE asset_id=? ORDER BY sort_order""",
                    (image["id"],),
                )
            ).fetchall()
            strength_fingerprints = {
                str(row[0])
                for row in await (
                    await db.execute(
                        """SELECT DISTINCT provider_fingerprint
                        FROM chunk_media_strengths WHERE document_id=? AND asset_id=?""",
                        (document["document_id"], image["id"]),
                    )
                ).fetchall()
            }
        finally:
            await db.close()
        target_revision = await context.manager.control.get_provider("embedding_b")
        assert target_revision is not None
        assert relation["bytes"] == 5 * 4
        assert relation["calibration_provider_fingerprint"] == target_revision.config_sha256
        assert len(asset_metadata) == 3
        assert [int(item["sort_order"]) for item in asset_metadata] == [0, 1, 2]
        assert {int(item["bytes"]) for item in asset_metadata} == {5 * 4}
        assert {
            str(item["provider_fingerprint"]) for item in asset_metadata
        } == {target_revision.config_sha256}
        assert strength_fingerprints == {target_revision.config_sha256}

        target_started.clear()
        target_release.clear()
        changing = await context.manager.create_provider(
            {
                "id": "embedding_changing",
                "display_name": "Embedding changing",
                "type": "vllm_embedding",
                "enabled": True,
                "model": "fixture-b",
                "dimensions": 6,
            }
        )
        stable_generation = rebuilt.indexes.status()["generation"]
        changing_job_id = await context.manager.jobs.start_resumable(
            "text_media_index_rebuild",
            {"provider_id": changing["id"], "reason": "test_provider_drift"},
            library_id=ref.id,
            database_type=TEXT_MEDIA_V1_TYPE,
            lease_runtime=False,
        )
        await asyncio.wait_for(target_started.wait(), timeout=2)
        pending_usage = await context.manager.provider_usage(
            "embedding_changing"
        )
        assert any(
            item["usage_kind"] == "embedding_rebuild_target"
            and item["active_task_id"] == changing_job_id
            for item in pending_usage
        )
        with pytest.raises(ValueError, match="不能删除"):
            await context.manager.delete_provider("embedding_changing")
        await context.manager.update_provider(
            "embedding_changing", {"model": "fixture-c-changed"}
        )
        target_release.set()
        changing_job = await asyncio.wait_for(
            context.manager.jobs.wait(changing_job_id), timeout=5
        )
        assert changing_job["status"] == "failed"
        assert "changed before activation" in changing_job["error"]
        after_drift = await text_manager.library_detail(ref.id)
        assert after_drift["provider_id"] == "embedding_b"
        assert after_drift["indexes"]["generation"] == stable_generation

        failed = await context.manager.create_provider(
            {
                "id": "embedding_failure",
                "display_name": "Embedding failure",
                "type": "vllm_embedding",
                "enabled": True,
                "model": "fixture-failure",
                "dimensions": 7,
            }
        )
        failed_job_id = await context.manager.jobs.start_resumable(
            "text_media_index_rebuild",
            {"provider_id": failed["id"], "reason": "test_failure"},
            library_id=ref.id,
            database_type=TEXT_MEDIA_V1_TYPE,
            lease_runtime=False,
        )
        failed_job = await asyncio.wait_for(
            context.manager.jobs.wait(failed_job_id), timeout=5
        )
        assert failed_job["status"] == "failed"
        assert "fixture provider rebuild failure" in failed_job["error"]
        after_failure = await text_manager.library_detail(ref.id)
        assert after_failure["provider_id"] == "embedding_b"
        assert after_failure["indexes"]["generation"] == stable_generation
        assert not text_manager.resumable_tasks.workspace(
            {"id": failed_job_id}
        ).exists()
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_text_media_rename_is_typed_and_protected(tmp_path: Path, monkeypatch) -> None:
    config = AppConfig(
        api_key="test",
        session_secret="test-session",
        library_psk_secret="rename-secret",
    )
    context = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "state",
        config=config,
        configure_logs=False,
    )
    await context.manager.initialize()
    try:
        await context.manager.create_library(
            {
                "database_type": TEXT_MEDIA_V1_TYPE,
                "id": "rename_source",
                "name": "Rename Source",
                "provider_id": config.provider.id,
            }
        )
        ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, "rename_source")
        connection = await context.manager.control.register_adapter_connection(
            ref,
            adapter_id="KnowledgeAdapter",
            instance_id="instance-a",
            adapter_type="knowledge-test",
        )
        manager = context.manager._managers[TEXT_MEDIA_V1_TYPE]
        with pytest.raises(ValueError, match="适配器连接"):
            await manager.update_library("rename_source", {"id": "rename_target"})

        await context.manager.control.force_disconnect_adapter(
            ref,
            "KnowledgeAdapter",
            expected_instance_id=connection["instance_id"],
        )
        active_job = AsyncMock(return_value={"id": "active-job"})
        monkeypatch.setattr(manager.jobs, "active_long_job", active_job)
        with pytest.raises(ValueError, match="进行中的任务"):
            await manager.update_library("rename_source", {"id": "rename_target"})
        active_job.return_value = None

        driver = database_type_registry.require(TEXT_MEDIA_V1_TYPE)
        old_key = driver.derive_access_key(config.library_psk_secret, "rename_source")
        renamed = await manager.update_library(
            "rename_source",
            {
                "id": "rename_target",
                "name": "Renamed",
                "description": "moved",
                "uniform_media_strength": 0.42,
                "retrieval_settings": {
                    **DEFAULT_RETRIEVAL_CONFIG,
                    "rrf_k": 75,
                },
            },
        )
        target_ref = DatabaseRef(TEXT_MEDIA_V1_TYPE, "rename_target")
        assert renamed["id"] == "rename_target"
        assert renamed["name"] == "Renamed"
        assert renamed["uniform_media_strength"] == pytest.approx(0.42)
        assert renamed["retrieval_settings"]["rrf_k"] == 75
        assert await context.manager.control.database_identity(ref) is None
        assert await context.manager.control.database_identity(target_ref)
        source_dir = database_type_registry.data_dir(context.manager.data_dir, ref)
        target_dir = database_type_registry.data_dir(context.manager.data_dir, target_ref)
        assert not source_dir.exists()
        assert target_dir.exists()
        storage = TextMediaStorage(target_dir)
        try:
            metadata = await storage.metadata()
            assert metadata["database_id"] == "rename_target"
            assert float(metadata["uniform_media_strength"]) == pytest.approx(0.42)
            assert json.loads(metadata["retrieval_config_json"])["rrf_k"] == 75
        finally:
            await storage.close()
        forced = await context.manager.control.adapter_connection(
            target_ref, "KnowledgeAdapter"
        )
        assert forced and forced["state"] == "forced_offline"
        assert await context.manager.control.adapter_connection(
            ref, "KnowledgeAdapter"
        ) is None
        assert old_key != driver.derive_access_key(
            config.library_psk_secret, "rename_target"
        )

        await context.manager.control.register_adapter_connection(
            target_ref,
            adapter_id="KnowledgeAdapter",
            instance_id="instance-b",
            adapter_type="knowledge-test",
            manual_reconnect=True,
        )
        with pytest.raises(ValueError, match="适配器连接"):
            await manager.delete_library("rename_target")
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_tmkb_plaintext_roundtrip_and_offline_provider_restore(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    storage = TextMediaStorage(source_root)
    await storage.initialize()
    await storage.create_library(
        database_id="portable",
        name="Portable",
        description="package fixture",
        provider_id="missing-provider",
        provider_revision=7,
        provider_fingerprint="missing-fingerprint",
        rerank_provider_id="missing-rerank",
        rerank_provider_revision=3,
        rerank_provider_fingerprint="missing-rerank-fingerprint",
    )
    source_service = TextMediaService(
        source_root,
        storage,
        TextMediaIndex(source_root),
        FixtureProvider(),
        FixtureReranker(),
        {
            "id": "missing-rerank",
            "revision": 3,
            "fingerprint": "missing-rerank-fingerprint",
        },
    )
    package = tmp_path / "portable.tmkb"
    try:
        document = await source_service.ingest_document(
            filename="knowledge.txt",
            title="Portable entry",
            data="Portable character illustration knowledge.".encode(),
        )
        image = await source_service.upload_image(
            filename="illustration.png",
            data=_png_bytes((640, 480)),
            media_descriptions=[
                "portable character 立绘",
                "full-body black and red armor",
                "standing portrait with a long sword",
            ],
        )
        await storage.link_asset(
            scope="entry",
            target_id=document["entry_id"],
            asset_id=image["id"],
            payload={"output_policy": "with_result"},
        )
        await storage.link_asset(
            scope="document",
            target_id=document["document_id"],
            asset_id=image["id"],
            payload={"output_policy": "auto"},
        )
        await source_service.recalibrate_document_media(
            document_id=document["document_id"],
            asset_id=image["id"],
            enabled=True,
            media_description="portable character 立绘",
        )
        await storage.update_metadata(
            {
                "retrieval_config_json": {
                    **DEFAULT_RETRIEVAL_CONFIG,
                    "media_candidate_limit": 77,
                }
            }
        )
        custom_policy = normalize_visual_intent_policy(
            DEFAULT_VISUAL_INTENT_POLICY
        )
        custom_policy["generation_action_terms"].append("创作")
        await source_service.update_visual_intent_policy(custom_policy)
        await export_tmkb(service=source_service, target=package)
    finally:
        await source_service.close()

    with zipfile.ZipFile(package) as archive:
        assert archive.read("manifest.json")
        assert all(not (info.flag_bits & 0x1) for info in archive.infolist())
    manifest = inspect_tmkb(package)
    assert manifest["database_id"] == "portable"
    assert manifest["files"]["database/textmediaknowledge.db"]
    assert manifest["files"]["settings/visual_intent_policy.csv"]
    assert manifest["media_vectors"]["count"] == 4
    assert manifest["version"] == 1
    assert manifest["database_type"] == "text_media_v1"
    assert manifest["rerank_provider"] == {
        "id": "missing-rerank",
        "revision": 3,
        "fingerprint": "missing-rerank-fingerprint",
    }

    config = AppConfig(api_key="target", session_secret="target-session")
    context = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "target",
        config=config,
        configure_logs=False,
    )
    await context.manager.initialize()
    try:
        restored = await install_tmkb(
            manager=context.manager._managers[TEXT_MEDIA_V1_TYPE],
            package_path=package,
            target_id="restored",
            name_override="Restored",
        )
        assert restored["id"] == "restored"
        assert restored["status"] == "provider_binding_required"
        assert restored["retrieval_settings"]["media_candidate_limit"] == 77
        assert restored["retrieval_settings"]["media_score_threshold_fallback"] == 0.35
        assert restored["retrieval_settings"]["media_threshold_evidence_limit"] == 5
        assert "创作" in restored["visual_intent_policy"][
            "generation_action_terms"
        ]
        assert restored["visual_intent_policy_is_default"] is False
        assert restored["rerank_provider_id"] == "missing-rerank"
        service = await context.manager.get_runtime(
            DatabaseRef(TEXT_MEDIA_V1_TYPE, "restored")
        )
        assert (await service.storage.statistics()) == {
            "documents": 1,
            "entries": 1,
            "chunks": 1,
            "images": 1,
        }
        validation = await service.storage.validate()
        assert validation["vector_count"] == 1
        assert validation["relation_media_vector_count"] == 1
        assert validation["asset_media_vector_count"] == 3
        restored_asset = await service.storage.get_asset_detail(image["id"])
        assert [
            item["media_description"]
            for item in restored_asset["media_descriptions"]
        ] == [
            "portable character 立绘",
            "full-body black and red armor",
            "standing portrait with a long sword",
        ]
        calibrations = await service.storage.list_document_media_calibrations()
        assert calibrations[0]["rerank_calibrated_chunk_count"] == 1
        assert calibrations[0][
            "calibration_rerank_provider_fingerprint"
        ] == "missing-rerank-fingerprint"
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_legacy_tmkb_json_visual_policy_migrates_on_import(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "legacy-source"
    storage = TextMediaStorage(source_root)
    await storage.initialize()
    await storage.create_library(
        database_id="legacy-policy",
        name="Legacy Policy",
        description="",
        provider_id="missing-provider",
        provider_revision=1,
        provider_fingerprint="missing-fingerprint",
    )
    custom = normalize_visual_intent_policy(DEFAULT_VISUAL_INTENT_POLICY)
    custom["generation_action_terms"].append("旧包创作")
    db = await storage.pool.acquire()
    try:
        await db.execute(
            "UPDATE library_meta SET retrieval_config_json=? WHERE singleton=1",
            (
                json.dumps(
                    {"rrf_k": 81, "visual_intent_policy": custom},
                    ensure_ascii=False,
                ),
            ),
        )
        await db.commit()
    finally:
        await db.close()
    service = TextMediaService(
        source_root,
        storage,
        TextMediaIndex(source_root),
        FixtureProvider(),
    )
    package = tmp_path / "legacy-policy.tmkb"
    try:
        await export_tmkb(service=service, target=package)
    finally:
        await service.close()
    assert "settings/visual_intent_policy.csv" not in inspect_tmkb(package)[
        "files"
    ]

    context = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "legacy-target",
        config=AppConfig(api_key="target", session_secret="target-session"),
        configure_logs=False,
    )
    await context.manager.initialize()
    try:
        restored = await install_tmkb(
            manager=context.manager._managers[TEXT_MEDIA_V1_TYPE],
            package_path=package,
            target_id="legacy-restored",
        )
        assert restored["retrieval_settings"]["rrf_k"] == 81
        assert "旧包创作" in restored["visual_intent_policy"][
            "generation_action_terms"
        ]
        assert restored["visual_intent_policy_is_default"] is False
        restored_root = database_type_registry.data_dir(
            context.manager.data_dir,
            DatabaseRef(TEXT_MEDIA_V1_TYPE, "legacy-restored"),
        )
        assert (restored_root / "visual_intent_policy.csv").exists()
        restored_storage = TextMediaStorage(restored_root)
        await restored_storage.initialize()
        try:
            stored = json.loads(
                (await restored_storage.metadata())["retrieval_config_json"]
            )
            assert "visual_intent_policy" not in stored
        finally:
            await restored_storage.close()
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_media_only_tmkb_roundtrip_keeps_asset_vector_without_text_generation(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "media-only-source"
    image_path = tmp_path / "media-only.png"
    image_path.write_bytes(_png_bytes((730, 490)))
    storage = TextMediaStorage(source_root)
    await storage.initialize()
    await storage.create_library(
        database_id="media_only_portable",
        name="Media-only portable",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    service = TextMediaService(
        source_root, storage, TextMediaIndex(source_root), FixtureProvider()
    )
    package = tmp_path / "media-only.tmkb"
    try:
        await service.ingest_batch(
            batch_id="media-only",
            documents=[],
            images=[
                {
                    "path": str(image_path),
                    "filename": "Gemini版劣等模型表情包.png",
                    "document_indexes": [],
                    "media_description": "Gemini版劣等模型表情包",
                }
            ],
            chunk_target=1200,
            chunk_overlap=150,
            embedding_batch_size=8,
            concurrency=1,
            max_retries=1,
        )
        await export_tmkb(service=service, target=package)
    finally:
        await service.close()

    manifest = inspect_tmkb(package)
    assert manifest["version"] == 1
    assert manifest["counts"]["documents"] == 0
    assert manifest["counts"]["assets"] == 1
    assert manifest["media_vectors"]["count"] == 1

    context = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "media-only-target",
        config=AppConfig(api_key="target", session_secret="target-session"),
        configure_logs=False,
    )
    await context.manager.initialize()
    try:
        await install_tmkb(
            manager=context.manager._managers[TEXT_MEDIA_V1_TYPE],
            package_path=package,
            target_id="media_only_restored",
            name_override="Media-only restored",
        )
        restored = await context.manager.get_runtime(
            DatabaseRef(TEXT_MEDIA_V1_TYPE, "media_only_restored")
        )
        validation = await restored.storage.validate()
        assert validation["vector_count"] == 0
        assert validation["asset_media_vector_count"] == 1
        assert restored.indexes.status()["generation"] is None
        assert restored.indexes.status()["media_vector_count"] == 1
    finally:
        await context.manager.close()


def test_tmkb_rejects_tampered_member(tmp_path: Path) -> None:
    package = tmp_path / "tampered.tmkb"
    manifest = {
        "format": "personalityrag.text_media_knowledge.tmkb",
        "version": 1,
        "database_type": "text_media_v1",
        "database_id": "demo",
        "files": {
            "database/textmediaknowledge.db": {
                "sha256": "0" * 64,
                "size": 3,
            }
        },
    }
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("database/textmediaknowledge.db", b"bad")
    with pytest.raises(TmkbPackageError):
        from personalityrag.library_types.text_media_v1.package import extract_tmkb

        extract_tmkb(package, tmp_path / "extract")


async def _build_tmkb_fixture(
    tmp_path: Path, database_id: str, *, with_media: bool = False
) -> Path:
    root = tmp_path / f"source-{database_id}"
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id=database_id,
        name=f"Library {database_id}",
        description="batch fixture",
        provider_id="missing-provider",
        provider_revision=1,
        provider_fingerprint="missing-fingerprint",
    )
    service = TextMediaService(root, storage, TextMediaIndex(root), FixtureProvider())
    package = tmp_path / f"{database_id}.tmkb"
    try:
        document = await service.ingest_document(
            filename=f"{database_id}.txt",
            title=f"Entry {database_id}",
            data=f"Knowledge for {database_id}.".encode(),
        )
        if with_media:
            image = await service.upload_image(
                filename=f"{database_id}-portrait.png",
                data=_png_bytes((640 + len(database_id), 480)),
                media_descriptions=[
                    f"portrait illustration for {database_id}"
                ],
            )
            await storage.link_asset(
                scope="document",
                target_id=document["document_id"],
                asset_id=image["id"],
                payload={"output_policy": "auto"},
            )
            await service.recalibrate_document_media(
                document_id=document["document_id"],
                asset_id=image["id"],
                enabled=True,
                media_description=f"portrait illustration for {database_id}",
            )
            await storage.update_metadata(
                {
                    "retrieval_config_json": {
                        **DEFAULT_RETRIEVAL_CONFIG,
                        "rrf_k": 61,
                        "media_relevance_pivot_fallback": 0.44,
                    }
                }
            )
        await export_tmkb(service=service, target=package)
    finally:
        await service.close()
    return package


@pytest.mark.asyncio
async def test_tmkbs_contains_independent_tmkb_packages_and_detects_tampering(
    tmp_path: Path,
) -> None:
    first = await _build_tmkb_fixture(tmp_path, "first", with_media=True)
    second = await _build_tmkb_fixture(tmp_path, "second", with_media=True)
    package = tmp_path / "libraries.tmkbs"
    write_tmkbs([first, second], package)

    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        assert names == {
            "manifest.json",
            "libraries/first.tmkb",
            "libraries/second.tmkb",
        }
        assert archive.getinfo("libraries/first.tmkb").compress_type == (
            zipfile.ZIP_STORED
        )
        assert archive.getinfo("libraries/second.tmkb").compress_type == (
            zipfile.ZIP_STORED
        )
        extracted_child = tmp_path / "extracted-first.tmkb"
        extracted_child.write_bytes(archive.read("libraries/first.tmkb"))
    child_manifest = inspect_tmkb(extracted_child)
    assert child_manifest["database_id"] == "first"
    assert child_manifest["media_vectors"]["count"] == 2
    assert [item["database_id"] for item in inspect_tmkbs(package)["libraries"]] == [
        "first",
        "second",
    ]

    tampered = tmp_path / "tampered.tmkbs"
    with zipfile.ZipFile(package) as source, zipfile.ZipFile(tampered, "w") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "libraries/second.tmkb":
                data += b"tampered"
            target.writestr(info.filename, data)
    with pytest.raises(TmkbPackageError):
        inspect_tmkbs(tampered)


@pytest.mark.asyncio
async def test_tmkbs_import_is_atomic_and_allows_per_library_renames(tmp_path: Path) -> None:
    first = await _build_tmkb_fixture(tmp_path, "first", with_media=True)
    second = await _build_tmkb_fixture(tmp_path, "second", with_media=True)
    package = tmp_path / "libraries.tmkbs"
    write_tmkbs([first, second], package)
    context = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "target",
        config=AppConfig(api_key="target", session_secret="target-session"),
        configure_logs=False,
    )
    await context.manager.initialize()
    text_manager = context.manager._managers[TEXT_MEDIA_V1_TYPE]
    try:
        imported = await install_tmkbs_atomic(
            manager=text_manager,
            package_path=package,
            items=[
                {"source_id": "first", "target_id": "restored_first", "name": "First restored"},
                {"source_id": "second", "target_id": "restored_second", "name": "Second restored"},
            ],
        )
        assert [item["id"] for item in imported] == ["restored_first", "restored_second"]
        assert all(item["status"] == "provider_binding_required" for item in imported)
        for database_id in ("restored_first", "restored_second"):
            service = await context.manager.get_runtime(
                DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id)
            )
            validation = await service.storage.validate()
            assert validation["relation_media_vector_count"] == 1
            assert validation["asset_media_vector_count"] == 1
            detail = await text_manager.library_detail(database_id)
            assert detail["retrieval_settings"]["rrf_k"] == 61
            assert detail["retrieval_settings"][
                "media_relevance_pivot_fallback"
            ] == 0.44
            assert detail["retrieval_settings"][
                "media_score_threshold_fallback"
            ] == 0.44

        with pytest.raises(TmkbPackageError, match="already exists"):
            await install_tmkbs_atomic(
                manager=text_manager,
                package_path=package,
                items=[
                    {"source_id": "first", "target_id": "new_first"},
                    {"source_id": "second", "target_id": "restored_second"},
                ],
            )
        assert await context.manager.control.database_identity(
            DatabaseRef(TEXT_MEDIA_V1_TYPE, "new_first")
        ) is None
        assert not database_type_registry.data_dir(
            text_manager.data_dir, DatabaseRef(TEXT_MEDIA_V1_TYPE, "new_first")
        ).exists()
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_tmkb_import_tasks_are_resumable_atomic_and_cleanup_owned_inputs(
    tmp_path: Path,
) -> None:
    first = await _build_tmkb_fixture(tmp_path, "first", with_media=True)
    second = await _build_tmkb_fixture(tmp_path, "second")
    batch = tmp_path / "libraries.tmkbs"
    write_tmkbs([first, second], batch)
    context = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "target",
        config=AppConfig(api_key="target", session_secret="target-session"),
        configure_logs=False,
    )
    await context.manager.initialize()
    text_manager = context.manager._managers[TEXT_MEDIA_V1_TYPE]
    upload_root = (
        text_manager.data_dir / "import_uploads" / TEXT_MEDIA_V1_TYPE
    )
    upload_root.mkdir(parents=True, exist_ok=True)
    single_upload = upload_root / "single.tmkb"
    single_upload.write_bytes(first.read_bytes())
    batch_upload = upload_root / "batch.tmkbs"
    batch_upload.write_bytes(batch.read_bytes())
    try:
        single_job_id = await context.manager.jobs.start_resumable(
            "tmkb_import",
            {
                "package_path": str(single_upload),
                "target_id": "restored_single",
                "name": "Restored single",
            },
            library_id="restored_single",
            database_type=TEXT_MEDIA_V1_TYPE,
            lease_runtime=False,
        )
        single_job = await asyncio.wait_for(
            context.manager.jobs.wait(single_job_id), timeout=10
        )
        assert single_job["status"] == "completed"
        assert single_job["resumable"] is True
        assert single_job["result"]["id"] == "restored_single"
        assert not single_upload.exists()
        assert not text_manager.resumable_tasks.workspace(
            {"id": single_job_id}
        ).exists()
        single_root = database_type_registry.data_dir(
            text_manager.data_dir,
            DatabaseRef(TEXT_MEDIA_V1_TYPE, "restored_single"),
        )
        assert not (
            single_root
            / text_manager.resumable_tasks.INSTALL_OWNER_FILENAME
        ).exists()

        batch_job_id = await context.manager.jobs.start_resumable(
            "tmkbs_import",
            {
                "package_path": str(batch_upload),
                "items": [
                    {
                        "source_id": "first",
                        "target_id": "restored_first_task",
                        "name": "First task import",
                    },
                    {
                        "source_id": "second",
                        "target_id": "restored_second_task",
                        "name": "Second task import",
                    },
                ],
            },
            library_id=None,
            database_type=TEXT_MEDIA_V1_TYPE,
            lease_runtime=False,
        )
        batch_job = await asyncio.wait_for(
            context.manager.jobs.wait(batch_job_id), timeout=15
        )
        assert batch_job["status"] == "completed"
        assert batch_job["resumable"] is True
        assert [
            item["id"] for item in batch_job["result"]["libraries"]
        ] == ["restored_first_task", "restored_second_task"]
        assert not batch_upload.exists()
        assert not text_manager.resumable_tasks.workspace(
            {"id": batch_job_id}
        ).exists()
        for database_id in ("restored_first_task", "restored_second_task"):
            root = database_type_registry.data_dir(
                text_manager.data_dir,
                DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id),
            )
            assert not (
                root / text_manager.resumable_tasks.INSTALL_OWNER_FILENAME
            ).exists()
            assert await context.manager.control.database_identity(
                DatabaseRef(TEXT_MEDIA_V1_TYPE, database_id)
            )
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_text_media_copy_naming_and_fixed_directory_backup(tmp_path: Path) -> None:
    package = await _build_tmkb_fixture(tmp_path, "source")
    context = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "target",
        config=AppConfig(api_key="target", session_secret="target-session"),
        configure_logs=False,
    )
    await context.manager.initialize()
    text_manager = context.manager._managers[TEXT_MEDIA_V1_TYPE]
    driver = database_type_registry.require(TEXT_MEDIA_V1_TYPE)
    assert {"copy", "backup"}.issubset(driver.descriptor.capabilities)
    try:
        await install_tmkb(
            manager=text_manager,
            package_path=package,
            target_id="source",
            name_override="Source Library",
        )
        backup = await text_manager.backup_library("source")
        backup_path = Path(backup["path"])
        assert backup_path.parent == database_type_registry.data_dir(
            text_manager.data_dir, DatabaseRef(TEXT_MEDIA_V1_TYPE, "source")
        ) / "backups"
        assert backup_path.suffix == ".tmkb"
        assert inspect_tmkb(backup_path)["database_id"] == "source"

        first = await text_manager.copy_library("source")
        second = await text_manager.copy_library("source")
        assert (first["id"], first["name"]) == ("source_copy", "Source Library(副本)")
        assert (second["id"], second["name"]) == ("source_copy2", "Source Library(副本2)")
        assert not (
            database_type_registry.data_dir(
                text_manager.data_dir, DatabaseRef(TEXT_MEDIA_V1_TYPE, "source_copy")
            )
            / "backups"
        ).exists()
        assert first["stats"] == second["stats"] == {
            "documents": 1,
            "entries": 1,
            "chunks": 1,
            "images": 0,
        }

        await text_manager.delete_library("source_copy")
        third = await text_manager.copy_library("source")
        assert (third["id"], third["name"]) == (
            "source_copy3",
            "Source Library(副本3)",
        )
    finally:
        await context.manager.close()


@pytest.mark.asyncio
async def test_prag_v2_embeds_and_restores_plaintext_tmkb(tmp_path: Path) -> None:
    config = AppConfig(api_key="source", session_secret="source-session")
    source = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "source-state",
        config=config,
        configure_logs=False,
    )
    package = tmp_path / "full.prag"
    await source.manager.initialize()
    try:
        await source.manager.create_library(
            {
                "database_type": TEXT_MEDIA_V1_TYPE,
                "id": "portable",
                "name": "Portable",
                "provider_id": config.provider.id,
            }
        )
        await source.manager.create_library(
            {
                "database_type": TEXT_MEDIA_V1_TYPE,
                "id": "portable_two",
                "name": "Portable Two",
                "provider_id": config.provider.id,
            }
        )
        portable_service = await source.manager.get_runtime(
            DatabaseRef(TEXT_MEDIA_V1_TYPE, "portable")
        )
        custom_policy = normalize_visual_intent_policy(
            DEFAULT_VISUAL_INTENT_POLICY
        )
        custom_policy["generation_action_terms"].append("创作")
        await portable_service.update_visual_intent_policy(custom_policy)
        await export_prag_package(
            root=source.state_root,
            config=config,
            manager=source.manager,
            target=package,
            password="secret",
            include_libraries=True,
            include_providers=True,
        )
    finally:
        await source.manager.close()

    with pyzipper.AESZipFile(package) as outer:
        outer.setpassword(b"secret")
        manifest = json.loads(outer.read("manifest.json"))
        assert manifest["version"] == 2
        names = set(outer.namelist())
        assert not any(name.endswith(".tmkbs") for name in names)
        assert {
            "databases/text_media_v1/portable.tmkb",
            "databases/text_media_v1/portable_two.tmkb",
        }.issubset(names)
        assert outer.getinfo(
            "databases/text_media_v1/portable.tmkb"
        ).compress_type == zipfile.ZIP_STORED
        assert outer.getinfo(
            "databases/text_media_v1/portable_two.tmkb"
        ).compress_type == zipfile.ZIP_STORED
        nested = outer.read("databases/text_media_v1/portable.tmkb")
    with zipfile.ZipFile(io.BytesIO(nested)) as inner:
        assert all(not (info.flag_bits & 0x1) for info in inner.infolist())
        assert json.loads(inner.read("manifest.json"))["database_id"] == "portable"

    target_config = AppConfig(api_key="target", session_secret="target-session")
    target = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "target-state",
        config=target_config,
        configure_logs=False,
    )
    await target.manager.initialize()
    try:
        next_config, result = await import_prag_package(
            root=target.state_root,
            config_path=target.config_path,
            config=target_config,
            manager=target.manager,
            package_path=package,
            password="secret",
        )
        assert next_config.api_key == "source"
        assert result["libraries_imported"] == 3
        detail = await target.manager.library_detail(
            DatabaseRef(TEXT_MEDIA_V1_TYPE, "portable")
        )
        assert detail["name"] == "Portable"
        assert detail["status"] == "ready"
        assert "创作" in detail["visual_intent_policy"][
            "generation_action_terms"
        ]
        assert detail["visual_intent_policy_is_default"] is False
        second_detail = await target.manager.library_detail(
            DatabaseRef(TEXT_MEDIA_V1_TYPE, "portable_two")
        )
        assert second_detail["name"] == "Portable Two"
        assert second_detail["visual_intent_policy_is_default"] is True
    finally:
        await target.manager.close()


def _minimal_docx_bytes() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
              <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
              <Default Extension="xml" ContentType="application/xml"/>
              <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
            </Types>""",
        )
        archive.writestr(
            "_rels/.rels",
            """<?xml version="1.0"?>
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
              <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
            </Relationships>""",
        )
        archive.writestr(
            "word/document.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body>
                <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>生态设定</w:t></w:r></w:p>
                <w:p><w:r><w:t>狱狼龙会吸引大量蝕龙虫。</w:t></w:r></w:p>
                <w:sectPr/>
              </w:body>
            </w:document>""",
        )
    return output.getvalue()


def test_docx_and_pdf_parsers_preserve_semantic_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docx = parse_document_bytes("ecology.docx", _minimal_docx_bytes())
    assert docx.format_hint == "markdown"
    assert "# 生态设定" in docx.content
    assert "狱狼龙会吸引大量蝕龙虫" in docx.content

    class Page:
        def __init__(self, number: int):
            self.number = number

        def extract_text(self) -> str:
                return (
                    "重复页眉\n"
                    f"第 {self.number} 页正文，包含足够长的可检索生态设定说明。\n"
                "https://example.invalid/page\n"
                "重复页脚"
            )

    class Reader:
        is_encrypted = False
        pages = [Page(index) for index in range(1, 7)]

    monkeypatch.setattr(document_parsers, "PdfReader", lambda _stream: Reader())
    pdf = parse_document_bytes("book.pdf", b"%PDF-fixture")
    assert pdf.page_count == 6
    assert "重复页眉" not in pdf.content
    assert "重复页脚" not in pdf.content
    assert "https://example.invalid/page" not in pdf.content
    assert "〔PDF 第 1 页〕" in pdf.content
    assert pdf.content.index("第 1 页正文") < pdf.content.index("第 6 页正文")

    class ScannedReader:
        is_encrypted = False
        pages = [type("BlankPage", (), {"extract_text": lambda self: ""})()]

    monkeypatch.setattr(
        document_parsers, "PdfReader", lambda _stream: ScannedReader()
    )
    with pytest.raises(ValueError, match="OCR"):
        parse_document_bytes("scan.pdf", b"%PDF-scan")


@pytest.mark.asyncio
async def test_text_only_and_media_only_batches_trim_provider_work(
    tmp_path: Path,
) -> None:
    class CountingProvider(FixtureProvider):
        def __init__(self):
            self.calls: list[list[str]] = []

        async def get_embedding(self, text: str) -> list[float]:
            self.calls.append([text])
            return self._vector(text)

        async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            return [self._vector(text) for text in texts]

    root = tmp_path / "trimmed"
    document = tmp_path / "only.txt"
    image = tmp_path / "only.png"
    document.write_text("纯文本内容与群友对话。" * 80, encoding="utf-8")
    image.write_bytes(_png_bytes((640, 480)))
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="trimmed",
        name="Trimmed",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    provider = CountingProvider()
    service = TextMediaService(root, storage, TextMediaIndex(root), provider)
    try:
        text_result = await service.ingest_batch(
            batch_id="text-only",
            documents=[{"path": str(document), "filename": document.name}],
            images=[],
            chunk_target=240,
            chunk_overlap=40,
            embedding_batch_size=8,
            concurrency=1,
            max_retries=1,
        )
        assert text_result["parameters"]["ingest_mode"] == "text_only"
        assert all("only.png" not in " ".join(call) for call in provider.calls)
        provider.calls.clear()
        text_search = await service.search(
            "纯文本群友内容",
            retrieval_mode="standard",
            rerank=False,
        )
        assert text_search["items"]
        assert text_search["media_decisions"] == []
        assert text_search["execution"]["bound_media_executed"] is False
        assert (
            text_search["execution"]["bound_media_skip_reason"]
            == "no_bound_chunk_in_fixed_candidate_window"
        )
        assert text_search["execution"]["asset_index_candidate_count"] == 0
        provider.calls.clear()

        media_result = await service.ingest_batch(
            batch_id="media-only",
            documents=[],
            images=[
                {
                    "path": str(image),
                    "filename": "原来是劣等模型 Claude版表情包.png",
                    "document_indexes": [],
                    "media_description": "",
                }
            ],
            chunk_target=1200,
            chunk_overlap=150,
            embedding_batch_size=8,
            concurrency=1,
            max_retries=1,
            media_semantic_calibration_enabled=True,
        )
        assert media_result["parameters"]["ingest_mode"] == "media_only"
        assert media_result["generation_id"] is None
        assert media_result["index"]["media_vector_count"] == 1
        assert len(provider.calls) == 1
        assert provider.calls[0] == ["原来是劣等模型 Claude版表情包"]
        db = await storage.pool.acquire()
        try:
            batches = await (
                await db.execute("SELECT id FROM ingest_batches ORDER BY id")
            ).fetchall()
        finally:
            await db.close()
        assert {str(item["id"]) for item in batches} == {
            "text-only", "media-only"
        }
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_provider_rebuild_refreshes_media_only_index_without_text_generation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "media-only-rebuild"
    image = tmp_path / "media.png"
    image.write_bytes(_png_bytes((700, 500)))
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="media-only-rebuild",
        name="Media-only rebuild",
        description="",
        provider_id="fixture-a",
        provider_revision=1,
        provider_fingerprint="fixture-a-sha",
    )
    service = TextMediaService(root, storage, TextMediaIndex(root), FixtureProvider())
    try:
        installed = await service.ingest_batch(
            batch_id="media-only",
            documents=[],
            images=[
                {
                    "path": str(image),
                    "filename": "GPT版劣等模型表情包.png",
                    "document_indexes": [],
                    "media_description": "GPT版劣等模型表情包",
                }
            ],
            chunk_target=1200,
            chunk_overlap=150,
            embedding_batch_size=8,
            concurrency=1,
            max_retries=1,
        )
        asset_id = installed["images"][0]["asset_id"]
        first_media_generation = service.indexes.status()["media_generation"]
        assert service.indexes.status()["generation"] is None

        rebuilt = await service.rebuild_embeddings(
            provider_id="fixture-b",
            provider_revision=2,
            provider_fingerprint="fixture-b-sha",
        )
        assert rebuilt["generation_id"] is None
        assert rebuilt["index"]["generation"] is None
        assert rebuilt["index"]["media_vector_count"] == 1
        assert rebuilt["index"]["media_generation"] != first_media_generation
        metadata = await storage.media_metadata_records()
        assert len(metadata) == 1
        assert metadata[0]["asset_id"] == asset_id
        assert metadata[0]["provider_fingerprint"] == "fixture-b-sha"
        found = await service.search(
            "GPT版劣等模型表情包",
            retrieval_mode="media_only",
            rerank=False,
            media_output_confidence_threshold=0,
        )
        assert found["media_outputs"][0]["asset_id"] == asset_id
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_media_only_retrieval_applies_universal_relevance_pivot(
    tmp_path: Path,
) -> None:
    root = tmp_path / "media-only-search"
    document = tmp_path / "character.md"
    bound_image = tmp_path / "bound.png"
    unbound_image = tmp_path / "unbound.png"
    document.write_text("# 角色\n\n角色的战斗立绘与武器。" * 30, encoding="utf-8")
    bound_image.write_bytes(_png_bytes((640, 480)))
    unbound_image.write_bytes(_png_bytes((641, 480)))
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="media-only-search",
        name="Media only search",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    service = TextMediaService(root, storage, TextMediaIndex(root), FixtureProvider())
    try:
        installed = await service.ingest_batch(
            batch_id="mixed",
            documents=[{"path": str(document), "filename": document.name}],
            images=[
                {
                    "path": str(bound_image),
                    "filename": bound_image.name,
                    "document_indexes": [0],
                    "media_description": "角色战斗立绘",
                },
                {
                    "path": str(unbound_image),
                    "filename": "原来是劣等模型 Claude版表情包.png",
                    "document_indexes": [],
                    "media_description": "原来是劣等模型 Claude版表情包",
                },
            ],
            chunk_target=300,
            chunk_overlap=30,
            embedding_batch_size=8,
            concurrency=1,
            max_retries=1,
            media_semantic_calibration_enabled=True,
        )
        bound_asset_id = installed["images"][0]["asset_id"]
        unbound_asset_id = installed["images"][1]["asset_id"]
        standard = await service.search(
            "Claude版劣等模型表情包",
            retrieval_mode="standard",
            rerank=False,
            media_output_confidence_threshold=0,
        )
        assert any(
            item["asset_id"] == unbound_asset_id
            for item in standard["media_decisions"]
        )
        equivalent_media_only = await service.search(
            "Claude版劣等模型表情包",
            retrieval_mode="media_only",
            rerank=False,
            media_output_confidence_threshold=0,
        )
        assert standard["media_outputs"] == equivalent_media_only["media_outputs"]
        assert standard["media_decisions"] == equivalent_media_only["media_decisions"]

        low_pivot = await service.search(
            "Claude版劣等模型表情包",
            retrieval_mode="media_only",
            rerank=False,
            top_k=5,
            media_score_threshold=0.1,
            media_output_confidence_threshold=0,
        )
        high_pivot = await service.search(
            "Claude版劣等模型表情包",
            retrieval_mode="media_only",
            rerank=False,
            top_k=50,
            media_score_threshold=0.9,
            media_output_confidence_threshold=0,
        )
        assert low_pivot["items"] == low_pivot["baseline_items"] == []
        assert low_pivot["visual_intent"]["detected"] is True
        low_decision = next(
            item for item in low_pivot["media_decisions"]
            if item["asset_id"] == unbound_asset_id
        )
        high_decision = next(
            item for item in high_pivot["media_decisions"]
            if item["asset_id"] == unbound_asset_id
        )
        assert low_decision["media_evidence_scope"] == "asset_metadata_only"
        assert low_decision["media_relevance_pivot"] == 0.1
        assert low_decision["media_score_threshold"] == 0.1
        assert low_decision["media_relevance_pivot_source"] == "request"
        assert low_decision["evidence"] == []
        assert low_decision["output_confidence"] > high_decision["output_confidence"]
        assert low_decision["raw_direct_relevance"] == pytest.approx(
            high_decision["raw_direct_relevance"]
        )
        assert low_decision["pivot_calibrated_direct_relevance"] > (
            high_decision["pivot_calibrated_direct_relevance"]
        )
        assert low_pivot["execution"]["text_retrieval_executed"] is False
        assert low_pivot["execution"]["unbound_media_executed"] is True

        # Bound chunk retrieval is an independent third candidate source.  A
        # bound image remains discoverable even when its asset description is
        # absent from both the media-vector and asset-FTS candidate heads.
        service.indexes.search_media = AsyncMock(return_value=[])
        storage.media_metadata_lexical_search = AsyncMock(return_value=[])
        bound_from_chunks = await service.search(
            "角色战斗立绘",
            retrieval_mode="media_only",
            rerank=False,
            media_output_confidence_threshold=0,
        )
        bound_decision = next(
            item
            for item in bound_from_chunks["media_decisions"]
            if item["asset_id"] == bound_asset_id
        )
        assert bound_decision["asset_metadata_only"] is False
        assert bound_decision["asset_bound_chunk_rrf_score"] > 0
        assert bound_decision["associated_chunk_count"] > 0
        assert bound_decision["confidence_algorithm"] == "bound_chunk_grounded"
        assert bound_decision["media_only_grounding_direct_exponent"] is None
        assert bound_decision["media_only_grounding_gate"] == 1.0
        assert bound_decision["media_only_gated_grounding_score"] == pytest.approx(
            bound_decision["adjusted_grounding_score"],
            abs=1e-6,
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_text_only_matches_standard_text_and_skips_all_media_work(
    tmp_path: Path,
) -> None:
    root = tmp_path / "text-only-channel"
    document = tmp_path / "character.md"
    image = tmp_path / "portrait.png"
    document.write_text(
        "# 澄月\n\n澄月的全身战斗立绘采用黑红铠甲与白发红瞳设计。" * 30,
        encoding="utf-8",
    )
    image.write_bytes(_png_bytes((640, 480)))
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="text-only-channel",
        name="Text-only channel",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
        rerank_provider_id="fixture-rerank",
        rerank_provider_revision=1,
        rerank_provider_fingerprint="rerank-sha",
    )
    reranker = FixtureReranker()
    service = TextMediaService(
        root,
        storage,
        TextMediaIndex(root),
        FixtureProvider(),
        reranker,
        {"id": "fixture-rerank", "revision": 1, "fingerprint": "rerank-sha"},
    )
    try:
        await service.ingest_batch(
            batch_id="mixed",
            documents=[{"path": str(document), "filename": document.name}],
            images=[
                {
                    "path": str(image),
                    "filename": image.name,
                    "document_indexes": [0],
                    "media_description": "澄月黑红铠甲全身战斗立绘",
                }
            ],
            chunk_target=300,
            chunk_overlap=30,
            embedding_batch_size=8,
            concurrency=1,
            max_retries=1,
            media_semantic_calibration_enabled=True,
        )
        standard = await service.search(
            "澄月的黑红铠甲是什么样子",
            retrieval_mode="standard",
            rerank=False,
        )
        text_only = await service.search(
            "澄月的黑红铠甲是什么样子",
            retrieval_mode="text_only",
            rerank=False,
        )
        def text_result_projection(items: list[dict[str, object]]) -> list[dict[str, object]]:
            return [
                {key: value for key, value in item.items() if key != "associated_media"}
                for item in items
            ]

        assert text_result_projection(text_only["items"]) == text_result_projection(
            standard["items"]
        )
        assert text_result_projection(
            text_only["baseline_items"]
        ) == text_result_projection(standard["baseline_items"])
        assert text_only["media_outputs"] == []
        assert text_only["media_decisions"] == []
        assert text_only["execution"]["text_retrieval_executed"] is True
        assert text_only["execution"]["media_channel_executed"] is False
        text_only_reranked = await service.search(
            "澄月的黑红铠甲是什么样子",
            retrieval_mode="text_only",
            rerank=True,
        )
        assert text_only_reranked["rerank"]["applied"] is True
        assert {
            item["scope"] for item in text_only_reranked["rerank"]["scopes"]
        } == {"text_chunks"}
        assert text_only_reranked["media_outputs"] == []
        assert text_only_reranked["media_decisions"] == []
        standard_reranked = await service.search(
            "展示澄月的黑红铠甲全身战斗立绘",
            retrieval_mode="standard",
            rerank=True,
            media_output_confidence_threshold=0,
        )
        media_only_reranked = await service.search(
            "展示澄月的黑红铠甲全身战斗立绘",
            retrieval_mode="media_only",
            rerank=True,
            media_output_confidence_threshold=0,
        )
        assert standard_reranked["media_outputs"] == media_only_reranked["media_outputs"]
        assert (
            standard_reranked["media_decisions"]
            == media_only_reranked["media_decisions"]
        )

        storage.media_bindings = AsyncMock(
            side_effect=AssertionError("text_only loaded media bindings")
        )
        storage.media_metadata_records = AsyncMock(
            side_effect=AssertionError("text_only loaded media metadata")
        )
        storage.media_token_statistics = AsyncMock(
            side_effect=AssertionError("text_only loaded media token statistics")
        )
        storage.media_lexical_search = AsyncMock(
            side_effect=AssertionError("text_only searched bound media")
        )
        storage.media_metadata_lexical_search = AsyncMock(
            side_effect=AssertionError("text_only searched media descriptions")
        )
        service.indexes.search_media = AsyncMock(
            side_effect=AssertionError("text_only searched media vectors")
        )
        service.indexes.score_subset = AsyncMock(
            side_effect=AssertionError("text_only scored media-bound chunks")
        )
        isolated = await service.search(
            "澄月的黑红铠甲是什么样子",
            retrieval_mode="text_only",
            rerank=False,
        )
        assert isolated["items"] == text_only["items"]
        assert isolated["baseline_items"] == text_only["baseline_items"]
        assert isolated["media_outputs"] == isolated["media_decisions"] == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_media_only_frequency_distinguishes_unique_models_and_collection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "media-frequency-search"
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="media-frequency-search",
        name="Media frequency search",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    service = TextMediaService(
        root, storage, TextMediaIndex(root), FixtureProvider()
    )
    models = ("gpt", "gemini", "claude", "deepseek")
    images: list[dict[str, object]] = []
    expected_assets: dict[str, str] = {}
    try:
        for index, model in enumerate(models):
            path = tmp_path / f"{model}.png"
            path.write_bytes(_png_bytes((640 + index, 480)))
            images.append(
                {
                    "path": str(path),
                    "filename": f"“原来是劣等模型”{model}版表情包.jpg",
                    "document_indexes": [],
                    "media_description": f"“原来是劣等模型”{model}版表情包",
                }
            )
        (tmp_path / "unrelated.png").write_bytes(_png_bytes((650, 480)))
        installed = await service.ingest_batch(
            batch_id="four-model-memes",
            documents=[],
            images=[
                *images,
                {
                    "path": str(tmp_path / "unrelated.png"),
                    "filename": "澄月战斗立绘.jpg",
                    "document_indexes": [],
                    "media_description": "澄月战斗场景全身立绘",
                },
            ],
            chunk_target=1200,
            chunk_overlap=150,
            embedding_batch_size=8,
            concurrency=1,
            max_retries=1,
            media_semantic_calibration_enabled=True,
        )
        expected_assets = {
            model: str(installed["images"][index]["asset_id"])
            for index, model in enumerate(models)
        }
        unrelated_asset_id = str(installed["images"][4]["asset_id"])

        for model in models:
            for query in (
                model,
                f"给我{model}表情包",
                f"给我一张{model}表情包",
                f"给我看看{model.upper()}版的梗图",
                f"show me a {model.upper()} meme",
                f"покажи мем {model}",
            ):
                result = await service.search(
                    query,
                    retrieval_mode="media_only",
                    rerank=False,
                    media_output_confidence_threshold=0.60,
                    max_media_outputs=5,
                )
                assert [
                    item["asset_id"] for item in result["media_outputs"]
                ] == [expected_assets[model]]
                decision = next(
                    item
                    for item in result["media_decisions"]
                    if item["asset_id"] == expected_assets[model]
                )
                assert decision["media_frequency_algorithm"] == (
                    "corpus_distinctive_bm25_v1"
                )
                assert decision["media_frequency_scope"] == "unbound_active_media"
                assert decision["media_frequency_corpus_size"] == 5
                assert decision["distinctive_support"] > 0
                model_token = next(
                    item
                    for item in decision["frequency_token_details"]
                    if item["token"] == model
                )
                assert model_token["document_frequency"] == 1
                assert model_token["rarity"] == pytest.approx(1.0)
                assert decision["media_only_advantage_margin"] > 0
                sibling_confidences = [
                    float(item["output_confidence"])
                    for item in result["media_decisions"]
                    if item["asset_id"] in expected_assets.values()
                    and item["asset_id"] != expected_assets[model]
                ]
                assert sibling_confidences
                assert min(sibling_confidences) > 0
                assert max(sibling_confidences) < 0.60
                assert decision["media_only_specificity_algorithm"] == (
                    "structure_aware_unbound_competition_v1"
                )
                assert decision["unbound_specificity_algorithm"] == (
                    "structure_aware_unbound_competition_v1"
                )
                assert decision["unbound_direct_score"] == pytest.approx(
                    decision["media_only_direct_score"]
                )
                assert decision["unbound_confidence_factor"] == pytest.approx(
                    decision["media_only_confidence_factor"]
                )
                assert decision["unbound_reliability_target"] == pytest.approx(0.25)
                assert decision["unbound_specificity_exponent"] == pytest.approx(1.0)
                assert decision["unbound_advantage_target"] == pytest.approx(0.04)

        generic = await service.search(
            "给我一张表情包",
            retrieval_mode="media_only",
            rerank=False,
            media_output_confidence_threshold=0.60,
            max_media_outputs=5,
        )
        assert generic["media_outputs"] == []
        generic_meme_confidences = [
            float(item["output_confidence"])
            for item in generic["media_decisions"]
            if item["asset_id"] in expected_assets.values()
        ]
        unrelated_confidence = next(
            float(item["output_confidence"])
            for item in generic["media_decisions"]
            if item["asset_id"] == unrelated_asset_id
        )
        assert min(generic_meme_confidences) > unrelated_confidence

        collection = await service.search(
            "把四种表情包都给我看",
            retrieval_mode="media_only",
            rerank=False,
            media_output_confidence_threshold=0.60,
            max_media_outputs=5,
        )
        assert {
            item["asset_id"] for item in collection["media_outputs"]
        } == set(expected_assets.values())
        assert all(
            item["asset_id"] != unrelated_asset_id
            for item in collection["media_outputs"]
        )
        assert all(
            item["media_only_collection_intent"] is True
            and item["collection_support"] > 0
            for item in collection["media_decisions"]
            if item["asset_id"] in expected_assets.values()
        )
        assert min(
            float(item["output_confidence"])
            for item in collection["media_decisions"]
            if item["asset_id"] in expected_assets.values()
        ) >= 0.64

        selected_collection = await service.search(
            "把gpt和claude表情包都给我",
            retrieval_mode="media_only",
            rerank=False,
            media_output_confidence_threshold=0.60,
            max_media_outputs=5,
        )
        assert {
            item["asset_id"] for item in selected_collection["media_outputs"]
        } == {
            expected_assets["gpt"],
            expected_assets["claude"],
        }

        standard = await service.search(
            "给我一张gpt表情包",
            retrieval_mode="standard",
            rerank=False,
            media_output_confidence_threshold=0.60,
        )
        assert [
            item["asset_id"] for item in standard["media_outputs"]
        ] == [expected_assets["gpt"]]
        standard_target = next(
            item
            for item in standard["media_decisions"]
            if item["asset_id"] == expected_assets["gpt"]
        )
        assert standard_target["confidence_algorithm"] == "unbound_asset_direct"
        assert standard_target["output_confidence"] >= 0.60
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_standard_distinctive_rescue_requires_bound_evidence_and_scope_isolation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bound-distinctive-rescue"
    storage = TextMediaStorage(root)
    await storage.initialize()
    await storage.create_library(
        database_id="bound-distinctive-rescue",
        name="Bound distinctive rescue",
        description="",
        provider_id="fixture",
        provider_revision=1,
        provider_fingerprint="fixture-sha",
    )
    service = TextMediaService(
        root, storage, TextMediaIndex(root), FixtureProvider()
    )
    try:
        target_document = await service.ingest_document(
            filename="zinogre.txt",
            title="狱狼龙生态",
            data=("狱狼龙是雷狼龙的特殊个体，拥有龙属性与蚀龙虫。 " * 80).encode(),
        )
        other_document = await service.ingest_document(
            filename="portrait.txt",
            title="澄月角色设定",
            data=("澄月的战斗立绘与角色外貌设定。 " * 80).encode(),
        )
        unbound_document = await service.ingest_document(
            filename="unbound.txt",
            title="无媒体文本",
            data=("普通文本候选，不具有任何图片关系。 " * 80).encode(),
        )
        target_image = await service.upload_image(
            filename="狱狼龙插图.jpg", data=_png_bytes((720, 480))
        )
        other_image = await service.upload_image(
            filename="澄月战斗立绘.jpg", data=_png_bytes((721, 480))
        )
        await storage.link_asset(
            scope="document",
            target_id=target_document["document_id"],
            asset_id=target_image["id"],
            payload={
                "relation_weight": 1.0,
                "output_policy": "auto",
                "media_description": "狱狼龙生态设定插图",
            },
        )
        await storage.link_asset(
            scope="document",
            target_id=other_document["document_id"],
            asset_id=other_image["id"],
            payload={
                "relation_weight": 1.0,
                "output_policy": "auto",
                "media_description": "澄月全身战斗立绘",
            },
        )

        # Force the public text candidate window to contain only an unrelated,
        # unbound chunk. The exact rare descriptor must rescue the target asset,
        # after which confidence still has to be grounded in its own chunks.
        service.indexes.search = AsyncMock(
            return_value=[(int(unbound_document["chunk_ids"][0]), 1.0)]
        )
        storage.lexical_search = AsyncMock(return_value=[])

        async def target_result() -> tuple[dict[str, object], dict[str, object]]:
            response = await service.search(
                "展示狱狼龙插图",
                retrieval_mode="standard",
                rerank=False,
                media_output_confidence_threshold=0.60,
                media_score_threshold=0.35,
            )
            decision = next(
                item
                for item in response["media_decisions"]
                if item["asset_id"] == target_image["id"]
            )
            return response, decision

        baseline_response, baseline = await target_result()
        assert [item["asset_id"] for item in baseline_response["media_outputs"]] == [
            target_image["id"]
        ]
        assert baseline["descriptor_rescue"] is True
        assert baseline["distinctive_support"] >= 0.8
        assert baseline["associated_chunk_count"] > 0
        assert {
            int(item["chunk_id"]) for item in baseline["evidence"]
        }.issubset(set(target_document["chunk_ids"]))
        assert baseline["grounding_reliability"] > 0

        unbound_image = await service.upload_image(
            filename="狱狼龙无绑定表情包.jpg",
            data=_png_bytes((722, 480)),
        )
        after_response, after = await target_result()
        assert after["output_confidence"] == pytest.approx(
            baseline["output_confidence"], abs=1e-6
        )
        assert after["association_score"] == pytest.approx(
            baseline["association_score"], abs=1e-6
        )
        assert after["evidence"] == baseline["evidence"]
        unbound_decision = next(
            item
            for item in after_response["media_decisions"]
            if item["asset_id"] == unbound_image["id"]
        )
        assert unbound_decision["confidence_algorithm"] == "unbound_asset_direct"
        assert unbound_decision["output_confidence"] > 0
        assert unbound_decision["associated_chunk_count"] == 0
        assert after["media_frequency_scope"] == "bound_active_media"
        assert after["media_frequency_corpus_size"] == 2
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_text_media_api_is_type_scoped_and_pkb_authenticates_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AppConfig(
        api_key="admin-key",
        session_secret="api-session",
        library_psk_secret="knowledge-secret",
    )
    context = ApplicationContext.create(
        source_root=Path(__file__).resolve().parents[1],
        state_root=tmp_path / "state",
        config=config,
        configure_logs=False,
    )
    monkeypatch.setattr(
        "personalityrag.library_types.text_media_v1.manager.build_provider",
        lambda _config: FixtureProvider(),
    )
    monkeypatch.setattr(
        "personalityrag.library_types.text_media_v1.resumable_tasks.build_provider",
        lambda _config: FixtureProvider(),
    )
    await context.manager.initialize()
    try:
        app = create_app(context)
        headers = {"Authorization": "Bearer admin-key"}
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test:8765"
        ) as client:
            created = await client.post(
                "/api/v1/knowledge-libraries/text_media_v1",
                headers=headers,
                json={
                    "id": "art",
                    "name": "Art Knowledge",
                    "provider_id": config.provider.id,
                },
            )
            assert created.status_code == 200, created.text
            initial_detail = created.json()
            visual_policy = dict(initial_detail["visual_intent_policy"])
            visual_policy["generation_action_terms"] = [
                *visual_policy["generation_action_terms"],
                "创作",
            ]
            policy_updated = await client.put(
                (
                    "/api/v1/knowledge-libraries/text_media_v1/art/"
                    "visual-intent-policy"
                ),
                headers=headers,
                json=visual_policy,
            )
            assert policy_updated.status_code == 200, policy_updated.text
            policy_payload = policy_updated.json()
            assert policy_payload["job_id"]
            assert policy_payload["provider_calls"] == 0
            assert policy_payload["index_rebuilt"] is False
            assert policy_payload["generation_unchanged"] is True
            assert policy_payload["media_generation_unchanged"] is True
            policy_job = await context.manager.jobs.get(
                policy_payload["job_id"]
            )
            assert policy_job and policy_job["task_type"] == {
                "lane": "short",
                "resumable": False,
                "adapter_blocking": False,
                "runtime_pause": True,
                "read_only": False,
                "embedding_context_policy": "none",
                "database_state_comparison": False,
            }
            detail_after_policy = await client.get(
                "/api/v1/knowledge-libraries/text_media_v1/art",
                headers=headers,
            )
            assert detail_after_policy.status_code == 200
            updated_detail = detail_after_policy.json()
            assert "创作" in updated_detail["visual_intent_policy"][
                "generation_action_terms"
            ]
            assert updated_detail["visual_intent_policy_is_default"] is False
            assert updated_detail["visual_intent_detector_version"] == (
                "lexicon_reference_rules_v1"
            )
            assert updated_detail["indexes"] == initial_detail["indexes"]
            export_draft = {
                key: list(values) for key, values in visual_policy.items()
            }
            export_draft["generation_action_terms"].append("只在草稿中")
            exported = await client.post(
                (
                    "/api/v1/knowledge-libraries/text_media_v1/art/"
                    "visual-intent-policy/export"
                ),
                headers=headers,
                json={
                    "categories": [
                        "visual_object_terms",
                        "generation_action_terms",
                    ],
                    "policy": export_draft,
                },
            )
            assert exported.status_code == 200, exported.text
            assert exported.content.startswith(b"\xef\xbb\xbf")
            assert exported.headers["content-type"].startswith("text/csv")
            assert (
                'filename="art-visual-intent-policy.csv"'
                in exported.headers["content-disposition"]
            )
            exported_policy, exported_categories = (
                parse_visual_intent_policy_csv(exported.content)
            )
            assert exported_categories == (
                "visual_object_terms",
                "generation_action_terms",
            )
            assert "只在草稿中" in exported_policy[
                "generation_action_terms"
            ]
            no_export_categories = await client.post(
                (
                    "/api/v1/knowledge-libraries/text_media_v1/art/"
                    "visual-intent-policy/export"
                ),
                headers=headers,
                json={"categories": [], "policy": export_draft},
            )
            assert no_export_categories.status_code == 422

            imported = await client.post(
                (
                    "/api/v1/knowledge-libraries/text_media_v1/art/"
                    "visual-intent-policy/imports/inspect"
                ),
                headers=headers,
                files={
                    "file": (
                        "draft.csv",
                        (
                            b"\xef\xbb\xbf"
                            b"generation_action_terms,lookup_action_terms\r\n"
                            b'"create, now",\r\n'
                        ),
                        "text/csv",
                    )
                },
            )
            assert imported.status_code == 200, imported.text
            assert imported.json() == {
                "filename": "draft.csv",
                "categories": [
                    {"key": "generation_action_terms", "count": 1},
                    {"key": "lookup_action_terms", "count": 0},
                ],
                "policy": {
                    "generation_action_terms": ["create, now"],
                    "lookup_action_terms": [],
                },
            }
            invalid_import = await client.post(
                (
                    "/api/v1/knowledge-libraries/text_media_v1/art/"
                    "visual-intent-policy/imports/inspect"
                ),
                headers=headers,
                files={"file": ("draft.txt", b"x", "text/plain")},
            )
            assert invalid_import.status_code == 400
            incomplete_policy = await client.put(
                (
                    "/api/v1/knowledge-libraries/text_media_v1/art/"
                    "visual-intent-policy"
                ),
                headers=headers,
                json={"visual_object_terms": ["立绘"]},
            )
            assert incomplete_policy.status_code == 422
            conflicting_pivots = await client.post(
                "/api/v1/knowledge-libraries/text_media_v1/art/search",
                headers=headers,
                json={
                    "query": "portrait",
                    "media_relevance_pivot": 0.2,
                    "media_score_threshold": 0.3,
                },
            )
            assert conflicting_pivots.status_code == 422
            uploaded = await client.post(
                "/api/v1/knowledge-libraries/text_media_v1/art/assets/images",
                headers=headers,
                data={
                    "media_description": "Art portrait",
                    "media_descriptions": json.dumps(
                        ["Art portrait", "Black armor character"]
                    ),
                },
                files={"file": ("art.png", _png_bytes((900, 600)), "image/png")},
            )
            assert uploaded.status_code == 200, uploaded.text
            asset_id = uploaded.json()["id"]
            assert uploaded.json()["job_id"]
            image_job = await context.manager.jobs.get(uploaded.json()["job_id"])
            assert image_job and image_job["task_type"]["lane"] == "short"
            descriptions_update = await client.put(
                (
                    "/api/v1/knowledge-libraries/text_media_v1/art/assets/"
                    f"{asset_id}/media-descriptions"
                ),
                headers=headers,
                json={
                    "media_descriptions": [
                        "Black armor character",
                        "Art portrait",
                        "Full-body standing illustration",
                    ]
                },
            )
            assert descriptions_update.status_code == 202, (
                descriptions_update.text
            )
            descriptions_job = await context.manager.jobs.wait(
                descriptions_update.json()["job_id"]
            )
            assert descriptions_job["status"] == "completed", descriptions_job
            assert descriptions_job["kind"] == (
                "text_media_media_descriptions_update"
            )
            assert descriptions_job["task_type"]["lane"] == "long"
            asset_detail = await client.get(
                (
                    "/api/v1/knowledge-libraries/text_media_v1/art/assets/"
                    f"{asset_id}"
                ),
                headers=headers,
            )
            assert asset_detail.status_code == 200, asset_detail.text
            assert asset_detail.json()["media_description"] == (
                "Black armor character"
            )
            assert [
                item["media_description"]
                for item in asset_detail.json()["media_descriptions"]
            ] == [
                "Black armor character",
                "Art portrait",
                "Full-body standing illustration",
            ]
            document_upload = await client.post(
                "/api/v1/knowledge-libraries/text_media_v1/art/documents",
                headers=headers,
                files={
                    "file": (
                        "art.md",
                        b"# Art\n\nportrait illustration details",
                        "text/markdown",
                    )
                },
            )
            assert document_upload.status_code == 202, document_upload.text
            document_job = await context.manager.jobs.wait(
                document_upload.json()["job_id"]
            )
            assert document_job["status"] == "completed"
            assert document_job["resumable"] is True
            assert document_job["task_type"]["lane"] == "long"

            pure_text = await client.post(
                "/api/v1/knowledge-libraries/text_media_v1/art/ingest-batches",
                headers=headers,
                data={
                    "manifest": json.dumps(
                        {
                            "images": [],
                            "chunk_target": 300,
                            "chunk_overlap": 30,
                        }
                    )
                },
                files=[
                    (
                        "documents[]",
                        (
                            "pure-text.txt",
                            b"pure text batch semantic content " * 20,
                            "text/plain",
                        ),
                    )
                ],
            )
            assert pure_text.status_code == 202, pure_text.text
            pure_text_job = await context.manager.jobs.wait(
                pure_text.json()["job_id"]
            )
            assert pure_text_job["status"] == "completed", pure_text_job
            assert (
                pure_text_job["result"]["parameters"]["ingest_mode"]
                == "text_only"
            )

            pure_media = await client.post(
                "/api/v1/knowledge-libraries/text_media_v1/art/ingest-batches",
                headers=headers,
                data={
                    "manifest": json.dumps(
                        {
                            "images": [
                                {
                                    "document_indexes": [],
                                    "media_description": "",
                                }
                            ]
                        }
                    )
                },
                files=[
                    (
                        "images[]",
                        (
                            "standalone-meme.png",
                            _png_bytes((901, 601)),
                            "image/png",
                        ),
                    )
                ],
            )
            assert pure_media.status_code == 202, pure_media.text
            pure_media_job = await context.manager.jobs.wait(
                pure_media.json()["job_id"]
            )
            assert pure_media_job["status"] == "completed", pure_media_job
            assert (
                pure_media_job["result"]["parameters"]["ingest_mode"]
                == "media_only"
            )
            assert pure_media_job["result"]["generation_id"] is None
            driver_key = context.manager._managers[TEXT_MEDIA_V1_TYPE].services.config.library_psk_secret
            from personalityrag.library_types.text_media_v1.driver import TextMediaV1Driver

            pkb = TextMediaV1Driver().derive_access_key(driver_key, "art")
            signed_path = (
                "/api/v1/knowledge-libraries/text_media_v1/art/assets/"
                f"{asset_id}/signed-url"
            )
            admin_signed = await client.post(
                signed_path,
                headers=headers,
                json={"variant": "content", "expires_in": 300},
            )
            assert admin_signed.status_code == 200, admin_signed.text
            assert "scope=" not in admin_signed.json()["url"]
            admin_signed_media = await client.get(admin_signed.json()["url"])
            assert admin_signed_media.status_code == 200
            assert admin_signed_media.headers["cache-control"] == (
                "private, max-age=3600"
            )

            missing_adapter_identity = await client.post(
                signed_path,
                headers={"Authorization": f"Bearer {pkb}"},
                json={"variant": "content"},
            )
            assert missing_adapter_identity.status_code == 400

            adapter_headers = {
                "Authorization": f"Bearer {pkb}",
                "X-PersonalityRAG-Adapter-ID": "Astrbot",
                "X-PersonalityRAG-Adapter-Instance-ID": "knowledge-instance",
                "X-PersonalityRAG-Adapter-Type": "astrbot-knowledge",
            }
            heartbeat = await client.post(
                (
                    "/api/v1/knowledge-libraries/text_media_v1/art/"
                    "adapters/heartbeat"
                ),
                headers=adapter_headers,
                json={},
            )
            assert heartbeat.status_code == 200, heartbeat.text
            first_connected_at = heartbeat.json()["connection"]["connected_at"]

            scoped_signed = await client.post(
                signed_path,
                headers=adapter_headers,
                json={"variant": "content", "expires_in": 60},
            )
            assert scoped_signed.status_code == 200, scoped_signed.text
            assert 295 <= scoped_signed.json()["expires"] - int(time.time()) <= 300
            scoped_url = scoped_signed.json()["url"]
            assert "scope=" in scoped_url
            scoped_media = await client.get(scoped_url)
            assert scoped_media.status_code == 200
            assert scoped_media.headers["cache-control"] == "private, no-store"
            repeated_scoped_media = await client.get(scoped_url)
            assert repeated_scoped_media.status_code == 200

            from personalityrag.library_types.text_media_v1.api import (
                _media_signature,
            )

            parsed_scoped_url = urlsplit(scoped_url)
            scoped_query = parse_qs(parsed_scoped_url.query)
            expired_at = int(time.time()) - 1
            expired_signature = _media_signature(
                "art",
                asset_id,
                "content",
                expired_at,
                scope=scoped_query["scope"][0],
            )
            expired_media = await client.get(
                (
                    f"{parsed_scoped_url.path}?expires={expired_at}"
                    f"&scope={scoped_query['scope'][0]}"
                    f"&signature={expired_signature}"
                )
            )
            assert expired_media.status_code == 401

            tampered_media = await client.get(scoped_url + "x")
            assert tampered_media.status_code == 401
            mismatched_instance = await client.post(
                signed_path,
                headers={
                    **adapter_headers,
                    "X-PersonalityRAG-Adapter-Instance-ID": "other-instance",
                },
                json={"variant": "content"},
            )
            assert mismatched_instance.status_code == 409
            assert (
                mismatched_instance.json()["detail"]["code"]
                == "adapter_connection_changed"
            )

            disconnected = await client.post(
                (
                    "/api/v1/knowledge-libraries/text_media_v1/art/"
                    "adapters/Astrbot/disconnect"
                ),
                headers=headers,
                json={"instance_id": "knowledge-instance"},
            )
            assert disconnected.status_code == 200, disconnected.text
            revoked_media = await client.get(scoped_url)
            assert revoked_media.status_code == 409
            assert (
                revoked_media.json()["detail"]["code"]
                == "adapter_forced_offline"
            )

            await asyncio.sleep(0.001)
            reconnected = await client.post(
                (
                    "/api/v1/knowledge-libraries/text_media_v1/art/"
                    "adapters/heartbeat"
                ),
                headers=adapter_headers,
                json={"manual_reconnect": True},
            )
            assert reconnected.status_code == 200, reconnected.text
            assert (
                reconnected.json()["connection"]["connected_at"]
                > first_connected_at
            )
            old_generation_media = await client.get(scoped_url)
            assert old_generation_media.status_code == 409
            assert (
                old_generation_media.json()["detail"]["code"]
                == "adapter_connection_changed"
            )

            replacement_signed = await client.post(
                signed_path,
                headers=adapter_headers,
                json={"variant": "content"},
            )
            assert replacement_signed.status_code == 200
            replacement_media = await client.get(replacement_signed.json()["url"])
            assert replacement_media.status_code == 200
            assert replacement_media.headers["cache-control"] == (
                "private, no-store"
            )
            control_db = await context.manager.control.connect()
            try:
                await control_db.execute(
                    """UPDATE database_adapter_connections
                    SET last_seen=?
                    WHERE database_type=? AND database_id=? AND adapter_id=?""",
                    (0, TEXT_MEDIA_V1_TYPE, "art", "Astrbot"),
                )
                await control_db.commit()
            finally:
                await control_db.close()
            stale_lease_media = await client.get(replacement_signed.json()["url"])
            assert stale_lease_media.status_code == 409
            assert (
                stale_lease_media.json()["detail"]["code"]
                == "adapter_connection_changed"
            )

            media = await client.get(
                f"/api/v1/knowledge-libraries/text_media_v1/art/assets/{asset_id}/content",
                headers={"Authorization": f"Bearer {pkb}"},
            )
            wrong = await client.get(
                "/api/v1/knowledge-libraries/text_media_v1/art",
                headers={"Authorization": "Bearer psk-invalid"},
            )
            generic = await client.get(
                "/api/v1/databases/text_media_v1/art", headers=headers
            )
        assert media.status_code == 200
        assert media.headers["content-type"] == "image/webp"
        assert media.content[:4] == b"RIFF"
        assert wrong.status_code == 401
        assert generic.status_code == 404
    finally:
        await context.manager.close()
