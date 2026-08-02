from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import httpx

from personalityrag.library_types.text_media_v1.retrieval import (
    calibrated_semantic,
    fuse_embedding_rerank,
    media_frequency_signals,
    noisy_or,
    unbound_media_confidence_factor,
)
from personalityrag.library_types.text_media_v1.text import (
    media_collection_intent,
    media_tokens,
    visual_intent,
)


OUTPUT_THRESHOLD = 0.60
EVIDENCE_THRESHOLD = 0.35
MEME_LABELS = frozenset({"claude", "deepseek", "gemini", "gpt"})


@dataclass(frozen=True)
class MediaCase:
    name: str
    query: str
    required: frozenset[str]
    allowed: frozenset[str]
    family: str


@dataclass(frozen=True)
class TextCase:
    query: str
    expected_title: str


def media_case_payload(case: MediaCase) -> dict[str, Any]:
    return {
        "name": case.name,
        "query": case.query,
        "required": sorted(case.required),
        "allowed": sorted(case.allowed),
        "family": case.family,
    }


def media_case_from_payload(payload: dict[str, Any]) -> MediaCase:
    return MediaCase(
        name=str(payload["name"]),
        query=str(payload["query"]),
        required=frozenset(str(value) for value in payload["required"]),
        allowed=frozenset(str(value) for value in payload["allowed"]),
        family=str(payload["family"]),
    )


MEDIA_CASES = (
    MediaCase(
        "claude_naked",
        "claude",
        frozenset({"claude"}),
        frozenset({"claude"}),
        "unbound",
    ),
    MediaCase(
        "claude_short",
        "给我一张 claude 表情包",
        frozenset({"claude"}),
        frozenset({"claude"}),
        "unbound",
    ),
    MediaCase(
        "deepseek_naked",
        "DEEPSEEK",
        frozenset({"deepseek"}),
        frozenset({"deepseek"}),
        "unbound",
    ),
    MediaCase(
        "deepseek_short",
        "给我一张 deepseek 表情包",
        frozenset({"deepseek"}),
        frozenset({"deepseek"}),
        "unbound",
    ),
    MediaCase(
        "deepseek_user_case",
        "给我deepseek表情包",
        frozenset({"deepseek"}),
        frozenset({"deepseek"}),
        "unbound",
    ),
    MediaCase(
        "deepseek_english",
        "show me a DEEPSEEK meme",
        frozenset({"deepseek"}),
        frozenset({"deepseek"}),
        "unbound",
    ),
    MediaCase(
        "deepseek_russian",
        "покажи мем deepseek",
        frozenset({"deepseek"}),
        frozenset({"deepseek"}),
        "unbound",
    ),
    MediaCase(
        "claude_direct",
        "看看 Claude 版‘原来是劣等模型’表情包",
        frozenset({"claude"}),
        frozenset({"claude"}),
        "unbound",
    ),
    MediaCase(
        "gemini_naked",
        "Gemini",
        frozenset({"gemini"}),
        frozenset({"gemini"}),
        "unbound",
    ),
    MediaCase(
        "gemini_short",
        "给我一张 gemini 表情包",
        frozenset({"gemini"}),
        frozenset({"gemini"}),
        "unbound",
    ),
    MediaCase(
        "claude_paraphrase",
        "给我那张嘲讽 Claude 是劣等模型的梗图",
        frozenset({"claude"}),
        frozenset({"claude"}),
        "unbound",
    ),
    MediaCase(
        "gpt_naked",
        "gpt",
        frozenset({"gpt"}),
        frozenset({"gpt"}),
        "unbound",
    ),
    MediaCase(
        "gpt_short",
        "给我一张gpt表情包",
        frozenset({"gpt"}),
        frozenset({"gpt"}),
        "unbound",
    ),
    MediaCase(
        "gpt_english",
        "show me the GPT meme",
        frozenset({"gpt"}),
        frozenset({"gpt"}),
        "unbound",
    ),
    MediaCase(
        "gpt_russian",
        "покажи мем GPT",
        frozenset({"gpt"}),
        frozenset({"gpt"}),
        "unbound",
    ),
    MediaCase(
        "deepseek_direct",
        "展示 DeepSeek 版劣等模型表情包",
        frozenset({"deepseek"}),
        frozenset({"deepseek"}),
        "unbound",
    ),
    MediaCase(
        "deepseek_paraphrase",
        "深度求索那个‘原来是劣等模型’的表情",
        frozenset({"deepseek"}),
        frozenset({"deepseek"}),
        "unbound",
    ),
    MediaCase(
        "gemini_direct",
        "Gemini 版本的劣等模型表情包",
        frozenset({"gemini"}),
        frozenset({"gemini"}),
        "unbound",
    ),
    MediaCase(
        "gemini_paraphrase",
        "谷歌 Gemini 被叫作劣等模型的那张梗图",
        frozenset({"gemini"}),
        frozenset({"gemini"}),
        "unbound",
    ),
    MediaCase(
        "gpt_direct",
        "看看 GPT 版原来是劣等模型表情包",
        frozenset({"gpt"}),
        frozenset({"gpt"}),
        "unbound",
    ),
    MediaCase(
        "gpt_paraphrase",
        "OpenAI GPT 被吐槽是劣等模型的表情",
        frozenset({"gpt"}),
        frozenset({"gpt"}),
        "unbound",
    ),
    MediaCase(
        "generic_meme",
        "给我一张表情包",
        frozenset(),
        frozenset(),
        "negative",
    ),
    MediaCase(
        "all_memes",
        "把四种‘原来是劣等模型’表情包都给我看",
        MEME_LABELS,
        MEME_LABELS,
        "unbound",
    ),
    MediaCase(
        "all_memes_short",
        "把四种表情包都给我看",
        MEME_LABELS,
        MEME_LABELS,
        "unbound",
    ),
    MediaCase(
        "selected_memes",
        "把 GPT 和 Claude 表情包都给我",
        frozenset({"gpt", "claude"}),
        frozenset({"gpt", "claude"}),
        "unbound",
    ),
    MediaCase(
        "generic_collection",
        "把所有图片都给我",
        frozenset(),
        frozenset(),
        "negative",
    ),
    MediaCase(
        "full_portrait",
        "描述一下你的全身立绘",
        frozenset({"full"}),
        frozenset({"full", "avatar"}),
        "bound",
    ),
    MediaCase(
        "avatar",
        "给我看看你的脸部头像",
        frozenset({"avatar"}),
        frozenset({"avatar"}),
        "bound",
    ),
    MediaCase(
        "battle_scene",
        "展示你在竞技场战斗的场景图",
        frozenset({"battle"}),
        frozenset({"battle"}),
        "bound",
    ),
    MediaCase(
        "stygian_ecology",
        "展示狱狼龙生态插图",
        frozenset({"stygian"}),
        frozenset({"stygian"}),
        "bound",
    ),
    MediaCase(
        "stygian_paraphrase",
        "给我看红黑色雷光环绕的狱狼龙图片",
        frozenset({"stygian"}),
        frozenset({"stygian"}),
        "bound",
    ),
    MediaCase(
        "negative_build",
        "给我看看黑龙太刀配装图",
        frozenset(),
        frozenset(),
        "negative",
    ),
    MediaCase(
        "negative_bankai",
        "展示斩魄刀卍解形态图片",
        frozenset(),
        frozenset(),
        "negative",
    ),
    MediaCase(
        "negative_novel",
        "找一张《闪光的哈萨维》小说封面",
        frozenset(),
        frozenset(),
        "negative",
    ),
    MediaCase(
        "negative_weather",
        "随便来一张今天的天气照片",
        frozenset(),
        frozenset(),
        "negative",
    ),
)

TEXT_CASES = (
    TextCase("不动衣装怎么获得和强化", "全特殊装备获取与强化深度解析"),
    TextCase("富野由悠季写的小说内容", "闪光的哈萨维"),
    TextCase("aibo 是谁", "群友"),
    TextCase("纳刀术对太刀有什么作用", "终盘太刀配装"),
    TextCase("金黄澄月卍解有什么能力", "斩魄刀的设定"),
    TextCase("狱狼龙如何吸引蝕龍蟲", "狱狼龙"),
)


def classify_asset(name: str) -> str:
    value = name.casefold()
    if "claude" in value:
        return "claude"
    if "deepseek" in value:
        return "deepseek"
    if "gemini" in value:
        return "gemini"
    if "gpt" in value:
        return "gpt"
    if "狱狼龙插图" in value:
        return "stygian"
    if "头像" in value:
        return "avatar"
    if "战斗场景" in value:
        return "battle"
    if "全身立绘" in value:
        return "full"
    return f"other:{value}"


def request_json(
    client: httpx.Client, method: str, url: str, **kwargs: Any
) -> dict[str, Any]:
    response = client.request(method, url, **kwargs)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError(f"unexpected response from {url}")
    return value


def search(
    client: httpx.Client,
    library_url: str,
    query: str,
    *,
    retrieval_mode: str,
    rerank: bool,
    top_k: int = 10,
    max_media_outputs: int = 8,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    result = request_json(
        client,
        "POST",
        f"{library_url}/search",
        json={
            "query": query,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "media_output_confidence_threshold": OUTPUT_THRESHOLD,
            "media_score_threshold": EVIDENCE_THRESHOLD,
            "max_media_outputs": max_media_outputs,
            "rerank": rerank,
        },
    )
    return result, (time.perf_counter() - started) * 1000.0


def capture_raw_media(
    client: httpx.Client,
    library_url: str,
    *,
    rerank: bool,
) -> tuple[list[dict[str, Any]], list[float]]:
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for case in MEDIA_CASES:
        result, elapsed = search(
            client,
            library_url,
            case.query,
            retrieval_mode="media_only",
            rerank=rerank,
        )
        latencies.append(elapsed)
        decisions = []
        for item in result.get("media_decisions", []):
            direct_relations = list(item.get("direct_relations") or [])
            matched_description_id = item.get("matched_media_description_id")
            direct = next(
                (
                    relation
                    for relation in direct_relations
                    if relation.get("media_description_id")
                    == matched_description_id
                ),
                direct_relations[0] if direct_relations else {},
            )
            decisions.append(
                {
                    "label": classify_asset(str(item.get("original_name") or "")),
                    "asset_metadata_only": bool(item.get("asset_metadata_only")),
                    "media_vector_similarity": item.get("media_vector_similarity"),
                    "lexical_coverage": float(item.get("lexical_coverage") or 0),
                    "matched_tokens": list(item.get("matched_tokens") or []),
                    "media_frequency_corpus_size": int(
                        item.get("media_frequency_corpus_size") or 0
                    ),
                    "frequency_token_details": list(
                        item.get("frequency_token_details") or []
                    ),
                    "base_lexical_score": float(
                        item.get("lexical_score") or 0
                    ),
                    "distinctive_support": float(
                        item.get("distinctive_support") or 0
                    ),
                    "distinctive_membership_support": float(
                        item.get("distinctive_membership_support") or 0
                    ),
                    "collection_support": float(
                        item.get("collection_support") or 0
                    ),
                    "collection_membership_support": float(
                        item.get("collection_membership_support") or 0
                    ),
                    "rerank_raw_score": direct.get("rerank_raw_score"),
                    "rerank_rank": direct.get("rerank_rank"),
                    "api_output_confidence": float(
                        item.get("output_confidence") or 0
                    ),
                    "api_output": bool(item.get("output")),
                }
            )
        rows.append(
            {
                "case": media_case_payload(case),
                "decisions": decisions,
                "rerank_candidate_count": len(decisions),
                "api_outputs": [
                    classify_asset(str(item.get("original_name") or ""))
                    for item in result.get("media_outputs", [])
                ],
                "rerank": dict(result.get("rerank") or {}),
            }
        )
    return rows, latencies


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def parameter_neighborhood(
    center: float,
    offsets: tuple[float, ...],
    *,
    lower: float,
    upper: float,
) -> tuple[float, ...]:
    return tuple(
        sorted(
            {
                round(max(lower, min(upper, center + offset)), 4)
                for offset in offsets
            }
        )
    )


def score_grid(
    snapshots: dict[str, list[dict[str, Any]]],
    rerank_settings: dict[str, Any],
    parameter_values: dict[str, tuple[float, ...]] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    values = parameter_values or {
        "distinctive_boost": (0.35, 0.40, 0.46, 0.50, 0.55),
        "collection_boost": (0.35, 0.40, 0.45, 0.50, 0.55),
        "competition_floor": (0.35, 0.40, 0.45, 0.50, 0.55),
        "reliability_target": (0.25, 0.35, 0.45),
        "specificity_exponent": (1.0, 1.25, 1.5, 2.0),
        "advantage_target": (0.04, 0.06, 0.08, 0.10),
    }
    fixed_semantic_floor = float(rerank_settings["media_semantic_floor"])
    fixed_semantic_weight = float(rerank_settings["media_semantic_weight"])
    fixed_lexical_boost = float(rerank_settings["media_lexical_boost"])
    fixed_lexical_exponent = float(
        rerank_settings["media_lexical_coverage_exponent"]
    )
    fixed_common_floor = float(rerank_settings["media_lexical_common_floor"])
    fixed_oov_penalty = float(rerank_settings["media_lexical_oov_penalty"])
    fixed_rarity_exponent = float(
        rerank_settings["media_distinctive_rarity_exponent"]
    )
    for (
        distinctive_boost,
        collection_boost,
        competition_floor,
        reliability_target,
        specificity_exponent,
        advantage_target,
    ) in itertools.product(
        values["distinctive_boost"],
        values["collection_boost"],
        values["competition_floor"],
        values["reliability_target"],
        values["specificity_exponent"],
        values["advantage_target"],
    ):
        tp = fp = fn = 0
        reciprocal_ranks: list[float] = []
        top1_hits = 0
        top1_total = 0
        required_hits = required_total = negative_outputs = 0
        margins: list[float] = []
        for library_id, rows in snapshots.items():
            library_uses_rerank = library_id == "beileite_test"
            for row in rows:
                case = media_case_from_payload(row["case"])
                if case.family not in {"unbound", "negative"}:
                    continue
                scored: list[tuple[str, float]] = []
                collection_membership_by_label: dict[str, float] = {}
                for decision in row["decisions"]:
                    if not decision["asset_metadata_only"]:
                        continue
                    similarity = decision["media_vector_similarity"]
                    semantic = calibrated_semantic(
                        float(similarity) if similarity is not None else None,
                        fixed_semantic_floor,
                    )
                    if (
                        library_uses_rerank
                        and decision["rerank_raw_score"] is not None
                        and decision["rerank_rank"] is not None
                    ):
                        semantic = float(
                            fuse_embedding_rerank(
                                semantic,
                                float(decision["rerank_raw_score"]),
                                rank=int(decision["rerank_rank"]),
                                candidate_count=int(row["rerank_candidate_count"]),
                                fusion_weight=float(
                                    rerank_settings["rerank_fusion_weight"]
                                ),
                                rank_bonus_weight=float(
                                    rerank_settings["rerank_rank_bonus_weight"]
                                ),
                                rank_reliability_exponent=float(
                                    rerank_settings[
                                        "rerank_rank_reliability_exponent"
                                    ]
                                ),
                            )["fused_relevance"]
                        )
                    token_details = list(
                        decision.get("frequency_token_details") or []
                    )
                    document_frequencies = {
                        str(item["token"]): int(
                            item.get("document_frequency") or 0
                        )
                        for item in token_details
                    }
                    signals = media_frequency_signals(
                        media_tokens(case.query),
                        list(decision.get("matched_tokens") or []),
                        document_frequencies,
                        int(decision.get("media_frequency_corpus_size") or 0),
                        coverage_exponent=fixed_lexical_exponent,
                        common_floor=fixed_common_floor,
                        oov_penalty=fixed_oov_penalty,
                        rarity_exponent=fixed_rarity_exponent,
                        collection_intent=media_collection_intent(case.query),
                    )
                    confidence = noisy_or(
                        (
                            fixed_semantic_weight * semantic,
                            fixed_lexical_boost
                            * float(signals["base_lexical_score"]),
                            distinctive_boost
                            * float(signals["distinctive_support"]),
                            collection_boost
                            * max(
                                float(signals["collection_support"]),
                                float(
                                    signals[
                                        "collection_membership_support"
                                    ]
                                ),
                                float(
                                    signals[
                                        "distinctive_membership_support"
                                    ]
                                ),
                            ),
                        )
                    )
                    label = str(decision["label"])
                    scored.append((label, confidence))
                    collection_membership_by_label[label] = max(
                        float(signals["distinctive_membership_support"]),
                        float(signals["collection_membership_support"]),
                        float(signals["collection_support"]),
                    )
                best_direct = max((value for _, value in scored), default=0.0)
                collection_intent = media_collection_intent(case.query)
                best_collection_membership = max(
                    collection_membership_by_label.values(), default=0.0
                )
                adjusted: list[tuple[str, float]] = []
                for label, value in scored:
                    strongest_competitor = max(
                        (
                            other_value
                            for other_label, other_value in scored
                            if other_label != label
                        ),
                        default=0.0,
                    )
                    adjustment = unbound_media_confidence_factor(
                        value,
                        best_direct,
                        strongest_competitor_score=strongest_competitor,
                        collection_membership_support=(
                            collection_membership_by_label.get(label, 0.0)
                            / best_collection_membership
                            if best_collection_membership > 0.0
                            else 0.0
                        ),
                        collection_intent=collection_intent,
                        competition_floor=competition_floor,
                        reliability_target=reliability_target,
                        specificity_exponent=specificity_exponent,
                        advantage_target=advantage_target,
                    )
                    adjusted.append(
                        (
                            label,
                            value * float(adjustment["confidence_factor"]),
                        )
                    )
                scored = adjusted
                scored.sort(key=lambda item: (-item[1], item[0]))
                predicted = {
                    label for label, value in scored if value >= OUTPUT_THRESHOLD
                }
                expected = set(case.required)
                tp += len(predicted & expected)
                fp += len(predicted - set(case.allowed))
                fn += len(expected - predicted)
                required_hits += len(predicted & expected)
                required_total += len(expected)
                if not expected:
                    negative_outputs += len(predicted)
                else:
                    expected_ranks = [
                        index
                        for index, (label, _) in enumerate(scored, start=1)
                        if label in expected
                    ]
                    reciprocal_ranks.append(
                        1.0 / min(expected_ranks) if expected_ranks else 0.0
                    )
                    if len(expected) == 1:
                        top1_total += 1
                        if scored and scored[0][0] in expected:
                            top1_hits += 1
                    true_values = [
                        value for label, value in scored if label in expected
                    ]
                    false_values = [
                        value for label, value in scored if label not in case.allowed
                    ]
                    if true_values:
                        margins.append(
                            min(true_values) - max(false_values, default=0.0)
                        )
        macro_f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0
        results.append(
            {
                "unbound_media_distinctive_boost": distinctive_boost,
                "unbound_media_collection_boost": collection_boost,
                "unbound_media_competition_floor": competition_floor,
                "unbound_media_reliability_target": reliability_target,
                "unbound_media_specificity_exponent": specificity_exponent,
                "unbound_media_advantage_target": advantage_target,
                "required_recall": (
                    required_hits / required_total if required_total else 1.0
                ),
                "negative_outputs": negative_outputs,
                "macro_f1": macro_f1,
                "mrr": (
                    sum(reciprocal_ranks) / len(reciprocal_ranks)
                    if reciprocal_ranks
                    else 1.0
                ),
                "top1_accuracy": top1_hits / top1_total if top1_total else 1.0,
                "minimum_margin": min(margins, default=0.0),
            }
        )
    results.sort(
        key=lambda item: (
            -int(
                item["required_recall"] == 1.0
                and item["negative_outputs"] == 0
            ),
            -item["required_recall"],
            item["negative_outputs"],
            -item["macro_f1"],
            -item["top1_accuracy"],
            -item["mrr"],
            -item["minimum_margin"],
            item["unbound_media_distinctive_boost"],
            item["unbound_media_collection_boost"],
            item["unbound_media_competition_floor"],
            item["unbound_media_specificity_exponent"],
            item["unbound_media_advantage_target"],
            item["unbound_media_reliability_target"],
        )
    )
    return results


def actual_media_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_hits = required_total = false_outputs = negative_outputs = 0
    top1_hits = top1_total = 0
    failures: list[dict[str, Any]] = []
    hierarchy_failures: list[dict[str, Any]] = []
    for row in rows:
        case = media_case_from_payload(row["case"])
        outputs = list(row["api_outputs"])
        predicted = set(outputs)
        expected = set(case.required)
        required_hits += len(predicted & expected)
        required_total += len(expected)
        false = predicted - set(case.allowed)
        false_outputs += len(false)
        if not expected:
            negative_outputs += len(predicted)
        elif len(expected) == 1:
            top1_total += 1
            if outputs and outputs[0] in expected:
                top1_hits += 1
        if not expected <= predicted or false:
            failures.append(
                {
                    "case": case.name,
                    "required": sorted(expected),
                    "allowed": sorted(case.allowed),
                    "outputs": outputs,
                }
            )
        confidence_by_label = {
            str(item["label"]): float(item.get("api_output_confidence") or 0.0)
            for item in row.get("decisions", [])
        }
        if (
            len(expected) == 1
            and expected <= MEME_LABELS
            and bool(visual_intent(case.query, gate_enabled=True)["detected"])
        ):
            target = next(iter(expected))
            siblings = [
                confidence_by_label.get(label, 0.0)
                for label in MEME_LABELS - {target}
            ]
            unrelated = [
                value
                for label, value in confidence_by_label.items()
                if label not in MEME_LABELS
            ]
            if (
                not siblings
                or min(siblings) <= 0.0
                or min(siblings) <= max(unrelated, default=0.0)
            ):
                hierarchy_failures.append(
                    {
                        "case": case.name,
                        "sibling_min": min(siblings, default=0.0),
                        "unrelated_max": max(unrelated, default=0.0),
                    }
                )
        if case.name == "generic_meme":
            meme_values = [
                confidence_by_label.get(label, 0.0) for label in MEME_LABELS
            ]
            unrelated = [
                value
                for label, value in confidence_by_label.items()
                if label not in MEME_LABELS
            ]
            if min(meme_values, default=0.0) <= max(unrelated, default=0.0):
                hierarchy_failures.append(
                    {
                        "case": case.name,
                        "meme_min": min(meme_values, default=0.0),
                        "unrelated_max": max(unrelated, default=0.0),
                    }
                )
    return {
        "required_recall": required_hits / required_total if required_total else 1.0,
        "false_outputs": false_outputs,
        "negative_outputs": negative_outputs,
        "top1_accuracy": top1_hits / top1_total if top1_total else 1.0,
        "failures": failures,
        "hierarchy_failures": hierarchy_failures,
    }


def text_quality(
    client: httpx.Client,
    library_url: str,
    *,
    rerank: bool,
) -> dict[str, Any]:
    hits = 0
    rows = []
    for case in TEXT_CASES:
        result, elapsed = search(
            client,
            library_url,
            case.query,
            retrieval_mode="standard",
            rerank=rerank,
            max_media_outputs=5,
        )
        titles = [str(item.get("title") or "") for item in result.get("items", [])]
        rank = next(
            (
                index
                for index, title in enumerate(titles, start=1)
                if case.expected_title in title
            ),
            None,
        )
        hits += int(rank is not None and rank <= 3)
        rows.append(
            {
                "query": case.query,
                "expected_title": case.expected_title,
                "rank": rank,
                "top_titles": titles[:5],
                "media_outputs": [
                    classify_asset(str(item.get("original_name") or ""))
                    for item in result.get("media_outputs", [])
                ],
                "execution": dict(result.get("execution") or {}),
                "elapsed_ms": elapsed,
            }
        )
    fast_path_count = sum(
        not bool(row["execution"].get("bound_media_executed")) for row in rows
    )
    unexpected_media_outputs = sum(len(row["media_outputs"]) for row in rows)
    return {
        "recall_at_3": hits / len(TEXT_CASES),
        "pure_text_fast_path_count": fast_path_count,
        "unexpected_media_outputs": unexpected_media_outputs,
        "rows": rows,
    }


def embedding_baseline_parity(
    client: httpx.Client,
    primary_url: str,
    control_url: str,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    maximum_score_delta = 0.0
    for case in TEXT_CASES:
        primary, _ = search(
            client,
            primary_url,
            case.query,
            retrieval_mode="standard",
            rerank=False,
        )
        control, _ = search(
            client,
            control_url,
            case.query,
            retrieval_mode="standard",
            rerank=False,
        )

        def text_signature(result: dict[str, Any]) -> list[tuple[str, str, float]]:
            return [
                (
                    str(item.get("title") or ""),
                    hashlib.sha256(
                        str(item.get("text") or "").encode("utf-8")
                    ).hexdigest(),
                    float(item.get("score") or 0),
                )
                for item in result.get("items", [])
            ]

        primary_signature = text_signature(primary)
        control_signature = text_signature(control)
        identities_match = [row[:2] for row in primary_signature] == [
            row[:2] for row in control_signature
        ]
        score_delta = max(
            (
                abs(left[2] - right[2])
                for left, right in zip(
                    primary_signature, control_signature, strict=False
                )
            ),
            default=0.0,
        )
        maximum_score_delta = max(maximum_score_delta, score_delta)
        if not identities_match or score_delta > 1e-4:
            failures.append(
                {
                    "scope": "text",
                    "query": case.query,
                    "identities_match": identities_match,
                    "score_delta": score_delta,
                }
            )

    for case in MEDIA_CASES:
        primary, _ = search(
            client,
            primary_url,
            case.query,
            retrieval_mode="media_only",
            rerank=False,
        )
        control, _ = search(
            client,
            control_url,
            case.query,
            retrieval_mode="media_only",
            rerank=False,
        )

        def media_signature(result: dict[str, Any]) -> list[tuple[str, bool, float]]:
            return [
                (
                    classify_asset(str(item.get("original_name") or "")),
                    bool(item.get("output")),
                    float(item.get("output_confidence") or 0),
                )
                for item in result.get("media_decisions", [])
            ]

        primary_signature = media_signature(primary)
        control_signature = media_signature(control)
        primary_by_label = {row[0]: row[1:] for row in primary_signature}
        control_by_label = {row[0]: row[1:] for row in control_signature}
        primary_outputs = [
            row[0] for row in primary_signature if row[1]
        ]
        control_outputs = [
            row[0] for row in control_signature if row[1]
        ]
        identities_match = (
            primary_outputs == control_outputs
            and primary_by_label.keys() == control_by_label.keys()
            and all(
                primary_by_label[label][0] == control_by_label[label][0]
                for label in primary_by_label
            )
        )
        score_delta = max(
            (
                abs(
                    float(primary_by_label[label][1])
                    - float(control_by_label[label][1])
                )
                for label in primary_by_label.keys() & control_by_label.keys()
            ),
            default=0.0,
        )
        maximum_score_delta = max(maximum_score_delta, score_delta)
        # The two libraries were embedded and recalibrated by independent real
        # GPU calls. Their asset/description hashes are identical, while tiny
        # vector drift can expand through several nonlinear confidence stages.
        # Output identities must remain exact; numeric drift is bounded here
        # and same-library mode parity remains exact elsewhere.
        if not identities_match or score_delta > 2e-2:
            failures.append(
                {
                    "scope": "media",
                    "case": case.name,
                    "identities_match": identities_match,
                    "score_delta": score_delta,
                }
            )
    return {
        "passed": not failures,
        "maximum_score_delta": maximum_score_delta,
        "text_score_tolerance": 1e-4,
        "media_score_tolerance": 2e-2,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate all three text_media_v1 retrieval modes."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PERSONALITYRAG_API_KEY"),
        help="Bearer token; defaults to PERSONALITYRAG_API_KEY.",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--confirm-libraries",
        nargs="*",
        default=[],
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error(
            "--api-key or PERSONALITYRAG_API_KEY is required"
        )
    library_ids = ("beileite_test", "beileite_test2")
    if args.apply and set(args.confirm_libraries) != set(library_ids):
        parser.error("--apply requires confirmation for both benchmark libraries")
    base_url = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.api_key}"}
    report: dict[str, Any] = {"created_at": time.time(), "libraries": {}}
    with httpx.Client(headers=headers, timeout=180.0) as client:
        details = {
            library_id: request_json(
                client,
                "GET",
                (
                    f"{base_url}/api/v1/knowledge-libraries/text_media_v1/"
                    f"{library_id}"
                ),
            )
            for library_id in library_ids
        }
        urls = {
            library_id: (
                f"{base_url}/api/v1/knowledge-libraries/text_media_v1/"
                f"{library_id}"
            )
            for library_id in library_ids
        }
        originals = {
            library_id: dict(detail["retrieval_settings"])
            for library_id, detail in details.items()
        }
        if details["beileite_test"].get("rerank_provider_id") == "":
            raise RuntimeError("beileite_test must retain its Rerank binding")
        if details["beileite_test2"].get("rerank_provider_id"):
            raise RuntimeError("beileite_test2 must remain the Embedding-only control")
        snapshots: dict[str, list[dict[str, Any]]] = {}
        all_latencies: dict[str, list[float]] = {}
        applied = False
        try:
            for library_id in library_ids:
                capture_settings = {
                    **originals[library_id],
                    "unbound_media_candidate_limit": 100,
                }
                request_json(
                    client,
                    "PATCH",
                    urls[library_id],
                    json={"retrieval_settings": capture_settings},
                )
                snapshots[library_id], all_latencies[library_id] = capture_raw_media(
                    client,
                    urls[library_id],
                    rerank=library_id == "beileite_test",
                )

            coarse_grid = score_grid(snapshots, originals["beileite_test"])
            coarse_winner = coarse_grid[0]
            fine_grid = score_grid(
                snapshots,
                originals["beileite_test"],
                {
                    "distinctive_boost": parameter_neighborhood(
                        float(
                            coarse_winner[
                                "unbound_media_distinctive_boost"
                            ]
                        ),
                        (-0.04, -0.02, 0.0, 0.02, 0.04),
                        lower=0.35,
                        upper=0.55,
                    ),
                    "collection_boost": parameter_neighborhood(
                        float(
                            coarse_winner[
                                "unbound_media_collection_boost"
                            ]
                        ),
                        (-0.10, -0.05, 0.0, 0.05, 0.10),
                        lower=0.35,
                        upper=0.55,
                    ),
                    "competition_floor": parameter_neighborhood(
                        float(
                            coarse_winner[
                                "unbound_media_competition_floor"
                            ]
                        ),
                        (-0.10, -0.05, 0.0, 0.05, 0.10),
                        lower=0.35,
                        upper=0.55,
                    ),
                    "reliability_target": parameter_neighborhood(
                        float(
                            coarse_winner[
                                "unbound_media_reliability_target"
                            ]
                        ),
                        (-0.10, -0.05, 0.0, 0.05, 0.10),
                        lower=0.25,
                        upper=0.45,
                    ),
                    "specificity_exponent": parameter_neighborhood(
                        float(
                            coarse_winner[
                                "unbound_media_specificity_exponent"
                            ]
                        ),
                        (-0.50, -0.25, 0.0, 0.25, 0.50),
                        lower=1.0,
                        upper=2.0,
                    ),
                    "advantage_target": parameter_neighborhood(
                        float(
                            coarse_winner[
                                "unbound_media_advantage_target"
                            ]
                        ),
                        (-0.04, -0.02, 0.0, 0.02, 0.04),
                        lower=0.04,
                        upper=0.10,
                    ),
                },
            )
            hard_passes = [
                item
                for item in fine_grid
                if item["required_recall"] == 1.0
                and item["negative_outputs"] == 0
                and item["macro_f1"] == 1.0
                and item["top1_accuracy"] == 1.0
            ]
            best_margin = max(
                (float(item["minimum_margin"]) for item in hard_passes),
                default=float("-inf"),
            )
            near_tied = [
                item
                for item in hard_passes
                if best_margin - float(item["minimum_margin"]) < 0.002
            ]
            near_tied.sort(
                key=lambda item: (
                    item["unbound_media_distinctive_boost"],
                    item["unbound_media_collection_boost"],
                    item["unbound_media_competition_floor"],
                    item["unbound_media_specificity_exponent"],
                    item["unbound_media_advantage_target"],
                    item["unbound_media_reliability_target"],
                )
            )

            def settings_for_candidate(
                base: dict[str, Any], candidate: dict[str, Any]
            ) -> dict[str, Any]:
                return {
                    **base,
                    "unbound_media_candidate_limit": 10,
                    "unbound_media_distinctive_boost": candidate[
                        "unbound_media_distinctive_boost"
                    ],
                    "unbound_media_collection_boost": candidate[
                        "unbound_media_collection_boost"
                    ],
                    "unbound_media_competition_floor": candidate[
                        "unbound_media_competition_floor"
                    ],
                    "unbound_media_reliability_target": candidate[
                        "unbound_media_reliability_target"
                    ],
                    "unbound_media_specificity_exponent": candidate[
                        "unbound_media_specificity_exponent"
                    ],
                    "unbound_media_advantage_target": candidate[
                        "unbound_media_advantage_target"
                    ],
                }

            original_candidate = {
                key: originals["beileite_test"][key]
                for key in (
                    "unbound_media_distinctive_boost",
                    "unbound_media_collection_boost",
                    "unbound_media_competition_floor",
                    "unbound_media_reliability_target",
                    "unbound_media_specificity_exponent",
                    "unbound_media_advantage_target",
                )
            }
            validation_candidates: list[dict[str, Any]] = []
            validation_keys: set[tuple[float, ...]] = set()
            for candidate in [
                *(near_tied[:10] if near_tied else fine_grid[:10]),
                coarse_winner,
                original_candidate,
            ]:
                key = tuple(
                    float(candidate[field])
                    for field in (
                        "unbound_media_distinctive_boost",
                        "unbound_media_collection_boost",
                        "unbound_media_competition_floor",
                        "unbound_media_reliability_target",
                        "unbound_media_specificity_exponent",
                        "unbound_media_advantage_target",
                    )
                )
                if key in validation_keys:
                    continue
                validation_keys.add(key)
                validation_candidates.append(candidate)

            provider_grid_validation: list[dict[str, Any]] = []
            provider_valid_candidates: list[dict[str, Any]] = []
            for candidate in validation_candidates:
                candidate_result = {
                    "settings": settings_for_candidate(
                        originals["beileite_test"], candidate
                    ),
                    "libraries": {},
                }
                candidate_passed = True
                for library_id in library_ids:
                    candidate_settings = settings_for_candidate(
                        originals[library_id], candidate
                    )
                    request_json(
                        client,
                        "PATCH",
                        urls[library_id],
                        json={"retrieval_settings": candidate_settings},
                    )
                    rounds = []
                    for _ in range(3):
                        provider_rows, provider_latencies = capture_raw_media(
                            client,
                            urls[library_id],
                            rerank=library_id == "beileite_test",
                        )
                        metrics = actual_media_metrics(provider_rows)
                        passed = (
                            metrics["required_recall"] == 1.0
                            and metrics["false_outputs"] == 0
                            and metrics["negative_outputs"] == 0
                            and metrics["top1_accuracy"] == 1.0
                            and not metrics["hierarchy_failures"]
                        )
                        candidate_passed = candidate_passed and passed
                        rounds.append(
                            {
                                "passed": passed,
                                "metrics": metrics,
                                "p95_ms": percentile(
                                    provider_latencies, 0.95
                                ),
                            }
                        )
                    candidate_result["libraries"][library_id] = rounds
                candidate_result["passed"] = candidate_passed
                provider_grid_validation.append(candidate_result)
                if candidate_passed:
                    provider_valid_candidates.append(candidate)

            winner = (
                provider_valid_candidates[0]
                if provider_valid_candidates
                else near_tied[0]
                if near_tied
                else fine_grid[0]
            )
            winning_settings = {
                **settings_for_candidate(
                    originals["beileite_test"], winner
                ),
            }
            candidate_limit_checks: dict[str, Any] = {}
            for library_id in library_ids:
                signatures: dict[str, Any] = {}
                for limit in (10, 20, 30, 50, 100):
                    settings = {
                        **winning_settings,
                        "unbound_media_candidate_limit": limit,
                    }
                    request_json(
                        client,
                        "PATCH",
                        urls[library_id],
                        json={"retrieval_settings": settings},
                    )
                    case_signatures: dict[str, Any] = {}
                    for case in MEDIA_CASES:
                        result, _ = search(
                            client,
                            urls[library_id],
                            case.query,
                            retrieval_mode="media_only",
                            rerank=library_id == "beileite_test",
                        )
                        case_signatures[case.name] = [
                            (
                                classify_asset(
                                    str(item.get("original_name") or "")
                                ),
                                float(item.get("output_confidence") or 0),
                                bool(item.get("output")),
                            )
                            for item in result.get("media_decisions", [])
                        ]
                    signatures[str(limit)] = case_signatures
                candidate_limit_checks[library_id] = {
                    "stable": len(
                        {
                            json.dumps(value, sort_keys=True)
                            for value in signatures.values()
                        }
                    )
                    == 1,
                    "signatures": signatures,
                }

            final_snapshots: dict[str, list[dict[str, Any]]] = {}
            for library_id in library_ids:
                request_json(
                    client,
                    "PATCH",
                    urls[library_id],
                    json={"retrieval_settings": winning_settings},
                )
                final_snapshots[library_id], final_latencies = capture_raw_media(
                    client,
                    urls[library_id],
                    rerank=library_id == "beileite_test",
                )
                report["libraries"][library_id] = {
                    "media_metrics": actual_media_metrics(
                        final_snapshots[library_id]
                    ),
                    "text_quality": text_quality(
                        client,
                        urls[library_id],
                        rerank=library_id == "beileite_test",
                    ),
                    "cold_capture_p95_ms": percentile(
                        all_latencies[library_id], 0.95
                    ),
                    "final_capture_p95_ms": percentile(final_latencies, 0.95),
                }

            standard_unbound_retrieval: dict[str, Any] = {}
            media_mode_parity: dict[str, Any] = {}
            text_mode_parity: dict[str, Any] = {}
            top_k_stability: dict[str, Any] = {}
            for library_id in library_ids:
                standard_failures = []
                parity_failures = []
                for case in MEDIA_CASES:
                    standard_result, _ = search(
                        client,
                        urls[library_id],
                        case.query,
                        retrieval_mode="standard",
                        rerank=library_id == "beileite_test",
                    )
                    if case.family == "unbound":
                        labels = {
                            classify_asset(str(item.get("original_name") or ""))
                            for item in standard_result.get("media_outputs", [])
                        }
                        if not set(case.required).issubset(labels) or (
                            labels - set(case.allowed)
                        ):
                            standard_failures.append(
                                {"case": case.name, "outputs": sorted(labels)}
                            )
                    intent = visual_intent(case.query)
                    if bool(intent.get("detected")):
                        media_result, _ = search(
                            client,
                            urls[library_id],
                            case.query,
                            retrieval_mode="media_only",
                            rerank=library_id == "beileite_test",
                        )
                        standard_signature = {
                            "outputs": standard_result.get("media_outputs", []),
                            "decisions": standard_result.get("media_decisions", []),
                        }
                        media_signature = {
                            "outputs": media_result.get("media_outputs", []),
                            "decisions": media_result.get("media_decisions", []),
                        }
                        if standard_signature != media_signature:
                            parity_failures.append(
                                {
                                    "case": case.name,
                                    "standard": standard_signature,
                                    "media_only": media_signature,
                                }
                            )
                standard_unbound_retrieval[library_id] = {
                    "passed": not standard_failures,
                    "failures": standard_failures,
                }
                media_mode_parity[library_id] = {
                    "passed": not parity_failures,
                    "failures": parity_failures,
                }

                text_failures = []
                for case in TEXT_CASES:
                    standard_text, _ = search(
                        client,
                        urls[library_id],
                        case.query,
                        retrieval_mode="standard",
                        rerank=library_id == "beileite_test",
                    )
                    text_only, _ = search(
                        client,
                        urls[library_id],
                        case.query,
                        retrieval_mode="text_only",
                        rerank=library_id == "beileite_test",
                    )

                    def text_signature(result: dict[str, Any]) -> list[tuple[Any, ...]]:
                        return [
                            (
                                item.get("chunk_id"),
                                item.get("rank"),
                                item.get("score"),
                                item.get("initial_rank"),
                                item.get("initial_score"),
                                item.get("rerank_raw_score"),
                                item.get("rerank_rank"),
                            )
                            for item in result.get("items", [])
                        ]

                    if text_signature(standard_text) != text_signature(text_only):
                        text_failures.append(
                            {
                                "query": case.query,
                                "standard": text_signature(standard_text),
                                "text_only": text_signature(text_only),
                            }
                        )
                    if (
                        text_only.get("media_outputs")
                        or text_only.get("media_decisions")
                        or bool(
                            (text_only.get("execution") or {}).get(
                                "media_channel_executed"
                            )
                        )
                    ):
                        text_failures.append(
                            {
                                "query": case.query,
                                "reason": "text_only_executed_media_channel",
                            }
                        )
                text_mode_parity[library_id] = {
                    "passed": not text_failures,
                    "failures": text_failures,
                }

                signatures = {}
                for top_k in (5, 10, 20, 50):
                    result, _ = search(
                        client,
                        urls[library_id],
                        "描述一下你的立绘",
                        retrieval_mode="standard",
                        rerank=library_id == "beileite_test",
                        top_k=top_k,
                    )
                    signatures[str(top_k)] = [
                        (
                            classify_asset(str(item.get("original_name") or "")),
                            float(item.get("output_confidence") or 0),
                            bool(item.get("output")),
                        )
                        for item in result.get("media_decisions", [])
                    ]
                top_k_stability[library_id] = {
                    "stable": len(
                        {
                            json.dumps(value, sort_keys=True)
                            for value in signatures.values()
                        }
                    )
                    == 1,
                    "signatures": signatures,
                }

            baseline_parity = embedding_baseline_parity(
                client,
                urls["beileite_test"],
                urls["beileite_test2"],
            )
            report.update(
                {
                    "coarse_grid_top_20": coarse_grid[:20],
                    "fine_grid_top_20": fine_grid[:20],
                    # Compatibility alias for older local report readers.
                    "grid_top_20": fine_grid[:20],
                    "coarse_winner": coarse_winner,
                    "fine_best_margin": best_margin,
                    "fine_near_tied_count": len(near_tied),
                    "provider_grid_validation": provider_grid_validation,
                    "winner": winner,
                    "winning_settings": winning_settings,
                    "candidate_limit_checks": candidate_limit_checks,
                    "standard_unbound_retrieval": standard_unbound_retrieval,
                    "media_mode_parity": media_mode_parity,
                    "text_mode_parity": text_mode_parity,
                    "top_k_stability": top_k_stability,
                    "embedding_baseline_parity": baseline_parity,
                    "raw_snapshots": snapshots,
                    "final_snapshots": final_snapshots,
                }
            )
            hard_failures = []
            for library_id in library_ids:
                metrics = report["libraries"][library_id]["media_metrics"]
                if (
                    metrics["required_recall"] < 1.0
                    or metrics["negative_outputs"]
                    or metrics["top1_accuracy"] < 1.0
                    or report["libraries"][library_id]["text_quality"][
                        "recall_at_3"
                    ]
                    < 1.0
                    or report["libraries"][library_id]["text_quality"][
                        "unexpected_media_outputs"
                    ]
                    or not candidate_limit_checks[library_id]["stable"]
                    or not standard_unbound_retrieval[library_id]["passed"]
                    or not media_mode_parity[library_id]["passed"]
                    or not text_mode_parity[library_id]["passed"]
                    or not top_k_stability[library_id]["stable"]
                ):
                    hard_failures.append(library_id)
            if not baseline_parity["passed"]:
                hard_failures.append("embedding_baseline_parity")
            report["hard_failures"] = hard_failures
            if args.apply and not hard_failures:
                applied = True
        finally:
            if not applied:
                for library_id in library_ids:
                    request_json(
                        client,
                        "PATCH",
                        urls[library_id],
                        json={"retrieval_settings": originals[library_id]},
                    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "winner": report["winner"],
                "winning_settings": report["winning_settings"],
                "libraries": {
                    key: value["media_metrics"]
                    for key, value in report["libraries"].items()
                },
                "hard_failures": report["hard_failures"],
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if report["hard_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
