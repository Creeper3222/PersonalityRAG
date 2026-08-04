from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_text_media_v1_rerank import (  # noqa: E402
    evaluate as evaluate_media_acceptance,
    recalibrate_all,
    signature as media_signature,
)
from personalityrag.library_types.text_media_v1.retrieval import (  # noqa: E402
    DEFAULT_RETRIEVAL_CONFIG,
)


DEFAULT_QRELS = Path("tests/fixtures/text_media_v1_semantic_qrels.json")
DEFAULT_SNAPSHOT = Path("reports/text_media_v1_semantic_rerank_snapshot.json")
DEFAULT_REPORT = Path("reports/text_media_v1_semantic_rerank_report.json")
DEFAULT_DATA_ROOT = Path("data/databases/knowledge_bases/text_media_v1")
MEDIA_LABELS = {
    "澄月全身立绘.png": "full",
    "澄月的立绘头像.png": "avatar",
    "澄月战斗场景插图.jpg": "battle",
}


@dataclass(frozen=True)
class QueryCase:
    id: str
    topic: str
    family: str
    style: str
    query: str
    qrels: dict[tuple[str, int], int]
    allowed_media: frozenset[str]
    required_media: frozenset[str]
    top_media: str | None
    sealed: bool
    hard_negative: bool


def load_cases(path: Path) -> tuple[dict[str, Any], list[QueryCase]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "personalityrag.text_media_v1.semantic_qrels":
        raise ValueError("unexpected qrels format")
    styles = tuple(str(value) for value in payload.get("styles") or ())
    if styles != ("direct", "second_person", "paraphrase", "indirect"):
        raise ValueError("semantic qrels must use the fixed four-style order")
    topics = list(payload.get("topics") or [])
    if len(topics) != 25:
        raise ValueError("semantic qrels must contain exactly 25 topics")

    def build(
        row: dict[str, Any],
        *,
        case_id: str,
        topic: str,
        style: str,
        query: str,
        sealed: bool,
        hard_negative: bool,
    ) -> QueryCase:
        qrels: dict[tuple[str, int], int] = {}
        for item in row.get("qrels") or []:
            key = (str(item["document"]), int(item["ordinal"]))
            grade = int(item["grade"])
            if key in qrels or grade < 0 or grade > 3:
                raise ValueError(f"invalid qrel in {case_id}")
            qrels[key] = grade
        media = dict(row.get("media") or {})
        allowed = frozenset(str(value) for value in media.get("allowed") or [])
        required = frozenset(str(value) for value in media.get("required") or [])
        top = str(media["top"]) if media.get("top") else None
        if not required.issubset(allowed) or (top is not None and top not in allowed):
            raise ValueError(f"invalid media labels in {case_id}")
        return QueryCase(
            id=case_id,
            topic=topic,
            family=str(row["family"]),
            style=style,
            query=query,
            qrels=qrels,
            allowed_media=allowed,
            required_media=required,
            top_media=top,
            sealed=sealed,
            hard_negative=hard_negative,
        )

    cases: list[QueryCase] = []
    for topic in topics:
        queries = dict(topic.get("queries") or {})
        if tuple(queries) != styles:
            raise ValueError(f"topic {topic.get('id')} does not contain four styles")
        for style in styles:
            cases.append(
                build(
                    topic,
                    case_id=f"{topic['id']}:{style}",
                    topic=str(topic["id"]),
                    style=style,
                    query=str(queries[style]),
                    sealed=style == "indirect",
                    hard_negative=False,
                )
            )
    for section, sealed, hard_negative in (
        ("conflicts", True, False),
        ("hard_negatives", True, True),
    ):
        for row in payload.get(section) or []:
            cases.append(
                build(
                    row,
                    case_id=str(row["id"]),
                    topic=str(row["id"]),
                    style=section,
                    query=str(row["query"]),
                    sealed=sealed,
                    hard_negative=hard_negative,
                )
            )
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("qrels contain duplicate case ids")
    return payload, cases


def _readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def corpus_snapshot(data_root: Path, library: str) -> dict[str, Any]:
    path = data_root / library / "textmediaknowledge.db"
    with _readonly_connection(path) as connection:
        documents = {
            str(row["id"]): dict(row)
            for row in connection.execute(
                "SELECT * FROM documents WHERE status='ready'"
            )
        }
        chunk_rows: list[dict[str, Any]] = []
        for row in connection.execute(
            """
            SELECT c.id,c.ordinal,c.text,c.content_sha256,e.document_id,
                   ce.vector,ce.vector_sha256
            FROM chunks c
            JOIN entries e ON e.id=c.entry_id
            JOIN chunk_embeddings ce ON ce.chunk_id=c.id
            JOIN active_generation ag ON ag.generation_id=ce.generation_id
            WHERE c.status='active'
            ORDER BY e.document_id,c.ordinal
            """
        ):
            item = dict(row)
            item["document"] = str(documents[str(item["document_id"])]["title"])
            item["key"] = (item["document"], int(item["ordinal"]))
            chunk_rows.append(item)
        assets = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM assets WHERE state='active' ORDER BY original_name"
            )
        ]
        relations = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM document_assets ORDER BY document_id,asset_id"
            )
        ]
        meta = dict(
            connection.execute(
                "SELECT * FROM library_meta WHERE singleton=1"
            ).fetchone()
        )
    return {
        "library": library,
        "path": str(path),
        "documents": documents,
        "chunks": chunk_rows,
        "assets": assets,
        "relations": relations,
        "meta": meta,
        "chunk_by_id": {int(row["id"]): row for row in chunk_rows},
        "chunk_by_key": {row["key"]: row for row in chunk_rows},
        "asset_by_id": {str(row["id"]): row for row in assets},
    }


def verify_corpus_parity(
    baseline: dict[str, Any], rerank: dict[str, Any]
) -> dict[str, Any]:
    baseline_chunks = baseline["chunk_by_key"]
    rerank_chunks = rerank["chunk_by_key"]
    keys_equal = set(baseline_chunks) == set(rerank_chunks)
    vector_sha_equal = sum(
        baseline_chunks[key]["vector_sha256"]
        == rerank_chunks[key]["vector_sha256"]
        for key in set(baseline_chunks) & set(rerank_chunks)
    )
    vector_cosines: list[float] = []
    if keys_equal:
        import numpy as np

        for key in sorted(baseline_chunks):
            first = np.frombuffer(baseline_chunks[key]["vector"], dtype=np.float32)
            second = np.frombuffer(rerank_chunks[key]["vector"], dtype=np.float32)
            vector_cosines.append(
                float(first @ second)
                / max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-12)
            )
    def document_hashes(data: dict[str, Any]) -> list[str]:
        return sorted(
            str(item["content_sha256"])
            for item in data["documents"].values()
        )

    asset_fields = (
        "kind",
        "sha256",
        "original_name",
        "mime_type",
        "size_bytes",
        "width",
        "height",
    )
    def assets(data: dict[str, Any]) -> list[tuple[Any, ...]]:
        return sorted(
            tuple(item[field] for field in asset_fields)
            for item in data["assets"]
        )

    relation_fields = (
        "role",
        "relation_weight",
        "caption",
        "alt_text",
        "output_policy",
        "sort_order",
        "semantic_mode",
        "media_description",
    )
    def relations(data: dict[str, Any]) -> list[tuple[Any, ...]]:
        return sorted(
            tuple(item[field] for field in relation_fields)
            for item in data["relations"]
        )

    retrieval_equal = (
        json.loads(str(baseline["meta"]["retrieval_config_json"]))
        == json.loads(str(rerank["meta"]["retrieval_config_json"]))
    )
    result = {
        "document_hashes_equal": document_hashes(baseline)
        == document_hashes(rerank),
        "chunk_keys_equal": keys_equal,
        "chunk_count": len(baseline_chunks),
        "vector_sha_equal_count": vector_sha_equal,
        "minimum_vector_cosine": min(vector_cosines, default=0.0),
        "maximum_vector_cosine_delta": max(
            (abs(1.0 - value) for value in vector_cosines), default=math.inf
        ),
        "assets_equal": assets(baseline) == assets(rerank),
        "relations_equal": relations(baseline) == relations(rerank),
        "retrieval_settings_equal": retrieval_equal,
    }
    result["passed"] = bool(
        result["document_hashes_equal"]
        and result["chunk_keys_equal"]
        and result["chunk_count"] == 28
        and result["vector_sha_equal_count"] == 28
        and result["maximum_vector_cosine_delta"] <= 1e-6
        and result["assets_equal"]
        and result["relations_equal"]
        and result["retrieval_settings_equal"]
    )
    return result


def _request_json(
    client: httpx.Client, method: str, url: str, **kwargs: Any
) -> dict[str, Any]:
    response = client.request(method, url, **kwargs)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected response from {url}")
    return payload


def _search(
    client: httpx.Client,
    library_url: str,
    query: str,
    *,
    rerank: bool,
    top_k: int = 50,
) -> dict[str, Any]:
    return _request_json(
        client,
        "POST",
        f"{library_url}/search",
        json={
            "query": query,
            "top_k": top_k,
            "media_output_confidence_threshold": 0.60,
            "media_score_threshold": 0.35,
            "max_media_outputs": 5,
            "rerank": rerank,
        },
    )


def _stable_items(
    result: dict[str, Any], corpus: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in result.get("items") or []:
        chunk = corpus["chunk_by_id"][int(item["chunk_id"])]
        rows.append(
            {
                "document": chunk["document"],
                "ordinal": int(chunk["ordinal"]),
                "content_sha256": str(chunk["content_sha256"]),
                "score": float(item.get("score") or 0.0),
                "initial_score": float(item.get("initial_score") or 0.0),
                "initial_rank": int(item.get("initial_rank") or item.get("rank") or 0),
                "rerank_raw_score": (
                    float(item["rerank_raw_score"])
                    if item.get("rerank_raw_score") is not None
                    else None
                ),
                "rerank_rank": (
                    int(item["rerank_rank"])
                    if item.get("rerank_rank") is not None
                    else None
                ),
                "fused_score": (
                    float(item["fused_score"])
                    if item.get("fused_score") is not None
                    else None
                ),
            }
        )
    return rows


def _media_labels(result: dict[str, Any]) -> list[str]:
    return [
        MEDIA_LABELS.get(str(item.get("original_name") or ""), "unknown")
        for item in result.get("media_outputs") or []
    ]


def verify_api_baseline_parity(
    client: httpx.Client,
    cases: Iterable[QueryCase],
    baseline_url: str,
    rerank_url: str,
    baseline_corpus: dict[str, Any],
    rerank_corpus: dict[str, Any],
) -> dict[str, Any]:
    order_mismatches: list[str] = []
    media_mismatches: list[str] = []
    max_score_delta = 0.0
    provider_candidates = 0
    for case in cases:
        first = _search(client, baseline_url, case.query, rerank=False)
        second = _search(client, rerank_url, case.query, rerank=False)
        first_items = _stable_items(first, baseline_corpus)
        second_items = _stable_items(second, rerank_corpus)
        first_keys = [
            (row["document"], row["ordinal"], row["content_sha256"])
            for row in first_items
        ]
        second_keys = [
            (row["document"], row["ordinal"], row["content_sha256"])
            for row in second_items
        ]
        if first_keys != second_keys:
            order_mismatches.append(case.id)
        for left, right in zip(first_items, second_items, strict=True):
            max_score_delta = max(
                max_score_delta, abs(left["score"] - right["score"])
            )
        if _media_labels(first) != _media_labels(second):
            media_mismatches.append(case.id)
        for result in (first, second):
            rerank_meta = dict(result.get("rerank") or {})
            provider_candidates += int(rerank_meta.get("provider_candidates") or 0)
            if rerank_meta.get("applied") or rerank_meta.get("scopes"):
                raise RuntimeError("rerank=false unexpectedly applied Rerank")
    result = {
        "queries": len(list(cases)) if not isinstance(cases, list) else len(cases),
        "order_mismatches": order_mismatches,
        "media_mismatches": media_mismatches,
        "maximum_score_delta": max_score_delta,
        "rerank_provider_candidates": provider_candidates,
    }
    result["passed"] = bool(
        not order_mismatches
        and not media_mismatches
        and max_score_delta <= 1e-6
        and provider_candidates == 0
    )
    return result


def _direct_rerank(
    client: httpx.Client,
    endpoint: str,
    model: str,
    query: str,
    documents: list[str],
) -> tuple[list[float], float]:
    started = time.perf_counter()
    payload = _request_json(
        client,
        "POST",
        endpoint,
        json={
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        },
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    results = list(payload.get("results") or [])
    scores: list[float | None] = [None] * len(documents)
    for item in results:
        index = int(item["index"])
        if index < 0 or index >= len(documents) or scores[index] is not None:
            raise RuntimeError("Rerank provider returned invalid indexes")
        score = float(item["relevance_score"])
        if not math.isfinite(score):
            raise RuntimeError("Rerank provider returned a non-finite score")
        scores[index] = score
    if any(score is None for score in scores):
        raise RuntimeError("Rerank provider returned an incomplete candidate set")
    return [float(score) for score in scores], elapsed


def collect_snapshot(
    client: httpx.Client,
    provider_client: httpx.Client,
    cases: list[QueryCase],
    baseline_url: str,
    baseline_corpus: dict[str, Any],
    *,
    rerank_endpoint: str,
    rerank_model: str,
    provider_rounds: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    provider_elapsed: list[float] = []
    maximum_round_delta = 0.0
    for index, case in enumerate(cases, start=1):
        baseline = _search(client, baseline_url, case.query, rerank=False)
        items = _stable_items(baseline, baseline_corpus)
        documents = [
            str(
                baseline_corpus["chunk_by_key"][(row["document"], row["ordinal"])][
                    "text"
                ]
            )
            for row in items
        ]
        rounds: list[list[float]] = []
        for _ in range(max(1, provider_rounds)):
            scores, elapsed = _direct_rerank(
                provider_client,
                rerank_endpoint,
                rerank_model,
                case.query,
                documents,
            )
            rounds.append(scores)
            provider_elapsed.append(elapsed)
        for other in rounds[1:]:
            maximum_round_delta = max(
                maximum_round_delta,
                max(abs(left - right) for left, right in zip(rounds[0], other)),
            )
        rows.append(
            {
                "id": case.id,
                "query": case.query,
                "baseline": items,
                "rerank_raw_scores": rounds[0],
                "provider_rounds": len(rounds),
            }
        )
        if index % 10 == 0 or index == len(cases):
            print(f"snapshot {index}/{len(cases)}", flush=True)
    ordered = sorted(provider_elapsed)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "format": "personalityrag.text_media_v1.semantic_rerank_snapshot",
        "version": 1,
        "created_at": time.time(),
        "rerank_endpoint": rerank_endpoint,
        "rerank_model": rerank_model,
        "query_count": len(rows),
        "maximum_provider_round_delta": maximum_round_delta,
        "provider_latency_ms": {
            "mean": statistics.fmean(provider_elapsed),
            "p95": ordered[p95_index],
        },
        "rows": rows,
    }


def canonical_rerank_score(raw_score: float) -> float:
    value = float(raw_score)
    if not math.isfinite(value):
        raise ValueError("rerank score must be finite")
    if 0.0 <= value <= 1.0:
        return value
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def noisy_or(*values: float) -> float:
    remaining = 1.0
    for value in values:
        remaining *= 1.0 - max(0.0, min(1.0, float(value)))
    return max(0.0, min(1.0, 1.0 - remaining))


def fused_relevance(
    embedding_signal: float,
    probability: float,
    *,
    rank: int,
    count: int,
    fusion_weight: float,
    rank_bonus_weight: float,
    exponent: float,
) -> float:
    percentile = 1.0 if count <= 1 else 1.0 - (rank - 1) / (count - 1)
    reliable_probability = probability ** exponent
    support = percentile * reliable_probability
    rerank_signal = noisy_or(
        reliable_probability, rank_bonus_weight * support
    )
    return max(
        0.0,
        min(
            1.0,
            (1.0 - fusion_weight) * embedding_signal
            + fusion_weight * rerank_signal,
        ),
    )


def monotonic_residual_relevance(
    embedding_signal: float,
    probability: float,
    *,
    rank: int,
    count: int,
    fusion_weight: float,
    rank_bonus_weight: float,
    exponent: float,
) -> float:
    percentile = 1.0 if count <= 1 else 1.0 - (rank - 1) / (count - 1)
    support = percentile * probability ** exponent
    rerank_signal = noisy_or(probability, rank_bonus_weight * support)
    return noisy_or(embedding_signal, fusion_weight * rerank_signal)


def rerank_order(
    row: dict[str, Any],
    *,
    strategy: str,
    candidate_limit: int,
    fusion_weight: float,
    rank_bonus_weight: float,
    exponent: float,
    gate: float,
) -> list[tuple[str, int]]:
    baseline = list(row["baseline"])
    count = min(len(baseline), max(1, int(candidate_limit)))
    head = baseline[:count]
    tail = baseline[count:]
    raw = [float(value) for value in row["rerank_raw_scores"][:count]]
    probability = [canonical_rerank_score(value) for value in raw]
    provider_order = sorted(range(count), key=lambda value: (-raw[value], value))
    provider_rank = {
        value: rank for rank, value in enumerate(provider_order, start=1)
    }
    embedding = [float(item["score"]) for item in head]
    fused = [
        fused_relevance(
            embedding[index],
            probability[index],
            rank=provider_rank[index],
            count=count,
            fusion_weight=fusion_weight,
            rank_bonus_weight=rank_bonus_weight,
            exponent=exponent,
        )
        for index in range(count)
    ]
    residual = [
        monotonic_residual_relevance(
            embedding[index],
            probability[index],
            rank=provider_rank[index],
            count=count,
            fusion_weight=fusion_weight,
            rank_bonus_weight=rank_bonus_weight,
            exponent=exponent,
        )
        for index in range(count)
    ]
    ordinary_convex = [
        max(
            0.0,
            min(
                1.0,
                (1.0 - fusion_weight) * embedding[index]
                + fusion_weight * probability[index],
            ),
        )
        for index in range(count)
    ]
    if strategy == "embedding":
        indexes = list(range(count))
    elif strategy == "production_gate":
        indexes = list(range(count))
        trusted_positions = [
            index for index, score in enumerate(probability) if score >= gate
        ]
        trusted = sorted(
            trusted_positions,
            key=lambda index: (provider_rank[index], -fused[index], index),
        )
        for position, index in zip(trusted_positions, trusted, strict=True):
            indexes[position] = index
    elif strategy == "dual_channel":
        indexes = sorted(
            range(count), key=lambda index: (-fused[index], index)
        )
    elif strategy == "convex":
        indexes = sorted(
            range(count), key=lambda index: (-ordinary_convex[index], index)
        )
    elif strategy == "monotonic_residual":
        indexes = sorted(
            range(count), key=lambda index: (-residual[index], index)
        )
    elif strategy == "rank_blend":
        values: list[float] = []
        for index in range(count):
            baseline_percentile = (
                1.0 if count <= 1 else 1.0 - index / (count - 1)
            )
            provider_percentile = (
                1.0
                if count <= 1
                else 1.0 - (provider_rank[index] - 1) / (count - 1)
            )
            reliability = (
                max(0.0, (probability[index] - gate) / max(1.0 - gate, 1e-12))
                ** exponent
            )
            effective_weight = min(
                1.0,
                fusion_weight
                + rank_bonus_weight * reliability * provider_percentile,
            )
            values.append(
                baseline_percentile
                + effective_weight
                * reliability
                * (provider_percentile - baseline_percentile)
            )
        indexes = sorted(range(count), key=lambda index: (-values[index], index))
    elif strategy == "pure_provider":
        indexes = provider_order
    else:
        raise ValueError(f"unknown strategy {strategy}")
    ordered = [head[index] for index in indexes] + tail
    return [(str(item["document"]), int(item["ordinal"])) for item in ordered]


def _dcg(grades: list[int]) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def case_metrics(case: QueryCase, order: list[tuple[str, int]]) -> dict[str, float]:
    grades = [int(case.qrels.get(key, 0)) for key in order]
    ideal = sorted(case.qrels.values(), reverse=True)
    ideal_dcg = _dcg(ideal[:10])
    ndcg = _dcg(grades[:10]) / ideal_dcg if ideal_dcg else 0.0
    best_grade = max(case.qrels.values(), default=0)
    best_rank = next(
        (rank for rank, grade in enumerate(grades, start=1) if grade == best_grade),
        0,
    )
    return {
        "ndcg_at_10": ndcg,
        "reciprocal_rank": 1.0 / best_rank if best_rank else 0.0,
        "recall_at_1": float(any(grade == best_grade for grade in grades[:1])),
        "recall_at_3": float(any(grade == best_grade for grade in grades[:3])),
        "recall_at_5": float(any(grade == best_grade for grade in grades[:5])),
        "best_rank": float(best_rank),
    }


def aggregate_metrics(
    cases: list[QueryCase], orders: dict[str, list[tuple[str, int]]]
) -> dict[str, Any]:
    rows: dict[str, dict[str, float]] = {}
    families: dict[str, list[float]] = {}
    sealed: list[float] = []
    for case in cases:
        if not case.qrels:
            continue
        metrics = case_metrics(case, orders[case.id])
        rows[case.id] = metrics
        families.setdefault(case.family, []).append(metrics["ndcg_at_10"])
        if case.sealed:
            sealed.append(metrics["ndcg_at_10"])
    return {
        "ndcg_at_10": statistics.fmean(
            value["ndcg_at_10"] for value in rows.values()
        ),
        "mrr": statistics.fmean(
            value["reciprocal_rank"] for value in rows.values()
        ),
        "recall_at_1": statistics.fmean(
            value["recall_at_1"] for value in rows.values()
        ),
        "recall_at_3": statistics.fmean(
            value["recall_at_3"] for value in rows.values()
        ),
        "recall_at_5": statistics.fmean(
            value["recall_at_5"] for value in rows.values()
        ),
        "sealed_ndcg_at_10": statistics.fmean(sealed) if sealed else 0.0,
        "worst_family_ndcg_at_10": min(
            statistics.fmean(values) for values in families.values()
        ),
        "families": {
            family: statistics.fmean(values)
            for family, values in sorted(families.items())
        },
        "rows": rows,
    }


def regression_count(
    cases: list[QueryCase],
    baseline: dict[str, list[tuple[str, int]]],
    candidate: dict[str, list[tuple[str, int]]],
) -> int:
    regressions = 0
    for case in cases:
        if not case.qrels:
            continue
        relevant = {
            key for key, grade in case.qrels.items() if grade == max(case.qrels.values())
        }
        baseline_top3 = relevant.intersection(baseline[case.id][:3])
        if baseline_top3 and not baseline_top3.intersection(candidate[case.id][:3]):
            regressions += 1
            continue
        if baseline[case.id][0] in relevant and not relevant.intersection(
            candidate[case.id][:3]
        ):
            regressions += 1
    return regressions


def bootstrap_interval(
    baseline_values: list[float],
    candidate_values: list[float],
    *,
    iterations: int = 2000,
    seed: int = 20260722,
) -> tuple[float, float]:
    if len(baseline_values) != len(candidate_values) or not baseline_values:
        return (0.0, 0.0)
    differences = [
        right - left for left, right in zip(baseline_values, candidate_values)
    ]
    randomizer = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [randomizer.choice(differences) for _ in differences]
        means.append(statistics.fmean(sample))
    means.sort()
    return (
        means[max(0, int(0.025 * iterations) - 1)],
        means[min(iterations - 1, int(0.975 * iterations))],
    )


def grid_evaluate(
    cases: list[QueryCase], snapshot: dict[str, Any]
) -> dict[str, Any]:
    row_by_id = {str(row["id"]): row for row in snapshot["rows"]}
    if set(row_by_id) != {case.id for case in cases}:
        raise ValueError("snapshot and qrels query sets differ")
    embedding_orders = {
        case.id: rerank_order(
            row_by_id[case.id],
            strategy="embedding",
            candidate_limit=50,
            fusion_weight=0.0,
            rank_bonus_weight=0.0,
            exponent=1.0,
            gate=0.0,
        )
        for case in cases
    }
    baseline_metrics = aggregate_metrics(cases, embedding_orders)
    values = itertools.product(
        ("dual_channel", "production_gate", "convex", "rank_blend"),
        (10, 14, 20, 30, 50),
        (
            0.0,
            0.005,
            0.01,
            0.02,
            0.025,
            0.03,
            0.0325,
            0.035,
            0.0375,
            0.04,
            0.08,
            0.12,
            0.20,
        ),
        (0.0, 0.05, 0.10, 0.20, 0.30),
        (1.0, 1.5, 2.0, 2.5, 3.0),
        (0.0, 0.002, 0.005, 0.008, 0.012, 0.02),
    )
    candidates: list[dict[str, Any]] = []
    seen_orders: set[str] = set()
    for strategy, limit, fusion, bonus, exponent, gate in values:
        orders = {
            case.id: rerank_order(
                row_by_id[case.id],
                strategy=strategy,
                candidate_limit=limit,
                fusion_weight=fusion,
                rank_bonus_weight=bonus,
                exponent=exponent,
                gate=gate,
            )
            for case in cases
        }
        signature = hashlib.sha256(
            json.dumps(orders, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if signature in seen_orders:
            continue
        seen_orders.add(signature)
        metrics = aggregate_metrics(cases, orders)
        regressions = regression_count(cases, embedding_orders, orders)
        candidates.append(
            {
                "strategy": strategy,
                "candidate_limit": limit,
                "fusion_weight": fusion,
                "rank_bonus_weight": bonus,
                "rank_reliability_exponent": exponent,
                "gate": gate,
                "regressions": regressions,
                "metrics": {
                    key: value
                    for key, value in metrics.items()
                    if key not in {"rows", "families"}
                },
                "families": metrics["families"],
                "orders": orders,
                "row_metrics": metrics["rows"],
            }
        )
    for limit in (10, 14, 20, 30, 50):
        orders = {
            case.id: rerank_order(
                row_by_id[case.id],
                strategy="pure_provider",
                candidate_limit=limit,
                fusion_weight=1.0,
                rank_bonus_weight=0.0,
                exponent=1.0,
                gate=0.0,
            )
            for case in cases
        }
        metrics = aggregate_metrics(cases, orders)
        candidates.append(
            {
                "strategy": "pure_provider",
                "candidate_limit": limit,
                "fusion_weight": 1.0,
                "rank_bonus_weight": 0.0,
                "rank_reliability_exponent": 1.0,
                "gate": 0.0,
                "regressions": regression_count(cases, embedding_orders, orders),
                "metrics": {
                    key: value
                    for key, value in metrics.items()
                    if key not in {"rows", "families"}
                },
                "families": metrics["families"],
                "orders": orders,
                "row_metrics": metrics["rows"],
            }
        )
    candidates.sort(
        key=lambda item: (
            -int(item["regressions"]),
            item["metrics"]["sealed_ndcg_at_10"],
            item["metrics"]["worst_family_ndcg_at_10"],
            item["metrics"]["ndcg_at_10"],
            item["metrics"]["mrr"],
            item["metrics"]["recall_at_1"],
            item["metrics"]["recall_at_3"],
            item["metrics"]["recall_at_5"],
            -float(item["fusion_weight"]),
            -int(item["candidate_limit"]),
        ),
        reverse=True,
    )
    best = candidates[0]
    sealed_cases = [case for case in cases if case.sealed and case.qrels]
    baseline_sealed = [
        baseline_metrics["rows"][case.id]["ndcg_at_10"] for case in sealed_cases
    ]
    best_sealed = [best["row_metrics"][case.id]["ndcg_at_10"] for case in sealed_cases]
    interval = bootstrap_interval(baseline_sealed, best_sealed)
    compact = []
    for candidate in candidates[:20]:
        compact.append(
            {
                key: candidate[key]
                for key in (
                    "strategy",
                    "candidate_limit",
                    "fusion_weight",
                    "rank_bonus_weight",
                    "rank_reliability_exponent",
                    "gate",
                    "regressions",
                    "metrics",
                    "families",
                )
            }
        )
    return {
        "evaluated_unique_orders": len(candidates),
        "baseline": {
            key: value
            for key, value in baseline_metrics.items()
            if key != "rows"
        },
        "best": compact[0],
        "best_sealed_paired_bootstrap_95": {
            "lower": interval[0],
            "upper": interval[1],
        },
        "top_twenty": compact,
        "best_orders": best["orders"],
        "best_row_metrics": best["row_metrics"],
        "baseline_orders": embedding_orders,
        "baseline_row_metrics": baseline_metrics["rows"],
    }


def current_live_evaluation(
    client: httpx.Client,
    cases: list[QueryCase],
    library_url: str,
    corpus: dict[str, Any],
) -> dict[str, Any]:
    baseline_orders: dict[str, list[tuple[str, int]]] = {}
    rerank_orders: dict[str, list[tuple[str, int]]] = {}
    media_records: list[dict[str, Any]] = []
    candidate_subset = True
    rank_or_score_changed = 0
    for index, case in enumerate(cases, start=1):
        first = _search(client, library_url, case.query, rerank=False)
        second = _search(client, library_url, case.query, rerank=True)
        third = _search(client, library_url, case.query, rerank=False)
        for key in ("items", "media_outputs", "media_decisions"):
            if first.get(key) != third.get(key):
                raise RuntimeError(f"A/B/A baseline drifted for {case.id}: {key}")
        first_items = _stable_items(first, corpus)
        second_items = _stable_items(second, corpus)
        baseline_orders[case.id] = [
            (row["document"], row["ordinal"]) for row in first_items
        ]
        rerank_orders[case.id] = [
            (row["document"], row["ordinal"]) for row in second_items
        ]
        initial = {
            (row["document"], row["ordinal"]) for row in first_items
        }
        candidate_subset = candidate_subset and all(
            (row["document"], row["ordinal"]) in initial for row in second_items
        )
        rank_or_score_changed += int(
            baseline_orders[case.id] != rerank_orders[case.id]
            or any(
                abs(left["score"] - right["score"]) > 1e-6
                for left, right in zip(first_items, second_items, strict=True)
            )
        )
        labels = _media_labels(second)
        media_records.append(
            {
                "id": case.id,
                "outputs": labels,
                "required_hit": case.required_media.issubset(labels),
                "unexpected": sorted(set(labels) - case.allowed_media),
                "top_hit": case.top_media is None
                or (bool(labels) and labels[0] == case.top_media),
                "hard_negative_outputs": len(labels) if case.hard_negative else 0,
            }
        )
        if index % 10 == 0 or index == len(cases):
            print(f"live A/B/A {index}/{len(cases)}", flush=True)
    baseline_metrics = aggregate_metrics(cases, baseline_orders)
    rerank_metrics = aggregate_metrics(cases, rerank_orders)
    regressions = regression_count(cases, baseline_orders, rerank_orders)
    sealed_cases = [case for case in cases if case.sealed and case.qrels]
    sealed_interval = bootstrap_interval(
        [
            baseline_metrics["rows"][case.id]["ndcg_at_10"]
            for case in sealed_cases
        ],
        [
            rerank_metrics["rows"][case.id]["ndcg_at_10"]
            for case in sealed_cases
        ],
    )
    return {
        "candidate_subset": candidate_subset,
        "baseline_restored": True,
        "rank_or_score_changed_queries": rank_or_score_changed,
        "regressions": regressions,
        "sealed_paired_bootstrap_95": {
            "lower": sealed_interval[0],
            "upper": sealed_interval[1],
        },
        "baseline": {
            key: value
            for key, value in baseline_metrics.items()
            if key != "rows"
        },
        "rerank": {
            key: value
            for key, value in rerank_metrics.items()
            if key != "rows"
        },
        "media": {
            "required_recall": statistics.fmean(
                float(item["required_hit"])
                for item in media_records
                if next(case for case in cases if case.id == item["id"]).required_media
            ),
            "top_accuracy": statistics.fmean(
                float(item["top_hit"])
                for item in media_records
                if next(case for case in cases if case.id == item["id"]).top_media
            ),
            "unexpected_outputs": sum(len(item["unexpected"]) for item in media_records),
            "hard_negative_outputs": sum(
                item["hard_negative_outputs"] for item in media_records
            ),
            "rows": media_records,
        },
    }


def evaluate_live_finalists(
    client: httpx.Client,
    library_url: str,
    *,
    apply_best: bool,
    finalists: list[tuple[int, float, float, float]] | None = None,
) -> dict[str, Any]:
    library = _request_json(client, "GET", library_url)
    original = dict(library["retrieval_settings"])
    baseline = evaluate_media_acceptance(client, library_url, rerank=False)
    finalists = finalists or [
        (10, 0.0200, 0.0, 1.0),
        (10, 0.0225, 0.0, 1.0),
        (10, 0.0250, 0.0, 1.0),
        (10, 0.0275, 0.0, 1.0),
    ]
    rows: list[dict[str, Any]] = []
    try:
        for index, (limit, fusion, bonus, exponent) in enumerate(
            finalists, start=1
        ):
            settings = {
                **original,
                "rerank_candidate_limit": limit,
                "rerank_fusion_weight": fusion,
                "rerank_rank_bonus_weight": bonus,
                "rerank_rank_reliability_exponent": exponent,
            }
            _request_json(
                client,
                "PATCH",
                library_url,
                json={"retrieval_settings": settings},
            )
            recalibrated = recalibrate_all(client, library_url)
            rounds = [
                evaluate_media_acceptance(client, library_url, rerank=True)
                for _ in range(3)
            ]
            stable = len({media_signature(round_) for round_ in rounds}) == 1
            compact_rounds = [
                {
                    key: round_[key]
                    for key in (
                        "core_recall",
                        "top_accuracy",
                        "negative_outputs",
                        "macro_f1",
                        "minimum_margin",
                        "text_mrr",
                        "text_ndcg_at_10",
                        "p95_wall_ms",
                        "mean_reported_rerank_ms",
                    )
                }
                for round_ in rounds
            ]
            hard_pass = bool(
                stable
                and all(
                    round_["core_recall"] == 1.0
                    and round_["top_accuracy"] == 1.0
                    and round_["negative_outputs"] == 0
                    and round_["text_mrr"] + 1e-12 >= baseline["text_mrr"]
                    and round_["text_ndcg_at_10"] + 1e-12
                    >= baseline["text_ndcg_at_10"]
                    for round_ in rounds
                )
            )
            rows.append(
                {
                    "settings": {
                        key: settings[key]
                        for key in (
                            "rerank_candidate_limit",
                            "rerank_fusion_weight",
                            "rerank_rank_bonus_weight",
                            "rerank_rank_reliability_exponent",
                        )
                    },
                    "recalibrated_relations": recalibrated,
                    "cross_round_stable": stable,
                    "hard_pass": hard_pass,
                    "rounds": compact_rounds,
                }
            )
            print(
                f"finalist {index}/{len(finalists)} "
                f"recall={compact_rounds[0]['core_recall']:.3f} "
                f"top={compact_rounds[0]['top_accuracy']:.3f} "
                f"negative={compact_rounds[0]['negative_outputs']} "
                f"macro_f1={compact_rounds[0]['macro_f1']:.3f}",
                flush=True,
            )
        rows.sort(
            key=lambda item: (
                int(item["hard_pass"]),
                item["rounds"][0]["core_recall"],
                item["rounds"][0]["top_accuracy"],
                -item["rounds"][0]["negative_outputs"],
                item["rounds"][0]["macro_f1"],
                item["rounds"][0]["minimum_margin"],
                item["rounds"][0]["text_ndcg_at_10"],
                item["rounds"][0]["text_mrr"],
                -statistics.fmean(
                    round_["p95_wall_ms"] for round_ in item["rounds"]
                ),
                -float(item["settings"]["rerank_rank_bonus_weight"]),
                -int(item["settings"]["rerank_candidate_limit"]),
            ),
            reverse=True,
        )
        best = rows[0]
        if apply_best and best["hard_pass"]:
            _request_json(
                client,
                "PATCH",
                library_url,
                json={
                    "retrieval_settings": {**original, **best["settings"]}
                },
            )
            recalibrate_all(client, library_url)
        return {
            "baseline": {
                key: baseline[key]
                for key in (
                    "core_recall",
                    "top_accuracy",
                    "negative_outputs",
                    "macro_f1",
                    "minimum_margin",
                    "text_mrr",
                    "text_ndcg_at_10",
                )
            },
            "hard_constraints_passed": bool(best["hard_pass"]),
            "best": best,
            "finalists": rows,
        }
    finally:
        if not (apply_best and rows and rows[0]["hard_pass"]):
            _request_json(
                client,
                "PATCH",
                library_url,
                json={"retrieval_settings": original},
            )
            recalibrate_all(client, library_url)


def select_live_finalists(
    offline_grid: dict[str, Any] | None,
) -> list[tuple[int, float, float, float]]:
    """Select nine distinct offline leaders plus the current safe incumbent.

    The offline grid contains many parameter combinations that collapse to the
    same order.  Live validation is deliberately limited to distinct settings
    supported by the production dual-channel algorithm, while the incumbent is
    always retained so a text-only offline winner cannot silently trade away
    media quality.
    """

    incumbent = (
        int(DEFAULT_RETRIEVAL_CONFIG["rerank_candidate_limit"]),
        float(DEFAULT_RETRIEVAL_CONFIG["rerank_fusion_weight"]),
        float(DEFAULT_RETRIEVAL_CONFIG["rerank_rank_bonus_weight"]),
        float(DEFAULT_RETRIEVAL_CONFIG["rerank_rank_reliability_exponent"]),
    )
    selected: list[tuple[int, float, float, float]] = []
    seen: set[tuple[int, float, float, float]] = set()
    for row in (offline_grid or {}).get("top_twenty", []):
        if row.get("strategy") != "dual_channel":
            continue
        if float(row.get("gate", 0.0)) != 0.0:
            continue
        settings = (
            int(row["candidate_limit"]),
            float(row["fusion_weight"]),
            float(row["rank_bonus_weight"]),
            float(row["rank_reliability_exponent"]),
        )
        if settings in seen:
            continue
        seen.add(settings)
        selected.append(settings)
        if len(selected) == 9:
            break
    if incumbent not in seen:
        selected.append(incumbent)
    if len(selected) < 10:
        for settings in (
            (10, 0.0200, 0.0, 1.0),
            (10, 0.0250, 0.0, 1.0),
            (10, 0.0275, 0.0, 1.0),
            (10, 0.0300, 0.0, 1.0),
        ):
            if settings not in seen:
                seen.add(settings)
                selected.append(settings)
            if len(selected) == 10:
                break
    return selected[:10]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate text_media_v1 semantic Rerank quality on a cloned baseline."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--baseline-library", default="beileite_test2")
    parser.add_argument("--rerank-library", default="beileite_test")
    parser.add_argument("--api-key", default=os.environ.get("PERSONALITYRAG_API_KEY"))
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--rerank-endpoint", default="http://127.0.0.1:8002/v1/rerank")
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--provider-rounds", type=int, default=1, choices=(1, 3))
    parser.add_argument(
        "--phase",
        choices=("parity", "collect", "grid", "finalists", "live", "all"),
        default="all",
    )
    parser.add_argument("--apply-best", action="store_true")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api-key or PERSONALITYRAG_API_KEY is required")
    _payload, cases = load_cases(args.qrels)
    baseline_corpus = corpus_snapshot(args.data_root, args.baseline_library)
    rerank_corpus = corpus_snapshot(args.data_root, args.rerank_library)
    keys = set(rerank_corpus["chunk_by_key"])
    missing_qrels = sorted(
        {
            key
            for case in cases
            for key in case.qrels
            if key not in keys
        }
    )
    if missing_qrels:
        raise RuntimeError(f"qrels reference missing chunks: {missing_qrels}")
    root = args.base_url.rstrip("/")
    prefix = f"{root}/api/v1/knowledge-libraries/text_media_v1"
    baseline_url = f"{prefix}/{args.baseline_library}"
    rerank_url = f"{prefix}/{args.rerank_library}"
    headers = {"Authorization": f"Bearer {args.api_key}"}
    report: dict[str, Any] = {}
    if args.report.exists():
        existing = json.loads(args.report.read_text(encoding="utf-8"))
        if existing.get("format") == "personalityrag.text_media_v1.semantic_rerank_report":
            report = existing
    report.update(
        {
            "format": "personalityrag.text_media_v1.semantic_rerank_report",
            "version": 1,
            "updated_at": time.time(),
            "qrels": str(args.qrels),
            "query_count": len(cases),
            "topic_count": 25,
            "sealed_count": sum(case.sealed for case in cases),
            "hard_negative_count": sum(case.hard_negative for case in cases),
            "corpus_parity": verify_corpus_parity(
                baseline_corpus, rerank_corpus
            ),
        }
    )
    if not report["corpus_parity"]["passed"]:
        raise RuntimeError("corpus parity failed")
    with httpx.Client(headers=headers, timeout=180.0) as client:
        if args.phase in {"parity", "all"}:
            report["api_baseline_parity"] = verify_api_baseline_parity(
                client,
                cases,
                baseline_url,
                rerank_url,
                baseline_corpus,
                rerank_corpus,
            )
            if not report["api_baseline_parity"]["passed"]:
                raise RuntimeError("API baseline parity failed")
        if args.phase in {"collect", "all"}:
            with httpx.Client(timeout=180.0) as provider_client:
                snapshot = collect_snapshot(
                    client,
                    provider_client,
                    cases,
                    baseline_url,
                    baseline_corpus,
                    rerank_endpoint=args.rerank_endpoint,
                    rerank_model=args.rerank_model,
                    provider_rounds=args.provider_rounds,
                )
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            args.snapshot.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            report["snapshot"] = {
                key: value for key, value in snapshot.items() if key != "rows"
            }
        if args.phase in {"grid", "all"}:
            snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
            report["offline_grid"] = grid_evaluate(cases, snapshot)
        if args.phase in {"finalists", "all"}:
            finalist_settings = select_live_finalists(
                report.get("offline_grid")
            )
            report["live_finalists"] = evaluate_live_finalists(
                client,
                rerank_url,
                apply_best=args.apply_best,
                finalists=finalist_settings,
            )
            report["live_finalists"]["selection_source"] = {
                "offline_leaders": 9,
                "safe_incumbent": {
                    key: DEFAULT_RETRIEVAL_CONFIG[key]
                    for key in (
                        "rerank_candidate_limit",
                        "rerank_fusion_weight",
                        "rerank_rank_bonus_weight",
                        "rerank_rank_reliability_exponent",
                    )
                },
                "evaluated_count": len(finalist_settings),
            }
        if args.phase in {"live", "all"}:
            report["live"] = current_live_evaluation(
                client, cases, rerank_url, rerank_corpus
            )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "report": str(args.report),
        "snapshot": str(args.snapshot),
        "query_count": len(cases),
        "corpus_parity": report["corpus_parity"]["passed"],
        "api_baseline_parity": report.get("api_baseline_parity", {}).get("passed"),
        "offline_best": report.get("offline_grid", {}).get("best"),
        "live_finalist_best": report.get("live_finalists", {}).get("best"),
        "live": {
            "regressions": report.get("live", {}).get("regressions"),
            "candidate_subset": report.get("live", {}).get(
                "candidate_subset"
            ),
            "baseline_restored": report.get("live", {}).get(
                "baseline_restored"
            ),
            "sealed_paired_bootstrap_95": report.get("live", {}).get(
                "sealed_paired_bootstrap_95"
            ),
            "baseline": report.get("live", {}).get("baseline"),
            "rerank": report.get("live", {}).get("rerank"),
            "media": {
                key: value
                for key, value in report.get("live", {}).get("media", {}).items()
                if key != "rows"
            },
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
