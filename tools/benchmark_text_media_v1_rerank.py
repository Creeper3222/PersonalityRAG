from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from benchmark_text_media_v1 import CASES, classify, macro_f1, request_json


TEXT_EXPECTATIONS = {
    **{case.query: "澄月" for case in CASES[:13]},
    **{case.query: "斩魄刀" for case in CASES[17:20]},
    **{case.query: "怪物猎人" for case in CASES[20:24]},
}


def _dcg(relevances: list[int]) -> float:
    return sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def _text_metrics(rows: list[dict[str, Any]]) -> tuple[float, float]:
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []
    for row in rows:
        expected = TEXT_EXPECTATIONS.get(str(row["query"]))
        if not expected:
            continue
        relevances = [
            int(expected in str(item.get("title") or ""))
            for item in row["result"].get("items", [])[:10]
        ]
        try:
            first = relevances.index(1) + 1
        except ValueError:
            first = 0
        reciprocal_ranks.append(1.0 / first if first else 0.0)
        ideal = sorted(relevances, reverse=True)
        denominator = _dcg(ideal)
        ndcg_values.append(_dcg(relevances) / denominator if denominator else 0.0)
    return (
        statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        statistics.fmean(ndcg_values) if ndcg_values else 0.0,
    )


def evaluate(
    client: httpx.Client,
    library_url: str,
    *,
    rerank: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    records: list[tuple[set[str], frozenset[str]]] = []
    required_hits = required_total = negative_outputs = top_hits = top_total = 0
    margins: list[float] = []
    wall_times: list[float] = []
    provider_times: list[float] = []
    for case in CASES:
        started = time.perf_counter()
        result = request_json(
            client,
            "POST",
            f"{library_url}/search",
            {
                "query": case.query,
                "top_k": 10,
                "media_output_confidence_threshold": 0.60,
                "media_score_threshold": 0.35,
                "max_media_outputs": 5,
                "rerank": rerank,
            },
        )
        wall_times.append((time.perf_counter() - started) * 1000.0)
        provider_times.append(float(result.get("rerank", {}).get("elapsed_ms") or 0))
        ordered = [
            classify(str(item.get("original_name") or ""))
            for item in result.get("media_outputs", [])
        ]
        predicted = set(ordered)
        records.append((predicted, case.positives))
        required_hits += len(predicted & case.required)
        required_total += len(case.required)
        if case.top_label is not None:
            top_total += 1
            top_hits += int(bool(ordered) and ordered[0] == case.top_label)
        if case.hard_negative:
            negative_outputs += len(predicted)
        confidences = {
            classify(str(item.get("original_name") or "")): float(
                item.get("output_confidence") or 0.0
            )
            for item in result.get("media_decisions", [])
        }
        true_scores = [confidences.get(label, 0.0) for label in case.required]
        false_scores = [
            score
            for label, score in confidences.items()
            if label not in case.positives
        ]
        if true_scores:
            margins.append(min(true_scores) - max(false_scores, default=0.0))
        rows.append({"query": case.query, "result": result})
    mrr, ndcg = _text_metrics(rows)
    ordered_wall = sorted(wall_times)
    p95_index = max(0, math.ceil(0.95 * len(ordered_wall)) - 1)
    return {
        "core_recall": required_hits / max(1, required_total),
        "top_accuracy": top_hits / max(1, top_total),
        "negative_outputs": negative_outputs,
        "macro_f1": macro_f1(records),
        "minimum_margin": min(margins, default=-math.inf),
        "text_mrr": mrr,
        "text_ndcg_at_10": ndcg,
        "p95_wall_ms": ordered_wall[p95_index],
        "mean_reported_rerank_ms": statistics.fmean(provider_times),
        "rows": [
            {
                "query": row["query"],
                "text": [
                    {
                        "chunk_id": item.get("chunk_id"),
                        "title": item.get("title"),
                        "rank": item.get("rank"),
                        "initial_rank": item.get("initial_rank"),
                        "score": item.get("score"),
                        "initial_score": item.get("initial_score"),
                        "rerank_raw_score": item.get("rerank_raw_score"),
                        "rerank_rank": item.get("rerank_rank"),
                    }
                    for item in row["result"].get("items", [])
                ],
                "media": [
                    {
                        "label": classify(
                            str(item.get("original_name") or "")
                        ),
                        "output": item.get("output"),
                        "confidence": item.get("output_confidence"),
                        "baseline_confidence": item.get(
                            "rerank_baseline_output_confidence"
                        ),
                        "confidence_delta": item.get(
                            "rerank_output_confidence_delta"
                        ),
                    }
                    for item in row["result"].get("media_decisions", [])
                ],
            }
            for row in rows
        ],
    }


def signature(result: dict[str, Any]) -> str:
    return json.dumps(
        {
            key: result[key]
            for key in (
                "core_recall",
                "top_accuracy",
                "negative_outputs",
                "macro_f1",
                "minimum_margin",
                "text_mrr",
                "text_ndcg_at_10",
                "rows",
            )
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def validate_aba(client: httpx.Client, library_url: str) -> dict[str, Any]:
    """Verify baseline -> rerank -> baseline reversibility and all four scopes."""

    changed_queries = 0
    changed_items = 0
    rank_changed_queries = 0
    rank_changed_items = 0
    score_changed_items = 0
    bound_evidence_changed = 0
    direct_semantic_changed = 0
    for case in CASES:
        payload = {
            "query": case.query,
            "top_k": 10,
            "media_output_confidence_threshold": 0.60,
            "media_score_threshold": 0.35,
            "max_media_outputs": 5,
        }
        first = request_json(
            client, "POST", f"{library_url}/search", {**payload, "rerank": False}
        )
        reranked = request_json(
            client, "POST", f"{library_url}/search", {**payload, "rerank": True}
        )
        third = request_json(
            client, "POST", f"{library_url}/search", {**payload, "rerank": False}
        )
        for result in (first, third):
            meta = result.get("rerank") or {}
            if meta.get("applied") or meta.get("provider_candidates") or meta.get("scopes"):
                raise RuntimeError("A phase unexpectedly invoked Rerank")
        for key in ("items", "media_outputs", "media_decisions"):
            if first.get(key) != third.get(key):
                raise RuntimeError(f"A/B/A baseline drifted: {key}")
        rerank_meta = reranked.get("rerank") or {}
        if not rerank_meta.get("applied") or rerank_meta.get("failed"):
            raise RuntimeError("B phase did not apply Rerank successfully")
        baseline_ids = {
            int(item["chunk_id"])
            for item in reranked.get("baseline_items", [])
        }
        reranked_ids = {
            int(item["chunk_id"])
            for item in reranked.get("items", [])
        }
        if not reranked_ids.issubset(baseline_ids):
            raise RuntimeError("Rerank introduced a chunk outside the initial candidates")
        first_items = first.get("items", [])
        reranked_items = reranked.get("items", [])
        rank_changed = [
            item.get("chunk_id") for item in first_items
        ] != [item.get("chunk_id") for item in reranked_items]
        score_changed = [
            item.get("score") for item in first_items
        ] != [item.get("score") for item in reranked_items]
        changed_queries += int(rank_changed or score_changed)
        rank_changed_queries += int(rank_changed)
        rank_changed_items += sum(
            item.get("initial_rank") != item.get("rank")
            for item in reranked_items
        )
        score_changed_items += sum(
            abs(
                float(item.get("score") or 0.0)
                - float(item.get("initial_score") or 0.0)
            ) > 1e-6
            for item in reranked_items
        )
        changed_items += sum(
            item.get("initial_rank") != item.get("rank")
            or abs(float(item.get("score") or 0.0) - float(item.get("initial_score") or 0.0)) > 1e-6
            for item in reranked_items
        )
        for decision in reranked.get("media_decisions", []):
            for evidence in decision.get("evidence", []):
                if evidence.get("rerank_raw_score") is not None and abs(
                    float(evidence.get("fused_relevance") or 0.0)
                    - float(evidence.get("dense_score") or 0.0)
                ) > 1e-6:
                    bound_evidence_changed += 1
            for relation in decision.get("direct_relations", []):
                baseline_semantic = relation.get("embedding_calibrated_semantic_score")
                if baseline_semantic is not None and abs(
                    float(relation.get("calibrated_semantic_score") or 0.0)
                    - float(baseline_semantic)
                ) > 1e-6:
                    direct_semantic_changed += 1
    calibrations = request_json(
        client, "GET", f"{library_url}/media-calibrations"
    ).get("items", [])
    calibration_changed = sum(
        bool(item.get("rerank_calibrated_chunk_count"))
        and (
            int(item.get("rerank_changed_chunk_count") or 0) > 0
            or float(item.get("maximum_rerank_strength_delta") or 0.0)
            > 1e-12
        )
        for item in calibrations
    )
    if not changed_queries or not changed_items:
        raise RuntimeError("Rerank did not change any text score or rank")
    if not rank_changed_queries or not rank_changed_items:
        raise RuntimeError("Rerank did not change any text rank")
    if not score_changed_items:
        raise RuntimeError("Rerank did not change any text score")
    if not bound_evidence_changed:
        raise RuntimeError("Rerank did not affect bound media evidence")
    if not direct_semantic_changed:
        raise RuntimeError("Rerank did not affect direct media semantics")
    if not calibration_changed:
        raise RuntimeError("Rerank did not affect persisted media calibration")
    return {
        "queries": len(CASES),
        "changed_queries": changed_queries,
        "changed_items": changed_items,
        "rank_changed_queries": rank_changed_queries,
        "rank_changed_items": rank_changed_items,
        "score_changed_items": score_changed_items,
        "candidate_subset": True,
        "baseline_restored": True,
        "bound_evidence_changed": bound_evidence_changed,
        "direct_semantic_changed": direct_semantic_changed,
        "calibration_relations_changed": calibration_changed,
    }


def validate_contract(
    client: httpx.Client, library_url: str
) -> dict[str, Any]:
    top_k_rows: dict[int, dict[str, Any]] = {}
    for top_k in (5, 10, 20, 50):
        result = request_json(
            client,
            "POST",
            f"{library_url}/search",
            {
                "query": "描述一下你的立绘",
                "top_k": top_k,
                "media_output_confidence_threshold": 0.60,
                "media_score_threshold": 0.35,
                "max_media_outputs": 5,
                "rerank": True,
            },
        )
        top_k_rows[top_k] = {
            "outputs": result.get("media_outputs", []),
            "decisions": result.get("media_decisions", []),
        }
    if any(top_k_rows[value] != top_k_rows[5] for value in (10, 20, 50)):
        raise RuntimeError("public top_k changed reranked media output or diagnostics")

    first = request_json(
        client,
        "POST",
        f"{library_url}/search",
        {"query": "描述一下你的立绘", "top_k": 10, "rerank": False},
    )
    second = request_json(
        client,
        "POST",
        f"{library_url}/search",
        {"query": "描述一下你的立绘", "top_k": 10, "rerank": False},
    )
    for result in (first, second):
        rerank_meta = result.get("rerank") or {}
        if (
            rerank_meta.get("applied")
            or rerank_meta.get("provider_candidates")
            or rerank_meta.get("scopes")
        ):
            raise RuntimeError("rerank=false invoked or applied rerank")
    for key in ("items", "media_outputs", "media_decisions"):
        if first.get(key) != second.get(key):
            raise RuntimeError(f"rerank=false baseline is unstable: {key}")
    return {
        "top_k_stable": True,
        "rerank_false_stable": True,
        "rerank_false_provider_candidates": 0,
        "aba": validate_aba(client, library_url),
    }


def recalibrate_all(client: httpx.Client, library_url: str) -> int:
    """Refresh persisted dual-path strengths for the active shared settings."""

    rows = request_json(
        client, "GET", f"{library_url}/media-calibrations"
    ).get("items", [])
    api_root = library_url.split("/api/v1/", 1)[0]
    completed = 0
    for row in rows:
        queued = request_json(
            client,
            "PUT",
            (
                f"{library_url}/document-media-relations/"
                f"{row['document_id']}/{row['asset_id']}/semantic-calibration"
            ),
            {
                "enabled": True,
                "media_description": str(row.get("media_description") or ""),
            },
        )
        deadline = time.monotonic() + 180.0
        while True:
            job = request_json(
                client,
                "GET",
                f"{api_root}/api/v1/jobs/{queued['job_id']}",
            )
            if job.get("status") in {
                "completed",
                "failed",
                "cancelled",
                "stopped",
            }:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("media recalibration job timed out")
            time.sleep(0.05)
        if job.get("status") != "completed":
            raise RuntimeError(
                f"media recalibration failed: {job.get('status')} "
                f"{job.get('error') or ''}"
            )
        completed += 1
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Grid-evaluate text_media_v1 full-chain rerank fusion."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--library", default="beileite_test")
    parser.add_argument("--api-key", default=os.environ.get("PERSONALITYRAG_API_KEY"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/text_media_v1_rerank_benchmark.json"),
    )
    parser.add_argument("--apply-best", action="store_true")
    parser.add_argument(
        "--grid-profile",
        choices=("center", "fine"),
        default="center",
        help="center tests the planned ranges; fine resolves the safe low-weight edge",
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api-key or PERSONALITYRAG_API_KEY is required")
    root = args.base_url.rstrip("/")
    library_url = (
        f"{root}/api/v1/knowledge-libraries/text_media_v1/{args.library}"
    )
    headers = {"Authorization": f"Bearer {args.api_key}"}
    with httpx.Client(headers=headers, timeout=180.0) as client:
        library = request_json(client, "GET", library_url)
        original = dict(library["retrieval_settings"])
        baseline = evaluate(client, library_url, rerank=False)
        candidates: list[dict[str, Any]] = []
        try:
            if args.grid_profile == "center":
                values = list(
                    itertools.product(
                        (10, 25, 50),
                        (0.005, 0.01, 0.025, 0.05),
                        (0.10, 0.20, 0.30),
                        (1.5, 2.0, 2.5),
                    )
                )
            else:
                values = list(
                    itertools.product(
                        (10, 25),
                        (0.006, 0.007, 0.008, 0.009, 0.01, 0.0125),
                        (0.10, 0.30),
                        (2.0, 3.0),
                    )
                )
            total = len(values)
            for index, (limit, fusion, bonus, exponent) in enumerate(
                values, start=1
            ):
                settings = {
                    **original,
                    "rerank_candidate_limit": limit,
                    "rerank_fusion_weight": fusion,
                    "rerank_rank_bonus_weight": bonus,
                    "rerank_rank_reliability_exponent": exponent,
                }
                request_json(
                    client,
                    "PATCH",
                    library_url,
                    {"retrieval_settings": settings},
                )
                recalibrate_all(client, library_url)
                result = evaluate(client, library_url, rerank=True)
                candidates.append({"settings": settings, "evaluation": result})
                print(
                    f"[{index}/{total}] recall={result['core_recall']:.3f} "
                    f"top={result['top_accuracy']:.3f} "
                    f"negative={result['negative_outputs']} "
                    f"mrr={result['text_mrr']:.3f} "
                    f"ndcg={result['text_ndcg_at_10']:.3f}",
                    flush=True,
                )
            candidates.sort(
                key=lambda item: (
                    int(item["evaluation"]["core_recall"] == 1.0),
                    int(item["evaluation"]["negative_outputs"] == 0),
                    int(item["evaluation"]["top_accuracy"] == 1.0),
                    int(
                        item["evaluation"]["text_mrr"] + 1e-12
                        >= baseline["text_mrr"]
                    ),
                    int(
                        item["evaluation"]["text_ndcg_at_10"] + 1e-12
                        >= baseline["text_ndcg_at_10"]
                    ),
                    item["evaluation"]["macro_f1"],
                    item["evaluation"]["minimum_margin"],
                    item["evaluation"]["text_mrr"],
                    item["evaluation"]["text_ndcg_at_10"],
                    -float(item["settings"]["rerank_fusion_weight"]),
                    -int(item["settings"]["rerank_candidate_limit"]),
                ),
                reverse=True,
            )
            verified: list[dict[str, Any]] = []
            for candidate in candidates[:10]:
                request_json(
                    client,
                    "PATCH",
                    library_url,
                    {"retrieval_settings": candidate["settings"]},
                )
                recalibrate_all(client, library_url)
                rounds = [evaluate(client, library_url, rerank=True) for _ in range(3)]
                candidate["live_rounds"] = rounds
                candidate["cross_round_stable"] = len(
                    {signature(round_) for round_ in rounds}
                ) == 1
                verified.append(candidate)
            passing = [
                item
                for item in verified
                if item["cross_round_stable"]
                and all(
                    round_["core_recall"] == 1.0
                    and round_["negative_outputs"] == 0
                    and round_["top_accuracy"] == 1.0
                    and round_["text_mrr"] + 1e-12 >= baseline["text_mrr"]
                    and round_["text_ndcg_at_10"] + 1e-12
                    >= baseline["text_ndcg_at_10"]
                    for round_ in item["live_rounds"]
                )
            ]
            best = passing[0] if passing else verified[0]
            request_json(
                client,
                "PATCH",
                library_url,
                {"retrieval_settings": best["settings"]},
            )
            recalibrate_all(client, library_url)
            contract = validate_contract(client, library_url)
            report = {
                "format": "personalityrag.text_media_v1.rerank_benchmark",
                "version": 1,
                "library": args.library,
                "grid_profile": args.grid_profile,
                "baseline": baseline,
                "best": best,
                "hard_constraints_passed": bool(passing),
                "verified_top_ten": verified,
                "contract": contract,
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "report": str(args.report),
                        "hard_constraints_passed": bool(passing),
                        "baseline": {
                            key: baseline[key]
                            for key in (
                                "text_mrr",
                                "text_ndcg_at_10",
                                "core_recall",
                                "top_accuracy",
                                "negative_outputs",
                            )
                        },
                        "best_settings": {
                            key: best["settings"][key]
                            for key in (
                                "rerank_candidate_limit",
                                "rerank_fusion_weight",
                                "rerank_rank_bonus_weight",
                                "rerank_rank_reliability_exponent",
                            )
                        },
                        "best": {
                            key: best["live_rounds"][0][key]
                            for key in (
                                "text_mrr",
                                "text_ndcg_at_10",
                                "core_recall",
                                "top_accuracy",
                                "negative_outputs",
                                "macro_f1",
                                "minimum_margin",
                                "p95_wall_ms",
                            )
                        },
                        "contract": contract,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        finally:
            if not args.apply_best:
                request_json(
                    client,
                    "PATCH",
                    library_url,
                    {"retrieval_settings": original},
                )
                recalibrate_all(client, library_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
