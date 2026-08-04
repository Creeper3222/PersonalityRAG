from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from personalityrag.library_types.text_media_v1.retrieval import (
    fuse_embedding_rerank,
)


@dataclass(frozen=True)
class TextCase:
    id: str
    query: str
    title_contains: str | None
    primary_markers: tuple[str, ...] = ()
    secondary_markers: tuple[str, ...] = ()
    family: str = "general"

    def grade(self, item: dict[str, Any]) -> int:
        if self.title_contains is None:
            return 0
        title = str(item.get("title") or "")
        if self.title_contains not in title:
            return 0
        text = str(item.get("text") or "")
        if self.primary_markers and all(
            marker in text for marker in self.primary_markers
        ):
            return 3
        if self.secondary_markers and any(
            marker in text for marker in self.secondary_markers
        ):
            return 2
        return 1


CASES = (
    TextCase("friend_direct", "萌依是谁", "群友", ("萌依",), family="friend"),
    TextCase(
        "friend_paraphrase",
        "介绍一下群里的萌依",
        "群友",
        ("萌依",),
        family="friend",
    ),
    TextCase(
        "identity_direct",
        "澄月是谁",
        "澄月",
        ("基础设定",),
        ("角色背景",),
        "identity",
    ),
    TextCase(
        "identity_second_person",
        "澄月，你究竟是什么人",
        "澄月",
        ("基础设定",),
        ("角色背景",),
        "identity",
    ),
    TextCase(
        "appearance_direct",
        "澄月平时穿什么",
        "澄月",
        ("相貌衣着(常服)",),
        ("神态外貌",),
        "appearance",
    ),
    TextCase(
        "appearance_paraphrase",
        "说说你的日常服装和外貌",
        "澄月",
        ("相貌衣着(常服)",),
        ("神态外貌",),
        "appearance",
    ),
    TextCase(
        "shikai_direct",
        "始解的名字和解放语是什么",
        "斩魄刀的设定",
        ("解放语",),
        ("始解",),
        "shikai",
    ),
    TextCase(
        "shikai_paraphrase",
        "金黄澄月要念什么才能解放",
        "斩魄刀的设定",
        ("解放语",),
        ("始解",),
        "shikai",
    ),
    TextCase(
        "bankai_direct",
        "卍解最终技叫什么",
        "斩魄刀的设定",
        ("最终技",),
        ("卍解",),
        "bankai",
    ),
    TextCase(
        "bankai_paraphrase",
        "终焉澄月最后的招式是什么",
        "斩魄刀的设定",
        ("最终技",),
        ("卍解",),
        "bankai",
    ),
    TextCase(
        "quick_sheath_direct",
        "纳刀术对太刀有什么作用",
        "终盘太刀配装",
        ("纳刀术",),
        ("居合",),
        "longsword",
    ),
    TextCase(
        "quick_sheath_paraphrase",
        "居合流太刀为什么要出快收刀",
        "终盘太刀配装",
        ("纳刀术",),
        ("居合",),
        "longsword",
    ),
    TextCase(
        "stygian_direct",
        "狱狼龙如何吸引蝕龙虫",
        "狱狼龙",
        ("蝕龙虫",),
        ("龙光",),
        "ecology",
    ),
    TextCase(
        "stygian_paraphrase",
        "雷狼龙亚种怎样聚集蚀龙虫",
        "狱狼龙",
        ("蚀龙虫",),
        ("龙光",),
        "ecology",
    ),
    TextCase(
        "pdf_author_direct",
        "闪光的哈萨维小说作者是谁",
        "闪光的哈萨维",
        ("作者:富野由悠季",),
        family="pdf",
    ),
    TextCase(
        "pdf_author_paraphrase",
        "富野由悠季写的闪光哈萨维是谁创作的",
        "闪光的哈萨维",
        ("作者:富野由悠季",),
        family="pdf",
    ),
    TextCase("unknown_weather", "火星明天下午会不会下雨", None, family="none"),
    TextCase("unknown_phone", "萌依的手机号码是多少", None, family="none"),
)


def _request_json(
    client: httpx.Client, method: str, url: str, **kwargs: Any
) -> dict[str, Any]:
    response = client.request(method, url, **kwargs)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError(f"unexpected JSON response from {url}")
    return value


def _dcg(grades: list[int]) -> float:
    return sum(
        (2**grade - 1) / math.log2(index + 2)
        for index, grade in enumerate(grades)
    )


def _metrics(
    cases: tuple[TextCase, ...], rows: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    reciprocal: list[float] = []
    ndcg: list[float] = []
    recall1: list[float] = []
    recall3: list[float] = []
    family_ndcg: dict[str, list[float]] = {}
    separations: list[float] = []
    unknown_maxima: list[float] = []
    for case in cases:
        items = rows[case.id][:10]
        grades = [case.grade(item) for item in items]
        scores = [float(item.get("score") or 0.0) for item in items]
        if case.title_contains is None:
            unknown_maxima.append(max(scores, default=0.0))
            continue
        relevant = [index for index, grade in enumerate(grades) if grade > 0]
        reciprocal.append(1.0 / (relevant[0] + 1) if relevant else 0.0)
        ideal = sorted(grades, reverse=True)
        denominator = _dcg(ideal)
        value = _dcg(grades) / denominator if denominator else 0.0
        ndcg.append(value)
        family_ndcg.setdefault(case.family, []).append(value)
        recall1.append(float(bool(grades and grades[0] > 0)))
        recall3.append(float(any(grade > 0 for grade in grades[:3])))
        true_scores = [score for score, grade in zip(scores, grades) if grade > 0]
        false_scores = [score for score, grade in zip(scores, grades) if grade == 0]
        if true_scores:
            separations.append(
                max(true_scores) - max(false_scores, default=0.0)
            )
    return {
        "mrr": statistics.fmean(reciprocal) if reciprocal else 0.0,
        "ndcg_at_10": statistics.fmean(ndcg) if ndcg else 0.0,
        "recall_at_1": statistics.fmean(recall1) if recall1 else 0.0,
        "recall_at_3": statistics.fmean(recall3) if recall3 else 0.0,
        "minimum_family_ndcg": min(
            (statistics.fmean(values) for values in family_ndcg.values()),
            default=0.0,
        ),
        "minimum_separation": min(separations, default=0.0),
        "maximum_unknown_score": max(unknown_maxima, default=0.0),
    }


def _rerank_snapshot(
    client: httpx.Client, library_url: str, case: TextCase
) -> list[dict[str, Any]]:
    result = _request_json(
        client,
        "POST",
        f"{library_url}/search",
        json={
            "query": case.query,
            "retrieval_mode": "text_only",
            "top_k": 50,
            "media_output_confidence_threshold": 0.60,
            "media_relevance_pivot": 0.35,
            "max_media_outputs": 0,
            "rerank": True,
        },
    )
    rerank = dict(result.get("rerank") or {})
    if not rerank.get("applied") or rerank.get("fallback"):
        raise RuntimeError(f"Rerank was not applied for {case.id}: {rerank}")
    rows = [dict(item) for item in result.get("items") or []]
    if len(rows) < 50 or any(item.get("rerank_raw_score") is None for item in rows):
        raise RuntimeError(f"incomplete 50-candidate snapshot for {case.id}")
    initial = sorted(rows, key=lambda item: int(item["initial_rank"]))
    # The API response is in the reranked stage, so ``score`` is fused even
    # after restoring initial order.  A snapshot's baseline copy must expose
    # the absolute Embedding relevance instead of accidentally comparing the
    # Rerank score against itself.
    for item in initial:
        item["score"] = float(item["embedding_relevance"])
    return initial


def _simulate(
    initial: list[dict[str, Any]],
    *,
    candidate_limit: int,
    fusion_weight: float,
    rank_bonus_weight: float,
    exponent: float,
) -> list[dict[str, Any]]:
    head = [dict(item) for item in initial[:candidate_limit]]
    provider_order = sorted(
        range(len(head)),
        key=lambda index: (
            -float(head[index]["rerank_raw_score"]),
            int(head[index]["initial_rank"]),
        ),
    )
    provider_rank = {
        index: rank for rank, index in enumerate(provider_order, start=1)
    }
    for index, item in enumerate(head):
        detail = fuse_embedding_rerank(
            float(item["embedding_relevance"]),
            float(item["rerank_raw_score"]),
            rank=provider_rank[index],
            candidate_count=len(head),
            fusion_weight=fusion_weight,
            rank_bonus_weight=rank_bonus_weight,
            rank_reliability_exponent=exponent,
        )
        item["score"] = float(detail["fused_relevance"])
        item["rerank_rank"] = provider_rank[index]
    head.sort(
        key=lambda item: (
            -float(item["score"]),
            int(item["rerank_rank"]),
            int(item["initial_rank"]),
            int(item["chunk_id"]),
        )
    )
    tail = [dict(item) for item in initial[candidate_limit:]]
    for item in tail:
        item["score"] = float(item["embedding_relevance"])
    return head + tail


def _first_primary_rank(case: TextCase, items: list[dict[str, Any]]) -> int:
    for rank, item in enumerate(items, start=1):
        if case.grade(item) == 3:
            return rank
    return 10_000


def _hard_constraints(
    cases: tuple[TextCase, ...],
    baseline: dict[str, list[dict[str, Any]]],
    reranked: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    regressions = 0
    for case in cases:
        if case.title_contains is None:
            continue
        before = _first_primary_rank(case, baseline[case.id])
        after = _first_primary_rank(case, reranked[case.id])
        if before <= 3 and after > 3:
            regressions += 1
        if before == 1 and after > 3:
            regressions += 1
    friend = next(case for case in cases if case.id == "friend_direct")
    friend_rows = reranked[friend.id][:10]
    friend_target = max(
        (
            float(item["score"])
            for item in friend_rows
            if friend.grade(item) > 0
        ),
        default=0.0,
    )
    friend_false = max(
        (
            float(item["score"])
            for item in friend_rows
            if friend.grade(item) == 0
        ),
        default=0.0,
    )
    baseline_metrics = _metrics(cases, baseline)
    reranked_metrics = _metrics(cases, reranked)
    quality_non_degradation = all(
        float(reranked_metrics[key]) + 1e-12
        >= float(baseline_metrics[key])
        for key in ("mrr", "ndcg_at_10", "recall_at_1", "recall_at_3")
    )
    return {
        "regressions": regressions,
        "friend_target_score": friend_target,
        "friend_false_max": friend_false,
        "quality_non_degradation": quality_non_degradation,
        "baseline_metrics": baseline_metrics,
        "passed": (
            regressions == 0
            and friend_target > 0.80
            and friend_false <= 0.20
            and quality_non_degradation
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {args.api_key}"}
    root = f"{args.base_url.rstrip('/')}/api/v1/knowledge-libraries/text_media_v1"
    library_url = f"{root}/{args.library}"
    with httpx.Client(headers=headers, timeout=args.timeout) as client:
        detail = _request_json(client, "GET", library_url)
        original = dict(detail["retrieval_settings"])
        capture = {
            **original,
            "rerank_candidate_limit": 50,
            "rerank_fusion_weight": 0.35,
            "rerank_rank_bonus_weight": 0.0,
            "rerank_rank_reliability_exponent": 3.0,
        }
        started = time.perf_counter()
        snapshots: dict[float, dict[str, list[dict[str, Any]]]] = {}
        try:
            for lexical_boost in args.lexical_boosts:
                settings = {**capture, "text_lexical_boost": lexical_boost}
                _request_json(
                    client,
                    "PATCH",
                    library_url,
                    json={"retrieval_settings": settings},
                )
                snapshots[lexical_boost] = {
                    case.id: _rerank_snapshot(client, library_url, case)
                    for case in CASES
                }

            results: list[dict[str, Any]] = []
            for lexical_boost, candidate_limit, weight, bonus, exponent in itertools.product(
                args.lexical_boosts,
                args.candidate_limits,
                args.fusion_weights,
                args.rank_bonuses,
                args.reliability_exponents,
            ):
                baseline = snapshots[lexical_boost]
                reranked = {
                    case.id: _simulate(
                        baseline[case.id],
                        candidate_limit=candidate_limit,
                        fusion_weight=weight,
                        rank_bonus_weight=bonus,
                        exponent=exponent,
                    )
                    for case in CASES
                }
                metrics = _metrics(CASES, reranked)
                hard = _hard_constraints(CASES, baseline, reranked)
                results.append(
                    {
                        "settings": {
                            "text_lexical_boost": lexical_boost,
                            "rerank_candidate_limit": candidate_limit,
                            "rerank_fusion_weight": weight,
                            "rerank_rank_bonus_weight": bonus,
                            "rerank_rank_reliability_exponent": exponent,
                        },
                        "metrics": metrics,
                        "hard_constraints": hard,
                    }
                )
            results.sort(
                key=lambda row: (
                    not bool(row["hard_constraints"]["passed"]),
                    -float(row["metrics"]["minimum_family_ndcg"]),
                    -float(row["metrics"]["ndcg_at_10"]),
                    -float(row["metrics"]["mrr"]),
                    -float(row["metrics"]["minimum_separation"]),
                    float(row["metrics"]["maximum_unknown_score"]),
                    float(row["settings"]["rerank_fusion_weight"]),
                    int(row["settings"]["rerank_candidate_limit"]),
                )
            )
            quality_leader = results[0]
            near_ties = [
                row
                for row in results
                if bool(row["hard_constraints"]["passed"])
                and float(row["metrics"]["minimum_family_ndcg"])
                >= float(quality_leader["metrics"]["minimum_family_ndcg"])
                - 0.002
                and float(row["metrics"]["ndcg_at_10"])
                >= float(quality_leader["metrics"]["ndcg_at_10"]) - 0.002
                and float(row["metrics"]["mrr"])
                >= float(quality_leader["metrics"]["mrr"]) - 0.002
            ]
            winner = min(
                near_ties or [quality_leader],
                key=lambda row: (
                    float(row["settings"]["rerank_fusion_weight"]),
                    int(row["settings"]["rerank_candidate_limit"]),
                    float(row["settings"]["rerank_rank_bonus_weight"]),
                    float(row["settings"]["text_lexical_boost"]),
                ),
            )
            best_by_candidate_limit = {
                str(limit): next(
                    row
                    for row in results
                    if int(row["settings"]["rerank_candidate_limit"]) == limit
                )
                for limit in args.candidate_limits
            }
            baseline_metrics = _metrics(
                CASES, snapshots[float(winner["settings"]["text_lexical_boost"])]
            )
            if args.apply_winner:
                winning_settings = {**original, **winner["settings"]}
                for library_id in args.apply_libraries:
                    target = f"{root}/{library_id}"
                    target_detail = _request_json(client, "GET", target)
                    _request_json(
                        client,
                        "PATCH",
                        target,
                        json={
                            "retrieval_settings": {
                                **dict(target_detail["retrieval_settings"]),
                                **winner["settings"],
                            }
                        },
                    )
                original = winning_settings
            return {
                "algorithm": "absolute_dense_lexical_plus_log_odds_rerank_v1",
                "library": args.library,
                "case_count": len(CASES),
                "combination_count": len(results),
                "elapsed_seconds": time.perf_counter() - started,
                "baseline_metrics_at_winner_lexical_boost": baseline_metrics,
                "winner": winner,
                "quality_leader": quality_leader,
                "best_by_candidate_limit": best_by_candidate_limit,
                "top_twenty": results[:20],
            }
        finally:
            if not args.apply_winner:
                _request_json(
                    client,
                    "PATCH",
                    library_url,
                    json={"retrieval_settings": original},
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--library", default="beileite_test")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--apply-winner", action="store_true")
    parser.add_argument(
        "--apply-libraries", nargs="+", default=["beileite_test", "beileite_test2"]
    )
    parser.add_argument(
        "--lexical-boosts",
        type=float,
        nargs="+",
        default=[0.25, 0.40, 0.50, 0.60, 0.75],
    )
    parser.add_argument(
        "--fusion-weights",
        type=float,
        nargs="+",
        default=[0.20, 0.30, 0.35, 0.40, 0.50],
    )
    parser.add_argument(
        "--rank-bonuses", type=float, nargs="+", default=[0.0, 0.05, 0.10]
    )
    parser.add_argument(
        "--reliability-exponents",
        type=float,
        nargs="+",
        default=[1.5, 2.0, 3.0],
    )
    parser.add_argument(
        "--candidate-limits", type=int, nargs="+", default=[10, 20, 50]
    )
    args = parser.parse_args()
    report = run(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
