from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from personalityrag.library_types.text_media_v1.retrieval import (
    DEFAULT_RETRIEVAL_CONFIG,
    aggregate_grounding,
    apply_evidence_threshold,
    calibrated_semantic,
    clamp01,
    noisy_or,
)


OUTPUT_THRESHOLD = 0.60
EVIDENCE_THRESHOLD = 0.35


@dataclass(frozen=True)
class QueryCase:
    query: str
    positives: frozenset[str]
    required: frozenset[str]
    top_label: str | None = None
    hard_negative: bool = False


CASES = (
    QueryCase("描述一下你的立绘", frozenset({"full", "avatar"}), frozenset({"full"}), "full"),
    QueryCase("给我看看你的全身设定图", frozenset({"full"}), frozenset({"full"}), "full"),
    QueryCase("你的头像是什么样子", frozenset({"avatar"}), frozenset({"avatar"}), "avatar"),
    QueryCase("描述一下你的脸部特征", frozenset({"avatar"}), frozenset({"avatar"}), "avatar"),
    QueryCase("展示你的战斗场景", frozenset({"battle"}), frozenset({"battle"}), "battle"),
    QueryCase("描述一下你的战斗立绘", frozenset({"full", "battle"}), frozenset({"full"}), "full"),
    QueryCase("你战斗时是什么样子", frozenset({"full", "battle"}), frozenset({"battle"})),
    QueryCase("描述一下你的外貌", frozenset({"avatar", "full"}), frozenset({"avatar"}), "avatar"),
    QueryCase("展示你的外观", frozenset({"avatar", "full"}), frozenset({"full"})),
    QueryCase("看看你手持暗红雷电太刀的战斗场景", frozenset({"battle"}), frozenset({"battle"}), "battle"),
    QueryCase("看看你身穿狱狼龙铠甲的全身形象", frozenset({"full"}), frozenset({"full"}), "full"),
    QueryCase("展示你的头像、立绘和战斗场景", frozenset({"avatar", "full", "battle"}), frozenset({"avatar", "full", "battle"})),
    QueryCase("展示你手里的狱牙刀", frozenset({"full", "battle"}), frozenset({"full"})),
    QueryCase("你喜欢战斗吗", frozenset(), frozenset(), hard_negative=True),
    QueryCase("战斗策略是什么", frozenset(), frozenset(), hard_negative=True),
    QueryCase("立绘是谁画的", frozenset(), frozenset(), hard_negative=True),
    QueryCase("图片上传失败怎么办", frozenset(), frozenset(), hard_negative=True),
    QueryCase("你的斩魄刀始解能力是什么", frozenset(), frozenset(), hard_negative=True),
    QueryCase("看看你的卍解形态", frozenset(), frozenset(), hard_negative=True),
    QueryCase("描述一下金黄澄月斩魄刀的外观", frozenset(), frozenset(), hard_negative=True),
    QueryCase("黑龙太刀怎么配装", frozenset(), frozenset(), hard_negative=True),
    QueryCase("纳刀术对居合有什么影响", frozenset(), frozenset(), hard_negative=True),
    QueryCase("属性太刀配装", frozenset(), frozenset(), hard_negative=True),
    QueryCase("给我看看黑龙太刀配装图", frozenset(), frozenset(), hard_negative=True),
)


def classify(name: str) -> str:
    lowered = name.casefold()
    if "头像" in lowered or "avatar" in lowered or "portrait" in lowered:
        return "avatar"
    if "战斗" in lowered or "battle" in lowered or "scene" in lowered:
        return "battle"
    if "全身" in lowered or "full" in lowered or "立绘" in lowered:
        return "full"
    return f"other:{lowered}"


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = client.request(method, url, json=payload)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError(f"unexpected response from {url}")
    return result


def macro_f1(records: list[tuple[set[str], frozenset[str]]]) -> float:
    values = []
    for label in ("full", "avatar", "battle"):
        tp = fp = fn = 0
        for predicted, expected in records:
            if label in predicted and label in expected:
                tp += 1
            elif label in predicted:
                fp += 1
            elif label in expected:
                fn += 1
        denominator = 2 * tp + fp + fn
        values.append((2 * tp / denominator) if denominator else 1.0)
    return sum(values) / len(values)


def capture_snapshot(
    client: httpx.Client,
    library_url: str,
    original_settings: dict[str, Any],
) -> dict[str, Any]:
    capture_settings = {
        **DEFAULT_RETRIEVAL_CONFIG,
        **original_settings,
        "media_candidate_limit": 30,
        "media_threshold_evidence_limit": 30,
    }
    request_json(
        client,
        "PATCH",
        library_url,
        {"retrieval_settings": capture_settings},
    )
    rows = []
    for case in CASES:
        result = request_json(
            client,
            "POST",
            f"{library_url}/search",
            {
                "query": case.query,
                "top_k": 50,
                "media_output_confidence_threshold": OUTPUT_THRESHOLD,
                "media_score_threshold": EVIDENCE_THRESHOLD,
                "max_media_outputs": 5,
            },
        )
        decisions = []
        for decision in result.get("media_decisions", []):
            decisions.append(
                {
                    "label": classify(str(decision.get("original_name") or "")),
                    "asset_id": str(decision.get("asset_id") or ""),
                    "visual_intent": bool(decision.get("visual_intent")),
                    "implicit_media_lookup": bool(
                        decision.get("implicit_media_lookup")
                    ),
                    "subject_anchor_required": bool(
                        decision.get("subject_anchor_required")
                    ),
                    "subject_anchor_met": bool(
                        decision.get("subject_anchor_met", True)
                    ),
                    "output_confidence": float(
                        decision.get("output_confidence") or 0.0
                    ),
                    "output": bool(decision.get("output")),
                    "reason": str(decision.get("reason") or ""),
                    "media_evidence_scope": str(
                        decision.get("media_evidence_scope") or ""
                    ),
                    "media_ranking_scope": str(
                        decision.get("media_ranking_scope") or ""
                    ),
                    "bound_chunk_count": int(
                        decision.get("bound_chunk_count") or 0
                    ),
                    "candidate_chunk_count": int(
                        decision.get("candidate_chunk_count") or 0
                    ),
                    "evidence": [
                        {
                            "rank": int(item["rank"]),
                            "score": float(item["media_evidence_score"]),
                            "scope": str(item.get("scope") or ""),
                        }
                        for item in decision.get("evidence", [])
                    ],
                    "direct_relations": [
                        {
                            "media_vector_similarity": item.get(
                                "media_vector_similarity"
                            ),
                            "lexical_coverage": float(
                                item.get("lexical_coverage") or 0.0
                            ),
                        }
                        for item in decision.get("direct_relations", [])
                    ],
                }
            )
        rows.append(
            {
                "query": case.query,
                "text_titles": [
                    str(item.get("title") or "")
                    for item in result.get("items", [])
                ],
                "decisions": decisions,
            }
        )
    return {
        "format": "personalityrag.text_media_v1.raw_retrieval_snapshot",
        "version": 1,
        "library": library_url.rsplit("/", 1)[-1],
        "candidate_limit": 30,
        "cases": rows,
    }


def write_read_only_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.chmod(path, stat.S_IREAD)


def media_invariance_signature(snapshot: dict[str, Any]) -> dict[str, Any]:
    signature: dict[str, Any] = {}
    for case in snapshot["cases"]:
        decisions = []
        for decision in case["decisions"]:
            if decision.get("media_evidence_scope") != "asset_bound_only":
                raise RuntimeError("media evidence is not restricted to bound chunks")
            if decision.get("media_ranking_scope") != "per_asset":
                raise RuntimeError("media evidence is not ranked per asset")
            decisions.append(
                {
                    "label": decision["label"],
                    "visual_intent": decision["visual_intent"],
                    "implicit_media_lookup": decision[
                        "implicit_media_lookup"
                    ],
                    "subject_anchor_required": decision[
                        "subject_anchor_required"
                    ],
                    "subject_anchor_met": decision["subject_anchor_met"],
                    "output_confidence": decision["output_confidence"],
                    "output": decision["output"],
                    "reason": decision["reason"],
                    "bound_chunk_count": decision["bound_chunk_count"],
                    "candidate_chunk_count": decision["candidate_chunk_count"],
                    "evidence": decision["evidence"],
                    "direct_relations": [
                        {
                            "media_vector_similarity": item[
                                "media_vector_similarity"
                            ],
                            "lexical_coverage": item["lexical_coverage"],
                        }
                        for item in decision["direct_relations"]
                    ],
                }
            )
        decisions.sort(key=lambda item: item["label"])
        signature[str(case["query"])] = decisions
    return signature


def compare_media_invariance(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    baseline_signature = media_invariance_signature(baseline)
    current_signature = media_invariance_signature(current)
    shared_queries = sorted(set(baseline_signature) & set(current_signature))
    changed: list[str] = []
    maximum_delta = 0.0

    def compare(left: Any, right: Any) -> bool:
        nonlocal maximum_delta
        if isinstance(left, bool) or isinstance(right, bool):
            return left == right
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            delta = abs(float(left) - float(right))
            maximum_delta = max(maximum_delta, delta)
            return delta <= tolerance
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(
                compare(left[key], right[key]) for key in left
            )
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(
                compare(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        return left == right

    for query in shared_queries:
        if not compare(baseline_signature[query], current_signature[query]):
            changed.append(query)
    if changed:
        raise RuntimeError(
            "unbound documents changed media diagnostics for: "
            + ", ".join(changed)
        )
    return {
        "shared_query_count": len(shared_queries),
        "changed_query_count": 0,
        "media_invariant": True,
        "tolerance": tolerance,
        "maximum_numeric_delta": maximum_delta,
    }


def direct_components(
    decision: dict[str, Any], settings: dict[str, Any]
) -> tuple[float, float]:
    if not (
        decision["visual_intent"] or decision.get("implicit_media_lookup")
    ):
        return 0.0, 0.0
    best = (0.0, 0.0, -1.0)
    for relation in decision["direct_relations"]:
        semantic = clamp01(
            float(settings["media_semantic_weight"])
            * calibrated_semantic(
                relation["media_vector_similarity"],
                float(settings["media_semantic_floor"]),
            )
        )
        lexical = clamp01(
            float(settings["media_lexical_boost"])
            * float(relation["lexical_coverage"])
            ** float(settings["media_lexical_coverage_exponent"])
        )
        candidate = (semantic, lexical, noisy_or((semantic, lexical)))
        if candidate[2] > best[2]:
            best = candidate
    return best[0], best[1]


def offline_confidence(
    decision: dict[str, Any], settings: dict[str, Any]
) -> dict[str, float]:
    evidence = decision["evidence"]
    ranked = [
        float(item["score"])
        / (
            math.log2(int(item["rank"]) + 1)
            ** float(settings["media_rank_decay_exponent"])
        )
        for item in evidence
    ]
    grounding = aggregate_grounding(
        ranked,
        corroboration_weight=float(settings["media_corroboration_weight"]),
        limit=int(settings["media_corroboration_limit"]),
    )
    threshold_result = apply_evidence_threshold(
        grounding,
        ((float(item["score"]), int(item["rank"])) for item in evidence),
        threshold=EVIDENCE_THRESHOLD,
        limit=int(settings["media_threshold_evidence_limit"]),
        rank_decay_exponent=float(
            settings["media_threshold_rank_decay_exponent"]
        ),
        negative_reliability_exponent=float(
            settings["media_threshold_negative_reliability_exponent"]
        ),
        reinforcement_weight=float(
            settings["media_threshold_reinforcement_weight"]
        ),
        weakening_weight=float(settings["media_threshold_weakening_weight"]),
        tail_rank=int(settings["media_corroboration_limit"]),
    )
    semantic, lexical = direct_components(decision, settings)
    media_gate_open = bool(
        decision["visual_intent"] or decision.get("implicit_media_lookup")
    ) and (
        not bool(decision.get("subject_anchor_required"))
        or bool(decision.get("subject_anchor_met"))
    )
    confidence = (
        noisy_or(
            (float(threshold_result["adjusted_grounding"]), semantic, lexical)
        )
        if media_gate_open
        else 0.0
    )
    return {
        "confidence": confidence,
        "tail_pressure": float(threshold_result["tail_negative_pressure"]),
    }


def evaluate_snapshot(
    snapshot: dict[str, Any], settings: dict[str, Any]
) -> dict[str, Any]:
    records: list[tuple[set[str], frozenset[str]]] = []
    required_hits = required_total = negative_outputs = top_hits = top_total = 0
    margins: list[float] = []
    tail_pressure = 0.0
    rows = []
    snapshot_cases = {str(item["query"]): item for item in snapshot["cases"]}
    for case in CASES:
        scored = []
        for decision in snapshot_cases[case.query]["decisions"]:
            result = offline_confidence(decision, settings)
            scored.append(
                (
                    str(decision["label"]),
                    str(decision["asset_id"]),
                    result["confidence"],
                )
            )
            tail_pressure += result["tail_pressure"]
        scored.sort(key=lambda item: (-item[2], item[1]))
        ordered = [label for label, _, score in scored if score >= OUTPUT_THRESHOLD]
        predicted = set(ordered)
        records.append((predicted, case.positives))
        required_hits += len(predicted & case.required)
        required_total += len(case.required)
        if case.top_label is not None:
            top_total += 1
            top_hits += int(bool(ordered) and ordered[0] == case.top_label)
        if case.hard_negative:
            negative_outputs += len(predicted)
        confidence_by_label = {label: score for label, _, score in scored}
        true_scores = [confidence_by_label.get(label, 0.0) for label in case.required]
        false_scores = [
            score for label, _, score in scored if label not in case.positives
        ]
        if true_scores:
            margins.append(min(true_scores) - max(false_scores, default=0.0))
        rows.append(
            {
                "query": case.query,
                "ordered_predictions": ordered,
                "confidences": {
                    label: round(score, 6) for label, _, score in scored
                },
            }
        )
    return {
        "settings": settings,
        "core_recall": required_hits / max(1, required_total),
        "top_accuracy": top_hits / max(1, top_total),
        "negative_outputs": negative_outputs,
        "macro_f1": macro_f1(records),
        "minimum_margin": min(margins, default=-math.inf),
        "tail_pressure": tail_pressure,
        "rows": rows,
    }


def result_key(item: dict[str, Any]) -> tuple[Any, ...]:
    settings = item["settings"]
    return (
        int(item["core_recall"] == 1.0),
        int(item["negative_outputs"] == 0),
        int(item["top_accuracy"] == 1.0),
        item["macro_f1"],
        item["minimum_margin"],
        -item["tail_pressure"],
        -float(settings["media_threshold_reinforcement_weight"]),
        -float(settings["media_threshold_weakening_weight"]),
        -int(settings["media_threshold_evidence_limit"]),
    )


def threshold_grid(base: dict[str, Any]) -> Iterable[dict[str, Any]]:
    keys = (
        "media_threshold_evidence_limit",
        "media_threshold_rank_decay_exponent",
        "media_threshold_negative_reliability_exponent",
        "media_threshold_reinforcement_weight",
        "media_threshold_weakening_weight",
    )
    values = (
        (5, 10, 20, 30),
        (1.0, 1.5, 2.0, 2.5),
        (1.5, 2.0, 2.5, 3.0),
        (0.20, 0.35, 0.50),
        (0.10, 0.25, 0.40, 0.60),
    )
    for row in itertools.product(*values):
        yield {**base, **dict(zip(keys, row, strict=True))}


def direct_grid(base: dict[str, Any]) -> Iterable[dict[str, Any]]:
    keys = (
        "media_semantic_floor",
        "media_semantic_weight",
        "media_lexical_boost",
        "media_lexical_coverage_exponent",
        "media_corroboration_weight",
    )
    values = (
        (0.25, 0.30, 0.35),
        (0.80, 0.90, 1.0),
        (0.20, 0.30, 0.40),
        (1.0, 1.5, 2.0, 2.5),
        (0.30, 0.40, 0.50),
    )
    for row in itertools.product(*values):
        yield {**base, **dict(zip(keys, row, strict=True))}


def offline_grid(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    base = {
        **DEFAULT_RETRIEVAL_CONFIG,
        "media_score_threshold_fallback": EVIDENCE_THRESHOLD,
        "media_candidate_limit": 30,
    }
    threshold_results = [
        evaluate_snapshot(snapshot, settings) for settings in threshold_grid(base)
    ]
    threshold_results.sort(key=result_key, reverse=True)
    refined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for threshold_result in threshold_results[:50]:
        for settings in direct_grid(threshold_result["settings"]):
            identity = json.dumps(settings, sort_keys=True)
            if identity in seen:
                continue
            seen.add(identity)
            refined.append(evaluate_snapshot(snapshot, settings))
    refined.sort(key=result_key, reverse=True)
    return refined[:10]


def live_evaluate(
    client: httpx.Client,
    library_url: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    request_json(
        client, "PATCH", library_url, {"retrieval_settings": settings}
    )
    rows = []
    records: list[tuple[set[str], frozenset[str]]] = []
    required_hits = required_total = negative_outputs = top_hits = top_total = 0
    margins = []
    for case in CASES:
        result = request_json(
            client,
            "POST",
            f"{library_url}/search",
            {
                "query": case.query,
                "top_k": 10,
                "media_output_confidence_threshold": OUTPUT_THRESHOLD,
                "media_score_threshold": EVIDENCE_THRESHOLD,
                "max_media_outputs": 5,
            },
        )
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
            score for label, score in confidences.items() if label not in case.positives
        ]
        if true_scores:
            margins.append(min(true_scores) - max(false_scores, default=0.0))
        rows.append(
            {
                "query": case.query,
                "outputs": ordered,
                "confidences": confidences,
            }
        )
    return {
        "core_recall": required_hits / max(1, required_total),
        "top_accuracy": top_hits / max(1, top_total),
        "negative_outputs": negative_outputs,
        "macro_f1": macro_f1(records),
        "minimum_margin": min(margins, default=-math.inf),
        "rows": rows,
    }


def validate_contract(
    client: httpx.Client, library_url: str
) -> dict[str, Any]:
    top_k_runs = {}
    for top_k in (5, 10, 20, 50):
        result = request_json(
            client,
            "POST",
            f"{library_url}/search",
            {
                "query": "描述一下你的立绘",
                "top_k": top_k,
                "media_output_confidence_threshold": OUTPUT_THRESHOLD,
                "media_score_threshold": EVIDENCE_THRESHOLD,
                "max_media_outputs": 5,
            },
        )
        top_k_runs[top_k] = {
            "outputs": [
                (str(item["asset_id"]), float(item["output_confidence"]))
                for item in result.get("media_outputs", [])
            ],
            "decisions": result.get("media_decisions", []),
        }
    if any(top_k_runs[value] != top_k_runs[5] for value in (10, 20, 50)):
        raise RuntimeError("public Top-K changed media diagnostics or output")

    limit_runs = {}
    for limit in (0, 1, 2, 5):
        result = request_json(
            client,
            "POST",
            f"{library_url}/search",
            {
                "query": "展示你的头像、全身立绘和战斗场景",
                "top_k": 10,
                "media_output_confidence_threshold": 0.30,
                "media_score_threshold": EVIDENCE_THRESHOLD,
                "max_media_outputs": limit,
            },
        )
        limit_runs[limit] = [
            str(item["asset_id"]) for item in result.get("media_outputs", [])
        ]
        if len(limit_runs[limit]) > limit:
            raise RuntimeError("maximum media output count was exceeded")
    if any(limit_runs[value] != limit_runs[5][:value] for value in (0, 1, 2)):
        raise RuntimeError("media output ordering changed across output limits")

    fallback = request_json(
        client,
        "POST",
        f"{library_url}/search",
        {
            "query": "描述一下你的立绘",
            "top_k": 10,
            "media_output_confidence_threshold": OUTPUT_THRESHOLD,
            "media_score_threshold": None,
            "max_media_outputs": 5,
        },
    )
    if fallback["thresholds"]["media_score_threshold_source"] != "library_fallback":
        raise RuntimeError("null media score threshold did not use library fallback")
    threshold_runs = {}
    for threshold in (0.2, 0.35, 0.6, 0.8):
        result = request_json(
            client,
            "POST",
            f"{library_url}/search",
            {
                "query": "描述一下你的立绘",
                "top_k": 10,
                "media_output_confidence_threshold": OUTPUT_THRESHOLD,
                "media_score_threshold": threshold,
                "max_media_outputs": 5,
            },
        )
        threshold_runs[threshold] = {
            classify(str(item.get("original_name") or "")): float(
                item.get("output_confidence") or 0.0
            )
            for item in result.get("media_decisions", [])
        }
    return {
        "top_k": {key: value["outputs"] for key, value in top_k_runs.items()},
        "limits": limit_runs,
        "fallback_source": "library_fallback",
        "threshold_sweep": threshold_runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate text_media_v1 from one real-provider snapshot and verify the top candidates live."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--library", default="beileite_test")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("reports/text_media_v1_raw_snapshot.json"),
    )
    parser.add_argument("--refresh-snapshot", action="store_true")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--compare-snapshot", type=Path)
    parser.add_argument("--invariance-tolerance", type=float, default=1e-6)
    parser.add_argument("--apply-best", action="store_true")
    args = parser.parse_args()
    root = args.base_url.rstrip("/")
    library_url = f"{root}/api/v1/knowledge-libraries/text_media_v1/{args.library}"
    headers = {"Authorization": f"Bearer {args.api_key}"}
    with httpx.Client(headers=headers, timeout=180.0) as client:
        library = request_json(client, "GET", library_url)
        original = dict(library["retrieval_settings"])
        completed = False
        try:
            if args.refresh_snapshot or not args.snapshot.exists():
                snapshot = capture_snapshot(client, library_url, original)
                write_read_only_snapshot(args.snapshot, snapshot)
            else:
                snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
            comparison = None
            if args.compare_snapshot is not None:
                baseline = json.loads(
                    args.compare_snapshot.read_text(encoding="utf-8")
                )
                comparison = compare_media_invariance(
                    baseline,
                    snapshot,
                    tolerance=max(0.0, args.invariance_tolerance),
                )
            if args.capture_only:
                print(
                    json.dumps(
                        {
                            "snapshot": str(args.snapshot),
                            "case_count": len(snapshot["cases"]),
                            "comparison": comparison,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
            offline_top = offline_grid(snapshot)
            live_candidates = []
            for index, candidate in enumerate(offline_top, start=1):
                rounds = [
                    live_evaluate(client, library_url, candidate["settings"])
                    for _ in range(3)
                ]
                signatures = [
                    json.dumps(round_["rows"], sort_keys=True, ensure_ascii=False)
                    for round_ in rounds
                ]
                live_candidates.append(
                    {
                        **candidate,
                        "live_rounds": rounds,
                        "cross_round_stable": len(set(signatures)) == 1,
                    }
                )
                print(
                    f"[{index}/10] recall={rounds[0]['core_recall']:.3f} "
                    f"top={rounds[0]['top_accuracy']:.3f} "
                    f"negative={rounds[0]['negative_outputs']} "
                    f"f1={rounds[0]['macro_f1']:.3f} "
                    f"stable={len(set(signatures)) == 1}",
                    file=sys.stderr,
                )
            valid = [
                item
                for item in live_candidates
                if item["cross_round_stable"]
                and all(
                    round_["core_recall"] == 1.0
                    and round_["negative_outputs"] == 0
                    and round_["top_accuracy"] == 1.0
                    for round_ in item["live_rounds"]
                )
            ]
            if not valid:
                raise RuntimeError("no live-verified parameter set passed hard constraints")
            valid.sort(
                key=lambda item: (
                    item["live_rounds"][0]["macro_f1"],
                    item["live_rounds"][0]["minimum_margin"],
                    -item["tail_pressure"],
                    -float(item["settings"]["media_threshold_reinforcement_weight"]),
                    -float(item["settings"]["media_threshold_weakening_weight"]),
                    -int(item["settings"]["media_threshold_evidence_limit"]),
                ),
                reverse=True,
            )
            best = valid[0]
            request_json(
                client,
                "PATCH",
                library_url,
                {"retrieval_settings": best["settings"]},
            )
            best["contract_validation"] = validate_contract(client, library_url)
            completed = True
        finally:
            if not (args.apply_best and completed):
                request_json(
                    client,
                    "PATCH",
                    library_url,
                    {"retrieval_settings": original},
                )
        print(json.dumps(best, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
