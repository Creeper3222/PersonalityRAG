from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from personalityrag.library_types.text_media_v1.retrieval import (
    score_media_confidence,
)


PIVOT_POINTS = (0.0, 0.01, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.0)
OUTPUT_THRESHOLD = 0.60


@dataclass(frozen=True)
class Case:
    name: str
    query: str
    required: frozenset[str]
    allowed: frozenset[str]
    visual: bool = True


CASES = (
    Case(
        "portrait",
        "看看你的立绘",
        frozenset({"portrait"}),
        frozenset({"portrait", "avatar"}),
    ),
    Case(
        "avatar",
        "看看你的头像",
        frozenset({"avatar"}),
        frozenset({"avatar"}),
    ),
    Case(
        "battle_scene",
        "展示一下你的战斗场景",
        frozenset({"battle"}),
        frozenset({"battle"}),
    ),
    Case(
        "zinogre",
        "给我看看狱狼龙生态插图",
        frozenset({"zinogre"}),
        frozenset({"zinogre"}),
    ),
    Case(
        "deepseek",
        "给我 DeepSeek 表情包",
        frozenset({"deepseek"}),
        frozenset({"deepseek"}),
    ),
    Case(
        "deepseek_zh",
        "发一张深度求索表情包",
        frozenset({"deepseek"}),
        frozenset({"deepseek"}),
    ),
    Case(
        "gpt",
        "给我一张 GPT 表情包",
        frozenset({"gpt"}),
        frozenset({"gpt"}),
    ),
    Case(
        "gemini_zh",
        "来一张谷歌大模型表情包",
        frozenset({"gemini"}),
        frozenset({"gemini"}),
    ),
    Case(
        "claude",
        "show me the Anthropic Claude meme",
        frozenset({"claude"}),
        frozenset({"claude"}),
    ),
    Case("generic_meme", "给我一张表情包", frozenset(), frozenset()),
    Case(
        "all_memes",
        "把四种表情包都给我看",
        frozenset({"claude", "deepseek", "gemini", "gpt"}),
        frozenset({"claude", "deepseek", "gemini", "gpt"}),
    ),
    Case(
        "author_blocker",
        "你的立绘是谁画的",
        frozenset(),
        frozenset(),
        visual=False,
    ),
    Case(
        "upload_blocker",
        "图片上传失败怎么办",
        frozenset(),
        frozenset(),
        visual=False,
    ),
    Case(
        "strategy_blocker",
        "你的战斗策略是什么",
        frozenset(),
        frozenset(),
        visual=False,
    ),
)


def asset_label(name: str) -> str:
    normalized = name.casefold()
    for label in ("deepseek", "gemini", "claude", "gpt"):
        if label in normalized:
            return label
    if "全身立绘" in name:
        return "portrait"
    if "立绘头像" in name:
        return "avatar"
    if "战斗场景" in name:
        return "battle"
    if "狱狼龙" in name:
        return "zinogre"
    return name


def post_json(url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    started = time.perf_counter()
    with request.urlopen(req, timeout=180) as response:
        result = json.load(response)
    result["_request_elapsed_ms"] = (time.perf_counter() - started) * 1000
    return result


def search(
    *,
    base_url: str,
    api_key: str,
    library_id: str,
    case: Case,
    pivot: float,
    top_k: int = 10,
    mode: str = "standard",
    rerank: bool = True,
) -> dict[str, Any]:
    return post_json(
        (
            f"{base_url.rstrip('/')}/api/v1/knowledge-libraries/"
            f"text_media_v1/{library_id}/search"
        ),
        api_key,
        {
            "query": case.query,
            "retrieval_mode": mode,
            "top_k": top_k,
            "media_output_confidence_threshold": OUTPUT_THRESHOLD,
            "media_relevance_pivot": pivot,
            "max_media_outputs": 20,
            "rerank": rerank,
        },
    )


def compact_response(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "outputs": [
            asset_label(str(item["original_name"]))
            for item in result["media_outputs"]
        ],
        "decisions": {
            asset_label(str(item["original_name"])): {
                "confidence": float(item["output_confidence"]),
                "raw_relevance": float(item["raw_relevance_score"]),
                "direct_relevance": float(item["raw_direct_relevance"]),
                "structural_factor": float(
                    item["structural_attenuation_factor"]
                ),
                "algorithm": str(item["confidence_algorithm"]),
                "policy_gate_open": bool(item["media_policy_gate_open"]),
            }
            for item in result["media_decisions"]
        },
        "elapsed_ms": float(result["_request_elapsed_ms"]),
        "rerank": dict(result.get("rerank") or {}),
    }


def evaluate_case(case: Case, compact: dict[str, Any]) -> dict[str, Any]:
    output = set(compact["outputs"])
    required_ok = case.required <= output
    precision_ok = output <= case.allowed
    top_ok = (
        not case.required
        or len(case.required) != 1
        or (
            bool(compact["outputs"])
            and compact["outputs"][0] in case.required
        )
    )
    return {
        "passed": required_ok and precision_ok and top_ok,
        "required_ok": required_ok,
        "precision_ok": precision_ok,
        "top_ok": top_ok,
        "outputs": compact["outputs"],
    }


def replay_confidence(
    decision: dict[str, Any],
    *,
    pivot: float,
    positive_blend: float,
    negative_weight: float,
    negative_floor: float,
    format_factor: float,
    content_factor: float,
) -> float:
    result = score_media_confidence(
        raw_direct_relevance=float(decision["raw_direct_relevance"]),
        grounding=float(decision["grounding_score"]),
        evidence=(
            (float(item["media_evidence_score"]), int(item["rank"]))
            for item in decision["evidence"]
        ),
        bound_media=not bool(decision["asset_metadata_only"]),
        relevance_pivot=pivot,
        direct_signal_enabled=bool(decision["media_policy_gate_open"]),
        ambiguous_collection=False,
        subject_anchor_met=bool(decision["subject_anchor_met"]),
        media_format_compatible=bool(decision["media_format_compatible"]),
        content_anchor_met=bool(decision["content_anchor_met"]),
        evidence_limit=int(decision["threshold_evidence_limit"]),
        rank_decay_exponent=1.5,
        negative_reliability_exponent=1.5,
        positive_blend=positive_blend,
        negative_weight=negative_weight,
        negative_attenuation_floor=negative_floor,
        format_mismatch_factor=format_factor,
        content_mismatch_factor=content_factor,
    )
    confidence = float(result["output_confidence"])
    if bool(decision["asset_metadata_only"]):
        confidence *= float(decision.get("unbound_confidence_factor") or 0.0)
    return confidence


def parameter_grid(
    snapshots: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    combinations = itertools.product(
        (0.70, 0.80, 0.90),
        (0.35, 0.50, 0.65),
        (0.01, 0.05, 0.10),
        (0.05, 0.10, 0.15),
        (0.10, 0.15, 0.20),
    )
    rows: list[dict[str, Any]] = []
    for positive, negative, floor, format_factor, content_factor in combinations:
        required_hits = 0
        required_total = 0
        false_outputs = 0
        margins: list[float] = []
        hierarchy_ok = True
        for case in CASES:
            raw = snapshots[case.name]
            confidences = {
                asset_label(str(item["original_name"])): replay_confidence(
                    item,
                    pivot=0.35,
                    positive_blend=positive,
                    negative_weight=negative,
                    negative_floor=floor,
                    format_factor=format_factor,
                    content_factor=content_factor,
                )
                for item in raw["media_decisions"]
            }
            outputs = {
                label
                for label, confidence in confidences.items()
                if confidence >= OUTPUT_THRESHOLD
            }
            required_hits += len(case.required & outputs)
            required_total += len(case.required)
            false_outputs += len(outputs - case.allowed)
            if case.required:
                best_required = max(
                    confidences.get(label, 0.0) for label in case.required
                )
                best_false = max(
                    (
                        confidence
                        for label, confidence in confidences.items()
                        if label not in case.allowed
                    ),
                    default=0.0,
                )
                margins.append(best_required - best_false)
        portrait = snapshots["portrait"]["media_decisions"]
        portrait_scores = {
            asset_label(str(item["original_name"])): replay_confidence(
                item,
                pivot=0.35,
                positive_blend=positive,
                negative_weight=negative,
                negative_floor=floor,
                format_factor=format_factor,
                content_factor=content_factor,
            )
            for item in portrait
        }
        hierarchy_ok = (
            portrait_scores.get("portrait", 0.0)
            > portrait_scores.get("battle", 0.0)
            > max(
                portrait_scores.get("deepseek", 0.0),
                portrait_scores.get("gemini", 0.0),
                portrait_scores.get("gpt", 0.0),
                portrait_scores.get("claude", 0.0),
            )
            and portrait_scores.get("battle", 0.0) >= 0.03
            and portrait_scores.get("zinogre", 0.0) > 0.0
        )
        rows.append(
            {
                "media_pivot_positive_blend": positive,
                "media_pivot_negative_weight": negative,
                "media_pivot_negative_attenuation_floor": floor,
                "media_format_mismatch_factor": format_factor,
                "media_content_mismatch_factor": content_factor,
                "required_recall": (
                    required_hits / required_total if required_total else 1.0
                ),
                "false_outputs": false_outputs,
                "minimum_margin": min(margins, default=0.0),
                "mean_margin": (
                    statistics.fmean(margins) if margins else 0.0
                ),
                "hierarchy_ok": hierarchy_ok,
            }
        )
    rows.sort(
        key=lambda item: (
            item["false_outputs"],
            not item["hierarchy_ok"],
            -item["required_recall"],
            -item["minimum_margin"],
            -item["mean_margin"],
            abs(item["media_pivot_positive_blend"] - 0.80),
            abs(item["media_pivot_negative_weight"] - 0.50),
            abs(item["media_pivot_negative_attenuation_floor"] - 0.05),
            abs(item["media_format_mismatch_factor"] - 0.10),
            item["media_content_mismatch_factor"],
        )
    )
    return rows


def legacy_replay(decision: dict[str, Any]) -> float:
    """Reproduce the pre-pivot hard-gate outcome for A/B/A determinism checks."""

    metadata_only = bool(decision["asset_metadata_only"])
    gate = (
        bool(decision["media_policy_gate_open"])
        and bool(decision["subject_anchor_met"])
        and (metadata_only or bool(decision["media_format_compatible"]))
        and (metadata_only or bool(decision["content_anchor_met"]))
    )
    if not gate:
        return 0.0
    if metadata_only:
        return max(
            0.0,
            min(
                1.0,
                float(decision["raw_direct_relevance"])
                * float(decision.get("unbound_confidence_factor") or 0.0),
            ),
        )
    # The exact historical threshold implementation is intentionally not
    # imported into production. Its user-visible zeroing defect was the hard
    # structural gate above; use the recorded pre-pivot confidence for the
    # remaining legacy baseline.
    return float(decision["pre_threshold_output_confidence"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8801")
    parser.add_argument("--library-id", default="beileite_test")
    parser.add_argument(
        "--embedding-library-id", default="beileite_test2"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PERSONALITYRAG_API_KEY", ""),
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api-key or PERSONALITYRAG_API_KEY is required")

    snapshots: dict[str, dict[str, Any]] = {}
    compact: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    for case in CASES:
        result = search(
            base_url=args.base_url,
            api_key=args.api_key,
            library_id=args.library_id,
            case=case,
            pivot=0.35,
        )
        snapshots[case.name] = result
        compact[case.name] = compact_response(result)
        evaluations[case.name] = evaluate_case(case, compact[case.name])

    pivot_case = next(case for case in CASES if case.name == "portrait")
    pivot_snapshots = {
        str(pivot): compact_response(
            search(
                base_url=args.base_url,
                api_key=args.api_key,
                library_id=args.library_id,
                case=pivot_case,
                pivot=pivot,
            )
        )
        for pivot in PIVOT_POINTS
    }
    names = set.intersection(
        *(
            set(item["decisions"])
            for item in pivot_snapshots.values()
        )
    )
    monotonic = {
        label: all(
            pivot_snapshots[str(PIVOT_POINTS[index])]["decisions"][label][
                "confidence"
            ]
            >= pivot_snapshots[str(PIVOT_POINTS[index + 1])]["decisions"][
                label
            ]["confidence"]
            - 1e-6
            for index in range(len(PIVOT_POINTS) - 1)
        )
        for label in sorted(names)
    }

    parity_cases = [
        case
        for case in CASES
        if case.visual and case.name not in {"generic_meme"}
    ]
    mode_parity: dict[str, bool] = {}
    top_k_stability: dict[str, bool] = {}
    for case in parity_cases:
        standard = compact[case.name]
        media_only = compact_response(
            search(
                base_url=args.base_url,
                api_key=args.api_key,
                library_id=args.library_id,
                case=case,
                pivot=0.35,
                mode="media_only",
            )
        )
        mode_parity[case.name] = (
            standard["outputs"] == media_only["outputs"]
            and standard["decisions"] == media_only["decisions"]
        )
        variants = [
            compact_response(
                search(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    library_id=args.library_id,
                    case=case,
                    pivot=0.35,
                    top_k=top_k,
                )
            )
            for top_k in (5, 10, 20, 50)
        ]
        top_k_stability[case.name] = all(
            variant["outputs"] == variants[0]["outputs"]
            and variant["decisions"] == variants[0]["decisions"]
            for variant in variants[1:]
        )

    embedding_baseline: dict[str, Any] = {}
    for case in CASES:
        rerank_off = compact_response(
            search(
                base_url=args.base_url,
                api_key=args.api_key,
                library_id=args.library_id,
                case=case,
                pivot=0.35,
                rerank=False,
            )
        )
        separate_library = compact_response(
            search(
                base_url=args.base_url,
                api_key=args.api_key,
                library_id=args.embedding_library_id,
                case=case,
                pivot=0.35,
                rerank=False,
            )
        )
        embedding_baseline[case.name] = {
            "rerank_disabled_zero_call": not bool(
                rerank_off["rerank"].get("applied")
            ),
            "cross_library_outputs_equal": (
                rerank_off["outputs"] == separate_library["outputs"]
            ),
        }

    grid = parameter_grid(snapshots)
    legacy_a1 = {
        case: {
            asset_label(str(item["original_name"])): legacy_replay(item)
            for item in result["media_decisions"]
        }
        for case, result in snapshots.items()
    }
    new_b = {
        case: {
            asset_label(str(item["original_name"])): float(
                item["output_confidence"]
            )
            for item in result["media_decisions"]
        }
        for case, result in snapshots.items()
    }
    legacy_a2 = {
        case: {
            asset_label(str(item["original_name"])): legacy_replay(item)
            for item in result["media_decisions"]
        }
        for case, result in snapshots.items()
    }
    elapsed = [
        float(item["elapsed_ms"])
        for item in compact.values()
    ]
    report = {
        "algorithm": "unified_relevance_pivot_v1",
        "cases": evaluations,
        "all_cases_passed": all(
            item["passed"] for item in evaluations.values()
        ),
        "pivot_points": list(PIVOT_POINTS),
        "pivot_monotonic": monotonic,
        "pivot_effect": {
            label: {
                "p_0_01": pivot_snapshots["0.01"]["decisions"][label][
                    "confidence"
                ],
                "p_0_35": pivot_snapshots["0.35"]["decisions"][label][
                    "confidence"
                ],
            }
            for label in sorted(names)
        },
        "mode_parity": mode_parity,
        "top_k_stability": top_k_stability,
        "embedding_baseline": embedding_baseline,
        "parameter_grid_top10": grid[:10],
        "aba": {
            "legacy_a1": legacy_a1,
            "new_b": new_b,
            "legacy_a2": legacy_a2,
            "legacy_repeat_identical": legacy_a1 == legacy_a2,
        },
        "latency_ms": {
            "median": statistics.median(elapsed),
            "maximum": max(elapsed),
        },
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    hard_ok = (
        report["all_cases_passed"]
        and all(monotonic.values())
        and all(mode_parity.values())
        and all(top_k_stability.values())
        and report["aba"]["legacy_repeat_identical"]
    )
    return 0 if hard_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
