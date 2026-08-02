from __future__ import annotations

import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from benchmark_text_media_v1_semantic_rerank import (  # noqa: E402
    QueryCase,
    aggregate_metrics,
    bootstrap_interval,
    load_cases,
    regression_count,
    rerank_order,
    select_live_finalists,
)


FIXTURE = Path(__file__).parent / "fixtures" / "text_media_v1_semantic_qrels.json"


def test_semantic_qrels_have_fixed_topics_styles_and_holdout_cases() -> None:
    _payload, cases = load_cases(FIXTURE)

    topic_cases = [
        case
        for case in cases
        if case.style in {"direct", "second_person", "paraphrase", "indirect"}
    ]
    assert len(topic_cases) == 100
    assert len({case.topic for case in topic_cases}) == 25
    assert {case.style for case in topic_cases} == {
        "direct",
        "second_person",
        "paraphrase",
        "indirect",
    }
    assert sum(case.sealed for case in cases) == 44
    assert sum(case.hard_negative for case in cases) == 10


def test_monotonic_residual_order_preserves_strong_embedding_lead() -> None:
    row = {
        "baseline": [
            {"document": "doc", "ordinal": 0, "score": 1.0},
            {"document": "doc", "ordinal": 1, "score": 0.90},
            {"document": "doc", "ordinal": 2, "score": 0.80},
        ],
        "rerank_raw_scores": [0.001, 0.002, 0.003],
    }

    ordered = rerank_order(
        row,
        strategy="monotonic_residual",
        candidate_limit=10,
        fusion_weight=0.60,
        rank_bonus_weight=0.0,
        exponent=1.0,
        gate=0.0,
    )

    assert ordered == [("doc", 0), ("doc", 1), ("doc", 2)]


def test_dual_channel_order_can_promote_candidate_without_escape() -> None:
    row = {
        "baseline": [
            {"document": "doc", "ordinal": 0, "score": 1.0},
            {"document": "doc", "ordinal": 1, "score": 0.995},
            {"document": "doc", "ordinal": 2, "score": 0.99},
            {"document": "doc", "ordinal": 3, "score": 0.5},
        ],
        "rerank_raw_scores": [0.01, 0.99, 0.02, 1.0],
    }

    ordered = rerank_order(
        row,
        strategy="dual_channel",
        candidate_limit=3,
        fusion_weight=0.035,
        rank_bonus_weight=0.0,
        exponent=1.0,
        gate=0.0,
    )

    assert ordered[0] == ("doc", 1)
    assert ordered[-1] == ("doc", 3)


def test_ordinary_convex_is_independent_from_reliability_gated_order() -> None:
    row = {
        "baseline": [
            {"document": "doc", "ordinal": 0, "score": 1.0},
            {"document": "doc", "ordinal": 1, "score": 0.99},
        ],
        "rerank_raw_scores": [0.0, 0.5],
    }

    dual = rerank_order(
        row,
        strategy="dual_channel",
        candidate_limit=10,
        fusion_weight=0.05,
        rank_bonus_weight=0.0,
        exponent=3.0,
        gate=0.0,
    )
    convex = rerank_order(
        row,
        strategy="convex",
        candidate_limit=10,
        fusion_weight=0.05,
        rank_bonus_weight=0.0,
        exponent=3.0,
        gate=0.0,
    )

    assert dual[0] == ("doc", 0)
    assert convex[0] == ("doc", 1)


def test_graded_metrics_and_regression_guard_use_stable_chunk_keys() -> None:
    case = QueryCase(
        id="case",
        topic="topic",
        family="family",
        style="direct",
        query="query",
        qrels={("doc", 0): 3, ("doc", 1): 1},
        allowed_media=frozenset(),
        required_media=frozenset(),
        top_media=None,
        sealed=False,
        hard_negative=False,
    )
    baseline = {"case": [("doc", 0), ("doc", 1), ("doc", 2)]}
    degraded = {"case": [("doc", 2), ("doc", 1), ("doc", 3), ("doc", 0)]}

    metrics = aggregate_metrics([case], baseline)
    assert metrics["ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["recall_at_1"] == pytest.approx(1.0)
    assert regression_count([case], baseline, degraded) == 1


def test_paired_bootstrap_is_deterministic() -> None:
    first = bootstrap_interval([0.0, 0.5, 1.0], [0.1, 0.5, 1.0])
    second = bootstrap_interval([0.0, 0.5, 1.0], [0.1, 0.5, 1.0])

    assert first == second
    assert first[0] >= 0.0


def test_live_finalist_selection_keeps_nine_leaders_and_safe_incumbent() -> None:
    leaders = [
        {
            "strategy": "dual_channel",
            "candidate_limit": 10,
            "fusion_weight": 0.03 + index / 1000,
            "rank_bonus_weight": 0.0,
            "rank_reliability_exponent": 2.5,
            "gate": 0.0,
        }
        for index in range(12)
    ]
    leaders.insert(
        0,
        {
            "strategy": "convex",
            "candidate_limit": 10,
            "fusion_weight": 1.0,
            "rank_bonus_weight": 0.0,
            "rank_reliability_exponent": 1.0,
            "gate": 0.0,
        },
    )

    selected = select_live_finalists({"top_twenty": leaders})

    assert len(selected) == 10
    assert selected[:9] == [
        (10, 0.03 + index / 1000, 0.0, 2.5) for index in range(9)
    ]
    assert (10, 0.30, 0.0, 1.5) in selected
