from __future__ import annotations

import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from benchmark_text_media_v1_text_relevance import (  # noqa: E402
    CASES,
    _hard_constraints,
    _metrics,
    _simulate,
)


def _item(
    chunk_id: int,
    title: str,
    text: str,
    embedding: float,
    rerank: float,
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "title": title,
        "text": text,
        "score": embedding,
        "embedding_relevance": embedding,
        "rerank_raw_score": rerank,
        "initial_rank": chunk_id,
    }


def test_text_benchmark_covers_required_semantic_families() -> None:
    assert len(CASES) >= 18
    assert {
        "friend",
        "identity",
        "appearance",
        "shikai",
        "bankai",
        "longsword",
        "ecology",
        "pdf",
        "none",
    } <= {case.family for case in CASES}


def test_offline_simulation_uses_absolute_log_odds_scores() -> None:
    rows = [
        _item(1, "群友", "萌依是一只 ai 猫娘", 0.73, 0.98),
        _item(2, "澄月", "无关内容", 0.34, 0.01),
    ]
    result = _simulate(
        rows,
        candidate_limit=2,
        fusion_weight=0.35,
        rank_bonus_weight=0.0,
        exponent=3.0,
    )

    assert result[0]["chunk_id"] == 1
    assert float(result[0]["score"]) > 0.80
    assert float(result[1]["score"]) < 0.20


def test_metrics_and_hard_constraints_reject_rank_regression() -> None:
    case = next(case for case in CASES if case.id == "friend_direct")
    relevant = _item(1, "群友", "萌依和你一样是 ai", 0.73, 0.98)
    irrelevant = _item(2, "澄月", "无关", 0.34, 0.01)
    baseline = {case.id: [relevant, irrelevant]}
    good = {
        case.id: [
            {**relevant, "score": 0.9},
            {**irrelevant, "score": 0.01},
        ]
    }
    bad = {
        case.id: [
            {**irrelevant, "score": 0.9},
            {**irrelevant, "chunk_id": 3, "score": 0.8},
            {**irrelevant, "chunk_id": 4, "score": 0.7},
            {**relevant, "score": 0.6},
        ]
    }

    assert _metrics((case,), good)["mrr"] == 1.0
    assert _hard_constraints((case,), baseline, good)["passed"] is True
    assert _hard_constraints((case,), baseline, bad)["passed"] is False
