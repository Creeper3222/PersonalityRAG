from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping

from .text import DEFAULT_VISUAL_INTENT_POLICY, normalize_visual_intent_policy


DEFAULT_RETRIEVAL_CONFIG: dict[str, Any] = {
    "rrf_k": 60,
    "rerank_candidate_limit": 10,
    "rerank_fusion_weight": 0.30,
    "rerank_rank_bonus_weight": 0.0,
    "rerank_rank_reliability_exponent": 1.5,
    "text_lexical_boost": 0.60,
    "media_candidate_limit": 30,
    "unbound_media_candidate_limit": 10,
    "media_relevance_pivot_fallback": 0.35,
    "media_pivot_positive_blend": 0.70,
    "media_pivot_negative_weight": 0.35,
    "media_pivot_negative_attenuation_floor": 0.05,
    "media_format_mismatch_factor": 0.10,
    "media_content_mismatch_factor": 0.10,
    # Deprecated compatibility projections. Normalization keeps these values
    # synchronized with the canonical relevance-pivot settings.
    "media_score_threshold_fallback": 0.35,
    "media_threshold_evidence_limit": 5,
    "media_threshold_rank_decay_exponent": 1.5,
    "media_threshold_negative_reliability_exponent": 1.5,
    "media_threshold_reinforcement_weight": 0.70,
    "media_threshold_weakening_weight": 0.35,
    "visual_intent_gate_enabled": True,
    "visual_intent_policy": normalize_visual_intent_policy(
        DEFAULT_VISUAL_INTENT_POLICY
    ),
    "media_semantic_floor": 0.35,
    "media_semantic_weight": 0.80,
    "media_lexical_boost": 0.30,
    "media_lexical_coverage_exponent": 1.20,
    "media_lexical_common_floor": 0.00,
    "media_lexical_oov_penalty": 0.30,
    "media_distinctive_rarity_exponent": 1.50,
    "unbound_media_distinctive_boost": 0.35,
    "unbound_media_collection_boost": 0.55,
    "unbound_media_competition_floor": 0.35,
    "unbound_media_reliability_target": 0.25,
    "unbound_media_specificity_exponent": 1.0,
    "unbound_media_advantage_target": 0.04,
    "media_bound_distinctive_boost": 0.10,
    "media_bound_distinctive_rescue_min": 0.80,
    "media_rank_decay_exponent": 1.0,
    "media_corroboration_weight": 0.50,
    "media_corroboration_limit": 5,
}

RERANK_CALIBRATION_CONFIG_KEYS = (
    "rerank_candidate_limit",
    "rerank_fusion_weight",
    "rerank_rank_bonus_weight",
    "rerank_rank_reliability_exponent",
)

RERANK_FUSION_ALGORITHM = "log_odds_absolute_relevance_v8"
MINIMUM_RERANK_REORDER_PROBABILITY = 0.0
LEGACY_INEFFECTIVE_RERANK_FUSION_WEIGHT = 0.00004
UNBOUND_MEDIA_SPECIFICITY_ALGORITHM = "structure_aware_unbound_competition_v1"
MEDIA_FREQUENCY_ALGORITHM = "corpus_distinctive_bm25_v1"
MEDIA_CONFIDENCE_ALGORITHM = "unified_relevance_pivot_v1"
TEXT_RELEVANCE_ALGORITHM = "absolute_dense_lexical_v1"

_LEGACY_CONFIG_ALIASES = {
    "unbound_media_candidate_limit": "media_only_asset_candidate_limit",
    "unbound_media_distinctive_boost": "media_only_distinctive_boost",
    "unbound_media_collection_boost": "media_only_collection_boost",
    "unbound_media_competition_floor": "media_only_competition_floor",
    "media_relevance_pivot_fallback": "media_score_threshold_fallback",
    "media_pivot_positive_blend": "media_threshold_reinforcement_weight",
    "media_pivot_negative_weight": "media_threshold_weakening_weight",
}

_INTEGER_RANGES = {
    "rrf_k": (1, 1000),
    "rerank_candidate_limit": (10, 200),
    "media_candidate_limit": (10, 500),
    "unbound_media_candidate_limit": (10, 500),
    "media_threshold_evidence_limit": (1, 100),
    "media_corroboration_limit": (1, 50),
}
_FLOAT_RANGES = {
    "rerank_fusion_weight": (0.0, 1.0),
    "rerank_rank_bonus_weight": (0.0, 1.0),
    "rerank_rank_reliability_exponent": (0.5, 4.0),
    "text_lexical_boost": (0.0, 1.0),
    "media_relevance_pivot_fallback": (0.0, 1.0),
    "media_pivot_positive_blend": (0.0, 1.0),
    "media_pivot_negative_weight": (0.0, 1.0),
    "media_pivot_negative_attenuation_floor": (0.001, 1.0),
    "media_format_mismatch_factor": (0.001, 1.0),
    "media_content_mismatch_factor": (0.001, 1.0),
    "media_score_threshold_fallback": (0.0, 1.0),
    "media_threshold_rank_decay_exponent": (0.0, 4.0),
    "media_threshold_negative_reliability_exponent": (1.0, 4.0),
    "media_threshold_reinforcement_weight": (0.0, 1.0),
    "media_threshold_weakening_weight": (0.0, 1.0),
    "media_semantic_floor": (0.0, 0.99),
    "media_semantic_weight": (0.0, 1.0),
    "media_lexical_boost": (0.0, 1.0),
    "media_lexical_coverage_exponent": (0.1, 4.0),
    "media_lexical_common_floor": (0.0, 1.0),
    "media_lexical_oov_penalty": (0.0, 2.0),
    "media_distinctive_rarity_exponent": (0.25, 4.0),
    "unbound_media_distinctive_boost": (0.0, 1.0),
    "unbound_media_collection_boost": (0.0, 1.0),
    "unbound_media_competition_floor": (0.0, 1.0),
    "unbound_media_reliability_target": (0.01, 1.0),
    "unbound_media_specificity_exponent": (0.25, 4.0),
    "unbound_media_advantage_target": (0.01, 1.0),
    "media_bound_distinctive_boost": (0.0, 1.0),
    "media_bound_distinctive_rescue_min": (0.0, 1.0),
    "media_rank_decay_exponent": (0.0, 4.0),
    "media_corroboration_weight": (0.0, 1.0),
}


def normalize_retrieval_config(value: Any = None) -> dict[str, Any]:
    """Return a complete, validated retrieval configuration.

    Unknown keys are deliberately ignored so a package written by a newer
    application can still be opened by an older build without changing the
    public retrieval surface.
    """

    if value in (None, ""):
        supplied: Mapping[str, Any] = {}
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("retrieval configuration is not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("retrieval configuration must be an object")
        supplied = decoded
    elif isinstance(value, Mapping):
        supplied = value
    else:
        raise ValueError("retrieval configuration must be an object")

    result = dict(DEFAULT_RETRIEVAL_CONFIG)
    supplied = dict(supplied)
    legacy_default_threshold_weights = (
        "media_pivot_positive_blend" not in supplied
        and "media_pivot_negative_weight" not in supplied
        and math.isclose(
            float(supplied.get("media_threshold_reinforcement_weight", 0.20)),
            0.20,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(supplied.get("media_threshold_weakening_weight", 0.10)),
            0.10,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and (
            "media_threshold_reinforcement_weight" in supplied
            or "media_threshold_weakening_weight" in supplied
        )
    )
    for canonical, legacy in _LEGACY_CONFIG_ALIASES.items():
        if canonical not in supplied and legacy in supplied:
            supplied[canonical] = supplied[legacy]
    if legacy_default_threshold_weights:
        supplied["media_pivot_positive_blend"] = float(
            DEFAULT_RETRIEVAL_CONFIG["media_pivot_positive_blend"]
        )
        supplied["media_pivot_negative_weight"] = float(
            DEFAULT_RETRIEVAL_CONFIG["media_pivot_negative_weight"]
        )
    for key, (minimum, maximum) in _INTEGER_RANGES.items():
        if key not in supplied:
            continue
        raw = supplied[key]
        if isinstance(raw, bool):
            raise ValueError(f"{key} must be an integer")
        number = int(raw)
        if float(raw) != number or not minimum <= number <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")
        result[key] = number
    for key, (minimum, maximum) in _FLOAT_RANGES.items():
        if key not in supplied:
            continue
        number = float(supplied[key])
        if not math.isfinite(number) or not minimum <= number <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")
        if (
            key == "rerank_fusion_weight"
            and math.isclose(
                number,
                LEGACY_INEFFECTIVE_RERANK_FUSION_WEIGHT,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            number = float(DEFAULT_RETRIEVAL_CONFIG[key])
        result[key] = number
    if "visual_intent_gate_enabled" in supplied:
        raw = supplied["visual_intent_gate_enabled"]
        if not isinstance(raw, bool):
            raise ValueError("visual_intent_gate_enabled must be a boolean")
        result["visual_intent_gate_enabled"] = raw
    result["visual_intent_policy"] = normalize_visual_intent_policy(
        supplied.get("visual_intent_policy")
    )
    result["media_threshold_evidence_limit"] = min(
        int(result["media_threshold_evidence_limit"]),
        int(result["media_candidate_limit"]),
    )
    result["media_score_threshold_fallback"] = float(
        result["media_relevance_pivot_fallback"]
    )
    result["media_threshold_reinforcement_weight"] = float(
        result["media_pivot_positive_blend"]
    )
    result["media_threshold_weakening_weight"] = float(
        result["media_pivot_negative_weight"]
    )
    return result


def retrieval_config_json(value: Any = None) -> str:
    normalized = normalize_retrieval_config(value)
    # The visual-intent lexicon is stored in a dedicated per-library CSV.
    # Keeping the field out of SQLite also makes absence unambiguously mean
    # type-level inheritance for old and new callers alike.
    normalized.pop("visual_intent_policy", None)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def rerank_calibration_settings_fingerprint(value: Any = None) -> str:
    """Fingerprint the shared settings that shape persisted Rerank calibration."""

    settings = normalize_retrieval_config(value)
    payload = {"fusion_algorithm": RERANK_FUSION_ALGORITHM}
    payload.update(
        {
            key: settings[key]
            for key in RERANK_CALIBRATION_CONFIG_KEYS
        }
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def media_relevance_pivot(score: float, pivot: float) -> float:
    """Calibrate relevance around a user-controlled odds pivot.

    The transform preserves the zero and one endpoints, maps ``score == pivot``
    to 0.5, and changes every real positive signal monotonically when the pivot
    moves. Explicit endpoint handling also makes the public 0 and 1 settings
    deterministic instead of relying on a vanishing denominator.
    """

    normalized_score = clamp01(score)
    normalized_pivot = clamp01(pivot)
    if normalized_score <= 0.0:
        return 0.0
    if normalized_score >= 1.0:
        return 1.0
    if normalized_pivot <= 0.0:
        return 1.0
    if normalized_pivot >= 1.0:
        return 0.0
    numerator = normalized_score * (1.0 - normalized_pivot)
    denominator = numerator + (1.0 - normalized_score) * normalized_pivot
    return clamp01(numerator / denominator) if denominator > 0.0 else 0.0


def noisy_or(values: Iterable[float]) -> float:
    remainder = 1.0
    for value in values:
        remainder *= 1.0 - clamp01(value)
    return clamp01(1.0 - remainder)


def text_embedding_relevance(
    dense_score: float,
    lexical_score: float,
    *,
    lexical_boost: float,
) -> float:
    """Return absolute text relevance without leaking RRF rank scale."""

    return noisy_or(
        (
            clamp01(dense_score),
            clamp01(lexical_boost) * clamp01(lexical_score),
        )
    )


def media_frequency_signals(
    query_tokens: Iterable[str],
    candidate_tokens: Iterable[str],
    document_frequencies: Mapping[str, int],
    corpus_size: int,
    *,
    coverage_exponent: float,
    common_floor: float,
    oov_penalty: float,
    rarity_exponent: float,
    collection_intent: bool,
) -> dict[str, Any]:
    """Score direct media terms by corpus frequency and query completeness.

    Document frequency is counted once per active media asset. Terms that are
    common across the current retrieval scope remain useful category anchors,
    while rare exact terms provide strong target-specific evidence. Query terms
    that are outside the media corpus reduce completeness without becoming
    artificial distinctive evidence.
    """

    query = list(dict.fromkeys(str(token) for token in query_tokens if str(token)))
    candidate = {str(token) for token in candidate_tokens if str(token)}
    size = max(0, int(corpus_size))
    normalized_common_floor = clamp01(common_floor)
    normalized_oov_penalty = max(0.0, min(2.0, float(oov_penalty)))
    normalized_rarity_exponent = max(0.25, min(4.0, float(rarity_exponent)))
    normalized_coverage_exponent = max(0.1, min(4.0, float(coverage_exponent)))

    def idf(document_frequency: int) -> float:
        if size <= 0:
            return 0.0
        frequency = max(0, min(size, int(document_frequency)))
        return math.log(
            1.0 + (size - frequency + 0.5) / (frequency + 0.5)
        )

    common_idf = idf(size) if size else 0.0
    unique_idf = idf(1) if size else 0.0
    rarity_span = max(0.0, unique_idf - common_idf)
    denominator = 0.0
    matched_weight = 0.0
    matched_rarity_weight = 0.0
    matched_tokens: list[str] = []
    rarity_values: list[float] = []
    collection_values: list[float] = []
    token_details: list[dict[str, Any]] = []
    selective_collection_frequencies = [
        int(document_frequencies.get(token, 0))
        for token in query
        if 0 < int(document_frequencies.get(token, 0)) < size
    ]
    selective_collection_frequency = min(
        selective_collection_frequencies,
        default=0,
    )
    query_has_selective_collection_terms = (
        collection_intent and selective_collection_frequency > 0
    )

    for token in query:
        document_frequency = max(
            0, min(size, int(document_frequencies.get(token, 0)))
        )
        in_corpus = document_frequency > 0
        token_idf = idf(document_frequency) if in_corpus else unique_idf
        length_weight = float(max(1, min(4, len(token))))
        weight = length_weight * token_idf
        if not in_corpus:
            weight *= normalized_oov_penalty
        matched = token in candidate
        rarity = 0.0
        if in_corpus and size > 1 and rarity_span > 0.0:
            rarity = clamp01(
                (token_idf - common_idf) / rarity_span
            ) ** normalized_rarity_exponent
        denominator += weight
        if matched:
            matched_tokens.append(token)
            matched_weight += weight
            matched_rarity_weight += weight * rarity
            rarity_values.append(rarity)
            if collection_intent:
                if query_has_selective_collection_terms:
                    # Explicit member terms ("GPT and Claude memes") define
                    # the set; common category terms must not add every sibling.
                    if document_frequency == selective_collection_frequency:
                        collection_values.append(1.0)
                elif document_frequency >= 2:
                    # A category shared by every asset remains valid membership
                    # evidence for an explicit "all four memes" request.
                    prevalence = document_frequency / size if size else 0.0
                    collection_values.append(
                        1.0
                        if document_frequency == size
                        else 4.0 * prevalence * (1.0 - prevalence)
                    )
        token_details.append(
            {
                "token": token,
                "matched": matched,
                "document_frequency": document_frequency,
                "idf": round(token_idf, 6),
                "rarity": round(rarity, 6),
                "oov": not in_corpus,
                "weight": round(weight, 6),
            }
        )

    completeness = clamp01(
        matched_weight / denominator if denominator > 0.0 else 0.0
    )
    informativeness = clamp01(
        matched_rarity_weight / matched_weight if matched_weight > 0.0 else 0.0
    )
    base_lexical = clamp01(
        completeness**normalized_coverage_exponent
        * (
            normalized_common_floor
            + (1.0 - normalized_common_floor) * informativeness
        )
    )
    distinctive_support = clamp01(
        completeness * noisy_or(rarity_values)
    )
    distinctive_membership_support = noisy_or(rarity_values)
    collection_membership_support = noisy_or(collection_values)
    collection_support = clamp01(
        completeness * collection_membership_support
    )
    candidate_lexical_score = noisy_or(
        (base_lexical, distinctive_support, collection_support)
    )
    return {
        "algorithm": MEDIA_FREQUENCY_ALGORITHM,
        "corpus_size": size,
        "matched_tokens": matched_tokens,
        "completeness": completeness,
        "informativeness": informativeness,
        "base_lexical_score": base_lexical,
        "distinctive_support": distinctive_support,
        "distinctive_membership_support": distinctive_membership_support,
        "collection_membership_support": collection_membership_support,
        "collection_support": collection_support,
        "candidate_lexical_score": candidate_lexical_score,
        "token_details": token_details,
    }


def unbound_media_confidence_factor(
    direct_score: float,
    best_direct_score: float,
    *,
    strongest_competitor_score: float = 0.0,
    collection_membership_support: float = 1.0,
    collection_intent: bool,
    competition_floor: float = 0.35,
    reliability_target: float = 0.25,
    specificity_exponent: float = 1.0,
    advantage_target: float = 0.04,
) -> dict[str, float]:
    """Adjust unbound-media confidence with direct support and soft competition.

    This function is intentionally limited to assets without an effective chunk
    binding. Bound media use their independent chunk-grounded confidence path.
    A clear single target keeps its lead while related non-winners retain a
    calibrated fraction of their own direct signal. Explicit collection queries
    use collection membership instead of single-target advantage.
    """

    normalized_direct = clamp01(direct_score)
    normalized_best = clamp01(best_direct_score)
    reliability = clamp01(
        normalized_direct / max(0.01, float(reliability_target))
    )
    relative = (
        1.0
        if normalized_best <= 0.0
        else clamp01(normalized_direct / normalized_best)
    )
    competitor = clamp01(strongest_competitor_score)
    advantage_margin = max(0.0, normalized_direct - competitor)
    advantage = clamp01(
        advantage_margin / max(0.01, float(advantage_target))
    )
    specificity = (
        clamp01(collection_membership_support)
        if collection_intent
        else (
            relative ** max(0.25, min(4.0, float(specificity_exponent)))
            * advantage
        )
    )
    winner_support = clamp01(reliability * specificity)
    normalized_floor = clamp01(competition_floor)
    applied_floor = 0.0 if collection_intent else normalized_floor
    confidence_factor = (
        winner_support
        if collection_intent
        else applied_floor + (1.0 - applied_floor) * winner_support
    )
    return {
        "direct_reliability": reliability,
        "relative_specificity": relative,
        "strongest_competitor_score": competitor,
        "advantage_margin": advantage_margin,
        "advantage_weight": advantage,
        "collection_membership_weight": (
            clamp01(collection_membership_support)
            if collection_intent
            else 0.0
        ),
        "specificity_weight": specificity,
        "winner_support": winner_support,
        "competition_floor": applied_floor,
        "confidence_factor": clamp01(confidence_factor),
    }


def media_only_confidence_factor(
    direct_score: float,
    best_direct_score: float,
    *,
    strongest_competitor_score: float = 0.0,
    collection_membership_support: float = 1.0,
    collection_intent: bool,
    competition_floor: float = 0.50,
) -> dict[str, float]:
    """Backward-compatible alias for the former mode-named helper."""

    return unbound_media_confidence_factor(
        direct_score,
        best_direct_score,
        strongest_competitor_score=strongest_competitor_score,
        collection_membership_support=collection_membership_support,
        collection_intent=collection_intent,
        competition_floor=competition_floor,
        # Preserve the compatibility helper's former implicit constants.
        reliability_target=0.35,
        specificity_exponent=1.25,
        advantage_target=0.06,
    )


def canonical_rerank_score(raw_score: float) -> float:
    """Map provider-specific rerank output to a finite probability-like score."""

    value = float(raw_score)
    if not math.isfinite(value):
        raise ValueError("rerank score must be finite")
    if 0.0 <= value <= 1.0:
        return value
    if value >= 0.0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def fuse_embedding_rerank(
    embedding_signal: float,
    raw_rerank_score: float,
    *,
    rank: int,
    candidate_count: int,
    fusion_weight: float,
    rank_bonus_weight: float,
    rank_reliability_exponent: float,
) -> dict[str, float]:
    """Fuse absolute Embedding relevance and Rerank evidence in log-odds space.

    A convex blend leaves almost all of an inflated Embedding score intact
    when the Rerank weight is small.  Log-odds pooling makes a confident
    Rerank rejection observable while still allowing a high Provider score to
    rescue a weak-but-valid initial candidate.  Both ordering and downstream
    evidence consume the same fused relevance across all four Rerank paths.
    """

    probability = canonical_rerank_score(raw_rerank_score)
    count = max(1, int(candidate_count))
    normalized_rank = max(1, min(count, int(rank)))
    rank_percentile = (
        1.0
        if count <= 1
        else 1.0 - (normalized_rank - 1) / (count - 1)
    )
    reliable_probability = probability ** max(
        0.5, float(rank_reliability_exponent)
    )
    rank_support = rank_percentile * reliable_probability
    # The provider probability remains the primary absolute relevance signal.
    # Reliability only gates the optional *rank bonus*: exponentiating the
    # probability itself would silently turn this setting into a second,
    # unconfigured score penalty and would diverge from the documented
    # cross-path fusion formula.
    rerank_signal = noisy_or(
        (probability, clamp01(rank_bonus_weight) * rank_support)
    )
    weight = clamp01(fusion_weight)
    embedding_relevance = clamp01(embedding_signal)
    epsilon = 1e-9

    def logit(value: float) -> float:
        bounded = min(1.0 - epsilon, max(epsilon, clamp01(value)))
        return math.log(bounded / (1.0 - bounded))

    mixed_log_odds = (
        (1.0 - weight) * logit(embedding_relevance)
        + weight * logit(rerank_signal)
    )
    if mixed_log_odds >= 0:
        exponent = math.exp(-mixed_log_odds)
        fused = 1.0 / (1.0 + exponent)
    else:
        exponent = math.exp(mixed_log_odds)
        fused = exponent / (1.0 + exponent)
    ordering_relevance = clamp01(fused)
    fused = ordering_relevance
    return {
        "rerank_probability": probability,
        "rerank_rank_percentile": rank_percentile,
        "rerank_reliable_probability": reliable_probability,
        "rerank_rank_support": rank_support,
        "rerank_signal": rerank_signal,
        "ordering_relevance": ordering_relevance,
        "fused_relevance": fused,
    }


def calibrated_semantic(similarity: float | None, floor: float) -> float:
    if similarity is None:
        return 0.0
    normalized_floor = max(0.0, min(0.99, float(floor)))
    return clamp01((float(similarity) - normalized_floor) / (1.0 - normalized_floor))


def aggregate_grounding(
    evidence: Iterable[float], *, corroboration_weight: float, limit: int
) -> float:
    ordered = sorted((clamp01(value) for value in evidence), reverse=True)
    if not ordered:
        return 0.0
    selected = ordered[: max(1, int(limit))]
    best = selected[0]
    corroboration = noisy_or(selected[1:])
    return clamp01(
        best
        + (1.0 - best)
        * clamp01(corroboration_weight)
        * corroboration
    )


def apply_evidence_threshold(
    grounding: float,
    evidence: Iterable[tuple[float, int]],
    *,
    threshold: float,
    limit: int,
    rank_decay_exponent: float,
    negative_reliability_exponent: float,
    reinforcement_weight: float,
    weakening_weight: float,
    negative_attenuation_floor: float = 0.0,
    tail_rank: int | None = None,
) -> dict[str, Any]:
    """Apply the universal relevance pivot to fixed-window bound evidence.

    Near-zero evidence is treated as little information rather than strong
    counter-evidence. The theoretical normalization constant depends only on
    the configured window, so missing or appended tail evidence cannot
    renormalize the influence of earlier ranks. ``reinforcement_weight`` and
    ``weakening_weight`` are retained as internal-call compatibility names; in
    the new algorithm they mean positive convex-blend weight and negative
    pressure weight respectively.
    """

    normalized_threshold = clamp01(threshold)
    normalized_limit = max(1, int(limit))
    rank_exponent = max(0.0, min(4.0, float(rank_decay_exponent)))
    reliability_exponent = max(
        1.0, min(4.0, float(negative_reliability_exponent))
    )
    normalization = sum(
        1.0 / (math.log2(rank + 1) ** rank_exponent)
        for rank in range(1, normalized_limit + 1)
    )
    selected = sorted(
        ((clamp01(score), max(1, int(rank))) for score, rank in evidence),
        key=lambda item: item[1],
    )[:normalized_limit]
    contributions: list[dict[str, Any]] = []
    positive_values: list[float] = []
    negative_sum = 0.0
    tail_negative_sum = 0.0
    for score, rank in selected:
        rank_weight = 1.0 / (
            math.log2(rank + 1) ** rank_exponent
        )
        pivoted_score = media_relevance_pivot(score, normalized_threshold)
        positive = 0.0
        negative = 0.0
        if score > normalized_threshold:
            # Measure signed distance above the pivot.  This keeps the
            # contribution continuous at ``score == pivot`` while preserving
            # the full 0..1 odds-calibration range.
            positive = rank_weight * max(
                2.0 * pivoted_score - 1.0, 0.0
            )
        elif score > 0.0 and normalized_threshold > 0.0:
            # Reliability is tied to the evidence itself, not to a changing
            # ``score / pivot`` ratio.  As the public pivot is lowered the
            # calibrated score can therefore only reduce this pressure.
            negative = rank_weight * (
                score**reliability_exponent
            ) * max(1.0 - 2.0 * pivoted_score, 0.0)
        positive = clamp01(positive)
        negative = max(0.0, float(negative))
        positive_values.append(positive)
        negative_sum += negative
        if tail_rank is not None and rank > int(tail_rank):
            tail_negative_sum += negative
        contributions.append(
            {
                "rank": rank,
                "evidence_score": round(score, 6),
                "pivot_calibrated_evidence_score": round(pivoted_score, 6),
                "threshold_rank_weight": round(rank_weight, 6),
                "threshold_positive_contribution": round(positive, 6),
                "threshold_negative_contribution": round(negative, 6),
                "threshold_effect": (
                    "reinforce"
                    if positive > 0.0
                    else "weaken"
                    if negative > 0.0
                    else "neutral"
                ),
            }
        )

    positive_support = noisy_or(positive_values)
    negative_pressure = clamp01(
        negative_sum / normalization if normalization > 0.0 else 0.0
    )
    tail_negative_pressure = clamp01(
        tail_negative_sum / normalization if normalization > 0.0 else 0.0
    )
    base = clamp01(grounding)
    positive_blend = clamp01(reinforcement_weight)
    reinforced = clamp01(
        (1.0 - positive_blend) * base
        + positive_blend * positive_support
    )
    negative_floor = clamp01(negative_attenuation_floor)
    negative_factor = max(
        negative_floor,
        1.0 - clamp01(weakening_weight) * negative_pressure,
    )
    adjusted = clamp01(
        reinforced * negative_factor
    )
    return {
        "grounding": base,
        "reinforced_grounding": reinforced,
        "adjusted_grounding": adjusted,
        "grounding_delta": adjusted - base,
        "positive_support": positive_support,
        "negative_pressure": negative_pressure,
        "negative_attenuation_factor": negative_factor,
        "positive_blend_weight": positive_blend,
        "tail_negative_pressure": tail_negative_pressure,
        "normalization_constant": normalization,
        "evidence_limit": normalized_limit,
        "evidence_count": len(selected),
        "contributions": contributions,
    }


def score_media_confidence(
    *,
    raw_direct_relevance: float,
    grounding: float,
    evidence: Iterable[tuple[float, int]],
    bound_media: bool,
    relevance_pivot: float,
    direct_signal_enabled: bool,
    ambiguous_collection: bool,
    subject_anchor_met: bool,
    media_format_compatible: bool,
    content_anchor_met: bool,
    evidence_limit: int,
    rank_decay_exponent: float,
    negative_reliability_exponent: float,
    positive_blend: float,
    negative_weight: float,
    negative_attenuation_floor: float,
    format_mismatch_factor: float,
    content_mismatch_factor: float,
    tail_rank: int | None = None,
) -> dict[str, Any]:
    """Pure structure-aware media confidence scoring shared by both baselines."""

    normalized_direct = clamp01(raw_direct_relevance)
    normalized_grounding = clamp01(grounding) if bound_media else 0.0
    threshold_result = apply_evidence_threshold(
        normalized_grounding,
        evidence if bound_media else (),
        threshold=relevance_pivot,
        limit=evidence_limit,
        rank_decay_exponent=rank_decay_exponent,
        negative_reliability_exponent=negative_reliability_exponent,
        reinforcement_weight=positive_blend,
        weakening_weight=negative_weight,
        negative_attenuation_floor=negative_attenuation_floor,
        tail_rank=tail_rank,
    )
    policy_enabled = bool(direct_signal_enabled) and not bool(
        ambiguous_collection
    )
    format_factor = (
        1.0
        if media_format_compatible
        else clamp01(format_mismatch_factor)
    )
    content_factor = (
        1.0
        if subject_anchor_met and content_anchor_met
        else clamp01(content_mismatch_factor)
    )
    structural_factor = (
        clamp01(format_factor * content_factor) if policy_enabled else 0.0
    )
    pivoted_direct = media_relevance_pivot(
        normalized_direct, relevance_pivot
    )
    raw_combined = (
        noisy_or((normalized_grounding, normalized_direct))
        if bound_media
        else normalized_direct
    )
    pivoted_combined = (
        noisy_or(
            (
                float(threshold_result["adjusted_grounding"]),
                pivoted_direct,
            )
        )
        if bound_media
        else pivoted_direct
    )
    return {
        "raw_direct_relevance": normalized_direct,
        "pivoted_direct_relevance": pivoted_direct,
        "raw_combined_confidence": raw_combined,
        "pivoted_combined_confidence": pivoted_combined,
        "structural_attenuation_factor": structural_factor,
        "format_attenuation_factor": (
            format_factor if policy_enabled else 0.0
        ),
        "content_attenuation_factor": (
            content_factor if policy_enabled else 0.0
        ),
        "policy_gate_open": policy_enabled,
        "output_confidence": clamp01(
            pivoted_combined * structural_factor
        ),
        "pre_pivot_output_confidence": clamp01(
            raw_combined * structural_factor
        ),
        "threshold_result": threshold_result,
    }
