from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Any


ATOM_TTL_CONFIG: dict[str, tuple[float, str]] = {
    "episodic": (7.0, "exponential"),
    "planned": (2.0, "step"),
    "factual": (180.0, "exponential"),
    "relational": (90.0, "linear"),
    "preference": (60.0, "exponential"),
    "unknown": (30.0, "exponential"),
}

_TIME_INDICATORS = re.compile(
    r"明天|后天|大后天|昨天|前天|今天|"
    r"(?:上周|本周|下下周|下周)?周[一二三四五六日天]|"
    r"上周|本周|下下周|下周|"
    r"下个?月|上个?月|明年|后年|去年|前年|"
    r"\d{1,2}月\d{1,2}[日号]|\d{4}年\d{1,2}月|"
    r"上午|下午|晚上|凌晨|早上|中午|傍晚|"
    r"\d{1,2}[点时：:]\d{1,2}"
)
_ACTION_VERBS = re.compile(
    r"开会|讨论|参加|组织|安排|举办|进行|执行|完成|提交|发送|发布|"
    r"去|来|到|做|要|准备|计划|打算"
)
_STATIVE_PATTERNS = re.compile(r"是|有|属于|等于|代表|意味|包含|包括|位于")
_RELATION_PATTERNS = re.compile(
    r"同事|朋友|同学|家人|亲戚|队友|搭档|伙伴|老板|上司|下属|"
    r"合作|合伙|夫妻|情侣|邻居|室友|老乡"
)
_PREFERENCE_PATTERNS = re.compile(
    r"喜欢|讨厌|爱|不爱|偏好|最爱|不喜欢|热衷于|沉迷|"
    r"爱吃|爱喝|喜欢喝|喜欢去|讨厌吃|讨厌去"
)
_WEEKDAY_INDEX = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
}


def _bounded_importance(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = value
    elif value:
        items = [value]
    else:
        items = []
    return [str(item).strip() for item in items if str(item).strip()]


def _parse_weekday_time(text: str, now: float) -> float | None:
    match = re.search(r"(上周|本周|下下周|下周)?周([一二三四五六日天])", text)
    if not match:
        return None

    prefix = match.group(1) or ""
    target_weekday = _WEEKDAY_INDEX[match.group(2)]
    now_dt = datetime.fromtimestamp(now)

    if prefix == "上周":
        days_delta = target_weekday - now_dt.weekday() - 7
    elif prefix == "本周":
        days_delta = target_weekday - now_dt.weekday()
    elif prefix == "下周":
        days_delta = target_weekday - now_dt.weekday() + 7
    elif prefix == "下下周":
        days_delta = target_weekday - now_dt.weekday() + 14
    else:
        days_delta = (target_weekday - now_dt.weekday()) % 7

    return (now_dt + timedelta(days=days_delta)).timestamp()


def parse_atom_event_time(text: str, now: float | None = None) -> float | None:
    if now is None:
        now = time.time()
    day_sec = 86400.0
    mapping: dict[str, float] = {
        "前天": -2 * day_sec,
        "昨天": -1 * day_sec,
        "今天": 0.0,
        "明天": day_sec,
        "后天": 2 * day_sec,
        "大后天": 3 * day_sec,
    }
    for word, offset in mapping.items():
        if word in text:
            return now + offset

    weekday_time = _parse_weekday_time(text, now)
    if weekday_time is not None:
        return weekday_time

    week_mapping: dict[str, float] = {
        "上周": -7 * day_sec,
        "本周": 0.0,
        "下下周": 14 * day_sec,
        "下周": 7 * day_sec,
    }
    for word, offset in week_mapping.items():
        if word in text:
            return now + offset

    match = re.search(r"(\d{1,2})月(\d{1,2})[日号]", text)
    if match:
        month, day = int(match.group(1)), int(match.group(2))
        now_dt = datetime.fromtimestamp(now)
        target = now_dt.replace(
            month=month, day=day, hour=0, minute=0, second=0, microsecond=0
        )
        if target < now_dt:
            target = target.replace(year=now_dt.year + 1)
        return target.timestamp()

    return None


def classify_atom_text(text: str) -> tuple[str, float, float | None]:
    has_time = bool(_TIME_INDICATORS.search(text))
    has_action = bool(_ACTION_VERBS.search(text))
    has_stative = bool(_STATIVE_PATTERNS.search(text))
    has_relation = bool(_RELATION_PATTERNS.search(text))
    has_preference = bool(_PREFERENCE_PATTERNS.search(text))
    event_time = parse_atom_event_time(text) if has_time else None

    if has_time and has_action:
        return "planned", 0.85, event_time
    if has_preference:
        return "preference", 0.82, None
    if has_relation:
        return "relational", 0.80, None
    if has_stative:
        return "factual", 0.78, None
    if has_action:
        return "episodic", 0.75, None
    return "unknown", 0.60, None


def compute_atom_ttl(
    atom_type: Any,
    importance: Any = 0.5,
    reinforcement_count: int = 0,
    event_time: Any = None,
) -> tuple[float, str]:
    kind = str(atom_type or "unknown").lower()
    base_ttl, decay_type = ATOM_TTL_CONFIG.get(kind, ATOM_TTL_CONFIG["unknown"])
    parsed_event_time: float | None = None
    try:
        if event_time is not None:
            parsed_event_time = float(event_time)
    except (TypeError, ValueError):
        parsed_event_time = None

    if kind == "planned" and parsed_event_time is not None:
        days_until_event = max(0.0, (parsed_event_time - time.time()) / 86400.0)
        base_ttl = days_until_event + base_ttl

    try:
        reinforcement = max(0, int(reinforcement_count))
    except (TypeError, ValueError):
        reinforcement = 0
    importance_factor = 0.5 + _bounded_importance(importance)
    reinforcement_factor = 1.0 + min(0.5, reinforcement * 0.1)
    return max(1.0, base_ttl * importance_factor * reinforcement_factor), decay_type


def classify_memory_atoms(
    *,
    key_facts: Any,
    topics: Any = None,
    participants: Any = None,
    parent_importance: Any = 0.5,
    session_id: str | None = None,
    persona_id: str | None = None,
    limit: int | None = 8,
) -> list[dict[str, Any]]:
    entities = [*_as_text_list(topics), *_as_text_list(participants)]
    facts = _as_text_list(key_facts)
    if limit is not None:
        facts = facts[:limit]
    importance = _bounded_importance(parent_importance)
    atoms: list[dict[str, Any]] = []
    for fact in facts:
        atom_type, confidence, event_time = classify_atom_text(fact)
        atoms.append(
            {
                "atom_type": atom_type,
                "content": fact,
                "entities": entities,
                "importance": importance,
                "confidence": confidence,
                "event_time": event_time,
                "session_id": session_id,
                "persona_id": persona_id,
                "metadata": {"source": "summary_key_fact"},
            }
        )
    return atoms
