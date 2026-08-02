from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Protocol
import unicodedata

import jieba
from markdown_it import MarkdownIt

from .visual_intent_policy import (
    DEFAULT_VISUAL_INTENT_POLICY,
    VISUAL_INTENT_POLICY_KEYS,
    normalize_visual_intent_policy,
    visual_intent_policy_fingerprint,
)


CHUNKER_ID = "markdown_hierarchy_v1"
DEFAULT_CHUNK_TARGET = 1200
DEFAULT_CHUNK_OVERLAP = 150

_MARKDOWN = MarkdownIt("commonmark", {"html": False}).enable("table")
_ROOT_BLOCK_TYPES = {
    "paragraph_open",
    "bullet_list_open",
    "ordered_list_open",
    "blockquote_open",
    "fence",
    "code_block",
    "hr",
    "table_open",
    "html_block",
}
_SEMANTIC_BOUNDARIES = (
    re.compile(r"(?<=\n\n)"),
    re.compile(r"(?<=\n)"),
    re.compile(r"(?<=[。！？!?；;])"),
    re.compile(r"(?<=\.)(?=\s|$)"),
    re.compile(r"(?<=[，、,:：])"),
    re.compile(r"(?<=\s)"),
)


@dataclass(frozen=True)
class _SourceBlock:
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class _Section:
    heading_path: tuple[str, ...]
    blocks: tuple[_SourceBlock, ...]


@dataclass(frozen=True)
class _Fragment:
    text: str
    char_start: int
    char_end: int


def normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def lexical_text(value: str) -> str:
    return " ".join(token.strip() for token in jieba.cut(value) if token.strip())


def fts_query_text(value: str) -> str:
    """Build a recall-oriented FTS query from meaningful query terms.

    FTS5 separates adjacent quoted terms with an implicit AND.  Natural
    questions such as ``萌依是谁`` therefore used to miss the exact entity
    whenever Jieba attached the copula to an unknown name.  Search tokens are
    normalized independently from media-intent tokens and joined with OR so
    lexical recall behaves like the LivingMemory v8 baseline.
    """

    return " OR ".join(
        f'"{token.replace(chr(34), chr(34) * 2)}"'
        for token in text_query_tokens(value)
    )


_MEDIA_STOP_WORDS = frozenset(
    {
        "的", "了", "吗", "呢", "吧", "啊", "呀", "一下", "一个", "一种",
        "一张", "来张", "发张", "全部", "所有", "四种", "多张", "几张",
        "一组", "合集", "一起", "都", "包都",
        "我", "你", "您", "他", "她", "它", "我们", "你们", "他们",
        "请", "帮", "帮我", "给我", "说说", "描述", "介绍", "展示", "看看",
        "what", "is", "are", "the", "a", "an", "your", "you", "my", "me",
        "please", "show", "describe", "tell", "about", "of", "to", "and",
        "all", "both", "every", "multiple", "several", "collection", "set",
        "как", "что", "это", "ты", "вы", "твой", "ваш", "мне", "покажи",
        "опиши", "расскажи", "про", "и", "все", "оба", "несколько",
        "набор", "коллекция",
    }
)
_TOKEN_PATTERN = re.compile(r"[\w\u3400-\u9fff]+", re.UNICODE)

_TEXT_QUERY_STOP_WORDS = frozenset(
    {
        "的", "了", "是", "谁", "吗", "呢", "吧", "啊", "呀", "么",
        "什么", "怎么", "怎样", "如何", "为何", "为什么", "请问", "问一下",
        "请", "帮我", "告诉我", "说说", "介绍", "描述",
        "what", "who", "is", "are", "the", "a", "an", "of", "to",
        "please", "tell", "me", "about", "how", "why",
        "кто", "что", "это", "как", "почему", "расскажи", "пожалуйста",
    }
)
_TEXT_QUERY_TRAILING_STOP_WORDS = (
    "为什么", "什么", "怎么", "怎样", "如何", "是谁", "请问", "的", "是", "吗",
    "呢", "吧", "啊", "呀",
)


def text_query_tokens(value: str) -> list[str]:
    """Return stable multilingual lexical tokens for text retrieval.

    Jieba does not know every proper name.  In particular, it may segment an
    unseen name plus ``是`` as one token.  Removing a trailing interrogative
    particle recovers the entity without maintaining a corpus-specific name
    dictionary.  Single-character content words are retained; only explicit
    stop words are removed.
    """

    normalized = normalize_media_text(value)
    seen: set[str] = set()
    result: list[str] = []
    for raw in jieba.cut_for_search(normalized):
        token = raw.strip()
        if not token or not _TOKEN_PATTERN.fullmatch(token):
            continue
        for suffix in _TEXT_QUERY_TRAILING_STOP_WORDS:
            if token.endswith(suffix):
                prefix = token[: -len(suffix)].strip()
                if len(prefix) >= 2 and _TOKEN_PATTERN.fullmatch(prefix):
                    token = prefix
                break
        if token in _TEXT_QUERY_STOP_WORDS or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result

_VISUAL_SUBJECT_ANCHOR_PATTERN = re.compile(
    r"斩魄刀|太刀|狱牙刀|武器|刀|剑|枪|弓|弩|配装|装备|铠甲|盔甲|始解|卍解"
)
_VISUAL_NEGATIVE_PATTERNS = (
    re.compile(r"谁画|谁绘|谁设计|作者|画师|来源|出处|什么意思|这个词|上传失败|怎么上传|战斗策略"),
    re.compile(
        r"\b(who (drew|painted|designed)|artist|author|source|meaning|means|"
        r"upload (failed|error)|how to upload|battle strateg)\w*\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(кто (нарисовал|создал)|автор|источник|значени|ошибк.*загруз|"
        r"как загрузить|стратег.*бо)\w*\b",
        re.IGNORECASE,
    ),
)

VISUAL_INTENT_DETECTOR_VERSION = "lexicon_reference_rules_v1"

PROTECTED_VISUAL_BLOCKER_TERMS = (
    "谁画的", "谁绘制的", "画师", "作者", "来源", "出处", "上传失败",
    "怎么上传", "战斗策略", "who drew", "artist", "author", "source",
    "upload failed", "how to upload", "battle strategy", "кто нарисовал",
    "автор", "источник", "ошибка загрузки", "стратегия боя",
)
_MEDIA_COLLECTION_PATTERNS = (
    re.compile(r"全部|所有|四种|多张|几张|一组|合集|一起|都(?:给我|发|展示|看看|列出)?"),
    re.compile(
        r"(?:立绘|头像|外貌|外观|场景|插图|图片|照片|表情包|梗图)"
        r".*(?:以及|和|与|、)"
        r".*(?:立绘|头像|外貌|外观|场景|插图|图片|照片|表情包|梗图)"
    ),
    re.compile(
        r"\b(all|both|every|multiple|several|collection|set of)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(все|оба|несколько|набор|коллекц)\w*\b", re.IGNORECASE),
)
_MEDIA_FORMAT_PATTERNS = {
    "meme": re.compile(
        r"表情(?:包)?|梗图|\bmemes?\b|\bмем\w*\b",
        re.IGNORECASE,
    ),
    "portrait": re.compile(
        r"立绘|\b(?:portrait|character\s+art|full[-\s]?body\s+portrait)\b|"
        r"\b(?:ростов|персонажн).*(?:портрет|арт)\w*\b",
        re.IGNORECASE,
    ),
    "avatar": re.compile(
        r"头像|脸部|面部|面容|脸|头部特写|\b(?:avatar|headshot|face)\b|"
        r"\b(?:аватар|лиц|портрет\s+лица)\w*\b",
        re.IGNORECASE,
    ),
    "scene": re.compile(
        r"场景|竞技场|\bscene\b|\bсцен\w*\b",
        re.IGNORECASE,
    ),
    "illustration": re.compile(
        r"插图|插画|\billustration\b|\bиллюстрац\w*\b",
        re.IGNORECASE,
    ),
    "cover": re.compile(
        r"封面|\bcover\b|\bобложк\w*\b",
        re.IGNORECASE,
    ),
    "generic_image": re.compile(
        r"图片|照片|图像|图|\b(?:image|picture|photo)\b|"
        r"\b(?:изображен|картин|фото)\w*\b",
        re.IGNORECASE,
    ),
}
_MEDIA_FORMAT_QUERY_TOKENS = frozenset(
    {
        "表情",
        "表情包",
        "梗图",
        "立绘",
        "头像",
        "脸部",
        "面部",
        "面容",
        "头部",
        "特写",
        "场景",
        "竞技场",
        "插图",
        "插画",
        "封面",
        "图片",
        "照片",
        "图像",
        "meme",
        "memes",
        "portrait",
        "avatar",
        "headshot",
        "face",
        "scene",
        "illustration",
        "cover",
        "image",
        "picture",
        "photo",
    }
)


def normalize_media_text(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold().strip()


@dataclass(frozen=True)
class _IntentMatch:
    category: str
    term: str
    start: int
    end: int


class VisualIntentDetector(Protocol):
    """Stable extension point for future rule or lightweight ML detectors."""

    version: str
    policy_fingerprint: str

    def analyze(self, value: str, *, gate_enabled: bool = True) -> dict[str, object]:
        ...


def _literal_term_matches(
    value: str,
    category: str,
    terms: list[str],
) -> list[_IntentMatch]:
    matches: list[_IntentMatch] = []
    for term in sorted(terms, key=lambda item: (-len(item), item)):
        if re.search(r"[\u3400-\u9fff]", term):
            offset = 0
            while True:
                index = value.find(term, offset)
                if index < 0:
                    break
                matches.append(_IntentMatch(category, term, index, index + len(term)))
                offset = index + max(1, len(term))
        else:
            pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
            matches.extend(
                _IntentMatch(category, term, match.start(), match.end())
                for match in pattern.finditer(value)
            )
    # Keep the longest literal when phrases in one category overlap.  This is
    # what protects `立绘` before Jieba sees the adjacent generation verb.
    selected: list[_IntentMatch] = []
    for match in sorted(matches, key=lambda item: (item.start, -len(item.term), item.term)):
        if any(
            match.start < existing.end and existing.start < match.end
            for existing in selected
        ):
            continue
        selected.append(match)
    return selected


def _unique_terms(matches: list[_IntentMatch]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for match in sorted(matches, key=lambda item: (item.start, item.end)):
        if match.term not in seen:
            seen.add(match.term)
            result.append(match.term)
    return result


def _clean_reference_span(value: str) -> str:
    cleaned = value.strip(" \t\r\n,，。.!！?？;；:：")
    cleaned = re.sub(
        r"^(?:先|再|然后|接着|请|麻烦|给我|让我|看看|看一下|展示|"
        r"show\s+|display\s+|let\s+me\s+see\s+|покажи\s+)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned.strip(" \t\r\n,，。.!！?？;；:：")


def _media_query_prefix(value: str) -> str:
    if re.search(r"[а-яё]", value, re.IGNORECASE):
        return "покажи "
    if re.search(r"[a-z]", value, re.IGNORECASE) and not re.search(
        r"[\u3400-\u9fff]", value
    ):
        return "show "
    return "看看"


class LexiconReferenceVisualIntentDetector:
    version = VISUAL_INTENT_DETECTOR_VERSION

    def __init__(self, policy: object = None) -> None:
        self.policy = normalize_visual_intent_policy(policy)
        self.policy_fingerprint = visual_intent_policy_fingerprint(self.policy)

    def _matches(self, value: str) -> dict[str, list[_IntentMatch]]:
        return {
            key: _literal_term_matches(value, key, self.policy[key])
            for key in VISUAL_INTENT_POLICY_KEYS
        }

    @staticmethod
    def _reference_projection(
        value: str,
        matches: dict[str, list[_IntentMatch]],
    ) -> tuple[str, str, str] | None:
        generations = sorted(
            matches["generation_action_terms"], key=lambda item: item.start
        )
        objects = sorted(matches["visual_object_terms"], key=lambda item: item.start)
        if not generations or not objects:
            return None
        generation = generations[0]
        connectors = sorted(
            matches["reference_connector_terms"], key=lambda item: item.start
        )
        lookups = sorted(matches["lookup_action_terms"], key=lambda item: item.start)
        before_objects = [item for item in objects if item.end <= generation.start]
        after_objects = [item for item in objects if item.start >= generation.end]
        reference = ""
        generation_target = value[generation.end :].strip()

        # `根据/参考/用 X 画 Y` and equivalent source-before-action forms.
        preceding_connectors = [item for item in connectors if item.end <= generation.start]
        if preceding_connectors and before_objects:
            connector = preceding_connectors[-1]
            reference = value[connector.end : generation.start]
        else:
            # `把 X 画成 Y` is intentionally a protected deterministic form,
            # not another editable category.
            ba_index = max(value.rfind("把", 0, generation.start), -1)
            if ba_index >= 0 and any(
                ba_index < item.start < generation.start for item in before_objects
            ):
                reference = value[ba_index + 1 : generation.start]

        if not reference and before_objects:
            preceding_lookups = [item for item in lookups if item.end <= generation.start]
            if preceding_lookups:
                lookup = preceding_lookups[-1]
                delimiter = re.search(r"[，,。.!！?？;；]", value[lookup.end : generation.start])
                end = lookup.end + delimiter.start() if delimiter else generation.start
                reference = value[lookup.end:end]

        # `draw Y based on X` / `нарисуй Y на основе X`.
        if not reference:
            following_connectors = [item for item in connectors if item.start >= generation.end]
            if following_connectors and after_objects:
                connector = following_connectors[0]
                if any(item.start >= connector.end for item in after_objects):
                    reference = value[connector.end :]
                    generation_target = value[generation.end : connector.start].strip()

        # A possessive media object after the verb can itself be the reference
        # (`重绘你的立绘`), while generic generated targets remain excluded.
        if not reference and after_objects:
            first_object = after_objects[0]
            between = value[generation.end : first_object.start]
            if re.search(r"你的|您的|这张|该|your\s+|тво[йяею]\s+", between):
                end_match = re.search(r"[，,。.!！?？;；]", value[first_object.end :])
                end = (
                    first_object.end + end_match.start()
                    if end_match
                    else first_object.end
                )
                reference = value[generation.end:end]

        reference = _clean_reference_span(reference)
        if not reference:
            return None
        # A valid source must still contain one of the configured media-object
        # literals.  This prevents a loose connector from selecting prose.
        if not any(
            _literal_term_matches(reference, "visual_object_terms", [item.term])
            for item in objects
        ):
            return None
        media_query = f"{_media_query_prefix(value)}{reference}".strip()
        return reference, generation_target, media_query

    def analyze(self, value: str, *, gate_enabled: bool = True) -> dict[str, object]:
        normalized = normalize_media_text(value)
        matches = self._matches(normalized)
        blocked = [
            match.group(0)
            for pattern in _VISUAL_NEGATIVE_PATTERNS
            if (match := pattern.search(normalized)) is not None
        ]
        subject_anchors = [
            match.group(0)
            for match in _VISUAL_SUBJECT_ANCHOR_PATTERN.finditer(normalized)
        ]
        projection = None if blocked else self._reference_projection(normalized, matches)
        generation_present = bool(matches["generation_action_terms"])
        visual_objects = matches["visual_object_terms"]
        lookup_present = bool(matches["lookup_action_terms"])

        if blocked:
            intent_kind = "blocked"
            detected = False
        elif projection is not None:
            intent_kind = "reference_generation"
            detected = True
        elif generation_present:
            # A generation target is not automatically a request for an
            # existing reference asset.
            intent_kind = "generation_without_reference"
            detected = False
        elif visual_objects or (lookup_present and subject_anchors):
            intent_kind = "lookup"
            detected = True
        else:
            intent_kind = "none"
            detected = False
        if not gate_enabled and intent_kind not in {
            "blocked",
            "generation_without_reference",
        }:
            detected = True

        reference_span = projection[0] if projection else ""
        generation_span = projection[1] if projection else ""
        media_query = projection[2] if projection else normalized
        matched_categories = {
            key: _unique_terms(category_matches)
            for key, category_matches in matches.items()
        }
        matched_terms: list[str] = []
        for key in VISUAL_INTENT_POLICY_KEYS:
            for term in matched_categories[key]:
                if term not in matched_terms:
                    matched_terms.append(term)
        return {
            "detected": bool(detected),
            "gate_enabled": bool(gate_enabled),
            "intent_kind": intent_kind,
            "matched_terms": matched_terms,
            "matched_categories": matched_categories,
            "blocked_terms": blocked,
            "subject_anchor_required": bool(subject_anchors),
            "subject_anchor_terms": subject_anchors,
            "protected_terms": matched_categories["visual_object_terms"],
            "original_query": normalized,
            "media_query": media_query,
            "reference_span": reference_span,
            "generation_span": generation_span,
            "ignored_output_terms": [generation_span] if generation_span else [],
            "projection_applied": bool(projection and media_query != normalized),
            "policy_fingerprint": self.policy_fingerprint,
            "detector_version": self.version,
        }


MAX_MEDIA_DESCRIPTIONS = 20
MAX_MEDIA_DESCRIPTION_LENGTH = 2000


def normalize_media_description(value: str) -> str:
    """Normalize one user-visible description for identity comparisons."""

    return re.sub(r"\s+", " ", normalize_media_text(value))


def media_description_list(
    values: object,
    *,
    fallback: str = "",
    allow_empty_input: bool = False,
) -> list[str]:
    """Validate an ordered, per-asset media-description set."""

    if values is None:
        raw_values: list[object] = []
    elif isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, (list, tuple)):
        raw_values = list(values)
    else:
        raise ValueError("media descriptions must be a list of strings")
    if not raw_values and fallback:
        raw_values = [fallback]
    if not raw_values and allow_empty_input:
        return []
    if not raw_values:
        raise ValueError("at least one media description is required")
    if len(raw_values) > MAX_MEDIA_DESCRIPTIONS:
        raise ValueError(
            f"an image supports at most {MAX_MEDIA_DESCRIPTIONS} media descriptions"
        )
    result: list[str] = []
    identities: set[str] = set()
    for value in raw_values:
        if not isinstance(value, str):
            raise ValueError("media descriptions must be strings")
        display = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()
        if not display:
            raise ValueError("media descriptions cannot be empty")
        if len(display) > MAX_MEDIA_DESCRIPTION_LENGTH:
            raise ValueError(
                f"media descriptions cannot exceed {MAX_MEDIA_DESCRIPTION_LENGTH} characters"
            )
        identity = normalize_media_description(display)
        if identity in identities:
            raise ValueError("duplicate media descriptions are not allowed")
        identities.add(identity)
        result.append(display)
    return result


def media_tokens(
    value: str,
    *,
    protected_terms: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    normalized = normalize_media_text(value)
    seen: set[str] = set()
    result: list[str] = []
    raw_tokens = list(jieba.cut_for_search(normalized))
    # Lexicon matching happens before Jieba so `立绘画` cannot collapse to the
    # unrelated token `绘画`.  Protected literals are added as exact tokens.
    raw_tokens.extend(normalize_media_text(item) for item in protected_terms or ())
    for raw in raw_tokens:
        token = raw.strip()
        if (
            not token
            or token in _MEDIA_STOP_WORDS
            or not _TOKEN_PATTERN.fullmatch(token)
            or (len(token) == 1 and not token.isascii())
        ):
            continue
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result


def visual_intent(
    value: str,
    *,
    gate_enabled: bool = True,
    policy: object = None,
    detector: VisualIntentDetector | None = None,
) -> dict[str, object]:
    active_detector = detector or LexiconReferenceVisualIntentDetector(policy)
    return active_detector.analyze(value, gate_enabled=gate_enabled)


def media_collection_intent(value: str) -> bool:
    """Return whether a media query explicitly asks for several assets.

    Single-target media search uses candidate competition to suppress near-
    duplicate false positives. Explicit collection requests must bypass that
    relative suppression so several independently eligible assets can remain.
    """

    normalized = normalize_media_text(value)
    return any(pattern.search(normalized) for pattern in _MEDIA_COLLECTION_PATTERNS)


def media_format_groups(value: str) -> set[str]:
    """Return explicit media-form groups named by a query or description."""

    normalized = normalize_media_text(value)
    return {
        group
        for group, pattern in _MEDIA_FORMAT_PATTERNS.items()
        if pattern.search(normalized)
    }


def media_subject_tokens(tokens: list[str]) -> list[str]:
    """Return content-identifying query tokens, excluding media form words."""

    return [
        token
        for token in tokens
        if normalize_media_text(token) not in _MEDIA_FORMAT_QUERY_TOKENS
    ]


def weighted_token_coverage(
    query_tokens: list[str], candidate_tokens: list[str]
) -> tuple[float, list[str]]:
    if not query_tokens:
        return 0.0, []
    candidate = set(candidate_tokens)
    matched = [token for token in query_tokens if token in candidate]
    weights = {token: float(max(1, min(4, len(token)))) for token in query_tokens}
    denominator = sum(weights.values())
    coverage = sum(weights[token] for token in matched) / denominator
    return max(0.0, min(1.0, coverage)), matched


def _line_offsets(value: str) -> list[int]:
    offsets = [0]
    for match in re.finditer("\n", value):
        offsets.append(match.end())
    offsets.append(len(value))
    return offsets


def _source_block(
    value: str, lines: list[str], offsets: list[int], start_line: int, end_line: int
) -> _SourceBlock | None:
    raw = "\n".join(lines[start_line:end_line]).strip()
    if not raw:
        return None
    absolute_start = offsets[start_line]
    relative_start = value.find(raw, absolute_start)
    if relative_start < 0:
        relative_start = absolute_start
    return _SourceBlock(raw, relative_start, relative_start + len(raw))


def _markdown_sections(value: str) -> list[_Section]:
    lines = value.splitlines()
    offsets = _line_offsets(value)
    headings: list[tuple[int, str]] = []
    current_path: tuple[str, ...] = ()
    current_blocks: list[_SourceBlock] = []
    sections: list[_Section] = []

    def flush() -> None:
        nonlocal current_blocks
        if current_blocks:
            sections.append(_Section(current_path, tuple(current_blocks)))
            current_blocks = []

    tokens = _MARKDOWN.parse(value)
    for index, token in enumerate(tokens):
        if token.type == "heading_open" and token.level == 0 and token.map:
            flush()
            level = int(token.tag[1:])
            title = ""
            if index + 1 < len(tokens) and tokens[index + 1].type == "inline":
                title = tokens[index + 1].content.strip()
            heading = f"{'#' * level} {title}".rstrip()
            headings = [item for item in headings if item[0] < level]
            headings.append((level, heading))
            current_path = tuple(item[1] for item in headings)
            continue
        if (
            token.level == 0
            and token.map
            and token.type in _ROOT_BLOCK_TYPES
            and token.nesting >= 0
        ):
            block = _source_block(
                value, lines, offsets, int(token.map[0]), int(token.map[1])
            )
            if block is not None:
                current_blocks.append(block)
    flush()
    return sections


def _plain_sections(value: str) -> list[_Section]:
    blocks: list[_SourceBlock] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", value, re.DOTALL):
        text = match.group(0).strip()
        if text:
            start = value.find(text, match.start())
            blocks.append(_SourceBlock(text, start, start + len(text)))
    return [_Section((), tuple(blocks))] if blocks else []


def _split_semantic_units(
    value: str, maximum: int, boundary_index: int = 0
) -> list[str]:
    text = value.strip()
    if not text:
        return []
    if len(text) <= maximum:
        return [text]
    if boundary_index >= len(_SEMANTIC_BOUNDARIES):
        return [text[index : index + maximum] for index in range(0, len(text), maximum)]
    pieces = [
        piece.strip()
        for piece in _SEMANTIC_BOUNDARIES[boundary_index].split(text)
        if piece.strip()
    ]
    if len(pieces) <= 1:
        return _split_semantic_units(text, maximum, boundary_index + 1)
    result: list[str] = []
    for piece in pieces:
        result.extend(_split_semantic_units(piece, maximum, boundary_index + 1))
    return result


def _block_fragments(block: _SourceBlock, maximum: int) -> list[_Fragment]:
    pieces = _split_semantic_units(block.text, maximum)
    fragments: list[_Fragment] = []
    cursor = 0
    for piece in pieces:
        relative = block.text.find(piece, cursor)
        if relative < 0:
            relative = block.text.find(piece)
        if relative < 0:
            relative = cursor
        start = block.char_start + relative
        fragments.append(_Fragment(piece, start, start + len(piece)))
        cursor = max(relative + len(piece), cursor)
    return fragments


def _joined_length(fragments: list[_Fragment]) -> int:
    return sum(len(item.text) for item in fragments) + 2 * max(0, len(fragments) - 1)


def _overlap_tail(fragments: list[_Fragment], budget: int) -> list[_Fragment]:
    if budget <= 0:
        return []
    selected: list[_Fragment] = []
    for fragment in reversed(fragments):
        candidate = [fragment, *selected]
        if _joined_length(candidate) > budget:
            break
        selected = candidate
    return selected


def _section_chunks(
    section: _Section, *, maximum: int, overlap: int
) -> list[tuple[str, int, int]]:
    prefix = "\n".join(section.heading_path).strip()
    body_maximum = maximum - len(prefix) - (2 if prefix else 0)
    if body_maximum < 32:
        raise ValueError("Markdown heading path leaves no room for chunk content")
    fragments: list[_Fragment] = []
    for block in section.blocks:
        fragments.extend(_block_fragments(block, body_maximum))

    chunks: list[tuple[str, int, int]] = []
    current: list[_Fragment] = []
    for fragment in fragments:
        candidate = [*current, fragment]
        if current and _joined_length(candidate) > body_maximum:
            body = "\n\n".join(item.text for item in current)
            text = f"{prefix}\n\n{body}" if prefix else body
            chunks.append(
                (
                    text,
                    min(item.char_start for item in current),
                    max(item.char_end for item in current),
                )
            )
            current = _overlap_tail(current, min(overlap, body_maximum // 2))
            while current and _joined_length([*current, fragment]) > body_maximum:
                current.pop(0)
        current.append(fragment)
    if current:
        body = "\n\n".join(item.text for item in current)
        text = f"{prefix}\n\n{body}" if prefix else body
        chunks.append(
            (
                text,
                min(item.char_start for item in current),
                max(item.char_end for item in current),
            )
        )
    return chunks


def chunk_text(
    value: str,
    *,
    target: int = DEFAULT_CHUNK_TARGET,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    format_hint: str = "markdown",
) -> list[dict[str, object]]:
    normalized = normalize_text(value)
    if not normalized:
        return []
    maximum = max(200, int(target))
    overlap_budget = max(0, min(int(overlap), maximum // 2))
    sections = (
        _markdown_sections(normalized)
        if format_hint in {"markdown", "md", ".md", ".markdown"}
        else _plain_sections(normalized)
    )

    raw_chunks: list[tuple[str, int, int]] = []
    for section in sections:
        raw_chunks.extend(
            _section_chunks(
                section, maximum=maximum, overlap=overlap_budget
            )
        )

    result: list[dict[str, object]] = []
    for ordinal, (text, start, end) in enumerate(raw_chunks):
        if len(text) > maximum:
            raise ValueError("generated chunk exceeds configured maximum")
        result.append(
            {
                "ordinal": ordinal,
                "text": text,
                "search_text": lexical_text(text),
                "char_start": start,
                "char_end": end,
                "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return result
