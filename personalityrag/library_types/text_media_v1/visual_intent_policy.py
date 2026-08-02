from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


VISUAL_INTENT_POLICY_KEYS = (
    "visual_object_terms",
    "lookup_action_terms",
    "generation_action_terms",
    "reference_connector_terms",
)
MAX_VISUAL_INTENT_TERMS_PER_CATEGORY = 256
MAX_VISUAL_INTENT_TERM_LENGTH = 64
MAX_VISUAL_INTENT_TERMS_TOTAL = 768
MAX_VISUAL_INTENT_POLICY_CSV_BYTES = 1024 * 1024

VISUAL_INTENT_POLICY_FILENAME = "visual_intent_policy.csv"
DEFAULT_VISUAL_INTENT_POLICY_PATH = (
    Path(__file__).with_name("resources") / "visual_intent_policy.default.csv"
)


def _normalize_term(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", normalized.casefold().strip())


def _normalize_policy_fields(
    value: Mapping[str, object],
    *,
    keys: tuple[str, ...],
) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    total = 0
    for key in keys:
        raw_terms = value[key]
        if not isinstance(raw_terms, (list, tuple)):
            raise ValueError(f"{key} must be a list of literal terms")
        if len(raw_terms) > MAX_VISUAL_INTENT_TERMS_PER_CATEGORY:
            raise ValueError(
                f"{key} supports at most "
                f"{MAX_VISUAL_INTENT_TERMS_PER_CATEGORY} terms"
            )
        terms: list[str] = []
        identities: set[str] = set()
        for raw_term in raw_terms:
            if not isinstance(raw_term, str):
                raise ValueError(f"{key} terms must be strings")
            term = _normalize_term(raw_term)
            if not term:
                raise ValueError(f"{key} terms cannot be empty")
            if len(term) > MAX_VISUAL_INTENT_TERM_LENGTH:
                raise ValueError(
                    f"{key} terms cannot exceed "
                    f"{MAX_VISUAL_INTENT_TERM_LENGTH} characters"
                )
            if term in identities:
                continue
            identities.add(term)
            terms.append(term)
        normalized[key] = terms
        total += len(terms)
    if total > MAX_VISUAL_INTENT_TERMS_TOTAL:
        raise ValueError(
            "visual intent policy supports at most "
            f"{MAX_VISUAL_INTENT_TERMS_TOTAL} total terms"
        )
    return normalized


def parse_visual_intent_policy_csv(
    payload: bytes,
    *,
    require_complete: bool = False,
) -> tuple[dict[str, list[str]], tuple[str, ...]]:
    """Parse UTF-8 CSV and return only categories present in its header."""

    if len(payload) > MAX_VISUAL_INTENT_POLICY_CSV_BYTES:
        raise ValueError("visual intent policy CSV cannot exceed 1 MiB")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("visual intent policy CSV must be UTF-8") from exc
    if not text:
        raise ValueError("visual intent policy CSV is empty")
    try:
        rows = csv.reader(io.StringIO(text, newline=""), strict=True)
        raw_header = next(rows)
    except StopIteration as exc:
        raise ValueError("visual intent policy CSV is empty") from exc
    except csv.Error as exc:
        raise ValueError(f"invalid visual intent policy CSV: {exc}") from exc

    header = tuple(cell.strip() for cell in raw_header)
    if not header or any(not cell for cell in header):
        raise ValueError("visual intent policy CSV contains an empty header")
    duplicates = sorted({cell for cell in header if header.count(cell) > 1})
    if duplicates:
        raise ValueError(
            "duplicate visual intent policy CSV columns: "
            + ", ".join(duplicates)
        )
    unknown = sorted(set(header) - set(VISUAL_INTENT_POLICY_KEYS))
    if unknown:
        raise ValueError(
            "unknown visual intent policy CSV columns: " + ", ".join(unknown)
        )
    if require_complete:
        missing = [key for key in VISUAL_INTENT_POLICY_KEYS if key not in header]
        if missing:
            raise ValueError(
                "visual intent policy CSV requires all columns: "
                + ", ".join(missing)
            )

    raw_policy: dict[str, list[str]] = {key: [] for key in header}
    try:
        for row_number, row in enumerate(rows, start=2):
            if len(row) != len(header):
                raise ValueError(
                    "visual intent policy CSV row "
                    f"{row_number} has {len(row)} fields; expected {len(header)}"
                )
            if tuple(cell.strip() for cell in row) == header:
                raise ValueError(
                    f"visual intent policy CSV repeats its header on row {row_number}"
                )
            for key, cell in zip(header, row, strict=True):
                if cell.strip():
                    raw_policy[key].append(cell)
    except csv.Error as exc:
        raise ValueError(f"invalid visual intent policy CSV: {exc}") from exc

    normalized = _normalize_policy_fields(raw_policy, keys=header)
    return normalized, header


def serialize_visual_intent_policy_csv(
    policy: Mapping[str, object],
    *,
    categories: tuple[str, ...] | list[str] | None = None,
) -> bytes:
    selected = tuple(categories or VISUAL_INTENT_POLICY_KEYS)
    if not selected:
        raise ValueError("select at least one visual intent policy category")
    if len(set(selected)) != len(selected):
        raise ValueError("visual intent policy categories cannot be duplicated")
    unknown = sorted(set(selected) - set(VISUAL_INTENT_POLICY_KEYS))
    if unknown:
        raise ValueError(
            "unknown visual intent policy categories: " + ", ".join(unknown)
        )
    missing = [key for key in selected if key not in policy]
    if missing:
        raise ValueError(
            "visual intent policy is missing categories: " + ", ".join(missing)
        )
    normalized = _normalize_policy_fields(policy, keys=selected)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerow(selected)
    row_count = max((len(normalized[key]) for key in selected), default=0)
    for index in range(row_count):
        writer.writerow(
            [
                normalized[key][index] if index < len(normalized[key]) else ""
                for key in selected
            ]
        )
    encoded = output.getvalue().encode("utf-8-sig")
    if len(encoded) > MAX_VISUAL_INTENT_POLICY_CSV_BYTES:
        raise ValueError("visual intent policy CSV cannot exceed 1 MiB")
    return encoded


@lru_cache(maxsize=1)
def _type_default_policy() -> dict[str, list[str]]:
    try:
        payload = DEFAULT_VISUAL_INTENT_POLICY_PATH.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            "text_media_v1 visual intent default CSV is unavailable"
        ) from exc
    policy, categories = parse_visual_intent_policy_csv(
        payload,
        require_complete=True,
    )
    if categories != VISUAL_INTENT_POLICY_KEYS:
        raise RuntimeError(
            "text_media_v1 visual intent default CSV columns are out of order"
        )
    return policy


DEFAULT_VISUAL_INTENT_POLICY: dict[str, tuple[str, ...]] = {
    key: tuple(values) for key, values in _type_default_policy().items()
}


def normalize_visual_intent_policy(
    value: object = None,
    *,
    require_complete: bool = False,
) -> dict[str, list[str]]:
    """Return a complete, deterministic literal visual-intent policy."""

    if value is None:
        source: Mapping[str, object] = {}
    elif isinstance(value, Mapping):
        source = value
    else:
        raise ValueError("visual_intent_policy must be an object")
    unexpected = sorted(set(source) - set(VISUAL_INTENT_POLICY_KEYS))
    if unexpected:
        raise ValueError(
            "unknown visual intent policy fields: " + ", ".join(unexpected)
        )
    if require_complete:
        missing = [key for key in VISUAL_INTENT_POLICY_KEYS if key not in source]
        if missing:
            raise ValueError(
                "visual intent policy requires all fields: " + ", ".join(missing)
            )
    complete = {
        key: source.get(key, DEFAULT_VISUAL_INTENT_POLICY[key])
        for key in VISUAL_INTENT_POLICY_KEYS
    }
    return _normalize_policy_fields(complete, keys=VISUAL_INTENT_POLICY_KEYS)


def visual_intent_policy_fingerprint(value: object = None) -> str:
    policy = normalize_visual_intent_policy(value)
    payload = json.dumps(
        policy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def visual_intent_policy_path(library_root: Path) -> Path:
    return Path(library_root) / VISUAL_INTENT_POLICY_FILENAME


def read_visual_intent_policy_override(
    library_root: Path,
) -> dict[str, list[str]] | None:
    target = visual_intent_policy_path(library_root)
    if not target.exists():
        return None
    try:
        payload = target.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"cannot read visual intent policy CSV: {target.name}"
        ) from exc
    policy, categories = parse_visual_intent_policy_csv(
        payload,
        require_complete=True,
    )
    if categories != VISUAL_INTENT_POLICY_KEYS:
        raise ValueError(
            "single-library visual intent policy CSV columns must use the "
            "canonical order"
        )
    return policy


def read_effective_visual_intent_policy(
    library_root: Path,
) -> dict[str, list[str]]:
    override = read_visual_intent_policy_override(library_root)
    return override or normalize_visual_intent_policy(DEFAULT_VISUAL_INTENT_POLICY)


def _atomic_write(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_visual_intent_policy_override(
    library_root: Path,
    policy: object,
) -> bool:
    normalized = normalize_visual_intent_policy(policy, require_complete=True)
    target = visual_intent_policy_path(library_root)
    if normalized == normalize_visual_intent_policy(DEFAULT_VISUAL_INTENT_POLICY):
        if target.exists():
            target.unlink()
        return False
    _atomic_write(target, serialize_visual_intent_policy_csv(normalized))
    return True


async def migrate_legacy_visual_intent_policy(
    storage: Any,
    library_root: Path,
) -> dict[str, list[str]]:
    """Move an old SQLite JSON policy to the per-library CSV override."""

    meta = await storage.metadata()
    raw = meta.get("retrieval_config_json")
    if isinstance(raw, Mapping):
        settings = dict(raw)
    else:
        try:
            settings = json.loads(str(raw or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("stored retrieval configuration is invalid JSON") from exc
    if not isinstance(settings, dict):
        raise ValueError("stored retrieval configuration must be a JSON object")

    target = visual_intent_policy_path(library_root)
    if target.exists():
        effective = read_effective_visual_intent_policy(library_root)
    elif "visual_intent_policy" in settings:
        legacy = normalize_visual_intent_policy(
            settings["visual_intent_policy"],
            require_complete=True,
        )
        write_visual_intent_policy_override(library_root, legacy)
        effective = legacy
    else:
        effective = normalize_visual_intent_policy(DEFAULT_VISUAL_INTENT_POLICY)

    if "visual_intent_policy" in settings:
        settings.pop("visual_intent_policy", None)
        await storage.update_metadata({"retrieval_config_json": settings})
    return effective
