from __future__ import annotations

import json
from pathlib import PurePath
from typing import Any


_REDACTED = "[redacted]"
_OMITTED = object()
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "access_key",
    "authorization",
    "credential",
    "credentials",
    "password",
    "pkb",
    "psk",
    "secret",
    "token",
}
_CONTENT_KEYS = {
    "base64",
    "bytes",
    "content",
    "data_uri",
    "document_text",
    "prompt",
    "query",
    "raw_text",
    "source_text",
    "text",
}
_PATH_KEYS = {
    "conversations_db",
    "package_path",
    "path",
    "source_db",
    "source_path",
    "target_path",
    "upload_path",
}
_MAX_DEPTH = 5
_MAX_COLLECTION_ITEMS = 24
_MAX_STRING_CHARS = 240
_MAX_DIFF_ITEMS = 300


def _key_tokens(key: str) -> set[str]:
    normalized = str(key or "").strip().lower().replace("-", "_")
    tokens = {normalized}
    tokens.update(part for part in normalized.split("_") if part)
    return tokens


def _matches_key(key: str, candidates: set[str]) -> bool:
    normalized = str(key or "").strip().lower().replace("-", "_")
    tokens = _key_tokens(normalized)
    return any(
        candidate in tokens
        or normalized.startswith(f"{candidate}_")
        or normalized.endswith(f"_{candidate}")
        or f"_{candidate}_" in normalized
        for candidate in candidates
    )


def _content_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"redacted": True, "bytes": len(value)}
    if isinstance(value, str):
        return {"redacted": True, "characters": len(value)}
    if isinstance(value, (list, tuple, set, dict)):
        return {"redacted": True, "items": len(value)}
    return {"redacted": True}


def _safe_path_name(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    normalized = value.replace("\\", "/")
    name = PurePath(normalized).name
    return name or "[local path]"


def sanitize_task_value(
    value: Any,
    *,
    key: str = "",
    depth: int = 0,
) -> Any:
    """Return bounded, JSON-safe task metadata without secrets or payload text."""

    if key.startswith("_"):
        return _OMITTED
    if _matches_key(key, _SENSITIVE_KEYS):
        return _REDACTED
    if _matches_key(key, _CONTENT_KEYS):
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return _content_summary(value)
    if _matches_key(key, _PATH_KEYS) or key.lower().endswith(("_path", "_file")):
        return _safe_path_name(value)
    if depth >= _MAX_DEPTH:
        return "[nested metadata omitted]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        lowered = value.lstrip().lower()
        if lowered.startswith(("data:", "bearer ")):
            return _REDACTED
        if len(value) <= _MAX_STRING_CHARS:
            return value
        return {
            "preview": value[:_MAX_STRING_CHARS] + "…",
            "characters": len(value),
            "truncated": True,
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"binary": True, "bytes": len(value)}
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for index, (child_key, child_value) in enumerate(value.items()):
            if index >= _MAX_COLLECTION_ITEMS:
                result["_truncated_items"] = len(value) - _MAX_COLLECTION_ITEMS
                break
            safe = sanitize_task_value(
                child_value,
                key=str(child_key),
                depth=depth + 1,
            )
            if safe is not _OMITTED:
                result[str(child_key)] = safe
        return result
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        safe_items = [
            sanitize_task_value(item, depth=depth + 1)
            for item in items[:_MAX_COLLECTION_ITEMS]
        ]
        if len(items) > _MAX_COLLECTION_ITEMS:
            return {
                "items": safe_items,
                "total_items": len(items),
                "truncated": True,
            }
        return safe_items
    return str(value)


def task_request_summary(operation: Any) -> Any:
    if operation is None:
        return None
    result = sanitize_task_value(operation)
    return None if result is _OMITTED else result


def _flatten_state(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            if str(key) == "captured_at":
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_state(child, child_prefix))
        return flattened
    return {prefix or "value": value}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def build_database_state_comparison(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    capture_errors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before_flat = _flatten_state(before or {})
    after_flat = _flatten_state(after or {})
    changes: list[dict[str, Any]] = []
    for path in sorted(set(before_flat) | set(after_flat)):
        before_present = path in before_flat
        after_present = path in after_flat
        before_value = before_flat.get(path)
        after_value = after_flat.get(path)
        if before_present and after_present and _canonical(before_value) == _canonical(after_value):
            continue
        change = "changed"
        if not before_present:
            change = "added"
        elif not after_present:
            change = "removed"
        changes.append(
            {
                "path": path,
                "change": change,
                "before": before_value if before_present else None,
                "after": after_value if after_present else None,
            }
        )
    truncated = len(changes) > _MAX_DIFF_ITEMS
    return {
        "available": before is not None or after is not None,
        "complete": before is not None and after is not None,
        "changed": bool(changes),
        "change_count": len(changes),
        "changes": changes[:_MAX_DIFF_ITEMS],
        "changes_truncated": truncated,
        "before": before,
        "after": after,
        "capture_errors": capture_errors or {},
    }
