from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import io
import json
import re
import secrets
import time
from pathlib import Path
from typing import Any, Iterable

from ...io_utils import atomic_write_json
from .source import serialize_source_messages


TRANSFER_FORMAT = "personalityrag-livingmemory-transfer"
TRANSFER_VERSION = 1
MAX_TRANSFER_BYTES = 50 * 1024 * 1024
MAX_TRANSFER_RECORDS = 10_000
PREVIEW_TTL_SECONDS = 30 * 60
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")
_CONTENT_KEYS = (
    "canonical_summary",
    "content",
    "text",
    "summary",
    "memory",
    "value",
    "memory_content",
)
_SOURCE_KEYS = (
    "source_messages",
    "original_messages",
    "raw_messages",
    "messages",
    "conversation",
    "dialogue",
    "dialog",
    "source",
)
_LONG_TERM_KEYS = (
    "memories",
    "long_term_memories",
    "longTermMemories",
    "long_term_memory",
    "items",
    "documents",
    "records",
)
_SHORT_TERM_KEYS = (
    "short_term_memories",
    "shortTermMemories",
    "conversations",
    "sessions",
)
_CSV_SCALAR_FIELDS = {
    "content",
    "canonical_summary",
    "persona_summary",
    "session_id",
    "persona_id",
    "memory_type",
    "status",
}


class MemoryTransferError(ValueError):
    pass


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def transfer_dedupe_key(
    content: Any, session_id: Any = None, persona_id: Any = None
) -> str:
    payload = "\0".join(
        (
            _normalize_space(content),
            _normalize_space(session_id),
            _normalize_space(persona_id),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _list_value(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [
                    str(item).strip() for item in parsed if str(item).strip()
                ]
        return [
            item.strip()
            for item in re.split(r"[;,，；\n]+", stripped)
            if item.strip()
        ]
    return [str(value).strip()] if str(value).strip() else []


def _dict_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _first_value(raw: dict[str, Any], keys: Iterable[str]) -> Any:
    return next(
        (raw.get(key) for key in keys if raw.get(key) not in (None, "")),
        None,
    )


def _source_value(
    raw: dict[str, Any], session_id: str | None
) -> list[dict[str, Any]]:
    value = _first_value(raw, _SOURCE_KEYS)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = []
    if isinstance(value, dict):
        value = value.get("messages", value.get("conversation", []))
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(value):
        if isinstance(message, str):
            message = {"role": "user", "content": message}
        if not isinstance(message, dict):
            continue
        role = str(
            message.get("role")
            or message.get("sender")
            or message.get("speaker")
            or "user"
        ).lower()
        if role in {"ai", "bot", "model"}:
            role = "assistant"
        elif role in {"human", "customer"}:
            role = "user"
        content = _first_value(message, ("content", "text", "message", "value"))
        if content in (None, ""):
            continue
        normalized.append(
            {
                "id": message.get("id", index + 1),
                "session_id": message.get("session_id") or session_id or "import",
                "role": role,
                "content": content,
                "sender_id": message.get("sender_id") or role,
                "sender_name": message.get("sender_name"),
                "group_id": message.get("group_id"),
                "platform": message.get("platform") or "import",
                "timestamp": message.get("timestamp", 0.0),
                "metadata": message.get("metadata")
                or {"is_bot_message": role == "assistant"},
            }
        )
    return serialize_source_messages(normalized)


def _importance(value: Any) -> float:
    try:
        number = float(value if value not in (None, "") else 0.5)
    except (TypeError, ValueError):
        number = 0.5
    if 1 < number <= 10:
        number /= 10
    return max(0.0, min(1.0, number))


def normalize_transfer_record(raw: Any, row_number: int) -> dict[str, Any]:
    if isinstance(raw, str):
        raw = {"content": raw}
    if not isinstance(raw, dict):
        raise MemoryTransferError("record must be an object or text")
    metadata = _dict_value(raw.get("metadata"))
    session_id = raw.get("session_id", metadata.get("session_id"))
    if session_id in (None, ""):
        session_id = raw.get("session")
    persona_id = raw.get("persona_id", metadata.get("persona_id"))
    if persona_id in (None, ""):
        persona_id = raw.get("persona")
    normalized_session_id = (
        str(session_id).strip() if session_id not in (None, "") else None
    )
    normalized_persona_id = (
        str(persona_id).strip() if persona_id not in (None, "") else None
    )
    content = _normalize_space(_first_value(raw, _CONTENT_KEYS))
    source_messages = _source_value(raw, normalized_session_id)
    if not source_messages:
        user_text = _first_value(raw, ("user", "human", "query", "input"))
        assistant_text = _first_value(
            raw, ("assistant", "ai", "bot", "response", "output")
        )
        if user_text not in (None, "") and assistant_text not in (None, ""):
            source_messages = _source_value(
                {
                    "messages": [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ]
                },
                normalized_session_id,
            )
    if not content and len(source_messages) < 2:
        raise MemoryTransferError(
            "record needs a summary or at least two source messages"
        )
    persona_summary = _normalize_space(
        raw.get("persona_summary")
        or raw.get("personalized_summary")
        or metadata.get("persona_summary")
        or content
    )
    status = str(raw.get("status") or metadata.get("status") or "active")
    if status not in {"active", "archived"}:
        status = "active"
    topics = _list_value(raw.get("topics", metadata.get("topics")))
    participants = _list_value(
        raw.get("participants", metadata.get("participants"))
    )
    key_facts = _list_value(raw.get("key_facts", metadata.get("key_facts")))
    importance = _importance(raw.get("importance", metadata.get("importance")))
    memory_type = raw.get("memory_type") or raw.get("type")
    if memory_type not in (None, ""):
        metadata["memory_type"] = str(memory_type).upper()
    for key in (
        "canonical_summary",
        "persona_summary",
        "session_id",
        "persona_id",
        "importance",
        "status",
        "topics",
        "participants",
        "key_facts",
        "has_source",
        "source_message_count",
        "atom_types",
        "previous_id",
    ):
        metadata.pop(key, None)
    original_id = raw.get("original_id", raw.get("id"))
    if original_id is not None:
        metadata["imported_from_id"] = original_id
    item_id = str(raw.get("preview_item_id") or original_id or row_number)
    return {
        "preview_item_id": item_id,
        "row_number": row_number,
        "content": content,
        "canonical_summary": content,
        "persona_summary": persona_summary,
        "session_id": normalized_session_id,
        "persona_id": normalized_persona_id,
        "importance": importance,
        "status": status,
        "topics": topics,
        "participants": participants,
        "key_facts": key_facts,
        "source_messages": source_messages,
        "source_time_strategy": str(raw.get("source_time_strategy") or "preserve"),
        "source_time_tags": _dict_value(raw.get("source_time_tags")) or None,
        "metadata": metadata,
        "needs_summary": not bool(content),
    }


def _looks_like_message_list(value: list[Any]) -> bool:
    return bool(value) and all(
        isinstance(item, dict)
        and any(key in item for key in ("role", "sender", "speaker"))
        and any(key in item for key in ("content", "text", "message", "value"))
        for item in value
    )


def _as_entry_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        return []
    entries: list[Any] = []
    for original_id, item in value.items():
        if isinstance(item, dict):
            entry = dict(item)
            entry.setdefault("original_id", original_id)
            entries.append(entry)
        elif isinstance(item, str):
            entries.append({"original_id": original_id, "content": item})
    return entries


def _as_conversation_entries(value: Any) -> list[Any]:
    if isinstance(value, dict):
        entries: list[Any] = []
        for session_id, item in value.items():
            if isinstance(item, list):
                entries.append({"session_id": session_id, "messages": item})
            elif isinstance(item, dict):
                entry = dict(item)
                entry.setdefault("session_id", session_id)
                entries.append(entry)
        return entries
    if not isinstance(value, list):
        return _as_entry_list(value)
    if _looks_like_message_list(value):
        return [{"messages": value}]
    return [
        {"messages": item} if isinstance(item, list) else item for item in value
    ]


def _raw_records(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        if _looks_like_message_list(payload):
            return [{"messages": payload}]
        return payload
    if not isinstance(payload, dict):
        raise MemoryTransferError("JSON transfer must be an array or object")
    entries: list[Any] = []
    found_collection = False
    for key in _LONG_TERM_KEYS:
        if key in payload:
            found_collection = True
            entries.extend(_as_entry_list(payload[key]))
    for key in _SHORT_TERM_KEYS:
        if key in payload:
            found_collection = True
            entries.extend(_as_conversation_entries(payload[key]))
    if found_collection:
        return entries
    if "data" in payload and isinstance(payload["data"], (dict, list)):
        return _raw_records(payload["data"])
    if any(key in payload for key in (*_CONTENT_KEYS, *_SOURCE_KEYS)):
        return [payload]
    mapped = _as_entry_list(payload)
    if mapped:
        return mapped
    raise MemoryTransferError("JSON transfer contains no memory records")


def _unescape_csv_formula(value: str) -> str:
    if len(value) >= 2 and value[0] == "'" and value[1] in "=+-@\t\r\n":
        return value[1:]
    return value


def _unescape_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _unescape_csv_formula(value)
        if key in _CSV_SCALAR_FIELDS and isinstance(value, str)
        else value
        for key, value in row.items()
    }


def parse_transfer_bytes(data: bytes, filename: str = "transfer.json") -> list[Any]:
    if len(data) > MAX_TRANSFER_BYTES:
        raise MemoryTransferError("transfer file exceeds 50 MiB")
    suffix = Path(filename).suffix.lower()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MemoryTransferError("transfer file must use UTF-8") from exc
    if suffix == ".csv":
        try:
            rows = [
                _unescape_csv_row(row)
                for row in csv.DictReader(io.StringIO(text, newline=""))
            ]
        except csv.Error as exc:
            raise MemoryTransferError("CSV transfer is malformed") from exc
        if not rows:
            raise MemoryTransferError("CSV transfer contains no records")
        return rows
    if suffix not in {".json", ""}:
        raise MemoryTransferError("only JSON and CSV transfer files are supported")
    try:
        return _raw_records(json.loads(text))
    except json.JSONDecodeError as exc:
        raise MemoryTransferError("JSON transfer is malformed") from exc


def inspect_transfer_records(
    raw_records: Iterable[Any], existing_keys: set[str]
) -> dict[str, Any]:
    records = list(raw_records)
    if len(records) > MAX_TRANSFER_RECORDS:
        raise MemoryTransferError("transfer contains more than 10,000 records")
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    seen = set(existing_keys)
    for index, raw in enumerate(records, start=1):
        try:
            item = normalize_transfer_record(raw, index)
        except MemoryTransferError as exc:
            invalid.append({"row_number": index, "error": str(exc)})
            continue
        key = (
            transfer_dedupe_key(
                item["content"], item.get("session_id"), item.get("persona_id")
            )
            if item["content"]
            else ""
        )
        duplicate = bool(key and key in seen)
        item["dedupe_key"] = key
        item["duplicate"] = duplicate
        if key:
            seen.add(key)
        valid.append(item)
    return {
        "items": valid,
        "invalid_items": invalid,
        "counts": {
            "input": len(records),
            "valid": len(valid),
            "invalid": len(invalid),
            "duplicates": sum(bool(item["duplicate"]) for item in valid),
            "needs_summary": sum(bool(item["needs_summary"]) for item in valid),
            "planned_skip": sum(bool(item["duplicate"]) for item in valid),
            "planned_import": sum(not bool(item["duplicate"]) for item in valid),
        },
    }


def existing_transfer_keys(records: Iterable[dict[str, Any]]) -> set[str]:
    return {
        transfer_dedupe_key(
            item.get("content"), item.get("session_id"), item.get("persona_id")
        )
        for item in records
        if _normalize_space(item.get("content"))
    }


def _preview_root(state_root: Path) -> Path:
    return state_root / "data" / "import_previews" / "livingmemory_v8"


def create_transfer_preview(
    *,
    state_root: Path,
    database_id: str,
    source_sha256: str,
    inspection: dict[str, Any],
    secret: str,
) -> str:
    nonce = secrets.token_urlsafe(24)
    now = time.time()
    payload = {
        "version": 1,
        "nonce": nonce,
        "database_id": database_id,
        "source_sha256": source_sha256,
        "created_at": now,
        "expires_at": now + PREVIEW_TTL_SECONDS,
    }
    encoded = _b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    signature = _b64encode(
        hmac.new(secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).digest()
    )
    root = _preview_root(state_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        root / f"{nonce}.json",
        {**payload, "inspection": inspection},
    )
    return f"{encoded}.{signature}"


def load_transfer_preview(
    *, state_root: Path, database_id: str, preview_id: str, secret: str
) -> dict[str, Any]:
    try:
        encoded, signature = preview_id.split(".", 1)
        expected = _b64encode(
            hmac.new(
                secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
            ).digest()
        )
        if not hmac.compare_digest(signature, expected):
            raise MemoryTransferError("import preview is invalid")
        payload = json.loads(_b64decode(encoded).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MemoryTransferError("import preview is invalid") from exc
    if (
        not isinstance(payload, dict)
        or str(payload.get("database_id") or "") != database_id
        or float(payload.get("expires_at") or 0) <= time.time()
    ):
        raise MemoryTransferError("import preview has expired or belongs to another library")
    path = _preview_root(state_root) / f"{payload.get('nonce')}.json"
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MemoryTransferError("import preview is missing") from exc
    for key in ("database_id", "source_sha256", "expires_at"):
        if stored.get(key) != payload.get(key):
            raise MemoryTransferError("import preview fingerprint changed")
    return {**stored, "path": str(path)}


def consume_transfer_preview(preview: dict[str, Any]) -> None:
    Path(str(preview.get("path") or "")).unlink(missing_ok=True)


def apply_transfer_summaries(
    items: list[dict[str, Any]], summaries: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {
        str(item.get("preview_item_id") or ""): item for item in summaries
    }
    ready: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for source in items:
        item = dict(source)
        if item.get("needs_summary"):
            summary = by_id.get(str(item.get("preview_item_id") or "")) or {}
            canonical = _normalize_space(summary.get("canonical_summary"))
            if not canonical:
                errors.append(
                    {
                        "preview_item_id": item.get("preview_item_id"),
                        "row_number": item.get("row_number"),
                        "error": "source-only record has no generated summary",
                    }
                )
                continue
            item["content"] = canonical
            item["canonical_summary"] = canonical
            item["persona_summary"] = _normalize_space(
                summary.get("persona_summary") or canonical
            )
            item["key_facts"] = _list_value(
                summary.get("key_facts") or item.get("key_facts")
            )
            item["topics"] = _list_value(
                summary.get("topics") or item.get("topics")
            )
            item["participants"] = _list_value(
                summary.get("participants") or item.get("participants")
            )
            item["participant_identities"] = list(
                summary.get("participant_identities")
                or item.get("participant_identities")
                or []
            )
            item["importance"] = _importance(
                summary.get("importance", item.get("importance"))
            )
            item["needs_summary"] = False
            item["dedupe_key"] = transfer_dedupe_key(
                canonical, item.get("session_id"), item.get("persona_id")
            )
        ready.append(item)
    return ready, errors


def _csv_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    elif value is None:
        text = ""
    else:
        text = str(value)
    return "'" + text if text.startswith(_FORMULA_PREFIXES) else text


def export_transfer_json(records: Iterable[dict[str, Any]]) -> bytes:
    payload = {
        "format": TRANSFER_FORMAT,
        "version": TRANSFER_VERSION,
        "created_at": time.time(),
        "memories": list(records),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def export_transfer_csv(records: Iterable[dict[str, Any]]) -> bytes:
    fields = (
        "content",
        "persona_summary",
        "importance",
        "status",
        "session_id",
        "persona_id",
        "topics",
        "participants",
        "key_facts",
        "source_messages",
        "metadata",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        metadata = dict(record.get("metadata") or {})
        writer.writerow(
            {
                field: _csv_cell(
                    record.get(field)
                    if field not in {"persona_summary", "status"}
                    else metadata.get(field, record.get(field))
                )
                for field in fields
            }
        )
    return ("\ufeff" + output.getvalue()).encode("utf-8")
