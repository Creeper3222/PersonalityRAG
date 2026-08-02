from __future__ import annotations

from typing import Any


_MESSAGE_FIELDS = (
    "id",
    "session_id",
    "role",
    "content",
    "sender_id",
    "sender_name",
    "group_id",
    "platform",
    "timestamp",
)


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts: list[str] = []
        saw_media = False
        for item in value:
            part_text, part_has_media = _content_part_to_text(item)
            if part_text:
                parts.append(part_text)
            saw_media = saw_media or part_has_media
        text = " ".join(parts).strip()
        return text or ("[图片消息]" if saw_media else "")
    text, saw_media = _content_part_to_text(value)
    return text or ("[图片消息]" if saw_media else str(value))


def _content_part_to_text(part: Any) -> tuple[str, bool]:
    if part is None:
        return "", False
    if isinstance(part, str):
        return part, False
    if isinstance(part, (int, float, bool)):
        return str(part), False
    if isinstance(part, list):
        text = _content_text(part)
        return ("" if text == "[图片消息]" else text), text == "[图片消息]"
    if not isinstance(part, dict):
        return str(part), False
    part_type = str(part.get("type") or "").lower()
    if part_type in {"text", "plain"}:
        return str(part.get("text") or part.get("content") or ""), False
    if isinstance(part.get("text"), str):
        return str(part["text"]), False
    if "content" in part:
        return _content_part_to_text(part.get("content"))
    if "message" in part:
        return _content_part_to_text(part.get("message"))
    media_keys = {"image_url", "image", "file", "audio", "video", "media"}
    if part_type in media_keys or any(key in part for key in media_keys):
        return "", True
    return "", False


def serialize_source_messages(messages: Any) -> list[dict[str, Any]]:
    """Normalize retained source messages without persisting arbitrary payloads."""

    if not isinstance(messages, list):
        return []
    serialized: list[dict[str, Any]] = []
    for raw in messages:
        if isinstance(raw, str):
            raw = {"role": "user", "content": raw}
        elif isinstance(raw, dict):
            raw = raw
        elif hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        else:
            raw = {field: getattr(raw, field, None) for field in _MESSAGE_FIELDS}
        content = _content_text(raw.get("content"))
        if not content:
            continue
        metadata = raw.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        try:
            message_id = int(raw.get("id") or 0)
        except (TypeError, ValueError):
            message_id = 0
        try:
            timestamp = float(raw.get("timestamp") or 0.0)
        except (TypeError, ValueError):
            timestamp = 0.0
        item = {field: raw.get(field) for field in _MESSAGE_FIELDS}
        item.update(
            {
                "id": message_id,
                "session_id": str(raw.get("session_id") or ""),
                "role": str(raw.get("role") or "user"),
                "content": content,
                "sender_id": str(raw.get("sender_id") or "unknown"),
                "timestamp": timestamp,
                "metadata": {
                    "is_bot_message": bool(metadata.get("is_bot_message", False))
                },
            }
        )
        serialized.append(item)
    return serialized


__all__ = ["serialize_source_messages"]
