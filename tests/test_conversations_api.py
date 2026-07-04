from __future__ import annotations

import pytest

from personalityrag.storage import Storage


@pytest.mark.asyncio
async def test_conversation_short_term_buffer_and_metadata(tmp_path):
    storage = Storage(tmp_path)
    await storage.initialize()

    first = await storage.add_conversation_message(
        {
            "session_id": "umo:test",
            "role": "user",
            "content": "你好",
            "sender_id": "u1",
            "sender_name": "用户",
            "platform": "astrbot",
            "timestamp": 100.0,
            "dedup_key": "m1",
        }
    )
    duplicate = await storage.add_conversation_message(
        {
            "session_id": "umo:test",
            "role": "user",
            "content": "你好",
            "sender_id": "u1",
            "platform": "astrbot",
            "timestamp": 100.0,
            "dedup_key": "m1",
        }
    )
    await storage.add_conversation_message(
        {
            "session_id": "umo:test",
            "role": "assistant",
            "content": "你好呀",
            "sender_id": "bot",
            "sender_name": "Bot",
            "platform": "astrbot",
            "timestamp": 101.0,
            "metadata": {"is_bot_message": True},
        }
    )

    assert first["duplicate"] is False
    assert duplicate["duplicate"] is True
    convo = await storage.get_conversation("umo:test", limit=10)
    assert convo is not None
    assert convo["session"]["message_count"] == 2
    assert convo["session"]["participants"] == ["u1", "bot"]
    assert [msg["role"] for msg in convo["messages"]] == ["user", "assistant"]

    updated = await storage.update_conversation_metadata(
        "umo:test", {"last_summarized_index": 2, "pending_summary": None}
    )
    assert updated is not None
    assert updated["metadata"]["last_summarized_index"] == 2
    assert "pending_summary" not in updated["metadata"]


@pytest.mark.asyncio
async def test_conversation_range_and_safe_trim(tmp_path):
    storage = Storage(tmp_path)
    await storage.initialize()
    for idx in range(6):
        await storage.add_conversation_message(
            {
                "session_id": "umo:test",
                "role": "assistant" if idx % 2 else "user",
                "content": f"msg-{idx}",
                "sender_id": "bot" if idx % 2 else "u1",
                "platform": "astrbot",
                "timestamp": float(idx),
            }
        )
    await storage.update_conversation_metadata(
        "umo:test", {"last_summarized_index": 4}
    )

    window = await storage.get_conversation_messages(
        "umo:test", start_index=1, end_index=5
    )
    assert [msg["content"] for msg in window] == ["msg-1", "msg-2", "msg-3", "msg-4"]

    trimmed = await storage.trim_conversation("umo:test", 10)
    assert trimmed["deleted"] == 4
    convo = await storage.get_conversation("umo:test", limit=10)
    assert convo is not None
    assert convo["session"]["message_count"] == 2
    assert convo["session"]["metadata"]["last_summarized_index"] == 0
    assert [msg["content"] for msg in convo["messages"]] == ["msg-4", "msg-5"]

    cleared = await storage.clear_conversation("umo:test")
    assert cleared["deleted"] == 2
    convo = await storage.get_conversation("umo:test", limit=10)
    assert convo is not None
    assert convo["session"]["message_count"] == 0
    assert convo["messages"] == []
