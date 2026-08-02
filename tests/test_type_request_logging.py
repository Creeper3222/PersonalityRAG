from __future__ import annotations

import logging

from fastapi import Request

from personalityrag.library_types.livingmemory_v8.request_logging import (
    log_recall_completed,
    log_recall_request_received,
)
from personalityrag.library_types.text_media_v1.api import SearchRequest
from personalityrag.library_types.text_media_v1.request_logging import (
    log_search_completed,
    log_search_request_received,
)
from personalityrag.schemas import RecallRequest


def _adapter_request(path: str, request_id: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [
                (b"x-personalityrag-adapter-id", b"Astrbot"),
                (
                    b"x-personalityrag-adapter-instance-id",
                    b"astrbot-fixture-instance",
                ),
                (
                    b"x-personalityrag-adapter-type",
                    b"astrbot_knowledge_base",
                ),
            ],
            "state": {"request_id": request_id},
        }
    )


def test_text_media_request_logging_is_type_specific_and_redacted(
    caplog,
) -> None:
    query = "不要把这段图文检索正文写入日志"
    request = _adapter_request(
        "/api/v1/knowledge-libraries/text_media_v1/beileite_test/search",
        "text-media-request",
    )
    payload = SearchRequest(
        query=query,
        retrieval_mode="standard",
        media_response_mode="descriptions_only",
        top_k=10,
        media_output_confidence_threshold=0.6,
        media_relevance_pivot=0.35,
        max_media_outputs=5,
        rerank=True,
    )
    caplog.set_level(logging.INFO, logger="personalityrag")

    request_id = log_search_request_received(
        request,
        knowledge_base_id="beileite_test",
        payload=payload,
    )
    log_search_completed(
        request_id=request_id,
        knowledge_base_id="beileite_test",
        elapsed_ms=12.5,
        result={
            "baseline_items": [{}, {}],
            "items": [{}],
            "media_decisions": [{}, {}, {}],
            "media_outputs": [{}, {}],
            "rerank": {
                "provider_available": True,
                "applied": True,
                "fallback": False,
            },
            "execution": {
                "text_retrieval_executed": True,
                "media_channel_executed": True,
            },
        },
    )

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "[text_media_v1] 收到图文知识检索请求" in messages
    assert "请求ID=text-media-request" in messages
    assert "知识库ID=beileite_test" in messages
    assert "适配器ID=Astrbot" in messages
    assert "检索模式=standard" in messages
    assert "媒体响应模式=descriptions_only" in messages
    assert f"查询字符数={len(query)}" in messages
    assert "请求Rerank=true" in messages
    assert "最终文本分块数=1" in messages
    assert "输出媒体数=2" in messages
    assert "Rerank回退=false" in messages
    assert query not in messages
    assert "library_id=" not in messages


def test_livingmemory_request_logging_uses_recall_parameters(
    caplog,
) -> None:
    query = "不要把这段记忆召回正文写入日志"
    request = _adapter_request(
        "/api/v1/memory-libraries/livingmemory_v8/beileite_test/recall",
        "livingmemory-request",
    )
    payload = RecallRequest(
        query=query,
        k=20,
        rerank_k=10,
        rerank=None,
        include_baseline=True,
        session_id="private-session",
        persona_id="private-persona",
    )
    caplog.set_level(logging.INFO, logger="personalityrag")

    request_id = log_recall_request_received(
        request,
        memory_store_id="beileite_test",
        payload=payload,
    )
    log_recall_completed(
        request_id=request_id,
        memory_store_id="beileite_test",
        elapsed_ms=8.25,
        embedding_candidates=20,
        returned_results=10,
        rerank_available=True,
        rerank_applied=True,
        rerank_failed=False,
        session_filter_applied=True,
        persona_filter_applied=True,
        generation="gen-fixture",
    )

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "[livingmemory_v8] 收到记忆召回请求" in messages
    assert "请求ID=livingmemory-request" in messages
    assert "记忆库ID=beileite_test" in messages
    assert "适配器ID=Astrbot" in messages
    assert f"查询字符数={len(query)}" in messages
    assert "嵌入召回数=20" in messages
    assert "Rerank输出数=10" in messages
    assert "请求Rerank=默认" in messages
    assert "包含嵌入基线=true" in messages
    assert "返回记忆数=10" in messages
    assert "Rerank失败=false" in messages
    assert query not in messages
    assert "private-session" not in messages
    assert "private-persona" not in messages
    assert "library_id=" not in messages
