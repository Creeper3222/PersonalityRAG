from __future__ import annotations

from typing import Any

from fastapi import Request

from ...logger import logger, safe_summary
from ...request_logging import (
    log_bool,
    log_optional_bool,
    log_optional_number,
    request_log_context,
)


def log_search_request_received(
    request: Request,
    *,
    knowledge_base_id: str,
    payload: Any,
) -> str:
    context = request_log_context(request)
    payload_data = payload.model_dump()
    logger.info(
        "[text_media_v1] 收到图文知识检索请求：请求ID=%s 知识库ID=%s "
        "请求来源=%s 适配器ID=%s 适配器实例ID=%s 适配器类型=%s "
        "查询字符数=%s 检索模式=%s 媒体响应模式=%s 文本Top-K=%s "
        "媒体输出置信阈值=%s 媒体相关性枢轴=%s 兼容媒体阈值=%s "
        "最大媒体输出数=%s 请求Rerank=%s",
        context.request_id,
        knowledge_base_id,
        context.source,
        context.adapter_id,
        context.adapter_instance_id,
        context.adapter_type,
        len(str(payload.query or "")),
        payload.retrieval_mode,
        payload.media_response_mode,
        int(payload.top_k),
        payload.media_output_confidence_threshold,
        log_optional_number(payload.media_relevance_pivot),
        log_optional_number(payload_data.get("media_score_threshold")),
        int(payload.max_media_outputs),
        log_optional_bool(payload.rerank),
    )
    return context.request_id


def log_search_rejected(
    *,
    request_id: str,
    knowledge_base_id: str,
    status_code: int,
    error: Exception,
) -> None:
    logger.warning(
        "[text_media_v1] 图文知识检索被拒绝：请求ID=%s 知识库ID=%s "
        "状态码=%s 错误=%s",
        request_id,
        knowledge_base_id,
        status_code,
        safe_summary(error),
    )


def log_search_failed(
    *,
    request_id: str,
    knowledge_base_id: str,
    error: Exception,
) -> None:
    logger.warning(
        "[text_media_v1] 图文知识检索失败：请求ID=%s 知识库ID=%s 错误=%s",
        request_id,
        knowledge_base_id,
        safe_summary(error),
    )


def log_search_completed(
    *,
    request_id: str,
    knowledge_base_id: str,
    elapsed_ms: float,
    result: dict[str, Any],
) -> None:
    rerank = result.get("rerank") if isinstance(result.get("rerank"), dict) else {}
    execution = (
        result.get("execution")
        if isinstance(result.get("execution"), dict)
        else {}
    )
    logger.info(
        "[text_media_v1] 图文知识检索完成：请求ID=%s 知识库ID=%s "
        "耗时毫秒=%.2f 嵌入基线分块数=%s 最终文本分块数=%s "
        "媒体候选数=%s 输出媒体数=%s Rerank可用=%s "
        "Rerank已应用=%s Rerank回退=%s 文本通道已执行=%s "
        "媒体通道已执行=%s",
        request_id,
        knowledge_base_id,
        elapsed_ms,
        len(result.get("baseline_items") or []),
        len(result.get("items") or []),
        len(result.get("media_decisions") or []),
        len(result.get("media_outputs") or []),
        log_bool(rerank.get("provider_available")),
        log_bool(rerank.get("applied")),
        log_bool(rerank.get("fallback")),
        log_bool(execution.get("text_retrieval_executed")),
        log_bool(execution.get("media_channel_executed")),
    )
