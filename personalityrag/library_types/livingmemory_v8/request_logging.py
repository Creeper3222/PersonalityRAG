from __future__ import annotations

from typing import Any

from fastapi import Request

from ...logger import logger, safe_summary
from ...request_logging import (
    log_bool,
    log_optional_bool,
    request_log_context,
)


def log_recall_request_received(
    request: Request,
    *,
    memory_store_id: str,
    payload: Any,
) -> str:
    context = request_log_context(request)
    logger.info(
        "[livingmemory_v8] 收到记忆召回请求：请求ID=%s 记忆库ID=%s "
        "请求来源=%s 适配器ID=%s 适配器实例ID=%s 适配器类型=%s "
        "查询字符数=%s 嵌入召回数=%s Rerank输出数=%s 请求Rerank=%s "
        "包含嵌入基线=%s 提供会话过滤ID=%s 提供人格过滤ID=%s",
        context.request_id,
        memory_store_id,
        context.source,
        context.adapter_id,
        context.adapter_instance_id,
        context.adapter_type,
        len(str(payload.query or "")),
        int(payload.k),
        int(payload.rerank_k or payload.k),
        log_optional_bool(payload.rerank),
        log_bool(payload.include_baseline),
        log_bool(bool(str(payload.session_id or "").strip())),
        log_bool(bool(str(payload.persona_id or "").strip())),
    )
    return context.request_id


def log_recall_rerank_fallback(
    *,
    request_id: str,
    memory_store_id: str,
    provider_id: str,
    error: Exception,
) -> None:
    logger.warning(
        "[livingmemory_v8] 记忆召回 Rerank 失败并回退嵌入结果："
        "请求ID=%s 记忆库ID=%s Rerank提供方ID=%s 错误=%s",
        request_id,
        memory_store_id,
        provider_id or "-",
        safe_summary(error),
    )


def log_recall_failed(
    *,
    request_id: str,
    memory_store_id: str,
    error: Exception,
) -> None:
    logger.warning(
        "[livingmemory_v8] 记忆召回失败：请求ID=%s 记忆库ID=%s 错误=%s",
        request_id,
        memory_store_id,
        safe_summary(error),
    )


def log_recall_completed(
    *,
    request_id: str,
    memory_store_id: str,
    elapsed_ms: float,
    embedding_candidates: int,
    returned_results: int,
    rerank_available: bool,
    rerank_applied: bool,
    rerank_failed: bool,
    session_filter_applied: bool,
    persona_filter_applied: bool,
    generation: object,
) -> None:
    logger.info(
        "[livingmemory_v8] 记忆召回完成：请求ID=%s 记忆库ID=%s "
        "耗时毫秒=%.2f 嵌入候选数=%s 返回记忆数=%s "
        "Rerank可用=%s Rerank已应用=%s Rerank失败=%s "
        "会话过滤已应用=%s 人格过滤已应用=%s 索引代次=%s",
        request_id,
        memory_store_id,
        elapsed_ms,
        embedding_candidates,
        returned_results,
        log_bool(rerank_available),
        log_bool(rerank_applied),
        log_bool(rerank_failed),
        log_bool(session_filter_applied),
        log_bool(persona_filter_applied),
        safe_summary(generation or "-", max_chars=96),
    )
