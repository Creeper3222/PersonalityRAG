from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from fastapi import Request

from .http_shared import (
    ADAPTER_ID_HEADER,
    ADAPTER_INSTANCE_HEADER,
    ADAPTER_TYPE_HEADER,
)
from .logger import safe_summary


_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,128}")


@dataclass(frozen=True)
class RequestLogContext:
    request_id: str
    source: str
    adapter_id: str
    adapter_instance_id: str
    adapter_type: str


def _safe_header(request: Request, name: str, *, max_chars: int = 96) -> str:
    return safe_summary(request.headers.get(name) or "-", max_chars=max_chars)


def request_log_context(request: Request) -> RequestLogContext:
    state = request.scope.setdefault("state", {})
    request_id = str(state.get("request_id") or "").strip()
    if not _REQUEST_ID_PATTERN.fullmatch(request_id):
        header_request_id = str(request.headers.get("x-request-id") or "").strip()
        request_id = (
            header_request_id
            if _REQUEST_ID_PATTERN.fullmatch(header_request_id)
            else uuid.uuid4().hex
        )
        state["request_id"] = request_id
    adapter_id = _safe_header(request, ADAPTER_ID_HEADER)
    has_adapter = adapter_id != "-"
    return RequestLogContext(
        request_id=request_id,
        source="适配器" if has_adapter else "管理端或其他客户端",
        adapter_id=adapter_id,
        adapter_instance_id=_safe_header(
            request,
            ADAPTER_INSTANCE_HEADER,
            max_chars=128,
        ),
        adapter_type=_safe_header(request, ADAPTER_TYPE_HEADER),
    )


def log_bool(value: object) -> str:
    return "true" if bool(value) else "false"


def log_optional_bool(value: bool | None) -> str:
    if value is None:
        return "默认"
    return log_bool(value)


def log_optional_number(value: object) -> str:
    if value is None:
        return "未提供"
    return str(value)
