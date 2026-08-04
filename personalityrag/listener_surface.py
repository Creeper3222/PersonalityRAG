from __future__ import annotations

import re
from typing import Any

from starlette.datastructures import MutableHeaders


WEBUI_SURFACE = "webui"
ADAPTER_ACCESS_SURFACE = "adapter-access"
SURFACE_HEADER = "X-PersonalityRAG-Surface"
PROTOCOL_HEADER = "X-PersonalityRAG-Adapter-Protocol"
ADAPTER_PROTOCOL_VERSION = "1"
SCOPE_STATE_KEY = "personalityrag_surface"

_IDENTIFIER = r"[A-Za-z0-9_-]+"
_PATH_SEGMENT = r"[^/]+"
_MEMORY_BASE = (
    rf"/api/v1/memory-libraries/livingmemory_v8/(?P<memory_store_id>{_IDENTIFIER})"
)
_KNOWLEDGE_BASE = (
    rf"/api/v1/knowledge-libraries/text_media_v1/"
    rf"(?P<knowledge_base_id>{_IDENTIFIER})"
)

_MEMORY_ACCESS_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("GET", re.compile(rf"^{_MEMORY_BASE}$")),
    ("PATCH", re.compile(rf"^{_MEMORY_BASE}$")),
    ("GET", re.compile(rf"^{_MEMORY_BASE}/stats$")),
    ("POST", re.compile(rf"^{_MEMORY_BASE}/adapters/heartbeat$")),
    ("POST", re.compile(rf"^{_MEMORY_BASE}/recall$")),
    ("POST", re.compile(rf"^{_MEMORY_BASE}/memories$")),
    ("DELETE", re.compile(rf"^{_MEMORY_BASE}/memories/[0-9]+$")),
    (
        "GET",
        re.compile(rf"^{_MEMORY_BASE}/memories/[0-9]+/source$"),
    ),
    (
        "POST",
        re.compile(
            rf"^{_MEMORY_BASE}/memories/[0-9]+/(?:archive|restore|resummary)$"
        ),
    ),
    (
        "GET",
        re.compile(rf"^{_MEMORY_BASE}/transfers/export$"),
    ),
    (
        "POST",
        re.compile(rf"^{_MEMORY_BASE}/transfers/imports/preview$"),
    ),
    (
        "POST",
        re.compile(
            rf"^{_MEMORY_BASE}/transfers/imports/{_PATH_SEGMENT}/commit$"
        ),
    ),
    ("POST", re.compile(rf"^{_MEMORY_BASE}/indexes/rebuild$")),
    ("POST", re.compile(rf"^{_MEMORY_BASE}/graph/rebuild$")),
    ("POST", re.compile(rf"^{_MEMORY_BASE}/conversations/messages$")),
    (
        "GET",
        re.compile(rf"^{_MEMORY_BASE}/conversations/{_PATH_SEGMENT}$"),
    ),
    (
        "GET",
        re.compile(
            rf"^{_MEMORY_BASE}/conversations/{_PATH_SEGMENT}/messages$"
        ),
    ),
    (
        "PATCH",
        re.compile(
            rf"^{_MEMORY_BASE}/conversations/{_PATH_SEGMENT}/metadata$"
        ),
    ),
    (
        "POST",
        re.compile(
            rf"^{_MEMORY_BASE}/conversations/{_PATH_SEGMENT}/(?:clear|trim)$"
        ),
    ),
)

_KNOWLEDGE_ACCESS_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("GET", re.compile(rf"^{_KNOWLEDGE_BASE}$")),
    ("POST", re.compile(rf"^{_KNOWLEDGE_BASE}/adapters/heartbeat$")),
    ("POST", re.compile(rf"^{_KNOWLEDGE_BASE}/search$")),
    (
        "POST",
        re.compile(
            rf"^{_KNOWLEDGE_BASE}/assets/{_IDENTIFIER}/signed-url$"
        ),
    ),
    (
        "GET",
        re.compile(
            rf"^{_KNOWLEDGE_BASE}/assets/{_IDENTIFIER}/(?:content|thumbnail)$"
        ),
    ),
)


def request_surface(scope_or_request: Any) -> str:
    scope = getattr(scope_or_request, "scope", scope_or_request)
    if not isinstance(scope, dict):
        return WEBUI_SURFACE
    state = scope.get("state")
    if not isinstance(state, dict):
        return WEBUI_SURFACE
    value = str(state.get(SCOPE_STATE_KEY) or "").strip()
    return value if value in {WEBUI_SURFACE, ADAPTER_ACCESS_SURFACE} else WEBUI_SURFACE


def is_adapter_access_request_allowed(method: str, path: str) -> bool:
    normalized_method = str(method or "").upper()
    normalized_path = str(path or "").rstrip("/") or "/"
    if normalized_method == "GET" and normalized_path == "/api/v1/health":
        return True
    return any(
        normalized_method == allowed_method and pattern.fullmatch(normalized_path)
        for allowed_method, pattern in (
            *_MEMORY_ACCESS_RULES,
            *_KNOWLEDGE_ACCESS_RULES,
        )
    )


class ListenerSurfaceApp:
    """Attach an immutable listener identity outside the shared FastAPI app."""

    def __init__(self, app: Any, *, surface: str) -> None:
        if surface not in {WEBUI_SURFACE, ADAPTER_ACCESS_SURFACE}:
            raise ValueError(f"unsupported listener surface: {surface}")
        self.app = app
        self.surface = surface

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        scoped = dict(scope)
        state = dict(scope.get("state") or {})
        state[SCOPE_STATE_KEY] = self.surface
        scoped["state"] = state

        async def send_with_surface(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers[SURFACE_HEADER] = self.surface
                headers["X-Content-Type-Options"] = "nosniff"
                headers["Referrer-Policy"] = "no-referrer"
                if self.surface == ADAPTER_ACCESS_SURFACE:
                    headers[PROTOCOL_HEADER] = ADAPTER_PROTOCOL_VERSION
                    if "Cache-Control" not in headers:
                        headers["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scoped, receive, send_with_surface)


def listener_app(app: Any, surface: str) -> ListenerSurfaceApp:
    return ListenerSurfaceApp(app, surface=surface)
